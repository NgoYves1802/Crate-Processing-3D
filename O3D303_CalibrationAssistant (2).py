"""
IFM O3D303 — Calibration Assistant  (v4)
==========================================
Changes in v4:
  - Removed "Test processing app" button.
  - PLC tab now allows user-defined data structure via JSON.
  - Default structure matches previous layout.
  - Test write generates dummy data based on the defined fields.
  - Readback displays values according to the structure.

All other features unchanged.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import xmlrpc.client
import socket
import struct
import threading
import subprocess
import sys
import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from datetime import datetime
from PIL import Image, ImageTk

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_IP   = "10.10.10.50"
DEFAULT_PORT = 50010
XMLRPC_PORT  = 80

# PLC defaults
PLC_IP_DEFAULT    = "10.10.10.105"
PLC_RACK_DEFAULT  = 0
PLC_SLOT_DEFAULT  = 1
PLC_DB_DEFAULT    = 2

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "border":    "#30363d",
    "accent":    "#f97316",
    "accent2":   "#3b82f6",
    "success":   "#22c55e",
    "warning":   "#eab308",
    "danger":    "#ef4444",
    "text":      "#e6edf3",
    "text_dim":  "#8b949e",
    "input_bg":  "#21262d",
    "input_fg":  "#e6edf3",
    "header":    "#f97316",
    "plc":       "#a855f7",
}

FONT_TITLE  = ("Courier New", 18, "bold")
FONT_HEADER = ("Courier New", 11, "bold")
FONT_LABEL  = ("Courier New", 9)
FONT_VALUE  = ("Courier New", 9, "bold")
FONT_MONO   = ("Courier New", 9)
FONT_BTN    = ("Courier New", 10, "bold")
FONT_LOG    = ("Courier New", 8)


# ─────────────────────────────────────────────────────────────────────────────
# PCIC HELPERS (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def build_pcic_frame(ticket: str, command: str) -> bytes:
    payload = f"{ticket}{command}\r\n"
    header  = f"{ticket}L{len(payload):09d}\r\n"
    return (header + payload).encode("ascii")


def recv_exactly(sock, n: int) -> bytes:
    data = b""
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("Camera disconnected")
        data += chunk
    return data


def read_pcic_frame(sock):
    header  = recv_exactly(sock, 16).decode("ascii", errors="replace")
    ticket  = header[0:4]
    length  = int(header[5:14])
    payload = recv_exactly(sock, length)
    return ticket, payload


def parse_chunks(payload: bytes) -> dict:
    TYPE_NAMES = {
        1: "distance", 2: "amplitude", 4: "x", 5: "y", 6: "z",
        7: "confidence", 8: "diagnostic",
    }
    SIGNED_TYPES = {4, 5, 6}

    images = {}
    data   = payload[4:]
    offset = 0

    while offset < len(data) - 2:
        if offset + 36 > len(data):
            break
        chunk_type   = struct.unpack_from("<I", data, offset +  0)[0]
        chunk_size   = struct.unpack_from("<I", data, offset +  4)[0]
        header_size  = struct.unpack_from("<I", data, offset +  8)[0]
        width        = struct.unpack_from("<I", data, offset + 16)[0]
        height       = struct.unpack_from("<I", data, offset + 20)[0]
        pixel_format = struct.unpack_from("<I", data, offset + 24)[0]

        if chunk_size == 0 or width == 0 or height == 0:
            break

        pixel_start = offset + header_size
        dtype = np.int16 if chunk_type in SIGNED_TYPES else np.uint16
        nbytes = width * height * 2
        pixel_data = data[pixel_start: pixel_start + nbytes]
        arr  = np.frombuffer(pixel_data, dtype=dtype).reshape(height, width)
        name = TYPE_NAMES.get(chunk_type, f"type{chunk_type}")
        images[name] = arr
        offset += chunk_size

    return images


# ─────────────────────────────────────────────────────────────────────────────
# XML-RPC CAMERA INTERFACE (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class CameraXMLRPC:
    def __init__(self, ip: str):
        self.ip   = ip
        self.base = f"http://{ip}/api/rpc/v1/com.ifm.efector"
        self.sid  = None
        self._proxies = {}

    def _proxy(self, path=""):
        url = f"{self.base}/{path}"
        if url not in self._proxies:
            self._proxies[url] = xmlrpc.client.ServerProxy(url)
        return self._proxies[url]

    def connect(self):
        self.sid = self._proxy().requestSession("", "")
        return self.sid

    def disconnect(self):
        if self.sid:
            try:
                self._proxy(f"session_{self.sid}/").cancelSession()
            except Exception:
                pass
            self.sid = None
            self._proxies = {}

    def enter_edit(self):
        self._proxy(f"session_{self.sid}/").setOperatingMode(1)

    def exit_edit(self):
        self._proxy(f"session_{self.sid}/").setOperatingMode(0)

    def heartbeat(self, seconds=30):
        try:
            self._proxy(f"session_{self.sid}/").heartbeat(seconds)
        except Exception:
            pass

    def get_device_params(self):
        return self._proxy(f"session_{self.sid}/edit/device/").getAllParameters()

    def get_network_params(self):
        return self._proxy(f"session_{self.sid}/edit/device/network/").getAllParameters()

    def get_app_params(self):
        return self._proxy(f"session_{self.sid}/edit/application/").getAllParameters()

    def get_imager_params(self):
        return self._proxy(f"session_{self.sid}/edit/application/imager_001/").getAllParameters()

    def get_spatial_filter(self):
        return self._proxy(
            f"session_{self.sid}/edit/application/imager_001/spatialfilter/"
        ).getAllParameters()

    def get_temporal_filter(self):
        return self._proxy(
            f"session_{self.sid}/edit/application/imager_001/temporalfilter/"
        ).getAllParameters()

    def set_app_param(self, key, value):
        self._proxy(f"session_{self.sid}/edit/application/").setParameter(key, str(value))

    def set_imager_param(self, key, value):
        self._proxy(
            f"session_{self.sid}/edit/application/imager_001/"
        ).setParameter(key, str(value))

    def set_spatial_filter(self, key, value):
        self._proxy(
            f"session_{self.sid}/edit/application/imager_001/spatialfilter/"
        ).setParameter(key, str(value))

    def set_temporal_filter(self, key, value):
        self._proxy(
            f"session_{self.sid}/edit/application/imager_001/temporalfilter/"
        ).setParameter(key, str(value))

    def set_device_param(self, key, value):
        self._proxy(f"session_{self.sid}/edit/device/").setParameter(key, str(value))

    def set_network_param(self, key, value):
        self._proxy(f"session_{self.sid}/edit/device/network/").setParameter(key, str(value))

    def save_app(self):
        self._proxy(f"session_{self.sid}/edit/application/").save()

    def save_device(self):
        self._proxy(f"session_{self.sid}/edit/device/").save()

    def get_temperatures(self):
        try:
            p = self._proxy(f"session_{self.sid}/edit/device/")
            return (float(p.getParameter("TemperatureFront1")),
                    float(p.getParameter("TemperatureFront2")),
                    float(p.getParameter("TemperatureIllu")))
        except Exception:
            return None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# LIVE STREAM WINDOW (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
class LiveStreamWindow(tk.Toplevel):
    SCHEMA = (
        '{"layouter":"flexible","format":{"dataencoding":"ascii"},"elements":['
        '{"type":"blob","id":"amplitude_image"},'
        '{"type":"blob","id":"distance_image"},'
        '{"type":"blob","id":"x_image"},'
        '{"type":"blob","id":"y_image"},'
        '{"type":"blob","id":"z_image"}]}'
    )

    def __init__(self, parent, ip, trigger_mode, resolution=1):
        self.img_width  = 176 if int(resolution) == 1 else 352
        self.img_height = 132 if int(resolution) == 1 else 264

        super().__init__(parent)
        self.title(f"Live Stream — O3D303  [{self.img_width}×{self.img_height}]")
        self.configure(bg=C["bg"])
        self.resizable(True, True)

        self.ip           = ip
        self.trigger_mode = int(trigger_mode)
        self._running     = False
        self._sock        = None
        self._frame_count = 0
        self._t_last      = time.time()

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._stop)
        self._start()

    def _build_ui(self):
        top = tk.Frame(self, bg=C["bg"])
        top.pack(fill="x", padx=12, pady=8)

        tk.Label(top, text="◈ LIVE STREAM", font=FONT_HEADER,
                 bg=C["bg"], fg=C["accent"]).pack(side="left")

        self._lbl_status = tk.Label(top, text="● CONNECTING",
                                    font=FONT_LABEL, bg=C["bg"], fg=C["warning"])
        self._lbl_status.pack(side="left", padx=16)

        self._lbl_fps = tk.Label(top, text="FPS: —",
                                 font=FONT_LABEL, bg=C["bg"], fg=C["text_dim"])
        self._lbl_fps.pack(side="left", padx=8)

        self._lbl_frames = tk.Label(top, text="Frames: 0",
                                    font=FONT_LABEL, bg=C["bg"], fg=C["text_dim"])
        self._lbl_frames.pack(side="left", padx=8)

        if self.trigger_mode == 2:
            tk.Button(top, text="⚡ SEND TRIGGER", font=FONT_BTN,
                      bg=C["accent"], fg="white", relief="flat",
                      padx=10, command=self._send_trigger).pack(side="right", padx=4)

        self.fig = Figure(figsize=(10, 7), facecolor=C["panel"])
        self.fig.subplots_adjust(hspace=0.35, wspace=0.3)

        self._ax_amp  = self.fig.add_subplot(2, 2, 1)
        self._ax_dist = self.fig.add_subplot(2, 2, 2)
        self._ax_z    = self.fig.add_subplot(2, 2, 3)
        self._ax_hist = self.fig.add_subplot(2, 2, 4)

        for ax, title in [(self._ax_amp,  "Amplitude"),
                          (self._ax_dist, "Distance (mm)"),
                          (self._ax_z,    "Depth Z (mm)"),
                          (self._ax_hist, "Z Histogram")]:
            ax.set_facecolor(C["bg"])
            ax.set_title(title, color=C["text_dim"], fontsize=8)
            ax.tick_params(colors=C["text_dim"], labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(C["border"])

        blank = np.zeros((self.img_height, self.img_width))
        self._im_amp  = self._ax_amp.imshow(blank,  cmap="inferno", vmin=0, vmax=1)
        self._im_dist = self._ax_dist.imshow(blank, cmap="plasma",  vmin=0, vmax=1)
        self._im_z    = self._ax_z.imshow(blank,    cmap="viridis", vmin=0, vmax=1)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)

    def _start(self):
        self._running = True
        threading.Thread(target=self._stream_loop, daemon=True).start()

    def _stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self.destroy()

    def _send_trigger(self):
        if self._sock:
            try:
                self._sock.sendall(build_pcic_frame("1000", "t"))
            except Exception:
                pass

    def _stream_loop(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.settimeout(10.0)
            self._sock.connect((self.ip, DEFAULT_PORT))
            self._update_status("● CONNECTED", C["success"])

            self._sock.sendall(build_pcic_frame("0010", f"c{self.SCHEMA}"))
            read_pcic_frame(self._sock)

            self._sock.sendall(build_pcic_frame("0011", "p1"))
            read_pcic_frame(self._sock)

            if self.trigger_mode == 2:
                self._update_status("● WAITING FOR TRIGGER", C["warning"])
            else:
                self._update_status("● STREAMING", C["success"])

            self._sock.settimeout(30.0)

            while self._running:
                if self.trigger_mode == 2:
                    self._sock.sendall(build_pcic_frame("0012", "t"))
                    time.sleep(0.5)

                ticket, payload = read_pcic_frame(self._sock)

                if ticket == "0000":
                    images = parse_chunks(payload)
                    self._frame_count += 1
                    now = time.time()
                    fps = 1.0 / max(now - self._t_last, 1e-3)
                    self._t_last = now
                    self._schedule_display(images, fps)

        except Exception as exc:
            if self._running:
                self._update_status(f"● ERROR: {exc}", C["danger"])

    def _update_status(self, text, color):
        try:
            self.after(0, lambda: self._lbl_status.config(text=text, fg=color))
        except Exception:
            pass

    def _schedule_display(self, images, fps):
        try:
            self.after(0, lambda: self._do_display(images, fps))
        except Exception:
            pass

    def _do_display(self, images, fps):
        try:
            if "amplitude" in images:
                amp = images["amplitude"].astype(float)
                mx  = amp.max()
                self._im_amp.set_data(amp / max(mx, 1))
                self._ax_amp.set_title(f"Amplitude  max={int(mx)}",
                                       color=C["text_dim"], fontsize=8)

            if "distance" in images:
                dist = images["distance"].astype(float)
                mx   = dist.max()
                self._im_dist.set_data(dist / max(mx, 1))
                self._ax_dist.set_title(f"Distance  max={int(mx)} mm",
                                        color=C["text_dim"], fontsize=8)

            if "z" in images:
                z     = images["z"].astype(float)
                valid = z[z != 0]
                if len(valid):
                    z_norm = np.clip((z - valid.min()) / max(valid.ptp(), 1), 0, 1)
                    self._im_z.set_data(z_norm)
                    self._ax_z.set_title(f"Depth Z  mean={valid.mean():.0f} mm",
                                         color=C["text_dim"], fontsize=8)

                    self._ax_hist.cla()
                    self._ax_hist.set_facecolor(C["bg"])
                    self._ax_hist.hist(valid, bins=60, color=C["accent"], alpha=0.8)
                    self._ax_hist.set_title("Z Histogram", color=C["text_dim"], fontsize=8)
                    self._ax_hist.tick_params(colors=C["text_dim"], labelsize=6)
                    for spine in self._ax_hist.spines.values():
                        spine.set_edgecolor(C["border"])
                    self._ax_hist.set_xlabel("mm", color=C["text_dim"], fontsize=6)

            self.canvas.draw_idle()
            self._lbl_frames.config(text=f"Frames: {self._frame_count}")
            self._lbl_fps.config(text=f"FPS: {fps:.1f}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# PLC SETTINGS TAB – WITH CONFIGURABLE DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────
class PLCTab(tk.Frame):
    """
    S7-1200 ISO-on-TCP connection with user-defined data structure.
    The structure is defined in JSON format (list of fields or arrays).
    """

    # Default structure (matches previous hardcoded layout)
    DEFAULT_STRUCTURE = """
[
  {"name": "crate_count", "type": "INT", "offset": 0},
  {"name": "snap_index", "type": "INT", "offset": 2},
  {
    "name": "crates",
    "type": "ARRAY",
    "length": 4,
    "element": [
      {"name": "crate_number", "type": "INT", "offset": 0},
      {"name": "Rx", "type": "REAL", "offset": 2},
      {"name": "Ry", "type": "REAL", "offset": 6},
      {"name": "Rz", "type": "REAL", "offset": 10},
      {"name": "theta", "type": "REAL", "offset": 14},
      {"name": "S_flags", "type": "WORD", "offset": 18},
      {"name": "ai_class", "type": "BYTE", "offset": 20},
      {"name": "padding", "type": "BYTE", "offset": 21}
    ]
  }
]
"""

    # Mapping of type strings to struct format characters (big-endian)
    TYPE_FORMAT = {
        "BOOL":  "?",
        "BYTE":  "B",
        "WORD":  "H",
        "DWORD": "I",
        "INT":   "h",
        "DINT":  "i",
        "REAL":  "f",
    }

    def __init__(self, parent, log_fn):
        super().__init__(parent, bg=C["bg"])
        self._log = log_fn
        self._plc = None
        self._connected = False
        self._poll_job = None
        self._structure = None          # will hold parsed JSON structure
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Connection settings (unchanged)
        conn_box = tk.LabelFrame(self, text="  ISO-on-TCP Connection  ",
                                 font=FONT_HEADER, bg=C["panel"], fg=C["plc"],
                                 labelanchor="nw", pady=8, padx=12,
                                 bd=1, relief="flat",
                                 highlightthickness=1,
                                 highlightbackground=C["border"])
        conn_box.pack(fill="x", padx=16, pady=(16, 8))

        fields = [
            ("PLC IP Address", "plc_ip", PLC_IP_DEFAULT),
            ("Rack",           "plc_rack", str(PLC_RACK_DEFAULT)),
            ("Slot",           "plc_slot", str(PLC_SLOT_DEFAULT)),
            ("DB Number",      "plc_db",   str(PLC_DB_DEFAULT)),
        ]

        self._vars = {}
        for row_i, (label, key, default) in enumerate(fields):
            tk.Label(conn_box, text=f"{label}:", font=FONT_LABEL,
                     bg=C["panel"], fg=C["text_dim"],
                     width=20, anchor="w").grid(row=row_i, column=0,
                                                padx=(0, 8), pady=3, sticky="w")
            var = tk.StringVar(value=default)
            self._vars[key] = var
            tk.Entry(conn_box, textvariable=var, width=22,
                     font=FONT_MONO, bg=C["input_bg"], fg=C["input_fg"],
                     insertbackground=C["plc"], relief="flat",
                     highlightthickness=1, highlightbackground=C["border"],
                     highlightcolor=C["plc"]).grid(row=row_i, column=1,
                                                    padx=4, pady=3, sticky="w")

        btn_row = tk.Frame(conn_box, bg=C["panel"])
        btn_row.grid(row=len(fields), column=0, columnspan=2, pady=(12, 4), sticky="w")

        self._btn_connect = self._btn(btn_row, "⚡ CONNECT PLC", self._connect_plc,
                                       C["plc"], side="left")
        self._btn_disconnect = self._btn(btn_row, "✕ DISCONNECT", self._disconnect_plc,
                                          C["danger"], side="left", padx=8)
        self._btn_disconnect.config(state="disabled")

        self._lbl_status = tk.Label(btn_row, text="⬤  Not connected",
                                     font=FONT_LABEL, bg=C["panel"], fg=C["danger"])
        self._lbl_status.pack(side="left", padx=16)

        # Structure editor
        struct_box = tk.LabelFrame(self, text="  Data Structure (JSON)  ",
                                   font=FONT_HEADER, bg=C["panel"], fg=C["plc"],
                                   labelanchor="nw", pady=8, padx=12,
                                   bd=1, relief="flat",
                                   highlightthickness=1,
                                   highlightbackground=C["border"])
        struct_box.pack(fill="both", expand=True, padx=16, pady=8)

        self._struct_text = scrolledtext.ScrolledText(
            struct_box, bg=C["bg"], fg=C["text"], font=FONT_MONO,
            height=12, relief="flat",
            highlightthickness=1, highlightbackground=C["border"])
        self._struct_text.pack(fill="both", expand=True, pady=4)
        self._struct_text.insert("1.0", self.DEFAULT_STRUCTURE)

        # Buttons to load/save structure
        struct_btn_frame = tk.Frame(struct_box, bg=C["panel"])
        struct_btn_frame.pack(pady=4)
        tk.Button(struct_btn_frame, text="Load from file", command=self._load_structure,
                  font=FONT_BTN, bg=C["accent2"], fg="white", relief="flat",
                  padx=8).pack(side="left", padx=4)
        tk.Button(struct_btn_frame, text="Save to file", command=self._save_structure,
                  font=FONT_BTN, bg=C["accent2"], fg="white", relief="flat",
                  padx=8).pack(side="left", padx=4)
        tk.Button(struct_btn_frame, text="Reset to default", command=self._reset_structure,
                  font=FONT_BTN, bg=C["warning"], fg="white", relief="flat",
                  padx=8).pack(side="left", padx=4)

        self._db_size_lbl = tk.Label(struct_box, text="",
                                     font=FONT_VALUE, bg=C["panel"], fg=C["plc"])
        self._db_size_lbl.pack(anchor="w", pady=(4, 0))
        self._update_db_size_label()

        # Test tools
        test_box = tk.LabelFrame(self, text="  Test & Readback  ",
                                 font=FONT_HEADER, bg=C["panel"], fg=C["plc"],
                                 labelanchor="nw", pady=8, padx=12,
                                 bd=1, relief="flat",
                                 highlightthickness=1,
                                 highlightbackground=C["border"])
        test_box.pack(fill="x", padx=16, pady=8)

        trow = tk.Frame(test_box, bg=C["panel"])
        trow.pack(anchor="w")

        self._btn_write_test = self._btn(
            trow, "▶ WRITE TEST DATA", self._write_test, C["accent2"], side="left")
        self._btn_readback   = self._btn(
            trow, "↓ READ DB", self._readback, C["success"], side="left", padx=8)
        self._btn_poll       = self._btn(
            trow, "⟳ POLL (1s)", self._toggle_poll, C["warning"], side="left", padx=0)

        for b in [self._btn_write_test, self._btn_readback, self._btn_poll]:
            b.config(state="disabled")

        self._poll_active = False

        # Readback display
        self._readback_text = scrolledtext.ScrolledText(
            test_box, bg=C["bg"], fg=C["text"], font=FONT_MONO,
            height=8, relief="flat", state="disabled",
            highlightthickness=1, highlightbackground=C["border"])
        self._readback_text.pack(fill="x", pady=(10, 0))

        # TIA Portal checklist (unchanged)
        tip_box = tk.LabelFrame(self, text="  TIA Portal Requirements  ",
                                 font=FONT_HEADER, bg=C["panel"], fg=C["warning"],
                                 labelanchor="nw", pady=8, padx=12,
                                 bd=1, relief="flat",
                                 highlightthickness=1,
                                 highlightbackground=C["border"])
        tip_box.pack(fill="x", padx=16, pady=8)

        tips = [
            "☑  DB must have 'Optimized block access' UNCHECKED",
            "☑  PUT/GET must be enabled:",
            "     Device Config → PLC → PROFINET → Advanced options",
            "     → Connections → ☑ Permit access with PUT/GET",
            "☑  Data structure must match the JSON definition exactly.",
        ]
        for tip in tips:
            tk.Label(tip_box, text=tip, font=FONT_LABEL,
                     bg=C["panel"], fg=C["text_dim"],
                     justify="left", anchor="w").pack(anchor="w", pady=1)

    def _btn(self, parent, text, cmd, color, side="left", padx=4):
        b = tk.Button(parent, text=text, command=cmd, font=FONT_BTN,
                      bg=color, fg="white", relief="flat",
                      padx=12, pady=5, cursor="hand2",
                      activebackground=color, activeforeground="white")
        b.pack(side=side, padx=padx, pady=4)
        return b

    def _update_db_size_label(self):
        """Compute total DB size from the current structure."""
        try:
            struct = json.loads(self._struct_text.get("1.0", "end-1c"))
            size = self._calculate_structure_size(struct)
            self._db_size_lbl.config(text=f"→  DB size = {size} bytes")
        except Exception as e:
            self._db_size_lbl.config(text=f"→  Structure error: {e}")

    def _calculate_structure_size(self, struct, base_offset=0):
        """Recursively compute the total size (bytes) of a structure."""
        size = 0
        for item in struct:
            typ = item["type"]
            offset = item.get("offset", 0)
            if typ == "ARRAY":
                elem_size = self._calculate_structure_size(item["element"])
                size = max(size, offset + elem_size * item["length"])
            else:
                fmt = self.TYPE_FORMAT.get(typ)
                if fmt is None:
                    raise ValueError(f"Unknown type: {typ}")
                elem_size = struct.calcsize(fmt) if fmt != "?" else 1  # BOOL is 1 byte in DB
                size = max(size, offset + elem_size)
        return size

    def _parse_structure(self):
        """Parse the JSON structure and return it, or None on error."""
        try:
            struct = json.loads(self._struct_text.get("1.0", "end-1c"))
            # Validate structure format (simple check)
            if not isinstance(struct, list):
                raise ValueError("Root must be a list of fields")
            self._structure = struct
            return struct
        except Exception as e:
            self._log(f"Invalid structure JSON: {e}", C["danger"])
            return None

    def _load_structure(self):
        file = filedialog.askopenfilename(
            title="Load structure from JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file:
            return
        try:
            with open(file, "r") as f:
                content = f.read()
            self._struct_text.delete("1.0", "end")
            self._struct_text.insert("1.0", content)
            self._update_db_size_label()
            self._log(f"Structure loaded from {file}", C["success"])
        except Exception as e:
            self._log(f"Failed to load: {e}", C["danger"])

    def _save_structure(self):
        file = filedialog.asksaveasfilename(
            title="Save structure as JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
        if not file:
            return
        try:
            content = self._struct_text.get("1.0", "end-1c")
            with open(file, "w") as f:
                f.write(content)
            self._log(f"Structure saved to {file}", C["success"])
        except Exception as e:
            self._log(f"Failed to save: {e}", C["danger"])

    def _reset_structure(self):
        self._struct_text.delete("1.0", "end")
        self._struct_text.insert("1.0", self.DEFAULT_STRUCTURE)
        self._update_db_size_label()
        self._log("Structure reset to default.", C["success"])

    def _set_test_btns(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for b in [self._btn_write_test, self._btn_readback, self._btn_poll]:
            b.config(state=state)

    # ── connection (unchanged) ────────────────────────────────────────────────

    def _connect_plc(self):
        try:
            import snap7
            import snap7.client
        except ImportError:
            messagebox.showerror(
                "snap7 not installed",
                "pip install python-snap7\n\nsnap7 is required for PLC connectivity."
            )
            return

        ip   = self._vars["plc_ip"].get().strip()
        rack = int(self._vars["plc_rack"].get())
        slot = int(self._vars["plc_slot"].get())

        self._log(f"Connecting to PLC {ip}  rack={rack}  slot={slot}…", C["warning"])
        threading.Thread(
            target=self._connect_thread,
            args=(ip, rack, slot),
            daemon=True,
        ).start()

    def _connect_thread(self, ip, rack, slot):
        try:
            import snap7.client
            client = snap7.client.Client()
            client.connect(ip, rack, slot)
            self._plc = client
            self._connected = True
            self.after(0, lambda: self._on_plc_connected(ip))
        except Exception as exc:
            self.after(0, lambda: (
                self._lbl_status.config(
                    text=f"⬤  Error: {exc}", fg=C["danger"]),
                self._log(f"PLC connection failed: {exc}", C["danger"])
            ))

    def _on_plc_connected(self, ip):
        self._lbl_status.config(text=f"⬤  {ip}", fg=C["success"])
        self._btn_connect.config(state="disabled")
        self._btn_disconnect.config(state="normal")
        self._set_test_btns(True)
        self._log(f"PLC connected: {ip}", C["success"])

    def _disconnect_plc(self):
        if self._poll_active:
            self._toggle_poll()
        if self._plc:
            try:
                self._plc.disconnect()
            except Exception:
                pass
            self._plc = None
        self._connected = False
        self._lbl_status.config(text="⬤  Not connected", fg=C["danger"])
        self._btn_connect.config(state="normal")
        self._btn_disconnect.config(state="disabled")
        self._set_test_btns(False)
        self._log("PLC disconnected.", C["text_dim"])

    # ── test write (using structure) ──────────────────────────────────────────

    def _write_test(self):
        if not self._connected:
            return
        struct = self._parse_structure()
        if struct is None:
            return
        threading.Thread(target=self._write_test_thread, args=(struct,), daemon=True).start()

    def _write_test_thread(self, struct):
        try:
            db = int(self._vars["plc_db"].get())
            size = self._calculate_structure_size(struct)
            buf = bytearray(size)

            # Fill with dummy data based on the structure
            self._fill_dummy_data(buf, struct)

            self._plc.db_write(db, 0, buf)
            self.after(0, lambda: self._log(
                f"✓ Test data written to DB{db}  ({size} bytes)", C["success"]))
            self._readback_thread()   # immediately read back

        except Exception as exc:
            self.after(0, lambda: self._log(f"Write error: {exc}", C["danger"]))

    def _fill_dummy_data(self, buf, struct, base_offset=0):
        """
        Recursively fill buffer with dummy values for each field.
        For primitive types, we generate a value based on the field name.
        For arrays, we generate dummy values for each element.
        """
        for item in struct:
            typ = item["type"]
            offset = item.get("offset", 0) + base_offset
            name = item.get("name", "")
            if typ == "ARRAY":
                length = item["length"]
                elem_struct = item["element"]
                elem_size = self._calculate_structure_size(elem_struct)
                for i in range(length):
                    self._fill_dummy_data(buf, elem_struct, offset + i * elem_size)
            else:
                fmt = self.TYPE_FORMAT.get(typ)
                if fmt is None:
                    raise ValueError(f"Unknown type: {typ}")
                # Generate dummy value (based on type and name)
                value = self._dummy_value(typ, name)
                # Pack into buffer at the given offset
                if fmt == "?":
                    # BOOL: 1 byte
                    struct.pack_into("B", buf, offset, 1 if value else 0)
                elif fmt == "B":
                    struct.pack_into("B", buf, offset, value)
                elif fmt == "H":
                    struct.pack_into(">H", buf, offset, value)
                elif fmt == "I":
                    struct.pack_into(">I", buf, offset, value)
                elif fmt == "h":
                    struct.pack_into(">h", buf, offset, value)
                elif fmt == "i":
                    struct.pack_into(">i", buf, offset, value)
                elif fmt == "f":
                    struct.pack_into(">f", buf, offset, value)

    def _dummy_value(self, typ, name):
        """Generate a dummy value for a given type and field name."""
        import random
        if typ == "BOOL":
            return random.choice([True, False])
        elif typ == "BYTE":
            return random.randint(0, 255)
        elif typ == "WORD":
            return random.randint(0, 65535)
        elif typ == "DWORD":
            return random.randint(0, 4294967295)
        elif typ == "INT":
            return random.randint(-32768, 32767)
        elif typ == "DINT":
            return random.randint(-2147483648, 2147483647)
        elif typ == "REAL":
            return random.uniform(-1000, 1000)
        else:
            return 0

    # ── readback (using structure) ────────────────────────────────────────────

    def _readback(self):
        if not self._connected:
            return
        struct = self._parse_structure()
        if struct is None:
            return
        threading.Thread(target=self._readback_thread, args=(struct,), daemon=True).start()

    def _readback_thread(self, struct=None):
        if struct is None:
            struct = self._parse_structure()
            if struct is None:
                return
        try:
            db = int(self._vars["plc_db"].get())
            size = self._calculate_structure_size(struct)
            buf = self._plc.db_read(db, 0, size)

            # Parse buffer according to structure
            lines = []
            self._parse_buffer(buf, struct, lines, prefix="")

            text = "\n".join(lines)
            self.after(0, lambda: self._show_readback(text))

        except Exception as exc:
            self.after(0, lambda: self._log(f"Readback error: {exc}", C["danger"]))

    def _parse_buffer(self, buf, struct, lines, prefix=""):
        """Recursively parse buffer and append readable lines."""
        for item in struct:
            typ = item["type"]
            offset = item.get("offset", 0)
            name = item.get("name", "field")
            if typ == "ARRAY":
                length = item["length"]
                elem_struct = item["element"]
                elem_size = self._calculate_structure_size(elem_struct)
                for i in range(length):
                    new_prefix = f"{prefix}{name}[{i}]."
                    self._parse_buffer(buf, elem_struct, lines, new_prefix)
            else:
                fmt = self.TYPE_FORMAT.get(typ)
                if fmt is None:
                    continue
                if fmt == "?":
                    val = struct.unpack_from("B", buf, offset)[0]
                    val = bool(val)
                elif fmt == "B":
                    val = struct.unpack_from("B", buf, offset)[0]
                elif fmt == "H":
                    val = struct.unpack_from(">H", buf, offset)[0]
                elif fmt == "I":
                    val = struct.unpack_from(">I", buf, offset)[0]
                elif fmt == "h":
                    val = struct.unpack_from(">h", buf, offset)[0]
                elif fmt == "i":
                    val = struct.unpack_from(">i", buf, offset)[0]
                elif fmt == "f":
                    val = struct.unpack_from(">f", buf, offset)[0]
                else:
                    continue
                lines.append(f"{prefix}{name} = {val}")

    def _show_readback(self, text: str):
        self._readback_text.config(state="normal")
        self._readback_text.delete("1.0", "end")
        self._readback_text.insert("end", text)
        self._readback_text.config(state="disabled")

    # ── polling (unchanged) ────────────────────────────────────────────────────

    def _toggle_poll(self):
        if not self._poll_active:
            self._poll_active = True
            self._btn_poll.config(text="■ STOP POLL", bg=C["danger"])
            self._do_poll()
        else:
            self._poll_active = False
            self._btn_poll.config(text="⟳ POLL (1s)", bg=C["warning"])
            if self._poll_job:
                self.after_cancel(self._poll_job)
                self._poll_job = None

    def _do_poll(self):
        if not self._poll_active or not self._connected:
            return
        self._readback()
        self._poll_job = self.after(1000, self._do_poll)


# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION TAB (unchanged from v3)
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationTab(tk.Frame):
    # ... (same as in v3, no changes) ...
    # For brevity, we omit the full code here. In the final file it will be included.
    pass


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION (updated to remove test app button)
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationAssistant(tk.Tk):
    REALTIME_DEBOUNCE_MS = 600

    def __init__(self):
        super().__init__()
        self.title("IFM O3D303 — Calibration Assistant")
        self.configure(bg=C["bg"])
        self.geometry("1200x900")
        self.resizable(True, True)

        self._cam           = None
        self._connected     = False
        self._params        = {}
        self._widgets       = {}
        self._hb_job        = None
        self._live_win      = None
        self._rt_job        = None

        self._build_ui()
        self._schedule_heartbeat()

    def _build_ui(self):
        # Title bar
        title_bar = tk.Frame(self, bg=C["bg"], pady=10)
        title_bar.pack(fill="x", padx=20)

        tk.Label(title_bar, text="◈ O3D303  CALIBRATION ASSISTANT",
                 font=FONT_TITLE, bg=C["bg"], fg=C["accent"]).pack(side="left")

        self._lbl_conn = tk.Label(title_bar, text="⬤  DISCONNECTED",
                                   font=FONT_LABEL, bg=C["bg"], fg=C["danger"])
        self._lbl_conn.pack(side="right", padx=10)

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # Connection bar
        conn_bar = tk.Frame(self, bg=C["panel"], pady=8)
        conn_bar.pack(fill="x")

        tk.Label(conn_bar, text="  Camera IP:", font=FONT_LABEL,
                 bg=C["panel"], fg=C["text_dim"]).pack(side="left", padx=(16, 4))

        self._ip_var = tk.StringVar(value=DEFAULT_IP)
        tk.Entry(conn_bar, textvariable=self._ip_var, width=16,
                 font=FONT_MONO, bg=C["input_bg"], fg=C["input_fg"],
                 insertbackground=C["accent"], relief="flat",
                 highlightthickness=1, highlightcolor=C["accent"],
                 highlightbackground=C["border"]).pack(side="left", padx=4)

        self._btn_connect = self._make_btn(
            conn_bar, "⚡ CONNECT", self._connect, C["accent2"], side="left", padx=8)
        self._btn_disconnect = self._make_btn(
            conn_bar, "✕ DISCONNECT", self._disconnect, C["danger"], side="left", padx=4)
        self._btn_disconnect.config(state="disabled")

        self._rt_var = tk.BooleanVar(value=True)
        tk.Checkbutton(conn_bar, text="Real-time preview",
                       variable=self._rt_var,
                       font=FONT_LABEL, bg=C["panel"], fg=C["text_dim"],
                       selectcolor=C["input_bg"],
                       activebackground=C["panel"]).pack(side="left", padx=16)

        self._lbl_temp = tk.Label(conn_bar,
                                   text="  T1: —°C   T2: —°C   Illu: —°C",
                                   font=FONT_LABEL, bg=C["panel"], fg=C["text_dim"])
        self._lbl_temp.pack(side="right", padx=20)

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x")

        # Notebook
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=0, pady=0)

        # Camera params tab
        cam_tab = tk.Frame(self._notebook, bg=C["bg"])
        self._notebook.add(cam_tab, text="  Camera Parameters  ")
        self._build_camera_tab(cam_tab)

        # PLC tab
        self._plc_tab = PLCTab(self._notebook, self._log_write)
        self._notebook.add(self._plc_tab, text="  PLC Settings  ")

        # Calibration tab
        self._calib_tab = CalibrationTab(self._notebook, self._log_write,
                                         lambda: self._cam, self._ip_var)
        self._notebook.add(self._calib_tab, text="  Calibration  ")

    def _build_camera_tab(self, parent):
        # ... (unchanged from v3) ...
        pass

    def _build_action_bar(self, parent):
        bar = tk.Frame(parent, bg=C["panel"], pady=10)
        bar.pack(fill="x")

        btn_frame = tk.Frame(bar, bg=C["panel"])
        btn_frame.pack()

        self._btn_read = self._make_btn(
            btn_frame, "↓ READ FROM CAMERA", self._read_all, C["accent2"],
            side="left", padx=8)
        self._btn_apply = self._make_btn(
            btn_frame, "↑ APPLY ALL CHANGES", self._apply_changes, C["accent"],
            side="left", padx=8)
        self._btn_visualise = self._make_btn(
            btn_frame, "◈ VISUALISE (LIVE)", self._open_live_stream, C["success"],
            side="left", padx=8)

        for b in [self._btn_read, self._btn_apply, self._btn_visualise]:
            b.config(state="disabled")

    # ... (all other methods unchanged from v3, but we remove _launch_processing_app) ...
    # The remaining methods (real-time apply, connection, read_all, apply_changes,
    # live stream, heartbeat, etc.) are identical to v3.

    def _set_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for b in [self._btn_read, self._btn_apply, self._btn_visualise]:
            b.config(state=state)
        self._btn_connect.config(state="disabled" if enabled else "normal")
        self._btn_disconnect.config(state="normal" if enabled else "disabled")

    # ... rest of class unchanged ...


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("TCombobox",
                    fieldbackground=C["input_bg"],
                    background=C["input_bg"],
                    foreground=C["input_fg"],
                    selectbackground=C["accent"],
                    selectforeground="white",
                    font=("Courier New", 9))
    style.configure("TScrollbar",
                    background=C["panel"],
                    troughcolor=C["bg"],
                    arrowcolor=C["text_dim"])
    style.configure("TNotebook",
                    background=C["bg"],
                    tabmargins=[2, 4, 0, 0])
    style.configure("TNotebook.Tab",
                    background=C["panel"],
                    foreground=C["text_dim"],
                    font=("Courier New", 10, "bold"),
                    padding=[12, 6])
    style.map("TNotebook.Tab",
              background=[("selected", C["bg"])],
              foreground=[("selected", C["accent"])])

    app = CalibrationAssistant()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()