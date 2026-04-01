"""
IFM O3D303 — Calibration Assistant  (v3)
==========================================
New in v3:
  - Calibration tab: capture frame, aim crosshair, pick 4 points,
    evaluate rectangle/perpendicularity/flatness, auto-adjust extrinsic angles,
    3D point cloud viewer.
  - All previous features (live stream, PLC tab, real‑time params) remain.

Requirements:
    pip install numpy matplotlib pillow snap7 (snap7 optional)

Usage:
    python O3D303_CalibrationAssistant.py
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
from PIL import Image, ImageTk   # Added for calibration tab image scaling

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
PLC_MAX_CRATES    = 4

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
    "plc":       "#a855f7",   # purple for PLC tab
}

FONT_TITLE  = ("Courier New", 18, "bold")
FONT_HEADER = ("Courier New", 11, "bold")
FONT_LABEL  = ("Courier New", 9)
FONT_VALUE  = ("Courier New", 9, "bold")
FONT_MONO   = ("Courier New", 9)
FONT_BTN    = ("Courier New", 10, "bold")
FONT_LOG    = ("Courier New", 8)


# ─────────────────────────────────────────────────────────────────────────────
# PCIC HELPERS
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
    """
    Parse image chunks from a PCIC frame payload.

    pixel_format field:
      0x00 = uint16  (amplitude, distance, confidence)
      0x01 = int16   (x, y, z — signed mm values)
    """
    TYPE_NAMES = {
        1: "distance", 2: "amplitude", 4: "x", 5: "y", 6: "z",
        7: "confidence", 8: "diagnostic",
    }
    SIGNED_TYPES = {4, 5, 6}   # x, y, z are signed int16

    images = {}
    data   = payload[4:]       # skip 4-byte ticket prefix
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
        # Use chunk type to decide dtype (more reliable than pixel_format field)
        dtype = np.int16 if chunk_type in SIGNED_TYPES else np.uint16
        nbytes = width * height * 2
        pixel_data = data[pixel_start: pixel_start + nbytes]
        arr  = np.frombuffer(pixel_data, dtype=dtype).reshape(height, width)
        name = TYPE_NAMES.get(chunk_type, f"type{chunk_type}")
        images[name] = arr
        offset += chunk_size

    return images


# ─────────────────────────────────────────────────────────────────────────────
# XML-RPC CAMERA INTERFACE
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
# LIVE STREAM WINDOW
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
        # ── Resolve dimensions FIRST — used in title ───────────────────────────
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
        """Send a software trigger (TriggerMode=2)."""
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

            # Upload blob schema
            self._sock.sendall(build_pcic_frame("0010", f"c{self.SCHEMA}"))
            read_pcic_frame(self._sock)   # schema ACK

            # Enable streaming
            self._sock.sendall(build_pcic_frame("0011", "p1"))
            read_pcic_frame(self._sock)   # streaming ACK

            if self.trigger_mode == 2:
                self._update_status("● WAITING FOR TRIGGER", C["warning"])
            else:
                self._update_status("● STREAMING", C["success"])

            self._sock.settimeout(30.0)

            while self._running:
                if self.trigger_mode == 2:
                    # Auto-fire at ~2 fps for preview in software trigger mode
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
        """Push display update to the Tk main thread."""
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
# PLC SETTINGS TAB
# ─────────────────────────────────────────────────────────────────────────────
class PLCTab(tk.Frame):
    """
    S7-1200 ISO-on-TCP connection settings and test tools.
    Requires python-snap7 (pip install python-snap7).
    """

    # DB layout constants (must match TIA Portal DB definition)
    HEADER_SIZE = 4     # crate_count (INT 2B) + snap_index (INT 2B)
    CRATE_SIZE  = 22    # bytes per crate record

    def __init__(self, parent, log_fn):
        super().__init__(parent, bg=C["bg"])
        self._log   = log_fn
        self._plc   = None
        self._connected = False
        self._poll_job  = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Connection settings ────────────────────────────────────────────────
        conn_box = tk.LabelFrame(self, text="  ISO-on-TCP Connection  ",
                                  font=FONT_HEADER, bg=C["panel"], fg=C["plc"],
                                  labelanchor="nw", pady=8, padx=12,
                                  bd=1, relief="flat",
                                  highlightthickness=1,
                                  highlightbackground=C["border"])
        conn_box.pack(fill="x", padx=16, pady=(16, 8))

        fields = [
            ("PLC IP Address",  "plc_ip",    PLC_IP_DEFAULT),
            ("Rack",            "plc_rack",  str(PLC_RACK_DEFAULT)),
            ("Slot",            "plc_slot",  str(PLC_SLOT_DEFAULT)),
            ("DB Number",       "plc_db",    str(PLC_DB_DEFAULT)),
            ("Max crates (N)",  "plc_max",   str(PLC_MAX_CRATES)),
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

        # ── DB layout info ────────────────────────────────────────────────────
        info_box = tk.LabelFrame(self, text="  DB Memory Layout  ",
                                  font=FONT_HEADER, bg=C["panel"], fg=C["plc"],
                                  labelanchor="nw", pady=8, padx=12,
                                  bd=1, relief="flat",
                                  highlightthickness=1,
                                  highlightbackground=C["border"])
        info_box.pack(fill="x", padx=16, pady=8)

        layout_text = (
            "Offset  Size  Type   Name\n"
            "─────────────────────────────────────────────\n"
            "  0       2    INT    crate_count\n"
            "  2       2    INT    snap_index\n"
            "─── Array[1..N] of Crate_Type at offset 4 ───\n"
            "  +0      2    INT    crate_number\n"
            "  +2      4    REAL   Rx  (mm)\n"
            "  +6      4    REAL   Ry  (mm)\n"
            " +10      4    REAL   Rz  (mm)\n"
            " +14      4    REAL   theta  (rad)\n"
            " +18      2    WORD   S_flags  S1=bit0 … S12=bit11\n"
            " +20      1    BYTE   ai_class (1=confirmed)\n"
            " +21      1    BYTE   padding\n"
            "─────────────────────────────────────────────\n"
            "Total = 4 + 22 × N bytes"
        )
        tk.Label(info_box, text=layout_text, font=FONT_MONO,
                 bg=C["panel"], fg=C["text_dim"],
                 justify="left", anchor="w").pack(anchor="w")

        self._db_size_lbl = tk.Label(info_box, text="",
                                      font=FONT_VALUE, bg=C["panel"], fg=C["plc"])
        self._db_size_lbl.pack(anchor="w", pady=(4, 0))
        self._update_db_size_label()

        # Recalculate when Max crates changes
        self._vars["plc_max"].trace_add("write", lambda *_: self._update_db_size_label())

        # ── Test tools ────────────────────────────────────────────────────────
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

        # ── TIA Portal checklist ──────────────────────────────────────────────
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
            "☑  DB offset 0 = crate_count (INT)",
            "☑  DB offset 2 = snap_index  (INT)",
            "☑  Each crate record = exactly 22 bytes",
        ]
        for tip in tips:
            tk.Label(tip_box, text=tip, font=FONT_LABEL,
                     bg=C["panel"], fg=C["text_dim"],
                     justify="left", anchor="w").pack(anchor="w", pady=1)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _btn(self, parent, text, cmd, color, side="left", padx=4):
        b = tk.Button(parent, text=text, command=cmd, font=FONT_BTN,
                      bg=color, fg="white", relief="flat",
                      padx=12, pady=5, cursor="hand2",
                      activebackground=color, activeforeground="white")
        b.pack(side=side, padx=padx, pady=4)
        return b

    def _update_db_size_label(self):
        try:
            n = int(self._vars["plc_max"].get())
            total = self.HEADER_SIZE + self.CRATE_SIZE * n
            self._db_size_lbl.config(
                text=f"→  DB size with N={n} crates = {total} bytes"
            )
        except Exception:
            pass

    def _set_test_btns(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for b in [self._btn_write_test, self._btn_readback, self._btn_poll]:
            b.config(state=state)

    # ── connection ────────────────────────────────────────────────────────────

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

    # ── test write ────────────────────────────────────────────────────────────

    def _write_test(self):
        if not self._connected:
            return
        threading.Thread(target=self._write_test_thread, daemon=True).start()

    def _write_test_thread(self):
        try:
            db  = int(self._vars["plc_db"].get())
            n   = int(self._vars["plc_max"].get())
            total = self.HEADER_SIZE + self.CRATE_SIZE * n
            buf = bytearray(total)

            # Header: crate_count=2, snap_index=42
            struct.pack_into(">h", buf, 0, 2)
            struct.pack_into(">h", buf, 2, 42)

            # Two test crates
            for idx, (crate_num, rx, ry, rz, theta, s_flags, ai) in enumerate([
                (1,  150.0,  -80.0, 1020.0, 0.05, 0b000000000111, 1),
                (2, -200.0,  120.0, 1020.0, -0.1, 0b000000111000, 0),
            ]):
                if idx >= n:
                    break
                base = self.HEADER_SIZE + idx * self.CRATE_SIZE
                struct.pack_into(">h", buf, base + 0,  crate_num)
                struct.pack_into(">f", buf, base + 2,  rx)
                struct.pack_into(">f", buf, base + 6,  ry)
                struct.pack_into(">f", buf, base + 10, rz)
                struct.pack_into(">f", buf, base + 14, theta)
                struct.pack_into(">H", buf, base + 18, s_flags)
                struct.pack_into(">B", buf, base + 20, ai)

            self._plc.db_write(db, 0, buf)
            self.after(0, lambda: self._log(
                f"✓ Test data written to DB{db}  ({len(buf)} bytes)", C["success"]))
            self._readback_thread()

        except Exception as exc:
            self.after(0, lambda: self._log(f"Write error: {exc}", C["danger"]))

    # ── readback ─────────────────────────────────────────────────────────────

    def _readback(self):
        if not self._connected:
            return
        threading.Thread(target=self._readback_thread, daemon=True).start()

    def _readback_thread(self):
        try:
            db    = int(self._vars["plc_db"].get())
            n_max = int(self._vars["plc_max"].get())
            total = self.HEADER_SIZE + self.CRATE_SIZE * n_max
            buf   = self._plc.db_read(db, 0, total)

            crate_count = struct.unpack_from(">h", buf, 0)[0]
            snap_index  = struct.unpack_from(">h", buf, 2)[0]

            lines = [
                f"DB{db} readback  [{datetime.now().strftime('%H:%M:%S')}]",
                f"  crate_count = {crate_count}",
                f"  snap_index  = {snap_index}",
                "",
            ]
            for i in range(min(crate_count, n_max)):
                base         = self.HEADER_SIZE + i * self.CRATE_SIZE
                crate_number = struct.unpack_from(">h", buf, base + 0)[0]
                rx    = struct.unpack_from(">f", buf, base + 2)[0]
                ry    = struct.unpack_from(">f", buf, base + 6)[0]
                rz    = struct.unpack_from(">f", buf, base + 10)[0]
                theta = struct.unpack_from(">f", buf, base + 14)[0]
                flags = struct.unpack_from(">H", buf, base + 18)[0]
                ai    = struct.unpack_from(">B", buf, base + 20)[0]

                slots = [f"S{j+1}" for j in range(12) if flags & (1 << j)]
                lines += [
                    f"  Crate[{i+1}]  #{crate_number}",
                    f"    Rx={rx:.1f}  Ry={ry:.1f}  Rz={rz:.1f}  θ={theta:.4f}",
                    f"    Slots filled: {', '.join(slots) if slots else 'none'}",
                    f"    AI: {'✓ confirmed' if ai else '✗ not confirmed'}",
                    "",
                ]

            self.after(0, lambda: self._show_readback("\n".join(lines)))

        except Exception as exc:
            self.after(0, lambda: self._log(f"Readback error: {exc}", C["danger"]))

    def _show_readback(self, text: str):
        self._readback_text.config(state="normal")
        self._readback_text.delete("1.0", "end")
        self._readback_text.insert("end", text)
        self._readback_text.config(state="disabled")

    # ── polling ───────────────────────────────────────────────────────────────

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


# =============================================================================
# CALIBRATION TAB (New)
# =============================================================================
class CalibrationTab(tk.Frame):
    """
    Provides a guided calibration workflow:
      - Capture a single frame (PCIC)
      - Display amplitude image with a centre crosshair
      - Pick up to 4 points
      - Evaluate rectangle, perpendicularity and flatness
      - Auto‑adjust extrinsic rotation angles
    """
    # PCIC constants (same as LiveStreamWindow)
    SCHEMA = (
        '{"layouter":"flexible","format":{"dataencoding":"ascii"},"elements":['
        '{"type":"blob","id":"amplitude_image"},'
        '{"type":"blob","id":"distance_image"},'
        '{"type":"blob","id":"x_image"},'
        '{"type":"blob","id":"y_image"},'
        '{"type":"blob","id":"z_image"}]}'
    )

    def __init__(self, parent, log_fn, cam_accessor, ip_var):
        super().__init__(parent, bg=C["bg"])
        self._log = log_fn                     # function to write to main log
        self._cam_accessor = cam_accessor      # callable that returns current CameraXMLRPC or None
        self._ip_var = ip_var                  # StringVar containing camera IP

        self._captured_images = None           # dict with keys 'amplitude','x','y','z','distance'
        self._points = []                      # list of (x, y, z) camera coordinates for picked points
        self._point_markers = []               # canvas ids for drawn circles
        self._current_resolution = 1            # 0 = full 352x264, 1 = binned 176x132

        self._build_ui()

    # -------------------------------------------------------------------------
    # UI building
    # -------------------------------------------------------------------------
    def _build_ui(self):
        # Left side: image display + controls
        left_frame = tk.Frame(self, bg=C["bg"])
        left_frame.pack(side="left", fill="both", expand=True, padx=12, pady=8)

        # Image display area
        self._canvas = tk.Canvas(left_frame, bg=C["bg"], highlightthickness=0,
                                 width=400, height=300)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Button-1>", self._on_canvas_click)

        # Status label for point count
        self._point_status = tk.Label(left_frame, text="Points: 0/4", font=FONT_LABEL,
                                      bg=C["bg"], fg=C["text_dim"])
        self._point_status.pack(pady=4)

        # Control buttons
        btn_frame = tk.Frame(left_frame, bg=C["bg"])
        btn_frame.pack(pady=8)

        self._btn_capture = tk.Button(btn_frame, text="📸 Capture Frame", command=self._capture_frame,
                                      font=FONT_BTN, bg=C["accent2"], fg="white", relief="flat",
                                      padx=12, pady=4)
        self._btn_capture.pack(side="left", padx=4)

        self._btn_reset_points = tk.Button(btn_frame, text="↺ Reset Points", command=self._reset_points,
                                           font=FONT_BTN, bg=C["warning"], fg="white", relief="flat",
                                           padx=12, pady=4)
        self._btn_reset_points.pack(side="left", padx=4)
        self._btn_reset_points.config(state="disabled")

        self._btn_show3d = tk.Button(btn_frame, text="🔍 3D Point Cloud", command=self._show_3d,
                                     font=FONT_BTN, bg=C["plc"], fg="white", relief="flat",
                                     padx=12, pady=4)
        self._btn_show3d.pack(side="left", padx=4)
        self._btn_show3d.config(state="disabled")

        # Right side: parameters and evaluation
        right_frame = tk.Frame(self, bg=C["bg"], width=350)
        right_frame.pack(side="right", fill="y", padx=12, pady=8)
        right_frame.pack_propagate(False)

        # Rectangle dimensions and tolerances
        dim_frame = tk.LabelFrame(right_frame, text="  Rectangle Parameters  ",
                                  font=FONT_HEADER, bg=C["panel"], fg=C["accent"],
                                  labelanchor="nw", pady=8, padx=12,
                                  bd=1, relief="flat", highlightthickness=1,
                                  highlightbackground=C["border"])
        dim_frame.pack(fill="x", pady=4)

        # Expected width and height
        tk.Label(dim_frame, text="Expected width (mm):", font=FONT_LABEL,
                 bg=C["panel"], fg=C["text_dim"]).grid(row=0, column=0, sticky="w", pady=2)
        self._width_var = tk.StringVar(value="500.0")
        tk.Entry(dim_frame, textvariable=self._width_var, width=12,
                 font=FONT_MONO, bg=C["input_bg"], fg=C["input_fg"]).grid(row=0, column=1, padx=8, pady=2)

        tk.Label(dim_frame, text="Expected height (mm):", font=FONT_LABEL,
                 bg=C["panel"], fg=C["text_dim"]).grid(row=1, column=0, sticky="w", pady=2)
        self._height_var = tk.StringVar(value="300.0")
        tk.Entry(dim_frame, textvariable=self._height_var, width=12,
                 font=FONT_MONO, bg=C["input_bg"], fg=C["input_fg"]).grid(row=1, column=1, padx=8, pady=2)

        tk.Label(dim_frame, text="Tolerance (mm):", font=FONT_LABEL,
                 bg=C["panel"], fg=C["text_dim"]).grid(row=2, column=0, sticky="w", pady=2)
        self._tol_var = tk.StringVar(value="20.0")
        tk.Entry(dim_frame, textvariable=self._tol_var, width=12,
                 font=FONT_MONO, bg=C["input_bg"], fg=C["input_fg"]).grid(row=2, column=1, padx=8, pady=2)

        # Evaluation result area
        self._eval_text = scrolledtext.ScrolledText(right_frame, bg=C["bg"], fg=C["text"],
                                                    font=FONT_MONO, height=12,
                                                    relief="flat", state="disabled",
                                                    highlightthickness=1,
                                                    highlightbackground=C["border"])
        self._eval_text.pack(fill="both", expand=True, pady=8)

        # Buttons for evaluation and auto-adjust
        action_frame = tk.Frame(right_frame, bg=C["bg"])
        action_frame.pack(pady=8)

        self._btn_evaluate = tk.Button(action_frame, text="📐 Evaluate", command=self._evaluate,
                                       font=FONT_BTN, bg=C["accent2"], fg="white", relief="flat",
                                       padx=12, pady=4)
        self._btn_evaluate.pack(side="left", padx=4)
        self._btn_evaluate.config(state="disabled")

        self._btn_auto = tk.Button(action_frame, text="🔄 Auto‑adjust", command=self._auto_adjust,
                                   font=FONT_BTN, bg=C["success"], fg="white", relief="flat",
                                   padx=12, pady=4)
        self._btn_auto.pack(side="left", padx=4)
        self._btn_auto.config(state="disabled")

        # Initially disable all interactive buttons
        self._set_interactive_state(False)

    def _set_interactive_state(self, enabled):
        state = "normal" if enabled else "disabled"
        self._btn_reset_points.config(state=state)
        self._btn_show3d.config(state=state)
        self._btn_evaluate.config(state=state)
        self._btn_auto.config(state=state)

    # -------------------------------------------------------------------------
    # Frame capture (PCIC)
    # -------------------------------------------------------------------------
    def _capture_frame(self):
        if self._cam_accessor() is None:
            self._log("Camera not connected. Connect first.", C["danger"])
            return

        ip = self._ip_var.get().strip()
        if not ip:
            self._log("Invalid IP address.", C["danger"])
            return

        # Read current resolution from camera
        try:
            cam = self._cam_accessor()
            cam.enter_edit()
            params = cam.get_imager_params()
            res_raw = params.get("Resolution", "1")
            self._current_resolution = int(res_raw)
            cam.exit_edit()
        except Exception as e:
            self._log(f"Could not read resolution: {e}", C["warning"])
            self._current_resolution = 1  # assume binned

        self._log(f"Capturing frame (resolution={'Full' if self._current_resolution==0 else 'Binned'})…", C["accent2"])
        threading.Thread(target=self._capture_thread, args=(ip,), daemon=True).start()

    def _capture_thread(self, ip):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10.0)
            sock.connect((ip, DEFAULT_PORT))

            # Upload schema
            sock.sendall(build_pcic_frame("0010", f"c{self.SCHEMA}"))
            read_pcic_frame(sock)   # ack

            # Enable streaming
            sock.sendall(build_pcic_frame("0011", "p1"))
            read_pcic_frame(sock)   # ack

            # Check current trigger mode
            cam = self._cam_accessor()
            cam.enter_edit()
            trig = cam.get_app_params().get("TriggerMode", "1")
            cam.exit_edit()
            trig_mode = int(trig)

            if trig_mode == 2:  # software trigger
                sock.sendall(build_pcic_frame("0012", "t"))
                time.sleep(0.1)  # allow camera to respond

            # Read one frame
            _, payload = read_pcic_frame(sock)
            images = parse_chunks(payload)
            sock.close()

            # Store images (ensure they are numpy arrays)
            self.after(0, lambda: self._on_frame_captured(images))

        except Exception as e:
            self.after(0, lambda: self._log(f"Capture failed: {e}", C["danger"]))

    def _on_frame_captured(self, images):
        if "amplitude" not in images:
            self._log("Frame captured but no amplitude image found.", C["danger"])
            return

        self._captured_images = images
        self._reset_points()          # clear previous points
        self._set_interactive_state(True)
        self._display_amplitude()
        self._log("Frame captured successfully. Click on the image to pick points.", C["success"])

    def _display_amplitude(self):
        """Show amplitude image on canvas with centre crosshair."""
        if "amplitude" not in self._captured_images:
            return

        amp = self._captured_images["amplitude"]
        # Normalize for display
        amp_norm = amp.astype(float)
        mx = amp_norm.max()
        if mx > 0:
            amp_norm = (amp_norm / mx) * 255
        else:
            amp_norm = amp_norm * 0
        amp_uint8 = np.clip(amp_norm, 0, 255).astype(np.uint8)

        # Convert to PIL Image, then to PhotoImage
        img = Image.fromarray(amp_uint8, mode="L")
        # Scale image to fit canvas (preserve aspect)
        canvas_width = self._canvas.winfo_width()
        canvas_height = self._canvas.winfo_height()
        if canvas_width < 10 or canvas_height < 10:
            canvas_width, canvas_height = 400, 300  # fallback
        img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)

        # Draw crosshair at centre of the image (in image coordinates)
        h, w = amp.shape
        cx = w // 2
        cy = h // 2
        # Convert image coordinates to canvas coordinates (scale may have changed)
        # We'll store the scale factor to map canvas clicks to image pixels
        img_w, img_h = img.size
        scale_x = w / img_w
        scale_y = h / img_h
        self._img_to_canvas_scale = (scale_x, scale_y)
        # Draw crosshair in canvas coordinates
        canvas_cx = cx / scale_x if scale_x else cx
        canvas_cy = cy / scale_y if scale_y else cy
        size = 20
        self._canvas.create_line(canvas_cx - size, canvas_cy, canvas_cx + size, canvas_cy,
                                 fill="red", width=2)
        self._canvas.create_line(canvas_cx, canvas_cy - size, canvas_cx, canvas_cy + size,
                                 fill="red", width=2)

        # Store for point picking
        self._amp_shape = (h, w)
        self._img_size = (img_w, img_h)

    def _on_canvas_click(self, event):
        if self._captured_images is None:
            return
        if len(self._points) >= 4:
            self._log("Already have 4 points. Reset to pick new points.", C["warning"])
            return

        # Convert canvas coordinates to image pixel coordinates
        x_canvas = event.x
        y_canvas = event.y
        if not hasattr(self, '_img_size'):
            return
        img_w, img_h = self._img_size
        if x_canvas < 0 or y_canvas < 0 or x_canvas >= img_w or y_canvas >= img_h:
            return  # outside image
        # Scale to original amplitude resolution
        h, w = self._amp_shape
        px = int(x_canvas * (w / img_w))
        py = int(y_canvas * (h / img_h))
        px = max(0, min(w-1, px))
        py = max(0, min(h-1, py))

        # Retrieve x,y,z from captured images
        x_img = self._captured_images.get("x")
        y_img = self._captured_images.get("y")
        z_img = self._captured_images.get("z")
        if x_img is None or y_img is None or z_img is None:
            self._log("X, Y, or Z images missing in captured frame.", C["danger"])
            return

        x_val = x_img[py, px]
        y_val = y_img[py, px]
        z_val = z_img[py, px]
        if z_val == 0:  # invalid
            self._log("Invalid point (Z=0). Pick a point with valid depth.", C["warning"])
            return

        self._points.append((float(x_val), float(y_val), float(z_val)))
        # Draw a circle on canvas at clicked position
        radius = 6
        self._point_markers.append(
            self._canvas.create_oval(x_canvas - radius, y_canvas - radius,
                                     x_canvas + radius, y_canvas + radius,
                                     outline=C["accent"], fill=None, width=2)
        )
        self._point_status.config(text=f"Points: {len(self._points)}/4")
        if len(self._points) == 4:
            self._log("4 points selected. Press 'Evaluate'.", C["success"])

    def _reset_points(self):
        self._points = []
        for marker in self._point_markers:
            self._canvas.delete(marker)
        self._point_markers = []
        self._point_status.config(text="Points: 0/4")

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    def _evaluate(self):
        if len(self._points) != 4:
            self._log("Need exactly 4 points to evaluate.", C["warning"])
            return

        try:
            exp_width = float(self._width_var.get())
            exp_height = float(self._height_var.get())
            tolerance = float(self._tol_var.get())
        except ValueError:
            self._log("Invalid numeric values in rectangle parameters.", C["danger"])
            return

        # Order points (we assume they were clicked in order around the rectangle)
        # For robustness, we can compute the convex hull and then distances between consecutive points.
        # Here we'll assume they are given in order (top-left, top-right, bottom-right, bottom-left).
        # We'll compute all 6 distances and check that 4 of them are close to exp_width/exp_height.
        p = self._points
        dists = []
        for i in range(4):
            dx = p[i][0] - p[(i+1)%4][0]
            dy = p[i][1] - p[(i+1)%4][1]
            dz = p[i][2] - p[(i+1)%4][2]
            dist = np.sqrt(dx*dx + dy*dy + dz*dz)
            dists.append(dist)
        # Also diagonals
        diag1 = np.linalg.norm(np.array(p[0]) - np.array(p[2]))
        diag2 = np.linalg.norm(np.array(p[1]) - np.array(p[3]))
        dists.extend([diag1, diag2])

        # Check if two distances match width and two match height (within tolerance)
        # We'll sort distances of sides (first 4)
        side_dists = sorted(dists[:4])
        # Expected side lengths: width, height, width, height in any order
        expected = sorted([exp_width, exp_height, exp_width, exp_height])
        rectangle_ok = True
        for a, b in zip(side_dists, expected):
            if abs(a - b) > tolerance:
                rectangle_ok = False
                break

        # Plane normal
        v1 = np.array(p[1]) - np.array(p[0])
        v2 = np.array(p[2]) - np.array(p[0])
        normal = np.cross(v1, v2)
        norm_n = np.linalg.norm(normal)
        if norm_n > 0:
            normal = normal / norm_n
        else:
            normal = np.array([0,0,1])
        # Ideal camera axis is (0,0,1) (Z forward)
        angle_to_z = np.arccos(np.clip(np.dot(normal, np.array([0,0,1])), -1, 1)) * 180 / np.pi
        perp_ok = angle_to_z < 10.0   # threshold 10 degrees

        # Flatness: standard deviation of Z values
        z_vals = [pt[2] for pt in p]
        z_std = np.std(z_vals)
        flatness_ok = z_std < tolerance   # same tolerance as rectangle

        # Build result text
        result = f"=== Evaluation Results ===\n"
        result += f"Rectangle dimensions:\n"
        for i, d in enumerate(dists[:4]):
            result += f"  Side {i+1}: {d:.1f} mm\n"
        result += f"  Expected: {exp_width} x {exp_height} mm ± {tolerance}\n"
        result += f"  Rectangle check: {'✓ PASS' if rectangle_ok else '✗ FAIL'}\n\n"
        result += f"Plane normal: ({normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f})\n"
        result += f"Angle from Z-axis: {angle_to_z:.1f}°\n"
        result += f"  Perpendicularity check: {'✓ PASS' if perp_ok else '✗ FAIL'}\n\n"
        result += f"Z values (mm): {', '.join(f'{z:.1f}' for z in z_vals)}\n"
        result += f"Z std dev: {z_std:.1f} mm\n"
        result += f"  Flatness check: {'✓ PASS' if flatness_ok else '✗ FAIL'}\n\n"

        overall = rectangle_ok and perp_ok and flatness_ok
        result += f"OVERALL: {'✓ Calibration OK' if overall else '✗ Needs adjustment'}\n"

        self._eval_text.config(state="normal")
        self._eval_text.delete("1.0", "end")
        self._eval_text.insert("end", result)
        self._eval_text.config(state="disabled")

        # Enable auto-adjust only if not OK
        if not overall:
            self._btn_auto.config(state="normal")
        else:
            self._btn_auto.config(state="disabled")

    # -------------------------------------------------------------------------
    # Auto-adjust extrinsic rotations
    # -------------------------------------------------------------------------
    def _auto_adjust(self):
        if len(self._points) != 4:
            self._log("Need 4 points to compute adjustment.", C["warning"])
            return

        # Compute plane normal again
        p = self._points
        v1 = np.array(p[1]) - np.array(p[0])
        v2 = np.array(p[2]) - np.array(p[0])
        normal = np.cross(v1, v2)
        norm_n = np.linalg.norm(normal)
        if norm_n > 0:
            normal = normal / norm_n
        else:
            normal = np.array([0,0,1])

        # Desired normal = (0,0,1)
        target = np.array([0,0,1])
        # Compute rotation that maps normal to target
        axis = np.cross(normal, target)
        dot = np.dot(normal, target)
        angle = np.arccos(np.clip(dot, -1, 1))
        if np.linalg.norm(axis) > 1e-6:
            axis = axis / np.linalg.norm(axis)
            # Rodrigues' rotation formula to get rotation matrix
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*K@K
        else:
            R = np.eye(3)

        # Convert rotation matrix to Euler angles (XYZ extrinsic order)
        # Assuming rotations are applied in order: Rx, Ry, Rz (extrinsic)
        # R = Rz(γ) * Ry(β) * Rx(α)
        # Compute α, β, γ (in degrees)
        if abs(R[2,0]) < 1-1e-6:
            β = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))
            α = np.arctan2(R[2,1]/np.cos(β), R[2,2]/np.cos(β))
            γ = np.arctan2(R[1,0]/np.cos(β), R[0,0]/np.cos(β))
        else:
            # Gimbal lock
            γ = 0
            if R[2,0] < -1+1e-6:
                β = np.pi/2
                α = np.arctan2(R[0,1], R[0,2])
            else:
                β = -np.pi/2
                α = np.arctan2(-R[0,1], -R[0,2])

        rx = np.degrees(α)
        ry = np.degrees(β)
        rz = np.degrees(γ)

        # Show proposed values to user
        msg = (f"Proposed extrinsic angles to align plane with camera:\n"
               f"RotX = {rx:.1f}°\n"
               f"RotY = {ry:.1f}°\n"
               f"RotZ = {rz:.1f}°\n\n"
               f"Apply these settings to the camera?")
        if messagebox.askyesno("Apply Calibration", msg):
            # Apply to camera
            threading.Thread(target=self._apply_rotation, args=(rx, ry, rz), daemon=True).start()

    def _apply_rotation(self, rx, ry, rz):
        cam = self._cam_accessor()
        if cam is None:
            self.after(0, lambda: self._log("Camera not connected.", C["danger"]))
            return
        try:
            cam.enter_edit()
            # Convert to string with 6 decimals (as camera expects)
            cam.set_device_param("ExtrinsicCalibRotX", f"{rx:.6f}")
            cam.set_device_param("ExtrinsicCalibRotY", f"{ry:.6f}")
            cam.set_device_param("ExtrinsicCalibRotZ", f"{rz:.6f}")
            cam.save_device()
            cam.exit_edit()
            self.after(0, lambda: self._log(f"Extrinsic angles set: RotX={rx:.1f}°, RotY={ry:.1f}°, RotZ={rz:.1f}°", C["success"]))
            # Optionally re-capture frame to verify
            self.after(1000, self._capture_frame)   # capture new frame after apply
        except Exception as e:
            self.after(0, lambda: self._log(f"Failed to set angles: {e}", C["danger"]))

    # -------------------------------------------------------------------------
    # 3D point cloud visualisation
    # -------------------------------------------------------------------------
    def _show_3d(self):
        if self._captured_images is None:
            return
        x_img = self._captured_images.get("x")
        y_img = self._captured_images.get("y")
        z_img = self._captured_images.get("z")
        if x_img is None or y_img is None or z_img is None:
            self._log("No XYZ data available.", C["danger"])
            return

        # Use only valid points (z != 0)
        mask = z_img != 0
        x_vals = x_img[mask]
        y_vals = y_img[mask]
        z_vals = z_img[mask]

        # Colour by amplitude if available
        amp_img = self._captured_images.get("amplitude")
        if amp_img is not None:
            colours = amp_img[mask]
        else:
            colours = None

        # Launch in a new thread to avoid blocking GUI
        def plot_3d():
            fig = plt.figure(figsize=(10,8), facecolor=C["bg"])
            ax = fig.add_subplot(111, projection='3d')
            # Downsample for performance if too many points
            step = max(1, len(x_vals) // 10000)
            sc = ax.scatter(x_vals[::step], y_vals[::step], z_vals[::step],
                            c=colours[::step] if colours is not None else None,
                            cmap="viridis", s=1, alpha=0.6)
            if colours is not None:
                plt.colorbar(sc, label="Amplitude")
            ax.set_xlabel("X (mm)", color=C["text_dim"])
            ax.set_ylabel("Y (mm)", color=C["text_dim"])
            ax.set_zlabel("Z (mm)", color=C["text_dim"])
            ax.set_title("3D Point Cloud", color=C["accent"])
            plt.show()

        threading.Thread(target=plot_3d, daemon=True).start()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
class CalibrationAssistant(tk.Tk):
    # How long to wait after a widget change before auto-sending (ms)
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
        self._live_win      = None    # reference to open LiveStreamWindow
        self._rt_job        = None    # debounce job for real-time apply

        self._build_ui()
        self._schedule_heartbeat()

    # ─────────────────────────────────────────────────────────────────
    # UI BUILD
    # ─────────────────────────────────────────────────────────────────
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

        # Real-time toggle
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

        # Tab notebook (camera params tabs + PLC tab + Calibration tab)
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
        """Build the scrollable parameter panels + right log panel."""
        content = tk.Frame(parent, bg=C["bg"])
        content.pack(fill="both", expand=True, padx=12, pady=8)

        left = tk.Frame(content, bg=C["bg"])
        left.pack(side="left", fill="both", expand=True)

        right = tk.Frame(content, bg=C["bg"], width=280)
        right.pack(side="right", fill="y", padx=(10, 0))
        right.pack_propagate(False)

        self._build_param_panels(left)
        self._build_right_panel(right)

        tk.Frame(parent, bg=C["border"], height=1).pack(fill="x")
        self._build_action_bar(parent)

    def _build_param_panels(self, parent):
        canvas = tk.Canvas(parent, bg=C["bg"], highlightthickness=0)
        scroll = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._param_frame = tk.Frame(canvas, bg=C["bg"])
        cw = canvas.create_window((0, 0), window=self._param_frame, anchor="nw")

        self._param_frame.bind("<Configure>",
                               lambda e: canvas.configure(
                                   scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(cw, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        self._build_section("ACQUISITION & TRIGGER", [
            ("TriggerMode",       "Trigger Mode",       "combo",
             ["1 — Free Run", "2 — Software (PCIC t)", "3 — HW Rising Edge", "4 — HW Falling Edge"],
             "Controls what causes the camera to capture a frame.\n"
             "Free Run = continuous, Software = Python sends 't' command,\n"
             "Hardware = voltage pulse on Pin 2."),
            ("Resolution",        "Resolution",         "combo",
             ["0 — Full  352×264", "1 — Binned 176×132"],
             "Image resolution mode.\n"
             "Full (0): 352×264 pixels — max detail, larger data, slower.\n"
             "Binned (1): 176×132 — 2×2 binning, 4× less data, higher FPS.\n"
             "⚠ Changing resolution requires reconnecting the PCIC stream."),
            ("ExposureTime",      "Exposure Time (µs)", "int",   None,
             "ToF integration time per frame.\n"
             "Higher = better SNR, slower max FPS, more heat.\n"
             "Range: 50 – 10 000 µs.  Typical: 2600 µs."),
            ("FrameRate",         "Frame Rate (fps)",   "float", None,
             "Frames per second in Free Run mode.\n"
             "Ignored in trigger modes. Max depends on exposure time."),
            ("Channel",           "Frequency Channel",  "combo",
             ["0", "1", "2", "3"],
             "Modulation frequency channel — set different values on\n"
             "cameras within range of each other to avoid interference."),
        ])

        self._build_section("IMAGE QUALITY & FILTERING", [
            ("MinimumAmplitude",  "Minimum Amplitude",  "int",   None,
             "Pixels below this amplitude are marked INVALID.\n"
             "Too high = valid points lost. Too low = noisy phantoms.\n"
             "Range: 0 – 65535.  Typical: 42."),
            ("SymmetryThreshold", "Symmetry Threshold", "float", None,
             "Rejects pixels with asymmetric ToF waveform.\n"
             "0 = accept all, 1 = strictest.  Typical: 0.4."),
            ("SpatialFilterType", "Spatial Filter",     "combo",
             ["0 — Off", "1 — Median", "2 — Mean", "3 — Bilateral"],
             "Per-frame pixel smoothing.\n"
             "Median: removes salt-and-pepper noise.\n"
             "Bilateral: smooths while preserving edges."),
            ("TemporalFilterType","Temporal Filter",    "combo",
             ["0 — Off", "1 — Mean", "2 — Adaptive Exponential"],
             "Smoothing across consecutive frames.\n"
             "Reduces flicker on static scenes; introduces motion blur."),
        ])

        self._build_section("REGION OF INTEREST — 2D PIXEL CLIPPING", [
            ("ClippingLeft",   "Left   (px)", "int", None, "Left pixel boundary.   Range: 0 – 175."),
            ("ClippingRight",  "Right  (px)", "int", None, "Right pixel boundary.  Range: 0 – 175."),
            ("ClippingTop",    "Top    (px)", "int", None, "Top pixel boundary.    Range: 0 – 131."),
            ("ClippingBottom", "Bottom (px)", "int", None, "Bottom pixel boundary. Range: 0 – 131."),
        ])

        self._build_section("REGION OF INTEREST — 3D SPATIAL CLIPPING (mm)", [
            ("ClipZMin", "Z Min  (mm)", "float", None, "Min depth — closer points discarded."),
            ("ClipZMax", "Z Max  (mm)", "float", None, "Max depth — farther points discarded."),
            ("ClipXMin", "X Min  (mm)", "float", None, "Left 3D boundary  (typically negative)."),
            ("ClipXMax", "X Max  (mm)", "float", None, "Right 3D boundary."),
            ("ClipYMin", "Y Min  (mm)", "float", None, "Top 3D boundary   (typically negative)."),
            ("ClipYMax", "Y Max  (mm)", "float", None, "Bottom 3D boundary."),
        ])

        self._build_section("EXTRINSIC CALIBRATION (world frame offset)", [
            ("ExtrinsicCalibTransX", "Trans X (mm)", "float", None, "Camera X offset from world origin."),
            ("ExtrinsicCalibTransY", "Trans Y (mm)", "float", None, "Camera Y offset from world origin."),
            ("ExtrinsicCalibTransZ", "Trans Z (mm)", "float", None, "Camera height above reference plane."),
            ("ExtrinsicCalibRotX",   "Rot X   (°)",  "float", None, "Pitch (rotation around X)."),
            ("ExtrinsicCalibRotY",   "Rot Y   (°)",  "float", None, "Roll  (rotation around Y)."),
            ("ExtrinsicCalibRotZ",   "Rot Z   (°)",  "float", None, "Yaw   (rotation around Z)."),
        ])

        self._build_section("NETWORK", [
            ("IPAddress",  "IP Address",  "str", None, "Camera IPv4 address. Change requires reboot."),
            ("SubnetMask", "Subnet Mask", "str", None, "Network subnet mask."),
            ("Gateway",    "Gateway",     "str", None, "Default network gateway."),
        ])

    def _build_section(self, title, params):
        wrapper = tk.Frame(self._param_frame, bg=C["bg"])
        wrapper.pack(fill="x", pady=(0, 2))

        hdr = tk.Frame(wrapper, bg=C["panel"], pady=6)
        hdr.pack(fill="x")
        tk.Label(hdr, text=f"  ▸ {title}", font=FONT_HEADER,
                 bg=C["panel"], fg=C["accent"]).pack(side="left", padx=8)

        body = tk.Frame(wrapper, bg=C["bg"], pady=4)
        body.pack(fill="x")

        for key, label, kind, choices, tooltip in params:
            row = tk.Frame(body, bg=C["bg"])
            row.pack(fill="x", padx=8, pady=2)

            lbl_frame = tk.Frame(row, bg=C["bg"], width=180)
            lbl_frame.pack(side="left")
            lbl_frame.pack_propagate(False)

            tk.Label(lbl_frame, text=label, font=FONT_LABEL,
                     bg=C["bg"], fg=C["text"], anchor="w").pack(side="left")
            tip_btn = tk.Label(lbl_frame, text=" ⓘ", font=FONT_LABEL,
                                bg=C["bg"], fg=C["text_dim"], cursor="question_arrow")
            tip_btn.pack(side="left")
            self._add_tooltip(tip_btn, tooltip)

            if kind == "combo":
                var = tk.StringVar()
                w   = ttk.Combobox(row, textvariable=var, values=choices,
                                   state="readonly", width=28, font=FONT_MONO)
                w.pack(side="left", padx=8)
                # Real-time: bind on selection change
                var.trace_add("write", lambda *_, k=key: self._on_widget_change(k))
                self._widgets[key] = (w, var, "combo")
            else:
                var = tk.StringVar(value="—")
                w   = tk.Entry(row, textvariable=var, width=20,
                               font=FONT_MONO, bg=C["input_bg"], fg=C["input_fg"],
                               insertbackground=C["accent"], relief="flat",
                               highlightthickness=1, highlightbackground=C["border"],
                               highlightcolor=C["accent"])
                w.pack(side="left", padx=8)
                # Real-time: bind on key release
                w.bind("<KeyRelease>", lambda e, k=key: self._on_widget_change(k))
                self._widgets[key] = (w, var, kind)

            cur = tk.Label(row, text="current: —", font=FONT_LABEL,
                           bg=C["bg"], fg=C["text_dim"])
            cur.pack(side="left", padx=4)
            self._widgets[f"{key}__cur"] = cur

    def _build_right_panel(self, parent):
        tk.Label(parent, text="  SYSTEM LOG", font=FONT_HEADER,
                 bg=C["bg"], fg=C["accent"]).pack(anchor="w", pady=(0, 4))

        self._log = scrolledtext.ScrolledText(
            parent, bg=C["panel"], fg=C["text_dim"], font=FONT_LOG,
            relief="flat", wrap="word", state="disabled",
            highlightthickness=1, highlightbackground=C["border"])
        self._log.pack(fill="both", expand=True)

        tk.Frame(parent, bg=C["border"], height=1).pack(fill="x", pady=8)

        tk.Label(parent, text="  CAMERA STATUS", font=FONT_HEADER,
                 bg=C["bg"], fg=C["accent"]).pack(anchor="w")

        status_frame = tk.Frame(parent, bg=C["panel"], pady=8, padx=8)
        status_frame.pack(fill="x", pady=4)

        self._status_items = {}
        for label in ["Connection", "Session", "Trigger Mode",
                       "Active Application", "Firmware"]:
            row = tk.Frame(status_frame, bg=C["panel"])
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{label}:", font=FONT_LABEL,
                     bg=C["panel"], fg=C["text_dim"],
                     width=18, anchor="w").pack(side="left")
            val = tk.Label(row, text="—", font=FONT_VALUE,
                           bg=C["panel"], fg=C["text"])
            val.pack(side="left")
            self._status_items[label] = val

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
        self._btn_test = self._make_btn(
            btn_frame, "▶ TEST PROCESSING APP", self._launch_processing_app,
            "#7c3aed", side="left", padx=8)

        for b in [self._btn_read, self._btn_apply, self._btn_visualise, self._btn_test]:
            b.config(state="disabled")

    # ─────────────────────────────────────────────────────────────────
    # REAL-TIME PARAMETER PREVIEW
    # ─────────────────────────────────────────────────────────────────
    def _on_widget_change(self, key: str):
        """
        Called on every widget change. Starts a debounce timer;
        when it fires, the changed parameter is sent to the camera
        and the live stream (if open) gets the next frame automatically.
        """
        if not self._connected or not self._rt_var.get():
            return
        # Cancel any pending debounce
        if self._rt_job:
            self.after_cancel(self._rt_job)
        self._rt_job = self.after(
            self.REALTIME_DEBOUNCE_MS,
            lambda: self._realtime_apply(key),
        )

    def _realtime_apply(self, key: str):
        """Send a single parameter to the camera without a full save."""
        self._rt_job = None
        if not self._connected:
            return

        entry = self._widgets.get(key)
        if entry is None or isinstance(entry, tk.Label):
            return

        w, var, kind = entry
        raw = var.get().strip()
        if raw in ("", "—"):
            return

        COMBO_EXTRACT = {
            "TriggerMode":        lambda v: v.split("—")[0].strip(),
            "Resolution":         lambda v: v.split("—")[0].strip(),
            "Channel":            lambda v: v.split("—")[0].strip(),
            "SpatialFilterType":  lambda v: v.split("—")[0].strip(),
            "TemporalFilterType": lambda v: v.split("—")[0].strip(),
        }
        if kind == "combo" and key in COMBO_EXTRACT:
            raw = COMBO_EXTRACT[key](raw)

        def _send():
            try:
                APP_KEYS    = {"TriggerMode"}
                IMG_KEYS    = {"ExposureTime","FrameRate","Channel","Resolution",
                               "MinimumAmplitude","SymmetryThreshold",
                               "ClippingLeft","ClippingRight","ClippingTop","ClippingBottom"}
                FILTER_SPATIAL  = {"SpatialFilterType"}
                FILTER_TEMPORAL = {"TemporalFilterType"}
                EXT_KEYS    = {"ExtrinsicCalibTransX","ExtrinsicCalibTransY",
                               "ExtrinsicCalibTransZ","ExtrinsicCalibRotX",
                               "ExtrinsicCalibRotY","ExtrinsicCalibRotZ"}

                if key in APP_KEYS:
                    self._cam.set_app_param(key, raw)
                elif key in IMG_KEYS:
                    self._cam.set_imager_param(key, raw)
                elif key in FILTER_SPATIAL:
                    self._cam.set_spatial_filter(key, raw)
                elif key in FILTER_TEMPORAL:
                    self._cam.set_temporal_filter(key, raw)
                elif key in EXT_KEYS:
                    self._cam.set_device_param(key, raw)
                else:
                    return   # network params — never send in real-time

                self.after(0, lambda: self._log_write(
                    f"  ↻ real-time: {key} = {raw}", C["text_dim"]))

                # Update current-value label
                cur_key = f"{key}__cur"
                if cur_key in self._widgets:
                    self.after(0, lambda: self._widgets[cur_key].config(
                        text=f"current: {raw}", fg=C["success"]))

            except Exception as exc:
                self.after(0, lambda: self._log_write(
                    f"  real-time error ({key}): {exc}", C["danger"]))

        threading.Thread(target=_send, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────
    def _make_btn(self, parent, text, cmd, color, side="left", padx=4, pady=4):
        btn = tk.Button(parent, text=text, command=cmd, font=FONT_BTN,
                        bg=color, fg="white", relief="flat",
                        padx=14, pady=6, cursor="hand2",
                        activebackground=color, activeforeground="white")
        btn.pack(side=side, padx=padx, pady=pady)
        btn.bind("<Enter>", lambda e: btn.config(bg=self._lighten(color)))
        btn.bind("<Leave>", lambda e: btn.config(bg=color))
        return btn

    @staticmethod
    def _lighten(hex_color):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"#{min(255,r+30):02x}{min(255,g+30):02x}{min(255,b+30):02x}"

    def _log_write(self, msg, color=None):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log.config(state="normal")
        self._log.insert("end", f"[{ts}] {msg}\n")
        if color:
            start = self._log.index("end-2l")
            end   = self._log.index("end-1l")
            tag   = f"c{color.replace('#','')}"
            self._log.tag_configure(tag, foreground=color)
            self._log.tag_add(tag, start, end)
        self._log.config(state="disabled")
        self._log.see("end")

    def _add_tooltip(self, widget, text):
        tip = None
        def show(e):
            nonlocal tip
            tip = tk.Toplevel(widget)
            tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{e.x_root+12}+{e.y_root+12}")
            tk.Label(tip, text=text, justify="left", font=FONT_LOG,
                     bg="#1c2128", fg=C["text"], relief="flat",
                     padx=8, pady=6, wraplength=320).pack()
        def hide(e):
            nonlocal tip
            if tip:
                tip.destroy()
                tip = None
        widget.bind("<Enter>", show)
        widget.bind("<Leave>", hide)

    def _set_status(self, key, value, color=None):
        if key in self._status_items:
            self._status_items[key].config(text=value, fg=color or C["text"])

    def _set_buttons_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for b in [self._btn_read, self._btn_apply, self._btn_visualise, self._btn_test]:
            b.config(state=state)
        self._btn_connect.config(state="disabled" if enabled else "normal")
        self._btn_disconnect.config(state="normal" if enabled else "disabled")

    # ─────────────────────────────────────────────────────────────────
    # CONNECTION
    # ─────────────────────────────────────────────────────────────────
    def _connect(self):
        ip = self._ip_var.get().strip()
        self._log_write(f"Connecting to {ip}…", C["warning"])
        threading.Thread(target=self._connect_thread, args=(ip,), daemon=True).start()

    def _connect_thread(self, ip):
        try:
            self._cam = CameraXMLRPC(ip)
            sid = self._cam.connect()
            self._cam.enter_edit()
            self._connected = True
            self.after(0, lambda: self._on_connected(ip, sid))
            self._read_all_thread()
        except Exception as exc:
            self.after(0, lambda: self._log_write(f"Connection failed: {exc}", C["danger"]))

    def _on_connected(self, ip, sid):
        self._lbl_conn.config(text=f"⬤  {ip}", fg=C["success"])
        self._set_status("Connection", "OK", C["success"])
        self._set_status("Session", sid[:12] + "…")
        self._set_buttons_enabled(True)
        self._log_write(f"Connected — session {sid[:12]}…", C["success"])

    def _disconnect(self):
        if self._cam:
            try:
                self._cam.exit_edit()
                self._cam.disconnect()
            except Exception:
                pass
        self._connected = False
        self._cam = None
        self._lbl_conn.config(text="⬤  DISCONNECTED", fg=C["danger"])
        self._set_buttons_enabled(False)
        self._set_status("Connection", "Disconnected", C["danger"])
        self._set_status("Session", "—")
        self._log_write("Disconnected.", C["text_dim"])

    # ─────────────────────────────────────────────────────────────────
    # READ ALL PARAMETERS
    # ─────────────────────────────────────────────────────────────────
    def _read_all(self):
        self._log_write("Reading all parameters from camera…", C["accent2"])
        threading.Thread(target=self._read_all_thread, daemon=True).start()

    def _read_all_thread(self):
        try:
            dev  = self._cam.get_device_params()
            net  = self._cam.get_network_params()
            app  = self._cam.get_app_params()
            img  = self._cam.get_imager_params()
            spat = self._cam.get_spatial_filter()
            temp = self._cam.get_temporal_filter()

            all_params = {}
            all_params.update(dev)
            all_params.update(net)
            all_params.update(app)
            all_params.update(img)
            all_params.update({f"Spatial_{k}": v for k, v in spat.items()})
            all_params.update({f"Temporal_{k}": v for k, v in temp.items()})

            # Unpack ClippingCuboid JSON
            try:
                cuboid = json.loads(img.get("ClippingCuboid", "{}"))
                all_params["ClipXMin"] = cuboid.get("XMin", -3.4e38)
                all_params["ClipXMax"] = cuboid.get("XMax",  3.4e38)
                all_params["ClipYMin"] = cuboid.get("YMin", -3.4e38)
                all_params["ClipYMax"] = cuboid.get("YMax",  3.4e38)
                all_params["ClipZMin"] = cuboid.get("ZMin", -3.4e38)
                all_params["ClipZMax"] = cuboid.get("ZMax",  3.4e38)
            except Exception:
                pass

            all_params["SpatialFilterType"]  = spat.get("SpatialFilterType",  "0")
            all_params["TemporalFilterType"] = temp.get("TemporalFilterType", "0")

            self._params = all_params
            self.after(0, lambda: self._populate_widgets(all_params))

            t1, t2, til = self._cam.get_temperatures()
            if t1 is not None:
                self.after(0, lambda: self._lbl_temp.config(
                    text=f"  T1: {t1:.1f}°C   T2: {t2:.1f}°C   Illu: {til:.1f}°C"))

            self.after(0, lambda: self._set_status(
                "Trigger Mode",
                {"1": "Free Run", "2": "Software",
                 "3": "HW Rising", "4": "HW Falling"}.get(
                    str(app.get("TriggerMode", "?")), "?")))
            self.after(0, lambda: self._set_status(
                "Active Application", str(dev.get("ActiveApplication", "?"))))
            self.after(0, lambda: self._log_write(
                f"Read {len(all_params)} parameters OK", C["success"]))

        except Exception as exc:
            self.after(0, lambda: self._log_write(f"Read error: {exc}", C["danger"]))

    def _populate_widgets(self, params):
        COMBO_MAP = {
            "TriggerMode":        {"1": "1 — Free Run", "2": "2 — Software (PCIC t)",
                                   "3": "3 — HW Rising Edge", "4": "4 — HW Falling Edge"},
            "Resolution":         {"0": "0 — Full  352×264", "1": "1 — Binned 176×132"},
            "Channel":            {"0": "0", "1": "1", "2": "2", "3": "3"},
            "SpatialFilterType":  {"0": "0 — Off", "1": "1 — Median",
                                   "2": "2 — Mean", "3": "3 — Bilateral"},
            "TemporalFilterType": {"0": "0 — Off", "1": "1 — Mean",
                                   "2": "2 — Adaptive Exponential"},
        }

        for key, entry in self._widgets.items():
            if key.endswith("__cur"):
                base = key.replace("__cur", "")
                raw  = params.get(base, "—")
                entry.config(text=f"current: {raw}", fg=C["text_dim"])
            else:
                w, var, kind = entry
                raw = params.get(key, "")
                if raw == "" or raw is None:
                    continue
                if kind == "combo":
                    display = COMBO_MAP.get(key, {}).get(str(raw), str(raw))
                    var.set(display)
                else:
                    try:
                        fval = float(raw)
                        if abs(fval) > 1e10:
                            var.set("—")
                        elif fval == int(fval):
                            var.set(str(int(fval)))
                        else:
                            var.set(f"{fval:.2f}")
                    except (ValueError, TypeError):
                        var.set(str(raw))

    # ─────────────────────────────────────────────────────────────────
    # APPLY ALL CHANGES
    # ─────────────────────────────────────────────────────────────────
    def _apply_changes(self):
        self._log_write("Applying all changes to camera…", C["warning"])
        threading.Thread(target=self._apply_thread, daemon=True).start()

    def _apply_thread(self):
        try:
            COMBO_EXTRACT = {
                "TriggerMode":        lambda v: v.split("—")[0].strip(),
                "Resolution":         lambda v: v.split("—")[0].strip(),
                "Channel":            lambda v: v.split("—")[0].strip(),
                "SpatialFilterType":  lambda v: v.split("—")[0].strip(),
                "TemporalFilterType": lambda v: v.split("—")[0].strip(),
            }

            values = {}
            for key, entry in self._widgets.items():
                if key.endswith("__cur"):
                    continue
                w, var, kind = entry
                raw = var.get().strip()
                if raw in ("", "—"):
                    continue
                if kind == "combo" and key in COMBO_EXTRACT:
                    raw = COMBO_EXTRACT[key](raw)
                values[key] = raw

            APP_KEYS = {"TriggerMode"}
            IMG_KEYS = {"ExposureTime","FrameRate","Channel","Resolution",
                        "MinimumAmplitude","SymmetryThreshold",
                        "ClippingLeft","ClippingRight","ClippingTop","ClippingBottom"}
            EXT_KEYS = {"ExtrinsicCalibTransX","ExtrinsicCalibTransY","ExtrinsicCalibTransZ",
                        "ExtrinsicCalibRotX","ExtrinsicCalibRotY","ExtrinsicCalibRotZ"}
            NET_KEYS = {"IPAddress","SubnetMask","Gateway"}

            for k in APP_KEYS:
                if k in values:
                    self._cam.set_app_param(k, values[k])
                    self.after(0, lambda kk=k, vv=values[k]:
                               self._log_write(f"  app.{kk} = {vv}"))

            for k in IMG_KEYS:
                if k in values:
                    self._cam.set_imager_param(k, values[k])
                    self.after(0, lambda kk=k, vv=values[k]:
                               self._log_write(f"  imager.{kk} = {vv}"))

            if "Resolution" in values:
                res  = values["Resolution"]
                dims = "352×264" if res == "0" else "176×132"
                old  = str(self._params.get("Resolution", "1"))
                if res != old:
                    self.after(0, lambda d=dims: self._log_write(
                        f"⚠  Resolution changed → image is now {d}px.\n"
                        f"   Reconnect the PCIC stream and update\n"
                        f"   imageWidth/imageHeight in your code.", C["warning"]))

            # 3D cuboid
            cuboid_keys = {"ClipXMin","ClipXMax","ClipYMin","ClipYMax","ClipZMin","ClipZMax"}
            if any(k in values for k in cuboid_keys):
                def _v(k, default):
                    try:
                        return float(values.get(k, default))
                    except Exception:
                        return float(default)
                cuboid = {
                    "XMin": _v("ClipXMin", -3.402823e+38),
                    "XMax": _v("ClipXMax",  3.402823e+38),
                    "YMin": _v("ClipYMin", -3.402823e+38),
                    "YMax": _v("ClipYMax",  3.402823e+38),
                    "ZMin": _v("ClipZMin", -3.402823e+38),
                    "ZMax": _v("ClipZMax",  3.402823e+38),
                }
                self._cam.set_imager_param("ClippingCuboid", json.dumps(cuboid))
                self.after(0, lambda: self._log_write("  imager.ClippingCuboid updated"))

            if "SpatialFilterType" in values:
                self._cam.set_spatial_filter("SpatialFilterType", values["SpatialFilterType"])
                self.after(0, lambda v=values["SpatialFilterType"]:
                           self._log_write(f"  spatialFilter.type = {v}"))

            if "TemporalFilterType" in values:
                self._cam.set_temporal_filter("TemporalFilterType", values["TemporalFilterType"])
                self.after(0, lambda v=values["TemporalFilterType"]:
                           self._log_write(f"  temporalFilter.type = {v}"))

            for k in EXT_KEYS:
                if k in values:
                    self._cam.set_device_param(k, values[k])
                    self.after(0, lambda kk=k, vv=values[k]:
                               self._log_write(f"  device.{kk} = {vv}"))

            net_changed = any(
                values.get(k) and values.get(k) != str(self._params.get(k, ""))
                for k in NET_KEYS)
            if net_changed:
                for k in NET_KEYS:
                    if k in values:
                        self._cam.set_network_param(k, values[k])
                self.after(0, lambda: self._log_write(
                    "Network params changed — reboot required!", C["warning"]))

            self._cam.save_app()
            self._cam.save_device()
            self.after(0, lambda: self._log_write(
                "✓ All changes applied and saved.", C["success"]))

            time.sleep(0.5)
            self._read_all_thread()

        except Exception as exc:
            self.after(0, lambda: self._log_write(f"Apply error: {exc}", C["danger"]))

    # ─────────────────────────────────────────────────────────────────
    # LIVE STREAM
    # ─────────────────────────────────────────────────────────────────
    def _open_live_stream(self):
        ip = self._ip_var.get().strip()

        w, var, _ = self._widgets.get("TriggerMode", (None, None, None))
        trigger_raw  = var.get() if var else "1"
        trigger_mode = trigger_raw.split("—")[0].strip() if "—" in trigger_raw else trigger_raw
        try:
            trigger_mode = int(trigger_mode)
        except ValueError:
            trigger_mode = 1

        w2, var2, _ = self._widgets.get("Resolution", (None, None, None))
        res_raw  = var2.get() if var2 else "1"
        res_mode = res_raw.split("—")[0].strip() if "—" in res_raw else res_raw
        try:
            res_mode = int(res_mode)
        except ValueError:
            res_mode = 1

        dims = "176×132" if res_mode == 1 else "352×264"
        self._log_write(
            f"Opening live stream — trigger={trigger_mode}, resolution={dims}",
            C["success"])

        # Keep a reference so we can detect if it's still open
        if self._live_win is not None:
            try:
                self._live_win._stop()
            except Exception:
                pass
        self._live_win = LiveStreamWindow(self, ip, trigger_mode, resolution=res_mode)

    # ─────────────────────────────────────────────────────────────────
    # LAUNCH PROCESSING APP
    # ─────────────────────────────────────────────────────────────────
    def _launch_processing_app(self):
        candidates = [
            "SnapAndSave_HWTrigger.py",
            "SnapAndSave_Triggered.py",
            "SnapAndSave.py",
        ]
        script_dir = os.path.dirname(os.path.abspath(__file__))
        found = next(
            (os.path.join(script_dir, n) for n in candidates
             if os.path.exists(os.path.join(script_dir, n))),
            None,
        )
        if found is None:
            found = filedialog.askopenfilename(
                title="Select Processing Application",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")])
            if not found:
                return

        self._log_write(f"Launching: {os.path.basename(found)}", C["success"])
        try:
            subprocess.Popen(
                [sys.executable, found],
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
            )
            self._log_write("Processing app launched in new window.", C["text_dim"])
        except Exception as exc:
            self._log_write(f"Launch error: {exc}", C["danger"])

    # ─────────────────────────────────────────────────────────────────
    # HEARTBEAT
    # ─────────────────────────────────────────────────────────────────
    def _schedule_heartbeat(self):
        if self._connected and self._cam:   # guard: cam may be None on first call
            self._cam.heartbeat(30)
        self._hb_job = self.after(20_000, self._schedule_heartbeat)

    def on_close(self):
        if self._hb_job:
            self.after_cancel(self._hb_job)
        self._disconnect()
        self.destroy()


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