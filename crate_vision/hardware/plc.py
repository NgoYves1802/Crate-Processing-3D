"""
crate_vision/hardware/plc.py
=============================
S7-1200 ISO-on-TCP client via Snap7.

Architecture
------------
  - PC is always the CLIENT — connects to the PLC on port 102
  - PLC is always the SERVER — always listening, no TCON block needed
  - No background threads, no accept loops, no raw sockets
  - PC writes directly into a DB using Snap7

DB1 "CrateData" layout (must match TIA Portal exactly):
  Offset  Size  Type   Name
  0       2     INT    crate_count
  2       2     INT    snap_index
  ── Array[1..MAX_CRATES] of Crate_Type at offset 4 ──
  Each crate = CRATE_SIZE bytes (see constants below).
"""

from __future__ import annotations

import struct
import threading
from datetime import datetime

try:
    import snap7
    import snap7.util
    _SNAP7_AVAILABLE = True
except ImportError:
    _SNAP7_AVAILABLE = False


# ---------------------------------------------------------------------------
# DB layout constants
# ---------------------------------------------------------------------------

MAX_CRATES   = 4
HEADER_SIZE  = 4    # crate_count INT (2B) + snap_index INT (2B)
CRATE_SIZE   = 22   # bytes per crate record
DB_TOTAL     = HEADER_SIZE + CRATE_SIZE * MAX_CRATES   # 92 bytes


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")


# ---------------------------------------------------------------------------
# PLCClient
# ---------------------------------------------------------------------------

class PLCClient:
    """
    ISO-on-TCP client that writes crate detection results directly into
    the S7-1200 DB using Snap7.

    If the connection drops, write_result() automatically reconnects.
    No PLC programming is needed for the communication layer — just read
    the DB variables in normal OB1/FB logic.

    Parameters
    ----------
    ip   : PLC IP address
    rack : S7-1200 rack number (always 0)
    slot : S7-1200 slot number (always 1)
    db   : DB number in TIA Portal (e.g. 2)
    """

    def __init__(self, ip: str, rack: int, slot: int, db: int):
        if not _SNAP7_AVAILABLE:
            raise ImportError("python-snap7 is not installed.")
        self.ip   = ip
        self.rack = rack
        self.slot = slot
        self.db   = db

        self._client       = snap7.client.Client()
        self._lock         = threading.Lock()
        self._snap_counter = 0

    # ── Connection management ─────────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to PLC. Returns True on success."""
        try:
            self._client.connect(self.ip, self.rack, self.slot)
            _log(f"PLC ◂ connected to {self.ip} (ISO-on-TCP port 102) ✓")
            return True
        except Exception as exc:
            _log(f"PLC ⚠  connection failed: {exc}")
            return False

    def disconnect(self) -> None:
        try:
            self._client.disconnect()
            _log("PLC ◂ disconnected.")
        except Exception:
            pass

    @property
    def connected(self) -> bool:
        try:
            return bool(self._client.get_connected())
        except Exception:
            return False

    def _ensure_connected(self) -> bool:
        if not self.connected:
            _log("PLC ⚠  connection lost — reconnecting...")
            return self.connect()
        return True

    # ── DB buffer builder ─────────────────────────────────────────────────────

    @staticmethod
    def _pack_s_flags(crate: dict) -> int:
        """Pack S1–S12 booleans into a 16-bit WORD. S1=bit0, S12=bit11."""
        word = 0
        for i in range(1, 13):
            if crate.get(f"S{i}", False):
                word |= (1 << (i - 1))
        return word

    def _build_db_buffer(self, result: dict) -> bytearray:
        """
        Convert detection result into a flat bytearray matching DB layout.
        """
        crates      = result.get("crates", [])
        crate_count = min(len(crates), MAX_CRATES)

        self._snap_counter = (self._snap_counter + 1) % 32767
        snap_index = self._snap_counter

        buf = bytearray(DB_TOTAL)
        struct.pack_into(">h", buf, 0, crate_count)
        struct.pack_into(">h", buf, 2, snap_index)

        for idx, crate in enumerate(crates[:MAX_CRATES]):
            base = HEADER_SIZE + idx * CRATE_SIZE
            struct.pack_into(">h", buf, base + 0,  int(crate.get("crate_number", idx + 1)))
            struct.pack_into(">f", buf, base + 2,  float(crate.get("Rx",    0.0)))
            struct.pack_into(">f", buf, base + 6,  float(crate.get("Ry",    0.0)))
            struct.pack_into(">f", buf, base + 10, float(crate.get("Rz",    0.0)))
            struct.pack_into(">f", buf, base + 14, float(crate.get("theta", 0.0)))
            struct.pack_into(">H", buf, base + 18, self._pack_s_flags(crate))
            struct.pack_into(">B", buf, base + 20, 1 if crate.get("ai_classification") else 0)
            # byte +21 = padding, stays 0

        return buf

    # ── Main write method ─────────────────────────────────────────────────────

    def write_result(self, result: dict) -> bool:
        """
        Write detection result into the PLC DB via ISO-on-TCP.

        Returns True if written successfully, False on error.
        """
        with self._lock:
            if not self._ensure_connected():
                _log("PLC ⚠  cannot write — not connected")
                return False
            try:
                buf = self._build_db_buffer(result)
                self._client.db_write(self.db, 0, buf)
                n = len(result.get("crates", []))
                _log(
                    f"PLC ▸ DB{self.db} written ✓  "
                    f"({n} crates, {len(buf)} bytes, snap_id={result.get('snap_id')})"
                )
                return True
            except Exception as exc:
                _log(f"PLC ⚠  db_write failed: {exc}")
                return False


# ---------------------------------------------------------------------------
# Debug readback
# ---------------------------------------------------------------------------

def readback_db(plc_client: PLCClient) -> None:
    """Read the DB back from the PLC and print decoded values (debug only)."""
    try:
        buf = plc_client._client.db_read(plc_client.db, 0, DB_TOTAL)
        crate_count = struct.unpack_from(">h", buf, 0)[0]
        snap_index  = struct.unpack_from(">h", buf, 2)[0]
        _log(f"  DB{plc_client.db} readback: crate_count={crate_count}  snap_index={snap_index}")

        for i in range(min(crate_count, MAX_CRATES)):
            base         = HEADER_SIZE + i * CRATE_SIZE
            crate_number = struct.unpack_from(">h", buf, base + 0)[0]
            Rx    = struct.unpack_from(">f", buf, base + 2)[0]
            Ry    = struct.unpack_from(">f", buf, base + 6)[0]
            Rz    = struct.unpack_from(">f", buf, base + 10)[0]
            theta = struct.unpack_from(">f", buf, base + 14)[0]
            flags = struct.unpack_from(">H", buf, base + 18)[0]
            ai    = struct.unpack_from(">B", buf, base + 20)[0]
            _log(
                f"  Crate[{i+1}]: #{crate_number}  "
                f"Rx={Rx:.1f} Ry={Ry:.1f} Rz={Rz:.1f}  "
                f"theta={theta:.3f}  S_flags={bin(flags)}  ai={ai}"
            )
    except Exception as exc:
        _log(f"  readback error: {exc}")
