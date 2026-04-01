"""
main.py
=======
IFM O3D303 — Normal Operating Mode  +  S7-1200 via ISO-on-TCP (Snap7)

Startup sequence:
  1. Connect to S7-1200 via Snap7 (ISO-on-TCP port 102)
  2. Configure O3D303 via XML-RPC  → TriggerMode = 3 (HW rising edge)
  3. Open PCIC stream (port 50010)
  4. Wait for hardware trigger on Pin 2
  5. Per trigger:
       a. Receive frame → save snapshot
       b. Call process_depth_layers() → result dict
       c. Write result into PLC DB via Snap7

Usage:
    python main.py
    python main.py --config my_config.json
"""

from __future__ import annotations

import signal
import sys
import threading
from datetime import datetime

# ── Optional o3d3xx import ────────────────────────────────────────────────────
try:
    import o3d3xx
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False


# ── Project imports ────────────────────────────────────────────────────────────
from crate_vision.config import get_config, load_config_from_json
from crate_vision.hardware.camera import GrabO3D300, configure_camera, setup_pcic_stream
from crate_vision.hardware.plc import PLCClient
from crate_vision.pipeline import process_depth_layers


# =============================================================================
# Configuration
# =============================================================================

# Camera
CAMERA_IP  = "10.10.10.50"
PCIC_PORT  = 50010

CAMERA_CONFIG = {
    "TriggerMode": "3",   # HW rising edge
}

# PLC
PLC_IP   = "10.10.10.105"
PLC_RACK = 0
PLC_SLOT = 1
PLC_DB   = 2

# PCIC blob schema (requires o3d3xx)
PCIC_SCHEMA = (
    o3d3xx.PCICFormat.blobs(
        "amplitude_image",
        "distance_image",
        "x_image",
        "y_image",
        "z_image",
    )
    if _O3D_AVAILABLE
    else None
)


# =============================================================================
# Logging
# =============================================================================

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")


# =============================================================================
# Main trigger loop
# =============================================================================

def run(cam, plc: PLCClient, shutdown_event: threading.Event) -> None:
    """Block-loop: receive frame → process → write PLC."""
    cfg     = get_config()
    grabber = GrabO3D300(
        image_width=cfg.image_px_width,
        image_height=cfg.image_px_height,
    )

    log("─" * 60)
    log("Ready — waiting for hardware trigger on Pin 2 (rising edge)")
    log("Press Ctrl+C to stop.")
    log("─" * 60)

    while not shutdown_event.is_set():
        try:
            frame = cam.readNextFrame()
            t_recv = datetime.now()
            log(f"▸ Frame received  {t_recv.strftime('%H:%M:%S.%f')[:-3]}")

            if "amplitude_image" not in frame or "z_image" not in frame:
                log("  ⚠  Incomplete frame — skipping.")
                continue

            grabber.load_from_frame(frame)

            log("▸ Saving snapshot...")
            snapshot_folder = grabber.save_snapshot()

            # ── CrateDetector ──────────────────────────────────────────────────
            log("=" * 56)
            log("  STEP 1 — CrateDetector")
            log("=" * 56)

            detection_result = None
            try:
                detection_result = process_depth_layers(snapshot_folder, cfg=cfg)
                n = len(detection_result.get("crates", []))
                log(f"  Result: snap_id={detection_result.get('snap_id')}  crates={n}")
            except Exception as exc:
                log(f"  ⚠  CrateDetector error: {exc}")
                detection_result = {"snap_id": snapshot_folder, "crates": []}

            # ── Write to PLC ───────────────────────────────────────────────────
            if detection_result is not None:
                log("─" * 56)
                log(f"  STEP 2 — Write result to S7-1200 DB{PLC_DB}")
                log("─" * 56)
                sent = plc.write_result(detection_result)
                if not sent:
                    log("  PLC ⚠  write failed — result logged only")

            elapsed = (datetime.now() - t_recv).total_seconds()
            log(f"◂ Cycle complete in {elapsed:.3f}s")
            log("─" * 60)
            log("Waiting for next trigger...")

        except Exception as exc:
            if shutdown_event.is_set():
                break
            log(f"  ⚠  Unexpected error: {exc}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    # ── Optional config file ───────────────────────────────────────────────────
    if "--config" in sys.argv:
        idx = sys.argv.index("--config")
        if idx + 1 < len(sys.argv):
            load_config_from_json(sys.argv[idx + 1])
            log(f"Config loaded from {sys.argv[idx + 1]}")

    cfg = get_config()

    log("=" * 60)
    log("  IFM O3D303 — Normal Operating Mode (ISO-on-TCP)")
    log(f"  Camera     : {CAMERA_IP}:{PCIC_PORT}")
    log(f"  Trigger    : Hardware Rising Edge (Pin 2)")
    log(f"  Resolution : {cfg.image_px_width}×{cfg.image_px_height} px")
    log(f"  PLC        : {PLC_IP}  rack={PLC_RACK}  slot={PLC_SLOT}  DB={PLC_DB}")
    log(f"  Protocol   : ISO-on-TCP (port 102, Snap7)")
    log("=" * 60)

    # ── Graceful shutdown ──────────────────────────────────────────────────────
    shutdown_event = threading.Event()

    def _on_signal(sig, frame):
        log("\nShutdown requested...")
        shutdown_event.set()

    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ── Phase 1: PLC connection ────────────────────────────────────────────────
    log("\n── Phase 1: S7-1200 ISO-on-TCP Connection ───────────────")
    plc = PLCClient(PLC_IP, PLC_RACK, PLC_SLOT, PLC_DB)
    if not plc.connect():
        log("FATAL: Cannot connect to PLC — check IP, PUT/GET setting, and DB number")
        # sys.exit(1)   # uncomment to make PLC connection mandatory
    log(f"  PLC DB{PLC_DB} ready — no OUC blocks needed on PLC side ✓")

    # ── Phase 2: XML-RPC camera configuration ──────────────────────────────────
    log("\n── Phase 2: XML-RPC Configuration ──────────────────────")
    try:
        configure_camera(CAMERA_IP, CAMERA_CONFIG)
    except Exception as exc:
        log(f"FATAL: XML-RPC failed — {exc}")
        plc.disconnect()
        sys.exit(1)

    # ── Phase 3: PCIC stream setup ─────────────────────────────────────────────
    log("\n── Phase 3: PCIC Stream Setup ───────────────────────────")
    try:
        cam = setup_pcic_stream(CAMERA_IP, PCIC_PORT, PCIC_SCHEMA)
    except Exception as exc:
        log(f"FATAL: PCIC setup failed — {exc}")
        plc.disconnect()
        sys.exit(1)

    # ── Phase 4: Hardware trigger loop ─────────────────────────────────────────
    log("\n── Phase 4: Hardware Trigger Loop ───────────────────────")
    try:
        run(cam, plc, shutdown_event)
    finally:
        plc.disconnect()
        log("Done.")


if __name__ == "__main__":
    main()
