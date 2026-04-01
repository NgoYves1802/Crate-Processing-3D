"""
crate_vision/io/serializer.py
==============================
Binary packing for the S7-1200 PLC and JSON disk output.

DB_CrateScan layout (20 bytes per crate)
-----------------------------------------
  Offset  Field          Type          Bytes
  0       crate_number   INT (S7 BE)   2
  2       Rx             REAL (S7 BE)  4
  6       Ry             REAL (S7 BE)  4
  10      Rz             REAL (S7 BE)  4
  14      theta          REAL (S7 BE)  4
  18      S1–S8          BYTE          1   bit0=S1 … bit7=S8
  19      S9–S12 + AI    BYTE          1   bit0=S9 … bit3=S12, bit4=ai
  Total                                20

Frame layout (variable length)
-------------------------------
  1 byte  : number of crates  (N)
  20×N    : crate records
"""

from __future__ import annotations
import json
import os
import struct


# ---------------------------------------------------------------------------
# Single-crate packer
# ---------------------------------------------------------------------------

def pack_crate(crate: dict) -> bytes:
    """
    Pack one crate dict into 20 bytes matching DB_CrateScan layout.

    Parameters
    ----------
    crate : dict
        Keys expected: crate_number, Rx, Ry, Rz, theta,
                       S1–S12 (bool), ai_classification (bool).
        Missing keys default to 0 / False.
    """
    slot_byte_0 = 0
    slot_byte_1 = 0

    for i, key in enumerate(["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]):
        if crate.get(key, False):
            slot_byte_0 |= (1 << i)

    for i, key in enumerate(["S9", "S10", "S11", "S12", "ai_classification"]):
        if crate.get(key, False):
            slot_byte_1 |= (1 << i)

    return struct.pack(
        ">hffff BB",
        int(crate.get("crate_number") or 0),
        float(crate.get("Rx") or 0.0),
        float(crate.get("Ry") or 0.0),
        float(crate.get("Rz") or 0.0),
        float(crate.get("theta") or 0.0),
        slot_byte_0,
        slot_byte_1,
    )


# ---------------------------------------------------------------------------
# Full-result packer
# ---------------------------------------------------------------------------

def pack_all_crates(result: dict) -> bytes:
    """
    Pack the full detection result into a binary frame.

    Frame = 1-byte header (crate count N) + 20 bytes × N.
    Max 241 bytes for 12 crates.

    Parameters
    ----------
    result : dict
        Must have key ``"crates"`` → list of crate dicts.
    """
    crates = result.get("crates", [])
    header = struct.pack("B", len(crates))
    payload = b"".join(pack_crate(c) for c in crates)
    return header + payload


# ---------------------------------------------------------------------------
# JSON output helpers
# ---------------------------------------------------------------------------

def build_crate_row(meta: dict) -> dict:
    """
    Convert one object meta dict (from ``save_object``) into a flat
    crate_scans row matching the DB schema.

    Parameters
    ----------
    meta : dict
        The dict returned by ``pose.build_meta()``.

    Returns
    -------
    dict with keys: snap_id, crate_number, Rx, Ry, Rz, theta,
                    S1–S12 (bool), ai_classification (bool).
    """
    bary = meta.get("barycenter_mm", {})
    pose = meta.get("pose_2d", {})
    ai = meta.get("ai_verification", {})
    slots = meta.get("slot_analysis", {}).get("slots", [])

    def _slot_bool(slot_id: int) -> bool:
        for s in slots:
            if s["slot_id"] == slot_id:
                return s["status"] == "filled"
        return False

    return {
        "snap_id":          meta.get("snap_id"),
        "crate_number":     meta.get("crate_id"),
        "Rx":               bary.get("x"),
        "Ry":               bary.get("y"),
        "Rz":               bary.get("z"),
        "theta":            pose.get("angle_deg"),
        "S1":               _slot_bool(0),
        "S2":               _slot_bool(1),
        "S3":               _slot_bool(2),
        "S4":               _slot_bool(3),
        "S5":               _slot_bool(4),
        "S6":               _slot_bool(5),
        "S7":               _slot_bool(6),
        "S8":               _slot_bool(7),
        "S9":               _slot_bool(8),
        "S10":              _slot_bool(9),
        "S11":              _slot_bool(10),
        "S12":              _slot_bool(11),
        "ai_classification": bool(ai.get("passed", False)),
    }


def write_crate_scans_json(result: dict, save_folder: str) -> str:
    """
    Write the full detection result to ``<save_folder>/crate_scans.json``.

    Parameters
    ----------
    result : dict
        ``{"snap_id": ..., "crates": [...]}``
    save_folder : str
        Output directory (must exist).

    Returns
    -------
    str
        Absolute path of the written file.
    """
    path = os.path.join(save_folder, "crate_scans.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path
