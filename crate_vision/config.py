"""
crate_vision/config.py
======================
Central configuration.  One source of truth.

Usage
-----
    from crate_vision.config import get_config, CONFIG

    # Read the live config anywhere
    cfg = get_config()
    cfg.layer_distances_mm          # [1020.0, 1300.0]

    # Override for testing / CLI
    from crate_vision.config import override_config
    override_config(snapshot_folder="snapshots/snap0042_...")
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


# ---------------------------------------------------------------------------
# Dataclass — every parameter has a type annotation and a default
# ---------------------------------------------------------------------------

@dataclass
class CrateVisionConfig:

    # ── Input / Output ────────────────────────────────────────────────────────
    snapshot_folder: str = r"snapshots\snap0001_20260323_171356"
    output_dpi: int = 150

    # ── Depth Layer Segmentation ──────────────────────────────────────────────
    layer_distances_mm: List[float] = field(default_factory=lambda: [1020.0, 1300.0])
    layer_half_widths_mm: List[float] = field(default_factory=lambda: [100.0, 100.0])

    # ── Crate Grid Layout ─────────────────────────────────────────────────────
    grid_n_cols: int = 2
    grid_n_rows: int = 2
    crate_width: float = 500.0      # mm
    crate_height: float = 400.0     # mm
    grid_corner_pixels: dict = field(default_factory=dict)  # Anchor pixels for grid calibration

    # ── Crate Size Filter ─────────────────────────────────────────────────────
    crate_min_size_mm: List[float] = field(default_factory=lambda: [380.0, 250.0, 200.0])
    crate_max_size_mm: List[float] = field(default_factory=lambda: [600.0, 500.0, 300.0])

    # ── CCL Blob Detection ────────────────────────────────────────────────────
    ccl_min_blob_size: int = 10
    ccl_min_px: int = 5
    ccl_max_px: int = 50000
    ccl_min_aspect: float = 0.2
    ccl_max_aspect: float = 5.0
    ccl_cell_padding_px: int = 5

    # ── Rectangle Fitting ─────────────────────────────────────────────────────
    rect_padding: int = 3
    rect_interior_threshold: float = 0.8

    # ── Corner Detection ──────────────────────────────────────────────────────
    corner_min_neighbours: int = 4
    corner_search_radius_px: int = 3

    # ── Slot Fill Analysis ────────────────────────────────────────────────────
    slot_n_cols: int = 6
    slot_n_rows: int = 4
    slot_fill_threshold_mm: float = 180.0
    slot_min_valid_frac: float = 0.05
    slot_sample_radius_px: int = 3
    slot_radius_mm: float = 30.0
    present_n_pixel: int = 3

    # ── OBB Visualisation ─────────────────────────────────────────────────────
    obb_canvas_margin: int = 8

    # ── Camera Intrinsics ─────────────────────────────────────────────────────
    image_px_width: int = 176
    image_px_height: int = 132
    fov_h_deg: float = 60.0
    fov_v_deg: float = 45.0

    # ── AI Verification ───────────────────────────────────────────────────────
    ai_model_pth: str = r"outputs\densenet121_crate_best.pth"
    ai_model_onnx: str = "densenet121_crate.onnx"
    ai_conf_threshold: float = 0.7
    ai_class_names: List[str] = field(default_factory=lambda: ["crate", "no_crate"])
    ai_crate_class_idx: int = 0

    # ── Helpers ───────────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CrateVisionConfig":
        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def from_dict(cls, d: dict) -> "CrateVisionConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_CONFIG: CrateVisionConfig = CrateVisionConfig()


def get_config() -> CrateVisionConfig:
    """Return the active configuration singleton."""
    return _CONFIG


def override_config(**kwargs) -> CrateVisionConfig:
    """
    Update specific fields on the singleton in-place.

    Example
    -------
        override_config(snapshot_folder="snapshots/test", ai_conf_threshold=0.8)
    """
    global _CONFIG
    for k, v in kwargs.items():
        if not hasattr(_CONFIG, k):
            raise AttributeError(f"CrateVisionConfig has no field '{k}'")
        setattr(_CONFIG, k, v)
    return _CONFIG


def load_config_from_json(path: str) -> CrateVisionConfig:
    """Replace the singleton from a JSON file and return it."""
    global _CONFIG
    _CONFIG = CrateVisionConfig.from_json(path)
    return _CONFIG


# Convenience alias — use get_config() in new code; CONFIG for compat with old code
CONFIG = _CONFIG
