"""
crate_vision/io/loader.py
=========================
Load amplitude images and XYZ point clouds from snapshot folders.

Every function returns plain numpy arrays or None — no side effects,
no global state, no CONFIG dependency at call time.
"""

from __future__ import annotations
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import zoom


def load_depth_and_amplitude(
    folder_path: str,
) -> tuple[
    np.ndarray | None,   # x_coords  (H, W) float
    np.ndarray | None,   # y_coords  (H, W) float
    np.ndarray | None,   # z_coords  (H, W) float
    np.ndarray | None,   # img_color (H, W, 3) uint8 RGB
    np.ndarray | None,   # img_gray  (H, W)    uint8
]:
    """
    Load ``amplitude.png`` and ``xyz_combined.npy`` from *folder_path*.

    ``xyz_combined.npy`` shape must be ``(H, W, 3)`` where:
      - channel 0 → X in mm
      - channel 1 → Y in mm
      - channel 2 → Z in mm

    If either file is missing or invalid, all five return values are ``None``.

    Parameters
    ----------
    folder_path : str or Path
        Directory containing ``amplitude.png`` and ``xyz_combined.npy``.

    Returns
    -------
    x_coords, y_coords, z_coords, img_color, img_gray
    """
    folder = Path(folder_path)

    depth_path = folder / "xyz_combined.npy"
    amp_path = folder / "amplitude.png"

    if not depth_path.exists() or not amp_path.exists():
        return None, None, None, None, None

    # ── Load XYZ ──────────────────────────────────────────────────────────────
    xyz = np.load(str(depth_path)).astype(float)
    if xyz.ndim != 3 or xyz.shape[2] != 3:
        return None, None, None, None, None

    # ── Load amplitude ────────────────────────────────────────────────────────
    img_color = cv2.imread(str(amp_path))
    if img_color is None:
        return None, None, None, None, None

    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    img_h, img_w = img_color.shape[:2]

    # ── Resize XYZ to match image if needed ───────────────────────────────────
    if xyz.shape[:2] != (img_h, img_w):
        resized = np.zeros((img_h, img_w, 3), dtype=float)
        for ch in range(3):
            resized[..., ch] = zoom(
                xyz[..., ch],
                (img_h / xyz.shape[0], img_w / xyz.shape[1]),
                order=1,
            )
        xyz = resized

    x_coords = xyz[..., 0]
    y_coords = xyz[..., 1]
    z_coords = xyz[..., 2]

    return x_coords, y_coords, z_coords, img_color, img_gray
