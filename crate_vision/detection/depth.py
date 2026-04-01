"""
crate_vision/detection/depth.py
================================
Depth-layer masking and small-blob removal.

All functions are pure (no global state, no I/O).
"""

from __future__ import annotations

import cv2
import numpy as np


def create_depth_masks(
    z_coord: np.ndarray,
    distances: list[float],
    half_width: float | list[float] = 5.0,
) -> dict[str, np.ndarray]:
    """
    Build binary masks for depth slices centred on each *distance*.

    Parameters
    ----------
    z_coord    : (H, W) float array — Z values in mm
    distances  : list of target depths in mm
    half_width : ± range around each target (mm). Can be a single float or list of floats (one per distance)

    Returns
    -------
    dict mapping a human-readable range name to a bool (H, W) mask.
    Example key: ``"1020.0mm ±100.0mm (920.0-1120.0mm)"``
    """
    # Convert single half_width to list if needed
    if isinstance(half_width, (int, float)):
        half_widths = [half_width] * len(distances)
    else:
        half_widths = list(half_width)
        # Pad with last value if lengths don't match
        if len(half_widths) < len(distances):
            half_widths.extend([half_widths[-1]] * (len(distances) - len(half_widths)))
    
    valid = np.isfinite(z_coord) & (z_coord > 0)
    masks: dict[str, np.ndarray] = {}

    for dist, hw in zip(distances, half_widths):
        lo = dist - hw
        hi = dist + hw
        name = f"{dist}mm \u00b1{hw}mm ({lo}-{hi}mm)"
        masks[name] = valid & (z_coord >= lo) & (z_coord < hi)

    return masks


def remove_small_blobs(
    mask: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """
    Remove connected components whose area is smaller than *min_size*.

    Parameters
    ----------
    mask     : (H, W) bool or uint8 mask
    min_size : minimum blob area in pixels to keep

    Returns
    -------
    (H, W) bool mask with small blobs removed.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8)
    )
    clean = np.zeros_like(mask, dtype=bool)
    for i in range(1, num_labels):          # skip label 0 (background)
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            clean[labels == i] = True
    return clean


def apply_mask_to_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Zero out pixels where *mask* is False.

    Works for both grayscale ``(H, W)`` and colour ``(H, W, C)`` images.

    Parameters
    ----------
    image : numpy array  (H, W) or (H, W, C)
    mask  : (H, W) bool or uint8

    Returns
    -------
    Copy of *image* with masked-out pixels set to zero.
    """
    masked = image.copy()
    m = mask.astype(np.uint8)
    if masked.ndim == 3:
        for ch in range(masked.shape[2]):
            masked[:, :, ch] = masked[:, :, ch] * m
    else:
        masked = masked * m
    return masked
