"""
crate_vision/detection/ccl.py
==============================
Connected-component labelling (CCL) on masked amplitude images,
plus the real-world size filter.

All functions are pure (no global state, no I/O, no CONFIG dependency).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as scipy_label
from skimage.measure import regionprops


# ---------------------------------------------------------------------------
# CCL on mask
# ---------------------------------------------------------------------------

def ccl_on_mask(
    mask: np.ndarray,
    amp_smooth: np.ndarray,
    min_px: int,
    max_px: int,
    min_aspect: float,
    max_aspect: float,
) -> tuple[list[dict], np.ndarray, list]:
    """
    Run 8-connected CCL on *amp_smooth* restricted to *mask*.

    Each surviving component is returned as a dict with keys:
      mask, bbox, area, width, height, aspect, centroid.

    Parameters
    ----------
    mask       : (H, W) bool — depth-layer mask for this cell
    amp_smooth : (H, W) float [0, 1] — amplitude image
    min_px     : minimum blob area (pixels)
    max_px     : maximum blob area (pixels)
    min_aspect : minimum width/height ratio
    max_aspect : maximum width/height ratio

    Returns
    -------
    objects     : list of dicts (kept blobs)
    labeled     : (H, W) int label array (all components, for debug)
    all_regions : list of skimage regionprops (all components, for debug)
    """
    masked_amp = np.where(mask, amp_smooth, 0)
    binary = masked_amp > 0

    labeled, _ = scipy_label(binary, structure=np.ones((3, 3)))
    all_regions = regionprops(labeled)

    objects: list[dict] = []
    for region in all_regions:
        area = region.area
        if area < min_px or area > max_px:
            continue

        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        aspect = width / max(height, 1)

        if aspect < min_aspect or aspect > max_aspect:
            continue

        objects.append(
            {
                "mask":     labeled == region.label,
                "bbox":     (minr, minc, maxr, maxc),
                "area":     area,
                "width":    width,
                "height":   height,
                "aspect":   aspect,
                "centroid": region.centroid,
            }
        )

    return objects, labeled, all_regions


# ---------------------------------------------------------------------------
# Real-world size filter
# ---------------------------------------------------------------------------

def size_filter(
    xs: np.ndarray,
    ys: np.ndarray,
    min_size_mm: list[float],
    max_size_mm: list[float],
) -> tuple[bool, float, float]:
    """
    Accept or reject an object based on its real-world XY extents.

    Parameters
    ----------
    xs          : 1-D array of X coordinates (mm) for valid pixels
    ys          : 1-D array of Y coordinates (mm) for valid pixels
    min_size_mm : [min_x, min_y, *ignored*] — minimum extents in mm
    max_size_mm : [max_x, max_y, *ignored*] — maximum extents in mm

    Returns
    -------
    accepted  : bool  — True if the object passes both extent and std checks
    x_size_mm : float — measured X extent in mm
    y_size_mm : float — measured Y extent in mm
    """
    min_x, min_y = min_size_mm[0], min_size_mm[1]
    max_x, max_y = max_size_mm[0], max_size_mm[1]

    x_size = float(xs.max() - xs.min())
    y_size = float(ys.max() - ys.min())

    x_std = np.std(xs)
    y_std = np.std(ys)

    accepted = (
        (min_x <= x_size <= max_x)
        and (min_y <= y_size <= max_y)
        and (x_std > 90 and y_std > 90)
    )

    return accepted, x_size, y_size
