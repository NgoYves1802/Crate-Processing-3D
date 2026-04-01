"""
crate_vision/detection/corners.py
===================================
Corner detection via contour curvature.

Why curvature, not Harris
--------------------------
The blob is a hollow rectangular ring (the crate rim).  Harris looks for
intensity L-junctions, but the ring is nearly uniform in amplitude AND in Z.
What IS present is a geometric corner: the boundary contour makes a sharp
90 degree turn at each crate corner.  Curvature is large at a bend and
near-zero on a straight edge.

Algorithm
---------
1. Extract the outer boundary contour with findContours (CHAIN_APPROX_NONE).
2. Compute signed curvature at each point via cross-product of forward/back
   tangent vectors over a sliding window.
3. NMS along the 1-D curvature profile -> four peaks, one per quadrant.
4. Refine to sub-pixel accuracy with cornerSubPix on the Z-gradient magnitude.
5. Bilinear-interpolate XYZ at the refined position.

Public API
----------
    from crate_vision.detection.corners import detect_corners_curvature

    corners_xyz = detect_corners_curvature(
        blob_mask = mask_crop,
        amp_crop  = amp_crop,
        Z_crop    = Z_crop,
        X_crop    = X_crop,
        Y_crop    = Y_crop,
        r0 = r0, c0 = c0,
        debug_dir = None,
    )
"""

from __future__ import annotations
import os
import cv2
import numpy as np


def detect_corners_curvature(
    blob_mask:  np.ndarray,
    amp_crop:   np.ndarray,
    Z_crop:     np.ndarray,
    X_crop:     np.ndarray,
    Y_crop:     np.ndarray,
    r0: int,
    c0: int,
    tangent_window:     int   = 5,
    curvature_smooth:   int   = 3,
    nms_radius:         int   = 8,
    subpix_window:      int   = 7,
    subpix_iterations:  int   = 40,
    subpix_epsilon:     float = 0.001,
    min_contour_length: int   = 20,
    debug_dir: str | None     = None,
) -> dict[str, dict | None]:
    """
    Find the four crate corners as contour-curvature peaks.

    Parameters
    ----------
    blob_mask         : (H, W) bool  crate rim mask in crop coords
    amp_crop          : (H, W) float [0,1]  amplitude crop
    Z_crop            : (H, W) float mm  depth values
    X_crop, Y_crop    : (H, W) float mm  lateral coords
    r0, c0            : crop top-left in full-image coordinates
    tangent_window    : half-window for tangent estimation (3-8 px)
    curvature_smooth  : Gaussian smoothing sigma along curvature profile
    nms_radius        : NMS half-window along the contour
    subpix_window     : half-window for cornerSubPix
    subpix_iterations : max iterations for cornerSubPix
    subpix_epsilon    : convergence for cornerSubPix
    min_contour_length: ignore contours shorter than this

    Returns
    -------
    dict with keys top_left, top_right, bottom_right, bottom_left.
    Each value is None or:
        col_row_crop : (col, row) float  sub-pixel crop-local
        col_row_full : (col, row) float  sub-pixel full-image
        xyz_mm       : (x, y, z) tuple in mm
    """
    H, W = blob_mask.shape

    contour = _extract_contour(blob_mask, min_contour_length)
    if contour is None:
        _save_debug(debug_dir, amp_crop, blob_mask, None, None, {})
        return {k: None for k in ["top_left", "top_right", "bottom_right", "bottom_left"]}

    curvature = _compute_curvature(contour, tangent_window, curvature_smooth)
    candidates = _nms_on_contour(curvature, nms_radius)
    corners_px = _assign_quadrants(contour, candidates, curvature, H, W)
    corners_px = _refine_subpix(corners_px, Z_crop, amp_crop,
                                  subpix_window, subpix_iterations, subpix_epsilon)
    result = _build_output(corners_px, X_crop, Y_crop, Z_crop, r0, c0)
    _save_debug(debug_dir, amp_crop, blob_mask, contour, curvature, result)
    return result


# ---------------------------------------------------------------------------
# Step 1 -- contour extraction
# ---------------------------------------------------------------------------

def _extract_contour(blob_mask: np.ndarray, min_length: int) -> np.ndarray | None:
    """
    Return the outer boundary of the blob as (N, 2) float array of (col, row).
    CHAIN_APPROX_NONE keeps every boundary pixel for dense curvature estimation.
    """
    mask_u8 = blob_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    longest = max(contours, key=lambda c: len(c))
    if len(longest) < min_length:
        return None
    return longest[:, 0, :].astype(np.float32)   # (N, 2): col, row


# ---------------------------------------------------------------------------
# Step 2 -- curvature
# ---------------------------------------------------------------------------

def _compute_curvature(
    contour: np.ndarray,
    window: int,
    smooth_sigma: int,
) -> np.ndarray:
    """
    Absolute curvature at each contour point.

    kappa_i = |cross(v_back, v_fwd)| / (|v_back| * |v_fwd|)
            = |sin(theta)|  where theta is the turning angle.

    Large at a 90-degree bend, near-zero on a straight edge.
    """
    N = len(contour)
    curvature = np.zeros(N, dtype=np.float32)

    for i in range(N):
        i_back = (i - window) % N
        i_fwd  = (i + window) % N
        v_back = contour[i]       - contour[i_back]
        v_fwd  = contour[i_fwd]   - contour[i]

        nb = np.linalg.norm(v_back)
        nf = np.linalg.norm(v_fwd)
        if nb < 1e-6 or nf < 1e-6:
            continue

        v_back /= nb
        v_fwd  /= nf
        cross = v_back[0] * v_fwd[1] - v_back[1] * v_fwd[0]
        curvature[i] = abs(cross)

    if smooth_sigma > 0:
        k = int(6 * smooth_sigma + 1) | 1
        tile = np.tile(curvature, 3)
        smoothed = cv2.GaussianBlur(
            tile.reshape(1, -1).astype(np.float32), (k, 1), smooth_sigma
        ).flatten()
        curvature = smoothed[N: 2 * N]

    mx = curvature.max()
    if mx > 1e-9:
        curvature /= mx
    return curvature


# ---------------------------------------------------------------------------
# Step 3 -- NMS + quadrant assignment
# ---------------------------------------------------------------------------

def _nms_on_contour(curvature: np.ndarray, nms_radius: int) -> list[int]:
    """Local maxima along the 1-D curvature profile, sorted by strength."""
    N = len(curvature)
    peaks = []
    for i in range(N):
        v = curvature[i]
        if v <= 0:
            continue
        if all(v >= curvature[(i + d) % N]
               for d in range(-nms_radius, nms_radius + 1) if d != 0):
            peaks.append(i)
    peaks.sort(key=lambda i: -curvature[i])
    return peaks


def _assign_quadrants(
    contour:       np.ndarray,
    candidate_idx: list[int],
    curvature:     np.ndarray,
    H: int, W: int,
) -> dict[str, tuple[float, float] | None]:
    """Best candidate per quadrant, split at image centre."""
    mid_r, mid_c = H / 2.0, W / 2.0

    preds = {
        "top_left":     lambda col, row: row < mid_r and col < mid_c,
        "top_right":    lambda col, row: row < mid_r and col >= mid_c,
        "bottom_right": lambda col, row: row >= mid_r and col >= mid_c,
        "bottom_left":  lambda col, row: row >= mid_r and col < mid_c,
    }

    corners: dict[str, tuple[float, float] | None] = {}
    for name, pred in preds.items():
        best_score = -1.0
        best: tuple[float, float] | None = None
        for idx in candidate_idx:
            col, row = float(contour[idx, 0]), float(contour[idx, 1])
            if pred(col, row) and curvature[idx] > best_score:
                best_score = curvature[idx]
                best = (col, row)
        corners[name] = best
    return corners


# ---------------------------------------------------------------------------
# Step 4 -- sub-pixel refinement
# ---------------------------------------------------------------------------

def _refine_subpix(
    corners_px:        dict[str, tuple[float, float] | None],
    Z_crop:            np.ndarray,
    amp_crop:          np.ndarray,               # kept for signature, but not used
    subpix_window:     int,                      # not used, kept for compatibility
    subpix_iterations: int,                      # not used
    subpix_epsilon:    float,                    # not used
    search_radius:     int = 1,                  # new parameter: half‑window for min‑Z search
) -> dict[str, tuple[float, float] | None]:
    """
    Replace each corner with the pixel having the lowest Z value in a local region.

    For each candidate corner, we search a square of size (2*search_radius+1) pixels
    centred at the rounded integer position of the corner. Within that window, we
    find the (col, row) of the pixel with the smallest Z value (ignoring invalid Z).
    This new coordinate is returned as a float.

    Parameters
    ----------
    corners_px : dict
        Current corner positions (col, row) as floats, or None.
    Z_crop : (H, W) float
        Depth map (mm). Invalid values (<=0, NaN) are ignored.
    amp_crop : (H, W) float   (unused)
        Amplitude map – kept only for signature compatibility.
    subpix_window, subpix_iterations, subpix_epsilon : (unused)
        Kept for signature compatibility.
    search_radius : int
        Half‑window size for the local search. Default 3 (7x7 window).

    Returns
    -------
    dict[str, tuple[float, float] | None]
        Updated dictionary with new coordinates.
    """
    H, W = Z_crop.shape
    valid_mask = np.isfinite(Z_crop) & (Z_crop > 0)

    out = dict(corners_px)
    for name, cr in corners_px.items():
        if cr is None:
            continue

        # Round to integer pixel coordinates
        col0 = int(round(cr[0]))
        row0 = int(round(cr[1]))

        # Define window bounds, clipping to image edges
        r_min = max(0, row0 - search_radius)
        r_max = min(H, row0 + search_radius + 1)
        c_min = max(0, col0 - search_radius)
        c_max = min(W, col0 + search_radius + 1)

        # Extract the region of interest
        roi_Z = Z_crop[r_min:r_max, c_min:c_max]
        roi_valid = valid_mask[r_min:r_max, c_min:c_max]

        if not np.any(roi_valid):
            # No valid depth in the region – keep original coordinate
            continue

        # Find the pixel with the minimum Z (lowest depth)
        # Only consider valid pixels; set invalid to +inf so they are ignored
        roi_Z_invalid = np.where(roi_valid, roi_Z, np.inf)
        min_idx = np.argmin(roi_Z_invalid)          # flat index
        min_row, min_col = np.unravel_index(min_idx, roi_Z_invalid.shape)

        # Convert back to global coordinates
        best_col = float(c_min + min_col)
        best_row = float(r_min + min_row)

        out[name] = (best_col, best_row)

    return out

# ---------------------------------------------------------------------------
# Step 5 -- XYZ lookup
# ---------------------------------------------------------------------------

def _build_output(
    corners_px: dict[str, tuple[float, float] | None],
    X_crop: np.ndarray,
    Y_crop: np.ndarray,
    Z_crop: np.ndarray,
    r0: int,
    c0: int,
) -> dict[str, dict | None]:
    H, W = X_crop.shape
    result: dict[str, dict | None] = {}
    for name, cr in corners_px.items():
        if cr is None:
            result[name] = None
            continue
        col_f = float(np.clip(cr[0], 0, W - 1))
        row_f = float(np.clip(cr[1], 0, H - 1))
        result[name] = {
            "col_row_crop": (col_f, row_f),
            "col_row_full": (col_f + c0, row_f + r0),
            "xyz_mm": (
                _bilinear(X_crop, row_f, col_f),
                _bilinear(Y_crop, row_f, col_f),
                _bilinear(Z_crop, row_f, col_f),
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-9:
        return np.zeros_like(a, dtype=np.uint8)
    return ((a - lo) / (hi - lo) * 255).astype(np.uint8)


def _bilinear(arr: np.ndarray, row_f: float, col_f: float) -> float:
    H, W = arr.shape
    r0 = int(row_f);  r1 = min(r0 + 1, H - 1)
    c0 = int(col_f);  c1 = min(c0 + 1, W - 1)
    dr = row_f - r0;  dc = col_f - c0
    return float(
        arr[r0, c0] * (1 - dr) * (1 - dc) +
        arr[r0, c1] * (1 - dr) *      dc  +
        arr[r1, c0] *      dr  * (1 - dc) +
        arr[r1, c1] *      dr  *      dc
    )


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def _save_debug(
    debug_dir:  str | None,
    amp_crop:   np.ndarray,
    blob_mask:  np.ndarray,
    contour:    np.ndarray | None,
    curvature:  np.ndarray | None,
    corners:    dict[str, dict | None],
) -> None:
    if debug_dir is None:
        return

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.cm as cm

    os.makedirs(debug_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#0d1117")

    COLORS = {
        "top_left":     "#ff6b6b",
        "top_right":    "#ffd93d",
        "bottom_right": "#6bcb77",
        "bottom_left":  "#4d96ff",
    }

    # Panel 0: amplitude + mask
    axes[0].imshow(amp_crop, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*blob_mask.shape, 4), dtype=np.float32)
    overlay[blob_mask] = [0.2, 0.8, 0.4, 0.35]
    axes[0].imshow(overlay)
    axes[0].set_title("Amplitude + blob mask", color="white", fontsize=10)
    axes[0].axis("off")

    # Panel 1: contour
    axes[1].imshow(amp_crop, cmap="gray", vmin=0, vmax=1)
    if contour is not None:
        axes[1].plot(contour[:, 0], contour[:, 1], "-",
                     color="#00e5ff", linewidth=0.8, alpha=0.8)
    n = len(contour) if contour is not None else 0
    axes[1].set_title(f"Boundary contour ({n} pts)", color="white", fontsize=10)
    axes[1].axis("off")

    # Panel 2: curvature heatmap
    axes[2].imshow(amp_crop, cmap="gray", vmin=0, vmax=1)
    if contour is not None and curvature is not None:
        cmap_hot = cm.get_cmap("hot")
        for i in range(len(contour) - 1):
            ca, cb = contour[i], contour[i + 1]
            axes[2].plot([ca[0], cb[0]], [ca[1], cb[1]],
                         "-", color=cmap_hot(float(curvature[i])),
                         linewidth=1.5, solid_capstyle="round")
    axes[2].set_title("Curvature (bright=high)", color="white", fontsize=10)
    axes[2].axis("off")

    # Panel 3: final corners
    axes[3].imshow(amp_crop, cmap="gray", vmin=0, vmax=1)
    patches = []
    for name, info in corners.items():
        if info is None:
            continue
        col_f, row_f = info["col_row_crop"]
        color = COLORS[name]
        axes[3].plot(col_f, row_f, "+", color=color,
                     markersize=16, markeredgewidth=2.5, zorder=5)
        circle = plt.Circle((col_f, row_f), 2.5,
                             color=color, fill=False, linewidth=1.2, zorder=4)
        axes[3].add_patch(circle)
        axes[3].annotate(name.replace("_", "\n"), xy=(col_f, row_f),
                         xytext=(col_f + 4, row_f - 4), color=color, fontsize=7,
                         arrowprops=dict(arrowstyle="-", color=color, lw=0.6))
        x, y, z = info["xyz_mm"]
        axes[3].annotate(f"({x:.0f},{y:.0f},{z:.0f}mm)",
                         xy=(col_f, row_f), xytext=(col_f + 4, row_f + 8),
                         color=color, fontsize=6)
        patches.append(mpatches.Patch(color=color, label=name))

    axes[3].set_title("Detected corners (sub-pixel)", color="white", fontsize=10)
    axes[3].axis("off")
    if patches:
        axes[3].legend(handles=patches, fontsize=7,
                       facecolor="#1a1a2e", labelcolor="white", loc="lower right")

    for ax in axes:
        ax.set_facecolor("#0d1117")

    plt.suptitle("Corner Detection — Contour Curvature",
                 color="white", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(debug_dir, "corners_debug.png"),
                dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()