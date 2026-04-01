"""
crate_vision/detection/slots.py
================================
Crate slot fill analysis, grid annotation, and figure output.

  analyze_crate_slots  — classify each slot as filled / empty / unknown
  draw_slot_grid       — annotate an amplitude crop with slot circles
  save_slot_figure     — write the annotated figure to disk
"""

from __future__ import annotations

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# 3-D slot centroid via bilinear interpolation between crate corners
# =============================================================================

def slot_centroid_3d(
    corners_xyz: dict,
    sr: int,
    sc: int,
    n_slot_rows: int,
    n_slot_cols: int,
) -> np.ndarray:
    """
    Bilinear interpolation between the four crate corners to get the
    3-D centroid of slot (sr, sc).

    Missing corners are estimated from the parallelogram assumption.
    """
    available = {
        k: np.array(v["xyz_mm"])
        for k, v in corners_xyz.items()
        if v is not None
    }

    u = (sc + 0.5) / n_slot_cols   # horizontal fraction
    v = (sr + 0.5) / n_slot_rows   # vertical   fraction

    if all(k in available for k in ["top_left", "top_right", "bottom_left", "bottom_right"]):
        top    = available["top_left"]    * (1 - u) + available["top_right"]    * u
        bottom = available["bottom_left"] * (1 - u) + available["bottom_right"] * u

    elif all(k in available for k in ["bottom_left", "bottom_right", "top_right"]):
        tl_est = available["bottom_left"] + (available["top_right"] - available["bottom_right"])
        top    = tl_est                   * (1 - u) + available["top_right"]    * u
        bottom = available["bottom_left"] * (1 - u) + available["bottom_right"] * u

    else:
        pts = list(available.values())
        return np.mean(pts, axis=0)

    return top * (1 - v) + bottom * v


# =============================================================================
# Slot fill analysis
# =============================================================================

def analyze_crate_slots(
    Z_full: np.ndarray,
    X_full: np.ndarray | None,
    Y_full: np.ndarray | None,
    bbox_pixels: tuple[int, int, int, int],
    corners_xyz: dict | None,
    ref_z_mm: float,
    plane_normal: list | None = None,
    plane_d: float | None = None,
    n_slot_cols: int = 6,
    n_slot_rows: int = 4,
    fill_threshold_mm: float = 180.0,
    min_valid_frac: float = 0.05,
    sample_radius_px: int = 3,
    slot_radius_mm: float = 30.0,
    present_n_pixel: int = 3,
) -> dict:
    """
    Classify each slot as ``"filled"`` / ``"empty"`` / ``"unknown"``.

    When *plane_normal* and *plane_d* are provided, uses plane-projection
    distance to determine validity and fill status.  Falls back to a
    simple Z threshold otherwise.

    Parameters
    ----------
    Z_full, X_full, Y_full : full-image coordinate arrays
    bbox_pixels            : (minr, minc, maxr, maxc) object bbox
    corners_xyz            : dict from detect_corners (or None to skip)
    ref_z_mm               : reference Z of the crate layer
    plane_normal           : (3,) list/array — fitted plane normal
    plane_d                : float — plane offset
    n_slot_cols, n_slot_rows : slot grid dimensions
    fill_threshold_mm      : height above plane considered "filled"
    min_valid_frac         : minimum fraction of circle pixels needed
    sample_radius_px       : pixel radius of sampling circle per slot
    slot_radius_mm         : radius on the plane for valid pixel acceptance
    present_n_pixel        : minimum depth-valid pixels to confirm "filled"

    Returns
    -------
    dict with keys: n_slot_cols, n_slot_rows, ref_z_mm, fill_threshold_mm,
                    sample_radius_px, slots, corner_crop
    """
    _empty_result = {
        "n_slot_cols":       n_slot_cols,
        "n_slot_rows":       n_slot_rows,
        "ref_z_mm":          float(ref_z_mm) if ref_z_mm is not None else 0.0,
        "fill_threshold_mm": float(fill_threshold_mm),
        "sample_radius_px":  sample_radius_px,
        "slots":             [],
        "corner_crop":       (None, None, None, None),
    }

    if corners_xyz is None:
        return _empty_result

    use_plane = (
        plane_normal is not None
        and plane_d is not None
        and X_full is not None
        and Y_full is not None
    )
    n_vec = np.array(plane_normal, dtype=float) if use_plane else None
    d_val = float(plane_d) if use_plane else None

    # ── Resolve corner pixel positions ────────────────────────────────────────
    def _get_px(name):
        v = corners_xyz.get(name)
        if v is None:
            return None
        cc, cr = v["col_row_full"]
        return np.array([cc, cr], dtype=float)

    tl = _get_px("top_left")
    tr = _get_px("top_right")
    bl = _get_px("bottom_left")
    br = _get_px("bottom_right")

    # Fill missing corners via parallelogram
    if tl is None and tr is not None and bl is not None and br is not None:
        tl = tr + bl - br
    if tr is None and tl is not None and br is not None and bl is not None:
        tr = tl + br - bl
    if bl is None and tl is not None and br is not None and tr is not None:
        bl = tl + br - tr
    if br is None and tr is not None and bl is not None and tl is not None:
        br = tr + bl - tl

    if any(c is None for c in [tl, tr, bl, br]):
        return _empty_result

    # ── Slot centroid in pixel space via bilinear interpolation ───────────────
    def _slot_px(sr: int, sc: int) -> np.ndarray:
        uf = (sc + 0.5) / n_slot_cols
        vf = (sr + 0.5) / n_slot_rows
        top    = tl * (1 - uf) + tr * uf
        bottom = bl * (1 - uf) + br * uf
        return top * (1 - vf) + bottom * vf   # (col, row)

    # ── Classify each slot ────────────────────────────────────────────────────
    slots = []
    rs = sample_radius_px

    for sr in range(n_slot_rows):
        for sc in range(n_slot_cols):
            cx, cy = _slot_px(sr, sc)
            cc_full = int(round(cx))
            cr_full = int(round(cy))

            # Guard: outside image
            if not (0 <= cr_full < Z_full.shape[0] and 0 <= cc_full < Z_full.shape[1]):
                slots.append(_unknown_slot(sr, sc, (cr_full, cc_full), n_slot_cols))
                continue

            # Circular sample window
            r0 = max(0, cr_full - rs); r1 = min(Z_full.shape[0], cr_full + rs + 1)
            c0 = max(0, cc_full - rs); c1 = min(Z_full.shape[1], cc_full + rs + 1)
            rr, cc_g = np.mgrid[r0:r1, c0:c1]
            circle = (rr - cr_full) ** 2 + (cc_g - cc_full) ** 2 <= rs ** 2

            patch_z = Z_full[r0:r1, c0:c1]
            has_xyz = circle & np.isfinite(patch_z) & (patch_z > 0)
            n_total = int(circle.sum())

            centroid_3d = slot_centroid_3d(corners_xyz, sr, sc, n_slot_rows, n_slot_cols)

            if use_plane:
                patch_x = X_full[r0:r1, c0:c1]
                patch_y = Y_full[r0:r1, c0:c1]
                has_xyz = has_xyz & np.isfinite(patch_x) & np.isfinite(patch_y)

                if not has_xyz.any():
                    slots.append(_unknown_slot(sr, sc, (cr_full, cc_full), n_slot_cols))
                    continue

                P = np.stack([patch_x[has_xyz], patch_y[has_xyz], patch_z[has_xyz]], axis=1)
                above = P @ n_vec - d_val
                P_proj = P - np.outer(above, n_vec)
                dist_on_plane = np.linalg.norm(P_proj - centroid_3d, axis=1)

                valid_mask = dist_on_plane <= slot_radius_mm
                n_valid = int(valid_mask.sum())

                depth_valid_mask = above[valid_mask] < fill_threshold_mm
                n_depth_valid = int(depth_valid_mask.sum())

                rr_idx = rr[has_xyz]
                cc_idx = cc_g[has_xyz]

                if n_total > rs:
                    median_above = np.mean(above[valid_mask])
                    status = (
                        "filled"
                        if (n_depth_valid >= present_n_pixel or median_above < fill_threshold_mm)
                        else "empty"
                    )
                    mean_above = above[valid_mask][depth_valid_mask].mean() if depth_valid_mask.any() else None
                else:
                    mean_above, status = None, "unknown"

                valid_rows  = rr_idx[valid_mask]
                valid_cols  = cc_idx[valid_mask]
                dv_rows     = rr_idx[valid_mask][depth_valid_mask]
                dv_cols     = cc_idx[valid_mask][depth_valid_mask]

                slots.append({
                    "slot_id":            sr * n_slot_cols + sc,
                    "row":                sr,
                    "col":                sc,
                    "label":              f"slot_{sr}_{sc}",
                    "centroid_3d":        centroid_3d.tolist(),
                    "centroid_px":        (cr_full, cc_full),
                    "mean_above_mm":      mean_above,
                    "n_valid_px":         n_valid,
                    "n_total_px":         n_total,
                    "status":             status,
                    "valid_px_full":      list(zip(valid_rows.tolist(), valid_cols.tolist())),
                    "depth_valid_px_full": list(zip(dv_rows.tolist(), dv_cols.tolist())),
                })

    return {
        "n_slot_cols":       n_slot_cols,
        "n_slot_rows":       n_slot_rows,
        "ref_z_mm":          float(ref_z_mm),
        "fill_threshold_mm": float(fill_threshold_mm),
        "sample_radius_px":  sample_radius_px,
        "slots":             slots,
        "corner_crop":       (tl, bl, br, tr),
    }


def _unknown_slot(sr: int, sc: int, centroid_px: tuple, n_slot_cols: int) -> dict:
    return {
        "slot_id":     sr * n_slot_cols + sc,
        "row":         sr,
        "col":         sc,
        "label":       f"slot_{sr}_{sc}",
        "centroid_px": centroid_px,
        "mean_z_mm":   None,
        "n_valid_px":  0,
        "n_total_px":  0,
        "status":      "unknown",
    }


# =============================================================================
# Slot grid visualisation
# =============================================================================

def draw_slot_grid(
    amp_crop: np.ndarray,
    slot_analysis: dict,
) -> np.ndarray:
    """
    Return a (H, W, 3) uint8 RGB image with slot circles and labels
    annotated over the amplitude crop.

    Colours: green = filled, blue = empty, dark-red = unknown.
    """
    base    = np.stack([(amp_crop * 255).astype(np.uint8)] * 3, axis=-1).copy()
    overlay = base.copy()
    result  = base.copy()

    rs = slot_analysis["sample_radius_px"]

    COLOR = {
        "filled":  (50,  200, 50),
        "empty":   (50,  100, 220),
        "unknown": (150,  0,  0),
    }

    for s in slot_analysis["slots"]:
        cr, cc = s["centroid_px"]
        color = COLOR.get(s["status"], COLOR["unknown"])

        cv2.circle(overlay, (cc, cr), 4, color, -1)
        cv2.circle(result,  (cc, cr), 4, color,  1, cv2.LINE_AA)

        ch = 3
        cv2.line(result, (cc - ch, cr), (cc + ch, cr), color, 1, cv2.LINE_AA)
        cv2.line(result, (cc, cr - ch), (cc, cr + ch), color, 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.30, result, 0.70, 0, result)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for s in slot_analysis["slots"]:
        cr, cc = s["centroid_px"]

        # Draw valid and depth-valid pixels
        for (r_f, c_f) in s.get("valid_px_full", []):
            if 0 <= r_f < result.shape[0] and 0 <= c_f < result.shape[1]:
                cv2.circle(result, (c_f, r_f), 1, (255, 140, 0), -1)

        for (r_f, c_f) in s.get("depth_valid_px_full", []):
            if 0 <= r_f < result.shape[0] and 0 <= c_f < result.shape[1]:
                cv2.circle(result, (c_f, r_f), 1, (0, 140, 0), -1)

        txt = str(s["slot_id"])
        (tw, th), _ = cv2.getTextSize(txt, font, 0.18, 1)
        ox, oy = cc - tw // 2, cr + th // 2
        for dr, dc, clr in [(1, 1, (0, 0, 0)), (0, 0, (255, 255, 255))]:
            cv2.putText(result, txt, (ox + dc, oy + dr), font, 0.18, clr, 1, cv2.LINE_AA)

    # Draw corner polygon
    corners = slot_analysis.get("corner_crop", (None,) * 4)
    if not all(c is None for c in corners):
        pts = np.array([corners], dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(result, [pts], isClosed=True,
                      color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    return result


def save_slot_figure(
    amp_crop: np.ndarray,
    slot_analysis: dict,
    bbox_pixels: tuple[int, int, int, int],
    out_path: str,
    output_dpi: int = 150,
) -> None:
    """
    Save the annotated slot grid figure to *out_path*.

    Parameters
    ----------
    amp_crop      : (H, W) float [0, 1] amplitude crop
    slot_analysis : dict from analyze_crate_slots
    bbox_pixels   : (minr, minc, maxr, maxc)
    out_path      : PNG output path
    output_dpi    : DPI for saved figure
    """
    annotated = draw_slot_grid(amp_crop, slot_analysis)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle(
        f"Slot Analysis  ({slot_analysis['n_slot_cols']}\u00d7{slot_analysis['n_slot_rows']} grid)\n"
        f"ref_Z={slot_analysis['ref_z_mm']:.1f} mm, "
        f"threshold={slot_analysis['fill_threshold_mm']:.1f} mm",
        fontsize=9,
    )
    ax.imshow(annotated)
    ax.set_title("Slot Grid (green=filled, blue=empty, grey=unknown)", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=output_dpi, facecolor="white")
    plt.close(fig)
