"""
crate_vision/pose.py
====================
Object pose estimation and persistence.

The original ``save_object`` god-function has been decomposed into:

    estimate_pose(...)   → pose_result dict  (pure, no I/O)
    build_meta(...)      → flat meta dict    (pure, no I/O)
    save_object(...)     → writes files + returns meta dict  (orchestrator)

This makes each step independently testable and the I/O boundary explicit.
"""

from __future__ import annotations

import json
import os

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image

from crate_vision.ai_verifier import get_verifier
from crate_vision.config import CrateVisionConfig, get_config
from crate_vision.detection.geometry import (
    compute_in_plane_rotation,
    compute_normal_from_corners,
    compute_orientation_angles,
    fit_min_area_rect,
    fit_plane_svd,
)
from crate_vision.detection.corners import detect_corners_curvature
from crate_vision.detection.slots import analyze_crate_slots, save_slot_figure


# =============================================================================
# estimate_pose  (pure — no I/O, no global state)
# =============================================================================

def estimate_pose(
    mask: np.ndarray,
    mask_crop: np.ndarray,
    amp_crop: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    cfg: CrateVisionConfig,
    debug_rect_dir: str | None = None,
) -> dict:
    """
    Estimate the 2-D/3-D pose of one detected object.

    Parameters
    ----------
    mask          : (H, W) bool — full-cell object mask
    mask_crop     : (crop_H, crop_W) bool — mask in crop coordinates
    amp_crop      : (crop_H, crop_W) float [0,1] — amplitude crop
    X, Y, Z       : (H, W) full-image coordinate arrays
    r0, c0, r1, c1: crop bounds in full-image coordinates
    cfg           : CrateVisionConfig
    debug_rect_dir: optional path for fit_min_area_rect debug images

    Returns
    -------
    dict with keys:
        pose2d         : output of fit_min_area_rect (or None)
        corners_xyz    : output of detect_corners (or None)
        rect_xyz       : (N, 3) float32 XYZ of rect interior pixels (or None)
        rect_mask_full : (H, W) bool (or None)
        normal         : (3,) float array
        d              : float — plane offset
        rmse           : float — plane fit RMSE (mm)
        residuals      : (N,) float — per-point residuals
        in_plane_rot   : float | None — in-plane rotation angle (degrees)
        yaw, pitch, roll : float — orientation angles (degrees)
        plane_info     : dict for JSON serialisation
    """
    result: dict = {
        "pose2d":         None,
        "corners_xyz":    None,
        "rect_xyz":       None,
        "rect_mask_full": None,
        "normal":         np.array([0.0, 0.0, 1.0]),
        "d":              0.0,
        "rmse":           0.0,
        "residuals":      np.array([]),
        "in_plane_rot":   0.0,
        "yaw":            0.0,
        "pitch":          0.0,
        "roll":           0.0,
        "plane_info":     None,
    }

    # ── 1. Fit interior rectangle ──────────────────────────────────────────────
    pose2d = fit_min_area_rect(
        mask_crop,
        padding=cfg.rect_padding,
        debug_dir=debug_rect_dir,
    )
    result["pose2d"] = pose2d

    if pose2d is None or X is None:
        return result

    # ── 2. Extract XYZ for rect interior pixels ────────────────────────────────
    rect_mask_crop = pose2d["rect_pixels_mask"]
    rect_mask_full = np.zeros_like(mask, dtype=bool)
    rect_mask_full[r0:r1, c0:c1] = rect_mask_crop

    rect_xyz = np.stack(
        [X[rect_mask_full], Y[rect_mask_full], Z[rect_mask_full]], axis=-1
    ).astype(np.float32)

    result["rect_xyz"] = rect_xyz
    result["rect_mask_full"] = rect_mask_full

    # ── 3. Corner detection (Harris fused 2D+3D, sub-pixel) ───────────────────
    Z_crop = Z[r0:r1, c0:c1]
    X_crop = X[r0:r1, c0:c1]
    Y_crop = Y[r0:r1, c0:c1]

    corners_xyz = detect_corners_curvature(
        amp_crop  = amp_crop,
        Z_crop    = Z_crop,
        X_crop    = X_crop,
        Y_crop    = Y_crop,
        blob_mask = rect_mask_crop,
        r0=r0, c0=c0,
        debug_dir = debug_rect_dir,
    )
    result["corners_xyz"] = corners_xyz

    # ── 4. Plane fitting ───────────────────────────────────────────────────────
    normal, d, residuals = fit_plane_svd(rect_xyz)
    rmse = float(np.sqrt((residuals ** 2).mean()))

    # Recompute from corners for better accuracy when enough are available
    normal_c = compute_normal_from_corners(corners_xyz)
    available = {k: v for k, v in corners_xyz.items() if v is not None}
    if len(available) >= 3:
        normal = normal_c
        d = float(np.mean([
            normal @ np.array(v["xyz_mm"])
            for v in available.values()
        ]))

    result.update({"normal": normal, "d": d, "rmse": rmse, "residuals": residuals})

    # ── 5. In-plane rotation ───────────────────────────────────────────────────
    in_plane_rot = compute_in_plane_rotation(corners_xyz, normal)
    result["in_plane_rot"] = in_plane_rot if in_plane_rot is not None else 0.0

    # ── 6. Orientation angles ──────────────────────────────────────────────────
    yaw, pitch, roll = compute_orientation_angles(normal)
    result.update({"yaw": yaw, "pitch": pitch, "roll": roll})

    # ── 7. Barycenter estimates from corners ───────────────────────────────────
    half_w = cfg.crate_width  / 2.0
    half_h = cfg.crate_height / 2.0

    tl = corners_xyz.get("top_left")
    tr = corners_xyz.get("top_right")
    br = corners_xyz.get("bottom_right")
    bl = corners_xyz.get("bottom_left")

    bary_estimates = [
        (tl["xyz_mm"][0] + half_w, tl["xyz_mm"][1] + half_h) if tl else None,
        (tr["xyz_mm"][0] - half_w, tr["xyz_mm"][1] + half_h) if tr else None,
        (br["xyz_mm"][0] - half_w, br["xyz_mm"][1] - half_h) if br else None,
        (bl["xyz_mm"][0] + half_w, bl["xyz_mm"][1] - half_h) if bl else None,
    ]
    valid_bary = [b for b in bary_estimates if b is not None]
    bary_x = float(np.mean([b[0] for b in valid_bary])) if valid_bary else None
    bary_y = float(np.mean([b[1] for b in valid_bary])) if valid_bary else None

    corners_list = [c for c in [tl, tr, br, bl] if c is not None]
    centroid_3d = (
        np.array([c["xyz_mm"] for c in corners_list]).mean(axis=0)
        if corners_list else None
    )

    # ── 8. Assemble plane_info ────────────────────────────────────────────────
    result["plane_info"] = {
        "normal":       normal.tolist(),
        "d_mm":         d,
        "rmse_mm":      rmse,
        "corners":      {
            name: info["xyz_mm"] if info else None
            for name, info in corners_xyz.items()
        },
        "bary_TL":      bary_estimates[0],
        "bary_TR":      bary_estimates[1],
        "bary_BR":      bary_estimates[2],
        "bary_BL":      bary_estimates[3],
        "centroid_xy":  (bary_x, bary_y),
        "centroid_3d":  centroid_3d.tolist() if centroid_3d is not None else None,
        "pitch":        pitch,
        "yaw":          yaw,
        "roll":         roll,
    }

    return result


# =============================================================================
# build_meta  (pure — assembles the JSON-serialisable meta dict)
# =============================================================================

def build_meta(
    obj_idx: int,
    obj_data: dict,
    layer_name: str,
    pose_result: dict,
    ai_result: dict,
    slot_analysis: dict | None,
    X: np.ndarray | None,
    Y: np.ndarray | None,
    Z: np.ndarray | None,
    mask: np.ndarray,
    r0: int, c0: int, r1: int, c1: int,
    corner_z_mm: float | None,
    crate_w_mm: float | None,
    crate_h_mm: float | None,
    mm_per_px_x: float,
    mm_per_px_y: float,
    snap_id: int | None,
    crate_id: int | None,
) -> dict:
    """
    Assemble the flat meta dict that goes into info.json and is returned
    from save_object.

    All parameters come from earlier pipeline steps — no computation here,
    just structure assembly.
    """
    pose2d      = pose_result.get("pose2d")
    in_plane_rot = pose_result.get("in_plane_rot", 0.0)
    plane_info  = pose_result.get("plane_info")

    minr, minc, maxr, maxc = obj_data["bbox"]

    # OBB size source
    if crate_w_mm is not None and crate_h_mm is not None:
        obb_w_px = crate_w_mm / mm_per_px_x
        obb_h_px = crate_h_mm / mm_per_px_y
        size_source = "model"
    elif pose2d is not None:
        obb_w_px = pose2d["width_px"]
        obb_h_px = pose2d["height_px"]
        size_source = "blob"
    else:
        obb_w_px = float(c1 - c0)
        obb_h_px = float(r1 - r0)
        size_source = "bbox"

    if pose2d is not None:
        centroid_col = pose2d["centroid_px"][0]
        centroid_row = pose2d["centroid_px"][1]
        angle_deg    = pose2d["angle_deg"]
    else:
        centroid_col = (c1 - c0) / 2.0
        centroid_row = (r1 - r0) / 2.0
        angle_deg    = 0.0

    # OBB corners
    angle_rad = np.radians(angle_deg)
    hw, hh = obb_w_px / 2.0, obb_h_px / 2.0
    lx, ly =  np.cos(angle_rad),  np.sin(angle_rad)
    sx, sy = -np.sin(angle_rad),  np.cos(angle_rad)
    cx, cy = centroid_col, centroid_row
    obb_corners = np.array([
        [cx - hw * lx - hh * sx, cy - hw * ly - hh * sy],
        [cx + hw * lx - hh * sx, cy + hw * ly - hh * sy],
        [cx + hw * lx + hh * sx, cy + hw * ly + hh * sy],
        [cx - hw * lx + hh * sx, cy - hw * ly + hh * sy],
    ], dtype=np.float32)

    # Barycenter from point cloud
    bary_x = bary_y = bary_z = None
    if X is not None:
        xs = X[mask]; ys = Y[mask]; zs = Z[mask]
        valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs) & (zs != 0)
        valid_xy = np.isfinite(xs) & np.isfinite(ys)
        if valid_xy.sum() > 0:
            bary_x = float(xs[valid_xy].mean())
            bary_y = float(ys[valid_xy].mean())
        bary_z = float(corner_z_mm) if corner_z_mm is not None else None

    meta: dict = {
        "snap_id":    snap_id,
        "crate_id":   crate_id,
        "layer":      layer_name,
        "object_id":  obj_idx,
        "bbox_pixels": {
            "row_min": int(minr), "row_max": int(maxr),
            "col_min": int(minc), "col_max": int(maxc),
            "height":  int(obj_data["height"]),
            "width":   int(obj_data["width"]),
        },
        "area_pixels":  int(obj_data["area"]),
        "aspect_ratio": float(obj_data["aspect"]),
        "centroid_pixels": {
            "row": float(obj_data["centroid"][0]),
            "col": float(obj_data["centroid"][1]),
        },
        "plane_info":     plane_info,
        "pose_2d": {
            "method":          "interior_rect + model size",
            "angle_deg":       float(in_plane_rot),
            "centroid_px":     {"col": float(centroid_col), "row": float(centroid_row)},
            "obb_size_source": size_source,
            "obb_w_px":        float(obb_w_px),
            "obb_h_px":        float(obb_h_px),
            "obb_w_mm":        float(crate_w_mm) if crate_w_mm else None,
            "obb_h_mm":        float(crate_h_mm) if crate_h_mm else None,
            "obb_corners_px":  obb_corners.tolist(),
        },
        "ai_verification": ai_result,
    }

    if bary_x is not None:
        meta["barycenter_mm"] = {"x": bary_x, "y": bary_y, "z": bary_z}

    if slot_analysis is not None:
        n_filled  = sum(1 for s in slot_analysis["slots"] if s["status"] == "filled")
        n_empty   = sum(1 for s in slot_analysis["slots"] if s["status"] == "empty")
        n_unknown = sum(1 for s in slot_analysis["slots"] if s["status"] == "unknown")
        meta["slot_analysis"] = {
            "grid":              f"{slot_analysis['n_slot_cols']}x{slot_analysis['n_slot_rows']}",
            "ref_z_mm":          slot_analysis["ref_z_mm"],
            "fill_threshold_mm": slot_analysis["fill_threshold_mm"],
            "n_filled":          n_filled,
            "n_empty":           n_empty,
            "n_unknown":         n_unknown,
            "slots":             slot_analysis["slots"],
        }

    return meta


# =============================================================================
# save_object  (I/O orchestrator — calls estimate_pose + build_meta + writes)
# =============================================================================

def save_object(
    obj_idx: int,
    obj_data: dict,
    layer_name: str,
    amp_gray: np.ndarray,
    dist_gray: np.ndarray,
    X: np.ndarray | None,
    Y: np.ndarray | None,
    Z: np.ndarray | None,
    out_dir: str,
    corner_z_mm: float | None = None,
    crate_w_mm: float | None = None,
    crate_h_mm: float | None = None,
    mm_per_px_x: float = 1.0,
    mm_per_px_y: float = 1.0,
    snap_id: int | None = None,
    crate_id: int | None = None,
    cfg: CrateVisionConfig | None = None,
) -> dict:
    """
    Save all data for one detected object and return its meta dict.

    Writes to *out_dir/<layer_name>_obj<NN>/*:
      amplitude_crop.png, distance_crop.png, amplitude_obb.png,
      mask.npy, pointcloud.npy, plane_fit.png, slot_analysis.png, info.json

    Parameters
    ----------
    obj_idx      : object index within the cell
    obj_data     : dict from ccl_on_mask
    layer_name   : human-readable layer label
    amp_gray     : (H, W) float [0,1] amplitude
    dist_gray    : (H, W) float [0,1] normalised depth
    X, Y, Z      : (H, W) full-image coordinate arrays (may be None)
    out_dir      : parent output directory
    corner_z_mm  : reference Z of the crate layer (mm)
    crate_w_mm   : known crate width  (mm) — for OBB sizing
    crate_h_mm   : known crate height (mm) — for OBB sizing
    mm_per_px_x  : horizontal mm/pixel
    mm_per_px_y  : vertical   mm/pixel
    snap_id      : snapshot integer ID
    crate_id     : global crate counter value
    cfg          : CrateVisionConfig (defaults to singleton)
    """
    if cfg is None:
        cfg = get_config()

    obj_folder = os.path.join(out_dir, f"{layer_name}_obj{obj_idx:02d}")
    os.makedirs(obj_folder, exist_ok=True)

    mask                    = obj_data["mask"]
    minr, minc, maxr, maxc  = obj_data["bbox"]
    H, W                    = amp_gray.shape

    # ── Crop bounds ────────────────────────────────────────────────────────────
    r0, r1 = max(0, minr), min(H, maxr)
    c0, c1 = max(0, minc), min(W, maxc)

    amp_crop  = amp_gray [r0:r1, c0:c1]
    dist_crop = dist_gray[r0:r1, c0:c1]
    mask_crop = mask     [r0:r1, c0:c1]

    # ── Save amplitude / depth crops ──────────────────────────────────────────
    np.save(os.path.join(obj_folder, "mask.npy"), mask)
    Image.fromarray((amp_crop  * 255).astype(np.uint8)).save(
        os.path.join(obj_folder, "amplitude_crop.png"))
    Image.fromarray((dist_crop * 255).astype(np.uint8)).save(
        os.path.join(obj_folder, "distance_crop.png"))

    # ── AI verification ────────────────────────────────────────────────────────
    ai_result = get_verifier(cfg).verify(amp_crop)

    # ── Pose estimation ────────────────────────────────────────────────────────
    debug_rect_dir = os.path.join(obj_folder, "rectangle_debug")
    pose_result = estimate_pose(
        mask=mask, mask_crop=mask_crop, amp_crop=amp_crop,
        X=X, Y=Y, Z=Z,
        r0=r0, c0=c0, r1=r1, c1=c1,
        cfg=cfg,
        debug_rect_dir=debug_rect_dir,
    )

    # ── Save plane-fit debug figure ────────────────────────────────────────────
    rect_xyz       = pose_result.get("rect_xyz")
    rect_mask_full = pose_result.get("rect_mask_full")
    corners_xyz    = pose_result.get("corners_xyz")
    residuals      = pose_result.get("residuals", np.array([]))
    rmse           = pose_result.get("rmse", 0.0)
    normal         = pose_result.get("normal")
    d              = pose_result.get("d", 0.0)

    if rect_xyz is not None and corners_xyz is not None:
        _save_plane_fit_figure(
            obj_folder=obj_folder,
            rect_xyz=rect_xyz,
            rect_mask_full=rect_mask_full,
            mask=mask,
            corners_xyz=corners_xyz,
            residuals=residuals,
            rmse=rmse,
            H=H, W=W,
        )

    # ── Point cloud ────────────────────────────────────────────────────────────
    if X is not None:
        xs = X[mask]; ys = Y[mask]; zs = Z[mask]
        valid = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(zs) & (zs != 0)
        if valid.sum() >= 3:
            pc = np.stack([xs[valid], ys[valid], zs[valid]], axis=1).astype(np.float32)
            np.save(os.path.join(obj_folder, "pointcloud.npy"), pc)

    # ── OBB image ──────────────────────────────────────────────────────────────
    pose2d = pose_result.get("pose2d")
    in_plane_rot = pose_result.get("in_plane_rot", 0.0)

    if crate_w_mm and crate_h_mm:
        obb_w_px = crate_w_mm / mm_per_px_x
        obb_h_px = crate_h_mm / mm_per_px_y
    elif pose2d:
        obb_w_px = pose2d["width_px"]
        obb_h_px = pose2d["height_px"]
    else:
        obb_w_px = float(c1 - c0)
        obb_h_px = float(r1 - r0)

    cx = pose2d["centroid_px"][0] if pose2d else (c1 - c0) / 2.0
    cy = pose2d["centroid_px"][1] if pose2d else (r1 - r0) / 2.0

    _save_obb_image(
        obj_folder=obj_folder,
        amp_crop=amp_crop,
        mask_crop=mask_crop,
        angle_deg=in_plane_rot,
        centroid_col=cx,
        centroid_row=cy,
        obb_w_px=obb_w_px,
        obb_h_px=obb_h_px,
        margin=cfg.obb_canvas_margin,
    )

    # ── Slot analysis ──────────────────────────────────────────────────────────
    slot_analysis = None
    if Z is not None:
        plane_info = pose_result.get("plane_info")
        slot_analysis = analyze_crate_slots(
            Z_full=Z, X_full=X, Y_full=Y,
            bbox_pixels=(r0, c0, r1, c1),
            ref_z_mm=corner_z_mm,
            corners_xyz=corners_xyz,
            plane_normal=plane_info.get("normal") if plane_info and rect_xyz is not None else None,
            plane_d=plane_info.get("d_mm")        if plane_info and rect_xyz is not None else None,
            n_slot_cols=cfg.slot_n_cols,
            n_slot_rows=cfg.slot_n_rows,
            fill_threshold_mm=cfg.slot_fill_threshold_mm,
            min_valid_frac=cfg.slot_min_valid_frac,
            sample_radius_px=cfg.slot_sample_radius_px,
            slot_radius_mm=cfg.slot_radius_mm,
            present_n_pixel=cfg.present_n_pixel,
        )
        save_slot_figure(
            amp_gray, slot_analysis, (r0, c0, r1, c1),
            os.path.join(obj_folder, "slot_analysis.png"),
            output_dpi=cfg.output_dpi,
        )

    # ── Assemble + write meta ──────────────────────────────────────────────────
    meta = build_meta(
        obj_idx=obj_idx,
        obj_data=obj_data,
        layer_name=layer_name,
        pose_result=pose_result,
        ai_result=ai_result,
        slot_analysis=slot_analysis,
        X=X, Y=Y, Z=Z,
        mask=mask,
        r0=r0, c0=c0, r1=r1, c1=c1,
        corner_z_mm=corner_z_mm,
        crate_w_mm=crate_w_mm,
        crate_h_mm=crate_h_mm,
        mm_per_px_x=mm_per_px_x,
        mm_per_px_y=mm_per_px_y,
        snap_id=snap_id,
        crate_id=crate_id,
    )

    with open(os.path.join(obj_folder, "info.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# =============================================================================
# Private figure helpers
# =============================================================================

def _save_plane_fit_figure(
    obj_folder, rect_xyz, rect_mask_full, mask,
    corners_xyz, residuals, rmse, H, W,
):
    res_norm = (residuals - residuals.min()) / (residuals.max() - residuals.min() + 1e-9)
    colors   = (cm.RdYlGn(res_norm)[:, :3] * 255).astype(np.uint8)

    canvas = np.full((H, W, 3), 30, dtype=np.uint8)
    canvas[mask] = [80, 80, 80]
    rect_rows_full, rect_cols_full = np.where(rect_mask_full)
    for i, (r, c) in enumerate(zip(rect_rows_full, rect_cols_full)):
        canvas[r, c] = colors[i]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    ax = axes[0]
    ax.imshow(canvas, interpolation="nearest")
    for name, info in corners_xyz.items():
        if info is None:
            continue
        fc, fr = info["col_row_full"]
        ax.plot(fc, fr, "w+", markersize=14, markeredgewidth=2)
        ax.annotate(name.replace("_", "\n"),
                    xy=(fc, fr), xytext=(fc + 6, fr - 6),
                    color="white", fontsize=7,
                    arrowprops=dict(arrowstyle="-", color="white", lw=0.8))
    ax.set_title("Plane residuals  (green=on-plane  red=off-plane)",
                 color="white", fontsize=10)
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax3d = fig.add_subplot(1, 2, 2, projection="3d")
    sc = ax3d.scatter(rect_xyz[:, 0], rect_xyz[:, 1], rect_xyz[:, 2],
                      c=rect_xyz[:, 2], cmap="viridis", s=2, alpha=0.6)
    valid_corners = [info["xyz_mm"] for info in corners_xyz.values() if info]
    if len(valid_corners) >= 3:
        ax3d.add_collection3d(
            Poly3DCollection([valid_corners], alpha=0.25, facecolor="cyan", edgecolor="white")
        )
    for name, info in corners_xyz.items():
        if info is None:
            continue
        x, y, z = info["xyz_mm"]
        ax3d.scatter([x], [y], [z], c="white", s=40, zorder=5)
        label = "".join(p[0].upper() for p in name.split("_"))
        ax3d.text(x, y, z, f"  {label}", color="white", fontsize=7)

    ax3d.set_xlabel("X mm", color="white", fontsize=8)
    ax3d.set_ylabel("Y mm", color="white", fontsize=8)
    ax3d.set_zlabel("Z mm", color="white", fontsize=8)
    ax3d.set_title(f"3-D plane fit  RMSE={rmse:.2f} mm", color="white", fontsize=10)
    ax3d.set_facecolor("#1a1a2e")
    ax3d.tick_params(colors="white")
    fig.colorbar(sc, ax=ax3d, shrink=0.5, label="Z mm")

    plt.tight_layout()
    plt.savefig(os.path.join(obj_folder, "plane_fit.png"),
                dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()


def _save_obb_image(
    obj_folder, amp_crop, mask_crop, angle_deg,
    centroid_col, centroid_row, obb_w_px, obb_h_px, margin,
):
    angle_rad = np.radians(angle_deg)
    hw, hh = obb_w_px / 2.0, obb_h_px / 2.0
    lx, ly =  np.cos(angle_rad),  np.sin(angle_rad)
    sx, sy = -np.sin(angle_rad),  np.cos(angle_rad)
    cx, cy = centroid_col, centroid_row

    corners_px = np.array([
        [cx - hw*lx - hh*sx, cy - hw*ly - hh*sy],
        [cx + hw*lx - hh*sx, cy + hw*ly - hh*sy],
        [cx + hw*lx + hh*sx, cy + hw*ly + hh*sy],
        [cx - hw*lx + hh*sx, cy - hw*ly + hh*sy],
    ], dtype=np.float32)

    obb_col_min = float(corners_px[:, 0].min())
    obb_col_max = float(corners_px[:, 0].max())
    obb_row_min = float(corners_px[:, 1].min())
    obb_row_max = float(corners_px[:, 1].max())

    canvas_w = max(int(np.ceil(obb_col_max - obb_col_min)) + 2 * margin + 1, 1)
    canvas_h = max(int(np.ceil(obb_row_max - obb_row_min)) + 2 * margin + 1, 1)

    off_col = -obb_col_min + margin
    off_row = -obb_row_min + margin

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    amp_u8 = (amp_crop * 255).astype(np.uint8)
    obj_rows, obj_cols = np.where(mask_crop)
    dst_c = np.clip(np.round(obj_cols + off_col).astype(int), 0, canvas_w - 1)
    dst_r = np.clip(np.round(obj_rows + off_row).astype(int), 0, canvas_h - 1)
    vals = amp_u8[obj_rows, obj_cols]
    canvas[dst_r, dst_c] = np.stack([vals, vals, vals], axis=1)

    shifted = corners_px.copy()
    shifted[:, 0] += off_col
    shifted[:, 1] += off_row
    cv2.polylines(canvas, [shifted.astype(np.int32).reshape((-1, 1, 2))],
                  isClosed=True, color=(0, 255, 0), thickness=1)

    cx_c = centroid_col + off_col
    cx_r = centroid_row + off_row
    for (dx, dy, half_len, color) in [
        (lx, ly, hw, (255, 60,  60)),
        (sx, sy, hh, (60,  255, 60)),
    ]:
        tip_c = int(np.clip(round(cx_c + dx * half_len), 0, canvas_w - 1))
        tip_r = int(np.clip(round(cx_r + dy * half_len), 0, canvas_h - 1))
        org_c = int(np.clip(round(cx_c), 0, canvas_w - 1))
        org_r = int(np.clip(round(cx_r), 0, canvas_h - 1))
        cv2.arrowedLine(canvas, (org_c, org_r), (tip_c, tip_r), color, 1, tipLength=0.25)

    cv2.circle(canvas,
               (int(np.clip(round(cx_c), 0, canvas_w - 1)),
                int(np.clip(round(cx_r), 0, canvas_h - 1))),
               2, (255, 255, 255), -1)

    Image.fromarray(canvas).save(os.path.join(obj_folder, "amplitude_obb.png"))
