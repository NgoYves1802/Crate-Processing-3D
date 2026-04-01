"""
crate_vision/pipeline.py
=========================
Main processing pipeline.

Entry point:
    from crate_vision.pipeline import process_depth_layers
    result = process_depth_layers("snapshots/snap0001_...")

The pipeline wires together all sub-modules:
  io/loader         → load images + XYZ
  detection/depth   → depth masks
  detection/ccl     → blob detection + size filter
  detection/geometry→ KDTree, grid anchor, mm/pixel scale
  pose              → per-object pose + slot analysis + file writing
  io/serializer     → JSON output
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

from crate_vision.config import CrateVisionConfig, get_config
from crate_vision.detection.ccl import ccl_on_mask, size_filter
from crate_vision.detection.depth import (
    apply_mask_to_image,
    create_depth_masks,
    remove_small_blobs,
)
from crate_vision.detection.geometry import (
    build_kdtree_for_mask,
    compute_mm_per_pixel_theoretical,
    find_closest_pixel,
    get_grid_anchor,
)
from crate_vision.io.loader import load_depth_and_amplitude
from crate_vision.io.serializer import build_crate_row, write_crate_scans_json
from crate_vision.pose import save_object


# =============================================================================
# Crate grid drawing
# =============================================================================

def draw_crate_grid(
    image: np.ndarray,
    corner_col: int,
    corner_row: int,
    crate_w_mm: float,
    crate_h_mm: float,
    n_cols: int,
    n_rows: int,
    mm_per_px_x: float,
    mm_per_px_y: float,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> tuple[np.ndarray, int, int, int, int, str]:
    """
    Draw a (n_cols × n_rows) crate grid on *image*.

    Returns
    -------
    out, crate_w_px, crate_h_px, anchor_col, anchor_row, zone_label
    """
    out = image.copy()
    H, W = out.shape[:2]

    crate_w_px = int(round(crate_w_mm / mm_per_px_x))
    crate_h_px = int(round(crate_h_mm / mm_per_px_y))

    anchor_col, anchor_row, zone_label, col_zone, row_zone = get_grid_anchor(
        corner_col, corner_row,
        crate_w_px, crate_h_px,
        n_cols, n_rows, W, H,
    )

    # Zone debug overlay
    zone_color = (180, 180, 0)
    for zc in range(1, 3):
        cv2.line(out, (int(W * zc / 3), 0), (int(W * zc / 3), H - 1), zone_color, 1)
    for zr in range(1, 3):
        cv2.line(out, (0, int(H * zr / 3)), (W - 1, int(H * zr / 3)), zone_color, 1)

    zx1, zy1 = int(W * col_zone / 3), int(H * row_zone / 3)
    zx2, zy2 = int(W * (col_zone + 1) / 3), int(H * (row_zone + 1) / 3)
    overlay = out.copy()
    cv2.rectangle(overlay, (zx1, zy1), (zx2, zy2), (255, 255, 0), -1)
    cv2.addWeighted(overlay, 0.12, out, 0.88, 0, out)

    for r in range(n_rows):
        for c in range(n_cols):
            x1 = anchor_col + c * crate_w_px
            y1 = anchor_row + r * crate_h_px
            x2, y2 = x1 + crate_w_px, y1 + crate_h_px
            if x1 >= W or y1 >= H:
                continue
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)

    cv2.putText(out, f"Zone: {zone_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    return out, crate_w_px, crate_h_px, anchor_col, anchor_row, zone_label


# =============================================================================
# Grid cell cropping + object detection
# =============================================================================

def save_grid_crops(
    image: np.ndarray,
    img_gray: np.ndarray,
    depth_mask: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    anchor_col: int,
    anchor_row: int,
    crate_w_px: int,
    crate_h_px: int,
    n_cols: int,
    n_rows: int,
    layer_name: str,
    save_folder: str,
    object_folder: str,
    padding_px: int,
    min_px: int,
    max_px: int,
    min_aspect: float,
    max_aspect: float,
    corner_z_mm: float | None,
    crate_w_mm: float | None,
    crate_h_mm: float | None,
    mm_per_px_x: float,
    mm_per_px_y: float,
    snap_id: int | None,
    layer_crate_id_start: int,
    cfg: CrateVisionConfig,
) -> list[dict]:
    """
    Crop each grid cell, run CCL + size filter on each cell, then call
    save_object for every accepted blob.

    Returns
    -------
    list of meta dicts from save_object
    """
    H, W = image.shape[:2]

    safe_layer = (
        layer_name.replace(" ", "_")
                  .replace("(", "").replace(")", "")
                  .replace("\u00b1", "pm")
                  .replace("/", "-")
    )

    amp_norm = (
        img_gray.astype(float) / 255.0
        if img_gray.max() > 1
        else img_gray.astype(float)
    )

    z_finite = np.where(np.isfinite(z_coords), z_coords, 0.0)
    z_max = z_finite.max()
    dist_norm = (z_finite / z_max) if z_max > 0 else z_finite

    collected_metas: list[dict] = []
    crate_id_counter = layer_crate_id_start

    for r in range(n_rows):
        for c in range(n_cols):
            x1_orig = anchor_col + c * crate_w_px
            y1_orig = anchor_row + r * crate_h_px
            x2_orig = x1_orig + crate_w_px
            y2_orig = y1_orig + crate_h_px

            if x1_orig >= W or y1_orig >= H or x2_orig <= 0 or y2_orig <= 0:
                continue

            x1c = max(0, x1_orig - padding_px);  x2c = min(W, x2_orig + padding_px)
            y1c = max(0, y1_orig - padding_px);  y2c = min(H, y2_orig + padding_px)

            crop_img  = image     [y1c:y2c, x1c:x2c]
            crop_mask = depth_mask[y1c:y2c, x1c:x2c]
            crop_amp  = amp_norm  [y1c:y2c, x1c:x2c]
            crop_dist = dist_norm [y1c:y2c, x1c:x2c]
            crop_X    = x_coords  [y1c:y2c, x1c:x2c]
            crop_Y    = y_coords  [y1c:y2c, x1c:x2c]
            crop_Z    = z_coords  [y1c:y2c, x1c:x2c]

            if crop_img.size == 0:
                continue

            cell_out_dir = os.path.join(object_folder, safe_layer, f"row{r}_col{c}")
            objects, _, _ = ccl_on_mask(
                crop_mask, crop_amp, min_px, max_px, min_aspect, max_aspect
            )

            for obj_idx, obj_data in enumerate(objects):
                obj_mask = obj_data["mask"]
                xs = crop_X[obj_mask]
                ys = crop_Y[obj_mask]
                valid = np.isfinite(xs) & np.isfinite(ys)
                if valid.sum() == 0:
                    continue

                accepted, _, _ = size_filter(
                    xs[valid], ys[valid],
                    cfg.crate_min_size_mm,
                    cfg.crate_max_size_mm,
                )
                if not accepted:
                    continue

                meta = save_object(
                    obj_idx=obj_idx,
                    obj_data=obj_data,
                    layer_name=layer_name,
                    amp_gray=crop_amp,
                    dist_gray=crop_dist,
                    X=crop_X, Y=crop_Y, Z=crop_Z,
                    out_dir=cell_out_dir,
                    corner_z_mm=corner_z_mm,
                    crate_w_mm=crate_w_mm,
                    crate_h_mm=crate_h_mm,
                    mm_per_px_x=mm_per_px_x,
                    mm_per_px_y=mm_per_px_y,
                    snap_id=snap_id,
                    crate_id=crate_id_counter,
                    cfg=cfg,
                )
                collected_metas.append(meta)
                crate_id_counter += 1

    return collected_metas


# =============================================================================
# Per-layer processing
# =============================================================================

def plot_layer_results(
    original_color: np.ndarray,
    img_gray: np.ndarray,
    z_coord: np.ndarray,
    mask: np.ndarray,
    layer_name: str,
    selected_corner: dict | None,
    grid_params: dict,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    save_folder: str,
    object_folder: str,
    snap_id: int | None,
    layer_crate_id_start: int,
    cfg: CrateVisionConfig,
) -> list[dict]:
    """
    Compute the pixel scale, build the crate grid, and run detection
    for one depth layer.

    Returns a list of meta dicts from save_grid_crops.
    """
    if selected_corner is None or grid_params is None:
        return []

    cx = selected_corner["pixel_col"]
    cy = selected_corner["pixel_row"]

    mm_per_px_x, mm_per_px_y = compute_mm_per_pixel_theoretical(
        z_coord, mask, cfg.fov_h_deg, cfg.fov_v_deg
    )

    _, crate_w_px, crate_h_px, anchor_col, anchor_row, _ = draw_crate_grid(
        original_color,
        cx, cy,
        grid_params["crate_w_mm"],
        grid_params["crate_h_mm"],
        grid_params["n_cols"],
        grid_params["n_rows"],
        mm_per_px_x, mm_per_px_y,
        color=(0, 220, 0), thickness=2,
    )

    return save_grid_crops(
        image=original_color,
        img_gray=img_gray,
        depth_mask=mask,
        x_coords=x_coords,
        y_coords=y_coords,
        z_coords=z_coord,
        anchor_col=anchor_col,
        anchor_row=anchor_row,
        crate_w_px=crate_w_px,
        crate_h_px=crate_h_px,
        n_cols=grid_params["n_cols"],
        n_rows=grid_params["n_rows"],
        layer_name=layer_name,
        save_folder=save_folder,
        object_folder=object_folder,
        padding_px=cfg.ccl_cell_padding_px,
        min_px=cfg.ccl_min_px,
        max_px=cfg.ccl_max_px,
        min_aspect=cfg.ccl_min_aspect,
        max_aspect=cfg.ccl_max_aspect,
        corner_z_mm=selected_corner["actual_z"],
        crate_w_mm=grid_params["crate_w_mm"],
        crate_h_mm=grid_params["crate_h_mm"],
        mm_per_px_x=mm_per_px_x,
        mm_per_px_y=mm_per_px_y,
        snap_id=snap_id,
        layer_crate_id_start=layer_crate_id_start,
        cfg=cfg,
    )


# =============================================================================
# Summary figure
# =============================================================================

def create_summary_figure(
    original_color: np.ndarray,
    z_coord: np.ndarray,
    masks: dict[str, np.ndarray],
    corners_by_layer: dict,
    grid_params: dict,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    save_folder: str,
    cfg: CrateVisionConfig,
) -> None:
    """Save the 3-row summary figure to save_folder/depth_layers_summary.png."""
    n_layers = len(masks)
    fig, axes = plt.subplots(3, n_layers + 1, figsize=(5 * (n_layers + 1), 12))

    axes[0, 0].imshow(original_color)
    axes[0, 0].set_title("Original Image", fontweight="bold")
    axes[0, 0].axis("off")

    depth_display = np.copy(z_coord)
    depth_display[~np.isfinite(depth_display)] = 0
    im = axes[1, 0].imshow(depth_display, cmap="viridis")
    axes[1, 0].set_title("Depth Map", fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[2, 0].axis("off")

    for col, (layer_name, mask) in enumerate(masks.items(), start=1):
        axes[0, col].imshow(mask, cmap="gray")
        axes[0, col].set_title(f"{layer_name}\nMask", fontweight="bold", fontsize=9)
        axes[0, col].axis("off")

        masked_img = apply_mask_to_image(original_color, mask)
        axes[1, col].imshow(masked_img)
        axes[1, col].set_title("Masked Image", fontweight="bold", fontsize=9)
        axes[1, col].axis("off")

        corner = corners_by_layer.get(layer_name)
        if corner is not None and grid_params is not None:
            cx, cy = corner["pixel_col"], corner["pixel_row"]
            mm_per_px_x, mm_per_px_y = compute_mm_per_pixel_theoretical(
                z_coord, mask, cfg.fov_h_deg, cfg.fov_v_deg
            )
            img_grid, _, _, _, _, _ = draw_crate_grid(
                original_color, cx, cy,
                grid_params["crate_w_mm"], grid_params["crate_h_mm"],
                grid_params["n_cols"], grid_params["n_rows"],
                mm_per_px_x, mm_per_px_y,
                color=(0, 220, 0), thickness=2,
            )
            axes[2, col].imshow(img_grid)
            axes[2, col].scatter([cx], [cy], c="red", s=40, marker="o",
                                 edgecolors="white", linewidths=1.0, zorder=5)
            axes[2, col].set_title(
                f"Grid ({grid_params['n_cols']}×{grid_params['n_rows']})\ncorner ({cx},{cy})",
                fontweight="bold", fontsize=8,
            )
        else:
            axes[2, col].imshow(original_color)
            axes[2, col].set_title("No Corner / Grid", fontweight="bold", fontsize=8)
        axes[2, col].axis("off")

    plt.suptitle("Depth Layer Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_folder, "depth_layers_summary.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=cfg.output_dpi, facecolor="white")
    plt.close()


# =============================================================================
# Main entry point
# =============================================================================

def process_depth_layers(
    folder_path: str,
    cfg: CrateVisionConfig | None = None,
) -> dict:
    """
    Full pipeline: load snapshot → depth masks → corner input → detection.

    Parameters
    ----------
    folder_path : str
        Path to the snapshot folder (must contain amplitude.png + xyz_combined.npy).
    cfg : CrateVisionConfig, optional
        Configuration to use.  Defaults to the module-level singleton.

    Returns
    -------
    dict  ``{"snap_id": int, "crates": [<crate_row>, ...]}``
    Also writes ``<save_folder>/crate_scans.json``.
    """
    if cfg is None:
        cfg = get_config()

    # ── Snap ID from folder name ───────────────────────────────────────────────
    snap_folder = Path(folder_path).resolve().name
    match = re.search(r"snap(\d+)", snap_folder)
    snap_id = int(match.group(1)) if match else 0

    save_folder   = os.path.join("delimiter", snap_folder)
    object_folder = os.path.join("object",    snap_folder)
    os.makedirs(save_folder,   exist_ok=True)
    os.makedirs(object_folder, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    x_coords, y_coords, z_coords, img_color, img_gray = load_depth_and_amplitude(folder_path)
    if x_coords is None:
        return {"snap_id": snap_id, "crates": []}

    if z_coords.shape != img_gray.shape:
        zoom_factors = (img_gray.shape[0] / z_coords.shape[0],
                        img_gray.shape[1] / z_coords.shape[1])
        z_coords = zoom(z_coords, zoom_factors, order=1)

    # ── Depth masks ────────────────────────────────────────────────────────────
    masks = create_depth_masks(z_coords, cfg.layer_distances_mm, cfg.layer_half_widths_mm)
    masks = {
        name: remove_small_blobs(mask, cfg.ccl_min_blob_size)
        for name, mask in masks.items()
    }

    grid_params = {
        "crate_w_mm": cfg.crate_width,
        "crate_h_mm": cfg.crate_height,
        "n_cols":     cfg.grid_n_cols,
        "n_rows":     cfg.grid_n_rows,
    }

    # Sort layers furthest-first so crate IDs start from the back
    sorted_layer_names = sorted(
        masks.keys(),
        key=lambda n: -cfg.layer_distances_mm[list(masks.keys()).index(n)],
    )

    corners_by_layer: dict = {}
    all_metas: list[dict] = []
    crate_id_counter = 1

    for layer_name in sorted_layer_names:
        mask = masks[layer_name]
        print(f"\nProcessing layer '{layer_name}' with {mask.sum()} masked pixels...")
        pixel_col, pixel_row = cfg.grid_corner_pixels.get(layer_name, (88, 66))
        print(f"Using corner pixel: ({pixel_col}, {pixel_row})")
        selected_corner = {
                "pixel_col": pixel_col,
                "pixel_row": pixel_row,
                "actual_x":  float(x_coords[pixel_row, pixel_col]),
                "actual_y":  float(y_coords[pixel_row, pixel_col]),
                "actual_z":  float(z_coords[pixel_row, pixel_col]) if z_coords is not None else None,
            }
        corners_by_layer[layer_name] = selected_corner

        layer_metas = plot_layer_results(
            original_color=img_color,
            img_gray=img_gray,
            z_coord=z_coords,
            mask=mask,
            layer_name=layer_name,
            selected_corner=selected_corner,
            grid_params=grid_params,
            x_coords=x_coords,
            y_coords=y_coords,
            save_folder=save_folder,
            object_folder=object_folder,
            snap_id=snap_id,
            layer_crate_id_start=crate_id_counter,
            cfg=cfg,
        )
        all_metas.extend(layer_metas)
        crate_id_counter += len(layer_metas)

    # ── Summary figure ─────────────────────────────────────────────────────────
    create_summary_figure(
        img_color, z_coords, masks,
        corners_by_layer, grid_params,
        x_coords, y_coords,
        save_folder=save_folder,
        cfg=cfg,
    )

    # ── Build result dict ──────────────────────────────────────────────────────
    crate_rows = [build_crate_row(meta) for meta in all_metas]
    result = {"snap_id": snap_id, "crates": crate_rows}
    write_crate_scans_json(result, save_folder)
    return result
