"""
crate_vision/detection/geometry.py
====================================
Pure geometry functions:
  - Interior rectangle fitting (flood-fill + bounding box)
  - Corner detection with quadrant constraint + min-Z refinement
  - Plane fitting (SVD) + normal / rotation estimation
  - KDTree helpers for point lookup
  - Grid anchor computation from image zone
  - mm-per-pixel scale from FOV

All functions are pure (no I/O, no global state, no CONFIG dependency).
"""

from __future__ import annotations

import os
from collections import deque

import cv2
import numpy as np
from scipy.spatial import KDTree


# =============================================================================
# mm-per-pixel scale
# =============================================================================

def compute_mm_per_pixel_theoretical(
    z_coords: np.ndarray,
    mask: np.ndarray,
    fov_h_deg: float,
    fov_v_deg: float,
) -> tuple[float, float]:
    """
    Estimate mm/pixel using the median Z of *mask* pixels and camera FOV.

    Parameters
    ----------
    z_coords  : (H, W) float — Z values in mm
    mask      : (H, W) bool — depth-layer mask
    fov_h_deg : horizontal field-of-view in degrees
    fov_v_deg : vertical   field-of-view in degrees

    Returns
    -------
    mm_per_px_x, mm_per_px_y : horizontal and vertical scales
    """
    H, W = z_coords.shape
    z_vals = z_coords[mask]
    z_median = np.median(z_vals[np.isfinite(z_vals)])

    fov_h_rad = np.radians(fov_h_deg)
    fov_v_rad = np.radians(fov_v_deg)

    mm_per_px_x = (2 * z_median * np.tan(fov_h_rad / 2)) / W
    mm_per_px_y = (2 * z_median * np.tan(fov_v_rad / 2)) / H

    return mm_per_px_x, mm_per_px_y


# =============================================================================
# KDTree point lookup
# =============================================================================

def build_kdtree_for_mask(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    mask: np.ndarray,
) -> tuple[KDTree, np.ndarray, np.ndarray]:
    """
    Build a KDTree restricted to pixels inside *mask*.

    Parameters
    ----------
    x_coords : (H, W) float
    y_coords : (H, W) float
    mask     : (H, W) bool — depth-layer mask

    Returns
    -------
    tree      : KDTree
    rows_flat : 1-D int array of row indices of masked pixels
    cols_flat : 1-D int array of col indices of masked pixels

    Raises
    ------
    ValueError if no finite XY pixels exist inside the mask.
    """
    rows_flat, cols_flat = np.where(mask)
    x_flat = x_coords[rows_flat, cols_flat]
    y_flat = y_coords[rows_flat, cols_flat]

    valid = np.isfinite(x_flat) & np.isfinite(y_flat)
    rows_flat = rows_flat[valid]
    cols_flat = cols_flat[valid]
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]

    if len(x_flat) == 0:
        raise ValueError(
            "No valid pixels with finite X/Y coordinates in this layer's mask."
        )

    tree = KDTree(np.column_stack([x_flat, y_flat]))
    return tree, rows_flat, cols_flat

def find_closest_pixel5(
    tree: KDTree,
    rows_flat: np.ndarray,
    cols_flat: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    target_pixel: tuple[int, int],
) -> tuple[int, int, float, float, float, float]:
    """
    Return the pixel in *tree* nearest to the specified target pixel.

    Parameters:
    -----------
    tree : KDTree
        The K-Dimensional Tree used for querying.
    rows_flat : np.ndarray
        A flattened array of row indices.
    cols_flat : np.ndarray
        A flattened array of column indices.
    x_coords : np.ndarray
        A 2D array containing x coordinates.
    y_coords : np.ndarray
        A 2D array containing y coordinates.
    z_coords : np.ndarray
        A 2D array containing z coordinates (if applicable).
    target_pixel : tuple[int, int]
        The target pixel coordinate as a tuple of (row, col).

    Returns
    -------
    pixel_col, pixel_row, actual_x, actual_y, actual_z, dist_mm
    """
    # Extract the target row and column from the target_pixel tuple
    pixel_row, pixel_col = target_pixel

    # Query the KDTree with the target coordinates
    target_point = np.array([x_coords[pixel_row, pixel_col], y_coords[pixel_row, pixel_col]])
    dist_mm, idx = tree.query(target_point)



    # Return the details of the closest pixel
    return (
        pixel_col,
        pixel_row,
        float(x_coords[pixel_row, pixel_col]),
        float(y_coords[pixel_row, pixel_col]),
        float(z_coords[pixel_row, pixel_col]) if z_coords is not None else None,
        float(dist_mm)
    )

def find_closest_pixel(
    tree: KDTree,
    rows_flat: np.ndarray,
    cols_flat: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: np.ndarray,
    target_x_mm: float,
    target_y_mm: float,
) -> tuple[int, int, float, float, float, float]:
    """
    Return the pixel in *tree* nearest to *(target_x_mm, target_y_mm)*.

    Returns
    -------
    pixel_col, pixel_row, actual_x, actual_y, actual_z, dist_mm
    """
    dist_mm, idx = tree.query([target_x_mm, target_y_mm])
    pixel_row = int(rows_flat[idx])
    pixel_col = int(cols_flat[idx])
    return (
        pixel_col,
        pixel_row,
        float(x_coords[pixel_row, pixel_col]),
        float(y_coords[pixel_row, pixel_col]),
        float(z_coords[pixel_row, pixel_col]),
        float(dist_mm),
    )


# =============================================================================
# Grid anchor
# =============================================================================

def get_grid_anchor(
    corner_col: int,
    corner_row: int,
    crate_w_px: int,
    crate_h_px: int,
    n_cols: int,
    n_rows: int,
    img_w: int,
    img_h: int,
) -> tuple[int, int, str, int, int]:
    """
    Compute the grid top-left anchor pixel from the detected corner pixel.

    The image is divided into a 3×3 zone grid.  The zone the corner falls
    in determines which "handle" of the grid the corner represents:

        TL(0,0) TC(1,0) TR(2,0)
        ML(0,1) CC(1,1) MR(2,1)
        BL(0,2) BC(1,2) BR(2,2)

    Returns
    -------
    anchor_col, anchor_row, zone_label, col_zone, row_zone
    """
    total_w = crate_w_px * n_cols
    total_h = crate_h_px * n_rows

    col_zone = min(int(corner_col / (img_w / 3)), 2)
    row_zone = min(int(corner_row / (img_h / 3)), 2)

    col_frac = col_zone / 2.0
    row_frac = row_zone / 2.0

    anchor_col = int(round(corner_col - col_frac * total_w))
    anchor_row = int(round(corner_row - row_frac * total_h))

    zone_names = {
        (0, 0): "Top-Left",    (1, 0): "Top-Center",    (2, 0): "Top-Right",
        (0, 1): "Mid-Left",    (1, 1): "Center",         (2, 1): "Mid-Right",
        (0, 2): "Bottom-Left", (1, 2): "Bottom-Center",  (2, 2): "Bottom-Right",
    }
    zone_label = zone_names.get((col_zone, row_zone), "Unknown")

    return anchor_col, anchor_row, zone_label, col_zone, row_zone


# =============================================================================
# Interior rectangle fitting
# =============================================================================

def fit_min_area_rect(
    pixel_mask: np.ndarray,
    padding: int = 5,
    debug_dir: str | None = None,
) -> dict | None:
    """
    Detect the interior of *pixel_mask* (flood fill, ≥80% black neighbours)
    and return an axis-aligned padded bounding box around it.

    Parameters
    ----------
    pixel_mask : (H, W) bool
    padding    : pixels to add around the interior bounding box
    debug_dir  : if given, saves four debug PNG files here

    Returns
    -------
    dict with keys:
        centroid_px      : (col, row) centre of padded rectangle
        angle_deg        : 0.0  (fixed axis-aligned)
        width_px, height_px
        corners_px       : (4, 2) float32  TL, TR, BR, BL in (col, row)
        bbox_px          : (min_col, min_row, max_col, max_row) interior bbox
        interior_mask    : (H, W) bool
        rect_pixels_mask : (H, W) bool — pixels with value 1 strictly inside rect
    None if fewer than 3 pixels or interior too small.
    """
    import matplotlib.pyplot as plt

    grid = np.asarray(pixel_mask, dtype=int)
    H, W = grid.shape
    deltas_8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    # ── Debug helper ──────────────────────────────────────────────────────────
    def _save_debug(name: str, rgb: np.ndarray, seed_rc=None, corners=None, title=""):
        if debug_dir is None:
            return
        os.makedirs(debug_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.imshow(rgb, interpolation="nearest")
        if seed_rc is not None:
            ax.plot(seed_rc[1], seed_rc[0], "r+", markersize=14, markeredgewidth=2)
        if corners is not None:
            poly = plt.Polygon(corners, closed=True, linewidth=2,
                               edgecolor="#ff4444", facecolor="none")
            ax.add_patch(poly)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, name), dpi=150,
                    bbox_inches="tight", facecolor="#1a1a2e")
        plt.close()

    def _make_rgb(g, interior=None, rect_pixels=None):
        rgb = np.ones((H, W, 3))
        rgb[g == 0] = [0.93, 0.93, 0.93]
        rgb[g == 1] = [0.75, 0.75, 0.75]
        if interior is not None:
            rgb[interior] = [0.55, 0.75, 0.95]
        if rect_pixels is not None:
            rgb[rect_pixels] = [0.35, 0.85, 0.45]
        return rgb

    # ── Step 1 — seed from centroid of white pixels ───────────────────────────
    white_coords = np.argwhere(grid == 1)
    if len(white_coords) < 3:
        return None

    seed_r = int(np.round(white_coords[:, 0].mean()))
    seed_c = int(np.round(white_coords[:, 1].mean()))
    _save_debug("01_grid.png", _make_rgb(grid), seed_rc=(seed_r, seed_c),
                title=f"01 — grid | seed=({seed_r},{seed_c})")

    # ── Step 2 — flood fill interior (≥80% black neighbours) ──────────────────
    def _qualifies(r, c):
        nb = [(r + dr, c + dc) for dr, dc in deltas_8
              if 0 <= r + dr < H and 0 <= c + dc < W]
        return sum(1 for nr, nc in nb if grid[nr, nc] == 0) / len(nb) >= 0.8

    interior = np.zeros((H, W), dtype=bool)
    visited = np.zeros((H, W), dtype=bool)
    queue = deque([(seed_r, seed_c)])
    visited[seed_r, seed_c] = True

    while queue:
        r, c = queue.popleft()
        if grid[r, c] != 0:
            continue
        if _qualifies(r, c):
            interior[r, c] = True
            for dr, dc in deltas_8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc] and grid[nr, nc] == 0:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

    _save_debug("02_interior.png", _make_rgb(grid, interior=interior),
                seed_rc=(seed_r, seed_c),
                title=f"02 — interior | {interior.sum()} px")

    if interior.sum() < 3:
        return None

    # ── Step 3 — padded bounding box ──────────────────────────────────────────
    ic = np.argwhere(interior)
    min_r = int(ic[:, 0].min()); max_r = int(ic[:, 0].max())
    min_c = int(ic[:, 1].min()); max_c = int(ic[:, 1].max())

    min_r_pad = max(0, min_r - padding)
    max_r_pad = min(H - 1, max_r + padding)
    min_c_pad = max(0, min_c - padding)
    max_c_pad = min(W - 1, max_c + padding)

    width  = max_c_pad - min_c_pad + 1
    height = max_r_pad - min_r_pad + 1

    corners = np.array([
        [min_c_pad, min_r_pad],
        [max_c_pad, min_r_pad],
        [max_c_pad, max_r_pad],
        [min_c_pad, max_r_pad],
    ], dtype=np.float32)

    centroid_c = (min_c_pad + max_c_pad) / 2.0
    centroid_r = (min_r_pad + max_r_pad) / 2.0

    # ── Step 4 — pixels with value 1 strictly inside padded rect ──────────────
    rect_pixels_mask = np.zeros((H, W), dtype=bool)
    rect_pixels_mask[
        min_r_pad + 1: max_r_pad,
        min_c_pad + 1: max_c_pad,
    ] = (grid[min_r_pad + 1: max_r_pad, min_c_pad + 1: max_c_pad] == 1)

    _save_debug("03_result.png", _make_rgb(grid, interior=interior),
                seed_rc=(seed_r, seed_c), corners=corners,
                title=f"03 — result | pad={padding} size={width}×{height}")
    _save_debug("04_rect_pixels.png",
                _make_rgb(grid, interior=interior, rect_pixels=rect_pixels_mask),
                seed_rc=(seed_r, seed_c), corners=corners,
                title=f"04 — rect pixels | {rect_pixels_mask.sum()} px")

    return {
        "centroid_px":       (float(centroid_c), float(centroid_r)),
        "angle_deg":          0.0,
        "width_px":           float(width),
        "height_px":          float(height),
        "corners_px":         corners,
        "bbox_px":            (int(min_c), int(min_r), int(max_c), int(max_r)),
        "interior_mask":      interior,
        "rect_pixels_mask":   rect_pixels_mask,
    }


# =============================================================================
# Corner detection
# =============================================================================

def detect_corners(
    pose2d: dict,
    mask_crop: np.ndarray,
    rect_mask_crop: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    r0: int,
    c0: int,
    min_neighbours: int,
    search_radius_px: int,
) -> dict[str, dict | None]:
    """
    Find the four crate corner pixels from the rectangle mask using
    quadrant constraints, then refine to the local min-Z pixel.

    Parameters
    ----------
    pose2d           : output of fit_min_area_rect
    mask_crop        : (crop_H, crop_W) bool — full object mask in crop coords
    rect_mask_crop   : (crop_H, crop_W) bool — rect interior pixels
    X, Y, Z          : (H, W) full-image coordinate arrays
    r0, c0           : crop top-left in full-image coordinates
    min_neighbours   : min 8-neighbour count to accept a corner candidate
    search_radius_px : search radius (px) for min-Z refinement

    Returns
    -------
    dict with keys "top_left", "top_right", "bottom_right", "bottom_left".
    Each value is either None or a dict:
        col_row_crop : (col, row) in crop-local coords
        col_row_full : (col, row) in full-image coords
        xyz_mm       : (x, y, z) tuple in mm
    """
    corners_crop = pose2d["corners_px"]          # (4,2) TL TR BR BL
    corner_names = ["top_left", "top_right", "bottom_right", "bottom_left"]

    rect_rows_crop, rect_cols_crop = np.where(rect_mask_crop)
    rect_rc_crop = np.stack([rect_rows_crop, rect_cols_crop], axis=1)

    mid_r = (corners_crop[:, 1].min() + corners_crop[:, 1].max()) / 2.0
    mid_c = (corners_crop[:, 0].min() + corners_crop[:, 0].max()) / 2.0

    quadrant = {
        "top_left":     (rect_rows_crop <  mid_r) & (rect_cols_crop <  mid_c),
        "top_right":    (rect_rows_crop <  mid_r) & (rect_cols_crop >= mid_c),
        "bottom_right": (rect_rows_crop >= mid_r) & (rect_cols_crop >= mid_c),
        "bottom_left":  (rect_rows_crop >= mid_r) & (rect_cols_crop <  mid_c),
    }

    corners_xyz: dict[str, dict | None] = {}

    for name, (col, row) in zip(corner_names, corners_crop):
        target = np.array([row, col], dtype=np.float32)
        qmask = quadrant[name]

        if qmask.sum() == 0:
            corners_xyz[name] = None
            continue

        candidates = rect_rc_crop[qmask]
        dists = np.linalg.norm(candidates - target, axis=1)
        order = np.argsort(dists)

        # Find first candidate with enough neighbours
        chosen = None
        for idx in order:
            cr, cc = candidates[idx]
            nb_count = sum(
                1 for dr in [-1, 0, 1] for dc in [-1, 0, 1]
                if (dr != 0 or dc != 0)
                and 0 <= cr + dr < rect_mask_crop.shape[0]
                and 0 <= cc + dc < rect_mask_crop.shape[1]
                and rect_mask_crop[cr + dr, cc + dc]
            )
            if nb_count >= min_neighbours:
                chosen = (int(cr), int(cc))
                break

        if chosen is None:
            corners_xyz[name] = None
            continue

        # Refine to min-Z in search_radius around chosen
        chosen_r, chosen_c = chosen
        chosen_center = np.array([chosen_r, chosen_c], dtype=np.float32)
        dists_from_chosen = np.linalg.norm(rect_rc_crop - chosen_center, axis=1)
        nearby_mask = dists_from_chosen <= search_radius_px

        if nearby_mask.sum() > 0:
            nearby = rect_rc_crop[nearby_mask]
            cand_z = np.array([float(Z[cr + r0, cc + c0]) for cr, cc in nearby])
            best = np.argmin(cand_z)
            chosen_r, chosen_c = int(nearby[best][0]), int(nearby[best][1])

        fr, fc = chosen_r + r0, chosen_c + c0
        corners_xyz[name] = {
            "col_row_crop": (chosen_c, chosen_r),
            "col_row_full": (fc, fr),
            "xyz_mm": (float(X[fr, fc]), float(Y[fr, fc]), float(Z[fr, fc])),
        }

    return corners_xyz


# =============================================================================
# Plane fitting + orientation
# =============================================================================

def fit_plane_svd(
    points: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Least-squares plane fit to *points* via SVD.

    Parameters
    ----------
    points : (N, 3) float array

    Returns
    -------
    normal    : (3,) unit normal pointing toward camera (+Z)
    d         : plane offset  (normal · point = d for any point on plane)
    residuals : (N,) signed distances of each point from the plane
    """
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    normal /= np.linalg.norm(normal)
    d = float(normal @ centroid)
    residuals = (points @ normal) - d
    return normal, d, residuals


def compute_normal_from_corners(
    corners_xyz: dict[str, dict | None],
) -> np.ndarray:
    """
    Compute the crate plane normal from the available corner XYZ points.

    Falls back to ``[0, 0, 1]`` if fewer than 3 corners are available.

    Parameters
    ----------
    corners_xyz : dict from detect_corners

    Returns
    -------
    (3,) unit normal, Z-component positive (pointing toward camera).
    """
    available = {
        k: np.array(v["xyz_mm"])
        for k, v in corners_xyz.items()
        if v is not None
    }

    if len(available) < 3:
        return np.array([0.0, 0.0, 1.0])

    pts = np.array(list(available.values()))

    if len(pts) == 3:
        p0, p1, p2 = pts
        normal = np.cross(p1 - p0, p2 - p0)
    else:
        centroid = pts.mean(axis=0)
        _, _, Vt = np.linalg.svd(pts - centroid)
        normal = Vt[-1]

    if normal[2] < 0:
        normal = -normal
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


def compute_in_plane_rotation(
    corners_xyz: dict[str, dict | None],
    normal: np.ndarray,
) -> float | None:
    """
    Estimate the crate's in-plane rotation angle (degrees) relative to
    the camera X axis, using all available adjacent corner pairs.

    Returns None if no edge pairs are available.
    """
    ADJACENT_PAIRS = [
        ("bottom_left",  "bottom_right", False),
        ("top_left",     "top_right",    False),
        ("bottom_left",  "top_left",     True),
        ("bottom_right", "top_right",    True),
    ]

    ref = np.array([1.0, 0.0, 0.0])
    ref_in_plane = ref - np.dot(ref, normal) * normal
    norm_ref = np.linalg.norm(ref_in_plane)
    if norm_ref < 1e-9:
        return None
    ref_in_plane /= norm_ref

    edge_angles = []
    for name_a, name_b, is_vertical in ADJACENT_PAIRS:
        ca = corners_xyz.get(name_a)
        cb = corners_xyz.get(name_b)
        if ca is None or cb is None:
            continue

        pa = np.array(ca["xyz_mm"])
        pb = np.array(cb["xyz_mm"])
        edge = pb - pa
        edge_in_plane = edge - np.dot(edge, normal) * normal
        norm_e = np.linalg.norm(edge_in_plane)
        if norm_e < 1e-6:
            continue
        edge_in_plane /= norm_e

        cos_a = np.clip(np.dot(edge_in_plane, ref_in_plane), -1, 1)
        cross = np.cross(ref_in_plane, edge_in_plane)
        sign = np.sign(np.dot(cross, normal))
        angle = float(np.degrees(np.arccos(cos_a))) * sign

        if is_vertical:
            angle -= 90.0
        angle = (angle + 90) % 180 - 90
        edge_angles.append(angle)

    if not edge_angles:
        return None

    angles = np.array(edge_angles)
    mean_sin = np.mean(np.sin(np.radians(angles)))
    mean_cos = np.mean(np.cos(np.radians(angles)))
    return float(np.degrees(np.arctan2(mean_sin, mean_cos)))


def compute_orientation_angles(
    normal: np.ndarray,
) -> tuple[float, float, float]:
    """
    Decompose *normal* into yaw / pitch / roll angles (degrees).

    Returns
    -------
    yaw_deg, pitch_deg, roll_deg
    """
    yaw_deg   = float(np.degrees(np.arctan2(normal[1], normal[0])))
    pitch_deg = float(np.degrees(np.arctan2(normal[0], normal[2])))
    roll_deg  = float(np.degrees(np.arctan2(normal[1], normal[2])))
    return yaw_deg, pitch_deg, roll_deg
