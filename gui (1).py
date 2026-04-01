"""
crate_vision/gui.py
====================
PyQt5 configuration and review interface.

Layout
------
┌─────────────────────────────────────────────────────────────────────┐
│  Left sidebar (280px)  │  Centre (depth preview)  │  Right panel    │
│                        │                          │                  │
│  Snapshot folder       │  Depth summary image     │  Detected crates │
│  ─────────────────     │  (click = set pixel)     │  (scrollable list)│
│  Layer distances       │                          │  ─────────────── │
│  [+] [−] rows          │  Active layer selector   │  Selected crate  │
│                        │  Pixel coords readout    │  preview tabs:   │
│  Half-width slider     │                          │  • Plane fit     │
│                        │                          │  • Slot analysis │
│  Grid layout           │                          │  • Info JSON     │
│  Slot parameters       │                          │                  │
│                        │                          │  Slot params     │
│  [Run Detection]       │                          │  (live sliders)  │
│  [Save Config]         │                          │                  │
└─────────────────────────────────────────────────────────────────────┘

Usage
-----
    python -m crate_vision.gui
    # or
    from crate_vision.gui import launch
    launch()
"""
from __future__ import annotations
# gui (1).py — FIRST LINES, before ANY other imports
import os
import sys

# Windows DLL hell fix: Preload torch before Qt can interfere
if sys.platform == 'win32':
    try:
        import torch
        print(f"Preloaded torch: {torch.__version__}")
    except Exception:
        pass  # Will handle in ai_verifier


import json
import threading
from pathlib import Path

import numpy as np

from PyQt5.QtCore import (
    Qt, QThread, QTimer, pyqtSignal, pyqtSlot
)
from PyQt5.QtGui import (
    QColor, QFont, QImage, QPixmap, QPainter, QPen, QBrush
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QSlider, QSpinBox,
    QDoubleSpinBox, QScrollArea, QFrame, QTabWidget,
    QFileDialog, QMessageBox, QListWidget, QListWidgetItem,
    QGroupBox, QSizePolicy, QProgressBar, QTextEdit,
    QAction, QMenuBar, QStatusBar, QToolButton,
)

from crate_vision.config import CrateVisionConfig, get_config, override_config
from crate_vision.detection.depth import create_depth_masks, remove_small_blobs
from crate_vision.io.loader import load_depth_and_amplitude


# ─────────────────────────────────────────────────────────────────────────────
# Colour palette (dark industrial theme)
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "bg":          "#1a1d23",
    "surface":     "#22262e",
    "surface2":    "#2a2f3a",
    "border":      "#363c4a",
    "accent":      "#4a9eff",
    "accent_dim":  "#2a5a99",
    "green":       "#3dd68c",
    "red":         "#ff5a5a",
    "amber":       "#f5a623",
    "text":        "#e8eaf0",
    "text_dim":    "#8892a4",
    "text_faint":  "#4a5568",
}

BASE_QSS = f"""
QMainWindow, QWidget {{
    background: {PALETTE['bg']};
    color: {PALETTE['text']};
    font-family: "Segoe UI", "SF Pro Display", sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    border: 1px solid {PALETTE['border']};
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    font-weight: 600;
    color: {PALETTE['text_dim']};
    font-size: 11px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}}
QPushButton {{
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['border']};
    border-radius: 5px;
    padding: 6px 14px;
    color: {PALETTE['text']};
    font-weight: 500;
}}
QPushButton:hover {{
    background: {PALETTE['accent_dim']};
    border-color: {PALETTE['accent']};
}}
QPushButton:pressed {{
    background: {PALETTE['accent']};
}}
QPushButton#run_btn {{
    background: {PALETTE['accent']};
    border: none;
    font-weight: 700;
    font-size: 14px;
    padding: 10px;
    color: white;
    border-radius: 6px;
}}
QPushButton#run_btn:hover {{
    background: #6ab4ff;
}}
QPushButton#run_btn:disabled {{
    background: {PALETTE['surface2']};
    color: {PALETTE['text_faint']};
}}
QLineEdit, QSpinBox, QDoubleSpinBox {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 4px;
    padding: 4px 8px;
    color: {PALETTE['text']};
    selection-background-color: {PALETTE['accent']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {PALETTE['accent']};
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {PALETTE['border']};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {PALETTE['accent']};
    border: none;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::sub-page:horizontal {{
    background: {PALETTE['accent']};
    border-radius: 2px;
}}
QListWidget {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 5px;
    outline: none;
}}
QListWidget::item {{
    padding: 8px 12px;
    border-bottom: 1px solid {PALETTE['border']};
}}
QListWidget::item:selected {{
    background: {PALETTE['accent_dim']};
    color: white;
}}
QListWidget::item:hover {{
    background: {PALETTE['surface2']};
}}
QTabWidget::pane {{
    border: 1px solid {PALETTE['border']};
    border-radius: 0 5px 5px 5px;
    background: {PALETTE['surface']};
}}
QTabBar::tab {{
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['border']};
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    padding: 5px 14px;
    color: {PALETTE['text_dim']};
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {PALETTE['surface']};
    color: {PALETTE['text']};
    border-bottom: 1px solid {PALETTE['surface']};
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: {PALETTE['surface']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {PALETTE['border']};
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QTextEdit {{
    background: {PALETTE['surface']};
    border: 1px solid {PALETTE['border']};
    border-radius: 4px;
    color: {PALETTE['text']};
    font-family: "Cascadia Code", "Fira Code", monospace;
    font-size: 12px;
}}
QLabel#section_label {{
    color: {PALETTE['text_dim']};
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}}
QFrame#divider {{
    background: {PALETTE['border']};
    max-height: 1px;
}}
QProgressBar {{
    background: {PALETTE['surface2']};
    border: 1px solid {PALETTE['border']};
    border-radius: 4px;
    text-align: center;
    color: {PALETTE['text']};
    height: 18px;
}}
QProgressBar::chunk {{
    background: {PALETTE['accent']};
    border-radius: 3px;
}}
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _divider() -> QFrame:
    f = QFrame()
    f.setObjectName("divider")
    f.setFrameShape(QFrame.HLine)
    f.setFixedHeight(1)
    return f


def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("section_label")
    return lbl


def _help_label(text: str) -> QLabel:
    """Helper text shown above parameter groups"""
    lbl = QLabel(text)
    lbl.setStyleSheet(f"color:{PALETTE['text_dim']}; font-size:10px; font-style:italic; margin:0 0 6px 0;")
    return lbl


def _badge(text: str, color: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet(
        f"background:{color}; color:white; border-radius:3px;"
        f"padding:1px 6px; font-size:11px; font-weight:700;"
    )
    lbl.setFixedHeight(18)
    return lbl


def _numpy_to_pixmap(arr: np.ndarray, max_w: int = 0, max_h: int = 0) -> QPixmap:
    """Convert a (H,W,3) uint8 RGB array to a QPixmap, optionally scaled."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    img = QImage(arr.data, w, h, w * 3, QImage.Format_RGB888).copy()
    pix = QPixmap.fromImage(img)
    if max_w > 0 or max_h > 0:
        pix = pix.scaled(
            max_w or pix.width(), max_h or pix.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
    return pix


def _load_png_pixmap(path: str, max_w: int = 0, max_h: int = 0) -> QPixmap | None:
    if not os.path.exists(path):
        return None
    pix = QPixmap(path)
    if pix.isNull():
        return None
    if max_w > 0 or max_h > 0:
        pix = pix.scaled(
            max_w or pix.width(), max_h or pix.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
    return pix


# ─────────────────────────────────────────────────────────────────────────────
# Clickable image label
# ─────────────────────────────────────────────────────────────────────────────

class ClickableImageLabel(QLabel):
    """
    QLabel that emits pixel_clicked(col, row) when the user clicks,
    accounting for the scaled pixmap offset inside the label.
    Also draws a crosshair at the last clicked position and a grid overlay.
    Supports zoom in/out.
    """
    pixel_clicked = pyqtSignal(int, int)   # col, row in original image coords

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.CrossCursor)
        self._click_col: int | None = None
        self._click_row: int | None = None
        self._orig_w: int = 1
        self._orig_h: int = 1
        self._grid_config: dict | None = None
        self._zoom_level: float = 1.0  # 1.0 = 100%
        self._orig_pixmap: QPixmap | None = None

    def set_image(self, arr: np.ndarray) -> None:
        self._orig_h, self._orig_w = arr.shape[:2]
        self._orig_pixmap = _numpy_to_pixmap(arr)
        self._zoom_level = 1.0
        self._update_display()

    def _update_display(self) -> None:
        """Update the displayed pixmap based on zoom level."""
        if self._orig_pixmap is None:
            return
        
        if self._zoom_level == 1.0:
            self.setPixmap(self._orig_pixmap)
        else:
            w = int(self._orig_pixmap.width() * self._zoom_level)
            h = int(self._orig_pixmap.height() * self._zoom_level)
            scaled = self._orig_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled)
        self.update()

    def set_zoom(self, zoom_level: float) -> None:
        """Set zoom level (1.0 = 100%, 2.0 = 200%, etc.)."""
        self._zoom_level = max(0.5, min(zoom_level, 4.0))  # Clamp between 50% and 400%
        self._update_display()

    def zoom_in(self) -> None:
        """Zoom in by 25%."""
        self.set_zoom(self._zoom_level * 1.25)

    def zoom_out(self) -> None:
        """Zoom out by 20%."""
        self.set_zoom(self._zoom_level / 1.25)

    def zoom_fit(self) -> None:
        """Reset to fit view (100%)."""
        self.set_zoom(1.0)

    def set_click(self, col: int, row: int, grid_config: dict | None = None) -> None:
        self._click_col = col
        self._click_row = row
        self._grid_config = grid_config
        self.update()

    def clear_click(self) -> None:
        self._click_col = None
        self._click_row = None
        self._grid_config = None
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.pixmap() is None:
            return

        pix = self.pixmap()
        pw, ph = pix.width(), pix.height()
        lw, lh = self.width(), self.height()

        # Offset of the pixmap inside the label (centred)
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2

        mx, my = event.x() - ox, event.y() - oy
        if mx < 0 or my < 0 or mx >= pw or my >= ph:
            return

        # Scale back to original image coordinates
        col = int(mx * self._orig_w / pw)
        row = int(my * self._orig_h / ph)
        col = max(0, min(col, self._orig_w - 1))
        row = max(0, min(row, self._orig_h - 1))

        self._click_col = col
        self._click_row = row
        self.update()
        self.pixel_clicked.emit(col, row)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._click_col is None or self.pixmap() is None:
            return

        pix = self.pixmap()
        pw, ph = pix.width(), pix.height()
        lw, lh = self.width(), self.height()
        ox = (lw - pw) // 2
        oy = (lh - ph) // 2

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw grid if configured
        if self._grid_config is not None:
            self._draw_grid(painter, ox, oy, pw, ph)

        # Crosshair in label coords
        cx = ox + int(self._click_col * pw / self._orig_w)
        cy = oy + int(self._click_row * ph / self._orig_h)

        # Outer ring
        painter.setPen(QPen(QColor("#000000"), 3))
        painter.drawEllipse(cx - 8, cy - 8, 16, 16)
        painter.setPen(QPen(QColor(PALETTE['accent']), 2))
        painter.drawEllipse(cx - 8, cy - 8, 16, 16)

        # Cross lines
        painter.setPen(QPen(QColor("#000000"), 3))
        painter.drawLine(cx - 14, cy, cx + 14, cy)
        painter.drawLine(cx, cy - 14, cx, cy + 14)
        painter.setPen(QPen(QColor(PALETTE['accent']), 1.5))
        painter.drawLine(cx - 14, cy, cx + 14, cy)
        painter.drawLine(cx, cy - 14, cx, cy + 14)
        painter.end()

    def _draw_grid(self, painter: QPainter, ox: int, oy: int, pw: int, ph: int):
        """Draw the crate grid overlay in green."""
        if not self._grid_config or self._click_col is None or self._click_row is None:
            return

        cfg = self._grid_config
        crate_w_px = cfg.get("crate_w_px", 0)
        crate_h_px = cfg.get("crate_h_px", 0)
        n_cols = cfg.get("n_cols", 2)
        n_rows = cfg.get("n_rows", 2)
        anchor_col = cfg.get("anchor_col", 0)
        anchor_row = cfg.get("anchor_row", 0)

        if crate_w_px <= 0 or crate_h_px <= 0:
            return

        # Scale from original coords to pixmap coords
        scale_x = pw / self._orig_w
        scale_y = ph / self._orig_h

        painter.setPen(QPen(QColor("#00aa00"), 2))
        painter.setBrush(QBrush(Qt.NoBrush))

        for r in range(n_rows):
            for c in range(n_cols):
                x1_orig = anchor_col + c * crate_w_px
                y1_orig = anchor_row + r * crate_h_px
                x2_orig = x1_orig + crate_w_px
                y2_orig = y1_orig + crate_h_px

                if x1_orig >= self._orig_w or y1_orig >= self._orig_h or x2_orig <= 0 or y2_orig <= 0:
                    continue

                x1_pix = ox + int(x1_orig * scale_x)
                y1_pix = oy + int(y1_orig * scale_y)
                x2_pix = ox + int(x2_orig * scale_x)
                y2_pix = oy + int(y2_orig * scale_y)

                painter.drawRect(x1_pix, y1_pix, x2_pix - x1_pix, y2_pix - y1_pix)


# ─────────────────────────────────────────────────────────────────────────────
# Background worker thread for detection
# ─────────────────────────────────────────────────────────────────────────────

class DetectionWorker(QThread):
    finished  = pyqtSignal(dict)          # result dict
    error     = pyqtSignal(str)           # error message
    progress  = pyqtSignal(str)           # status text

    def __init__(self, folder_path: str, cfg: CrateVisionConfig):
        super().__init__()
        self._folder = folder_path
        self._cfg    = cfg

    def run(self):
        try:
            from crate_vision.pipeline import process_depth_layers
            self.progress.emit("Loading depth data…")
            result = process_depth_layers(self._folder, cfg=self._cfg)
            self.finished.emit(result)
        except Exception as exc:
            import traceback
            self.error.emit(traceback.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# Focusable spin boxes for parameter explanations
# ─────────────────────────────────────────────────────────────────────────────

class FocusableSpinBox(QSpinBox):
    """QSpinBox that tracks focus for showing parameter explanations"""
    param_focused = pyqtSignal(str)  # param name
    
    def __init__(self, param_name: str = ""):
        super().__init__()
        self.param_name = param_name
    
    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self.param_name:
            self.param_focused.emit(self.param_name)


class FocusableDoubleSpinBox(QDoubleSpinBox):
    """QDoubleSpinBox that tracks focus for showing parameter explanations"""
    param_focused = pyqtSignal(str)  # param name
    
    def __init__(self, param_name: str = ""):
        super().__init__()
        self.param_name = param_name
    
    def focusInEvent(self, event):
        super().focusInEvent(event)
        if self.param_name:
            self.param_focused.emit(self.param_name)


# ─────────────────────────────────────────────────────────────────────────────
# Left sidebar
# ─────────────────────────────────────────────────────────────────────────────

class SidebarWidget(QWidget):
    config_changed   = pyqtSignal()     # any config param changed
    run_requested    = pyqtSignal()
    save_requested   = pyqtSignal()
    folder_changed   = pyqtSignal(str)

    def __init__(self, cfg: CrateVisionConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        # Removed fixed width to allow resizing via splitter
        self._layer_rows: list[tuple[QDoubleSpinBox, QToolButton]] = []
        self._building = False
        
        # Parameter explanations
        self.param_descriptions = {
            # Core grid parameters
            "grid_cols": "Number of columns in the detection grid. Typically 2-4 for crate layouts.",
            "grid_rows": "Number of rows in the detection grid. Typically 2-4 for crate layouts.",
            "crate_w": "Physical width of the crate in millimeters. Measured horizontally.",
            "crate_h": "Physical height of the crate in millimeters. Measured vertically.",
            
            # Camera parameters
            "image_width": "Camera image width in pixels. Used for grid calculations and coordinate mapping.",
            "image_height": "Camera image height in pixels. Used for grid calculations and coordinate mapping.",
            "fov_h": "Horizontal field of view in degrees. Determines how wide the camera can see.",
            "fov_v": "Vertical field of view in degrees. Determines how tall the camera can see.",
            "output_dpi": "Resolution for saving detection output images (DPI).",
            
            # Slot analysis parameters
            "slot_cols": "Number of slot columns within each crate cell. Advanced setting for slot analysis.",
            "slot_rows": "Number of slot rows within each crate cell. Advanced setting for slot analysis.",
            "slot_thresh": "Fill detection threshold in mm. Objects below this height are considered filled slots.",
            "slot_radius": "Radius in mm for searching nearby points. Used in slot occupancy detection.",
            "slot_sample_r": "Sample radius in pixels for slot analysis computations.",
            "slot_min_frac": "Minimum valid fraction of pixels required for slot fill analysis.",
            "present_n": "Minimum pixels required to determine a slot is occupied.",
            
            # CCL parameters
            "ccl_min_blob": "Minimum blob size in pixels for connected component labeling.",
            "ccl_min_px": "Minimum pixel threshold for CCL analysis.",
            "ccl_max_px": "Maximum pixel threshold for CCL analysis.",
            "ccl_min_aspect": "Minimum aspect ratio allowed for blobs (width/height).",
            "ccl_max_aspect": "Maximum aspect ratio allowed for blobs (width/height).",
            "ccl_cell_pad": "Padding in pixels around each cell for independent blob detection.",
            
            # Rectangle fitting
            "rect_pad": "Padding in pixels applied around fitted rectangles.",
            "rect_interior": "Minimum fraction of rectangle that must be interior to the blob (0-1).",
            
            # Corner detection
            "corner_neighbors": "Minimum neighbors required for corner detection (Harris corner detection).",
            "corner_search": "Search radius in pixels for corner detection.",
            
            # Crate filtering
            "crate_min_w": "Minimum detected crate width in mm. Crates smaller than this are filtered out.",
            "crate_max_w": "Maximum detected crate width in mm. Crates larger than this are filtered out.",
            "crate_min_h": "Minimum detected crate height in mm.",
            "crate_max_h": "Maximum detected crate height in mm.",
            "crate_min_d": "Minimum detected crate depth in mm.",
            "crate_max_d": "Maximum detected crate depth in mm.",
            
            # OBB visualization
            "obb_margin": "Canvas margin in pixels for OBB (Oriented Bounding Box) visualization.",
            
            # AI verification
            "ai_conf": "Confidence threshold for AI model predictions (0-1). Higher = stricter.",
        }
        
        self._build_ui()
        self._populate_from_config()

    # ── Parameter explanation system ──────────────────────────────────────────
    
    def _show_help(self, param_name: str):
        """Display explanation for a parameter"""
        text = self.param_descriptions.get(param_name, "")
        if text:
            self.help_label.setText(f"<b>{param_name}:</b> {text}")
    
    # ── build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Scrollable content ────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"QScrollArea {{ border:none; background:{PALETTE['bg']}; }}")
        
        scroll_widget = QWidget()
        scroll_widget.setStyleSheet(f"background:{PALETTE['bg']};")
        root = QVBoxLayout(scroll_widget)
        root.setContentsMargins(12, 14, 12, 14)
        root.setSpacing(14)

        # ── Snapshot folder ───────────────────────────────────────────────────
        root.addWidget(_section_label("Snapshot"))
        folder_row = QHBoxLayout()
        self.folder_edit = QLineEdit()
        self.folder_edit.setPlaceholderText("snapshots/snap0001_…")
        self.folder_edit.textChanged.connect(self._on_folder_changed)
        btn_browse = QToolButton()
        btn_browse.setText("…")
        btn_browse.setFixedWidth(28)
        btn_browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.folder_edit)
        folder_row.addWidget(btn_browse)
        root.addLayout(folder_row)
        root.addWidget(_divider())

        # ── Layer distances ───────────────────────────────────────────────────
        root.addWidget(_section_label("Depth Layers (mm)"))
        root.addWidget(_help_label("Set target depth and tolerance range for each layer"))

        self.layers_container = QVBoxLayout()
        self.layers_container.setSpacing(4)
        root.addLayout(self.layers_container)

        btn_add = QPushButton("+ Add layer")
        btn_add.setFixedHeight(28)
        btn_add.clicked.connect(self._add_layer_row)
        root.addWidget(btn_add)
        root.addWidget(_divider())

        # ── Grid layout ───────────────────────────────────────────────────────
        root.addWidget(_section_label("Crate Grid"))
        root.addWidget(_help_label("Configure grid layout and crate dimensions"))
        gform = QFormLayout()
        gform.setSpacing(6)

        self.grid_cols = FocusableSpinBox("grid_cols"); self.grid_cols.setRange(1, 10)
        self.grid_rows = FocusableSpinBox("grid_rows"); self.grid_rows.setRange(1, 10)
        self.crate_w   = FocusableDoubleSpinBox("crate_w"); self.crate_w.setRange(1, 9999); self.crate_w.setSuffix(" mm")
        self.crate_h   = FocusableDoubleSpinBox("crate_h"); self.crate_h.setRange(1, 9999); self.crate_h.setSuffix(" mm")

        for w in (self.grid_cols, self.grid_rows, self.crate_w, self.crate_h):
            w.valueChanged.connect(self._emit_config_changed)
            w.param_focused.connect(self._show_help)

        gform.addRow("Columns:", self.grid_cols)
        gform.addRow("Rows:",    self.grid_rows)
        gform.addRow("Width:",   self.crate_w)
        gform.addRow("Height:",  self.crate_h)
        root.addLayout(gform)
        root.addWidget(_divider())

        # ── Camera Intrinsics ─────────────────────────────────────────────────
        root.addWidget(_section_label("Camera"))
        root.addWidget(_help_label("Camera resolution and field of view settings"))
        cform = QFormLayout()
        cform.setSpacing(6)

        self.image_width  = FocusableSpinBox("image_width");      self.image_width.setRange(1, 9999)
        self.image_height = FocusableSpinBox("image_height");      self.image_height.setRange(1, 9999)
        self.fov_h        = FocusableDoubleSpinBox("fov_h"); self.fov_h.setRange(0, 180); self.fov_h.setSuffix(" °")
        self.fov_v        = FocusableDoubleSpinBox("fov_v"); self.fov_v.setRange(0, 180); self.fov_v.setSuffix(" °")
        self.output_dpi   = FocusableSpinBox("output_dpi"); self.output_dpi.setRange(72, 600)

        for w in (self.image_width, self.image_height, self.fov_h, self.fov_v, self.output_dpi):
            w.valueChanged.connect(self._emit_config_changed)
            w.param_focused.connect(self._show_help)

        cform.addRow("Width (px):", self.image_width)
        cform.addRow("Height (px):", self.image_height)
        cform.addRow("FOV H:", self.fov_h)
        cform.addRow("FOV V:", self.fov_v)
        cform.addRow("Output DPI:", self.output_dpi)
        root.addLayout(cform)
        root.addWidget(_divider())

        # ── Slot parameters ───────────────────────────────────────────────────
        root.addWidget(_section_label("Slot Analysis"))
        root.addWidget(_help_label("Advanced: Configure slot grid and fill detection"))
        sform = QFormLayout()
        sform.setSpacing(6)

        self.slot_cols      = FocusableSpinBox("slot_cols");      self.slot_cols.setRange(1, 20)
        self.slot_rows      = FocusableSpinBox("slot_rows");      self.slot_rows.setRange(1, 20)
        self.slot_thresh    = FocusableDoubleSpinBox("slot_thresh"); self.slot_thresh.setRange(0, 9999); self.slot_thresh.setSuffix(" mm")
        self.slot_radius    = FocusableDoubleSpinBox("slot_radius"); self.slot_radius.setRange(0, 999);  self.slot_radius.setSuffix(" mm")
        self.slot_sample_r  = FocusableSpinBox("slot_sample_r");      self.slot_sample_r.setRange(1, 50)
        self.slot_min_frac  = FocusableDoubleSpinBox("slot_min_frac"); self.slot_min_frac.setRange(0, 1); self.slot_min_frac.setDecimals(3); self.slot_min_frac.setSingleStep(0.01)
        self.present_n      = FocusableSpinBox("present_n");      self.present_n.setRange(1, 100)

        for w in (self.slot_cols, self.slot_rows, self.slot_thresh,
                  self.slot_radius, self.slot_sample_r, self.slot_min_frac, self.present_n):
            w.valueChanged.connect(self._emit_config_changed)
            w.param_focused.connect(self._show_help)

        sform.addRow("Cols:",           self.slot_cols)
        sform.addRow("Rows:",           self.slot_rows)
        sform.addRow("Fill threshold:", self.slot_thresh)
        sform.addRow("Radius:",         self.slot_radius)
        sform.addRow("Sample radius:",  self.slot_sample_r)
        sform.addRow("Min valid frac:", self.slot_min_frac)
        sform.addRow("Min pixels:",     self.present_n)
        root.addLayout(sform)
        root.addWidget(_divider())

        # ── More advanced settings ────────────────────────────────────────────
        root.addWidget(_section_label("Crate Size Filtering"))
        root.addWidget(_help_label("Advanced: Filter detected crates by size"))
        csform = QFormLayout()
        csform.setSpacing(6)

        self.crate_min_w = FocusableDoubleSpinBox("crate_min_w"); self.crate_min_w.setRange(1, 9999); self.crate_min_w.setSuffix(" mm")
        self.crate_max_w = FocusableDoubleSpinBox("crate_max_w"); self.crate_max_w.setRange(1, 9999); self.crate_max_w.setSuffix(" mm")
        self.crate_min_h = FocusableDoubleSpinBox("crate_min_h"); self.crate_min_h.setRange(1, 9999); self.crate_min_h.setSuffix(" mm")
        self.crate_max_h = FocusableDoubleSpinBox("crate_max_h"); self.crate_max_h.setRange(1, 9999); self.crate_max_h.setSuffix(" mm")
        self.crate_min_d = FocusableDoubleSpinBox("crate_min_d"); self.crate_min_d.setRange(1, 9999); self.crate_min_d.setSuffix(" mm")
        self.crate_max_d = FocusableDoubleSpinBox("crate_max_d"); self.crate_max_d.setRange(1, 9999); self.crate_max_d.setSuffix(" mm")
        
        for w in (self.crate_min_w, self.crate_max_w, self.crate_min_h, self.crate_max_h, self.crate_min_d, self.crate_max_d):
            w.valueChanged.connect(self._emit_config_changed)
            w.param_focused.connect(self._show_help)
        
        csform.addRow("Min width (W):", self.crate_min_w)
        csform.addRow("Max width (W):", self.crate_max_w)
        csform.addRow("Min height (H):", self.crate_min_h)
        csform.addRow("Max height (H):", self.crate_max_h)
        csform.addRow("Min depth (D):", self.crate_min_d)
        csform.addRow("Max depth (D):", self.crate_max_d)
        root.addLayout(csform)
        root.addWidget(_divider())
        
        # ── CCL Blob Detection ────────────────────────────────────────────────
        root.addWidget(_section_label("Blob Detection (CCL)"))
        root.addWidget(_help_label("Connected component labeling parameters"))
        ccform = QFormLayout()
        ccform.setSpacing(6)
        
        self.ccl_min_blob = FocusableSpinBox("ccl_min_blob"); self.ccl_min_blob.setRange(1, 10000)
        self.ccl_min_px = FocusableSpinBox("ccl_min_px"); self.ccl_min_px.setRange(1, 100000)
        self.ccl_max_px = FocusableSpinBox("ccl_max_px"); self.ccl_max_px.setRange(1, 100000)
        self.ccl_min_aspect = FocusableDoubleSpinBox("ccl_min_aspect"); self.ccl_min_aspect.setRange(0.01, 10); self.ccl_min_aspect.setDecimals(2); self.ccl_min_aspect.setSingleStep(0.1)
        self.ccl_max_aspect = FocusableDoubleSpinBox("ccl_max_aspect"); self.ccl_max_aspect.setRange(0.01, 10); self.ccl_max_aspect.setDecimals(2); self.ccl_max_aspect.setSingleStep(0.1)
        self.ccl_cell_pad = FocusableSpinBox("ccl_cell_pad"); self.ccl_cell_pad.setRange(0, 100)
        
        for w in (self.ccl_min_blob, self.ccl_min_px, self.ccl_max_px, self.ccl_min_aspect, self.ccl_max_aspect, self.ccl_cell_pad):
            w.valueChanged.connect(self._emit_config_changed)
            w.param_focused.connect(self._show_help)
        
        ccform.addRow("Min blob size:", self.ccl_min_blob)
        ccform.addRow("Min pixels:", self.ccl_min_px)
        ccform.addRow("Max pixels:", self.ccl_max_px)
        ccform.addRow("Min aspect:", self.ccl_min_aspect)
        ccform.addRow("Max aspect:", self.ccl_max_aspect)
        ccform.addRow("Cell padding:", self.ccl_cell_pad)
        root.addLayout(ccform)
        root.addWidget(_divider())
        
        # ── Rectangle Fitting ────────────────────────────────────────────────
        root.addWidget(_section_label("Rectangle Fitting"))
        root.addWidget(_help_label("Advanced: Fit rectangles to blobs"))
        rform = QFormLayout()
        rform.setSpacing(6)
        
        self.rect_pad = FocusableSpinBox("rect_pad"); self.rect_pad.setRange(0, 100)
        self.rect_interior = FocusableDoubleSpinBox("rect_interior"); self.rect_interior.setRange(0, 1); self.rect_interior.setDecimals(2); self.rect_interior.setSingleStep(0.05)
        
        for w in (self.rect_pad, self.rect_interior):
            w.valueChanged.connect(self._emit_config_changed)
            w.param_focused.connect(self._show_help)
        
        rform.addRow("Padding (px):", self.rect_pad)
        rform.addRow("Interior thresh:", self.rect_interior)
        root.addLayout(rform)
        root.addWidget(_divider())
        
        # ── Corner Detection ──────────────────────────────────────────────────
        root.addWidget(_section_label("Corner Detection"))
        root.addWidget(_help_label("Advanced: Harris corner detection"))
        coform = QFormLayout()
        coform.setSpacing(6)
        
        self.corner_neighbors = FocusableSpinBox("corner_neighbors"); self.corner_neighbors.setRange(1, 100)
        self.corner_search = FocusableSpinBox("corner_search"); self.corner_search.setRange(1, 50)
        
        for w in (self.corner_neighbors, self.corner_search):
            w.valueChanged.connect(self._emit_config_changed)
            w.param_focused.connect(self._show_help)
        
        coform.addRow("Min neighbors:", self.corner_neighbors)
        coform.addRow("Search radius (px):", self.corner_search)
        root.addLayout(coform)
        root.addWidget(_divider())
        
        # ── Visualization & AI ────────────────────────────────────────────────
        root.addWidget(_section_label("Visualization & AI"))
        root.addWidget(_help_label("Advanced: Output and AI verification"))
        aiform = QFormLayout()
        aiform.setSpacing(6)
        
        self.obb_margin = FocusableSpinBox("obb_margin"); self.obb_margin.setRange(0, 100)
        self.ai_conf = FocusableDoubleSpinBox("ai_conf"); self.ai_conf.setRange(0, 1); self.ai_conf.setDecimals(2); self.ai_conf.setSingleStep(0.05)
        
        for w in (self.obb_margin, self.ai_conf):
            w.valueChanged.connect(self._emit_config_changed)
            w.param_focused.connect(self._show_help)
        
        aiform.addRow("OBB margin (px):", self.obb_margin)
        aiform.addRow("AI confidence:", self.ai_conf)
        root.addLayout(aiform)

        root.addStretch()

        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll)

        # ── Parameter explanation label ───────────────────────────────────────
        self.help_label = QLabel("Select a parameter to see explanation")
        self.help_label.setStyleSheet(
            f"color:{PALETTE['text_dim']}; font-size:11px; font-style:italic;"
            f"padding:8px; background:{PALETTE['surface']}; border-radius:4px;"
            f"border:1px solid {PALETTE['border']};"
        )
        self.help_label.setWordWrap(True)
        self.help_label.setMinimumHeight(60)
        main_layout.addWidget(self.help_label)

        # ── Action buttons ────────────────────────────────────────────────────
        button_layout = QVBoxLayout()
        button_layout.setContentsMargins(12, 8, 12, 12)
        button_layout.setSpacing(6)
        
        self.run_btn = QPushButton("▶  Run Detection")
        self.run_btn.setObjectName("run_btn")
        self.run_btn.clicked.connect(self.run_requested)

        save_btn = QPushButton("Save Config JSON")
        save_btn.clicked.connect(self.save_requested)

        button_layout.addWidget(self.run_btn)
        button_layout.addWidget(save_btn)
        main_layout.addLayout(button_layout)

    # ── populate from config ───────────────────────────────────────────────────

    def _populate_from_config(self):
        self._building = True
        self.folder_edit.setText(self.cfg.snapshot_folder)

        for i, dist in enumerate(self.cfg.layer_distances_mm):
            hw = self.cfg.layer_half_widths_mm[i] if i < len(self.cfg.layer_half_widths_mm) else 100.0
            self._add_layer_row(dist, hw)

        self.grid_cols.setValue(self.cfg.grid_n_cols)
        self.grid_rows.setValue(self.cfg.grid_n_rows)
        self.crate_w.setValue(self.cfg.crate_width)
        self.crate_h.setValue(self.cfg.crate_height)

        self.image_width.setValue(self.cfg.image_px_width)
        self.image_height.setValue(self.cfg.image_px_height)
        self.fov_h.setValue(self.cfg.fov_h_deg)
        self.fov_v.setValue(self.cfg.fov_v_deg)
        self.output_dpi.setValue(self.cfg.output_dpi)

        self.slot_cols.setValue(self.cfg.slot_n_cols)
        self.slot_rows.setValue(self.cfg.slot_n_rows)
        self.slot_thresh.setValue(self.cfg.slot_fill_threshold_mm)
        self.slot_radius.setValue(self.cfg.slot_radius_mm)
        self.slot_sample_r.setValue(self.cfg.slot_sample_radius_px)
        self.slot_min_frac.setValue(self.cfg.slot_min_valid_frac)
        self.present_n.setValue(self.cfg.present_n_pixel)

        # Crate filtering (use first 3 values from lists)
        self.crate_min_w.setValue(self.cfg.crate_min_size_mm[0] if len(self.cfg.crate_min_size_mm) > 0 else 380.0)
        self.crate_max_w.setValue(self.cfg.crate_max_size_mm[0] if len(self.cfg.crate_max_size_mm) > 0 else 600.0)
        self.crate_min_h.setValue(self.cfg.crate_min_size_mm[1] if len(self.cfg.crate_min_size_mm) > 1 else 250.0)
        self.crate_max_h.setValue(self.cfg.crate_max_size_mm[1] if len(self.cfg.crate_max_size_mm) > 1 else 500.0)
        self.crate_min_d.setValue(self.cfg.crate_min_size_mm[2] if len(self.cfg.crate_min_size_mm) > 2 else 200.0)
        self.crate_max_d.setValue(self.cfg.crate_max_size_mm[2] if len(self.cfg.crate_max_size_mm) > 2 else 300.0)

        # CCL parameters
        self.ccl_min_blob.setValue(self.cfg.ccl_min_blob_size)
        self.ccl_min_px.setValue(self.cfg.ccl_min_px)
        self.ccl_max_px.setValue(self.cfg.ccl_max_px)
        self.ccl_min_aspect.setValue(self.cfg.ccl_min_aspect)
        self.ccl_max_aspect.setValue(self.cfg.ccl_max_aspect)
        self.ccl_cell_pad.setValue(self.cfg.ccl_cell_padding_px)

        # Rectangle fitting
        self.rect_pad.setValue(self.cfg.rect_padding)
        self.rect_interior.setValue(self.cfg.rect_interior_threshold)

        # Corner detection
        self.corner_neighbors.setValue(self.cfg.corner_min_neighbours)
        self.corner_search.setValue(self.cfg.corner_search_radius_px)

        # Visualization & AI
        self.obb_margin.setValue(self.cfg.obb_canvas_margin)
        self.ai_conf.setValue(self.cfg.ai_conf_threshold)

        self._building = False

    # ── layer row management ───────────────────────────────────────────────────

    def _add_layer_row(self, value: float = 1000.0, half_width: float = 100.0):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(4)

        spin = QDoubleSpinBox()
        spin.setRange(1, 99999)
        spin.setValue(float(value))
        spin.setSuffix(" mm")
        spin.setDecimals(1)
        spin.valueChanged.connect(self._emit_config_changed)

        hw_spin = QDoubleSpinBox()
        hw_spin.setRange(1, 9999)
        hw_spin.setValue(float(half_width))
        hw_spin.setSuffix(" mm")
        hw_spin.setDecimals(1)
        hw_spin.setMaximumWidth(90)
        hw_spin.valueChanged.connect(self._emit_config_changed)

        remove_btn = QToolButton()
        remove_btn.setText("−")
        remove_btn.setFixedWidth(24)

        row_layout.addWidget(QLabel("Depth:"))
        row_layout.addWidget(spin, 1)
        row_layout.addWidget(QLabel("Half-W:"))
        row_layout.addWidget(hw_spin)
        row_layout.addWidget(remove_btn)

        self._layer_rows.append((spin, hw_spin, remove_btn))
        self.layers_container.addWidget(row_widget)

        def _remove():
            self._layer_rows.remove((spin, hw_spin, remove_btn))
            row_widget.setParent(None)
            row_widget.deleteLater()
            self._emit_config_changed()

        remove_btn.clicked.connect(_remove)

        if not self._building:
            self._emit_config_changed()

    # ── read current values → config ──────────────────────────────────────────

    def read_into_config(self) -> CrateVisionConfig:
        """Push all sidebar values into cfg and return it."""
        self.cfg.snapshot_folder     = self.folder_edit.text().strip()
        self.cfg.layer_distances_mm  = [s.value() for s, _, _ in self._layer_rows]
        self.cfg.layer_half_widths_mm = [hw.value() for _, hw, _ in self._layer_rows]
        self.cfg.output_dpi          = self.output_dpi.value()
        
        self.cfg.grid_n_cols         = self.grid_cols.value()
        self.cfg.grid_n_rows         = self.grid_rows.value()
        self.cfg.crate_width         = self.crate_w.value()
        self.cfg.crate_height        = self.crate_h.value()

        self.cfg.image_px_width      = self.image_width.value()
        self.cfg.image_px_height     = self.image_height.value()
        self.cfg.fov_h_deg           = self.fov_h.value()
        self.cfg.fov_v_deg           = self.fov_v.value()

        self.cfg.slot_n_cols         = self.slot_cols.value()
        self.cfg.slot_n_rows         = self.slot_rows.value()
        self.cfg.slot_fill_threshold_mm = self.slot_thresh.value()
        self.cfg.slot_radius_mm      = self.slot_radius.value()
        self.cfg.slot_sample_radius_px = self.slot_sample_r.value()
        self.cfg.slot_min_valid_frac = self.slot_min_frac.value()
        self.cfg.present_n_pixel     = self.present_n.value()

        # Crate filtering (store back into lists)
        self.cfg.crate_min_size_mm = [
            self.crate_min_w.value(),
            self.crate_min_h.value(),
            self.crate_min_d.value()
        ]
        self.cfg.crate_max_size_mm = [
            self.crate_max_w.value(),
            self.crate_max_h.value(),
            self.crate_max_d.value()
        ]

        # CCL parameters
        self.cfg.ccl_min_blob_size  = self.ccl_min_blob.value()
        self.cfg.ccl_min_px        = self.ccl_min_px.value()
        self.cfg.ccl_max_px        = self.ccl_max_px.value()
        self.cfg.ccl_min_aspect    = self.ccl_min_aspect.value()
        self.cfg.ccl_max_aspect    = self.ccl_max_aspect.value()
        self.cfg.ccl_cell_padding_px = self.ccl_cell_pad.value()

        # Rectangle fitting
        self.cfg.rect_padding      = self.rect_pad.value()
        self.cfg.rect_interior_threshold = self.rect_interior.value()

        # Corner detection
        self.cfg.corner_min_neighbours = self.corner_neighbors.value()
        self.cfg.corner_search_radius_px = self.corner_search.value()

        # Visualization & AI
        self.cfg.obb_canvas_margin = self.obb_margin.value()
        self.cfg.ai_conf_threshold = self.ai_conf.value()

        return self.cfg

    # ── slots ─────────────────────────────────────────────────────────────────

    def _emit_config_changed(self):
        if not self._building:
            self.read_into_config()
            self.config_changed.emit()

    def _on_folder_changed(self, text: str):
        if not self._building:
            self.folder_changed.emit(text)

    def _browse_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select snapshot folder")
        if path:
            self.folder_edit.setText(path)

    def set_running(self, running: bool):
        self.run_btn.setEnabled(not running)
        self.run_btn.setText("⏳  Running…" if running else "▶  Run Detection")


# ─────────────────────────────────────────────────────────────────────────────
# Centre panel — depth preview + click-to-set-pixel
# ─────────────────────────────────────────────────────────────────────────────

class CentrePanel(QWidget):
    """
    Shows the depth summary (amplitude + depth map + per-layer masks).
    User clicks on the image to set the grid anchor pixel for a layer.
    """
    pixel_set = pyqtSignal(str, int, int)   # layer_name, col, row

    def __init__(self, cfg: CrateVisionConfig, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._masks: dict = {}
        self._img_color: np.ndarray | None = None
        self._z_coords: np.ndarray | None  = None
        self._active_layer: str | None     = None
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        # ── Layer selector tabs ────────────────────────────────────────────────
        selector_row = QHBoxLayout()
        selector_row.setSpacing(6)
        lbl = QLabel("Active layer:")
        lbl.setStyleSheet(f"color:{PALETTE['text_dim']}; font-size:12px;")
        selector_row.addWidget(lbl)
        self.layer_tabs_layout = QHBoxLayout()
        self.layer_tabs_layout.setSpacing(4)
        selector_row.addLayout(self.layer_tabs_layout)
        selector_row.addStretch()
        root.addLayout(selector_row)

        # ── Splitter: summary (top) + clickable camera image (bottom) ─────────
        inner_split = QSplitter(Qt.Vertical)
        inner_split.setHandleWidth(4)
        inner_split.setStyleSheet(
            f"QSplitter::handle {{ background:{PALETTE['border']}; }}"
        )

        # Top: matplotlib depth summary (read-only, for context)
        self.summary_label = QLabel()
        self.summary_label.setAlignment(Qt.AlignCenter)
        self.summary_label.setStyleSheet(f"background:{PALETTE['surface']};")
        self.summary_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        inner_split.addWidget(self.summary_label)

        # Bottom: actual camera image at native resolution → click here
        bottom_wrap = QWidget()
        bw_layout = QVBoxLayout(bottom_wrap)
        bw_layout.setContentsMargins(0, 0, 0, 0)
        bw_layout.setSpacing(4)

        click_header = QLabel("Click below to set anchor pixel for active layer")
        click_header.setStyleSheet(
            f"color:{PALETTE['text_dim']}; font-size:11px;"
            f" padding:2px 0; letter-spacing:0.3px;"
        )
        bw_layout.addWidget(click_header)

        # ── Zoom controls ─────────────────────────────────────────────────────
        zoom_row = QHBoxLayout()
        zoom_row.setSpacing(4)
        zoom_minus = QPushButton("−")
        zoom_minus.setFixedWidth(32)
        zoom_minus.clicked.connect(self._zoom_out)
        zoom_fit = QPushButton("Fit")
        zoom_fit.setFixedWidth(48)
        zoom_fit.clicked.connect(self._zoom_fit)
        zoom_plus = QPushButton("+")
        zoom_plus.setFixedWidth(32)
        zoom_plus.clicked.connect(self._zoom_in)
        self.zoom_label = QLabel("100%")
        self.zoom_label.setStyleSheet(f"color:{PALETTE['text_dim']}; font-size:11px;")
        self.zoom_label.setFixedWidth(50)
        
        zoom_row.addWidget(QLabel("Zoom:"))
        zoom_row.addWidget(zoom_minus)
        zoom_row.addWidget(zoom_fit)
        zoom_row.addWidget(zoom_plus)
        zoom_row.addWidget(self.zoom_label)
        zoom_row.addStretch()
        bw_layout.addLayout(zoom_row)

        self.img_label = ClickableImageLabel()
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_label.setMinimumHeight(120)
        self.img_label.pixel_clicked.connect(self._on_pixel_clicked)
        bw_layout.addWidget(self.img_label, 1)

        inner_split.addWidget(bottom_wrap)
        inner_split.setSizes([320, 200])
        root.addWidget(inner_split, 1)

        # ── Pixel readout ──────────────────────────────────────────────────────
        readout_row = QHBoxLayout()
        self.pixel_coord_lbl = QLabel("No anchor set — click the camera image above")
        self.pixel_coord_lbl.setStyleSheet(
            f"color:{PALETTE['text_dim']}; font-size:12px;"
        )
        readout_row.addWidget(self.pixel_coord_lbl)
        readout_row.addStretch()
        root.addLayout(readout_row)

        # ── Progress bar ───────────────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        root.addWidget(self.progress)

    # ── public API ────────────────────────────────────────────────────────────

    def load_snapshot(self, folder: str, cfg: CrateVisionConfig) -> bool:
        """
        Load amplitude + depth from a snapshot folder and render the preview.
        Returns True on success.
        """
        x, y, z, img_color, img_gray = load_depth_and_amplitude(folder)
        if x is None:
            return False

        self._img_color = img_color
        self._z_coords  = z
        self._masks = {}

        from crate_vision.detection.depth import create_depth_masks, remove_small_blobs
        raw = create_depth_masks(z, cfg.layer_distances_mm, cfg.layer_half_widths_mm)
        self._masks = {
            name: remove_small_blobs(mask, cfg.ccl_min_blob_size)
            for name, mask in raw.items()
        }

        self._rebuild_layer_tabs(cfg)
        self._render_summary(cfg)
        self._render_click_view()   # show camera image at native resolution
        return True

    def refresh_masks(self, cfg: CrateVisionConfig):
        """Recompute masks with new half-width and re-render."""
        if self._z_coords is None:
            return
        from crate_vision.detection.depth import create_depth_masks, remove_small_blobs
        raw = create_depth_masks(
            self._z_coords, cfg.layer_distances_mm, cfg.layer_half_widths_mm
        )
        self._masks = {
            name: remove_small_blobs(mask, cfg.ccl_min_blob_size)
            for name, mask in raw.items()
        }
        self._rebuild_layer_tabs(cfg)
        self._render_summary(cfg)
        self._render_click_view()

    def set_busy(self, busy: bool):
        self.progress.setVisible(busy)

    def get_active_layer(self) -> str | None:
        return self._active_layer

    def refresh_grid_visualization(self, cfg: CrateVisionConfig):
        """Redraw grid if an anchor pixel is currently set."""
        if (self.img_label._click_col is None or self.img_label._click_row is None or 
            self._active_layer is None or self._z_coords is None):
            return
        
        grid_config = self._calculate_grid_config(self.img_label._click_col, self.img_label._click_row)
        if grid_config is not None:
            self.img_label.set_click(self.img_label._click_col, self.img_label._click_row, grid_config)

    def _zoom_in(self) -> None:
        """Zoom in."""
        self.img_label.zoom_in()
        self.zoom_label.setText(f"{int(self.img_label._zoom_level * 100)}%")

    def _zoom_out(self) -> None:
        """Zoom out."""
        self.img_label.zoom_out()
        self.zoom_label.setText(f"{int(self.img_label._zoom_level * 100)}%")

    def _zoom_fit(self) -> None:
        """Reset zoom to fit."""
        self.img_label.zoom_fit()
        self.zoom_label.setText("100%")

    # ── private ───────────────────────────────────────────────────────────────

    def _rebuild_layer_tabs(self, cfg: CrateVisionConfig):
        # Clear existing tab buttons
        while self.layer_tabs_layout.count():
            item = self.layer_tabs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        self._layer_btns: dict[str, QPushButton] = {}
        layer_names = list(self._masks.keys())

        if not layer_names:
            return

        for name in layer_names:
            # Short label: just the distance part, e.g. "1020 mm"
            short = name.split("mm")[0].strip() + " mm"
            btn = QPushButton(short)
            btn.setFixedHeight(26)
            btn.setCheckable(True)
            btn.setStyleSheet(
                f"QPushButton {{ border-radius:4px; padding:2px 10px;"
                f" font-size:12px; }}"
            )
            btn.clicked.connect(lambda checked, n=name: self._select_layer(n))
            self.layer_tabs_layout.addWidget(btn)
            self._layer_btns[name] = btn

        # Select first layer by default if nothing active
        if self._active_layer not in layer_names:
            self._active_layer = layer_names[0]
        self._update_tab_styles()

    def _select_layer(self, name: str):
        self._active_layer = name
        self._update_tab_styles()
        
        # Check if this layer has a saved anchor pixel
        if name in self.cfg.grid_corner_pixels:
            col, row = self.cfg.grid_corner_pixels[name]
            z_val = ""
            if self._z_coords is not None:
                z = float(self._z_coords[row, col])
                z_val = f"   Z = {z:.1f} mm" if np.isfinite(z) and z > 0 else "   Z = —"
            
            layer_short = name.split("mm")[0].strip() + " mm"
            self.pixel_coord_lbl.setStyleSheet(
                f"color:{PALETTE['accent']}; font-size:12px; font-weight:600;"
            )
            self.pixel_coord_lbl.setText(
                f"✓  {layer_short}  →  col = {col},  row = {row}{z_val}"
            )
            
            # Restore grid visualization for this layer
            grid_config = self._calculate_grid_config(col, row)
            self.img_label.set_click(col, row, grid_config)
        else:
            # No anchor set for this layer
            self.pixel_coord_lbl.setText("No anchor set — click the camera image above")
            self.img_label.clear_click()
        
        # Refresh click view to show this layer's mask
        self._render_click_view()

    def _update_tab_styles(self):
        for name, btn in getattr(self, "_layer_btns", {}).items():
            active = name == self._active_layer
            btn.setChecked(active)
            btn.setStyleSheet(
                f"QPushButton {{ border-radius:4px; padding:2px 10px; font-size:12px;"
                f" background:{'#4a9eff' if active else PALETTE['surface2']};"
                f" color:{'white' if active else PALETTE['text_dim']};"
                f" border:1px solid {'#4a9eff' if active else PALETTE['border']}; }}"
            )

    def _render_click_view(self):
        """
        Display the active layer's masked amplitude at native camera resolution.
        This is the image the user clicks on — coordinates map 1:1 to camera pixels.
        If no active layer, show the plain amplitude image.
        """
        if self._img_color is None:
            return

        mask = self._masks.get(self._active_layer) if self._active_layer else None
        if mask is not None:
            from crate_vision.detection.depth import apply_mask_to_image
            display = apply_mask_to_image(self._img_color, mask).copy()
            # Tint unmasked pixels slightly so the active region is obvious
            outside = ~mask
            display[outside] = (display[outside] * 0.25).astype(np.uint8)
        else:
            display = self._img_color.copy()

        # img_label stores orig_w/orig_h = camera dimensions → clicks are correct
        self.img_label.set_image(display)

    def _render_summary(self, cfg: CrateVisionConfig):
        """Compose the depth summary figure (matplotlib) and show in summary_label."""
        if self._img_color is None or not self._masks:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        n = len(self._masks)
        fig = Figure(figsize=(3 * (n + 1), 5.5), dpi=96, facecolor="#1a1d23")
        canvas = FigureCanvasAgg(fig)
        axes = fig.subplots(2, n + 1)
        if n + 1 == 1:
            axes = [[axes[r]] for r in range(2)]

        def _style(ax):
            ax.axis("off")
            ax.set_facecolor("#1a1d23")

        # Col 0: original + depth map
        axes[0][0].imshow(self._img_color)
        axes[0][0].set_title("Amplitude", color="white", fontsize=8, pad=3)
        _style(axes[0][0])

        z_disp = np.copy(self._z_coords)
        z_disp[~np.isfinite(z_disp)] = 0
        im = axes[1][0].imshow(z_disp, cmap="viridis")
        axes[1][0].set_title("Depth map", color="white", fontsize=8, pad=3)
        _style(axes[1][0])
        fig.colorbar(im, ax=axes[1][0], fraction=0.046, pad=0.04)

        for col_i, (layer_name, mask) in enumerate(self._masks.items(), start=1):
            # Row 0: mask
            axes[0][col_i].imshow(mask, cmap="gray")
            short = layer_name.split("mm")[0].strip() + " mm"
            axes[0][col_i].set_title(f"{short}\nMask", color="white", fontsize=8, pad=3)
            _style(axes[0][col_i])

            # Row 1: masked amplitude + pixel stats in title
            from crate_vision.detection.depth import apply_mask_to_image
            masked = apply_mask_to_image(self._img_color, mask)
            axes[1][col_i].imshow(masked)
            pct = 100 * mask.sum() / mask.size
            axes[1][col_i].set_title(f"Masked\n{mask.sum():,} px ({pct:.1f}%)", color="white", fontsize=7, pad=3)
            _style(axes[1][col_i])

        fig.tight_layout(pad=0.5)
        canvas.draw()
        buf = canvas.buffer_rgba()
        w, h = canvas.get_width_height()
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        rgb = arr[:, :, :3].copy()
        plt.close(fig)

        self._preview_rgb = rgb
        # Show in summary_label (read-only overview) — not the clickable view
        pix = _numpy_to_pixmap(rgb)
        self.summary_label.setPixmap(pix)

    def _on_pixel_clicked(self, col: int, row: int):
        if self._active_layer is None:
            return

        # col, row are now true camera pixel coordinates (img_label was set
        # with the camera image at native resolution, so the scaling in
        # ClickableImageLabel maps back to camera pixels correctly).
        z_val = ""
        if self._z_coords is not None:
            z = float(self._z_coords[row, col])
            z_val = f"   Z = {z:.1f} mm" if np.isfinite(z) and z > 0 else "   Z = —"

        layer_short = self._active_layer.split("mm")[0].strip() + " mm"
        self.pixel_coord_lbl.setStyleSheet(
            f"color:{PALETTE['accent']}; font-size:12px; font-weight:600;"
        )
        self.pixel_coord_lbl.setText(
            f"✓  {layer_short}  →  col = {col},  row = {row}{z_val}"
        )
        
        # Calculate grid config and pass it to the image label for visualization
        grid_config = self._calculate_grid_config(col, row)
        self.img_label.set_click(col, row, grid_config)
        
        self.pixel_set.emit(self._active_layer, col, row)

    def _calculate_grid_config(self, corner_col: int, corner_row: int) -> dict | None:
        """Calculate grid parameters for visualization."""
        if self._z_coords is None or self._active_layer is None:
            return None
        
        # Get z-depth at this layer
        z_at_layer = float(self._z_coords[corner_row, corner_col])
        if not np.isfinite(z_at_layer) or z_at_layer <= 0:
            return None
        
        # Import what we need for calculation
        from crate_vision.detection.geometry import compute_mm_per_pixel_theoretical, get_grid_anchor
        from crate_vision.config import get_config
        
        cfg = get_config()
        mask = self._masks.get(self._active_layer)
        if mask is None:
            return None
        
        # Calculate pixel scale
        mm_per_px_x, mm_per_px_y = compute_mm_per_pixel_theoretical(
            self._z_coords, mask, cfg.fov_h_deg, cfg.fov_v_deg
        )
        
        if mm_per_px_x <= 0 or mm_per_px_y <= 0:
            return None
        
        # Calculate grid dimensions in pixels
        crate_w_px = int(round(cfg.crate_width / mm_per_px_x))
        crate_h_px = int(round(cfg.crate_height / mm_per_px_y))
        
        # Calculate anchor point for the grid
        H, W = self._z_coords.shape[:2]
        anchor_col, anchor_row, _, _, _ = get_grid_anchor(
            corner_col, corner_row,
            crate_w_px, crate_h_px,
            cfg.grid_n_cols, cfg.grid_n_rows, W, H,
        )
        
        return {
            "crate_w_px": crate_w_px,
            "crate_h_px": crate_h_px,
            "n_cols": cfg.grid_n_cols,
            "n_rows": cfg.grid_n_rows,
            "anchor_col": anchor_col,
            "anchor_row": anchor_row,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Right panel — results list + object preview
# ─────────────────────────────────────────────────────────────────────────────

class ResultsPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._object_dirs: list[str]  = []   # path to each object folder
        self._metas:       list[dict] = []
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background:{PALETTE['border']}; }}"
        )

        # ── Top: crate list ────────────────────────────────────────────────────
        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(8, 8, 8, 8)
        top_layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.addWidget(_section_label("Detected crates"))
        header_row.addStretch()
        self.count_lbl = QLabel("0 objects")
        self.count_lbl.setStyleSheet(f"color:{PALETTE['text_dim']}; font-size:12px;")
        header_row.addWidget(self.count_lbl)
        top_layout.addLayout(header_row)

        self.list_widget = QListWidget()
        self.list_widget.currentRowChanged.connect(self._on_selection_changed)
        top_layout.addWidget(self.list_widget, 1)

        splitter.addWidget(top)

        # ── Bottom: object detail ──────────────────────────────────────────────
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 8, 8, 8)
        bottom_layout.setSpacing(6)

        self.detail_tabs = QTabWidget()
        self.detail_tabs.setMinimumHeight(220)

        # Tab 1 — Plane fit image
        self.plane_tab = QLabel()
        self.plane_tab.setAlignment(Qt.AlignCenter)
        self.plane_tab.setStyleSheet(
            f"background:{PALETTE['surface']}; color:{PALETTE['text_faint']};"
        )
        self.plane_tab.setText("No object selected")
        self.detail_tabs.addTab(self._scroll(self.plane_tab), "Plane fit")

        # Tab 2 — Slot analysis image
        self.slot_tab = QLabel()
        self.slot_tab.setAlignment(Qt.AlignCenter)
        self.slot_tab.setStyleSheet(
            f"background:{PALETTE['surface']}; color:{PALETTE['text_faint']};"
        )
        self.slot_tab.setText("No object selected")
        self.detail_tabs.addTab(self._scroll(self.slot_tab), "Slot analysis")

        # Tab 3 — Info JSON
        self.info_tab = QTextEdit()
        self.info_tab.setReadOnly(True)
        self.detail_tabs.addTab(self.info_tab, "Info JSON")

        bottom_layout.addWidget(self.detail_tabs, 1)

        # ── Slot summary row ──────────────────────────────────────────────────
        self.slot_summary = QLabel("")
        self.slot_summary.setStyleSheet(
            f"color:{PALETTE['text_dim']}; font-size:12px; padding:4px 0;"
        )
        bottom_layout.addWidget(self.slot_summary)

        splitter.addWidget(bottom)
        splitter.setSizes([250, 320])

        root.addWidget(splitter)

    def _scroll(self, widget: QWidget) -> QScrollArea:
        sa = QScrollArea()
        sa.setWidget(widget)
        sa.setWidgetResizable(True)
        sa.setAlignment(Qt.AlignCenter)
        return sa

    # ── public API ────────────────────────────────────────────────────────────

    def clear(self):
        self.list_widget.clear()
        self.plane_tab.setText("No object selected")
        self.plane_tab.setPixmap(QPixmap())
        self.slot_tab.setText("No object selected")
        self.slot_tab.setPixmap(QPixmap())
        self.info_tab.clear()
        self.slot_summary.setText("")
        self.count_lbl.setText("0 objects")
        self._object_dirs.clear()
        self._metas.clear()

    def populate(self, result: dict, object_root: str):
        """
        Fill the list from a detection result dict.
        object_root is the base folder where object sub-folders were written.
        """
        self.clear()
        crates = result.get("crates", [])
        self.count_lbl.setText(f"{len(crates)} object{'s' if len(crates) != 1 else ''}")

        for crate in crates:
            item = QListWidgetItem()
            item.setSizeHint(item.sizeHint().__class__(0, 52))

            crate_id  = crate.get("crate_number", "?")
            layer     = crate.get("layer", "")
            n_filled  = sum(1 for k in [f"S{i}" for i in range(1, 13)] if crate.get(k))
            ai_pass   = crate.get("ai_classification", False)

            # Build display widget
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(8, 4, 8, 4)
            row_l.setSpacing(8)

            id_lbl = QLabel(f"#{crate_id}")
            id_lbl.setStyleSheet(f"color:{PALETTE['accent']}; font-weight:700; font-size:14px;")
            id_lbl.setFixedWidth(32)

            info_col = QVBoxLayout()
            info_col.setSpacing(1)
            layer_lbl = QLabel(layer.split("mm")[0].strip() + " mm" if layer else "—")
            layer_lbl.setStyleSheet(f"color:{PALETTE['text']}; font-size:12px;")
            slot_lbl  = QLabel(f"{n_filled}/12 slots filled")
            slot_lbl.setStyleSheet(f"color:{PALETTE['text_dim']}; font-size:11px;")
            info_col.addWidget(layer_lbl)
            info_col.addWidget(slot_lbl)

            ai_badge = _badge("AI ✓" if ai_pass else "AI ✗",
                              PALETTE['green'] if ai_pass else PALETTE['red'])

            row_l.addWidget(id_lbl)
            row_l.addLayout(info_col, 1)
            row_l.addWidget(ai_badge)

            self.list_widget.addItem(item)
            self.list_widget.setItemWidget(item, row_w)

            # Try to find the object folder
            obj_dir = self._find_object_dir(object_root, crate)
            self._object_dirs.append(obj_dir or "")
            self._metas.append(crate)

    def _find_object_dir(self, object_root: str, crate: dict) -> str | None:
        """
        Walk the object output tree to find the folder for this crate.
        Matches on crate_id stored in info.json.
        """
        crate_id = crate.get("crate_number")
        if crate_id is None or not os.path.isdir(object_root):
            return None
        for dirpath, dirnames, filenames in os.walk(object_root):
            if "info.json" in filenames:
                try:
                    with open(os.path.join(dirpath, "info.json")) as f:
                        meta = json.load(f)
                    if meta.get("crate_id") == crate_id:
                        return dirpath
                except Exception:
                    pass
        return None

    # ── selection ─────────────────────────────────────────────────────────────

    def _on_selection_changed(self, idx: int):
        if idx < 0 or idx >= len(self._object_dirs):
            return

        obj_dir = self._object_dirs[idx]
        meta    = self._metas[idx]

        # ── Plane fit tab ──────────────────────────────────────────────────────
        plane_path = os.path.join(obj_dir, "plane_fit.png") if obj_dir else ""
        pix = _load_png_pixmap(plane_path, max_w=600, max_h=400)
        if pix:
            self.plane_tab.setText("")
            self.plane_tab.setPixmap(pix)
        else:
            self.plane_tab.setPixmap(QPixmap())
            self.plane_tab.setText("plane_fit.png not found")

        # ── Slot analysis tab ─────────────────────────────────────────────────
        slot_path = os.path.join(obj_dir, "slot_analysis.png") if obj_dir else ""
        pix2 = _load_png_pixmap(slot_path, max_w=400, max_h=400)
        if pix2:
            self.slot_tab.setText("")
            self.slot_tab.setPixmap(pix2)
        else:
            self.slot_tab.setPixmap(QPixmap())
            self.slot_tab.setText("slot_analysis.png not found")

        # ── Info JSON tab ─────────────────────────────────────────────────────
        self.info_tab.setPlainText(json.dumps(meta, indent=2))

        # ── Slot summary row ──────────────────────────────────────────────────
        slot_data = meta.get("slot_analysis", {})
        if slot_data:
            nf = slot_data.get("n_filled", 0)
            ne = slot_data.get("n_empty", 0)
            nu = slot_data.get("n_unknown", 0)
            grid = slot_data.get("grid", "")
            green  = PALETTE['green']
            tdim   = PALETTE['text_dim']
            amber  = PALETTE['amber']
            self.slot_summary.setText(
                f"Grid {grid}  ·  "
                f"<span style='color:{green}'>{nf} filled</span>  "
                f"<span style='color:{tdim}'>{ne} empty</span>  "
                f"<span style='color:{amber}'>{nu} unknown</span>"
            )
            self.slot_summary.setTextFormat(Qt.RichText)
        else:
            self.slot_summary.setText("")


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg    = get_config()
        self._worker: DetectionWorker | None = None
        self._corner_pixels: dict[str, list[int]] = {}
        self._object_root   = ""
        self._refresh_timer = QTimer()
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self._on_config_changed_debounced)

        self.setWindowTitle("Crate Vision — Configuration & Review")
        self.setMinimumSize(1200, 700)
        self.resize(1440, 820)

        self._build_ui()
        self._load_snapshot_if_valid()

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.setStyleSheet(BASE_QSS)

        # ── Menu bar ──────────────────────────────────────────────────────────
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction("Open snapshot folder…", self._browse_snapshot)
        file_menu.addSeparator()
        file_menu.addAction("Load config JSON…",  self._load_config_json)
        file_menu.addAction("Save config JSON…",  self._save_config_json)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        # ── Status bar ────────────────────────────────────────────────────────
        self.status_bar = self.statusBar()
        self.status_bar.setStyleSheet(
            f"background:{PALETTE['surface']}; color:{PALETTE['text_dim']};"
            f"border-top:1px solid {PALETTE['border']};"
        )
        self.status_bar.showMessage("Ready")

        # ── Central splitter ──────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(4)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background:{PALETTE['border']}; }}"
        )

        # Left sidebar
        self.sidebar = SidebarWidget(self.cfg)
        self.sidebar.config_changed.connect(self._on_config_changed)
        self.sidebar.run_requested.connect(self._run_detection)
        self.sidebar.save_requested.connect(self._save_config_json)
        self.sidebar.folder_changed.connect(self._on_folder_changed)
        splitter.addWidget(self.sidebar)

        # Centre
        centre_wrap = QWidget()
        centre_wrap.setStyleSheet(
            f"background:{PALETTE['surface']};"
            f"border-left:1px solid {PALETTE['border']};"
            f"border-right:1px solid {PALETTE['border']};"
        )
        cw_layout = QVBoxLayout(centre_wrap)
        cw_layout.setContentsMargins(10, 10, 10, 10)
        self.centre = CentrePanel(self.cfg)
        self.centre.pixel_set.connect(self._on_pixel_set)
        cw_layout.addWidget(self.centre)
        splitter.addWidget(centre_wrap)

        # Right panel
        right_wrap = QWidget()
        right_wrap.setStyleSheet(f"background:{PALETTE['bg']};")
        rw_layout = QVBoxLayout(right_wrap)
        rw_layout.setContentsMargins(8, 8, 8, 8)
        self.results = ResultsPanel()
        rw_layout.addWidget(self.results)
        splitter.addWidget(right_wrap)

        splitter.setSizes([288, 680, 380])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        h_layout.addWidget(splitter)

    # ── snapshot loading ───────────────────────────────────────────────────────

    def _load_snapshot_if_valid(self):
        folder = self.cfg.snapshot_folder
        if folder and os.path.isdir(folder):
            self._reload_preview()

    def _reload_preview(self):
        self.sidebar.read_into_config()
        folder = self.cfg.snapshot_folder
        if not folder or not os.path.isdir(folder):
            self.status_bar.showMessage("Snapshot folder not found")
            return
        self.status_bar.showMessage("Loading preview…")
        ok = self.centre.load_snapshot(folder, self.cfg)
        if ok:
            self.status_bar.showMessage(
                f"Loaded: {Path(folder).name}  — "
                f"click the depth image to set anchor pixels"
            )
        else:
            self.status_bar.showMessage("Could not load depth data from folder")

    # ── config change handling ────────────────────────────────────────────────

    def _on_config_changed(self):
        # Debounce — wait 400 ms after last change before refreshing
        self._refresh_timer.start(400)

    def _on_config_changed_debounced(self):
        """Called 400 ms after the last config change."""
        self.sidebar.read_into_config()
        self.centre.refresh_masks(self.cfg)
        self.centre.refresh_grid_visualization(self.cfg)

    def _on_folder_changed(self, path: str):
        self.cfg.snapshot_folder = path
        if os.path.isdir(path):
            self._reload_preview()

    # ── pixel click ───────────────────────────────────────────────────────────

    def _on_pixel_set(self, layer_name: str, col: int, row: int):
        self._corner_pixels[layer_name] = [col, row]
        self.cfg.grid_corner_pixels = dict(self._corner_pixels)
        n = len(self._corner_pixels)
        self.status_bar.showMessage(
            f"Anchor set — {layer_name.split('mm')[0].strip()} mm → col={col}, row={row}  "
            f"({n} layer{'s' if n != 1 else ''} configured)"
        )

    # ── detection ─────────────────────────────────────────────────────────────

    def _run_detection(self):
        self.sidebar.read_into_config()

        folder = self.cfg.snapshot_folder
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "No snapshot", "Please select a valid snapshot folder.")
            return

        if not self.cfg.grid_corner_pixels:
            QMessageBox.warning(
                self, "No anchor pixels",
                "Click on the depth image to set at least one grid anchor pixel "
                "before running detection."
            )
            return

        self.results.clear()
        self.sidebar.set_running(True)
        self.centre.set_busy(True)
        self.status_bar.showMessage("Running detection…")

        # Object output root (mirrors pipeline.py convention)
        snap_name = Path(folder).resolve().name
        self._object_root = os.path.join("object", snap_name)

        self._worker = DetectionWorker(folder, self.cfg)
        self._worker.finished.connect(self._on_detection_finished)
        self._worker.error.connect(self._on_detection_error)
        self._worker.progress.connect(lambda msg: self.status_bar.showMessage(msg))
        self._worker.start()

    @pyqtSlot(dict)
    def _on_detection_finished(self, result: dict):
        self.sidebar.set_running(False)
        self.centre.set_busy(False)
        n = len(result.get("crates", []))
        self.status_bar.showMessage(
            f"Detection complete — {n} crate{'s' if n != 1 else ''} found"
        )
        self.results.populate(result, self._object_root)

    @pyqtSlot(str)
    def _on_detection_error(self, tb: str):
        self.sidebar.set_running(False)
        self.centre.set_busy(False)
        self.status_bar.showMessage("Detection failed — see error dialog")
        QMessageBox.critical(self, "Detection error", tb)

    # ── file actions ──────────────────────────────────────────────────────────

    def _browse_snapshot(self):
        path = QFileDialog.getExistingDirectory(self, "Select snapshot folder")
        if path:
            self.sidebar.folder_edit.setText(path)

    def _save_config_json(self):
        self.sidebar.read_into_config()
        self.cfg.grid_corner_pixels = dict(self._corner_pixels)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", "crate_config.json", "JSON (*.json)"
        )
        if path:
            self.cfg.save_json(path)
            self.status_bar.showMessage(f"Config saved → {path}")

    def _load_config_json(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            from crate_vision.config import load_config_from_json
            self.cfg = load_config_from_json(path)
            self._corner_pixels = dict(self.cfg.grid_corner_pixels or {})
            # Rebuild sidebar with new values
            self.sidebar.cfg = self.cfg
            self.sidebar._populate_from_config()
            self._reload_preview()
            self.status_bar.showMessage(f"Config loaded ← {path}")
        except Exception as exc:
            QMessageBox.warning(self, "Load error", str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def launch():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    launch()
