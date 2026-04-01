"""
crate_vision/hardware/camera.py
================================
IFM O3D303 camera interface.

  GrabO3D300      — frame capture, snapshot save
  configure_camera — XML-RPC camera configuration
  setup_pcic_stream — o3d3xx.FormatClient setup
"""

from __future__ import annotations

import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xmlrpc.client

# o3d3xx is required at runtime but omitted from imports so the rest of
# the package can be imported on machines without it.
try:
    import o3d3xx
    _O3D_AVAILABLE = True
except ImportError:
    _O3D_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_WIDTH  = 176
DEFAULT_IMAGE_HEIGHT = 132
DEFAULT_OUTPUT_DIR   = "snapshots"
DEFAULT_PCIC_PORT    = 50010

# Camera parameters settable via XML-RPC
_APP_PARAMS    = {"TriggerMode"}
_IMAGER_PARAMS = {
    "ExposureTime", "FrameRate", "MinimumAmplitude",
    "SymmetryThreshold", "Channel", "Resolution",
}


# ---------------------------------------------------------------------------
# GrabO3D300
# ---------------------------------------------------------------------------

class GrabO3D300:
    """
    Holds one frame worth of data and handles snapshot persistence.

    Usage
    -----
        grabber = GrabO3D300()
        grabber.load_from_frame(frame_dict)
        folder = grabber.save_snapshot()
    """

    def __init__(
        self,
        image_width:  int = DEFAULT_IMAGE_WIDTH,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        output_dir:   str = DEFAULT_OUTPUT_DIR,
    ):
        self.image_width  = image_width
        self.image_height = image_height
        self.output_dir   = output_dir
        self.snap_counter = 0

        self.Amplitude = np.zeros((image_height, image_width), dtype=np.uint16)
        self.Distance  = np.zeros((image_height, image_width), dtype=np.uint16)
        self.X         = np.zeros((image_height, image_width), dtype=np.int16)
        self.Y         = np.zeros((image_height, image_width), dtype=np.int16)
        self.Z         = np.zeros((image_height, image_width), dtype=np.int16)
        self.illuTemp  = 20.0

        os.makedirs(self.output_dir, exist_ok=True)

    def load_from_frame(self, frame: dict) -> None:
        """
        Populate arrays from an o3d3xx.FormatClient.readNextFrame() dict.

        Expected keys: amplitude_image, distance_image, x_image, y_image, z_image.
        """
        H, W = self.image_height, self.image_width

        self.Amplitude = (
            np.frombuffer(frame["amplitude_image"], dtype="uint16").reshape(H, W)
        )
        self.Distance = (
            np.frombuffer(frame["distance_image"], dtype="uint16").reshape(H, W)
        )
        for attr, key in [("X", "x_image"), ("Y", "y_image"), ("Z", "z_image")]:
            if key in frame:
                setattr(self, attr,
                        np.frombuffer(frame[key], dtype="int16").reshape(H, W))
        self.illuTemp = 20.0

    def save_snapshot(self) -> str:
        """
        Save all arrays and visualisations to a timestamped sub-folder.

        Returns
        -------
        str : path of the created snapshot folder.
        """
        self.snap_counter += 1
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_id  = f"snap{self.snap_counter:04d}_{timestamp}"
        folder       = os.path.join(self.output_dir, snapshot_id)
        os.makedirs(folder, exist_ok=True)

        # Raw arrays
        np.save(os.path.join(folder, "amplitude.npy"),    self.Amplitude)
        np.save(os.path.join(folder, "distance.npy"),     self.Distance)
        np.save(os.path.join(folder, "x_coords.npy"),     self.X)
        np.save(os.path.join(folder, "y_coords.npy"),     self.Y)
        np.save(os.path.join(folder, "z_coords.npy"),     self.Z)

        xyz_combined = np.stack([self.X, self.Y, self.Z], axis=2)
        np.save(os.path.join(folder, "xyz_combined.npy"), xyz_combined)

        # Point clouds
        self._save_pointcloud_xyz(os.path.join(folder, "pointcloud.xyz"))
        self._save_pointcloud_ply(os.path.join(folder, "pointcloud.ply"))

        # PNG previews
        plt.imsave(os.path.join(folder, "amplitude.png"), self.Amplitude, cmap="viridis")
        plt.imsave(os.path.join(folder, "distance.png"),  self.Distance,  cmap="viridis")
        plt.imsave(os.path.join(folder, "depth_z.png"),   self.Z,         cmap="jet")

        # Metadata text
        self._save_metadata_txt(folder, snapshot_id, timestamp)

        return folder

    # ── Private helpers ───────────────────────────────────────────────────────

    def _save_metadata_txt(self, folder: str, snapshot_id: str, timestamp: str) -> None:
        valid_points = int(np.sum(self.Z != 0))
        lines = [
            f"Snapshot ID: {snapshot_id}",
            f"Snap Number: {self.snap_counter}",
            f"Timestamp: {timestamp}",
            f"Datetime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Image Width: {self.image_width}",
            f"Image Height: {self.image_height}",
            f"Illumination Temperature: {self.illuTemp}",
            "",
            "Amplitude Stats:",
            f"  Min: {np.min(self.Amplitude)}",
            f"  Max: {np.max(self.Amplitude)}",
            f"  Mean: {np.mean(self.Amplitude):.2f}",
            f"  Std: {np.std(self.Amplitude):.2f}",
            "",
            "Distance Stats:",
            f"  Min: {np.min(self.Distance)}",
            f"  Max: {np.max(self.Distance)}",
            f"  Mean: {np.mean(self.Distance):.2f}",
            f"  Std: {np.std(self.Distance):.2f}",
            "",
            f"Valid points: {valid_points}",
        ]
        with open(os.path.join(folder, "metadata.txt"), "w") as f:
            f.write("\n".join(lines))

    def _save_pointcloud_xyz(self, filename: str) -> None:
        H, W = self.image_height, self.image_width
        with open(filename, "w") as f:
            for i in range(H):
                for j in range(W):
                    if self.Z[i, j] != 0:
                        f.write(f"{self.X[i,j]} {self.Y[i,j]} {self.Z[i,j]}\n")

    def _save_pointcloud_ply(self, filename: str) -> None:
        H, W = self.image_height, self.image_width
        valid_mask = self.Z != 0
        num_valid  = int(np.sum(valid_mask))
        amp_max    = float(np.max(self.Amplitude)) or 1.0
        amp_norm   = np.clip(self.Amplitude.astype(float) / amp_max * 255, 0, 255).astype(np.uint8)

        with open(filename, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {num_valid}\n")
            for prop in ("float x", "float y", "float z",
                         "uchar red", "uchar green", "uchar blue"):
                f.write(f"property {prop}\n")
            f.write("end_header\n")
            for i in range(H):
                for j in range(W):
                    if valid_mask[i, j]:
                        c = amp_norm[i, j]
                        f.write(f"{self.X[i,j]} {self.Y[i,j]} {self.Z[i,j]} {c} {c} {c}\n")


# ---------------------------------------------------------------------------
# XML-RPC camera configuration
# ---------------------------------------------------------------------------

def configure_camera(ip: str, camera_config: dict) -> None:
    """
    Configure the O3D303 via XML-RPC.

    Correct navigation sequence:
      1. requestSession()
      2. setOperatingMode(1)  — enter edit
      3. editApplication(1)   — attach application object
      4. setParameter() calls on app / imager
      5. app.save()
      6. stopEditingApplication()
      7. setOperatingMode(0)  — exit edit
      8. cancelSession()

    Parameters
    ----------
    ip            : camera IP address
    camera_config : dict of parameter name → string value
                    e.g. {"TriggerMode": "3"}
    """
    base    = f"http://{ip}/api/rpc/v1/com.ifm.efector"
    main    = xmlrpc.client.ServerProxy(f"{base}/")

    sid     = main.requestSession("", "")
    session = xmlrpc.client.ServerProxy(f"{base}/session_{sid}/")
    edit    = xmlrpc.client.ServerProxy(f"{base}/session_{sid}/edit/")
    app     = xmlrpc.client.ServerProxy(f"{base}/session_{sid}/edit/application/")
    imager  = xmlrpc.client.ServerProxy(
        f"{base}/session_{sid}/edit/application/imager_001/"
    )

    try:
        session.setOperatingMode(1)
        edit.editApplication(1)

        for key, value in camera_config.items():
            if key in _APP_PARAMS:
                app.setParameter(key, value)
            elif key in _IMAGER_PARAMS:
                imager.setParameter(key, value)

        actual = app.getParameter("TriggerMode")
        if str(actual) != "3":
            raise RuntimeError(f"TriggerMode mismatch: got {actual}")

        app.save()
        edit.stopEditingApplication()

    finally:
        session.setOperatingMode(0)
        session.cancelSession()


# ---------------------------------------------------------------------------
# PCIC stream setup
# ---------------------------------------------------------------------------

def setup_pcic_stream(
    camera_ip: str,
    pcic_port: int,
    pcic_schema,
):
    """
    Create and return an o3d3xx.FormatClient.

    Parameters
    ----------
    camera_ip   : camera IP address
    pcic_port   : PCIC TCP port (default 50010)
    pcic_schema : o3d3xx.PCICFormat blob schema

    Returns
    -------
    o3d3xx.FormatClient instance, connected and ready.

    Raises
    ------
    ImportError if o3d3xx is not installed.
    RuntimeError if connection fails.
    """
    if not _O3D_AVAILABLE:
        raise ImportError("o3d3xx is not installed — cannot connect to camera.")
    return o3d3xx.FormatClient(camera_ip, pcic_port, pcic_schema)
