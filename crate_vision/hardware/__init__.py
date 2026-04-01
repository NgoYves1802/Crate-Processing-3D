"""
crate_vision.hardware
====================
Hardware interfaces for camera and PLC communication.

Modules:
    camera - IFM O3D303 camera interface and configuration
    plc    - S7-1200 PLC communication via Snap7
"""

from . import camera
from . import plc

__all__ = ["camera", "plc"]