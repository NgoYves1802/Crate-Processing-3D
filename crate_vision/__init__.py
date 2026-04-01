"""
crate_vision package
===================
3-D crate detection pipeline for the IFM O3D303 depth camera + S7-1200 PLC.

This package provides:
- Main pipeline orchestration
- Depth layer processing
- Object pose estimation
- AI verification
- Hardware interfaces (camera + PLC)
- Data I/O and serialization

Usage:
    from crate_vision.pipeline import process_depth_layers
    result = process_depth_layers("snapshots/snap0001_20260323_171356")
"""

__version__ = "1.0.0"
__author__ = "Crate Vision Team"

# Import main modules for convenience
from . import config
from . import pipeline
from . import pose
from . import ai_verifier

# Hardware modules
from . import hardware

# I/O modules  
from . import io

# Detection modules
from . import detection

__all__ = [
    "config",
    "pipeline", 
    "pose",
    "ai_verifier",
    "hardware",
    "io",
    "detection",
]