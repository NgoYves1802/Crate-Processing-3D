"""
crate_vision.detection
======================
Object detection and geometric analysis.

Modules:
    depth    - Depth layer masking and blob removal
    ccl      - Connected component labeling and size filtering
    geometry - Geometric fitting, corner detection, plane estimation
    slots    - Crate slot analysis and grid detection
"""

from . import depth
from . import ccl
from . import geometry
from . import slots

__all__ = ["depth", "ccl", "geometry", "slots"]