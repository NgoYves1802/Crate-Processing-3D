"""
crate_vision.io
==============
Data loading and serialization utilities.

Modules:
    loader     - Load amplitude images and XYZ point clouds
    serializer - Pack and write crate detection results
"""

from . import loader
from . import serializer

__all__ = ["loader", "serializer"]