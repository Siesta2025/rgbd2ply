"""rgbd2ply — multi-camera RGBD → labelled pointcloud pipeline.

Tools for building multi-view labelled pointclouds from ego-centric RGBD
recordings, using SAM3 open-vocabulary segmentation + manual review.
"""

__version__ = "2.0.0"

from .config import cfg
from .discovery import discover, SegmentInfo
