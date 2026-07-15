"""Stable label helpers for the rgbd2ply split pipeline.

Canonical ids live in object_registry.json:
  0 bg, 1 hand, 2 ice_box, 3 transparent_cup, 4 bucket_a,
  5 bucket_b, 6 metal_pot, 7 metal_funnel.
"""
import numpy as np

LABEL_NAMES = {
    0: "bg",
    1: "hand",
    2: "ice_box",
    3: "transparent_cup",
    4: "bucket_a",
    5: "bucket_b",
    6: "metal_pot",
    7: "metal_funnel",
}
OBJECT_LABELS = (2, 3, 4, 5, 6, 7)
N_LABELS = 8

# Objects first, hand last so a hand covering an object wins overlap.
PAINT_ORDER = (2, 3, 4, 5, 6, 7, 1)

LABEL_RGB = np.array([
    [160, 160, 160],  # bg
    [220, 40, 40],    # hand
    [40, 200, 40],    # ice_box
    [60, 180, 230],   # transparent_cup
    [200, 60, 200],   # bucket_a
    [240, 130, 40],   # bucket_b
    [20, 70, 190],    # metal_pot
    [200, 40, 160],   # metal_funnel
], np.uint8)


def label_name(label):
    return LABEL_NAMES.get(int(label), "label_%d" % int(label))


def rgb(label):
    """Stable RGB colour for a label, including labels not in LABEL_RGB."""
    label = int(label)
    if 0 <= label < len(LABEL_RGB):
        return LABEL_RGB[label]
    x = (label * 1103515245 + 12345) & 0x7fffffff
    return np.array([80 + (x & 127), 80 + ((x >> 7) & 127), 80 + ((x >> 14) & 127)], np.uint8)


def bgr(label):
    r, g, b = rgb(label)
    return int(b), int(g), int(r)
