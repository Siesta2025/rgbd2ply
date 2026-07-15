"""Fuse cam1 + cam3 RGBD into multi-label pointclouds.

Merges the functionality previously split across fusion_multilabel.py (bag-mode)
and fusion_split_images.py (prepared-frames mode). Both paths lead to the same
per-frame processing pipeline:

  depth + labels + colour  →  back_project_labeled  →  declutter  →  hand filter
       →  cluster largest  →  cam3→cam1 transform  →  SOR  →  PLY

Supports two input modes, auto-detected:
  - "frames" mode: reads prepared cam{N}_frames/meta.json + labels npz
  - "bags" mode:   reads raw camera_{N}_rgb_depth.bag + labels npz

Usage (CLI):
    # Frames mode (most common — from prepare.py output)
    python fusion.py runs/recording_xyz /data/recording_xyz --out pointclouds/masked_rgb

    # Bags mode
    python fusion.py --cam1-bag cam1.bag --cam3-bag cam3.bag \\
        --cam1-intr cam1.json --cam3-intr cam3.json \\
        --extrinsic extrinsic.json --cam1-labels l1.npz --cam3-labels l3.npz \\
        --out pointclouds/

Usage (module):
    from rgbd2ply.fusion import fuse_run, process_frame, save_ply
"""

import argparse
import bisect
import json
import os
from pathlib import Path

import cv2
import numpy as np

try:
    from .config import cfg
    from .labelspec import label_name, rgb as label_rgb, OBJECT_LABELS
except ImportError:
    from config import cfg
    from labelspec import label_name, rgb as label_rgb, OBJECT_LABELS

try:
    from .camera_utils import obbag
    from .camera_utils import pipeline as base_pipeline
except ImportError:
    from camera_utils import obbag
    from camera_utils import pipeline as base_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAM_RGB = np.array([[220, 60, 60], [60, 120, 220]], np.uint8)  # cam1 red, cam3 blue
HAND_LABEL = 1

# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def _active_labels(labels: np.ndarray) -> list[int]:
    return [int(x) for x in np.unique(labels) if int(x) > 0]


def _label_counts(labels: np.ndarray) -> dict[str, int]:
    return {label_name(x): int((labels == x).sum()) for x in _active_labels(labels)}


# ---------------------------------------------------------------------------
# PLY writer
# ---------------------------------------------------------------------------

def save_ply(
    path: str | Path,
    pts: np.ndarray,
    labels: np.ndarray,
    point_rgb: np.ndarray,
    cam: np.ndarray | None = None,
    color_mode: str = "rgb",
):
    """Write an ASCII .ply with per-point xyz, display colour, and integer label.

    Parameters
    ----------
    color_mode:
        "rgb"        — use original point colour
        "tint"       — original colour overlaid with label colour (same as masked_rgb)
        "masked_rgb" — label-coloured where labelled, original where background
        "label"      — pure label colour (useful for mask-only exports)
        "camera"     — red=cam1, blue=cam3
    """
    labels = labels.astype(np.int32, copy=False)

    if color_mode == "label":
        disp = np.vstack([label_rgb(x) for x in labels]).astype(np.uint8)
    elif color_mode == "camera" and cam is not None:
        disp = CAM_RGB[cam.astype(int)]
    elif color_mode in ("tint", "masked_rgb"):
        disp = point_rgb.copy()
        for lab in _active_labels(labels):
            disp[labels == lab] = label_rgb(lab)
    else:
        disp = point_rgb

    data = np.column_stack([pts, disp, labels])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property int label\nend_header\n")
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d %d")


# ---------------------------------------------------------------------------
# Pointcloud cleaning filters
# ---------------------------------------------------------------------------

def depth_declutter(pts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-label Tukey fence on Z; relabel depth outliers to background."""
    z = pts[:, 2]
    labels = labels.copy()
    for lab in _active_labels(labels):
        m = labels == lab
        if m.sum() < 30:
            continue
        q1, q3 = np.percentile(z[m], [25, 75])
        iqr = q3 - q1
        labels[m & ((z < q1 - 1.5 * iqr) | (z > q3 + 1.5 * iqr))] = 0
    return labels


def hand_depth_filter(
    pts: np.ndarray,
    labels: np.ndarray,
    hand_label: int = HAND_LABEL,
) -> np.ndarray:
    """One-sided depth cleanup for hand masks.

    Hand masks often include background depth at mask boundaries. Keeps the near
    hand surface, relabels unusually far hand pixels as background.
    """
    labels = labels.copy()
    idx = np.where(labels == int(hand_label))[0]
    if len(idx) < 50:
        return labels
    z = pts[idx, 2]
    ok = np.isfinite(z)
    if ok.sum() < 50:
        return labels
    valid_idx = idx[ok]
    z = z[ok]
    q10, q50, q90 = np.percentile(z, [10, 50, 90])
    far_margin = max(0.12, min(0.25, 2.0 * (q90 - q50)))
    near_margin = 0.06
    keep = (z >= q10 - near_margin) & (z <= q50 + far_margin)
    labels[valid_idx[~keep]] = 0
    labels[idx[~ok]] = 0
    return labels


def spatial_declutter(
    pts: np.ndarray,
    labels: np.ndarray,
    k: int = 16,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Statistical outlier removal per label; sparse points become background."""
    from scipy.spatial import cKDTree

    labels = labels.copy()
    for lab in _active_labels(labels):
        idx = np.where(labels == lab)[0]
        if len(idx) < k + 5:
            continue
        p = pts[idx]
        d, _ = cKDTree(p).query(p, k=k + 1)
        md = d[:, 1:].mean(1)
        labels[idx[md > md.mean() + std_ratio * md.std()]] = 0
    return labels


def cluster_keep_largest(
    pts: np.ndarray,
    labels: np.ndarray,
    target: int,
    eps: float = 0.035,
) -> np.ndarray:
    """Keep the largest voxel-connected cluster for a single-instance object label."""
    labels = labels.copy()
    idx = np.where(labels == int(target))[0]
    if len(idx) < 30:
        return labels

    vox = np.floor(pts[idx] / float(eps)).astype(np.int32)
    uniq, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
    if len(uniq) <= 1:
        return labels

    lookup = {tuple(v): i for i, v in enumerate(uniq)}
    seen = np.zeros(len(uniq), dtype=bool)
    best: list = []
    best_count = -1
    neighbors = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                 for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]

    for start in range(len(uniq)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = []
        total = 0
        while stack:
            cur = stack.pop()
            comp.append(cur)
            total += int(counts[cur])
            vx, vy, vz = uniq[cur]
            for dx, dy, dz in neighbors:
                nb = lookup.get((int(vx + dx), int(vy + dy), int(vz + dz)))
                if nb is not None and not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        if total > best_count:
            best_count = total
            best = comp

    keep_vox = np.zeros(len(uniq), dtype=bool)
    keep_vox[np.array(best, dtype=np.int32)] = True
    labels[idx[~keep_vox[inv]]] = 0
    return labels


# ---------------------------------------------------------------------------
# Core per-frame fusion
# ---------------------------------------------------------------------------

def process_frame(
    dep1: np.ndarray,
    lab1: np.ndarray,
    col1: np.ndarray,
    params1: tuple,
    dep3: np.ndarray,
    lab3: np.ndarray,
    col3: np.ndarray,
    params3: tuple,
    r13: np.ndarray,
    t13: np.ndarray,
    declutter: bool = True,
    hand_depth: bool = True,
    sor: bool = True,
    sor_std: float = 2.0,
    cluster_labels: tuple = OBJECT_LABELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fuse a single pair of cam1+cam3 frames into a labelled pointcloud.

    Returns (points, labels, point_rgb, cam_id).
    """
    pc1, l1, rgb1 = base_pipeline.back_project_labeled(dep1, lab1, col1, *params1)
    pc3, l3, rgb3 = base_pipeline.back_project_labeled(dep3, lab3, col3, *params3)

    if declutter:
        l1 = depth_declutter(pc1, l1)
        l3 = depth_declutter(pc3, l3)
    if hand_depth:
        l1 = hand_depth_filter(pc1, l1)
        l3 = hand_depth_filter(pc3, l3)
    for lab in cluster_labels or ():
        l1 = cluster_keep_largest(pc1, l1, int(lab))
        l3 = cluster_keep_largest(pc3, l3, int(lab))

    pc3_in1 = pc3 @ r13.T + t13
    pts = np.vstack([pc1, pc3_in1])
    labs = np.concatenate([l1, l3])
    point_rgb = np.vstack([rgb1, rgb3])
    cam = np.concatenate([np.zeros(len(pc1), np.uint8), np.ones(len(pc3), np.uint8)])

    if sor:
        labs = spatial_declutter(pts, labs, std_ratio=sor_std)

    return pts, labs, point_rgb, cam


# ---------------------------------------------------------------------------
# Frames-mode I/O
# ---------------------------------------------------------------------------

def _load_meta(frames_dir: Path) -> dict:
    return json.load(open(frames_dir / "meta.json"))


def _nearest_index(sorted_values: list, value: float) -> int:
    pos = bisect.bisect_left(sorted_values, value)
    if pos == 0:
        return 0
    if pos == len(sorted_values):
        return len(sorted_values) - 1
    before = pos - 1
    return before if abs(sorted_values[before] - value) <= abs(sorted_values[pos] - value) else pos


def _read_bgr(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return im


def _read_depth(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    return im


def fuse_from_frames(
    run_dir: Path,
    recording_dir: Path,
    extrinsic_json: str | Path | None = None,
    cam1_npz: str | Path | None = None,
    cam3_npz: str | Path | None = None,
    out_dir: str | Path | None = None,
    color_mode: str = "masked_rgb",
    labeled_only: bool = False,
    max_frames: int | None = None,
    max_pair_dt_ms: float = 200.0,
    declutter: bool = True,
    hand_depth: bool = True,
    sor: bool = True,
    sor_std: float = 2.0,
    cluster_labels: tuple = OBJECT_LABELS,
) -> Path:
    """Fuse prepared cam1/cam3 frames into per-frame .ply pointclouds.

    This is the most common mode — used after prepare.py + multi_concept_masks.py.
    """
    run_dir = Path(run_dir)
    recording_dir = Path(recording_dir)
    out_dir = Path(out_dir) if out_dir else run_dir / "pointclouds" / color_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    extrinsic_json = Path(extrinsic_json) if extrinsic_json else Path(cfg.paths.extrinsic)

    # Resolve label npz paths
    cam1_npz = Path(cam1_npz) if cam1_npz else run_dir / "cam1_labels_final.npz"
    cam3_npz = Path(cam3_npz) if cam3_npz else run_dir / "cam3_labels_final.npz"
    if not cam1_npz.exists():
        # Fall back to auto labels
        cam1_npz = run_dir / "cam1_labels_auto.npz"
    if not cam3_npz.exists():
        cam3_npz = run_dir / "cam3_labels_auto.npz"

    m1 = _load_meta(run_dir / "cam1_frames")
    m3 = _load_meta(run_dir / "cam3_frames")
    L1 = np.load(cam1_npz)["labels"]
    L3 = np.load(cam3_npz)["labels"]

    if len(L1) != m1["n_frames"]:
        raise ValueError(f"cam1 labels length {len(L1)} != meta n_frames {m1['n_frames']}")
    if len(L3) != m3["n_frames"]:
        raise ValueError(f"cam3 labels length {len(L3)} != meta n_frames {m3['n_frames']}")

    params1 = base_pipeline.camera_params(
        recording_dir / "camera_1_intrinsics.json",
        recording_dir / "camera_1_rgb_depth.bag",
    )
    params3 = base_pipeline.camera_params(
        recording_dir / "camera_3_intrinsics.json",
        recording_dir / "camera_3_rgb_depth.bag",
    )
    ext = json.load(open(extrinsic_json))["cam3_to_cam1"]
    R13 = np.array(ext["rotation_matrix"])
    T13 = np.array(ext["translation_m"])

    t1 = [int(x) for x in m1["source_rgb_timestamp_us"]]
    wrote = 0
    skipped = 0
    for i3, ts3 in enumerate(m3["source_rgb_timestamp_us"]):
        i1 = _nearest_index(t1, int(ts3))
        dt_ms = abs(t1[i1] - int(ts3)) / 1000.0
        if dt_ms > max_pair_dt_ms:
            skipped += 1
            continue

        dep1 = _read_depth(m1["source_depth"][i1])
        dep3 = _read_depth(m3["source_depth"][i3])
        col1 = _read_bgr(m1["source_rgb"][i1])
        col3 = _read_bgr(m3["source_rgb"][i3])

        pts, labs, point_rgb, cam = process_frame(
            dep1, L1[i1], col1, params1,
            dep3, L3[i3], col3, params3,
            R13, T13,
            declutter=declutter, hand_depth=hand_depth,
            sor=sor, sor_std=sor_std, cluster_labels=cluster_labels,
        )

        if labeled_only:
            keep = labs > 0
            pts, labs, point_rgb, cam = pts[keep], labs[keep], point_rgb[keep], cam[keep]

        frame_no = int(m3["frame_indices"][i3])
        save_ply(
            str(out_dir / f"frame_{frame_no:06d}.ply"),
            pts, labs, point_rgb, cam, color_mode,
        )
        print(f"frame {frame_no:06d}  cam1={int(m1['frame_indices'][i1]):06d}  "
              f"dt={dt_ms:.1f}ms  pts={len(labs):>7}  {_label_counts(labs)}",
              flush=True)
        wrote += 1
        if max_frames and wrote >= max_frames:
            break

    print(f"Wrote {wrote} clouds → {out_dir}  (skipped {skipped} frames by dt threshold)",
          flush=True)
    return out_dir


# ---------------------------------------------------------------------------
# Bags-mode I/O
# ---------------------------------------------------------------------------

def fuse_from_bags(
    cam1_bag: str | Path,
    cam3_bag: str | Path,
    cam1_intr_json: str | Path,
    cam3_intr_json: str | Path,
    extrinsic_json: str | Path,
    cam1_npz: str | Path,
    cam3_npz: str | Path,
    out_dir: str | Path,
    stride: int = 30,
    color_mode: str = "rgb",
    labeled_only: bool = False,
    max_frames: int | None = None,
    declutter: bool = True,
    hand_depth: bool = True,
    sor: bool = True,
    sor_std: float = 2.0,
    cluster_labels: tuple = OBJECT_LABELS,
) -> Path:
    """Fuse directly from ROS bags (older interface, for unprocessed recordings)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    l1_all = np.load(cam1_npz)["labels"]
    l3_all = np.load(cam3_npz)["labels"]
    params1 = base_pipeline.camera_params(cam1_intr_json, cam1_bag)
    params3 = base_pipeline.camera_params(cam3_intr_json, cam3_bag)

    ext = json.load(open(extrinsic_json))["cam3_to_cam1"]
    r13 = np.array(ext["rotation_matrix"])
    t13 = np.array(ext["translation_m"])

    s1d = obbag.frame_stream(str(cam1_bag), "depth", stride)
    s3d = obbag.frame_stream(str(cam3_bag), "depth", stride)
    s1c = obbag.frame_stream(str(cam1_bag), "color", stride)
    s3c = obbag.frame_stream(str(cam3_bag), "color", stride)
    n_labels = min(len(l1_all), len(l3_all))

    n = 0
    for t, ((i1, _, dep1), (_, _, dep3), (_, _, col1), (_, _, col3)) in enumerate(
        zip(s1d, s3d, s1c, s3c)
    ):
        if t >= n_labels:
            break
        pts, labs, point_rgb, cam = process_frame(
            dep1, l1_all[t], col1, params1,
            dep3, l3_all[t], col3, params3,
            r13, t13,
            declutter=declutter, hand_depth=hand_depth,
            sor=sor, sor_std=sor_std, cluster_labels=cluster_labels,
        )
        if labeled_only:
            keep = labs > 0
            pts, labs, point_rgb, cam = pts[keep], labs[keep], point_rgb[keep], cam[keep]
        save_ply(
            str(out_dir / f"frame_{i1:06d}.ply"),
            pts, labs, point_rgb, cam, color_mode,
        )
        print(f"frame {i1:6d}: {len(labs):7d} pts  {_label_counts(labs)}", flush=True)
        n += 1
        if max_frames and n >= max_frames:
            break

    print(f"Wrote {n} clouds → {out_dir}  (color_mode={color_mode})", flush=True)
    return out_dir


def fuse_run(
    run_dir: str | Path,
    recording_dir: str | Path | None = None,
    **kwargs,
) -> list[Path]:
    """Convenience: fuse a run directory with defaults for both colour modes.

    Returns [real_color_dir, mask_only_dir].
    """
    run_dir = Path(run_dir)
    recording_dir = Path(recording_dir) if recording_dir else run_dir

    out_dirs = []
    for mode, labeled_only in [("masked_rgb", False), ("label", True)]:
        out = fuse_from_frames(
            run_dir=run_dir,
            recording_dir=recording_dir,
            out_dir=run_dir / "pointclouds" / mode,
            color_mode=mode,
            labeled_only=labeled_only,
            **kwargs,
        )
        out_dirs.append(out)
    return out_dirs


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fuse cam1+cam3 RGBD into multi-label pointclouds."
    )

    # ---- Frames mode args (positional) ----
    ap.add_argument("run_dir", nargs="?", default=None,
                    help="Run directory (from prepare.py). Frames mode.")
    ap.add_argument("recording_dir", nargs="?", default=None,
                    help="Recording directory with intrinsics/bags. Frames mode.")

    # ---- Bags mode args (all explicit) ----
    ap.add_argument("--cam1-bag", default=None)
    ap.add_argument("--cam3-bag", default=None)
    ap.add_argument("--cam1-intr", default=None)
    ap.add_argument("--cam3-intr", default=None)
    ap.add_argument("--extrinsic", default=None,
                    help="Cam3→Cam1 extrinsic JSON (default from config)")
    ap.add_argument("--cam1-labels", default=None)
    ap.add_argument("--cam3-labels", default=None)

    # ---- Shared args ----
    ap.add_argument("--out", default=None, help="Output directory for PLY files")
    ap.add_argument("--color-mode", choices=["rgb", "tint", "masked_rgb", "camera", "label"],
                    default="masked_rgb")
    ap.add_argument("--labeled-only", action="store_true")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--max-pair-dt-ms", type=float, default=200.0)
    ap.add_argument("--stride", type=int, default=30, help="Stride for bag mode")
    ap.add_argument("--no-declutter", action="store_true")
    ap.add_argument("--no-hand-depth-filter", action="store_true")
    ap.add_argument("--no-object-cluster", action="store_true")
    ap.add_argument("--no-sor", action="store_true")
    ap.add_argument("--sor-std", type=float, default=2.0)
    ap.add_argument("--cluster-label", type=int, action="append", default=None)
    args = ap.parse_args()

    clusters = ([] if args.no_object_cluster
                else (tuple(args.cluster_label) if args.cluster_label else OBJECT_LABELS))

    if args.cam1_bag and args.cam3_bag:
        # ---- Bags mode ----
        if not args.out:
            ap.error("--out is required in bags mode")
        extrinsic = args.extrinsic or cfg.paths.extrinsic
        fuse_from_bags(
            cam1_bag=args.cam1_bag,
            cam3_bag=args.cam3_bag,
            cam1_intr_json=args.cam1_intr,
            cam3_intr_json=args.cam3_intr,
            extrinsic_json=extrinsic,
            cam1_npz=args.cam1_labels,
            cam3_npz=args.cam3_labels,
            out_dir=args.out,
            stride=args.stride,
            color_mode=args.color_mode,
            labeled_only=args.labeled_only,
            max_frames=args.max_frames,
            declutter=not args.no_declutter,
            hand_depth=not args.no_hand_depth_filter,
            sor=not args.no_sor,
            sor_std=args.sor_std,
            cluster_labels=clusters,
        )
    elif args.run_dir:
        # ---- Frames mode ----
        if not args.out:
            ap.error("--out is required")
        rec_dir = args.recording_dir or args.run_dir
        fuse_from_frames(
            run_dir=args.run_dir,
            recording_dir=Path(rec_dir),
            extrinsic_json=args.extrinsic,
            cam1_npz=args.cam1_labels,
            cam3_npz=args.cam3_labels,
            out_dir=args.out,
            color_mode=args.color_mode,
            labeled_only=args.labeled_only,
            max_frames=args.max_frames,
            max_pair_dt_ms=args.max_pair_dt_ms,
            declutter=not args.no_declutter,
            hand_depth=not args.no_hand_depth_filter,
            sor=not args.no_sor,
            sor_std=args.sor_std,
            cluster_labels=clusters,
        )
    else:
        ap.error("Provide run_dir+recording_dir (frames mode) or --cam1-bag+--cam3-bag (bags mode)")
