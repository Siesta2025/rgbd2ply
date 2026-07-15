"""Frame extraction for the rgbd2ply pipeline.

Extracts time-windowed cam1/cam3 frames and depth maps from recording folders
or rosbags into standardized frame directories for downstream SAM3 processing.

Input sources (auto-detected):
  - Image folders:  recording/camera_{N}_image/rgb/frame_times.csv + depth/
  - ROS bags:       recording/camera_{N}_rgb_depth.bag
  - Mixed:          cam1 from bag, cam3 from images (or vice versa)

Output per run:
  runs/<id>/cam{N}_frames/
    00000.jpg ... 000NN.jpg    (colour frames for SAM3)
    meta.json                  (source paths, timestamps, intrinsics refs)
  runs/<id>/cam{N}_depth/
    00000.png ... 000NN.png    (depth frames for fusion)

Usage (CLI):
    python prepare.py recording_20260701_195739
    python prepare.py --manifest segments.csv --session-root /data/sessions

Usage (module):
    from rgbd2ply.prepare import prepare, prepare_recording, prepare_manifest_row
"""

import csv
import json
import os
from pathlib import Path

import cv2

try:
    from .config import cfg
    from .discovery import (discover, SegmentInfo,
                            _manifest_filename, _manifest_segment_id,
                            _manifest_recording_dir)
except ImportError:
    from config import cfg
    from discovery import (discover, SegmentInfo,
                           _manifest_filename, _manifest_segment_id,
                           _manifest_recording_dir)

try:
    from .camera_utils import obbag
except ImportError:
    from camera_utils import obbag


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_times(csv_path: Path) -> list[dict]:
    """Read a frame_times.csv into a list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def by_file(rows: list[dict]) -> dict[str, dict]:
    """Index rows by 'file_name'."""
    return {r["file_name"]: r for r in rows}


def nearest_depth(depth_rows: list[dict], ts_us: int) -> dict:
    """Find the depth row closest to a given colour timestamp."""
    return min(depth_rows, key=lambda r: abs(int(r["timestamp_us"]) - int(ts_us)))


def choose_rgb_rows(
    rec_dir: Path,
    cam: int,
    start_us: int,
    end_us: int,
    cam3_names: list[str] | None = None,
) -> list[dict]:
    """Select colour frame rows within a time window.

    When cam3_names is given, match by file name first (used for manifest mode
    where the exact cam3 frames are known); otherwise filter by timestamp range.
    """
    rgb_dir = rec_dir / f"camera_{cam}_image" / "rgb"
    rows = read_times(rgb_dir / "frame_times.csv")
    if cam3_names is not None:
        by = by_file(rows)
        chosen = [by[name] for name in cam3_names if name in by]
        if chosen:
            return chosen
    return [r for r in rows if start_us <= int(r["timestamp_us"]) <= end_us]


# ---------------------------------------------------------------------------
# Bag extraction
# ---------------------------------------------------------------------------

def write_bag_frames(
    rec_dir: Path,
    cam: int,
    out_dir: Path,
    quality: int = 90,
    stride: int = 30,
) -> dict | None:
    """Extract frames from a camera_{cam}_rgb_depth.bag.

    Returns meta dict, or None if no bag was found.
    """
    bag = rec_dir / f"camera_{cam}_rgb_depth.bag"
    if not bag.exists():
        return None

    out_dir = Path(out_dir)
    depth_out = out_dir.parent / f"cam{cam}_depth"
    out_dir.mkdir(parents=True, exist_ok=True)
    depth_out.mkdir(parents=True, exist_ok=True)

    depth_rows = []
    for di, dts, dep in obbag.frame_stream(str(bag), "depth", stride=stride):
        name = f"{len(depth_rows):05d}.png"
        path = depth_out / name
        cv2.imwrite(str(path), dep)
        depth_rows.append({
            "frame_index": int(di),
            "timestamp_us": int(dts) // 1000,
            "file_name": name,
            "path": str(path),
        })

    if not depth_rows:
        raise ValueError(f"no depth frames extracted from {bag}")

    meta = {
        "camera": cam,
        "width": 0,
        "height": 0,
        "n_frames": 0,
        "frame_indices": [],
        "timestamps": [],
        "source_rgb": [],
        "source_depth": [],
        "source_rgb_frame_name": [],
        "source_depth_frame_name": [],
        "source_rgb_timestamp_us": [],
        "source_depth_timestamp_us": [],
        "source_bag": str(bag),
        "source_stride": int(stride),
    }

    for ci, cts, im in obbag.frame_stream(str(bag), "color", stride=stride):
        n = meta["n_frames"]
        rgb_name = f"{n:05d}.jpg"
        rgb_path = out_dir / rgb_name
        cv2.imwrite(str(rgb_path), im, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        c_us = int(cts) // 1000
        drow = nearest_depth(depth_rows, c_us)
        h, w = im.shape[:2]
        meta["width"], meta["height"] = w, h
        meta["frame_indices"].append(int(ci))
        meta["timestamps"].append(c_us)
        meta["source_rgb"].append(str(rgb_path))
        meta["source_depth"].append(drow["path"])
        meta["source_rgb_frame_name"].append(rgb_name)
        meta["source_depth_frame_name"].append(drow["file_name"])
        meta["source_rgb_timestamp_us"].append(c_us)
        meta["source_depth_timestamp_us"].append(int(drow["timestamp_us"]))
        meta["n_frames"] += 1

    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# ---------------------------------------------------------------------------
# Image-folder extraction
# ---------------------------------------------------------------------------

def write_frames(
    rec_dir: Path,
    cam: int,
    rows: list[dict],
    out_dir: Path,
    quality: int = 90,
) -> dict:
    """Copy frames from an image-folder recording into a numbered JPG sequence.

    `rows` come from frame_times.csv, filtered to the desired time window.
    """
    rgb_dir = rec_dir / f"camera_{cam}_image" / "rgb"
    depth_dir = rec_dir / f"camera_{cam}_image" / "depth"
    depth_rows = read_times(depth_dir / "frame_times.csv")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "camera": cam,
        "width": 0,
        "height": 0,
        "n_frames": 0,
        "frame_indices": [],
        "timestamps": [],
        "source_rgb": [],
        "source_depth": [],
        "source_rgb_frame_name": [],
        "source_depth_frame_name": [],
        "source_rgb_timestamp_us": [],
        "source_depth_timestamp_us": [],
    }

    for i, row in enumerate(rows):
        src = rgb_dir / row["file_name"]
        im = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if im is None:
            continue
        drow = nearest_depth(depth_rows, int(row["timestamp_us"]))
        dst = out_dir / f"{i:05d}.jpg"
        cv2.imwrite(str(dst), im, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        h, w = im.shape[:2]
        meta["width"], meta["height"] = w, h
        meta["frame_indices"].append(int(Path(row["file_name"]).stem))
        meta["timestamps"].append(int(row["timestamp_us"]))
        meta["source_rgb"].append(str(src))
        meta["source_depth"].append(str(depth_dir / drow["file_name"]))
        meta["source_rgb_frame_name"].append(row["file_name"])
        meta["source_depth_frame_name"].append(drow["file_name"])
        meta["source_rgb_timestamp_us"].append(int(row["timestamp_us"]))
        meta["source_depth_timestamp_us"].append(int(drow["timestamp_us"]))

    meta["n_frames"] = len(meta["frame_indices"])
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


# ---------------------------------------------------------------------------
# Top-level prepare functions
# ---------------------------------------------------------------------------

def prepare_manifest_row(
    row: dict,
    out_root: Path,
    session_root: Path,
    quality: int = 90,
) -> dict:
    """Prepare cam1 + cam3 frames from a split_segments manifest CSV row."""
    rec_dir = _manifest_recording_dir(row, session_root)
    seg_dir = out_root / _manifest_segment_id(row)

    cam3_rgb_dir = rec_dir / "camera_3_image" / "rgb"
    cam3_times = by_file(read_times(cam3_rgb_dir / "frame_times.csv"))
    sname = row["original_start_frame_name"] + ".png"
    ename = row["original_end_frame_name_inclusive"] + ".png"
    start_us = int(cam3_times[sname]["timestamp_us"])
    end_us = int(cam3_times[ename]["timestamp_us"])
    start_num = int(row["original_start_frame_name"])
    end_num = int(row["original_end_frame_name_inclusive"])
    cam3_names = [f"{n:06d}.png" for n in range(start_num, end_num + 1)]

    out = {
        "segment_id": _manifest_segment_id(row),
        "session": row["session"],
        "segment_dir": str(seg_dir),
        "source_manifest_row": row,
    }
    cam3_rows = choose_rgb_rows(rec_dir, 3, start_us, end_us, cam3_names=cam3_names)
    cam1_rows = choose_rgb_rows(rec_dir, 1, start_us, end_us, cam3_names=None)
    out["cam3"] = write_frames(rec_dir, 3, cam3_rows, seg_dir / "cam3_frames", quality=quality)
    out["cam1"] = write_frames(rec_dir, 1, cam1_rows, seg_dir / "cam1_frames", quality=quality)
    (seg_dir / "segment.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return out


def prepare_recording(
    rec_dir: Path,
    out_root: Path | None = None,
    quality: int = 90,
    stride: int = 30,
) -> dict:
    """Prepare cam1 + cam3 frames from a whole recording folder.

    Auto-detects image-folder vs bag vs mixed sources per camera.
    """
    rec_dir = Path(rec_dir)
    out_root = Path(out_root) if out_root else Path(cfg.paths.runs_root)
    seg_dir = out_root / rec_dir.name

    out = {
        "segment_id": rec_dir.name,
        "session": rec_dir.name,
        "recording_dir": str(rec_dir),
        "segment_dir": str(seg_dir),
        "source_mode": "recording",
        "recording_stride": int(stride),
        "cameras": [],
    }

    image_csv = rec_dir / "camera_3_image" / "rgb" / "frame_times.csv"
    if image_csv.exists():
        cam3_rows = read_times(image_csv)[::max(1, int(stride))]
        if cam3_rows:
            start_us = min(int(r["timestamp_us"]) for r in cam3_rows)
            end_us = max(int(r["timestamp_us"]) for r in cam3_rows)
            out["cam3"] = write_frames(rec_dir, 3, cam3_rows, seg_dir / "cam3_frames", quality=quality)
            out["cameras"].append(3)

            cam1_csv = rec_dir / "camera_1_image" / "rgb" / "frame_times.csv"
            if cam1_csv.exists():
                cam1_rows = choose_rgb_rows(rec_dir, 1, start_us, end_us, cam3_names=None)
                cam1_rows = cam1_rows[::max(1, int(stride))]
                out["cam1"] = write_frames(rec_dir, 1, cam1_rows, seg_dir / "cam1_frames", quality=quality)
                out["cameras"].append(1)
            else:
                meta = write_bag_frames(rec_dir, 1, seg_dir / "cam1_frames", quality=quality, stride=stride)
                if meta:
                    out["cam1"] = meta
                    out["cameras"].append(1)
    else:
        # No image CSVs — try bags for both cameras
        for cam in (1, 3):
            meta = write_bag_frames(rec_dir, cam, seg_dir / f"cam{cam}_frames", quality=quality, stride=stride)
            if meta:
                out[f"cam{cam}"] = meta
                out["cameras"].append(cam)

    if not out["cameras"]:
        raise FileNotFoundError(
            f"No cam1/cam3 image csv or rgb_depth.bag found in {rec_dir}"
        )
    out["cameras"] = sorted(set(out["cameras"]))
    (seg_dir / "segment.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    return out


def prepare(
    seg: SegmentInfo,
    out_root: Path | None = None,
    quality: int = 90,
    stride: int = 30,
) -> dict:
    """Unified entry point: prepare frames for one SegmentInfo.

    Returns the output dict (see prepare_recording / prepare_manifest_row).
    """
    if seg.mode == "manifest":
        row = seg.metadata.get("manifest_row", {})
        session_root = Path(cfg.paths.data_root)
        return prepare_manifest_row(row, out_root or Path(cfg.paths.runs_root), session_root, quality)
    else:
        return prepare_recording(seg.recording_dir, out_root, quality, stride)


# ---------------------------------------------------------------------------
# CLI entry point
# Standalone usage: python -m rgbd2ply run <source> --steps prepare
