"""Input source discovery for the rgbd2ply pipeline.

Unifies the scattered manifest/recording/glob logic across the old codebase into a
single `discover()` entry point.

Supports:
  - Recording folder (auto-detected: image frames, rosbag, or mixed)
  - Manifest CSV (split_segments style with column mapping)
  - Simple CSV (one recording per row, column mapping)
  - Glob pattern over directories
  - Explicit list of paths

Usage:
    from rgbd2ply.discovery import discover, SegmentInfo

    segs = discover("/data/recordings/camera_glove_recording_*")
    segs = discover("manifest.csv", csv_style="split_segments")
    segs = discover("list.csv", csv_map={"recording_dir": "path"})
"""

import csv
import glob as _glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SegmentInfo:
    """A single data unit ready for the pipeline.

    After discovery, every segment has a unique id, a path to its recording
    directory, a mode tag, and optional source metadata.
    """
    id: str
    recording_dir: Path
    mode: str = "recording"          # "recording" | "manifest" | "bag_list"
    metadata: dict = field(default_factory=dict)

    def __repr__(self):
        return f"SegmentInfo(id={self.id!r}, mode={self.mode!r}, dir={self.recording_dir!r})"


# ---------------------------------------------------------------------------
# Internal detectors
# ---------------------------------------------------------------------------

def _recording_has_images(rec: Path) -> bool:
    """True if `rec` has camera_*_image/rgb/frame_times.csv."""
    return ((rec / "camera_1_image" / "rgb" / "frame_times.csv").exists() or
            (rec / "camera_3_image" / "rgb" / "frame_times.csv").exists())

def _recording_has_bags(rec: Path) -> bool:
    """True if `rec` has camera_*_rgb_depth.bag."""
    return ((rec / "camera_1_rgb_depth.bag").exists() or
            (rec / "camera_3_rgb_depth.bag").exists())

def _detect_recording_type(rec: Path) -> str:
    """Return 'images' | 'bags' | 'mixed' | 'empty'."""
    has_img = _recording_has_images(rec)
    has_bag = _recording_has_bags(rec)
    if has_img and has_bag:
        return "mixed"
    if has_img:
        return "images"
    if has_bag:
        return "bags"
    return "empty"


def _is_manifest_csv(path: Path) -> bool:
    """Heuristic: does this CSV look like a split_segments manifest?"""
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
    except Exception:
        return False
    return "session" in fields and ("split_video_filename" in fields or
                                    "final_unmarked_video_filename" in fields or
                                    "final_unmarked_video" in fields)


def _find_recording_dirs(root: Path, names: list[str]) -> list[Path]:
    """Resolve recording names relative to root."""
    result = []
    for name in names:
        p = Path(name)
        if not p.is_absolute():
            p = root / name
        if p.is_dir():
            result.append(p)
    return result


# ---------------------------------------------------------------------------
# CSV readers
# ---------------------------------------------------------------------------

def _read_manifest_csv(path: Path, include_cali: bool = False) -> list[dict]:
    """Read a split_segments-style manifest CSV.

    Returns list of rows as dicts. Each row must have 'session' and a video
    filename column.
    """
    rows = list(csv.DictReader(open(path)))
    if not include_cali:
        rows = [r for r in rows
                if not r.get("session", "").startswith("camera_glove_recording_cali")]
    return rows


def _manifest_filename(row: dict) -> str:
    """Extract the video filename from a manifest row."""
    return (row.get("split_video_filename") or
            row.get("final_unmarked_video_filename") or
            Path(row.get("final_unmarked_video", "")).name)


def _manifest_segment_id(row: dict) -> str:
    """Derive a stable segment id from a manifest row."""
    fname = _manifest_filename(row)
    return Path(fname).stem if fname else row.get("session", "unknown")


def _manifest_recording_dir(row: dict, session_root: Path) -> Path:
    """Resolve the recording directory for a manifest row."""
    session = row.get("session", "")
    rec = session_root / session
    if not rec.exists() and "recordingi_" in session:
        rec = session_root / session.replace("recordingi_", "recording_")
    return rec


def _read_simple_csv(path: Path, col_map: dict[str, str] | None = None) -> list[dict]:
    """Read a generic CSV, optionally renaming columns via col_map {from: to}."""
    rows = list(csv.DictReader(open(path)))
    if col_map:
        mapped = []
        for r in rows:
            nr = dict(r)
            for src, dst in col_map.items():
                if src in nr:
                    nr[dst] = nr.pop(src)
            mapped.append(nr)
        rows = mapped
    return rows


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _apply_filters(segments: list[SegmentInfo],
                   only: str | None = None,
                   limit: int | None = None) -> list[SegmentInfo]:
    """Apply `only` substring filter and `limit`."""
    if only:
        segments = [s for s in segments
                    if only in s.id or only in str(s.recording_dir)]
    if limit is not None and limit > 0:
        segments = segments[:limit]
    return segments


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def discover(
    source: str | Path | None = None,
    *,
    # Explicit CSV mode
    manifest: str | Path | None = None,
    csv_style: str = "auto",             # "auto" | "split_segments" | "simple"
    csv_map: dict[str, str] | None = None,  # column renaming for simple csv
    col_recording: str = "recording_dir",   # column name for recording dir (simple csv)
    # Explicit recording mode
    recordings: list[str] | None = None,
    recording_dirs: list[str] | None = None,
    recording_root: str | Path | None = None,
    session_root: str | Path | None = None,  # for manifest csv
    # Filters
    only: str | None = None,
    limit: int | None = None,
    include_cali: bool = False,
) -> list[SegmentInfo]:
    """Discover pipeline input segments from a variety of sources.

    Parameters
    ----------
    source:
        A path to a CSV file, a recording directory, or a glob pattern.
        Auto-detects the type.
    manifest:
        Explicit path to a split_segments manifest CSV.
    csv_style:
        "auto" — detect manifest vs simple by inspecting columns.
        "split_segments" — treat as a split_segments CSV with session/video cols.
        "simple" — treat as a generic CSV; use csv_map + col_recording.
    csv_map:
        For simple CSV: rename source columns to expected names. E.g.
        {"path": "recording_dir", "name": "id"}.
    col_recording:
        For simple CSV: which (mapped) column contains the recording directory.
        Default "recording_dir".
    recordings:
        List of recording names (resolved under recording_root).
    recording_dirs:
        List of absolute recording directory paths.
    recording_root:
        Root directory to resolve relative recording names against.
    session_root:
        Root directory for sessions referenced in manifest CSVs.
    only:
        Substring filter on id or recording_dir.
    limit:
        Max number of segments to return.
    include_cali:
        Include calibration recordings (excluded by default).

    Returns
    -------
    list[SegmentInfo]
    """
    segments: list[SegmentInfo] = []

    # --- 1. Manifest CSV (explicit) ---
    if manifest:
        path = Path(manifest)
        rows = _read_manifest_csv(path, include_cali=include_cali)
        root = Path(session_root) if session_root else Path(os.getcwd())
        for r in rows:
            sid = _manifest_segment_id(r)
            rec_dir = _manifest_recording_dir(r, root)
            segments.append(SegmentInfo(
                id=sid, recording_dir=rec_dir, mode="manifest",
                metadata={"manifest_row": r},
            ))
        return _apply_filters(segments, only, limit)

    # --- 2. Recording dirs (explicit) ---
    if recordings or recording_dirs:
        rec_root = Path(recording_root) if recording_root else Path(os.getcwd())
        dirs = list(recording_dirs or [])
        dirs += [str(rec_root / n) if not Path(n).is_absolute() else n
                 for n in (recordings or [])]
        for d in dirs:
            p = Path(d)
            if p.is_dir():
                segments.append(SegmentInfo(id=p.name, recording_dir=p, mode="recording"))
        return _apply_filters(segments, only, limit)

    # --- 3. Source-based auto-detection ---
    if source is None:
        raise ValueError(
            "No input source provided. Use one of: source, manifest, recordings, recording_dirs."
        )

    path = Path(source)
    path_str = str(source)

    # 3a. Glob pattern
    if "*" in path_str or "?" in path_str:
        matches = sorted(_glob.glob(path_str))
        for m in matches:
            p = Path(m)
            if p.is_dir():
                rec_type = _detect_recording_type(p)
                if rec_type != "empty":
                    segments.append(SegmentInfo(
                        id=p.name, recording_dir=p, mode=f"recording_{rec_type}",
                    ))
        return _apply_filters(segments, only, limit)

    # 3b. CSV file
    if path.suffix == ".csv" and path.is_file():
        if csv_style == "auto":
            csv_style = "split_segments" if _is_manifest_csv(path) else "simple"

        if csv_style == "split_segments":
            rows = _read_manifest_csv(path, include_cali=include_cali)
            root = Path(session_root) if session_root else Path(os.getcwd())
            for r in rows:
                sid = _manifest_segment_id(r)
                rec_dir = _manifest_recording_dir(r, root)
                segments.append(SegmentInfo(
                    id=sid, recording_dir=rec_dir, mode="manifest",
                    metadata={"manifest_row": r},
                ))
        else:
            rows = _read_simple_csv(path, csv_map)
            for r in rows:
                rec_dir = Path(r.get(col_recording, ""))
                if not rec_dir.is_absolute() and recording_root:
                    rec_dir = Path(recording_root) / rec_dir
                sid = r.get("id") or rec_dir.name
                segments.append(SegmentInfo(
                    id=sid, recording_dir=rec_dir, mode="csv_list",
                    metadata={"csv_row": r},
                ))
        return _apply_filters(segments, only, limit)

    # 3c. Single recording directory
    if path.is_dir():
        rec_type = _detect_recording_type(path)
        if rec_type != "empty":
            segments.append(SegmentInfo(
                id=path.name, recording_dir=path, mode=f"recording_{rec_type}",
            ))
        return _apply_filters(segments, only, limit)

    raise ValueError(
        f"Cannot interpret source '{source}'. "
        "Expected a recording directory, CSV file, or glob pattern."
    )
