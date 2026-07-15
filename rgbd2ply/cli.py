"""Unified CLI for the rgbd2ply pipeline.

Usage:
    python -m rgbd2ply run <name>              # end-to-end on one recording
    python -m rgbd2ply batch <source>          # batch process multiple
    python -m rgbd2ply review <run_dir>        # launch review UI
    python -m rgbd2ply sweep <frames_dir>      # prompt exploration
    python -m rgbd2ply discover <source>       # list what's available
    python -m rgbd2ply config                  # print current config
"""

import argparse
import sys
from pathlib import Path

try:
    from . import __version__
    from .config import cfg
    from .discovery import discover
except ImportError:
    __version__ = "2.0.0"
    from config import cfg
    from discovery import discover


def _get_runner():
    """Lazy-import PipelineRunner (needs numpy)."""
    try:
        from .pipeline import PipelineRunner
    except ImportError:
        from pipeline import PipelineRunner
    return PipelineRunner


def _get_batch_run():
    """Lazy-import batch_run."""
    try:
        from .pipeline import batch_run
    except ImportError:
        from pipeline import batch_run
    return batch_run


def _cmd_run(args):
    """Run pipeline on a single data source."""
    PipelineRunner = _get_runner()

    segs = discover(
        args.source,
        manifest=args.manifest,
        recordings=args.recording or None,
        recording_dirs=args.recording_dir or None,
        recording_root=args.input_root,
        session_root=args.session_root,
        only=args.only,
        limit=args.limit,
        include_cali=args.include_cali,
    )

    if not segs:
        print("No segments found.", file=sys.stderr)
        sys.exit(1)

    runner = PipelineRunner(
        thr=args.thr,
        chunk=args.chunk,
        stride=args.stride,
        quality=args.quality,
        max_pair_dt_ms=args.max_pair_dt_ms,
        max_fuse_frames=args.max_fuse_frames,
        concepts_path=args.concepts,
    )

    for seg in segs:
        runner.run(seg, steps=args.steps, rerun=args.rerun)


def _cmd_batch(args):
    """Batch process multiple data sources."""
    batch_run = _get_batch_run()

    segs = discover(
        args.source,
        manifest=args.manifest,
        recordings=args.recording or None,
        recording_dirs=args.recording_dir or None,
        recording_root=args.input_root,
        session_root=args.session_root,
        only=args.only,
        limit=args.limit,
        include_cali=args.include_cali,
    )

    if not segs:
        print("No segments found.", file=sys.stderr)
        sys.exit(1)

    print(f"Batch: {len(segs)} segment(s), steps={args.steps}")
    results = batch_run(
        segs,
        steps=args.steps,
        rerun=args.rerun,
        thr=args.thr,
        chunk=args.chunk,
        stride=args.stride,
        quality=args.quality,
        max_pair_dt_ms=args.max_pair_dt_ms,
        max_fuse_frames=args.max_fuse_frames,
        concepts_path=args.concepts,
    )

    # Summary
    ok = sum(1 for r in results if "error" not in r)
    failed = len(results) - ok
    if failed:
        print(f"\n{ok}/{len(results)} succeeded, {failed} failed:")
        for r in results:
            if "error" in r:
                print(f"  ✗ {r['id']}: {r['error']}")


def _cmd_review(args):
    """Launch review UI."""
    import subprocess
    review_script = Path(__file__).resolve().parent / "review_ui.py"
    cmd = [
        cfg.envs.sam3,
        str(review_script),
        str(args.run_dir),
        "--cam", str(args.cam),
        "--port", str(args.port),
        "--host", args.host,
    ]
    print(f"Starting review UI at http://{args.host}:{args.port}/")
    subprocess.run(cmd)


def _cmd_sweep(args):
    """Run prompt sweep."""
    import subprocess
    sweep_script = Path(__file__).resolve().parent / "sweep.py"
    cmd = [
        cfg.envs.sam3,
        str(sweep_script),
        str(args.frames_dir),
        "--thr", str(args.thr),
    ]
    if args.prompts:
        cmd += ["--prompts", args.prompts]
    if args.prompts_from_registry:
        cmd.append("--prompts-from-registry")
    if args.out_dir:
        cmd += ["--out-dir", args.out_dir]
    subprocess.run(cmd)


def _cmd_discover(args):
    """List discoverable data sources."""
    segs = discover(
        args.source,
        manifest=args.manifest,
        recordings=args.recording or None,
        recording_dirs=args.recording_dir or None,
        recording_root=args.input_root,
        session_root=args.session_root,
        only=args.only,
        limit=args.limit,
        include_cali=args.include_cali,
    )

    if not segs:
        print("(none)")
        return

    print(f"{'id':<50} {'mode':<18} {'dir'}")
    print("-" * 100)
    for s in segs:
        has_cam1 = (s.recording_dir / "camera_1_image").exists() or \
                   (s.recording_dir / "camera_1_rgb_depth.bag").exists()
        has_cam3 = (s.recording_dir / "camera_3_image").exists() or \
                   (s.recording_dir / "camera_3_rgb_depth.bag").exists()
        cams = []
        if has_cam1: cams.append("cam1")
        if has_cam3: cams.append("cam3")
        print(f"{s.id:<50} {s.mode:<18} {str(s.recording_dir):<60} ({','.join(cams)})")
    print(f"\n{len(segs)} segment(s)")


def _cmd_config(args):
    """Print current configuration."""
    import json
    data = {
        "paths": dict(cfg.paths._data),
        "envs": dict(cfg.envs._data),
        "defaults": dict(cfg.defaults._data),
    }
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        for section, items in data.items():
            print(f"[{section}]")
            for k, v in items.items():
                print(f"  {k:25s} = {v}")


def main():
    ap = argparse.ArgumentParser(
        description="rgbd2ply — multi-camera RGBD → labelled pointcloud pipeline"
    )
    ap.add_argument("--version", action="version", version=f"rgbd2ply {__version__}")
    sub = ap.add_subparsers(dest="command", title="commands")

    # ---- run ----
    p_run = sub.add_parser("run", help="Run pipeline on one data source")
    p_run.add_argument("source", nargs="?", default=None,
                       help="Recording dir, manifest CSV, or glob pattern")
    p_run.add_argument("--steps", default="all",
                       help="Comma-separated: prepare,concepts,auto,fuse (default: all)")
    p_run.add_argument("--rerun", action="store_true", help="Force re-execution")
    _add_common_io_args(p_run)
    _add_pipeline_tuning_args(p_run)
    p_run.set_defaults(func=_cmd_run)

    # ---- batch ----
    p_batch = sub.add_parser("batch", help="Batch process multiple data sources")
    p_batch.add_argument("source", nargs="?", default=None,
                         help="Glob pattern, manifest CSV, or recording dir")
    p_batch.add_argument("--steps", default="all",
                         help="Comma-separated: prepare,concepts,auto,fuse (default: all)")
    p_batch.add_argument("--rerun", action="store_true", help="Force re-execution")
    _add_common_io_args(p_batch)
    _add_pipeline_tuning_args(p_batch)
    p_batch.set_defaults(func=_cmd_batch)

    # ---- review ----
    p_rev = sub.add_parser("review", help="Launch interactive review UI")
    p_rev.add_argument("run_dir", help="Run directory path")
    p_rev.add_argument("--cam", type=int, choices=[1, 3], default=3,
                       help="Default camera (1 or 3)")
    p_rev.add_argument("--port", type=int, default=8899)
    p_rev.add_argument("--host", default="0.0.0.0")
    p_rev.set_defaults(func=_cmd_review)

    # ---- sweep ----
    p_sw = sub.add_parser("sweep", help="Sweep text prompts against a frames_dir")
    p_sw.add_argument("frames_dir", help="Prepared frames directory")
    p_sw.add_argument("--thr", type=float, default=0.4, help="Detection threshold")
    p_sw.add_argument("--prompts", default=None, help="Comma-separated prompts")
    p_sw.add_argument("--prompts-from-registry", action="store_true")
    p_sw.add_argument("--out-dir", default=None)
    p_sw.set_defaults(func=_cmd_sweep)

    # ---- discover ----
    p_disc = sub.add_parser("discover", help="List available data sources")
    p_disc.add_argument("source", nargs="?", default=None,
                        help="Glob pattern, manifest CSV, or recording dir")
    _add_common_io_args(p_disc)
    p_disc.set_defaults(func=_cmd_discover)

    # ---- config ----
    p_cfg = sub.add_parser("config", help="Print current configuration")
    p_cfg.add_argument("--json", action="store_true", help="JSON format output")
    p_cfg.set_defaults(func=_cmd_config)

    args = ap.parse_args()
    if args.command is None:
        ap.print_help()
        sys.exit(1)

    args.func(args)


def _add_common_io_args(parser):
    """Add common I/O arguments shared across run/batch/discover."""
    grp = parser.add_argument_group("I/O")
    grp.add_argument("--manifest", default=None,
                     help="Path to split_segments manifest CSV")
    grp.add_argument("--input-root", default=None,
                     help="Root for recording folder names")
    grp.add_argument("--recording", action="append", default=[],
                     help="Recording name under --input-root (repeatable)")
    grp.add_argument("--recording-dir", action="append", default=[],
                     help="Absolute recording dir path (repeatable)")
    grp.add_argument("--session-root", default=None,
                     help="Root for sessions in manifest (default: config data_root)")
    grp.add_argument("--out-root", default=None,
                     help="Output root (default: config runs_root)")
    grp.add_argument("--only", default=None,
                     help="Substring filter on segment id or dir")
    grp.add_argument("--limit", type=int, default=None,
                     help="Limit number of segments")
    grp.add_argument("--include-cali", action="store_true",
                     help="Include calibration recordings")


def _add_pipeline_tuning_args(parser):
    """Add pipeline parameter tuning arguments."""
    grp = parser.add_argument_group("Pipeline parameters")
    grp.add_argument("--thr", type=float, default=None,
                     help=f"SAM3 confidence threshold (default: {cfg.defaults.thr})")
    grp.add_argument("--chunk", type=int, default=None,
                     help=f"SAM3 chunk size (default: {cfg.defaults.chunk})")
    grp.add_argument("--stride", type=int, default=None,
                     help=f"Frame subsampling stride (default: {cfg.defaults.stride})")
    grp.add_argument("--quality", type=int, default=None,
                     help=f"JPEG quality (default: {cfg.defaults.quality})")
    grp.add_argument("--max-pair-dt-ms", type=float, default=None,
                     help=f"Max cam1-cam3 timestamp delta ms (default: {cfg.defaults.max_pair_dt_ms})")
    grp.add_argument("--max-fuse-frames", type=int, default=None,
                     help="Max pointclouds per camera per colour mode")
    grp.add_argument("--concepts", default=None,
                     help="Path to concepts JSON (default: rgbd2ply/auto_concepts.json)")
