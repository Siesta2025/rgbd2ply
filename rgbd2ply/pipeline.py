"""Pipeline orchestrator for rgbd2ply.

Encapsulates the step dependency graph and execution logic. Each step is
idempotent — skips if outputs already exist (unless rerun=True).

Usage:
    from rgbd2ply.pipeline import PipelineRunner
    from rgbd2ply.discovery import discover

    segs = discover("/data/recordings/camera_glove_*")
    runner = PipelineRunner()
    for seg in segs:
        runner.run(seg, steps="all")
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    from .config import cfg
    from .discovery import SegmentInfo
    from .prepare import prepare
    from .concepts import build_concepts, save_concepts
except ImportError:
    from config import cfg
    from discovery import SegmentInfo
    from prepare import prepare
    from concepts import build_concepts, save_concepts

# Make rgbd2ply dir importable for subprocess scripts
_RGBD2PLY = Path(__file__).resolve().parent


class PipelineRunner:
    """Orchestrate the rgbd2ply pipeline steps with dependency awareness."""

    def __init__(
        self,
        sam3_python: str | None = None,
        wilor_python: str | None = None,
        thr: float | None = None,
        chunk: int | None = None,
        stride: int | None = None,
        quality: int | None = None,
        max_pair_dt_ms: float | None = None,
        max_fuse_frames: int | None = None,
        concepts_path: str | Path | None = None,
        sam3_ckpt: str | Path | None = None,
    ):
        self.sam3_py = sam3_python or cfg.envs.sam3
        self.wilor_py = wilor_python or cfg.envs.wilor
        self.thr = thr if thr is not None else cfg.defaults.thr
        self.chunk = chunk if chunk is not None else cfg.defaults.chunk
        self.stride = stride if stride is not None else cfg.defaults.stride
        self.quality = quality if quality is not None else cfg.defaults.quality
        self.max_pair_dt_ms = max_pair_dt_ms if max_pair_dt_ms is not None else cfg.defaults.max_pair_dt_ms
        self.max_fuse_frames = max_fuse_frames
        self.concepts_path = Path(concepts_path) if concepts_path else (_RGBD2PLY / "auto_concepts.json")
        self.sam3_ckpt = str(sam3_ckpt or cfg.paths.sam3_checkpoint)

        self._run_root = Path(cfg.paths.runs_root)

    # ------------------------------------------------------------------
    # Step: concepts
    # ------------------------------------------------------------------
    def step_concepts(self, rerun: bool = False) -> Path:
        """Generate auto_concepts.json from object_registry.json."""
        out = self.concepts_path
        if out.exists() and not rerun:
            print(f"  [concepts] skip — {out} exists")
            return out

        t0 = time.time()
        concepts = build_concepts()
        save_concepts(concepts, out)
        elapsed = time.time() - t0
        print(f"  [concepts] {len(concepts)} concepts → {out}  ({elapsed:.1f}s)")
        return out

    # ------------------------------------------------------------------
    # Step: prepare
    # ------------------------------------------------------------------
    def step_prepare(self, seg: SegmentInfo, rerun: bool = False) -> Path:
        """Extract frames for one segment. Returns run_dir."""
        run_dir = self._run_root / seg.id
        ready = ((run_dir / "cam1_frames" / "meta.json").exists() or
                 (run_dir / "cam3_frames" / "meta.json").exists())
        if ready and not rerun:
            print(f"  [prepare] skip — {run_dir} exists")
            return run_dir

        t0 = time.time()
        out = prepare(seg, out_root=self._run_root, quality=self.quality, stride=self.stride)
        elapsed = time.time() - t0
        c1 = out.get("cam1", {}).get("n_frames", 0)
        c3 = out.get("cam3", {}).get("n_frames", 0)
        print(f"  [prepare] {seg.id}  cam1={c1}  cam3={c3}  →  {run_dir}  ({elapsed:.1f}s)")
        return run_dir

    # ------------------------------------------------------------------
    # Step: auto label (SAM3 — cross-env via subprocess)
    # ------------------------------------------------------------------
    def step_auto(self, run_dir: Path, cam: int, rerun: bool = False) -> Path:
        """Run SAM3 multi-concept labelling for camera `cam`."""
        frames_dir = run_dir / f"cam{cam}_frames"
        out_npz = run_dir / f"cam{cam}_labels_auto.npz"
        overlay_dir = run_dir / "overlays"
        overlay_mp4 = overlay_dir / f"cam{cam}_auto_overlay.mp4"

        if not (frames_dir / "meta.json").exists():
            print(f"  [auto:cam{cam}] skip — no frames dir {frames_dir}")
            return out_npz

        if out_npz.exists() and not rerun:
            print(f"  [auto:cam{cam}] skip — {out_npz} exists")
            return out_npz

        overlay_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        cmd = [
            self.sam3_py,
            str(_RGBD2PLY / "multi_concept_masks.py"),
            str(frames_dir),
            "--out", str(out_npz),
            "--overlay", str(overlay_mp4),
            "--concepts", str(self.concepts_path),
            "--thr", str(self.thr),
            "--chunk", str(self.chunk),
            "--ckpt", self.sam3_ckpt,
        ]
        print(f"  [auto:cam{cam}] running SAM3...")
        subprocess.run(cmd, check=True)
        elapsed = time.time() - t0

        # Load result for summary
        data = dict(np.load(out_npz))
        labels = data.get("labels")
        if labels is not None:
            counts = {int(l): int((labels == l).sum()) for l in np.unique(labels) if int(l) > 0}
            print(f"  [auto:cam{cam}] done in {elapsed:.1f}s  pixel_counts={counts}")
        else:
            print(f"  [auto:cam{cam}] done in {elapsed:.1f}s")

        return out_npz

    # ------------------------------------------------------------------
    # Step: fuse
    # ------------------------------------------------------------------
    def step_fuse(self, run_dir: Path, recording_dir: Path, rerun: bool = False) -> tuple[Path, Path]:
        """Fuse cam1+cam3 into pointclouds (two colour modes)."""
        # Check prerequisites
        label_modes = []
        for cam in (1, 3):
            if (run_dir / f"cam{cam}_labels_final.npz").exists():
                label_modes.append("final")
                break
        for cam in (1, 3):
            if (run_dir / f"cam{cam}_labels_auto.npz").exists():
                label_modes.append("auto")
                break
        if not label_modes:
            print(f"  [fuse] skip — no label npz found in {run_dir}")
            return (run_dir / "pointclouds" / "masked_rgb", run_dir / "pointclouds" / "mask_only")

        # Copy auto labels as final if no corrections exist
        for cam in (1, 3):
            final = run_dir / f"cam{cam}_labels_final.npz"
            auto = run_dir / f"cam{cam}_labels_auto.npz"
            if not final.exists() and auto.exists():
                auto_data = dict(np.load(auto))
                np.savez_compressed(final, **auto_data)

        # Check both exist now
        for cam in (1, 3):
            if not (run_dir / f"cam{cam}_labels_final.npz").exists():
                print(f"  [fuse] skip — missing cam{cam}_labels_final.npz")
                return (run_dir / "pointclouds" / "masked_rgb", run_dir / "pointclouds" / "mask_only")

        real_dir = run_dir / "pointclouds" / "masked_rgb"
        mask_dir = run_dir / "pointclouds" / "mask_only"

        if real_dir.exists() and mask_dir.exists() and not rerun:
            n_real = len(list(real_dir.glob("*.ply")))
            n_mask = len(list(mask_dir.glob("*.ply")))
            print(f"  [fuse] skip — {n_real}+{n_mask} clouds exist")
            return (real_dir, mask_dir)

        # Build and run fusion commands via subprocess (wilor env)
        t0 = time.time()
        for out_dir, color_mode, labeled_only in [
            (real_dir, "masked_rgb", False),
            (mask_dir, "label", True),
        ]:
            cmd = [
                self.wilor_py,
                str(_RGBD2PLY / "fusion.py"),
                str(run_dir),
                str(recording_dir),
                "--out", str(out_dir),
                "--color-mode", color_mode,
                "--max-pair-dt-ms", str(self.max_pair_dt_ms),
            ]
            if labeled_only:
                cmd.append("--labeled-only")
            if self.max_fuse_frames:
                cmd += ["--max-frames", str(self.max_fuse_frames)]
            print(f"  [fuse:{color_mode}] fusing...")
            subprocess.run(cmd, check=True)

        elapsed = time.time() - t0
        n_real = len(list(real_dir.glob("*.ply")))
        n_mask = len(list(mask_dir.glob("*.ply")))
        print(f"  [fuse] done in {elapsed:.1f}s  →  {n_real} + {n_mask} clouds")
        return (real_dir, mask_dir)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------
    def run(
        self,
        seg: SegmentInfo,
        steps: str = "all",
        rerun: bool = False,
    ) -> dict:
        """Execute pipeline steps for one segment.

        Parameters
        ----------
        seg: SegmentInfo from discovery.
        steps: "all" or comma-separated: "prepare,concepts,auto,fuse".
        rerun: Force re-execution even if outputs exist.

        Returns
        -------
        dict with keys: id, run_dir, concepts, cam1_auto, cam3_auto, fuse_real, fuse_mask.
        """
        step_set = {s.strip() for s in steps.split(",")} if steps != "all" else {"all"}
        run_all = "all" in step_set
        result = {"id": seg.id}

        start = time.time()

        # ---- concepts (global, not per-segment) ----
        if run_all or "concepts" in step_set:
            result["concepts"] = self.step_concepts(rerun=rerun)

        # ---- prepare ----
        if run_all or "prepare" in step_set:
            result["run_dir"] = self.step_prepare(seg, rerun=rerun)
        else:
            result["run_dir"] = self._run_root / seg.id

        run_dir = result["run_dir"]

        # ---- auto label ----
        if run_all or "auto" in step_set:
            for cam in (1, 3):
                result[f"cam{cam}_auto"] = self.step_auto(run_dir, cam, rerun=rerun)

        # ---- fuse ----
        if run_all or "fuse" in step_set:
            rec_dir = seg.recording_dir
            result["fuse_real"], result["fuse_mask"] = self.step_fuse(run_dir, rec_dir, rerun=rerun)

        elapsed = time.time() - start
        step_list = steps if not run_all else "prepare,concepts,auto,fuse"
        print(f"  ✔ {seg.id}  [{step_list}]  done in {elapsed:.1f}s")
        return result


def batch_run(
    segments: list[SegmentInfo],
    steps: str = "all",
    rerun: bool = False,
    **runner_kwargs,
) -> list[dict]:
    """Run the pipeline over a batch of segments.

    Returns a list of result dicts, one per segment.
    """
    runner = PipelineRunner(**runner_kwargs)
    results = []
    for i, seg in enumerate(segments):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(segments)}] {seg.id}")
        print(f"{'='*60}")
        try:
            r = runner.run(seg, steps=steps, rerun=rerun)
            results.append(r)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            results.append({"id": seg.id, "error": str(e)})
    ok = sum(1 for r in results if "error" not in r)
    print(f"\n{'='*60}")
    print(f"Done: {ok}/{len(results)} succeeded")
    return results
