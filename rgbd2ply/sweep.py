"""Prompt sweep utility: test many text prompts against a frames_dir.

Runs SAM3 once per prompt candidate and reports detection consistency. Helps
find the best prompt for each object before running the full pipeline.

Usage:
    python sweep.py runs/xyz/cam3_frames --thr 0.4
    python sweep.py runs/xyz/cam3_frames --prompts "metal pot,funnel,plastic box"
    python sweep.py runs/xyz/cam3_frames --prompts-from-registry
"""

import argparse
import glob as _glob
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .config import cfg
    from .concepts import load_registry
    from .multi_concept_masks import (
        build_sam3_video_predictor,
        _chunk_dirs,
        _run_concept,
        DEFAULT_CKPT,
    )
except ImportError:
    from config import cfg
    from concepts import load_registry
    from multi_concept_masks import (
        build_sam3_video_predictor,
        _chunk_dirs,
        _run_concept,
        DEFAULT_CKPT,
    )


def run_sweep(
    frames_dir: str | Path,
    prompts: list[str],
    thr: float = 0.4,
    chunk: int = 500,
    ckpt: str | None = None,
    out_dir: str | Path | None = None,
    save_sheets: bool = True,
) -> list[dict]:
    """Run SAM3 detection for each prompt and return per-prompt stats.

    Returns a list of dicts: {prompt, n_frames_detected, avg_area_pct}.
    Saves contact-sheet images to `out_dir` for visual comparison.
    """
    frames_dir = Path(frames_dir)
    meta = json.load(open(frames_dir / "meta.json"))
    W, H, T = meta["width"], meta["height"], meta["n_frames"]
    files = sorted(
        _glob.glob(f"{frames_dir}/[0-9]*.jpg"),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0]),
    )

    ckpt = ckpt or DEFAULT_CKPT

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pred = build_sam3_video_predictor(
        checkpoint_path=ckpt, async_loading_frames=True, gpus_to_use=[0]
    )

    tag = os.path.basename(os.path.normpath(frames_dir))
    chunks = _chunk_dirs(files, T, tag, chunk)

    # Sample 8 frames for contact sheet
    samp = [int(i * (T - 1) / 7) for i in range(8)] if T > 8 else list(range(T))

    results = []
    for prompt in prompts:
        labels = np.zeros((T, H, W), np.int32)
        _run_concept(pred, chunks, labels, prompt, 9, thr, 1)

        det = np.array([(labels[t] == 9).sum() for t in range(T)])
        nf = int((det > 0).sum())
        areapct = 100.0 * det[det > 0].mean() / (H * W) if nf else 0.0

        results.append({
            "prompt": prompt,
            "n_frames_detected": nf,
            "avg_area_pct": round(areapct, 2),
            "coverage": f"{nf}/{T}",
        })

        if save_sheets and out_dir:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            tiles = []
            for t in samp:
                im = cv2.imread(files[t])
                m = labels[t] == 9
                if m.any():
                    im[m] = (0.5 * np.array([0, 200, 40]) + 0.5 * im[m]).astype(np.uint8)
                tiles.append(cv2.resize(im, (300, int(300 * H / W))))
            row = np.hstack(tiles)
            cv2.imwrite(str(out_dir / f"{prompt.replace(' ', '_')}.png"), row)

    for _, cd in chunks:
        shutil.rmtree(cd, ignore_errors=True)

    return results


def prompts_from_registry() -> list[str]:
    """Collect all prompt strings from the label registry."""
    reg = load_registry()
    prompts = []
    for item in reg.get("labels", []):
        if not item.get("auto", True):
            continue
        for p in (item.get("prompts") or []):
            if p not in prompts:
                prompts.append(p)
    return prompts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Sweep SAM3 text prompts against a frames_dir."
    )
    ap.add_argument("frames_dir", help="Prepared frames directory (cam{N}_frames)")
    ap.add_argument("--thr", type=float, default=0.4, help="SAM3 detection threshold")
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", default=None,
                    help="Directory for contact-sheet images (default: frames_dir/../sweep)")
    ap.add_argument("--prompts", default=None,
                    help="Comma-separated prompt list (default: built-in candidates)")
    ap.add_argument("--prompts-from-registry", action="store_true",
                    help="Use all prompts from object_registry.json")
    args = ap.parse_args()

    if args.prompts_from_registry:
        prompts = prompts_from_registry()
        print(f"Loaded {len(prompts)} prompts from registry")
    elif args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    else:
        # Reasonable built-in set for common kitchen objects
        prompts = [
            "metal pot", "cooking pot", "silver pot", "steel bucket",
            "funnel", "metal funnel", "stainless steel funnel",
            "plastic box", "clear plastic container", "storage box",
            "bucket", "plastic bucket", "white bucket", "water jug",
            "transparent plastic cup", "clear plastic cup", "cup",
        ]

    out_dir = args.out_dir or str(Path(args.frames_dir).parent / "sweep")

    results = run_sweep(
        args.frames_dir,
        prompts=prompts,
        thr=args.thr,
        chunk=args.chunk,
        ckpt=args.ckpt,
        out_dir=out_dir,
        save_sheets=True,
    )

    # Print summary table
    print(f"\n{'prompt':<30} {'frames':>7} {'area%':>8}")
    print("-" * 48)
    for r in results:
        print(f"{r['prompt']:<30} {r['coverage']:>7} {r['avg_area_pct']:>7.2f}%")
    print(f"\nContact sheets saved to {out_dir}/")
