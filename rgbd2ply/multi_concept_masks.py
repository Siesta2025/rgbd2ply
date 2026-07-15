"""Multi-concept SAM3 labelmap (sam3 env).

Runs the SAM3 open-vocab concept detector once per concept over a frames_dir and
merges every concept into one per-frame label map:

    0 = background, 1 = hand, 2..N = object labels

Same engine as hand_pipeline/sam3_hand_masks.py (text prompt -> whole-video
auto-detect + track, including objects entering mid-video). The model is built
once and reused for all concepts so the checkpoint loads a single time.

Output .npz schema consumed by fusion_multilabel.py:
    labels        int32 [T,H,W]
    frame_indices int64 [T]
    timestamps    int64 [T]

Objects are painted first and hand should be the last concept, so a hand covering
an object wins overlap.

    USE_PERFLIB=0 ~/miniconda3/envs/sam3/bin/python multi_concept_masks.py \
        <frames_dir> --out <labels.npz> --overlay <qc.mp4> \
        --concepts /home/descfly/chiwan/rgbd2ply/concepts.default.json
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# SAM3 repo path: prefer env var, fall back to config, then hardcoded.
_SAM3_REPO = os.environ.get("SAM3_REPO", "")
if not _SAM3_REPO:
    try:
        from config import cfg
        _SAM3_REPO = cfg.paths.sam3_repo
    except Exception:
        _SAM3_REPO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps", "sam3")
sys.path.insert(0, _SAM3_REPO)

import cv2
import numpy as np
import torch
from sam3.model_builder import build_sam3_video_predictor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labelspec import bgr, label_name

DEFAULT_CKPT = os.environ.get("SAM3_CKPT", "")
if not DEFAULT_CKPT:
    try:
        from config import cfg
        DEFAULT_CKPT = str(cfg.paths.sam3_checkpoint)
    except Exception:
        DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deps", "sam3", "ckpt", "sam3.pt")
_CFG_TMP = ""
try:
    from config import cfg
    _CFG_TMP = str(cfg.paths.tmp)
except Exception:
    pass
TMPB = os.environ.get("SAM3_TMP", _CFG_TMP or os.path.join(
    os.environ.get("TMPDIR", "/tmp"), "rgbd2ply_mcmask"))


def load_concepts(path=None):
    """Load concepts JSON; accepts {"concepts": [...]} or a raw list.

    Each concept entry:
      {"label": 4, "prompt": "transparent plastic box", "max_instances": 1}

    If path is None, falls back to <package_dir>/auto_concepts.json.
    """
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_concepts.json")
    raw = json.load(open(path))
    if isinstance(raw, dict):
        raw = raw.get("concepts", [])
    concepts = []
    for item in raw:
        if isinstance(item, dict):
            if item.get("enabled", True) is False:
                continue
            label = int(item["label"])
            prompt = str(item["prompt"])
            max_instances = int(item.get("max_instances", item.get("maxh", 1)))
            name = item.get("name", label_name(label))
        else:
            label, prompt, max_instances = item[:3]
            label, prompt, max_instances = int(label), str(prompt), int(max_instances)
            name = label_name(label)
        concepts.append({"label": label, "prompt": prompt,
                         "max_instances": max_instances, "name": name})
    if not concepts:
        raise ValueError("concept list is empty")
    return concepts


def _masks_np(m):
    m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
    return m.astype(bool).reshape(-1, m.shape[-2], m.shape[-1]) if m.ndim >= 2 else m


def _chunk_dirs(files, T, tag, chunk):
    """Make clean image-only symlink dirs SAM3 wants; return [(c0, chunk_dir), ...]."""
    dirs = []
    for ci, c0 in enumerate(range(0, T, chunk)):
        c1 = min(T, c0 + chunk)
        cd = f"{TMPB}/{tag}_{ci}"
        if os.path.isdir(cd):
            shutil.rmtree(cd)
        os.makedirs(cd)
        for j in range(c0, c1):
            os.symlink(files[j], f"{cd}/{j - c0:05d}.jpg")
        dirs.append((c0, cd))
    return dirs


def _run_concept(pred, chunks, labels, prompt, label, thr, max_instances):
    """Detect+track one concept over all chunks; paint its masks as `label`."""
    T, H, W = labels.shape
    ndet = 0
    for c0, cd in chunks:
        sid = pred.handle_request(dict(type="start_session", resource_path=cd,
                                       offload_video_to_cpu=True, offload_state_to_cpu=True))["session_id"]
        pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=0,
                                 text=prompt, output_prob_thresh=thr))
        for r in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid,
                                                 propagation_direction="forward", output_prob_thresh=thr)):
            gi = c0 + r["frame_index"]
            out = r["outputs"]
            pr = np.asarray(out["out_probs"]).reshape(-1)
            if not len(pr):
                continue
            masks = _masks_np(out["out_binary_masks"])
            order = np.argsort(-pr)
            if max_instances > 0:
                order = order[:max_instances]
            hit = False
            for k in order:
                m = masks[k]
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                labels[gi][m] = label
                hit = True
            ndet += int(hit)
        pred.handle_request(dict(type="close_session", session_id=sid))
        torch.cuda.empty_cache()
    print(f"    label {label:>2} '{prompt}': {ndet}/{T} frames detected", flush=True)


def build_labelmap(frames_dir, concepts=None, thr=0.5, ckpt=DEFAULT_CKPT, chunk=500):
    meta = json.load(open(os.path.join(frames_dir, "meta.json")))
    W, H, T = meta["width"], meta["height"], meta["n_frames"]
    files = sorted(glob.glob(f"{frames_dir}/[0-9]*.jpg"),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    assert len(files) == T, f"meta n_frames={T} but found {len(files)} jpgs"

    if concepts is None:
        concepts = load_concepts()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    pred = build_sam3_video_predictor(checkpoint_path=ckpt, async_loading_frames=True, gpus_to_use=[0])

    tag = os.path.basename(os.path.normpath(frames_dir))
    chunks = _chunk_dirs(files, T, tag, chunk)
    labels = np.zeros((T, H, W), np.int32)
    for c in concepts:
        _run_concept(pred, chunks, labels, c["prompt"], int(c["label"]), thr, int(c["max_instances"]))

    for _, cd in chunks:
        shutil.rmtree(cd, ignore_errors=True)
    return labels, meta


def _to_h264(path):
    tmp = path[:-4] + "_h264.mp4"
    if subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path,
                       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", tmp]).returncode == 0:
        os.replace(tmp, path)


def write_overlay(frames_dir, labels, meta, path):
    W, H, T = meta["width"], meta["height"], meta["n_frames"]
    ow, oh = 960, int(round(960 * H / W))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (ow, oh))
    active = [int(x) for x in np.unique(labels) if int(x) > 0]
    for t in range(T):
        im = cv2.imread(os.path.join(frames_dir, "%05d.jpg" % t))
        if im is None:
            continue
        for label in active:
            m = labels[t] == label
            if m.any():
                im[m] = (0.45 * np.array(bgr(label)) + 0.55 * im[m]).astype(np.uint8)
        vw.write(cv2.resize(im, (ow, oh)))
    vw.release()
    _to_h264(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Multi-concept SAM3 labelmap for fusion.")
    ap.add_argument("frames_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay", default=None, help="optional QC mp4, each label a colour")
    ap.add_argument("--concepts", default=None, help="JSON concept list; default is built in")
    ap.add_argument("--thr", type=float, default=float(os.environ.get("THR", "0.5")))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--chunk", type=int, default=int(os.environ.get("CHUNK", "500")))
    a = ap.parse_args()

    concepts = load_concepts(a.concepts)
    print("concepts:", [(c["label"], c["prompt"], c["max_instances"]) for c in concepts], flush=True)
    labels, meta = build_labelmap(a.frames_dir, concepts=concepts, thr=a.thr, ckpt=a.ckpt, chunk=a.chunk)
    np.savez_compressed(a.out, labels=labels,
                        frame_indices=np.array(meta["frame_indices"], np.int64),
                        timestamps=np.array(meta["timestamps"], np.int64))
    counts = {int(l): int((labels == l).sum()) for l in np.unique(labels)}
    print(f"saved {a.out}  labels={labels.shape}  pixel counts={counts}", flush=True)
    if a.overlay:
        write_overlay(a.frames_dir, labels, meta, a.overlay)
        print(f"saved overlay {a.overlay}", flush=True)
