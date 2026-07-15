"""Box/point-seeded SAM3 tracking for the multi-label rgbd2ply scheme (sam3 env).

Use this when text-prompt SAM3 misses a hard object. You provide one or more
seeded objects, each with a stable integer label, and SAM3 tracks them across the
whole frames_dir.

Object schema:
    {"label": 1, "box": [x1,y1,x2,y2]}
    {"label": 6, "points": [[x,y,1], [x,y,0]]}   # p=1 positive, p=0 negative

Examples:
    USE_PERFLIB=0 ~/miniconda3/envs/sam3/bin/python seed_masks.py \
      <frames_dir> --box 3 1000 520 1300 900 --out funnel.npz --overlay-dir overlays

    USE_PERFLIB=0 ~/miniconda3/envs/sam3/bin/python seed_masks.py \
      <frames_dir> --objects-json seeds_213442_cam3.json --out labels_seeded.npz
"""
import argparse
import json
import os

import cv2
import numpy as np

try:
    from .camera_utils import sam3_masking_click as smc
except ImportError:
    from camera_utils import sam3_masking_click as smc
from labelspec import bgr, label_name

DEFAULT_CKPT = smc.DEFAULT_CKPT


def load_objects(path):
    raw = json.load(open(path))
    if isinstance(raw, dict):
        raw = raw.get("objects", [])
    objects = []
    for obj in raw:
        if obj.get("enabled", True) is False:
            continue
        label = int(obj["label"])
        clean = {"label": label}
        if obj.get("box") is not None:
            clean["box"] = [float(x) for x in obj["box"]]
        if obj.get("points"):
            clean["points"] = [[float(p[0]), float(p[1]), int(p[2])] for p in obj["points"]]
        if "box" not in clean and "points" not in clean:
            raise ValueError("seed object needs box or points: %r" % obj)
        objects.append(clean)
    if not objects:
        raise ValueError("no seed objects loaded from %s" % path)
    return objects


def _paint_order(obj_label):
    labels = sorted({int(x) for x in obj_label.values() if int(x) != 1})
    if any(int(x) == 1 for x in obj_label.values()):
        labels.append(1)
    return labels


def track_objects(frames_dir, objects, out_npz, ckpt=DEFAULT_CKPT, overlay_dir=None, predictor=None):
    """objects: list of {'label': int, 'box':[..]?, 'points':[[x,y,p],..]?}.
    Reuse a pre-built predictor (from smc._build) to avoid reloading the model."""
    import torch

    meta = json.load(open(os.path.join(frames_dir, "meta.json")))
    W, H, T = meta["width"], meta["height"], meta["n_frames"]

    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if predictor is None:
        predictor = smc._build(ckpt)
    state = predictor.init_state(video_path=frames_dir)
    predictor.clear_all_points_in_video(state)

    obj_label = {}
    for oid, obj in enumerate(objects, start=1):
        smc._add_obj(predictor, state, oid, obj, W, H)
        obj_label[oid] = int(obj.get("label", 1))

    labels = np.zeros((T, H, W), np.int32)
    order = _paint_order(obj_label)
    for fidx, obj_ids, low, video_res_masks, scores in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=T, reverse=False, propagate_preflight=True):
        fm = {int(o): smc._mask_np(video_res_masks[i] > 0.0) for i, o in enumerate(obj_ids)}
        for target in order:
            for oid, m in fm.items():
                if obj_label.get(oid) == target:
                    labels[fidx][m] = target

    np.savez_compressed(out_npz, labels=labels,
                        frame_indices=np.array(meta["frame_indices"], np.int64),
                        timestamps=np.array(meta["timestamps"], np.int64))

    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        active = [int(x) for x in np.unique(labels) if int(x) > 0]
        for t in range(T):
            over = cv2.imread(os.path.join(frames_dir, "%05d.jpg" % t))
            if over is None:
                continue
            for lab in active:
                m = labels[t] == lab
                if m.any():
                    over[m] = (0.5 * np.array(bgr(lab)) + 0.5 * over[m]).astype(np.uint8)
            cv2.imwrite(os.path.join(overlay_dir, "overlay_%06d.jpg" % meta["frame_indices"][t]), over)
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Seed objects and track with SAM3 into a multi-label npz.")
    ap.add_argument("frames_dir")
    ap.add_argument("--objects-json", default=None)
    ap.add_argument("--box", nargs=5, action="append", default=[],
                    metavar=("LABEL", "X1", "Y1", "X2", "Y2"),
                    help="seed box in pixels: label x1 y1 x2 y2; repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay-dir", default=None)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    a = ap.parse_args()

    objs = []
    if a.objects_json:
        objs.extend(load_objects(a.objects_json))
    objs.extend({"label": int(b[0]), "box": [float(b[1]), float(b[2]), float(b[3]), float(b[4])]}
                for b in a.box)
    if not objs:
        raise SystemExit("Provide --objects-json or at least one --box.")
    lab = track_objects(a.frames_dir, objs, a.out, ckpt=a.ckpt, overlay_dir=a.overlay_dir)
    print("saved %s  labels=%s" % (a.out, lab.shape))
    print("pixel counts:", {label_name(i): int((lab == i).sum()) for i in np.unique(lab)})
