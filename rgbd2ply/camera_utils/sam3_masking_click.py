"""Step 2 (sam3 env), CLICK version: base SAM3 video tracker over a frames/ folder.

Uses the BASE SAM3 video model (needs sam3.pt), which accepts clicks/boxes AND
interactive refinement (positive/negative points). You pass a list of OBJECTS,
each tagged with a label (1=hand, 2=object) and defined by a box and/or a set of
points (each point is [x, y, p] with p=1 positive / p=0 negative). Output .npz
matches the other maskers so pipeline.py fusion is unchanged.

An "object" (used by the refinement UI):
    {"label": 1|2, "box": [x1,y1,x2,y2] (optional), "points": [[x,y,1],[x,y,0],...] (optional)}
"""
import os
import json
import argparse
import numpy as np
import cv2
import torch

_CKPT = os.environ.get("SAM3_CKPT", "")
if not _CKPT:
    try:
        from rgbd2ply.config import cfg
        _CKPT = str(cfg.paths.sam3_checkpoint)
    except Exception:
        _ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _CKPT = os.path.join(_ROOT, "deps", "sam3", "ckpt", "sam3.pt")
DEFAULT_CKPT = _CKPT


def _mask_np(m):
    m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
    return np.squeeze(m).astype(bool)


def _build(ckpt):
    from sam3.model_builder import build_sam3_video_model
    model = build_sam3_video_model(checkpoint_path=ckpt)
    predictor = model.tracker
    predictor.backbone = model.detector.backbone
    for m in model.modules():                     # FA3 not installed -> SDPA
        if hasattr(m, "use_fa3"):
            m.use_fa3 = False
    return predictor


def _add_obj(predictor, state, obj_id, obj, W, H):
    """Register one object from its box and/or its (positive/negative) points."""
    kwargs = {}
    if obj.get("box") is not None:
        x1, y1, x2, y2 = obj["box"]
        kwargs["box"] = np.array([[x1 / W, y1 / H, x2 / W, y2 / H]], dtype=np.float32)
    pts = obj.get("points") or []
    if pts:
        kwargs["points"] = torch.tensor([[p[0] / W, p[1] / H] for p in pts], dtype=torch.float32)
        kwargs["labels"] = torch.tensor([int(p[2]) for p in pts], dtype=torch.int32)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=obj_id, **kwargs)


def track_labelmap(frames_dir, objects, out_npz, ckpt=DEFAULT_CKPT, overlay_dir=None, predictor=None):
    """objects: list of {'label':1|2, 'box':[..]?, 'points':[[x,y,p],..]?}.
    Pass a pre-built ``predictor`` (from ``_build``) to avoid reloading the model."""
    meta = json.load(open(os.path.join(frames_dir, "meta.json")))
    W, H, T = meta["width"], meta["height"], meta["n_frames"]

    if torch.cuda.is_available():
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    if predictor is None:
        predictor = _build(ckpt)
    state = predictor.init_state(video_path=frames_dir)
    predictor.clear_all_points_in_video(state)

    obj_label = {}
    for oid, obj in enumerate(objects, start=1):
        _add_obj(predictor, state, oid, obj, W, H)
        obj_label[oid] = int(obj.get("label", 1))

    labels = np.zeros((T, H, W), np.int32)
    for fidx, obj_ids, low, video_res_masks, scores in predictor.propagate_in_video(
            state, start_frame_idx=0, max_frame_num_to_track=T, reverse=False, propagate_preflight=True):
        fm = {int(o): _mask_np(video_res_masks[i] > 0.0) for i, o in enumerate(obj_ids)}
        for oid, m in fm.items():
            if obj_label.get(oid) == 2:
                labels[fidx][m] = 2
        for oid, m in fm.items():
            if obj_label.get(oid) == 1:
                labels[fidx][m] = 1

    np.savez_compressed(out_npz, labels=labels,
                        frame_indices=np.array(meta["frame_indices"], np.int64),
                        timestamps=np.array(meta["timestamps"], np.int64))

    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
        for t in range(T):
            over = cv2.imread(os.path.join(frames_dir, "%05d.jpg" % t))
            lab = labels[t]
            for lv, col in [(1, (0, 0, 220)), (2, (40, 200, 40))]:
                m = lab == lv
                if m.any():
                    over[m] = (0.5 * np.array(col) + 0.5 * over[m]).astype(np.uint8)
            cv2.imwrite(os.path.join(overlay_dir, "overlay_%06d.jpg" % meta["frame_indices"][t]), over)
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--hand-bbox", type=float, nargs=4, action="append", default=[])
    ap.add_argument("--hand-point", type=float, nargs=2, action="append", default=[])
    ap.add_argument("--object-bbox", type=float, nargs=4, action="append", default=[])
    ap.add_argument("--object-point", type=float, nargs=2, action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--overlay-dir", default=None)
    a = ap.parse_args()
    objs = []
    for b in a.hand_bbox:    objs.append({"label": 1, "box": b})
    for p in a.hand_point:   objs.append({"label": 1, "points": [[p[0], p[1], 1]]})
    for b in a.object_bbox:  objs.append({"label": 2, "box": b})
    for p in a.object_point: objs.append({"label": 2, "points": [[p[0], p[1], 1]]})
    lab = track_labelmap(a.frames_dir, objs, a.out, ckpt=a.ckpt, overlay_dir=a.overlay_dir)
    print("saved %s  labels=%s" % (a.out, lab.shape))
