"""
bag_to_ply_pipeline — end-to-end: SAM label maps + RGBD -> fused, coloured point cloud.

Each `.ply` point carries: xyz, a display colour, and a raw integer label
(0=bg, 1=hand, 2=object). Display colour is chosen by `--color-mode`:
  rgb    (default) — the point's REAL colour from the colour image  ← check camera alignment
  camera           — cam1 red / cam3 blue                            ← starkest alignment check
  label            — 0 grey / 1 hand-red / 2 object-green            ← segmentation view

Geometry per point: depth pixel -> 3D (depth intrinsics) -> colour frame
(depth->colour extrinsic, taken from the BAG) -> colour pixel (read label + RGB);
then cam3 -> cam1 (cam1_cam3_extrinsic); concatenate. Cleanup filters: per-camera
depth fence, spatial SOR, and largest-cluster keep for the object.
Use the SAME `stride` here as when the `.npz` masks were made.
"""
import os
import json
import numpy as np

try:
    from . import obbag
except ImportError:
    import obbag

LABEL_RGB = np.array([[160, 160, 160], [220, 40, 40], [40, 200, 40]], np.uint8)  # bg, hand, object
CAM_RGB = np.array([[220, 60, 60], [60, 120, 220]], np.uint8)                    # cam1 red, cam3 blue


def load_depth_profile(bag):
    """depth->colour extrinsic for `bag`. A few recordings had their profile topic
    flooded with junk, wiping the real profile — read_profiles now raises instead
    of returning garbage. In that case borrow the extrinsic from a sibling
    recording of the SAME camera: depth->colour is a fixed factory calibration,
    bit-for-bit identical across every recording in the session."""
    try:
        return obbag.read_profiles(bag)['depth']
    except ValueError:
        pass
    name = os.path.basename(bag)                       # e.g. camera_1_rgb_depth.bag
    session = os.path.dirname(os.path.dirname(bag))    # .../<session>/<recording>/<bag>
    for rec in sorted(os.listdir(session)):
        donor = os.path.join(session, rec, name)
        if os.path.abspath(donor) == os.path.abspath(bag) or not os.path.exists(donor):
            continue
        try:
            prof = obbag.read_profiles(donor)['depth']
        except ValueError:
            continue
        print("  [profile] %s has no valid depth profile; borrowed extrinsic from %s" % (name, rec))
        return prof
    raise ValueError("no valid depth profile in %s or any sibling recording" % bag)


def camera_params(intrinsics_json, bag):
    """Intrinsics from the JSON; depth->colour extrinsic from the BAG (the JSON's
    rotation_matrix is EMPTY, and identity mis-registers depth vs colour by ~7 deg)."""
    p = json.load(open(intrinsics_json))['camera_param']
    r, d = p['rgb_intrinsic'], p['depth_intrinsic']
    cK = (r['fx'], r['fy'], r['cx'], r['cy'])
    dK = (d['fx'], d['fy'], d['cx'], d['cy'])
    prof = load_depth_profile(bag)
    return dK, cK, np.asarray(prof['R'], float), np.asarray(prof['T'], float) / 1000.0


def back_project_labeled(depth, label_map, color, dK, cK, R_d2c, T_d2c):
    """Depth frame + colour-frame label map + colour image -> (points in COLOUR
    frame, labels, per-point RGB)."""
    dfx, dfy, dcx, dcy = dK
    cfx, cfy, ccx, ccy = cK
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    Z = depth.astype(np.float64) / 1000.0
    keep = Z > 0
    xs, ys, Z = xs[keep], ys[keep], Z[keep]

    Xd = (xs - dcx) * Z / dfx
    Yd = (ys - dcy) * Z / dfy
    P_depth = np.stack([Xd, Yd, Z], axis=1)
    P_color = P_depth @ R_d2c.T + T_d2c

    good = P_color[:, 2] > 0
    u = cfx * P_color[:, 0] / np.where(good, P_color[:, 2], 1.0) + ccx
    v = cfy * P_color[:, 1] / np.where(good, P_color[:, 2], 1.0) + ccy
    ui = np.round(u).astype(int)
    vi = np.round(v).astype(int)
    CH, CW = label_map.shape
    inb = good & (ui >= 0) & (ui < CW) & (vi >= 0) & (vi < CH)

    labels = np.zeros(len(P_color), np.int32)
    labels[inb] = label_map[vi[inb], ui[inb]]
    rgb = np.full((len(P_color), 3), 130, np.uint8)         # neutral for out-of-view points
    rgb[inb] = color[vi[inb], ui[inb]][:, ::-1]             # colour is BGR -> store RGB
    return P_color, labels, rgb


def save_ply(path, pts, labels, rgb, cam, color_mode="rgb"):
    """Ascii .ply; display colour per `color_mode`, always with a raw label field."""
    labels = np.clip(labels, 0, 2)
    if color_mode == "label":
        disp = LABEL_RGB[labels]
    elif color_mode == "camera":
        disp = CAM_RGB[cam.astype(int)]
    elif color_mode == "tint":                       # real-colour background + SOLID hand/object
        disp = rgb.copy()
        disp[labels == 1] = LABEL_RGB[1]             # solid red   (hand)
        disp[labels == 2] = LABEL_RGB[2]             # solid green (object)
    else:
        disp = rgb
    data = np.column_stack([pts, disp, labels])
    with open(path, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex %d\n" % len(pts))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property uchar label\nend_header\n")
        np.savetxt(f, data, fmt="%.4f %.4f %.4f %d %d %d %d")


def depth_declutter(pts, labels):
    """Per label, relabel far depth-outliers to bg (Tukey 1.5*IQR on Z) — the
    "hand-mask edge lands on the far table" case."""
    Z = pts[:, 2]
    for L in (1, 2):
        m = labels == L
        if m.sum() < 30:
            continue
        q1, q3 = np.percentile(Z[m], [25, 75])
        iqr = q3 - q1
        labels[m & ((Z < q1 - 1.5 * iqr) | (Z > q3 + 1.5 * iqr))] = 0
    return labels


def spatial_declutter(pts, labels, k=16, std_ratio=2.0):
    """Statistical outlier removal per label: relabel spatially-isolated (sparse)
    points to bg. Cleans scatter from noisy depth that a 1-D depth fence misses."""
    from scipy.spatial import cKDTree
    for L in (1, 2):
        idx = np.where(labels == L)[0]
        if len(idx) < k + 5:
            continue
        P = pts[idx]
        d, _ = cKDTree(P).query(P, k=k + 1)
        md = d[:, 1:].mean(1)
        labels[idx[md > md.mean() + std_ratio * md.std()]] = 0
    return labels


def cluster_keep_largest(pts, labels, target=2, eps=0.03):
    """Keep only the largest connected cluster of `target` (object=2); relabel the
    rest to bg. Removes a transparent object's dense false-depth second blob.
    Object-only — hands can be two legit gloves."""
    from scipy.spatial import cKDTree
    idx = np.where(labels == target)[0]
    if len(idx) < 30:
        return labels
    P = pts[idx]
    parent = np.arange(len(P))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in cKDTree(P).query_pairs(eps, output_type="ndarray"):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(i) for i in range(len(P))])
    u, c = np.unique(roots, return_counts=True)
    labels[idx[roots != u[c.argmax()]]] = 0
    return labels


def process_frame(dep1, lab1, col1, params1, dep3, lab3, col3, params3, R13, T13,
                  declutter=True, sor=True, sor_std=2.0, cluster_obj=True):
    """Back-project + colour + label both cameras, move cam3 into cam1's frame, concatenate."""
    Pc1, l1, rgb1 = back_project_labeled(dep1, lab1, col1, *params1)
    Pc3, l3, rgb3 = back_project_labeled(dep3, lab3, col3, *params3)
    if declutter:
        l1 = depth_declutter(Pc1, l1)
        l3 = depth_declutter(Pc3, l3)
    Pc3_in1 = Pc3 @ R13.T + T13
    pts = np.vstack([Pc1, Pc3_in1])
    labs = np.concatenate([l1, l3])
    rgb = np.vstack([rgb1, rgb3])
    cam = np.concatenate([np.zeros(len(Pc1), np.uint8), np.ones(len(Pc3), np.uint8)])
    if sor:
        labs = spatial_declutter(pts, labs, std_ratio=sor_std)
    if cluster_obj:
        labs = cluster_keep_largest(pts, labs, target=2)
    return pts, labs, rgb, cam


def run_pipeline(cam1_bag, cam3_bag, cam1_json, cam3_json, extrinsic_json,
                 cam1_npz, cam3_npz, out_dir, stride=30, labeled_only=False,
                 max_frames=None, color_mode="rgb"):
    """Fuse cam1 + cam3 into one coloured point cloud per frame; write .ply files."""
    os.makedirs(out_dir, exist_ok=True)
    L1 = np.load(cam1_npz)['labels']
    L3 = np.load(cam3_npz)['labels']
    params1 = camera_params(cam1_json, cam1_bag)
    params3 = camera_params(cam3_json, cam3_bag)

    ext = json.load(open(extrinsic_json))['cam3_to_cam1']
    R13 = np.array(ext['rotation_matrix'])
    T13 = np.array(ext['translation_m'])

    s1d = obbag.frame_stream(cam1_bag, 'depth', stride)
    s3d = obbag.frame_stream(cam3_bag, 'depth', stride)
    s1c = obbag.frame_stream(cam1_bag, 'color', stride)
    s3c = obbag.frame_stream(cam3_bag, 'color', stride)
    n_labels = min(len(L1), len(L3))

    n = 0
    for t, ((i1, _, dep1), (i3, _, dep3), (_, _, col1), (_, _, col3)) in enumerate(zip(s1d, s3d, s1c, s3c)):
        if t >= n_labels:
            break
        pts, labs, rgb, cam = process_frame(dep1, L1[t], col1, params1,
                                            dep3, L3[t], col3, params3, R13, T13)
        if labeled_only:
            m = labs > 0
            pts, labs, rgb, cam = pts[m], labs[m], rgb[m], cam[m]
        save_ply(os.path.join(out_dir, "frame_%06d.ply" % i1), pts, labs, rgb, cam, color_mode)
        n += 1
        print("frame %6d: %7d pts (%6d hand, %6d object)"
              % (i1, len(labs), int((labs == 1).sum()), int((labs == 2).sum())))
        if max_frames and n >= max_frames:
            break
    print("wrote %d clouds -> %s  (color_mode=%s)" % (n, out_dir, color_mode))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Fuse cam1+cam3 RGBD into coloured point clouds.")
    ap.add_argument("cam1_bag")
    ap.add_argument("cam3_bag")
    ap.add_argument("cam1_json", help="camera_1_intrinsics.json")
    ap.add_argument("cam3_json", help="camera_3_intrinsics.json")
    ap.add_argument("extrinsic_json", help="cam1_cam3_extrinsic.json")
    ap.add_argument("cam1_npz")
    ap.add_argument("cam3_npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=30)
    ap.add_argument("--color-mode", choices=["rgb", "tint", "camera", "label"], default="rgb",
                    help="rgb=real colour, tint=real+mask highlight, camera=cam1/cam3, label=0/1/2")
    ap.add_argument("--labeled-only", action="store_true",
                    help="keep only hand+object points (drop background)")
    ap.add_argument("--max-frames", type=int, default=None)
    a = ap.parse_args()
    run_pipeline(a.cam1_bag, a.cam3_bag, a.cam1_json, a.cam3_json, a.extrinsic_json,
                 a.cam1_npz, a.cam3_npz, a.out, stride=a.stride, labeled_only=a.labeled_only,
                 max_frames=a.max_frames, color_mode=a.color_mode)
