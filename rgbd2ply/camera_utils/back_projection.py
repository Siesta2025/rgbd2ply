"""Back-project one camera's depth frame into a 3D point cloud (.ply).

Demo version: plain loops, simple syntax. One depth pixel -> one 3D point.
"""
import numpy as np
try:
    from . import obbag
except ImportError:
    import obbag


def back_projection(Z_mm, x, y, cx, cy, fx, fy):
    """One depth pixel -> one 3D point in metres (the pinhole formula)."""
    Z = Z_mm / 1000.0            
    X = (x - cx) * Z / fx
    Y = (y - cy) * Z / fy
    return X, Y, Z


def depth_to_ply(bag_path, out_path):
    """Read the depth stream's first frame, back-project every valid pixel,
    and save the result as an ascii .ply. Returns the (N, 3) point array."""
    ts, depth = obbag.read_frame(bag_path, which='depth')   # (H, W) uint16, mm
    prof = obbag.read_profiles(bag_path)['depth']            # depth intrinsics
    fx, fy, cx, cy = prof['fx'], prof['fy'], prof['cx'], prof['cy']

    H, W = depth.shape
    pts = []                                    # grows one point at a time
    for y in range(H):                          # y = row
        for x in range(W):                      # x = column
            Z = depth[y, x]
            if Z == 0:
                continue                        # no depth reading here -> skip
            pts.append(back_projection(Z, x, y, cx, cy, fx, fy))

    pts = np.array(pts)                         # list -> (N, 3) array
    save_ply(out_path, pts)
    return pts


def save_ply(path, pts):
    """Write an (N, 3) point array as a simple ascii .ply file."""
    with open(path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex %d\n" % len(pts))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in pts:
            f.write("%.4f %.4f %.4f\n" % (p[0], p[1], p[2]))


if __name__ == "__main__":
    import sys
    pts = depth_to_ply(sys.argv[1], sys.argv[2])
    print("wrote %d points -> %s" % (len(pts), sys.argv[2]))
    print("X range: %.2f .. %.2f" % (pts[:, 0].min(), pts[:, 0].max()))
    print("Y range: %.2f .. %.2f" % (pts[:, 1].min(), pts[:, 1].max()))
    print("Z range: %.2f .. %.2f" % (pts[:, 2].min(), pts[:, 2].max()))
