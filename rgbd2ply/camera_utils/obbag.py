"""Minimal pure-Python reader for Orbbec ROS1 .bag (Femto Bolt), no Orbbec SDK.
Parses sensor_msgs/Image and custom_msg/OBStreamProfileInfo by hand (LE, packed).
Uses `rosbags` only as the bag container (raw message bytes per connection).

Stable copy for the rgb_masking module (was a session scratchpad). Adds
`frame_stream` — an O(n) streaming generator — on top of the original API.
"""
import struct, numpy as np, cv2
from rosbags.rosbag1 import Reader

_COLOR_TOPIC = '/cam/sensor_2/frameType_2'
_DEPTH_TOPIC = '/cam/sensor_3/frameType_3'


def _hdr(b, o):
    # std_msgs/Header: uint32 seq, uint32 sec, uint32 nsec, string frame_id
    seq, sec, nsec = struct.unpack_from('<III', b, o); o += 12
    (n,) = struct.unpack_from('<I', b, o); o += 4
    frame_id = b[o:o+n].decode('ascii', 'replace'); o += n
    return o, (seq, sec, nsec, frame_id)


def parse_image(b):
    o, _ = _hdr(b, 0)
    height, width = struct.unpack_from('<II', b, o); o += 8
    (n,) = struct.unpack_from('<I', b, o); o += 4
    encoding = b[o:o+n].decode('ascii', 'replace'); o += n
    (is_be,) = struct.unpack_from('<B', b, o); o += 1
    (step,) = struct.unpack_from('<I', b, o); o += 4
    (dlen,) = struct.unpack_from('<I', b, o); o += 4
    data = b[o:o+dlen]; o += dlen
    return dict(height=height, width=width, encoding=encoding, step=step, data=data)


def parse_stream_profile(b):
    o, _ = _hdr(b, 0)
    streamType, fmt = struct.unpack_from('<BB', b, o); o += 2
    R = struct.unpack_from('<9f', b, o); o += 36
    T = struct.unpack_from('<3f', b, o); o += 12
    width, height, fps = struct.unpack_from('<3H', b, o); o += 6
    fx, fy, cx, cy = struct.unpack_from('<4f', b, o); o += 16
    dist = struct.unpack_from('<8f', b, o); o += 32
    (distModel,) = struct.unpack_from('<B', b, o); o += 1
    return dict(streamType=streamType, fmt=fmt,
                R=np.array(R, np.float64).reshape(3, 3),
                T=np.array(T, np.float64),  # mm, this stream -> color(reference)
                width=width, height=height, fps=fps,
                K=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64),
                fx=fx, fy=fy, cx=cx, cy=cy,
                dist=np.array(dist, np.float64), distModel=distModel)


def _conns(r, topic):
    return [c for c in r.connections if c.topic == topic]


def _decode(m, which):
    """Turn a parsed image message into a numpy array (color->BGR uint8, depth->uint16 mm)."""
    if which == 'color':
        buf = np.frombuffer(m['data'], np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:  # maybe raw
            img = buf.reshape(m['height'], m['width'], -1)
        return img
    return np.frombuffer(m['data'], np.uint16).reshape(m['height'], m['width'])


# A full OBStreamProfileInfo is 121 bytes: 16 hdr + 2 type/fmt + 36 R + 12 T
# + 6 wh/fps + 16 fx..cy + 32 dist + 1 distModel. Shorter messages are partial
# reads (seen when several processes hammer the same bag at once) — skip them.
_PROFILE_MIN_LEN = 121


def _valid_profile(p):
    """True only for a genuine OBStreamProfileInfo. Some bags had thousands of
    junk messages flooded onto the profile topic (the real one wiped out); a junk
    blob that is merely long enough parses without a struct.error but yields a
    garbage extrinsic (e.g. width=height=0, non-rotation R). Reject those so we
    never silently return a bad depth->color extrinsic — a real profile has a
    plausible resolution and either a proper rotation R (orthonormal, det ~ +1) or
    an all-zero R (the colour/reference stream's self-relative sentinel)."""
    w, h = p['width'], p['height']
    if not (0 < w <= 4096 and 0 < h <= 4096):
        return False
    R = p['R']
    if not np.all(np.isfinite(R)):
        return False
    if np.allclose(R, 0.0):                         # reference (colour) stream: R/T are zero
        return True
    # Loose bound: real factory R is a stored float32 rotation (slightly
    # denormalized, row norms ~0.99); junk R has values ~1e25 and det ~0 or huge.
    return (np.allclose(R @ R.T, np.eye(3), atol=5e-2)
            and abs(np.linalg.det(R) - 1.0) < 5e-2)


def read_profiles(bag):
    out = {}
    with Reader(bag) as r:
        for key, tp in [('color', _COLOR_TOPIC.replace('sensor_2/frameType_2', 'streamProfileType_2')),
                        ('depth', _DEPTH_TOPIC.replace('sensor_3/frameType_3', 'streamProfileType_3'))]:
            cc = _conns(r, tp)
            for conn, t, raw in r.messages(connections=cc):
                if len(raw) < _PROFILE_MIN_LEN:
                    continue                       # partial/garbage message — keep looking
                try:
                    p = parse_stream_profile(raw)
                except struct.error:
                    continue                       # not a valid profile — keep looking
                if not _valid_profile(p):
                    continue                       # parsed but garbage (junk on the topic) — keep looking
                out[key] = p; break
            if key not in out:
                raise ValueError("no valid %s stream profile found in %s" % (key, bag))
    return out


def read_frame(bag, which='color', index=0):
    """Return (timestamp_ns, image) for a SINGLE frame. Re-scans from the start
    each call (O(index)); use `frame_stream` for many frames."""
    topic = _COLOR_TOPIC if which == 'color' else _DEPTH_TOPIC
    with Reader(bag) as r:
        cc = _conns(r, topic)
        for i, (conn, t, raw) in enumerate(r.messages(connections=cc)):
            if i < index:
                continue
            return t, _decode(parse_image(raw), which)
    return None, None


def frame_stream(bag, which='color', stride=1):
    """Yield (frame_index, timestamp_ns, image) for every `stride`-th frame,
    opening the bag ONCE.

    Streaming counterpart to `read_frame`: one linear pass (O(n)) instead of an
    O(index) re-scan per call. `frame_index` is the true position in the stream
    (0, stride, 2*stride, ...); `timestamp_ns` lets callers time-align separate
    cameras. color -> BGR uint8; depth -> uint16 (mm).

    Usage::
        for i, ts, img in frame_stream(bag, stride=30):
            ...  # one frame in memory at a time
    """
    topic = _COLOR_TOPIC if which == 'color' else _DEPTH_TOPIC
    with Reader(bag) as r:
        cc = _conns(r, topic)
        for i, (conn, t, raw) in enumerate(r.messages(connections=cc)):
            if i % stride != 0:
                continue
            yield i, t, _decode(parse_image(raw), which)


def all_timestamps(bag, which='color'):
    topic = _COLOR_TOPIC if which == 'color' else _DEPTH_TOPIC
    ts = []
    with Reader(bag) as r:
        for conn, t, raw in r.messages(connections=_conns(r, topic)):
            ts.append(t)
    return np.array(ts, dtype=np.int64)
