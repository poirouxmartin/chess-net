"""CSNN binary format reader/writer, aligned with crates/nn/src/lib.rs.

Header (24 bytes, little-endian):
  magic   4 bytes  "CSNN"
  version u32     1 = value only, 2 = value + policy
  feat    u32     1 = HalfKP, 2 = KP768
  l0      u32     first hidden size
  l1      u32     second hidden size
  feat_count u32  number of features

Then float32 arrays (v1 and v2):
  fw  feat_count * l0     (row-major, feature-major)
  fb  l0
  w1  l1 * l0             (row-major)
  b1  l1
  wo  l1
  bo  1

v2 appends the policy head (shares h1 = relu(b1 + w1 . relu(acc))):
  policy_size u32
  wp  policy_size * l1    (row-major, row = move index from*64+to)
  bp  policy_size

Score in the Rust engine = (bo + wo . relu(h1)) * 400.
"""

import struct

MAGIC = b"CSNN"
VERSION = 2

FEAT_HALFKP = 1
FEAT_KP768 = 2


def load(path):
    """Return dict {feat, l0, l1, feat_count, fw, fb, w1, b1, wo, bo,
    policy_size?, wp?, bp?}."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != MAGIC:
        raise ValueError("bad magic")
    version, feat, l0, l1, feat_count = struct.unpack_from("<IIIII", data, 4)
    if version not in (1, 2):
        raise ValueError(f"unsupported version {version}")
    arrays = {}
    pos = 24
    for name, n in [
        ("fw", feat_count * l0),
        ("fb", l0),
        ("w1", l1 * l0),
        ("b1", l1),
        ("wo", l1),
        ("bo", 1),
    ]:
        arrays[name] = struct.unpack_from(f"<{n}f", data, pos)
        pos += 4 * n
    if version == 2:
        (policy_size,) = struct.unpack_from("<I", data, pos)
        pos += 4
        arrays["wp"] = struct.unpack_from(f"<{policy_size * l1}f", data, pos)
        pos += 4 * policy_size * l1
        arrays["bp"] = struct.unpack_from(f"<{policy_size}f", data, pos)
        arrays["policy_size"] = policy_size
    arrays["feat"] = feat
    arrays["l0"] = l0
    arrays["l1"] = l1
    arrays["feat_count"] = feat_count
    return arrays


def save(path, feat, l0, l1, feat_count, fw, fb, w1, b1, wo, bo,
         policy=None):
    """Write arrays as lists/tuples of floats to a CSNN file. If policy is a
    (wp, bp) pair, version 2 (value + policy) is written, otherwise v1."""
    if policy is None:
        version = 1
        wp = bp = ()
        policy_size = 0
    else:
        version = VERSION
        wp, bp = policy
        policy_size = len(bp)
    header = MAGIC + struct.pack("<IIIII", version, feat, l0, l1, feat_count)
    parts = [header]
    for arr in (fw, fb, w1, b1, wo, [bo]):
        parts.append(struct.pack(f"<{len(arr)}f", *arr))
    if policy is not None:
        parts.append(struct.pack("<I", policy_size))
        for arr in (wp, bp):
            parts.append(struct.pack(f"<{len(arr)}f", *arr))
    with open(path, "wb") as f:
        f.write(b"".join(parts))