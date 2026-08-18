"""CSNN binary format reader/writer, aligned with crates/nn/src/lib.rs.

Header (24 bytes, little-endian):
  magic   4 bytes  "CSNN"
  version u32 = 1
  feat    u32     1 = HalfKP, 2 = KP768
  l0      u32     first hidden size
  l1      u32     second hidden size
  feat_count u32  number of features

Then float32 arrays:
  fw  feat_count * l0     (row-major, feature-major)
  fb  l0
  w1  l1 * l0             (row-major)
  b1  l1
  wo  l1
  bo  1

Score in the Rust engine = (bo + wo . relu(b1 + w1 . relu(fb + fw^T onehot))) * 400.
"""

import struct

MAGIC = b"CSNN"
VERSION = 1

FEAT_HALFKP = 1
FEAT_KP768 = 2


def load(path):
    """Return dict {feat, l0, l1, feat_count, fw, fb, w1, b1, wo, bo}."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != MAGIC:
        raise ValueError("bad magic")
    version, feat, l0, l1, feat_count = struct.unpack_from("<IIIII", data, 4)
    if version != VERSION:
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
    arrays["feat"] = feat
    arrays["l0"] = l0
    arrays["l1"] = l1
    arrays["feat_count"] = feat_count
    return arrays


def save(path, feat, l0, l1, feat_count, fw, fb, w1, b1, wo, bo):
    """Write arrays as lists/tuples of floats to a CSNN file."""
    header = MAGIC + struct.pack("<IIIII", VERSION, feat, l0, l1, feat_count)
    parts = [header]
    for arr in (fw, fb, w1, b1, wo, [bo]):
        parts.append(struct.pack(f"<{len(arr)}f", *arr))
    with open(path, "wb") as f:
        f.write(b"".join(parts))