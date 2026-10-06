"""One-off: split 70k shards into <=20k chunks (worker memory quota).

Reads games_*/elite_*.npz, writes <name>_p0/p1...npz, verifies row counts,
deletes the original. Run once.
"""
import glob
import os

import numpy as np

CHUNK = 20000
base = os.path.dirname(os.path.abspath(__file__))

total_in = total_out = 0
for pat in ("../teacher/data/games_*.npz", "../teacher/elite/elite_*.npz"):
    for src in sorted(glob.glob(os.path.join(base, pat))):
        if "_p0" in src or "_p1" in src:
            continue
        with np.load(src, allow_pickle=False) as d:
            fens, hists = d["fens"], d["hists"]
            n = len(fens)
            stem = os.path.splitext(src)[0]
            # Skip if chunks already exist (previous partial run).
            if glob.glob(stem + "_p*.npz"):
                print(f"skip {os.path.basename(src)} (chunks exist)", flush=True)
                total_in += n
                total_out += n
                continue
            k = 0
            for s in range(0, n, CHUNK):
                e = min(n, s + CHUNK)
                dst = f"{stem}_p{k}.npz"
                np.savez_compressed(
                    dst, fens=fens[s:e], hists=hists[s:e])
                total_out += e - s
                k += 1
            print(f"{os.path.basename(src)}: {n} -> {k} chunks", flush=True)
            total_in += n
        os.remove(src)
print(f"total in={total_in} out={total_out}")
assert total_in == total_out
print("RESHARD OK")
