"""Phase 2: query teacher on game shards -> distillation targets.

Reads games_XXXX.npz (fens + hists), encodes Lc0-112, queries the teacher,
maps policy 1858 -> our 4288 indices (softmax T=1 over legal moves),
Q = (w - l + 1) / 2 (STM win-proba, matches train z in [0,1]).
Writes distill_XXXX.npz: fens, hists, pi_idx [N,128], pi_val [N,128],
pi_n [N], q [N].

Usage:
  py -m chessnet.query_teacher --data-dir ../teacher/data --temp 1.0
"""
import argparse
import glob
import os

import chess
import numpy as np

from .lc0_teacher import TeacherSession, encode_112, lc0_index
from .resnet import policy_index

MAX_MOVES = 128


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../teacher/data")
    ap.add_argument("--pattern", default="games_*.npz")
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--query-batch", type=int, default=1024)
    args = ap.parse_args()

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.data_dir)
    teacher = TeacherSession(batch=args.query_batch)
    total = 0
    for src in sorted(glob.glob(os.path.join(base, args.pattern))):
        tag = os.path.splitext(os.path.basename(src))[0]
        dst = os.path.join(base, f"distill_{tag}.npz")
        if os.path.exists(dst):
            print(f"skip {dst} (exists)", flush=True)
            total += int(np.load(dst)["q"].shape[0])
            continue
        d = np.load(src, allow_pickle=False)
        fens = [str(f) for f in d["fens"]]
        hists = [[str(h) for h in row if h] for row in d["hists"]]
        n = len(fens)
        planes = np.stack([encode_112(chess.Board(f), h) for f, h in zip(fens, hists)])
        tpol, twdl = teacher.query(planes)
        pi_idx = np.zeros((n, MAX_MOVES), dtype=np.int32)
        pi_val = np.zeros((n, MAX_MOVES), dtype=np.float32)
        pi_n = np.zeros(n, dtype=np.int32)
        q = np.zeros(n, dtype=np.float32)
        dropped = 0
        for i, (f, lp, w) in enumerate(zip(fens, tpol, twdl)):
            b = chess.Board(f)
            legal = list(b.legal_moves)
            pairs = []
            for m in legal:
                li = lc0_index(b, m)
                if li is None or li >= len(lp):
                    dropped += 1
                    continue
                pairs.append((policy_index(b, m), float(lp[li])))
            if not pairs:
                continue
            ids = np.array([p[0] for p in pairs])
            lg = np.array([p[1] for p in pairs]) / max(0.05, args.temp)
            lg -= lg.max()
            e = np.exp(lg)
            e /= e.sum()
            k = min(len(ids), MAX_MOVES)
            ek = e[:k]
            ek /= ek.sum()  # renormalize (truncate only past 128 legal moves)
            pi_idx[i, :k] = ids[:k]
            pi_val[i, :k] = ek
            pi_n[i] = k
            q[i] = (float(w[0]) - float(w[2]) + 1.0) / 2.0
        np.savez_compressed(dst, fens=d["fens"], hists=d["hists"],
                            pi_idx=pi_idx, pi_val=pi_val,
                            pi_n=pi_n, q=q)
        print(f"{dst}: {n} pos, dropped-moves={dropped}", flush=True)
        total += n
    print(f"DONE positions={total}")


if __name__ == "__main__":
    main()
