"""Parse Elite PGNs -> shards with (FEN + history), same format as gen_distill.

Samples fixed plies per game (skips opening theory, keeps middlegame
density where tactics live). Skips broken games.

Usage:
  py -m chessnet.pgn_to_shards --pgn ../teacher/pgn/elite_2025-11/lichess_elite_2025-11.pgn --out-dir ../../teacher/elite --max-games 400000
"""
import argparse
import os

import chess
import chess.pgn
import numpy as np

PLIES = (12, 20, 28, 40, 55, 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", required=True)
    ap.add_argument("--out-dir", default="../../teacher/elite")
    ap.add_argument("--max-games", type=int, default=400000)
    ap.add_argument("--shard-pos", type=int, default=70000)
    ap.add_argument("--prefix", default="elite")
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    pgn_path = os.path.join(base, args.pgn)
    out = os.path.join(base, args.out_dir)
    os.makedirs(out, exist_ok=True)
    # Resume: skip shard indices already on disk.
    import glob as _glob
    import re as _re
    taken = set()
    for fp in _glob.glob(os.path.join(out, f"{args.prefix}_*.npz")):
        m = _re.search(r"_(\d+)\.npz$", fp)
        if m:
            taken.add(int(m.group(1)))
    shard = max(taken) + 1 if taken else 0
    if taken:
        print(f"resuming at shard {shard}", flush=True)

    buf = []
    ngames = npos = skipped = 0

    def flush():
        nonlocal buf, npos, shard
        if not buf:
            return
        fens = np.array([f for f, _ in buf], dtype="<U128")
        h7 = np.full((len(buf), 7), "", dtype="<U128")
        for i, (_, h) in enumerate(buf):
            h7[i, :len(h)] = list(h)[-7:]
        path = os.path.join(out, f"{args.prefix}_{shard:04d}.npz")
        np.savez_compressed(path, fens=fens, hists=h7)
        npos += len(buf)
        print(f"shard {shard}: {len(buf)} pos (total {npos})", flush=True)
        shard += 1
        buf = []

    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        while ngames < args.max_games:
            try:
                game = chess.pgn.read_game(f)
            except Exception:
                skipped += 1
                continue
            if game is None:
                break
            ngames += 1
            try:
                b = game.board()
                fens, hists = [], []
                seen = []
                for mv in game.mainline_moves():
                    seen.append(b.fen())
                    b.push(mv)
                    if len(seen) in PLIES:
                        fens.append(b.fen())
                        hists.append(tuple(seen[-7:]))
            except Exception:
                skipped += 1
                continue
            for x, h in zip(fens, hists):
                buf.append((x, h))
            if len(buf) >= args.shard_pos:
                flush()
            if ngames % 20000 == 0:
                print(f"  games {ngames} pos={npos + len(buf)} skipped={skipped}",
                      flush=True)
    flush()
    print(f"DONE games={ngames} positions={npos} skipped={skipped}")


if __name__ == "__main__":
    main()
