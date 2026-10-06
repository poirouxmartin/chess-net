"""Phase 1: fast self-play games -> shards with FENs + histories.

Positions feed phase 2 (teacher queries). Uses our (random) net for
move selection so positions look like real play, not piece soup.
Shard format (.npz): fens [N] str, hists [N,7] str ("" if missing).

Usage:
  py -m chessnet.gen_distill --out-dir teacher/data --games 3000 --threads 8
"""
import argparse
import os
import threading
import time

import chess
import numpy as np
import torch

from .fastbatch import InferenceServer, play_fast_game
from .resnet import create_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="../../teacher/data")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--shard-games", type=int, default=200)
    ap.add_argument("--max-plies", type=int, default=350)
    ap.add_argument("--temp-plies", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.out_dir)
    os.makedirs(base, exist_ok=True)
    # Resume: skip shard indices already on disk (crashed runs keep shards).
    import glob as _glob
    import re as _re
    taken = set()
    for fp in _glob.glob(os.path.join(base, "games_*.npz")):
        m = _re.search(r"games_(\d+)\.npz$", fp)
        if m:
            taken.add(int(m.group(1)))
    shard_n = [max(taken) + 1 if taken else 0]
    if taken:
        print(f"resuming: {len(taken)} shards exist, starting at {shard_n[0]}",
              flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model().to(device)
    server = InferenceServer(model, device, max_batch=512)
    server.start()

    lock = threading.Lock()
    made = [0]
    buf = []  # (fen, hist tuple)

    def worker(seed):
        import random
        rng = random.Random(seed)
        while True:
            with lock:
                if made[0] >= args.games:
                    return
                made[0] += 1
                gid = made[0]
            fens, ucis, hists, result, term = play_fast_game(
                server, args.temp_plies, args.max_plies, 1.0)
            with lock:
                for f, h in zip(fens, hists):
                    buf.append((f, tuple(h)))
                if gid % 20 == 0:
                    print(f"  games {gid}/{args.games} pos={len(buf)}", flush=True)

    threads = [threading.Thread(target=worker, args=(args.seed + i,), daemon=True)
               for i in range(args.threads)]
    for t in threads:
        t.start()

    def flush():
        with lock:
            if not buf:
                return 0
            fens = np.array([f for f, _ in buf], dtype="<U128")
            h7 = np.full((len(buf), 7), "", dtype="<U128")
            for i, (_, h) in enumerate(buf):
                h7[i, :len(h)] = list(h)[-7:]
            path = os.path.join(base, f"games_{shard_n[0]:04d}.npz")
            np.savez_compressed(path, fens=fens, hists=h7)
            n = len(buf)
            buf.clear()
            shard_n[0] += 1
            return n

    saved_pos = 0
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        with lock:
            big = len(buf) >= args.shard_games * args.max_plies
        if big:
            n = flush()
            saved_pos += n
            print(f"shard {shard_n[0]-1}: {n} pos (total {saved_pos})", flush=True)
    n = flush()
    saved_pos += n
    if n:
        print(f"shard {shard_n[0]-1}: {n} pos (total {saved_pos})", flush=True)
    server.stop()
    print(f"DONE games={made[0]} positions={saved_pos} shards={shard_n[0]}")


if __name__ == "__main__":
    main()
