"""Differential test: my encode_112 vs reference LeelaBoard.lcz_features().

Plays seeded random games (both colors, castling, EP, promotions,
repetitions) and asserts bitwise equality on every position.
"""
import os
import random
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../teacher"))
from leela_board import LeelaBoard  # noqa: E402  (reference implementation)

from chessnet.lc0_teacher import encode_112


def main():
    rng = random.Random(20260908)
    tested = 0
    for game in range(12):
        ref = LeelaBoard()
        cur = chess.Board()
        hists = []
        for ply in range(120):
            mine = encode_112(cur, hists)
            theirs = ref.lcz_features().astype(np.float32)
            assert mine.shape == theirs.shape == (112, 8, 8), (mine.shape, theirs.shape)
            if not np.array_equal(mine, theirs):
                d = np.abs(mine - theirs).sum(axis=(1, 2))
                bad = [i for i, v in enumerate(d) if v > 0]
                raise SystemExit(
                    f"MISMATCH game {game} ply {ply} fen={cur.fen()} planes={bad}")
            tested += 1
            if cur.is_game_over():
                break
            mv = rng.choice(list(cur.legal_moves))
            ref.push_uci(mv.uci())
            hists.append(cur.fen())
            cur.push(mv)
    print(f"differential OK: {tested} positions bitwise identical")


if __name__ == "__main__":
    main()
