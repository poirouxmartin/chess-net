"""Generate a supervised training dataset for train_nnue.py.

Output format (one position per line, compatible with train_nnue.py):
  <fen> <label>
where label is the win probability for WHITE in [0,1] (1.0/0.5/0.0).

Modes:
  random  -> games played with random moves (fast, low quality)
  net     -> self-play with MCTS using a trained net checkpoint

Usage:
  python -m chessnet.gen_data --mode random --games 500 --out data.txt
  python -m chessnet.gen_data --mode net --checkpoint model.pt --games 200 \
      --mcts-iters 80 --workers 4 --out data.txt
"""

import argparse
import random

import chess
import torch

from .alpha import AlphaTrainer


def _random_game(max_plies=400):
    board = chess.Board()
    history = []
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        history.append(board.copy())
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(random.choice(moves))
        ply += 1
    if board.is_checkmate():
        winner = not board.turn
        white_result = 1.0 if winner == chess.WHITE else -1.0
    else:
        white_result = 0.0
    return [
        (b.fen(), (white_result + 1.0) / 2.0)
        for b in history
    ]


def gen_random(games, out):
    with open(out, "w") as f:
        for g in range(games):
            for fen, label in _random_game():
                f.write(f"{fen} {label}\n")
            if (g + 1) % 100 == 0:
                print(f"{g + 1} games")
    print(f"done -> {out}")


def gen_net(checkpoint, games, mcts_iters, workers, out):
    sd = torch.load(checkpoint, map_location="cpu")
    l0 = sd["fb"].numel()
    l1 = sd["wo.weight"].shape[1]
    feat_count = sd["fw"].shape[0]
    trainer = AlphaTrainer(feat_count=feat_count, l0=l0, l1=l1)
    trainer.net.load_state_dict(sd)
    data = trainer.self_play(games, mcts_iters, mcts_workers=workers)
    with open(out, "w") as f:
        for fen, _visits, label in data:
            f.write(f"{fen} {label}\n")
    print(f"{len(data)} positions -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="random", choices=["random", "net"])
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--out", default="data.txt")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--mcts-iters", type=int, default=80)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    if args.mode == "random":
        gen_random(args.games, args.out)
    else:
        if not args.checkpoint:
            ap.error("--mode net requires --checkpoint model.pt")
        gen_net(args.checkpoint, args.games, args.mcts_iters, args.workers, args.out)


if __name__ == "__main__":
    main()