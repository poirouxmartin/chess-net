"""AlphaZero-style self-play training.

Usage:
  python -m chessnet.train_alpha --cycles 5 --games 40 --mcts-iters 100
                                 --out model.pt --csnn net.csnn
"""

import argparse
import os

import torch

from .alpha import AlphaTrainer
from .nnue import save_csnn
from . import features as F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=5)
    ap.add_argument("--games", type=int, default=40, help="self-play games per cycle")
    ap.add_argument("--mcts-iters", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--feat", type=int, default=2, choices=[1, 2])
    ap.add_argument("--l0", type=int, default=256)
    ap.add_argument("--l1", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--csnn", default="net.csnn")
    args = ap.parse_args()

    feat_count = F.KP768_FEAT_COUNT if args.feat == 2 else F.HALFKP_FEAT_COUNT
    trainer = AlphaTrainer(feat_count=feat_count, l0=args.l0, l1=args.l1)
    print(f"device: {trainer.device}")

    for cycle in range(args.cycles):
        print(f"cycle {cycle + 1}/{args.cycles}: self-play {args.games} games...")
        data = trainer.self_play(args.games, args.mcts_iters)
        trainer.train(data, epochs=args.epochs, batch_size=args.batch_size)
        trainer.net = trainer.net.cpu()
        torch.save(trainer.net.state_dict(), args.out)
        save_csnn(trainer.net, args.csnn, feat=args.feat)
        trainer.net = trainer.net.to(trainer.device)
        print(f"  {len(data)} positions -> saved {args.out} and {args.csnn}")


if __name__ == "__main__":
    main()