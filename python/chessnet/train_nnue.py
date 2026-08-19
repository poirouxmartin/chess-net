"""Supervised NNUE training.

Dataset format (one position per line):
  <fen> <label>
where label is the raw target for the net output:
  --loss bce -> win probability for WHITE, in [0,1]
  --loss mse -> score in centipawns / 400 (logit*400 == centipawns)

Example lines:
  rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 0.5
  ... b - - 0 12 1.25

Usage:
  python -m chessnet.train_nnue --data data.txt --out model.pt --csnn net.csnn
"""

import argparse
import math

import chess
import torch
from torch.utils.data import DataLoader, TensorDataset

from .nnue import NNUE, encode_batch, save_csnn
from . import features as F


def load_dataset(path):
    boards, labels = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.rsplit(" ", 1)
            boards.append(chess.Board(parts[0]))
            labels.append(float(parts[1]))
    return boards, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--csnn", default="net.csnn")
    ap.add_argument("--feat", type=int, default=2, choices=[1, 2])
    ap.add_argument("--l0", type=int, default=256)
    ap.add_argument("--l1", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", default="bce", choices=["bce", "mse"])
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    feat_count = F.KP768_FEAT_COUNT if args.feat == 2 else F.HALFKP_FEAT_COUNT
    model = NNUE(feat_count, args.l0, args.l1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    boards, labels = load_dataset(args.data)
    idx, mask = encode_batch(boards, args.feat)
    y = torch.tensor(labels, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(idx, mask, y), batch_size=args.batch_size, shuffle=True
    )

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for bi, bm, by in loader:
            bi, bm, by = bi.to(device), bm.to(device), by.to(device)
            logit = model(bi, bm)
            if args.loss == "bce":
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, by)
            else:
                loss = torch.nn.functional.mse_loss(logit, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"epoch {epoch + 1}/{args.epochs} loss {total / len(loader):.4f}")

    model = model.cpu()
    torch.save(model.state_dict(), args.out)
    save_csnn(model, args.csnn, feat=args.feat)
    print(f"saved {args.out} and {args.csnn}")


if __name__ == "__main__":
    main()