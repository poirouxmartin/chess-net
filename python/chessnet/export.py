"""Export a trained NNUE checkpoint to a CSNN file for the Rust engine.

Usage: python -m chessnet.export --checkpoint model.pt --out net.csnn
"""

import argparse

import torch

from .nnue import NNUE, save_csnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feat", type=int, default=2, choices=[1, 2])
    ap.add_argument("--feat-count", type=int, default=None)
    args = ap.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    l0 = state["fb"].numel()
    l1 = state["wo.weight"].shape[1]
    feat_count = state["fw"].shape[0]
    if args.feat_count is not None and args.feat_count != feat_count:
        raise SystemExit(
            f"--feat-count {args.feat_count} mismatches checkpoint ({feat_count})")
    model = NNUE(feat_count, l0, l1)
    model.load_state_dict(state)
    save_csnn(model, args.out, feat=args.feat)
    print(f"wrote {args.out} (l0={l0}, l1={l1}, feat_count={feat_count})")


if __name__ == "__main__":
    main()