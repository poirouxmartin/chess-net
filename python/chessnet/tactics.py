"""Tactics probe: mate-in-1 suite, solutions verified programmatically.
Policy top-1 == mating move ? + tanh scalar (STM frame). Seconds to run.

Usage: py -m chessnet.tactics
"""
import json
import os
import sys

import chess
import numpy as np
import torch
import torch.nn.functional as F

from .train_alpha import create_model
from .features_lc0 import encode_position
from .resnet import policy_index

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# (name, fen, side-to-move mates with...) — solution found by search below.
CANDIDATES = [
    ("back-rank", "7k/6pp/8/8/8/8/8/R6K w - - 0 1"),
    ("scholar", "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"),
    ("fool", "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 1"),
    # Hand-built mates (loader keeps only real M1s — see skip messages).
    ("back-rank-2", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1"),
    ("qd8", "6k1/5ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1"),
    ("arabian-2", "k7/8/2N5/8/8/8/8/R6K w - - 0 1"),
    ("promo", "7k/1P4pp/8/8/8/8/5PPP/6K1 w - - 0 1"),
    ("black-back-2", "r6k/6pp/8/8/8/8/5PPP/6K1 b - - 0 1"),
]


def find_mate(fen):
    """Return UCI of a mating move if one exists, else None."""
    b = chess.Board(fen)
    for mv in b.legal_moves:
        b.push(mv)
        if b.is_checkmate():
            b.pop()
            return mv.uci()
        b.pop()
    return None


@torch.no_grad()
def probe(model, fen):
    b = chess.Board(fen)
    x = torch.tensor(np.stack([encode_position(b)]), dtype=torch.float32, device=DEV)
    pl, v = model(x)
    legal = list(b.legal_moves)
    idx = torch.tensor([policy_index(b, m) for m in legal])
    probs = F.softmax(pl[0][idx], dim=0).cpu().numpy()
    # tanh scalar [-1,1] STM frame -> white POV.
    t = v[0].item()
    w = (t + 1.0) / 2.0 if b.turn == chess.WHITE else 1.0 - (t + 1.0) / 2.0
    top = sorted(zip(legal, probs), key=lambda t: -t[1])[:3]
    return ([(m.uci(), round(float(p), 3)) for m, p in top], round(w, 3))


def main():
    suite = []
    for name, fen in CANDIDATES:
        sol = find_mate(fen)
        if sol:
            suite.append((name, fen, sol))
        else:
            print(f"(skip, no M1: {name})")
    print(f"suite: {len(suite)} verified M1")
    ckpt_dir = os.path.join(os.path.dirname(__file__), "../../checkpoints")
    import re as _re
    cycles = []
    try:
        for f in os.listdir(ckpt_dir):
            m = _re.match(r"model_cycle(\d+)\.pt$", f)
            if m:
                cycles.append(int(m.group(1)))
    except OSError:
        pass
    cycles.sort()
    nets = {"random": None}
    for c in cycles[-2:]:
        nets[f"cycle{c:04d}"] = os.path.join(ckpt_dir, f"model_cycle{c:04d}.pt")
    results = {}
    for tag, path in nets.items():
        model = create_model().to(DEV)
        if path:
            model.load_state_dict(torch.load(path, map_location=DEV, weights_only=True))
        model.eval()
        hits = 0
        details = []
        print(f"--- {tag} ---")
        for name, fen, sol in suite:
            top, w = probe(model, fen)
            ok = top and top[0][0] == sol
            hits += bool(ok)
            details.append({"pos": name, "sol": sol,
                            "top1": top[0][0] if top else None,
                            "top1p": top[0][1] if top else 0.0,
                            "white": w,
                            "hit": bool(ok)})
            print(f"  {name:10} sol={sol:6} top1={top[0] if top else None} W={w} {'HIT' if ok else ''}")
        print(f"  => {hits}/{len(suite)} mates found by policy top-1")
        results[tag] = {"hits": hits, "total": len(suite), "details": details}
    with open(os.path.join(os.path.dirname(__file__), "../../checkpoints/tactics.json"), "w") as f:
        json.dump({"nets": results}, f, indent=1)
    print("saved -> checkpoints/tactics.json")


if __name__ == "__main__":
    main()
