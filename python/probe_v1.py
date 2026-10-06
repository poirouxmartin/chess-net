"""Tactics probe on distilled v1 (CPU)."""
import os
import torch
from chessnet.resnet import create_model
from chessnet.tactics import CANDIDATES, find_mate, probe
import chessnet.tactics as T

T.DEV = "cpu"
ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../teacher/distill_v1.pt")
model = create_model()
model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
model.to("cpu").eval()
hits = total = 0
for name, fen in CANDIDATES:
    sol = find_mate(fen)
    if not sol:
        print(f"(skip, no M1: {name})")
        continue
    total += 1
    top, w = probe(model, fen)
    ok = bool(top) and top[0][0] == sol
    hits += ok
    print(f"  {name:10} sol={sol:6} top1={top[0] if top else None} W={w} {'HIT' if ok else ''}")
print(f"=> {hits}/{total} mates found by policy top-1")
