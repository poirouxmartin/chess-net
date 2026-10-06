"""Build lc0_policy.json (4 uci->idx dicts) from the reference implementation,
cross-checked against kMoveStrs extracted from lc0 encoder.cc."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../teacher"))
import leela_board as ref

base = os.path.join(HERE, "../teacher")
mine = json.load(open(os.path.join(base, "lc0_moves.json")))
print("lc0_moves.json:", len(mine))

# Reference base table must equal encoder.cc ground truth.
assert ref._idx_to_move_wn == mine, "reference table != encoder.cc table"
print("reference base == encoder.cc: OK (1858)")

names = ["wn", "wc", "bn", "bc"]
out = {}
for name, d in zip(names, ref.uci_to_idx):
    out[name] = d
    print(name, len(d))
json.dump(out, open(os.path.join(base, "lc0_policy.json"), "w"))
# Spot checks.
assert out["wn"]["e2e4"] == mine.index("e2e4")
assert out["bn"]["e7e5"] == mine.index("e2e4"), "black rank-flip symmetry"
assert out["wc"]["e1g1"] == mine.index("e1h1"), "castle swap"
assert out["bc"]["e8c8"] == mine.index("e1a1"), "castle swap (flipped)"
print("spot checks OK")
