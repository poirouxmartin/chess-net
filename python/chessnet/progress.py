"""Live training dashboard: parse the training log + checkpoints, print a
table and save progress curves (loss, result, Elo if measured).

Usage: py -m chessnet.progress [--log training_big.log] [--out-dir checkpoints]
"""
import argparse
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(path):
    cycles = []
    cur = {}
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return cycles
    # Flatten \r progress lines (keep last epoch of each training phase).
    text = re.sub(r"[^\n]*\r", "", text)
    for line in text.splitlines():
        m = re.search(r"Cycle (\d+)/\d+ \| Self-play \((\d+) games\)\.\.\. ([\d,]+) positions, avg_result=([\d.]+)(?: \[(.*)\])?", line)
        if m:
            terms = {}
            if m.group(5):
                for kv in m.group(5).split(","):
                    if ":" in kv:
                        k, v = kv.strip().split(":", 1)
                        try:
                            terms[k.strip()] = int(v)
                        except ValueError:
                            pass
            cur = {"cycle": int(m.group(1)), "games": int(m.group(2)),
                   "positions": int(m.group(3).replace(",", "")),
                   "avg_result": float(m.group(4)), "terms": terms}
            continue
        m = re.search(r"loss=([\d.]+) \(v=([\d.]+) p=([\d.]+)\) lr=([\deE.+-]+) ([\d.]+)s", line)
        if m and cur:
            cur.update({"loss": float(m.group(1)), "vloss": float(m.group(2)),
                        "ploss": float(m.group(3)), "lr": float(m.group(4)),
                        "seconds": float(m.group(5))})
            cycles.append(cur)
            cur = {}
    return cycles


def parse_steps(path):
    """Dense live points from async heartbeats:
    `step N games=G buf=B loss=X (v=Y p=Z ent=W)`. One point per ~100 steps,
    so the dashboard shows learning live between snapshots."""
    pts = []
    try:
        with open(path, errors="replace") as f:
            text = f.read()
    except OSError:
        return pts
    for line in text.splitlines():
        m = re.search(r"step (\d+) games=(\d+) buf=(\d+) loss=([\d.eE+-]+) "
                      r"\(v=([\d.eE+-]+) p=([\d.eE+-]+) ent=([\d.eE+-]+)\)", line)
        if m:
            try:
                pts.append({"step": int(m.group(1)), "games": int(m.group(2)),
                            "buf": int(m.group(3)), "loss": float(m.group(4)),
                            "vloss": float(m.group(5)), "ploss": float(m.group(6)),
                            "ent": float(m.group(7))})
            except ValueError:
                pass
    return pts


def load_elo(out_dir):
    try:
        with open(os.path.join(out_dir, "elo.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="training_big.log")
    ap.add_argument("--out-dir", default="checkpoints")
    args = ap.parse_args()

    cycles = parse_log(args.log)
    print(f"=== training progress ({args.log}) ===")
    print(f"{'cyc':>4} {'games':>6} {'pos':>7} {'result':>7} {'loss':>8} {'vloss':>8} {'ploss':>8} {'time':>8}")
    for c in cycles[-25:]:
        print(f"{c['cycle']:>4} {c.get('games', 0):>6} {c.get('positions', 0):>7} "
              f"{c.get('avg_result', float('nan')):>7.3f} {c.get('loss', float('nan')):>8.4f} "
              f"{c.get('vloss', float('nan')):>8.4f} {c.get('ploss', float('nan')):>8.4f} "
              f"{c.get('seconds', 0):>7.0f}s")

    elos = load_elo(args.out_dir)
    if elos:
        print("\n=== Elo (new vs ref) ===")
        print(f"{'new':>22} {'ref':>22} {'games':>6} {'score':>6} {'elo':>8}")
        for e in elos[-15:]:
            a = os.path.basename(e.get("a", "?"))[-22:]
            b = os.path.basename(e.get("b", "?"))[-22:]
            print(f"{a:>22} {b:>22} {e.get('games', 0):>6} {e.get('score', 0):>6.3f} {e.get('elo', 0):>+8.0f}")

    if not cycles:
        print("(no completed cycles yet)")
        return

    xs = [c["cycle"] for c in cycles]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(xs, [c["loss"] for c in cycles], "o-", label="loss")
    ax1.plot(xs, [c["vloss"] for c in cycles], "s-", label="value")
    ax1.plot(xs, [c["ploss"] for c in cycles], "^-", label="policy")
    ax1.set_ylabel("loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.plot(xs, [c["avg_result"] for c in cycles], "o-", color="green")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel("avg_result (self-play)")
    ax2.set_xlabel("cycle")
    ax2.grid(True, alpha=0.3)
    if elos:
        ax3 = ax2.twinx()
        ex = list(range(1, len(elos) + 1))
        ax3.plot(ex, [e["elo"] for e in elos], "D-", color="orange", label="Elo")
        ax3.set_ylabel("Elo (new vs ref)", color="orange")
    fig.tight_layout()
    out = os.path.join(args.out_dir, "training_progress.png")
    fig.savefig(out, dpi=100)
    print(f"\ncurves -> {out}")


if __name__ == "__main__":
    main()
