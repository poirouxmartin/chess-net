"""Distill Lc0 teacher into our 20x256 conv net (supervised).

Reads teacher/data/distill_*.npz (fens + hists + pi_idx/pi_val/pi_n + q),
encodes OUR 105 planes on the fly, trains policy CE + value MSE by reusing
train_alpha.train_step. Supervised setting -> AdamW (the SGD recipe is for
the RL loop that follows).

Usage:
  py -m chessnet.distill --data-dir ../../teacher/data --epochs 5 --out ../teacher/distilled.pt
"""
import argparse
import functools
import glob
import os
import time

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .features_lc0 import encode_position
from .resnet import create_model
from .train_alpha import train_step, export_onnx

MAX_MOVES = 128


class DistillDataset(Dataset):
    """Lazy sharded dataset: only an index is pickled to DataLoader workers
    (spawn re-pickles the whole object per worker — preloading GBs breaks
    the pipe). Each worker LRU-caches a few shards locally."""

    def __init__(self, files):
        self.files = list(files)
        counts = []
        for fp in self.files:
            with np.load(fp, allow_pickle=False) as d:
                counts.append(int(d["q"].shape[0]))
        self.cut = np.cumsum(counts)
        self.total = int(self.cut[-1])
        print(f"distill set: {self.total:,} positions from {len(files)} shards",
              flush=True)

    @staticmethod
    @functools.lru_cache(maxsize=2)
    def _shard(path):
        with np.load(path, allow_pickle=False) as d:
            return {k: np.asarray(d[k]) for k in
                    ("fens", "hists", "pi_idx", "pi_val", "pi_n", "q")}

    def __len__(self):
        return self.total

    def shard_ranges(self):
        """(start, end) global row ranges per shard (for shard-local batches)."""
        out = []
        prev = 0
        for c in self.cut:
            out.append((prev, int(c)))
            prev = int(c)
        return out

    def __getitem__(self, i):
        fi = int(np.searchsorted(self.cut, i, side="right"))
        row = i - (int(self.cut[fi - 1]) if fi else 0)
        s = self._shard(self.files[fi])
        f = str(s["fens"][row])
        h = [str(x) for x in s["hists"][row] if x]
        b = chess.Board(f)
        past = [chess.Board(x) for x in h[-7:]]
        x = encode_position(b, past)
        k = int(s["pi_n"][row])
        return (x.astype(np.float32), np.float32(s["q"][row]),
                np.asarray(s["pi_idx"][row], dtype=np.int64),
                np.asarray(s["pi_val"][row], dtype=np.float32), k)


class ShardSampler:
    """Batches drawn from ONE shard at a time (shard order + within-shard
    order shuffled per epoch). Global shuffle would thrash the workers'
    small shard cache (106 shards): every fetch would evict+reload ~115MB."""

    def __init__(self, ranges, batch_size, seed):
        self.ranges = ranges
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(self.ranges))
        for r in order:
            s, e = self.ranges[r]
            idx = np.arange(s, e)
            rng.shuffle(idx)
            for k in range(0, len(idx), self.batch_size):
                yield idx[k:k + self.batch_size].tolist()

    def __len__(self):
        return sum((e - s + self.batch_size - 1) // self.batch_size
                   for s, e in self.ranges)


def collate(batch):
    xs = np.stack([b[0] for b in batch])
    z = np.array([b[1] for b in batch], dtype=np.float32)
    ii = np.stack([b[2] for b in batch])
    vv = np.stack([b[3] for b in batch])
    kk = np.array([b[4] for b in batch])
    mask = np.zeros_like(vv, dtype=bool)
    for i, k in enumerate(kk):
        mask[i, :k] = True
    return (torch.from_numpy(xs), torch.from_numpy(z),
            torch.from_numpy(ii), torch.from_numpy(vv),
            torch.from_numpy(mask))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../teacher/data",
                    help="Comma-separated shard dirs (e.g. self-play + elite).")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="../../teacher/distilled.pt")
    ap.add_argument("--val-frac", type=float, default=0.02)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    files = []
    for dd in args.data_dir.split(","):
        files.extend(sorted(glob.glob(os.path.join(base, dd.strip(), "distill_*.npz"))))
    assert files, "no distill shards (run query_teacher first)"
    # Split by whole shards (no leakage, keeps shard-locality for workers).
    nva = max(1, int(round(len(files) * args.val_frac)))
    tr_files, va_files = files[:-nva] or files, files[-nva:]
    tr_ds = DistillDataset(tr_files)
    va_ds = DistillDataset(va_files)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    ldk = dict(num_workers=args.num_workers,
               pin_memory=(device == "cuda"),
               persistent_workers=(args.num_workers > 0))
    va_loader = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate, **ldk)

    model = create_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    nbatches = sum((e - s + args.batch_size - 1) // args.batch_size
                   for s, e in tr_ds.shard_ranges())
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * max(1, nbatches))
    scaler = torch.amp.GradScaler(device) if use_amp else None

    out_path = os.path.join(base, args.out)
    best = float("inf")
    for ep in range(args.epochs):
        t0 = time.time()
        model.train()
        tl = tv = tp = nb = 0.0
        tr_loader = DataLoader(
            tr_ds, collate_fn=collate,
            batch_sampler=ShardSampler(tr_ds.shard_ranges(), args.batch_size,
                                       seed=1000 + ep), **ldk)
        for x, z, ii, vv, mm in tr_loader:
            x, z, ii, vv, mm = (t.to(device, non_blocking=True)
                                for t in (x, z, ii, vv, mm))
            loss, vl, pl, _, skipped = train_step(
                model, opt, x, z, ii, vv, mm, scaler, entropy_coef=0.0)
            if skipped:
                continue
            sched.step()
            tl += loss
            tv += vl
            tp += pl
            nb += 1
        # Validation (no grad).
        model.eval()
        vl_sum = pl_sum = nv_b = 0.0
        with torch.no_grad():
            for x, z, ii, vv, mm in va_loader:
                x, z, ii, vv, mm = (t.to(device) for t in (x, z, ii, vv, mm))
                pl_, v_ = model(x)
                vl_sum += F.mse_loss(v_.squeeze(1), z * 2.0 - 1.0).item()
                lp = F.log_softmax(pl_, dim=1).gather(1, ii.masked_fill(~mm, 0))
                vv2 = vv.masked_fill(~mm, 0.0)
                pl_sum += -(vv2 * lp).sum(dim=1).mean().item()
                nv_b += 1
        vloss, ploss = vl_sum / max(1, nv_b), pl_sum / max(1, nv_b)
        tot = vloss + ploss
        print(f"epoch {ep+1}/{args.epochs} train={tl/max(1,nb):.4f} "
              f"(v={tv/max(1,nb):.4f} p={tp/max(1,nb):.4f}) "
              f"val={tot:.4f} (v={vloss:.4f} p={ploss:.4f}) "
              f"lr={opt.param_groups[0]['lr']:.2e} {time.time()-t0:.0f}s",
              flush=True)
        if tot < best:
            best = tot
            torch.save(model.state_dict(), out_path)
            print(f"  saved {out_path}", flush=True)
    export_onnx(model, os.path.join(os.path.dirname(out_path), "distilled.onnx"))
    print(f"DONE best val={best:.4f}")


if __name__ == "__main__":
    main()
