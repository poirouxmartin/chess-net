"""Supervised training utility for the shared ResNet (105 planes, 4288 policy,
tanh scalar value in STM frame).

Usage:
  python -m chessnet.train_resnet --data data.txt --out model.pt --onnx net.onnx

Dataset format (one position per line):
  <fen> <label>
where label is win probability for WHITE in [0,1]. Converted to the
side-to-move frame inside __getitem__ (the net speaks STM).

NOTE: value-head only (the dataset carries no policy targets). For the
live AlphaZero loop use async_train.py. Export is ONNX (the Rust engine
reads ONNX; the old CSNN v1/v2 MLP format is unrelated to this ResNet).
"""

import argparse
import os
import time

import chess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .features_lc0 import encode_position, NUM_PLANES
from .resnet import create_model


class ChessDataset(Dataset):
    """Memory-mapped chess position dataset."""

    def __init__(self, path, max_positions=None):
        self.fens = []
        self.labels = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_positions and i >= max_positions:
                    break
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit(" ", 1)
                if len(parts) != 2:
                    continue
                try:
                    # Validate FEN
                    chess.Board(parts[0])
                    self.fens.append(parts[0])
                    self.labels.append(float(parts[1]))
                except (ValueError, chess.InvalidMoveError):
                    continue

        print(f"Loaded {len(self.fens):,} positions from {path}")

    def __len__(self):
        return len(self.fens)

    def __getitem__(self, idx):
        board = chess.Board(self.fens[idx])
        planes = encode_position(board)  # [105, 8, 8]
        # Labels are white-POV; the net speaks side-to-move.
        label = self.labels[idx]
        if board.turn == chess.BLACK:
            label = 1.0 - label
        return planes, label


def collate_fn(batch):
    """Stack planes and labels into batches."""
    planes = np.stack([b[0] for b in batch], axis=0)
    labels = np.array([b[1] for b in batch], dtype=np.float32)
    return (
        torch.from_numpy(planes).float(),
        torch.from_numpy(labels).float(),
    )


def _stm_targets(labels):
    """STM win-probabilities in [0,1] -> tanh-frame targets in [-1,1].

    NOTE: this tool trains the VALUE head only; the policy head never gets
    a gradient here (the dataset carries no policy targets). For policy
    training use the self-play loop (train_alpha/async_train).
    """
    return 2.0 * labels - 1.0


def train_epoch(model, loader, optimizer, scaler, device, use_amp):
    """One training epoch with optional AMP."""
    model.train()
    total_loss = 0.0
    total_batches = 0

    for planes, labels in loader:
        planes = planes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                _, v = model(planes)
                # MSE on the tanh scalar (AlphaZero value loss).
                loss = F.mse_loss(v.squeeze(1), _stm_targets(labels))
        else:
            _, v = model(planes)
            loss = F.mse_loss(v.squeeze(1), _stm_targets(labels))

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    return total_loss / max(1, total_batches)


@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    """Evaluate accuracy and loss."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for planes, labels in loader:
        planes = planes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                _, v = model(planes)
                loss = F.mse_loss(v.squeeze(1), _stm_targets(labels))
        else:
            _, v = model(planes)
            loss = F.mse_loss(v.squeeze(1), _stm_targets(labels))

        total_loss += loss.item()
        # Sign agreement: predicted winner == labelled winner (draws excluded).
        pred_win = v.squeeze(1) > 0
        true_win = labels > 0.5
        true_loss = labels < 0.5
        correct += ((pred_win) & (true_win)).sum().item()
        correct += ((~pred_win) & (true_loss)).sum().item()
        total += (true_win | true_loss).sum().item()

    accuracy = correct / max(1, total)
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, accuracy


def export_onnx(model, path, channels=224, fp16=False):
    """Export model to ONNX format for Rust inference.

    Outputs:
      - policy_logits: [B, 4288] move logits (untrained here: value-only tool)
      - value:  [B, 1] tanh scalar, side-to-move frame
    """
    import onnx
    from onnxconverter_common import float16

    model = model.cpu().eval()
    dummy = torch.randn(1, NUM_PLANES, 8, 8)

    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["planes"],
        output_names=["policy_logits", "value"],
        dynamic_axes={
            "planes": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )

    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)

    if fp16:
        onnx_model = float16.convert_float_to_float16(
            onnx_model,
            keep_io_types=True,
            min_positive_val=1e-7,
            max_finite_val=1e4,
        )
        onnx.save(onnx_model, path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"Exported ONNX ({'FP16' if fp16 else 'FP32'}): {path} ({size_mb:.1f} MB)")

    _benchmark_onnx(path, fp16=fp16)


def _benchmark_onnx(path, fp16=False):
    """Benchmark ONNX inference speed."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  (onnxruntime not installed, skipping benchmark)")
        return

    providers = []
    if "CUDAExecutionProvider" in ort.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    sess = ort.InferenceSession(path, opts, providers=providers)
    provider = sess.get_providers()[0]
    print(f"  Provider: {provider}")

    # Warmup
    dummy = np.random.randn(32, NUM_PLANES, 8, 8).astype(np.float32)
    for _ in range(10):
        sess.run(None, {"planes": dummy})

    # Benchmark
    import time
    N = 100
    t0 = time.time()
    for _ in range(N):
        sess.run(None, {"planes": dummy})
    dt = time.time() - t0
    print(f"  Batch 32: {dt/N*1000:.1f}ms/batch, {32*N/dt:.0f} positions/s")


# NOTE: the old CSNN-v3 ResNet exporter was removed here. Nothing reads that
# format (the Rust engine loads CSNN v1/v2 MLP nets and ONNX ResNets);
# supervised ResNets export via export_onnx() above.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset file (fen label)")
    ap.add_argument("--out", default="model.pt", help="PyTorch checkpoint")
    ap.add_argument("--onnx", default=None, help="ONNX export path (also exports _fp16.onnx)")
    ap.add_argument("--channels", type=int, default=224)
    ap.add_argument("--blocks", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-every", type=int, default=5)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device == "cuda"

    if use_amp:
        torch.backends.cudnn.benchmark = True

    print(f"Device: {device}")
    print(f"AMP: {use_amp}")

    # Load dataset
    dataset = ChessDataset(args.data, max_positions=args.max_positions)
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(
        train_set, shuffle=True, collate_fn=collate_fn, **loader_kwargs
    )
    val_loader = DataLoader(
        val_set, shuffle=False, collate_fn=collate_fn, **loader_kwargs
    )

    # Create model
    model = create_model(channels=args.channels, num_blocks=args.blocks).to(device)

    # Optimizer + scaler (AlphaZero-style SGD + momentum; step drops below).
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay
    )
    drop_epochs = {args.epochs // 2, args.epochs * 3 // 4,
                   args.epochs * 7 // 8}
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    print(f"\nTraining:")
    print(f"  Train: {train_size:,} positions")
    print(f"  Val:   {val_size:,} positions")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  LR: {args.lr}")
    print(f"  Workers: {args.num_workers}")
    print()

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        val_loss, val_acc = evaluate(model, val_loader, device, use_amp)
        if (epoch + 1) in drop_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1
        dt = time.time() - t0

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch+1:3d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"acc {val_acc:.3f} | "
            f"lr {lr:.2e} | "
            f"{dt:.1f}s"
        )

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0 or val_loss < best_val_loss:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            torch.save(model.state_dict(), args.out)
            print(f"  saved {args.out}")

    # Export to ONNX (FP32 + FP16)
    if args.onnx:
        export_onnx(model, args.onnx, args.channels, fp16=False)
        export_onnx(model, args.onnx.replace(".onnx", "_fp16.onnx"), args.channels, fp16=True)

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
