"""Async AlphaZero: actors + replay buffer + continuous learner (no cycles).

- MCTS actor PROCESSES loop: reload live weights on change -> play from a
  book start (opening/middlegame/endgame) -> push the finished game.
- Fast actor THREADS share one InferenceServer, same loop with policy games.
- The ingester (main thread) encodes games on CPU into the replay buffer
  while the learner trains on GPU: generation and training overlap.
- The learner runs SGD non-stop, publishes live_model.pt every K steps
  (actors pick it up by mtime), snapshots + dashboard-format metrics every
  snapshot_steps (history.csv rows + loss= log lines, so progress.py and
  dashboard.py work unchanged). Elo/tactics stay manual (elo.py, tactics.py).

Usage:
  py -m chessnet.async_train --out-dir checkpoints_async --max-steps 50000
"""
import argparse
import os
import queue
import random
import re
import threading
import time
import chess
import torch.multiprocessing as mp
from collections import Counter

import torch

from .resnet import create_model
from .replay import ReplayBuffer
from .fastbatch import InferenceServer, play_fast_game
from .train_alpha import (
    play_game, train_step, export_onnx,
)


# ---------------------------------------------------------------------------
# Actors (top-level for spawn pickling)
# ---------------------------------------------------------------------------

def _live_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


def _live_config(path, defaults):
    """Read live hyperparams (static config published by the learner).
    Missing or corrupt file -> defaults. Cheap enough to call once per game."""
    try:
        with open(path) as f:
            import json as _json
            cfg = _json.load(f)
        out = dict(defaults)
        for k in out:
            if k in cfg:
                out[k] = float(cfg[k])
        return out
    except (OSError, ValueError):
        return dict(defaults)


def mcts_actor_loop(wid, live_path, games_q, stop_ev, book, mix,
                    mcts_iters, mcts_batch, c_puct, temp, temp_drop,
                    max_plies, n_cuda, dir_alpha, policy_temp, resign):
    """One MCTS actor: play games forever, pushing
    (fens, policies, hists, result, term) to the queue. Reloads live weights
    whenever the file changes."""
    import torch as _torch
    dev = f"cuda:{wid % n_cuda}" if _torch.cuda.is_available() else "cpu"
    model = create_model().to(dev)
    model.eval()
    rng = random.Random((os.getpid() << 16) ^ wid)
    seen = -1.0
    fails = 0
    cfg_path = os.path.join(os.path.dirname(live_path), "live_config.json")
    cfg0 = {"dirichlet_alpha": dir_alpha, "temp": temp,
            "policy_temp": policy_temp}
    while not stop_ev.is_set():
        mt = _live_mtime(live_path)
        if mt != seen:
            try:
                model.load_state_dict(
                    _torch.load(live_path, map_location=dev, weights_only=True))
                model.eval()
                seen = mt
            except (OSError, RuntimeError):
                pass
        cfg = _live_config(cfg_path, cfg0)
        # AlphaZero: every game from the initial position, no book.
        fen, tag, start_value = chess.STARTING_FEN, "startpos", None
        t = cfg["temp"]
        try:
            history, result, term = play_game(
                model, dev, mcts_iters, mcts_batch, c_puct, t, temp_drop,
                max_plies=max_plies, fen=fen, dir_alpha=cfg["dirichlet_alpha"],
                policy_temp=cfg["policy_temp"], resign=resign)
        except Exception:
            fails += 1
            if fails % 20 == 1:
                print(f"  [mcts-{wid}] {fails} consecutive game failures",
                      flush=True)
            continue
        fails = 0
        fens = [f for f, _, _ in history]
        pols = [p for _, p, _ in history]
        hists = [h for _, _, h in history]
        try:
            games_q.put((fens, pols, hists, result, term), timeout=30)
        except Exception:
            pass  # queue full: actor-global drop counter lives in learner


def fast_actor_loop(server, games_q, stop_ev, book, mix, seed,
                    temp_plies, max_plies, temp, live_config=None,
                    policy_temp=1.0):
    """One fast (policy-only) actor thread sharing the InferenceServer."""
    rng = random.Random(seed)
    fails = 0
    cfg0 = {"fast_temp": temp, "fast_temp_plies": temp_plies,
            "policy_temp": policy_temp}
    while not stop_ev.is_set():
        cfg = _live_config(live_config, cfg0) if live_config else cfg0
        # AlphaZero: every game from the initial position, no book.
        fen = chess.STARTING_FEN
        t = cfg["fast_temp"]
        try:
            fens, ucis, hists, result, term = play_fast_game(
                server, int(cfg["fast_temp_plies"]), max_plies, t, fen=fen,
                policy_temp=cfg["policy_temp"])
        except Exception:
            fails += 1
            if fails % 50 == 1:
                print(f"  [fast-{seed}] {fails} consecutive game failures",
                      flush=True)
            continue
        fails = 0
        pols = [{u: 1.0} for u in ucis]
        try:
            games_q.put((fens, pols, hists, result, term), timeout=30)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    try:
        os.replace(tmp, path)
    except PermissionError:
        pass


def write_live_cfg(path, cfg):
    tmp = path + ".tmp"
    try:
        import json as _json
        with open(tmp, "w") as f:
            _json.dump(cfg, f)
        os.replace(tmp, path)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="checkpoints_async")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.0)
    ap.add_argument("--buffer-cap", type=int, default=500000)
    ap.add_argument("--min-fill", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fast-threads", type=int, default=8)
    ap.add_argument("--infer-batch", type=int, default=512)
    ap.add_argument("--mcts-iters", type=int, default=800)
    ap.add_argument("--mcts-batch", type=int, default=128)
    ap.add_argument("--c-puct", type=float, default=1.4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--temp-drop", type=int, default=30)
    ap.add_argument("--dirichlet-alpha", type=float, default=0.3)
    ap.add_argument("--policy-temp", type=float, default=1.0)
    ap.add_argument("--fast-temp", type=float, default=1.0)
    ap.add_argument("--fast-temp-plies", type=int, default=30)
    ap.add_argument("--max-plies", type=int, default=350)
    ap.add_argument("--resign", action="store_true", default=False)
    ap.add_argument("--live-every", type=int, default=200)
    ap.add_argument("--snapshot-steps", type=int, default=2000)
    args = ap.parse_args()
    if args.buffer_cap < args.min_fill:
        raise SystemExit(
            f"--buffer-cap ({args.buffer_cap}) < --min-fill ({args.min_fill}): "
            f"the learner would wait forever")
    if args.snapshot_steps <= 0 or args.max_steps <= 0 or args.batch_size <= 0:
        raise SystemExit("snapshot-steps, max-steps and batch-size must be > 0")

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")
    # benchmark=False: the inference server sees ever-changing batch shapes
    # (1..infer-batch in 2ms windows); autotuning every new shape costs more
    # than it saves and stalls the shared GPU. Fixed heuristics win here.
    torch.backends.cudnn.benchmark = False

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}", flush=True)
        model = create_model()
        try:
            model.load_state_dict(torch.load(args.resume, map_location=device))
        except RuntimeError as e:
            raise SystemExit(f"--resume incompatible (arch mismatch?): {e}")
        model = model.to(device)
    else:
        model = create_model().to(device)
    # AlphaZero optimizer: SGD + momentum, lr 0.2 with 3 step drops.
    # L2 via weight_decay (paper's c=1e-4).
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay)
    drop_at = {args.max_steps // 2, args.max_steps * 3 // 4,
               args.max_steps * 7 // 8}
    drops_done = 0
    scaler = torch.amp.GradScaler(device) if use_amp else None

    print("Async AlphaZero Training", flush=True)
    print(f"  Device: {device}  AMP: {use_amp}", flush=True)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters",
          flush=True)
    print(f"  MCTS iters: {args.mcts_iters} x{args.workers} actors + "
          f"{args.fast_threads} fast threads", flush=True)
    print(f"  Batch: {args.batch_size}  Entropy: {args.entropy_coef}", flush=True)
    pseudo_total = args.max_steps // args.snapshot_steps
    print(f"Cycles: {pseudo_total} (saving 0001..{pseudo_total:04d})",
          flush=True)

    # AlphaZero: no opening book (dormant file kept for experiments).
    book, mix = None, None
    buf = ReplayBuffer(capacity=args.buffer_cap, min_fill=args.min_fill)
    mx = 0
    try:
        for f in os.listdir(args.out_dir):
            m = re.match(r"model_cycle(\d+)\.pt$", f)
            if m:
                mx = max(mx, int(m.group(1)))
    except OSError:
        pass
    snap_n = mx
    if mx:
        print(f"  Resuming snapshots at {mx + 1:04d}", flush=True)
    ctx = mp.get_context("spawn")
    games_q = ctx.Queue(maxsize=64)
    stop_ev = ctx.Event()

    live_path = os.path.join(args.out_dir, "live_model.pt")
    atomic_save(model.state_dict(), live_path)
    live_cfg_path = os.path.join(args.out_dir, "live_config.json")
    live_cfg = {"entropy_coef": args.entropy_coef,
                "dirichlet_alpha": args.dirichlet_alpha,
                "policy_temp": args.policy_temp,
                "fast_temp": args.fast_temp,
                "fast_temp_plies": args.fast_temp_plies}
    write_live_cfg(live_cfg_path, live_cfg)

    # Fast inference model: separate copy, refreshed from live weights.
    fast_model = create_model().to(device)
    fast_model.load_state_dict(model.state_dict())
    fast_model.eval()
    server = InferenceServer(fast_model, device, max_batch=args.infer_batch)
    server.start()

    actors = []
    for wid in range(args.workers):
        p = ctx.Process(target=mcts_actor_loop, args=(
            wid, live_path, games_q, stop_ev, book, mix,
            args.mcts_iters, args.mcts_batch, args.c_puct, args.temp,
            args.temp_drop, args.max_plies,
            torch.cuda.device_count() if device == "cuda" else 0,
            args.dirichlet_alpha, args.policy_temp, args.resign))
        p.start()
        actors.append(p)
    fast_threads = []
    for i in range(args.fast_threads):
        t = threading.Thread(target=fast_actor_loop, args=(
            server, games_q, stop_ev, book, mix, 1000 + i,
            args.fast_temp_plies, args.max_plies, args.fast_temp,
            live_cfg_path, args.policy_temp), daemon=True)
        t.start()
        fast_threads.append(t)

    hist_path = os.path.join(args.out_dir, "history.csv")
    if not os.path.exists(hist_path):
        with open(hist_path, "w") as hf:
            hf.write("cycle,games,positions,avg_result,loss,vloss,ploss,lr,seconds,terms,timestamp\n")

    steps = 0
    skipped_steps = 0
    drops_done = 0
    dropped_games = 0
    games = terms = 0
    pos_window = 0
    results_sum = 0.0
    loss_sum = vloss_sum = ploss_sum = ent_sum = 0.0
    term_counts: Counter = Counter()
    t0 = time.time()
    # snap_n initialized earlier (auto-detect past checkpoints).

    def snapshot():
        nonlocal snap_n, games, terms, results_sum
        nonlocal loss_sum, vloss_sum, ploss_sum, ent_sum, term_counts, t0
        nonlocal skipped_steps, pos_window
        snap_n += 1
        n = max(1, steps % args.snapshot_steps or args.snapshot_steps)
        lr = optimizer.param_groups[0]["lr"]
        avg_loss = loss_sum / n
        avg_v = vloss_sum / n
        avg_p = ploss_sum / n
        avg_r = results_sum / max(1, terms)
        term_str = " ".join(f"{k}:{v}" for k, v in sorted(term_counts.items()))
        atomic_save(model.state_dict(),
                    os.path.join(args.out_dir, f"model_cycle{snap_n:04d}.pt"))
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(hist_path, "a") as hf:
            hf.write(f"{snap_n},{games},{pos_window},{avg_r:.4f},{avg_loss:.4f},"
                     f"{avg_v:.4f},{avg_p:.4f},{lr:.2e},{time.time()-t0:.0f},"
                     f"\"{term_str}\",{stamp}\n")
        print(f"[snap {snap_n:04d}] steps={steps} games={games} "
              f"loss={avg_loss:.4f} (v={avg_v:.4f} p={avg_p:.4f}) "
              f"lr={lr:.2e} {time.time()-t0:.0f}s [{term_str}]", flush=True)
        if skipped_steps:
            print(f"  (skipped {skipped_steps} non-finite batches this window)",
                  flush=True)
            skipped_steps = 0
        games = terms = 0
        pos_window = 0
        results_sum = 0.0
        loss_sum = vloss_sum = ploss_sum = ent_sum = 0.0
        term_counts = Counter()
        t0 = time.time()

    print("Learner live: waiting for buffer fill...", flush=True)
    last_fill_msg = -1.0
    try:
        while steps < args.max_steps:
            # Ingest finished games (CPU encode, parallel with GPU train).
            while True:
                try:
                    fens, pols, hists, result, term = games_q.get_nowait()
                except queue.Empty:
                    break
                buf.add_game(fens, pols, hists, result)
                games += 1
                terms += 1
                pos_window += len(fens)
                results_sum += result
                term_counts[term] += 1
            if not buf.ready():
                if steps == 0 and time.time() - last_fill_msg > 30:
                    print(f"  fill {len(buf)}/{args.min_fill} "
                          f"({games} games)...", flush=True)
                    last_fill_msg = time.time()
                time.sleep(2.0)
                continue
            x, z, pi, pv, pm = buf.sample(args.batch_size)
            x = x.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            pi = pi.to(device, non_blocking=True)
            pv = pv.to(device, non_blocking=True)
            pm = pm.to(device, non_blocking=True)
            loss, vl, pl, ent, skipped = train_step(
                model, optimizer, x, z, pi, pv, pm, scaler,
                entropy_coef=live_cfg["entropy_coef"])
            if skipped:
                skipped_steps += 1
                if skipped_steps % 10 == 1:
                    print(f"  WARNING: non-finite loss, batch skipped "
                          f"(total {skipped_steps})", flush=True)
                continue
            steps += 1
            # AZ step schedule: lr x0.1 at 1/2, 3/4, 7/8 of max_steps.
            if steps in drop_at and drops_done < 3:
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.1
                drops_done += 1
                print(f"  [lr] drop to {optimizer.param_groups[0]['lr']:.2e} "
                      f"at step {steps}", flush=True)
            loss_sum += loss
            vloss_sum += vl
            ploss_sum += pl
            ent_sum += ent
            if steps % 100 == 0:
                print(f"  step {steps} games={games} buf={len(buf)} "
                      f"loss={loss:.4f} (v={vl:.4f} p={pl:.4f} ent={ent:.3f})",
                      flush=True)
            if steps % args.live_every == 0:
                atomic_save(model.state_dict(), live_path)
                try:
                    with server.mu:
                        fast_model.load_state_dict(model.state_dict())
                        fast_model.eval()
                except RuntimeError:
                    pass
            if steps % args.snapshot_steps == 0:
                snapshot()
                # ONNX export is slow: only every 2000 steps.
                if snap_n % max(1, 2000 // args.snapshot_steps) == 0:
                    try:
                        export_onnx(model, os.path.join(
                            args.out_dir, f"net_cycle{snap_n:04d}.onnx"))
                    except Exception as e:
                        print(f"  ONNX export failed: {e}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted, final snapshot...", flush=True)
    finally:
        stop_ev.set()
        # Don't let the mp.Queue feeder thread hang process exit.
        try:
            games_q.cancel_join_thread()
        except Exception:
            pass
        snapshot()
        try:
            export_onnx(model, os.path.join(
                args.out_dir, f"net_cycle{snap_n:04d}.onnx"))
        except Exception as e:
            print(f"  ONNX export failed: {e}", flush=True)
        server.stop()
        for p in actors:
            p.join(timeout=30)
        # Stragglers (long MCTS games ignoring stop): terminate, never linger
        # as GPU-stealing orphans next to a future run.
        for p in actors:
            if p.is_alive():
                p.terminate()
        print(f"Done. steps={steps} games={games} buffer={len(buf)}", flush=True)


if __name__ == "__main__":
    main()
