"""Elo check between two checkpoints: MCTS vs MCTS at fixed strength.

Both nets play with identical settings; only the weights differ. Colors
alternate. Appends {a, b, games, score, elo} to <out-dir>/elo.json.

Usage: py -m chessnet.elo --a model_cycle0005.pt --b model_cycle0004.pt
       [--games 12] [--iters 150] [--out-dir checkpoints]
"""
import argparse
import json
import math
import os
import time

import chess
import chess.pgn
import torch

from .train_alpha import MCTS, create_model, pick_avoiding_repetition
from .fastbatch import find_mate


def load_net(path, device):
    net = create_model().to(device)
    if path != "random":
        net.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    net.eval()
    return MCTS(net, device)


def play_pair(mcts_a, mcts_b, iters, a_white, max_plies=300, temp_plies=20):
    """One game. Returns (result from A's perspective, plies, termination).
    Also records the move list on the board object for PGN export."""
    board = chess.Board()
    mcts_a.root, mcts_b.root = None, None
    ply = 0
    r, term = None, None
    sans = []
    while not board.is_game_over() and ply < max_plies:
        m = mcts_a if (board.turn == chess.WHITE) == a_white else mcts_b
        mate = find_mate(board)
        if mate is not None:
            move = mate
        else:
            visits = m.search(board, iters)
            if not visits:
                break
            t = 1.0 if ply < temp_plies else 0.0
            move = pick_avoiding_repetition(board, visits, t)
        sans.append(board.san(move))
        board.push(move)
        mcts_a.make_move(move)
        mcts_b.make_move(move)
        ply += 1
    else:
        r, term = None, None
    if r is None and board.is_checkmate():
        white_won = not board.turn
        r = 1.0 if white_won else 0.0
        term = "mate"
    elif board.is_stalemate():
        r, term = 0.5, "stalemate"
    elif board.is_insufficient_material():
        r, term = 0.5, "material"
    elif board.is_repetition(3):
        r, term = 0.5, "repetition"
    elif board.is_fifty_moves():
        r, term = 0.5, "fifty"
    elif board.is_seventyfive_moves() or board.is_fivefold_repetition():
        r, term = 0.5, "auto-draw"
    elif ply >= max_plies:
        r, term = 0.5, "plies-cap"
    else:
        r, term = 0.5, "other-draw"
    return (r if a_white else 1.0 - r), ply, term, sans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--temp-plies", type=int, default=20,
                    help="Sampled opening plies (0 = fully greedy, fair strength)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--fast", action="store_true",
                    help="Policy-only instant games (no MCTS): ~100x faster, "
                         "measures policy sharpness + mating, not search.")
    args = ap.parse_args()
    if args.games <= 0:
        raise SystemExit("--games must be > 0")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.fast:
        from .fastbatch import InferenceServer, play_fast_match
        ma = load_net(args.a, device)
        mb = load_net(args.b, device)
        sa = InferenceServer(ma.net, device)
        sb = InferenceServer(mb.net, device)
        sa.start()
        sb.start()
        mcts_a = mcts_b = None
    else:
        mcts_a = load_net(args.a, device)
        mcts_b = load_net(args.b, device)
        mcts_a.batch_size = args.batch
        mcts_b.batch_size = args.batch
        sa = sb = None

    score = 0.0
    w = d = l = 0
    t0 = time.time()
    for g in range(args.games):
        a_white = (g % 2 == 0)
        if args.fast:
            r, plies, term = play_fast_match(sa, sb, a_white)
        else:
            r, plies, term, sans = play_pair(mcts_a, mcts_b, args.iters, a_white,
                                            temp_plies=args.temp_plies)
            game = chess.pgn.Game()
            game.headers["White"] = os.path.basename(args.a) if a_white else os.path.basename(args.b)
            game.headers["Black"] = os.path.basename(args.b) if a_white else os.path.basename(args.a)
            game.headers["Result"] = "1-0" if r == 1.0 else ("0-1" if r == 0.0 else "1/2-1/2")
            node = game
            bb = chess.Board()
            for s in sans:
                node = node.add_variation(bb.parse_san(s))
                bb.push_san(s)
            pgn_path = os.path.join(args.out_dir, f"elo_game{g+1:02d}.pgn")
            with open(pgn_path, "w") as pf:
                pf.write(str(game))
        score += r
        w, d, l = (w + 1, d, l) if r == 1.0 else ((w, d + 1, l) if r == 0.5 else (w, d, l + 1))
        tag = "Awhite" if a_white else "Ablack"
        res = "1-0" if r == 1.0 else ("0-1" if r == 0.0 else "1/2 ")
        print(f"  game {g+1}/{args.games} {tag} {res} {plies}plies {term} (A:{w}={d}-{l})", flush=True)
    dt = time.time() - t0
    s = score / args.games
    elo = -400.0 * math.log10(1.0 / max(1e-6, min(1 - 1e-6, s)) - 1.0)
    mode = "fast-policy" if args.fast else f"mcts-{args.iters}"
    print(f"A({args.a}) vs B({args.b}) [{mode}]: {w}W {d}D {l}L score={s:.3f} Elo={elo:+.0f} ({dt:.0f}s)")
    if args.fast:
        sa.stop()
        sb.stop()

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, "elo.json")
    try:
        with open(path) as f:
            hist = json.load(f)
    except (OSError, ValueError):
        hist = []
    hist.append({"a": args.a, "b": args.b, "games": args.games,
                 "iters": args.iters, "mode": mode, "score": s, "elo": elo, "time": dt})
    with open(path, "w") as f:
        json.dump(hist, f, indent=1)
    print(f"appended -> {path}")


if __name__ == "__main__":
    main()
