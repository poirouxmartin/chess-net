"""Fast batched self-play: many threaded policy-only games sharing one GPU
inference server (dynamic batching). Throughput target: 10+ games/s on a
single GPU (vs ~0.03/s for full MCTS) — the volume lever for training.

Policy targets are HARD (played move index); value targets are game results
(white-POV; converted to the STM frame on ingest in replay.py). Used for
bulk data and instant Elo matches (elo --fast).
"""
import math
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import numpy as np
import torch

from .features_lc0 import encode_position
from .resnet import policy_index


class InferenceServer:
    """One GPU, many games: worker threads enqueue single positions, the
    server forwards them in large batches and scatters results."""

    def __init__(self, net, device, max_batch=512, timeout=0.002):
        self.net = net
        self.device = device
        self.max_batch = max_batch
        self.timeout = timeout
        # Guards forward vs live weight refresh (torn reads -> NaNs).
        self.mu = threading.Lock()
        self.q: "queue.Queue" = queue.Queue()
        self._stop = False
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()

    def stop(self):
        self._stop = True

    def infer(self, planes):
        """planes: np [105,8,8] float32. Returns (policy[P], value scalar)."""
        box = {}
        ev = threading.Event()
        self.q.put((planes, box, ev))
        ev.wait()
        if "exc" in box:
            raise box["exc"]
        return box["pol"], box["val"]

    def _loop(self):
        self.net.eval()
        with torch.no_grad():
            while not self._stop:
                try:
                    first = self.q.get(timeout=0.05)
                except queue.Empty:
                    continue
                batch = [first]
                deadline = time.perf_counter() + self.timeout
                while len(batch) < self.max_batch:
                    try:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        batch.append(self.q.get(timeout=remaining))
                    except queue.Empty:
                        break
                try:
                    x = torch.stack(
                        [torch.from_numpy(p) for p, _, _ in batch]
                    ).to(self.device, non_blocking=True).float()
                    with self.mu:
                        self.net.eval()
                        pl, vl = self.net(x)
                    pls = pl.float().cpu().numpy()
                    vls = vl.float().cpu().numpy()
                    for (_, box, _ev), p, v in zip(batch, pls, vls):
                        box["pol"], box["val"] = p, v
                except Exception as e:  # noqa - must release waiters
                    for _, box, _ev in batch:
                        box["exc"] = e
                finally:
                    for _, _, ev in batch:
                        ev.set()


def pick_policy_move(board, probs, temp):
    """Sample (temp>0) or greedy (temp<=0) from per-legal-move probs dict,
    refusing 3rd repetitions unless forced, and NEVER walking into an
    opponent mate-in-1 (exact check on the chosen move only)."""
    cands = dict(probs)
    while cands:
        if temp > 0 and len(cands) > 1:
            tot = sum(math.pow(v, 1.0 / temp) for v in cands.values())
            r = random.random() * tot
            acc, mv = 0.0, None
            for m, v in cands.items():
                acc += math.pow(v, 1.0 / temp)
                if r <= acc:
                    mv = m
                    break
            mv = mv if mv is not None else max(cands, key=cands.get)
        else:
            mv = max(cands, key=cands.get)
        if len(cands) > 1 and allows_opp_mate(board, mv):
            del cands[mv]
            continue
        board.push(mv)
        bad = board.is_repetition(3)
        board.pop()
        if not bad or len(cands) == 1:
            return mv
        del cands[mv]
    return max(probs, key=probs.get)


def find_mate(board):
    """Immediate mate (1-ply exact check, movegen only, zero GPU)."""
    for mv in board.legal_moves:
        board.push(mv)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return mv
    return None


def allows_opp_mate(board, mv):
    """True if playing mv lets the opponent mate in 1 (exact 1-ply check).
    The PUCT-minimax guarantee needs full width or infinite budget; at
    finite budgets a low-prior mating reply is never visited, so the search
    happily walks into M1. This exact check closes the hole for free."""
    board.push(mv)
    try:
        for reply in board.legal_moves:
            board.push(reply)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return True
        return False
    finally:
        board.pop()


def play_fast_game(server, temp_plies=30, max_plies=350, temp=1.0, fen=None,
                   policy_temp=1.0):
    """One policy-only game. Returns (fens, ucis, hists, result, term):
    hists[i] = FENs of previous game positions (oldest first, up to 7)."""
    board = chess.Board(fen) if fen else chess.Board()
    fens, ucis, hists = [], [], []
    game_fens, game_boards = [], []
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        mv = find_mate(board)
        if mv is None:
            pol, _val = server.infer(encode_position(board, game_boards[-7:]))
            legal = list(board.legal_moves)
            if not legal:
                break
            idx = np.array([policy_index(board, m) for m in legal])
            logits = pol[idx] / max(0.05, policy_temp)
            mx = logits.max()
            probs = np.exp(logits - mx)
            probs /= probs.sum()
            t = temp if ply < temp_plies else 0.0
            mv = pick_policy_move(board, dict(zip(legal, probs.tolist())), t)
        fens.append(board.fen())
        ucis.append(mv.uci())
        hists.append(tuple(game_fens[-7:]))
        game_fens.append(board.fen())
        game_boards.append(board.copy())
        board.push(mv)
        ply += 1
    if board.is_checkmate():
        result, term = (1.0, "mate") if not board.turn else (0.0, "mate")
    elif board.is_stalemate():
        result, term = 0.5, "stalemate"
    elif board.is_insufficient_material():
        result, term = 0.5, "material"
    elif board.is_repetition(3):
        result, term = 0.5, "repetition"
    elif board.is_fifty_moves():
        result, term = 0.5, "fifty"
    elif board.is_seventyfive_moves() or board.is_fivefold_repetition():
        result, term = 0.5, "auto-draw"
    elif ply >= max_plies:
        result, term = 0.5, "plies-cap"
    else:
        result, term = 0.5, "other-draw"
    return fens, ucis, hists, result, term


def play_fast_batch(server, games, temp_plies=30, max_plies=350, temp=1.0,
                    threads=8, progress=None, fen_fn=None, policy_temp=1.0):
    """Play `games` policy games on `threads` threads sharing `server`.
    fen_fn: optional callable returning a starting FEN per game.
    Returns (fens, ucis, hists, results_per_position, terms_per_game)."""
    all_fens, all_uci, all_hist, all_res, terms = [], [], [], [], []
    done = [0]

    def one(_):
        fen = fen_fn() if fen_fn else None
        f, u, h, r, t = play_fast_game(server, temp_plies, max_plies, temp,
                                       fen=fen, policy_temp=policy_temp)
        return f, u, h, r, t

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for f, u, h, r, t in pool.map(one, range(games)):
            all_fens.extend(f)
            all_uci.extend(u)
            all_hist.extend(h)
            all_res.extend([r] * len(f))
            terms.append(t)
            done[0] += 1
            if progress:
                progress(done[0], games)
    return all_fens, all_uci, all_hist, all_res, terms


def play_fast_match(server_a, server_b, a_white=True, temp_plies=30,
                    max_plies=350, temp=1.0):
    """One policy game between two nets (for fast Elo). Returns
    (result from A's perspective, plies, termination)."""
    board = chess.Board()
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        server = server_a if (board.turn == chess.WHITE) == a_white else server_b
        mv = find_mate(board)
        if mv is None:
            pol, _val = server.infer(encode_position(board))
            legal = list(board.legal_moves)
            if not legal:
                break
            idx = np.array([policy_index(board, m) for m in legal])
            logits = pol[idx]
            mx = logits.max()
            probs = np.exp(logits - mx)
            probs /= probs.sum()
            t = temp if ply < temp_plies else 0.0
            mv = pick_policy_move(board, dict(zip(legal, probs.tolist())), t)
        board.push(mv)
        ply += 1
    if board.is_checkmate():
        r = 1.0 if not board.turn else 0.0
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
    return (r if a_white else 1.0 - r), ply, term
