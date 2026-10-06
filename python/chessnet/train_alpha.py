"""Pure AlphaZero self-play training loop.

NOTE (legacy): the active runner is async_train.py (continuous actors +
replay, no cycles). This sync cycle-based loop is kept for smoke tests
and small experiments.

No heuristics, no PeSTO, no supervised data. Everything from self-play.

Usage:
  python -m chessnet.train_alpha --cycles 100 --games 500 --mcts-iters 800

Pipeline per cycle:
  1. Self-play: current model plays games via MCTS -> (pos, policy, result)
  2. Train: update model on self-play data (policy + value loss)
  3. Export: save ONNX for Rust engine
  4. Repeat
"""

import argparse
import math
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import chess
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .features_lc0 import encode_position, NUM_PLANES
from .resnet import ResNetSE, create_model, POLICY_SIZE, policy_index
from .fastbatch import find_mate


# ---------------------------------------------------------------------------
# MCTS with batched GPU evaluation (pure Python, no heuristics)
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ("parent", "move", "children", "visits", "value_sum", "prior")

    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior

    def is_leaf(self):
        return not self.children

    def q(self):
        return self.value_sum / max(1, self.visits)


class MCTS:
    """Batched MCTS with GPU neural network evaluation."""

    def __init__(self, net, device, c_puct=1.4, batch_size=64, dir_alpha=0.3,
                 policy_temp=1.0):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.dir_alpha = dir_alpha
        self.policy_temp = policy_temp
        self.root = None

    def _encode_batch(self, boards, histories=None):
        """Encode a batch of boards into [B, 105, 8, 8] tensors.
        histories: optional per-board past-position lists (game history;
        search-path moves are excluded by design)."""
        if histories is None:
            planes = np.stack([encode_position(b) for b in boards], axis=0)
        else:
            planes = np.stack([encode_position(b, h)
                               for b, h in zip(boards, histories)], axis=0)
        return torch.tensor(planes, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def _forward(self, boards, histories=None):
        """Batched net eval: returns (value, policy_logits)."""
        x = self._encode_batch(boards, histories)
        policy_logits, value = self.net(x)
        return value, policy_logits

    def _ucb(self, node):
        # Node stores its OWN side-to-move frame (see _backup), so the
        # choosing parent must negate: 1-q = P(parent-side wins).
        q = 1.0 - node.q()
        u = self.c_puct * node.prior * math.sqrt(node.parent.visits) / (1 + node.visits)
        return q + u

    def search(self, board, iterations, history=()):
        """Run MCTS for `iterations` playouts. Returns visit counts.
        history: game positions before `board` (oldest first) for the
        history planes."""
        if self.root is None:
            self.root = MCTSNode()
        if not self.root.children:
            self._expand_root(board, history)
        if not self.root.children:
            return {}

        remaining = iterations
        while remaining > 0:
            b = min(self.batch_size, remaining)
            leaves = []
            for _ in range(b):
                leaf, path = self._select(board)
                leaves.append((leaf, path))
            self._evaluate_and_backup(board, leaves, history)
            remaining -= b

        return {m: c.visits for m, c in self.root.children.items()}

    def _expand_root(self, board, history=(), add_noise=True):
        """Expand root node with policy priors (+Dirichlet noise, AlphaZero:
        25% noise keeps self-play diverse across games in a cycle)."""
        boards = [board]
        _value, policy_logits = self._forward(boards, [history])
        self._expand_node(self.root, board, policy_logits[0], add_noise=add_noise)

    def _expand_node(self, node, board, logits, add_noise=False):
        """Create children from legal moves using policy logits."""
        legal = list(board.legal_moves)
        if not legal:
            return False
        indices = torch.tensor([policy_index(board, m) for m in legal],
                               device=self.device)
        t = max(0.05, self.policy_temp)
        probs = F.softmax(logits[indices] / t, dim=0)
        if add_noise and len(legal) > 1:
            noise = torch.tensor(np.random.dirichlet([self.dir_alpha] * len(legal)),
                                 dtype=torch.float32, device=self.device)
            probs = 0.75 * probs + 0.25 * noise
        for mv, p in zip(legal, probs.tolist()):
            node.children[mv] = MCTSNode(parent=node, move=mv, prior=p)
        return True

    def _select(self, board):
        """One traversal from root to a leaf. Returns (leaf_node, path)."""
        node = self.root
        path = [node]
        moves_played = []
        while node.children:
            best_mv = max(node.children, key=lambda m: self._ucb(node.children[m]))
            board.push(best_mv)
            moves_played.append(best_mv)
            node = node.children[best_mv]
            path.append(node)
        for _ in moves_played:
            board.pop()
        return node, path

    def _evaluate_and_backup(self, board, leaves, history=()):
        """Batch-evaluate leaves and backup values."""
        boards_to_eval = []
        eval_info = []

        for leaf, path in leaves:
            for node in path[1:]:
                board.push(node.move)

            if leaf.children:
                for _ in path[1:]:
                    board.pop()
                self._backup(path, leaf.q())
                continue

            legal = list(board.legal_moves)
            if not legal:
                # STM frame: the side to move is mated (0.0) or stalemated
                # (0.5). _backup alternates perspectives, so the incoming
                # value must be leaf-side-to-move POV like the NN branch
                # below — NEVER white POV (a white-delivered mate backed up
                # as 1.0 would read as a loss at a white root).
                if board.is_checkmate():
                    value = 0.0
                else:
                    value = 0.5
                for _ in path[1:]:
                    board.pop()
                self._backup(path, value)
                continue

            boards_to_eval.append(board.copy())
            eval_info.append((leaf, path, board.turn == chess.WHITE))

            for _ in path[1:]:
                board.pop()

        if not boards_to_eval:
            return

        value, policy_logits = self._forward(
            boards_to_eval, [history] * len(boards_to_eval))

        for i, (leaf, path, stm_is_white) in enumerate(eval_info):
            # Net outputs tanh already: map [-1,1] -> probability [0,1].
            # The tree works in probabilities (backup alternates 1-v).
            v = (value[i].item() + 1.0) / 2.0
            self._expand_node(leaf, boards_to_eval[i], policy_logits[i])
            self._backup(path, v)

    def _backup(self, path, value):
        """Backup value along path, alternating perspectives."""
        for node in reversed(path):
            node.visits += 1
            node.value_sum += value
            value = 1.0 - value

    def make_move(self, move):
        """Reuse subtree for next search."""
        child = self.root.children.get(move) if self.root else None
        self.root = child
        if self.root is not None:
            self.root.parent = None
            self.root.prior = 0.0

    def reset(self):
        self.root = None


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------

def pick_move(visits, temp=1.0):
    """Sample move from visit counts."""
    if temp <= 0 or len(visits) == 1:
        return max(visits, key=visits.get)
    probs = {m: math.pow(v, 1.0 / temp) for m, v in visits.items()}
    total = sum(probs.values())
    r = random.random() * total
    acc = 0.0
    for m, p in probs.items():
        acc += p
        if r <= acc:
            return m
    return max(visits, key=visits.get)


def pick_avoiding_repetition(board, visits, temp=1.0):
    """Pick a move, refusing moves that repeat a position 3rd time (unless
    forced). Prevents dead shuffle loops that teach nothing. Also never
    walks into an opponent mate-in-1 (exact check beats any budget)."""
    from .fastbatch import allows_opp_mate
    cands = dict(visits)
    safe = {m: v for m, v in cands.items()
            if not allows_opp_mate(board, m)}
    pool = safe or cands
    while pool:
        mv = pick_move(pool, temp) if temp > 0 else max(pool, key=pool.get)
        board.push(mv)
        bad = board.is_repetition(3)
        board.pop()
        if not bad or len(pool) == 1:
            return mv
        del pool[mv]
    return max(visits, key=visits.get)


@torch.no_grad()
def white_win_prob(net, device, board):
    """Direct net eval: P(white wins), from the tanh scalar."""
    planes = np.stack([encode_position(board)], axis=0)
    x = torch.tensor(planes, dtype=torch.float32, device=device)
    _, value = net(x)
    v = value[0].item()
    return (v + 1.0) / 2.0 if board.turn == chess.WHITE else 1.0 - (v + 1.0) / 2.0


def resign_winner(net, device, board):
    """Adjudicate a hopeless position (AZ-style 5% threshold on the side
    to move). Returns 'w', 'b', or None."""
    try:
        wprob = white_win_prob(net, device, board)
    except Exception:
        return None
    if wprob < 0.05:
        return "b"
    if wprob > 0.95:
        return "w"
    return None


def play_game(net, device, mcts_iters=800, mcts_batch=64, c_puct=1.4,
              temp=1.0, temp_drop=30, max_plies=512, resign=False, fen=None,
              dir_alpha=0.3, policy_temp=1.0):
    """Self-play one game. Returns (fens, policies, hists, result) tuples:
    hists[i] = FENs of previous game positions (oldest first, up to 7)."""
    board = chess.Board(fen) if fen else chess.Board()
    history = []
    game_fens = []  # past FENs (records, picklable)
    game_boards = []  # past positions (encoder input)
    mcts = MCTS(net, device, c_puct=c_puct, batch_size=mcts_batch,
                dir_alpha=dir_alpha, policy_temp=policy_temp)
    ply = 0

    while not board.is_game_over() and ply < max_plies:
        hist = tuple(game_fens[-7:])
        hist_boards = game_boards[-7:]
        mate = find_mate(board)
        if mate is not None:
            # Mate-in-1: always play it (exact 1-ply check beats any search
            # budget/temperature). Hard target teaches the policy the mate.
            history.append((board.fen(), {mate.uci(): 1.0}, hist))
            game_fens.append(board.fen())
            game_boards.append(board.copy())
            board.push(mate)
            mcts.make_move(mate)
            ply += 1
            continue
        visits = mcts.search(board, mcts_iters, hist_boards)
        if not visits:
            break
        t = temp if ply < temp_drop else 0.0
        move = pick_avoiding_repetition(board, visits, t)

        # Visit COUNTS must be normalized to probabilities: the KL loss
        # assumes targets sum to 1. Raw counts (≈mcts_iters) would scale
        # policy gradients x300 and slam the policy onto one move.
        total = sum(visits.values())
        policy_dict = {m.uci(): v / total for m, v in visits.items()}
        history.append((board.fen(), policy_dict, hist))

        game_fens.append(board.fen())
        game_boards.append(board.copy())
        board.push(move)
        mcts.make_move(move)
        ply += 1

        # Adjudicate hopeless positions instead of shuffling to fivefold.
        # Only with calibrated values (else resigns fire on noise): off early.
        if resign and ply >= temp_drop and ply % 5 == 0:
            rw = resign_winner(net, device, board)
            if rw == "w":
                return history, 1.0, "resign"
            if rw == "b":
                return history, 0.0, "resign"

    if board.is_checkmate():
        winner = not board.turn
        result = 1.0 if winner == chess.WHITE else 0.0
        term = "mate"
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

    return history, result, term


def _quiet_worker():
    """Silence worker stdout/stderr: spawned workers inherit the log file
    handle, and concurrent writes trample each other's bytes (corrupted
    progress lines). Exceptions still surface via futures in the parent."""
    import sys as _sys
    import os as _os
    _sys.stdout = open(_os.devnull, "w")
    _sys.stderr = open(_os.devnull, "w")


def _play_game_worker(args):
    """Top-level function for multiprocessing workers."""
    (wid, state_dict_path, mcts_iters, mcts_batch, c_puct, temp, temp_drop,
     resign, channels, blocks) = args
    dev = f"cuda:{wid % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu"
    model = create_model(channels=channels, num_blocks=blocks).to(dev)
    model.load_state_dict(torch.load(state_dict_path, map_location=dev, weights_only=True))
    model.eval()
    history, result, term = play_game(
        model, dev, mcts_iters, mcts_batch, c_puct, temp, temp_drop,
        resign=resign,
    )
    return ([(f, p, h, result) for f, p, h in history], term)


def self_play_batch(net, device, games, mcts_iters, mcts_batch, c_puct,
                    temp, temp_drop, workers=1, resign=False,
                    channels=224, blocks=15):
    """Self-play multiple games. Returns training data."""
    all_fens = []
    all_policies = []
    all_hists = []
    all_results = []
    all_terms = []

    if workers <= 1:
        term_counts: Counter = Counter()
        for i in range(games):
            history, result, term = play_game(
                net, device, mcts_iters, mcts_batch, c_puct, temp, temp_drop,
                resign=resign,
            )
            all_terms.append(term)
            term_counts[term] += 1
            for fen, policy_dict, hist in history:
                all_fens.append(fen)
                all_policies.append(policy_dict)
                all_hists.append(hist)
                all_results.append(result)
            tstr = " ".join(f"{k}:{v}" for k, v in sorted(term_counts.items()))
            print(f"\r  Game {i+1}/{games} · {len(all_fens):,} pos [{tstr}]", end="", flush=True)
        print()
    else:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        torch.save(net.state_dict(), tmp.name)
        tmp.close()
        state_dict_path = tmp.name

        worker_args = [
            (i, state_dict_path, mcts_iters, mcts_batch, c_puct, temp, temp_drop,
             resign, channels, blocks)
            for i in range(games)
        ]

        done = 0
        term_counts: Counter = Counter()
        with ProcessPoolExecutor(max_workers=workers, initializer=_quiet_worker) as pool:
            futures = {pool.submit(_play_game_worker, wa): i for i, wa in enumerate(worker_args)}
            for future in as_completed(futures):
                results, term = future.result()
                all_terms.append(term)
                term_counts[term] += 1
                for fen, policy_dict, hist, result in results:
                    all_fens.append(fen)
                    all_policies.append(policy_dict)
                    all_hists.append(hist)
                    all_results.append(result)
                done += 1
                tstr = " ".join(f"{k}:{v}" for k, v in sorted(term_counts.items()))
                print(f"\r  Game {done}/{games} · {len(all_fens):,} pos [{tstr}]", end="", flush=True)
        print()

        os.unlink(state_dict_path)

    return all_fens, all_policies, all_hists, all_results, all_terms


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def make_training_batch(fens, policies, hists, results, device):
    """Convert self-play data to training tensors. hists[i] = past FENs.
    z is stored STM-relative (white-POV result flipped for black to move)."""
    n = len(fens)
    boards = [chess.Board(f) for f in fens]
    planes = np.stack(
        [encode_position(b, [chess.Board(h) for h in hist])
         for b, hist in zip(boards, hists)], axis=0)
    x = torch.tensor(planes, dtype=torch.float32, device=device)
    z = torch.tensor(
        [r if b.turn == chess.WHITE else 1.0 - r
         for b, r in zip(boards, results)],
        dtype=torch.float32, device=device)

    policy_indices = []
    policy_values = []
    policy_mask = []
    max_moves = 0

    for fen, pol in zip(fens, policies):
        board = chess.Board(fen)
        legal = list(board.legal_moves)
        indices = []
        values = []
        for mv in legal:
            uci = mv.uci()
            if uci in pol and pol[uci] > 0:
                indices.append(policy_index(board, mv))
                values.append(pol[uci])
        s = sum(values)
        if s > 0:
            values = [v / s for v in values]
        policy_indices.append(indices)
        policy_values.append(values)
        max_moves = max(max_moves, len(indices))

    padded_indices = torch.zeros(n, max_moves, dtype=torch.long, device=device)
    padded_values = torch.zeros(n, max_moves, dtype=torch.float32, device=device)
    padded_mask = torch.zeros(n, max_moves, dtype=torch.bool, device=device)

    for i, (inds, vals) in enumerate(zip(policy_indices, policy_values)):
        if inds:
            padded_indices[i, :len(inds)] = torch.tensor(inds, dtype=torch.long)
            padded_values[i, :len(vals)] = torch.tensor(vals, dtype=torch.float32)
            padded_mask[i, :len(inds)] = True

    return x, z, padded_indices, padded_values, padded_mask


def train_step(net, optimizer, x, z, pol_idx, pol_val, pol_mask, scaler=None,
               entropy_coef=0.0):
    """AlphaZero loss: MSE(value, z) + CE(policy) - entropy bonus + NaN guard.
    z is STM-relative in [0,1] (1 = side to move wins); value head is tanh,
    so the MSE target is z mapped to [-1,1]. Returns
    (loss, vloss, ploss, entropy, skipped)."""
    net.train()
    optimizer.zero_grad(set_to_none=True)

    use_amp = scaler is not None
    device_type = "cuda" if next(net.parameters()).is_cuda else "cpu"
    if use_amp:
        with torch.amp.autocast(device_type=device_type):
            policy_logits, value = net(x)
            loss_v = F.mse_loss(value.squeeze(1), z * 2.0 - 1.0)
            loss_p = _policy_loss(policy_logits, pol_idx, pol_val, pol_mask)
            ent = _policy_entropy(policy_logits)
            loss = loss_v + loss_p - entropy_coef * ent
        if not torch.isfinite(loss):
            return float("nan"), float("nan"), float("nan"), float("nan"), True
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        policy_logits, value = net(x)
        loss_v = F.mse_loss(value.squeeze(1), z * 2.0 - 1.0)
        loss_p = _policy_loss(policy_logits, pol_idx, pol_val, pol_mask)
        ent = _policy_entropy(policy_logits)
        loss = loss_v + loss_p - entropy_coef * ent
        if not torch.isfinite(loss):
            return float("nan"), float("nan"), float("nan"), float("nan"), True
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

    return loss.item(), loss_v.item(), loss_p.item(), ent.item(), False


def _policy_entropy(policy_logits):
    """Mean Shannon entropy of the full policy (spread measure)."""
    log_probs = F.log_softmax(policy_logits, dim=1)
    return -(log_probs.exp() * log_probs).sum(dim=1).mean()


def _policy_loss(policy_logits, pol_idx, pol_val, pol_mask):
    """Mean cross-entropy between target policy and network policy.

    pol_val already sums to 1 per position (normalized visits / one-hot),
    so the sum IS the cross-entropy — no division by move count (that
    would downscale MCTS-visit targets 5-40x vs one-hot targets).
    """
    log_probs = F.log_softmax(policy_logits, dim=1)
    gathered = log_probs.gather(1, pol_idx.masked_fill(~pol_mask, 0))
    gathered = gathered.masked_fill(~pol_mask, 0.0)
    pol_val = pol_val.masked_fill(~pol_mask, 0.0)
    ce = -(pol_val * gathered).sum(dim=1)
    return ce.mean()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_onnx(model, path, fp16=False):
    """Export model to ONNX. Operates on a CPU copy: the live training
    model must stay on CUDA in fp32 (a previous version did
    model.cpu().eval() in place, which poisoned all later training with a
    device — then autocast dtype — mismatch crash)."""
    import copy

    import onnx

    export_model = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(1, NUM_PLANES, 8, 8)

    torch.onnx.export(
        export_model, dummy, path,
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
        from onnxconverter_common import float16
        onnx_model = float16.convert_float_to_float16(
            onnx_model, keep_io_types=True,
            min_positive_val=1e-7, max_finite_val=1e4,
        )
        onnx.save(onnx_model, path)

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Exported ONNX ({'FP16' if fp16 else 'FP32'}): {path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pure AlphaZero self-play training")
    ap.add_argument("--channels", type=int, default=224)
    ap.add_argument("--blocks", type=int, default=15)
    ap.add_argument("--cycles", type=int, default=100)
    ap.add_argument("--games-per-cycle", type=int, default=500)
    ap.add_argument("--train-epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=0.2)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.0,
                    help="Policy entropy bonus (anti single-move attractor). "
                         "NOTE: the sync loop is legacy (async_train is the "
                         "active runner); keep 0.0 unless experimenting.")
    ap.add_argument("--mcts-iters", type=int, default=800)
    ap.add_argument("--mcts-batch", type=int, default=64)
    ap.add_argument("--c-puct", type=float, default=1.4)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--temp-drop", type=int, default=30)
    ap.add_argument("--max-plies", type=int, default=512)
    ap.add_argument("--resign", action="store_true", default=False,
                    help="Adjudicate hopeless positions (needs calibrated values)")
    ap.add_argument("--fast-ratio", type=float, default=0.0,
                    help="Fraction of games played fast policy-only (hard targets). "
                         "Bulk volume; MCTS games carry the visit distributions.")
    ap.add_argument("--fast-threads", type=int, default=12)
    ap.add_argument("--infer-batch", type=int, default=512)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--resume", default=None, help="Resume from checkpoint .pt")
    ap.add_argument("--start-cycle", type=int, default=0,
                    help="First cycle number for saving (0 = auto: after existing)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=1, help="Parallel self-play workers")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device == "cuda"

    # Throughput: static input shapes -> autotune convs; TF32 matmuls.
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    os.makedirs(args.out_dir, exist_ok=True)

    # Continue cycle numbering past existing checkpoints (never overwrite).
    def detect_start(out_dir):
        import re
        mx = 0
        try:
            for f in os.listdir(out_dir):
                m = re.match(r"model_cycle(\d+)\.pt$", f)
                if m:
                    mx = max(mx, int(m.group(1)))
        except OSError:
            pass
        return mx + 1

    start_cycle = args.start_cycle or detect_start(args.out_dir)

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        model = create_model(channels=args.channels, num_blocks=args.blocks)
        model.load_state_dict(torch.load(args.resume, map_location=device))
        model = model.to(device)
    else:
        model = create_model(channels=args.channels, num_blocks=args.blocks).to(device)

    # AlphaZero optimizer (same as async runner): SGD + momentum 0.9,
    # lr 0.2, L2 via weight_decay (paper c=1e-4).
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=args.weight_decay
    )
    drop_cycles = {args.cycles // 2, args.cycles * 3 // 4,
                   args.cycles * 7 // 8}
    scaler = torch.amp.GradScaler(device) if use_amp else None

    total_games = 0
    total_positions = 0

    print(f"\nAlphaZero Training")
    print(f"  Device: {device}")
    print(f"  AMP: {use_amp}")
    print(f"  Model: {model.count_parameters():,} parameters")
    print(f"  Games/cycle: {args.games_per_cycle}")
    print(f"  MCTS iters: {args.mcts_iters}")
    print(f"  MCTS batch: {args.mcts_batch}")
    print(f"  Workers: {args.workers}")
    print(f"  Fast ratio: {args.fast_ratio} (threads={args.fast_threads}, infer-batch={args.infer_batch})")
    print(f"  Cycles: {args.cycles} (saving {start_cycle:04d}..{start_cycle + args.cycles - 1:04d})")
    print()

    for i in range(args.cycles):
        cycle = start_cycle + i
        t0 = time.time()

        # --- Self-play (mixed MCTS + fast policy games) ---
        print(f"Cycle {cycle}/{start_cycle + args.cycles - 1} | Self-play ({args.games_per_cycle} games)...", end=" ", flush=True)
        model.eval()
        n_fast = int(args.games_per_cycle * args.fast_ratio)
        n_mcts = args.games_per_cycle - n_fast
        fens, policies, hists, results, terms = [], [], [], [], []
        if n_mcts > 0:
            fens, policies, hists, results, terms = self_play_batch(
                model, device,
                games=n_mcts,
                mcts_iters=args.mcts_iters,
                mcts_batch=args.mcts_batch,
                c_puct=args.c_puct,
                temp=args.temp,
                temp_drop=args.temp_drop,
                workers=args.workers,
                resign=args.resign,
                channels=args.channels,
                blocks=args.blocks,
            )
        if n_fast > 0:
            from .fastbatch import InferenceServer, play_fast_batch
            server = InferenceServer(model, device, max_batch=args.infer_batch)
            server.start()
            ff, fu, fh, fr, ft = play_fast_batch(
                server, n_fast, temp_plies=args.temp_drop, max_plies=args.max_plies,
                temp=args.temp, threads=args.fast_threads,
                progress=lambda d, g: print(f"\r  Fast {d}/{g}", end="", flush=True),
            )
            server.stop()
            print()
            # Hard targets {played_move: 1.0} flow through the same KL loss.
            fens += ff
            policies += [{u: 1.0} for u in fu]
            hists += fh
            results += fr
            terms += ft
        n_games = len(set(
            chess.Board(f).fen().split(" ")[0] for f in fens
        ))
        total_games += args.games_per_cycle
        total_positions += len(fens)
        avg_result = sum(results) / len(results)
        term_str = " ".join(f"{k}:{v}" for k, v in sorted(Counter(terms).items()))
        print(f"{len(fens):,} positions, avg_result={avg_result:.3f} [{term_str}]")

        # --- Training ---
        print(f"  Training ({args.train_epochs} epochs)...", end=" ", flush=True)
        x, z, pol_idx, pol_val, pol_mask = make_training_batch(
            fens, policies, hists, results, device
        )
        if device == "cuda" and x.numel() * 4 < 6_000_000_000:
            # Whole dataset fits VRAM: keep it resident, kill per-batch H2D.
            dataset = TensorDataset(x, z, pol_idx, pol_val, pol_mask)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=False)
        else:
            dataset = TensorDataset(x.cpu(), z.cpu(), pol_idx.cpu(), pol_val.cpu(), pol_mask.cpu())
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=0, pin_memory=(device == "cuda"))

        for epoch in range(args.train_epochs):
            epoch_loss = 0.0
            epoch_vloss = 0.0
            epoch_ploss = 0.0
            n_batches = 0
            for batch_x, batch_z, batch_pi, batch_pv, batch_pm in loader:
                batch_x = batch_x.to(device, non_blocking=True)
                batch_z = batch_z.to(device, non_blocking=True)
                batch_pi = batch_pi.to(device, non_blocking=True)
                batch_pv = batch_pv.to(device, non_blocking=True)
                batch_pm = batch_pm.to(device, non_blocking=True)
                loss, vloss, ploss, _ent, _skipped = train_step(
                    model, optimizer, batch_x, batch_z,
                    batch_pi, batch_pv, batch_pm, scaler,
                    entropy_coef=args.entropy_coef
                )
                epoch_loss += loss
                epoch_vloss += vloss
                epoch_ploss += ploss
                n_batches += 1
            print(f"\r  Epoch {epoch+1}/{args.train_epochs} loss={epoch_loss/max(1,n_batches):.4f}", end="", flush=True)
        print()

        # AZ step schedule: lr x0.1 at 1/2, 3/4, 7/8 of cycles.
        if cycle in drop_cycles:
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1
        avg_loss = epoch_loss / max(1, n_batches)
        avg_vloss = epoch_vloss / max(1, n_batches)
        avg_ploss = epoch_ploss / max(1, n_batches)
        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        print(f"loss={avg_loss:.4f} (v={avg_vloss:.4f} p={avg_ploss:.4f}) lr={lr:.2e} {dt:.1f}s")

        # --- Save checkpoint ---
        ckpt_path = os.path.join(args.out_dir, f"model_cycle{cycle:04d}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # --- Append-only history (survives log overwrites/restarts) ---
        hist_path = os.path.join(args.out_dir, "history.csv")
        new_hist = not os.path.exists(hist_path)
        with open(hist_path, "a") as hf:
            if new_hist:
                hf.write("cycle,games,positions,avg_result,loss,vloss,ploss,lr,seconds,terms,timestamp\n")
            hf.write(f"{cycle},{args.games_per_cycle},{len(fens)},{avg_result:.4f},"
                     f"{avg_loss:.4f},{avg_vloss:.4f},{avg_ploss:.4f},{lr:.2e},{dt:.0f},"
                     f"\"{term_str}\",{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Export ONNX for Rust engine
        if cycle % 5 == 0 or i == args.cycles - 1:
            onnx_path = os.path.join(args.out_dir, f"net_cycle{cycle:04d}.onnx")
            export_onnx(model, onnx_path, fp16=False)
            onnx_fp16_path = os.path.join(args.out_dir, f"net_cycle{cycle:04d}_fp16.onnx")
            export_onnx(model, onnx_fp16_path, fp16=True)

    print(f"\nDone. Total: {total_games} games, {total_positions:,} positions")


if __name__ == "__main__":
    main()
