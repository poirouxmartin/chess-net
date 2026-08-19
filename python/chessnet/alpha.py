"""AlphaZero-style self-play reinforcement learning.

The value head is the NNUE net (logit -> sigmoid = win prob for WHITE).
MCTS uses a uniform prior over legal moves (no policy head yet).
Self-play games produce (board, white_result) pairs used to train the net:
  label = 1.0 white wins, 0.5 draw, 0.0 black wins (always white POV).
The side to move only flips the leaf value inside MCTS, never the label.

Search is bulk-parallel: leaf positions of a batch of traversals are
evaluated in a single GPU forward pass, deduplicated through a zobrist value
cache, and the tree is reused between plies (the child of the played move
becomes the new root).
"""

import math
import multiprocessing as mp
import random

import chess
import torch

from .nnue import NNUE, encode_batch, FEAT_HALFKP, FEAT_KP768, HALFKP_FEAT_COUNT, KP768_FEAT_COUNT


def _feat_id(feat_count):
    return FEAT_HALFKP if feat_count == HALFKP_FEAT_COUNT else FEAT_KP768


class MCTSNode:
    __slots__ = ("parent", "move", "children", "visits", "value", "prior")

    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.prior = prior

    def is_leaf(self):
        return not self.children


class MCTS:
    """Bulk MCTS with batched leaf evaluation and a zobrist value cache.

    `search(board, iterations)` runs `iterations` traversals in batches of
    `batch_size`: each batch selects batch_size leaves, evaluates the unique
    ones in a single forward pass (cache hits are skipped), then backs up every
    path. `make_move(move)` reparents the child subtree as the new root so the
    next `search` keeps the previous work.
    """

    def __init__(self, net, device, c_puct=1.4, batch_size=64, cache_size=200_000):
        self.net = net
        self.device = device
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.feat = _feat_id(net.feat_count)
        self.cache = {}
        self.cache_max = cache_size
        self.root = None

    def _policy_prior(self, board, legal):
        """Heuristic prior over legal moves (captures MVV-LVA + promotions).
        Uniform would explore every move equally; this biases the search toward
        tactically promising moves. A learned policy head can replace it by
        returning per-move logits (sum stays 1)."""
        scores = []
        for mv in legal:
            s = 1.0
            if board.is_capture(mv):
                attacker = board.piece_at(mv.from_square)
                victim = board.piece_at(mv.to_square)
                a = attacker.piece_type if attacker else 0
                v = victim.piece_type if victim else 0
                s += 10.0 + v - 0.1 * a  # MVV-LVA
            if mv.promotion:
                s += 8.0 + mv.promotion
            scores.append(s)
        total = sum(scores)
        return [s / total for s in scores]

    def _expand(self, node, board):
        """Create children for `node` from the legal moves of `board`.
        Returns False when the position is terminal (mate/stalemate)."""
        legal = list(board.legal_moves)
        if not legal:
            return False
        for mv, p in zip(legal, self._policy_prior(board, legal)):
            node.children[mv] = MCTSNode(parent=node, move=mv, prior=p)
        return True

    def _ucb(self, node):
        q = node.value / max(1, node.visits)
        u = self.c_puct * node.prior * math.sqrt(node.parent.visits) / (1 + node.visits)
        return q + u

    def search(self, board, iterations):
        if self.root is None:
            self.root = MCTSNode()
        if not self.root.children and not self._expand(self.root, board):
            return {}
        remaining = iterations
        while remaining > 0:
            b = min(self.batch_size, remaining)
            leaves = [self._select(board) for _ in range(b)]
            self._evaluate_and_backup(board, leaves)
            remaining -= b
        return {m: c.visits for m, c in self.root.children.items()}

    def _select(self, board):
        """One traversal from root to a leaf, replaying `board` along the path
        (left at the root afterwards). Returns (leaf_node, path_nodes)."""
        node = self.root
        path = [node]
        while node.children:
            move = max(node.children.values(), key=self._ucb).move
            board.push(move)
            node = node.children[move]
            path.append(node)
        for _ in range(len(path) - 1):
            board.pop()
        return node, path

    def _evaluate_and_backup(self, board, leaves):
        # board is at the root here. Replay each path to materialize the leaf
        # board, expand it, dedupe net evals by zobrist, then backup.
        ready = []  # (node, path, stm_value)
        to_forward = []  # (node, path, turn, zkey, leaf_board)
        for node, path in leaves:
            for n in path[1:]:
                board.push(n.move)
            if not node.children:
                self._expand(node, board)
            if not node.children:
                # Terminal leaf: mate (0.0) or stalemate (0.5) for the stm.
                ready.append((node, path, 0.0 if board.is_checkmate() else 0.5))
            else:
                zkey = board._transposition_key()
                ww = self.cache.get(zkey)
                if ww is not None:
                    ready.append((node, path, ww if board.turn == chess.WHITE else 1.0 - ww))
                else:
                    to_forward.append((node, path, board.turn, zkey, board.copy()))
            for _ in range(len(path) - 1):
                board.pop()

        if to_forward:
            idx, mask = encode_batch([t[4] for t in to_forward], self.feat)
            with torch.no_grad():
                logits = self.net(idx.to(self.device), mask.to(self.device))
                white_wins = torch.sigmoid(logits).flatten()
            for (node, path, turn, zkey, _), ww in zip(to_forward, white_wins):
                ww = ww.item()
                self.cache[zkey] = ww
                ready.append((node, path, ww if turn == chess.WHITE else 1.0 - ww))
            if len(self.cache) >= self.cache_max:
                self.cache.clear()

        for node, path, stm_value in ready:
            self._backup(node, path, stm_value)

    def _backup(self, node, path, value):
        for n in reversed(path):
            n.visits += 1
            n.value += value
            value = 1.0 - value

    def make_move(self, move):
        """Reuse the child subtree as the new root (caller pushed the move on
        its own board already)."""
        child = self.root.children.get(move) if self.root else None
        self.root = child
        if self.root is not None:
            self.root.parent = None
            self.root.prior = 0.0

    def reset(self):
        self.root = None
        self.cache.clear()


def mcts(root_board, net, iterations, device):
    """Run MCTS from root_board using net for leaf values. Returns visit counts."""
    return MCTS(net, device).search(root_board, iterations)


def pick_move(visits, temp=1.0):
    """Choose a move from visit counts; temp=0 -> argmax."""
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


def play_game(net, iterations, device, max_plies=400, temp=1.0, temp_drop=12, batch_size=64, c_puct=1.4):
    """Self-play one game. Returns list of (board, white_result) data pairs."""
    board = chess.Board()
    history = []
    search = MCTS(net, device, c_puct=c_puct, batch_size=batch_size)
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        visits = search.search(board, iterations)
        if not visits:
            break
        t = temp if ply < temp_drop else 0.0
        move = pick_move(visits, t)
        history.append(board.copy())
        board.push(move)
        search.make_move(move)
        ply += 1

    if board.is_checkmate():
        winner = not board.turn  # side that delivered mate
        white_result = 1.0 if winner == chess.WHITE else -1.0
    else:
        white_result = 0.0  # draw / insufficient material / stalemate / repetition

    data = []
    for b in history:
        # Labels are always in WHITE's perspective: 1.0/0.5/0.0.
        data.append((b.fen(), (white_result + 1.0) / 2.0))
    return data


class AlphaTrainer:
    def __init__(self, feat_count=KP768_FEAT_COUNT, l0=256, l1=32, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = NNUE(feat_count, l0, l1).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def self_play(self, games, iterations, mcts_workers=1):
        """Self-play `games` games. With mcts_workers>1, play in parallel
        subprocesses (each builds its own copy of the net)."""
        if mcts_workers <= 1:
            return self._serial_play(games, iterations)
        state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        ctx = mp.get_context("spawn")
        sizes = [games // mcts_workers] * mcts_workers
        for i in range(games % mcts_workers):
            sizes[i] += 1
        with ctx.Pool(mcts_workers) as pool:
            results = pool.starmap(
                _play_worker,
                [
                    (state, s, iterations, self.device, self.net.feat_count,
                     self.net.fb.numel(), self.net.wo.in_features)
                    for s in sizes
                ],
            )
        return [item for r in results for item in r]

    def _serial_play(self, games, iterations):
        data = []
        for _ in range(games):
            data.extend(play_game(self.net, iterations, self.device))
        return data

    def train(self, data, epochs=1, batch_size=1024):
        boards = [chess.Board(fen) for fen, _ in data]
        labels = torch.tensor([y for _, y in data], dtype=torch.float32)
        idx, mask = encode_batch(boards, _feat_id(self.net.feat_count))
        n = len(data)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                sel = perm[i : i + batch_size]
                b = idx[sel].to(self.device)
                m = mask[sel].to(self.device)
                y = labels[sel].to(self.device)
                logit = self.net(b, m)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return self.net


def _play_worker(state, games, iterations, device, feat_count, l0, l1):
    """Subprocess entry point for parallel self-play (Windows spawn-safe)."""
    net = NNUE(feat_count, l0, l1)
    net.load_state_dict(state)
    net.eval().to(device)
    data = []
    for _ in range(games):
        data.extend(play_game(net, iterations, device))
    return data