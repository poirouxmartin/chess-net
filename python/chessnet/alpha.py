"""AlphaZero-style self-play reinforcement learning.

The value head is the NNUE net (logit -> sigmoid = win prob for WHITE) and the
policy head outputs a logit per from*64+to move, giving MCTS learned priors
over legal moves (softmax-masked). Self-play games produce
(board, visits, white_result) triples: the value label is the game outcome
(always white POV) and the policy label is the root visit distribution.

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

from .nnue import (
    NNUE,
    encode_batch,
    move_policy_index,
    FEAT_HALFKP,
    FEAT_KP768,
    HALFKP_FEAT_COUNT,
    KP768_FEAT_COUNT,
)


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

    def _expand(self, node, board, logits):
        """Create children for `node` from the legal moves of `board`, using
        the policy logits (4096-vector from the net) as masked-softmax priors.
        Returns False when the position is terminal (mate/stalemate)."""
        legal = list(board.legal_moves)
        if not legal:
            return False
        inds = torch.tensor([move_policy_index(m) for m in legal])
        probs = torch.softmax(logits[inds], dim=0)
        for mv, p in zip(legal, probs.tolist()):
            node.children[mv] = MCTSNode(parent=node, move=mv, prior=p)
        return True

    def _forward(self, boards):
        """Batched net eval: returns (white_win_probs, policy_logits) as CPU
        tensors, aligned with `boards`."""
        idx, mask = encode_batch(boards, self.feat)
        with torch.no_grad():
            logits, policy = self.net.forward_with_policy(
                idx.to(self.device), mask.to(self.device)
            )
        return torch.sigmoid(logits).flatten().cpu(), policy.cpu()

    def _ucb(self, node):
        q = node.value / max(1, node.visits)
        u = self.c_puct * node.prior * math.sqrt(node.parent.visits) / (1 + node.visits)
        return q + u

    def search(self, board, iterations):
        if self.root is None:
            self.root = MCTSNode()
        if not self.root.children:
            _, policy = self._forward([board])
            if not self._expand(self.root, board, policy[0]):
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
        # board, detect terminals, dedupe net evals by zobrist, then expand the
        # non-terminal leaves with their policy logits and backup.
        ready = []  # (node, path, stm_value)
        to_forward = []  # (node, path, turn, zkey, leaf_board)
        for node, path in leaves:
            for n in path[1:]:
                board.push(n.move)
            if not node.children and not any(board.legal_moves):
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
            values, policies = self._forward([t[4] for t in to_forward])
            for (node, path, turn, zkey, leaf_board), ww, pl in zip(to_forward, values, policies):
                ww = ww.item()
                self.cache[zkey] = ww
                if not node.children:
                    self._expand(node, leaf_board, pl)
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
    """Self-play one game. Returns list of (board_fen, visits, white_result)
    triples: `visits` is the root visit distribution {uci: count} used as the
    policy label, `white_result` in {1.0, 0.5, 0.0} is the value label."""
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
        history.append((board.copy(), {m.uci(): v for m, v in visits.items()}))
        board.push(move)
        search.make_move(move)
        ply += 1

    if board.is_checkmate():
        winner = not board.turn  # side that delivered mate
        white_result = 1.0 if winner == chess.WHITE else -1.0
    else:
        white_result = 0.0  # draw / insufficient material / stalemate / repetition

    data = []
    for b, visits in history:
        # Labels are always in WHITE's perspective: 1.0/0.5/0.0.
        data.append((b.fen(), visits, (white_result + 1.0) / 2.0))
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

    def train(self, data, epochs=1, batch_size=1024, lambda_policy=1.0):
        has_policy = len(data[0]) == 3
        boards = [chess.Board(row[0]) for row in data]
        labels = torch.tensor([row[-1] for row in data], dtype=torch.float32)
        idx, mask = encode_batch(boards, _feat_id(self.net.feat_count))
        if has_policy and lambda_policy > 0:
            # Policy targets: per position, distinct legal (from*64+to) indices
            # and their visit-probabilities (promotion variants share an index).
            tgt_inds, tgt_probs, tgt_mask = self._policy_targets(data)
        n = len(data)
        for _ in range(epochs):
            perm = torch.randperm(n)
            for i in range(0, n, batch_size):
                sel = perm[i : i + batch_size]
                b = idx[sel].to(self.device)
                m = mask[sel].to(self.device)
                y = labels[sel].to(self.device)
                logit, policy = self.net.forward_with_policy(b, m)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
                if has_policy and lambda_policy > 0:
                    g = policy.gather(1, tgt_inds[sel].to(self.device))
                    g = g.masked_fill(~tgt_mask[sel].to(self.device), -float("inf"))
                    logp = g - torch.logsumexp(g, dim=1, keepdim=True)
                    ce = -(tgt_probs[sel].to(self.device) * logp)
                    ce = ce.masked_fill(~tgt_mask[sel].to(self.device), 0.0)
                    loss = loss + lambda_policy * ce.sum(dim=1).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
        return self.net

    def _policy_targets(self, data):
        n = len(data)
        inds_list, probs_list = [], []
        for row in data:
            b = chess.Board(row[0])
            visits = row[1]
            agg = {}
            for mv in b.legal_moves:
                cnt = visits.get(mv.uci(), 0)
                if cnt:
                    ind = move_policy_index(mv)
                    agg[ind] = agg.get(ind, 0) + cnt
            total = sum(agg.values())
            if total <= 0:
                inds_list.append(torch.zeros(0, dtype=torch.long))
                probs_list.append(torch.zeros(0))
                continue
            inds = torch.tensor(list(agg.keys()), dtype=torch.long)
            probs = torch.tensor([c / total for c in agg.values()])
            inds_list.append(inds)
            probs_list.append(probs)
        k = max(len(t) for t in inds_list)
        inds = torch.zeros(n, k, dtype=torch.long)
        probs = torch.zeros(n, k)
        mask = torch.zeros(n, k, dtype=torch.bool)
        for i, (iv, pv) in enumerate(zip(inds_list, probs_list)):
            inds[i, : len(iv)] = iv
            probs[i, : len(pv)] = pv
            mask[i, : len(iv)] = True
        return inds, probs, mask


def _play_worker(state, games, iterations, device, feat_count, l0, l1):
    """Subprocess entry point for parallel self-play (Windows spawn-safe)."""
    net = NNUE(feat_count, l0, l1)
    net.load_state_dict(state)
    net.eval().to(device)
    data = []
    for _ in range(games):
        data.extend(play_game(net, iterations, device))
    return data