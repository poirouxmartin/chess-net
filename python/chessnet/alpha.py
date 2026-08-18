"""AlphaZero-style self-play reinforcement learning.

The value head is the NNUE net (logit -> sigmoid = win prob for stm).
MCTS uses a uniform prior over legal moves (no policy head yet).
Self-play games produce (board, result-from-stm) pairs used to train the net.
"""

import math
import random

import chess
import torch

from .nnue import NNUE, encode_batch, FEAT_KP768, KP768_FEAT_COUNT


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

    def ucb(self, c_puct=1.4):
        if self.parent is None:
            return float("inf")
        q = self.value / max(1, self.visits)
        u = c_puct * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return q + u


def mcts(root_board, net, iterations, device):
    """Run MCTS from root_board using net for leaf values. Returns visit counts."""
    root = MCTSNode()
    legal = list(root_board.legal_moves)
    if not legal:
        return {}
    for mv in legal:
        root.children[mv] = MCTSNode(parent=root, move=mv, prior=1.0 / len(legal))

    for _ in range(iterations):
        node, board = root, root_board.copy()
        # Selection.
        while not node.is_leaf():
            move = max(node.children.values(), key=lambda c: c.ucb()).move
            board.push(move)
            node = node.children[move]
        # Expansion + evaluation (leaf).
        legal = list(board.legal_moves)
        if not legal:
            value = -1.0  # leaf is checkmate against side to move
        else:
            idx, mask = encode_batch([board], FEAT_KP768)
            with torch.no_grad():
                logit = net(idx.to(device), mask.to(device)).item()
            value = 1.0 / (1.0 + math.exp(-logit))
            for mv in legal:
                node.children[mv] = MCTSNode(parent=node, move=mv, prior=1.0 / len(legal))
        # Backup.
        while node is not None:
            node.visits += 1
            node.value += value
            value = 1.0 - value  # flip perspective
            node = node.parent
    return {m: c.visits for m, c in root.children.items()}


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


def play_game(net, iterations, device, max_plies=400, temp=1.0, temp_drop=12):
    """Self-play one game. Returns list of (board, stm_result) data pairs."""
    board = chess.Board()
    history = []
    ply = 0
    while not board.is_game_over() and ply < max_plies:
        visits = mcts(board, net, iterations, device)
        if not visits:
            break
        t = temp if ply < temp_drop else 0.0
        move = pick_move(visits, t)
        history.append(board.copy())
        board.push(move)
        ply += 1

    if board.is_checkmate():
        winner = not board.turn  # side that delivered mate
        white_result = 1.0 if winner == chess.WHITE else -1.0
    else:
        white_result = 0.0  # draw / insufficient material / stalemate / repetition

    data = []
    for b in history:
        stm = b.turn
        result = white_result if stm == chess.WHITE else -white_result
        data.append((b, (result + 1.0) / 2.0))
    return data


class AlphaTrainer:
    def __init__(self, feat_count=KP768_FEAT_COUNT, l0=256, l1=32, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = NNUE(feat_count, l0, l1).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def self_play(self, games, iterations, mcts_workers=1):
        data = []
        for _ in range(games):
            data.extend(play_game(self.net, iterations, self.device))
        return data

    def train(self, data, epochs=1, batch_size=1024):
        boards = [b for b, _ in data]
        labels = torch.tensor([y for _, y in data], dtype=torch.float32)
        idx, mask = encode_batch(boards, FEAT_KP768)
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