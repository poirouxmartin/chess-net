"""Async replay buffer: actors push finished games, the learner samples.

Ingest encodes positions to uint8 planes on CPU (in parallel with the
learner's GPU work). Sampling returns CPU tensors; the learner moves them
to CUDA. Uniform sampling over the recent window (AlphaZero-style).
"""
import threading

import chess
import numpy as np
import torch

from .features_lc0 import encode_position
from .resnet import policy_index


class ReplayBuffer:
    def __init__(self, capacity=150_000, min_fill=8_000):
        self.cap = capacity
        self.min_fill = min_fill
        self.lock = threading.Lock()
        # Plain lists: O(1) random access for sampling. deque[i] is O(n)
        # and turned every train step into minutes once the buffer grew.
        # Eviction by chunked del (amortized O(1)).
        self.planes = []   # np uint8 [105,8,8]
        self.z = []        # float, STM win-proba in [0,1] (train_step maps to tanh)
        self.pi_idx = []   # list[int] policy indices (STM-oriented)
        self.pi_val = []   # list[float] target probs
        self.games_total = 0
        self.pos_total = 0

    def __len__(self):
        with self.lock:
            return len(self.planes)

    def ready(self):
        return len(self) >= self.min_fill

    def add_game(self, fens, policies, hists, result):
        """Encode one finished game (CPU). policies: list[{uci: prob}],
        hists: list of past-FEN tuples (oldest first, up to 7)."""
        from .fastbatch import find_mate
        pl, zl, il, vl = [], [], [], []
        for fen, pol, hist in zip(fens, policies, hists):
            board = chess.Board(fen)
            past = [chess.Board(h) for h in hist]
            pl.append(encode_position(board, past).astype(np.uint8))
            r = float(result)
            if board.turn == chess.BLACK:
                r = 1.0 - r
            zl.append(r)
            pol = dict(pol)
            idx, val = [], []
            for mv in board.legal_moves:
                uci = mv.uci()
                if uci in pol and pol[uci] > 0:
                    idx.append(policy_index(board, mv))
                    val.append(float(pol[uci]))
            il.append(idx)
            # Defensive: the KL loss assumes targets summing to 1. Renormalize
            # so a future caller passing raw counts can never silently replay
            # the x300-gradient collapse.
            s = sum(val)
            vl.append([v / s for v in val] if s > 0 else val)
        with self.lock:
            self.planes.extend(pl)
            self.z.extend(zl)
            self.pi_idx.extend(il)
            self.pi_val.extend(vl)
            over = len(self.planes) - self.cap
            if over > 0:
                del self.planes[:over]
                del self.z[:over]
                del self.pi_idx[:over]
                del self.pi_val[:over]
            self.games_total += 1
            self.pos_total += len(pl)

    def sample(self, batch_size):
        """CPU tensors: x float32 [B,105,8,8], z [B], padded pi idx/val/mask."""
        with self.lock:
            n = len(self.planes)
            take = np.random.randint(0, n, size=batch_size)
            planes = [self.planes[i] for i in take]
            zs = [self.z[i] for i in take]
            iis = [self.pi_idx[i] for i in take]
            vvs = [self.pi_val[i] for i in take]
        x = torch.from_numpy(np.stack(planes).astype(np.float32))
        z = torch.tensor(zs, dtype=torch.float32)
        m = max((len(v) for v in vvs), default=1)
        pi_idx = torch.zeros(batch_size, m, dtype=torch.long)
        pi_val = torch.zeros(batch_size, m, dtype=torch.float32)
        pi_mask = torch.zeros(batch_size, m, dtype=torch.bool)
        for i, (ii, vv) in enumerate(zip(iis, vvs)):
            if ii:
                pi_idx[i, :len(ii)] = torch.tensor(ii, dtype=torch.long)
                pi_val[i, :len(vv)] = torch.tensor(vv, dtype=torch.float32)
                pi_mask[i, :len(ii)] = True
        return x, z, pi_idx, pi_val, pi_mask
