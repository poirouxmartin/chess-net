"""PyTorch NNUE module and supervised training helpers.

The network matches the CSNN format read by the Rust engine:
  acc  = fb + sum over active features of fw[feature]     (sparse L0)
  h1   = relu(b1 + w1 @ acc)
  logit = bo + wo @ relu(h1)
  score_cp = logit * 400   (applied by the engine at inference)
"""

import chess
import torch
import torch.nn as nn

from . import features as F
from . import csnn

FEAT_HALFKP = F.FEAT_HALFKP
FEAT_KP768 = F.FEAT_KP768
HALFKP_FEAT_COUNT = F.HALFKP_FEAT_COUNT
KP768_FEAT_COUNT = F.KP768_FEAT_COUNT

_MAX_ACTIVE = 96


class NNUE(nn.Module):
    """3-layer sparse MLP. forward() takes active feature indices."""

    def __init__(self, feat_count, l0, l1):
        super().__init__()
        self.feat_count = feat_count
        self.fw = nn.Parameter(torch.randn(feat_count, l0) * 0.05)
        self.fb = nn.Parameter(torch.zeros(l0))
        self.w1 = nn.Linear(l0, l1)
        self.wo = nn.Linear(l1, 1)

    def forward(self, indices, mask):
        # indices: (B, max_active) int64, mask: (B, max_active) bool
        B, K = indices.shape
        fw = self.fw.index_select(0, indices.reshape(-1)).reshape(B, K, -1)
        acc = self.fb.unsqueeze(0) + (fw * mask.unsqueeze(-1)).sum(dim=1)
        h = torch.relu(self.w1(torch.relu(acc)))
        logit = self.wo(torch.relu(h)).squeeze(-1)
        return logit

    @torch.no_grad()
    def evaluate(self, board, stm=True):
        """Single-position logit from stm perspective (stm ignored for features)."""
        feats = F.features_halfkp(board) if False else F.features_kp768(board)
        idx = torch.tensor([feats], dtype=torch.long)
        mask = torch.ones(1, len(feats), dtype=torch.bool)
        return self.forward(idx, mask).item()

    @torch.no_grad()
    def evaluate_cp(self, board):
        return self.evaluate(board) * 400.0


def encode_batch(boards, feat=FEAT_KP768):
    """Build (indices, mask) tensors for a list of python-chess boards."""
    enc = F.features_kp768 if feat == FEAT_KP768 else F.features_halfkp
    seqs = [enc(b) for b in boards]
    K = max(len(s) for s in seqs) if seqs else 1
    idx = torch.zeros(len(seqs), K, dtype=torch.long)
    mask = torch.zeros(len(seqs), K, dtype=torch.bool)
    for i, s in enumerate(seqs):
        idx[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = True
    return idx, mask


def train_epoch(model, loader, opt, device, loss_mode="bce", scale=1.0):
    """One epoch. Labels: bce -> probability p (stm), mse -> centipawns/400."""
    model.train()
    total = 0.0
    for idx, mask, y in loader:
        idx, mask, y = idx.to(device), mask.to(device), y.to(device)
        logit = model(idx, mask)
        if loss_mode == "bce":
            loss = nn.functional.binary_cross_entropy_with_logits(logit, y)
        else:
            loss = nn.functional.mse_loss(logit, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * idx.shape[0]
    return total / max(1, len(loader.dataset))


def save_csnn(model, path, feat=FEAT_KP768):
    """Export trained weights to a CSNN file consumable by the Rust engine."""
    feat_count = model.feat_count
    l0 = model.fb.numel()
    l1 = model.wo.in_features
    w = model.state_dict()
    csnn.save(
        path,
        feat,
        l0,
        l1,
        feat_count,
        w["fw"].detach().cpu().numpy().reshape(-1).tolist(),
        w["fb"].detach().cpu().numpy().tolist(),
        w["w1.weight"].detach().cpu().numpy().reshape(-1).tolist(),
        w["w1.bias"].detach().cpu().numpy().tolist(),
        w["wo.weight"].detach().cpu().numpy().reshape(-1).tolist(),
        w["wo.bias"].item(),
    )