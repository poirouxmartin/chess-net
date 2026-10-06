"""ResNet chess engine network (AlphaZero-style: 20x256, no SE).

Architecture:
  Input:   105 planes × 8×8 (8 history frames + rules planes, STM-oriented)
  Stem:    Conv2d(105→C, 3×3) + BN + ReLU
  Body:    N × ResBlock(C), SE optional (off by default, AZ has none)
  Policy:  Conv2d(C→64, 1×1) + Conv2d(C→3, 1×1), flatten+concat → 4288
  Value:   Conv2d(C→8, 1×1) + Flatten + FC(512→256) + FC(256→1) + tanh

The network outputs:
  - policy_logits: [B, 4288] move logits (STM-oriented + underpromo slices)
  - value: [B, 1] tanh in [-1, 1], side-to-move frame (AZ convention)

Score in centipawns = value * 400 (STM frame; negate for white POV).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .features_lc0 import NUM_PLANES  # single source of truth (105)

# Move encoding: from*64+to on the side-to-move-oriented board (see
# features_lc0.flip_square: black's board is mirrored so the net always
# plays "up"). Queen-promotion shares the base index; underpromotions
# (N/B/R) live in their own 64-slot slices past 4096.
PROMO_BASE = 4096
POLICY_SIZE = 4096 + 3 * 64  # = 4288


def policy_index(board, mv):
    """Move -> policy index, side-to-move oriented (flip when black)."""
    import chess
    from .features_lc0 import flip_square
    f = mv.from_square
    t = mv.to_square
    if board.turn == chess.BLACK:
        f = flip_square(f)
        t = flip_square(t)
    base = f * 64 + t
    if mv.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        return PROMO_BASE + (mv.promotion - 2) * 64 + f
    return base


def index_to_move(board, idx):
    """Inverse of policy_index: (from_square, to_square, promotion|None)
    in ABSOLUTE squares."""
    import chess
    from .features_lc0 import flip_square
    if idx >= PROMO_BASE:
        k = idx - PROMO_BASE
        promo = (k // 64) + 2
        f = k % 64
        if board.turn == chess.BLACK:
            f = flip_square(f)
        # to-square of an underpromotion is implied by the from-square
        # only up to file: resolve by scanning legal moves.
        for mv in board.legal_moves:
            if (mv.from_square == f and mv.promotion == promo
                    and policy_index(board, mv) == idx):
                return mv.from_square, mv.to_square, mv.promotion
        return f, -1, promo
    f, t = divmod(idx, 64)
    if board.turn == chess.BLACK:
        f = flip_square(f)
        t = flip_square(t)
    # A pawn stepping onto the last rank is always a promotion: the base
    # index then means queen (underpromotions have their own slice).
    for mv in board.legal_moves:
        if mv.from_square == f and mv.to_square == t:
            return f, t, mv.promotion
    return f, t, None


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""

    def __init__(self, channels, se_ratio=4):
        super().__init__()
        se_channels = channels // se_ratio
        self.fc1 = nn.Linear(channels, se_channels)
        self.fc2 = nn.Linear(se_channels, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        # Squeeze: global average pooling
        h = x.mean(dim=(2, 3))  # [B, C]
        h = F.relu(self.fc1(h))  # [B, se_channels]
        h = torch.sigmoid(self.fc2(h))  # [B, C]
        return x * h.unsqueeze(-1).unsqueeze(-1)


class ResBlock(nn.Module):
    """Residual block, SE optional (AlphaZero has none)."""

    def __init__(self, channels, se_ratio=4, use_se=True):
        super().__init__()
        self.use_se = use_se
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        if use_se:
            self.se = SEBlock(channels, se_ratio)

    def forward(self, x):
        residual = x
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if self.use_se:
            h = self.se(h)
        return F.relu(residual + h)


class ResNetSE(nn.Module):
    """Full ResNet+SE network for chess.

    Input:  [B, 105, 8, 8] float32 planes
    Output: policy_logits [B, policy_size], value [B, 1] tanh scalar
    """

    def __init__(self, in_planes=NUM_PLANES, channels=224, num_blocks=16,
                 se_ratio=4, policy_size=POLICY_SIZE, use_se=True):
        super().__init__()
        # The conv head bakes in the 64+3 plane layout: no other size fits.
        assert policy_size == POLICY_SIZE, policy_size
        self.channels = channels
        self.policy_size = policy_size

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Residual tower
        self.tower = nn.Sequential(
            *[ResBlock(channels, se_ratio, use_se) for _ in range(num_blocks)]
        )

        # Policy head: fully convolutional (AlphaZero-style). 64 planes =
        # to-square maps, one per from-square (64*64 = 4096 flat indices
        # f*64+t); 3 planes = underpromotion from-square maps, one per
        # piece N/B/R (3*64 = 192 flat indices 4096+k*64+f). Flattened and
        # concatenated downstream = the exact 4288 layout in policy_index.
        self.policy_to = nn.Sequential(
            nn.Conv2d(channels, 64, 1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.policy_under = nn.Sequential(
            nn.Conv2d(channels, 3, 1, bias=False),
            nn.BatchNorm2d(3),
        )

        # Value head: Conv reduces channels, then FC to scalar tanh
        # (AlphaZero: v in [-1,1] from side-to-move perspective).
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(8 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, 105, 8, 8] input planes
        Returns:
            policy_logits: [B, policy_size] move logits
            value: [B, 1] tanh in [-1, 1], side-to-move frame
        """
        # Stem + tower
        h = self.stem(x)
        h = self.tower(h)

        # Policy (raw logits, no activation: negatives are legal scores).
        to_maps = self.policy_to(h)        # [B, 64, 8, 8]
        under_maps = self.policy_under(h)  # [B, 3, 8, 8]
        policy_logits = torch.cat(
            [to_maps.flatten(1), under_maps.flatten(1)], dim=1)  # [B, 4288]

        # Value
        v = self.value_head(h)
        v = v.view(v.size(0), -1)
        value = torch.tanh(self.value_fc(v))  # [B, 1] in [-1, 1], STM frame

        return policy_logits, value

    def forward_value(self, x):
        """Value only (for evaluation in search)."""
        _, value = self.forward(x)
        return value

    def forward_policy(self, x):
        """Policy only (for MCTS prior)."""
        policy_logits, _ = self.forward(x)
        return policy_logits

    @torch.no_grad()
    def evaluate(self, board):
        """Single-position evaluation. Returns (score_cp, policy_probs)."""
        from .features_lc0 import encode_position
        import numpy as np

        planes = encode_position(board)
        x = torch.tensor(planes, dtype=torch.float32).unsqueeze(0)
        if next(self.parameters()).is_cuda:
            x = x.cuda()

        policy_logits, value = self.forward(x)

        # Score in centipawns, WHITE point of view.
        import chess

        v = value.item()
        if board.turn != chess.WHITE:
            v = -v
        score_cp = v * 400.0

        # Policy probs
        policy_probs = F.softmax(policy_logits, dim=-1).squeeze(0)

        return score_cp, policy_probs.cpu().numpy()

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(channels=256, num_blocks=20, use_se=False):
    """Create a ResNet model (AlphaZero: 20x256, no SE)."""
    model = ResNetSE(channels=channels, num_blocks=num_blocks, use_se=use_se)
    print(f"Created ResNet: {model.count_parameters():,} parameters")
    print(f"  Channels: {channels}, Blocks: {num_blocks}, SE: {use_se}")
    print(f"  Policy: {POLICY_SIZE} moves (STM-oriented)")
    print(f"  Value: tanh scalar [-1, 1] (STM frame)")
    return model


if __name__ == "__main__":
    model = create_model()
    # Quick sanity check
    x = torch.randn(2, NUM_PLANES, 8, 8)
    policy, value = model(x)
    print(f"\nSanity check:")
    print(f"  Input:  {x.shape}")
    print(f"  Policy: {policy.shape}")
    print(f"  Value:  {value.shape}")
    print(f"  Params: {model.count_parameters():,}")
