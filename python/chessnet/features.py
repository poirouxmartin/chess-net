"""Feature encoders aligned with the Rust engine (crates/nn/src/lib.rs).

Indexing conventions (mirror the Rust code exactly):
  sq = rank * 8 + file, a1 = 0, h8 = 63
  piece types: PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5

HalfKP (feat=1), feature_count = 2 * 64 * 641 = 82048.
  For each king perspective color p (0=white, 1=black), base = king_sq[p] * 641:
    own non-king piece type li (0..5):  base + li*64 + sq
    enemy non-king piece type li (0..5): base + (5+li)*64 + sq
    kings constant:                     base + 640
  Both perspectives are always emitted (no side-to-move dependency).

KP768 (feat=2), feature_count = 768.
  Every piece: index = color*6*64 + pt*64 + sq.
"""

import chess

FEAT_HALFKP = 1
FEAT_KP768 = 2
HALFKP_PER_KING = 641
HALFKP_FEAT_COUNT = 2 * 64 * HALFKP_PER_KING
KP768_FEAT_COUNT = 768

PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]


def sq_index(square):
    return chess.square_rank(square) * 8 + chess.square_file(square)


def features_halfkp(board):
    """Return list of active feature indices (HalfKP, feat=1)."""
    out = []
    for color in (chess.WHITE, chess.BLACK):
        king_sq = sq_index(board.king(color))
        base = king_sq * HALFKP_PER_KING
        for li, pt in enumerate(PIECE_ORDER):
            for s in board.pieces(pt, color):
                out.append(base + li * 64 + sq_index(s))
            for s in board.pieces(pt, not color):
                out.append(base + (5 + li) * 64 + sq_index(s))
        out.append(base + HALFKP_PER_KING - 1)
    return out


def features_kp768(board):
    """Return list of active feature indices (KP768, feat=2)."""
    out = []
    for color in (chess.WHITE, chess.BLACK):
        rust_color = 0 if color == chess.WHITE else 1
        for pt in range(chess.PAWN, chess.KING + 1):
            for s in board.pieces(pt, color):
                out.append(rust_color * 6 * 64 + (pt - 1) * 64 + sq_index(s))
    return out