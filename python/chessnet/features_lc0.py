"""AlphaZero-style 105-plane feature encoder.

Layout (side-to-move oriented AND STM-orientation: black mirrored):
  Planes  0- 95: Piece history (12 planes × 8 frames, current + 7 past)
    12 piece types: own P,N,B,R,Q,K + enemy P,N,B,R,Q,K
  Planes  96- 97: Repetition count (position occurred >=1 / >=2 before)
  Plane   98:     Colour (1 = white to move)
  Plane   99:     Total move count (min(ply,512)/512)
  Planes 100-101: Own castling rights (K, Q)
  Planes 102-103: Enemy castling rights (K, Q)
  Plane  104:     No-progress count (halfmove_clock/100)
"""

import chess
import numpy as np

NUM_PLANES = 105
PLANE_SIZE = 64  # 8 * 8

# Piece type to plane offset (from side-to-move perspective)
# own pieces: 0-5, enemy pieces: 6-11
PIECE_PLANE = {
    (chess.PAWN, True): 0,
    (chess.KNIGHT, True): 1,
    (chess.BISHOP, True): 2,
    (chess.ROOK, True): 3,
    (chess.QUEEN, True): 4,
    (chess.KING, True): 5,
    (chess.PAWN, False): 6,
    (chess.KNIGHT, False): 7,
    (chess.BISHOP, False): 8,
    (chess.ROOK, False): 9,
    (chess.QUEEN, False): 10,
    (chess.KING, False): 11,
}


def _sq_to_idx(square: int) -> int:
    """Convert python-chess square to [0,63] index (a1=0, h8=63)."""
    return chess.square_rank(square) * 8 + chess.square_file(square)


def flip_square(square: int) -> int:
    """Mirror a square vertically (rank r -> 7-r). Black's board is flipped
    so the net always sees its own back rank at row 0 ("plays up")."""
    return (7 - chess.square_rank(square)) * 8 + chess.square_file(square)


# Past frames encoded (0 = current). 12 planes each: frames 1..HIST_FRAMES
# hold previous game positions (own/enemy relative to the CURRENT stm).
HIST_FRAMES = 7


def encode_position(board: chess.Board, history=()) -> np.ndarray:
    """Encode a python-chess Board as [105, 8, 8] float32 planes.

    Returns numpy array in CHW format (105 planes of 8x8).
    All planes are from the side-to-move perspective AND orientation:
    when black is to move, the board is mirrored vertically.

    history: iterable of previous chess.Board positions, OLDEST first
    (current board excluded); the last HIST_FRAMES are used.
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    stm = board.turn  # True = white to move
    flip = (stm == chess.BLACK)

    def put(plane_idx, sq):
        idx = _sq_to_idx(sq)
        if flip:
            idx = _sq_to_idx(flip_square(sq))
        r, c = divmod(idx, 8)
        planes[plane_idx, r, c] = 1.0

    # --- Piece planes (0-95): current + history frames ---
    frames = [board] + [h for h in list(history)[-HIST_FRAMES:]]
    for frame, b in enumerate(frames[: HIST_FRAMES + 1]):
        base = frame * 12
        for piece_type in chess.PIECE_TYPES:
            for color in (chess.WHITE, chess.BLACK):
                is_own = (color == stm)
                plane_idx = base + PIECE_PLANE[(piece_type, is_own)]
                for sq in b.pieces(piece_type, color):
                    put(plane_idx, sq)

    # --- Castling rights (100-103): own K/Q, enemy K/Q ---
    for color in (chess.WHITE, chess.BLACK):
        is_own = (color == stm)
        base = 100 if is_own else 102
        for has_right, offset in [
            (board.has_kingside_castling_rights(color), 0),
            (board.has_queenside_castling_rights(color), 1),
        ]:
            if has_right:
                planes[base + offset, :, :] = 1.0

    # --- Repetitions (96-97) ---
    # shredder_fen = board + turn + castling + ep (no clocks): exactly the
    # repetition-relevant parts, via public API (transposition_key() is
    # version-dependent in python-chess).
    past_keys = [h.shredder_fen() for h in history]
    cur_key = board.shredder_fen()
    n = sum(1 for k in past_keys if k == cur_key)
    if n >= 1:
        planes[96, :, :] = 1.0
    if n >= 2:
        planes[97, :, :] = 1.0

    # --- Colour (98): 1 = white to move ---
    if stm == chess.WHITE:
        planes[98, :, :] = 1.0

    # --- Move count (99): ply number scaled ---
    try:
        parts = board.fen().split()
        fullmove = int(parts[5])
        ply = (fullmove - 1) * 2 + (0 if stm == chess.WHITE else 1)
    except (IndexError, ValueError):
        ply = 0
    planes[99, :, :] = min(ply, 512) / 512.0

    # --- No-progress (104): halfmove clock scaled ---
    planes[104, :, :] = min(board.halfmove_clock, 100) / 100.0

    return planes


def encode_batch_boards(boards):
    """Encode a list of python-chess Boards as a [B, 105, 8, 8] tensor."""
    batch = np.stack([encode_position(b) for b in boards], axis=0)
    return batch
