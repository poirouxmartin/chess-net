"""Lc0 classical-112 encoder + teacher session + 1858->4288 policy map.

Ground truth:
  - input layout: lc0 src/neural/encoder.cc (INPUT_CLASSICAL_112_PLANE)
  - black orientation (rank-flip + color-swap) + 4 policy dicts: verified
    against patrik-ha/minimal-lc0-for-research (see teacher/lc0_policy.json,
    built by python/build_policy.py)
  - bit-correctness: proven by test_lc0_112.py (differential vs reference)

Layout [112,8,8], plane[r,c] with r=0 = rank 1:
  0-103:  history, 8 frames x 13 planes (ours P/N/B/R/Q/K, theirs P..K,
           repetition>=1). Frame 0 = current. All frames from the CURRENT
           side-to-move perspective; when black is to move every frame is
           rank-flipped (r -> 7-r).
  104:    our queenside castling | 105: our kingside
  106:    their queenside        | 107: their kingside
  108:    side to move (1 = black) | 109: halfmove clock RAW
  110:    zeros | 111: ones (board edges)
"""

import json
import os

import chess
import numpy as np
import onnxruntime as ort

NUM_PLANES = 112
HISTORY_FRAMES = 8

_TABLES = None


def _tables():
    global _TABLES
    if _TABLES is None:
        path = os.path.join(os.path.dirname(__file__), "../../teacher/lc0_policy.json")
        with open(path) as f:
            _TABLES = json.load(f)
    return _TABLES


def policy_dict(board):
    """uci -> 1858 index dict for this position (castling/black variants)."""
    t = _tables()
    idx = (1 if board.has_castling_rights(board.turn) else 0)
    idx += 2 * (0 if board.turn == chess.WHITE else 1)
    return t[["wn", "wc", "bn", "bc"][idx]]


def lc0_index(board, move):
    """chess.Move -> 1858 policy index (None if unmapped).

    Knight underpromotions have no Lc0 slot (upstream wart, ~0.01% of
    moves): callers should remap them to the queen-promo index.
    """
    uci = move.uci()
    d = policy_dict(board)
    if uci in d:
        return d[uci]
    if uci[-1] == "n":
        return d.get(uci[:-1] + "q")
    return None


def _sq_planes(bb, flip):
    """SquareSet -> 8x8 float plane (r=0 is rank 1; rank-flip if black stm)."""
    plane = np.zeros((8, 8), dtype=np.float32)
    for sq in bb:
        r, c = divmod(sq, 8)
        if flip:
            r = 7 - r
        plane[r, c] = 1.0
    return plane


def encode_112(board, history=()):
    """Encode (board + past FENs, oldest first, current excluded).

    Repetitions use the transposition key (pieces + turn + castling + EP),
    like Lc0's counter: 1 if the frame state appeared earlier in the game.
    """
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    stm = board.turn
    flip = (stm == chess.BLACK)
    past = [chess.Board(h) for h in list(history)[-7:]]
    frames = past + [board]
    keys = [b._transposition_key() for b in frames]
    for i, b in enumerate(frames):
        base = (len(frames) - 1 - i) * 13
        for pt in range(6):
            planes[base + pt] = _sq_planes(b.pieces(pt + 1, stm), flip)
            planes[base + 6 + pt] = _sq_planes(b.pieces(pt + 1, not stm), flip)
        if keys[i] in keys[:i]:
            planes[base + 12, :, :] = 1.0
    # Aux planes (current position only).
    if board.has_queenside_castling_rights(stm):
        planes[104, :, :] = 1.0
    if board.has_kingside_castling_rights(stm):
        planes[105, :, :] = 1.0
    if board.has_queenside_castling_rights(not stm):
        planes[106, :, :] = 1.0
    if board.has_kingside_castling_rights(not stm):
        planes[107, :, :] = 1.0
    if stm == chess.BLACK:
        planes[108, :, :] = 1.0
    planes[109, :, :] = float(board.halfmove_clock)
    planes[111, :, :] = 1.0
    return planes


class TeacherSession:
    """ORT wrapper around the converted Lc0 teacher (policy/WDL heads)."""

    def __init__(self, path=None, batch=1024):
        if path is None:
            path = os.path.join(os.path.dirname(__file__),
                                "../../teacher/t1-256x10.onnx")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        outs = {o.name: o for o in self.sess.get_outputs()}
        self.pol_name = [n for n in outs if "policy" in n][0]
        self.wdl_name = [n for n in outs if "wdl" in n][0]
        self.batch = batch

    def query(self, planes):
        """[N,112,8,8] float32 -> (policy_logits [N,1858], wdl [N,3])."""
        pol, wdl = [], []
        for i in range(0, len(planes), self.batch):
            x = np.ascontiguousarray(planes[i:i + self.batch], dtype=np.float32)
            outs = self.sess.run(None, {self.in_name: x})
            names = [o.name for o in self.sess.get_outputs()]
            d = dict(zip(names, outs))
            pol.append(np.asarray(d[self.pol_name]))
            wdl.append(np.asarray(d[self.wdl_name]))
        return np.concatenate(pol), np.concatenate(wdl)
