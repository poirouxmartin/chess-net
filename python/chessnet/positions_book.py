"""Starting-position book: openings / middlegames / endgames.

Self-play from startpos only ever sees a razor-thin manifold of positions
(and a 1.Nh3 repertoire...), so the policy collapses onto universal moves.
Sampling starts across all game phases forces a position-dependent policy
AND feeds the value head real decisive endgames (conversion supervision).

Opening/middlegame entries are SAN move-lists played out at load: legal by
construction, dropped with a warning if anything is off. Endgames are FENs
validated by python-chess; mate entries are self-checked with find_mate.
"""
import random

import chess

OPENING_LINES = [
    ("ruy", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"]),
    ("italian", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3"]),
    ("sicilian", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"]),
    ("french", ["e4", "e6", "d4", "d5", "Nc3", "Nf6", "e5", "Nfd7"]),
    ("caro", ["e4", "c6", "d4", "d5", "exd5", "cxd5", "Nf3", "Nc6"]),
    ("qgd", ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7"]),
    ("kid", ["d4", "Nf6", "c4", "g6", "Nc3", "Bg7", "e4", "d6", "Nf3"]),
    ("english", ["c4", "e5", "Nc3", "Nf6", "Nf3", "Nc6", "g3", "d5"]),
    ("london", ["d4", "d5", "Bf4", "Nf6", "e3", "e6", "Nf3", "c5", "c3", "Nc6"]),
    ("scotch", ["e4", "e5", "Nf3", "Nc6", "d4", "exd4", "Nxd4", "Nf6"]),
]

# Longer book lines (12-18 plies): real middlegames, still legal by play-out.
MIDDLEGAME_LINES = [
    ("ruy-mid", ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O",
                 "Be7", "Re1", "b5", "Bb3", "d6", "c3", "O-O", "h3"]),
    ("italian-mid", ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d3",
                     "d6", "O-O", "O-O", "Re1", "a6", "Bb3"]),
    ("sicilian-mid", ["e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6",
                      "Nc3", "a6", "Be3", "e5", "Nb3", "Be7", "Be2", "O-O", "O-O"]),
    ("qgd-mid", ["d4", "d5", "c4", "e6", "Nc3", "Nf6", "Bg5", "Be7", "e3",
                 "O-O", "Nf3", "h6", "Bh4", "b6", "cxd5", "Nxd5"]),
]

ENDGAME_FENS = [
    # (tag, fen, theoretical white-POV outcome or None if unknown/drawn)
    ("KQvK-w", "8/8/8/4k3/8/8/5Q2/4K3 w - - 0 1", 1.0),
    ("KQvK-b", "8/8/8/4k3/8/8/5Q2/4K3 b - - 0 1", 1.0),
    ("KRvK-w", "8/8/8/4k3/8/8/5R2/4K3 w - - 0 1", 1.0),
    ("KRvK-b", "8/8/8/4k3/8/8/5R2/4K3 b - - 0 1", 1.0),
    ("KPvK-w", "8/8/8/3k4/8/8/3P4/3K4 w - - 0 1", None),
    ("KPvK-b", "8/8/8/3k4/8/8/3P4/3K4 b - - 0 1", None),
    ("RvR-pawn", "8/8/5k2/8/8/5R2/5P2/5K2 w - - 0 1", None),
    ("pawn-race", "8/2p5/8/4k3/8/8/2P5/4K3 w - - 0 1", None),
    ("bishops-draw", "8/8/8/4k3/8/3b4/3B4/3K4 w - - 0 1", None),
]

# Mating nets: verified at load with find_mate (the wrapper plays the mate,
# the hard target teaches the policy). Candidates that are not actually
# mate-in-1 are dropped.
MATE_FENS = [
    ("scholar", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1"),
    ("back-rank", "6k1/5ppp/8/8/8/8/5PPP/5RK1 w - - 0 1"),
    ("arabian", "7k/8/5R2/8/8/8/5N2/6K1 w - - 0 1"),
]


def _play_out(name, sans):
    b = chess.Board()
    try:
        for s in sans:
            b.push_san(s)
    except (chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
        print(f"  [book] dropping illegal line {name}", flush=True)
        return None
    if b.is_game_over():
        print(f"  [book] dropping finished line {name}", flush=True)
        return None
    return b.fen()


def _massacre_variants(fen, tag):
    """Derive materially-unbalanced starts by removing one enemy piece.
    Programmatic (never hand-written FENs). Returns [(fen, tag, value)]."""
    out = []
    try:
        b = chess.Board(fen)
    except ValueError:
        return out
    for color, win in ((chess.BLACK, 1.0), (chess.WHITE, 0.0)):
        for pt in (chess.QUEEN, chess.ROOK, chess.KNIGHT, chess.BISHOP):
            sqs = list(b.pieces(pt, color))
            if not sqs:
                continue
            sq = sqs[len(sqs) // 2]
            b2 = b.copy()
            b2.remove_piece_at(sq)
            if b2.is_valid() and not b2.is_game_over():
                pname = chess.piece_name(pt)
                out.append((b2.fen(), f"mass-{tag}-{pname}", win))
    return out


def build_book():
    """Returns {tier: [(fen, tag, value)]}. value = theoretical white-POV
    outcome or None. Never raises: bad entries are dropped."""
    book = {"opening": [], "middlegame": [], "endgame": [], "massacre": []}
    for name, sans in OPENING_LINES:
        fen = _play_out(name, sans)
        if fen:
            book["opening"].append((fen, f"op-{name}", None))
            for mf, mt, mv in _massacre_variants(fen, f"op-{name}"):
                book["massacre"].append((mf, mt, mv))
    for name, sans in MIDDLEGAME_LINES:
        fen = _play_out(name, sans)
        if fen:
            book["middlegame"].append((fen, f"mid-{name}", None))
            for mf, mt, mv in _massacre_variants(fen, f"mid-{name}"):
                book["massacre"].append((mf, mt, mv))
    for tag, fen, value in ENDGAME_FENS:
        try:
            b = chess.Board(fen)
            if b.is_valid() and not b.is_game_over():
                book["endgame"].append((fen, f"eg-{tag}", value))
            else:
                print(f"  [book] dropping endgame {tag}", flush=True)
        except ValueError:
            print(f"  [book] dropping bad fen {tag}", flush=True)
    try:
        from .fastbatch import find_mate
        for tag, fen in MATE_FENS:
            try:
                b = chess.Board(fen)
                if find_mate(b) is not None:
                    v = 1.0 if b.turn == chess.WHITE else 0.0
                    book["endgame"].append((fen, f"mate-{tag}", v))
                else:
                    print(f"  [book] not actually mate: {tag}", flush=True)
            except ValueError:
                print(f"  [book] dropping bad fen {tag}", flush=True)
    except ImportError:
        pass
    # Mirror ~half for black-to-move balance (validated like the rest:
    # a mirrored opening can in principle be terminal or illegal).
    for tier in ("opening", "middlegame"):
        mirrored = []
        for fen, tag, _v in book[tier][: len(book[tier]) // 2]:
            try:
                b = chess.Board(fen).mirror()
                if b.is_valid() and not b.is_game_over():
                    mirrored.append((b.fen(), tag + "-m", None))
                else:
                    print(f"  [book] dropping bad mirror {tag}", flush=True)
            except ValueError:
                print(f"  [book] dropping bad mirror {tag}", flush=True)
        book[tier].extend(mirrored)
    total = sum(len(v) for v in book.values())
    print(f"  [book] {total} starts: " +
          ", ".join(f"{k}={len(v)}" for k, v in book.items()), flush=True)
    return book


def sample_start(book, rng, mix=(0.25, 0.25, 0.2, 0.3)):
    """Stratified sample over tiers in book order. mix length must match.
    Returns (fen, tag, value). Falls back to startpos if a tier is empty."""
    tiers = [t for t in book if book.get(t)]
    if not tiers:
        return chess.STARTING_FEN, "startpos", None
    weights = list(mix[: len(tiers)])
    tot = sum(weights) or 1.0
    r = rng.random() * tot
    acc = 0.0
    for tier, w in zip(tiers, weights):
        acc += w
        if r <= acc:
            entries = book[tier]
            return rng.choice(entries) if entries else (chess.STARTING_FEN, "startpos", None)
    entries = book[tiers[-1]]
    return rng.choice(entries) if entries else (chess.STARTING_FEN, "startpos", None)
