//! Perft: move counting for correctness validation.
//!
//! Uses in-place make/unmake without zobrist updates (perft needs no keys,
//! halfmove clocks or repetition detection) plus bulk counting at depth 2.

use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use crate::bitboard::bit;
use crate::move_::{
    Move, FLAG_CAPTURE, FLAG_CASTLE_KS, FLAG_CASTLE_QS, FLAG_DOUBLE, FLAG_EN_PASSANT, FLAG_PROMO,
    FLAG_PROMO_CAPTURE,
};
use crate::movegen::generate_legal;
use crate::position::{CASTLE_CLEAR, KING, PAWN, PIECE_TYPES, Position, ROOK, WHITE};

/// Count legal move paths. `perft(pos, 1)` == number of legal moves.
pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = generate_legal(pos);
    if depth == 1 {
        return moves.len as u64;
    }
    let mut nodes = 0;
    for m in moves.iter() {
        let undo = make_fast(pos, m);
        nodes += if depth == 2 {
            generate_legal(pos).len as u64
        } else {
            perft(pos, depth - 1)
        };
        unmake_fast(pos, undo);
    }
    nodes
}

/// Parallel perft: split the root move list across `threads` workers.
pub fn perft_parallel(pos: &mut Position, depth: u32, threads: usize) -> u64 {
    if depth <= 1 || threads <= 1 {
        return perft(pos, depth);
    }
    let moves = generate_legal(pos);
    let root = *pos;
    let total = AtomicU64::new(0);
    thread::scope(|s| {
        let chunks = moves.len.div_ceil(threads);
        for chunk in moves.moves[..moves.len].chunks(chunks) {
            let total = &total;
            s.spawn(move || {
                let mut local = root;
                let mut n = 0;
                for &m in chunk {
                    let undo = make_fast(&mut local, m);
                    n += if depth == 2 {
                        generate_legal(&local).len as u64
                    } else {
                        perft(&mut local, depth - 1)
                    };
                    unmake_fast(&mut local, undo);
                }
                total.fetch_add(n, Ordering::Relaxed);
            });
        }
    });
    total.load(Ordering::Relaxed)
}

pub fn divide(pos: &mut Position, depth: u32) -> u64 {
    let moves = generate_legal(pos);
    let mut total = 0;
    for m in moves.iter() {
        let n = if depth <= 1 {
            1
        } else {
            let undo = make_fast(pos, m);
            let n = perft(pos, depth - 1);
            unmake_fast(pos, undo);
            n
        };
        println!("{}: {}", m.to_uci(), n);
        total += n;
    }
    total
}

struct FastUndo {
    m: Move,
    captured: Option<(usize, usize)>,
    castle: u8,
    ep: Option<usize>,
}

/// Make a move in place, updating only what perft/movegen need.
#[inline(always)]
fn make_fast(pos: &mut Position, m: Move) -> FastUndo {
    let us = pos.side;
    let them = us ^ 1;
    let from = m.from();
    let to = m.to();
    let flags = m.flags();
    let promo = flags == FLAG_PROMO || flags == FLAG_PROMO_CAPTURE;

    let mut undo = FastUndo {
        m,
        captured: None,
        castle: pos.castle,
        ep: pos.ep,
    };

    let from_pt = piece_pt_at(pos, from);
    let moving_pt = if promo { m.promo_pt() } else { from_pt };
    let base_pt = if promo { PAWN } else { from_pt };

    pos.pieces[us * PIECE_TYPES + base_pt] &= !bit(from);
    pos.occ &= !bit(from);

    if flags == FLAG_EN_PASSANT {
        let cap_sq = if us == WHITE { to - 8 } else { to + 8 };
        pos.pieces[them * PIECE_TYPES + PAWN] &= !bit(cap_sq);
        pos.occ &= !bit(cap_sq);
        undo.captured = Some((them, PAWN));
    } else if flags == FLAG_CAPTURE || flags == FLAG_PROMO_CAPTURE {
        let cap_pt = piece_pt_at(pos, to);
        pos.pieces[them * PIECE_TYPES + cap_pt] &= !bit(to);
        pos.occ &= !bit(to);
        undo.captured = Some((them, cap_pt));
    }

    pos.pieces[us * PIECE_TYPES + moving_pt] |= bit(to);
    pos.occ |= bit(to);
    if moving_pt == KING {
        pos.king_sq[us] = to;
    }

    if flags == FLAG_CASTLE_KS {
        let (rf, rt) = if us == WHITE { (7, 5) } else { (63, 61) };
        move_piece(pos, us, rf, rt, ROOK);
    } else if flags == FLAG_CASTLE_QS {
        let (rf, rt) = if us == WHITE { (0, 3) } else { (56, 59) };
        move_piece(pos, us, rf, rt, ROOK);
    }

    pos.castle &= !(CASTLE_CLEAR[from] | CASTLE_CLEAR[to]);
    pos.ep = if flags == FLAG_DOUBLE { Some((from + to) / 2) } else { None };
    pos.side = them;
    undo
}

#[inline(always)]
fn unmake_fast(pos: &mut Position, u: FastUndo) {
    let mover = pos.side ^ 1;
    let m = u.m;
    let from = m.from();
    let to = m.to();
    let flags = m.flags();
    let promo = flags == FLAG_PROMO || flags == FLAG_PROMO_CAPTURE;

    let moving_pt = if promo { m.promo_pt() } else { piece_pt_at(pos, to) };

    pos.pieces[mover * PIECE_TYPES + moving_pt] &= !bit(to);
    pos.occ &= !bit(to);

    if flags == FLAG_EN_PASSANT {
        let cap_sq = if mover == WHITE { to - 8 } else { to + 8 };
        pos.pieces[(mover ^ 1) * PIECE_TYPES + PAWN] |= bit(cap_sq);
        pos.occ |= bit(cap_sq);
    } else if let Some((c, pt)) = u.captured {
        pos.pieces[c * PIECE_TYPES + pt] |= bit(to);
        pos.occ |= bit(to);
    }

    let restore_pt = if promo { PAWN } else { moving_pt };
    pos.pieces[mover * PIECE_TYPES + restore_pt] |= bit(from);
    pos.occ |= bit(from);
    if restore_pt == KING {
        pos.king_sq[mover] = from;
    }

    if flags == FLAG_CASTLE_KS {
        let (rt, rf) = if mover == WHITE { (7, 5) } else { (63, 61) };
        move_piece(pos, mover, rf, rt, ROOK);
    } else if flags == FLAG_CASTLE_QS {
        let (rt, rf) = if mover == WHITE { (0, 3) } else { (56, 59) };
        move_piece(pos, mover, rf, rt, ROOK);
    }

    pos.castle = u.castle;
    pos.ep = u.ep;
    pos.side = mover;
}

#[inline(always)]
fn piece_pt_at(pos: &Position, sq: usize) -> usize {
    let b = bit(sq);
    for pt in 0..PIECE_TYPES {
        if pos.pieces[pt] & b != 0 {
            return pt;
        }
    }
    for pt in 0..PIECE_TYPES {
        if pos.pieces[PIECE_TYPES + pt] & b != 0 {
            return pt;
        }
    }
    KING // unreachable for an occupied square
}

#[inline(always)]
fn move_piece(pos: &mut Position, color: usize, from: usize, to: usize, pt: usize) {
    let idx = color * PIECE_TYPES + pt;
    let b = bit(from);
    pos.pieces[idx] = (pos.pieces[idx] & !b) | bit(to);
    pos.occ = (pos.occ & !b) | bit(to);
    if pt == KING {
        pos.king_sq[color] = to;
    }
}
