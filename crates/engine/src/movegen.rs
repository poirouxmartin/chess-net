//! Legal move generation with pin detection and check-evasion filtering.

use crate::attack::{between, king_attacks, knight_attacks, line, pawn_attacks, step_sq};
use crate::bitboard::{bit, pop_lsb, rank_of};
use crate::move_::*;
use crate::position::*;

pub const MAX_MOVES: usize = 256;

#[derive(Clone, Copy)]
pub struct MoveList {
    pub moves: [Move; MAX_MOVES],
    pub len: usize,
}

impl MoveList {
    pub fn new() -> Self {
        MoveList {
            moves: [Move::null(); MAX_MOVES],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        self.moves[self.len] = m;
        self.len += 1;
    }

    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = Move> + '_ {
        self.moves[..self.len].iter().copied()
    }

    #[inline(always)]
    pub fn get(&self, i: usize) -> Move {
        self.moves[i]
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate all legal moves for the side to move.
pub fn generate_legal(pos: &Position) -> MoveList {
    let us = pos.side;
    let them = us ^ 1;
    let k = pos.king_sq[us];
    let occ = pos.occ;
    let occ_no_king = occ & !bit(k);
    let enemies = pos.pieces_of(them);
    let empty = !occ;
    let targets = empty | enemies;
    let pinned = pos.pinned();
    let checkers = pos.checkers();
    let in_check = checkers != 0;
    let double_check = checkers.count_ones() > 1;
    let checker_sq = if checkers != 0 {
        pop_lsb(&mut checkers.clone())
    } else {
        0
    };
    let between_block = if in_check && !double_check {
        between(k, checker_sq)
    } else {
        0
    };

    let mut list = MoveList::new();

    // ---- King moves ----
    let mut kb = king_attacks(k) & targets;
    while kb != 0 {
        let sq = pop_lsb(&mut kb);
        if !pos.square_attacked(sq, them, occ_no_king) {
            let flags = if enemies & bit(sq) != 0 { FLAG_CAPTURE } else { FLAG_QUIET };
            list.push(Move::new(k, sq, 0, flags).with_piece(KING));
        }
    }

    if double_check {
        return list;
    }

    // ---- Castling ----
    if !in_check {
        gen_castling(pos, &mut list, occ, occ_no_king, them);
    }

    // ---- Pawns ----
    let mut pb = pos.piece_bb(us, PAWN);
    let forward: i32 = if us == WHITE { 8 } else { -8 };
    let promo_rank: usize = if us == WHITE { 7 } else { 0 };
    let start_rank: usize = if us == WHITE { 1 } else { 6 };
    while pb != 0 {
        let sq = pop_lsb(&mut pb);
        let pinned_sq = pinned & bit(sq) != 0;
        let on_line = |to: usize| !pinned_sq || line(k, sq) & bit(to) != 0;

        // Pushes.
        if let Some(ps) = step_sq(sq, forward) {
            if empty & bit(ps) != 0 {
                if rank_of(ps) == promo_rank {
                    if on_line(ps) {
                        for p in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT] {
                            list.push(Move::new(sq, ps, p, FLAG_PROMO).with_piece(PAWN));
                        }
                    }
                } else if on_line(ps) {
                    list.push(Move::new(sq, ps, 0, FLAG_QUIET).with_piece(PAWN));
                    if rank_of(sq) == start_rank {
                        if let Some(ps2) = step_sq(ps, forward) {
                            if empty & bit(ps2) != 0 && on_line(ps2) {
                                list.push(Move::new(sq, ps2, 0, FLAG_DOUBLE).with_piece(PAWN));
                            }
                        }
                    }
                }
            }
        }

        // Captures.
        let mut cb = pawn_attacks(us, sq) & enemies;
        while cb != 0 {
            let t = pop_lsb(&mut cb);
            if on_line(t) {
                if rank_of(t) == promo_rank {
                    for p in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT] {
                        list.push(Move::new(sq, t, p, FLAG_PROMO_CAPTURE).with_piece(PAWN));
                    }
                } else {
                    list.push(Move::new(sq, t, 0, FLAG_CAPTURE).with_piece(PAWN));
                }
            }
        }

        // En passant.
        if let Some(ep) = pos.ep {
            if pawn_attacks(us, sq) & bit(ep) != 0 {
                let m = Move::new(sq, ep, 0, FLAG_EN_PASSANT).with_piece(PAWN);
                if ep_legal(pos, m) {
                    list.push(m);
                }
            }
        }
    }

    // ---- Knights ----
    let mut kb = pos.piece_bb(us, KNIGHT);
    while kb != 0 {
        let sq = pop_lsb(&mut kb);
        if pinned & bit(sq) != 0 {
            continue;
        }
        let mut tb = knight_attacks(sq) & targets;
        while tb != 0 {
            let t = pop_lsb(&mut tb);
            push_simple(&mut list, sq, t, enemies, KNIGHT);
        }
    }

    // ---- Bishops / rooks / queens ----
    for pt in [BISHOP, ROOK, QUEEN] {
        let mut sb = pos.piece_bb(us, pt);
        while sb != 0 {
            let sq = pop_lsb(&mut sb);
            let mut tb = sliding_attacks(pt, sq, occ) & targets;
            if pinned & bit(sq) != 0 {
                tb &= line(k, sq);
            }
            while tb != 0 {
                let t = pop_lsb(&mut tb);
                push_simple(&mut list, sq, t, enemies, pt);
            }
        }
    }

    // ---- Check filtering: non-king moves must capture the checker or block ----
    if in_check {
        let mut out = 0usize;
        for i in 0..list.len {
            let m = list.get(i);
            let is_king = m.from() == k;
            // An en passant capture removes the pawn on to +/- 8 (not `to`).
            let ep_captures_checker = m.is_en_passant()
                && (m.to() == checker_sq.wrapping_add(8)
                    || m.to() == checker_sq.wrapping_sub(8));
            let ok = is_king
                || m.to() == checker_sq
                || between_block & bit(m.to()) != 0
                || ep_captures_checker;
            if ok {
                list.moves[out] = m;
                out += 1;
            }
        }
        list.len = out;
    }

    list
}

/// Count legal moves without building a move list. Used by perft leaf nodes
/// (bulk counting), which dominate the node budget.
#[inline(always)]
pub fn count_legal(pos: &Position) -> usize {
    let us = pos.side;
    let them = us ^ 1;
    let k = pos.king_sq[us];
    let occ = pos.occ;
    let occ_no_king = occ & !bit(k);
    let enemies = pos.pieces_of(them);
    let empty = !occ;
    let pinned = pos.pinned();
    let checkers = pos.checkers();
    let in_check = checkers != 0;
    let double_check = checkers.count_ones() > 1;
    let mut n = 0;

    // ---- King moves ----
    let mut kb = king_attacks(k) & (empty | enemies);
    while kb != 0 {
        let sq = pop_lsb(&mut kb);
        if !pos.square_attacked(sq, them, occ_no_king) {
            n += 1;
        }
    }

    if double_check {
        return n;
    }

    // ---- Castling ----
    if !in_check {
        n += count_castling(pos, occ, occ_no_king, them);
    }

    // ---- Pawns ----
    let mut pb = pos.piece_bb(us, PAWN);
    let forward: i32 = if us == WHITE { 8 } else { -8 };
    let promo_rank: usize = if us == WHITE { 7 } else { 0 };
    let start_rank: usize = if us == WHITE { 1 } else { 6 };
    while pb != 0 {
        let sq = pop_lsb(&mut pb);
        let pinned_sq = pinned & bit(sq) != 0;
        let on_line = |to: usize| !pinned_sq || line(k, sq) & bit(to) != 0;

        if let Some(ps) = step_sq(sq, forward) {
            if empty & bit(ps) != 0 {
                if rank_of(ps) == promo_rank {
                    if on_line(ps) {
                        n += 4;
                    }
                } else if on_line(ps) {
                    n += 1;
                    if rank_of(sq) == start_rank {
                        if let Some(ps2) = step_sq(ps, forward) {
                            if empty & bit(ps2) != 0 && on_line(ps2) {
                                n += 1;
                            }
                        }
                    }
                }
            }
        }

        let mut cb = pawn_attacks(us, sq) & enemies;
        while cb != 0 {
            let t = pop_lsb(&mut cb);
            if on_line(t) {
                n += if rank_of(t) == promo_rank { 4 } else { 1 };
            }
        }

        if let Some(ep) = pos.ep {
            if pawn_attacks(us, sq) & bit(ep) != 0 {
                let m = Move::new(sq, ep, 0, FLAG_EN_PASSANT).with_piece(PAWN);
                if ep_legal(pos, m) {
                    n += 1;
                }
            }
        }
    }

    // ---- Knights ----
    let mut kb = pos.piece_bb(us, KNIGHT);
    while kb != 0 {
        let sq = pop_lsb(&mut kb);
        if pinned & bit(sq) != 0 {
            continue;
        }
        n += (knight_attacks(sq) & (empty | enemies)).count_ones() as usize;
    }

    // ---- Bishops / rooks / queens ----
    for pt in [BISHOP, ROOK, QUEEN] {
        let mut sb = pos.piece_bb(us, pt);
        while sb != 0 {
            let sq = pop_lsb(&mut sb);
            let mut tb = sliding_attacks(pt, sq, occ) & (empty | enemies);
            if pinned & bit(sq) != 0 {
                tb &= line(k, sq);
            }
            n += tb.count_ones() as usize;
        }
    }

    // ---- Check filtering ----
    if in_check {
        // Re-count with the same filter generate_legal applies: keep moves that
        // capture the checker, block the line, or are king moves (already in n
        // separately). We cannot distinguish move types from raw bitboards, so
        // reuse generate_legal for the rare in-check nodes.
        return generate_legal(pos).len;
    }

    n
}

fn count_castling(pos: &Position, occ: u64, occ_no_king: u64, them: usize) -> usize {
    // Same gate as generation (see castle_available): counts must agree
    // with generated moves for perft bulk counting.
    let mut n = 0;
    if castle_available(pos, occ, occ_no_king, them, true) {
        n += 1;
    }
    if castle_available(pos, occ, occ_no_king, them, false) {
        n += 1;
    }
    n
}

fn sliding_attacks(pt: usize, sq: usize, occ: u64) -> u64 {
    match pt {
        BISHOP => crate::magic::bishop_attacks(sq, occ),
        ROOK => crate::magic::rook_attacks(sq, occ),
        _ => crate::magic::queen_attacks(sq, occ),
    }
}

fn push_simple(list: &mut MoveList, from: usize, to: usize, enemies: u64, pt: usize) {
    let flags = if enemies & bit(to) != 0 {
        FLAG_CAPTURE
    } else {
        FLAG_QUIET
    };
    list.push(Move::new(from, to, 0, flags).with_piece(pt));
}

/// Shared castling gate for generate and count paths (they must agree).
/// Beyond rights/emptiness/safety, requires the king on its home square
/// and an own rook on the corner: malformed FENs (rights without pieces)
/// must not yield castle moves that corrupt state on make/unmake.
fn castle_available(
    pos: &Position,
    occ: u64,
    occ_no_king: u64,
    them: usize,
    kingside: bool,
) -> bool {
    let us = pos.side;
    let (empty, king_path, rook_sq, right, from) = if kingside {
        if us == WHITE {
            (bit(5) | bit(6), bit(4) | bit(5) | bit(6), 7usize, CASTLE_WK, 4usize)
        } else {
            (bit(61) | bit(62), bit(60) | bit(61) | bit(62), 63usize, CASTLE_BK, 60usize)
        }
    } else if us == WHITE {
        (bit(1) | bit(2) | bit(3), bit(2) | bit(3) | bit(4), 0usize, CASTLE_WQ, 4usize)
    } else {
        (bit(57) | bit(58) | bit(59), bit(58) | bit(59) | bit(60), 56usize, CASTLE_BQ, 60usize)
    };
    if pos.castle & right == 0 || occ & empty != 0 {
        return false;
    }
    if pos.king_sq[us] != from {
        return false;
    }
    if pos.piece_bb(us, ROOK) & bit(rook_sq) == 0 {
        return false;
    }
    let mut b = king_path;
    while b != 0 {
        let s = pop_lsb(&mut b);
        if pos.square_attacked(s, them, occ_no_king) {
            return false;
        }
    }
    true
}

fn gen_castling(pos: &Position, list: &mut MoveList, occ: u64, occ_no_king: u64, them: usize) {
    let us = pos.side;
    let (ks_from, ks_to) = if us == WHITE { (4, 6) } else { (60, 62) };
    let (qs_from, qs_to) = if us == WHITE { (4, 2) } else { (60, 58) };
    if castle_available(pos, occ, occ_no_king, them, true) {
        list.push(Move::new(ks_from, ks_to, 0, FLAG_CASTLE_KS).with_piece(KING));
    }
    if castle_available(pos, occ, occ_no_king, them, false) {
        list.push(Move::new(qs_from, qs_to, 0, FLAG_CASTLE_QS).with_piece(KING));
    }
}

/// En passant is legal only if the king is not in check after removing both pawns.
fn ep_legal(pos: &Position, m: Move) -> bool {
    let us = pos.side;
    let cap_sq = if us == WHITE { m.to() - 8 } else { m.to() + 8 };
    let occ = pos.occ ^ bit(m.from()) ^ bit(m.to()) ^ bit(cap_sq);
    let k = pos.king_sq[us];
    let them = us ^ 1;
    // square_attacked ignores occ for pawns; here the captured pawn is gone,
    // so probe the king manually with the pawn set minus the captured pawn.
    let pawns = pos.piece_bb(them, PAWN) & !bit(cap_sq);
    let attacked = pawn_attacks(us, k) & pawns != 0
        || knight_attacks(k) & pos.piece_bb(them, KNIGHT) != 0
        || king_attacks(k) & pos.piece_bb(them, KING) != 0
        || crate::magic::bishop_attacks(k, occ)
            & (pos.piece_bb(them, BISHOP) | pos.piece_bb(them, QUEEN))
            != 0
        || crate::magic::rook_attacks(k, occ)
            & (pos.piece_bb(them, ROOK) | pos.piece_bb(them, QUEEN))
            != 0;
    !attacked
}

/// Captures + promotions for quiescence search.
pub fn generate_captures(pos: &Position) -> MoveList {
    let full = generate_legal(pos);
    let mut out = MoveList::new();
    for m in full.iter() {
        if m.is_capture() || m.is_promotion() {
            out.push(m);
        }
    }
    out
}

/// Pseudo-legal move generation without pin/check filtering. Legality must be
/// verified by the caller (make the move, then check the mover's king is not
/// attacked). `captures_only` limits generation to captures + promotions + en
/// passant (quiescence); when `in_check`, quiescence needs quiet evasions too,
/// so the caller passes `captures_only = !in_check`. Castling is generated
/// only when not in check and its path squares are already validated.
pub fn generate_pseudo(pos: &Position, captures_only: bool, in_check: bool) -> MoveList {
    let mut out = MoveList::new();
    generate_pseudo_into(pos, captures_only, in_check, &mut out);
    out
}

/// Fills `out` (cleared first) with the pseudo-legal moves. See
/// `generate_pseudo`. The caller supplies the buffer so the hot search path
/// reuses a per-ply `MoveList` instead of zeroing a fresh 1 KB array per node.
///
/// Contract: never call with (captures_only=true, in_check=true) — in check
/// the evasions needed are king moves and full replies, which only
/// captures_only=false provides. Production call shape is always
/// `generate_pseudo_into(pos, !in_check, in_check, ...)` (quiescence).
pub fn generate_pseudo_into(pos: &Position, captures_only: bool, in_check: bool, out: &mut MoveList) {
    out.len = 0;
    let us = pos.side;
    let them = us ^ 1;
    let k = pos.king_sq[us];
    let occ = pos.occ;
    let enemies = pos.pieces_of(them);
    let empty = !occ;
    let targets = empty | enemies;

    // ---- King moves ----
    let mut kb = king_attacks(k) & targets;
    while kb != 0 {
        let sq = pop_lsb(&mut kb);
        let is_cap = enemies & bit(sq) != 0;
        if !captures_only || is_cap {
            let flags = if is_cap { FLAG_CAPTURE } else { FLAG_QUIET };
            out.push(Move::new(k, sq, 0, flags).with_piece(KING));
        }
    }

    // ---- Castling (quiet; square safety already checked by gen_castling) ----
    if !captures_only && !in_check {
        gen_castling(pos, out, occ, occ & !bit(k), them);
    }

    // ---- Pawns ----
    let mut pb = pos.piece_bb(us, PAWN);
    let forward: i32 = if us == WHITE { 8 } else { -8 };
    let promo_rank: usize = if us == WHITE { 7 } else { 0 };
    let start_rank: usize = if us == WHITE { 1 } else { 6 };
    while pb != 0 {
        let sq = pop_lsb(&mut pb);

        if !captures_only {
            if let Some(ps) = step_sq(sq, forward) {
                if empty & bit(ps) != 0 {
                    if rank_of(ps) == promo_rank {
                        for p in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT] {
                            out.push(Move::new(sq, ps, p, FLAG_PROMO).with_piece(PAWN));
                        }
                    } else {
                        out.push(Move::new(sq, ps, 0, FLAG_QUIET).with_piece(PAWN));
                        if rank_of(sq) == start_rank {
                            if let Some(ps2) = step_sq(ps, forward) {
                                if empty & bit(ps2) != 0 {
                                    out.push(Move::new(sq, ps2, 0, FLAG_DOUBLE).with_piece(PAWN));
                                }
                            }
                        }
                    }
                }
            }
        } else if let Some(ps) = step_sq(sq, forward) {
            // Quiet promotions are material gains and must be seen by the
            // quiescence search even in captures-only mode.
            if empty & bit(ps) != 0 && rank_of(ps) == promo_rank {
                for p in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT] {
                    out.push(Move::new(sq, ps, p, FLAG_PROMO).with_piece(PAWN));
                }
            }
        }

        let mut cb = pawn_attacks(us, sq) & enemies;
        while cb != 0 {
            let t = pop_lsb(&mut cb);
            if rank_of(t) == promo_rank {
                for p in [PROMO_QUEEN, PROMO_ROOK, PROMO_BISHOP, PROMO_KNIGHT] {
                    out.push(Move::new(sq, t, p, FLAG_PROMO_CAPTURE).with_piece(PAWN));
                }
            } else {
                out.push(Move::new(sq, t, 0, FLAG_CAPTURE).with_piece(PAWN));
            }
        }

        // En passant: pseudo-legal here; legality (king safe after both pawns
        // are gone) is verified lazily by the caller.
        if let Some(ep) = pos.ep {
            if pawn_attacks(us, sq) & bit(ep) != 0 {
                out.push(Move::new(sq, ep, 0, FLAG_EN_PASSANT).with_piece(PAWN));
            }
        }
    }

    // ---- Knights ----
    let mut kb = pos.piece_bb(us, KNIGHT);
    while kb != 0 {
        let sq = pop_lsb(&mut kb);
        let mut tb = knight_attacks(sq) & targets;
        while tb != 0 {
            let t = pop_lsb(&mut tb);
            if !captures_only || enemies & bit(t) != 0 {
                push_simple(out, sq, t, enemies, KNIGHT);
            }
        }
    }

    // ---- Bishops / rooks / queens ----
    for pt in [BISHOP, ROOK, QUEEN] {
        let mut sb = pos.piece_bb(us, pt);
        while sb != 0 {
            let sq = pop_lsb(&mut sb);
            let mut tb = sliding_attacks(pt, sq, occ) & targets;
            while tb != 0 {
                let t = pop_lsb(&mut tb);
                if !captures_only || enemies & bit(t) != 0 {
                    push_simple(out, sq, t, enemies, pt);
                }
            }
        }
    }
}

