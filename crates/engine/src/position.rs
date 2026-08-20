//! Board representation: 12 piece bitboards, occupancy, zobrist hashing,
//! make/unmake with incremental hash updates.

use std::sync::OnceLock;

use crate::attack::{is_aligned, king_attacks, knight_attacks, pawn_attacks};
use crate::bitboard::{bit, file_of, pop_lsb, rank_of};
use crate::magic::{bishop_attacks, rook_attacks};
use crate::move_::{parse_sq_checked, Move, FLAG_CASTLE_KS, FLAG_CASTLE_QS, FLAG_CAPTURE, FLAG_DOUBLE, FLAG_EN_PASSANT, FLAG_PROMO, FLAG_PROMO_CAPTURE};

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;
pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;
pub const PIECE_TYPES: usize = 6;

pub const PIECE_CHARS: [char; 6] = ['p', 'n', 'b', 'r', 'q', 'k'];

// Castling rights bits.
pub const CASTLE_WK: u8 = 1;
pub const CASTLE_WQ: u8 = 2;
pub const CASTLE_BK: u8 = 4;
pub const CASTLE_BQ: u8 = 8;

// Castling rights cleared when from/to square moves a king/rook.
pub const CASTLE_CLEAR: [u8; 64] = {
    let mut t = [0u8; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = 0;
        i += 1;
    }
    t[4] = CASTLE_WK | CASTLE_WQ; // e1
    t[60] = CASTLE_BK | CASTLE_BQ; // e8
    t[7] = CASTLE_WK; // h1
    t[0] = CASTLE_WQ; // a1
    t[63] = CASTLE_BK; // h8
    t[56] = CASTLE_BQ; // a8
    t
};

#[derive(Clone, Copy)]
pub struct Zobrist {
    pub piece: [[[u64; 64]; PIECE_TYPES]; 2],
    pub castle: [u64; 16],
    pub ep: [u64; 8],
    pub side: u64,
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

impl Zobrist {
    fn new() -> Self {
        let mut rng = SplitMix64::new(0xD00D_FEED_CAFE_BABE);
        let mut z = Zobrist {
            piece: [[[0; 64]; PIECE_TYPES]; 2],
            castle: [0; 16],
            ep: [0; 8],
            side: 0,
        };
        for c in 0..2 {
            for p in 0..PIECE_TYPES {
                for sq in 0..64 {
                    z.piece[c][p][sq] = rng.next();
                }
            }
        }
        for i in 0..16 {
            z.castle[i] = rng.next();
        }
        for i in 0..8 {
            z.ep[i] = rng.next();
        }
        z.side = rng.next();
        z
    }
}

static ZOB: OnceLock<Zobrist> = OnceLock::new();

pub fn init() {
    let _ = ZOB.get_or_init(Zobrist::new);
}

fn zob() -> &'static Zobrist {
    ZOB.get().expect("zobrist not initialized")
}

/// State captured to undo a move.
#[derive(Clone, Copy)]
pub struct Undo {
    pub m: Move,
    pub captured: Option<(usize, usize)>,
    pub castle: u8,
    pub ep: Option<usize>,
    pub halfmove: u32,
}

#[derive(Clone, Copy)]
pub struct Position {
    /// pieces[color * 6 + piece_type] bitboard.
    pub pieces: [u64; 12],
    pub occ: u64,
    pub side: usize,
    pub castle: u8,
    pub ep: Option<usize>,
    pub halfmove: u32,
    pub fullmove: u32,
    pub key: u64,
    pub king_sq: [usize; 2],
    /// Square -> piece map: bits 0..2 = piece type, bit 6 = color, 0xFF empty.
    /// O(1) alternative to scanning the 12 bitboards in `piece_pt_at`.
    sq_piece: [u8; 64],
    /// Incremental static evaluation (white perspective): material + PST
    /// tapered totals and game phase, maintained in `add_piece`/`remove_piece`.
    /// Lets `evaluate` avoid rescanning the 12 bitboards.
    pub mg: i32,
    pub eg: i32,
    pub phase: i32,
}

/// Packed "empty square" marker for `sq_piece`.
const SQ_EMPTY: u8 = 0xFF;

#[inline(always)]
fn sq_code(color: usize, pt: usize) -> u8 {
    ((color as u8) << 6) | pt as u8
}

impl Position {
    pub fn startpos() -> Self {
        Self::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    }

    pub fn from_fen(fen: &str) -> Self {
        let parts: Vec<&str> = fen.split_whitespace().collect();
        let board = parts.first().copied().unwrap_or("");
        let side = if parts.get(1).copied() == Some("w") { WHITE } else { BLACK };
        let mut castle = 0u8;
        if let Some(p) = parts.get(2) {
            for c in p.chars() {
                match c {
                    'K' => castle |= CASTLE_WK,
                    'Q' => castle |= CASTLE_WQ,
                    'k' => castle |= CASTLE_BK,
                    'q' => castle |= CASTLE_BQ,
                    _ => {}
                }
            }
        }
        let ep = match parts.get(3).copied() {
            Some("-") | None => None,
            Some(s) => parse_sq_checked(s),
        };
        let halfmove = parts.get(4).map_or(0, |s| s.parse().unwrap_or(0));
        let fullmove = parts.get(5).map_or(1, |s| s.parse().unwrap_or(1));

        let mut p = Position {
            pieces: [0; 12],
            occ: 0,
            side,
            castle,
            ep,
            halfmove,
            fullmove,
            key: 0,
            king_sq: [0; 2],
            sq_piece: [SQ_EMPTY; 64],
            mg: 0,
            eg: 0,
            phase: 0,
        };

        let mut rank = 7usize;
        let mut file = 0usize;
        for ch in board.chars() {
            match ch {
                '1'..='8' => file += ch.to_digit(10).unwrap() as usize,
                '/' => {
                    if rank == 0 {
                        continue;
                    }
                    rank -= 1;
                    file = 0;
                }
                _ => {
                    let sq = rank * 8 + file;
                    if sq >= 64 {
                        file += 1;
                        continue;
                    }
                    let color = if ch.is_uppercase() { WHITE } else { BLACK };
                    let Some(pt) = PIECE_CHARS
                        .iter()
                        .position(|&c| c == ch.to_ascii_lowercase())
                    else {
                        file += 1;
                        continue;
                    };
                    p.add_piece(color, sq, pt);
                    file += 1;
                }
            }
        }
        if side == BLACK {
            p.key ^= zob().side;
        }
        if let Some(e) = p.ep {
            p.key ^= zob().ep[file_of(e)];
        }
        p.key ^= zob().castle[castle as usize];
        p
    }

    pub fn to_fen(&self) -> String {
        let mut board = String::new();
        let mut empty = 0;
        for rank in (0..8).rev() {
            for file in 0..8 {
                let sq = rank * 8 + file;
                match self.piece_at(sq) {
                    Some((c, pt)) => {
                        if empty > 0 {
                            board.push_str(&empty.to_string());
                            empty = 0;
                        }
                        let ch = PIECE_CHARS[pt];
                        if c == WHITE {
                            board.push(ch.to_ascii_uppercase());
                        } else {
                            board.push(ch);
                        }
                    }
                    None => empty += 1,
                }
            }
            if empty > 0 {
                board.push_str(&empty.to_string());
                empty = 0;
            }
            if rank > 0 {
                board.push('/');
            }
        }
        let mut castle = String::new();
        if self.castle & CASTLE_WK != 0 {
            castle.push('K');
        }
        if self.castle & CASTLE_WQ != 0 {
            castle.push('Q');
        }
        if self.castle & CASTLE_BK != 0 {
            castle.push('k');
        }
        if self.castle & CASTLE_BQ != 0 {
            castle.push('q');
        }
        if castle.is_empty() {
            castle.push('-');
        }
        let ep = match self.ep {
            Some(sq) => crate::move_::sq_to_name(sq),
            None => "-".to_string(),
        };
        format!(
            "{board} {} {castle} {ep} {} {}",
            if self.side == WHITE { "w" } else { "b" },
            self.halfmove,
            self.fullmove
        )
    }

    #[inline(always)]
    pub fn piece_idx(color: usize, pt: usize) -> usize {
        color * PIECE_TYPES + pt
    }

    #[inline(always)]
    pub fn pieces_of(&self, color: usize) -> u64 {
        self.pieces[color * PIECE_TYPES] 
            | self.pieces[color * PIECE_TYPES + 1]
            | self.pieces[color * PIECE_TYPES + 2]
            | self.pieces[color * PIECE_TYPES + 3]
            | self.pieces[color * PIECE_TYPES + 4]
            | self.pieces[color * PIECE_TYPES + 5]
    }

    #[inline(always)]
    pub fn piece_bb(&self, color: usize, pt: usize) -> u64 {
        self.pieces[color * PIECE_TYPES + pt]
    }

    #[inline(always)]
    pub fn piece_at(&self, sq: usize) -> Option<(usize, usize)> {
        let v = self.sq_piece[sq];
        if v == SQ_EMPTY {
            None
        } else {
            Some((((v >> 6) & 1) as usize, (v & 0x07) as usize))
        }
    }

    /// Piece type on `sq` (either color). `sq` must be occupied.
    #[inline(always)]
    pub fn piece_pt_at(&self, sq: usize) -> usize {
        (self.sq_piece[sq] & 0x07) as usize
    }

    /// Color of the piece on `sq`. `sq` must be occupied.
    #[inline(always)]
    pub fn piece_color_at(&self, sq: usize) -> usize {
        ((self.sq_piece[sq] >> 6) & 1) as usize
    }

    fn add_piece(&mut self, color: usize, sq: usize, pt: usize) {
        let idx = Self::piece_idx(color, pt);
        self.pieces[idx] |= bit(sq);
        self.occ |= bit(sq);
        self.sq_piece[sq] = sq_code(color, pt);
        let (mg, eg, ph) = crate::evaluate::piece_eval_delta(color, sq, pt);
        self.mg += mg;
        self.eg += eg;
        self.phase += ph;
        self.key ^= zob().piece[color][pt][sq];
        if pt == KING {
            self.king_sq[color] = sq;
        }
    }

    fn remove_piece(&mut self, color: usize, sq: usize, pt: usize) {
        let idx = Self::piece_idx(color, pt);
        self.pieces[idx] &= !bit(sq);
        self.occ &= !bit(sq);
        self.sq_piece[sq] = SQ_EMPTY;
        let (mg, eg, ph) = crate::evaluate::piece_eval_delta(color, sq, pt);
        self.mg -= mg;
        self.eg -= eg;
        self.phase -= ph;
        self.key ^= zob().piece[color][pt][sq];
        if pt == KING {
            self.king_sq[color] = usize::MAX;
        }
    }

    pub fn make_move(&mut self, m: Move) -> Undo {
        let us = self.side;
        let them = us ^ 1;
        let from = m.from();
        let to = m.to();
        let flags = m.flags();

        let mut undo = Undo {
            m,
            captured: None,
            castle: self.castle,
            ep: self.ep,
            halfmove: self.halfmove,
        };

        self.key ^= zob().side;
        if let Some(e) = self.ep {
            self.key ^= zob().ep[file_of(e)];
        }

        let moving_pt = if flags == FLAG_PROMO || flags == FLAG_PROMO_CAPTURE {
            m.promo_pt()
        } else {
            self.piece_pt_at(from)
        };
        let base_pt = if flags == FLAG_PROMO || flags == FLAG_PROMO_CAPTURE {
            PAWN
        } else {
            moving_pt
        };

        self.remove_piece(us, from, base_pt);

        if flags == FLAG_EN_PASSANT {
            let cap_sq = if us == WHITE { to - 8 } else { to + 8 };
            self.remove_piece(them, cap_sq, PAWN);
            undo.captured = Some((them, PAWN));
        } else if flags == FLAG_CAPTURE || flags == FLAG_PROMO_CAPTURE {
            let cap_pt = self.piece_pt_at(to);
            self.remove_piece(them, to, cap_pt);
            undo.captured = Some((them, cap_pt));
        }

        self.add_piece(us, to, moving_pt);

        if flags == FLAG_CASTLE_KS {
            let (rf, rt) = if us == WHITE { (7, 5) } else { (63, 61) };
            self.move_piece(us, rf, rt, ROOK);
        } else if flags == FLAG_CASTLE_QS {
            let (rf, rt) = if us == WHITE { (0, 3) } else { (56, 59) };
            self.move_piece(us, rf, rt, ROOK);
        }

        self.castle &= !(CASTLE_CLEAR[from] | CASTLE_CLEAR[to]);
        self.key ^= zob().castle[undo.castle as usize];
        self.key ^= zob().castle[self.castle as usize];
        if flags == FLAG_DOUBLE {
            self.ep = Some((from + to) / 2);
            self.key ^= zob().ep[file_of((from + to) / 2)];
        } else {
            self.ep = None;
        }
        if flags == FLAG_PROMO || flags == FLAG_PROMO_CAPTURE || flags == FLAG_CAPTURE
            || flags == FLAG_EN_PASSANT || moving_pt == PAWN
        {
            self.halfmove = 0;
        } else {
            self.halfmove += 1;
        }
        if us == BLACK {
            self.fullmove += 1;
        }
        self.side = them;
        undo
    }

    pub fn unmake_move(&mut self, undo: Undo) {
        let mover = self.side ^ 1;
        let m = undo.m;
        let from = m.from();
        let to = m.to();
        let flags = m.flags();

        let moving_pt = if flags == FLAG_PROMO || flags == FLAG_PROMO_CAPTURE {
            m.promo_pt()
        } else {
            self.piece_pt_at(to)
        };

        self.remove_piece(mover, to, moving_pt);

        if flags == FLAG_EN_PASSANT {
            let cap_sq = if mover == WHITE { to - 8 } else { to + 8 };
            self.add_piece(mover ^ 1, cap_sq, PAWN);
        } else if flags == FLAG_CAPTURE || flags == FLAG_PROMO_CAPTURE {
            if let Some((c, pt)) = undo.captured {
                self.add_piece(c, to, pt);
            }
        }

        if flags == FLAG_PROMO || flags == FLAG_PROMO_CAPTURE {
            self.add_piece(mover, from, PAWN);
        } else {
            self.add_piece(mover, from, moving_pt);
        }

        if flags == FLAG_CASTLE_KS {
            let (rt, rf) = if mover == WHITE { (7, 5) } else { (63, 61) };
            self.move_piece(mover, rf, rt, ROOK);
        } else if flags == FLAG_CASTLE_QS {
            let (rt, rf) = if mover == WHITE { (0, 3) } else { (56, 59) };
            self.move_piece(mover, rf, rt, ROOK);
        }

        self.key ^= zob().castle[self.castle as usize];
        self.castle = undo.castle;
        self.key ^= zob().castle[undo.castle as usize];
        if let Some(e) = self.ep {
            self.key ^= zob().ep[file_of(e)];
        }
        self.ep = undo.ep;
        if let Some(e) = undo.ep {
            self.key ^= zob().ep[file_of(e)];
        }
        self.halfmove = undo.halfmove;
        self.side = mover;
        self.key ^= zob().side;
        if mover == BLACK {
            self.fullmove -= 1;
        }
    }

    fn move_piece(&mut self, color: usize, from: usize, to: usize, pt: usize) {
        self.remove_piece(color, from, pt);
        self.add_piece(color, to, pt);
    }

    /// Is `sq` attacked by `by`? `occ` may differ from the real occupancy
    /// (e.g. the king removed) for check/legality probing.
    #[inline(always)]
    pub fn square_attacked(&self, sq: usize, by: usize, occ: u64) -> bool {
        if pawn_attacks(by ^ 1, sq) & self.piece_bb(by, PAWN) != 0 {
            return true;
        }
        if knight_attacks(sq) & self.piece_bb(by, KNIGHT) != 0 {
            return true;
        }
        if king_attacks(sq) & self.piece_bb(by, KING) != 0 {
            return true;
        }
        if bishop_attacks(sq, occ) & (self.piece_bb(by, BISHOP) | self.piece_bb(by, QUEEN)) != 0 {
            return true;
        }
        if rook_attacks(sq, occ) & (self.piece_bb(by, ROOK) | self.piece_bb(by, QUEEN)) != 0 {
            return true;
        }
        false
    }

    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.square_attacked(self.king_sq[self.side], self.side ^ 1, self.occ & !bit(self.king_sq[self.side]))
    }

    /// After `make_move` by side `us`, is the king of `us` attacked by the new
    /// side to move (`us ^ 1`)? Used for lazy move legality.
    #[inline(always)]
    pub fn king_safe(&self, us: usize) -> bool {
        let k = self.king_sq[us];
        !self.square_attacked(k, us ^ 1, self.occ & !bit(k))
    }

    /// Squares of enemy pieces giving check to the side to move.
    pub fn checkers(&self) -> u64 {
        let us = self.side;
        let them = us ^ 1;
        let k = self.king_sq[us];
        let occ = self.occ & !bit(k);
        let mut c = pawn_attacks(them ^ 1, k) & self.piece_bb(them, PAWN);
        c |= knight_attacks(k) & self.piece_bb(them, KNIGHT);
        c |= bishop_attacks(k, occ) & (self.piece_bb(them, BISHOP) | self.piece_bb(them, QUEEN));
        c |= rook_attacks(k, occ) & (self.piece_bb(them, ROOK) | self.piece_bb(them, QUEEN));
        c
    }

    /// Pieces of the side to move that are pinned against their king.
    pub fn pinned(&self) -> u64 {
        let us = self.side;
        let them = us ^ 1;
        let k = self.king_sq[us];
        let occ = self.occ;
        let mut pinned = 0u64;
        let rooks_qs = self.piece_bb(them, ROOK) | self.piece_bb(them, QUEEN);
        let bishops_qs = self.piece_bb(them, BISHOP) | self.piece_bb(them, QUEEN);

        let mut sliders = rooks_qs;
        while sliders != 0 {
            let s = pop_lsb(&mut sliders);
            if is_aligned(k, s)
                && (file_of(k) == file_of(s) || rank_of(k) == rank_of(s))
            {
                let bt = crate::attack::between(k, s) & occ;
                if bt != 0 && bt.count_ones() == 1 && bt & self.pieces_of(us) != 0 {
                    pinned |= bt;
                }
            }
        }
        let mut sliders = bishops_qs;
        while sliders != 0 {
            let s = pop_lsb(&mut sliders);
            if is_aligned(k, s)
                && file_of(k) != file_of(s)
                && rank_of(k) != rank_of(s)
            {
                let bt = crate::attack::between(k, s) & occ;
                if bt != 0 && bt.count_ones() == 1 && bt & self.pieces_of(us) != 0 {
                    pinned |= bt;
                }
            }
        }
        pinned
    }

    pub fn insufficient_material(&self) -> bool {
        // No pawns/rooks/queens and at most one minor.
        let minors = self.pieces_of(WHITE) | self.pieces_of(BLACK);
        if self.pieces[Self::piece_idx(WHITE, PAWN)] != 0
            || self.pieces[Self::piece_idx(BLACK, PAWN)] != 0
            || self.pieces[Self::piece_idx(WHITE, ROOK)] != 0
            || self.pieces[Self::piece_idx(BLACK, ROOK)] != 0
            || self.pieces[Self::piece_idx(WHITE, QUEEN)] != 0
            || self.pieces[Self::piece_idx(BLACK, QUEEN)] != 0
        {
            return false;
        }
        let non_king = minors & !(bit(self.king_sq[WHITE]) | bit(self.king_sq[BLACK]));
        non_king.count_ones() <= 1
    }

    /// First rank for the side to move's pawns (rank 1 for white, rank 6 for black).
    #[inline(always)]
    pub fn pawn_start_rank(&self) -> usize {
        if self.side == WHITE {
            1
        } else {
            6
        }
    }

    /// Make a null move (pass): toggle side, clear ep. Used by null-move pruning.
    pub fn make_null(&mut self) {
        self.key ^= zob().side;
        if let Some(e) = self.ep {
            self.key ^= zob().ep[file_of(e)];
            self.ep = None;
        }
        self.side ^= 1;
    }

    pub fn unmake_null(&mut self) {
        self.key ^= zob().side;
        self.side ^= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fen_roundtrip() {
        crate::init();
        let pos = Position::startpos();
        assert_eq!(
            pos.to_fen(),
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        );
        let pos2 = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        );
        assert_eq!(
            pos2.to_fen(),
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"
        );
    }
}