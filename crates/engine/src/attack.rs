//! Precomputed attack tables (non-sliding) and between/line tables.

use std::sync::OnceLock;

use crate::bitboard::{bit, file_of, rank_of};

pub const KNIGHT_DIRS: [i32; 8] = [17, 15, 10, 6, -6, -10, -15, -17];
pub const KING_DIRS: [i32; 8] = [8, -8, 1, -1, 9, 7, -7, -9];
pub const PAWN_DIRS_W: [i32; 2] = [9, 7];
pub const PAWN_DIRS_B: [i32; 2] = [-7, -9];

/// Move one square in a direction, None when off board.
#[inline(always)]
pub fn step_sq(s: usize, dir: i32) -> Option<usize> {
    let r = (s >> 3) as i32;
    let f = (s & 7) as i32;
    let (dr, df) = match dir {
        8 => (1, 0),
        -8 => (-1, 0),
        1 => (0, 1),
        -1 => (0, -1),
        9 => (1, 1),
        7 => (1, -1),
        -7 => (-1, 1),
        -9 => (-1, -1),
        17 => (2, 1),
        15 => (2, -1),
        10 => (1, 2),
        6 => (1, -2),
        -6 => (-1, 2),
        -10 => (-1, -2),
        -15 => (-2, 1),
        -17 => (-2, -1),
        _ => unreachable!(),
    };
    let nr = r + dr;
    let nf = f + df;
    if nr < 0 || nr > 7 || nf < 0 || nf > 7 {
        None
    } else {
        Some((nr * 8 + nf) as usize)
    }
}

pub struct Attacks {
    pub knight: [u64; 64],
    pub king: [u64; 64],
    /// pawn[color][sq]: squares attacked by a pawn of `color` standing on `sq`.
    pub pawn: [[u64; 64]; 2],
    /// Strictly between two squares (0 when not aligned).
    pub between: [[u64; 64]; 64],
    /// Full line through two aligned squares, extending in both directions
    /// (includes `b` and beyond, but NOT `a` itself).
    pub line: [[u64; 64]; 64],
}

fn direction(a: usize, b: usize) -> Option<i32> {
    for &dir in &KING_DIRS {
        let mut cur = a;
        while let Some(n) = step_sq(cur, dir) {
            if n == b {
                return Some(dir);
            }
            cur = n;
        }
    }
    None
}

impl Attacks {
    pub fn build() -> Self {
        let mut knight = [0u64; 64];
        let mut king = [0u64; 64];
        let mut pawn = [[0u64; 64]; 2];
        let mut between = [[0u64; 64]; 64];
        let mut line = [[0u64; 64]; 64];

        for sq in 0..64 {
            for &d in &KNIGHT_DIRS {
                if let Some(n) = step_sq(sq, d) {
                    knight[sq] |= bit(n);
                }
            }
            for &d in &KING_DIRS {
                if let Some(n) = step_sq(sq, d) {
                    king[sq] |= bit(n);
                }
            }
            for &d in &PAWN_DIRS_W {
                if let Some(n) = step_sq(sq, d) {
                    pawn[0][sq] |= bit(n);
                }
            }
            for &d in &PAWN_DIRS_B {
                if let Some(n) = step_sq(sq, d) {
                    pawn[1][sq] |= bit(n);
                }
            }
        }

        for a in 0..64 {
            for b in 0..64 {
                if let Some(dir) = direction(a, b) {
                    let mut cur = a;
                    let mut bt = 0u64;
                    while let Some(n) = step_sq(cur, dir) {
                        if n == b {
                            break;
                        }
                        bt |= bit(n);
                        cur = n;
                    }
                    between[a][b] = bt;
                    // Full line through a and b, extending in both directions.
                    let mut ln = 0u64;
                    let mut cur = a;
                    while let Some(n) = step_sq(cur, dir) {
                        ln |= bit(n);
                        cur = n;
                    }
                    let mut cur = a;
                    while let Some(n) = step_sq(cur, -dir) {
                        ln |= bit(n);
                        cur = n;
                    }
                    line[a][b] = ln;
                }
            }
        }

        Attacks {
            knight,
            king,
            pawn,
            between,
            line,
        }
    }
}

static ATTACKS: OnceLock<Attacks> = OnceLock::new();

pub fn init() {
    let _ = ATTACKS.get_or_init(Attacks::build);
}

#[inline]
pub fn knight_attacks(sq: usize) -> u64 {
    ATTACKS.get().expect("attacks not initialized").knight[sq]
}

#[inline]
pub fn king_attacks(sq: usize) -> u64 {
    ATTACKS.get().expect("attacks not initialized").king[sq]
}

#[inline]
pub fn pawn_attacks(color: usize, sq: usize) -> u64 {
    ATTACKS.get().expect("attacks not initialized").pawn[color][sq]
}

#[inline]
pub fn between(sq1: usize, sq2: usize) -> u64 {
    ATTACKS.get().expect("attacks not initialized").between[sq1][sq2]
}

#[inline]
pub fn line(sq1: usize, sq2: usize) -> u64 {
    ATTACKS.get().expect("attacks not initialized").line[sq1][sq2]
}

#[inline]
pub fn is_aligned(a: usize, b: usize) -> bool {
    file_of(a) == file_of(b)
        || rank_of(a) == rank_of(b)
        || (file_of(a) as i32 - file_of(b) as i32).abs()
            == (rank_of(a) as i32 - rank_of(b) as i32).abs()
}