//! AlphaZero-style 105-plane feature encoder.
//!
//! Layout (side-to-move oriented, black mirrored — matches
//! python/chessnet/features_lc0.py EXACTLY):
//!   0-95:   Piece history (12 planes x 8 frames; engine: frame 0 only)
//!   96-97:  Repetition count (position seen >=1 / >=2 before; engine: zeros)
//!   98:     Colour (1 = white to move)
//!   99:     Total move count (ply/512)
//!   100-101: Own castling rights (K, Q)
//!   102-103: Enemy castling rights (K, Q)
//!   104:    No-progress count (halfmove/100)

use engine::position::Position;
use engine::position::{CASTLE_WK, CASTLE_WQ, CASTLE_BK, CASTLE_BQ};

pub const NUM_PLANES: usize = 105;
pub const PLANE_SIZE: usize = 64; // 8 * 8

/// Piece type to plane offset (from side-to-move perspective).
/// Own pieces: 0-5, enemy pieces: 6-11.
const OWN_BASE: usize = 0;
const ENEMY_BASE: usize = 6;

/// Mirror a square vertically (rank r -> 7-r). Black's board is flipped so
/// the net always sees its own back rank at row 0, matching the Python
/// training encoder (python/chessnet/features_lc0.py).
#[inline]
pub fn flip_square(sq: usize) -> usize {
    let rank = sq / 8;
    let file = sq % 8;
    (7 - rank) * 8 + file
}

/// Encode a position as [105, 8, 8] float32 planes (CHW format).
/// All planes are side-to-move oriented: own/enemy pieces AND, when black
/// is to move, mirrored ranks. History frames (12-95) are zeros here — the
/// engine evaluates single positions without game history (frame 0 only).
pub fn encode_position(pos: &Position) -> [f32; NUM_PLANES * PLANE_SIZE] {
    let mut planes = [0.0f32; NUM_PLANES * PLANE_SIZE];
    let stm = pos.side; // 0=white, 1=black
    let flip = stm == 1;

    // Piece planes (0-95): current position only
    for pt in 0..6 {
        // Own pieces (color == stm)
        let mut bb = pos.piece_bb(stm, pt);
        while bb != 0 {
            let sq = engine::bitboard::pop_lsb(&mut bb);
            let sq = if flip { flip_square(sq) } else { sq };
            let plane_idx = OWN_BASE + pt;
            planes[plane_idx * PLANE_SIZE + sq] = 1.0;
        }

        // Enemy pieces (color == stm ^ 1)
        let enemy = stm ^ 1;
        let mut bb = pos.piece_bb(enemy, pt);
        while bb != 0 {
            let sq = engine::bitboard::pop_lsb(&mut bb);
            let sq = if flip { flip_square(sq) } else { sq };
            let plane_idx = ENEMY_BASE + pt;
            planes[plane_idx * PLANE_SIZE + sq] = 1.0;
        }
    }

    // Castling rights (100-103): own K/Q, enemy K/Q.
    let castle = pos.castle;
    let own_k = if stm == 0 { CASTLE_WK } else { CASTLE_BK };
    let own_q = if stm == 0 { CASTLE_WQ } else { CASTLE_BQ };
    let enemy_k = if stm == 0 { CASTLE_BK } else { CASTLE_WK };
    let enemy_q = if stm == 0 { CASTLE_BQ } else { CASTLE_WQ };

    if castle & own_k != 0 {
        for i in 0..PLANE_SIZE { planes[100 * PLANE_SIZE + i] = 1.0; }
    }
    if castle & own_q != 0 {
        for i in 0..PLANE_SIZE { planes[101 * PLANE_SIZE + i] = 1.0; }
    }
    if castle & enemy_k != 0 {
        for i in 0..PLANE_SIZE { planes[102 * PLANE_SIZE + i] = 1.0; }
    }
    if castle & enemy_q != 0 {
        for i in 0..PLANE_SIZE { planes[103 * PLANE_SIZE + i] = 1.0; }
    }

    // Repetitions (96-97): zeros here (engine evaluates single positions
    // without game history; frame 0 only like the piece planes).
    // Colour (98) - 1 when white to move
    if stm == 0 {
        for i in 0..PLANE_SIZE {
            planes[98 * PLANE_SIZE + i] = 1.0;
        }
    }

    // Total move count (99) - ply number scaled
    let ply = (pos.fullmove.saturating_sub(1)) * 2 + (stm as u32);
    let movecount = (ply.min(512) as f32) / 512.0;
    for i in 0..PLANE_SIZE {
        planes[99 * PLANE_SIZE + i] = movecount;
    }

    // No-progress count (104) - normalized
    let halfmove = pos.halfmove.min(100) as f32 / 100.0;
    for i in 0..PLANE_SIZE {
        planes[104 * PLANE_SIZE + i] = halfmove;
    }

    planes
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::position::Position;

    fn plane_sum(planes: &[f32], p: usize) -> f32 {
        planes[p * PLANE_SIZE..(p + 1) * PLANE_SIZE].iter().sum()
    }

    /// Locks the training-layout convention: startpos (white to move, all
    /// castling rights) must put own rights on 100/101 and enemy rights on
    /// 102/103 — exactly like the Python training encoder.
    /// See python/chessnet/features_lc0.py.
    #[test]
    fn startpos_castling_layout_matches_training() {
        engine::init();
        let pos = Position::startpos();
        let planes = encode_position(&pos);
        assert_eq!(plane_sum(&planes, 96), 0.0, "reps empty");
        assert_eq!(plane_sum(&planes, 97), 0.0, "reps empty");
        assert_eq!(plane_sum(&planes, 98), 64.0, "white to move");
        assert_eq!(plane_sum(&planes, 100), 64.0, "own K");
        assert_eq!(plane_sum(&planes, 101), 64.0, "own Q");
        assert_eq!(plane_sum(&planes, 102), 64.0, "enemy K");
        assert_eq!(plane_sum(&planes, 103), 64.0, "enemy Q");
        assert_eq!(plane_sum(&planes, 104), 0.0, "no-progress start");
    }

    /// Flip symmetry: a white-to-move position and its color-swapped,
    /// rank-mirrored black-to-move twin must encode identically except
    /// plane 98 (colour) and 99 (move count: ply 0 vs 1 by construction).
    /// Mirrors the Python test in test_flip.py.
    #[test]
    fn black_flip_matches_white_orientation() {
        engine::init();
        let white = Position::from_fen(
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w kq - 0 1",
        );
        let black = Position::from_fen(
            "rnbq1rk1/ppp2ppp/3p1n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R b KQ - 0 1",
        );
        let a = encode_position(&white);
        let b = encode_position(&black);
        assert_eq!(plane_sum(&a, 98), 64.0);
        assert_eq!(plane_sum(&b, 98), 0.0);
        assert_eq!(plane_sum(&a, 99), 0.0);
        assert!((plane_sum(&b, 99) - 64.0 / 512.0).abs() < 1e-6);
        for p in 0..NUM_PLANES {
            if p == 98 || p == 99 {
                continue;
            }
            for i in 0..PLANE_SIZE {
                assert_eq!(
                    a[p * PLANE_SIZE + i],
                    b[p * PLANE_SIZE + i],
                    "plane {p} cell {i} differs after flip"
                );
            }
        }
    }
}
