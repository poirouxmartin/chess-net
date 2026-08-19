//! Standard Algebraic Notation for moves.

use crate::move_::{sq_to_name, Move, FLAG_CASTLE_QS, FLAG_PROMO_CAPTURE, FLAG_PROMO};
use crate::movegen::generate_legal;
use crate::position::{BISHOP, KING, KNIGHT, Position, QUEEN, ROOK};

/// SAN text for `m` played from `pos` (e.g. "Nf3", "exd5", "O-O-O+", "e8=Q#").
pub fn to_san(pos: &Position, m: Move) -> String {
    if m.is_castle() {
        let mut s = if m.flags() == FLAG_CASTLE_QS { "O-O-O" } else { "O-O" }.to_string();
        s.push_str(&check_suffix(pos, m));
        return s;
    }

    let from = m.from();
    let pt = pos.piece_at(from).map_or(KING, |(_, p)| p);

    let mut s = String::new();
    if pt != crate::position::PAWN {
        s.push(piece_letter(pt));
        s.push_str(&disambiguation(pos, m, pt));
    }

    if m.is_capture() {
        if pt == crate::position::PAWN {
            s.push((b'a' + (from & 7) as u8) as char);
        }
        s.push('x');
    }

    s.push_str(&sq_to_name(m.to()));

    if m.flags() == FLAG_PROMO || m.flags() == FLAG_PROMO_CAPTURE {
        s.push('=');
        s.push(crate::move_::promo_char(m.promo()).to_ascii_uppercase());
    }

    s.push_str(&check_suffix(pos, m));
    s
}

fn piece_letter(pt: usize) -> char {
    match pt {
        KNIGHT => 'N',
        BISHOP => 'B',
        ROOK => 'R',
        QUEEN => 'Q',
        KING => 'K',
        _ => '?',
    }
}

/// File/rank/coordinate disambiguation when another piece of the same type
/// can also reach `m.to()`.
fn disambiguation(pos: &Position, m: Move, pt: usize) -> String {
    let legal = generate_legal(pos);
    let mut same_file = false;
    let mut same_rank = false;
    let mut any = false;
    for i in 0..legal.len {
        let c = legal.get(i);
        if c.from() == m.from() || c.to() != m.to() {
            continue;
        }
        if pos.piece_at(c.from()).is_some_and(|(col, p)| col == pos.side && p == pt) {
            any = true;
            same_file |= c.from() & 7 == m.from() & 7;
            same_rank |= c.from() >> 3 == m.from() >> 3;
        }
    }
    if !any {
        return String::new();
    }
    let mut s = String::new();
    if !same_file {
        s.push((b'a' + (m.from() & 7) as u8) as char);
    } else if !same_rank {
        s.push((b'1' + (m.from() >> 3) as u8) as char);
    } else {
        s.push((b'a' + (m.from() & 7) as u8) as char);
        s.push((b'1' + (m.from() >> 3) as u8) as char);
    }
    s
}

fn check_suffix(pos: &Position, m: Move) -> String {
    let mut p = *pos;
    p.make_move(m);
    if !p.in_check() {
        return String::new();
    }
    if generate_legal(&p).len == 0 {
        "#".to_string()
    } else {
        "+".to_string()
    }
}