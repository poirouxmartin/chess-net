//! Perft: move counting for correctness validation.

use crate::movegen::generate_legal;
use crate::position::Position;

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
        let undo = pos.make_move(m);
        nodes += perft(pos, depth - 1);
        pos.unmake_move(undo);
    }
    nodes
}

pub fn divide(pos: &mut Position, depth: u32) -> u64 {
    let moves = generate_legal(pos);
    let mut total = 0;
    for m in moves.iter() {
        let undo = pos.make_move(m);
        let n = perft(pos, depth - 1);
        pos.unmake_move(undo);
        println!("{}: {}", m.to_uci(), n);
        total += n;
    }
    total
}