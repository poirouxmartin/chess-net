//! Differential test: pseudo-legal generation + lazy verify must equal
//! generate_legal on every position (catches pseudo-path-only bugs that
//! perft never exercises, since perft uses the legal path).

use engine::movegen::{generate_legal, generate_pseudo_into, MoveList};
use engine::position::Position;

fn pseudo_filtered(pos: &mut Position) -> Vec<String> {
    let us = pos.side;
    let in_check = pos.in_check();
    let pinned = pos.pinned();
    let king = pos.king_sq[us];
    let mut moves = MoveList::new();
    // Same call shape as search (captures_only=false here to cover all).
    generate_pseudo_into(pos, false, in_check, &mut moves);
    let mut out = Vec::new();
    for i in 0..moves.len {
        let m = moves.get(i);
        let verify = in_check
            || m.is_en_passant()
            || m.from() == king
            || (pinned & engine::bitboard::bit(m.from())) != 0;
        let undo = pos.make_move(m);
        let ok = !verify || pos.king_safe(us);
        pos.unmake_move(undo);
        if ok {
            out.push(m.to_uci());
        }
    }
    out.sort();
    out
}

/// Quiescence shape: captures-only pseudo + the same lazy verify must
/// equal the legal captures (this is the path perft never covers).
fn pseudo_captures_filtered(pos: &mut Position) -> Vec<String> {
    let us = pos.side;
    let in_check = pos.in_check();
    let pinned = pos.pinned();
    let king = pos.king_sq[us];
    let mut moves = MoveList::new();
    // Production call shape (quiescence): captures-only outside check.
    generate_pseudo_into(pos, !in_check, in_check, &mut moves);
    let mut out = Vec::new();
    for i in 0..moves.len {
        let m = moves.get(i);
        // Mirror quiescence: it searches captures and promotions.
        if !m.is_capture() && !m.is_promotion() && !in_check {
            continue;
        }
        let verify = in_check
            || m.is_en_passant()
            || m.from() == king
            || (pinned & engine::bitboard::bit(m.from())) != 0;
        let undo = pos.make_move(m);
        let ok = !verify || pos.king_safe(us);
        pos.unmake_move(undo);
        if ok {
            out.push(m.to_uci());
        }
    }
    out.sort();
    out
}

#[test]
fn pseudo_matches_legal_everywhere() {
    engine::init();
    let fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b kq - 0 1",
        "7k/6pp/8/8/8/8/8/R6K w - - 0 1",
        "8/8/8/8/1b6/8/2N5/4K2k w - - 0 1",
        "r5k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1",
    ];
    for fen in fens {
        let mut pos = Position::from_fen(fen);
        let legal = generate_legal(&pos);
        let mut a: Vec<String> = legal.moves[..legal.len].iter().map(|m| m.to_uci()).collect();
        a.sort();
        let b = pseudo_filtered(&mut pos);
        assert_eq!(a, b, "pseudo/verify diverges from legal on {fen}");
        // Captures path (quiescence shape): in check it must equal all
        // legal moves, otherwise exactly the legal captures+promotions.
        let in_check = pos.in_check();
        let mut expected: Vec<String> = legal.moves[..legal.len]
            .iter()
            .filter(|m| in_check || m.is_capture() || m.is_promotion())
            .map(|m| m.to_uci())
            .collect();
        expected.sort();
        let d = pseudo_captures_filtered(&mut pos);
        assert_eq!(expected, d, "captures pseudo diverges on {fen}");
    }
}
