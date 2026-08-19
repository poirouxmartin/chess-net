//! Regression test: every reported PV must be a legal move sequence, even when
//! the search is stopped mid-iteration. A stale `pv_len[ply]` (not reset before
//! early returns) used to leak moves from an unrelated position into the PV,
//! which panicked the UI when rendering the line.

use std::sync::atomic::AtomicBool;

use engine::evaluate::evaluate;
use engine::move_::Move;
use engine::movegen::generate_legal;
use engine::position::Position;
use engine::search::{Limits, Searcher};

fn check_legal_chain(pos0: &Position, moves: &[Move], label: &str) {
    let mut p = *pos0;
    for (i, m) in moves.iter().enumerate() {
        let legal = generate_legal(&p);
        if !legal.moves[..legal.len].contains(m) {
            panic!(
                "{label}: illegal move {i} {} in chain {:?}",
                m.to_uci(),
                moves.iter().map(|x| x.to_uci()).collect::<Vec<_>>()
            );
        }
        p.make_move(*m);
    }
}

#[test]
fn pv_always_legal_under_stops() {
    engine::init();
    let fens = [
        "startpos",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    ];
    for rep in 0..6 {
        for f in &fens {
            let stop = AtomicBool::new(false);
            let mut pos = if *f == "startpos" {
                Position::startpos()
            } else {
                Position::from_fen(f)
            };
            let mut s = Searcher::new(64);
            let mut seen_pv = Vec::new();
            let mut seen_lines = Vec::new();
            let r = s.think_cb(
                &mut pos,
                &Limits { multi_pv: 5, movetime: Some(60), ..Default::default() },
                &stop,
                evaluate,
                Some(&mut |it: &engine::search::SearchIter| {
                    seen_pv.push(it.pv.clone());
                    for l in &it.lines {
                        seen_lines.push(l.pv.clone());
                    }
                }),
            );
            let label = format!("rep {rep} fen={f}");
            check_legal_chain(&pos, &r.pv, &format!("{label} result"));
            let best_line = r
                .lines
                .iter()
                .find(|l| l.mv == r.best)
                .map(|l| l.pv.clone())
                .unwrap_or_default();
            check_legal_chain(&pos, &best_line, &format!("{label} result-line"));
            for (i, pv) in seen_pv.iter().enumerate() {
                check_legal_chain(&pos, pv, &format!("{label} iter {i}"));
            }
            for (i, pv) in seen_lines.iter().enumerate() {
                check_legal_chain(&pos, pv, &format!("{label} line {i}"));
            }
        }
    }
}