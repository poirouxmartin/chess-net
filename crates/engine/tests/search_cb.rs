//! Verify `think_cb` reports one live iteration per depth, in order.

use std::sync::atomic::{AtomicBool, Ordering};

use engine::evaluate::evaluate;
use engine::position::Position;
use engine::search::{Limits, Searcher};

#[test]
fn think_cb_reports_each_depth() {
    engine::init();
    let mut pos = Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    let mut searcher = Searcher::new(8);
    let stop = AtomicBool::new(false);
    let mut iters: Vec<(i32, i32, u64)> = Vec::new();
    let limits = Limits {
        depth: Some(6),
        ..Default::default()
    };
    let result = searcher.think_cb(
        &mut pos,
        &limits,
        &stop,
        evaluate,
        Some(&mut |it| {
            iters.push((it.depth, it.score, it.nodes));
        }),
    );

    assert_eq!(result.depth, 6);
    assert_eq!(iters.len(), 6, "one callback per completed depth");
    for (i, (d, _, _)) in iters.iter().enumerate() {
        assert_eq!(*d, (i + 1) as i32, "depths reported in order");
    }
    assert!(iters.last().map(|(_, _, n)| *n > 0).unwrap_or(false));
    assert!(!result.pv.is_empty());
}

#[test]
fn think_cb_stops_on_flag() {
    engine::init();
    let mut pos = Position::startpos();
    let mut searcher = Searcher::new(8);
    let stop = AtomicBool::new(true); // already stopped: must not hang
    let limits = Limits {
        depth: Some(64),
        ..Default::default()
    };
    let result = searcher.think_cb(&mut pos, &limits, &stop, evaluate, None);
    assert!(result.best.to_uci().len() == 4);
}