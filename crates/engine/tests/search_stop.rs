//! Regression: a search whose stop flag is already set (e.g. UI stopped a
//! previous search) must never return an uninitialized -INF score.

use std::sync::atomic::{AtomicBool, Ordering};

use engine::evaluate::{evaluate, INF};
use engine::move_::Move;
use engine::position::Position;
use engine::search::{Limits, Searcher};

#[test]
fn stopped_at_start_returns_sane_root_score() {
    engine::init();
    let mut pos = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    let stop = AtomicBool::new(true);
    let mut searcher = Searcher::new(64);
    let limits = Limits { depth: None, movetime: None, ..Default::default() };
    let res = searcher.think_cb(&mut pos, &limits, &stop, evaluate, None);
    assert!(res.score > -INF, "score must not be uninitialized -INF, got {}", res.score);
    assert!(res.best != Move::null(), "best must be a legal move");
    assert!(res.depth >= 1);
    assert!(!res.pv.is_empty(), "PV must not be empty: best must be a searched move");
    assert_eq!(res.pv[0], res.best, "PV must start with the returned best move");
}