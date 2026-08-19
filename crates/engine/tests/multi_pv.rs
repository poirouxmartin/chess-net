//! Multi-PV: top lines reported when `Limits::multi_pv > 1`.

use std::sync::atomic::AtomicBool;

use engine::evaluate::evaluate;
use engine::move_::Move;
use engine::position::Position;
use engine::search::{Limits, Searcher};

#[test]
fn multi_pv_reports_top_lines() {
    engine::init();
    let mut pos = Position::startpos();
    let mut searcher = Searcher::new(8);
    let stop = AtomicBool::new(false);
    let limits = Limits { depth: Some(3), multi_pv: 5, ..Default::default() };
    let res = searcher.think_cb(&mut pos, &limits, &stop, evaluate, None);

    assert_eq!(res.lines.len(), 5, "expected 5 lines");
    assert_eq!(res.lines[0].score, res.score, "top line must share the search score");
    assert!(!res.lines[0].pv.is_empty(), "first line must have a PV");
    for w in res.lines.windows(2) {
        assert!(w[0].score >= w[1].score, "lines must be sorted by score");
    }
    let mut distinct = Vec::new();
    for l in &res.lines {
        if !distinct.contains(&l.mv) {
            distinct.push(l.mv);
        }
    }
    assert_eq!(distinct.len(), 5, "all root moves distinct");
    assert!(res.lines[0].mv != Move::null());
}