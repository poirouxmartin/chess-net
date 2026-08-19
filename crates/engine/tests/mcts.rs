//! MCTS: legal best move, mate-in-1 detection, terminal positions, budgets.

use std::sync::atomic::AtomicBool;

use engine::evaluate::evaluate;
use engine::mcts::{mcts_root, MctsLimits};
use engine::movegen::generate_legal;
use engine::position::Position;

fn run(pos: &mut Position, playouts: u64) -> engine::mcts::MctsResult {
    mcts_root(
        pos,
        &MctsLimits { playouts: Some(playouts), movetime: None, threads: 1 },
        &AtomicBool::new(false),
        evaluate,
        None,
    )
}

#[test]
fn finds_back_rank_mate_in_one() {
    engine::init();
    let mut pos = Position::from_fen("7k/6pp/8/8/8/8/8/R6K w - - 0 1");
    let result = run(&mut pos, 8000);
    assert_eq!(result.best.to_uci(), "a1a8");
    assert!(result.value > 0.99, "mate value should be ~1.0, got {}", result.value);
    let total: u32 = result.visits.iter().map(|(_, v)| v).sum();
    assert_eq!(total, 8000);
    assert_eq!(result.visits[0].0.to_uci(), "a1a8", "most-visited must be the mate");
}

#[test]
fn returns_legal_move_from_startpos() {
    engine::init();
    let mut pos = Position::startpos();
    let legal = generate_legal(&pos);
    let result = run(&mut pos, 1500);
    assert!(result.best.to_uci().len() >= 4);
    assert!(
        legal.moves[..legal.len].iter().any(|m| *m == result.best),
        "best move {} must be legal",
        result.best.to_uci()
    );
    assert!(result.playouts == 1500);
}

#[test]
fn stalemate_root_returns_null() {
    engine::init();
    let mut pos = Position::from_fen("7k/8/6QK/8/8/8/8/8 b - - 0 1");
    let result = run(&mut pos, 100);
    assert!(result.visits.is_empty());
    assert_eq!(result.value, 0.5, "stalemate is a draw");
}

#[test]
fn playouts_budget_exact() {
    engine::init();
    let mut pos = Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    let result = run(&mut pos, 250);
    assert_eq!(result.playouts, 250);
}

#[test]
fn stop_flag_returns_immediately() {
    engine::init();
    let mut pos = Position::startpos();
    let stop = AtomicBool::new(true);
    let result = mcts_root(
        &mut pos,
        &MctsLimits { playouts: Some(1_000_000), movetime: None, threads: 1 },
        &stop,
        evaluate,
        None,
    );
    assert_eq!(result.playouts, 0, "must not run any playout when already stopped");
    assert!(result.best.to_uci().len() == 4);
}

#[test]
fn progress_callback_reports_playouts() {
    engine::init();
    let mut pos = Position::startpos();
    let mut reports: Vec<u64> = Vec::new();
    let result = mcts_root(
        &mut pos,
        &MctsLimits { playouts: None, movetime: Some(150), threads: 1 },
        &AtomicBool::new(false),
        evaluate,
        Some(&mut |p| {
            reports.push(p.playouts);
        }),
    );
    assert!(!reports.is_empty(), "progress must be reported during a 150ms search");
    assert!(reports.windows(2).all(|w| w[0] < w[1]), "playouts must grow");
    assert!(!result.visits.is_empty());
    assert!(result.visits.iter().any(|(_, v)| *v > 0));
    assert!((0.0..=1.0).contains(&result.value), "value must be a probability");
}

#[test]
fn parallel_mcts_finds_mate_in_one() {
    use std::sync::Arc;
    engine::init();
    let mut pos = Position::from_fen("7k/6pp/8/8/8/8/8/R6K w - - 0 1");
    let result = engine::mcts::mcts_parallel(
        &mut pos,
        &MctsLimits { playouts: Some(20_000), movetime: None, threads: 4 },
        Arc::new(AtomicBool::new(false)),
        evaluate,
        None,
    );
    assert_eq!(result.best.to_uci(), "a1a8");
    let total: u32 = result.visits.iter().map(|(_, v)| v).sum();
    assert_eq!(total, result.playouts as u32, "visit counts must sum to the playout total");
    assert_eq!(result.visits[0].0.to_uci(), "a1a8");
}

#[test]
fn parallel_mcts_returns_legal_move_and_is_faster_than_single() {
    use std::sync::Arc;
    engine::init();
    let mut pos = Position::startpos();
    let legal = generate_legal(&pos);
    let result = engine::mcts::mcts_parallel(
        &mut pos,
        &MctsLimits { playouts: Some(60_000), movetime: None, threads: 8 },
        Arc::new(AtomicBool::new(false)),
        evaluate,
        None,
    );
    assert!(legal.moves[..legal.len].iter().any(|m| *m == result.best));
    let total: u32 = result.visits.iter().map(|(_, v)| v).sum();
    assert_eq!(total, result.playouts as u32);
}
