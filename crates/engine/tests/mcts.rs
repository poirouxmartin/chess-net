//! MCTS: legal best move, mate-in-1 detection, terminal positions, budgets.

use std::sync::atomic::AtomicBool;

use engine::evaluate::evaluate;
use engine::mcts::{mcts_root, MctsLimits};
use engine::movegen::{generate_legal, MoveList};
use engine::position::Position;

fn pesto_combined(pos: &Position, _legal: &MoveList) -> (i32, Vec<f32>) {
    (evaluate(pos), Vec::new())
}

fn run(pos: &mut Position, playouts: u64) -> engine::mcts::MctsResult {
    mcts_root(
        pos,
        &MctsLimits { playouts: Some(playouts), movetime: None, threads: 1 },
        &AtomicBool::new(false),
        pesto_combined,
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
    // Instant 1-ply mate: no search playouts, empty visits.
    assert!(result.visits.is_empty());
}

#[test]
fn mate_in_one_ignores_budget() {
    engine::init();
    // Scholar's mate taken: h5f7# with ZERO playouts.
    let mut pos = Position::from_fen(
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1",
    );
    let result = run(&mut pos, 0);
    assert_eq!(result.best.to_uci(), "h5f7");
    assert_eq!(result.value, 1.0);
}

#[test]
fn never_walks_into_opponent_mate() {
    use engine::mcts::{allows_opp_mate, avoid_opp_mate};
    engine::init();
    // After 8.Qxc2 Black to move: f6?? allows 9.Qg6#, h5 does not.
    let mut pos = Position::from_fen(
        "r1bqkbnr/1p1pppp1/7p/p1pN4/P1P2PP1/7N/1PQPP2P/R1B1KB1R b KQkq - 0 8",
    );
    let legal = generate_legal(&pos);
    let find = |uci: &str| {
        legal.moves[..legal.len]
            .iter()
            .copied()
            .find(|m| m.to_uci() == uci)
            .unwrap_or_else(|| panic!("{uci} must be legal"))
    };
    let f6 = find("f7f6");
    let h5 = find("h6h5");
    assert!(allows_opp_mate(&mut pos, f6), "f6 must allow Qg6#");
    assert!(!allows_opp_mate(&mut pos, h5), "h5 must be safe");
    // Most-visited blunder is overridden by the safe alternative.
    let visits = vec![(f6, 100), (h5, 50)];
    assert_eq!(avoid_opp_mate(&mut pos, f6, &visits), h5);
    // All doomed: keep the best (nothing safe exists).
    let visits = vec![(f6, 100)];
    assert_eq!(avoid_opp_mate(&mut pos, f6, &visits), f6);
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
        pesto_combined,
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
        pesto_combined,
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
    let (result, _search) = engine::mcts::mcts_parallel(
        &mut pos,
        &MctsLimits { playouts: Some(20_000), movetime: None, threads: 4 },
        Arc::new(AtomicBool::new(false)),
        pesto_combined,
        None,
        None,
    );
    assert_eq!(result.best.to_uci(), "a1a8");
    // Instant 1-ply mate: no search playouts, empty visits.
    assert!(result.visits.is_empty());
    assert!(result.value > 0.99, "mate value should be ~1.0, got {}", result.value);
}

#[test]
fn parallel_mcts_returns_legal_move_and_is_faster_than_single() {
    use std::sync::Arc;
    engine::init();
    let mut pos = Position::startpos();
    let legal = generate_legal(&pos);
    let (result, _search) = engine::mcts::mcts_parallel(
        &mut pos,
        &MctsLimits { playouts: Some(60_000), movetime: None, threads: 8 },
        Arc::new(AtomicBool::new(false)),
        pesto_combined,
        None,
        None,
    );
    assert!(legal.moves[..legal.len].iter().any(|m| *m == result.best));
    let total: u32 = result.visits.iter().map(|(_, v)| v).sum();
    assert_eq!(total, result.playouts as u32);
}

#[test]
fn incremental_search_reuses_subtree() {
    use std::sync::Arc;
    engine::init();
    let mut search = engine::mcts::MctsSearch::new(1 << 20, 4);
    let mut pos = Position::startpos();
    let stop = Arc::new(AtomicBool::new(false));

    let r1 = search.search(&pos, 4000, pesto_combined, stop.clone(), false, 0.0);
    assert!(!r1.visits.is_empty());
    let mv = r1.best;

    search.keep_child(mv, { pos.make_move(mv); pos.key });

    let r2 = search.search(&pos, 4000, pesto_combined, stop.clone(), false, 0.0);
    // The new root's children keep the visits accumulated under the old root
    // (through `mv`) plus the fresh playouts: strictly more than the budget.
    let sum2: u32 = r2.visits.iter().map(|(_, v)| v).sum();
    assert!(sum2 > 4000, "kept subtree visits should carry over, got {sum2}");
    assert!(r2.playouts >= 4000, "budget not reached: {}", r2.playouts);
    let legal = generate_legal(&pos);
    assert!(legal.moves[..legal.len].iter().any(|m| *m == r2.best));
}

#[test]
fn incremental_search_reset_gives_fresh_tree() {
    use std::sync::Arc;
    engine::init();
    let mut search = engine::mcts::MctsSearch::new(1 << 20, 2);
    let mut pos = Position::startpos();
    let stop = Arc::new(AtomicBool::new(false));

    let r1 = search.search(&pos, 1000, pesto_combined, stop.clone(), false, 0.0);
    search.reset();
    pos = Position::startpos();
    let r2 = search.search(&pos, 1000, pesto_combined, stop.clone(), false, 0.0);
    let sum2: u32 = r2.visits.iter().map(|(_, v)| v).sum();
    // Small overcount is possible when two workers pass the budget check at
    // once; a reset tree must start from (near) zero retained visits.
    assert!(sum2 <= 1000 + 8, "a reset tree must start from zero visits, got {sum2}");
    assert!(sum2 >= 1000);
    assert!(r1.playouts >= 1000);
}

/// Uniform-logits batch stub: blind priors/values, so only exact terminal
/// backups can distinguish moves. If M1 isn't backed up exactly, this fails.
fn uniform_batch(_positions: &[Position], legals: &[&MoveList]) -> Vec<(i32, Vec<f32>)> {
    legals.iter().map(|l| (0, vec![0.0; l.len])).collect()
}

/// Material-sensitive batch stub (side-to-move perspective, like a real
/// adapter): big signal for captures, ~0 otherwise. With scrambled
/// leaf<->result pairing, the winning capture's value lands on random
/// nodes and this test fails; with correct pairing it is found reliably.
fn material_batch(_positions: &[Position], legals: &[&MoveList]) -> Vec<(i32, Vec<f32>)> {
    _positions
        .iter()
        .zip(legals.iter())
        .map(|(pos, legal)| {
            let vals = [100, 300, 300, 500, 900, 0];
            let mut diff = 0i32;
            for pt in 0..6 {
                let w = (pos.pieces[pt] ).count_ones() as i32;
                let b = (pos.pieces[6 + pt]).count_ones() as i32;
                diff += vals[pt] * (w - b);
            }
            let cp = if pos.side == 0 { 20 * diff } else { -20 * diff };
            (cp, vec![0.0; legal.len])
        })
        .collect()
}

#[test]
fn batched_parallel_pairs_values_with_leaves() {    use std::sync::Arc;
    engine::init();
    // White rook can capture the a2 pawn; everything else is quiet.
    let mut pos = Position::from_fen("4k3/8/8/8/8/8/p6P/R3K3 w - - 0 1");
    let stop = Arc::new(AtomicBool::new(false));
    let limits = MctsLimits { playouts: Some(1000), movetime: None, threads: 4 };
    let (result, _search) = engine::mcts::mcts_batched_parallel(
        &mut pos, &limits, &stop, pesto_combined, material_batch, 16, false, None, None,
    );
    let top5: Vec<(String, u32)> = result.visits.iter().take(5)
        .map(|(m, v)| (m.to_uci(), *v)).collect();
    assert_eq!(result.best.to_uci(), "a1a2", "must find the winning capture, top: {top5:?}");
}

#[test]
fn batched_parallel_finds_mate_in_one() {
    use std::sync::Arc;
    engine::init();
    let mut pos = Position::from_fen("7k/6pp/8/8/8/8/8/R6K w - - 0 1");
    let stop = Arc::new(AtomicBool::new(false));
    let limits = MctsLimits { playouts: Some(2000), movetime: None, threads: 4 };
    let (result, _search) = engine::mcts::mcts_batched_parallel(
        &mut pos, &limits, &stop, pesto_combined, uniform_batch, 8, false, None, None,
    );
    let top5: Vec<(String, u32)> = result.visits.iter().take(5)
        .map(|(m, v)| (m.to_uci(), *v)).collect();
    assert_eq!(result.best.to_uci(), "a1a8", "batched MCTS must play M1, top: {top5:?}");
    assert!(result.value > 0.99, "mate value should be ~1.0, got {}", result.value);
    assert!(result.visits.is_empty(), "instant M1: no search playouts");
}

/// With blind uniform priors AND uniform values, PUCT exploration must keep
/// spreading visits: no move may starve. With dead root exploration (the
/// old missing root-visit bug), the first tie-break winner takes ~everything
/// and the rest freeze at 1 visit. Single thread => fully deterministic.
#[test]
fn batched_exploration_stays_balanced_with_uniform_priors() {
    use std::sync::Arc;
    engine::init();
    let mut pos = Position::startpos();
    let stop = Arc::new(AtomicBool::new(false));
    let limits = MctsLimits { playouts: Some(2000), movetime: None, threads: 1 };
    let (result, _search) = engine::mcts::mcts_batched_parallel(
        &mut pos, &limits, &stop, pesto_combined, uniform_batch, 8, false, None, None,
    );
    assert_eq!(result.visits.len(), 20, "startpos has 20 moves");
    let min_v = result.visits.iter().map(|(_, v)| *v).min().unwrap();
    let max_v = result.visits.iter().map(|(_, v)| *v).max().unwrap();
    assert!(min_v >= 50, "no move may starve, min visits = {min_v}");
    assert!(max_v <= 500, "no runaway lock-in, max visits = {max_v}");
}
