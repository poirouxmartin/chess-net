//! Benchmark: Rust MCTS forest playouts/s with the loaded net.
//! Run from workspace root: cargo run --release -p nn --example mcts_bench

use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use engine::mcts::{mcts_parallel, MctsLimits};

fn main() {
    engine::init();
    let path = std::env::args().nth(1).unwrap_or_else(|| "python/net.csnn".into());
    nn::load(&path).unwrap();
    let playouts: u64 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let threads: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(8);

    let mut pos = engine::position::Position::startpos();
    let stop = Arc::new(AtomicBool::new(false));
    let start = Instant::now();
    let (res, _search) = mcts_parallel(
        &mut pos,
        &MctsLimits { playouts: Some(playouts), movetime: None, threads },
        stop.clone(),
        nn::evaluate_loaded_combined,
        None,
        None,
    );
    let dt = start.elapsed();
    println!(
        "{} playouts in {:.2}s = {:.0} playouts/s ({} threads), best={}",
        res.playouts,
        dt.as_secs_f64(),
        res.playouts as f64 / dt.as_secs_f64(),
        threads,
        res.best.to_uci()
    );
}