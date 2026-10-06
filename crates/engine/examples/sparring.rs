//! Sparring harness: PV-leaf value vs subtree-mean move choice.
//!
//! Both sides run the IDENTICAL search (PeSTO eval, uniform priors, fixed
//! playouts, tree reuse, deterministic single thread); only the final move
//! choice differs:
//! - AVG picks the most-visited move (`MctsResult.best`, current default);
//! - PV  picks the move with the best PV-leaf value (`MctsSearch::pv_value`).
//!
//! Openings rotate and colors alternate. Score + Elo diff are reported from
//! AVG's perspective.
//!
//! Usage: `cargo run --release -p engine --example sparring [games] [playouts] [threads]`
//! Defaults: 50 games, 1000 playouts, 1 thread.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use engine::evaluate::evaluate;
use engine::mcts::MctsSearch;
use engine::movegen::{generate_legal, MoveList};
use engine::move_::Move;
use engine::position::Position;

fn pesto(pos: &Position, _legal: &MoveList) -> (i32, Vec<f32>) {
    (evaluate(pos), Vec::new())
}

const OPENINGS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1",
    "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq c3 0 1",
    "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",
    "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2",
];

const MAX_PLIES: u32 = 400;

fn pick_pv(search: &MctsSearch, legal: &MoveList, fallback: Move) -> Move {
    let mut best_mv = fallback;
    let mut best_v = f32::NEG_INFINITY;
    for m in legal.moves[..legal.len].iter().copied() {
        if let Some(v) = search.pv_value(m) {
            if v > best_v {
                best_v = v;
                best_mv = m;
            }
        }
    }
    best_mv
}

fn main() {
    engine::init();
    let args: Vec<String> = std::env::args().collect();
    let games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let playouts: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let threads: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
    println!("sparring: {games} games, {playouts} playouts, {threads} thread(s)");
    println!("AVG = most visits | PV = best PV-leaf value (both PeSTO, uniform priors)");

    let (mut w, mut d, mut l) = (0u32, 0u32, 0u32);
    let t0 = std::time::Instant::now();
    for g in 0..games {
        let fen = OPENINGS[g % OPENINGS.len()];
        let mut pos = Position::from_fen(fen);
        let avg_is_white = g % 2 == 0;
        let mut avg = MctsSearch::new(1 << 20, threads);
        let mut pv = MctsSearch::new(1 << 20, threads);
        let stop = Arc::new(AtomicBool::new(false));
        let mut result: Option<f32> = None; // white POV
        let mut final_ply = 0u32;
        for ply in 0..MAX_PLIES {
            final_ply = ply;
            let legal = generate_legal(&pos);
            if legal.len == 0 {
                result = Some(if pos.in_check() {
                    if pos.side == 0 { 0.0 } else { 1.0 }
                } else {
                    0.5
                });
                break;
            }
            if pos.insufficient_material() || pos.halfmove >= 100 {
                result = Some(0.5);
                break;
            }
            let white_to_move = pos.side == 0;
            let mv = if (white_to_move && avg_is_white) || (!white_to_move && !avg_is_white) {
                let r = avg.search(&pos, playouts, pesto, stop.clone(), false, 0.0);
                if r.best == Move::null() {
                    result = Some(0.5);
                    break;
                }
                r.best
            } else {
                let r = pv.search(&pos, playouts, pesto, stop.clone(), false, 0.0);
                if r.best == Move::null() {
                    result = Some(0.5);
                    break;
                }
                pick_pv(&pv, &legal, r.best)
            };
            pos.make_move(mv);
            avg.keep_child(mv, pos.key);
            pv.keep_child(mv, pos.key);
        }
        let r = result.unwrap_or(0.5);
        let avg_score = if avg_is_white { r } else { 1.0 - r };        match (avg_score - 0.5).signum() as i32 {
            1 => w += 1,
            -1 => l += 1,
            _ => d += 1,
        }
        let tag = if avg_is_white { "AVG white" } else { "AVG black" };
        println!(
            "game {:>3}: {} {:>4} {} plies (W:{w} D:{d} L:{l} for AVG)",
            g + 1,
            tag,
            if r == 1.0 { "1-0" } else if r == 0.0 { "0-1" } else { "1/2" },
            final_ply + 1,
        );
    }
    let n = games as f64;
    let s = (w as f64 + 0.5 * d as f64) / n;
    let elo = if s <= 0.0 || s >= 1.0 {
        format!("+/-inf (sweep {s:.3})")
    } else {
        format!("{:+.0}", -400.0 * (1.0 / s - 1.0).log10())
    };
    println!("----");
    println!("AVG: {w}W {d}D {l}L over {games} | score {s:.3} | Elo diff (AVG-PV): {elo}");
    println!("elapsed: {:.0}s", t0.elapsed().as_secs_f32());
}
