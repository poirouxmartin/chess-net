//! Self-play game generation with the loaded net + parallel MCTS.
//!
//! Each game: incremental MCTS (the tree is kept and re-rooted after each move,
//! AlphaZero-style), a move is picked from the root visit distribution
//! (temperature 1 until `temp_drop` plies, then greedy), and every position is
//! recorded with its active features, all legal moves + visit counts, and the
//! game result in white POV.

use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use engine::mcts::MctsSearch;
use engine::move_::Move;
use engine::position::Position;

use crate::data::{self, Sample};
use crate::Rng;

pub const MAX_ACTIVE: usize = data::MAX_ACTIVE;

/// Per-position record: active features, then (legal move, visit count).
type Record = (Vec<u32>, Vec<(Move, u32)>);

pub struct GenOptions {
    pub net_path: String,
    pub games: u32,
    /// MCTS playouts per position.
    pub iters: u32,
    /// MCTS worker threads.
    pub threads: usize,
    /// Plies played with temperature 1 before switching to greedy.
    pub temp_drop: u32,
    pub max_plies: u32,
    pub out_path: String,
}

pub fn generate(opts: &GenOptions) -> Result<(), String> {
    nn::load(&opts.net_path)?;
    let feat = nn::loaded_feat().unwrap_or(2);
    let mut rng = Rng::new(opts.games as u64 ^ 0x1234_5678_9abc_def1);

    let mut search = MctsSearch::new(opts.iters as usize * 24 + (1 << 21), opts.threads);
    let mut all: Vec<Sample> = Vec::new();
    let mut total_rebuilds = 0u64;
    for g in 0..opts.games {
        let before = search.rebuilds();
        let samples = play_game(&mut rng, &mut search, opts, feat, g == 0);
        total_rebuilds += search.rebuilds() - before;
        all.extend(samples);
        if g % 10 == 0 {
            println!("game {} ({} positions so far, {} tree rebuilds)", g, all.len(), total_rebuilds);
        }
    }
    data::write_samples(&opts.out_path, &all)?;
    println!("wrote {} positions to {}", all.len(), opts.out_path);
    Ok(())
}

fn play_game(rng: &mut Rng, search: &mut MctsSearch, opts: &GenOptions, feat: u32, first: bool) -> Vec<Sample> {
    let mut pos = Position::startpos();
    let mut records: Vec<Record> = Vec::new();
    let stop = Arc::new(AtomicBool::new(false));
    search.reset();

    for ply in 0..opts.max_plies {
        let legal = engine::movegen::generate_legal(&pos);
        if legal.len == 0 {
            break;
        }
        let res = search.search(&pos, opts.iters as u64, nn::evaluate_loaded_combined, stop.clone(), true, 1.0);
        if res.visits.is_empty() {
            break;
        }

        let mut feats = [0usize; MAX_ACTIVE];
        let n = nn::active_features(feat, &pos, &mut feats);
        records.push((
            feats[..n].iter().map(|f| *f as u32).collect(),
            res.visits.clone(),
        ));

        let temp = if ply < opts.temp_drop { 1.0 } else { 0.0 };
        let mv = match pick_move(&res.visits, temp, rng) {
            Some(m) => m,
            None => break,
        };
        search.keep_child(mv);
        pos.make_move(mv);
    }

    // White POV result for the whole game.
    let white_pov = if legal_is_empty(&pos) {
        if pos.in_check() {
            // Side to move is mated: the other side won.
            if pos.side == 0 { 0.0 } else { 1.0 }
        } else {
            0.5
        }
    } else {
        0.5
    };

    let mut samples = Vec::with_capacity(records.len());
    for (active, visits) in records {
        let moves: Vec<Move> = visits.iter().map(|(m, _)| *m).collect();
        let counts: Vec<u32> = visits.iter().map(|(_, v)| *v).collect();
        samples.push(Sample { active, moves, visits: counts, result: white_pov });
    }
    if first && !samples.is_empty() {
        println!("  sample: n_active={} n_moves={} result={}", samples[0].active.len(), samples[0].moves.len(), white_pov);
    }
    samples
}

fn legal_is_empty(pos: &Position) -> bool {
    engine::movegen::generate_legal(pos).len == 0
}

/// Pick a root move from the visit distribution. `temp == 0` picks the
/// most-visited move (ties resolve to the first of the max).
fn pick_move(visits: &[(Move, u32)], temp: f32, rng: &mut Rng) -> Option<Move> {
    let total: u64 = visits.iter().map(|(_, v)| *v as u64).sum();
    if total == 0 {
        return None;
    }
    if temp <= 0.0 {
        return visits
            .iter()
            .max_by_key(|(_, v)| *v)
            .map(|(m, _)| *m);
    }
    let inv = 1.0 / temp as f64;
    let weights: Vec<f64> = visits.iter().map(|(_, v)| (*v as f64).powf(inv)).collect();
    let sum: f64 = weights.iter().sum();
    let mut r = rng.f64() * sum;
    for (i, w) in weights.iter().enumerate() {
        r -= w;
        if r < 0.0 {
            return Some(visits[i].0);
        }
    }
    visits.last().map(|(m, _)| *m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_greedy() {
        let mut rng = Rng::new(7);
        let visits = vec![(Move(1), 3u32), (Move(2), 10u32), (Move(3), 5u32)];
        for _ in 0..20 {
            assert_eq!(pick_move(&visits, 0.0, &mut rng), Some(Move(2)));
        }
    }

    #[test]
    fn pick_empty() {
        let mut rng = Rng::new(7);
        assert_eq!(pick_move(&[], 1.0, &mut rng), None);
    }

    #[test]
    fn pick_temperature_samples_all() {
        let mut rng = Rng::new(42);
        let visits = vec![(Move(1), 3u32), (Move(2), 2u32), (Move(3), 1u32)];
        let mut seen = [false; 4];
        for _ in 0..2000 {
            let m = pick_move(&visits, 1.0, &mut rng).unwrap();
            seen[m.from()] = true;
        }
        assert!(seen[1] && seen[2] && seen[3]);
    }
}