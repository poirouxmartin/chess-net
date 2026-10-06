//! UCI protocol loop.

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::evaluate::{evaluate, MATE};
use crate::movegen::generate_legal;
use crate::move_::Move;
use crate::position::Position;
use crate::mcts::{MctsLimits, MctsProgress};
use crate::search::{Limits, SearchResult, Searcher, EvalFn};
use crate::tt::TT;

static EVAL_FN: Mutex<Option<EvalFn>> = Mutex::new(None);

/// Register a custom evaluation backend (e.g. NNUE) before calling run().
pub fn set_eval(f: EvalFn) {
    *EVAL_FN.lock().unwrap() = Some(f);
}

pub fn run() {
    run_with_hook(None);
}

/// Run the UCI loop. `hook(name, value)` returns true when it handled a
/// non-standard option (used to wire up external eval backends).
pub fn run_with_hook(hook: Option<&dyn Fn(&str, &str) -> bool>) {
    let stdin = std::io::stdin();
    let stop = Arc::new(AtomicBool::new(false));
    let mut search: Option<thread::JoinHandle<()>> = None;
    let mut pos = Position::startpos();
    let mut hash_mb = 32;
    let mut threads: usize = 1;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }
        match tokens[0] {
            "uci" => {
                out("id name chess-net");
                out("id author chess-net");
                out("option name Hash type spin default 32 min 1 max 4096");
                out("option name Threads type spin default 1 min 1 max 64");
                out("option name EvalFile type string default <empty>");
                out("uciok");
            }
            "isready" => out("readyok"),
            "ucinewgame" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
            }
            "setoption" => {
                let mut i = 1;
                let mut name = String::new();
                if i < tokens.len() && tokens[i] == "name" {
                    i += 1;
                }
                while i < tokens.len() && tokens[i] != "value" {
                    if name.is_empty() {
                        name.push_str(tokens[i]);
                    } else {
                        name.push(' ');
                        name.push_str(tokens[i]);
                    }
                    i += 1;
                }
                // Skip the "value" separator if present; a bare `setoption`
                // or `setoption name X` (button-type) has no value part and
                // must not index out of bounds.
                if i < tokens.len() && tokens[i] == "value" {
                    i += 1;
                }
                let value = tokens.get(i..).map(|s| s.join(" ")).unwrap_or_default();
                if name == "Hash" {
                    if let Ok(v) = value.parse::<usize>() {
                        hash_mb = v.clamp(1, 4096);
                    }
                } else if name == "Threads" {
                    if let Ok(v) = value.parse::<usize>() {
                        threads = v.clamp(1, 64);
                    }
                } else if name == "EvalFile" {
                    if let Some(h) = hook {
                        if h("EvalFile", &value) {
                            continue;
                        }
                    }
                }
            }
            "position" => {
                // A new position invalidates any running search: stop it
                // first so no stale bestmove is ever printed afterwards.
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
                pos = parse_position(&tokens[1..]);
            }
            "go" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
                let limits = parse_go(&tokens[1..]);
                let eval = (*EVAL_FN.lock().unwrap()).unwrap_or(evaluate);
                // Also update the MCTS eval to match
                crate::mcts::set_mcts_eval(eval);
                let hash = hash_mb;
                let n_threads = threads;
                let stop_clone = stop.clone();
                let pos_clone = pos.clone();
                search = Some(thread::spawn(move || {
                    lazy_smp_search(pos_clone, limits, stop_clone, eval, hash, n_threads);
                }));
            }
            "stop" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
            }
            "perft" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
                let depth: u32 = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
                let threads: usize = tokens
                    .iter()
                    .position(|&t| t == "threads")
                    .and_then(|i| tokens.get(i + 1))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1);
                let n = crate::perft::perft_parallel(&mut pos, depth, threads);
                out(&format!("nodes {}", n));
            }
            "divide" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
                let depth: u32 = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
                let n = crate::perft::divide(&mut pos, depth);
                out(&format!("nodes {}", n));
            }
            "moves" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
                let ml = generate_legal(&pos);
                let s = ml
                    .iter()
                    .map(|m| m.to_uci())
                    .collect::<Vec<_>>()
                    .join(" ");
                out(&s);
            }
            "d" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
                out(&pos.to_fen());
            }
            "mcts" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
                // mcts [playouts N] [threads N] [temperature F] [batch N]
                let playouts = tokens
                    .iter()
                    .position(|&t| t == "playouts")
                    .and_then(|i| tokens.get(i + 1))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(800u64);
                let threads = tokens
                    .iter()
                    .position(|&t| t == "threads")
                    .and_then(|i| tokens.get(i + 1))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1usize);
                let _temperature = tokens
                    .iter()
                    .position(|&t| t == "temperature")
                    .and_then(|i| tokens.get(i + 1))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0.0f32);
                let batch_size = tokens
                    .iter()
                    .position(|&t| t == "batch")
                    .and_then(|i| tokens.get(i + 1))
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(32usize);
                // Ensure the MCTS eval matches the current eval setting
                let eval = (*EVAL_FN.lock().unwrap()).unwrap_or(evaluate);
                crate::mcts::set_mcts_eval(eval);
                let limits = MctsLimits {
                    playouts: Some(playouts),
                    movetime: None,
                    threads,
                };
                // Share the GLOBAL stop flag (not a local one) so `stop` /
                // `quit` interrupt this synchronous command, and the next
                // `go` starts from a clean state.
                let stop = stop.clone();
                let mut pos_clone = pos.clone();
                let mut on_progress = |p: &MctsProgress| {
                    let pv: Vec<String> = p.moves.iter().map(|info| info.mv.to_uci()).collect();
                    out(&format!(
                        "info depth 1 score cp {} time {} nodes {} pv {}",
                        crate::mcts::cp_from_prob(p.value),
                        p.time_ms,
                        p.playouts,
                        pv.join(" ")
                    ));
                };
                // Use batched MCTS when a batch eval function is available
                let r = if let Some(batch_eval) = crate::mcts::get_batch_eval() {
                    if threads > 1 {
                        let (res, _search) = crate::mcts::mcts_batched_parallel(
                            &mut pos_clone,
                            &limits,
                            &stop,
                            crate::mcts::mcts_value_fn,
                            batch_eval,
                            batch_size,
                            false,
                            Some(&mut on_progress),
                            None,
                        );
                        res
                    } else {
                        crate::mcts::mcts_batched(
                            &mut pos_clone,
                            &limits,
                            &stop,
                            crate::mcts::mcts_value_fn,
                            batch_eval,
                            batch_size,
                            false,
                            Some(&mut on_progress),
                        )
                    }
                } else {
                    crate::mcts::mcts_root(
                        &mut pos_clone,
                        &limits,
                        &stop,
                        crate::mcts::mcts_value_fn,
                        Some(&mut on_progress),
                    )
                };
                let _pv: Vec<String> = r.visits.iter().take(5).map(|(m, _)| m.to_uci()).collect();
                // Terminal positions have no legal move: spec says 0000,
                // never a pseudo-legal-looking square pair.
                if r.best == Move::null() {
                    out("bestmove 0000");
                } else {
                    out(&format!("bestmove {}", r.best.to_uci()));
                }
            }
            "eval" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
                let eval = (*EVAL_FN.lock().unwrap()).unwrap_or(evaluate);
                out(&format!("eval {}", eval(&pos)));
            }
            "quit" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                break;
            }
            _ => {}
        }
    }
}

fn lazy_smp_search(
    pos: Position,
    limits: Limits,
    stop: Arc<AtomicBool>,
    eval: EvalFn,
    hash_mb: usize,
    n_threads: usize,
) {
    let tt = Arc::new(TT::new(hash_mb));
    let best = Arc::new(Mutex::<Option<SearchResult>>::new(None));
    let handles: Vec<_> = (0..n_threads)
        .map(|t| {
            let tt = tt.clone();
            let stop = stop.clone();
            let limits = limits.clone();
            let pos = pos.clone();
            let best = best.clone();
            // Depth diversity across helper threads (0/1/2): threads that
            // would otherwise skip the whole loop (start_depth > max_depth)
            // return depth 0 and are ignored in the aggregation below.
            let offset = (t as i32).min(2);
            thread::spawn(move || {
                let mut pos = pos;
                let mut s = Searcher::with_tt(tt, offset);
                let r = s.think(&mut pos, &limits, &stop, eval);
                // Skip threads that never completed an iteration (e.g. an
                // offset past a shallow depth limit): their null bestmove
                // must never outrank a real result by score.
                if r.depth == 0 {
                    return;
                }
                let mut guard = best.lock().unwrap();
                let dominated = guard.as_ref().map_or(true, |old| {
                    r.score > old.score || (r.score == old.score && r.depth > old.depth)
                });
                if dominated {
                    *guard = Some(r);
                }
            })
        })
        .collect();
    for h in handles {
        let _ = h.join();
    }
    let result = best.lock().unwrap().take();
    if let Some(r) = result {
        report(&r);
    }
}

fn join(search: &mut Option<thread::JoinHandle<()>>) {
    if let Some(h) = search.take() {
        let _ = h.join();
    }
}

fn out(s: &str) {
    let mut stdout = std::io::stdout();
    let _ = writeln!(stdout, "{s}");
    let _ = stdout.flush();
}

fn report(r: &SearchResult) {
    let score = if r.score >= MATE - 200 {
        format!("mate {}", (MATE - r.score + 1) / 2)
    } else if r.score <= -MATE + 200 {
        format!("mate -{}", (MATE + r.score + 1) / 2)
    } else {
        format!("cp {}", r.score)
    };
    let pv: Vec<String> = r.pv.iter().map(|m| m.to_uci()).collect();
    out(&format!(
        "info depth {} score {} nodes {} time {} pv {}",
        r.depth,
        score,
        r.nodes,
        r.time_ms,
        pv.join(" ")
    ));
    // Terminal positions have no legal move: spec says 0000.
    if r.best == Move::null() {
        out("bestmove 0000");
    } else {
        out(&format!("bestmove {}", r.best.to_uci()));
    }
}

fn parse_position(tokens: &[&str]) -> Position {
    let mut pos = Position::startpos();
    let mut i = 0;
    if i < tokens.len() && tokens[i] == "startpos" {
        i += 1;
    } else if i < tokens.len() && tokens[i] == "fen" {
        let mut fen = Vec::new();
        i += 1;
        while i < tokens.len() && tokens[i] != "moves" {
            fen.push(tokens[i]);
            i += 1;
        }
        pos = Position::from_fen(&fen.join(" "));
    }
    if i < tokens.len() && tokens[i] == "moves" {
        for m in &tokens[i + 1..] {
            if let Some(mv) = parse_move(&pos, m) {
                pos.make_move(mv);
            }
        }
    }
    pos
}

fn parse_move(pos: &Position, s: &str) -> Option<Move> {
    let moves = generate_legal(pos);
    for m in moves.iter() {
        if m.to_uci() == s {
            return Some(m);
        }
    }
    None
}

fn parse_go(tokens: &[&str]) -> Limits {
    let mut l = Limits::default();
    let mut i = 0;
    while i < tokens.len() {
        match tokens[i] {
            "movetime" => {
                l.movetime = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "wtime" => {
                l.wtime = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "btime" => {
                l.btime = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "winc" => {
                l.winc = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "binc" => {
                l.binc = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            "depth" => {
                l.depth = tokens.get(i + 1).and_then(|s| s.parse().ok());
                i += 2;
            }
            _ => i += 1,
        }
    }
    l
}