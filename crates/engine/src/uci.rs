//! UCI protocol loop.

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::evaluate::{evaluate, MATE};
use crate::movegen::generate_legal;
use crate::move_::Move;
use crate::position::Position;
use crate::search::{Limits, SearchResult, Searcher, EvalFn};

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
                i += 1;
                let value = tokens[i..].join(" ");
                if name == "Hash" {
                    if let Ok(v) = value.parse() {
                        hash_mb = v;
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
                pos = parse_position(&tokens[1..]);
            }
            "go" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
                let limits = parse_go(&tokens[1..]);
                let mut target = pos.clone();
                let s = stop.clone();
                let eval = (*EVAL_FN.lock().unwrap()).unwrap_or(evaluate);
                let hash = hash_mb;
                search = Some(thread::spawn(move || {
                    let mut searcher = Searcher::new(hash);
                    let r = searcher.think(&mut target, &limits, &s, eval);
                    report(&r);
                }));
            }
            "stop" => {
                stop.store(true, Ordering::Relaxed);
                join(&mut search);
                stop.store(false, Ordering::Relaxed);
            }
            "perft" => {
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
                let depth: u32 = tokens.get(1).and_then(|s| s.parse().ok()).unwrap_or(1);
                let n = crate::perft::divide(&mut pos, depth);
                out(&format!("nodes {}", n));
            }
            "moves" => {
                let ml = generate_legal(&pos);
                let s = ml
                    .iter()
                    .map(|m| m.to_uci())
                    .collect::<Vec<_>>()
                    .join(" ");
                out(&s);
            }
            "d" => {
                out(&pos.to_fen());
            }
            "eval" => {
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
    out(&format!("bestmove {}", r.best.to_uci()));
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