//! Temporary single-thread search speed bench (removed after optimization).

use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use engine::evaluate::{evaluate, evaluate_breakdown};
use engine::movegen::{generate_legal, generate_pseudo, MoveList};
use engine::perft::perft;
use engine::position::Position;
use engine::search::{EvalFn, Limits, Searcher};

#[test]
fn micro_bench() {
    engine::init();
    let mut pos = Position::startpos();
    let mut kp = Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    let mut acc = 0i64;

    // make+unmake only: cycle through a fixed legal move list.
    let fixed = generate_legal(&pos);
    let t = Instant::now();
    let mut occ = 0u64;
    for i in 0..10_000_000u64 {
        let m = fixed.get((i % fixed.len as u64) as usize);
        let undo = pos.make_move(m);
        pos.unmake_move(undo);
        occ |= pos.occ;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("make+unmake: {:.1}M/s ({:.1} ns)", 1e7 / dt / 1e6, dt * 1e9 / 1e7);

    let t = Instant::now();
    for _ in 0..3_000_000 {
        let ml = generate_pseudo(&pos, false, false);
        acc += ml.len as i64;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("generate_pseudo: {:.1}M/s ({:.1} ns)", 3e6 / dt / 1e6, dt * 1e9 / 3e6);

    let t = Instant::now();
    for _ in 0..3_000_000 {
        let ml = generate_legal(&pos);
        acc += ml.len as i64;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("generate_legal: {:.1}M/s ({:.1} ns)", 3e6 / dt / 1e6, dt * 1e9 / 3e6);

    let t = Instant::now();
    for _ in 0..10_000_000 {
        let ml = std::hint::black_box(MoveList::new());
        acc += std::hint::black_box(&ml).moves[7].0 as i64;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("MoveList::new: {:.1}M/s ({:.2} ns)", 1e7 / dt / 1e6, dt * 1e9 / 1e7);

    let t = Instant::now();
    for i in 0..10_000_000u64 {
        let m = fixed.get((i % fixed.len as u64) as usize);
        let undo = pos.make_move(m);
        acc += evaluate(&pos) as i64;
        pos.unmake_move(undo);
    }
    let dt = t.elapsed().as_secs_f64();
    println!("make+eval+unmake (startpos): {:.1}M/s ({:.2} ns)", 1e7 / dt / 1e6, dt * 1e9 / 1e7);

    let kfixed = generate_legal(&kp);
    let t = Instant::now();
    for i in 0..10_000_000u64 {
        let m = kfixed.get((i % kfixed.len as u64) as usize);
        let undo = kp.make_move(m);
        acc += evaluate(&kp) as i64;
        kp.unmake_move(undo);
    }
    let dt = t.elapsed().as_secs_f64();
    println!("make+eval+unmake (kiwipete): {:.1}M/s ({:.2} ns)", 1e7 / dt / 1e6, dt * 1e9 / 1e7);

    let t = Instant::now();
    for i in 0..10_000_000u64 {
        let m = kfixed.get((i % kfixed.len as u64) as usize);
        let undo = kp.make_move(m);
        acc += kp.in_check() as i64;
        kp.unmake_move(undo);
    }
    let dt = t.elapsed().as_secs_f64();
    println!("make+in_check+unmake: {:.1}M/s ({:.2} ns)", 1e7 / dt / 1e6, dt * 1e9 / 1e7);

    let t = Instant::now();
    for i in 0..10_000_000u64 {
        let m = kfixed.get((i % kfixed.len as u64) as usize);
        let undo = kp.make_move(m);
        acc += kp.pinned().count_ones() as i64;
        kp.unmake_move(undo);
    }
    let dt = t.elapsed().as_secs_f64();
    println!("make+pinned+unmake: {:.1}M/s ({:.2} ns)", 1e7 / dt / 1e6, dt * 1e9 / 1e7);

    let mut s = Searcher::new(64);
    let mut tt_acc = 0u32;
    let t = Instant::now();
    for i in 0..20_000_000u64 {
        tt_acc = tt_acc.wrapping_add(s.tt.probe(pos.key ^ i).map_or(0, |e| e.mv));
    }
    let dt = t.elapsed().as_secs_f64();
    println!("tt probe (miss): {:.1}M/s ({:.2} ns) acc={}", 2e7 / dt / 1e6, dt * 1e9 / 2e7, tt_acc & 0xFF);

    let t = Instant::now();
    for _ in 0..5_000_000 {
        let mut ml = generate_pseudo(&pos, false, false);
        s.order_moves(&pos, &mut ml, None, 0);
        acc += ml.get(0).0 as i64;
    }
    let dt = t.elapsed().as_secs_f64();
    println!("pseudo+order_moves: {:.1}M/s ({:.1} ns)", 5e6 / dt / 1e6, dt * 1e9 / 5e6);
    let _ = (acc, occ);
}

static EVAL_CALLS: AtomicU64 = AtomicU64::new(0);
fn eval_counted(pos: &Position) -> i32 {
    EVAL_CALLS.fetch_add(1, Ordering::Relaxed);
    evaluate(pos)
}

#[test]
fn pseudo_legality_agrees() {
    engine::init();
    let mut rng = 0x12345678u64.wrapping_mul(0x9E3779B97F4A7C15);
    let mut pos = Position::startpos();
    for i in 0..20000 {
        let in_check = pos.in_check();
        let pseudo = generate_pseudo(&pos, false, in_check);
        let mut accepted = Vec::new();
        let us = pos.side;
        let pinned = pos.pinned();
        let king = pos.king_sq[us];
        for k in 0..pseudo.len {
            let m = pseudo.get(k);
            let verify = in_check || m.is_en_passant() || m.from() == king || (pinned & (1u64 << m.from())) != 0;
            let undo = pos.make_move(m);
            let ok = !verify || pos.king_safe(us);
            pos.unmake_move(undo);
            if ok {
                accepted.push(m);
            }
        }
        let legal = generate_legal(&pos);
        assert_eq!(accepted.len(), legal.len, "move count mismatch at move {i}: pseudo {} vs legal {}", accepted.len(), legal.len);
        let mut legal_set = legal.iter().collect::<Vec<_>>();
        legal_set.sort_by_key(|m| m.0);
        let mut acc = accepted.clone();
        acc.sort_by_key(|m| m.0);
        for (a, l) in acc.iter().zip(legal_set.iter()) {
            assert_eq!(a.0, l.0, "move mismatch at move {i}");
        }
        if legal.len == 0 {
            break;
        }
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let m = legal.get((rng as usize) % legal.len);
        pos.make_move(m);
    }
    println!("pseudo+king_safe == generate_legal over 20000 random moves");
}

#[test]
fn eval_consistency() {
    engine::init();
    let mut rng = 0x12345678u64.wrapping_mul(0x9E3779B97F4A7C15);
    let mut pos = Position::startpos();
    for i in 0..5000 {
        let inc = evaluate(&pos);
        let bd = evaluate_breakdown(&pos);
        assert_eq!(inc, bd.score, "eval mismatch at move {i}: {} vs {}", inc, bd.score);
        let legal = generate_legal(&pos);
        if legal.len == 0 {
            break;
        }
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let m = legal.get((rng as usize) % legal.len);
        pos.make_move(m);
    }
    println!("eval consistent over 5000 random moves");
}

fn bench(depth: i32) {
    let mut best_dt = f64::INFINITY;
    let mut best_nodes = 0;
    let mut best_move = 0;
    for _ in 0..5 {
        let mut pos = Position::startpos();
        let stop = AtomicBool::new(false);
        let mut s = Searcher::new(64);
        let t = Instant::now();
        let r = s.think(&mut pos, &Limits { depth: Some(depth), ..Default::default() }, &stop, evaluate);
        let dt = t.elapsed().as_secs_f64();
        if dt < best_dt {
            best_dt = dt;
            best_nodes = r.nodes;
            best_move = r.best.0;
        }
    }
    println!(
        "depth {}: {} nodes in {:.3}s = {:.1} Mn/s (best-of-5) best {}",
        depth,
        best_nodes,
        best_dt,
        best_nodes as f64 / best_dt / 1e6,
        best_move
    );
}

#[test]
fn bench_search() {
    engine::init();
    bench(9);
    bench(12);
}

#[cfg(feature = "profiling")]
#[test]
fn prof_search() {
    engine::init();
    use engine::search::PROF;
    for mb in [64usize, 4, 2, 1] {
        let mut pos = Position::startpos();
        let stop = AtomicBool::new(false);
        let mut s = Searcher::new(mb);
        let t = Instant::now();
        let r = s.think(&mut pos, &Limits { depth: Some(12), ..Default::default() }, &stop, evaluate);
        let dt = t.elapsed().as_secs_f64();
        println!(
            "tt={}MB depth 12: {} nodes in {:.3}s = {:.1} Mn/s best {}",
            mb, r.nodes, dt, r.nodes as f64 / dt / 1e6, r.best.0
        );
    }
    let mut pos = Position::startpos();
    let stop = AtomicBool::new(false);
    let mut s = Searcher::new(64);
    let t = Instant::now();
    let r = s.think(&mut pos, &Limits { depth: Some(9), ..Default::default() }, &stop, evaluate);
    let dt = t.elapsed().as_secs_f64();
    let n = r.nodes as f64;
    println!("nodes {} in {:.3}s = {:.1} Mn/s best {}", r.nodes, dt, r.nodes as f64 / dt / 1e6, r.best.0);
    let load = |v: u64| v as f64 / n;
    println!(
        "negamax {:.1} qnode {:.1} | movegen {:.1} movegen_q {:.1} | order {:.1} order_cap {:.1} | make {:.1} make_q {:.1} | see {:.1} rep {:.1}",
        load(PROF.negamax.load(Ordering::Relaxed)),
        load(PROF.qnode.load(Ordering::Relaxed)),
        load(PROF.movegen.load(Ordering::Relaxed)),
        load(PROF.movegen_q.load(Ordering::Relaxed)),
        load(PROF.order.load(Ordering::Relaxed)),
        load(PROF.order_cap.load(Ordering::Relaxed)),
        load(PROF.make.load(Ordering::Relaxed)),
        load(PROF.make_q.load(Ordering::Relaxed)),
        load(PROF.see.load(Ordering::Relaxed)),
        load(PROF.rep.load(Ordering::Relaxed)),
    );
}

#[test]
fn debug_key() {
    engine::init();
    let mut pos = Position::startpos();
    let orig = pos.key;
    println!("startpos key {:#016x}", orig);
    let mut m = engine::move_::Move::null();
    for i in 0..generate_legal(&pos).len {
        let cand = generate_legal(&pos).get(i);
        if cand.to_uci() == "d2d3" {
            m = cand;
        }
    }
    let undo = pos.make_move(m);
    let mid = pos.key;
    println!("after make d2d3: {:#016x}", mid);
    pos.unmake_move(undo);
    println!("after unmake: {:#016x} expect {:#016x}", pos.key, orig);
    assert_eq!(pos.key, orig);
}

#[test]
fn zobrist_roundtrip() {
    engine::init();
    let mut rng = 0x12345678u64.wrapping_mul(0x9E3779B97F4A7C15);
    let mut pos = Position::startpos();
    let orig = pos.key;
    for i in 0..2000 {
        let legal = generate_legal(&pos);
        if legal.len == 0 {
            break;
        }
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let m = legal.get((rng as usize) % legal.len);
        let undo = pos.make_move(m);
        let after_make = pos.key;
        pos.unmake_move(undo);
        assert_eq!(pos.key, orig, "key mismatch after unmake at move {i} ({}): make key {:#016x}", m.to_uci(), after_make);
    }
    println!("zobrist round-trip OK over 2000 random make/unmake");
}

#[test]
fn debug_m1() {
    engine::init();
    let mut pos = Position::from_fen("6k1/5ppp/8/8/8/8/8/1K1R4 w - - 0 1");
    println!("king_sq W={} B={} occ={:#x}", pos.king_sq[0], pos.king_sq[1], pos.occ);
    let pseudo = generate_pseudo(&pos, false, false);
    println!("pseudo {} moves", pseudo.len);
    for i in 0..pseudo.len {
        let m = pseudo.get(i);
        let undo = pos.make_move(m);
        if pos.king_sq[0] == usize::MAX || pos.king_sq[1] == usize::MAX {
            println!("KING CORRUPT after {}", m.to_uci());
        }
        let ok = pos.king_safe(0);
        println!("  {} king_safe={} ksq={}/{}", m.to_uci(), ok, pos.king_sq[0], pos.king_sq[1]);
        pos.unmake_move(undo);
    }
    for d in 1..=3 {
        let stop = AtomicBool::new(false);
        let mut s = Searcher::new(64);
        let r = s.think(&mut pos, &Limits { depth: Some(d), ..Default::default() }, &stop, evaluate);
        println!("depth {}: best {} score {} nodes {}", d, r.best.to_uci(), r.score, r.nodes);
    }
}

#[test]
fn sanity_positions() {
    engine::init();
    // Mate in 1 (back-rank): 1. Rd8#.
    let mut m1 = Position::from_fen("6k1/5ppp/8/8/8/8/8/1K1R4 w - - 0 1");
    let stop = AtomicBool::new(false);
    let mut s = Searcher::new(64);
    let r = s.think(&mut m1, &Limits { depth: Some(3), ..Default::default() }, &stop, evaluate);
    println!("M1 pos depth 3: best {} nodes {}", r.best.to_uci(), r.nodes);
    assert_eq!(r.best.to_uci(), "d1d8", "expected Rd8# to be found");

    // Kiwipete at depth 8: engine must not hang and should return a move.
    let mut kp = Position::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    let stop = AtomicBool::new(false);
    let mut s = Searcher::new(64);
    let r = s.think(&mut kp, &Limits { depth: Some(8), ..Default::default() }, &stop, evaluate);
    println!("kiwipete depth 8: best {} score {} nodes {}", r.best.to_uci(), r.score, r.nodes);
    assert!(r.best.0 != 0, "kiwipete must return a move");
}

#[test]
fn tt_consistency() {
    // The TT is a pure cache: enabling it must never change the reported root
    // score (fail-soft alpha-beta returns the exact depth-d value regardless of
    // move order). Any mismatch means a stale/colliding entry or a bad bound.
    engine::init();
    let cases: &[(&str, i32)] = &[
        ("rnbqkbnr/pppppppp/8/8/8/8/8/RNBQKBNR w KQkq - 0 1", 5),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 5),
        ("6k1/5ppp/8/8/8/8/8/1K1R4 w - - 0 1", 4),
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", 4),
    ];
    for &(fen, depth) in cases {
        let stop = AtomicBool::new(false);
        let mut pos = Position::from_fen(fen);
        let mut s_on = Searcher::new(64);
        let r_on = s_on.think(&mut pos, &Limits { depth: Some(depth), ..Default::default() }, &stop, evaluate);
        let mut pos = Position::from_fen(fen);
        let mut s_off = Searcher::new(64);
        s_off.use_tt = false;
        let r_off = s_off.think(&mut pos, &Limits { depth: Some(depth), ..Default::default() }, &stop, evaluate);
        assert_eq!(
            r_on.score, r_off.score,
            "TT on/off score mismatch at depth {depth} on {fen}: tt={} ({} nodes) no-tt={} ({} nodes)",
            r_on.score, r_on.nodes, r_off.score, r_off.nodes
        );
        println!("tt_consistency ok {fen} d{depth}: tt={} ({}n) no-tt={} ({}n) best {}",
            r_on.score, r_on.nodes, r_off.score, r_off.nodes, r_on.best.to_uci());
    }
}

#[test]
fn pv_sanity() {
    // PV moves must be legal at every step, and a mate score must correspond
    // to a PV that really ends in checkmate. Catches TT mate-score corruption,
    // PV-table bugs and blind TT-move plays.
    engine::init();
    let m1 = "6k1/5ppp/8/8/8/8/8/1K1R4 w - - 0 1";
    for use_tt in [true, false] {
        let mut pos = Position::from_fen(m1);
        let stop = AtomicBool::new(false);
        let mut s = Searcher::new(64);
        s.use_tt = use_tt;
        let r = s.think(&mut pos, &Limits { depth: Some(3), ..Default::default() }, &stop, evaluate);
        assert_eq!(r.score, engine::evaluate::MATE - 1, "tt={use_tt}: expected mate in 1, got score {}", r.score);
        assert_eq!(r.best.to_uci(), "d1d8", "tt={use_tt}: expected Rd8#");
        let mut p = pos;
        for m in &r.pv {
            let legal = generate_legal(&p);
            assert!(legal.iter().any(|x| x == *m), "tt={use_tt}: illegal PV move {} at {}", m.to_uci(), p.to_fen());
            p.make_move(*m);
        }
        assert!(
            p.in_check() && generate_legal(&p).len == 0,
            "tt={use_tt}: PV does not end in checkmate: {}",
            p.to_fen()
        );
    }

    let mut rng = 0xDEADBEEFu64.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..8 {
        let mut pos = Position::startpos();
        for _ in 0..10 {
            let legal = generate_legal(&pos);
            if legal.len == 0 {
                break;
            }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let m = legal.get((rng as usize) % legal.len);
            pos.make_move(m);
        }
        let fen = pos.to_fen();
        let stop = AtomicBool::new(false);
        let mut s = Searcher::new(64);
        let r = s.think(&mut pos, &Limits { depth: Some(4), ..Default::default() }, &stop, evaluate);
        let mut p = pos;
        for m in &r.pv {
            let legal = generate_legal(&p);
            assert!(
                legal.iter().any(|x| x == *m),
                "illegal PV move {} at {} ({fen})",
                m.to_uci(),
                p.to_fen()
            );
            p.make_move(*m);
        }
        if (engine::evaluate::MATE - r.score).abs() <= 64 {
            assert!(
                p.in_check() && generate_legal(&p).len == 0,
                "mate score {} but PV does not end in checkmate: {} ({fen})",
                r.score,
                p.to_fen()
            );
        }
        println!(
            "pv_sanity ok {} best {} score {} pv {:?} nodes {}",
            fen,
            r.best.to_uci(),
            r.score,
            r.pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>(),
            r.nodes
        );
    }
}

#[test]
fn bench_eval_calls() {
    engine::init();
    EVAL_CALLS.store(0, Ordering::Relaxed);
    let mut pos = Position::startpos();
    let stop = AtomicBool::new(false);
    let mut s = Searcher::new(64);
    let t = Instant::now();
    let r = s.think(&mut pos, &Limits { depth: Some(9), ..Default::default() }, &stop, eval_counted as EvalFn);
    let dt = t.elapsed().as_secs_f64();
    let calls = EVAL_CALLS.load(Ordering::Relaxed);
    println!(
        "depth 9 (counted eval): {} nodes, {} eval calls = {:.2} eval/node, {:.3}s = {:.1} Mn/s best {}",
        r.nodes,
        calls,
        calls as f64 / r.nodes as f64,
        dt,
        r.nodes as f64 / dt / 1e6,
        r.best.0
    );
}

#[test]
fn bench_perft() {
    engine::init();
    for d in [6u32, 7] {
        let mut pos = Position::startpos();
        let t = Instant::now();
        let n = perft(&mut pos, d);
        let dt = t.elapsed().as_secs_f64();
        println!("perft depth {}: {} nodes in {:.3}s = {:.1} Mn/s", d, n, dt, n as f64 / dt / 1e6);
    }
}