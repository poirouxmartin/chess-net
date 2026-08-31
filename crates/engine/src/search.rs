//! Alpha-beta search: iterative deepening, PVS, quiescence, TT, killers,
//! history heuristic, null-move pruning, time management.

use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "profiling")]
use std::sync::atomic::AtomicU64;
use std::cmp::Reverse;
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use crate::bitboard::{bit, pop_lsb};
use crate::attack::{knight_attacks, pawn_attacks};
use crate::magic::{bishop_attacks, queen_attacks, rook_attacks};
use crate::evaluate::{MATE, INF};
use crate::movegen::{generate_legal, generate_pseudo_into, MoveList};
use crate::move_::Move;
use crate::position::*;
use crate::tt::{TT, FLAG_EXACT, FLAG_LOWER, FLAG_UPPER};

pub const MAX_PLY: usize = 128;

/// Per-search composition counters (compilation profiling), enabled with the
/// `profiling` feature. Not part of the engine contract.
#[cfg(feature = "profiling")]
pub static PROF: Prof = Prof {
    negamax: AtomicU64::new(0),
    qnode: AtomicU64::new(0),
    movegen: AtomicU64::new(0),
    movegen_q: AtomicU64::new(0),
    order: AtomicU64::new(0),
    order_cap: AtomicU64::new(0),
    make: AtomicU64::new(0),
    make_q: AtomicU64::new(0),
    see: AtomicU64::new(0),
    rep: AtomicU64::new(0),
};

#[cfg(feature = "profiling")]
pub struct Prof {
    pub negamax: AtomicU64,
    pub qnode: AtomicU64,
    pub movegen: AtomicU64,
    pub movegen_q: AtomicU64,
    pub order: AtomicU64,
    pub order_cap: AtomicU64,
    pub make: AtomicU64,
    pub make_q: AtomicU64,
    pub see: AtomicU64,
    pub rep: AtomicU64,
}

pub type EvalFn = fn(&Position) -> i32;

#[derive(Clone, Copy, Default)]
pub struct Limits {
    pub movetime: Option<u64>,
    pub wtime: Option<u64>,
    pub btime: Option<u64>,
    pub winc: Option<u64>,
    pub binc: Option<u64>,
    pub depth: Option<i32>,
    /// Number of principal variations to report (0 or 1 = single PV).
    pub multi_pv: u8,
}

/// One alternative line: root move, its score and full PV. `visits` is only
/// meaningful for MCTS results.
#[derive(Clone)]
pub struct MultiLine {
    pub mv: Move,
    pub score: i32,
    pub pv: Vec<Move>,
    pub visits: u32,
}

pub struct SearchResult {
    pub best: Move,
    pub score: i32,
    pub depth: i32,
    pub nodes: u64,
    pub pv: Vec<Move>,
    pub time_ms: u64,
    /// Filled only when `Limits::multi_pv > 1`.
    pub lines: Vec<MultiLine>,
}

/// Snapshot of one completed iterative-deepening iteration, for live UI.
pub struct SearchIter {
    pub depth: i32,
    pub score: i32,
    pub nodes: u64,
    pub time_ms: u64,
    pub pv: Vec<Move>,
    /// Top lines when `Limits::multi_pv > 1`.
    pub lines: Vec<MultiLine>,
}

struct RootResult {
    best: Move,
    score: i32,
}

pub struct Searcher {
    pub tt: Arc<TT>,
    /// When false, the transposition table is ignored (probe/store bypassed).
    /// Used for A/B correctness checks and fair speed comparisons.
    pub use_tt: bool,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    /// Capture history: indexed by [side][from][to].
    cap_history: [[[i32; 64]; 64]; 2],
    pv_table: Box<[[Move; MAX_PLY]; MAX_PLY]>,
    pv_len: [usize; MAX_PLY],
    nodes: u64,
    start: Instant,
    budget: Duration,
    /// Keys of the positions in the current line (pushed after each move).
    key_hist: [u64; MAX_PLY],
    key_hist_len: usize,
    /// Scratch sort scores for `order_moves`/`order_captures`. Reused so the
    /// hot path never zeroes a fresh 1 KB array per node.
    scratch: [i32; 256],
    /// Depth offset for Lazy SMP diversity (added to each iteration's depth).
    depth_offset: i32,
}

const PIECE_ORDER: [i32; 6] = [1, 2, 3, 4, 5, 6]; // P,N,B,R,Q,K for MVV-LVA
/// Piece values in centipawns, indexed by piece type (P=0..K=5). Used by SEE
/// and delta pruning. The king is never a recapture target in SEE.
const SEE_VAL: [i32; 6] = [100, 320, 330, 500, 900, 0];

/// Precomputed LMR reduction table: `LMR[depth][move_index]`. Log-based like
/// Stockfish: `0.75 + ln(d) * ln(i) / 2.25`, clamped to [0, 4].
static LMR: LazyLock<[[i32; 64]; 64]> = LazyLock::new(|| {
    let mut table = [[0i32; 64]; 64];
    for d in 1..64 {
        for i in 1..64 {
            let r = 0.75 + (d as f64).ln() * (i as f64).ln() / 2.25;
            table[d][i] = (r as i32).clamp(0, 4);
        }
    }
    table
});

/// Least valuable piece of `side` that attacks square `to` under occupancy
/// `occ` (sliding rays re-open as pieces are removed). Returns 0 if none.
fn see_attacker(pos: &Position, to: usize, side: usize, occ: u64) -> usize {
    for pt in [PAWN, KNIGHT, BISHOP, ROOK, QUEEN] {
        let mut bb = match pt {
            PAWN => pawn_attacks(side ^ 1, to) & pos.piece_bb(side, PAWN) & occ,
            KNIGHT => knight_attacks(to) & pos.piece_bb(side, KNIGHT) & occ,
            _ => {
                let mut b = pos.piece_bb(side, pt) & occ;
                let mut hit = 0;
                while b != 0 {
                    let sq = pop_lsb(&mut b);
                    let attacks = match pt {
                        BISHOP => bishop_attacks(sq, occ),
                        ROOK => rook_attacks(sq, occ),
                        _ => queen_attacks(sq, occ),
                    };
                    if attacks & bit(to) != 0 {
                        hit = bit(sq);
                        break;
                    }
                }
                hit
            }
        };
        if bb != 0 {
            return pop_lsb(&mut bb);
        }
    }
    0
}

/// Value of the exchange on `to` when `side` is to recapture the piece
/// `victim` standing on `to` under occupancy `occ`. Both sides recapture with
/// their cheapest attacker and may decline if the recapture loses material.
fn see_rec(pos: &Position, to: usize, victim: usize, side: usize, occ: u64) -> i32 {
    let att = see_attacker(pos, to, side, occ);
    if att == 0 {
        return 0;
    }
    let value = SEE_VAL[victim];
    let gain = value - see_rec(pos, to, pos.piece_pt_at(att), side ^ 1, occ & !bit(att));
    gain.max(0)
}

/// Static exchange evaluation of a capture from `from` to `to`: the net
/// material balance for the side to move (negative = losing capture).
pub fn see(pos: &Position, from: usize, to: usize) -> i32 {
    let value = SEE_VAL[pos.piece_pt_at(to)];
    value - see_rec(pos, to, pos.piece_pt_at(from), pos.side ^ 1, pos.occ & !bit(from))
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Self {
        Searcher {
            tt: Arc::new(TT::new(hash_mb)),
            use_tt: true,
            killers: [[Move::null(); 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            cap_history: [[[0; 64]; 64]; 2],
            pv_table: Box::new([[Move::null(); MAX_PLY]; MAX_PLY]),
            pv_len: [0; MAX_PLY],
            nodes: 0,
            start: Instant::now(),
            budget: Duration::from_secs(3600),
            key_hist: [0; MAX_PLY],
            key_hist_len: 0,
            scratch: [0; 256],
            depth_offset: 0,
        }
    }

    /// Create a searcher sharing the same TT (for Lazy SMP).
    pub fn with_tt(tt: Arc<TT>, depth_offset: i32) -> Self {
        Searcher {
            tt,
            use_tt: true,
            killers: [[Move::null(); 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            cap_history: [[[0; 64]; 64]; 2],
            pv_table: Box::new([[Move::null(); MAX_PLY]; MAX_PLY]),
            pv_len: [0; MAX_PLY],
            nodes: 0,
            start: Instant::now(),
            budget: Duration::from_secs(3600),
            key_hist: [0; MAX_PLY],
            key_hist_len: 0,
            scratch: [0; 256],
            depth_offset,
        }
    }

    pub fn think(&mut self, pos: &mut Position, limits: &Limits, stop: &AtomicBool, eval: EvalFn) -> SearchResult {
        self.think_cb(pos, limits, stop, eval, None)
    }

    /// Like `think`, but calls `on_iter` after each completed depth iteration
    /// with live (depth, score, nodes, pv). Used by GUIs to stream progress.
    pub fn think_cb(
        &mut self,
        pos: &mut Position,
        limits: &Limits,
        stop: &AtomicBool,
        eval: EvalFn,
        mut on_iter: Option<&mut dyn FnMut(&SearchIter)>,
    ) -> SearchResult {
        self.budget = compute_budget(pos, limits);
        self.start = Instant::now();
        self.nodes = 0;
        self.key_hist_len = 0;

        for k in &mut self.killers {
            *k = [Move::null(); 2];
        }
        for h in self.history.iter_mut() {
            for r in h.iter_mut() {
                for v in r.iter_mut() {
                    *v = 0;
                }
            }
        }
        for ch in self.cap_history.iter_mut() {
            for r in ch.iter_mut() {
                for v in r.iter_mut() {
                    *v = 0;
                }
            }
        }

        let max_depth = limits.depth.unwrap_or(64).clamp(1, 64) as usize;
        let multi = limits.multi_pv.max(1) as usize;
        let mut best = Move::null();
        let mut score = 0;
        let mut done = 0;
        let mut pv = Vec::new();
        let mut lines = Vec::new();
        // Per-ply move buffers, reused by `generate_pseudo_into`. Threaded as a
        // separate parameter (disjoint from `self`) so the parent's move list
        // can stay alive while children are searched. One extra slot guards the
        // deepest allowed ply.
        let mut moves_pool = Box::new([MoveList::new(); MAX_PLY + 1]);

        // Lazy SMP: start from a different depth for each thread to explore
        // different parts of the tree. The offset wraps around so threads
        // with offset 0, 1, 2 search depths starting at 1, 2, 3.
        let start_depth = (1 + self.depth_offset).max(1) as usize;

        let mut prev = 0i32;
        for d in start_depth..=max_depth {
            // Aspiration windows from depth 6 on: search around the previous
            // iteration's score. On a fail, re-center the window on the fail
            // score and widen (never fall back to a full window), so the
            // re-search stays narrow and the node cost of a re-search is small.
            let aspire = d >= 6 && (MATE - prev).abs() > MAX_PLY as i32;
            let mut delta = 25i32;
            let (mut alpha, mut beta) = if aspire {
                ((prev - delta).max(-INF), (prev + delta).min(INF))
            } else {
                (-INF, INF)
            };
            let mut r;
            loop {
                r = self.root_search(pos, d as i32, alpha, beta, stop, eval, &mut moves_pool[..]);
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                if r.score >= beta {
                    beta = (r.score + delta).min(INF);
                    alpha = (alpha + beta) / 2;
                } else if r.score <= alpha {
                    alpha = (r.score - delta).max(-INF);
                    beta = (alpha + beta) / 2;
                } else {
                    break;
                }
                delta += delta / 2;
            }
            best = r.best;
            score = r.score;
            done = d as i32;
            prev = score;
            pv = self.pv_table[0][..self.pv_len[0]].to_vec();
            if multi > 1 {
                lines = self.compute_multi_pv(pos, d as i32, stop, eval, multi, &mut moves_pool[..]);
            }
            if let Some(cb) = on_iter.as_mut() {
                cb(&SearchIter {
                    depth: done,
                    score,
                    nodes: self.nodes,
                    time_ms: self.start.elapsed().as_millis() as u64,
                    pv: pv.clone(),
                    lines: lines.clone(),
                });
            }
            if stop.load(Ordering::Relaxed) {
                break;
            }
            if (MATE - score).abs() <= 2 {
                break;
            }
        }

        SearchResult {
            best,
            score,
            depth: done,
            nodes: self.nodes,
            pv,
            time_ms: self.start.elapsed().as_millis() as u64,
            lines,
        }
    }

    fn root_search(&mut self, pos: &mut Position, depth: i32, alpha: i32, beta: i32, stop: &AtomicBool, eval: EvalFn, pool: &mut [MoveList]) -> RootResult {
        let mut moves = generate_legal(pos);
        let tt_move = if self.use_tt { self.tt.probe(pos.key).map(|(_, mv, _, _)| mv) } else { None };
        self.order_moves(pos, &mut moves, tt_move, 0);

        if moves.len == 0 {
            return RootResult {
                best: Move::null(),
                score: if pos.in_check() { -MATE } else { 0 },
            };
        }

        let mut alpha = alpha;
        let mut best_move = moves.get(0);
        self.pv_len[0] = 0;

        for i in 0..moves.len {
            let m = moves.get(i);
            let undo = pos.make_move(m);
            self.nodes += 1;
            self.key_hist[self.key_hist_len] = pos.key;
            self.key_hist_len += 1;
            let s = if i == 0 {
                -self.negamax(pos, depth - 1, -beta, -alpha, 1, stop, eval, pool)
            } else {
                let s = -self.negamax(pos, depth - 1, -alpha - 1, -alpha, 1, stop, eval, pool);
                if s > alpha && s < beta {
                    -self.negamax(pos, depth - 1, -beta, -alpha, 1, stop, eval, pool)
                } else {
                    s
                }
            };
            self.key_hist_len -= 1;
            pos.unmake_move(undo);
            if stop.load(Ordering::Relaxed) {
                break;
            }
            if s > alpha {
                alpha = s;
                best_move = m;
                self.pv_table[0][0] = m;
                let len = self.pv_len[1];
                let (left, right) = self.pv_table.split_at_mut(1);
                left[0][1..1 + len].copy_from_slice(&right[0][..len]);
                self.pv_len[0] = len + 1;
            }
        }

        // Fallback si la recherche est interrompue avant le premier coup bouclÃ©.
        if alpha == -INF {
            alpha = eval(pos);
        }

        RootResult { best: best_move, score: alpha }
    }

    /// Re-searches every legal root move at `depth` with a full window to
    /// collect the top `multi` lines (score + PV). Used for analysis UIs.
    fn compute_multi_pv(
        &mut self,
        pos: &mut Position,
        depth: i32,
        stop: &AtomicBool,
        eval: EvalFn,
        multi: usize,
        pool: &mut [MoveList],
    ) -> Vec<MultiLine> {
        let mut moves = generate_legal(pos);
        if moves.len == 0 {
            return Vec::new();
        }
        self.order_moves(pos, &mut moves, None, 0);

        let mut lines = Vec::new();
        for i in 0..moves.len {
            let m = moves.get(i);
            self.pv_len[1] = 0;
            let undo = pos.make_move(m);
            self.key_hist[self.key_hist_len] = pos.key;
            self.key_hist_len += 1;
            let s = -self.negamax(pos, depth - 1, -INF, INF, 1, stop, eval, pool);
            self.key_hist_len -= 1;
            pos.unmake_move(undo);
            let mut pv = vec![m];
            pv.extend_from_slice(&self.pv_table[1][..self.pv_len[1]]);
            lines.push(MultiLine { mv: m, score: s, pv, visits: 0 });
            if stop.load(Ordering::Relaxed) {
                break;
            }
        }
        lines.sort_by_key(|l| Reverse(l.score));
        lines.truncate(multi);
        lines
    }

    /// Convert a node score to a root-relative TT score: mate scores depend on
    /// the ply at which they were produced, so they are stored relative to the
    /// root and restored relative to the retrieving node's ply.
    fn score_to_tt(score: i32, ply: usize) -> i32 {
        if score >= MATE - MAX_PLY as i32 {
            score + ply as i32
        } else if score <= -MATE + MAX_PLY as i32 {
            score - ply as i32
        } else {
            score
        }
    }

    fn score_from_tt(score: i32, ply: usize) -> i32 {
        if score >= MATE - MAX_PLY as i32 {
            score - ply as i32
        } else if score <= -MATE + MAX_PLY as i32 {
            score + ply as i32
        } else {
            score
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        pos: &mut Position,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: usize,
        stop: &AtomicBool,
        eval: EvalFn,
        pool: &mut [MoveList],
    ) -> i32 {
        self.nodes += 1;
        #[cfg(feature = "profiling")]
        PROF.negamax.fetch_add(1, Ordering::Relaxed);
        // Reset the PV slot first: any early return (time-up, TT cutoff,
        // null-move, RFP, quiescence, mate) leaves `pv_len[ply] == 0` so the
        // parent never copies a stale PV from an unrelated position.
        self.pv_len[ply] = 0;
        if (self.nodes & 1023) == 0 && self.time_up(stop) {
            return 0;
        }

        if ply > 0 {
            #[cfg(feature = "profiling")]
            PROF.rep.fetch_add(1, Ordering::Relaxed);
            if pos.halfmove >= 100 || self.is_repetition(pos.key, pos.halfmove) {
                return 0;
            }
        }

        let in_check = pos.in_check();
        if depth <= 0 && !in_check {
            return self.quiescence(pos, alpha, beta, ply, stop, eval, pool);
        }

        let key = pos.key;
        let tt_move = match if self.use_tt { self.tt.probe(key) } else { None } {
            Some((tt_score, tt_mv, tt_flag, tt_depth)) => {
                if ply > 0 && tt_depth >= depth {
                    match tt_flag {
                        FLAG_EXACT => return Self::score_from_tt(tt_score, ply),
                        FLAG_LOWER => {
                            if tt_score >= beta {
                                return Self::score_from_tt(tt_score, ply);
                            }
                        }
                        FLAG_UPPER => {
                            if tt_score <= alpha {
                                return Self::score_from_tt(tt_score, ply);
                            }
                        }
                        _ => {}
                    }
                }
                Some(tt_mv)
            }
            None => None,
        };

        // Internal iterative reduction: without a TT move and sufficient depth,
        // reduce by 1 to avoid a full-window search on a mostly uninformative
        // node (Stockfish IIR, depth >= 4).
        let mut iir_depth = depth;
        if self.use_tt && tt_move.is_none() && depth >= 4 && !in_check {
            iir_depth -= 1;
        }

        let occ_count = pos.occ.count_ones();
        let mut eval_cache = 0i32;
        let mut have_eval = false;
        if depth >= 3 && !in_check && occ_count > 5 {
            eval_cache = eval(pos);
            have_eval = true;
            if eval_cache >= beta {
                pos.make_null();
                let score = -self.negamax(pos, depth - 1 - 2, -beta, -beta + 1, ply + 1, stop, eval, pool);
                pos.unmake_null();
                if score >= beta {
                    return score;
                }
            }
        }

        // Reverse futility pruning: position is clearly above beta, skip movegen.
        if depth <= 7 && !in_check && ply > 0 {
            if !have_eval {
                eval_cache = eval(pos);
            }
            if eval_cache - 90 * depth >= beta {
                return eval_cache;
            }
        }

        // Razoring: at low depth, if eval is far below alpha, the position is
        // likely losing. Drop into quiescence to confirm (Stockfish razoring).
        if depth <= 1 && !in_check && ply > 0 {
            if !have_eval {
                eval_cache = eval(pos);
            }
            if eval_cache + 300 < alpha {
                let razor = self.quiescence(pos, alpha, beta, ply, stop, eval, pool);
                return razor.max(eval_cache);
            }
        }

        let us = pos.side;
        let pinned = pos.pinned();
        let king = pos.king_sq[us];

        // Singular extensions: if the TT move is the only move that can exceed
        // alpha (all others fail low), extend the search by 1 ply.
        let mut se_ext = 0i32;
        if let Some(tt_m) = tt_move {
            if depth >= 5 && self.use_tt {
                let margin = 2 * depth as i32;
                let se_depth = (depth / 2).max(1);
                if !have_eval {
                    eval_cache = eval(pos);
                }
                let mut se_best = eval_cache;
                let verify = in_check
                    || tt_m.is_en_passant()
                    || tt_m.from() == king
                    || (pinned & bit(tt_m.from()) != 0);
                let undo = pos.make_move(tt_m);
                if verify && !pos.king_safe(us) {
                    pos.unmake_move(undo);
                } else {
                    self.key_hist[self.key_hist_len] = pos.key;
                    self.key_hist_len += 1;
                    se_best = -self.negamax(pos, se_depth, -(alpha + margin), -(alpha + margin - 1), ply + 1, stop, eval, pool);
                    self.key_hist_len -= 1;
                    pos.unmake_move(undo);
                }
                if se_best <= alpha + margin - 1 {
                    se_ext = 1;
                }
            }
        }

        let mut best = -INF;
        let mut best_move = Move::null();
        let mut alpha0 = alpha;
        let mut pruned = false;

        // TT move first: searched before any movegen/ordering. A cutoff here
        // skips generation entirely (the common cut-node case). The move is
        // then skipped in the main loop; its score is already exact (it was
        // searched with the full window).
let mut searched: Option<Move> = None;
        if let Some(tt_m) = tt_move {
            // A TT move can be stale (entry from another position sharing the
            // slot, or a null move stored by a buggy mate node): only search it
            // when its `from` square actually holds a piece of the side to move.
            if pos.pieces_of(us) & bit(tt_m.from()) != 0 {
                let verify = in_check
                || tt_m.is_en_passant()
                || tt_m.from() == king
                || (pinned & bit(tt_m.from()) != 0);
            let undo = pos.make_move(tt_m);
            #[cfg(feature = "profiling")]
            PROF.make.fetch_add(1, Ordering::Relaxed);
            if verify && !pos.king_safe(us) {
                pos.unmake_move(undo);
            } else {
                self.key_hist[self.key_hist_len] = pos.key;
            self.key_hist_len += 1;
                let s = -self.negamax(pos, depth - 1 + se_ext, -beta, -alpha0, ply + 1, stop, eval, pool);
                self.key_hist_len -= 1;
                pos.unmake_move(undo);
                if stop.load(Ordering::Relaxed) {
                    return 0;
                }
                if s > best {
                    best = s;
                    best_move = tt_m;
                    if s > alpha0 {
                        alpha0 = s;
                        self.pv_table[ply][0] = tt_m;
                        let len = self.pv_len[ply + 1];
                        let (left, right) = self.pv_table.split_at_mut(ply + 1);
                        left[ply][1..1 + len].copy_from_slice(&right[0][..len]);
                        self.pv_len[ply] = len + 1;
                    }
                }
                if alpha0 >= beta {
                    if best_move.is_quiet() {
                        self.update_history(pos, best_move, depth);
                        if self.killers[ply][0] != best_move {
                            self.killers[ply][1] = self.killers[ply][0];
                            self.killers[ply][0] = best_move;
                        }
                    }
                    if self.use_tt {
                        self.tt.store(key, Self::score_to_tt(best, ply), depth, FLAG_LOWER, best_move);
                    }
                    return best;
                }
                searched = Some(tt_m);
            }
        }
        }

        let (moves, rest) = pool.split_first_mut().expect("move pool exhausted");
        generate_pseudo_into(pos, false, in_check, moves);
        #[cfg(feature = "profiling")]
        PROF.movegen.fetch_add(1, Ordering::Relaxed);
        if moves.len == 0 {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }

        order_moves_impl(pos, &self.killers, &self.history, &self.cap_history, &mut self.scratch, moves, tt_move, ply);
        #[cfg(feature = "profiling")]
        PROF.order.fetch_add(1, Ordering::Relaxed);

        let num_moves = moves.len;
        for i in 0..num_moves {
            let m = moves.get(i);
            if searched == Some(m) {
                continue;
            }
            // Reduction index: the skipped TT move occupied list slot 0, so
            // restore the original numbering for LMR thresholds.
            let idx = i + if searched.is_some() { 1 } else { 0 };
            // Late Move Reductions: log-based table, same as Stockfish.
            let reduction = if idx > 0 && m.is_quiet() && !in_check && iir_depth >= 3 {
                LMR[iir_depth.min(63) as usize][idx.min(63) as usize]
            } else {
                0
            };
            // A move is unconditionally legal only for a non-pinned piece that
            // is not the king (and not en passant); anything else is verified
            // by making it and checking the mover's king afterwards.
            let verify = in_check
                || m.is_en_passant()
                || m.from() == king
                || (pinned & bit(m.from()) != 0);
            // Shallow node pruning for quiet moves: at low depth a quiet move
            // whose static evaluation is far below alpha cannot improve it
            // (futility), and late quiet moves are unlikely to matter (LMP).
            // Skipped near mate, when in check, or for the TT/killer moves
            // that are already searched before the loop. Runs before the
            // make_move below: pruned moves never touch the board.
            if !in_check && m.is_quiet() && iir_depth <= 3 && alpha0 > -MATE + MAX_PLY as i32 {
                if !have_eval {
                    eval_cache = eval(pos);
                    have_eval = true;
                }
                if eval_cache + 110 * iir_depth <= alpha0 || idx >= (5 + iir_depth * iir_depth) as usize {
                    pruned = true;
                    continue;
                }
            }
            let undo = pos.make_move(m);
            #[cfg(feature = "profiling")]
            PROF.make.fetch_add(1, Ordering::Relaxed);
            if verify && !pos.king_safe(us) {
                pos.unmake_move(undo);
                continue;
            }
            self.key_hist[self.key_hist_len] = pos.key;
            self.key_hist_len += 1;
            let s = if i == 0 && searched.is_none() {
                -self.negamax(pos, iir_depth - 1, -beta, -alpha0, ply + 1, stop, eval, rest)
            } else {
                let s = -self.negamax(
                    pos,
                    iir_depth - 1 - reduction,
                    -alpha0 - 1,
                    -alpha0,
                    ply + 1,
                    stop,
                    eval,
                    rest,
                );
                if s > alpha0 && s < beta {
                    -self.negamax(pos, iir_depth - 1, -beta, -alpha0, ply + 1, stop, eval, rest)
                } else {
                    s
                }
            };
            self.key_hist_len -= 1;
            pos.unmake_move(undo);
            if stop.load(Ordering::Relaxed) {
                return 0;
            }
            if s > best {
                best = s;
                best_move = m;
                if s > alpha0 {
                    alpha0 = s;
                    self.pv_table[ply][0] = m;
                    let len = self.pv_len[ply + 1];
                    let (left, right) = self.pv_table.split_at_mut(ply + 1);
                    left[ply][1..1 + len].copy_from_slice(&right[0][..len]);
                    self.pv_len[ply] = len + 1;
                }
            }
            if alpha0 >= beta {
                if best_move.is_quiet() {
                    self.update_history(pos, best_move, iir_depth);
                    if self.killers[ply][0] != best_move {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = best_move;
                    }
                    self.update_cap_history(pos, best_move, iir_depth);
                } else if best_move.is_capture() {
                    let ch = &mut self.cap_history[us][best_move.from()][best_move.to()];
                    *ch += iir_depth * iir_depth;
                    if *ch > 16_000 {
                        for r in self.cap_history[us].iter_mut() {
                            for v in r.iter_mut() {
                                *v /= 2;
                            }
                        }
                    }
                }
                break;
            }
        }

        let flag = if best <= alpha {
            FLAG_UPPER
        } else if best >= beta {
            FLAG_LOWER
        } else {
            FLAG_EXACT
        };
        if best == -INF {
            // Some moves were skipped by futility/LMP: the position has legal
            // moves, so the node is a fail-low bound, not a terminal node.
            if pruned {
                return alpha0;
            }
            // Every pseudo-move was illegal: the position is mate or
            // stalemate even though generate_pseudo was non-empty. Without
            // this the node would return -INF and store a null TT move.
            return if in_check { -MATE + ply as i32 } else { 0 };
        }
        if self.use_tt {
            self.tt.store(key, Self::score_to_tt(best, ply), depth, flag, best_move);
        }
        best
    }

    fn quiescence(
        &mut self,
        pos: &mut Position,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        stop: &AtomicBool,
        eval: EvalFn,
        pool: &mut [MoveList],
    ) -> i32 {
        self.nodes += 1;
        #[cfg(feature = "profiling")]
        PROF.qnode.fetch_add(1, Ordering::Relaxed);
        // Same PV-slot reset as negamax: a quiescence leaf is never part of a
        // PV, so a stale length from an earlier position must not leak up.
        self.pv_len[ply] = 0;
        if (self.nodes & 1023) == 0 && self.time_up(stop) {
            return 0;
        }

        let in_check = pos.in_check();
        let stand = eval(pos);
        if !in_check {
            if stand >= beta {
                return stand;
            }
            if stand > alpha {
                alpha = stand;
            }
        }

        let (moves, rest) = pool.split_first_mut().expect("move pool exhausted");
        generate_pseudo_into(pos, !in_check, in_check, moves);
        #[cfg(feature = "profiling")]
        PROF.movegen_q.fetch_add(1, Ordering::Relaxed);
        if moves.len == 0 {
            return if in_check { -MATE + ply as i32 } else { alpha };
        }

        order_captures_impl(pos, &mut self.scratch, moves);
        #[cfg(feature = "profiling")]
        PROF.order_cap.fetch_add(1, Ordering::Relaxed);

        let mut best = if in_check { -INF } else { alpha };
        let mut alpha0 = if in_check { -INF } else { alpha };
        let us = pos.side;
        let pinned = pos.pinned();
        let king = pos.king_sq[us];

        let num_moves = moves.len;
        for i in 0..num_moves {
            let m = moves.get(i);
            let verify = in_check
                || m.is_en_passant()
                || m.from() == king
                || (pinned & bit(m.from()) != 0);
            // Skip losing captures without touching the board. Delta pruning
            // rejects captures that cannot reach alpha even if free; SEE
            // rejects captures that lose material. Evasions and promotions
            // (whose gain can exceed the victim) are always searched.
            if !in_check && m.is_capture() && !m.is_en_passant() && !m.is_promotion() {
                let victim = pos.piece_pt_at(m.to());
                if stand + SEE_VAL[victim] + 200 < alpha0 {
                    continue;
                }
                #[cfg(feature = "profiling")]
                PROF.see.fetch_add(1, Ordering::Relaxed);
                if see(pos, m.from(), m.to()) < 0 {
                    continue;
                }
            }
            let undo = pos.make_move(m);
            #[cfg(feature = "profiling")]
            PROF.make_q.fetch_add(1, Ordering::Relaxed);
            if verify && !pos.king_safe(us) {
                pos.unmake_move(undo);
                continue;
            }
            let s = -self.quiescence(pos, -beta, -alpha0, ply + 1, stop, eval, rest);
            pos.unmake_move(undo);
            if stop.load(Ordering::Relaxed) {
                return 0;
            }
            if s > best {
                best = s;
            }
            if s > alpha0 {
                alpha0 = s;
                if alpha0 >= beta {
                    break;
                }
            }
        }
        if best == -INF {
            return if in_check { -MATE + ply as i32 } else { alpha };
        }
        best
    }

    pub fn order_moves(&mut self, pos: &Position, moves: &mut MoveList, tt_move: Option<Move>, ply: usize) {
        order_moves_impl(pos, &self.killers, &self.history, &self.cap_history, &mut self.scratch, moves, tt_move, ply);
    }

    fn update_history(&mut self, pos: &Position, m: Move, depth: i32) {
        let h = &mut self.history[pos.side][m.from()][m.to()];
        *h += depth * depth;
        if *h > 16_000 {
            for r in self.history[pos.side].iter_mut() {
                for v in r.iter_mut() {
                    *v /= 2;
                }
            }
        }
    }

    fn update_cap_history(&mut self, pos: &Position, m: Move, depth: i32) {
        let h = &mut self.cap_history[pos.side][m.from()][m.to()];
        *h += depth * depth;
        if *h > 16_000 {
            for r in self.cap_history[pos.side].iter_mut() {
                for v in r.iter_mut() {
                    *v /= 2;
                }
            }
        }
    }

    /// Repetition: the current key is in `key_hist` (pushed by the parent); a
    /// draw when it appeared at least once before. Only the plies since the
    /// last irreversible move (`halfmove`) are scanned: the same position
    /// cannot recur across an irreversible move, so earlier entries cannot
    /// match.
    fn is_repetition(&self, key: u64, halfmove: u32) -> bool {
        // A position can only recur after at least 4 plies (two full moves).
        // Skipping the scan for small halfmove windows avoids most iterations.
        if halfmove < 4 {
            return false;
        }
        let len = self.key_hist_len;
        let start = len.saturating_sub(halfmove as usize + 1);
        let mut count = 0;
        for &h in &self.key_hist[start..len] {
            if h == key {
                count += 1;
                if count >= 2 {
                    return true;
                }
            }
        }
        false
    }

    fn time_up(&self, stop: &AtomicBool) -> bool {
        if stop.load(Ordering::Relaxed) {
            return true;
        }
        if self.start.elapsed() >= self.budget {
            stop.store(true, Ordering::Relaxed);
            return true;
        }
        false
    }
}

fn compute_budget(pos: &Position, limits: &Limits) -> Duration {
    if let Some(ms) = limits.movetime {
        return Duration::from_millis(ms);
    }
    let (t, inc) = if pos.side == WHITE {
        (limits.wtime, limits.winc)
    } else {
        (limits.btime, limits.binc)
    };
    if let Some(t) = t {
        let inc = inc.unwrap_or(0);
        let budget = ((t as i64 + inc as i64) / 20).max(1) as u64;
        return Duration::from_millis(budget);
    }
    Duration::from_secs(3600)
}

/// Scores and insertion-sorts `moves` by descending priority: TT move first,
/// then MVV-LVA captures/promotions, killers, history, capture history.
fn order_moves_impl(
    pos: &Position,
    killers: &[[Move; 2]; MAX_PLY],
    history: &[[[i32; 64]; 64]; 2],
    cap_history: &[[[i32; 64]; 64]; 2],
    scratch: &mut [i32],
    moves: &mut MoveList,
    tt_move: Option<Move>,
    ply: usize,
) {
    for i in 0..moves.len {
        let m = moves.get(i);
        let mut s = 0;
        if tt_move == Some(m) {
            s = 1_000_000;
        } else {
            if m.is_promotion() {
                s += 800_000 + PIECE_ORDER[m.promo_pt()];
            }
            if m.is_capture() {
                let victim = if m.is_en_passant() {
                    PAWN
                } else {
                    pos.piece_pt_at(m.to())
                };
                let attacker = if m.is_promotion() {
                    PAWN
                } else {
                    pos.piece_pt_at(m.from())
                };
                s += 500_000 + PIECE_ORDER[victim] * 16 - PIECE_ORDER[attacker];
                s += cap_history[pos.side][m.from()][m.to()];
            }
            if killers[ply][0] == m {
                s += 300_000;
            } else if killers[ply][1] == m {
                s += 290_000;
            }
            s += history[pos.side][m.from()][m.to()];
        }
        scratch[i] = s;
    }
    for i in 1..moves.len {
        let key = scratch[i];
        let mv = moves.get(i);
        let mut j = i;
        while j > 0 && scratch[j - 1] < key {
            moves.moves[j] = moves.moves[j - 1];
            scratch[j] = scratch[j - 1];
            j -= 1;
        }
        moves.moves[j] = mv;
        scratch[j] = key;
    }
}

/// Scores and insertion-sorts the quiescence move list (MVV-LVA, promotions
/// first) using `scratch` as the per-call score buffer.
fn order_captures_impl(pos: &Position, scratch: &mut [i32], moves: &mut MoveList) {
    for i in 0..moves.len {
        let m = moves.get(i);
        scratch[i] = if m.is_promotion() {
            700_000 + PIECE_ORDER[m.promo_pt()]
        } else if m.is_capture() {
            let victim = if m.is_en_passant() {
                PAWN
            } else {
                pos.piece_pt_at(m.to())
            };
            let attacker = pos.piece_pt_at(m.from());
            PIECE_ORDER[victim] * 16 - PIECE_ORDER[attacker]
        } else {
            0
        };
    }
    for i in 1..moves.len {
        let key = scratch[i];
        let mv = moves.get(i);
        let mut j = i;
        while j > 0 && scratch[j - 1] < key {
            moves.moves[j] = moves.moves[j - 1];
            scratch[j] = scratch[j - 1];
            j -= 1;
        }
        moves.moves[j] = mv;
        scratch[j] = key;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::move_::parse_sq;

    fn see_fen(fen: &str, from: &str, to: &str) -> i32 {
        crate::position::init();
        crate::attack::init();
        crate::magic::init();
        let pos = Position::from_fen(fen);
        see(&pos, parse_sq(from), parse_sq(to))
    }

    #[test]
    fn see_undefended_pawn() {
        assert_eq!(see_fen("4k3/8/8/8/8/8/4p3/2N1K3 w - - 0 1", "c1", "e2"), 100);
    }

    #[test]
    fn see_queen_defended_by_pawn() {
        // White pawn captures a queen defended by a black pawn: +900 - 100.
        assert_eq!(see_fen("4k3/8/7p/6q1/5P2/8/8/4K3 w - - 0 1", "f4", "g5"), 800);
    }

    #[test]
    fn see_queen_defended_by_knight() {
        // QxQ then NxQ: equal exchange.
        assert_eq!(see_fen("3qk3/8/4n3/8/8/8/8/2Q1K3 w - - 0 1", "c1", "d8"), 0);
    }

    #[test]
    fn see_rook_defended_by_rook() {
        // QxR then RxQ: white nets +500 - 900 = -400 (bad capture).
        assert_eq!(see_fen("3rr2k/8/8/8/8/8/8/3Q3K w - - 0 1", "d1", "e8"), -400);
    }

    #[test]
    fn see_queen_trade_with_recapture() {
        // QxR, RxQ, RxR: white nets +500 - 900 + 500 = +100.
        assert_eq!(see_fen("3r1r1k/8/8/8/8/8/3R4/R2QK3 w - - 0 1", "d1", "d8"), 100);
    }

    #[test]
    fn see_queen_captures_undefended_rook() {
        assert_eq!(see_fen("4r2k/8/8/8/8/8/8/3Q3K w - - 0 1", "d1", "e8"), 500);
    }
}





