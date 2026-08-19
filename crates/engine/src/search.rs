//! Alpha-beta search: iterative deepening, PVS, quiescence, TT, killers,
//! history heuristic, null-move pruning, time management.

use std::sync::atomic::{AtomicBool, Ordering};
use std::cmp::Reverse;
use std::time::{Duration, Instant};

use crate::bitboard::bit;
use crate::evaluate::{MATE, INF};
use crate::movegen::{generate_legal, generate_pseudo, MoveList};
use crate::move_::Move;
use crate::position::*;
use crate::tt::{TT, FLAG_EXACT, FLAG_LOWER, FLAG_UPPER};

pub const MAX_PLY: usize = 128;

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
    pub tt: TT,
    /// When false, the transposition table is ignored (probe/store bypassed).
    /// Used for A/B correctness checks and fair speed comparisons.
    pub use_tt: bool,
    killers: [[Move; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    pv_table: Box<[[Move; MAX_PLY]; MAX_PLY]>,
    pv_len: [usize; MAX_PLY],
    nodes: u64,
    start: Instant,
    budget: Duration,
    /// Keys of the positions in the current line (pushed after each move).
    key_hist: Vec<u64>,
}

const PIECE_ORDER: [i32; 6] = [1, 2, 3, 4, 5, 6]; // P,N,B,R,Q,K for MVV-LVA

impl Searcher {
    pub fn new(hash_mb: usize) -> Self {
        Searcher {
            tt: TT::new(hash_mb),
            use_tt: true,
            killers: [[Move::null(); 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            pv_table: Box::new([[Move::null(); MAX_PLY]; MAX_PLY]),
            pv_len: [0; MAX_PLY],
            nodes: 0,
            start: Instant::now(),
            budget: Duration::from_secs(3600),
            key_hist: Vec::new(),
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
        self.key_hist.clear();

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

        let max_depth = limits.depth.unwrap_or(64).clamp(1, 64) as usize;
        let multi = limits.multi_pv.max(1) as usize;
        let mut best = Move::null();
        let mut score = 0;
        let mut done = 0;
        let mut pv = Vec::new();
        let mut lines = Vec::new();

        for d in 1..=max_depth {
            let r = self.root_search(pos, d as i32, stop, eval);
            best = r.best;
            score = r.score;
            done = d as i32;
            pv = self.pv_table[0][..self.pv_len[0]].to_vec();
            if multi > 1 {
                lines = self.compute_multi_pv(pos, d as i32, stop, eval, multi);
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

    fn root_search(&mut self, pos: &mut Position, depth: i32, stop: &AtomicBool, eval: EvalFn) -> RootResult {
        let mut moves = generate_legal(pos);
        let tt_move = if self.use_tt { self.tt.probe(pos.key).map(|e| Move(e.mv)) } else { None };
        self.order_moves(pos, &mut moves, tt_move, 0);

        if moves.len == 0 {
            return RootResult {
                best: Move::null(),
                score: if pos.in_check() { -MATE } else { 0 },
            };
        }

        let mut alpha = -INF;
        let beta = INF;
        let mut best_move = moves.get(0);
        self.pv_len[0] = 0;

        for i in 0..moves.len {
            let m = moves.get(i);
            let undo = pos.make_move(m);
            self.nodes += 1;
            self.key_hist.push(pos.key);
            let s = if i == 0 {
                -self.negamax(pos, depth - 1, -beta, -alpha, 1, stop, eval)
            } else {
                let s = -self.negamax(pos, depth - 1, -alpha - 1, -alpha, 1, stop, eval);
                if s > alpha {
                    -self.negamax(pos, depth - 1, -beta, -alpha, 1, stop, eval)
                } else {
                    s
                }
            };
            self.key_hist.pop();
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
            self.key_hist.push(pos.key);
            let s = -self.negamax(pos, depth - 1, -INF, INF, 1, stop, eval);
            self.key_hist.pop();
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
    ) -> i32 {
        self.nodes += 1;
        // Reset the PV slot first: any early return (time-up, TT cutoff,
        // null-move, RFP, quiescence, mate) leaves `pv_len[ply] == 0` so the
        // parent never copies a stale PV from an unrelated position.
        self.pv_len[ply] = 0;
        if (self.nodes & 1023) == 0 && self.time_up(stop) {
            return 0;
        }

        if ply > 0 {
            if pos.halfmove >= 100 || self.is_repetition(pos.key, pos.halfmove) {
                return 0;
            }
        }

        let in_check = pos.in_check();
        if depth <= 0 && !in_check {
            return self.quiescence(pos, alpha, beta, ply, stop, eval);
        }

        let key = pos.key;
        let tt_move = match if self.use_tt { self.tt.probe(key) } else { None } {
            Some(e) => {
                if ply > 0 && (e.depth as i32) >= depth {
                    match e.flag {
                        FLAG_EXACT => return Self::score_from_tt(e.score, ply),
                        FLAG_LOWER => {
                            if e.score >= beta {
                                return Self::score_from_tt(e.score, ply);
                            }
                        }
                        FLAG_UPPER => {
                            if e.score <= alpha {
                                return Self::score_from_tt(e.score, ply);
                            }
                        }
                        _ => {}
                    }
                }
                Some(Move(e.mv))
            }
            None => None,
        };

        let occ_count = pos.occ.count_ones();
        let mut eval_cache = 0i32;
        let mut have_eval = false;
        if depth >= 3 && !in_check && occ_count > 5 {
            eval_cache = eval(pos);
            have_eval = true;
            if eval_cache >= beta {
                pos.make_null();
                let score = -self.negamax(pos, depth - 1 - 2, -beta, -beta + 1, ply + 1, stop, eval);
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

        let us = pos.side;
        let pinned = pos.pinned();
        let king = pos.king_sq[us];

        let mut best = -INF;
        let mut best_move = Move::null();
        let mut alpha0 = alpha;

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
            if verify && !pos.king_safe(us) {
                pos.unmake_move(undo);
            } else {
                self.key_hist.push(pos.key);
                let s = -self.negamax(pos, depth - 1, -beta, -alpha0, ply + 1, stop, eval);
                self.key_hist.pop();
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

        let mut moves = generate_pseudo(pos, false, in_check);
        if moves.len == 0 {
            return if in_check { -MATE + ply as i32 } else { 0 };
        }

        self.order_moves(pos, &mut moves, tt_move, ply);

        for i in 0..moves.len {
            let m = moves.get(i);
            if searched == Some(m) {
                continue;
            }
            // Reduction index: the skipped TT move occupied list slot 0, so
            // restore the original numbering for LMR thresholds.
            let idx = i + if searched.is_some() { 1 } else { 0 };
            // Late Move Reductions: cut depth for quiet moves searched late.
            let reduction = if idx > 0 && m.is_quiet() && !in_check && depth >= 3 {
                if idx >= 6 {
                    2
                } else if idx >= 3 {
                    1
                } else {
                    0
                }
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
            let undo = pos.make_move(m);
            if verify && !pos.king_safe(us) {
                pos.unmake_move(undo);
                continue;
            }
            self.key_hist.push(pos.key);
            let s = if i == 0 && searched.is_none() {
                -self.negamax(pos, depth - 1, -beta, -alpha0, ply + 1, stop, eval)
            } else {
                let s = -self.negamax(
                    pos,
                    depth - 1 - reduction,
                    -alpha0 - 1,
                    -alpha0,
                    ply + 1,
                    stop,
                    eval,
                );
                if s > alpha0 && s < beta {
                    -self.negamax(pos, depth - 1, -beta, -alpha0, ply + 1, stop, eval)
                } else {
                    s
                }
            };
            self.key_hist.pop();
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
                    self.update_history(pos, best_move, depth);
                    if self.killers[ply][0] != best_move {
                        self.killers[ply][1] = self.killers[ply][0];
                        self.killers[ply][0] = best_move;
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
    ) -> i32 {
        self.nodes += 1;
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

        let mut moves = generate_pseudo(pos, !in_check, in_check);
        if moves.len == 0 {
            return if in_check { -MATE + ply as i32 } else { alpha };
        }

        self.order_captures(pos, &mut moves);

        let mut best = if in_check { -INF } else { alpha };
        let mut alpha0 = if in_check { -INF } else { alpha };
        let us = pos.side;
        let pinned = pos.pinned();
        let king = pos.king_sq[us];

        for i in 0..moves.len {
            let m = moves.get(i);
            let verify = in_check
                || m.is_en_passant()
                || m.from() == king
                || (pinned & bit(m.from()) != 0);
            let undo = pos.make_move(m);
            if verify && !pos.king_safe(us) {
                pos.unmake_move(undo);
                continue;
            }
            let s = -self.quiescence(pos, -beta, -alpha0, ply + 1, stop, eval);
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
        let mut scores = [0i32; 256];
        for i in 0..moves.len {
            scores[i] = self.move_score(pos, moves.get(i), tt_move, ply);
        }
        for i in 1..moves.len {
            let key = scores[i];
            let mv = moves.get(i);
            let mut j = i;
            while j > 0 && scores[j - 1] < key {
                moves.moves[j] = moves.moves[j - 1];
                scores[j] = scores[j - 1];
                j -= 1;
            }
            moves.moves[j] = mv;
            scores[j] = key;
        }
    }

    fn order_captures(&self, pos: &Position, moves: &mut MoveList) {
        let mut scores = [0i32; 256];
        for i in 0..moves.len {
            let m = moves.get(i);
            scores[i] = if m.is_capture() {
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
            let key = scores[i];
            let mv = moves.get(i);
            let mut j = i;
            while j > 0 && scores[j - 1] < key {
                moves.moves[j] = moves.moves[j - 1];
                scores[j] = scores[j - 1];
                j -= 1;
            }
            moves.moves[j] = mv;
            scores[j] = key;
        }
    }

    fn move_score(&self, pos: &Position, m: Move, tt_move: Option<Move>, ply: usize) -> i32 {
        if tt_move == Some(m) {
            return 1_000_000;
        }
        let mut s = 0;
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
        }
        if self.killers[ply][0] == m {
            s += 300_000;
        } else if self.killers[ply][1] == m {
            s += 290_000;
        }
        s += self.history[pos.side][m.from()][m.to()];
        s
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

    /// Repetition: the current key is in `key_hist` (pushed by the parent); a
    /// draw when it appeared at least once before. Only the plies since the
    /// last irreversible move (`halfmove`) are scanned: the same position
    /// cannot recur across an irreversible move, so earlier entries cannot
    /// match.
    fn is_repetition(&self, key: u64, halfmove: u32) -> bool {
        let len = self.key_hist.len();
        let start = len.saturating_sub(halfmove as usize + 1);
        let mut count = 0;
        for &h in &self.key_hist[start..] {
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
