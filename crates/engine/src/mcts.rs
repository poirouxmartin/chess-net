//! Monte-Carlo Tree Search (UCT) with a value function for leaf evaluation.
//!
//! The value function maps a position to centipawns for the side to move
//! (PeSTO or the NN); the leaf value is `2*sigmoid(cp/400) - 1`, a zero-sum
//! score in [-1, 1]. A policy function can provide per-move priors from the NN
//! policy head (softmax-masked over legal moves); when it returns no logits the
//! prior is uniform over legal moves.
//!
//! Nodes live in a preallocated arena (u32 indices) of atomics so the tree can
//! be shared across threads: selection applies a virtual loss on the chosen
//! path, workers expand leaves under a per-node spinlock, and the backup undoes
//! the virtual loss and adds the real value. When the arena is full, expansion
//! stops and the leaf is evaluated instead (graceful degradation).

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::move_::Move;
use crate::movegen::{generate_legal, MoveList, MAX_MOVES};
use crate::position::Position;
use crate::search::EvalFn;

/// Alpha, epsilon for Dirichlet noise at root (AlphaZero defaults).
pub const DIRICHLET_ALPHA: f32 = 0.03;
pub const DIRICHLET_EPSILON: f32 = 0.25;

/// Sample Gamma(shape, 1.0) using Marsaglia-Tsang method.
/// Valid for shape >= 1 (we use shape = 1 for Dirichlet).
fn sample_gamma(shape: f32, rng: &mut u64) -> f32 {
    let d = shape - 1.0 / 3.0;
    loop {
        let mut x;
        loop {
            x = sample_normal(rng);
            if x > -1.0 / (3.0 * shape).sqrt() {
                break;
            }
        }
        let v = 1.0 + x / (3.0 * shape).sqrt();
        let v3 = v * v * v;
        let u = sample_uniform(rng);
        if u < 1.0 - 0.0331 * x * x * x * x {
            return d * v3;
        }
        if (u.ln()) < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

/// Sample Gamma(alpha, 1) for alpha < 1 using the alpha-exp trick:
/// Gamma(a) = Gamma(a+1) * U^(1/a).
fn sample_gamma_small(alpha: f32, rng: &mut u64) -> f32 {
    let big = sample_gamma(alpha + 1.0, rng);
    let u = sample_uniform(rng);
    big * u.powf(1.0 / alpha)
}

/// Box-Muller normal(0,1).
fn sample_normal(rng: &mut u64) -> f32 {
    let u1 = sample_uniform(rng).max(1e-10);
    let u2 = sample_uniform(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Simple xorshift64 PRNG.
#[inline]
fn sample_uniform(rng: &mut u64) -> f32 {
    *rng ^= *rng << 13;
    *rng ^= *rng >> 7;
    *rng ^= *rng << 17;
    (*rng as f32) / (u64::MAX as f32)
}

/// Sample Dirichlet(alpha) distribution, returning `n` probabilities summing to 1.
fn sample_dirichlet(n: usize, alpha: f32, rng: &mut u64) -> Vec<f32> {
    let mut samples: Vec<f32> = (0..n).map(|_| {
        if alpha < 1.0 {
            sample_gamma_small(alpha, rng)
        } else {
            sample_gamma(alpha, rng)
        }
    }).collect();
    let sum: f32 = samples.iter().sum();
    let inv = 1.0 / sum.max(f32::EPSILON);
    for s in samples.iter_mut() {
        *s *= inv;
    }
    samples
}

/// Value function: centipawns from the side-to-move perspective (same contract
/// as the alpha-beta `EvalFn`).
pub type ValueFn = EvalFn;

/// Combined value + move policy: centipawns for the side to move and raw
/// policy logits, one per move in `legal` (same order; empty when no policy is
/// available -> uniform priors). The eval function computes both in a single
/// forward. Passing an empty `legal` requests the value only.
pub type ValuePolicyFn = fn(&Position, &MoveList) -> (i32, Vec<f32>);

/// ValuePolicyFn adapter that wraps an EvalFn (value only, no policy logits).
/// Uses a static to bridge between EvalFn and ValuePolicyFn since fn pointers
/// can't capture other fn pointers. Mutex (not OnceLock): EvalFile switches
/// and per-go updates must take effect instead of sticking on first write.
static MCTS_EVAL: std::sync::Mutex<Option<EvalFn>> = std::sync::Mutex::new(None);

/// Optional MCTS value+policy function (set from engine-uci when ONNX is loaded).
/// Mutex for the same replaceability reason as MCTS_EVAL.
static MCTS_VALUE_POLICY: std::sync::Mutex<Option<ValuePolicyFn>> =
    std::sync::Mutex::new(None);

/// Set the evaluation function to be used by MCTS when no ONNX is available.
pub fn set_mcts_eval(eval: EvalFn) {
    *MCTS_EVAL.lock().unwrap_or_else(|e| e.into_inner()) = Some(eval);
}

/// Set the value+policy function for MCTS (used when ONNX is loaded).
pub fn set_mcts_value_policy(vp: ValuePolicyFn) {
    *MCTS_VALUE_POLICY.lock().unwrap_or_else(|e| e.into_inner()) = Some(vp);
}

/// Policy output size of the NN (must match nn::POLICY_SIZE and
/// python/chessnet/resnet.py POLICY_SIZE): 4096 from*64+to slots on the
/// side-to-move oriented board + 3x64 underpromotion slices. The engine
/// crate cannot depend on nn (nn depends on engine), so this is mirrored
/// here — keep the two in sync.
const POLICY_SIZE: usize = 4288;
const PROMO_BASE: usize = 4096;

/// Move -> NN policy index (mirrors nn::policy_index).
fn nn_policy_index(side: usize, m: Move) -> usize {
    use crate::move_::{PROMO_BISHOP, PROMO_KNIGHT, PROMO_ROOK};
    let (mut f, mut t) = (m.from(), m.to());
    if side == 1 {
        f = (7 - f / 8) * 8 + f % 8;
        t = (7 - t / 8) * 8 + t % 8;
    }
    let base = f * 64 + t;
    if m.is_promotion() {
        match m.promo() {
            PROMO_KNIGHT => return PROMO_BASE + 0 * 64 + f,
            PROMO_BISHOP => return PROMO_BASE + 1 * 64 + f,
            PROMO_ROOK => return PROMO_BASE + 2 * 64 + f,
            _ => return base,
        }
    }
    base
}

/// MCTS-compatible value function. Uses the ONNX policy head when available,
/// falls back to the stored EvalFn (no policy) otherwise.
pub fn mcts_value_fn(pos: &Position, legal: &MoveList) -> (i32, Vec<f32>) {
    let vp: Option<ValuePolicyFn> =
        *MCTS_VALUE_POLICY.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(vp) = vp {
        let (cp, all_logits) = vp(pos, legal);
        if all_logits.len() == POLICY_SIZE && legal.len > 0 {
            let mut filtered = Vec::with_capacity(legal.len);
            for i in 0..legal.len {
                let mv = legal.moves[i];
                filtered.push(all_logits[nn_policy_index(pos.side, mv)]);
            }
            return (cp, filtered);
        }
        return (cp, all_logits);
    }
    let eval: EvalFn = (*MCTS_EVAL.lock().unwrap_or_else(|e| e.into_inner()))
        .unwrap_or(crate::evaluate::evaluate);
    (eval(pos), Vec::new())
}

#[derive(Default)]
pub struct MctsLimits {
    pub playouts: Option<u64>,
    pub movetime: Option<u64>,
    /// Worker threads (0 = auto = available parallelism).
    pub threads: usize,
}

pub struct MctsProgress {
    pub playouts: u64,
    /// Win probability for the side to move, from the most-visited child.
    pub value: f32,
    pub best: Move,
    /// Top moves by visits (at most 8), most-visited first.
    pub moves: Vec<MctsChildInfo>,
    /// Playouts that ended in a checkmate.
    pub mates: u64,
    /// Playouts that ended in a draw (stalemate or 50-move rule).
    pub draws: u64,
    pub time_ms: u64,
}

/// Rich per-child information for the variant panel.
#[derive(Clone, Debug)]
pub struct MctsChildInfo {
    pub mv: Move,
    pub visits: u32,
    pub prior: f32,
    pub q: f32,
    pub w: f32,
    pub d: f32,
    pub l: f32,
    pub pv: Vec<Move>,
}

pub struct MctsResult {
    pub best: Move,
    /// Completed search playouts (counter). NOTE: this is not always equal
    /// to visits.sum(): instant mate/terminal returns use playouts =
    /// legal-move count with empty visits, and parallel paths bump visits
    /// (virtual loss) for leaves discarded on stop/budget.
    pub playouts: u64,
    /// Visit counts for each root move, most-visited first. EMPTY on
    /// instant mate/terminal returns (best still holds the move).
    pub visits: Vec<(Move, u32)>,
    /// Win probability for the side to move (from the most-visited child).
    pub value: f32,
    /// Playouts that ended in a checkmate.
    pub mates: u64,
    /// Playouts that ended in a draw (stalemate or 50-move rule).
    pub draws: u64,
    pub time_ms: u64,
}

/// Rough win/draw/loss probabilities from a zero-sum value `q` in [-1, 1].
/// Self-consistent: W + 0.5·D == (q+1)/2, so the displayed expected score
/// always matches the backed-up value. Draw mass peaks near equality and
/// vanishes at decisive scores.
pub fn wdl_from_q(q: f32) -> (f32, f32, f32) {
    let p = q_to_prob(q);
    let draw = 0.6 * (1.0 - q.abs());
    let w = (p - 0.5 * draw).clamp(0.0, 1.0);
    let l = (1.0 - p - 0.5 * draw).clamp(0.0, 1.0);
    (w, draw, l)
}

/// Win-probability scaling: `sigmoid(cp / 400)`.
#[inline]
pub fn prob_from_cp(cp: i32) -> f32 {
    1.0 / (1.0 + (-(cp as f32) / 400.0).exp())
}

/// Zero-sum score (in [-1, 1]) from `sigmoid(cp / 400)`.
#[inline]
fn score_from_cp(cp: i32) -> f32 {
    2.0 * prob_from_cp(cp) - 1.0
}

/// Display mapping from win probability to centipawns, Lc0-style (the same
/// numbers Nibbler shows): `111.714640912 * tan(1.5620688421 * Q)` with
/// Q = 2p-1. Much flatter than the binary logistic near equality, so
/// drawish positions show modest pawn scores (e.g. 57% -> ~0.26, not 1.16).
/// Display only — search backup still uses the logistic `score_from_cp`.
#[inline]
pub fn cp_from_prob(p: f32) -> i32 {
    let q = (2.0 * p - 1.0).clamp(-1.0, 1.0);
    (111.714640912 * (1.5620688421 * q).tan()).round() as i32
}

/// Zero-sum score to win probability for the side to move.
#[inline]
pub fn q_to_prob(q: f32) -> f32 {
    (q + 1.0) * 0.5
}

pub const ROOT_ID: u32 = 0;
pub const NO_ID: u32 = u32::MAX;
pub const C_PUCT: f32 = 1.4;
/// Selection never descends deeper than this; such nodes are evaluated as leaves.
pub const MAX_DEPTH: u32 = 96;
/// Fixed-point scale for backed-up values (q in [-1, 1] -> [-SCALE, SCALE]).
pub const SCALE: i32 = 128;

/// Proven outcome, side-to-move frame (Lc0 "sticky endgames": a proven
/// result sticks instead of being averaged away by NN noise).
pub const PROVEN_NONE: u8 = 0;
/// Side to move has a forced win.
pub const PROVEN_WIN: u8 = 1;
/// Side to move is lost (mated).
pub const PROVEN_LOSS: u8 = 2;
/// Proven draw (stalemate, 50-move rule).
pub const PROVEN_DRAW: u8 = 3;

/// A node of the search tree. All mutable state is atomic so the tree is shared
/// across worker threads without locks on the hot path. `val_visits` packs
/// `value` (i32, high 32 bits) and `visits` (u32, low 32 bits) so a backup is a
/// single read-modify-write.
pub struct Node {
    mv: AtomicU32,
    parent: AtomicU32,
    first_child: AtomicU32,
    next_sibling: AtomicU32,
    n_children: AtomicU32,
    terminal: AtomicBool,
    val_visits: AtomicU64,
    prior: AtomicU32,
    /// Expansion spinlock.
    lock: AtomicBool,
    /// Proven outcome (side-to-move frame); see PROVEN_*.
    proven: AtomicU8,
    /// True when expansion created children for ALL legal moves (needed to
    /// soundly infer a loss from "all children decided").
    complete: AtomicBool,
}

impl Node {
    fn new() -> Self {
        Node {
            mv: AtomicU32::new(Move::null().0),
            parent: AtomicU32::new(NO_ID),
            first_child: AtomicU32::new(NO_ID),
            next_sibling: AtomicU32::new(NO_ID),
            n_children: AtomicU32::new(0),
            terminal: AtomicBool::new(false),
            val_visits: AtomicU64::new(0),
            prior: AtomicU32::new(0),
            lock: AtomicBool::new(false),
            proven: AtomicU8::new(PROVEN_NONE),
            complete: AtomicBool::new(false),
        }
    }

    #[inline]
    fn mv(&self) -> Move {
        Move(self.mv.load(Ordering::Relaxed))
    }

    #[inline]
    fn prior(&self) -> f32 {
        f32::from_bits(self.prior.load(Ordering::Relaxed))
    }

    #[inline]
    fn visits(&self) -> u32 {
        (self.val_visits.load(Ordering::Relaxed) & 0xFFFF_FFFF) as u32
    }

    #[inline]
    fn q(&self) -> f32 {
        // Proven results stick (Lc0-style): report the exact outcome instead
        // of the NN-noise-diluted average. Stored values are parent-side
        // perspective, i.e. the negation of the side-to-move frame.
        match self.proven.load(Ordering::Relaxed) {
            PROVEN_WIN => -1.0,
            PROVEN_LOSS => 1.0,
            PROVEN_DRAW => 0.0,
            _ => {
                let packed = self.val_visits.load(Ordering::Relaxed);
                let visits = (packed & 0xFFFF_FFFF) as u32;
                let value = ((packed >> 32) as u32) as i32;
                (value as f32 / SCALE as f32) / visits.max(1) as f32
            }
        }
    }
}

/// Packs `value` (i32) and `visits` (u32) into one u64.
#[inline]
fn pack(value: i32, visits: u32) -> u64 {
    ((value as u32 as u64) << 32) | visits as u64
}

/// Fixed-size arena shared across workers.
struct Tree {
    nodes: Box<[Node]>,
    len: AtomicU32,
    /// Playouts that ended in a checkmate.
    mates: AtomicU64,
    /// Playouts that ended in a draw (stalemate or 50-move rule).
    draws: AtomicU64,
}

impl Tree {
    /// Allocates the arena (the root is node 0) and preallocates `cap` slots.
    fn new(cap: usize) -> Self {
        let nodes: Box<[Node]> = (0..cap).map(|_| Node::new()).collect::<Vec<_>>().into_boxed_slice();
        let tree = Tree { nodes, len: AtomicU32::new(0), mates: AtomicU64::new(0), draws: AtomicU64::new(0) };
        tree.alloc(); // root, id 0
        tree
    }

    #[inline]
    fn node(&self, id: u32) -> &Node {
        &self.nodes[id as usize]
    }

    /// Bumps the arena; `NO_ID` when full.
    fn alloc(&self) -> u32 {
        let id = self.len.fetch_add(1, Ordering::Relaxed);
        if (id as usize) >= self.nodes.len() {
            NO_ID
        } else {
            id
        }
    }

    #[inline]
    fn add_scaled(&self, id: u32, delta: i32) {
        let a = &self.node(id).val_visits;
        a.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |packed| {
            let visits = (packed & 0xFFFF_FFFF) as u32;
            let value = ((packed >> 32) as u32) as i32;
            Some(pack(value.wrapping_add(delta), visits))
        })
        .unwrap();
    }

    /// Increments `visits` by one (single read-modify-write).
    #[inline]
    fn add_visit(&self, id: u32) {
        let a = &self.node(id).val_visits;
        a.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |packed| {
            let visits = (packed & 0xFFFF_FFFF) as u32;
            let value = ((packed >> 32) as u32) as i32;
            Some(pack(value, visits.wrapping_add(1)))
        })
        .unwrap();
    }

    fn ucb(&self, node_id: u32, parent_visits: u32) -> f32 {
        let n = self.node(node_id);
        let visits = n.visits();
        if visits == 0 {
            return f32::INFINITY;
        }
        n.q() + C_PUCT * n.prior() * (parent_visits as f32).sqrt() / (1.0 + visits as f32)
    }

    fn best_child(&self, node: u32) -> u32 {
        let mut best = self.node(node).first_child.load(Ordering::Relaxed);
        let mut best_visits = u32::MIN;
        let mut c = best;
        while c != NO_ID {
            let v = self.node(c).visits();
            if v > best_visits {
                best_visits = v;
                best = c;
            }
            c = self.node(c).next_sibling.load(Ordering::Relaxed);
        }
        best
    }

    /// Best child move + q, or `None` when the node has no children (e.g.
    /// immediate arena exhaustion at the root). Prefer this over
    /// `node(best_child(..))`: with panic=abort, an unchecked NO_ID index
    /// kills the whole process.
    fn best_child_info(&self, node: u32) -> Option<(Move, f32)> {
        let best = self.best_child(node);
        if best == NO_ID {
            None
        } else {
            let n = self.node(best);
            Some((n.mv(), n.q()))
        }
    }

    fn visits_of(&self, node: u32) -> Vec<(Move, u32)> {
        let mut out = Vec::new();
        let mut c = self.node(node).first_child.load(Ordering::Relaxed);
        while c != NO_ID {
            out.push((self.node(c).mv(), self.node(c).visits()));
            c = self.node(c).next_sibling.load(Ordering::Relaxed);
        }
        out.sort_by_key(|b| std::cmp::Reverse(b.1));
        out
    }

    /// q of the child of `node` matching `mv`, if any.
    fn q_of_child(&self, node: u32, mv: Move) -> Option<f32> {
        let mut c = self.node(node).first_child.load(Ordering::Relaxed);
        while c != NO_ID {
            let n = self.node(c);
            if n.mv() == mv {
                return Some(n.q());
            }
            c = n.next_sibling.load(Ordering::Relaxed);
        }
        None
    }

    /// Extract the principal variation starting from `node`, following the most-visited child at each level.
    fn pv_from(&self, node: u32, max_depth: usize) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut current = node;
        for _ in 0..max_depth {
            let best = self.best_child(current);
            if best == NO_ID || self.node(best).visits() == 0 {
                break;
            }
            pv.push(self.node(best).mv());
            current = best;
        }
        pv
    }

    /// Per-child stats for each child of `node`.
    fn child_stats(&self, node: u32, pv_depth: usize) -> Vec<MctsChildInfo> {
        let mut out = Vec::new();
        let mut c = self.node(node).first_child.load(Ordering::Relaxed);
        while c != NO_ID {
            let n = self.node(c);
            let visits = n.visits();
            let q = n.q();
            let (w, d, l) = wdl_from_q(q);
            let pv = if visits > 0 { self.pv_from(c, pv_depth) } else { Vec::new() };
            out.push(MctsChildInfo {
                mv: n.mv(),
                visits,
                prior: n.prior(),
                q,
                w: w * 100.0,
                d: d * 100.0,
                l: l * 100.0,
                pv,
            });
            c = n.next_sibling.load(Ordering::Relaxed);
        }
        out.sort_by(|a, b| b.visits.cmp(&a.visits));
        out
    }
}

/// Writes softmax priors for `logits` (one per legal move, `legal_len` of them)
/// into `priors`. When `logits` is empty (no policy head), uses uniform priors.
fn softmax_priors(logits: &[f32], legal_len: usize, priors: &mut [f32]) {
    if logits.len() != legal_len {
        priors[..legal_len].fill(1.0 / legal_len as f32);
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &l in &logits[..legal_len] {
        max = max.max(l);
    }
    let mut sum = 0.0;
    for (i, &l) in logits[..legal_len].iter().enumerate() {
        let e = (l - max).exp();
        priors[i] = e;
        sum += e;
    }
    let inv = 1.0 / sum.max(f32::EPSILON);
    for p in &mut priors[..legal_len] {
        *p *= inv;
    }
}

/// Value-only eval (leaf paths that don't expand): value for the side to move.
#[inline]
fn eval_value(eval_fn: ValuePolicyFn, pos: &Position) -> f32 {
    let empty = MoveList::new();
    score_from_cp(eval_fn(pos, &empty).0)
}

/// Mark `node` terminal with an exact side-to-move outcome (`1` win,
/// `-1` loss, `0` draw) and propagate proof-like results upward (Lc0
/// "sticky endgames"). Only the fresh transition propagates; concurrent
/// duplicate marks are idempotent, so this is thread-safe.
fn mark_terminal(tree: &Tree, node: u32, outcome_stm: i8) {
    let flag = match outcome_stm {
        1 => PROVEN_WIN,
        -1 => PROVEN_LOSS,
        _ => PROVEN_DRAW,
    };
    tree.node(node).terminal.store(true, Ordering::Release);
    if tree.node(node).proven.compare_exchange(PROVEN_NONE, flag, Ordering::AcqRel, Ordering::Relaxed).is_ok() {
        let mut cur = tree.node(node).parent.load(Ordering::Relaxed);
        while cur != NO_ID {
            if !try_decide(tree, cur) {
                break;
            }
            cur = tree.node(cur).parent.load(Ordering::Relaxed);
        }
    }
}

/// Try to prove `node` from its children's proven flags. Returns true when
/// the node is decided (already or newly):
/// - some child proven lost-for-its-side  => node proven won (mate found);
/// - every child decided (requires full expansion) and all won-for-their
///   side => node proven lost; mixed won/drawn => proven draw.
/// An undecided node, or one with unexpanded moves and no proven-loss
/// child, returns false.
fn try_decide(tree: &Tree, node: u32) -> bool {
    let n = tree.node(node);
    if n.proven.load(Ordering::Acquire) != PROVEN_NONE {
        return true;
    }
    if n.n_children.load(Ordering::Acquire) == 0 {
        return false;
    }
    let mut all_decided = true;
    let mut all_won = true;
    // Acquire loads: a sibling proven concurrently by another worker must
    // become visible here, or a fully-decided parent could be missed forever
    // (both climbers seeing each other undecided). Later terminals climbing
    // through this node re-run the check, which self-heals any residue.
    let mut c = n.first_child.load(Ordering::Acquire);
    while c != NO_ID {
        match tree.node(c).proven.load(Ordering::Acquire) {
            PROVEN_LOSS => {
                n.proven.store(PROVEN_WIN, Ordering::Release);
                return true;
            }
            PROVEN_WIN => {}
            PROVEN_DRAW => {
                all_won = false;
            }
            _ => {
                all_decided = false;
                all_won = false;
            }
        }
        c = tree.node(c).next_sibling.load(Ordering::Acquire);
    }
    // Sound only with all legal moves expanded: uncreated children might win.
    if all_decided && n.complete.load(Ordering::Acquire) {
        n.proven.store(if all_won { PROVEN_LOSS } else { PROVEN_DRAW }, Ordering::Release);
        return true;
    }
    false
}

/// Fresh RNG seed per call (nanos + atomic counter): temperature sampling
/// and Dirichlet noise must differ across games, not replay one fixed draw.
fn next_rng() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static CTR: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9E3779B97F4A7C15);
    nanos.wrapping_add(CTR.fetch_add(0x9E3779B97F4A7C15, Ordering::Relaxed))
}

/// Re-apply Dirichlet noise to an already-expanded root's priors (used
/// when a kept subtree is reused with dirichlet=true: fresh noise per
/// search, like a fresh expansion).
fn renoise_root(tree: &Tree, root: u32) {
    let mut c = tree.node(root).first_child.load(Ordering::Relaxed);
    if c == NO_ID {
        return;
    }
    let mut n = 0usize;
    let mut cur = c;
    while cur != NO_ID {
        n += 1;
        cur = tree.node(cur).next_sibling.load(Ordering::Relaxed);
    }
    if n < 2 {
        return;
    }
    let mut rng = next_rng();
    let noise = sample_dirichlet(n, DIRICHLET_ALPHA, &mut rng);
    let eps = DIRICHLET_EPSILON;
    let mut i = 0usize;
    while c != NO_ID {
        let node = tree.node(c);
        let p = f32::from_bits(node.prior.load(Ordering::Relaxed));
        node.prior.store(((1.0 - eps) * p + eps * noise[i]).to_bits(), Ordering::Relaxed);
        i += 1;
        c = node.next_sibling.load(Ordering::Relaxed);
    }
}

/// Creates the root's children (single-threaded, before any worker starts).
/// `root` is the node whose children are created (ROOT_ID for a fresh tree, or
/// a reused subtree root). When `dirichlet` is true, adds Dirichlet noise to
/// the priors for exploration (AlphaZero-style).
fn expand_root(tree: &Tree, root: u32, pos: &Position, eval_fn: ValuePolicyFn, dirichlet: bool) {
    let legal = generate_legal(pos);
    let (_, logits) = eval_fn(pos, &legal);
    let mut priors = [0f32; MAX_MOVES];
    softmax_priors(&logits, legal.len, &mut priors);
    if dirichlet && legal.len > 0 {
        let mut rng = next_rng();
        let noise = sample_dirichlet(legal.len, DIRICHLET_ALPHA, &mut rng);
        let eps = DIRICHLET_EPSILON;
        for i in 0..legal.len {
            priors[i] = (1.0 - eps) * priors[i] + eps * noise[i];
        }
    }
    let mut first = NO_ID;
    let mut count = 0u32;
    for (i, &mv) in legal.moves[..legal.len].iter().enumerate() {
        let id = tree.alloc();
        if id == NO_ID {
            break;
        }
        let c = tree.node(id);
        c.mv.store(mv.0, Ordering::Relaxed);
        c.parent.store(root, Ordering::Relaxed);
        c.prior.store(priors[i].to_bits(), Ordering::Relaxed);
        c.next_sibling.store(first, Ordering::Relaxed);
        first = id;
        count += 1;
    }
    tree.node(root).first_child.store(first, Ordering::Release);
    tree.node(root).n_children.store(count, Ordering::Release);
    tree.node(root).complete.store(count as usize == legal.len, Ordering::Release);
}

/// Expands `idx` under its spinlock (once) and returns the leaf value for the
/// side to move. Concurrent workers spin briefly, then evaluate as a leaf.
fn expand(tree: &Tree, idx: u32, pos: &mut Position, eval_fn: ValuePolicyFn) -> f32 {
    let lock = &tree.node(idx).lock;
    // If another thread is already expanding this leaf, don't wait: evaluate it
    // as a leaf instead (the tree is still correct, just shallower there).
    if lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_err() {
        return eval_value(eval_fn, pos);
    }

    let node = tree.node(idx);
    let v = if node.n_children.load(Ordering::Acquire) > 0 || node.terminal.load(Ordering::Acquire) {
        // Another thread already expanded this node (or it turned terminal).
        eval_value(eval_fn, pos)
    } else {
        let legal = generate_legal(pos);
        if legal.len == 0 {
            // Exact terminal outcome + sticky propagation (never NN noise).
            let mate = pos.in_check();
            mark_terminal(tree, idx, if mate { -1 } else { 0 });
            if mate { -1.0 } else { 0.0 }
} else {
            let (v_cp, logits) = eval_fn(pos, &legal);
            let v = score_from_cp(v_cp);
            let mut priors = [0f32; MAX_MOVES];
            softmax_priors(&logits, legal.len, &mut priors);
            let mut first = NO_ID;
            let mut count = 0u32;
            for (i, &mv) in legal.moves[..legal.len].iter().enumerate() {
                let id = tree.alloc();
                if id == NO_ID {
                    break;
                }
                let c = tree.node(id);
                c.mv.store(mv.0, Ordering::Relaxed);
                c.parent.store(idx, Ordering::Relaxed);
                c.prior.store(priors[i].to_bits(), Ordering::Relaxed);
                c.next_sibling.store(first, Ordering::Relaxed);
                first = id;
                count += 1;
            }
            node.first_child.store(first, Ordering::Release);
            node.n_children.store(count, Ordering::Release);
            node.complete.store(count as usize == legal.len, Ordering::Release);
            v
        }
    };
    lock.store(false, Ordering::Release);
    v
}

/// One selection -> expansion/evaluation -> backup cycle. `pos` is a scratch
/// copy that is mutated as the tree is descended. `root` is the search root
/// (ROOT_ID for a fresh tree); when reusing a kept subtree it is that node and
/// the backup stops there.
fn playout_from(tree: &Tree, root: u32, pos: &mut Position, eval_fn: ValuePolicyFn) {
    let mut idx = root;
    let mut ply = 0u32;

    // The root's visit for this playout.
    tree.add_visit(root);

    // Selection: descend while the current node is expanded and not at the
    // depth cap.
    loop {
        let node = tree.node(idx);
        if node.n_children.load(Ordering::Acquire) == 0 || node.terminal.load(Ordering::Acquire) || ply >= MAX_DEPTH
        {
            break;
        }
        let parent_visits = node.visits();
        let mut best = node.first_child.load(Ordering::Relaxed);
        let mut best_u = f32::NEG_INFINITY;
        let mut c = best;
        while c != NO_ID {
            let u = tree.ucb(c, parent_visits);
            if u > best_u {
                best_u = u;
                best = c;
            }
            c = tree.node(c).next_sibling.load(Ordering::Relaxed);
        }
        // The visit for the chosen child is placed at selection time.
        tree.add_visit(best);
        pos.make_move(tree.node(best).mv());
        idx = best;
        ply += 1;
    }

// Leaf: fixed terminal result, depth-cap evaluation, or expand + evaluate.
    // `v_leaf` is the zero-sum outcome for the side to move at the leaf.
    let node = tree.node(idx);
    let v_leaf = if node.terminal.load(Ordering::Acquire) {
        // Already proven: re-marking is idempotent (no propagation repeat).
        let mate = pos.in_check();
        mark_terminal(tree, idx, if mate { -1 } else { 0 });
        if mate {
            tree.mates.fetch_add(1, Ordering::Relaxed);
            -1.0
        } else {
            tree.draws.fetch_add(1, Ordering::Relaxed);
            0.0
        }
    } else if pos.halfmove >= 100 {
        // 50-move rule: the position is drawn.
        mark_terminal(tree, idx, 0);
        tree.draws.fetch_add(1, Ordering::Relaxed);
        0.0
    } else if node.n_children.load(Ordering::Acquire) > 0 {
        // Internal node reached through the depth cap: evaluate as a leaf.
        eval_value(eval_fn, pos)
    } else {
        expand(tree, idx, pos, eval_fn)
    };

// Backup: only values are adjusted here (visits were placed during
    // selection). Each node accumulates the outcome for the side that moved
    // into it (its parent's perspective). Stops at the search root.
    let mut n = idx;
    let mut v_scaled = (-v_leaf * SCALE as f32).round() as i32;
    while n != root {
        let node = tree.node(n);
        tree.add_scaled(n, v_scaled);
        v_scaled = v_scaled.wrapping_neg();
        n = node.parent.load(Ordering::Relaxed);
    }
}

#[inline]
fn playout(tree: &Tree, pos: &mut Position, eval_fn: ValuePolicyFn) {
    playout_from(tree, ROOT_ID, pos, eval_fn);
}

fn arena_cap(limits: &MctsLimits) -> usize {
    match limits.playouts {
        Some(p) => ((p as usize).saturating_mul(24).saturating_add(4096)).min(1 << 23),
        None => 1 << 22,
    }
}

/// 1-ply exact mate check: look at every legal move once; the first move
/// that leaves the opponent with no legal reply while in check mates.
/// The engine must ALWAYS play mate-in-1 when it exists, regardless of
/// search budget, priors, or temperature.
pub fn find_mate_in_one(pos: &mut Position) -> Option<Move> {
    let legal = generate_legal(pos);
    for i in 0..legal.len {
        let m = legal.moves[i];
        let undo = pos.make_move(m);
        let reply = generate_legal(pos);
        let mate = reply.len == 0 && pos.in_check();
        pos.unmake_move(undo);
        if mate {
            return Some(m);
        }
    }
    None
}

/// True if playing `mv` lets the opponent mate in 1 (exact 1-ply check
/// from the opponent's perspective).
pub fn allows_opp_mate(pos: &mut Position, mv: Move) -> bool {
    let undo = pos.make_move(mv);
    let reply = generate_legal(pos);
    let mut mated = false;
    for i in 0..reply.len {
        let r = reply.moves[i];
        let u2 = pos.make_move(r);
        let is_mate = generate_legal(pos).len == 0 && pos.in_check();
        pos.unmake_move(u2);
        if is_mate {
            mated = true;
            break;
        }
    }
    pos.unmake_move(undo);
    mated
}

/// Best move that never walks into an opponent mate-in-1: keep `best`
/// unless it allows an immediate mate, then fall back to the most-visited
/// safe alternative (or `best` if every move is doomed).
/// PUCT exploration is prior-weighted, so at finite budgets a low-prior
/// mating reply may never be visited and the search can prefer a blunder;
/// this exact filter closes the hole (mirror of find_mate_in_one).
pub fn avoid_opp_mate(pos: &mut Position, best: Move, visits: &[(Move, u32)]) -> Move {
    if best == Move::null() || !allows_opp_mate(pos, best) {
        return best;
    }
    let mut alt = Move::null();
    let mut alt_v = 0u32;
    for &(m, v) in visits {
        if m != best && v > alt_v && !allows_opp_mate(pos, m) {
            alt = m;
            alt_v = v;
        }
    }
    if alt == Move::null() {
        best
    } else {
        alt
    }
}

/// Single-threaded MCTS (deterministic, no virtual loss). Mainly used by tests.
pub fn mcts_root(
    pos: &mut Position,
    limits: &MctsLimits,
    stop: &AtomicBool,
    eval_fn: ValuePolicyFn,
    mut on_progress: Option<&mut dyn FnMut(&MctsProgress)>,
) -> MctsResult {
    let start = Instant::now();
    let tree = Tree::new(arena_cap(limits));

    let legal = generate_legal(pos);
    if legal.len == 0 {
        let value = if pos.in_check() { 0.0 } else { 0.5 };
return MctsResult {
            best: Move::null(),
            playouts: 0,
            visits: Vec::new(),
            value,
            mates: 0,
            draws: 0,
            time_ms: start.elapsed().as_millis() as u64,
        };
    }
    // Mate-in-1 first: play it immediately, no search needed.
    if let Some(mate) = find_mate_in_one(pos) {
        return MctsResult {
            best: mate,
            playouts: legal.len as u64,
            visits: Vec::new(),
            value: 1.0,
            mates: 1,
            draws: 0,
            time_ms: start.elapsed().as_millis() as u64,
        };
    }
    expand_root(&tree, ROOT_ID, pos, eval_fn, false);

    let mut playouts = 0u64;
    let mut last_report = start;
    let root_pos = *pos;
    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if let Some(p) = limits.playouts {
            if playouts >= p {
                break;
            }
        }
        if let Some(t) = limits.movetime {
            if start.elapsed().as_millis() as u64 >= t {
                break;
            }
        }

        let mut p = root_pos;
        playout(&tree, &mut p, eval_fn);
        playouts += 1;

        if on_progress.is_some() && last_report.elapsed().as_millis() >= 16 {
            last_report = Instant::now();
            let best_id = tree.best_child(ROOT_ID);
            // best_id can be NO_ID when the arena is exhausted at the root:
            // never index the tree with it (own safety contract).
            if best_id != NO_ID {
                let value = q_to_prob(tree.node(best_id).q());
                let child_info = tree.child_stats(ROOT_ID, 20);
                if let Some(cb) = on_progress.as_mut() {
                    cb(&MctsProgress {
                        playouts,
                        value,
                        best: tree.node(best_id).mv(),
                        moves: child_info,
                        mates: tree.mates.load(Ordering::Relaxed),
                        draws: tree.draws.load(Ordering::Relaxed),
                        time_ms: start.elapsed().as_millis() as u64,
                    });
                }
            }
        }
    }

    let (best, value) = match tree.best_child_info(ROOT_ID) {
        Some((m, q)) => (m, q_to_prob(q)),
        None => (Move::null(), 0.5),
    };
    let visits = tree.visits_of(ROOT_ID);
    // Never walk into an opponent mate-in-1 (see avoid_opp_mate).
    let best = avoid_opp_mate(pos, best, &visits);
    MctsResult {
        best,
        playouts,
        visits,
        value,
        mates: tree.mates.load(Ordering::Relaxed),
        draws: tree.draws.load(Ordering::Relaxed),
        time_ms: start.elapsed().as_millis() as u64,
    }
}

/// Forest parallel MCTS: each worker runs an independent single-threaded tree
/// (no shared state, so it scales almost linearly), and the root visit counts
/// are merged at the end. `stop` is shared with the caller (e.g. the UI's
/// "stop" button); `on_progress` is called from the calling thread while the
/// workers run.
pub fn mcts_parallel(
    pos: &mut Position,
    limits: &MctsLimits,
    stop: Arc<AtomicBool>,
    eval_fn: ValuePolicyFn,
    mut on_progress: Option<&mut dyn FnMut(&MctsProgress)>,
    prev: Option<MctsSearch>,
) -> (MctsResult, MctsSearch) {
    let start = Instant::now();

    let legal = generate_legal(pos);
    if legal.len == 0 {
        let value = if pos.in_check() { 0.0 } else { 0.5 };
        let empty = MctsSearch::new(1024, 1);
        return (MctsResult {
            best: Move::null(),
            playouts: 0,
            visits: Vec::new(),
            value,
            mates: 0,
            draws: 0,
            time_ms: start.elapsed().as_millis() as u64,
        }, empty);
    }
    // Mate-in-1 first (guaranteed on every entry point).
    if let Some(mate) = find_mate_in_one(pos) {
        let mut ms = MctsSearch::new(1024, 1);
        ms.root_key = pos.key;
        return (MctsResult {
            best: mate,
            playouts: legal.len as u64,
            visits: Vec::new(),
            value: 1.0,
            mates: 1,
            draws: 0,
            time_ms: start.elapsed().as_millis() as u64,
        }, ms);
    }

    let threads = if limits.threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        limits.threads
    };
    let threads = threads.max(1);

    let root_pos = *pos;
    let counter = Arc::new(AtomicU64::new(0));
    let active = Arc::new(AtomicUsize::new(0));
    let budget_playouts = limits.playouts;
    let budget_movetime = limits.movetime;

    let mut trees: Vec<Arc<Tree>> = Vec::with_capacity(threads);
    let cap = per_thread_cap(limits, threads);
    let mut root = ROOT_ID;

    if let Some(mut prev_search) = prev {
        // Defense in depth: only reuse the kept subtree when it actually
        // belongs to this position. A stale tree would run tree moves
        // against a mismatched position (panic in make_move on a worker).
        if prev_search.root_key() != pos.key {
            prev_search = MctsSearch::new(cap, threads);
        }
        root = prev_search.root;
        let tree = prev_search.tree.clone();
        // Ensure root is expanded for the current position
        if tree.node(root).n_children.load(Ordering::Acquire) == 0 {
            expand_root(&tree, root, &root_pos, eval_fn, false);
        }
        // All threads share the same tree
        for _ in 0..threads {
            trees.push(tree.clone());
        }
    } else {
        for _ in 0..threads {
            let tree = Arc::new(Tree::new(cap));
            expand_root(&tree, ROOT_ID, &root_pos, eval_fn, false);
            trees.push(tree);
        }
    }

    let mut handles = Vec::with_capacity(threads);
    for (i, tree) in trees.iter().enumerate() {
        let stop = stop.clone();
        let counter = counter.clone();
        let active = active.clone();
        let tree = tree.clone();
        let tree_root = root;
        handles.push(std::thread::spawn(move || {
            active.fetch_add(1, Ordering::Relaxed);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                loop {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Some(p) = budget_playouts {
                        if counter.load(Ordering::Relaxed) >= p {
                            break;
                        }
                    }
                    if let Some(t) = budget_movetime {
                        if start.elapsed().as_millis() as u64 >= t {
                            break;
                        }
                    }
                    let mut p = root_pos;
                    playout_from(&tree, tree_root, &mut p, eval_fn);
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            }));
            if let Err(e) = result {
                eprintln!("[mcts] worker panicked: {:?}", e);
            }
            active.fetch_sub(1, Ordering::Relaxed);
        }));
    }

    // Wait for at least one worker to start before entering progress loop
    while active.load(Ordering::Relaxed) == 0 {
        std::thread::yield_now();
    }

    if let Some(cb) = on_progress.as_mut() {
        while active.load(Ordering::Relaxed) > 0 {
            std::thread::sleep(Duration::from_millis(16));
            if stop.load(Ordering::Relaxed) {
                break;
            }
            // Same aliasing caveat as the final merge below: dedupe.
            let mut seen: Vec<&Arc<Tree>> = Vec::with_capacity(trees.len());
            for t in &trees {
                if !seen.iter().any(|u| Arc::ptr_eq(u, t)) {
                    seen.push(t);
                }
            }
            let owned: Vec<Arc<Tree>> = seen.iter().map(|t| (*t).clone()).collect();
            let (best, value, _moves) = merge_trees(&owned, root);
            let child_info = merge_child_stats(&owned, root, 20);
            cb(&MctsProgress {
                playouts: counter.load(Ordering::Relaxed),
                value,
                best,
                moves: child_info,
                mates: owned.iter().map(|t| t.mates.load(Ordering::Relaxed)).sum(),
                draws: owned.iter().map(|t| t.draws.load(Ordering::Relaxed)).sum(),
                time_ms: start.elapsed().as_millis() as u64,
            });
        }
    }

    for h in handles {
        let _ = h.join();
    }

    // Dedupe aliased Arcs: with `prev` reuse all threads share ONE tree,
    // and naive summation would count every visit N times.
    let mut uniq: Vec<Arc<Tree>> = Vec::with_capacity(trees.len());
    for t in &trees {
        if !uniq.iter().any(|u| Arc::ptr_eq(u, t)) {
            uniq.push(t.clone());
        }
    }
    let (best, value, visits) = merge_trees(&uniq, root);
    let mates: u64 = uniq.iter().map(|t| t.mates.load(Ordering::Relaxed)).sum();
    let draws: u64 = uniq.iter().map(|t| t.draws.load(Ordering::Relaxed)).sum();
    // Never walk into an opponent mate-in-1 (see avoid_opp_mate).
    let best = avoid_opp_mate(pos, best, &visits);
    let result = MctsResult {
        best,
        playouts: counter.load(Ordering::Relaxed),
        visits,
        value,
        mates,
        draws,
        time_ms: start.elapsed().as_millis() as u64,
    };

    let search = MctsSearch {
        tree: trees.into_iter().next().unwrap(),
        root,
        threads,
        cap,
        rebuilds: 0,
        root_key: root_pos.key,
    };

    (result, search)
}

/// Arena size for one worker of a `threads`-way forest search: each tree only
/// needs its share of the playout budget (the tree grows ~24 nodes/playout),
/// bounded to keep total memory sane.
fn per_thread_cap(limits: &MctsLimits, threads: usize) -> usize {
    match limits.playouts {
        Some(p) => {
            let share = p.div_ceil(threads as u64);
            ((share as usize).saturating_mul(24).saturating_add(4096)).min(1 << 17)
        }
        None => 1 << 17,
    }
}

/// Merges the root visit counts of a forest into a single (best, value,
/// sorted-visits) result. `value` is the visit-weighted q of the best move.
fn merge_trees(trees: &[Arc<Tree>], root: u32) -> (Move, f32, Vec<(Move, u32)>) {
    use std::collections::HashMap;
    let mut visits: HashMap<Move, u32> = HashMap::new();
    let mut q_sum: HashMap<Move, f64> = HashMap::new();
    for tree in trees {
        for (m, v) in tree.visits_of(root) {
            if v == 0 {
                continue;
            }
            *visits.entry(m).or_insert(0) += v;
            if let Some(q) = tree.q_of_child(root, m) {
                *q_sum.entry(m).or_insert(0.0) += q as f64 * v as f64;
            }
        }
    }
    if visits.is_empty() {
        return (Move::null(), 0.5, Vec::new());
    }
    let mut sorted: Vec<(Move, u32)> = visits.into_iter().collect();
    sorted.sort_by_key(|b| std::cmp::Reverse(b.1));
    let best = sorted[0].0;
    let total = sorted[0].1.max(1) as f64;
    let q = (q_sum.get(&best).copied().unwrap_or(0.0) / total).clamp(-1.0, 1.0) as f32;
    (best, q_to_prob(q), sorted)
}

/// Merge per-child stats across trees: returns MctsChildInfo sorted by visits.
fn merge_child_stats(trees: &[Arc<Tree>], root: u32, pv_depth: usize) -> Vec<MctsChildInfo> {
    use std::collections::HashMap;
    let mut visits: HashMap<Move, u32> = HashMap::new();
    let mut q_sum: HashMap<Move, f64> = HashMap::new();
    let mut priors: HashMap<Move, f32> = HashMap::new();

    for tree in trees {
        let mut c = tree.node(root).first_child.load(Ordering::Relaxed);
        while c != NO_ID {
            let n = tree.node(c);
            let mv = n.mv();
            let v = n.visits();
            *visits.entry(mv).or_insert(0) += v;
            if v > 0 {
                if let Some(q) = tree.q_of_child(root, mv) {
                    *q_sum.entry(mv).or_insert(0.0) += q as f64 * v as f64;
                }
            }
            priors.entry(mv).or_insert(n.prior());
            c = n.next_sibling.load(Ordering::Relaxed);
        }
    }

    let mut out: Vec<MctsChildInfo> = visits.into_iter().map(|(mv, total_v)| {
        let total = total_v.max(1) as f64;
        let q = (q_sum.get(&mv).copied().unwrap_or(0.0) / total).clamp(-1.0, 1.0) as f32;
        let (w, d, l) = wdl_from_q(q);

        // PV from first tree that has this child
        let pv = trees.iter().find_map(|tree| {
            let mut c = tree.node(root).first_child.load(Ordering::Relaxed);
            while c != NO_ID {
                if tree.node(c).mv() == mv && tree.node(c).visits() > 0 {
                    return Some(tree.pv_from(c, pv_depth));
                }
                c = tree.node(c).next_sibling.load(Ordering::Relaxed);
            }
            None
        }).unwrap_or_default();

        MctsChildInfo {
            mv,
            visits: total_v,
            prior: priors.get(&mv).copied().unwrap_or(0.0),
            q,
            w: (w * 100.0) as f32,
            d: (d * 100.0) as f32,
            l: (l * 100.0) as f32,
            pv,
        }
    }).collect();

    out.sort_by(|a, b| b.visits.cmp(&a.visits));
    out
}

/// Incremental parallel MCTS with a reusable tree (AlphaZero-style): after a
/// search, `keep_child` re-points the root at the chosen child so the next
/// search builds on the existing subtree. The tree is shared across worker
/// threads (all state is atomic); a full arena triggers a rebuild. This is the
/// workhorse for self-play generation.
pub struct MctsSearch {
    tree: Arc<Tree>,
    root: u32,
    threads: usize,
    cap: usize,
    rebuilds: u64,
    /// Zobrist key of the position `root` corresponds to. Used to detect
    /// a stale tree (e.g. new game / undo / FEN while analysing) so a
    /// search never runs tree moves against a mismatched position
    /// (which panics in `make_move`). `u64::MAX` = unknown / empty.
    root_key: u64,
}

impl MctsSearch {
    pub fn new(cap: usize, threads: usize) -> Self {
        let cap = cap.max(1024);
        MctsSearch {
            tree: Arc::new(Tree::new(cap)),
            root: ROOT_ID,
            threads: threads.max(1),
            cap,
            rebuilds: 0,
            root_key: u64::MAX,
        }
    }

    /// Zobrist key of the position the current root corresponds to.
    pub fn root_key(&self) -> u64 {
        self.root_key
    }

    /// Number of times the arena was full and the tree had to be rebuilt.
    pub fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    /// Starts a fresh tree (new game).
    pub fn reset(&mut self) {
        self.tree = Arc::new(Tree::new(self.cap));
        self.root = ROOT_ID;
        self.root_key = u64::MAX;
    }

    /// Runs up to `playouts` playouts (or until `stop`) from the current root,
    /// which must correspond to `pos`. Terminal positions return immediately.
    /// When `dirichlet` is true, adds Dirichlet noise to root priors.
    /// `temperature` controls move selection: 0 = pick most visited,
    /// >0 = sample proportional to visits^(1/temp).
    pub fn search(
        &mut self,
        pos: &Position,
        playouts: u64,
        eval_fn: ValuePolicyFn,
        stop: Arc<AtomicBool>,
        dirichlet: bool,
        temperature: f32,
    ) -> MctsResult {
        let start = Instant::now();
        // Contract: the current root corresponds to `pos`; record it so a
        // later caller can detect a stale tree via `root_key()`.
        self.root_key = pos.key;
        let legal = generate_legal(pos);
        if legal.len == 0 {
            let value = if pos.in_check() { 0.0 } else { 0.5 };
            return MctsResult {
                best: Move::null(),
                playouts: 0,
                visits: Vec::new(),
                value,
                mates: 0,
                draws: 0,
                time_ms: 0,
            };
        }
        // Mate-in-1 first: play it immediately, no search needed.
        if let Some(mate) = find_mate_in_one(&mut pos.clone()) {
            return MctsResult {
                best: mate,
                playouts: legal.len as u64,
                visits: Vec::new(),
                value: 1.0,
                mates: 1,
                draws: 0,
                time_ms: start.elapsed().as_millis() as u64,
            };
        }

        // Rebuild when the arena is nearly full (rare with a generous cap).
        if self.tree.len.load(Ordering::Relaxed) as usize + 4096 > self.cap {
            self.tree = Arc::new(Tree::new(self.cap));
            self.root = ROOT_ID;
            self.rebuilds += 1;
        }

        // Make sure the root's children exist for this position.
        if self.tree.node(self.root).n_children.load(Ordering::Acquire) == 0 {
            expand_root(&self.tree, self.root, pos, eval_fn, dirichlet);
        } else if dirichlet {
            // Kept subtree: refresh the exploration noise for this search.
            renoise_root(&self.tree, self.root);
        }

        let tree = self.tree.clone();
        let root = self.root;
        let root_pos = *pos;
        let counter = Arc::new(AtomicU64::new(0));
        let mut handles = Vec::with_capacity(self.threads);
        for _ in 0..self.threads {
            let tree = tree.clone();
            let counter = counter.clone();
            let stop = stop.clone();
            handles.push(std::thread::spawn(move || loop {
                if stop.load(Ordering::Relaxed) || counter.load(Ordering::Relaxed) >= playouts {
                    break;
                }
                let mut p = root_pos;
                playout_from(&tree, root, &mut p, eval_fn);
                counter.fetch_add(1, Ordering::Relaxed);
            }));
        }
        for h in handles {
            let _ = h.join();
        }

        let (best, value) = if temperature > 0.0
            && tree.node(self.root).first_child.load(Ordering::Relaxed) != NO_ID
        {
            let best_id = self.select_with_temperature(&tree, self.root, temperature);
            let n = tree.node(best_id);
            (n.mv(), q_to_prob(n.q()))
        } else {
            match tree.best_child_info(root) {
                Some((m, q)) => (m, q_to_prob(q)),
                None => (Move::null(), 0.5),
            }
        };
        let visits = tree.visits_of(root);
        // Never walk into an opponent mate-in-1 (see avoid_opp_mate).
        let mut tmp = *pos;
        let best = avoid_opp_mate(&mut tmp, best, &visits);
        MctsResult {
            best,
            playouts: counter.load(Ordering::Relaxed),
            visits,
            value,
            mates: tree.mates.load(Ordering::Relaxed),
            draws: tree.draws.load(Ordering::Relaxed),
            time_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Value at the end of the principal variation starting with `mv`:
    /// walk most-visited children while visited, return the deepest node's q
    /// converted to the CURRENT ROOT's perspective (stored q is always
    /// parent-perspective, so frames alternate each ply: odd depth returns
    /// q as-is, even depth negated). Returns `None` when `mv` is not a root
    /// child. This is the "PV-only" alternative to the subtree mean for
    /// scoring moves (used by the sparring harness, not by default search).
    pub fn pv_value(&self, mv: Move) -> Option<f32> {
        let mut cur = NO_ID;
        let mut c = self.tree.node(self.root).first_child.load(Ordering::Relaxed);
        while c != NO_ID {
            if self.tree.node(c).mv() == mv {
                cur = c;
                break;
            }
            c = self.tree.node(c).next_sibling.load(Ordering::Relaxed);
        }
        if cur == NO_ID {
            return None;
        }
        let mut depth = 1u32;
        loop {
            let next = self.tree.best_child(cur);
            if next == NO_ID || self.tree.node(next).visits() == 0 {
                break;
            }
            cur = next;
            depth += 1;
        }
        let q = self.tree.node(cur).q();
        Some(if depth % 2 == 1 { q } else { -q })
    }

    /// Re-points the search root at the child matching `mv` (the played move),
    /// keeping its subtree for the next search. `new_key` must be the zobrist
    /// key of the position after `mv`; it is recorded so a stale tree can be
    /// detected via `root_key()`. Falls back to a fresh tree when the move is
    /// not found (should not happen).
    pub fn keep_child(&mut self, mv: Move, new_key: u64) {
        let mut c = self.tree.node(self.root).first_child.load(Ordering::Relaxed);
        while c != NO_ID {
            let n = self.tree.node(c);
            if n.mv() == mv {
                // Sever the link to the discarded ancestors: proof
                // propagation (mark_terminal) climbs via parent links and
                // must stop at the new root.
                self.tree.node(c).parent.store(NO_ID, Ordering::Relaxed);
                self.root = c;
                self.root_key = new_key;
                return;
            }
            c = n.next_sibling.load(Ordering::Relaxed);
        }
        self.reset();
    }

    /// Returns child stats for the current root (for Descend without starting a new search).
    pub fn current_child_stats(&self) -> (u64, Vec<MctsChildInfo>) {
        let root_visits = self.tree.node(self.root).visits() as u64;
        let child_info = self.tree.child_stats(self.root, 20);
        (root_visits, child_info)
    }

    /// Select a child of `node` proportional to visits^(1/temperature).
    /// temperature > 0; lower temperature = more greedy.
    fn select_with_temperature(&self, tree: &Tree, node: u32, temperature: f32) -> u32 {
        let inv_temp = 1.0 / temperature;
        let mut total = 0.0f32;
        let mut c = tree.node(node).first_child.load(Ordering::Relaxed);
        while c != NO_ID {
            let v = tree.node(c).visits() as f32;
            total += v.powf(inv_temp);
            c = tree.node(c).next_sibling.load(Ordering::Relaxed);
        }
        if total <= 0.0 {
            return tree.best_child(node);
        }
        let mut rng = next_rng();
        let mut r = sample_uniform(&mut rng) * total;
        let mut c = tree.node(node).first_child.load(Ordering::Relaxed);
        let mut prev = c;
        while c != NO_ID {
            let v = tree.node(c).visits() as f32;
            r -= v.powf(inv_temp);
            if r <= 0.0 {
                return c;
            }
            prev = c;
            c = tree.node(c).next_sibling.load(Ordering::Relaxed);
        }
        prev
    }
}

/// Optional batch evaluation function (set from engine-uci when ONNX is loaded).
/// Mutex (not OnceLock) so EvalFile switches take effect.
static BATCH_EVAL_FN: std::sync::Mutex<Option<BatchEvalFn>> = std::sync::Mutex::new(None);

/// Set the batch evaluation function for use by batched MCTS.
pub fn set_batch_eval(f: BatchEvalFn) {
    *BATCH_EVAL_FN.lock().unwrap_or_else(|e| e.into_inner()) = Some(f);
}

/// Get the batch evaluation function, if set.
pub fn get_batch_eval() -> Option<BatchEvalFn> {
    *BATCH_EVAL_FN.lock().unwrap_or_else(|e| e.into_inner())
}

// ---------------------------------------------------------------------------
// Batched MCTS — collect N leaves → 1 GPU call → expand all → backup all.
// ---------------------------------------------------------------------------

/// Batch evaluation function: takes a slice of positions and their legal move
/// lists, returns (cp, policy_logits) for each. The caller provides this;
/// the engine crate doesn't depend on nn/ort.
pub type BatchEvalFn = fn(&[Position], &[&MoveList]) -> Vec<(i32, Vec<f32>)>;

/// Split items into `n` per-worker chunks, preserving order.
/// Workers pop whole chunks from the end; order within and across chunks
/// is unchanged, so leaf↔result pairing survives distribution.
fn split_chunks<T>(items: Vec<T>, n: usize) -> Vec<Vec<T>> {
    let n = n.max(1);
    let per = items.len().div_ceil(n);
    let mut out = Vec::with_capacity(n);
    let mut it = items.into_iter();
    for _ in 0..n {
        out.push(it.by_ref().take(per).collect());
    }
    out
}

/// Result of one selection pass (one leaf).
struct BatchLeaf {
    /// Node id of the leaf in the tree.
    node_id: u32,
    /// Position at the leaf.
    pos: Position,
    /// Legal moves at the leaf (empty when terminal).
    legal: MoveList,
    /// Path from root to this leaf (node ids, excluding root).
    path: Vec<u32>,
}

/// Single-threaded batched MCTS. Each round collects `batch_size` leaves,
/// evaluates them in one GPU call, then expands and backs up.
pub fn mcts_batched(
    pos: &mut Position,
    limits: &MctsLimits,
    stop: &AtomicBool,
    eval_fn: ValuePolicyFn,
    batch_eval: BatchEvalFn,
    batch_size: usize,
    dirichlet: bool,
    mut on_progress: Option<&mut dyn FnMut(&MctsProgress)>,
) -> MctsResult {
    let start = Instant::now();
    let tree = Tree::new(arena_cap(limits));

    let legal = generate_legal(pos);
    if legal.len == 0 {
        let value = if pos.in_check() { 0.0 } else { 0.5 };
        return MctsResult {
            best: Move::null(),
            playouts: 0,
            visits: Vec::new(),
            value,
            mates: 0,
            draws: 0,
            time_ms: start.elapsed().as_millis() as u64,
        };
    }
    // Mate-in-1 first (guaranteed on every entry point).
    if let Some(mate) = find_mate_in_one(pos) {
        return MctsResult {
            best: mate,
            playouts: legal.len as u64,
            visits: Vec::new(),
            value: 1.0,
            mates: 1,
            draws: 0,
            time_ms: start.elapsed().as_millis() as u64,
        };
    }
    expand_root(&tree, ROOT_ID, pos, eval_fn, dirichlet);

    let mut playouts = 0u64;
    let mut last_report = start;
    let root_pos = *pos;
    let bs = batch_size.max(1);

    loop {
        if stop.load(Ordering::Relaxed) {
            break;
        }
        if let Some(p) = limits.playouts {
            if playouts >= p {
                break;
            }
        }
        if let Some(t) = limits.movetime {
            if start.elapsed().as_millis() as u64 >= t {
                break;
            }
        }

        // --- Phase 1: Collect leaves via selection ---
        let mut leaves: Vec<BatchLeaf> = Vec::with_capacity(bs);
        for _ in 0..bs {
            if let Some(p) = limits.playouts {
                if playouts + leaves.len() as u64 >= p {
                    break;
                }
            }
            if let Some(t) = limits.movetime {
                if start.elapsed().as_millis() as u64 >= t {
                    break;
                }
            }
            if let Some(leaf) = select_leaf(&tree, ROOT_ID, &root_pos, eval_fn) {
                leaves.push(leaf);
            } else {
                break;
            }
        }

        if leaves.is_empty() {
            break;
        }

        // --- Phase 2: Batch GPU evaluation ---
        let positions: Vec<Position> = leaves.iter().map(|l| l.pos).collect();
        let legal_refs: Vec<&MoveList> = leaves.iter().map(|l| &l.legal).collect();
        let results = batch_eval(&positions, &legal_refs);

        // --- Phase 3: Expand and backup ---
        for (leaf, (cp, logits)) in leaves.iter().zip(results.iter()) {
            let node = tree.node(leaf.node_id);
            if leaf.legal.len == 0 {
                // Terminal leaf: exact outcome for the side to move at the
                // leaf (not the root) + sticky propagation.
                let mate = leaf.pos.in_check();
                mark_terminal(&tree, leaf.node_id, if mate { -1 } else { 0 });
                if mate {
                    tree.mates.fetch_add(1, Ordering::Relaxed);
                    backup(&tree, &leaf.path, -1.0);
                } else {
                    tree.draws.fetch_add(1, Ordering::Relaxed);
                    backup(&tree, &leaf.path, 0.0);
                }
            } else if leaf.pos.halfmove >= 100 {
                mark_terminal(&tree, leaf.node_id, 0);
                tree.draws.fetch_add(1, Ordering::Relaxed);
                backup(&tree, &leaf.path, 0.0);
            } else {
                // `cp` is side-to-move-at-leaf perspective: no depth flip.
                let v = score_from_cp(*cp);
                // Expand children with policy priors (single-threaded: skip
                // if already expanded, e.g. depth-cap revisits).
                if node.n_children.load(Ordering::Acquire) == 0
                    && !node.terminal.load(Ordering::Acquire)
                {
                    let mut priors = [0f32; MAX_MOVES];
                    softmax_priors(logits, leaf.legal.len, &mut priors);
                    let mut first = NO_ID;
                    let mut count = 0u32;
                    for (i, &mv) in leaf.legal.moves[..leaf.legal.len].iter().enumerate() {
                        let id = tree.alloc();
                        if id == NO_ID {
                            break;
                        }
                        let c = tree.node(id);
                        c.mv.store(mv.0, Ordering::Relaxed);
                        c.parent.store(leaf.node_id, Ordering::Relaxed);
                        c.prior.store(priors[i].to_bits(), Ordering::Relaxed);
                        c.next_sibling.store(first, Ordering::Relaxed);
                        first = id;
                        count += 1;
                    }
                    node.first_child.store(first, Ordering::Release);
                    node.n_children.store(count, Ordering::Release);
                    node.complete.store(count as usize == leaf.legal.len, Ordering::Release);
                }
                backup(&tree, &leaf.path, v);
            }
            playouts += 1;
        }

        if on_progress.is_some() && last_report.elapsed().as_millis() >= 16 {
            last_report = Instant::now();
            let best_id = tree.best_child(ROOT_ID);
            // best_id can be NO_ID when the arena is exhausted at the root:
            // never index the tree with it (own safety contract).
            if best_id != NO_ID {
                let value = q_to_prob(tree.node(best_id).q());
                let child_info = tree.child_stats(ROOT_ID, 20);
                if let Some(cb) = on_progress.as_mut() {
                    cb(&MctsProgress {
                        playouts,
                        value,
                        best: tree.node(best_id).mv(),
                        moves: child_info,
                        mates: tree.mates.load(Ordering::Relaxed),
                        draws: tree.draws.load(Ordering::Relaxed),
                        time_ms: start.elapsed().as_millis() as u64,
                    });
                }
            }
        }
    }

    let (best, value) = match tree.best_child_info(ROOT_ID) {
        Some((m, q)) => (m, q_to_prob(q)),
        None => (Move::null(), 0.5),
    };
    let visits = tree.visits_of(ROOT_ID);
    // Never walk into an opponent mate-in-1 (see avoid_opp_mate).
    let best = avoid_opp_mate(pos, best, &visits);
    MctsResult {
        best,
        playouts,
        visits,
        value,
        mates: tree.mates.load(Ordering::Relaxed),
        draws: tree.draws.load(Ordering::Relaxed),
        time_ms: start.elapsed().as_millis() as u64,
    }
}

/// Select one leaf from the tree: descend via PUCT, apply virtual loss,
/// return the leaf position + legal moves + path for backup.
/// Returns None when the tree is fully expanded or terminal.
/// `root` is the search root (kept subtree roots are supported).
fn select_leaf(tree: &Tree, root: u32, root_pos: &Position, _eval_fn: ValuePolicyFn) -> Option<BatchLeaf> {
    let mut pos = *root_pos;
    let mut idx = root;
    let mut path = Vec::new();
    let mut depth = 0u32;

    // Every selection passes through the root exactly once: bump its visits
    // so parent_visits > 0 below. Without this the PUCT exploration term
    // (prior·sqrt(N)/(1+n)) is zero at the root and the search greedily
    // locks onto whatever move got a slightly higher Q first.
    tree.add_visit(root);

    loop {
        let node = tree.node(idx);
        let nc = node.n_children.load(Ordering::Acquire);

        if nc == 0 || node.terminal.load(Ordering::Acquire) || depth >= MAX_DEPTH {
            // This is a leaf. Terminal leaves are returned (with empty legal
            // movelists) so the caller backs up the exact mate/draw value —
            // never NN noise. Mate handling is algorithmic, not learned.
            let legal = generate_legal(&pos);
            return Some(BatchLeaf {
                node_id: idx,
                pos,
                legal,
                path,
            });
        }

        // Select best child via PUCT
        let parent_visits = node.visits();
        let mut best = node.first_child.load(Ordering::Relaxed);
        let mut best_u = f32::NEG_INFINITY;
        let mut c = best;
        while c != NO_ID {
            let u = tree.ucb(c, parent_visits);
            if u > best_u {
                best_u = u;
                best = c;
            }
            c = tree.node(c).next_sibling.load(Ordering::Relaxed);
        }

        // Apply virtual loss
        tree.add_visit(best);
        let mv = tree.node(best).mv();
        pos.make_move(mv);
        path.push(best);
        idx = best;
        depth += 1;
    }
}

/// Backup value along the path from leaf to root, undoing virtual loss
/// and adding the real value. Each node accumulates the outcome for the side
/// that moved into it (its parent's perspective).
fn backup(tree: &Tree, path: &[u32], v_leaf: f32) {
    let mut v_scaled = (-v_leaf * SCALE as f32).round() as i32;
    for &node_id in path.iter().rev() {
        tree.add_scaled(node_id, v_scaled);
        v_scaled = v_scaled.wrapping_neg();
    }
}

// ---------------------------------------------------------------------------
// Parallel batched MCTS — N threads share a tree, batch GPU evaluations.
// ---------------------------------------------------------------------------

use std::sync::Barrier;

/// Parallel batched MCTS: N workers share a tree and batch their leaf
/// evaluations into single GPU calls for maximum throughput.
///
/// Each round:
///   1. All workers select leaves concurrently (virtual loss via visit bumps)
///   2. Barrier → main thread collects all leaves into one batch
///   3. Single GPU evaluation call
///   4. Barrier → all workers expand and backup their leaves
///   5. Repeat
pub fn mcts_batched_parallel(
    pos: &mut Position,
    limits: &MctsLimits,
    stop: &Arc<AtomicBool>,
    eval_fn: ValuePolicyFn,
    batch_eval: BatchEvalFn,
    batch_size: usize,
    dirichlet: bool,
    mut on_progress: Option<&mut dyn FnMut(&MctsProgress)>,
    prev: Option<MctsSearch>,
) -> (MctsResult, MctsSearch) {
    use std::sync::Mutex;
    let start = Instant::now();

    let n_threads = if limits.threads == 0 {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
    } else {
        limits.threads
    };
    let n_threads = n_threads.max(1);
    let bs = batch_size.max(n_threads);

    let legal = generate_legal(pos);
    if legal.len == 0 {
        let value = if pos.in_check() { 0.0 } else { 0.5 };
        return (MctsResult {
            best: Move::null(),
            playouts: 0,
            visits: Vec::new(),
            value,
            mates: 0,
            draws: 0,
            time_ms: start.elapsed().as_millis() as u64,
        }, MctsSearch::new(1024, 1));
    }
    // Mate-in-1 first: play it immediately, no search needed.
    if let Some(mate) = find_mate_in_one(pos) {
        let mut ms = MctsSearch::new(1024, 1);
        ms.root_key = pos.key;
        return (MctsResult {
            best: mate,
            playouts: legal.len as u64,
            visits: Vec::new(),
            value: 1.0,
            mates: 1,
            draws: 0,
            time_ms: start.elapsed().as_millis() as u64,
        }, ms);
    }

    let root_pos = *pos;
    // Tree setup: reuse the kept subtree when provided (all workers share
    // it, like mcts_parallel), otherwise start a fresh tree. The kept
    // subtree is only reused when it belongs to this position.
    let (tree, root) = match prev {
        Some(prev_search) if prev_search.root_key() == pos.key => {
            let r = prev_search.root;
            let t = prev_search.tree.clone();
            // Ensure the kept root is expanded for the current position.
            if t.node(r).n_children.load(Ordering::Acquire) == 0 {
                expand_root(&t, r, &root_pos, eval_fn, false);
            } else if dirichlet {
                renoise_root(&t, r);
            }
            (t, r)
        }
        _ => {
            let t = Arc::new(Tree::new(arena_cap(limits)));
            expand_root(&t, ROOT_ID, &root_pos, eval_fn, dirichlet);
            (t, ROOT_ID)
        }
    };
    let cap = arena_cap(limits);
    // Rebuild when the (possibly reused) arena is nearly full; a fresh
    // tree restarts at ROOT_ID.
    let (tree, root, rebuilt) = {
        if tree.len.load(Ordering::Relaxed) as usize + 4096 > cap {
            let fresh = Arc::new(Tree::new(cap));
            expand_root(&fresh, ROOT_ID, &root_pos, eval_fn, dirichlet);
            (fresh, ROOT_ID, 1u64)
        } else {
            (tree, root, 0u64)
        }
    };
    let counter = Arc::new(AtomicU64::new(0));
    let budget_playouts = limits.playouts;
    let budget_movetime = limits.movetime;
    // Termination is decided ONLY by the main thread (see main loop below).
    // Workers exit solely on `done`, after a final rendezvous — this keeps
    // every barrier paired so stop/budget can never deadlock the join.
    let done = Arc::new(AtomicBool::new(false));

    let batch = Arc::new(Mutex::new(Vec::<BatchLeaf>::with_capacity(bs)));
    // Per-worker chunks of (leaf, eval-result) pairs. Leaves and results
    // travel together in one chunk so a worker can never back a value up
    // to the wrong node (a LIFO-pop/FIFO-drain split previously rotated
    // every round's results across the wrong leaves).
    let batch_out = Arc::new(Mutex::new(Vec::<Vec<(BatchLeaf, (i32, Vec<f32>))>>::new()));

    let barrier_select = Arc::new(Barrier::new(n_threads + 1));
    let barrier_eval = Arc::new(Barrier::new(n_threads + 1));

    let handles: Vec<_> = (0..n_threads)
        .map(|_tid| {
            let tree = tree.clone();
            let stop = stop.clone();
            let done = done.clone();
            let counter = counter.clone();
            let batch = batch.clone();
            let batch_out = batch_out.clone();
            let barrier_select = barrier_select.clone();
            let barrier_eval = barrier_eval.clone();

            std::thread::spawn(move || loop {
                if done.load(Ordering::Relaxed) {
                    break;
                }

                // Phase 1: Select leaves
                let mut my_leaves = Vec::new();
                let target = bs / n_threads;
                for _ in 0..target {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Some(leaf) = select_leaf(&tree, root, &root_pos, eval_fn) {
                        my_leaves.push(leaf);
                    } else {
                        break;
                    }
                }
                {
                    let mut guard = batch.lock().unwrap();
                    guard.extend(my_leaves);
                }

                // Wait for main to collect and evaluate
                barrier_select.wait();
                barrier_eval.wait();

                // Phase 2: Backup my results. The chunk carries each leaf
                // together with its own eval result — pairing is structural.
                let chunk: Vec<(BatchLeaf, (i32, Vec<f32>))> =
                    batch_out.lock().unwrap().pop().unwrap_or_default();

                for (leaf, (cp, logits)) in chunk.iter() {
                    let node = tree.node(leaf.node_id);

                    // Terminal leaves: exact outcome for the side to move,
                    // never NN noise. Mate/stalemate detection is algorithmic.
                    if leaf.legal.len == 0 {
                        // Exact terminal outcome + sticky propagation.
                        let mate = leaf.pos.in_check();
                        mark_terminal(&tree, leaf.node_id, if mate { -1 } else { 0 });
                        if mate {
                            tree.mates.fetch_add(1, Ordering::Relaxed);
                            backup(&tree, &leaf.path, -1.0);
                        } else {
                            tree.draws.fetch_add(1, Ordering::Relaxed);
                            backup(&tree, &leaf.path, 0.0);
                        }
                        counter.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    if leaf.pos.halfmove >= 100 {
                        mark_terminal(&tree, leaf.node_id, 0);
                        tree.draws.fetch_add(1, Ordering::Relaxed);
                        backup(&tree, &leaf.path, 0.0);
                        counter.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }

                    // `cp` is side-to-move-at-leaf perspective: backup takes
                    // it as-is (a depth-parity flip here inverted odd-depth
                    // values — same convention as expand()/playout_from).
                    let v = score_from_cp(*cp);

                    // Expand children, exactly once: skip if another worker
                    // already expanded this node (mirror expand()'s spinlock).
                    let claimed = node.n_children.load(Ordering::Acquire) == 0
                        && !node.terminal.load(Ordering::Acquire)
                        && node.lock.compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed).is_ok();
                    if claimed {
                        if node.n_children.load(Ordering::Acquire) == 0
                            && !node.terminal.load(Ordering::Acquire)
                        {
                            let mut priors = [0f32; MAX_MOVES];
                            softmax_priors(logits, leaf.legal.len, &mut priors);
                            let mut first = NO_ID;
                            let mut count = 0u32;
                            for (i, &mv) in leaf.legal.moves[..leaf.legal.len].iter().enumerate() {
                                let id = tree.alloc();
                                if id == NO_ID { break; }
                                let c = tree.node(id);
                                c.mv.store(mv.0, Ordering::Relaxed);
                                c.parent.store(leaf.node_id, Ordering::Relaxed);
                                c.prior.store(priors[i].to_bits(), Ordering::Relaxed);
                                c.next_sibling.store(first, Ordering::Relaxed);
                                first = id;
                                count += 1;
                            }
                            node.first_child.store(first, Ordering::Release);
                            node.n_children.store(count, Ordering::Release);
                            node.complete.store(count as usize == leaf.legal.len, Ordering::Release);
                        }
                        node.lock.store(false, Ordering::Release);
                    }
                    backup(&tree, &leaf.path, v);
                    counter.fetch_add(1, Ordering::Relaxed);
                }

                // Wait for all workers to finish backup
                barrier_select.wait();
            })
        })
        .collect();

    // Main thread: coordinate batch evaluation.
    // Termination is decided ONLY here, right after the select rendezvous,
    // and workers are always released through the remaining two rendezvous
    // before we break. Workers exit solely on `done`, so every barrier
    // stays paired: stop/budget can never deadlock the final join, even
    // when set before the first round.
    // Split items into per-worker chunks, preserving order. Workers pop
    // whole chunks, so each leaf stays paired with its own eval result.
    let release_empty = |batch_out: &Arc<Mutex<Vec<Vec<(BatchLeaf, (i32, Vec<f32>))>>>>| {
        let mut b_guard = batch_out.lock().unwrap();
        b_guard.clear();
        b_guard.extend(split_chunks(Vec::new(), n_threads));
    };
    loop {
        // Rendezvous 1: workers finished selection.
        barrier_select.wait();

        let over = stop.load(Ordering::Relaxed)
            || budget_playouts.map(|p| counter.load(Ordering::Relaxed) >= p).unwrap_or(false)
            || budget_movetime.map(|t| start.elapsed().as_millis() as u64 >= t).unwrap_or(false);
        if over {
            // Drain leftovers, release workers through the remaining
            // rendezvous, then exit. Workers see `done` at their loop top.
            // NOTE: leaves selected this round keep their select-time visit
            // bumps without backup (virtual loss never refunded). At most
            // one round's worth per search; negligible next to thousands of
            // backed-up playouts, and it decays in meaning across reuses.
            batch.lock().unwrap().clear();
            release_empty(&batch_out);
            done.store(true, Ordering::Relaxed);
            barrier_eval.wait();
            barrier_select.wait();
            break;
        }

        // Collect batch
        let leaves: Vec<BatchLeaf> = {
            let mut guard = batch.lock().unwrap();
            guard.drain(..).collect()
        };

        if leaves.is_empty() {
            // Nothing to evaluate (all selections failed); release workers
            // for another round instead of breaking the protocol.
            release_empty(&batch_out);
            barrier_eval.wait();
            barrier_select.wait();
            continue;
        }

        // Batch GPU evaluation
        let positions: Vec<Position> = leaves.iter().map(|l| l.pos).collect();
        let legal_refs: Vec<&MoveList> = leaves.iter().map(|l| &l.legal).collect();
        let eval_results = batch_eval(&positions, &legal_refs);
        debug_assert_eq!(leaves.len(), eval_results.len());

        // Pair each leaf with its own result, then split into per-worker
        // chunks (order-preserving). Workers pop whole chunks.
        let paired: Vec<(BatchLeaf, (i32, Vec<f32>))> =
            leaves.into_iter().zip(eval_results).collect();
        {
            let mut b_guard = batch_out.lock().unwrap();
            b_guard.clear();
            b_guard.extend(split_chunks(paired, n_threads));
        }

        // Release workers to backup
        barrier_eval.wait();

        // Wait for all workers to finish backup
        barrier_select.wait();

        if on_progress.is_some() {
            let elapsed = start.elapsed().as_millis() as u64;
            if elapsed > 0 && (elapsed < 1000 || elapsed % 500 < 16) {
                let best_id = tree.best_child(root);
                // NO_ID guard (arena exhaustion): never index with it.
                if best_id != NO_ID {
                    let value = q_to_prob(tree.node(best_id).q());
                    let child_info = tree.child_stats(root, 20);
                    if let Some(cb) = on_progress.as_mut() {
                        cb(&MctsProgress {
                            playouts: counter.load(Ordering::Relaxed),
                            value,
                            best: tree.node(best_id).mv(),
                            moves: child_info,
                            mates: tree.mates.load(Ordering::Relaxed),
                            draws: tree.draws.load(Ordering::Relaxed),
                            time_ms: elapsed,
                        });
                    }
                }
            }
        }
    }

    for h in handles {
        let _ = h.join();
    }

    let (best, value) = match tree.best_child_info(root) {
        Some((m, q)) => (m, q_to_prob(q)),
        None => (Move::null(), 0.5),
    };
    let visits = tree.visits_of(root);
    // Never walk into an opponent mate-in-1 (see avoid_opp_mate).
    let best = avoid_opp_mate(pos, best, &visits);
    let result = MctsResult {
        best,
        playouts: counter.load(Ordering::Relaxed),
        visits,
        value,
        mates: tree.mates.load(Ordering::Relaxed),
        draws: tree.draws.load(Ordering::Relaxed),
        time_ms: start.elapsed().as_millis() as u64,
    };
    let search = MctsSearch {
        tree,
        root,
        threads: n_threads,
        cap,
        rebuilds: rebuilt,
        root_key: root_pos.key,
    };
    (result, search)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate;

    #[test]
    fn split_chunks_preserves_order() {
        let v: Vec<(u32, char)> = (0..10).map(|i| (i, (b'a' + i as u8) as char)).collect();
        let chunks = split_chunks(v, 4);
        assert_eq!(chunks.len(), 4);
        // Flattened chunks must reproduce the original order ...
        let flat: Vec<u32> = chunks.iter().flatten().map(|(i, _)| *i).collect();
        assert_eq!(flat, (0..10).collect::<Vec<_>>());
        // ... so popping whole chunks from the end keeps every pair intact.
        let mut c = chunks;
        let last = c.pop().unwrap();
        assert!(last.windows(2).all(|w| w[0].0 + 1 == w[1].0));
        assert_eq!(last.last().unwrap().0, 9);
    }
    fn cb(pos: &Position, _legal: &MoveList) -> (i32, Vec<f32>) {
        (evaluate::evaluate(pos), Vec::new())
    }

    /// Serial (threads=1): the kept subtree must keep the old children visits.
    #[test]
    fn reuse_serial_keeps_subtree() {
        crate::init();
        let mut s = MctsSearch::new(1 << 20, 1);
        let stop = Arc::new(AtomicBool::new(false));
        let mut pos = Position::startpos();
        let r1 = s.search(&pos, 4000, cb, stop.clone(), false, 0.0);
        let mv = r1.best;
        pos.make_move(mv);
        s.keep_child(mv, pos.key);
        let r2 = s.search(&pos, 4000, cb, stop.clone(), false, 0.0);
        let sum2: u32 = r2.visits.iter().map(|(_, v)| v).sum();
        assert!(sum2 > 4000, "kept subtree visits must carry over, got {sum2}");
        assert_eq!(s.rebuilds(), 0);
    }

    /// root_key tracks the search position: unknown when fresh, set by
    /// search, advanced by keep_child, cleared by reset. Callers use it to
    /// drop stale trees instead of running tree moves on a wrong position.
    #[test]
    fn root_key_tracks_position() {
        crate::init();
        let mut s = MctsSearch::new(65536, 1);
        assert_eq!(s.root_key(), u64::MAX);
        let stop = Arc::new(AtomicBool::new(false));
        let mut pos = Position::startpos();
        let r = s.search(&pos, 200, cb, stop.clone(), false, 0.0);
        assert_eq!(s.root_key(), pos.key);
        let mv = r.best;
        pos.make_move(mv);
        s.keep_child(mv, pos.key);
        assert_eq!(s.root_key(), pos.key);
        s.reset();
        assert_eq!(s.root_key(), u64::MAX);
    }

    /// pv_value reports the PV-leaf value in root frame: the mating move
    /// scores ~+1, and unknown moves give None. Uses a hand-built proven
    /// tree (mate-in-1 now returns instantly with no search tree).
    #[test]
    fn pv_value_reports_mate_in_root_frame() {
        crate::init();
        let tree = Tree::new(1024);
        let c = tree.alloc();
        let mate = Move::new(8, 16, 0, 0);
        tree.node(c).mv.store(mate.0, Ordering::Relaxed);
        tree.node(c).parent.store(ROOT_ID, Ordering::Relaxed);
        tree.node(c).next_sibling.store(NO_ID, Ordering::Relaxed);
        tree.node(ROOT_ID).first_child.store(c, Ordering::Release);
        tree.node(ROOT_ID).n_children.store(1, Ordering::Release);
        mark_terminal(&tree, c, -1);
        let s = MctsSearch {
            tree: std::sync::Arc::new(tree),
            root: ROOT_ID,
            threads: 1,
            cap: 1024,
            rebuilds: 0,
            root_key: u64::MAX,
        };
        let v = s.pv_value(mate).expect("mate must be a root child");
        assert!(v > 0.99, "PV-leaf value of mate must be ~+1, got {v}");
        assert!(s.pv_value(crate::move_::Move::null()).is_none());
    }

    /// Sticky endgames: a proven-mate child flips the parent immediately
    /// (Lc0-style), without waiting for thousands of visits to average out.    /// Conventions: proven flags are side-to-move frame; q() reports exact
    /// parent-perspective values (-S) once proven.
    #[test]
    fn sticky_propagates_proven_results() {
        crate::init();

        // Hand-built tree: root with two fully-expanded children.
        fn two_child_tree() -> Tree {
            let tree = Tree::new(1024);
            let c1 = tree.alloc();
            let c2 = tree.alloc();
            for (c, from, to) in [(c1, 8usize, 16usize), (c2, 8usize, 24usize)] {
                tree.node(c).mv.store(Move::new(from, to, 0, 0).0, Ordering::Relaxed);
                tree.node(c).parent.store(ROOT_ID, Ordering::Relaxed);
            }
            tree.node(c1).next_sibling.store(c2, Ordering::Relaxed);
            tree.node(c2).next_sibling.store(NO_ID, Ordering::Relaxed);
            tree.node(ROOT_ID).first_child.store(c1, Ordering::Release);
            tree.node(ROOT_ID).n_children.store(2, Ordering::Release);
            tree.node(ROOT_ID).complete.store(true, Ordering::Release);
            tree
        }

        // One mated child => parent proven won (mate found, shown exact).
        let tree = two_child_tree();
        mark_terminal(&tree, tree.node(ROOT_ID).first_child.load(Ordering::Relaxed), -1);
        assert_eq!(tree.node(ROOT_ID).proven.load(Ordering::Relaxed), PROVEN_WIN);
        assert_eq!(tree.node(ROOT_ID).q(), -1.0);

        // All children won-for-their-side => parent proven lost.
        let tree = two_child_tree();
        let c1 = tree.node(ROOT_ID).first_child.load(Ordering::Relaxed);
        let c2 = tree.node(c1).next_sibling.load(Ordering::Relaxed);
        mark_terminal(&tree, c1, 1);
        mark_terminal(&tree, c2, 1);
        assert_eq!(tree.node(ROOT_ID).proven.load(Ordering::Relaxed), PROVEN_LOSS);
        assert_eq!(tree.node(ROOT_ID).q(), 1.0);

        // Won + drawn children => parent proven drawn.
        let tree = two_child_tree();
        let c1 = tree.node(ROOT_ID).first_child.load(Ordering::Relaxed);
        let c2 = tree.node(c1).next_sibling.load(Ordering::Relaxed);
        mark_terminal(&tree, c1, 1);
        mark_terminal(&tree, c2, 0);
        assert_eq!(tree.node(ROOT_ID).proven.load(Ordering::Relaxed), PROVEN_DRAW);
        assert_eq!(tree.node(ROOT_ID).q(), 0.0);
    }

    /// Serial: mid-depth nodes must have expanded children with visits.    #[test]
    fn serial_descends_below_root() {
        crate::init();
        let mut s = MctsSearch::new(65536, 1);
        let stop = Arc::new(AtomicBool::new(false));
        let pos = Position::startpos();
        s.search(&pos, 4000, cb, stop.clone(), false, 0.0);
        let root = s.tree.node(ROOT_ID);
        let mut total_grand_visits = 0u32;
        let mut c = root.first_child.load(Ordering::Relaxed);
        while c != NO_ID {
            let n = s.tree.node(c);
            if n.visits() > 10 {
                let mut cc = n.first_child.load(Ordering::Relaxed);
                while cc != NO_ID {
                    total_grand_visits += s.tree.node(cc).visits();
                    cc = s.tree.node(cc).next_sibling.load(Ordering::Relaxed);
                }
            }
            c = n.next_sibling.load(Ordering::Relaxed);
        }
        assert!(total_grand_visits > 3000, "playouts must descend past depth 1, got {total_grand_visits}");
    }
}

