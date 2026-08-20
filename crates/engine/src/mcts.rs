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

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::move_::Move;
use crate::movegen::{generate_legal, MoveList, MAX_MOVES};
use crate::position::Position;
use crate::search::EvalFn;

/// Value function: centipawns from the side-to-move perspective (same contract
/// as the alpha-beta `EvalFn`).
pub type ValueFn = EvalFn;

/// Combined value + move policy: centipawns for the side to move and raw
/// policy logits, one per move in `legal` (same order; empty when no policy is
/// available -> uniform priors). The eval function computes both in a single
/// forward. Passing an empty `legal` requests the value only.
pub type ValuePolicyFn = fn(&Position, &MoveList) -> (i32, Vec<f32>);

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
    /// Top moves by visits (at most 5), most-visited first.
    pub moves: Vec<(Move, u32)>,
    /// Playouts that ended in a checkmate.
    pub mates: u64,
    /// Playouts that ended in a draw (stalemate or 50-move rule).
    pub draws: u64,
    pub time_ms: u64,
}

pub struct MctsResult {
    pub best: Move,
    pub playouts: u64,
    /// Visit counts for each root move, most-visited first.
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
/// The draw probability peaks near equality and vanishes at decisive scores.
pub fn wdl_from_q(q: f32) -> (f32, f32, f32) {
    let win_raw = (q + 1.0) / 2.0;
    let draw = 0.6 * (1.0 - q.abs());
    (win_raw * (1.0 - draw), draw, (1.0 - win_raw) * (1.0 - draw))
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

/// Inverse of `score_from_cp`, for display.
#[inline]
pub fn cp_from_prob(p: f32) -> i32 {
    (400.0 * (p / (1.0 - p).max(1e-6)).ln()).round() as i32
}

/// Zero-sum score to win probability for the side to move.
#[inline]
pub fn q_to_prob(q: f32) -> f32 {
    (q + 1.0) * 0.5
}

const ROOT_ID: u32 = 0;
const NO_ID: u32 = u32::MAX;
const C_PUCT: f32 = 1.4;
/// Selection never descends deeper than this; such nodes are evaluated as leaves.
const MAX_DEPTH: u32 = 96;
/// Fixed-point scale for backed-up values (q in [-1, 1] -> [-SCALE, SCALE]).
const SCALE: i32 = 128;

/// A node of the search tree. All mutable state is atomic so the tree is shared
/// across worker threads without locks on the hot path. `val_visits` packs
/// `value` (i32, high 32 bits) and `visits` (u32, low 32 bits) so a backup is a
/// single read-modify-write.
struct Node {
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
        let packed = self.val_visits.load(Ordering::Relaxed);
        let visits = (packed & 0xFFFF_FFFF) as u32;
        let value = ((packed >> 32) as u32) as i32;
        (value as f32 / SCALE as f32) / visits.max(1) as f32
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

/// Creates the root's children (single-threaded, before any worker starts).
/// `root` is the node whose children are created (ROOT_ID for a fresh tree, or
/// a reused subtree root).
fn expand_root(tree: &Tree, root: u32, pos: &Position, eval_fn: ValuePolicyFn) {
    let legal = generate_legal(pos);
    let (_, logits) = eval_fn(pos, &legal);
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
        c.parent.store(root, Ordering::Relaxed);
        c.prior.store(priors[i].to_bits(), Ordering::Relaxed);
        c.next_sibling.store(first, Ordering::Relaxed);
        first = id;
        count += 1;
    }
    tree.node(root).first_child.store(first, Ordering::Release);
    tree.node(root).n_children.store(count, Ordering::Release);
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
            node.terminal.store(true, Ordering::Release);
            if pos.in_check() { -1.0 } else { 0.0 }
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
        if pos.in_check() {
            tree.mates.fetch_add(1, Ordering::Relaxed);
            -1.0
        } else {
            tree.draws.fetch_add(1, Ordering::Relaxed);
            0.0
        }
    } else if pos.halfmove >= 100 {
        // 50-move rule: the position is drawn.
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
        Some(p) => (p as usize * 24 + 4096).min(1 << 23),
        None => 1 << 22,
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
    expand_root(&tree, ROOT_ID, pos, eval_fn);

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
            let value = q_to_prob(tree.node(best_id).q());
            let moves = tree.visits_of(ROOT_ID);
            if let Some(cb) = on_progress.as_mut() {
cb(&MctsProgress {
                    playouts,
                    value,
                    best: tree.node(best_id).mv(),
                    moves: moves.into_iter().take(5).collect(),
                    mates: tree.mates.load(Ordering::Relaxed),
                    draws: tree.draws.load(Ordering::Relaxed),
                    time_ms: start.elapsed().as_millis() as u64,
                });
            }
        }
    }

    let best_id = tree.best_child(ROOT_ID);
    let value = q_to_prob(tree.node(best_id).q());
MctsResult {
        best: tree.node(best_id).mv(),
        playouts,
        visits: tree.visits_of(ROOT_ID),
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
) -> MctsResult {
    let start = Instant::now();

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
    let mut handles = Vec::with_capacity(threads);
    for _ in 0..threads {
        let tree = Arc::new(Tree::new(per_thread_cap(limits, threads)));
        expand_root(&tree, ROOT_ID, &root_pos, eval_fn);
        trees.push(tree.clone());
        let stop = stop.clone();
let counter = counter.clone();
        let active = active.clone();
        handles.push(std::thread::spawn(move || {
            active.fetch_add(1, Ordering::Relaxed);
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
                playout(&tree, &mut p, eval_fn);
                counter.fetch_add(1, Ordering::Relaxed);
            }
            active.fetch_sub(1, Ordering::Relaxed);
        }));
    }

    if let Some(cb) = on_progress.as_mut() {
        while active.load(Ordering::Relaxed) > 0 {
            std::thread::sleep(Duration::from_millis(16));
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let (best, value, moves) = merge_trees(&trees);
cb(&MctsProgress {
                playouts: counter.load(Ordering::Relaxed),
                value,
                best,
                moves: moves.into_iter().take(5).collect(),
                mates: trees.iter().map(|t| t.mates.load(Ordering::Relaxed)).sum(),
                draws: trees.iter().map(|t| t.draws.load(Ordering::Relaxed)).sum(),
                time_ms: start.elapsed().as_millis() as u64,
            });
        }
    }

    for h in handles {
        let _ = h.join();
    }

    let (best, value, visits) = merge_trees(&trees);
MctsResult {
        best,
        playouts: counter.load(Ordering::Relaxed),
        visits,
        value,
        mates: trees.iter().map(|t| t.mates.load(Ordering::Relaxed)).sum(),
        draws: trees.iter().map(|t| t.draws.load(Ordering::Relaxed)).sum(),
        time_ms: start.elapsed().as_millis() as u64,
    }
}

/// Arena size for one worker of a `threads`-way forest search: each tree only
/// needs its share of the playout budget (the tree grows ~24 nodes/playout),
/// bounded to keep total memory sane.
fn per_thread_cap(limits: &MctsLimits, threads: usize) -> usize {
    match limits.playouts {
        Some(p) => {
            let share = p.div_ceil(threads as u64);
            ((share as usize) * 24 + 4096).min(1 << 17)
        }
        None => 1 << 17,
    }
}

/// Merges the root visit counts of a forest into a single (best, value,
/// sorted-visits) result. `value` is the visit-weighted q of the best move.
fn merge_trees(trees: &[Arc<Tree>]) -> (Move, f32, Vec<(Move, u32)>) {
    use std::collections::HashMap;
    let mut visits: HashMap<Move, u32> = HashMap::new();
    let mut q_sum: HashMap<Move, f64> = HashMap::new();
    for tree in trees {
        for (m, v) in tree.visits_of(ROOT_ID) {
            if v == 0 {
                continue;
            }
            *visits.entry(m).or_insert(0) += v;
            if let Some(q) = tree.q_of_child(ROOT_ID, m) {
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
        }
    }

    /// Number of times the arena was full and the tree had to be rebuilt.
    pub fn rebuilds(&self) -> u64 {
        self.rebuilds
    }

    /// Starts a fresh tree (new game).
    pub fn reset(&mut self) {
        self.tree = Arc::new(Tree::new(self.cap));
        self.root = ROOT_ID;
    }

    /// Runs up to `playouts` playouts (or until `stop`) from the current root,
    /// which must correspond to `pos`. Terminal positions return immediately.
    pub fn search(
        &mut self,
        pos: &Position,
        playouts: u64,
        eval_fn: ValuePolicyFn,
        stop: Arc<AtomicBool>,
    ) -> MctsResult {
        let start = Instant::now();
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

        // Rebuild when the arena is nearly full (rare with a generous cap).
        if self.tree.len.load(Ordering::Relaxed) as usize + 4096 > self.cap {
            self.tree = Arc::new(Tree::new(self.cap));
            self.root = ROOT_ID;
            self.rebuilds += 1;
        }

        // Make sure the root's children exist for this position.
        if self.tree.node(self.root).n_children.load(Ordering::Acquire) == 0 {
            expand_root(&self.tree, self.root, pos, eval_fn);
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

        let best_id = tree.best_child(root);
        let best = tree.node(best_id).mv();
        let value = q_to_prob(tree.node(best_id).q());
        MctsResult {
            best,
            playouts: counter.load(Ordering::Relaxed),
            visits: tree.visits_of(root),
            value,
            mates: tree.mates.load(Ordering::Relaxed),
            draws: tree.draws.load(Ordering::Relaxed),
            time_ms: start.elapsed().as_millis() as u64,
        }
    }

    /// Re-points the search root at the child matching `mv` (the played move),
    /// keeping its subtree for the next search. The caller must then make the
    /// move so `pos` matches the new root. Falls back to a fresh tree when the
    /// move is not found (should not happen).
    pub fn keep_child(&mut self, mv: Move) {
        let mut c = self.tree.node(self.root).first_child.load(Ordering::Relaxed);
        while c != NO_ID {
            let n = self.tree.node(c);
            if n.mv() == mv {
                self.root = c;
                return;
            }
            c = n.next_sibling.load(Ordering::Relaxed);
        }
        self.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evaluate;

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
        let r1 = s.search(&pos, 4000, cb, stop.clone());
        let mv = r1.best;
        s.keep_child(mv);
        pos.make_move(mv);
        let r2 = s.search(&pos, 4000, cb, stop.clone());
        let sum2: u32 = r2.visits.iter().map(|(_, v)| v).sum();
        assert!(sum2 > 4000, "kept subtree visits must carry over, got {sum2}");
        assert_eq!(s.rebuilds(), 0);
    }

    /// Serial: mid-depth nodes must have expanded children with visits.
    #[test]
    fn serial_descends_below_root() {
        crate::init();
        let mut s = MctsSearch::new(65536, 1);
        let stop = Arc::new(AtomicBool::new(false));
        let pos = Position::startpos();
        s.search(&pos, 4000, cb, stop.clone());
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

