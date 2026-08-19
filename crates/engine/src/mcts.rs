//! Monte-Carlo Tree Search (UCT) with a value function for leaf evaluation.
//!
//! The value function maps a position to centipawns for the side to move
//! (PeSTO or the NN); the leaf value is `2*sigmoid(cp/400) - 1`, a zero-sum
//! score in [-1, 1]. There is no policy head yet, so the prior is uniform over
//! legal moves, matching the Python self-play trainer.
//!
//! Nodes live in a preallocated arena (u32 indices) so the tree can later be
//! shared across threads with atomic virtual-loss counters.

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::move_::Move;
use crate::movegen::generate_legal;
use crate::position::Position;
use crate::search::EvalFn;

/// Value function: centipawns from the side-to-move perspective (same contract
/// as the alpha-beta `EvalFn`).
pub type ValueFn = EvalFn;

pub struct MctsLimits {
    pub playouts: Option<u64>,
    pub movetime: Option<u64>,
}

pub struct MctsProgress {
    pub playouts: u64,
    /// Win probability for the side to move, from the most-visited child.
    pub value: f32,
    pub best: Move,
    /// Top moves by visits (at most 5), most-visited first.
    pub moves: Vec<(Move, u32)>,
    pub time_ms: u64,
}

pub struct MctsResult {
    pub best: Move,
    pub playouts: u64,
    /// Visit counts for each root move, most-visited first.
    pub visits: Vec<(Move, u32)>,
    /// Win probability for the side to move (from the most-visited child).
    pub value: f32,
    pub time_ms: u64,
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
fn q_to_prob(q: f32) -> f32 {
    (q + 1.0) * 0.5
}

const ROOT_ID: u32 = 0;
const NO_ID: u32 = u32::MAX;
const C_PUCT: f32 = 1.4;
/// Selection never descends deeper than this; such nodes are evaluated as leaves.
const MAX_DEPTH: u32 = 96;

#[derive(Clone, Copy)]
struct Node {
    mv: Move,
    parent: u32,
    first_child: u32,
    next_sibling: u32,
    n_children: u16,
    terminal: bool,
    visits: u32,
    /// Accumulated backed-up values (win prob for the node's side to move).
    value: f32,
    prior: f32,
}

impl Default for Node {
    fn default() -> Self {
        Node {
            mv: Move::null(),
            parent: NO_ID,
            first_child: NO_ID,
            next_sibling: NO_ID,
            n_children: 0,
            terminal: false,
            visits: 0,
            value: 0.0,
            prior: 0.0,
        }
    }
}

impl Node {
    #[inline]
    fn q(&self) -> f32 {
        self.value / self.visits.max(1) as f32
    }
}

struct Tree {
    nodes: Vec<Node>,
}

impl Tree {
    fn new(cap: usize) -> Self {
        Tree { nodes: Vec::with_capacity(cap) }
    }

    #[inline]
    fn alloc(&mut self) -> u32 {
        if self.nodes.len() == self.nodes.capacity() {
            self.nodes.reserve(self.nodes.capacity().max(1 << 18));
        }
        let id = self.nodes.len() as u32;
        self.nodes.push(Node::default());
        id
    }

    fn new_root(&mut self) -> u32 {
        let id = self.alloc();
        debug_assert_eq!(id, ROOT_ID);
        id
    }

    fn new_child(&mut self, parent: u32, mv: Move, prior: f32) -> u32 {
        let id = self.alloc();
        self.nodes[id as usize].mv = mv;
        self.nodes[id as usize].parent = parent;
        self.nodes[id as usize].prior = prior;
        let old = self.nodes[parent as usize].first_child;
        self.nodes[id as usize].next_sibling = old;
        let p = &mut self.nodes[parent as usize];
        p.n_children += 1;
        p.first_child = id;
        id
    }

    #[inline]
    fn ucb(&self, node: u32, parent_visits: u32) -> f32 {
        let n = &self.nodes[node as usize];
        if n.visits == 0 {
            return f32::INFINITY;
        }
        n.q() + C_PUCT * n.prior * (parent_visits as f32).sqrt() / (1.0 + n.visits as f32)
    }

    fn best_child(&self, node: u32) -> u32 {
        let mut best = self.nodes[node as usize].first_child;
        let mut best_visits = u32::MIN;
        let mut c = best;
        while c != NO_ID {
            let v = self.nodes[c as usize].visits;
            if v > best_visits {
                best_visits = v;
                best = c;
            }
            c = self.nodes[c as usize].next_sibling;
        }
        best
    }

    fn visits_of(&self, node: u32) -> Vec<(Move, u32)> {
        let mut out = Vec::new();
        let mut c = self.nodes[node as usize].first_child;
        while c != NO_ID {
            out.push((self.nodes[c as usize].mv, self.nodes[c as usize].visits));
            c = self.nodes[c as usize].next_sibling;
        }
        out.sort_by_key(|b| std::cmp::Reverse(b.1));
        out
    }
}

/// One selection -> expansion/evaluation -> backup cycle. Returns the value
/// backed up to the root (win prob for the root's side to move). `pos` is
/// mutated as the tree is descended; callers must pass a scratch copy.
fn playout(pos: &mut Position, tree: &mut Tree, root: u32, value_fn: ValueFn) -> f32 {
    let mut idx = root;
    let mut parent_visits = tree.nodes[root as usize].visits;
    let mut ply = 0u32;

    // Selection: descend while the current node is expanded and not at the
    // depth cap.
    loop {
        let node = &tree.nodes[idx as usize];
        if node.n_children == 0 || node.terminal || ply >= MAX_DEPTH {
            break;
        }
        let mut best = node.first_child;
        let mut best_u = f32::NEG_INFINITY;
        let mut c = best;
        while c != NO_ID {
            let u = tree.ucb(c, parent_visits);
            if u > best_u {
                best_u = u;
                best = c;
            }
            c = tree.nodes[c as usize].next_sibling;
        }
        pos.make_move(tree.nodes[best as usize].mv);
        idx = best;
        parent_visits = tree.nodes[idx as usize].visits;
        ply += 1;
    }

    // Leaf: fixed terminal result, depth-cap evaluation, or expand + evaluate.
    // `v_leaf` is the zero-sum outcome for the side to move at the leaf.
    let node = &tree.nodes[idx as usize];
    let v_leaf = if node.terminal {
        if pos.in_check() { -1.0 } else { 0.0 }
    } else if node.n_children > 0 {
        // Internal node reached through the depth cap: evaluate as a leaf.
        score_from_cp(value_fn(pos))
    } else {
        let legal = generate_legal(pos);
        if legal.len == 0 {
            let v = if pos.in_check() { -1.0 } else { 0.0 };
            tree.nodes[idx as usize].terminal = true;
            v
        } else {
            let v = score_from_cp(value_fn(pos));
            let prior = 1.0 / legal.len as f32;
            for i in 0..legal.len {
                tree.new_child(idx, legal.moves[i], prior);
            }
            v
        }
    };

    // Backup: each node accumulates the outcome for the side that moved into
    // it (i.e. its parent's perspective), so sibling values are comparable.
    let mut n = idx;
    let mut v = -v_leaf;
    while n != NO_ID {
        let node = &mut tree.nodes[n as usize];
        node.visits += 1;
        node.value += v;
        v = -v;
        n = node.parent;
    }
    -v
}

/// Run MCTS from the given position. `value_fn(pos)` must return centipawns
/// from the side-to-move perspective. Returns the best root move and the visit
/// distribution.
pub fn mcts_root(
    pos: &mut Position,
    limits: &MctsLimits,
    stop: &AtomicBool,
    value_fn: ValueFn,
    mut on_progress: Option<&mut dyn FnMut(&MctsProgress)>,
) -> MctsResult {
    let start = Instant::now();
    let mut tree = Tree::new(1 << 18);
    let root = tree.new_root();

    let legal = generate_legal(pos);
    if legal.len == 0 {
        let value = if pos.in_check() { 0.0 } else { 0.5 };
        return MctsResult {
            best: Move::null(),
            playouts: 0,
            visits: Vec::new(),
            value,
            time_ms: start.elapsed().as_millis() as u64,
        };
    }
    let prior = 1.0 / legal.len as f32;
    for i in 0..legal.len {
        tree.new_child(root, legal.moves[i], prior);
    }

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
        playout(&mut p, &mut tree, root, value_fn);
        playouts += 1;

        if on_progress.is_some() && last_report.elapsed().as_millis() >= 16 {
            last_report = Instant::now();
            let best_id = tree.best_child(root);
            let value = q_to_prob(tree.nodes[best_id as usize].q());
            let moves = tree.visits_of(root);
            if let Some(cb) = on_progress.as_mut() {
                cb(&MctsProgress {
                    playouts,
                    value,
                    best: tree.nodes[best_id as usize].mv,
                    moves: moves.into_iter().take(5).collect(),
                    time_ms: start.elapsed().as_millis() as u64,
                });
            }
        }
    }

    let best_id = tree.best_child(root);
    let value = q_to_prob(tree.nodes[best_id as usize].q());
    MctsResult {
        best: tree.nodes[best_id as usize].mv,
        playouts,
        visits: tree.visits_of(root),
        value,
        time_ms: start.elapsed().as_millis() as u64,
    }
}
