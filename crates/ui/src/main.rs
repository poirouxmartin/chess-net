//! chess-net GUI: dark theme, analysis-only, Nibbler-style variant panel.
//! Continuous MCTS analysis with persistent tree, full PVs, WDL bar.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use eframe::egui::{
    self, Align2, Color32, FontId, Pos2, Rect, RichText, Sense, Stroke, Vec2,
};

use engine::evaluate::{evaluate, MATE};
use engine::mcts::{cp_from_prob, mcts_batched_parallel, q_to_prob, wdl_from_q, MctsLimits, MctsChildInfo, MctsSearch};
use engine::move_::Move;
use engine::movegen::{generate_legal, MoveList};
use engine::position::{Position, WHITE};
use engine::san::to_san;
use engine::search::{Limits, SearchIter, Searcher};

const CONFIG_FILE: &str = "chess-net.conf";

fn config_path() -> std::path::PathBuf {
    std::env::current_dir().unwrap_or_default().join(CONFIG_FILE)
}
fn load_config() -> String {
    std::fs::read_to_string(config_path()).unwrap_or_default().trim().to_owned()
}
fn save_config(nn_file: &str) {
    let _ = std::fs::write(config_path(), nn_file);
}
/// Rank a candidate net file: highest training cycle first, then FP32 over
/// FP16 (no speed gain, lower precision), then onnx over csnn.
fn net_rank(p: &std::path::Path) -> (u64, u8, u8) {
    let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("").to_lowercase();
    let cycle = name.find("cycle").and_then(|i| {
        name[i + 5..].chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse::<u64>().ok()
    }).unwrap_or(0);
    let fp16 = if name.contains("fp16") { 1 } else { 0 };
    let ext = match p.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "onnx" => 0,
        _ => 1,
    };
    (cycle, fp16, ext)
}

/// All discoverable net files, best first (see `net_rank`).
fn find_all_nets(dir: &std::path::Path) -> Vec<String> {
    // Search roots: working dir + exe dir + their ancestors (covers
    // launching from target/release), each with and without checkpoints/.
    let mut roots: Vec<std::path::PathBuf> = vec![dir.to_path_buf()];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            roots.push(exe_dir.to_path_buf());
        }
    }
    let mut search_dirs: Vec<std::path::PathBuf> = Vec::new();
    for root in &roots {
        let mut cur = root.clone();
        for _ in 0..6 {
            search_dirs.push(cur.clone());
            search_dirs.push(cur.join("checkpoints"));
            match cur.parent() {
                Some(p) => cur = p.to_path_buf(),
                None => break,
            }
        }
    }
    search_dirs.sort();
    search_dirs.dedup();
    let mut candidates = Vec::new();
    for search_dir in &search_dirs {
        if let Ok(entries) = std::fs::read_dir(search_dir) {
            for entry in entries.filter_map(|e| e.ok()) {
                let p = entry.path();
                if let Some(ext) = p.extension().and_then(|e| e.to_str()) {
                    if ext == "onnx" || ext == "csnn" {
                        candidates.push(p);
                    }
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        let (ra, rb) = (net_rank(a), net_rank(b));
        // cycle desc, then fp16-last, then onnx-first
        rb.0.cmp(&ra.0).then(ra.1.cmp(&rb.1)).then(ra.2.cmp(&rb.2))
    });
    candidates.into_iter().map(|p| p.to_string_lossy().to_string()).collect()
}

fn find_onnx_in_dir(dir: &std::path::Path) -> Option<String> {
    find_all_nets(dir).into_iter().next()
}

const TEXT_DIM: Color32 = Color32::from_rgb(140, 140, 150);
const TEXT_LIGHT: Color32 = Color32::from_rgb(210, 210, 215);
const ACCENT: Color32 = Color32::from_rgb(80, 180, 120);
const WARN: Color32 = Color32::from_rgb(220, 170, 60);
const ERR: Color32 = Color32::from_rgb(220, 80, 80);
const BOARD_LIGHT: Color32 = Color32::from_rgb(0xF0, 0xD9, 0xB5);
const BOARD_DARK_C: Color32 = Color32::from_rgb(0xB5, 0x88, 0x63);
const WIN_COLOR: Color32 = Color32::from_rgb(235, 235, 230);
const DRAW_COLOR: Color32 = Color32::from_rgb(170, 170, 170);
const LOSS_COLOR: Color32 = Color32::from_rgb(50, 50, 55);
// Readable text variants for WDL on the dark panel (LOSS_COLOR as text
// would be invisible on dark background).
const WIN_TXT: Color32 = Color32::from_rgb(150, 225, 170);
const DRAW_TXT: Color32 = Color32::from_rgb(185, 185, 190);
const LOSS_TXT: Color32 = Color32::from_rgb(240, 140, 140);

/// GPU batch size for batched MCTS leaf evaluation (measured optimum).
const MCTS_BATCH: usize = 64;

/// Batch eval for the legacy CSNN backend (no GPU batching there: one
/// combined eval per position). Used when no ONNX session is loaded so
/// MCTS can never hit the ONNX `expect` (engine-thread panic = dead UI).
fn batch_eval_csnn(positions: &[Position], legal_lists: &[&MoveList]) -> Vec<(i32, Vec<f32>)> {
    positions
        .iter()
        .zip(legal_lists.iter().copied())
        .map(|(pos, legal)| nn::evaluate_loaded_combined(pos, legal))
        .collect()
}

fn nn_combined(pos: &Position, legal: &MoveList) -> (i32, Vec<f32>) {
    if nn::onnx::OnnxEvaluator::is_ready() {
        let (cp, all_logits) = nn::onnx::evaluate_onnx(pos, legal);
        let mut filtered = Vec::with_capacity(legal.len);
        for i in 0..legal.len {
            let mv = legal.moves[i];
            // Guarded: a size-mismatched net must yield neutral priors,
            // never an indexing panic in the engine thread.
            filtered.push(all_logits.get(nn::policy_index(pos.side, mv)).copied().unwrap_or(0.0));
        }
        (cp, filtered)
    } else if nn::loaded().is_some() {
        nn::evaluate_loaded_combined(pos, legal)
    } else {
        eprintln!("[nn] WARNING: no NN loaded, using uniform priors");
        (0, Vec::new())
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum SearchKind { AlphaBeta, Mcts }

#[derive(Default, Clone)]
#[allow(dead_code)]
struct AnalysisState {
    playouts: u64,
    value: f32,
    score_cp: i32,
    depth: i32,
    nodes: u64,
    time_ms: u64,
    mates: u64,
    draws: u64,
    w: f32,
    d: f32,
    l: f32,
    variants: Vec<MctsChildInfo>,
}

struct ChessApp {
    /// Game tree arena (root = initial position). Playing a different move
    /// from a visited node creates a branch instead of erasing history.
    tree: Vec<GameNode>,
    /// Currently viewed node.
    cursor: usize,
    /// Zobrist key of the position the engine is analysing (if any). Arrows
    /// and variants only render when it matches the viewed position.
    analyzed: Option<u64>,
    selected: Option<usize>,
    targets: Vec<Move>,
    game_over: Option<String>,
    search_kind: SearchKind,
    flip: bool,
    nn_file: String,
    nn_error: Option<String>,
    fen_input: String,
    status: String,
    live: Arc<Mutex<AnalysisState>>,
    stop: Arc<AtomicBool>,
    analysis_running: bool,
    engine_tx: mpsc::Sender<EngineCmd>,
    engine_rx: mpsc::Receiver<AnalysisState>,
    live_policy: Vec<PolicyEntry>,
    threads: usize,
    playouts_limit: String,
    movetime_limit: String,
    show_settings: bool,
    show_net_list: bool,
    net_candidates: Vec<String>,
}

#[derive(Clone)]
struct PolicyEntry { mv: Move, prob: f32 }

/// One node of the game tree: the position AFTER `mv` (root has `mv=None`).
#[derive(Clone)]
struct GameNode {
    pos: Position,
    mv: Option<Move>,
    parent: Option<usize>,
    children: Vec<usize>,
}

impl GameNode {
    fn root(pos: Position) -> Self {
        GameNode { pos, mv: None, parent: None, children: Vec::new() }
    }
}

enum EngineCmd {
    Search {
        pos: Position,
        kind: SearchKind,
        keep_child_mv: Option<Move>,
        threads: usize,
        playouts: Option<u64>,
        movetime: Option<u64>,
    },
    /// User played `Move` from the given PRE-move position. The engine only
    /// descends its tree when the tree root matches that position's key.
    Descend(Move, Position),
}

impl ChessApp {
    fn new() -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel();
        let (state_tx, state_rx) = mpsc::channel();
        let live = Arc::new(Mutex::new(AnalysisState::default()));
        let stop = Arc::new(AtomicBool::new(false));
        let saved_nn = load_config();

        let live2 = live.clone();
        let stop2 = stop.clone();
        std::thread::spawn(move || engine_loop(cmd_rx, state_tx, live2, stop2));

        let mut app = Self {
            tree: vec![GameNode::root(Position::startpos())],
            cursor: 0,
            analyzed: None,
            selected: None,
            targets: Vec::new(),
            game_over: None,
            search_kind: SearchKind::Mcts,
            flip: false,
            nn_file: saved_nn,
            nn_error: None,
            fen_input: "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".into(),
            status: String::new(),
            live,
            stop,
            analysis_running: false,
            engine_tx: cmd_tx,
            engine_rx: state_rx,
            live_policy: Vec::new(),
            threads: 4,
            playouts_limit: String::new(),
            movetime_limit: String::new(),
            show_settings: false,
            show_net_list: false,
            net_candidates: Vec::new(),
        };
        if !app.nn_file.is_empty() && std::path::Path::new(&app.nn_file).exists() {
            app.load_nn_file(&app.nn_file.clone());
        }
        if !nn::is_loaded() {
            app.auto_load_onnx();
        }
        app
    }

    fn auto_load_onnx(&mut self) {
        let cwd = std::env::current_dir().unwrap_or_default();
        eprintln!("[nn] searching for ONNX in {:?} and checkpoints/", cwd);
        if let Some(path) = find_onnx_in_dir(cwd.as_path()) {
            eprintln!("[nn] found: {path}");
            match nn::onnx::OnnxEvaluator::init(&path, 512) {
                Ok(_) => {
                    self.nn_file = path.clone();
                    self.nn_error = None;
                    self.status = format!("ONNX auto-chargé: {path}");
                    engine::uci::set_eval(nn::onnx::eval_fn_onnx);
                    engine::mcts::set_mcts_value_policy(nn::onnx::evaluate_onnx);
                    eprintln!("[nn] ONNX loaded successfully");
                    return;
                }
                Err(e) => {
                    eprintln!("[nn] ONNX init failed: {e}");
                    self.nn_error = Some(format!("ONNX: {e}"));
                }
            }
        } else {
            eprintln!("[nn] no .onnx or .csnn found");
            self.nn_error = Some("Aucun .onnx trouvé".into());
        }
    }

    fn load_nn_file(&mut self, path: &str) {
        let p = path.to_owned();
        if path.ends_with(".onnx") {
            match nn::onnx::OnnxEvaluator::init(&p, 512) {
                Ok(_) => {
                    self.nn_error = None;
                    self.status = format!("ONNX chargé: {p}");
                    engine::uci::set_eval(nn::onnx::eval_fn_onnx);
                    engine::mcts::set_mcts_value_policy(nn::onnx::evaluate_onnx);
                    save_config(&p);
                }
                Err(e) => self.nn_error = Some(format!("ONNX: {e}")),
            }
        } else {
            match nn::load(&p) {
                Ok(_) => {
                    // Take ONNX out of the loop (same franken-eval issue
                    // as the UCI backend) and give MCTS the CSNN priors.
                    nn::onnx::OnnxEvaluator::unload();
                    self.nn_error = None;
                    self.status = format!("CSNN chargé: {p}");
                    engine::uci::set_eval(nn::evaluate_loaded_stm);
                    engine::mcts::set_mcts_value_policy(nn::evaluate_loaded_combined);
                    save_config(&p);
                }
                Err(e) => self.nn_error = Some(format!("CSNN: {e}")),
            }
        }
    }

    fn start_analysis(&mut self, keep_child_mv: Option<Move>) {
        if self.game_over.is_some() { return; }
        if self.search_kind == SearchKind::Mcts && !nn::is_loaded() {
            self.status = "MCTS nécessite un .onnx".into();
            return;
        }
        // Signal any running search to stop. Do NOT clear the flag here:
        // the engine clears it when it starts the new Search (engine_loop).
        // Clearing it immediately would race the old search, which would
        // never observe `true` and keep analysing the old position forever.
        self.stop.store(true, Ordering::SeqCst);
        // The engine is about to analyse the viewed position: gate arrows
        // and variants on this key until then.
        self.analyzed = Some(self.view_pos().key);
        let playouts = self.playouts_limit.parse::<u64>().ok();
        let movetime = self.movetime_limit.parse::<u64>().ok();
        let _ = self.engine_tx.send(EngineCmd::Search {
            pos: self.view_pos(),
            kind: self.search_kind,
            keep_child_mv,
            threads: self.threads,
            playouts,
            movetime,
        });
        self.analysis_running = true;
    }

    fn stop_analysis(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        self.analysis_running = false;
        // NOTE: `analyzed` is intentionally kept: pausing must leave the
        // last results visible (they still match the viewed position).
    }

    fn poll_engine(&mut self) {
        let mut latest = None;
        while let Ok(state) = self.engine_rx.try_recv() {
            latest = Some(state);
        }
        if let Some(state) = latest {
            // Feed the policy overlay (arrows/circles) from the search
            // priors; most-probable first (consumers use [0] as max).
            let mut pol: Vec<PolicyEntry> = state
                .variants
                .iter()
                .map(|c| PolicyEntry { mv: c.mv, prob: c.prior })
                .collect();
            pol.sort_by(|a, b| b.prob.partial_cmp(&a.prob).unwrap_or(std::cmp::Ordering::Equal));
            self.live_policy = pol;
            *self.live.lock().unwrap_or_else(|e| e.into_inner()) = state;
        }
    }

    fn reset_tree(&mut self, pos: Position) {
        self.tree = vec![GameNode::root(pos)];
        self.cursor = 0;
        self.selected = None;
        self.targets.clear();
        self.game_over = None;
        *self.live.lock().unwrap_or_else(|e| e.into_inner()) = AnalysisState::default();
    }

    fn new_game(&mut self) {
        self.stop_analysis();
        self.fen_input = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".into();
        self.reset_tree(Position::startpos());
        self.start_analysis(None);
    }

    fn load_fen(&mut self) {
        self.stop_analysis();
        if self.fen_input.split_whitespace().count() != 6 {
            self.status = "FEN invalide".into();
            return;
        }
        let pos = Position::from_fen(&self.fen_input);
        self.reset_tree(pos);
        self.start_analysis(None);
    }

    fn undo(&mut self) {
        let parent = self.tree[self.cursor].parent;
        if parent.is_none() { return; }
        self.stop_analysis();
        self.cursor = parent.unwrap();
        self.selected = None;
        self.targets.clear();
        self.update_game_over();
        *self.live.lock().unwrap_or_else(|e| e.into_inner()) = AnalysisState::default();
        self.start_analysis(None);
    }

    fn view_pos(&self) -> Position {
        self.tree[self.cursor].pos
    }

    /// Depth of the cursor in plies from the root.
    fn depth(&self) -> usize {
        let mut d = 0;
        let mut n = self.cursor;
        while let Some(p) = self.tree[n].parent {
            d += 1;
            n = p;
        }
        d
    }

    /// Node ids from root to cursor (inclusive).
    fn path_to_cursor(&self) -> Vec<usize> {
        let mut path = Vec::new();
        let mut n = self.cursor;
        loop {
            path.push(n);
            match self.tree[n].parent {
                Some(p) => n = p,
                None => break,
            }
        }
        path.reverse();
        path
    }

    /// Jump the viewed node (game navigation). Analysis display follows
    /// via the analyzed-key gate; nothing is erased (branches persist).
    fn goto_node(&mut self, id: usize) {
        if id >= self.tree.len() { return; }
        self.cursor = id;
        self.selected = None;
        self.targets.clear();
        self.update_game_over();
    }

    fn update_game_over(&mut self) {
        let pos = self.view_pos();
        let legal = generate_legal(&pos);
        if legal.len == 0 {
            self.game_over = Some(if pos.in_check() {
                format!("{} gagne", if pos.side == WHITE { "Noirs" } else { "Blancs" })
            } else { "Pat".into() });
        } else {
            self.game_over = None;
        }
    }

    /// Play `m` from the cursor node. Reuses the existing child when the
    /// move was already played (branch navigation); otherwise grows a new
    /// branch. The MCTS side stays consistent through Descend+Search.
    fn apply(&mut self, m: Move) {
        if let Some(&child) = self.tree[self.cursor]
            .children
            .iter()
            .find(|&&c| self.tree[c].mv == Some(m))
        {
            self.cursor = child;
        } else {
            let mut pos = self.view_pos();
            pos.make_move(m);
            let id = self.tree.len();
            self.tree.push(GameNode { pos, mv: Some(m), parent: Some(self.cursor), children: Vec::new() });
            self.tree[self.cursor].children.push(id);
            self.cursor = id;
        }
        self.selected = None;
        self.targets.clear();
        self.update_game_over();
    }

    fn human_move(&mut self, m: Move) {
        let was_running = self.analysis_running;
        // Pre-move position: the engine verifies its tree against this key
        // before descending, so a stale tree can never be advanced by a
        // coincidental move-encoding match.
        let pre_pos = self.view_pos();
        self.apply(m);
        // The engine is about to report on the new position (Descend, then
        // Search if running): gate the display on its key right away.
        self.analyzed = Some(self.view_pos().key);
        // Clear stale variants immediately — position changed but engine
        // hasn't sent updated variants yet. Prevents to_san crash on
        // moves that don't match the new position.
        {
            let mut live = self.live.lock().unwrap_or_else(|e| e.into_inner());
            live.variants.clear();
            live.playouts = 0;
        }
        if was_running {
            self.stop_analysis();
        }
        let _ = self.engine_tx.send(EngineCmd::Descend(m, pre_pos));
        // Keep the previous running state: if analysis was enabled,
        // restart it on the new position (tree is reused via Descend+Search).
        if was_running && self.game_over.is_none() {
            self.start_analysis(None);
        }
    }

    fn press_square(&mut self, sq: usize) {
        if self.game_over.is_some() { return; }
        // No nav guard: playing from a visited node grows a branch.
        if let Some(sel) = self.selected {
            if let Some(m) = self.targets.iter().copied().find(|m| m.to() == sq) {
                self.human_move(m);
                return;
            }
            if sel == sq { self.selected = None; self.targets.clear(); return; }
        }
        let pos = self.view_pos();
        if let Some((c, _)) = pos.piece_at(sq) {
            if c == pos.side {
                self.selected = Some(sq);
                let legal = generate_legal(&pos);
                self.targets = legal.moves[..legal.len].iter().copied().filter(|m| m.from() == sq).collect();
                return;
            }
        }
        self.selected = None;
        self.targets.clear();
    }

    fn release_square(&mut self, sq: usize) {
        if self.game_over.is_some() { return; }
        if self.selected.is_some() {
            if let Some(m) = self.targets.iter().copied().find(|m| m.to() == sq) {
                self.human_move(m);
            }
        }
    }

    /// Whether the live analysis results describe the viewed position.
    /// Arrows, variants, bar and stats only render on a match — otherwise
    /// stale moves would be shown against the wrong board.
    fn analysis_matches_view(&self) -> bool {
        self.analyzed == Some(self.view_pos().key)
    }

    /// Variants currently shown as analysis arrows (single source of truth
    /// for drawing AND one-click play): every analyzed move from the
    /// selected square, else the top 8 by visits (visits order kept).
    fn arrow_infos<'a>(&self, variants: &'a [MctsChildInfo]) -> Vec<&'a MctsChildInfo> {
        if let Some(sq) = self.selected {
            variants.iter().filter(|v| v.mv.from() == sq).collect()
        } else {
            variants.iter().take(8).collect()
        }
    }

    fn draw_board(&mut self, ui: &mut egui::Ui, board_size: f32) {
        let sq_px = board_size / 8.0;
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(board_size), Sense::click_and_drag());
        let painter = ui.painter_at(rect);

        let flip = self.flip;
        let to_xy = |sq: usize| -> Pos2 {
            let file = (sq & 7) as f32;
            let rank = (sq >> 3) as f32;
            let (fx, fy) = if flip { (7.0 - file, 7.0 - rank) } else { (file, rank) };
            Pos2::new(rect.min.x + fx * sq_px, rect.min.y + (7.0 - fy) * sq_px)
        };
        let sq_at = |p: Pos2| -> usize {
            let dx = ((p.x - rect.min.x) / sq_px).floor() as i32;
            let dy = ((p.y - rect.min.y) / sq_px).floor() as i32;
            if !(0..=7).contains(&dx) || !(0..=7).contains(&dy) { return usize::MAX; }
            let file = dx as usize;
            let rank = (7 - dy) as usize;
            if flip { rank * 8 + (7 - file) } else { rank * 8 + file }
        };

        for rank in 0..8 {
            for file in 0..8 {
                let s = rank * 8 + file;
                let color = if (rank + file) % 2 == 0 { BOARD_LIGHT } else { BOARD_DARK_C };
                painter.rect_filled(Rect::from_min_size(to_xy(s), Vec2::splat(sq_px)), 0.0, color);
            }
        }

        if let Some(m) = self.tree[self.cursor].mv {
            for s in [m.from(), m.to()] {
                let r = Rect::from_min_size(to_xy(s), Vec2::splat(sq_px));
                painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(255, 255, 80, 50));
            }
        }
        if let Some(s) = self.selected {
            let r = Rect::from_min_size(to_xy(s), Vec2::splat(sq_px));
            painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(80, 220, 80, 70));
        }

        let pos = self.view_pos();
        if pos.in_check() {
            let c = to_xy(pos.king_sq[pos.side]) + Vec2::splat(sq_px / 2.0);
            painter.circle_stroke(c, sq_px * 0.42, Stroke::new(3.5, Color32::from_rgb(220, 50, 50)));
        }

        // policy circles
        if !self.live_policy.is_empty() {
            let max_prob = self.live_policy[0].prob;
            for entry in &self.live_policy {
                let center = to_xy(entry.mv.to()) + Vec2::splat(sq_px / 2.0);
                let ratio = entry.prob / max_prob;
                let r = sq_px * 0.08 + sq_px * 0.22 * ratio;
                let alpha = (40.0 + 120.0 * ratio) as u8;
                painter.circle_filled(center, r, Color32::from_rgba_unmultiplied(70, 150, 230, alpha));
            }
        }

        // pieces
        let dragging = ui.input(|i| i.pointer.is_decidedly_dragging());
        let font_size = sq_px * 0.82;
        for sq in 0..64 {
            if dragging && self.selected == Some(sq) { continue; }
            if let Some((c, pt)) = pos.piece_at(sq) {
                draw_piece(&painter, to_xy(sq) + Vec2::splat(sq_px / 2.0), pt, c == WHITE, font_size);
            }
        }
        if dragging {
            if let Some(sel) = self.selected {
                if let Some(p) = ui.input(|i| i.pointer.interact_pos()) {
                    if let Some((c, pt)) = pos.piece_at(sel) {
                        draw_piece(&painter, p, pt, c == WHITE, font_size);
                    }
                }
            }
        }

        let targets = self.targets.clone();
        for m in &targets {
            let c = to_xy(m.to()) + Vec2::splat(sq_px / 2.0);
            if m.is_capture() {
                painter.circle_stroke(c, sq_px * 0.44, Stroke::new(2.5, Color32::from_rgba_unmultiplied(0, 0, 0, 140)));
            } else {
                painter.circle_filled(c, sq_px * 0.13, Color32::from_rgba_unmultiplied(0, 0, 0, 140));
            }
        }

        // analysis arrows, Nibbler/GrogZero style: uniform thick lines,
        // dot heads with the expected score (0-100) in black inside.
        // Longest arrows drawn first (underneath), best move on top.
        // Only for the analysed position (see analysis_matches_view).
        let state = self.live.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if !state.variants.is_empty() && self.analysis_matches_view() {
            let best_mv = state.variants[0].mv;
            // Selected piece: every arrow from that square (GrogZero-style).
            // Otherwise: top arrows by visits only.
            let shortlist: Vec<&MctsChildInfo> = self.arrow_infos(&state.variants);
            let mut order: Vec<&MctsChildInfo> = shortlist.iter().copied().collect();
            order.sort_by_key(|v| {
                let df = (v.mv.from() % 8) as i32 - (v.mv.to() % 8) as i32;
                let dr = (v.mv.from() / 8) as i32 - (v.mv.to() / 8) as i32;
                -(df.abs() + dr.abs())
            });
            let width = sq_px * 0.15;
            let head_r = width * 1.6;
            let tail_r = width * 0.9;
            let color_of = |mv: Move| if mv == best_mv {
                Color32::from_rgb(80, 200, 80)
            } else {
                Color32::from_rgb(100, 160, 220)
            };
            // Pass 1: lines, longest underneath (Nibbler).
            for var in &order {
                let a = to_xy(var.mv.from()) + Vec2::splat(sq_px / 2.0);
                let b = to_xy(var.mv.to()) + Vec2::splat(sq_px / 2.0);
                let dir = b - a;
                let len = dir.length().max(1.0);
                let unit = dir / len;
                let color = color_of(var.mv);
                painter.line_segment([a + unit * tail_r, b - unit * head_r], Stroke::new(width, color));
                painter.circle_filled(a, tail_r, color);
            }
            // Pass 2: dot heads with expected score, one label per target
            // square (best move claims it first — avoids stacked numbers).
            let mut labeled: Vec<usize> = Vec::new();
            for var in &shortlist {
                let to = var.mv.to();
                if labeled.contains(&to) {
                    continue;
                }
                labeled.push(to);
                let b = to_xy(to) + Vec2::splat(sq_px / 2.0);
                let color = color_of(var.mv);
                painter.circle_filled(b, head_r, color);
                // expected score 0-100 (== (q+1)/2 now that WDL is consistent)
                let mut label = format!("{:.0}", var.w + var.d * 0.5);
                if label == "100" && var.q < 1.0 {
                    label = "99".to_string();
                }
                painter.text(b, Align2::CENTER_CENTER, label, FontId::monospace(width * 1.05), Color32::BLACK);
            }
        } else if !self.live_policy.is_empty() {
            for (i, entry) in self.live_policy.iter().take(5).enumerate() {
                let a = to_xy(entry.mv.from()) + Vec2::splat(sq_px / 2.0);
                let b = to_xy(entry.mv.to()) + Vec2::splat(sq_px / 2.0);
                let dir = b - a;
                let ratio = entry.prob / self.live_policy[0].prob;
                let width = 2.0 + 5.0 * ratio;
                let alpha = (60.0 + 160.0 * ratio) as u8;
                let color = if i == 0 {
                    Color32::from_rgba_unmultiplied(70, 180, 110, alpha)
                } else {
                    Color32::from_rgba_unmultiplied(80, 140, 210, alpha)
                };
                painter.arrow(a + dir * 0.12, dir * 0.76, Stroke::new(width, color));
                let pct = format!("{:.0}%", entry.prob * 100.0);
                draw_label(&painter, a + dir * 0.30, &pct, 11.0, Color32::from_rgb(20, 20, 20));
            }
        }

        if resp.hovered() && ui.input(|i| i.pointer.primary_pressed()) {
            if let Some(p) = ui.input(|i| i.pointer.press_origin()) {
                // One-click arrowhead (Nibbler): clicking an analysis
                // arrowhead plays the best variant targeting that square.
                let mut arrow_pick: Option<Move> = None;
                if self.game_over.is_none() && self.analysis_matches_view() && !state.variants.is_empty() {
                    let width = sq_px * 0.15;
                    let mut best_dist = width * 1.6 + 8.0;
                    for var in self.arrow_infos(&state.variants) {
                        let b = to_xy(var.mv.to()) + Vec2::splat(sq_px / 2.0);
                        let d = (p - b).length();
                        if d <= best_dist {
                            best_dist = d;
                            arrow_pick = Some(var.mv);
                        }
                    }
                }
                if let Some(mv) = arrow_pick {
                    // Guard: only play moves legal in the viewed position.
                    let legal = engine::movegen::generate_legal(&self.view_pos());
                    if legal.moves[..legal.len].contains(&mv) {
                        self.human_move(mv);
                    } else {
                        let s = sq_at(p);
                        if s != usize::MAX { self.press_square(s); }
                    }
                } else {
                    let s = sq_at(p);
                    if s != usize::MAX { self.press_square(s); }
                }
            }
        }
        if resp.drag_stopped() || resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                let s = sq_at(p);
                if s != usize::MAX { self.release_square(s); }
            }
        }
    }

    fn draw_eval_bar(&self, painter: &egui::Painter, bar_rect: Rect, board_height: f32) {
        let state = self.live.lock().unwrap_or_else(|e| e.into_inner()).clone();

        // Dark backdrop so the bar reads on any panel background.
        painter.rect_filled(bar_rect.expand(2.0), 4.0, Color32::from_rgb(24, 24, 28));

        // Only reflect live analysis when it describes the viewed position.
        let show = self.analysis_matches_view();

        // The bar follows the board orientation: the side at the bottom of
        // the board gets the bottom segment (like lichess/chess.com).
        let white_on_top = self.flip;
        if show && state.playouts > 0 {
            // WDL from white's perspective (always: white at top, black at bottom)
            // state.w/d/l are from side-to-move perspective, convert to white's POV
            let vpos = self.view_pos();
            let (w_white, d_white, _l_white) = if vpos.side == WHITE {
                (state.w, state.d, state.l)
            } else {
                (state.l, state.d, state.w) // flip: black's loss = white's win
            };

            // Bar layout: white at top, draws (gray) middle, black at bottom
            let w_h = board_height * (w_white / 100.0);
            let d_h = board_height * (d_white / 100.0);
            let y_top = bar_rect.min.y;
            let y_bot = bar_rect.max.y;

            if white_on_top {
                // White wins = white at top
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(bar_rect.min.x, y_top), Pos2::new(bar_rect.max.x, y_top + w_h)),
                    2.0, WIN_COLOR,
                );
                // Draws = gray in middle
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(bar_rect.min.x, y_top + w_h), Pos2::new(bar_rect.max.x, y_top + w_h + d_h)),
                    2.0, DRAW_COLOR,
                );
                // Black wins = black at bottom
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(bar_rect.min.x, y_top + w_h + d_h), Pos2::new(bar_rect.max.x, y_bot)),
                    2.0, LOSS_COLOR,
                );
            } else {
                // Black wins = black at top
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(bar_rect.min.x, y_top), Pos2::new(bar_rect.max.x, y_bot - w_h - d_h)),
                    2.0, LOSS_COLOR,
                );
                // Draws = gray in middle
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(bar_rect.min.x, y_bot - w_h - d_h), Pos2::new(bar_rect.max.x, y_bot - w_h)),
                    2.0, DRAW_COLOR,
                );
                // White wins = white at bottom
                painter.rect_filled(
                    Rect::from_min_max(Pos2::new(bar_rect.min.x, y_bot - w_h), Pos2::new(bar_rect.max.x, y_bot)),
                    2.0, WIN_COLOR,
                );
            }
        } else {
            // No WDL data (idle or alpha-beta mode): binary bar from the
            // pawn score, oriented like the board.
            let white_frac = ((state.score_cp as f32 / 800.0).tanh() * 0.5 + 0.5).clamp(0.02, 0.98);
            let h_white = bar_rect.height() * white_frac;
            if white_on_top {
                painter.rect_filled(Rect::from_min_max(Pos2::new(bar_rect.min.x, bar_rect.min.y + h_white), bar_rect.max), 2.0, LOSS_COLOR);
                painter.rect_filled(Rect::from_min_max(bar_rect.min, Pos2::new(bar_rect.max.x, bar_rect.min.y + h_white)), 2.0, WIN_COLOR);
            } else {
                painter.rect_filled(Rect::from_min_max(bar_rect.min, Pos2::new(bar_rect.max.x, bar_rect.max.y - h_white)), 2.0, LOSS_COLOR);
                painter.rect_filled(Rect::from_min_max(Pos2::new(bar_rect.min.x, bar_rect.max.y - h_white), bar_rect.max), 2.0, WIN_COLOR);
            }
        }

        // Score chip: dark pill + bold white text, readable on any segment.
        // Shown only for the analysed position.
        if show {
            let score_text = fmt_score(state.score_cp);
            let galley = painter.layout_no_wrap(score_text, FontId::monospace(12.0), Color32::WHITE);
            let pad = Vec2::new(4.0, 2.0);
            let rect = galley.rect.translate(
                (bar_rect.center() - galley.rect.center().to_vec2()).to_vec2()
            );
            painter.rect_filled(rect.expand2(pad), 4.0, Color32::from_rgba_unmultiplied(15, 15, 18, 220));
            painter.galley(rect.min, galley, Color32::WHITE);
        }
        painter.rect_stroke(bar_rect.expand(2.0), 4.0, Stroke::new(1.0, Color32::from_gray(80)), egui::StrokeKind::Inside);
    }

    fn right_panel(&mut self, ui: &mut egui::Ui, board_height: f32) {
        ui.set_min_width(320.0);
        ui.set_max_width(380.0);

        ui.heading(RichText::new("♟ chess-net").size(18.0).color(TEXT_LIGHT));
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            if ui.button(RichText::new("⟳").color(TEXT_LIGHT)).on_hover_text("Nouvelle partie").clicked() {
                self.new_game();
            }
            if ui.button(RichText::new("↩").color(TEXT_LIGHT)).on_hover_text("Annuler").clicked() {
                self.undo();
            }
            if ui.button(RichText::new("↕").color(TEXT_LIGHT)).on_hover_text("Retourner le plateau").clicked() {
                self.flip = !self.flip;
            }
        });

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        // Search kind
        ui.label(RichText::new("Moteur").color(TEXT_DIM).size(11.0));
        ui.horizontal(|ui| {
            let sk = self.search_kind;
            let ab = sk == SearchKind::AlphaBeta;
            let mc = sk == SearchKind::Mcts;
            if ui.selectable_label(ab, RichText::new("Alpha-beta").color(if ab { ACCENT } else { TEXT_DIM })).clicked() {
                self.search_kind = SearchKind::AlphaBeta;
                self.start_analysis(None);
            }
            if ui.selectable_label(mc, RichText::new("MCTS").color(if mc { ACCENT } else { TEXT_DIM })).clicked() {
                self.search_kind = SearchKind::Mcts;
                self.start_analysis(None);
            }
        });
        ui.horizontal(|ui| {
            if self.analysis_running {
                if ui.button(RichText::new("⏸ Pause").color(WARN)).clicked() {
                    self.stop_analysis();
                }
            } else {
                let label = if self.live.lock().unwrap_or_else(|e| e.into_inner()).playouts > 0 || self.live.lock().unwrap_or_else(|e| e.into_inner()).nodes > 0 {
                    "▶ Reprendre"
                } else {
                    "▶ Analyser"
                };
                if ui.button(RichText::new(label).color(ACCENT)).clicked() && self.game_over.is_none() {
                    self.start_analysis(None);
                }
            }
        });

        // Settings
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Paramètres").color(TEXT_DIM).size(11.0));
            if ui.small_button(RichText::new(if self.show_settings { "▲" } else { "▼" }).color(TEXT_DIM)).clicked() {
                self.show_settings = !self.show_settings;
            }
        });
        if self.show_settings {
            ui.horizontal(|ui| {
                ui.label(RichText::new("Threads:").color(TEXT_DIM).size(11.0));
                ui.add(egui::DragValue::new(&mut self.threads).speed(1).range(1..=32));
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new("Playouts max:").color(TEXT_DIM).size(11.0));
                ui.add(egui::TextEdit::singleline(&mut self.playouts_limit).desired_width(80.0).hint_text("∞"));
            });
            ui.horizontal(|ui| {
                ui.label(RichText::new("Temps max (ms):").color(TEXT_DIM).size(11.0));
                ui.add(egui::TextEdit::singleline(&mut self.movetime_limit).desired_width(80.0).hint_text("∞"));
            });
            if self.search_kind == SearchKind::Mcts {
                ui.label(RichText::new("Conseil: 4 threads, pas de limite → analyse continue").color(TEXT_DIM).size(10.0));
            }
        }

        // Global stats (only for the analysed position).
        let state = self.live.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let show_stats = self.analysis_matches_view() && (state.playouts > 0 || state.nodes > 0);
        if show_stats {
            ui.add_space(4.0);
            ui.label(RichText::new("Stats globales").color(TEXT_DIM).size(11.0));
            if self.search_kind == SearchKind::Mcts {
                // Selective depth = principal-variation length in plies.
                let depth = state.variants.first().map(|v| v.pv.len() + 1).unwrap_or(0);
                ui.horizontal(|ui| {
                    ui.label(format!("{} coups", fmt_u64(state.playouts)));
                    ui.label(format!("p{}", depth));
                    ui.label(format!("{:.1}s", state.time_ms as f32 / 1000.0));
                });
                ui.label(format!("{:.0} coups/s", nps(state.playouts, state.time_ms)));
            } else {
                ui.horizontal(|ui| {
                    ui.label(format!("d{}", state.depth));
                    ui.label(format!("{} n", fmt_u64(state.nodes)));
                    ui.label(format!("{:.1}s", state.time_ms as f32 / 1000.0));
                });
            }
            // WDL from white's perspective (MCTS only — alpha-beta lines
            // carry no WDL, showing zeros would look broken).
            if self.search_kind == SearchKind::Mcts {
                let vpos = self.view_pos();
                let (w_w, d_w, l_w) = if vpos.side == WHITE {
                    (state.w, state.d, state.l)
                } else {
                    (state.l, state.d, state.w)
                };
                ui.horizontal(|ui| {
                    ui.colored_label(WIN_TXT, format!("W:{:.0}%", w_w));
                    ui.colored_label(DRAW_TXT, format!("D:{:.0}%", d_w));
                    ui.colored_label(LOSS_TXT, format!("L:{:.0}%", l_w));
                });
            }
            ui.label(RichText::new(fmt_score(state.score_cp)).size(15.0).strong());
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(4.0);

        // NN
        ui.label(RichText::new("Réseau").color(TEXT_DIM).size(11.0));
        ui.horizontal(|ui| {
            if ui.button(RichText::new("📂").color(TEXT_LIGHT)).on_hover_text("Choisir un réseau").clicked() {
                let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("."));
                self.net_candidates = find_all_nets(&cwd);
                if self.net_candidates.is_empty() {
                    self.nn_error = Some("Aucun .onnx/.csnn trouvé.".into());
                    self.show_net_list = false;
                } else {
                    self.nn_error = None;
                    self.show_net_list = !self.show_net_list;
                }
            }
            ui.add(egui::TextEdit::singleline(&mut self.nn_file).desired_width(180.0).hint_text("fichier NN"));
            if ui.button(RichText::new("→").color(TEXT_LIGHT)).clicked() && !self.nn_file.is_empty() {
                let p = self.nn_file.clone();
                self.load_nn_file(&p);
            }
        });
        if self.show_net_list && !self.net_candidates.is_empty() {
            egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                let mut picked: Option<String> = None;
                for cand in &self.net_candidates {
                    let name = cand.rsplit(['/', '\\']).next().unwrap_or(cand);
                    let sel = *cand == self.nn_file;
                    if ui.selectable_label(sel, RichText::new(name).color(if sel { ACCENT } else { TEXT_LIGHT }).size(11.0)).clicked() {
                        picked = Some(cand.clone());
                    }
                }
                if let Some(p) = picked {
                    self.load_nn_file(&p);
                    self.nn_file = p;
                    self.show_net_list = false;
                }
            });
        }
        if let Some(e) = &self.nn_error {
            ui.colored_label(ERR, e);
        } else if nn::is_loaded() {
            ui.colored_label(ACCENT, "● NN active");
        }

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(4.0);

        // FEN
        ui.label(RichText::new("Position").color(TEXT_DIM).size(11.0));
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.fen_input).desired_width(230.0).hint_text("FEN"));
            if ui.button("→").clicked() { self.load_fen(); }
        });

        ui.add_space(6.0);
        ui.separator();
        ui.add_space(4.0);

        // VARIANTS — Nibbler-style: value first, then the full PV starting
        // with the candidate move itself (best move of every ply), then a
        // gray stats line. Best line highlighted, others dimmed.
        // Only for the analysed position (see analysis_matches_view).
        ui.label(RichText::new("Variantes").color(TEXT_DIM).size(11.0));
        if !self.analysis_matches_view() {
            ui.label(RichText::new("Analyse d'une autre position — cliquez Analyser").color(TEXT_DIM));
        } else if state.variants.is_empty() {
            ui.label(RichText::new("Aucune analyse en cours").color(TEXT_DIM));
        } else {
            let total_v: u64 = state.variants.iter().map(|v| v.visits as u64).sum::<u64>().max(1);
            egui::ScrollArea::vertical().max_height(board_height - 250.0).show(ui, |ui| {
                for (i, var) in state.variants.iter().enumerate() {
                    let is_best = i == 0;
                    let exp = var.w + var.d * 0.5;
                    // Share of search spent on this move + PV depth in plies.
                    let pct = var.visits as f64 * 100.0 / total_v as f64;
                    let pv_depth = var.pv.len() + 1;

                    // Header: rank + candidate move BIG first.
                    ui.horizontal(|ui| {
                        ui.monospace(RichText::new(format!("{}.", i + 1)).color(TEXT_DIM).size(12.0));
                        ui.monospace(
                            RichText::new(to_san(&self.view_pos(), var.mv))
                                .color(if is_best { ACCENT } else { TEXT_LIGHT })
                                .strong()
                                .size(14.0),
                        );
                        ui.monospace(
                            RichText::new(format!("{:.1}%", pct))
                                .color(if is_best { ACCENT } else { TEXT_LIGHT })
                                .size(12.0),
                        );
                        ui.monospace(
                            RichText::new(format!("p{}", pv_depth))
                                .color(TEXT_DIM)
                                .size(11.0),
                        );
                    });

                    // Replies: wrap onto the following lines automatically.
                    if !var.pv.is_empty() {
                        let mut p = self.view_pos();
                        let legal = engine::movegen::generate_legal(&p);
                        if legal.moves[..legal.len].contains(&var.mv) {
                            p.make_move(var.mv);
                            let line = san_line(p, &var.pv);
                            if !line.is_empty() {
                                ui.label(
                                    RichText::new(line)
                                        .monospace()
                                        .color(if is_best { TEXT_LIGHT } else { TEXT_DIM })
                                        .size(12.0),
                                );
                            }
                        }
                    }

                    // Stats gray line (MCTS: with WDL + expected score).
                    ui.horizontal(|ui| {
                        let mut stats =
                            format!("N:{} P:{:.1}% Q:{:.3}", fmt_u64(var.visits as u64), var.prior * 100.0, var.q);
                        if self.search_kind == SearchKind::Mcts {
                            stats.push_str(&format!(
                                " W:{:.0} D:{:.0} L:{:.0} E:{:.0}",
                                var.w, var.d, var.l, exp
                            ));
                        }
                        ui.label(RichText::new(format!("({stats})")).color(TEXT_DIM).size(10.0));
                    });

                    if i < state.variants.len() - 1 {
                        ui.separator();
                    }
                }
            });
        }

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        // Status + move list (current line root -> cursor)
        let vpos = self.view_pos();
        ui.label(RichText::new(if vpos.side == WHITE { "Trait: Blancs" } else { "Trait: Noirs" }).color(TEXT_LIGHT));
        if let Some(go) = &self.game_over { ui.colored_label(ERR, go); }
        if !self.status.is_empty() { ui.colored_label(WARN, &self.status); }

        ui.add_space(2.0);
        ui.horizontal(|ui| {
            if ui.button("<<").on_hover_text("Début").clicked() { self.goto_node(0); }
            if ui.button("<").on_hover_text("Coup précédent").clicked() {
                if let Some(p) = self.tree[self.cursor].parent { self.goto_node(p); }
            }
            if ui.button(">").on_hover_text("Coup suivant (1re variante)").clicked() {
                if let Some(&c) = self.tree[self.cursor].children.first() { self.goto_node(c); }
            }
            if ui.button(">>").on_hover_text("Fin de ligne").clicked() {
                let mut n = self.cursor;
                while let Some(&c) = self.tree[n].children.first() { n = c; }
                self.goto_node(n);
            }
            let kids = self.tree[self.cursor].children.len();
            let mut counter = format!("p{}", self.depth());
            if kids > 1 {
                counter.push_str(&format!(" ({} var)", kids));
            }
            ui.label(RichText::new(counter).color(TEXT_DIM));
        });

        {
            let path = self.path_to_cursor();
            if path.len() > 1 {
                egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                    for &id in &path[1..] {
                        let node = &self.tree[id];
                        let parent_pos = self.tree[node.parent.unwrap()].pos;
                        let m = node.mv.unwrap();
                        let san = to_san(&parent_pos, m);
                        let text = if parent_pos.side == WHITE {
                            format!("{}. {}", parent_pos.fullmove, san)
                        } else {
                            format!("{}... {}", parent_pos.fullmove, san)
                        };
                        let sel = id == self.cursor;
                        let txt = if sel { RichText::new(text).color(ACCENT).strong() } else { RichText::new(text).color(TEXT_LIGHT) };
                        if ui.selectable_label(sel, txt).clicked() { self.goto_node(id); }
                    }
                });
            }
        }

        ui.add_space(4.0);
        ui.separator();
        ui.add_space(4.0);

        // Combined game + exploration tree. Game nodes are clickable to
        // jump; under the cursor, the engine's top variants hang as
        // playable pseudo-branches (same data as Variantes).
        ui.label(RichText::new("Arbre").color(TEXT_DIM).size(11.0));
        self.draw_game_tree(ui);
    }

    /// Recursive game-tree rendering from `id` (indented, clickable).
    /// Under the cursor node, engine variants appear as branches.
    fn draw_game_tree(&mut self, ui: &mut egui::Ui) {
        let state = self.live.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let show_engine = self.analysis_matches_view();
        egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
            self.draw_tree_node(ui, 0, 0, &state, show_engine);
        });
    }

    fn draw_tree_node(&mut self, ui: &mut egui::Ui, id: usize, indent: usize, state: &AnalysisState, show_engine: bool) {
        // SAN label from the parent position (always legal by construction).
        let label = if let Some(mv) = self.tree[id].mv {
            let parent_pos = self.tree[self.tree[id].parent.unwrap()].pos;
            let san = to_san(&parent_pos, mv);
            if parent_pos.side == WHITE {
                format!("{}. {}", parent_pos.fullmove, san)
            } else {
                format!("{}... {}", parent_pos.fullmove, san)
            }
        } else {
            "début".to_string()
        };
        let nkids = self.tree[id].children.len();
        let txt = if nkids > 1 && id != self.cursor {
            format!("{label} (+{} var)", nkids - 1)
        } else {
            label
        };
        ui.horizontal(|ui| {
            ui.add_space(indent as f32 * 12.0);
            let sel = id == self.cursor;
            let col = if sel { ACCENT } else if id == 0 { TEXT_DIM } else { TEXT_LIGHT };
            let mut rt = RichText::new(txt).color(col).size(12.0);
            if sel { rt = rt.strong(); }
            if ui.selectable_label(sel, rt).clicked() { self.goto_node(id); }
        });
        // Engine variations grafted under the cursor node (click = play).
        if id == self.cursor && show_engine {
            let mut picked: Option<Move> = None;
            for var in state.variants.iter().take(6) {
                ui.horizontal(|ui| {
                    ui.add_space(indent as f32 * 12.0 + 12.0);
                    ui.label(RichText::new("▸").color(ACCENT).size(11.0));
                    if ui.selectable_label(
                        false,
                        RichText::new(format!(
                            "{} {:.0}% E:{:.0}",
                            to_san(&self.view_pos(), var.mv),
                            var.visits as f64 * 100.0
                                / state.variants.iter().map(|v| v.visits as u64).sum::<u64>().max(1) as f64,
                            var.w + var.d * 0.5
                        ))
                        .color(TEXT_LIGHT)
                        .size(11.0),
                    ).clicked() {
                        picked = Some(var.mv);
                    }
                });
            }
            if let Some(mv) = picked {
                let legal = engine::movegen::generate_legal(&self.view_pos());
                if legal.moves[..legal.len].contains(&mv) {
                    self.human_move(mv);
                }
            }
        }
        // Recurse into game children (cap breadth to keep the view readable,
        // and depth: a 300-ply game would otherwise nest 300 UI frames deep).
        if indent < 64 {
            let kids: Vec<usize> = self.tree[id].children.iter().take(8).copied().collect();
            for c in kids {
                self.draw_tree_node(ui, c, indent + 1, state, show_engine);
            }
        }
    }
}

/// Background engine thread: runs MCTS or alpha-beta continuously, reports state via channel.
fn engine_loop(
    rx: mpsc::Receiver<EngineCmd>,
    state_tx: mpsc::Sender<AnalysisState>,
    live: Arc<Mutex<AnalysisState>>,
    stop: Arc<AtomicBool>,
) {
    eprintln!("[engine] thread started");
    let mut prev_mcts: Option<MctsSearch> = None;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
    loop {
        match rx.recv() {
            Ok(EngineCmd::Search { pos, kind, keep_child_mv, threads, playouts: playouts_limit, movetime: movetime_limit }) => {
                stop.store(false, Ordering::SeqCst);
                // Send initial "searching" state immediately
                let _ = state_tx.send(AnalysisState {
                    playouts: 0, score_cp: 0, time_ms: 0,
                    ..AnalysisState::default()
                });
                match kind {
                    SearchKind::Mcts => {
                        let limits = MctsLimits {
                            playouts: playouts_limit,
                            movetime: movetime_limit,
                            threads,
                        };
                        let mut pos = pos;
                        if keep_child_mv.is_none() {
                            // Drop a stale tree: new game / undo / FEN changed
                            // the position while analysis was running. Reusing
                            // it would run tree moves against a mismatched
                            // position (panic in make_move on a worker).
                            let stale = prev_mcts.as_ref().map(|s| s.root_key() != pos.key).unwrap_or(false);
                            if stale {
                                prev_mcts = None;
                            }
                        }
                        if let Some(mv) = keep_child_mv {
                            if let Some(ref mut search) = prev_mcts {
                                pos.make_move(mv);
                                search.keep_child(mv, pos.key);
                            }
                        }
                        // Backend-aware batch eval: ONNX batches on GPU, CSNN
                        // falls back to per-position CPU eval. Never call the
                        // ONNX batch path without a session (its expect()
                        // would kill the whole engine thread).
                        let batch_fn = if nn::onnx::OnnxEvaluator::is_ready() {
                            nn::onnx::batch_evaluate_onnx
                        } else {
                            batch_eval_csnn
                        };
                        let (result, search) = mcts_batched_parallel(&mut pos, &limits, &stop, nn_combined, batch_fn, MCTS_BATCH, false, Some(&mut |p| {
                            let q = 2.0 * p.value - 1.0;
                            let (w, d, l) = wdl_from_q(q);
                            let state = AnalysisState {
                                playouts: p.playouts,
                                value: p.value,
                                score_cp: cp_from_prob(p.value),
                                time_ms: p.time_ms,
                                mates: p.mates,
                                draws: p.draws,
                                w: w * 100.0,
                                d: d * 100.0,
                                l: l * 100.0,
                                variants: p.moves.iter().map(|c| MctsChildInfo {
                                    mv: c.mv, visits: c.visits,
                                    prior: c.prior, q: c.q,
                                    w: c.w, d: c.d, l: c.l,
                                    pv: c.pv.clone(),
                                }).collect(),
                                ..AnalysisState::default()
                            };
                            *live.lock().unwrap_or_else(|e| e.into_inner()) = state.clone();
                            let _ = state_tx.send(state);
                        }), prev_mcts.take());
                        prev_mcts = Some(search);
                        // Instant mate/terminal: no progress callback fired
                        // (empty visits), so publish the final state here —
                        // otherwise the UI sticks on "searching" forever.
                        if result.visits.is_empty() && result.best != Move::null() {
                            let q = 2.0 * result.value - 1.0;
                            let (w, d, l) = wdl_from_q(q);
                            let state = AnalysisState {
                                playouts: result.playouts,
                                value: result.value,
                                score_cp: cp_from_prob(result.value),
                                time_ms: result.time_ms,
                                mates: result.mates,
                                draws: result.draws,
                                w: w * 100.0,
                                d: d * 100.0,
                                l: l * 100.0,
                                variants: vec![MctsChildInfo {
                                    mv: result.best,
                                    visits: result.playouts as u32,
                                    prior: 1.0,
                                    q,
                                    w: w * 100.0,
                                    d: d * 100.0,
                                    l: l * 100.0,
                                    pv: vec![result.best],
                                }],
                                ..AnalysisState::default()
                            };
                            *live.lock().unwrap_or_else(|e| e.into_inner()) = state.clone();
                            let _ = state_tx.send(state);
                        }
                    }
                    SearchKind::AlphaBeta => {
                        // No time limit — runs until stopped
                        let mut searcher = Searcher::new(64);
                        let limits = Limits { multi_pv: 8, depth: None, movetime: None, ..Default::default() };
                        let mut pos = pos;
                        searcher.think_cb(&mut pos, &limits, &stop, evaluate, Some(&mut |it: &SearchIter| {
                            let state = AnalysisState {
                                score_cp: it.score,
                                depth: it.depth as i32,
                                nodes: it.nodes,
                                time_ms: it.time_ms,
                                variants: it.lines.iter().map(|x| MctsChildInfo {
                                    mv: x.mv, visits: x.visits,
                                    prior: 0.0, q: 0.0,
                                    w: 0.0, d: 0.0, l: 0.0,
                                    pv: x.pv.clone(),
                                }).collect(),
                                ..AnalysisState::default()
                            };
                            *live.lock().unwrap_or_else(|e| e.into_inner()) = state.clone();
                            let _ = state_tx.send(state);
                        }));
                    }
                }
            }
            Ok(EngineCmd::Descend(mv, pre_pos)) => {
                // Descend only when the tree matches the pre-move position.
                // A stale tree (rapid moves, reset races) is dropped instead:
                // advancing it on a coincidental move-encoding match would run
                // tree moves against the wrong position (worker panic).
                let matches = prev_mcts.as_ref().map(|s| s.root_key() == pre_pos.key).unwrap_or(false);
                if matches {
                    // Belt and braces (panic=abort profile): only make the
                    // move if it is actually legal in the pre-move position.
                    let legal = engine::movegen::generate_legal(&pre_pos);
                    if !legal.moves[..legal.len].contains(&mv) {
                        prev_mcts = None;
                    } else {
                        let mut post = pre_pos;
                        post.make_move(mv);
                        if let Some(ref mut search) = prev_mcts {
                            search.keep_child(mv, post.key);
                            // Populate variants from the subtree so the UI shows them immediately
                            let (root_visits, child_info) = search.current_child_stats();
                            let q = child_info.first().map(|c| c.q).unwrap_or(0.0);
                            let value = q_to_prob(q);
                            let state = AnalysisState {
                                playouts: root_visits,
                                value,
                                score_cp: cp_from_prob(value),
                                variants: child_info,
                                ..AnalysisState::default()
                            };
                            *live.lock().unwrap_or_else(|e| e.into_inner()) = state.clone();
                            let _ = state_tx.send(state);
                        }
                    }
                } else {
                    // No usable subtree: drop and clear. If analysis was
                    // running, the Search command right behind this one
                    // starts a fresh search on the new position.
                    prev_mcts = None;
                    let state = AnalysisState::default();
                    *live.lock().unwrap_or_else(|e| e.into_inner()) = state.clone();
                    let _ = state_tx.send(state);
                }
            }
            Err(_) => break,
        }
    }
    }));
    if let Err(e) = result {
        if let Some(s) = e.downcast_ref::<String>() {
            eprintln!("[engine] PANIC: {s}");
        } else if let Some(s) = e.downcast_ref::<&str>() {
            eprintln!("[engine] PANIC: {s}");
        } else {
            eprintln!("[engine] PANIC: unknown error");
        }
    }
}

impl eframe::App for ChessApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll_engine();

        // Keyboard game navigation (ignored while typing in a text field):
        // Left = parent, Right = first child, Up = root, Down = deepest.
        if ui.memory(|m| m.focused().is_none()) {
            let (left, right, up, down) = ui.input(|i| {
                (
                    i.key_pressed(egui::Key::ArrowLeft),
                    i.key_pressed(egui::Key::ArrowRight),
                    i.key_pressed(egui::Key::ArrowUp),
                    i.key_pressed(egui::Key::ArrowDown),
                )
            });
            if left {
                if let Some(p) = self.tree[self.cursor].parent { self.goto_node(p); }
            }
            if right {
                if let Some(&c) = self.tree[self.cursor].children.first() { self.goto_node(c); }
            }
            if up {
                self.goto_node(0);
            }
            if down {
                let mut n = self.cursor;
                while let Some(&c) = self.tree[n].children.first() { n = c; }
                self.goto_node(n);
            }
        }

        let available = ui.available_size();
        let side_panel_w = 340.0;
        let bar_w = 28.0;
        let board_size = (available.x - side_panel_w - bar_w - 40.0).min(available.y - 40.0).max(300.0);

        egui::Panel::left("side").show(ui, |ui| {
            ui.set_min_width(side_panel_w - 20.0);
            ui.set_max_width(side_panel_w - 20.0);
            self.right_panel(ui, board_size);
        });

        ui.horizontal_centered(|ui| {
            ui.add_space(8.0);
            self.draw_board(ui, board_size);
            let bar_rect = Rect::from_min_size(
                ui.cursor().min - Vec2::new(bar_w + 4.0, 0.0),
                Vec2::new(bar_w, board_size),
            );
            self.draw_eval_bar(ui.painter(), bar_rect, board_size);
        });

        ui.ctx().request_repaint();
    }
}

fn fmt_score(s: i32) -> String {
    if s >= MATE - 200 { format!("mat {}", (MATE - s + 1) / 2) }
    else if s <= -MATE + 200 { format!("mat -{}", (MATE + s + 1) / 2) }
    else { format!("{:.2}", s as f32 / 100.0) }
}

fn nps(n: u64, time_ms: u64) -> f64 {
    if time_ms == 0 { 0.0 } else { n as f64 * 1000.0 / time_ms as f64 }
}

fn san_line(mut pos: Position, moves: &[Move]) -> String {
    let mut out = Vec::new();
    for m in moves {
        let legal = engine::movegen::generate_legal(&pos);
        if !legal.moves[..legal.len].contains(m) {
            break;
        }
        let san = to_san(&pos, *m);
        let prefix = if pos.side == WHITE { format!("{}. ", pos.fullmove) }
        else if out.is_empty() { format!("{}... ", pos.fullmove) } else { String::new() };
        out.push(format!("{prefix}{san}"));
        pos.make_move(*m);
    }
    out.join(" ")
}

fn fmt_u64(n: u64) -> String {
    if n >= 1_000_000 { format!("{:.1}M", n as f32 / 1_000_000.0) }
    else if n >= 1_000 { format!("{:.1}k", n as f32 / 1_000.0) }
    else { n.to_string() }
}

fn draw_label(painter: &egui::Painter, pos: Pos2, text: &str, size: f32, bg: Color32) {
    let galley = painter.layout_no_wrap(text.into(), FontId::monospace(size), Color32::from_rgb(20, 20, 20));
    let pad = Vec2::splat(3.0);
    let rect = galley.rect.translate(pos.to_vec2() - galley.rect.center().to_vec2());
    let bg_dim = Color32::from_rgba_unmultiplied(bg.r(), bg.g(), bg.b(), 180);
    painter.rect_filled(rect.expand2(pad), 3.0, bg_dim);
    painter.galley(rect.min, galley, Color32::from_rgb(20, 20, 20));
}

const GLYPHS: [char; 6] = ['♟', '♞', '♝', '♜', '♛', '♚'];

fn draw_piece(painter: &egui::Painter, center: Pos2, pt: usize, white: bool, font_size: f32) {
    let glyph = GLYPHS[pt];
    let font = FontId::proportional(font_size);
    let (fg, outline) = if white {
        (Color32::WHITE, Color32::from_rgb(60, 60, 60))
    } else {
        (Color32::from_rgb(20, 20, 25), Color32::from_rgb(230, 230, 230))
    };
    let shadow = center + Vec2::new(1.5, 1.5);
    painter.text(shadow, Align2::CENTER_CENTER, glyph, font.clone(), outline);
    painter.text(center, Align2::CENTER_CENTER, glyph, font, fg);
}

fn main() -> eframe::Result {
    engine::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1200.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("chess-net"),
        ..Default::default()
    };
    eframe::run_native(
        "chess-net",
        options,
        Box::new(|cc| {
            let mut fonts = egui::FontDefinitions::default();
            for path in ["C:/Windows/Fonts/seguisym.ttf", "C:/Windows/Fonts/segoeui.ttf"] {
                if let Ok(bytes) = std::fs::read(path) {
                    fonts.font_data.insert("ui-sym".into(), egui::FontData::from_owned(bytes).into());
                    if let Some(fam) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                        fam.push("ui-sym".into());
                    }
                    break;
                }
            }
            cc.egui_ctx.set_fonts(fonts);
            Ok(Box::new(ChessApp::new()))
        }),
    )
}
