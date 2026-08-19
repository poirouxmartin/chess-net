//! chess-net GUI: playable chessboard + live engine info panel.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use eframe::egui::{
    self, Align2, Color32, ComboBox, FontId, Pos2, Rect, Sense, Stroke, Vec2,
};

use engine::evaluate::{evaluate, evaluate_breakdown, MATE};
use engine::mcts::{cp_from_prob, mcts_parallel, wdl_from_q, MctsLimits};
use engine::move_::Move;
use engine::movegen::generate_legal;
use engine::position::{BLACK, Position, WHITE};
use engine::san::to_san;
use engine::search::{Limits, MultiLine, SearchIter, Searcher};

const BOARD_PX: f32 = 640.0;
const BOARD_BG: Color32 = Color32::from_rgb(45, 45, 48);
const PIECE_NAMES: [&str; 6] = ["P", "N", "B", "R", "Q", "K"];

#[derive(Clone, Copy, PartialEq)]
enum EvalKind {
    PeSTO,
    NN,
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    HumanEngine,
    HumanHuman,
    EngineEngine,
}

#[derive(Clone, Copy, PartialEq)]
enum SearchKind {
    AlphaBeta,
    Mcts,
}

#[derive(Clone, Copy, PartialEq)]
enum GameKind {
    New,
    Setup,
}

#[derive(Clone)]
struct LiveLine {
    mv: Move,
    score: i32,
    pv: Vec<Move>,
    visits: u32,
}

#[derive(Default)]
struct LiveInfo {
    mcts: bool,
    playouts: u64,
    value: f32,
    depth: i32,
    score: i32,
    nodes: u64,
    time_ms: u64,
    mates: u64,
    draws: u64,
    pv: Vec<Move>,
    lines: Vec<LiveLine>,
}

struct ChessApp {
    pos: Position,
    history: Vec<Position>,
    played: Vec<Move>,
    nav: usize,
    selected: Option<usize>,
    targets: Vec<Move>,
    last_from: Option<usize>,
    last_to: Option<usize>,
    game_over: Option<String>,
    mode: Mode,
    search_kind: SearchKind,
    analyzing: bool,
    analysis_pending: bool,
    flip: bool,
    movetime_ms: u64,
    mcts_threads: usize,
    mcts_analysis_ms: u64,
    eval_kind: EvalKind,
    nn_file: String,
    nn_error: Option<String>,
    fen_input: String,
    status: String,

    gen: u64,
    active: Option<u64>,
    live: Arc<Mutex<LiveInfo>>,
    stop: Arc<AtomicBool>,
    tx: mpsc::Sender<(u64, Move, Vec<MultiLine>)>,
    rx: mpsc::Receiver<(u64, Move, Vec<MultiLine>)>,
    eval_fn: engine::search::EvalFn,
}

impl ChessApp {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            pos: Position::startpos(),
            history: Vec::new(),
            played: Vec::new(),
            nav: 0,
            selected: None,
            targets: Vec::new(),
            last_from: None,
            last_to: None,
            game_over: None,
            mode: Mode::HumanEngine,
            search_kind: SearchKind::AlphaBeta,
            analyzing: false,
            analysis_pending: false,
            flip: false,
            movetime_ms: 1000,
            mcts_threads: 0,
            mcts_analysis_ms: 2000,
            eval_kind: EvalKind::PeSTO,
            nn_file: String::new(),
            nn_error: None,
            fen_input: String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            status: String::new(),
            gen: 0,
            active: None,
            live: Arc::new(Mutex::new(LiveInfo::default())),
            stop: Arc::new(AtomicBool::new(false)),
            tx,
            rx,
            eval_fn: evaluate,
        }
    }

    fn new_game(&mut self, kind: GameKind) {
        self.stop.store(true, Ordering::Relaxed);
        self.gen += 1;
        let fen = match kind {
            GameKind::New => String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
            GameKind::Setup => self.fen_input.clone(),
        };
        if fen.split_whitespace().count() != 6 {
            self.status = "FEN invalide : il faut 6 champs".to_owned();
            return;
        }
        let p = Position::from_fen(&fen);
        self.pos = p;
        self.fen_input = fen;
        self.status = format!(
            "Position chargée ({})",
            if self.pos.side == 0 { "Blancs" } else { "Noirs" }
        );
        self.history.clear();
        self.played.clear();
        self.nav = 0;
        self.selected = None;
        self.targets.clear();
        self.last_from = None;
        self.last_to = None;
        self.game_over = None;
        self.active = None;
        let _ = self.rx.try_recv();
        let mut l = self.live.lock().unwrap();
        *l = LiveInfo::default();
        if self.analyzing {
            self.analysis_pending = true;
        }
    }

    fn undo(&mut self) {
        if self.history.is_empty() {
            return;
        }
        self.stop.store(true, Ordering::Relaxed);
        self.gen += 1;
        self.pos = self.history.pop().unwrap();
        self.played.pop();
        self.selected = None;
        self.targets.clear();
        self.game_over = None;
        self.last_from = None;
        self.last_to = None;
        self.active = None;
        let _ = self.rx.try_recv();
        let mut l = self.live.lock().unwrap();
        *l = LiveInfo::default();
        if self.analyzing {
            self.analysis_pending = true;
        }
        // reculer d'un coup de plus si le moteur vient de jouer et c'est au tour du moteur
        if self.mode == Mode::HumanEngine && self.pos.side == BLACK && !self.history.is_empty() {
            self.pos = self.history.pop().unwrap();
            self.played.pop();
        }
        self.nav = self.nav.min(self.played.len());
    }

    /// Position currently displayed (live end of game, or the nav point).
    fn view_pos(&self) -> Position {
        if self.nav < self.played.len() {
            self.history[self.nav]
        } else {
            self.pos
        }
    }

    fn is_searching(&self) -> bool {
        self.active.is_some()
    }

    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.gen += 1;
        self.active = None;
        let _ = self.rx.try_recv();
    }

    fn trigger_engine(&mut self, analysis: bool) {
        if self.active.is_some() || self.game_over.is_some() {
            return;
        }
        let mut pos = self.pos;
        let stop = Arc::new(AtomicBool::new(false));
        self.stop = stop.clone();
        let live = self.live.clone();
        let tx = self.tx.clone();
        let eval = self.eval_fn;
        let ms = self.movetime_ms;
        let mcts_threads = self.mcts_threads;
        let mcts_analysis_ms = self.mcts_analysis_ms;
        let gen = self.gen;
        let kind = self.search_kind;
        self.active = Some(gen);
        *self.live.lock().unwrap() = LiveInfo::default();
        std::thread::spawn(move || {
            let (best, lines) = match kind {
                SearchKind::AlphaBeta => {
                    let mut searcher = Searcher::new(64);
                    let limits = if analysis {
                        Limits { multi_pv: 5, depth: None, ..Default::default() }
                    } else {
                        Limits { movetime: Some(ms), ..Default::default() }
                    };
                    let result = searcher.think_cb(&mut pos, &limits, &stop, eval, Some(&mut |it: &SearchIter| {
                        let mut l = live.lock().unwrap();
                        *l = LiveInfo {
                            mcts: false,
                            depth: it.depth,
                            score: it.score,
                            nodes: it.nodes,
                            time_ms: it.time_ms,
                            pv: it.pv.clone(),
                            lines: it.lines.iter()
                                .map(|x| LiveLine { mv: x.mv, score: x.score, pv: x.pv.clone(), visits: 0 })
                                .collect(),
                            ..LiveInfo::default()
                        };
                    }));
                    (result.best, result.lines)
                }
                SearchKind::Mcts => {
                    let limits = if analysis {
                        MctsLimits { playouts: None, movetime: Some(mcts_analysis_ms), threads: mcts_threads }
                    } else {
                        MctsLimits { playouts: None, movetime: Some(ms), threads: mcts_threads }
                    };
                    let result = mcts_parallel(&mut pos, &limits, stop.clone(), eval, Some(&mut |p| {
                        let mut l = live.lock().unwrap();
                        *l = LiveInfo {
                            mcts: true,
                            playouts: p.playouts,
                            value: p.value,
                            score: cp_from_prob(p.value),
                            nodes: p.playouts,
                            time_ms: p.time_ms,
                            mates: p.mates,
                            draws: p.draws,
                            pv: p.moves.iter().map(|(m, _)| *m).collect(),
                            lines: p
                                .moves
                                .iter()
                                .map(|(m, v)| LiveLine { mv: *m, score: 0, pv: vec![*m], visits: *v })
                                .collect(),
                            ..LiveInfo::default()
                        };
                    }));
                    let lines = result
                        .visits
                        .iter()
                        .map(|(m, v)| MultiLine { mv: *m, score: 0, pv: vec![*m], visits: *v })
                        .collect();
                    (result.best, lines)
                }
            };
            let _ = tx.send((gen, best, lines));
        });
    }

    fn engine_move_arrived(&mut self) {
        if let Ok((gen, m, lines)) = self.rx.try_recv() {
            if self.active != Some(gen) {
                return;
            }
            self.active = None;
            if self.analyzing {
                // Analyse : garder le plateau, afficher les variantes finales.
                let mut l = self.live.lock().unwrap();
                l.lines = lines
                    .into_iter()
                    .map(|x| LiveLine { mv: x.mv, score: x.score, pv: x.pv, visits: x.visits })
                    .collect();
                return;
            }
            self.apply(m);
        }
    }

    fn apply(&mut self, m: Move) {
        self.history.push(self.pos);
        self.played.push(m);
        self.nav = self.played.len();
        self.last_from = Some(m.from());
        self.last_to = Some(m.to());
        self.selected = None;
        self.targets.clear();
        self.pos.make_move(m);
        play_move_sound();
        let legal = generate_legal(&self.pos);
        if legal.len == 0 {
            self.game_over = Some(if self.pos.in_check() {
                format!("Échec et mat — {} gagne", if self.pos.side == 0 { "Noirs" } else { "Blancs" })
            } else {
                String::from("Pat")
            });
        }
    }

    fn human_move(&mut self, m: Move) {
        self.stop_search();
        self.gen += 1;
        self.apply(m);
        if self.analyzing {
            self.analysis_pending = true;
        }
    }

    fn human_has_turn(&self) -> bool {
        if self.analyzing {
            return true;
        }
        match self.mode {
            Mode::HumanHuman => true,
            Mode::HumanEngine => self.pos.side == WHITE,
            Mode::EngineEngine => false,
        }
    }

    fn maybe_engine_plays(&mut self) {
        if self.nav < self.played.len() {
            return;
        }
        if self.analyzing {
            if self.analysis_pending && !self.is_searching() && self.game_over.is_none() {
                self.analysis_pending = false;
                self.trigger_engine(true);
            }
            return;
        }
        let engine_turn = match self.mode {
            Mode::HumanEngine => self.pos.side == BLACK,
            Mode::EngineEngine => true,
            Mode::HumanHuman => false,
        };
        if engine_turn && !self.is_searching() && self.game_over.is_none() {
            self.trigger_engine(false);
        }
    }

    fn static_eval_panel(&mut self, ui: &mut egui::Ui) {
        let b = evaluate_breakdown(&self.view_pos());
        egui::CollapsingHeader::new("Éval statique").default_open(false).show(ui, |ui| {
            ui.monospace(format!("score (au trait): {:+.0} cp", b.score));
            ui.monospace(format!("tapered: {:+.0} cp | phase: {}/24 | tempo: +{}", b.tapered, b.phase, b.tempo));
            let w_mat: i32 = (0..6).map(|pt| b.detail[WHITE][pt].material).sum();
            let b_mat: i32 = (0..6).map(|pt| b.detail[BLACK][pt].material).sum();
            let w_pst = b.tapered - (w_mat - b_mat);
            ui.monospace(format!(
                "matériel: B {:+} / N {:+} | diff {:+}",
                w_mat, b_mat, w_mat - b_mat
            ));
            ui.monospace(format!("PST (moyenné): {:+} cp", w_pst));
            ui.separator();
            for pt in 0..6 {
                let w = b.detail[WHITE][pt];
                let bl = b.detail[BLACK][pt];
                if w.count == 0 && bl.count == 0 {
                    continue;
                }
                ui.monospace(format!(
                    "{}  B x{} mat{:+} pst{:+}/{:+}   N x{} mat{:+} pst{:+}/{:+}",
                    PIECE_NAMES[pt],
                    w.count, w.material, w.pst_mg, w.pst_eg,
                    bl.count, bl.material, bl.pst_mg, bl.pst_eg,
                ));
            }
        });
    }

    fn draw_board(&mut self, ui: &mut egui::Ui) {
        let sq_px = BOARD_PX / 8.0;
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(BOARD_PX), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 4.0, BOARD_BG);
        let board_rect = Rect::from_min_size(rect.min, Vec2::splat(BOARD_PX));

        let flip = self.flip;
        let to_xy = |sq: usize| -> Pos2 {
            let file = (sq & 7) as f32;
            let rank = (sq >> 3) as f32;
            let (fx, fy) = if flip { (7.0 - file, 7.0 - rank) } else { (file, rank) };
            Pos2::new(board_rect.min.x + fx * sq_px, board_rect.min.y + (7.0 - fy) * sq_px)
        };
        let sq_at = |p: Pos2| -> usize {
            let dx = ((p.x - board_rect.min.x) / sq_px).floor() as i32;
            let dy = ((p.y - board_rect.min.y) / sq_px).floor() as i32;
            if !(0..=7).contains(&dx) || !(0..=7).contains(&dy) {
                return usize::MAX;
            }
            let file = dx as usize;
            let rank = (7 - dy) as usize;
            if flip {
                rank * 8 + (7 - file)
            } else {
                rank * 8 + file
            }
        };

        // squares (a1 is dark)
        for rank in 0..8 {
            for file in 0..8 {
                let s = rank * 8 + file;
                let color = if (rank + file) % 2 == 0 {
                    Color32::from_rgb(0xB5, 0x88, 0x63)
                } else {
                    Color32::from_rgb(0xF0, 0xD9, 0xB5)
                };
                let r = Rect::from_min_size(to_xy(s), Vec2::splat(sq_px));
                painter.rect_filled(r, 0.0, color);
            }
        }

        // highlights: last move (or currently viewed move)
        let (hl_from, hl_to) = if self.nav > 0 {
            let m = self.played[self.nav - 1];
            (Some(m.from()), Some(m.to()))
        } else {
            (None, None)
        };
        if let (Some(f), Some(t)) = (hl_from, hl_to) {
            let tint = Color32::from_rgba_unmultiplied(255, 255, 0, 55);
            for s in [f, t] {
                let r = Rect::from_min_size(to_xy(s), Vec2::splat(sq_px));
                painter.rect_filled(r, 0.0, tint);
            }
        }
        // selected
        if let Some(s) = self.selected {
            let r = Rect::from_min_size(to_xy(s), Vec2::splat(sq_px));
            painter.rect_filled(r, 0.0, Color32::from_rgba_unmultiplied(80, 220, 80, 90));
        }
        // check (viewed position)
        let pos = self.view_pos();
        let check_sq = pos.in_check().then(|| pos.king_sq[pos.side]);
        if let Some(s) = check_sq {
            let c = to_xy(s) + Vec2::splat(sq_px / 2.0);
            painter.circle_stroke(c, sq_px * 0.42, Stroke::new(4.0, Color32::from_rgb(220, 40, 40)));
        }

        // pieces (skip the one being dragged)
        let dragging = ui.input(|i| i.pointer.is_decidedly_dragging());
        let font_size = sq_px * 0.82;
        for sq in 0..64 {
            if dragging && self.selected == Some(sq) {
                continue;
            }
            if let Some((c, pt)) = pos.piece_at(sq) {
                let center = to_xy(sq) + Vec2::splat(sq_px / 2.0);
                draw_piece(&painter, center, pt, c == WHITE, font_size);
            }
        }
        // drag ghost: piece follows the cursor
        if dragging {
            if let Some(sel) = self.selected {
                if let Some(p) = ui.input(|i| i.pointer.interact_pos()) {
                    if let Some((c, pt)) = pos.piece_at(sel) {
                        draw_piece(&painter, p, pt, c == WHITE, font_size);
                    }
                }
            }
        }

        // targets (dots / capture rings)
        let targets = self.targets.clone();
        for m in &targets {
            let c = to_xy(m.to()) + Vec2::splat(sq_px / 2.0);
            if m.is_capture() {
                painter.circle_stroke(c, sq_px * 0.44, Stroke::new(3.0, Color32::from_rgba_unmultiplied(0, 0, 0, 160)));
            } else {
                painter.circle_filled(c, sq_px * 0.14, Color32::from_rgba_unmultiplied(0, 0, 0, 160));
            }
        }

        // analysis: best-move highlight + numbered arrows (root moves only)
        if self.analyzing {
            let info = self.live.lock().unwrap().clone_into_info();
            let arrows: Vec<(Move, String)> = if !info.lines.is_empty() {
                info.lines
                    .iter()
                    .take(5)
                    .map(|l| {
                        let num = if info.mcts {
                            let total = info.lines.iter().map(|x| x.visits as u64).sum::<u64>().max(1);
                            format!("{:.0}%", l.visits as f64 * 100.0 / total as f64)
                        } else {
                            fmt_score(l.score)
                        };
                        (l.mv, num)
                    })
                    .collect()
            } else if self.active.is_some() && !info.pv.is_empty() {
                vec![(info.pv[0], fmt_score(info.score))]
            } else {
                Vec::new()
            };
            if !arrows.is_empty() {
                let best = arrows[0].0;
                let tint = Color32::from_rgba_unmultiplied(80, 200, 80, 80);
                for s in [best.from(), best.to()] {
                    let r = Rect::from_min_size(to_xy(s), Vec2::splat(sq_px));
                    painter.rect_filled(r, 0.0, tint);
                }
                for (i, (m, num)) in arrows.iter().enumerate() {
                    let a = to_xy(m.from()) + Vec2::splat(sq_px / 2.0);
                    let b = to_xy(m.to()) + Vec2::splat(sq_px / 2.0);
                    let dir = b - a;
                    let width = if i == 0 { 6.0 } else { 3.0 };
                    let alpha = if i == 0 { 220 } else { 150 };
                    painter.arrow(
                        a + dir * 0.12,
                        dir * 0.76,
                        Stroke::new(width, Color32::from_rgba_unmultiplied(90, 210, 90, alpha)),
                    );
                    // eval label near the from-square
                    let label_pos = a + dir * 0.30;
                    let galley = painter.layout_no_wrap(
                        num.clone(),
                        FontId::monospace(13.0),
                        Color32::from_rgb(20, 20, 20),
                    );
                    let pad = Vec2::splat(2.0);
                    let text_rect = galley.rect.translate(label_pos.to_vec2() - galley.rect.center().to_vec2());
                    painter.rect_filled(text_rect.expand2(pad), 3.0, Color32::from_rgba_unmultiplied(230, 230, 230, 210));
                    painter.galley(text_rect.min, galley, Color32::from_rgb(20, 20, 20));
                }
            }
        }

        // click handling: press selects (or immediate move), release completes.
        if resp.hovered() && ui.input(|i| i.pointer.primary_pressed()) {
            if let Some(p) = ui.input(|i| i.pointer.press_origin()) {
                let s = sq_at(p);
                if s != usize::MAX {
                    self.press_square(s);
                }
            }
        }
        if resp.drag_stopped() || resp.clicked() {
            if let Some(p) = resp.interact_pointer_pos() {
                let s = sq_at(p);
                if s != usize::MAX {
                    self.release_square(s);
                }
            }
        }
    }

    fn press_square(&mut self, sq: usize) {
        if !self.human_has_turn() || self.game_over.is_some() {
            return;
        }
        if self.nav < self.played.len() {
            return;
        }
        // press on a target of the selected piece: move immediately
        if let Some(sel) = self.selected {
            if let Some(m) = self.targets.iter().copied().find(|m| m.to() == sq) {
                self.human_move(m);
                return;
            }
            if sel == sq {
                self.selected = None;
                self.targets.clear();
                return;
            }
        }
        // select own piece
        if let Some((c, _)) = self.pos.piece_at(sq) {
            if c == self.pos.side {
                self.selected = Some(sq);
                let legal = generate_legal(&self.pos);
                self.targets = legal.moves[..legal.len]
                    .iter()
                    .copied()
                    .filter(|m| m.from() == sq)
                    .collect();
                return;
            }
        }
        self.selected = None;
        self.targets.clear();
    }

    fn release_square(&mut self, sq: usize) {
        if !self.human_has_turn() || self.game_over.is_some() {
            return;
        }
        if self.nav < self.played.len() {
            return;
        }
        if self.selected.is_some() {
            if let Some(m) = self.targets.iter().copied().find(|m| m.to() == sq) {
                self.human_move(m);
            }
        }
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("chess-net");

        ui.horizontal(|ui| {
            if ui.button("Nouvelle partie").clicked() {
                self.new_game(GameKind::New);
            }
            if ui.button("Annuler").clicked() {
                self.undo();
            }
            if ui.button("↔").on_hover_text("Retourner l'échiquier").clicked() {
                self.flip = !self.flip;
            }
        });
        ui.horizontal(|ui| {
            if ui.selectable_label(self.analyzing, "Analyse").clicked() {
                self.analyzing = !self.analyzing;
                self.stop_search();
                self.gen += 1;
                if self.analyzing {
                    self.analysis_pending = true;
                    self.status = "Mode analyse : jouez un coup des deux côtés".to_owned();
                } else {
                    self.status = String::new();
                }
            }
            if ui.button("Arrêter").clicked() {
                self.stop_search();
            }
            if ui.button("Réfléchir").clicked() && self.game_over.is_none() {
                self.trigger_engine(false);
            }
        });

        ui.separator();
        ui.label("Mode");
        ComboBox::from_id_salt("mode")
            .selected_text(match self.mode {
                Mode::HumanEngine => "Humain (Blancs) vs Moteur",
                Mode::HumanHuman => "Humain vs Humain",
                Mode::EngineEngine => "Moteur vs Moteur",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.mode, Mode::HumanEngine, "Humain (Blancs) vs Moteur");
                ui.selectable_value(&mut self.mode, Mode::HumanHuman, "Humain vs Humain");
                ui.selectable_value(&mut self.mode, Mode::EngineEngine, "Moteur vs Moteur");
            });
        ui.add(egui::Slider::new(&mut self.movetime_ms, 50..=5000).text("temps/coup (ms)"));
        ui.label("Recherche");
        ComboBox::from_id_salt("search")
            .selected_text(match self.search_kind {
                SearchKind::AlphaBeta => "Alpha-beta (PVS)",
                SearchKind::Mcts => "MCTS (NN)",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.search_kind, SearchKind::AlphaBeta, "Alpha-beta (PVS)");
                ui.selectable_value(&mut self.search_kind, SearchKind::Mcts, "MCTS (NN)");
            });
        if self.search_kind == SearchKind::Mcts {
            ui.add(egui::Slider::new(&mut self.mcts_threads, 0..=32).text("threads MCTS (0 = auto)"));
            ui.add(egui::Slider::new(&mut self.mcts_analysis_ms, 100..=60000).text("temps analyse MCTS (ms)"));
        }

        ui.separator();
        ui.label("Évaluation");
        ui.horizontal(|ui| {
            ui.radio_value(&mut self.eval_kind, EvalKind::PeSTO, "PeSTO");
            ui.radio_value(&mut self.eval_kind, EvalKind::NN, "NN");
        });
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut self.nn_file).desired_width(150.0));
            if ui.button("Charger").clicked() {
                match nn::load(&self.nn_file) {
                    Ok(()) => {
                        self.nn_error = None;
                        self.status = format!("NN chargée : {}", self.nn_file);
                    }
                    Err(e) => self.nn_error = Some(format!("NN : {e}")),
                }
            }
        });
        if let Some(e) = &self.nn_error {
            ui.colored_label(Color32::from_rgb(230, 90, 90), e);
        }
        if self.eval_kind == EvalKind::NN && !nn::is_loaded() {
            ui.colored_label(Color32::from_rgb(230, 170, 60), "Aucune NN chargée — PeSTO utilisé");
            self.eval_kind = EvalKind::PeSTO;
        }
        if self.eval_kind == EvalKind::NN {
            self.eval_fn = nn::evaluate_loaded;
        } else {
            self.eval_fn = evaluate;
        }
        self.static_eval_panel(ui);

        ui.separator();
        ui.label("Position (FEN)");
        ui.add(egui::TextEdit::singleline(&mut self.fen_input).desired_width(220.0));
        if ui.button("Charger la position").clicked() {
            self.new_game(GameKind::Setup);
        }

        ui.separator();
        ui.label("Moteur");
        if self.analyzing {
            ui.colored_label(Color32::from_rgb(120, 200, 255), "mode analyse actif");
        }
        let info = self.live.lock().unwrap().clone_into_info();
        let frac = if info.mcts {
            info.value
        } else {
            ((info.score as f32 + 6000.0) / 12000.0).clamp(0.0, 1.0)
        };
        let (rect, _) = ui.allocate_exact_size(egui::vec2(200.0, 14.0), egui::Sense::hover());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 3.0, Color32::from_gray(40));
        if frac > 0.0 {
            painter.rect_filled(
                egui::Rect::from_min_size(rect.min, egui::vec2(rect.width() * frac, rect.height())),
                3.0,
                Color32::WHITE,
            );
        }
        painter.rect_stroke(rect, 3.0, egui::Stroke::new(1.0, Color32::from_gray(110)), egui::StrokeKind::Inside);
        if self.active.is_some() {
            ui.colored_label(Color32::from_rgb(120, 220, 120), "● recherche en cours");
        } else {
            ui.label("○ au repos");
        }
        if info.mcts {
            ui.horizontal(|ui| {
                ui.label(format!("plouts {}", fmt_u64(info.playouts)));
                ui.label(format!("{:.1} s", info.time_ms as f32 / 1000.0));
            });
            ui.label(format!("{:.0} plouts/s", nps(info.playouts, info.time_ms)));
            ui.label(format!("P(gagne) {:.1}%", info.value * 100.0));
            let q = 2.0 * info.value - 1.0;
            let (w, d, l) = wdl_from_q(q);
            ui.label(format!("WDL {:.1}% / {:.1}% / {:.1}%", w * 100.0, d * 100.0, l * 100.0));
            if info.mates + info.draws > 0 {
                ui.label(format!("terminaux: {} mat, {} nulle", info.mates, info.draws));
            }
            ui.label(format!("score {}", fmt_score(info.score)));
        } else {
            ui.horizontal(|ui| {
                ui.label(format!("prof {}", info.depth));
                ui.label(format!("nœuds {}", fmt_u64(info.nodes)));
                ui.label(format!("{:.1} s", info.time_ms as f32 / 1000.0));
            });
            ui.label(format!("{:.0} nœuds/s", nps(info.nodes, info.time_ms)));
            ui.label(format!("score {}", fmt_score(info.score)));
        }

        if !info.lines.is_empty() {
            ui.separator();
            ui.label("Variantes");
            egui::ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                for (i, line) in info.lines.iter().enumerate() {
                    let num = if info.mcts {
                        let total = info.lines.iter().map(|l| l.visits as u64).sum::<u64>().max(1);
                        format!("{:.0}%", line.visits as f64 * 100.0 / total as f64)
                    } else {
                        fmt_score(line.score)
                    };
                    ui.horizontal(|ui| {
                        ui.monospace(format!("{}.", i + 1));
                        ui.monospace(to_san(&self.pos, line.mv));
                        ui.monospace(num);
                    });
                    let pv_text = san_line(self.pos, &line.pv);
                    if !line.pv.is_empty() {
                        ui.monospace(pv_text);
                    }
                }
            });
        }

        ui.separator();
        let vpos = self.view_pos();
        ui.label(format!(
            "Au trait : {}",
            if vpos.side == 0 { "Blancs" } else { "Noirs" }
        ));
        if let Some(go) = &self.game_over {
            ui.colored_label(Color32::from_rgb(230, 90, 90), go);
        }
        if !self.status.is_empty() {
            ui.label(&self.status);
        }
        ui.monospace(vpos.to_fen());

        ui.separator();
        ui.label(format!("Coups joués ({})", self.played.len()));
        if self.nav < self.played.len() {
            ui.colored_label(
                Color32::from_rgb(230, 170, 60),
                format!("navigation — coup {}/{} (revenir à la fin pour jouer)", self.nav, self.played.len()),
            );
        }
        if !self.played.is_empty() {
            ui.horizontal(|ui| {
                if ui.button("<<").clicked() {
                    self.nav = 0;
                }
                if ui.button("<").clicked() {
                    self.nav = self.nav.saturating_sub(1);
                }
                if ui.button(">").clicked() {
                    self.nav = (self.nav + 1).min(self.played.len());
                }
                if ui.button(">>").clicked() {
                    self.nav = self.played.len();
                }
            });
            if let Some(start) = self.history.first().copied() {
                egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                    let mut p = start;
                    for (i, m) in self.played.iter().enumerate() {
                        let san = to_san(&p, *m);
                        let text = if p.side == WHITE {
                            format!("{}. {san}", p.fullmove)
                        } else {
                            san
                        };
                        let sel = self.nav == i + 1;
                        if ui.selectable_label(sel, text).clicked() {
                            self.nav = i + 1;
                        }
                        p.make_move(*m);
                    }
                });
            }
        }
    }
}

impl LiveInfo {
    fn clone_into_info(&self) -> LiveInfo {
        LiveInfo {
            mcts: self.mcts,
            playouts: self.playouts,
            value: self.value,
            depth: self.depth,
            score: self.score,
            nodes: self.nodes,
            time_ms: self.time_ms,
            mates: self.mates,
            draws: self.draws,
            pv: self.pv.clone(),
            lines: self.lines.clone(),
        }
    }
}

fn fmt_score(s: i32) -> String {
    if s >= MATE - 200 {
        format!("mat en {}", (MATE - s + 1) / 2)
    } else if s <= -MATE + 200 {
        format!("mat en -{}", (MATE + s + 1) / 2)
    } else {
        format!("{:.2}", s as f32 / 100.0)
    }
}

fn nps(n: u64, time_ms: u64) -> f64 {
    if time_ms == 0 {
        0.0
    } else {
        n as f64 * 1000.0 / time_ms as f64
    }
}

/// Numbered SAN line for `moves` played from `pos`, e.g. "1. e4 e5 2. Nf3 Nc6".
fn san_line(mut pos: Position, moves: &[Move]) -> String {
    let mut out = Vec::new();
    for m in moves {
        let san = to_san(&pos, *m);
        let prefix = if pos.side == WHITE {
            format!("{}. ", pos.fullmove)
        } else if out.is_empty() {
            format!("{}... ", pos.fullmove)
        } else {
            String::new()
        };
        out.push(format!("{prefix}{san}"));
        pos.make_move(*m);
    }
    out.join(" ")
}

fn fmt_u64(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f32 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f32 / 1_000.0)
    } else {
        n.to_string()
    }
}

const GLYPHS: [char; 6] = ['♟', '♞', '♝', '♜', '♛', '♚'];

fn play_move_sound() {
    #[cfg(windows)]
    unsafe {
        use windows_sys::Win32::System::Diagnostics::Debug::MessageBeep;
        // 0xFFFFFFFF = default beep (plays the default sound)
        MessageBeep(0xFFFFFFFF);
    }
}

fn draw_piece(painter: &egui::Painter, center: Pos2, pt: usize, white: bool, font_size: f32) {
    let glyph = GLYPHS[pt];
    let font = FontId::proportional(font_size);
    let (fg, outline) = if white {
        (Color32::WHITE, Color32::from_gray(70))
    } else {
        (Color32::from_rgb(28, 28, 32), Color32::from_rgb(235, 235, 235))
    };
    let shadow = Pos2::new(center.x + 2.0, center.y + 2.0);
    painter.text(shadow, Align2::CENTER_CENTER, glyph, font.clone(), outline);
    painter.text(center, Align2::CENTER_CENTER, glyph, font, fg);
}

impl eframe::App for ChessApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.engine_move_arrived();
        self.maybe_engine_plays();

        egui::Panel::left("side").show(ui, |ui| {
            self.side_panel(ui);
        });
        ui.vertical_centered(|ui| {
            self.draw_board(ui);
        });
        ui.ctx().request_repaint();
    }
}

fn main() -> eframe::Result {
    engine::init();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([BOARD_PX + 300.0, BOARD_PX + 120.0])
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
                    fonts.font_data.insert("ui-sym".to_owned(), egui::FontData::from_owned(bytes).into());
                    if let Some(fam) = fonts.families.get_mut(&egui::FontFamily::Proportional) {
                        fam.push("ui-sym".to_owned());
                    }
                    break;
                }
            }
            cc.egui_ctx.set_fonts(fonts);
            Ok(Box::new(ChessApp::new()))
        }),
    )
}