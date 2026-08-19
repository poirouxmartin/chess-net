//! chess-net GUI: playable chessboard + live engine info panel.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use eframe::egui::{
    self, Align2, Color32, ComboBox, FontId, Pos2, Rect, Sense, Stroke, Vec2,
};

use engine::evaluate::{evaluate, MATE};
use engine::move_::Move;
use engine::movegen::generate_legal;
use engine::position::{BLACK, Position, WHITE};
use engine::search::{Limits, SearchIter, Searcher};

const BOARD_PX: f32 = 640.0;
const BOARD_BG: Color32 = Color32::from_rgb(45, 45, 48);

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
enum GameKind {
    New,
    Setup,
}

#[derive(Default)]
struct LiveInfo {
    depth: i32,
    score: i32,
    nodes: u64,
    time_ms: u64,
    pv: Vec<Move>,
}

struct ChessApp {
    pos: Position,
    history: Vec<Position>,
    played: Vec<Move>,
    selected: Option<usize>,
    targets: Vec<Move>,
    last_from: Option<usize>,
    last_to: Option<usize>,
    check_sq: Option<usize>,
    game_over: Option<String>,
    mode: Mode,
    flip: bool,
    movetime_ms: u64,
    eval_kind: EvalKind,
    nn_file: String,
    nn_error: Option<String>,
    fen_input: String,
    status: String,

    gen: u64,
    active: Option<u64>,
    live: Arc<Mutex<LiveInfo>>,
    stop: Arc<AtomicBool>,
    tx: mpsc::Sender<(u64, Move)>,
    rx: mpsc::Receiver<(u64, Move)>,
    eval_fn: engine::search::EvalFn,
}

impl ChessApp {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        Self {
            pos: Position::startpos(),
            history: Vec::new(),
            played: Vec::new(),
            selected: None,
            targets: Vec::new(),
            last_from: None,
            last_to: None,
            check_sq: None,
            game_over: None,
            mode: Mode::HumanEngine,
            flip: false,
            movetime_ms: 1000,
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
        self.selected = None;
        self.targets.clear();
        self.last_from = None;
        self.last_to = None;
        self.game_over = None;
        self.check_sq = None;
        self.active = None;
        let _ = self.rx.try_recv();
        let mut l = self.live.lock().unwrap();
        *l = LiveInfo::default();
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
        self.check_sq = None;
        self.last_from = None;
        self.last_to = None;
        self.active = None;
        let _ = self.rx.try_recv();
        let mut l = self.live.lock().unwrap();
        *l = LiveInfo::default();
        // reculer d'un coup de plus si le moteur vient de jouer et c'est au tour du moteur
        if self.mode == Mode::HumanEngine && self.pos.side == BLACK && !self.history.is_empty() {
            self.pos = self.history.pop().unwrap();
            self.played.pop();
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

    fn trigger_engine(&mut self) {
        if self.active.is_some() || self.game_over.is_some() {
            return;
        }
        let mut pos = self.pos;
        let stop = self.stop.clone();
        let live = self.live.clone();
        let tx = self.tx.clone();
        let eval = self.eval_fn;
        let ms = self.movetime_ms;
        let gen = self.gen;
        self.active = Some(gen);
        *self.live.lock().unwrap() = LiveInfo::default();
        std::thread::spawn(move || {
            let mut searcher = Searcher::new(64);
            let limits = Limits { movetime: Some(ms), ..Default::default() };
            let result = searcher.think_cb(&mut pos, &limits, &stop, eval, Some(&mut |it: &SearchIter| {
                let mut l = live.lock().unwrap();
                l.depth = it.depth;
                l.score = it.score;
                l.nodes = it.nodes;
                l.time_ms = it.time_ms;
                l.pv = it.pv.clone();
            }));
            let _ = tx.send((gen, result.best));
        });
    }

    fn engine_move_arrived(&mut self) {
        if let Ok((gen, m)) = self.rx.try_recv() {
            if self.active != Some(gen) {
                return;
            }
            self.active = None;
            self.apply(m);
        }
    }

    fn apply(&mut self, m: Move) {
        self.history.push(self.pos);
        self.played.push(m);
        self.last_from = Some(m.from());
        self.last_to = Some(m.to());
        self.selected = None;
        self.targets.clear();
        self.pos.make_move(m);
        self.check_sq = self.pos.in_check().then(|| self.pos.king_sq[self.pos.side]);
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
    }

    fn human_has_turn(&self) -> bool {
        match self.mode {
            Mode::HumanHuman => true,
            Mode::HumanEngine => self.pos.side == WHITE,
            Mode::EngineEngine => false,
        }
    }

    fn maybe_engine_plays(&mut self) {
        let engine_turn = match self.mode {
            Mode::HumanEngine => self.pos.side == BLACK,
            Mode::EngineEngine => true,
            Mode::HumanHuman => false,
        };
        if engine_turn && !self.is_searching() && self.game_over.is_none() {
            self.trigger_engine();
        }
    }

    fn draw_board(&mut self, ui: &mut egui::Ui) {
        let sq_px = BOARD_PX / 8.0;
        let (rect, resp) = ui.allocate_exact_size(Vec2::splat(BOARD_PX), Sense::click());
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

        // squares
        for rank in 0..8 {
            for file in 0..8 {
                let s = rank * 8 + file;
                let color = if (rank + file) % 2 == 0 {
                    Color32::from_rgb(0xF0, 0xD9, 0xB5)
                } else {
                    Color32::from_rgb(0xB5, 0x88, 0x63)
                };
                let r = Rect::from_min_size(to_xy(s), Vec2::splat(sq_px));
                painter.rect_filled(r, 0.0, color);
            }
        }

        // highlights: last move
        if let (Some(f), Some(t)) = (self.last_from, self.last_to) {
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
        // check
        if let Some(s) = self.check_sq {
            let c = to_xy(s) + Vec2::splat(sq_px / 2.0);
            painter.circle_stroke(c, sq_px * 0.42, Stroke::new(4.0, Color32::from_rgb(220, 40, 40)));
        }

        // pieces
        let font_size = sq_px * 0.82;
        for sq in 0..64 {
            if let Some((c, pt)) = self.pos.piece_at(sq) {
                let center = to_xy(sq) + Vec2::splat(sq_px / 2.0);
                draw_piece(&painter, center, pt, c == WHITE, font_size);
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

        // click handling
        if let Some(click) = resp.interact_pointer_pos() {
            let s = sq_at(click);
            if s != usize::MAX {
                self.handle_click(s);
            }
        }
    }

    fn handle_click(&mut self, sq: usize) {
        if !self.human_has_turn() || self.game_over.is_some() {
            return;
        }
        // try to move the selected piece there
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
            if ui.button("Arrêter").clicked() {
                self.stop_search();
            }
            if ui.button("Réfléchir").clicked() && self.game_over.is_none() {
                self.trigger_engine();
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

        ui.separator();
        ui.label("Position (FEN)");
        ui.add(egui::TextEdit::singleline(&mut self.fen_input).desired_width(220.0));
        if ui.button("Charger la position").clicked() {
            self.new_game(GameKind::Setup);
        }

        ui.separator();
        ui.label("Moteur");
        let info = self.live.lock().unwrap().clone_into_info();
        if self.active.is_some() {
            ui.colored_label(Color32::from_rgb(120, 220, 120), "● recherche en cours");
            ui.horizontal(|ui| {
                ui.label(format!("prof {}", info.depth));
                ui.label(format!("nœuds {}", fmt_u64(info.nodes)));
                ui.label(format!("{:.1} s", info.time_ms as f32 / 1000.0));
            });
            ui.label(format!("score {}", fmt_score(info.score)));
            ui.monospace(format!("PV  {}", info.pv.iter().map(|m| m.to_uci()).collect::<Vec<_>>().join(" ")));
        } else {
            ui.label("○ au repos");
        }

        ui.separator();
        ui.label(format!(
            "Au trait : {}",
            if self.pos.side == 0 { "Blancs" } else { "Noirs" }
        ));
        if let Some(go) = &self.game_over {
            ui.colored_label(Color32::from_rgb(230, 90, 90), go);
        }
        if !self.status.is_empty() {
            ui.label(&self.status);
        }
        ui.monospace(self.pos.to_fen());

        ui.separator();
        ui.label(format!("Coups joués ({})", self.played.len()));
        egui::ScrollArea::vertical().max_height(180.0).show(ui, |ui| {
            for (i, m) in self.played.iter().enumerate() {
                if i % 2 == 0 {
                    let black = self.played.get(i + 1).map(|b| b.to_uci());
                    ui.monospace(format!(
                        "{:>2}. {}  {}",
                        i / 2 + 1,
                        m.to_uci(),
                        black.as_deref().unwrap_or("")
                    ));
                }
            }
            if self.played.is_empty() {
                ui.weak("—");
            }
        });
    }
}

impl LiveInfo {
    fn clone_into_info(&self) -> LiveInfo {
        LiveInfo {
            depth: self.depth,
            score: self.score,
            nodes: self.nodes,
            time_ms: self.time_ms,
            pv: self.pv.clone(),
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