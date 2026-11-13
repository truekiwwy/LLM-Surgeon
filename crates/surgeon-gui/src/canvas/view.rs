//! The canvas: an infinite plane drawn on its own layer under a pan / zoom transform (so the
//! card widgets scale with it), a dot grid, Bézier wires from output to input ports, the
//! cards and a minimap.
//!
//! Wheel or Ctrl + wheel zooms on the pointer; middle drag or Space + drag pans; right click
//! adds a card; dragging a card moves it, dragging its output port wires it.

use std::collections::HashMap;
use std::ops::RangeInclusive;

use eframe::egui::emath::TSTransform;
use eframe::egui::epaint::CubicBezierShape;
use eframe::egui::{
    self, Align, Align2, Color32, ComboBox, CornerRadius, DragValue, FontId, Frame, Grid, Id, Key,
    LayerId, Layout, Order, PointerButton, Pos2, Rect, RichText, Sense, Stroke, StrokeKind,
    TextEdit, UiBuilder, Vec2,
};
use weight_surgeon::bench::score::Backend;
use weight_surgeon::patch::{Axis, OperationKind, Projection};

use super::{summary, verdict, Card, CardKind, Verdict, Workspace, EXTENSION, KINDS};
use crate::runner::Finished;
use crate::theme::{self, mono};

pub const CARD_WIDTH: f32 = 272.0;
const ZOOM: RangeInclusive<f32> = 0.2..=2.5;
const GRID: f32 = 24.0;
/// Ports sit on the header line.
const PORT_Y: f32 = 14.0;

pub struct Canvas {
    pub ws: Workspace,
    /// Path of the workspace file.
    pub file: String,
    /// Last run of each card.
    pub results: HashMap<u32, Finished>,
    pub selected: Option<u32>,
    pub message: Option<String>,
    /// Card rectangles of the last frame, canvas coordinates (wires, drops, minimap).
    rects: HashMap<u32, Rect>,
    /// A wire being dragged out of this card.
    wiring: Option<u32>,
    menu_at: Pos2,
    fit: bool,
}

impl Default for Canvas {
    fn default() -> Self {
        Self {
            ws: Workspace::default(),
            file: format!("studio{EXTENSION}"),
            results: HashMap::new(),
            selected: None,
            message: None,
            rects: HashMap::new(),
            wiring: None,
            menu_at: Pos2::ZERO,
            fit: false,
        }
    }
}

/// What the canvas asks the app to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    Run(u32),
    OpenInEditor(u32),
    ShowGraph(u32),
    Rollback,
}

/// The app state a card shows.
pub struct Status<'a> {
    pub running: Option<u32>,
    pub busy: bool,
    pub can_rollback: bool,
    /// One line about the NDJSON stream, when one is followed.
    pub live: Option<&'a str>,
}

/// Structural edits, applied after the cards are drawn.
enum Edit {
    Duplicate(u32),
    Fork(u32),
    Unplug(u32),
    Remove(u32),
}

/// Zooms by `factor` keeping the canvas point under `at` (relative to the canvas corner)
/// where it is.
pub fn zoom_at(ws: &mut Workspace, at: Vec2, factor: f32) {
    let zoom = (ws.zoom * factor).clamp(*ZOOM.start(), *ZOOM.end());
    let pan = Vec2::from(ws.pan);
    let point = (at - pan) / ws.zoom;
    ws.pan = (at - point * zoom).into();
    ws.zoom = zoom;
}

fn out_port(r: Rect) -> Pos2 {
    Pos2::new(r.right(), r.top() + PORT_Y)
}

fn in_port(r: Rect) -> Pos2 {
    Pos2::new(r.left(), r.top() + PORT_Y)
}

fn wire(painter: &egui::Painter, from: Pos2, to: Pos2, color: Color32) {
    let reach = ((to.x - from.x).abs() * 0.5).max(40.0);
    let points = [
        from,
        from + Vec2::new(reach, 0.0),
        to - Vec2::new(reach, 0.0),
        to,
    ];
    painter.add(CubicBezierShape::from_points_stroke(
        points,
        false,
        Color32::TRANSPARENT,
        Stroke::new(1.5, color),
    ));
}

pub fn show(ui: &mut egui::Ui, c: &mut Canvas, status: &Status) -> Vec<Action> {
    let mut actions = Vec::new();
    toolbar(ui, c, status);

    let (outer, bg) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
    let origin = outer.min.to_vec2();
    if c.fit {
        fit(c, outer.size());
        c.fit = false;
    }

    // Pan and zoom.
    let space = ui.input(|i| i.key_down(Key::Space));
    if bg.dragged_by(PointerButton::Middle) || (space && bg.dragged_by(PointerButton::Primary)) {
        c.ws.pan = (Vec2::from(c.ws.pan) + bg.drag_delta()).into();
    }
    let layer = LayerId::new(ui.layer_id().order, ui.id().with("canvas-layer"));
    let pointer = ui.input(|i| i.pointer.hover_pos());
    if let Some(p) = pointer.filter(|p| outer.contains(*p)) {
        let ours = ui
            .ctx()
            .layer_id_at(p)
            .is_none_or(|l| l == layer || l == ui.layer_id());
        if ours {
            // Ctrl + wheel arrives as zoom_delta, the bare wheel as a scroll.
            let (pinch, scroll) = ui.input(|i| (i.zoom_delta(), i.smooth_scroll_delta.y));
            let factor = pinch * (scroll * 0.0015).exp();
            if factor != 1.0 {
                zoom_at(&mut c.ws, p.to_vec2() - origin, factor);
            }
        }
    }
    let to_global = TSTransform::new(origin + Vec2::from(c.ws.pan), c.ws.zoom);
    let view = to_global.inverse() * outer;

    // Background and dot grid, in screen space.
    let painter = ui.painter_at(outer);
    painter.rect_filled(outer, CornerRadius::ZERO, theme::BG);
    let step = GRID * c.ws.zoom;
    if step >= 10.0 {
        let pan = Vec2::from(c.ws.pan);
        let start = outer.min + Vec2::new(pan.x.rem_euclid(step), pan.y.rem_euclid(step));
        let mut y = start.y;
        while y < outer.bottom() {
            let mut x = start.x;
            while x < outer.right() {
                painter.rect_filled(
                    Rect::from_min_size(Pos2::new(x, y), Vec2::splat(1.0)),
                    CornerRadius::ZERO,
                    theme::BORDER,
                );
                x += step;
            }
            y += step;
        }
    }

    // The plane itself.
    ui.ctx().set_sublayer(ui.layer_id(), layer);
    ui.ctx().set_transform_layer(layer, to_global);
    let mut plane = ui.new_child(UiBuilder::new().layer_id(layer).max_rect(view));
    plane.set_clip_rect(view);

    for w in &c.ws.wires {
        if let (Some(a), Some(b)) = (c.rects.get(&w.from), c.rects.get(&w.to)) {
            let lit = c.selected.is_some_and(|s| s == w.from || s == w.to);
            let color = if lit { theme::SELECTION } else { theme::DIM };
            wire(plane.painter(), out_port(*a), in_port(*b), color);
        }
    }

    let mut edits = Vec::new();
    let ids: Vec<u32> = c.ws.cards.iter().map(|card| card.id).collect();
    for id in ids {
        let Some(mut card) = c.ws.card(id).cloned() else {
            continue;
        };
        let rect = card_ui(&mut plane, c, &mut card, status, &mut actions, &mut edits);
        c.rects.insert(id, rect);
        if let Some(slot) = c.ws.card_mut(id) {
            *slot = card;
        }
    }

    // A wire being dragged, dropped on the card under the pointer.
    if let (Some(from), Some(p)) = (c.wiring, ui.input(|i| i.pointer.latest_pos())) {
        let p = to_global.inverse() * p;
        if let Some(r) = c.rects.get(&from) {
            wire(plane.painter(), out_port(*r), p, theme::SELECTION);
        }
        if ui.input(|i| i.pointer.any_released()) {
            let target = c
                .rects
                .iter()
                .find(|(_, r)| r.contains(p))
                .map(|(id, _)| *id);
            if let Some(to) = target {
                c.message = c.ws.connect(from, to).err();
            }
            c.wiring = None;
        }
    }

    for edit in edits {
        match edit {
            Edit::Duplicate(id) => c.selected = c.ws.duplicate(id),
            Edit::Fork(id) => c.selected = c.ws.fork(id),
            Edit::Unplug(id) => c.ws.wires.retain(|w| w.to != id),
            Edit::Remove(id) => remove(c, id),
        }
    }

    if bg.secondary_clicked() {
        if let Some(p) = bg.interact_pointer_pos() {
            c.menu_at = to_global.inverse() * p;
        }
    }
    bg.context_menu(|ui| {
        ui.label(RichText::new("AJOUTER").small().color(theme::DIM));
        for (k, name) in KINDS.iter().enumerate() {
            if ui.button(*name).clicked() {
                let id = c.ws.add(CardKind::new(k), [c.menu_at.x, c.menu_at.y]);
                c.selected = Some(id);
                ui.close();
            }
        }
    });
    if bg.clicked() {
        c.selected = None;
    }
    let typing = ui.ctx().egui_wants_keyboard_input();
    if !typing && ui.input(|i| i.key_pressed(Key::Delete)) {
        if let Some(id) = c.selected {
            remove(c, id);
        }
    }

    if c.ws.cards.is_empty() {
        painter.text(
            outer.center(),
            Align2::CENTER_CENTER,
            "clic droit : ajouter une carte (Model, Extract, Patch, Graph Diff, Eval / Gate)",
            FontId::monospace(11.5),
            theme::DIM,
        );
    } else {
        minimap(ui, c, outer, view);
    }
    painter.text(
        outer.left_bottom() + Vec2::new(10.0, -8.0),
        Align2::LEFT_BOTTOM,
        "molette : zoom   clic milieu ou Espace + glisser : pan   clic droit : ajouter   Suppr : supprimer",
        FontId::proportional(10.5),
        theme::DIM,
    );
    actions
}

fn remove(c: &mut Canvas, id: u32) {
    c.ws.remove(id);
    c.rects.remove(&id);
    c.results.remove(&id);
    if c.selected == Some(id) {
        c.selected = None;
    }
}

fn toolbar(ui: &mut egui::Ui, c: &mut Canvas, status: &Status) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("workspace").color(theme::DIM));
        ui.add(TextEdit::singleline(&mut c.file).desired_width(240.0));
        if ui.button("Ouvrir").clicked() {
            match Workspace::load(std::path::Path::new(c.file.trim())) {
                Ok(ws) => {
                    *c = Canvas {
                        ws,
                        file: std::mem::take(&mut c.file),
                        fit: false,
                        ..Default::default()
                    };
                }
                Err(e) => c.message = Some(e),
            }
        }
        if ui.button("Enregistrer").clicked() {
            c.message = c.ws.save(std::path::Path::new(c.file.trim())).err();
        }
        if ui.button("Recentrer").clicked() {
            c.fit = true;
        }
        ui.label(mono(format!("{:>3.0} %", c.ws.zoom * 100.0)).color(theme::DIM));
        if let Some(live) = status.live {
            ui.label(mono(live).color(theme::AGENT));
        }
        if let Some(m) = &c.message {
            ui.label(RichText::new(m).small().color(theme::BAD));
        }
    });
}

/// Frames every card.
fn fit(c: &mut Canvas, size: Vec2) {
    let Some(bounds) = c.rects.values().copied().reduce(|a, b| a.union(b)) else {
        c.ws.pan = [0.0; 2];
        c.ws.zoom = 1.0;
        return;
    };
    let zoom = (size.x / (bounds.width() + 80.0))
        .min(size.y / (bounds.height() + 80.0))
        .clamp(*ZOOM.start(), 1.0);
    c.ws.zoom = zoom;
    c.ws.pan = (size / 2.0 - bounds.center().to_vec2() * zoom).into();
}

fn minimap(ui: &egui::Ui, c: &mut Canvas, outer: Rect, view: Rect) {
    let Some(cards) = c.rects.values().copied().reduce(|a, b| a.union(b)) else {
        return;
    };
    let bounds = cards.union(view);
    let mini = Rect::from_min_size(
        outer.right_bottom() - Vec2::new(172.0, 120.0),
        Vec2::new(160.0, 108.0),
    );
    let scale = (mini.width() / bounds.width()).min(mini.height() / bounds.height());
    let map = |p: Pos2| mini.center() + (p - bounds.center()) * scale;
    let painter = ui
        .ctx()
        .layer_painter(LayerId::new(Order::Foreground, Id::new("canvas-minimap")))
        .with_clip_rect(outer);
    painter.rect_filled(mini, CornerRadius::same(2), theme::PANEL);
    painter.rect_stroke(
        mini,
        CornerRadius::same(2),
        Stroke::new(1.0, theme::BORDER),
        StrokeKind::Inside,
    );
    for (id, r) in &c.rects {
        let color = match c.selected == Some(*id) {
            true => theme::SELECTION,
            false => theme::DIM.gamma_multiply(0.6),
        };
        painter.rect_filled(
            Rect::from_min_max(map(r.min), map(r.max)),
            CornerRadius::ZERO,
            color,
        );
    }
    painter.rect_stroke(
        Rect::from_min_max(map(view.min), map(view.max)),
        CornerRadius::ZERO,
        Stroke::new(1.0, theme::TEXT.gamma_multiply(0.5)),
        StrokeKind::Inside,
    );
    let response = ui.interact(mini, Id::new("canvas-minimap-nav"), Sense::click_and_drag());
    if let Some(p) = response.interact_pointer_pos() {
        if response.clicked() || response.dragged() {
            let target = bounds.center() + (p - mini.center()) / scale;
            c.ws.pan = (outer.size() / 2.0 - target.to_vec2() * c.ws.zoom).into();
        }
    }
}

fn text(ui: &mut egui::Ui, value: &mut String, hint: &str) {
    ui.add(
        TextEdit::singleline(value)
            .hint_text(hint)
            .desired_width(CARD_WIDTH - 110.0),
    );
}

fn combo<T: PartialEq + Copy>(
    ui: &mut egui::Ui,
    salt: (&str, u32),
    value: &mut T,
    options: &[(T, &str)],
) {
    let current = options
        .iter()
        .find(|(v, _)| v == value)
        .map_or("?", |(_, label)| label);
    ComboBox::from_id_salt(salt)
        .selected_text(mono(current))
        .width(CARD_WIDTH - 110.0)
        .show_ui(ui, |ui| {
            for (v, label) in options {
                ui.selectable_value(value, *v, mono(*label));
            }
        });
}

const PROJECTIONS: [(Projection, &str); 5] = [
    (Projection::Gate, "gate"),
    (Projection::Up, "up"),
    (Projection::Down, "down"),
    (Projection::AttnOut, "o_proj"),
    (Projection::EmbedTokens, "embed"),
];

/// Draws one card; returns its rectangle (canvas coordinates).
fn card_ui(
    plane: &mut egui::Ui,
    c: &mut Canvas,
    card: &mut Card,
    status: &Status,
    actions: &mut Vec<Action>,
    edits: &mut Vec<Edit>,
) -> Rect {
    let id = card.id;
    // The card body moves and selects it; registered first, so the fields stay on top.
    if let Some(last) = c.rects.get(&id) {
        let body = plane.interact(*last, Id::new(("card-body", id)), Sense::click_and_drag());
        if body.dragged_by(PointerButton::Primary) {
            let d = body.drag_delta();
            card.pos = [card.pos[0] + d.x, card.pos[1] + d.y];
        }
        if body.clicked() || body.drag_started() {
            c.selected = Some(id);
        }
    }
    let selected = c.selected == Some(id);
    let max = Rect::from_min_size(Pos2::from(card.pos), Vec2::new(CARD_WIDTH, 4000.0));
    let mut ui = plane.new_child(UiBuilder::new().max_rect(max).id_salt(("card", id)));
    let border = if selected {
        theme::SELECTION
    } else {
        theme::BORDER
    };
    let result = c.results.get(&id);
    let frame = Frame::new()
        .fill(theme::PANEL)
        .stroke(Stroke::new(1.0, border))
        .corner_radius(CornerRadius::same(3))
        .inner_margin(6);
    let response = frame.show(&mut ui, |ui| {
        ui.set_width(CARD_WIDTH - 12.0);
        ui.horizontal(|ui| {
            ui.label(mono(card.kind.title()).strong());
            ui.label(mono(format!("#{id}")).color(theme::DIM));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.menu_button(mono("..."), |ui| {
                    if ui.button("Dupliquer").clicked() {
                        edits.push(Edit::Duplicate(id));
                    }
                    if ui.button("Forker la branche").clicked() {
                        edits.push(Edit::Fork(id));
                    }
                    if ui.button("Débrancher les entrées").clicked() {
                        edits.push(Edit::Unplug(id));
                    }
                    if ui.button("Supprimer").clicked() {
                        edits.push(Edit::Remove(id));
                    }
                });
                let run = ui
                    .add_enabled(!status.busy, egui::Button::new("Run"))
                    .on_hover_text("exécute les cartes en amont puis celle-ci");
                if run.clicked() {
                    actions.push(Action::Run(id));
                }
            });
        });
        ui.separator();
        fields(ui, id, &mut card.kind, status, result, actions);
        state_lines(ui, id, &card.kind, status, result);
    });
    let rect = response.response.rect;

    // Ports: input on the left for cards that take one, output on the right.
    let painter = plane.painter();
    let takes_input = !matches!(card.kind, CardKind::Model(_));
    if takes_input {
        painter.circle(in_port(rect), 4.0, theme::BG, Stroke::new(1.0, theme::DIM));
    }
    if !matches!(card.kind, CardKind::Gate(_)) {
        let port = Rect::from_center_size(out_port(rect), Vec2::splat(14.0));
        let handle = plane.interact(port, Id::new(("card-out", id)), Sense::drag());
        let color = if handle.hovered() || c.wiring == Some(id) {
            theme::SELECTION
        } else {
            theme::DIM
        };
        plane
            .painter()
            .circle(out_port(rect), 4.0, theme::PANEL, Stroke::new(1.0, color));
        if handle.drag_started() {
            c.wiring = Some(id);
        }
    }
    rect
}

fn fields(
    ui: &mut egui::Ui,
    id: u32,
    kind: &mut CardKind,
    status: &Status,
    result: Option<&Finished>,
    actions: &mut Vec<Action>,
) {
    Grid::new(("card-fields", id))
        .num_columns(2)
        .spacing([6.0, 3.0])
        .show(ui, |ui| {
            let row = |ui: &mut egui::Ui, key: &str, add: &mut dyn FnMut(&mut egui::Ui)| {
                ui.label(RichText::new(key).small().color(theme::DIM));
                add(ui);
                ui.end_row();
            };
            match kind {
                CardKind::Model(m) => {
                    row(ui, "modèle", &mut |ui| {
                        text(ui, &mut m.path, "dossier ou fichier")
                    });
                }
                CardKind::Extract(e) => {
                    row(ui, "prompts +", &mut |ui| {
                        text(ui, &mut e.pos_prompts, "pos.json")
                    });
                    row(ui, "prompts -", &mut |ui| {
                        text(ui, &mut e.neg_prompts, "neg.json")
                    });
                    row(ui, "couche", &mut |ui| {
                        ui.add(DragValue::new(&mut e.layer).range(0..=512));
                    });
                    row(ui, "méthode", &mut |ui| {
                        let registry = weight_surgeon::plugin::Registry::with_builtin();
                        ComboBox::from_id_salt(("method", id))
                            .selected_text(mono(&e.method))
                            .width(CARD_WIDTH - 110.0)
                            .show_ui(ui, |ui| {
                                for (m, _) in registry.list() {
                                    ui.selectable_value(
                                        &mut e.method,
                                        m.id().to_owned(),
                                        mono(m.id()),
                                    );
                                }
                            });
                    });
                    row(ui, "rang k", &mut |ui| {
                        ui.add(DragValue::new(&mut e.rank).range(1..=64));
                    });
                    row(ui, "sortie", &mut |ui| text(ui, &mut e.out, ".safetensors"));
                }
                CardKind::Patch(p) => {
                    row(ui, "opération", &mut |ui| {
                        combo(
                            ui,
                            ("op", id),
                            &mut p.operation,
                            &[
                                (OperationKind::Subtract, "subtract"),
                                (OperationKind::Add, "add"),
                                (OperationKind::ProjectOut, "project_out"),
                            ],
                        );
                    });
                    let layers_ok = p
                        .layers
                        .parse::<weight_surgeon::patch::LayerSelection>()
                        .is_ok();
                    row(ui, "couches", &mut |ui| {
                        let edit = TextEdit::singleline(&mut p.layers)
                            .desired_width(CARD_WIDTH - 110.0)
                            .text_color(if layers_ok { theme::TEXT } else { theme::BAD });
                        ui.add(edit);
                    });
                    row(ui, "projections", &mut |ui| {
                        ui.horizontal_wrapped(|ui| {
                            ui.set_max_width(CARD_WIDTH - 110.0);
                            for (proj, label) in PROJECTIONS {
                                let mut on = p.projections.contains(&proj);
                                if ui.toggle_value(&mut on, mono(label).small()).changed() {
                                    match on {
                                        true => p.projections.push(proj),
                                        false => p.projections.retain(|q| *q != proj),
                                    }
                                    p.projections.sort();
                                }
                            }
                        });
                    });
                    row(ui, "alpha", &mut |ui| {
                        ui.add(
                            DragValue::new(&mut p.alpha)
                                .speed(0.01)
                                .range(-100.0..=100.0),
                        );
                    });
                    row(ui, "axe", &mut |ui| {
                        combo(
                            ui,
                            ("axis", id),
                            &mut p.axis,
                            &[(Axis::Cols, "cols"), (Axis::Rows, "rows")],
                        );
                    });
                    row(ui, "vecteur", &mut |ui| {
                        text(ui, &mut p.vector_file, "vide : carte Extract")
                    });
                    row(ui, "tenseur", &mut |ui| {
                        text(ui, &mut p.vector_tensor, "direction")
                    });
                    row(ui, "", &mut |ui| {
                        ui.checkbox(&mut p.backup, "backup vérifié");
                    });
                    row(ui, "", &mut |ui| {
                        ui.checkbox(&mut p.apply, "Run écrit le modèle")
                            .on_hover_text("décoché : Run fait un dry-run, rien n'est écrit");
                    });
                }
                CardKind::GraphDiff(g) => {
                    row(ui, "baseline", &mut |ui| {
                        text(ui, &mut g.baseline, "modèle ou graph.json")
                    });
                    row(ui, "sortie", &mut |ui| text(ui, &mut g.out_dir, "dossier"));
                }
                CardKind::Gate(g) => {
                    row(ui, "prompts", &mut |ui| {
                        text(ui, &mut g.prompts, "harmful.json")
                    });
                    row(ui, "backend", &mut |ui| {
                        combo(
                            ui,
                            ("backend", id),
                            &mut g.backend,
                            &[
                                (Backend::Native, "native"),
                                (Backend::MistralRs, "mistralrs"),
                                (Backend::Hipfire, "hipfire"),
                                (Backend::LlamaServer, "llama-server"),
                            ],
                        );
                    });
                    row(ui, "baseline", &mut |ui| {
                        text(ui, &mut g.baseline, "facultatif")
                    });
                    row(ui, "refus max", &mut |ui| {
                        ui.add(
                            DragValue::new(&mut g.max_refusal)
                                .speed(0.01)
                                .range(0.0..=1.0),
                        );
                    });
                }
            }
        });
    match kind {
        CardKind::Patch(_) => {
            ui.horizontal(|ui| {
                if ui
                    .button("Éditeur")
                    .on_hover_text("ouvre ce patch dans l'éditeur JSON")
                    .clicked()
                {
                    actions.push(Action::OpenInEditor(id));
                }
                let enabled = !status.busy && status.can_rollback;
                if ui
                    .add_enabled(enabled, egui::Button::new("Rollback"))
                    .on_hover_text("restaure les backups du dernier Apply")
                    .clicked()
                {
                    actions.push(Action::Rollback);
                }
            });
        }
        CardKind::GraphDiff(_) => {
            let done = result.is_some_and(|r| r.code == 0);
            if ui
                .add_enabled(done, egui::Button::new("Voir en 3D"))
                .on_hover_text("charge le graph.json et centre la vue 3D sur les nœuds modifiés")
                .clicked()
            {
                actions.push(Action::ShowGraph(id));
            }
        }
        _ => {}
    }
}

fn state_lines(
    ui: &mut egui::Ui,
    id: u32,
    kind: &CardKind,
    status: &Status,
    result: Option<&Finished>,
) {
    if status.running == Some(id) {
        ui.label(mono("en cours...").color(theme::SELECTION));
        return;
    }
    if let (CardKind::GraphDiff(_), Some(live)) = (kind, status.live) {
        ui.label(mono(live).small().color(theme::AGENT));
    }
    let Some(run) = result else {
        return;
    };
    ui.separator();
    match kind {
        CardKind::Gate(g) => {
            let (text, color) = match verdict(run, g.max_refusal) {
                Verdict::Go => ("GO", theme::GOOD),
                Verdict::NoGo => ("NO-GO", theme::BAD),
                Verdict::Error => ("ERREUR", theme::WARN),
            };
            ui.label(
                RichText::new(text)
                    .monospace()
                    .size(16.0)
                    .strong()
                    .color(color),
            );
        }
        _ => {
            let (text, color) = match run.code {
                0 => (format!("ok · {:.1} s", run.seconds), theme::GOOD),
                code => (format!("échec (code {code})"), theme::BAD),
            };
            ui.label(mono(text).small().color(color));
        }
    }
    Grid::new(("card-summary", id))
        .num_columns(2)
        .spacing([6.0, 2.0])
        .show(ui, |ui| {
            for (key, value) in summary(kind, run) {
                ui.label(RichText::new(key).small().color(theme::DIM));
                ui.label(mono(value).small());
                ui.end_row();
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zooming_keeps_the_point_under_the_pointer() {
        let mut ws = Workspace {
            pan: [40.0, -10.0],
            zoom: 0.8,
            ..Default::default()
        };
        let at = Vec2::new(300.0, 200.0);
        let before = (at - Vec2::from(ws.pan)) / ws.zoom;
        zoom_at(&mut ws, at, 1.5);
        assert!((ws.zoom - 1.2).abs() < 1e-6);
        let after = (at - Vec2::from(ws.pan)) / ws.zoom;
        assert!((before - after).length() < 1e-3, "{before:?} {after:?}");
        zoom_at(&mut ws, at, 100.0);
        assert_eq!(ws.zoom, *ZOOM.end());
        zoom_at(&mut ws, at, 1e-6);
        assert_eq!(ws.zoom, *ZOOM.start());
    }
}
