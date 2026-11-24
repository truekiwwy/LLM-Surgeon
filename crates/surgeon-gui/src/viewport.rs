//! The central 3D canvas: shaded nodes colored by role at their `scene` positions (blended
//! between the column and the helix), edges running from their source's color to their
//! target's, layer marks, orbit / pan / zoom, picking and a compact hover card. The selected
//! node pulses with its name and measures on a card and its edges flow; the layers of the
//! agent's running tool ripple under a banner naming it. A run (live or replayed) draws over
//! the nodes it reached: a glyph per state, never the color alone, and for an apply the |ΔW|
//! of each tensor on the report's color scale.
//!
//! Cost per frame stays flat on large graphs: the strongest edges only (the step sorts them
//! once), fewer of them past [`LOD_NODES`] nodes, sub-pixel edges skipped, and the per-frame
//! vectors kept in [`Buffers`] from one frame to the next.

use std::collections::HashMap;

use eframe::egui::{
    self, Align2, Color32, CornerRadius, Mesh, PointerButton, Pos2, Rect, Sense, Shape, Stroke,
    StrokeKind, Vec2,
};
use weight_surgeon::events::NodeStatus;
use weight_surgeon::graph::diff::{EdgeDiff, NodeDiff};
use weight_surgeon::graph::layout::Position;
use weight_surgeon::graph::output::NodeReport;
use weight_surgeon::graph::GraphRole;

use crate::agent::activity::Activity;
use crate::camera::{self, Orbit, Projected};
use crate::scene;
use crate::source::Step;
use crate::studio::live::JobKind;
use crate::studio::live::Layer;
use crate::theme::{self, RoleShape};
use crate::ui::i18n::{t, tf};
use crate::ui::{components, icons};

/// Past this many nodes, the lighter edge budget.
pub const LOD_NODES: usize = 400;

/// Weighted edges drawn per frame (structural ones and those of the selection always are).
pub fn edge_budget(nodes: usize) -> usize {
    match nodes > LOD_NODES {
        true => 1_200,
        false => 4_000,
    }
}

/// The tensor a role stands for, as checkpoints name it.
pub fn role_label(role: GraphRole) -> &'static str {
    match role {
        GraphRole::Embed => "embed_tokens",
        GraphRole::Q => "q_proj",
        GraphRole::K => "k_proj",
        GraphRole::V => "v_proj",
        GraphRole::AttnOut => "o_proj",
        GraphRole::Gate => "gate_proj",
        GraphRole::Up => "up_proj",
        GraphRole::MlpOut => "down_proj",
        GraphRole::LmHead => "lm_head",
    }
}

/// `Layer 14 · o_proj`, `Layer 3 · down_proj · e5`, `embed_tokens`.
pub fn node_title(node: &NodeReport) -> String {
    let expert = node.expert.map(|e| format!(" · e{e}")).unwrap_or_default();
    match node.layer {
        Some(layer) => format!("Layer {layer} · {}{expert}", role_label(node.role)),
        None => format!("{}{expert}", role_label(node.role)),
    }
}

/// How a run state reads without color.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glyph {
    /// Dotted ring.
    Dots,
    /// Square outline.
    Square,
    /// Ring and check mark.
    Check,
    /// Thick ring and cross.
    Cross,
}

pub fn state_style(state: NodeStatus) -> (Glyph, Color32, &'static str) {
    match state {
        NodeStatus::Pending => (Glyph::Dots, theme::DIM, t("state.pending")),
        // The accent is kept for the selection: a running node reads by its square glyph.
        NodeStatus::Running => (Glyph::Square, theme::TEXT, t("state.running")),
        NodeStatus::Done => (Glyph::Check, theme::GOOD, t("state.done")),
        NodeStatus::Failed => (Glyph::Cross, theme::BAD, t("state.failed")),
    }
}

pub const STATES: [NodeStatus; 4] = [
    NodeStatus::Pending,
    NodeStatus::Running,
    NodeStatus::Done,
    NodeStatus::Failed,
];

/// The glyph of `state` around a node of radius `r` at `c`.
fn glyph(out: &mut Vec<Shape>, state: NodeStatus, c: Pos2, r: f32) {
    let (glyph, color, _) = state_style(state);
    let ring = r + 3.0;
    match glyph {
        Glyph::Dots => out.extend((0..6).map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 6.0;
            Shape::circle_filled(c + ring * Vec2::angled(a), 1.2, color)
        })),
        Glyph::Square => out.push(Shape::rect_stroke(
            Rect::from_center_size(c, Vec2::splat(2.0 * ring)),
            CornerRadius::ZERO,
            Stroke::new(1.5, color),
            StrokeKind::Middle,
        )),
        Glyph::Check | Glyph::Cross => {
            let width = if glyph == Glyph::Cross { 2.0 } else { 1.5 };
            out.push(Shape::circle_stroke(c, ring, Stroke::new(width, color)));
            // The mark sits on the ring, top right, where it does not hide the node.
            let m = c + Vec2::new(ring, -ring) * 0.75;
            let s = (0.6 * r).max(2.5);
            let stroke = Stroke::new(1.5, color);
            match glyph {
                Glyph::Check => out.push(Shape::line(
                    vec![
                        m + Vec2::new(-s, 0.0),
                        m + Vec2::new(-0.3 * s, 0.7 * s),
                        m + Vec2::new(s, -0.8 * s),
                    ],
                    stroke,
                )),
                _ => out.extend([
                    Shape::line_segment([m - Vec2::splat(s), m + Vec2::splat(s)], stroke),
                    Shape::line_segment([m + Vec2::new(-s, s), m + Vec2::new(s, -s)], stroke),
                ]),
            }
        }
    }
}

/// Δ coloring of a node: coral for a small relative change up to magenta from 10 %, the
/// role color pushed back toward the canvas when the node did not change.
pub fn delta_color(diff: Option<&NodeDiff>, role: Color32) -> Color32 {
    match diff.filter(|d| d.changed) {
        Some(diff) => theme::diverge(diff.relative_change),
        None => theme::shade(role, -0.6),
    }
}

fn lerp(a: Position, b: Position, t: f32) -> Position {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

fn node_radius(p: &Projected) -> f32 {
    (scene::NODE_RADIUS * p.scale).clamp(2.5, 11.0)
}

pub struct View<'a> {
    pub step: &'a Step,
    /// 0: the column, 1: the helix.
    pub blend: f32,
    pub diff: bool,
    /// Nodes kept by the filters; the others are drawn faint and cannot be picked.
    pub visible: &'a [bool],
    pub selected: Option<usize>,
    /// Nodes the patch being edited will touch (empty: no preview).
    pub targets: &'a [bool],
    /// The run drawn over the graph, filled for this step.
    pub layer: Option<&'a Layer>,
    /// The agent's running tool: its layers pulse, its label heads the canvas.
    pub activity: Option<&'a Activity>,
    /// Pulses and flows move (off: `--no-motion`, captures, tests).
    pub animate: bool,
    /// Live mode: the nodes kept lit; the others step back. `None`: every node is lit.
    pub focus: Option<&'a [bool]>,
    /// The origin, drawn as a translucent wireframe over this state.
    pub ghost: Option<&'a Step>,
}

/// What a frame allocates, kept between frames.
#[derive(Default)]
pub struct Buffers {
    projected: Vec<Option<Projected>>,
    colors: Vec<Color32>,
    fades: Vec<f32>,
    order: Vec<usize>,
    pickable: Vec<(usize, Pos2, f32)>,
    shapes: Vec<Shape>,
    /// Edges drawn by the last frame.
    pub edges_drawn: usize,
}

/// What a click did: `Some(Some(i))` picked node `i`, `Some(None)` clicked empty space.
pub type Click = Option<Option<usize>>;

/// A segment from `a` to `b` whose color runs from `ca` to `cb`, `w` wide, with a feathered
/// pixel on each side (epaint does not antialias a mesh).
fn gradient_segment(mesh: &mut Mesh, [a, b]: [Pos2; 2], [ca, cb]: [Color32; 2], w: f32) {
    let d = b - a;
    let len = d.length();
    if len < 0.5 {
        return;
    }
    let n = Vec2::new(-d.y, d.x) / len;
    let (inner, outer) = (n * (w * 0.5), n * (w * 0.5 + 1.0));
    let base = mesh.vertices.len() as u32;
    for (p, c) in [(a, ca), (b, cb)] {
        mesh.colored_vertex(p - outer, theme::TRANSPARENT);
        mesh.colored_vertex(p - inner, c);
        mesh.colored_vertex(p + inner, c);
        mesh.colored_vertex(p + outer, theme::TRANSPARENT);
    }
    for k in 0..3 {
        let (i, j) = (base + k, base + 4 + k);
        mesh.add_triangle(i, i + 1, j);
        mesh.add_triangle(i + 1, j + 1, j);
    }
}

/// A shaded node: a soft halo, the shadowed body, the lit face and its highlight. Small
/// nodes keep the body and the face only.
fn ball(out: &mut Vec<Shape>, shape: RoleShape, c: Pos2, r: f32, color: Color32) {
    let detailed = r >= 3.5;
    if detailed {
        out.push(Shape::circle_filled(c, 2.2 * r, color.gamma_multiply(0.09)));
    }
    out.push(icons::role_fill(shape, c, r, theme::shade(color, -0.5)));
    out.push(icons::role_fill(
        shape,
        c + Vec2::new(-0.12, -0.14) * r,
        0.78 * r,
        color,
    ));
    if detailed {
        let light = c + Vec2::new(-0.34, -0.38) * r;
        out.push(Shape::circle_filled(
            light,
            0.3 * r,
            theme::shade(color, 0.7),
        ));
    }
}

/// A line of text on a plate of the canvas color, so that it reads over the edges.
fn plate(painter: &egui::Painter, at: Pos2, align: Align2, text: String, color: Color32) {
    let galley = painter.layout_no_wrap(text, components::font_mono(), color);
    let rect = align.anchor_size(at, galley.size());
    painter.rect_filled(rect.expand2(Vec2::new(4.0, 1.0)), 3.0, theme::BG);
    painter.galley(rect.min, galley, color);
}

/// The name of the selected node and its measures, on a card tied to it.
fn badge(painter: &egui::Painter, rect: Rect, at: Pos2, r: f32, node: &NodeReport, measured: bool) {
    let m = &node.metrics;
    let second = match (measured, m.sigma1) {
        (true, Some(s)) => format!(
            "σ1 {s:.3} · erank {:.1}-{:.1} · κ {:.1}",
            m.erank_energy_lower, m.erank_energy_upper, m.kurtosis
        ),
        _ => format!("{} × {} · {}", node.rows, node.cols, node.dtype),
    };
    let font = components::font_mono();
    let first = painter.layout_no_wrap(node.id.clone(), font.clone(), theme::TEXT);
    let second = painter.layout_no_wrap(second, font, theme::DIM);
    let size = Vec2::new(
        first.size().x.max(second.size().x) + 16.0,
        first.size().y + second.size().y + 10.0,
    );
    let right = at.x + r + 14.0 + size.x <= rect.right() - 4.0;
    let x = match right {
        true => at.x + r + 14.0,
        false => at.x - r - 14.0 - size.x,
    };
    let y = (at.y - size.y / 2.0).clamp(rect.top() + 4.0, rect.bottom() - size.y - 4.0);
    let card = Rect::from_min_size(Pos2::new(x, y), size);
    let tie = Pos2::new(
        if right { card.left() } else { card.right() },
        card.center().y,
    );
    painter.line_segment([at, tie], Stroke::new(1.0, theme::SELECTION));
    painter.rect_filled(card, 4.0, theme::PANEL);
    painter.rect_stroke(
        card,
        4.0,
        Stroke::new(1.0, theme::SELECTION),
        StrokeKind::Inside,
    );
    let h = first.size().y;
    painter.galley(card.min + Vec2::new(8.0, 5.0), first, theme::TEXT);
    painter.galley(card.min + Vec2::new(8.0, 5.0 + h), second, theme::DIM);
}

/// The label of the running tool at the top of the canvas, with a beating dot.
fn activity_banner(painter: &egui::Painter, rect: Rect, label: &str, beat: f32) {
    let galley = painter.layout_no_wrap(label.to_owned(), components::font_mono(), theme::TEXT);
    let size = Vec2::new(galley.size().x + 36.0, 26.0);
    let card = Rect::from_center_size(
        Pos2::new(rect.center().x, rect.top() + 10.0 + size.y / 2.0),
        size,
    );
    painter.rect_filled(card, 13.0, theme::PANEL);
    painter.rect_stroke(
        card,
        13.0,
        Stroke::new(1.0, theme::ACTIVE.gamma_multiply(0.6)),
        StrokeKind::Inside,
    );
    let dot = Pos2::new(card.left() + 14.0, card.center().y);
    painter.circle_filled(dot, 6.0, theme::ACTIVE.gamma_multiply(0.25 * beat));
    painter.circle_filled(dot, 3.5, theme::ACTIVE);
    let y = card.center().y - galley.size().y / 2.0;
    painter.galley(Pos2::new(card.left() + 26.0, y), galley, theme::TEXT);
}

pub fn show(ui: &mut egui::Ui, view: &View, camera: &mut Orbit, buf: &mut Buffers) -> Click {
    let (rect, response) = ui.allocate_exact_size(ui.available_size(), Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CornerRadius::ZERO, theme::BG);

    let delta = response.drag_delta();
    if response.dragged_by(PointerButton::Primary) {
        camera.rotate(delta.x, delta.y);
    } else if response.dragged_by(PointerButton::Secondary) {
        camera.pan(delta.x, delta.y, rect);
    }
    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            camera.zoom(scroll);
        }
    }
    // The clock of pulses and flows; still frames when motion is off.
    let time = match view.animate {
        true => ui.input(|i| i.time) as f32,
        false => 0.0,
    };
    if view.animate && (view.selected.is_some() || view.activity.is_some()) {
        ui.ctx().request_repaint();
    }

    let report = &view.step.report;
    let scene = &view.step.scene;
    let at = |i: usize| lerp(scene.column[i], scene.helix[i], view.blend);
    buf.projected.clear();
    buf.projected
        .extend((0..report.nodes.len()).map(|i| camera.project(at(i), rect)));
    let projected = &buf.projected;
    let node_diffs: HashMap<&str, &NodeDiff> = match view.diff {
        // The baseline diff first; in a stream, the delta of the patch that made this state.
        true => report
            .baseline_diff
            .iter()
            .flat_map(|d| &d.nodes)
            .chain(view.step.touched.values().flatten())
            .map(|d| (d.id.as_str(), d))
            .collect(),
        false => HashMap::new(),
    };
    let edge_diffs: HashMap<(&str, &str), &EdgeDiff> = match view.diff {
        true => report
            .baseline_diff
            .iter()
            .flat_map(|d| &d.edges)
            .filter(|e| e.changed)
            .map(|e| ((e.source.as_str(), e.target.as_str()), e))
            .collect(),
        false => HashMap::new(),
    };
    let state_of = |i: usize| view.layer.and_then(|l| l.states.get(i).copied().flatten());

    // Depth of field: the far nodes step back toward the canvas, more in the helix.
    let depths = projected.iter().flatten().map(|p| p.depth);
    let (near, far) = depths.fold((f32::MAX, f32::MIN), |(lo, hi), d| (lo.min(d), hi.max(d)));
    let reach = 0.3 + 0.35 * view.blend;
    buf.fades.clear();
    buf.fades.extend(projected.iter().map(|p| match p {
        Some(p) if far > near => 1.0 - reach * (p.depth - near) / (far - near),
        _ => 1.0,
    }));
    let fades = &buf.fades;

    // Each node's color: its role, or the diff, or the run's value; then faded.
    buf.colors.clear();
    buf.colors
        .extend(report.nodes.iter().enumerate().map(|(i, node)| {
            let role = theme::role_color(node.role);
            let color = match view.diff {
                true => delta_color(node_diffs.get(node.id.as_str()).copied(), role),
                false => role,
            };
            // A run in view: its values color the nodes they cover, unreached nodes step back.
            let color = match view.layer {
                Some(layer) => match (layer.colors.get(i).copied().flatten(), state_of(i)) {
                    (Some(c), _) => c,
                    (None, Some(_)) => color,
                    (None, None) => theme::shade(color, -0.6),
                },
                None => color,
            };
            let color = match view.visible[i] {
                true => color,
                false => theme::shade(color, -0.88),
            };
            let color = match view.focus.is_some_and(|f| !f[i]) {
                true => theme::shade(color, -0.8),
                false => color,
            };
            theme::mix(color, theme::BG, fades[i])
        }));
    let colors = &buf.colors;

    let shapes = &mut buf.shapes;
    shapes.clear();
    // The bones of the layout: rows (column) or rings (helix), and the residual stream.
    let guide = theme::BORDER.gamma_multiply(0.7);
    for &(_, y) in &scene.layers {
        let points: Vec<Pos2> = match view.blend < 0.5 {
            true => [-0.9, 0.7]
                .iter()
                .filter_map(|&x| camera.project([x, y, 0.0], rect))
                .map(|p| p.pos)
                .collect(),
            false => (0..=32)
                .filter_map(|k| {
                    let a = k as f32 * std::f32::consts::TAU / 32.0;
                    camera.project([scene::RING * a.cos(), y, scene::RING * a.sin()], rect)
                })
                .map(|p| p.pos)
                .collect(),
        };
        let fade = (1.0 - 2.0 * view.blend).abs();
        shapes.push(Shape::line(
            points,
            Stroke::new(1.0, guide.gamma_multiply(fade)),
        ));
    }
    let spine: Vec<Pos2> = scene
        .spine
        .iter()
        .filter_map(|p| camera.project(*p, rect))
        .map(|p| p.pos)
        .collect();
    shapes.push(Shape::line(
        spine,
        Stroke::new(2.0, theme::DIM.gamma_multiply(0.25)),
    ));

    // Edges under the nodes, in one mesh: the step lists them structural first, then
    // strongest. Each runs from its source's color to its target's.
    let max_weight = report
        .edges
        .iter()
        .map(|e| e.weight.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    let budget = edge_budget(report.nodes.len());
    let mut mesh = Mesh::default();
    let mut flows = Vec::new();
    buf.edges_drawn = 0;
    for (rank, &(k, a, b)) in view.step.edges.iter().enumerate() {
        let edge = &report.edges[k];
        let touches_selection = view.selected.is_some_and(|s| s == a || s == b);
        if rank >= budget && !edge.structural && !touches_selection {
            continue;
        }
        let (Some(pa), Some(pb)) = (projected[a], projected[b]) else {
            continue;
        };
        let strength = (edge.weight.abs() / max_weight) as f32;
        let (mut alpha, width) = match (touches_selection, edge.structural) {
            (true, _) => (0.95, 2.2),
            (false, true) => (0.35, 1.0),
            (false, false) => (0.22 + 0.5 * strength, 1.0 + 0.8 * strength),
        };
        if view.selected.is_some() && !touches_selection {
            alpha *= 0.45;
        }
        if !(view.visible[a] || view.visible[b]) {
            alpha *= 0.2;
        }
        if view.focus.is_some_and(|f| !(f[a] || f[b])) {
            alpha *= 0.25;
        }
        let ends = match edge_diffs.get(&(edge.source.as_str(), edge.target.as_str())) {
            Some(diff) => [theme::diverge(diff.alignment_delta); 2],
            None => [colors[a], colors[b]],
        };
        let alpha = alpha * (fades[a] + fades[b]) / 2.0;
        gradient_segment(
            &mut mesh,
            [pa.pos, pb.pos],
            ends.map(|c| c.gamma_multiply(alpha)),
            width,
        );
        buf.edges_drawn += 1;
        let active = view
            .activity
            .is_some_and(|act| act.covers(report.nodes[b].layer));
        if touches_selection || active {
            flows.push((pa.pos, pb.pos, ends[1]));
        }
    }
    shapes.push(Shape::mesh(mesh));
    // The flow: two beads per lit edge, running from the writer to the reader.
    for (a, b, color) in flows {
        for k in 0..2 {
            let t = (time * 0.6 + k as f32 * 0.5).fract();
            shapes.push(Shape::circle_filled(
                a + (b - a) * t,
                1.8,
                theme::shade(color, 0.4),
            ));
        }
    }

    // Nodes far to near, so the nearest stay on top.
    buf.order.clear();
    buf.order
        .extend((0..report.nodes.len()).filter(|&i| projected[i].is_some()));
    buf.order.sort_by(|&a, &b| {
        let depth = |i: usize| projected[i].map_or(0.0, |p| p.depth);
        depth(b).total_cmp(&depth(a))
    });
    buf.pickable.clear();
    let ripple = match view.animate {
        true => (time * 1.2).fract(),
        false => 0.3,
    };
    for &i in &buf.order {
        let Some(p) = projected[i] else { continue };
        let node = &report.nodes[i];
        let radius = node_radius(&p);
        let shape = theme::role_shape(node.role);
        if view.visible[i] && view.activity.is_some_and(|a| a.covers(node.layer)) {
            shapes.push(Shape::circle_stroke(
                p.pos,
                radius + 3.0 + 6.0 * ripple,
                Stroke::new(1.5, theme::ACTIVE.gamma_multiply(0.85 * (1.0 - ripple))),
            ));
        }
        ball(shapes, shape, p.pos, radius, colors[i]);
        if view.step.touched.contains_key(&node.id) {
            shapes.push(Shape::circle_stroke(
                p.pos,
                radius + 2.5,
                Stroke::new(1.5, theme::AGENT),
            ));
        }
        if view.targets.get(i).copied().unwrap_or(false) {
            shapes.push(Shape::circle_stroke(
                p.pos,
                radius + 2.0,
                Stroke::new(1.0, theme::SELECTION),
            ));
        }
        if let Some(state) = state_of(i).filter(|_| view.visible[i]) {
            glyph(shapes, state, p.pos, radius);
        }
        if view.visible[i] {
            buf.pickable.push((i, p.pos, radius));
        }
    }
    painter.extend(shapes.drain(..));
    if let Some(ghost) = view.ghost {
        origin_overlay(&painter, view, ghost, camera, rect);
    }

    // The layer marks at the left of each row, without overlapping one another. In the
    // helix, left of the ring as the camera sees it (its right is (cos yaw, 0, -sin yaw)).
    let lit = view.selected.and_then(|i| report.nodes[i].layer);
    let mut taken: Vec<Rect> = Vec::new();
    let side = -(scene::RING + 0.12);
    let ring_left = [side * camera.yaw.cos(), 0.0, -side * camera.yaw.sin()];
    for &(l, y) in &scene.layers {
        let mark = lerp([-0.9, y, 0.0], [ring_left[0], y, ring_left[2]], view.blend);
        let Some(p) = camera.project(mark, rect) else {
            continue;
        };
        let color = match (
            view.activity.is_some_and(|act| act.covers(Some(l))),
            lit == Some(l),
        ) {
            (true, _) => theme::ACTIVE,
            (_, true) => theme::TEXT,
            _ => theme::DIM,
        };
        let galley = painter.layout_no_wrap(format!("L{l}"), components::font_meta(), color);
        let r = Align2::RIGHT_CENTER.anchor_size(p.pos, galley.size());
        if taken.iter().any(|t| t.expand(2.0).intersects(r)) || !rect.contains_rect(r) {
            continue;
        }
        taken.push(r);
        painter.galley(r.min, galley, color);
    }

    let hovered = response
        .hover_pos()
        .and_then(|pointer| camera::pick(&buf.pickable, pointer, 3.0));
    if let Some(i) = view.selected.filter(|&i| view.visible[i]) {
        if let Some(p) = projected[i] {
            let radius = node_radius(&p);
            let beat = match view.animate {
                true => 0.5 + 0.5 * (time * 4.0).sin(),
                false => 0.6,
            };
            painter.circle_filled(p.pos, 3.0 * radius, theme::SELECTION.gamma_multiply(0.16));
            painter.circle_stroke(
                p.pos,
                radius + 5.0 + 3.0 * beat,
                Stroke::new(2.0, theme::SELECTION.gamma_multiply(1.0 - 0.6 * beat)),
            );
            painter.circle_stroke(p.pos, radius + 2.5, Stroke::new(1.5, theme::SELECTION));
            let node = &report.nodes[i];
            let mut lit = Vec::new();
            ball(
                &mut lit,
                theme::role_shape(node.role),
                p.pos,
                radius,
                buf.colors[i],
            );
            painter.extend(lit);
            badge(
                &painter,
                rect,
                p.pos,
                radius,
                node,
                report.parameters.rank > 0,
            );
        }
    }
    if let (Some(i), Some(pointer)) = (hovered, response.hover_pos()) {
        if let Some(p) = projected[i] {
            let radius = node_radius(&p);
            painter.circle_stroke(p.pos, radius + 1.5, Stroke::new(1.0, theme::TEXT));
            if view.selected != Some(i) {
                let above = p.pos - Vec2::new(0.0, radius + 6.0);
                plate(
                    &painter,
                    above,
                    Align2::CENTER_BOTTOM,
                    report.nodes[i].id.clone(),
                    theme::TEXT,
                );
            }
        }
        let run = view.layer.map(|l| run_line(l, i));
        let above = egui::LayerId::new(
            egui::Order::Middle,
            painter.layer_id().id.with("node-hover"),
        );
        let top = painter.ctx().layer_painter(above).with_clip_rect(rect);
        hover_card(&top, rect, pointer, &report.nodes[i], run);
    }
    if let Some(activity) = view.activity {
        let beat = match view.animate {
            true => 0.5 + 0.5 * (time * 5.0).sin(),
            false => 1.0,
        };
        activity_banner(&painter, rect, &activity.label, beat);
    }

    let mut top = rect.left_top() + Vec2::new(10.0, 8.0);
    let mut caption = |text: String, color: Color32| {
        let r = painter.text(top, Align2::LEFT_TOP, text, components::font_mono(), color);
        top.y += r.height() + 2.0;
    };
    let targeted = view.targets.iter().filter(|t| **t).count();
    // The small preview keeps its room for the graph; the 3D tab carries the captions.
    let wide = rect.width() >= 560.0;
    if targeted > 0 && wide {
        caption(tf("view3d.targets", &[("n", &targeted)]), theme::TEXT);
    }
    if let Some(layer) = view.layer.filter(|_| wide) {
        caption(job_caption(layer.kind).into(), theme::TEXT);
        if let (Some(metric), Some((lo, hi))) = (layer.kind.metric(), layer.range) {
            caption(format!("{metric} : {lo:.3e} – {hi:.3e}"), theme::DIM);
        }
        if layer.unmatched > 0 {
            caption(
                tf("view3d.unmatched", &[("n", &layer.unmatched)]),
                theme::DIM,
            );
        }
        legend(&painter, rect, layer);
    }
    axis_gizmo(&painter, rect, camera);
    // The help line only where it fits (not in the Atelier's small preview).
    if !wide {
        return match response.clicked() {
            true => Some(hovered),
            false => None,
        };
    }
    painter.text(
        rect.left_bottom() + Vec2::new(10.0, -8.0),
        Align2::LEFT_BOTTOM,
        t("view3d.help"),
        components::font_meta(),
        theme::DIM,
    );

    match response.clicked() {
        true => Some(hovered),
        false => None,
    }
}

/// `ghost`, the origin, as a translucent wireframe over the state on screen: its structural
/// edges, and a ring per node sized by its `‖W‖F` against the current one (the baseline when
/// the report compared to it, else the ghost's own metrics). Nothing changed: rings at rest.
fn origin_overlay(painter: &egui::Painter, view: &View, ghost: &Step, camera: &Orbit, rect: Rect) {
    let ink = Stroke::new(1.0, theme::TEXT.gamma_multiply(0.35));
    let projected: Vec<Option<Projected>> = (0..ghost.report.nodes.len())
        .map(|j| {
            let at = lerp(ghost.scene.column[j], ghost.scene.helix[j], view.blend);
            camera.project(at, rect)
        })
        .collect();
    let baseline: HashMap<&str, &NodeDiff> = view
        .step
        .report
        .baseline_diff
        .iter()
        .flat_map(|d| &d.nodes)
        .map(|d| (d.id.as_str(), d))
        .collect();
    let mut shapes = Vec::new();
    for &(k, a, b) in &ghost.edges {
        if let (true, Some(pa), Some(pb)) =
            (ghost.report.edges[k].structural, projected[a], projected[b])
        {
            shapes.push(Shape::line(vec![pa.pos, pb.pos], ink));
        }
    }
    for (node, p) in ghost.report.nodes.iter().zip(&projected) {
        let Some(p) = p else { continue };
        let now = view
            .step
            .index
            .get(&node.id)
            .map(|&i| view.step.report.nodes[i].metrics.frobenius_norm);
        let origin = match baseline.get(node.id.as_str()) {
            Some(diff) => now.map(|n| n - diff.norm_delta),
            None => Some(node.metrics.frobenius_norm),
        };
        let ratio = match (origin, now) {
            (Some(o), Some(n)) if o > 0.0 && n > 0.0 => (o / n).clamp(0.5, 2.0) as f32,
            _ => 1.0,
        };
        shapes.push(Shape::circle_stroke(
            p.pos,
            node_radius(p) * ratio + 1.5,
            ink,
        ));
    }
    painter.extend(shapes);
}

/// The drafting-table mark of the view: three orthogonal axes in the top right corner, in
/// the secondary text color (a role or an axis never gets a hue of its own).
fn axis_gizmo(painter: &egui::Painter, rect: Rect, camera: &Orbit) {
    let t = &theme::TOKENS;
    let origin = rect.right_top() + Vec2::new(-2.0 * t.space.xl, 2.0 * t.space.xl);
    let reach = t.space.l + t.space.xs;
    for ((dx, dy), name) in camera.axes().into_iter().zip(["x", "y", "z"]) {
        let tip = origin + Vec2::new(dx, dy) * reach;
        // A path, not a segment: segments are the graph's edges (the edge budget counts them).
        painter.add(Shape::line(
            vec![origin, tip],
            Stroke::new(t.stroke, t.color.text_secondary),
        ));
        let label = origin + Vec2::new(dx, dy) * (reach + t.space.s);
        painter.text(
            label,
            Align2::CENTER_CENTER,
            name,
            components::font_meta(),
            t.color.text_secondary,
        );
    }
}

/// What the reached nodes mean for this kind of run.
fn job_caption(kind: JobKind) -> &'static str {
    match kind {
        JobKind::Graph => t("job.graph"),
        JobKind::Extract => t("job.extract"),
        JobKind::Apply => t("job.apply"),
        JobKind::Bench => t("job.bench"),
        JobKind::Pipeline => t("job.pipeline"),
        JobKind::Other => t("job.other"),
    }
}

/// `terminé · max_abs_delta 1.250e-2`, or `hors run`.
pub fn run_line(layer: &Layer, i: usize) -> String {
    let state = layer.states.get(i).copied().flatten();
    let mut line = state.map_or(t("state.outside").into(), |s| state_style(s).2.to_owned());
    if let (Some(metric), Some(Some(v))) = (layer.kind.metric(), layer.values.get(i)) {
        line.push_str(&format!(" · {metric} {v:.3e}"));
    }
    line
}

/// Glyph, label and count of each state, bottom right.
fn legend(painter: &egui::Painter, rect: Rect, layer: &Layer) {
    let font = components::font_mono();
    let row = 16.0;
    let size = Vec2::new(150.0, row * STATES.len() as f32 + 8.0);
    let card = Rect::from_min_size(rect.right_bottom() - size - Vec2::splat(10.0), size);
    painter.rect_filled(card, CornerRadius::same(2), theme::PANEL);
    painter.rect_stroke(
        card,
        CornerRadius::same(2),
        Stroke::new(1.0, theme::BORDER),
        StrokeKind::Inside,
    );
    let mut shapes = Vec::new();
    for (k, state) in STATES.into_iter().enumerate() {
        let y = card.top() + 4.0 + row * (k as f32 + 0.5);
        let c = Pos2::new(card.left() + 14.0, y);
        let (_, color, label) = state_style(state);
        shapes.push(Shape::circle_filled(c, 3.0, theme::DIM));
        glyph(&mut shapes, state, c, 3.0);
        painter.text(
            Pos2::new(card.left() + 30.0, y),
            Align2::LEFT_CENTER,
            format!("{label} {}", layer.count(state)),
            font.clone(),
            color,
        );
    }
    painter.extend(shapes);
}

/// A few monospace lines next to the pointer, kept inside the canvas.
fn hover_card(
    painter: &egui::Painter,
    rect: Rect,
    pointer: Pos2,
    node: &NodeReport,
    run: Option<String>,
) {
    let m = &node.metrics;
    let sigma = m.sigma1.map_or("-".into(), |s| format!("{s:.4}"));
    let mut lines = vec![
        (node_title(node), theme::TEXT),
        (
            format!("{} × {}  {}", node.rows, node.cols, node.dtype),
            theme::DIM,
        ),
        (
            format!(
                "σ1 {sigma}  erank {:.2}-{:.2}",
                m.erank_energy_lower, m.erank_energy_upper
            ),
            theme::DIM,
        ),
    ];
    if let Some(run) = run {
        lines.push((tf("view3d.run", &[("run", &run)]), theme::TEXT));
    }
    let font = components::font_mono();
    let galleys: Vec<_> = lines
        .into_iter()
        .map(|(text, color)| painter.layout_no_wrap(text, font.clone(), color))
        .collect();
    let width = galleys.iter().map(|g| g.size().x).fold(0.0, f32::max);
    let height: f32 = galleys.iter().map(|g| g.size().y).sum();
    let size = Vec2::new(width + 12.0, height + 8.0);
    let mut min = pointer + Vec2::new(14.0, 14.0);
    min.x = min.x.min(rect.right() - size.x - 4.0);
    min.y = min.y.min(rect.bottom() - size.y - 4.0);
    let card = Rect::from_min_size(min, size);
    painter.rect_filled(
        card,
        CornerRadius::same(2),
        theme::TOKENS.color.surface_hover,
    );
    painter.rect_stroke(
        card,
        CornerRadius::same(2),
        Stroke::new(1.0, theme::TOKENS.color.text_disabled),
        StrokeKind::Inside,
    );
    let mut y = card.top() + 4.0;
    for galley in galleys {
        let h = galley.size().y;
        painter.galley(Pos2::new(card.left() + 6.0, y), galley, theme::TEXT);
        y += h;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster;
    use crate::source::Timeline;
    use crate::studio::bus::RunView;
    use crate::testkit;
    use weight_surgeon::events::{EventLine, SurgeonEvent};

    fn diff(rank: f64, relative: f64, changed: bool) -> NodeDiff {
        NodeDiff {
            id: "L1.attn_out".into(),
            role: GraphRole::AttnOut,
            layer: Some(1),
            effective_rank_delta: rank,
            norm_delta: relative,
            relative_change: relative,
            sigma1_delta: None,
            changed,
        }
    }

    #[test]
    fn changed_nodes_run_from_coral_to_magenta_with_the_change() {
        let role = theme::ROLE_MLP;
        let unchanged = delta_color(None, role);
        assert_eq!(delta_color(Some(&diff(1.0, 0.5, false)), role), unchanged);
        assert!(
            theme::luminance(unchanged) < theme::luminance(role),
            "stepped back"
        );
        // Warm and red in both directions of the rank; bluer (magenta) when larger.
        let weak = delta_color(Some(&diff(-0.5, 0.001, true)), role);
        let strong = delta_color(Some(&diff(0.5, 0.2, true)), role);
        for c in [weak, strong] {
            assert!(c.r() > 200 && c.g() < 130, "{c:?}");
        }
        assert_eq!(strong, theme::DIVERGE_HIGH);
        assert!(strong.b() > weak.b() + 60);
    }

    #[test]
    fn every_state_has_its_own_glyph_and_label() {
        let styles: Vec<_> = STATES.into_iter().map(state_style).collect();
        for (a, x) in styles.iter().enumerate() {
            for y in &styles[a + 1..] {
                assert_ne!(x.0, y.0, "glyph {} / {}", x.2, y.2);
                assert_ne!(x.2, y.2);
            }
        }
        // Each glyph draws something different in kind, not only in color.
        let kinds = |state| {
            let mut out = Vec::new();
            glyph(&mut out, state, Pos2::ZERO, 5.0);
            out.iter().map(std::mem::discriminant).collect::<Vec<_>>()
        };
        let all: Vec<_> = STATES.into_iter().map(kinds).collect();
        for (a, x) in all.iter().enumerate() {
            for y in &all[a + 1..] {
                assert_ne!(x, y);
            }
        }
    }

    /// A run where every node of `step` has `state`, with a |ΔW| when it is done.
    fn layer_for(step: &Step, state: NodeStatus) -> Layer {
        let mut view = RunView::default();
        let mut seq = 0;
        let mut push = |event| {
            view.apply(&EventLine {
                v: 1,
                seq,
                t_ms: seq,
                event,
            });
            seq += 1;
        };
        push(SurgeonEvent::JobStarted {
            command: "surgeon_apply".into(),
        });
        for (i, node) in step.report.nodes.iter().enumerate() {
            push(SurgeonEvent::NodeState {
                node_id: node.tensor.clone(),
                state,
            });
            if state == NodeStatus::Done {
                push(SurgeonEvent::Metric {
                    node_id: Some(node.id.clone()),
                    name: "max_abs_delta".into(),
                    value: i as f64,
                });
            }
        }
        let mut layer = Layer::default();
        layer.fill(&view, step);
        layer
    }

    /// One frame of the viewport alone at 960×640: its painted texts and line count.
    fn frame(
        step: &Step,
        layer: Option<&Layer>,
        selected: Option<usize>,
    ) -> (raster::Frame, usize) {
        frame_with(step, layer, selected, None)
    }

    /// One still frame (no motion) and the number of edges it drew.
    fn frame_with(
        step: &Step,
        layer: Option<&Layer>,
        selected: Option<usize>,
        activity: Option<&Activity>,
    ) -> (raster::Frame, usize) {
        let n = step.report.nodes.len();
        let (visible, targets) = (vec![true; n], vec![false; n]);
        let view = View {
            step,
            blend: 0.0,
            diff: false,
            visible: &visible,
            selected,
            targets: &targets,
            layer,
            activity,
            animate: false,
            focus: None,
            ghost: None,
        };
        render(&view)
    }

    /// One still frame of `view` at 960×640 and the number of edges it drew.
    fn render(view: &View) -> (raster::Frame, usize) {
        let ctx = egui::Context::default();
        let (mut camera, mut buf) = (Orbit::default(), Buffers::default());
        let frame = raster::render(&ctx, [960, 640], 1.0, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                show(ui, view, &mut camera, &mut buf);
            });
        });
        (frame, buf.edges_drawn)
    }

    #[test]
    fn live_focus_steps_back_the_untouched_layers_and_the_ghost_adds_its_wireframe() {
        let timeline = Timeline::from_report(testkit::graph(28));
        let step = timeline.current().unwrap();
        let n = step.report.nodes.len();
        let (visible, targets) = (vec![true; n], vec![false; n]);
        let lit: Vec<bool> = step
            .report
            .nodes
            .iter()
            .map(|x| x.layer == Some(2))
            .collect();
        let view = |focus, ghost| View {
            step,
            blend: 0.0,
            diff: false,
            visible: &visible,
            selected: None,
            targets: &targets,
            layer: None,
            activity: None,
            animate: false,
            focus,
            ghost,
        };
        // The role hues (indigo, amber) are counted, as the depth of field shifts the shades.
        let saturated = |frame: &raster::Frame| {
            frame
                .image
                .rgba
                .chunks_exact(4)
                .filter(|p| {
                    let (r, g, b) = (i32::from(p[0]), i32::from(p[1]), i32::from(p[2]));
                    (b > 140 && b > r + 50) || (r > 150 && r > g + 40 && g > b + 60)
                })
                .count()
        };
        let (all, _) = render(&view(None, None));
        let (focused, _) = render(&view(Some(&lit), None));
        assert!(
            saturated(&focused) * 3 < saturated(&all),
            "{} vs {}",
            saturated(&focused),
            saturated(&all)
        );
        assert!(saturated(&focused) > 0, "layer 2 stays lit");
        // The wireframe lands on the nodes' halos: count the pixels it changed, not the ink.
        let (ghosted, _) = render(&view(Some(&lit), Some(step)));
        let (before, after) = (&focused.image.rgba, &ghosted.image.rgba);
        let pairs = before.chunks_exact(4).zip(after.chunks_exact(4));
        let changed = pairs.filter(|(a, b)| a != b).count();
        assert!(changed > 500, "{changed}");
    }

    /// Pixels within 24 (Chebyshev, RGB) of `color`.
    fn pixels_near(image: &raster::Image, color: Color32) -> usize {
        image
            .rgba
            .chunks_exact(4)
            .filter(|p| {
                [color.r(), color.g(), color.b()]
                    .iter()
                    .zip(p.iter())
                    .all(|(a, b)| a.abs_diff(*b) < 24)
            })
            .count()
    }

    #[test]
    fn roles_read_in_color_with_layer_marks_the_selection_badge_and_the_running_tool() {
        crate::ui::i18n::set_lang(crate::settings::Lang::Fr);
        let timeline = Timeline::from_report(testkit::graph(28));
        let step = timeline.current().unwrap();
        let activity =
            crate::agent::activity::of("surgeon_extract", &serde_json::json!({"layer": 2}))
                .unwrap();
        let selected = step.index["L13.n"];
        let (frame, edges) = frame_with(step, None, Some(selected), Some(&activity));
        assert!(edges > 0);
        // Depth of field darkens the far nodes: the hues are counted, not the exact colors.
        let count = |hue: fn(i32, i32, i32) -> bool| {
            let rgba = frame.image.rgba.chunks_exact(4);
            rgba.filter(|p| hue(p[0].into(), p[1].into(), p[2].into()))
                .count()
        };
        let indigo = count(|r, g, b| b > 140 && b > r + 50 && b > g + 50);
        let amber = count(|r, g, b| r > 150 && g > 80 && r > g + 40 && g > b + 60);
        assert!(
            indigo > 150 && amber > 150,
            "indigo {indigo}, amber {amber}"
        );
        assert!(
            pixels_near(&frame.image, theme::ACTIVE) > 20,
            "the ripple and the banner"
        );
        for text in ["L0", "L3", "L13.n", "Extraction L2…"] {
            assert!(
                frame.texts.iter().any(|t| t == text),
                "{text}: {:?}",
                frame.texts
            );
        }
    }

    #[test]
    fn an_empty_graph_renders_its_frame_and_help() {
        let timeline = Timeline::from_report(testkit::graph(0));
        let (frame, lines) = frame(timeline.current().unwrap(), None, None);
        assert_eq!(lines, 0);
        assert!(frame.texts.iter().any(|t| t.contains("orbite")));
        assert!(frame.image.ink() > 0.0);
    }

    #[test]
    fn five_hundred_nodes_render_within_the_edge_budget() {
        let timeline = Timeline::from_report(testkit::graph(600));
        let step = timeline.current().unwrap();
        let structural = step.report.edges.iter().filter(|e| e.structural).count();
        assert!(
            step.edges.len() > edge_budget(600) + structural,
            "the budget bites"
        );
        let layer = layer_for(step, NodeStatus::Done);
        let (frame, lines) = frame(step, Some(&layer), Some(300));
        // Weighted edges stop at the budget; structural ones and the selection's stay.
        let selection = step
            .edges
            .iter()
            .filter(|e| e.1 == 300 || e.2 == 300)
            .count();
        assert!(
            lines <= edge_budget(600) + structural + selection,
            "{lines}"
        );
        assert!(lines >= edge_budget(600), "{lines}");
        assert!(frame.texts.iter().any(|t| t.starts_with("terminé 600")));
        assert!(frame.texts.iter().any(|t| t.contains("max_abs_delta")));
        assert!(frame.image.ink() > 0.1, "{}", frame.image.ink());
    }

    #[test]
    fn a_failed_run_reads_in_the_legend_and_the_glyphs() {
        let timeline = Timeline::from_report(testkit::graph(40));
        let step = timeline.current().unwrap();
        let layer = layer_for(step, NodeStatus::Failed);
        let (frame, _) = frame(step, Some(&layer), None);
        assert!(frame.texts.iter().any(|t| t == "échoué 40"));
        assert!(frame.texts.iter().any(|t| t == "terminé 0"));
        // The red of the failure glyphs is on screen.
        let bad = frame
            .image
            .rgba
            .chunks_exact(4)
            .filter(|p| p[0] > 180 && p[1] < 90 && p[2] < 100)
            .count();
        assert!(bad > 200, "{bad}");
    }

    #[test]
    fn layouts_and_picking_are_deterministic() {
        let a = Timeline::from_report(testkit::graph(64));
        let b = Timeline::from_report(testkit::graph(64));
        let (sa, sb) = (a.current().unwrap(), b.current().unwrap());
        assert_eq!(sa.edges, sb.edges, "same edge order");
        let rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
        let camera = Orbit::default();
        let project = |s: &Step| -> Vec<_> {
            s.scene
                .column
                .iter()
                .map(|p| camera.project(*p, rect))
                .collect()
        };
        assert_eq!(project(sa), project(sb));
        // The node under its own center is the one picked.
        let points: Vec<_> = project(sa)
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|p| (i, p.pos, node_radius(&p))))
            .collect();
        for &(i, pos, _) in points.iter().step_by(7) {
            let picked = camera::pick(&points, pos, 3.0).unwrap();
            assert!(points[picked].1.distance(pos) <= 1e-3, "{i} -> {picked}");
        }
    }

    /// `cargo test --release -- --ignored frame_cost`: the CPU side of a frame (layout and
    /// tessellation; the GPU upload is not measured) on 600 nodes stays under 60 Hz.
    #[test]
    #[ignore]
    fn frame_cost() {
        let timeline = Timeline::from_report(testkit::graph(600));
        let step = timeline.current().unwrap();
        let layer = layer_for(step, NodeStatus::Running);
        let n = step.report.nodes.len();
        let (visible, targets) = (vec![true; n], vec![false; n]);
        let view = View {
            step,
            blend: 0.5,
            diff: false,
            visible: &visible,
            selected: Some(10),
            targets: &targets,
            layer: Some(&layer),
            activity: None,
            animate: true,
            focus: None,
            ghost: None,
        };
        let ctx = egui::Context::default();
        let (mut camera, mut buf) = (Orbit::default(), Buffers::default());
        let input = || egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1440.0, 900.0))),
            ..Default::default()
        };
        let frames = 120;
        let start = std::time::Instant::now();
        for _ in 0..frames {
            camera.rotate(3.0, 0.0);
            let mut output = ctx.run_ui(input(), |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    show(ui, &view, &mut camera, &mut buf);
                });
            });
            output.textures_delta.clear();
            ctx.tessellate(output.shapes, 1.0);
        }
        let per_frame = start.elapsed() / frames;
        println!("frame_cost 600 nodes: {per_frame:?}");
        assert!(per_frame.as_secs_f64() < 1.0 / 60.0, "{per_frame:?}");
    }
}
