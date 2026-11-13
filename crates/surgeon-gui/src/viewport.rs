//! The central 3D canvas: nodes at their Rust-computed positions (blended between the two
//! layouts), coupling edges, orbit / pan / zoom, picking and a compact hover card. A run
//! (live or replayed) draws over the nodes it reached: a glyph per state, never the color
//! alone, and for an apply the |ΔW| of each tensor on the report's color scale.
//!
//! Cost per frame stays flat on large graphs: the strongest edges only (the step sorts them
//! once), fewer of them past [`LOD_NODES`] nodes, sub-pixel edges skipped, and the per-frame
//! vectors kept in [`Buffers`] from one frame to the next.

use std::collections::HashMap;

use eframe::egui::{
    self, Align2, Color32, CornerRadius, PointerButton, Pos2, Rect, Sense, Shape, Stroke,
    StrokeKind, Vec2,
};
use weight_surgeon::events::NodeStatus;
use weight_surgeon::graph::diff::{EdgeDiff, NodeDiff};
use weight_surgeon::graph::layout::Position;
use weight_surgeon::graph::output::NodeReport;
use weight_surgeon::graph::GraphRole;

use crate::camera::{self, Orbit, Projected};
use crate::source::Step;
use crate::studio::live::JobKind;
use crate::studio::live::Layer;
use crate::theme;
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

/// Δ coloring of a node: emerald when its effective rank grows, carmine when it shrinks (a
/// rank collapse is the usual mark of a destructive edit), plain text when only its norm
/// moved, dim when unchanged. Stronger with the relative change, full at 10 %.
pub fn delta_color(diff: Option<&NodeDiff>) -> Color32 {
    let Some(diff) = diff.filter(|d| d.changed) else {
        return theme::DIM.gamma_multiply(0.6);
    };
    let base = match diff.effective_rank_delta {
        d if d > 0.0 => theme::GOOD,
        d if d < 0.0 => theme::BAD,
        _ => theme::TEXT,
    };
    base.gamma_multiply(0.45 + 0.55 * (diff.relative_change.abs() / 0.1).min(1.0) as f32)
}

fn edge_delta_color(diff: &EdgeDiff) -> Color32 {
    match diff.alignment_delta >= 0.0 {
        true => theme::GOOD,
        false => theme::BAD,
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
    (0.02 * p.scale).clamp(2.5, 9.0)
}

pub struct View<'a> {
    pub step: &'a Step,
    /// 0: `pos_column`, 1: `pos_constellation`.
    pub blend: f32,
    pub diff: bool,
    /// Nodes kept by the filters; the others are drawn faint and cannot be picked.
    pub visible: &'a [bool],
    pub selected: Option<usize>,
    /// Nodes the patch being edited will touch (empty: no preview).
    pub targets: &'a [bool],
    /// The run drawn over the graph, filled for this step.
    pub layer: Option<&'a Layer>,
}

/// What a frame allocates, kept between frames.
#[derive(Default)]
pub struct Buffers {
    projected: Vec<Option<Projected>>,
    order: Vec<usize>,
    pickable: Vec<(usize, Pos2, f32)>,
    shapes: Vec<Shape>,
}

/// What a click did: `Some(Some(i))` picked node `i`, `Some(None)` clicked empty space.
pub type Click = Option<Option<usize>>;

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

    let report = &view.step.report;
    buf.projected.clear();
    buf.projected.extend(
        report
            .nodes
            .iter()
            .map(|n| camera.project(lerp(n.pos_column, n.pos_constellation, view.blend), rect)),
    );
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

    // Edges first, under the nodes; the step lists them structural first, then strongest.
    let shapes = &mut buf.shapes;
    shapes.clear();
    let max_weight = report
        .edges
        .iter()
        .map(|e| e.weight.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    let budget = edge_budget(report.nodes.len());
    for (rank, &(k, a, b)) in view.step.edges.iter().enumerate() {
        let edge = &report.edges[k];
        let touches_selection = view.selected.is_some_and(|s| s == a || s == b);
        if rank >= budget && !edge.structural && !touches_selection {
            continue;
        }
        let (Some(pa), Some(pb)) = (projected[a], projected[b]) else {
            continue;
        };
        if pa.pos.distance_sq(pb.pos) < 0.25 {
            continue;
        }
        let shown = view.visible[a] || view.visible[b];
        let (color, width) = if touches_selection {
            (theme::SELECTION.gamma_multiply(0.85), 1.5)
        } else if let Some(diff) = edge_diffs.get(&(edge.source.as_str(), edge.target.as_str())) {
            (edge_delta_color(diff).gamma_multiply(0.7), 1.2)
        } else if edge.structural {
            (theme::DIM.gamma_multiply(0.18), 1.0)
        } else {
            let strength = (edge.weight.abs() / max_weight) as f32;
            (theme::TEXT.gamma_multiply(0.05 + 0.35 * strength), 1.0)
        };
        let color = if shown {
            color
        } else {
            color.gamma_multiply(0.2)
        };
        shapes.push(Shape::line_segment(
            [pa.pos, pb.pos],
            Stroke::new(width, color),
        ));
    }

    // Nodes far to near, so the nearest stay on top.
    buf.order.clear();
    buf.order
        .extend((0..report.nodes.len()).filter(|&i| projected[i].is_some()));
    buf.order.sort_by(|&a, &b| {
        let depth = |i: usize| projected[i].map_or(0.0, |p| p.depth);
        depth(b).total_cmp(&depth(a))
    });
    let state_of = |i: usize| view.layer.and_then(|l| l.states.get(i).copied().flatten());
    buf.pickable.clear();
    for &i in &buf.order {
        let Some(p) = projected[i] else { continue };
        let node = &report.nodes[i];
        let radius = node_radius(&p);
        let fill = match view.diff {
            true => delta_color(node_diffs.get(node.id.as_str()).copied()),
            false => theme::DIM,
        };
        // A run in view: its values color the nodes they cover, unreached nodes step back.
        let fill = match view.layer {
            Some(layer) => match (layer.colors.get(i).copied().flatten(), state_of(i)) {
                (Some(color), _) => color,
                (None, Some(_)) => fill,
                (None, None) => fill.gamma_multiply(0.35),
            },
            None => fill,
        };
        let fill = match view.visible[i] {
            true => fill,
            false => fill.gamma_multiply(0.12),
        };
        // Wireframe: the role reads by the outline's shape, the color is data (diff, run).
        let shape = theme::role_shape(node.role);
        shapes.push(icons::role_fill(shape, p.pos, radius, theme::BG));
        shapes.push(icons::role_outline(
            shape,
            p.pos,
            radius,
            Stroke::new(1.25, fill),
        ));
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

    let hovered = response
        .hover_pos()
        .and_then(|pointer| camera::pick(&buf.pickable, pointer, 3.0));
    if let Some(i) = view.selected.filter(|&i| view.visible[i]) {
        if let Some(p) = projected[i] {
            let radius = node_radius(&p);
            let shape = theme::role_shape(report.nodes[i].role);
            painter.add(icons::role_fill(shape, p.pos, radius, theme::SELECTION));
            // On a canvas-colored plate, so the name reads over the edges behind it.
            let galley = painter.layout_no_wrap(
                node_title(&report.nodes[i]),
                components::font_mono(),
                theme::TEXT,
            );
            let min = p.pos + Vec2::new(radius + 8.0, -galley.size().y / 2.0);
            let plate = Rect::from_min_size(min, galley.size()).expand2(Vec2::new(4.0, 0.0));
            painter.rect_filled(plate, 0.0, theme::BG);
            painter.galley(min, galley, theme::TEXT);
        }
    }
    if let (Some(i), Some(pointer)) = (hovered, response.hover_pos()) {
        if let Some(p) = projected[i] {
            painter.circle_stroke(p.pos, node_radius(&p) + 1.5, Stroke::new(1.0, theme::TEXT));
        }
        let run = view.layer.map(|l| run_line(l, i));
        hover_card(&painter, rect, pointer, &report.nodes[i], run);
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
    painter.rect_filled(card, CornerRadius::same(2), theme::PANEL);
    painter.rect_stroke(
        card,
        CornerRadius::same(2),
        Stroke::new(1.0, theme::BORDER),
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
    fn delta_colors_follow_the_rank_direction() {
        let unchanged = delta_color(None);
        assert_eq!(delta_color(Some(&diff(1.0, 0.5, false))), unchanged);
        let hue = |c: Color32| (c.r() > c.g(), c.g() > c.r());
        assert_eq!(hue(delta_color(Some(&diff(0.5, 0.2, true)))), (false, true));
        assert_eq!(
            hue(delta_color(Some(&diff(-0.5, 0.2, true)))),
            (true, false)
        );
        // Larger changes are drawn stronger.
        let weak = delta_color(Some(&diff(-0.5, 0.001, true)));
        let strong = delta_color(Some(&diff(-0.5, 0.2, true)));
        assert!(strong.r() > weak.r());
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
        };
        let ctx = egui::Context::default();
        let (mut camera, mut buf) = (Orbit::default(), Buffers::default());
        let frame = raster::render(&ctx, [960, 640], 1.0, |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                show(ui, &view, &mut camera, &mut buf);
            });
        });
        // The edges drawn are the line segments under the first node.
        let ctx = egui::Context::default();
        let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
            egui::CentralPanel::default().show(ui, |ui| {
                show(ui, &view, &mut camera, &mut buf);
            });
        });
        output.textures_delta.clear();
        let lines = output
            .shapes
            .iter()
            .filter(|s| matches!(s.shape, Shape::LineSegment { .. }))
            .count();
        (frame, lines)
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
            s.report
                .nodes
                .iter()
                .map(|n| camera.project(n.pos_column, rect))
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
