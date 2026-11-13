//! The central 3D canvas: nodes at their Rust-computed positions (blended between the two
//! layouts), coupling edges, orbit / pan / zoom, picking and a compact hover card.

use std::collections::HashMap;

use eframe::egui::{
    self, Align2, Color32, CornerRadius, FontId, PointerButton, Pos2, Rect, Sense, Stroke,
    StrokeKind, Vec2,
};
use weight_surgeon::graph::diff::{EdgeDiff, NodeDiff};
use weight_surgeon::graph::layout::Position;
use weight_surgeon::graph::output::NodeReport;
use weight_surgeon::graph::GraphRole;

use crate::camera::{self, Orbit};
use crate::source::Step;
use crate::theme;

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

pub struct View<'a> {
    pub step: &'a Step,
    /// 0: `pos_column`, 1: `pos_constellation`.
    pub blend: f32,
    pub diff: bool,
    /// Nodes kept by the filters; the others are drawn faint and cannot be picked.
    pub visible: &'a [bool],
    pub selected: Option<usize>,
}

/// What a click did: `Some(Some(i))` picked node `i`, `Some(None)` clicked empty space.
pub type Click = Option<Option<usize>>;

pub fn show(ui: &mut egui::Ui, view: &View, camera: &mut Orbit) -> Click {
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
    let projected: Vec<_> = report
        .nodes
        .iter()
        .map(|n| camera.project(lerp(n.pos_column, n.pos_constellation, view.blend), rect))
        .collect();
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

    // Edges first, under the nodes.
    let max_weight = report
        .edges
        .iter()
        .map(|e| e.weight.abs())
        .fold(f64::MIN_POSITIVE, f64::max);
    for &(k, a, b) in &view.step.edges {
        let (Some(pa), Some(pb)) = (projected[a], projected[b]) else {
            continue;
        };
        let edge = &report.edges[k];
        let shown = view.visible[a] || view.visible[b];
        let touches_selection = view.selected.is_some_and(|s| s == a || s == b);
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
        painter.line_segment([pa.pos, pb.pos], Stroke::new(width, color));
    }

    // Nodes far to near, so the nearest stay on top.
    let mut order: Vec<usize> = (0..report.nodes.len())
        .filter(|&i| projected[i].is_some())
        .collect();
    order.sort_by(|&a, &b| {
        let depth = |i: usize| projected[i].map_or(0.0, |p| p.depth);
        depth(b).total_cmp(&depth(a))
    });
    let mut targets = Vec::new();
    for &i in &order {
        let Some(p) = projected[i] else { continue };
        let node = &report.nodes[i];
        let radius = (0.02 * p.scale).clamp(2.5, 9.0);
        let fill = match view.diff {
            true => delta_color(node_diffs.get(node.id.as_str()).copied()),
            false => theme::role_color(node.role),
        };
        let fill = match view.visible[i] {
            true => fill,
            false => fill.gamma_multiply(0.12),
        };
        painter.circle_filled(p.pos, radius, fill);
        if view.step.touched.contains_key(&node.id) {
            painter.circle_stroke(p.pos, radius + 2.5, Stroke::new(1.5, theme::AGENT));
        }
        if view.visible[i] {
            targets.push((i, p.pos, radius));
        }
    }

    let hovered = response
        .hover_pos()
        .and_then(|pointer| camera::pick(&targets, pointer, 3.0));
    if let Some(i) = view.selected.filter(|&i| view.visible[i]) {
        if let Some(p) = projected[i] {
            let radius = (0.02 * p.scale).clamp(2.5, 9.0);
            painter.circle_stroke(p.pos, radius + 3.5, Stroke::new(2.0, theme::SELECTION));
            painter.text(
                p.pos + Vec2::new(radius + 7.0, 0.0),
                Align2::LEFT_CENTER,
                node_title(&report.nodes[i]),
                FontId::monospace(11.0),
                theme::SELECTION,
            );
        }
    }
    if let (Some(i), Some(pointer)) = (hovered, response.hover_pos()) {
        if let Some(p) = projected[i] {
            painter.circle_stroke(
                p.pos,
                (0.02 * p.scale).clamp(2.5, 9.0) + 1.5,
                Stroke::new(1.0, theme::TEXT),
            );
        }
        hover_card(&painter, rect, pointer, &report.nodes[i]);
    }

    painter.text(
        rect.left_bottom() + Vec2::new(10.0, -8.0),
        Align2::LEFT_BOTTOM,
        "gauche : orbite   droit : pan   molette : zoom   R : recentrer",
        FontId::proportional(10.5),
        theme::DIM,
    );

    match response.clicked() {
        true => Some(hovered),
        false => None,
    }
}

/// A few monospace lines next to the pointer, kept inside the canvas.
fn hover_card(painter: &egui::Painter, rect: Rect, pointer: Pos2, node: &NodeReport) {
    let m = &node.metrics;
    let sigma = m.sigma1.map_or("-".into(), |s| format!("{s:.4}"));
    let lines = [
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
    let font = FontId::monospace(11.0);
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
}
