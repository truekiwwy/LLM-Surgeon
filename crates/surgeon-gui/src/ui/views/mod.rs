//! The five views, and what they share: the explorer of the left column (the project, its
//! working copy, the tensor tree), number formats and the per-tensor values the histogram,
//! the delta table and the targets read.

pub mod agents;
pub mod analyses;
pub mod atelier;
pub mod graph;
pub mod system;

use std::collections::BTreeMap;

use eframe::egui::{self, Rect, ScrollArea, Sense, Ui, Vec2};
use weight_surgeon::graph::output::GraphReport;

use super::components::{self as c, ButtonKind};
use super::i18n::{t, tf};
use super::icons::{self, Icon};
use super::theme::{self, TOKENS as T};
use crate::app::SurgeonApp;
use crate::viewport;

// ------------------------------------------------------------------ numbers

/// Fixed precision in a readable range, scientific outside.
pub fn num(v: f64) -> String {
    match v.abs() {
        a if a != 0.0 && !(1e-3..1e5).contains(&a) => format!("{v:.3e}"),
        _ => format!("{v:.4}"),
    }
}

pub fn opt(v: Option<f64>) -> String {
    v.map_or("-".into(), num)
}

/// Seconds as `42 s`, `1 min 05 s`.
pub fn duration(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s} s"),
        s => format!("{} min {:02} s", s / 60, s % 60),
    }
}

// ------------------------------------------------------------------ per-tensor values

/// What the histogram measures: the change against the baseline when the report carries
/// one, else the norm of the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// |Δ‖W‖F| / ‖W‖F against `--baseline`.
    Delta,
    /// ‖W‖F.
    Norm,
}

pub fn metric(report: &GraphReport) -> Metric {
    match report.baseline_diff {
        Some(_) => Metric::Delta,
        None => Metric::Norm,
    }
}

pub fn metric_label(metric: Metric) -> &'static str {
    match metric {
        Metric::Delta => t("metric.delta"),
        Metric::Norm => t("metric.norm"),
    }
}

/// The value of every node for [`metric`] (`None`: not compared).
pub fn delta_values(report: &GraphReport) -> Vec<Option<f64>> {
    match &report.baseline_diff {
        Some(diff) => {
            let by_id: BTreeMap<&str, f64> = diff
                .nodes
                .iter()
                .map(|d| (d.id.as_str(), d.relative_change.abs()))
                .collect();
            report
                .nodes
                .iter()
                .map(|n| by_id.get(n.id.as_str()).copied())
                .collect()
        }
        None => report
            .nodes
            .iter()
            .map(|n| Some(n.metrics.frobenius_norm))
            .collect(),
    }
}

// ------------------------------------------------------------------ explorer

/// The left column of the work tabs: the project and its copy, then the tensor tree.
pub fn explorer(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect) {
    left_column(ui, app, rect, "explorer", t("panel.explorer"), |ui, app| {
        context(ui, app);
        c::separator(ui);
        tree(ui, app);
    });
}

/// The left column of every tab: a panel whose header folds it (`Ctrl+B` too), or, folded,
/// a rail holding the chevron that unfolds it.
pub fn left_column(
    ui: &mut Ui,
    app: &mut SurgeonApp,
    rect: Rect,
    name: &'static str,
    title: &str,
    body: impl FnOnce(&mut Ui, &mut SurgeonApp),
) {
    if app.columns.collapsed {
        c::Panel::new("left-rail").flush().show(ui, rect, |ui| {
            let top = Rect::from_min_size(rect.min, Vec2::new(rect.width(), T.size.header));
            c::region(ui, top, "left-rail-button", |ui| {
                ui.centered_and_justified(|ui| {
                    if c::icon_button(ui, Icon::ChevronRight, t("left.expand")).clicked() {
                        app.toggle_left();
                    }
                });
            });
        });
        return;
    }
    c::Panel::new(name).title(title).show_with(
        ui,
        rect,
        app,
        |ui, app| {
            if c::icon_button(ui, Icon::ChevronLeft, t("left.collapse")).clicked() {
                app.toggle_left();
            }
        },
        body,
    );
}

/// The project picker, or the creation form, and the working copy.
pub fn context(ui: &mut Ui, app: &mut SurgeonApp) {
    let ctx = ui.ctx().clone();
    let width = ui.available_width();
    let current = app
        .studio
        .manager
        .as_ref()
        .map(|m| m.project.manifest.name.clone());
    let projects: Vec<(std::path::PathBuf, String)> = app
        .studio
        .projects
        .iter()
        .filter_map(|(dir, m)| m.as_ref().ok().map(|m| (dir.clone(), m.name.clone())))
        .collect();
    let shown = current.clone().unwrap_or_else(|| t("project.none").into());
    let mut open = None;
    if !projects.is_empty() {
        c::select(ui, "project-select", &shown, width, |ui| {
            for (dir, name) in &projects {
                let mut picked = current.clone();
                if c::option(ui, &mut picked, Some(name.clone()), name) {
                    open = Some(dir.clone());
                }
            }
        });
    }
    if let Some(dir) = open {
        app.studio.open_project(&ctx, &dir);
    }
    if current.is_none() {
        ui.add_space(T.space.s);
        c::field(ui, &mut app.studio.new_name, t("project.name.hint"), width);
        ui.add_space(T.space.xs);
        c::field_mono(
            ui,
            &mut app.studio.new_source,
            t("project.source.hint"),
            width,
        );
        ui.add_space(T.space.s);
        let ready = !app.studio.new_name.trim().is_empty() && !app.studio.creating();
        let text = match app.studio.creating() {
            true => t("project.creating"),
            false => t("project.create"),
        };
        if c::button_if(
            ui,
            ready,
            text,
            ButtonKind::Secondary,
            t("project.create.why"),
        )
        .clicked()
        {
            app.studio.create_project(&ctx);
        }
        return;
    }
    ui.add_space(T.space.s);
    let entries = app.ledger().to_vec();
    let copy = app.studio.active_copy(&entries);
    ui.horizontal(|ui| match &copy {
        Some(copy) => {
            let (state, word) = match copy.writes {
                0 => (icons::State::Ok, t("copy.intact")),
                _ => (icons::State::Warn, t("copy.modified")),
            };
            c::status(ui, state, word);
            c::mono_secondary(ui, copy.path.as_str());
        }
        None => {
            c::status(ui, icons::State::Idle, t("copy.none"));
        }
    });
    ui.add_space(T.space.s);
    if c::button(ui, t("copy.new"), ButtonKind::Secondary)
        .on_hover_text(t("copy.new.tip"))
        .clicked()
    {
        app.studio.submit_copy();
    }
}

/// The tensors of the graph on screen, by layer; Up / Down walk it, Escape clears.
pub fn tree(ui: &mut Ui, app: &mut SurgeonApp) {
    c::search(
        ui,
        egui::Id::new("tree-search"),
        &mut app.filter,
        t("tree.search"),
    );
    ui.add_space(T.space.xs);
    ui.horizontal(|ui| {
        let names = [t("group.attn"), t("group.mlp"), t("group.embed")];
        for (g, name) in names.into_iter().enumerate() {
            c::checkbox(ui, &mut app.groups[g], name);
        }
    });
    ui.add_space(T.space.xs);
    let Some(step) = app.timeline.current() else {
        c::empty(ui, t("tree.empty"), t("tree.empty.hint"));
        return;
    };
    let visible = app.visible(&step.report);
    let mut layers: BTreeMap<Option<usize>, Vec<usize>> = BTreeMap::new();
    for (i, node) in step.report.nodes.iter().enumerate() {
        if visible[i] {
            layers.entry(node.layer).or_default().push(i);
        }
    }
    let shown = visible.iter().filter(|v| **v).count();
    c::meta(
        ui,
        tf(
            "tree.count",
            &[("shown", &shown), ("total", &visible.len())],
        ),
    );
    let order: Vec<usize> = layers.values().flatten().copied().collect();
    let before = app.selected_index();
    let ctx = ui.ctx().clone();
    app.arrow_selection(&ctx, &order);
    let Some(step) = app.timeline.current() else {
        return;
    };
    let selected = app.selected_index();
    let moved = selected != before;
    let filtering = !app.filter.trim().is_empty();
    let run_states = app.run_view().map(|_| app.layer.states.clone());
    let mut picked: Option<Option<String>> = None;
    let mut toggled = None;
    ScrollArea::vertical()
        .id_salt("tree")
        .auto_shrink(false)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for (layer, nodes) in &layers {
                let open = filtering || app.open_layers.contains(layer);
                let title = match layer {
                    Some(l) => format!("layers.{l}"),
                    None => t("tree.outside").to_owned(),
                };
                let header = c::row(ui, false, &title, |ui| {
                    let icon = if open {
                        Icon::ChevronDown
                    } else {
                        Icon::ChevronRight
                    };
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(T.size.icon), Sense::hover());
                    icons::paint(ui.painter(), r, icon, T.color.text_secondary);
                    c::mono(ui, title.as_str());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        c::meta(ui, nodes.len().to_string());
                    });
                });
                if header.clicked() {
                    toggled = Some(*layer);
                }
                if !open {
                    continue;
                }
                for &i in nodes {
                    let node = &step.report.nodes[i];
                    let is_selected = Some(i) == selected;
                    let response = c::row(ui, is_selected, &node.tensor, |ui| {
                        ui.add_space(T.space.l);
                        let (r, _) =
                            ui.allocate_exact_size(Vec2::splat(T.size.icon), Sense::hover());
                        let shape = theme::role_shape(node.role);
                        let ink = if is_selected {
                            T.color.accent
                        } else {
                            T.color.text_secondary
                        };
                        ui.painter().add(icons::role_outline(
                            shape,
                            r.center(),
                            T.size.dot / 2.0 - 1.0,
                            egui::Stroke::new(T.stroke, ink),
                        ));
                        let mut text = viewport::role_label(node.role).to_owned();
                        if let Some(e) = node.expert {
                            text.push_str(&format!(" · e{e}"));
                        }
                        c::mono(ui, text);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if let Some(state) = run_states
                                .as_ref()
                                .and_then(|s| s.get(i).copied().flatten())
                            {
                                c::meta(ui, viewport::state_style(state).2);
                            } else if step.touched.contains_key(&node.id) {
                                c::meta(ui, t("tree.patched"));
                            }
                        });
                    });
                    let response = response.on_hover_text(&node.tensor);
                    if is_selected && moved {
                        response.scroll_to_me(None);
                    }
                    if response.clicked() {
                        picked = Some(Some(node.id.clone()));
                    }
                }
            }
        });
    if let Some(layer) = toggled {
        if !app.open_layers.remove(&layer) {
            app.open_layers.insert(layer);
        }
    }
    if let Some(picked) = picked {
        app.selected = picked;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use weight_surgeon::graph::stream::GraphPatch;

    pub fn report() -> GraphReport {
        let line = include_str!("../../../tests/fixtures/qwen2-mini.ndjson");
        match serde_json::from_str(line.trim()).unwrap() {
            GraphPatch::Snapshot { report } => *report,
            GraphPatch::Update { .. } => unreachable!(),
        }
    }

    #[test]
    fn numbers_switch_to_scientific_outside_the_readable_range() {
        assert_eq!(num(0.0), "0.0000");
        assert_eq!(num(12.5), "12.5000");
        assert_eq!(num(1.5e-5), "1.500e-5");
        assert_eq!(num(-2.5e6), "-2.500e6");
        assert_eq!(duration(42), "42 s");
        assert_eq!(duration(65), "1 min 05 s");
    }

    #[test]
    fn the_histogram_reads_the_baseline_when_there_is_one() {
        let mut report = report();
        assert_eq!(metric(&report), Metric::Delta);
        let values = delta_values(&report);
        assert_eq!(values.len(), report.nodes.len());
        let changed = report
            .nodes
            .iter()
            .position(|n| n.id == "L1.attn_out")
            .unwrap();
        let top = values.iter().flatten().fold(0.0_f64, |a, b| a.max(*b));
        assert_eq!(
            values[changed],
            Some(top),
            "the one changed node is the largest"
        );
        report.baseline_diff = None;
        assert_eq!(metric(&report), Metric::Norm);
        assert!(delta_values(&report).iter().all(Option::is_some));
    }
}
