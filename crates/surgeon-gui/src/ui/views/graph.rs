//! Graphe 3D: the explorer, the 3D view (Column / Constellation, the diff, the replay of a
//! run, the stream timeline), the inspector of the selected tensor and the chronology of the
//! graph's states.

use eframe::egui::{self, ScrollArea, Ui};
use weight_surgeon::graph::diff::NodeDiff;
use weight_surgeon::graph::metrics::NodeMetrics;
use weight_surgeon::graph::output::{GraphReport, NodeReport};

use super::super::components::{self as c, Cell, Column, Segment, Table, Width};
use super::super::i18n::{t, tf};
use super::super::icons::{Icon, State};
use super::super::layout::{self, Part, Regions};
use super::super::theme::TOKENS as T;
use super::{explorer, num, opt};
use crate::agent::activity;
use crate::app::{LayoutMode, SurgeonApp};
use crate::camera::Orbit;
use crate::source::Step;
use crate::studio::bus::RunView;
use crate::studio::live::Layer;
use crate::studio::vectors::LayerScores;
use crate::viewport::{self, node_title, View};

pub fn show(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    explorer(ui, app, regions.left);
    if !ui.ctx().egui_wants_keyboard_input() && ui.input(|i| i.key_pressed(egui::Key::R)) {
        app.camera = Orbit::default();
    }
    let has_bar = app.replay.is_some() || app.feed.is_some();
    let parts = match has_bar {
        true => layout::stack(
            regions.center,
            &[Part::Flex(1.0), Part::Fixed(T.size.header)],
        ),
        false => vec![regions.center],
    };
    c::Panel::new("graph-view")
        .title(t("panel.graph"))
        .canvas()
        .show_with(ui, parts[0], app, view_controls, |ui, app| {
            viewport(ui, app, false)
        });
    if has_bar {
        c::Panel::new("graph-bar")
            .flush()
            .divided()
            .show(ui, parts[1], |ui| match app.replay.is_some() {
                true => replay_bar(ui, app),
                false => timeline_bar(ui, app),
            });
    }
    let right = layout::stack(regions.right, &[Part::Flex(1.6), Part::Flex(1.0)]);
    c::Panel::new("inspector")
        .title(t("panel.inspector"))
        .show(ui, right[0], |ui| inspector(ui, app));
    c::Panel::new("chronology")
        .title(t("panel.chronology"))
        .flush()
        .divided()
        .show(ui, right[1], |ui| chronology(ui, app));
}

/// Header of the 3D panel, right to left: recenter, diff, layout.
fn view_controls(ui: &mut Ui, app: &mut SurgeonApp) {
    if c::icon_button(ui, Icon::Recenter, t("graph.recenter")).clicked() {
        app.camera = Orbit::default();
    }
    let has_diff = app.timeline.current().is_some_and(|s| {
        s.report.baseline_diff.is_some() || s.touched.values().any(Option::is_some)
    });
    let mut diff = app.diff;
    ui.add_enabled_ui(has_diff, |ui| {
        c::segmented(
            ui,
            &mut diff,
            &[
                Segment {
                    value: false,
                    label: t("graph.weights"),
                    tooltip: t("graph.weights.tip"),
                },
                Segment {
                    value: true,
                    label: t("graph.diff"),
                    tooltip: t("graph.diff.tip"),
                },
            ],
            c::font_body(),
        );
    });
    app.diff = diff && has_diff;
    c::segmented(
        ui,
        &mut app.layout,
        &[
            Segment {
                value: LayoutMode::Column,
                label: t("graph.column"),
                tooltip: t("graph.column.tip"),
            },
            Segment {
                value: LayoutMode::Constellation,
                label: t("graph.constellation"),
                tooltip: t("graph.constellation.tip"),
            },
        ],
        c::font_body(),
    );
}

/// The 3D canvas. `live`: the layers the agent works on stay lit, the others step back, and
/// the origin is overlaid when asked.
pub(super) fn viewport(ui: &mut Ui, app: &mut SurgeonApp, live: bool) {
    let target = match app.layout {
        LayoutMode::Column => 0.0,
        LayoutMode::Constellation => 1.0,
    };
    // The morph explains where each node goes, within the 120 ms every transition keeps.
    let time = if app.motion { T.motion } else { 0.0 };
    let t_ = ui
        .ctx()
        .animate_value_with_time(egui::Id::new("layout-blend"), target, time);
    let blend = t_ * t_ * (3.0 - 2.0 * t_);
    let scores = match live {
        true => app.layer_scores().clone(),
        false => LayerScores::default(),
    };
    let Some(step) = app.timeline.current() else {
        let (title, hint) = match app.feed {
            Some(_) => (t("graph.waiting"), t("graph.waiting.hint")),
            // Why the project's graph could not be read, when it could not.
            None => (
                t("graph.empty"),
                app.stream_error.as_deref().unwrap_or(t("graph.empty.hint")),
            ),
        };
        c::empty(ui, Icon::Cube, title, hint);
        return;
    };
    let visible = app.visible(&step.report);
    // No target set: the agent points at a tensor by selecting it (show_in_3d).
    let targets = vec![false; step.report.nodes.len()];
    let selected = app.selected_index();
    // Field by field, not run_view(): the layer is refilled while the run is borrowed.
    let run = match &app.replay {
        Some(replay) => Some(replay.view()),
        None => app.studio.live_view(),
    };
    match run {
        Some(run) => app.layer.fill(run, step),
        None => app.layer = Layer::default(),
    }
    let activity = match app.agents.busy() {
        true => app.agents.conversation().and_then(activity::current),
        false => None,
    };
    let layer = run.is_some().then_some(&app.layer);
    let focus = match live {
        true => {
            let last = app.agents.conversation().and_then(activity::last);
            super::live::focus(step, layer, last.as_ref(), &scores)
        }
        false => None,
    };
    let view = View {
        step,
        blend,
        diff: app.diff,
        visible: &visible,
        selected,
        targets: &targets,
        layer,
        activity: activity.as_ref(),
        animate: app.motion,
        focus: focus.as_deref(),
        ghost: (live && app.ghost).then(|| app.timeline.get(0)).flatten(),
    };
    if let Some(click) = viewport::show(ui, &view, &mut app.camera, &mut app.buffers) {
        app.selected = click.map(|i| step.report.nodes[i].id.clone());
    }
}

fn replay_bar(ui: &mut Ui, app: &mut SurgeonApp) {
    let Some(replay) = app.replay.as_mut() else {
        return;
    };
    let mut close = false;
    let inner = ui.max_rect().shrink2(egui::vec2(T.space.s, 0.0));
    c::region(ui, inner, "replay-bar", |ui| {
        ui.horizontal_centered(|ui| {
            let last = replay.len();
            let mut cursor = replay.cursor();
            if c::icon_button(ui, Icon::SkipStart, t("replay.start")).clicked() {
                cursor = 0;
            }
            if c::icon_button(ui, Icon::StepBack, t("replay.back")).clicked() {
                cursor = cursor.saturating_sub(1);
            }
            let (icon, tip) = match replay.playing {
                true => (Icon::Pause, t("replay.pause")),
                false => (Icon::Play, t("replay.play")),
            };
            if c::icon_button(ui, icon, tip).clicked() {
                replay.toggle();
                cursor = replay.cursor();
            }
            if c::icon_button(ui, Icon::StepForward, t("replay.forward")).clicked() {
                cursor = (cursor + 1).min(last);
            }
            if c::icon_button(ui, Icon::SkipEnd, t("replay.end")).clicked() {
                cursor = last;
            }
            c::mono(
                ui,
                tf(
                    "replay.position",
                    &[
                        ("line", &replay.cursor()),
                        ("lines", &last),
                        ("t", &format!("{:.1}", replay.t_ms() as f64 / 1000.0)),
                        (
                            "total",
                            &format!("{:.1}", replay.duration_ms() as f64 / 1000.0),
                        ),
                    ],
                ),
            );
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                close = c::icon_button(ui, Icon::Close, t("replay.close")).clicked();
                let mut speed = replay.speed;
                c::select(
                    ui,
                    "replay-speed",
                    &format!("×{speed}"),
                    8.0 * T.space.s,
                    |ui| {
                        for s in [0.25, 1.0, 4.0, 16.0, 64.0] {
                            c::option(ui, &mut speed, s, &format!("×{s}"));
                        }
                    },
                );
                replay.speed = speed;
                c::slider(ui, &mut cursor, last);
            });
            if cursor != replay.cursor() {
                replay.seek(cursor);
            }
        });
    });
    if close {
        app.replay = None;
    }
}

fn timeline_bar(ui: &mut Ui, app: &mut SurgeonApp) {
    let inner = ui.max_rect().shrink2(egui::vec2(T.space.s, 0.0));
    c::region(ui, inner, "timeline-bar", |ui| {
        ui.horizontal_centered(|ui| {
            let len = app.timeline.len();
            let mut cursor = app.timeline.cursor();
            let last = len.saturating_sub(1);
            ui.add_enabled_ui(len > 1, |ui| {
                if c::icon_button(ui, Icon::SkipStart, t("replay.start")).clicked() {
                    cursor = 0;
                }
                if c::icon_button(ui, Icon::StepBack, t("replay.back")).clicked() {
                    cursor = cursor.saturating_sub(1);
                }
                if c::icon_button(ui, Icon::StepForward, t("replay.forward")).clicked() {
                    cursor = (cursor + 1).min(last);
                }
                if c::icon_button(ui, Icon::SkipEnd, t("replay.end")).clicked() {
                    cursor = last;
                }
            });
            let touched = app.timeline.current().map_or(0, |s| s.touched.len());
            c::mono(
                ui,
                tf(
                    "timeline.position",
                    &[
                        ("i", &(cursor + usize::from(len > 0))),
                        ("n", &len),
                        ("touched", &touched),
                    ],
                ),
            );
            let state = match app.timeline.is_live() {
                true => (State::Active, t("timeline.live")),
                false => (State::Idle, t("timeline.replay")),
            };
            c::status(ui, state.0, state.1);
            if let Some(e) = &app.stream_error {
                c::status(ui, State::Error, e.as_str());
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_enabled_ui(len > 1, |ui| c::slider(ui, &mut cursor, last));
            });
            app.timeline.seek(cursor);
        });
    });
}

/// The states of the graph: one per stream patch, the replay's position, or the snapshot.
fn chronology(ui: &mut Ui, app: &mut SurgeonApp) {
    let columns = [
        Column {
            title: t("chrono.state"),
            width: Width::Fixed(48.0),
            right: false,
        },
        Column {
            title: t("chrono.touched"),
            width: Width::Flex(1.0),
            right: false,
        },
        Column {
            title: t("chrono.count"),
            width: Width::Fixed(48.0),
            right: true,
        },
    ];
    let table = Table { columns: &columns };
    table.header(ui);
    if app.timeline.is_empty() {
        ui.add_space(T.space.s);
        ui.horizontal(|ui| {
            ui.add_space(T.space.m);
            c::secondary(ui, t("chrono.empty"));
        });
        return;
    }
    let cursor = app.timeline.cursor();
    let mut seek = None;
    ScrollArea::vertical()
        .id_salt("chronology")
        .auto_shrink(false)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for i in 0..app.timeline.len() {
                let Some(step) = app.timeline.get(i) else {
                    continue;
                };
                let first = step
                    .touched
                    .keys()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| t("chrono.snapshot").into());
                let index = format!("{:03}", i + 1);
                let count = step.touched.len().to_string();
                let cells = [Cell::dim(&index), Cell::new(&first), Cell::new(&count)];
                if table.row(ui, i == cursor, &cells).clicked() {
                    seek = Some(i);
                }
            }
        });
    if let Some(i) = seek {
        app.timeline.seek(i);
    }
}

// ------------------------------------------------------------------ inspector

pub fn inspector(ui: &mut Ui, app: &mut SurgeonApp) {
    let scores = app.layer_scores().clone();
    let Some(step) = app.timeline.current() else {
        c::empty(
            ui,
            Icon::Search,
            t("inspector.empty"),
            t("inspector.empty.hint"),
        );
        return;
    };
    let run = app.run_view();
    let selected = app.selected_index();
    let layer = &app.layer;
    ScrollArea::vertical()
        .id_salt("inspector")
        .auto_shrink(false)
        .show(ui, |ui| match selected {
            Some(i) => {
                node_inspector(ui, step, &step.report.nodes[i]);
                layer_scores(ui, &scores, step.report.nodes[i].layer);
                if let Some(run) = run {
                    run_inspector(ui, run, layer, step, i);
                }
            }
            None => model_summary(ui, &step.report),
        });
}

/// `σ1 / σ_last` over the singular values the report keeps: the condition number when the
/// spectrum is complete, the spread of the top-r spectrum otherwise (`None` if singular).
pub fn spread(m: &NodeMetrics) -> Option<(String, f64)> {
    let first = f64::from(*m.singular_values.first()?);
    let last = f64::from(*m.singular_values.last()?);
    let label = match m.full_spectrum {
        true => "κ (σ1/σmin)".into(),
        false => format!("σ1/σ{}", m.singular_values.len()),
    };
    (last > 0.0).then(|| (label, first / last))
}

/// What deserves attention on a node, from its metrics and the report warnings.
pub fn node_warnings(report: &GraphReport, node: &NodeReport) -> Vec<String> {
    let m = &node.metrics;
    let mut out = Vec::new();
    if m.non_finite > 0 {
        out.push(tf("warn.non_finite", &[("n", &m.non_finite)]));
    }
    let threshold = report.parameters.kurtosis_threshold;
    if m.kurtosis > threshold {
        out.push(tf(
            "warn.kurtosis",
            &[("k", &num(m.kurtosis)), ("max", &num(threshold))],
        ));
    }
    if !m.channels.outliers.is_empty() {
        out.push(tf("warn.outliers", &[("n", &m.channels.outliers.len())]));
    }
    out.extend(
        report
            .model
            .warnings
            .iter()
            .chain(&report.global.warnings)
            .filter(|w| w.contains(&node.id) || w.contains(&node.tensor))
            .cloned(),
    );
    out
}

fn warnings(ui: &mut Ui, list: &[String]) {
    if list.is_empty() {
        return;
    }
    c::section(ui, t("inspector.warnings"));
    for w in list {
        c::status(ui, State::Warn, w.as_str());
    }
}

fn layer_scores(ui: &mut Ui, scores: &LayerScores, layer: Option<usize>) {
    let Some(layer) = layer else {
        return;
    };
    let causal = scores.causal.get(&layer);
    let auc = scores.auc.get(&layer);
    if causal.is_none() && auc.is_none() {
        return;
    }
    c::section(ui, t("inspector.scores"));
    if let Some(score) = causal {
        let component = scores.component.as_deref().unwrap_or("-");
        c::kv(
            ui,
            t("inspector.causal"),
            &format!("{score:+.3} · {component}"),
        );
        let window = match scores.lit(Some(layer)) {
            true => t("inspector.window.in"),
            false => t("inspector.window.out"),
        };
        c::kv(ui, t("inspector.window"), window);
    }
    if let Some(auc) = auc {
        c::kv(ui, "AUC", &format!("{auc:.3}"));
    }
    if let Some(margin) = scores.margin.get(&layer) {
        c::kv(ui, t("inspector.margin"), &format!("{margin:+.3}"));
    }
}

fn run_inspector(ui: &mut Ui, run: &RunView, layer: &Layer, step: &Step, i: usize) {
    let node = &step.report.nodes[i];
    c::section(ui, t("inspector.run"));
    c::kv(
        ui,
        t("inspector.command"),
        run.command.as_deref().unwrap_or("-"),
    );
    c::kv(ui, t("inspector.state"), &viewport::run_line(layer, i));
    for ((n, name), value) in &run.metrics {
        if n.as_ref()
            .is_some_and(|n| *n == node.id || *n == node.tensor)
        {
            c::kv(ui, name, &num(*value));
        }
    }
}

fn node_inspector(ui: &mut Ui, step: &Step, node: &NodeReport) {
    let m = &node.metrics;
    c::strong(ui, node_title(node));
    c::mono_secondary(ui, node.tensor.as_str());
    let measured = step.report.parameters.rank > 0;
    if !measured {
        c::secondary(ui, t("inspector.unmeasured"));
    }

    c::section(ui, t("inspector.tensor"));
    c::kv(
        ui,
        t("inspector.shape"),
        &format!("{} × {}", node.rows, node.cols),
    );
    c::kv(ui, "dtype", &node.dtype);
    c::kv(ui, t("inspector.role"), viewport::role_label(node.role));
    c::kv(
        ui,
        t("inspector.layer"),
        &node.layer.map_or("-".into(), |l| l.to_string()),
    );
    if let Some(e) = node.expert {
        c::kv(ui, t("inspector.expert"), &e.to_string());
    }
    c::kv(ui, t("inspector.bytes"), &node.weight_bytes.to_string());
    if !measured {
        return;
    }

    c::section(ui, t("inspector.spectrum"));
    let interval = |lo: Option<f64>, hi: Option<f64>| match (lo, hi) {
        (Some(lo), Some(hi)) => format!("[{}, {}]", num(lo), num(hi)),
        _ => "-".into(),
    };
    c::kv(ui, t("inspector.sigma1"), &opt(m.sigma1));
    c::kv(
        ui,
        t("inspector.sigma1.certified"),
        &interval(m.sigma1_lower, m.sigma1_upper),
    );
    c::kv(ui, t("inspector.stable_rank"), &opt(m.d_spec));
    c::kv(
        ui,
        t("inspector.erank"),
        &format!(
            "[{}, {}]",
            num(m.erank_energy_lower),
            num(m.erank_energy_upper)
        ),
    );
    c::kv(ui, "‖W‖F", &num(m.frobenius_norm));
    c::kv(ui, t("inspector.rank"), &m.rank.to_string());
    c::kv(ui, t("inspector.energy"), &num(m.energy));
    c::kv(ui, "kurtosis", &num(m.kurtosis));
    if let Some((label, value)) = spread(m) {
        c::kv(ui, &label, &num(value));
    }
    warnings(ui, &node_warnings(&step.report, node));

    let baseline = step
        .report
        .baseline_diff
        .as_ref()
        .and_then(|d| d.nodes.iter().find(|n| n.id == node.id));
    if let Some(diff) = baseline {
        c::section(ui, t("inspector.baseline"));
        delta_table(ui, m, diff);
    }
    if let Some(Some(diff)) = step.touched.get(&node.id) {
        c::section(ui, t("inspector.patch"));
        delta_table(ui, m, diff);
    }
}

/// Rows `metric | before | now | Δ`; before is now minus the delta.
fn delta_table(ui: &mut Ui, m: &NodeMetrics, diff: &NodeDiff) {
    let erank = (m.erank_energy_lower + m.erank_energy_upper) / 2.0;
    let mut lines = vec![
        ("‖W‖F", m.frobenius_norm, diff.norm_delta),
        ("erank", erank, diff.effective_rank_delta),
    ];
    if let (Some(s), Some(d)) = (m.sigma1, diff.sigma1_delta) {
        lines.push(("σ1", s, d));
    }
    let columns = [
        Column {
            title: "",
            width: Width::Flex(1.0),
            right: false,
        },
        Column {
            title: t("delta.base"),
            width: Width::Fixed(64.0),
            right: true,
        },
        Column {
            title: t("delta.now"),
            width: Width::Fixed(64.0),
            right: true,
        },
        Column {
            title: "Δ",
            width: Width::Fixed(64.0),
            right: true,
        },
    ];
    let table = Table { columns: &columns };
    table.header(ui);
    for (name, now, delta) in lines {
        let (base, now_s, d) = (num(now - delta), num(now), format!("{delta:+.4}"));
        table.row(
            ui,
            false,
            &[
                Cell::dim(name),
                Cell::new(&base),
                Cell::new(&now_s),
                Cell::new(&d),
            ],
        );
    }
    let rel = format!("{:+.2} %", diff.relative_change * 100.0);
    let (state, word) = match diff.changed {
        true => (State::Warn, t("delta.changed")),
        false => (State::Idle, t("delta.unchanged")),
    };
    c::status(
        ui,
        state,
        tf("delta.relative", &[("rel", &rel), ("word", &word)]),
    );
}

fn model_summary(ui: &mut Ui, report: &GraphReport) {
    let (m, g) = (&report.model, &report.global);
    c::strong(ui, m.architecture.as_str());
    c::mono_secondary(ui, m.path.as_str());
    if report.parameters.rank == 0 {
        c::secondary(ui, t("inspector.unmeasured"));
    }
    c::section(ui, t("inspector.model"));
    c::kv(ui, t("inspector.format"), &m.format);
    c::kv(ui, t("inspector.layers"), &m.layers.to_string());
    c::kv(ui, "hidden", &m.hidden.to_string());
    c::kv(ui, t("inspector.tensors"), &m.tensor_count.to_string());
    c::kv(ui, "sha256", &m.sha256.chars().take(12).collect::<String>());
    c::section(ui, t("inspector.graph"));
    c::kv(ui, t("inspector.nodes"), &g.nodes.to_string());
    c::kv(ui, t("inspector.edges"), &g.edges.to_string());
    c::kv(ui, t("inspector.kurtosis_median"), &num(g.kurtosis_median));
    c::kv(ui, t("inspector.kurtosis_max"), &num(g.kurtosis_max));
    c::kv(
        ui,
        t("inspector.outliers"),
        &g.outlier_channels.len().to_string(),
    );
    c::kv(
        ui,
        t("inspector.non_finite"),
        &g.non_finite_values.to_string(),
    );
    if let Some(diff) = &report.baseline_diff {
        c::section(ui, t("inspector.baseline"));
        c::kv(
            ui,
            t("inspector.nodes_changed"),
            &format!("{}/{}", diff.nodes_changed, diff.nodes_compared),
        );
        c::kv(
            ui,
            t("inspector.edges_changed"),
            &format!("{}/{}", diff.edges_changed, diff.edges_compared),
        );
        for layer in &diff.top_changed_layers {
            let name = layer
                .layer
                .map_or(t("tree.outside").to_owned(), |l| format!("layers.{l}"));
            // max_relative_change is an absolute value: no sign to show.
            c::kv(
                ui,
                &format!("{name} · {}", layer.top_node),
                &format!("|Δ| {:.2} %", layer.max_relative_change * 100.0),
            );
        }
    }
    let list: Vec<String> = m.warnings.iter().chain(&g.warnings).cloned().collect();
    warnings(ui, &list);
    ui.add_space(T.space.s);
    c::secondary(ui, t("inspector.pick"));
}

#[cfg(test)]
mod tests {
    use super::super::tests::report;
    use super::*;
    use crate::settings::Lang;
    use crate::ui::i18n;

    #[test]
    fn spread_is_a_condition_number_only_with_the_full_spectrum() {
        let mut m = report().nodes[0].metrics.clone();
        m.singular_values = vec![4.0, 2.0, 0.5];
        m.full_spectrum = true;
        assert_eq!(spread(&m), Some(("κ (σ1/σmin)".into(), 8.0)));
        m.full_spectrum = false;
        assert_eq!(spread(&m).unwrap().0, "σ1/σ3");
        m.singular_values = vec![1.0, 0.0];
        assert_eq!(spread(&m), None);
    }

    #[test]
    fn warnings_come_from_the_metrics_and_the_report() {
        i18n::set_lang(Lang::Fr);
        let mut report = report();
        let node = report.nodes[2].clone();
        assert!(node_warnings(&report, &node).is_empty());
        report
            .global
            .warnings
            .push(format!("{} : spectre tronqué", node.id));
        let mut noisy = node;
        noisy.metrics.non_finite = 3;
        noisy.metrics.kurtosis = report.parameters.kurtosis_threshold + 1.0;
        let found = node_warnings(&report, &noisy);
        assert_eq!(found.len(), 3, "{found:?}");
        assert!(found[0].starts_with("3 valeurs"));
        assert!(found[1].starts_with("kurtosis"));
    }

    #[test]
    fn the_fixture_carries_the_baseline_the_diff_view_needs() {
        let report = report();
        let diff = report.baseline_diff.as_ref().unwrap();
        assert_eq!(diff.nodes_changed, 1);
        let changed = diff.nodes.iter().find(|n| n.changed).unwrap();
        assert_eq!(changed.id, "L1.attn_out");
        let node = report.nodes.iter().find(|n| n.id == changed.id).unwrap();
        assert_eq!(node_title(node), "Layer 1 · o_proj");
    }
}
