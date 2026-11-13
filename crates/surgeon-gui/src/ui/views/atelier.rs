//! Atelier: the explorer; in the center the work (in `[SCALPING]` the ‖ΔW‖ histogram, its cut
//! and the surgery form on the working copy; in `[ADVANCE]` the read-only audit by the agent;
//! or the card canvas, or the patch editor) over the table of deltas; on the right the 3D
//! preview, the live activity of the followed run and the approval waiting for the user.

use eframe::egui::{self, Align2, Pos2, Rect, ScrollArea, Sense, Ui, Vec2};
use weight_surgeon::events::{LogLevel, SurgeonEvent};

use super::super::components::{self as c, ButtonKind, Cell, Column, Segment, Table, Width};
use super::super::i18n::{t, tf};
use super::super::icons::State;
use super::super::layout::{self, Part, Regions};
use super::super::theme::TOKENS as T;
use super::{agents, analyses, delta_values, explorer, graph, metric, metric_label, num, Metric};
use crate::app::{Bench, Mode, SurgeonApp, Tab};
use crate::canvas::view::{self as canvas_view, Status};
use crate::viewport;

pub fn show(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    explorer(ui, app, regions.left);
    let center = layout::stack(regions.center, &[Part::Flex(1.5), Part::Flex(1.0)]);
    let title = match (app.bench, app.mode) {
        (Bench::Surgery, Mode::Scalping) => t("panel.surgery"),
        (Bench::Surgery, Mode::Advance) => t("panel.audit"),
        (Bench::Canvas, _) => t("panel.canvas"),
        (Bench::Editor, _) => t("panel.editor"),
    };
    let flush = app.bench != Bench::Surgery;
    let mut panel = c::Panel::new("work").title(title);
    if flush {
        panel = panel.flush();
    }
    panel.show_with(ui, center[0], app, bench_switch, |ui, app| {
        match app.bench {
            Bench::Surgery => match app.mode {
                Mode::Scalping => surgery(ui, app),
                Mode::Advance => audit(ui, app),
            },
            Bench::Canvas => canvas(ui, app),
            Bench::Editor => editor(ui, app),
        }
    });
    deltas(ui, app, center[1]);

    let right = layout::stack(
        regions.right,
        &[
            Part::Flex(1.25),
            Part::Flex(1.0),
            Part::Fixed(8.0 * T.size.row + T.size.header),
        ],
    );
    c::Panel::new("preview")
        .title(t("panel.preview"))
        .canvas()
        .show(ui, right[0], |ui| graph::viewport(ui, app, true));
    c::Panel::new("activity")
        .title(t("panel.activity"))
        .show(ui, right[1], |ui| activity(ui, app));
    c::Panel::new("approval")
        .title(t("panel.approval"))
        .show(ui, right[2], |ui| agents::approval(ui, app, true));
}

fn bench_switch(ui: &mut Ui, app: &mut SurgeonApp) {
    c::segmented(
        ui,
        &mut app.bench,
        &[
            Segment {
                value: Bench::Surgery,
                label: t("bench.surgery"),
                tooltip: t("bench.surgery.tip"),
            },
            Segment {
                value: Bench::Canvas,
                label: t("bench.canvas"),
                tooltip: t("bench.canvas.tip"),
            },
            Segment {
                value: Bench::Editor,
                label: t("bench.editor"),
                tooltip: t("bench.editor.tip"),
            },
        ],
        c::font_meta(),
    );
}

// ------------------------------------------------------------------ scalping

fn surgery(ui: &mut Ui, app: &mut SurgeonApp) {
    let height = ui.available_height();
    let plot = (height * 0.45).max(T.size.row * 5.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), plot), Sense::hover());
    histogram(ui, app, rect);
    c::separator(ui);
    if app.studio.manager.is_none() {
        c::empty(ui, t("project.missing"), t("project.missing.hint"));
        return;
    }
    let rest = ui.available_rect_before_wrap();
    let half = ((rest.width() - T.space.l) / 2.0).floor();
    let left = Rect::from_min_size(rest.min, Vec2::new(half, rest.height()));
    let right = Rect::from_min_max(Pos2::new(left.right() + T.space.l, rest.top()), rest.max);
    c::region(ui, left, "surgery-fields", |ui| {
        analyses::tool_picker(ui, app, true);
        ui.add_space(T.space.s);
        analyses::fields(ui, app, true);
    });
    c::region(ui, right, "surgery-launch", |ui| {
        analyses::launch(ui, app, true, t("surgery.apply"));
    });
}

/// The histogram of the metric over the visible tensors, as a stepped wire; a click or a
/// drag places the cut, the tensors at or above it become the targets.
fn histogram(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect) {
    c::record(ui.ctx(), "histogram", rect);
    let painter = ui.painter_at(rect);
    painter.rect_filled(c::snap(&painter, rect), 0.0, T.color.canvas);
    c::border(&painter, rect, T.color.separator);
    let Some(step) = app.timeline.current() else {
        let r = rect.shrink(T.space.m);
        c::region(ui, r, "histogram-empty", |ui| {
            c::empty(ui, t("hist.empty"), t("hist.empty.hint"))
        });
        return;
    };
    let which = metric(&step.report);
    let visible = app.visible(&step.report);
    let values: Vec<f64> = delta_values(&step.report)
        .into_iter()
        .zip(&visible)
        .filter_map(|(v, shown)| v.filter(|_| *shown))
        .collect();
    let plot = Rect::from_min_max(
        rect.min + Vec2::new(T.space.l, T.space.l),
        rect.max - Vec2::new(T.space.l, T.size.row + T.space.s),
    );
    let max = values.iter().copied().fold(0.0_f64, f64::max);
    let bins = 32usize;
    let mut counts = vec![0usize; bins];
    for v in &values {
        let k = match max > 0.0 {
            true => ((v / max) * bins as f64) as usize,
            false => 0,
        };
        counts[k.min(bins - 1)] += 1;
    }
    let top = counts.iter().copied().max().unwrap_or(0).max(1) as f32;
    let x_of = |f: f64| plot.left() + f as f32 * plot.width();
    let y_of = |n: usize| plot.bottom() - n as f32 / top * plot.height();
    // Axis, then the stepped outline: wire, no fill.
    c::hline(&painter, plot.x_range(), plot.bottom(), T.color.border);
    c::vline(&painter, plot.left(), plot.y_range(), T.color.border);
    let mut points = vec![Pos2::new(plot.left(), plot.bottom())];
    for (k, n) in counts.iter().enumerate() {
        let (x0, x1) = (
            x_of(k as f64 / bins as f64),
            x_of((k + 1) as f64 / bins as f64),
        );
        points.push(Pos2::new(x0, y_of(*n)));
        points.push(Pos2::new(x1, y_of(*n)));
    }
    points.push(Pos2::new(plot.right(), plot.bottom()));
    painter.add(egui::Shape::line(
        points,
        egui::Stroke::new(T.stroke, T.color.data[5]),
    ));

    // The cut.
    let response = ui.interact(plot, ui.id().with("histogram"), Sense::click_and_drag());
    if let Some(p) = response.interact_pointer_pos().filter(|_| max > 0.0) {
        let f = ((p.x - plot.left()) / plot.width()).clamp(0.0, 1.0);
        app.threshold = Some(f as f64 * max);
    }
    let above = app
        .threshold
        .map_or(0, |cut| values.iter().filter(|v| **v >= cut).count());
    if let Some(cut) = app.threshold.filter(|_| max > 0.0) {
        let x = x_of((cut / max).min(1.0));
        c::vline(&painter, x, plot.y_range(), T.color.accent);
        let handle = Rect::from_center_size(Pos2::new(x, plot.top()), Vec2::splat(T.space.s));
        painter.rect_filled(c::snap(&painter, handle), 0.0, T.color.accent);
    }
    if let Some(p) = response.hover_pos() {
        let f = ((p.x - plot.left()) / plot.width()).clamp(0.0, 0.999);
        let k = (f * bins as f32) as usize;
        let (lo, hi) = (
            max * k as f64 / bins as f64,
            max * (k + 1) as f64 / bins as f64,
        );
        response.on_hover_text(tf(
            "hist.bin",
            &[("lo", &num(lo)), ("hi", &num(hi)), ("n", &counts[k])],
        ));
    }
    // Legend under the axis: the metric, the range, the targets.
    let y = rect.bottom() - T.size.row / 2.0 - T.space.xs;
    painter.text(
        Pos2::new(plot.left(), y),
        Align2::LEFT_CENTER,
        "0",
        c::font_mono(),
        T.color.text_secondary,
    );
    painter.text(
        Pos2::new(plot.center().x, y),
        Align2::CENTER_CENTER,
        tf(
            "hist.legend",
            &[("metric", &metric_label(which)), ("n", &values.len())],
        ),
        c::font_meta(),
        T.color.text_secondary,
    );
    painter.text(
        Pos2::new(plot.right(), y),
        Align2::RIGHT_CENTER,
        num(max),
        c::font_mono(),
        T.color.text_secondary,
    );
    if let Some(cut) = app.threshold {
        painter.text(
            plot.right_top(),
            Align2::RIGHT_TOP,
            tf("hist.cut", &[("cut", &num(cut)), ("n", &above)]),
            c::font_mono(),
            T.color.text,
        );
    } else {
        painter.text(
            plot.right_top(),
            Align2::RIGHT_TOP,
            t("hist.cut.hint"),
            c::font_meta(),
            T.color.text_secondary,
        );
    }
}

// ------------------------------------------------------------------ advance

/// The agent's audit: read only, on the project's connection.
fn audit(ui: &mut Ui, app: &mut SurgeonApp) {
    c::status(ui, State::Ok, t("audit.readonly"));
    c::secondary(ui, t("audit.explain"));
    ui.add_space(T.space.s);
    if !app.agents.connected() {
        c::empty(ui, t("audit.no_connection"), t("audit.no_connection.hint"));
        ui.add_space(T.space.s);
        if c::button(ui, t("audit.open_agents"), ButtonKind::Secondary).clicked() {
            app.tab = Tab::Agents;
        }
        return;
    }
    let footer = 2.0 * T.size.row + T.size.cta;
    let height = (ui.available_height() - footer).max(T.size.row * 2.0);
    ui.allocate_ui(egui::vec2(ui.available_width(), height), |ui| {
        agents::transcript(ui, app, "audit-transcript");
    });
    c::separator(ui);
    if app.agents.input.is_empty() {
        app.agents.input = t("audit.prompt").into();
    }
    c::field(
        ui,
        &mut app.agents.input,
        t("chat.hint"),
        ui.available_width(),
    );
    ui.add_space(T.space.s);
    let busy = app.agents.busy();
    ui.add_enabled_ui(!busy, |ui| {
        if c::button_wide(ui, t("audit.run"), ButtonKind::Primary).clicked() {
            app.agents.send_input();
        }
    });
}

// ------------------------------------------------------------------ canvas and editor

fn canvas(ui: &mut Ui, app: &mut SurgeonApp) {
    let touched = app.timeline.current().map_or(0, |s| s.touched.len());
    let live = app.feed.as_ref().map(|_| {
        tf(
            "canvas.stream",
            &[("states", &app.timeline.len()), ("touched", &touched)],
        )
    });
    let status = Status {
        running: app
            .job
            .as_ref()
            .filter(|j| !j.finished)
            .and_then(|j| j.cards.get(j.current).copied().flatten()),
        busy: app.busy(),
        can_rollback: !app.backups.is_empty(),
        live: live.as_deref(),
    };
    let actions = canvas_view::show(ui, &mut app.canvas, &status);
    let ctx = ui.ctx().clone();
    for action in actions {
        app.canvas_action(&ctx, action);
    }
}

fn editor(ui: &mut Ui, app: &mut SurgeonApp) {
    let (busy, can_rollback) = (app.busy(), !app.backups.is_empty());
    let inner = ui.max_rect().shrink(T.space.m);
    let request = c::region(ui, inner, "editor", |ui| {
        crate::editor::view::show(ui, &mut app.editor, busy, can_rollback)
    });
    if let Some(request) = request {
        let ctx = ui.ctx().clone();
        app.editor_request(&ctx, request);
    }
}

// ------------------------------------------------------------------ deltas

/// A row of the delta table: the node, its five cells, the value it is sorted by.
type DeltaRow = (usize, [String; 5], f64);

/// The tensors by decreasing change against the baseline (or by norm without one); a row
/// selects its tensor.
fn deltas(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect) {
    let which = app.timeline.current().map(|s| metric(&s.report));
    let title = match which {
        Some(Metric::Norm) => t("panel.norms"),
        _ => t("panel.deltas"),
    };
    c::Panel::new("deltas")
        .title(title)
        .flush()
        .show(ui, rect, |ui| {
            let Some(step) = app.timeline.current() else {
                ui.add_space(T.space.s);
                ui.horizontal(|ui| {
                    ui.add_space(T.space.m);
                    c::secondary(ui, t("deltas.empty"));
                });
                return;
            };
            let report = &step.report;
            let selected = app.selected_index();
            let cut = app.threshold;
            let (columns, mut rows): ([Column; 6], Vec<DeltaRow>) = match &report.baseline_diff {
                Some(diff) => {
                    let cols = [
                        Column {
                            title: "#",
                            width: Width::Fixed(48.0),
                            right: false,
                        },
                        Column {
                            title: t("deltas.tensor"),
                            width: Width::Flex(1.0),
                            right: false,
                        },
                        Column {
                            title: t("deltas.role"),
                            width: Width::Fixed(80.0),
                            right: false,
                        },
                        Column {
                            title: "Δ‖W‖F",
                            width: Width::Fixed(88.0),
                            right: true,
                        },
                        Column {
                            title: t("deltas.relative"),
                            width: Width::Fixed(80.0),
                            right: true,
                        },
                        Column {
                            title: "Δ erank",
                            width: Width::Fixed(72.0),
                            right: true,
                        },
                    ];
                    let rows = diff
                        .nodes
                        .iter()
                        .filter_map(|d| {
                            let i = *step.index.get(&d.id)?;
                            let n = &report.nodes[i];
                            Some((
                                i,
                                [
                                    n.tensor.clone(),
                                    viewport::role_label(n.role).into(),
                                    format!("{:+.4}", d.norm_delta),
                                    format!("{:+.2} %", d.relative_change * 100.0),
                                    format!("{:+.3}", d.effective_rank_delta),
                                ],
                                d.relative_change.abs(),
                            ))
                        })
                        .collect();
                    (cols, rows)
                }
                None => {
                    let cols = [
                        Column {
                            title: "#",
                            width: Width::Fixed(48.0),
                            right: false,
                        },
                        Column {
                            title: t("deltas.tensor"),
                            width: Width::Flex(1.0),
                            right: false,
                        },
                        Column {
                            title: t("deltas.role"),
                            width: Width::Fixed(80.0),
                            right: false,
                        },
                        Column {
                            title: "‖W‖F",
                            width: Width::Fixed(88.0),
                            right: true,
                        },
                        Column {
                            title: "σ1",
                            width: Width::Fixed(80.0),
                            right: true,
                        },
                        Column {
                            title: "kurtosis",
                            width: Width::Fixed(72.0),
                            right: true,
                        },
                    ];
                    let rows = report
                        .nodes
                        .iter()
                        .enumerate()
                        .map(|(i, n)| {
                            let m = &n.metrics;
                            (
                                i,
                                [
                                    n.tensor.clone(),
                                    viewport::role_label(n.role).into(),
                                    num(m.frobenius_norm),
                                    super::opt(m.sigma1),
                                    num(m.kurtosis),
                                ],
                                m.frobenius_norm,
                            )
                        })
                        .collect();
                    (cols, rows)
                }
            };
            rows.sort_by(|a, b| b.2.total_cmp(&a.2));
            let table = Table { columns: &columns };
            table.header(ui);
            let mut pick = None;
            ScrollArea::vertical()
                .id_salt("deltas")
                .auto_shrink(false)
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    for (k, (i, cells, value)) in rows.iter().enumerate() {
                        let index = format!("{:03}", k + 1);
                        // Past the cut: the row's number carries the state, in words on hover.
                        let target = cut.is_some_and(|cut| *value >= cut);
                        let first = match target {
                            true => Cell::state(State::Active, &index),
                            false => Cell::dim(&index),
                        };
                        let row = [
                            first,
                            Cell::new(&cells[0]),
                            Cell::dim(&cells[1]),
                            Cell::new(&cells[2]),
                            Cell::new(&cells[3]),
                            Cell::new(&cells[4]),
                        ];
                        let response = table.row(ui, selected == Some(*i), &row);
                        let response = match target {
                            true => response.on_hover_text(t("deltas.target")),
                            false => response,
                        };
                        if response.clicked() {
                            pick = Some(report.nodes[*i].id.clone());
                        }
                    }
                });
            if let Some(id) = pick {
                app.selected = Some(id);
            }
        });
}

// ------------------------------------------------------------------ activity

/// The followed run (else the running one): its progress and its log lines.
fn activity(ui: &mut Ui, app: &mut SurgeonApp) {
    let followed = app.studio.followed_view().and_then(Result::ok).cloned();
    let view = followed.or_else(|| app.studio.live_view().cloned());
    let Some(view) = view else {
        c::empty(ui, t("activity.empty"), t("activity.empty.hint"));
        return;
    };
    if let Some(command) = &view.command {
        c::mono(ui, command.as_str());
    }
    for (name, (done, total)) in &view.progress {
        let line = match total {
            Some(total) => format!("{name}  {done}/{total}"),
            None => format!("{name}  {done}"),
        };
        c::mono_secondary(ui, line);
    }
    ScrollArea::vertical()
        .id_salt("activity")
        .auto_shrink(false)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for (level, text) in &view.logs {
                let (state, tag) = match level {
                    LogLevel::Info => (None, "INFO"),
                    LogLevel::Warn => (Some(State::Warn), "WARN"),
                    LogLevel::Error => (Some(State::Error), "ERR "),
                };
                let line = format!("{tag} {text}");
                match state {
                    Some(state) => c::status(ui, state, line),
                    None => c::mono_secondary(ui, line),
                };
            }
            if let Some(end) = &view.end {
                let (state, word) = match end {
                    SurgeonEvent::Finished { .. } => (State::Ok, t("activity.finished")),
                    SurgeonEvent::Cancelled => (State::Idle, t("activity.cancelled")),
                    _ => (State::Error, t("activity.failed")),
                };
                c::status(ui, state, word);
            }
        });
}
