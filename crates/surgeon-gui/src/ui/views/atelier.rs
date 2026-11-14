//! Atelier, three zones: the explorer; Targeting in the center (the ‖ΔW‖ histogram and its
//! cut over the table of targets, or in `[ADVANCE]` the card canvas or the patch editor);
//! Operation on the right (the surgery form, then the run it started and its log). The
//! `[SCALPING]` / `[ADVANCE]` switch sits in the center's header and sets the form's depth.

use eframe::egui::{self, Align2, Pos2, Rect, ScrollArea, Sense, Ui, Vec2};
use weight_surgeon::events::{LogLevel, SurgeonEvent};

use super::super::components::{self as c, ButtonKind, Cell, Column, Segment, Table, Width};
use super::super::i18n::{t, tf};
use super::super::icons::State;
use super::super::layout::{self, Part, Regions};
use super::super::theme::TOKENS as T;
use super::{analyses, delta_values, explorer, metric, metric_label, num, Metric};
use crate::app::{Bench, Mode, SurgeonApp, Tab};
use crate::canvas::view::{self as canvas_view, Status};
use crate::viewport;

pub fn show(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    explorer(ui, app, regions.left);
    match app.bench {
        Bench::Surgery => {
            let center = layout::stack(regions.center, &[Part::Flex(1.0), Part::Flex(1.2)]);
            c::Panel::new("targeting")
                .title(t("panel.surgery"))
                .show_with(ui, center[0], app, work_actions, |ui, app| {
                    let rect = ui.max_rect();
                    histogram(ui, app, rect);
                });
            deltas(ui, app, center[1]);
        }
        Bench::Canvas | Bench::Editor => {
            let title = match app.bench {
                Bench::Canvas => t("panel.canvas"),
                _ => t("panel.editor"),
            };
            c::Panel::new("work").title(title).flush().show_with(
                ui,
                regions.center,
                app,
                work_actions,
                |ui, app| match app.bench {
                    Bench::Canvas => canvas(ui, app),
                    _ => editor(ui, app),
                },
            );
        }
    }

    let right = layout::stack(regions.right, &[Part::Flex(1.6), Part::Flex(1.0)]);
    c::Panel::new("operation")
        .title(t("panel.operation"))
        .show(ui, right[0], |ui| {
            if app.studio.manager.is_none() {
                c::empty(ui, t("project.missing"), t("project.missing.hint"));
                return;
            }
            let all = app.mode == Mode::Advance;
            analyses::form(ui, app, true, all, t("surgery.apply"));
        });
    c::Panel::new("activity")
        .title(t("panel.activity"))
        .show_with(
            ui,
            right[1],
            app,
            |ui, app| {
                if c::button(ui, t("activity.history"), ButtonKind::Ghost).clicked() {
                    app.tab = Tab::Analyses;
                }
            },
            |ui, app| activity(ui, app),
        );
}

/// Right to left: the mode switch, then in `[ADVANCE]` the bench switch.
fn work_actions(ui: &mut Ui, app: &mut SurgeonApp) {
    let mut mode = app.mode;
    c::segmented(
        ui,
        &mut mode,
        &[
            Segment {
                value: Mode::Scalping,
                label: t("mode.scalping"),
                tooltip: t("mode.scalping.tip"),
            },
            Segment {
                value: Mode::Advance,
                label: t("mode.advance"),
                tooltip: t("mode.advance.tip"),
            },
        ],
        c::font_meta(),
    );
    if mode != app.mode {
        app.set_mode(mode);
    }
    if app.mode == Mode::Scalping {
        return;
    }
    ui.add_space(T.space.m);
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

// ------------------------------------------------------------------ targeting

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
        // A band of 24 above the plot carries its count and its cut.
        rect.min + Vec2::new(T.space.l, T.space.xl + T.space.s),
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
    // A quiet count grid in quarters, its top labelled, under the axis and the wire.
    for k in 1..4 {
        let y = plot.bottom() - plot.height() * k as f32 / 4.0;
        c::hline(&painter, plot.x_range(), y, T.color.separator);
    }
    painter.text(
        Pos2::new(plot.left(), rect.top() + T.space.s),
        Align2::LEFT_TOP,
        tf("hist.count", &[("n", &(top as usize))]),
        c::font_meta(),
        T.color.text_secondary,
    );
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
            Pos2::new(plot.right(), rect.top() + T.space.s),
            Align2::RIGHT_TOP,
            tf("hist.cut", &[("cut", &num(cut)), ("n", &above)]),
            c::font_mono(),
            T.color.text,
        );
    } else {
        painter.text(
            Pos2::new(plot.right(), rect.top() + T.space.s),
            Align2::RIGHT_TOP,
            t("hist.cut.hint"),
            c::font_meta(),
            T.color.text_secondary,
        );
    }
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

/// Below this center width the delta table drops two columns.
const NARROW_TABLE: f32 = 640.0;

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
    let targeted = app
        .timeline
        .current()
        .is_some_and(|s| app.targets(&s.report).contains(&true));
    c::Panel::new("deltas").title(title).flush().show_with(
        ui,
        rect,
        app,
        |ui, app| {
            if c::button_if(
                ui,
                targeted,
                t("targets.show3d"),
                ButtonKind::Ghost,
                t("targets.none"),
            )
            .clicked()
            {
                app.show_targets();
            }
        },
        |ui, app| {
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
            // A narrow center keeps the tensor, its change and its share; the role and the
            // third measure come back with the room.
            let keep: &[usize] = match ui.available_width() < NARROW_TABLE {
                true => &[0, 1, 3, 4],
                false => &[0, 1, 2, 3, 4, 5],
            };
            let columns: Vec<Column> = keep
                .iter()
                .map(|&k| Column {
                    title: columns[k].title,
                    width: columns[k].width,
                    right: columns[k].right,
                })
                .collect();
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
                        let mut all = vec![
                            first,
                            Cell::new(&cells[0]),
                            Cell::dim(&cells[1]),
                            Cell::new(&cells[2]),
                            Cell::new(&cells[3]),
                            Cell::new(&cells[4]),
                        ];
                        let mut k = 0;
                        all.retain(|_| {
                            k += 1;
                            keep.contains(&(k - 1))
                        });
                        let response = table.row(ui, selected == Some(*i), &all);
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
        },
    );
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
