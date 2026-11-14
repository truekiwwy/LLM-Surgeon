//! Analyses: an engine tool, its typed form (generated from the tool's schema), the budget
//! and the estimate of these exact arguments, the launch; on the right, the queue and the
//! history of every run of the project, surgeries included, filtered by kind. The Operation
//! column of the Atelier reuses the same form.

use eframe::egui::{self, ScrollArea, Ui};

use super::super::components::{self as c, ButtonKind, Cell, Column, Table, Width};
use super::super::i18n::{t, tf};
use super::super::icons::State;
use super::super::layout::Regions;
use super::super::theme::TOKENS as T;
use super::{duration, explorer};
use crate::app::{RunKind, SurgeonApp};
use crate::studio::estimate;
use crate::studio::forms::FieldKind;
use crate::studio::jobs::RunStatus;
use crate::studio::view::{is_surgery, Target};

/// The widest a launch form grows, on the 8 grid.
const FORM_WIDTH: f32 = 720.0;

pub fn show(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    explorer(ui, app, regions.left);
    c::Panel::new("analysis")
        .title(t("panel.analysis"))
        .show(ui, regions.center, |ui| {
            if app.studio.manager.is_none() {
                c::empty(ui, t("project.missing"), t("project.missing.hint"));
                return;
            }
            // A form keeps a reading measure; a wide window does not stretch its action.
            ui.set_max_width(FORM_WIDTH);
            form(ui, app, false, true, t("analysis.launch"));
        });
    queue(ui, app, regions.right);
}

/// Tool, fields, budget and launch of one form; `all`: every field, else the essential ones.
pub fn form(ui: &mut Ui, app: &mut SurgeonApp, surgery: bool, all: bool, action: &str) {
    c::section(ui, t("analysis.step.tool"));
    tool_picker(ui, app, surgery);
    c::section(ui, t("analysis.step.params"));
    // The fields take the rows they need, up to what leaves room for the budget.
    let footer = 8.0 * T.size.row;
    let room = (ui.available_height() - footer).max(T.size.row * 3.0);
    let count = app
        .studio
        .forms
        .get(app.studio.launch(surgery).tool)
        .map_or(1, |f| {
            f.fields.iter().filter(|x| all || f.essential(x)).count()
        });
    let need = count.max(1) as f32 * (T.size.control + T.space.xs);
    ui.allocate_ui(egui::vec2(ui.available_width(), need.min(room)), |ui| {
        fields(ui, app, surgery, all);
    });
    c::separator(ui);
    c::section(ui, t("analysis.step.budget"));
    launch(ui, app, surgery, action);
}

/// The tools of one form kind, and the description of the picked one.
fn tool_picker(ui: &mut Ui, app: &mut SurgeonApp, surgery: bool) {
    let current = app.studio.launch(surgery).tool;
    let title = app
        .studio
        .forms
        .get(current)
        .map(|f| f.title.clone())
        .unwrap_or_default();
    let mut picked = current;
    let options: Vec<(usize, String)> = app
        .studio
        .forms
        .iter()
        .enumerate()
        .filter(|(_, f)| is_surgery(&f.tool) == surgery)
        .map(|(i, f)| (i, f.title.clone()))
        .collect();
    c::select(ui, ("tool", surgery), &title, ui.available_width(), |ui| {
        for (i, name) in &options {
            c::option(ui, &mut picked, *i, name);
        }
    });
    if picked != current {
        app.studio.select_tool(surgery, picked);
    }
    if let Some(form) = app.studio.forms.get(picked) {
        ui.add_space(T.space.xs);
        c::meta(ui, form.description.lines().next().unwrap_or("")).on_hover_text(&form.description);
    }
}

/// One row per field of the tool's schema (`all`, else the essential ones): name on the left,
/// a control of its type.
fn fields(ui: &mut Ui, app: &mut SurgeonApp, surgery: bool, all: bool) {
    let tool = app.studio.launch(surgery).tool;
    let Some(form) = app.studio.forms.get(tool).cloned() else {
        return;
    };
    let launch = app.studio.launch_mut(surgery);
    ScrollArea::vertical()
        .id_salt(("fields", surgery))
        .auto_shrink([false, true])
        .show(ui, |ui| {
            let label_width = (ui.available_width() * 0.4).floor();
            for field in form.fields.iter().filter(|f| all || form.essential(f)) {
                ui.horizontal(|ui| {
                    ui.allocate_ui(egui::vec2(label_width, T.size.control), |ui| {
                        ui.set_width(label_width);
                        let name = match field.required {
                            true => format!("{} *", field.name),
                            false => field.name.clone(),
                        };
                        c::mono_secondary(ui, name).on_hover_text(&field.description);
                    });
                    let width = ui.available_width();
                    let value = launch.values.entry(field.name.clone()).or_default();
                    let hint = field
                        .default
                        .as_ref()
                        .map(|d| d.to_string())
                        .unwrap_or_default();
                    match &field.kind {
                        FieldKind::Bool => {
                            let mut on = value == "true";
                            let before = on;
                            c::checkbox(ui, &mut on, "");
                            if on != before {
                                *value = on.to_string();
                            }
                        }
                        FieldKind::Choice(options) => {
                            let mut picked = value.clone();
                            let shown = match value.is_empty() {
                                true => t("form.choose"),
                                false => value.as_str(),
                            };
                            c::select(ui, ("choice", surgery, &field.name), shown, width, |ui| {
                                for o in options {
                                    c::option(ui, &mut picked, o.clone(), o);
                                }
                            });
                            *value = picked;
                        }
                        _ => {
                            c::field_mono(ui, value, &hint, width)
                                .on_hover_text(&field.description);
                        }
                    }
                });
                ui.add_space(T.space.xs);
            }
        });
}

/// Budget, estimate and the primary action of the form.
fn launch(ui: &mut Ui, app: &mut SurgeonApp, surgery: bool, action: &str) {
    let launch = app.studio.launch(surgery);
    let Some(form) = app.studio.forms.get(launch.tool) else {
        return;
    };
    let prepared = launch.prepare(form);
    let estimating = launch.estimating.is_some();
    let estimate = launch.current(&prepared.key).cloned();
    let max_wall = launch.max_wall_secs();
    let gate = estimate::gate(prepared.preview.is_some(), max_wall, estimate.as_ref());

    let launch = app.studio.launch_mut(surgery);
    ui.horizontal(|ui| {
        let half = (ui.available_width() - T.space.s) / 2.0;
        c::field_mono(ui, &mut launch.max_wall, t("budget.wall.hint"), half)
            .on_hover_text(t("budget.wall.tip"));
        c::field_mono(
            ui,
            &mut launch.max_candidates,
            t("budget.candidates.hint"),
            half,
        )
        .on_hover_text(t("budget.candidates.tip"));
    });
    ui.add_space(T.space.s);
    match (&estimate, max_wall) {
        (Some(e), Some(max)) => {
            let secs = e.seconds.unwrap_or(0.0);
            c::progress(ui, (secs / max as f64) as f32, secs > max as f64);
        }
        _ => c::progress(ui, 0.0, false),
    }
    let line = match (&estimate, estimating) {
        (_, true) => t("estimate.running").to_owned(),
        (Some(e), _) => tf("estimate.line", &[("summary", &e.summary())]),
        (None, false) if prepared.preview.is_none() => t("estimate.none").to_owned(),
        (None, false) => t("estimate.missing").to_owned(),
    };
    c::mono_wrapped(ui, line, T.color.text_secondary);
    if let Some(e) = &estimate {
        for w in &e.warnings {
            c::status(ui, State::Warn, w.as_str());
        }
    }
    let problem = prepared.args.as_ref().err().cloned().or(gate.err());
    if let Some(problem) = &problem {
        c::message(ui, problem, true);
    }
    ui.add_space(T.space.s);
    ui.horizontal(|ui| {
        let can_estimate = prepared.preview.is_some() && !estimating;
        if c::button_if(
            ui,
            can_estimate,
            t("estimate.action"),
            ButtonKind::Secondary,
            t("estimate.why"),
        )
        .clicked()
        {
            app.studio.request_estimate(surgery);
        }
        let ready = problem.is_none();
        let wide = ui.available_width();
        ui.add_enabled_ui(ready, |ui| {
            ui.set_width(wide);
            if c::button_wide(ui, action, ButtonKind::Primary)
                .on_disabled_hover_text(problem.as_deref().unwrap_or(""))
                .clicked()
            {
                app.studio.queue(surgery);
            }
        });
    });
}

/// The right column: running, queued, then finished runs, newest first.
fn queue(ui: &mut Ui, app: &mut SurgeonApp, rect: egui::Rect) {
    let ctx = ui.ctx().clone();
    c::Panel::new("queue")
        .title(t("panel.queue"))
        .flush()
        .show_with(
            ui,
            rect,
            app,
            |ui, app| {
                if c::button(ui, t("ledger.markdown"), ButtonKind::Ghost)
                    .on_hover_text(t("ledger.export.tip"))
                    .clicked()
                {
                    app.studio.export_ledger(true);
                }
                if c::button(ui, t("ledger.json"), ButtonKind::Ghost)
                    .on_hover_text(t("ledger.export.tip"))
                    .clicked()
                {
                    app.studio.export_ledger(false);
                }
                ui.add_space(T.space.s);
                c::segmented_small(
                    ui,
                    &mut app.history,
                    &[
                        (RunKind::All, t("history.all")),
                        (RunKind::Surgery, t("history.surgeries")),
                        (RunKind::Analysis, t("history.analyses")),
                    ],
                    t("history.filter"),
                );
            },
            |ui, app| {
                let Some(m) = app.studio.manager.as_ref() else {
                    ui.add_space(T.space.s);
                    ui.horizontal(|ui| {
                        ui.add_space(T.space.m);
                        c::secondary(ui, t("queue.empty"));
                    });
                    return;
                };
                let kind = app.history;
                let shown = |tool: &str| match kind {
                    RunKind::All => true,
                    RunKind::Surgery => is_surgery(tool),
                    RunKind::Analysis => !is_surgery(tool),
                };
                let mut rows: Vec<(String, String, State, String, String)> = Vec::new();
                for r in m.running.iter().filter(|r| shown(&r.spec.tool)) {
                    rows.push((
                        r.id.clone(),
                        r.spec.label.clone(),
                        State::Active,
                        t("run.running").into(),
                        duration(r.started.elapsed().as_secs()),
                    ));
                }
                for q in m.queue.iter().filter(|q| shown(&q.spec.tool)) {
                    rows.push((
                        q.id.clone(),
                        q.spec.label.clone(),
                        State::Idle,
                        t("run.queued").into(),
                        "-".into(),
                    ));
                }
                // shortcut: the history is read from runs/ every frame; cache it past a few
                // hundred runs.
                for r in m
                    .history()
                    .into_iter()
                    .rev()
                    .filter(|r| shown(&r.spec.tool))
                {
                    let (state, word) = match r.status.status {
                        RunStatus::Finished => (State::Ok, t("run.finished")),
                        RunStatus::Failed => (State::Error, t("run.failed")),
                        RunStatus::Cancelled => (State::Idle, t("run.cancelled")),
                        RunStatus::Interrupted => (State::Warn, t("run.interrupted")),
                        RunStatus::Queued | RunStatus::Running => continue,
                    };
                    let secs = r
                        .status
                        .started_unix
                        .zip(r.status.finished_unix)
                        .map_or("-".into(), |(a, b)| duration(b.saturating_sub(a)));
                    rows.push((r.id.clone(), r.spec.label.clone(), state, word.into(), secs));
                }
                let runs_dir = m.project.runs_dir();
                let running: Vec<String> = m.running.iter().map(|r| r.id.clone()).collect();
                let columns = [
                    Column {
                        title: t("queue.task"),
                        width: Width::Flex(1.0),
                        right: false,
                    },
                    Column {
                        title: t("queue.status"),
                        width: Width::Fixed(96.0),
                        right: false,
                    },
                    Column {
                        title: t("queue.duration"),
                        width: Width::Fixed(64.0),
                        right: true,
                    },
                ];
                let table = Table { columns: &columns };
                table.header(ui);
                if rows.is_empty() {
                    ui.add_space(T.space.s);
                    ui.horizontal(|ui| {
                        ui.add_space(T.space.m);
                        c::secondary(ui, t("queue.none"));
                    });
                }
                let mut follow = None;
                let footer = 3.0 * T.size.row;
                let list_height = (ui.available_height() - footer).max(T.size.row * 3.0);
                ScrollArea::vertical()
                    .id_salt("queue")
                    .max_height(list_height)
                    .auto_shrink(false)
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        for (id, label, state, word, secs) in &rows {
                            let selected = app.studio.followed.as_deref() == Some(id.as_str());
                            let cells =
                                [Cell::new(label), Cell::state(*state, word), Cell::new(secs)];
                            if table.row(ui, selected, &cells).on_hover_text(id).clicked() {
                                follow = Some(id.clone());
                            }
                        }
                    });
                if let Some(id) = follow {
                    app.studio.followed = Some(id);
                }
                let Some(id) = app.studio.followed.clone() else {
                    return;
                };
                c::separator(ui);
                ui.horizontal(|ui| {
                    ui.add_space(T.space.m);
                    let live = running.contains(&id);
                    if c::button(ui, t("run.show3d"), ButtonKind::Secondary).clicked() {
                        app.open_run(&ctx, &runs_dir.join(&id), live, Target::Graph);
                    }
                    if c::button(ui, t("run.diff3d"), ButtonKind::Ghost).clicked() {
                        app.open_run(&ctx, &runs_dir.join(&id), live, Target::Diff);
                    }
                    if live {
                        if let Some(m) = app.studio.manager.as_mut() {
                            if c::button(ui, t("action.cancel"), ButtonKind::Ghost).clicked() {
                                if let Err(e) = m.cancel(&id) {
                                    app.studio.message = Some((e, true));
                                }
                            }
                        }
                    }
                });
            },
        );
}
