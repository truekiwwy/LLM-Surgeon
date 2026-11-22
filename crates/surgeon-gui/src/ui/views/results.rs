//! The Results panel of the agent: the project's measuring runs, newest first. A bench run
//! unfolds into its dataset, the copy's score against the origin's with their 95 %
//! intervals, the items gained and lost, then each item; a graph run opens in 3D.

use eframe::egui::{self, ScrollArea, Sense, Ui, Vec2};

use super::super::components as c;
use super::super::i18n::{t, tf};
use super::super::icons::{self, Icon, State};
use super::super::theme::TOKENS as T;
use super::{age, duration};
use crate::agent::experiments;
use crate::app::{SurgeonApp, Tab};
use crate::studio::ledger::{self, Verdict};
use crate::studio::project::now_unix;
use crate::studio::results::{Answer, Bench, Mode, Run, Score};

pub fn show(ui: &mut Ui, app: &mut SurgeonApp) {
    let entries = app.runs();
    let runs = app.results().to_vec();
    if runs.is_empty() {
        c::empty(ui, Icon::Check, t("results.none"), t("results.none.hint"));
        return;
    }
    let title = app
        .studio
        .manager
        .as_ref()
        .map(|m| m.project.manifest.name.clone())
        .unwrap_or_default();
    export_buttons(ui, app, &title, &entries);
    let mut open3d = None;
    ScrollArea::vertical()
        .id_salt("results")
        .auto_shrink(false)
        .show(ui, |ui| {
            for run in &runs {
                if let Some(path) = run_block(ui, run) {
                    open3d = Some(path);
                }
            }
        });
    if let Some(path) = open3d {
        app.open_graph(ui.ctx(), &path);
        app.tab = Tab::Graph;
    }
}

/// "Copy Markdown" and "Copy JSON": the pure exports of the experiment report, to the clipboard.
fn export_buttons(ui: &mut Ui, app: &mut SurgeonApp, title: &str, entries: &[ledger::Entry]) {
    ui.horizontal(|ui| {
        if c::button(ui, t("results.export.md"), c::ButtonKind::Secondary).clicked() {
            let text = experiments::export_experiments_markdown(title, entries);
            to_clipboard(ui, app, text);
        }
        if c::button(ui, t("results.export.json"), c::ButtonKind::Secondary).clicked() {
            match experiments::export_experiments_json(entries) {
                Ok(text) => to_clipboard(ui, app, text),
                Err(e) => {
                    app.desktop.message =
                        Some(tf("results.export.failed", &[("error", &e.to_string())]))
                }
            }
        }
    });
    ui.add_space(T.space.s);
}

fn to_clipboard(ui: &Ui, app: &mut SurgeonApp, text: String) {
    let lines = text.lines().count();
    ui.ctx().copy_text(text);
    app.desktop.message = Some(tf("report.copied", &[("n", &lines)]));
}

/// A row that folds `body`, its state kept by egui under `id`.
fn fold(ui: &mut Ui, id: egui::Id, label: &str, head: impl FnOnce(&mut Ui)) -> bool {
    let open = ui.data(|d| d.get_temp::<bool>(id)).unwrap_or(false);
    let clicked = c::row(ui, false, label, |ui| {
        let (r, _) = ui.allocate_exact_size(Vec2::splat(T.size.icon), Sense::hover());
        let icon = match open {
            true => Icon::ChevronDown,
            false => Icon::ChevronRight,
        };
        icons::paint(ui.painter(), r, icon, T.color.text_secondary);
        head(ui);
    })
    .clicked();
    let open = open ^ clicked;
    ui.data_mut(|d| d.insert_temp(id, open));
    open
}

/// One run; the graph to open when its button was clicked.
fn run_block(ui: &mut Ui, run: &Run) -> Option<std::path::PathBuf> {
    let state = match run.verdict {
        Verdict::Ok => State::Ok,
        Verdict::Failed => State::Error,
        Verdict::Cancelled | Verdict::Interrupted => State::Warn,
    };
    let tool = run.tool.trim_start_matches("surgeon_");
    let title = match run.bench.as_ref().and_then(|b| b.dataset.as_deref()) {
        Some(dataset) => format!("{tool} · {dataset}"),
        None => tool.to_owned(),
    };
    let when = format!(
        "{} · {}",
        duration(run.seconds),
        age(now_unix().saturating_sub(run.finished_unix))
    );
    let id = egui::Id::new(("result", &run.id));
    let open = fold(ui, id, &title, |ui| {
        c::status(ui, state, title.as_str());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            c::meta(ui, when.as_str());
        });
    });
    if !open {
        return None;
    }
    ui.indent(id, |ui| {
        c::mono_secondary(ui, run.id.as_str());
        if let Some(e) = &run.error {
            c::status(ui, State::Error, e.as_str());
        }
        c::kv(ui, t("results.duration"), &duration(run.seconds));
        if let Some(bench) = &run.bench {
            bench_block(ui, id, bench);
        }
        let opened = run
            .graph
            .as_ref()
            .filter(|_| c::button(ui, t("results.open3d"), c::ButtonKind::Secondary).clicked());
        ui.add_space(T.space.s);
        opened.cloned()
    })
    .inner
}

fn rate(s: &Score) -> String {
    format!(
        "{:.1} % [{:.1}, {:.1}] · n={}",
        s.rate * 100.0,
        s.low * 100.0,
        s.high * 100.0,
        s.n
    )
}

fn bench_block(ui: &mut Ui, id: egui::Id, b: &Bench) {
    let metric = match b.mode {
        Mode::Eval => "accuracy",
        Mode::Refusal => "refusal_rate",
    };
    c::kv(
        ui,
        t("results.dataset"),
        b.dataset.as_deref().unwrap_or(t("results.unknown")),
    );
    c::kv(
        ui,
        &format!("{} · {metric}", t("results.copy")),
        &rate(&b.after),
    );
    if let Some(before) = &b.before {
        c::kv(
            ui,
            &format!("{} · {metric}", t("results.origin")),
            &rate(before),
        );
        c::kv(
            ui,
            t("results.gained"),
            &format!("+{} / -{}", b.gained, b.lost),
        );
    }
    let label = tf("results.items", &[("n", &b.items.len())]);
    if !fold(ui, id.with("items"), &label, |ui| {
        c::line(ui, label.as_str());
    }) {
        return;
    }
    for item in &b.items {
        let mut line = item.id.clone();
        if let Some(gold) = &item.gold {
            line.push_str(&format!(" · {}", tf("results.expected", &[("gold", gold)])));
        }
        c::status(ui, answer_state(&item.after), line.as_str());
        if let Some(q) = &item.question {
            c::secondary(ui, q.as_str());
        }
        if let Some(before) = &item.before {
            answer(ui, t("results.origin"), before);
        }
        answer(ui, t("results.copy"), &item.after);
        ui.add_space(T.space.xs);
    }
}

fn answer_state(a: &Answer) -> State {
    match a.good {
        true => State::Ok,
        false => State::Error,
    }
}

/// The model and its answer on one line (the whole reply on hover).
fn answer(ui: &mut Ui, model: &str, a: &Answer) {
    let first = a.text.lines().next().unwrap_or_default();
    c::status(ui, answer_state(a), format!("{model} : {first}")).on_hover_text(a.text.as_str());
}
