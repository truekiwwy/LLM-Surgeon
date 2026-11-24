//! The Results panel of the agent: the project's measuring runs, newest first. A bench run
//! unfolds into its dataset, the copy's score against the origin's with their 95 %
//! intervals, the items gained and lost, then each item; a graph run opens in 3D.

use eframe::egui::{self, ScrollArea, Sense, Ui, Vec2};
use weight_surgeon::eval::GgufQuant;

use super::super::components as c;
use super::super::i18n::{t, tf};
use super::super::icons::{self, Icon, State};
use super::super::theme::TOKENS as T;
use super::{age, elapsed};
use crate::agent::experiments;
use crate::agent::paths::ProjectPaths;
use crate::app::{SurgeonApp, Tab};
use crate::studio::ledger::{self, Verdict};
use crate::studio::ollama::Artifact;
use crate::studio::project::now_unix;
use crate::studio::results::{self as runs, Answer, Bench, Lora, Mode, Ppl, Run, Score};

pub fn show(ui: &mut Ui, app: &mut SurgeonApp) {
    let entries = app.runs();
    let runs = app.results().to_vec();
    let title = app
        .studio
        .manager
        .as_ref()
        .map(|m| m.project.manifest.name.clone())
        .unwrap_or_default();
    // The exports need a working copy, not a measuring run: they come first.
    export_buttons(ui, app, &title, &entries);
    if runs.is_empty() {
        c::empty(ui, Icon::Check, t("results.none"), t("results.none.hint"));
        return;
    }
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

/// "Copy Markdown" and "Copy JSON": the pure exports of the experiment report, to the
/// clipboard; "Export to Ollama": the project's `Modelfile` and its `ollama create` command.
fn export_buttons(ui: &mut Ui, app: &mut SurgeonApp, title: &str, entries: &[ledger::Entry]) {
    ui.horizontal_wrapped(|ui| {
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
        let id = egui::Id::new("ollama-format");
        let mut format = ui.data(|d| d.get_temp::<Option<GgufQuant>>(id)).flatten();
        c::select(
            ui,
            "ollama-format",
            &format_label(format),
            8.0 * T.space.l,
            |ui| {
                for choice in [None, Some(GgufQuant::Q8_0), Some(GgufQuant::Q4KM)] {
                    c::option(ui, &mut format, choice, &format_label(choice));
                }
            },
        );
        ui.data_mut(|d| d.insert_temp(id, format));
        if c::button(ui, t("ollama.export"), c::ButtonKind::Secondary).clicked() {
            app.export_ollama(format);
        }
    });
    // The rank and its button on a row of their own, so that they wrap together.
    ui.horizontal_wrapped(|ui| lora_export(ui, app));
    ollama_command(ui, app);
    ui.add_space(T.space.s);
}

/// The rank of the adapter, then "Extract as LoRA": a studio job of the working copy against
/// the origin, listed below with its SVD fidelity once it ends.
fn lora_export(ui: &mut Ui, app: &mut SurgeonApp) {
    let id = egui::Id::new("lora-rank");
    let mut rank: u32 = ui.data(|d| d.get_temp(id)).unwrap_or(16);
    c::select(
        ui,
        "lora-rank",
        &format!("r = {rank}"),
        6.0 * T.space.l,
        |ui| {
            for r in runs::LORA_RANKS {
                c::option(ui, &mut rank, r, &format!("r = {r}"));
            }
        },
    );
    ui.data_mut(|d| d.insert_temp(id, rank));
    if !c::button(ui, t("lora.extract"), c::ButtonKind::Secondary)
        .on_hover_text(t("lora.extract.tip"))
        .clicked()
    {
        return;
    }
    let paths = app.agents.project.as_deref().map(ProjectPaths::of);
    let job = match paths {
        Some(ProjectPaths {
            project,
            origin: Some(origin),
            copy: Some(copy),
            ..
        }) => runs::lora_job(&project, &origin, &copy, rank),
        _ => Err(t("lora.none").to_owned()),
    };
    let submitted = job.and_then(|job| match app.studio.manager.as_mut() {
        Some(manager) => manager.submit(job).map(drop),
        None => Err(t("lora.none").to_owned()),
    });
    if let Err(e) = submitted {
        app.studio.message = Some((e, true));
    }
}

fn format_label(format: Option<GgufQuant>) -> String {
    match format {
        None => "SafeTensors".into(),
        Some(quant) => format!("GGUF {}", quant.label()),
    }
}

/// The artifact the last export points at, its command and the button that copies it.
fn ollama_command(ui: &mut Ui, app: &mut SurgeonApp) {
    let command = match &app.ollama {
        None => return,
        Some(Err(e)) => {
            c::status(ui, State::Error, e.as_str());
            return;
        }
        Some(Ok(export)) => {
            let (kind, path) = match &export.artifact {
                Artifact::Gguf(path) => ("GGUF", path),
                Artifact::SafeTensors(path) => ("SafeTensors", path),
            };
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            c::kv(ui, t("ollama.artifact"), &format!("{kind} · {name}"));
            c::mono_wrapped(ui, export.command.as_str(), T.color.text);
            export.command.clone()
        }
    };
    if c::button(ui, t("ollama.copy"), c::ButtonKind::Secondary).clicked() {
        ui.ctx().copy_text(command);
        app.desktop.message = Some(t("ollama.copied").into());
    }
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

const TOOK_WIDTH: f32 = 72.0;
const AGE_WIDTH: f32 = 52.0;

fn cell(ui: &mut Ui, width: f32, text: &str) {
    ui.allocate_ui_with_layout(
        Vec2::new(width, T.size.row),
        egui::Layout::right_to_left(egui::Align::Center),
        |ui| c::meta(ui, text),
    );
}

fn ppl_state(delta_pct: f64) -> State {
    match delta_pct {
        d if d <= 2.0 => State::Ok,
        d if d <= 5.0 => State::Warn,
        _ => State::Error,
    }
}

fn ppl_text(p: &Ppl) -> String {
    match (p.before, p.delta_pct) {
        (Some(before), Some(delta)) => format!("{before:.2} → {:.2} · {delta:+.2} %", p.after),
        _ => format!("{:.2}", p.after),
    }
}

/// One run; the graph to open when its button was clicked.
fn run_block(ui: &mut Ui, run: &Run) -> Option<std::path::PathBuf> {
    let state = match (run.verdict, run.ppl.and_then(|p| p.delta_pct)) {
        (Verdict::Ok, Some(delta_pct)) => ppl_state(delta_pct),
        (Verdict::Ok, None) => State::Ok,
        (Verdict::Failed, _) => State::Error,
        (Verdict::Cancelled | Verdict::Interrupted, _) => State::Warn,
    };
    let tool = run.tool.trim_start_matches("surgeon_");
    let title = match (
        &run.lora,
        &run.ppl,
        run.bench.as_ref().and_then(|b| b.dataset.as_deref()),
    ) {
        (Some(l), _, _) => format!("{tool} · r{} · {:.1} %", l.rank, l.fidelity * 100.0),
        (None, Some(p), _) => format!("{tool} · {} {}", t("results.ppl"), ppl_text(p)),
        (None, None, Some(dataset)) => format!("{tool} · {dataset}"),
        (None, None, None) => tool.to_owned(),
    };
    let took = elapsed(run.millis);
    let ago = age(now_unix().saturating_sub(run.finished_unix));
    let id = egui::Id::new(("result", &run.id));
    let open = fold(ui, id, &title, |ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            cell(ui, AGE_WIDTH, &ago);
            cell(ui, TOOK_WIDTH, &took);
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Center), |ui| {
                c::status(ui, state, title.as_str()).on_hover_text(title.as_str());
            });
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
        c::kv(ui, t("results.duration"), &took);
        if let Some(bench) = &run.bench {
            bench_block(ui, id, bench);
        }
        if let Some(lora) = &run.lora {
            lora_block(ui, lora);
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

/// The fidelity of the adapter's SVD, its rank and size, the tensor it fits worst, its file.
fn lora_block(ui: &mut Ui, l: &Lora) {
    c::kv(
        ui,
        t("lora.fidelity"),
        &format!("{:.2} %", l.fidelity * 100.0),
    );
    c::kv(
        ui,
        t("lora.rank"),
        &tf("lora.rank.value", &[("rank", &l.rank), ("n", &l.tensors)]),
    );
    if let Some((name, error)) = &l.worst {
        c::kv(ui, t("lora.worst"), &format!("{:.2} %", error * 100.0));
        c::mono_secondary(ui, name.as_str());
    }
    c::mono_wrapped(ui, l.out.display().to_string(), T.color.text_secondary);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_perplexity_turns_amber_past_two_percent_and_red_past_five() {
        let states: Vec<State> = [-3.0, 0.0, 2.0, 2.01, 5.0, 5.01]
            .into_iter()
            .map(ppl_state)
            .collect();
        assert_eq!(
            states,
            [
                State::Ok,
                State::Ok,
                State::Ok,
                State::Warn,
                State::Warn,
                State::Error
            ]
        );
        let paired = Ppl {
            before: Some(17.0724),
            after: 17.4998,
            delta_pct: Some(2.5034),
        };
        assert_eq!(ppl_text(&paired), "17.07 → 17.50 · +2.50 %");
        let alone = Ppl {
            before: None,
            after: 9.5,
            delta_pct: None,
        };
        assert_eq!(ppl_text(&alone), "9.50");
    }
}
