//! The Projects workspace: the project list and creation, the open project (state, source,
//! working copies), the launch form of any engine tool, the queue, the run being followed
//! and the ledger exports. Only drawing and wiring: the work is in the sibling modules.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::Instant;

use eframe::egui::{self, ComboBox, ProgressBar, RichText, ScrollArea, TextEdit};
use weight_surgeon::events::{LogLevel, NodeStatus, SurgeonEvent};

use super::bus::{RunView, EVENTS_FILE, FRAME};
use super::estimate::{self, Estimate};
use super::exec::EngineExecutor;
use super::forms::{self, FieldKind, Form};
use super::jobs::{Budgets, Ended, JobSpec, Manager, COPY_TOOL, OUT};
use super::ledger;
use super::project::{Manifest, Project, ProjectState};
use crate::theme::{self, mono};

/// What the window does for this workspace.
pub enum Action {
    /// Open a `graph.json` / `graph.ndjson` without a project (ephemeral).
    OpenGraph(PathBuf),
    /// Show the run of `runs/<id>/` in the 3D view: live while it runs, else replayed from
    /// its `events.ndjson`, over the graph it wrote if any.
    OpenRun {
        dir: PathBuf,
        live: bool,
        target: Target,
    },
}

/// Where a run opens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Graph,
    Diff,
    /// The node the run reached last, in the inspector.
    Inspector,
}

pub struct Studio {
    pub workspace: PathBuf,
    projects: Vec<(PathBuf, Result<Manifest, String>)>,
    new_name: String,
    new_source: String,
    creating: Option<Receiver<Result<Project, String>>>,
    pub manager: Option<Manager>,
    forms: Vec<Form>,
    tool: usize,
    values: BTreeMap<String, String>,
    label: String,
    priority: i32,
    max_wall: String,
    max_candidates: String,
    /// The run shown under the queue: a live one, or one replayed from its file.
    followed: Option<String>,
    replayed: Option<(String, Result<RunView, String>)>,
    /// The preview run in progress and the key of the arguments it estimates.
    estimating: Option<(String, String)>,
    /// The last estimate and the key of its arguments: it holds only while they stay.
    estimate: Option<(String, Estimate)>,
    graph_path: String,
    pub message: Option<(String, bool)>,
}

impl Studio {
    pub fn new(workspace: PathBuf) -> Self {
        let mut studio = Self {
            projects: Vec::new(),
            workspace,
            new_name: String::new(),
            new_source: String::new(),
            creating: None,
            manager: None,
            forms: forms::forms(),
            tool: 0,
            values: BTreeMap::new(),
            label: String::new(),
            priority: 0,
            max_wall: String::new(),
            max_candidates: String::new(),
            followed: None,
            replayed: None,
            estimating: None,
            estimate: None,
            graph_path: String::new(),
            message: None,
        };
        studio.refresh();
        studio
    }

    fn refresh(&mut self) {
        self.projects = Project::list(&self.workspace);
    }

    /// Advances the jobs; keeps the window repainting at 60 Hz while one runs. Returns the
    /// runs that just ended (the agent hub waits for its own).
    pub fn tick(&mut self, ctx: &egui::Context) -> Vec<Ended> {
        if let Some(rx) = &self.creating {
            if let Ok(result) = rx.try_recv() {
                self.creating = None;
                match result {
                    Ok(p) => {
                        self.open(ctx, &p.dir);
                        self.refresh();
                    }
                    Err(e) => self.message = Some((e, true)),
                }
            }
        }
        let Some(m) = &mut self.manager else {
            return Vec::new();
        };
        let all = m.tick(Instant::now());
        for ended in &all {
            let line = format!("{} · {} : {:?}", ended.id, ended.spec.label, ended.verdict);
            let bad = ended.verdict != ledger::Verdict::Ok;
            self.message = Some((line, bad));
            if let Some((_, key)) = self
                .estimating
                .take_if(|(id, _)| *id == ended.id)
                .filter(|_| !bad)
            {
                let e = estimate::read(&ended.envelope);
                self.message = Some((format!("estimation : {}", e.summary()), false));
                self.estimate = Some((key, e));
            }
            if self.followed.as_deref() == Some(ended.id.as_str()) {
                self.replayed = Some((ended.id.clone(), Ok(ended.view.clone())));
            }
        }
        if m.busy() {
            ctx.request_repaint_after(FRAME);
        }
        all
    }

    /// The run the 3D view follows live: the followed one if it runs, else the first running.
    pub fn live_view(&self) -> Option<&RunView> {
        let running = &self.manager.as_ref()?.running;
        running
            .iter()
            .find(|r| self.followed.as_deref() == Some(r.id.as_str()))
            .or(running.first())
            .map(|r| &r.view)
    }

    fn open(&mut self, ctx: &egui::Context, dir: &Path) {
        let wake = ctx.clone();
        match Manager::open(
            dir,
            Arc::new(EngineExecutor::default()),
            Arc::new(move || wake.request_repaint()),
        ) {
            Ok(m) => {
                self.manager = Some(m);
                self.followed = None;
                self.replayed = None;
                self.reset_form();
            }
            Err(e) => self.message = Some((e, true)),
        }
    }

    fn reset_form(&mut self) {
        self.values = match (self.forms.get(self.tool), &self.manager) {
            (Some(form), Some(m)) => default_values(form, &m.project),
            _ => BTreeMap::new(),
        };
        self.label = self
            .forms
            .get(self.tool)
            .map(|f| f.title.clone())
            .unwrap_or_default();
    }

    pub fn show(&mut self, ui: &mut egui::Ui) -> Option<Action> {
        let mut action = None;
        if let Some((text, bad)) = &self.message {
            let color = if *bad { theme::BAD } else { theme::GOOD };
            ui.label(mono(text).color(color));
        }
        ui.columns(3, |cols| {
            action = self.projects_column(&mut cols[0]);
            ScrollArea::vertical()
                .id_salt("launch")
                .show(&mut cols[1], |ui| self.project_column(ui));
            let opened = ScrollArea::vertical()
                .id_salt("runs")
                .show(&mut cols[2], |ui| self.runs_column(ui))
                .inner;
            action = action.take().or(opened);
        });
        action
    }

    fn projects_column(&mut self, ui: &mut egui::Ui) -> Option<Action> {
        ui.heading("Projets");
        ui.label(
            mono(self.workspace.display().to_string())
                .small()
                .color(theme::DIM),
        );
        let ctx = ui.ctx().clone();
        let mut open = None;
        for (dir, manifest) in &self.projects {
            let name = dir.file_name().unwrap_or_default().to_string_lossy();
            match manifest {
                Ok(m) => {
                    let current = self.manager.as_ref().is_some_and(|x| x.project.dir == *dir);
                    let text = format!("{name} · {}", m.state.label());
                    if ui.selectable_label(current, mono(text)).clicked() && !current {
                        open = Some(dir.clone());
                    }
                }
                Err(e) => {
                    ui.label(mono(format!("{name} · illisible")).color(theme::BAD))
                        .on_hover_text(e);
                }
            }
        }
        if let Some(dir) = open {
            if self.manager.as_ref().is_some_and(Manager::busy) {
                self.message = Some(("un job tourne : attendez ou annulez-le".into(), true));
            } else {
                self.open(&ctx, &dir);
            }
        }
        ui.separator();
        ui.label(RichText::new("Nouveau projet").strong());
        ui.add(TextEdit::singleline(&mut self.new_name).hint_text("nom"));
        ui.add(
            TextEdit::singleline(&mut self.new_source)
                .hint_text("modèle d'origine (dossier ou fichier)"),
        );
        let creating = self.creating.is_some();
        let create = ui.add_enabled(!creating, egui::Button::new("Créer"));
        if creating {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(mono("empreinte SHA-256 du modèle…"));
            });
        }
        if create
            .on_hover_text("le modèle est référencé et haché, jamais modifié")
            .clicked()
        {
            let (tx, rx) = std::sync::mpsc::channel();
            let (ws, name) = (self.workspace.clone(), self.new_name.trim().to_owned());
            let source =
                Some(PathBuf::from(self.new_source.trim())).filter(|p| !p.as_os_str().is_empty());
            let wake = ctx.clone();
            std::thread::spawn(move || {
                let created = std::fs::create_dir_all(&ws)
                    .map_err(|e| e.to_string())
                    .and_then(|()| Project::create(&ws, &name, source.as_deref()));
                let _ = tx.send(created);
                wake.request_repaint();
            });
            self.creating = Some(rx);
        }
        ui.separator();
        ui.label(RichText::new("Graphe éphémère").strong())
            .on_hover_text("graph.json ou graph.ndjson, sans projet");
        ui.add(TextEdit::singleline(&mut self.graph_path).hint_text("chemin du graphe"));
        let path = self.graph_path.trim();
        ui.add_enabled(!path.is_empty(), egui::Button::new("Ouvrir dans la vue 3D"))
            .clicked()
            .then(|| Action::OpenGraph(PathBuf::from(path)))
    }

    fn project_column(&mut self, ui: &mut egui::Ui) {
        let Some(m) = &mut self.manager else {
            ui.label(mono("aucun projet ouvert").color(theme::DIM));
            return;
        };
        let manifest = m.project.manifest.clone();
        ui.heading(&manifest.name);
        let state_color = match manifest.state {
            ProjectState::Ready => theme::GOOD,
            ProjectState::Running => theme::SELECTION,
            ProjectState::Interrupted | ProjectState::NeedsAttention => theme::WARN,
        };
        ui.label(mono(manifest.state.label()).color(state_color));
        match manifest.state {
            ProjectState::Interrupted => {
                ui.label("Le studio s'est arrêté pendant des runs :");
                for run in m.resumable() {
                    ui.horizontal(|ui| {
                        ui.label(mono(format!("{} · {}", run.id, run.spec.label)));
                        if ui.button("Reprendre").clicked() {
                            if let Err(e) = m.resume(&run.id) {
                                self.message = Some((e, true));
                            }
                        }
                    });
                }
                if ui.button("Ignorer").clicked() {
                    let _ = m.acknowledge();
                }
            }
            ProjectState::NeedsAttention => {
                ui.label("Un run a échoué ou dépassé son budget (voir l'historique).");
                if ui.button("Vu").clicked() {
                    let _ = m.acknowledge();
                }
            }
            _ => {}
        }
        if let Some(source) = &manifest.source {
            ui.label(mono(format!("origine : {}", source.path.display())))
                .on_hover_text(format!("SHA-256 {}", source.sha256));
            ui.label(
                mono(format!(
                    "{:.2} Gio, lecture seule",
                    source.bytes as f64 / f64::from(1u32 << 30)
                ))
                .color(theme::DIM),
            );
            if ui
                .button("Copie de travail")
                .on_hover_text("copies/<id>/ : le seul endroit où le studio écrit un modèle")
                .clicked()
            {
                let spec = JobSpec {
                    tool: COPY_TOOL.into(),
                    label: "copie de travail".into(),
                    args: serde_json::json!({}),
                    priority: 0,
                    budgets: Budgets::default(),
                    resumed_from: None,
                    read_only: false,
                };
                if let Err(e) = m.submit(spec) {
                    self.message = Some((e, true));
                }
            }
        }
        ui.separator();
        ui.label(RichText::new("Lancer").strong());
        let before = self.tool;
        ComboBox::from_id_salt("tool")
            .selected_text(mono(
                self.forms.get(self.tool).map_or("", |f| f.tool.as_str()),
            ))
            .show_ui(ui, |ui| {
                for (i, f) in self.forms.iter().enumerate() {
                    ui.selectable_value(&mut self.tool, i, mono(&f.tool))
                        .on_hover_text(&f.title);
                }
            });
        let Some(form) = self.forms.get(self.tool) else {
            return;
        };
        if before != self.tool {
            self.values = default_values(form, &m.project);
            self.label = form.title.clone();
        }
        ui.label(RichText::new(&form.description).small().color(theme::DIM));
        ui.label(
            mono(format!("{OUT} = dossier de sortie du run"))
                .small()
                .color(theme::DIM),
        );
        egui::Grid::new("form")
            .num_columns(2)
            .striped(true)
            .show(ui, |ui| {
                for field in &form.fields {
                    let name = if field.required {
                        format!("{} *", field.name)
                    } else {
                        field.name.clone()
                    };
                    ui.label(mono(name)).on_hover_text(&field.description);
                    field_widget(
                        ui,
                        field,
                        self.values.entry(field.name.clone()).or_default(),
                    );
                    ui.end_row();
                }
            });
        ui.horizontal(|ui| {
            ui.label("libellé");
            ui.text_edit_singleline(&mut self.label);
        });
        ui.horizontal(|ui| {
            ui.label("priorité");
            ui.add(egui::DragValue::new(&mut self.priority).range(-10..=10));
            ui.label("durée max (s)");
            ui.add(TextEdit::singleline(&mut self.max_wall).desired_width(60.0));
            if form.tool == "surgeon_pipeline" {
                ui.label("candidats max");
                ui.add(TextEdit::singleline(&mut self.max_candidates).desired_width(40.0));
            }
        });
        // The estimate holds for these exact arguments only.
        let args = form.args(&self.values);
        let key = args
            .as_ref()
            .map(|a| format!("{} {a}", form.tool))
            .unwrap_or_default();
        let preview = args
            .as_ref()
            .ok()
            .filter(|a| !estimate::is_preview(&form.tool, a))
            .and_then(|a| estimate::preview_args(&form.tool, a));
        let current = self.estimate.as_ref().filter(|(k, _)| *k == key);
        match (&preview, current, &self.estimating) {
            (_, _, Some(_)) => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(mono("estimation en cours…"));
                });
            }
            (_, Some((_, e)), _) => {
                ui.label(mono(format!("estimation : {}", e.summary())));
                for w in &e.warnings {
                    ui.label(mono(w).small().color(theme::WARN));
                }
            }
            (Some(_), None, None) => {
                ui.label(
                    mono("pas d'estimation pour ces paramètres")
                        .small()
                        .color(theme::DIM),
                );
            }
            (None, None, None) => {}
        }
        let max_wall: Option<u64> = self.max_wall.trim().parse().ok();
        let mut queue = false;
        ui.horizontal(|ui| {
            let can_estimate = preview.is_some() && self.estimating.is_none();
            if ui
                .add_enabled(can_estimate, egui::Button::new("Estimer"))
                .on_hover_text(
                    "le mode aperçu de l'outil (estimate, dry_run, plan) : rien n'est écrit",
                )
                .on_disabled_hover_text("pas de mode aperçu pour cet outil, ou estimation en cours")
                .clicked()
            {
                if let Some(preview) = preview.clone() {
                    let submitted = m.submit(JobSpec {
                        tool: form.tool.clone(),
                        label: format!("estimation · {}", self.label),
                        args: preview,
                        priority: 10,
                        budgets: Budgets::default(),
                        resumed_from: None,
                        read_only: false,
                    });
                    match submitted {
                        Ok(id) => self.estimating = Some((id, key.clone())),
                        Err(e) => self.message = Some((e, true)),
                    }
                }
            }
            queue = ui.button("Mettre en file").clicked();
        });
        if queue {
            let submitted = args.and_then(|args| {
                estimate::gate(preview.is_some(), max_wall, current.map(|(_, e)| e))?;
                let parse = |s: &str| s.trim().parse().ok();
                m.submit(JobSpec {
                    tool: form.tool.clone(),
                    label: self.label.clone(),
                    args,
                    priority: self.priority,
                    budgets: Budgets {
                        max_wall_time_secs: max_wall,
                        max_candidates: parse(&self.max_candidates).map(|n: u64| n as u32),
                    },
                    resumed_from: None,
                    read_only: false,
                })
            });
            match submitted {
                Ok(id) => {
                    self.followed = Some(id.clone());
                    self.message = Some((format!("{id} en file"), false));
                }
                Err(e) => self.message = Some((e, true)),
            }
        }
    }

    fn runs_column(&mut self, ui: &mut egui::Ui) -> Option<Action> {
        let mut action = None;
        let m = self.manager.as_mut()?;
        ui.heading("Jobs");
        let mut cancel = None;
        for run in &m.running {
            ui.horizontal(|ui| {
                ui.spinner();
                let followed = self.followed.as_deref() == Some(run.id.as_str());
                if ui
                    .selectable_label(followed, mono(format!("{} · {}", run.id, run.spec.label)))
                    .clicked()
                {
                    self.followed = Some(run.id.clone());
                }
                let label = if run.cancelling() {
                    "annulation…"
                } else {
                    "Annuler"
                };
                if ui
                    .add_enabled(!run.cancelling(), egui::Button::new(label))
                    .clicked()
                {
                    cancel = Some(run.id.clone());
                }
            });
        }
        for q in &m.queue {
            ui.horizontal(|ui| {
                ui.label(
                    mono(format!("en file · p{} · {}", q.spec.priority, q.spec.label))
                        .color(theme::DIM),
                );
                if ui.small_button("×").clicked() {
                    cancel = Some(q.id.clone());
                }
            });
        }
        if let Some(id) = cancel {
            if let Err(e) = m.cancel(&id) {
                self.message = Some((e, true));
            }
        }
        if let Some(id) = &self.followed {
            ui.separator();
            let live = m.running.iter().any(|r| &r.id == id);
            ui.horizontal(|ui| {
                ui.label(mono("ouvrir :").color(theme::DIM));
                for (label, target, hint) in [
                    (
                        "3D",
                        Target::Graph,
                        "les nœuds atteints, avec rejeu temporel",
                    ),
                    (
                        "Diff Δ",
                        Target::Diff,
                        "la vue 3D colorée par le Δ du graphe",
                    ),
                    ("Inspecteur", Target::Inspector, "le dernier nœud atteint"),
                ] {
                    if ui.button(label).on_hover_text(hint).clicked() {
                        action = Some(Action::OpenRun {
                            dir: m.project.runs_dir().join(id),
                            live,
                            target,
                        });
                    }
                }
            });
            match m.running.iter().find(|r| &r.id == id) {
                Some(run) => run_view(ui, &run.view),
                None => {
                    let path = m.project.runs_dir().join(id).join(EVENTS_FILE);
                    if self.replayed.as_ref().is_none_or(|(rid, _)| rid != id) && path.exists() {
                        self.replayed = Some((id.clone(), RunView::replay(&path)));
                    }
                    match self.replayed.as_ref().filter(|(rid, _)| rid == id) {
                        Some((_, Ok(view))) => run_view(ui, view),
                        Some((_, Err(e))) => {
                            ui.label(mono(e).color(theme::BAD));
                        }
                        None => {
                            ui.label(mono("en attente").color(theme::DIM));
                        }
                    }
                }
            }
        }
        ui.separator();
        ui.label(RichText::new("Historique").strong());
        for run in m.history().iter().rev().take(30) {
            let text = format!("{} · {:?} · {}", run.id, run.status.status, run.spec.label);
            let followed = self.followed.as_deref() == Some(run.id.as_str());
            if ui
                .selectable_label(followed, mono(text).small())
                .on_hover_text(run.status.note.clone().unwrap_or_default())
                .clicked()
            {
                self.followed = Some(run.id.clone());
            }
        }
        ui.separator();
        ui.horizontal(|ui| {
            for (label, file, render) in [
                (
                    "Exporter JSON",
                    "ledger.json",
                    ledger::to_json as fn(&[ledger::Entry]) -> String,
                ),
                ("Exporter Markdown", "ledger.md", ledger::to_markdown),
            ] {
                if ui.button(label).clicked() {
                    let path = m.project.dir.join(file);
                    let written = ledger::read(&m.project.ledger_path()).and_then(|entries| {
                        std::fs::write(&path, render(&entries)).map_err(|e| e.to_string())
                    });
                    self.message = Some(match written {
                        Ok(()) => (format!("journal exporté : {}", path.display()), false),
                        Err(e) => (e, true),
                    });
                }
            }
        });
        action
    }
}

fn field_widget(ui: &mut egui::Ui, field: &forms::Field, value: &mut String) {
    let hint = field
        .default
        .as_ref()
        .map(|d| d.to_string())
        .unwrap_or_default();
    match &field.kind {
        FieldKind::Bool => {
            ui.horizontal(|ui| {
                let mut on = value == "true";
                if ui.checkbox(&mut on, "").changed() {
                    *value = on.to_string();
                }
                if !value.is_empty()
                    && ui
                        .small_button("×")
                        .on_hover_text("valeur par défaut")
                        .clicked()
                {
                    value.clear();
                }
            });
        }
        FieldKind::Choice(options) => {
            let shown = if value.is_empty() {
                "—"
            } else {
                value.as_str()
            };
            ComboBox::from_id_salt(("field", &field.name))
                .selected_text(mono(shown))
                .show_ui(ui, |ui| {
                    ui.selectable_value(value, String::new(), mono("—"));
                    for o in options {
                        ui.selectable_value(value, o.clone(), mono(o));
                    }
                });
        }
        _ => {
            ui.add(
                TextEdit::singleline(value)
                    .hint_text(hint)
                    .desired_width(220.0),
            );
        }
    }
}

fn run_view(ui: &mut egui::Ui, view: &RunView) {
    ui.label(mono(view.command.as_deref().unwrap_or("…")).strong());
    for (unit, (done, total)) in &view.progress {
        match total {
            Some(t) if *t > 0 => {
                ui.add(
                    ProgressBar::new(*done as f32 / *t as f32)
                        .text(mono(format!("{unit} {done}/{t}"))),
                );
            }
            _ => {
                ui.label(mono(format!("{unit} {done}")));
            }
        }
    }
    if !view.nodes.is_empty() {
        let count = |s: NodeStatus| view.nodes.values().filter(|v| **v == s).count();
        ui.label(mono(format!(
            "nœuds : {} terminés · {} en cours · {} échoués",
            count(NodeStatus::Done),
            count(NodeStatus::Running),
            count(NodeStatus::Failed)
        )));
    }
    for ((node, name), value) in view.metrics.iter().take(12) {
        let node = node
            .as_deref()
            .map(|n| format!("{n} · "))
            .unwrap_or_default();
        ui.label(mono(format!("{node}{name} = {value:.4}")).small());
    }
    for (kind, path, _) in &view.artifacts {
        ui.label(
            mono(format!("{kind} : {}", path.display()))
                .small()
                .color(theme::DIM),
        );
    }
    if view.sampled > 0 {
        ui.label(
            mono(format!(
                "{} ligne(s) de progression échantillonnée(s) à l'écran",
                view.sampled
            ))
            .small()
            .color(theme::DIM),
        );
    }
    match &view.end {
        Some(SurgeonEvent::Finished { .. }) => {
            ui.label(mono("terminé").color(theme::GOOD));
        }
        Some(SurgeonEvent::Cancelled) => {
            ui.label(mono("annulé").color(theme::WARN));
        }
        Some(SurgeonEvent::Failed { kind, message, .. }) => {
            ui.label(mono(format!("échec ({kind}) : {message}")).color(theme::BAD));
        }
        _ => {}
    }
    ScrollArea::vertical()
        .id_salt("logs")
        .max_height(180.0)
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for (level, message) in &view.logs {
                let color = match level {
                    LogLevel::Info => theme::TEXT,
                    LogLevel::Warn => theme::WARN,
                    LogLevel::Error => theme::BAD,
                };
                ui.label(mono(message).small().color(color));
            }
        });
}

/// Prefilled values of a form: outputs in the run folder, the model on the newest working
/// copy (or the source when there is none).
pub fn default_values(form: &Form, project: &Project) -> BTreeMap<String, String> {
    let newest_copy = std::fs::read_dir(project.copies_dir())
        .ok()
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().is_dir() && !e.file_name().to_string_lossy().ends_with(".partial"))
        .max_by_key(|e| e.metadata().and_then(|m| m.modified()).ok())
        .map(|e| format!("copies/{}", e.file_name().to_string_lossy()));
    let model = newest_copy.or_else(|| {
        project
            .manifest
            .source
            .as_ref()
            .map(|s| s.path.display().to_string())
    });
    let out_file = match form.tool.as_str() {
        "surgeon_extract" => "direction.safetensors",
        "surgeon_bench" => "logits.safetensors",
        "surgeon_autotune" => "patch.json",
        "surgeon_diff" => "model.surgery",
        "surgeon_lora" => "adapter_model.safetensors",
        _ => "result",
    };
    let mut values = BTreeMap::new();
    for field in &form.fields {
        let value = match field.name.as_str() {
            "model" => model.clone(),
            "out" => Some(format!("{OUT}/{out_file}")),
            "out_dir" => Some(OUT.into()),
            "job_dir" => Some(format!("{OUT}/job")),
            _ => None,
        };
        if let Some(v) = value {
            values.insert(field.name.clone(), v);
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outputs_default_to_the_run_folder_and_the_model_to_the_newest_copy() {
        let ws = tempfile::tempdir().unwrap();
        let model = ws.path().join("m.gguf");
        std::fs::write(&model, b"w").unwrap();
        let p = Project::create(ws.path(), "p", Some(&model)).unwrap();
        let forms = forms::forms();
        let form = |t: &str| forms.iter().find(|f| f.tool == t).unwrap();
        let v = default_values(form("surgeon_extract"), &p);
        assert_eq!(v["out"], "{out}/direction.safetensors");
        assert_eq!(
            v["model"],
            p.manifest
                .source
                .as_ref()
                .unwrap()
                .path
                .display()
                .to_string()
        );
        std::fs::create_dir_all(p.copies_dir().join("c1")).unwrap();
        std::fs::create_dir_all(p.copies_dir().join("c2.partial")).unwrap();
        let v = default_values(form("surgeon_graph"), &p);
        assert_eq!(v["model"], "copies/c1");
        assert_eq!(v["out_dir"], OUT);
        assert!(form("surgeon_graph").args(&v).is_ok());
    }
}
