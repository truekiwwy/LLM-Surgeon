//! The window state and what the views ask it to do. Drawing lives in `ui`: the shell (title
//! bar, five tabs, status bar, palette) and one view per tab. Here: the graph on screen and
//! its stream, the selection, the camera, the card canvas and the patch editor with their
//! job runner, the studio (projects, jobs, ledger), the agent hub and the desktop services.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::{Duration, SystemTime};

use eframe::egui::{self, Key};
use weight_surgeon::backend::{self, config::BackendConfig, detect::BackendReport};
use weight_surgeon::bench::score::Backend;
use weight_surgeon::graph::diff::NodeDiff;
use weight_surgeon::graph::output::GraphReport;
use weight_surgeon::graph::stream::GraphPatch;
use weight_surgeon::graph::GraphRole;

use crate::agent::view::Hub;
use crate::backend_dialog::BackendDialog;
use crate::camera::Orbit;
use crate::canvas::view::{Action, Canvas};
use crate::canvas::{self, CardKind};
use crate::cli::Item;
use crate::desktop::{Desktop, Event};
use crate::editor::{self, Editor};
use crate::runner::{self, Finished, Job, Task};
use crate::source::{self, Input, Timeline};
use crate::studio::bus::RunView;
use crate::studio::ledger;
use crate::studio::live::{Layer, Replay};
use crate::studio::view::{Studio, Target};
use crate::ui::layout::Columns;
use crate::ui::palette::Palette;
use crate::ui::{fonts, i18n, shell, theme};
use crate::viewport;

/// The five tabs of the title bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Atelier,
    Graph,
    Analyses,
    Agents,
    System,
}

impl Tab {
    pub const ALL: [Tab; 5] = [
        Tab::Atelier,
        Tab::Graph,
        Tab::Analyses,
        Tab::Agents,
        Tab::System,
    ];

    /// The `--screen` name of the tab.
    pub fn name(self) -> &'static str {
        match self {
            Tab::Atelier => "atelier",
            Tab::Graph => "graph",
            Tab::Analyses => "analyses",
            Tab::Agents => "agents",
            Tab::System => "system",
        }
    }

    /// `diagnostics`, the name before G6, still reaches the System tab.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "diagnostics" => Some(Tab::System),
            _ => Self::ALL.into_iter().find(|t| t.name() == name),
        }
    }
}

pub use crate::settings::Mode;

/// What the center of the Atelier shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bench {
    /// Targeting, the ‖ΔW‖ histogram and the surgery form.
    Surgery,
    /// The card canvas.
    Canvas,
    /// The patch.json editor.
    Editor,
}

/// The runs the history lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunKind {
    All,
    Surgery,
    Analysis,
}

/// The four sections of the System tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Machine,
    Providers,
    Updates,
    Journal,
}

impl Section {
    pub const ALL: [Section; 4] = [
        Section::Machine,
        Section::Providers,
        Section::Updates,
        Section::Journal,
    ];

    /// The `--screen system:<name>` part.
    pub fn from_name(name: &str) -> Option<Self> {
        let names = ["machine", "providers", "updates", "journal"];
        names.iter().position(|n| *n == name).map(|k| Self::ALL[k])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayoutMode {
    Column,
    Constellation,
}

/// The role filters of the tensor tree.
pub const GROUPS: [&[GraphRole]; 3] = [
    &[GraphRole::Q, GraphRole::K, GraphRole::V, GraphRole::AttnOut],
    &[GraphRole::Gate, GraphRole::Up, GraphRole::MlpOut],
    &[GraphRole::Embed, GraphRole::LmHead],
];

pub struct SurgeonApp {
    pub timeline: Timeline,
    /// Patches of an NDJSON stream; `None` for a single `graph.json` (no timeline).
    pub feed: Option<Receiver<Result<GraphPatch, String>>>,
    pub stream_error: Option<String>,
    pub layout: LayoutMode,
    pub diff: bool,
    pub camera: Orbit,
    pub selected: Option<String>,
    pub filter: String,
    pub groups: [bool; 3],
    /// Layers of the tensor tree the user opened (`None`: embeddings and head).
    pub open_layers: BTreeSet<Option<usize>>,
    pub dialog: BackendDialog,
    probe: Option<Receiver<BackendReport>>,
    pub hardware: Option<BackendReport>,
    pub tab: Tab,
    pub mode: Mode,
    pub bench: Bench,
    pub canvas: Canvas,
    pub editor: Editor,
    pub job: Option<Job>,
    /// The step that finished last.
    pub last_run: Option<Finished>,
    /// Backups of the last apply: what Rollback restores.
    pub backups: Vec<PathBuf>,
    /// Where card patches are written before `apply` reads them.
    scratch: PathBuf,
    pub studio: Studio,
    pub agents: Hub,
    pub desktop: Desktop,
    /// A recorded run on the 3D view, with its time cursor.
    pub replay: Option<Replay>,
    /// The run on the 3D view mapped onto the current graph, refilled every frame.
    pub layer: Layer,
    pub buffers: viewport::Buffers,
    pub columns: Columns,
    pub palette: Palette,
    /// The cut of the ‖ΔW‖ histogram: tensors at or above it are the targets.
    pub threshold: Option<f64>,
    /// `false` (`--no-motion`): no transition at all.
    pub motion: bool,
    /// The section the System view shows.
    pub system_section: Section,
    /// Which runs the history of Analyses lists.
    pub history: RunKind,
    ledger_cache: Option<(PathBuf, Option<SystemTime>, Vec<ledger::Entry>)>,
}

impl SurgeonApp {
    /// `workspace` is the projects folder of the studio.
    pub fn with_shell(
        ctx: &egui::Context,
        input: Option<Input>,
        desktop: Desktop,
        workspace: PathBuf,
    ) -> Self {
        theme::apply(ctx);
        fonts::install(ctx);
        let mut columns =
            Columns::from_saved(desktop.settings.left_width, desktop.settings.right_width);
        columns.collapsed = desktop.settings.left_collapsed;
        let tab = match input {
            Some(_) => Tab::Graph,
            None => Tab::Atelier,
        };
        let mode = desktop.settings.mode;
        let mut app = Self {
            timeline: Timeline::default(),
            feed: None,
            stream_error: None,
            layout: LayoutMode::Column,
            diff: false,
            camera: Orbit::default(),
            selected: None,
            filter: String::new(),
            groups: [true; 3],
            open_layers: BTreeSet::new(),
            dialog: BackendDialog::load(),
            probe: None,
            hardware: None,
            tab,
            mode,
            bench: Bench::Surgery,
            canvas: Canvas::default(),
            editor: Editor::default(),
            job: None,
            last_run: None,
            backups: Vec::new(),
            scratch: std::env::temp_dir().join(format!("surgeon-studio-{}", std::process::id())),
            studio: Studio::new(workspace),
            agents: Hub::system(ctx),
            desktop,
            replay: None,
            layer: Layer::default(),
            buffers: viewport::Buffers::default(),
            columns,
            palette: Palette::default(),
            threshold: None,
            motion: true,
            system_section: Section::Machine,
            history: RunKind::All,
            ledger_cache: None,
        };
        if app
            .agents
            .cfg
            .adopt_local_only(&mut app.desktop.settings.local_only)
        {
            app.desktop.save_settings();
        }
        if let Some(input) = input {
            app.set_input(ctx, input);
        }
        app.spawn_probe(ctx);
        app
    }

    /// Local only: no request leaves the machine (remote AI, CLI runtimes, downloads).
    pub fn set_local_only(&mut self, on: bool) {
        self.desktop.settings.local_only = on;
        self.desktop.save_settings();
        self.agents.cfg.local_only = on;
    }

    /// No transition at all (`--no-motion`, captures).
    pub fn set_motion(&mut self, ctx: &egui::Context, on: bool) {
        self.motion = on;
        let time = if on { theme::TOKENS.motion } else { 0.0 };
        ctx.all_styles_mut(|s| s.animation_time = time);
    }

    /// Switches the depth of the Atelier and keeps it for the next launch. The agent's own
    /// mode is never touched: it is chosen in Agents only.
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        // The cards and the patch editor belong to `[ADVANCE]`.
        if mode == Mode::Scalping {
            self.bench = Bench::Surgery;
        }
        self.desktop.settings.mode = mode;
        self.desktop.save_settings();
    }

    /// Folds or unfolds the left column (`Ctrl+B`, its chevron) and keeps the choice.
    pub fn toggle_left(&mut self) {
        self.columns.collapsed = !self.columns.collapsed;
        self.desktop.settings.left_collapsed = self.columns.collapsed;
        self.desktop.save_settings();
    }

    /// An external graph, outside any project (ephemeral): replaces the one shown.
    pub fn open_graph(&mut self, ctx: &egui::Context, path: &Path) {
        match source::open(path) {
            Ok(input) => self.set_input(ctx, input),
            Err(e) => self.studio.message = Some((e, true)),
        }
    }

    /// Replays a run's `events.ndjson` over the graph on screen (`--replay`, the run history).
    /// Opens on the end of the run, its last node selected.
    pub fn open_replay(&mut self, path: &Path) -> Result<(), String> {
        let replay = Replay::open(path)?;
        if let Some(last) = &replay.view().last_node {
            self.selected = Some(last.clone());
        }
        self.replay = Some(replay);
        self.tab = Tab::Graph;
        Ok(())
    }

    /// A run of the project on the 3D view, over the graph it wrote when it wrote one
    /// (`out/graph.json`, else `out/graph.ndjson`), else over the graph already open.
    pub fn open_run(&mut self, ctx: &egui::Context, dir: &Path, live: bool, target: Target) {
        let graph = ["graph.json", "graph.ndjson"]
            .into_iter()
            .map(|name| dir.join("out").join(name))
            .find(|p| p.is_file());
        if let Some(graph) = &graph {
            self.open_graph(ctx, graph);
        }
        if graph.is_none() && self.timeline.is_empty() && self.feed.is_none() {
            self.studio.message = Some((i18n::t("run.no_graph").into(), true));
            return;
        }
        self.replay = None;
        let opened = match live {
            // Live: the 3D view follows the running job (Studio::live_view).
            true => Ok(()),
            false => self.open_replay(&dir.join(crate::studio::bus::EVENTS_FILE)),
        };
        if let Err(e) = opened {
            self.studio.message = Some((e, true));
            return;
        }
        self.tab = Tab::Graph;
        self.diff = target == Target::Diff;
        if target == Target::Inspector {
            if let Some(last) = self.run_view().and_then(|v| v.last_node.clone()) {
                self.selected = Some(last);
            }
        }
    }

    /// The run drawn over the 3D view: the replay, else the job running in the project.
    pub fn run_view(&self) -> Option<&RunView> {
        match &self.replay {
            Some(replay) => Some(replay.view()),
            None => self.studio.live_view(),
        }
    }

    /// Shows `input` (a graph or a stream) on the 3D view.
    pub fn set_input(&mut self, ctx: &egui::Context, input: Input) {
        (self.timeline, self.feed) = match input {
            Input::Report(report) => (Timeline::from_report(*report), None),
            Input::Stream { reader, follow } => {
                let wake = ctx.clone();
                let rx = source::spawn_reader(reader, follow, move || wake.request_repaint());
                (Timeline::default(), Some(rx))
            }
        };
        self.stream_error = None;
        self.selected = None;
        self.threshold = None;
    }

    /// `llama-server --list-devices` can take seconds: probe off the UI thread.
    fn spawn_probe(&mut self, ctx: &egui::Context) {
        let (tx, probe) = std::sync::mpsc::channel();
        let user = self.dialog.saved().clone();
        let wake = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(backend::detect::detect_hardware_and_backends(&user));
            wake.request_repaint();
        });
        self.probe = Some(probe);
    }

    /// Opens what the command line carried besides the first graph.
    pub fn open_items(&mut self, ctx: &egui::Context, items: Vec<Item>) {
        if !items.is_empty() {
            self.desktop_event(ctx, Event::Open(items));
        }
    }

    /// What the desktop services asked for: open items, or the backend settings changed.
    fn desktop_event(&mut self, ctx: &egui::Context, event: Event) {
        match event {
            Event::BackendChanged => {
                self.dialog = BackendDialog::load();
                self.hardware = None;
                self.spawn_probe(ctx);
            }
            Event::Open(items) => {
                for item in items {
                    if let Err(e) = self.open_item(ctx, item) {
                        self.desktop.notify(e);
                    }
                }
            }
        }
    }

    fn open_item(&mut self, ctx: &egui::Context, item: Item) -> Result<(), String> {
        match item {
            Item::Graph(path) => {
                let input = source::open(&path)?;
                self.set_input(ctx, input);
                self.tab = Tab::Graph;
            }
            Item::Studio(path) => {
                self.canvas.ws = canvas::Workspace::load(&path)?;
                self.canvas.file = path.display().to_string();
                self.canvas.results.clear();
                self.tab = Tab::Atelier;
                self.bench = Bench::Canvas;
            }
            Item::Folder(dir) => {
                self.desktop.use_workspace(&dir)?;
                self.studio = Studio::new(dir);
            }
            Item::Unknown(path) => {
                return Err(i18n::tf("open.unknown", &[("path", &path.display())]))
            }
        }
        Ok(())
    }

    pub fn busy(&self) -> bool {
        self.job.as_ref().is_some_and(|j| !j.finished)
    }

    fn start(&mut self, ctx: &egui::Context, steps: Vec<runner::Step>) {
        if self.busy() || steps.is_empty() {
            return;
        }
        for id in steps.iter().filter_map(|s| s.card) {
            self.canvas.results.remove(&id);
        }
        let wake = ctx.clone();
        self.job = Some(Job::spawn(steps, move || wake.request_repaint()));
    }

    pub fn rollback(&mut self, ctx: &egui::Context) {
        let step = runner::Step {
            card: None,
            label: "rollback".into(),
            task: Task::Restore(self.backups.clone()),
        };
        self.start(ctx, vec![step]);
    }

    /// Records a finished step on its card, the editor and the rollback state.
    fn finish(&mut self, done: Finished) {
        let data = &done.envelope["data"];
        if done.code == 0 && data["report"]["dry_run"] == false {
            self.backups = canvas::backups_of(&done);
        }
        if done.code == 0 && data["restored"].is_array() {
            self.backups.clear();
        }
        match done.card {
            Some(id) => {
                self.canvas.results.insert(id, done.clone());
            }
            None => self.editor.message = Some((outcome_line(&done), done.code != 0)),
        }
        self.last_run = Some(done);
    }

    pub fn canvas_action(&mut self, ctx: &egui::Context, action: Action) {
        match action {
            Action::Run(id) => match self.canvas.ws.plan(id, &self.scratch) {
                Ok(steps) => self.start(ctx, steps),
                Err(e) => self.canvas.message = Some(e),
            },
            Action::OpenInEditor(id) => match self.canvas.ws.patch_json(id) {
                Ok(text) => {
                    let file = self.scratch.join(format!("card-{id}.patch.json"));
                    self.editor.file = file.display().to_string();
                    self.editor.model = self.canvas.ws.model_of(id).unwrap_or_default().into();
                    self.editor.set_text(text);
                    self.bench = Bench::Editor;
                }
                Err(e) => self.canvas.message = Some(e),
            },
            Action::ShowGraph(id) => {
                if let Err(e) = self.show_graph(id) {
                    self.canvas.message = Some(e);
                }
            }
            Action::Rollback => self.rollback(ctx),
        }
    }

    /// What the patch editor asked for.
    pub fn editor_request(&mut self, ctx: &egui::Context, request: crate::editor::view::Request) {
        match request {
            crate::editor::view::Request::Run(run) => match self.editor.step(run) {
                Ok(step) => self.start(ctx, vec![step]),
                Err(e) => self.editor.message = Some((e, true)),
            },
            crate::editor::view::Request::Rollback => self.rollback(ctx),
        }
    }

    /// Loads the `graph.json` of a Graph Diff card as a new state and frames its changes.
    fn show_graph(&mut self, id: u32) -> Result<(), String> {
        let Some(CardKind::GraphDiff(card)) = self.canvas.ws.card(id).map(|c| &c.kind) else {
            return Ok(());
        };
        let path = PathBuf::from(card.out_dir.trim()).join("graph.json");
        let Input::Report(report) = source::open(&path)? else {
            return Err(i18n::tf("open.not_graph", &[("path", &path.display())]));
        };
        let changed: Vec<&NodeDiff> = report
            .baseline_diff
            .iter()
            .flat_map(|d| &d.nodes)
            .filter(|n| n.changed)
            .collect();
        let marked: Vec<bool> = report
            .nodes
            .iter()
            .map(|n| changed.iter().any(|d| d.id == n.id))
            .collect();
        let center = centroid(&report, &marked, self.layout);
        let top = report
            .baseline_diff
            .as_ref()
            .and_then(|d| d.top_changed_layers.first())
            .map(|l| l.top_node.clone());
        self.timeline.push(&GraphPatch::snapshot(&report))?;
        self.timeline.seek(usize::MAX);
        self.diff = true;
        self.selected = top;
        if let Some(center) = center {
            self.camera.target = center;
        }
        self.tab = Tab::Graph;
        Ok(())
    }

    /// `Voir en 3D` of the Atelier: the 3D graph framed on the targets, which it highlights.
    pub fn show_targets(&mut self) {
        let center = self.timeline.current().and_then(|step| {
            let targets = self.targets(&step.report);
            centroid(&step.report, &targets, self.layout)
        });
        if let Some(center) = center {
            self.camera.target = center;
        }
        self.tab = Tab::Graph;
    }

    /// The 3D nodes the patch being edited would touch (the editor's selectors in the
    /// editor, else those of the selected Patch card), and in `[SCALPING]` the tensors at or
    /// above the histogram cut.
    pub fn targets(&self, report: &GraphReport) -> Vec<bool> {
        let from_editor = self.bench == Bench::Editor && !self.editor.checked.selections.is_empty();
        let selections = match from_editor {
            true => self.editor.checked.selections.clone(),
            false => self
                .canvas
                .selected
                .and_then(|id| self.canvas.ws.card(id))
                .and_then(|c| match &c.kind {
                    CardKind::Patch(p) => p.selection(),
                    _ => None,
                })
                .into_iter()
                .collect(),
        };
        let mut targets = editor::targets(&selections, report);
        if let (Some(cut), Mode::Scalping) = (self.threshold, self.mode) {
            let values = crate::ui::views::delta_values(report);
            for (t, v) in targets.iter_mut().zip(values) {
                *t |= v.is_some_and(|v| v >= cut);
            }
        }
        targets
    }

    /// The project's ledger, read again only when the file changed.
    pub fn ledger(&mut self) -> &[ledger::Entry] {
        let path = self
            .studio
            .manager
            .as_ref()
            .map(|m| m.project.ledger_path());
        let Some(path) = path else {
            self.ledger_cache = None;
            return &[];
        };
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        let stale = self
            .ledger_cache
            .as_ref()
            .is_none_or(|(p, m, _)| *p != path || *m != modified);
        if stale {
            let entries = ledger::read(&path).unwrap_or_default();
            self.ledger_cache = Some((path, modified, entries));
        }
        self.ledger_cache.as_ref().map_or(&[], |(_, _, e)| e)
    }

    /// Drains the background channels and runs the debounced validation.
    fn poll(&mut self, ctx: &egui::Context) {
        if let Some(feed) = &self.feed {
            for patch in feed.try_iter() {
                if let Err(e) = patch.and_then(|p| self.timeline.push(&p)) {
                    self.stream_error = Some(e);
                }
            }
        }
        if let Some(report) = self.probe.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.dialog.prefill(&report);
            self.hardware = Some(report);
            self.probe = None;
        }
        let ended = self.studio.tick(ctx);
        self.agents.tick(&mut self.studio, &ended);
        if let Some(target) = self.agents.take_focus() {
            self.selected = Some(target);
        }
        if self.agents.busy() {
            ctx.request_repaint_after(crate::studio::bus::FRAME);
        }
        if let Some(replay) = self.replay.as_mut().filter(|r| r.playing) {
            replay.advance(Duration::from_secs_f32(ctx.input(|i| i.stable_dt)));
            ctx.request_repaint();
        }
        let finished = self.job.as_mut().map(Job::poll).unwrap_or_default();
        for done in finished {
            self.finish(done);
        }
        if self.busy() {
            ctx.request_repaint_after(Duration::from_millis(250));
        }
        if let Some(wait) = self.editor.tick() {
            ctx.request_repaint_after(wait);
        }
    }

    /// Nodes kept by the role filters and the search field.
    pub fn visible(&self, report: &GraphReport) -> Vec<bool> {
        let needle = self.filter.trim().to_lowercase();
        report
            .nodes
            .iter()
            .map(|n| {
                let group = GROUPS
                    .iter()
                    .position(|roles| roles.contains(&n.role))
                    .is_none_or(|g| self.groups[g]);
                group
                    && (needle.is_empty()
                        || n.id.to_lowercase().contains(&needle)
                        || n.tensor.to_lowercase().contains(&needle))
            })
            .collect()
    }

    /// The selected node's index in the graph on screen.
    pub fn selected_index(&self) -> Option<usize> {
        let step = self.timeline.current()?;
        self.selected
            .as_ref()
            .and_then(|s| step.index.get(s))
            .copied()
    }

    /// `None` while the probe runs; the first GPU, else the CPU.
    pub fn hardware_label(&self) -> Option<String> {
        let r = self.hardware.as_ref()?;
        Some(match r.llama.as_ref().and_then(|l| l.devices.first()) {
            Some(gpu) => format!("{} · {}", gpu.name, gpu.api()),
            None => "CPU".into(),
        })
    }

    pub fn backend_label(&self) -> String {
        let saved: &BackendConfig = self.dialog.saved();
        let kind = saved.backend.unwrap_or(Backend::Native);
        let api = match kind {
            Backend::LlamaServer => self
                .hardware
                .as_ref()
                .and_then(|r| r.llama.as_ref())
                .and_then(|l| l.devices.first())
                .map(|d| d.api()),
            _ => None,
        };
        backend::describe(
            kind.label(),
            api,
            saved.gpu_layers,
            saved.batch_size.unwrap_or(1),
        )
    }

    /// A selection made by tensor name (a run's `node_id`) becomes the graph id.
    fn normalize_selection(&mut self) {
        let id = self
            .timeline
            .current()
            .zip(self.selected.as_ref())
            .and_then(|(step, s)| step.index.get(s).map(|&i| &step.report.nodes[i].id))
            .filter(|id| Some(*id) != self.selected.as_ref())
            .cloned();
        if id.is_some() {
            self.selected = id;
        }
    }

    /// Up / Down walk `order` (the tree as shown), Escape clears; not while a field has the
    /// keys.
    pub fn arrow_selection(&mut self, ctx: &egui::Context, order: &[usize]) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        let Some(step) = self.timeline.current() else {
            return;
        };
        let current = self.selected_index();
        let (down, up, escape) = ctx.input(|i| {
            (
                i.key_pressed(Key::ArrowDown),
                i.key_pressed(Key::ArrowUp),
                i.key_pressed(Key::Escape),
            )
        });
        let moved = match (down, up) {
            (true, false) => step_selection(order, current, 1),
            (false, true) => step_selection(order, current, -1),
            _ => None,
        };
        if let Some(i) = moved {
            let node = &step.report.nodes[i];
            self.open_layers.insert(node.layer);
            self.selected = Some(node.id.clone());
        } else if escape {
            self.selected = None;
        }
    }

    /// Persists the column widths after a drag.
    pub fn columns_changed(&mut self) {
        self.desktop.settings.left_width = Some(self.columns.left);
        self.desktop.settings.right_width = Some(self.columns.right);
        self.desktop.save_settings();
    }

    /// One frame of the whole window; no `eframe::Frame` needed, so that off-screen rendering
    /// (`--screenshot`, tests) can drive it.
    pub fn draw(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        i18n::set_lang(self.desktop.lang());
        self.poll(&ctx);
        for event in self.desktop.ui(ui) {
            self.desktop_event(&ctx, event);
        }
        self.normalize_selection();
        shell::show(ui, self);
        self.dialog.show(&ctx, self.hardware.as_ref());
    }
}

impl eframe::App for SurgeonApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.draw(ui);
    }
}

/// One line about a finished step: what an apply touched, what a rollback restored, or the
/// first line of the error.
pub fn outcome_line(run: &Finished) -> String {
    let data = &run.envelope["data"];
    if run.code != 0 {
        let message = run.envelope["error"]["message"].as_str().unwrap_or("");
        return i18n::tf(
            "outcome.failed",
            &[
                ("code", &run.code),
                ("message", &message.lines().next().unwrap_or("")),
            ],
        );
    }
    if let Some(files) = data["restored"].as_array() {
        return i18n::tf("outcome.restored", &[("n", &files.len())]);
    }
    let tensors: usize = data["report"]["operations"].as_array().map_or(0, |ops| {
        ops.iter()
            .filter_map(|o| o["tensors"].as_array())
            .map(Vec::len)
            .sum()
    });
    match data["report"]["dry_run"].as_bool() {
        Some(true) => i18n::tf("outcome.dry_run", &[("n", &tensors)]),
        Some(false) => i18n::tf(
            "outcome.applied",
            &[
                ("n", &tensors),
                ("backups", &data["backups"].as_array().map_or(0, Vec::len)),
            ],
        ),
        None => i18n::tf("outcome.ok", &[("s", &format!("{:.1}", run.seconds))]),
    }
}

/// The mean position of the `marked` nodes in `layout`; `None` when none is marked.
pub fn centroid(report: &GraphReport, marked: &[bool], layout: LayoutMode) -> Option<[f32; 3]> {
    let points: Vec<[f32; 3]> = report
        .nodes
        .iter()
        .zip(marked)
        .filter(|(_, m)| **m)
        .map(|(n, _)| match layout {
            LayoutMode::Column => n.pos_column,
            LayoutMode::Constellation => n.pos_constellation,
        })
        .collect();
    let n = points.len() as f32;
    let mean = |k: usize| points.iter().map(|p| p[k]).sum::<f32>() / n;
    (!points.is_empty()).then(|| [mean(0), mean(1), mean(2)])
}

/// The node `delta` places after `current` in the list as shown, held at both ends. With no
/// selection, or one the filters hide, Down starts at the top and Up at the bottom.
pub fn step_selection(order: &[usize], current: Option<usize>, delta: isize) -> Option<usize> {
    let at = current.and_then(|c| order.iter().position(|&i| i == c));
    let next = match (at, delta > 0) {
        (Some(at), _) => at
            .saturating_add_signed(delta)
            .min(order.len().checked_sub(1)?),
        (None, true) => 0,
        (None, false) => order.len().checked_sub(1)?,
    };
    order.get(next).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Lang;

    #[test]
    fn arrow_keys_walk_the_shown_list_and_stop_at_its_ends() {
        let order = [4, 9, 2];
        assert_eq!(step_selection(&order, Some(9), 1), Some(2));
        assert_eq!(step_selection(&order, Some(9), -1), Some(4));
        assert_eq!(
            step_selection(&order, Some(2), 1),
            Some(2),
            "held at the end"
        );
        assert_eq!(
            step_selection(&order, Some(4), -1),
            Some(4),
            "held at the top"
        );
        assert_eq!(step_selection(&order, None, 1), Some(4));
        assert_eq!(
            step_selection(&order, Some(7), -1),
            Some(2),
            "hidden: from the end"
        );
        assert_eq!(step_selection(&[], None, 1), None);
    }

    #[test]
    fn outcome_lines_describe_apply_rollback_and_errors() {
        i18n::set_lang(Lang::Fr);
        let run = |code: i32, envelope: serde_json::Value| Finished {
            card: None,
            label: String::new(),
            code,
            envelope,
            seconds: 0.5,
        };
        let report = serde_json::json!({"data": {"report": {"dry_run": false,
            "operations": [{"tensors": [{}, {}]}]}, "backups": ["a.bak-1"]}});
        assert_eq!(
            outcome_line(&run(0, report)),
            "appliqué : 2 tenseur(s), 1 sauvegarde(s)"
        );
        let restored = serde_json::json!({"data": {"restored": ["a", "b"]}});
        assert_eq!(outcome_line(&run(0, restored)), "2 fichier(s) restauré(s)");
        let failed = serde_json::json!({"error": {"message": "patch invalide :\n  - x"}});
        assert_eq!(
            outcome_line(&run(2, failed)),
            "échec (code 2) : patch invalide :"
        );
    }

    #[test]
    fn the_centroid_is_the_mean_of_the_marked_nodes_only() {
        let report = crate::ui::views::tests::report();
        let marked: Vec<bool> = (0..report.nodes.len()).map(|i| i < 2).collect();
        let [a, b] = [&report.nodes[0], &report.nodes[1]];
        let c = centroid(&report, &marked, LayoutMode::Column).unwrap();
        for k in 0..3 {
            assert!((c[k] - (a.pos_column[k] + b.pos_column[k]) / 2.0).abs() < 1e-5);
        }
        let none = vec![false; report.nodes.len()];
        assert_eq!(centroid(&report, &none, LayoutMode::Column), None);
    }

    #[test]
    fn tabs_round_trip_through_their_screen_names() {
        for tab in Tab::ALL {
            assert_eq!(Tab::from_name(tab.name()), Some(tab));
        }
        assert_eq!(Tab::from_name("home"), None);
        assert_eq!(Tab::from_name("diagnostics"), Some(Tab::System));
    }
}
