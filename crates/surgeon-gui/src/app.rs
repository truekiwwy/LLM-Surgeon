//! The window state and what the views ask it to do. Drawing lives in `ui`: the shell (title
//! bar, activity bar, palette, settings dialog) and the two views. Here: the graph on screen
//! and its stream, the selection, the camera, the studio (projects, jobs, ledger), the agent
//! hub and the desktop services.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;
use std::time::{Duration, SystemTime};

use eframe::egui::{self, Key};
use weight_surgeon::backend::{self, config::BackendConfig, detect::BackendReport, hipfire};
use weight_surgeon::bench::score::Backend;
use weight_surgeon::graph::output::GraphReport;
use weight_surgeon::graph::stream::GraphPatch;
use weight_surgeon::graph::GraphRole;

use crate::agent::evidence;
use crate::agent::mission::Mission;
use crate::agent::paths::ProjectPaths;
use crate::agent::session::Dirs;
use crate::agent::view::Hub;
use crate::backend_choice;
use crate::backend_dialog::BackendDialog;
use crate::camera::Orbit;
use crate::cli::Item;
use crate::desktop::{Desktop, Event};
use crate::hipfire_daemon::HipfireDaemon;
use crate::logging;
use crate::platform::Picker;
use crate::source::{self, Input, Timeline};
use crate::studio::bus::RunView;
use crate::studio::ledger;
use crate::studio::live::{Layer, Replay};
use crate::studio::results::{self, Run};
use crate::studio::view::{Studio, Target};
use crate::task::Task;
use crate::ui::layout::Columns;
use crate::ui::palette::Palette;
use crate::ui::{fonts, i18n, shell, theme};
use crate::viewport;

/// The two views of the activity bar: the agent (where the work happens) and the 3D graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Agent,
    Graph,
}

impl Tab {
    pub const ALL: [Tab; 2] = [Tab::Agent, Tab::Graph];

    /// The `--screen` name of the view.
    pub fn name(self) -> &'static str {
        match self {
            Tab::Agent => "agent",
            Tab::Graph => "graph",
        }
    }

    /// `agents`, the name before G7, still reaches the agent.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "agents" => Some(Tab::Agent),
            _ => Self::ALL.into_iter().find(|t| t.name() == name),
        }
    }
}

pub use crate::settings::Mode;

/// The sections of the settings dialog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    General,
    Providers,
    /// External harnesses: the skill and the MCP configuration.
    Harnesses,
    Engine,
    Updates,
    Journal,
}

impl Section {
    pub const ALL: [Section; 6] = [
        Section::General,
        Section::Providers,
        Section::Harnesses,
        Section::Engine,
        Section::Updates,
        Section::Journal,
    ];

    /// The `--screen settings:<name>` part; `machine` (before G7) is the engine.
    pub fn from_name(name: &str) -> Option<Self> {
        let names = [
            "general",
            "providers",
            "harnesses",
            "engine",
            "updates",
            "journal",
        ];
        match name {
            "machine" => Some(Section::Engine),
            _ => names.iter().position(|n| *n == name).map(|k| Self::ALL[k]),
        }
    }
}

/// What the right column of the agent shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RightPane {
    Evidence,
    Results,
}

/// A file read again only when its modification time changed: path, time, entries.
type RunCache = Option<(PathBuf, Option<SystemTime>, Vec<ledger::Entry>)>;
type RunStamp = [Option<(PathBuf, Option<SystemTime>)>; 2];

/// `cache` for `path`, read again with `read` when the file changed.
fn cached(
    cache: &mut RunCache,
    path: Option<PathBuf>,
    read: impl FnOnce(&Path) -> Vec<ledger::Entry>,
) -> &[ledger::Entry] {
    let Some(path) = path else {
        *cache = None;
        return &[];
    };
    let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
    let stale = cache
        .as_ref()
        .is_none_or(|(p, m, _)| *p != path || *m != modified);
    if stale {
        let entries = read(&path);
        *cache = Some((path, modified, entries));
    }
    cache.as_ref().map_or(&[], |(_, _, e)| e)
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
    probe: Option<Receiver<(BackendReport, bool)>>,
    pub hardware: Option<BackendReport>,
    /// The hipfire daemon answered at the last probe.
    pub hipfire_up: bool,
    /// The hipfire daemon this window started, if any.
    pub hipfire: HipfireDaemon,
    pub tab: Tab,
    /// The kind of session the new-session form offers first, kept between launches.
    pub mode: Mode,
    pub studio: Studio,
    pub agents: Hub,
    /// A session that starts once its working copy (this run) is made.
    pub pending_start: Option<(Mission, String)>,
    pub desktop: Desktop,
    /// A recorded run on the 3D view, with its time cursor.
    pub replay: Option<Replay>,
    /// The run on the 3D view mapped onto the current graph, refilled every frame.
    pub layer: Layer,
    pub buffers: viewport::Buffers,
    pub columns: Columns,
    pub palette: Palette,
    /// `false` (`--no-motion`): no transition at all.
    pub motion: bool,
    /// The settings dialog, open on this section.
    pub settings: Option<Section>,
    /// What the last skill install or MCP configuration did (`true`: an error).
    pub harness_note: Option<(String, bool)>,
    /// The system picker of the workspace folder (wizard, settings).
    pub workspace_picker: Picker,
    pub right: RightPane,
    /// The last node an agent showed (`show_in_3d`), and whether the camera still has to
    /// reach it (the graph holding it may not be loaded yet).
    pub focus: Option<String>,
    center_pending: bool,
    /// The project whose graph the 3D view follows, the graph run shown (`None`: the
    /// architecture), and whether the graph on screen was loaded for the project rather than
    /// opened by hand.
    graph_for: Option<PathBuf>,
    graph_run: Option<String>,
    auto_graph: bool,
    skeleton: Option<Task<Result<GraphReport, String>>>,
    ledger_cache: RunCache,
    mcp_cache: RunCache,
    results_cache: Option<(RunStamp, Vec<Run>)>,
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
            None => Tab::Agent,
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
            hipfire_up: false,
            hipfire: HipfireDaemon::default(),
            tab,
            mode,
            studio: Studio::new(workspace),
            agents: Hub::system(ctx),
            pending_start: None,
            desktop,
            replay: None,
            layer: Layer::default(),
            buffers: viewport::Buffers::default(),
            columns,
            palette: Palette::default(),
            motion: true,
            settings: None,
            harness_note: None,
            workspace_picker: Picker::default(),
            right: RightPane::Evidence,
            focus: None,
            center_pending: false,
            graph_for: None,
            graph_run: None,
            auto_graph: false,
            skeleton: None,
            ledger_cache: None,
            mcp_cache: None,
            results_cache: None,
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

    /// The kind of session the form offers, kept for the next launch. What the agent may do
    /// (its contract) is not touched.
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode = mode;
        self.desktop.settings.mode = mode;
        self.desktop.save_settings();
    }

    /// Makes `dir` the projects folder (created, checked writable) and opens it, unless the
    /// open project cannot be left ([`Studio::can_switch`]): nothing changes then.
    pub fn use_workspace(&mut self, dir: PathBuf) -> Result<(), String> {
        self.studio.can_switch()?;
        self.desktop.use_workspace(&dir)?;
        self.studio = Studio::new(dir);
        Ok(())
    }

    /// Opens the system picker of the workspace folder; the folder picked is used at once.
    pub fn pick_workspace(&mut self, ctx: &egui::Context, title: &str) {
        let wake = ctx.clone();
        self.workspace_picker
            .open(true, title, move || wake.request_repaint());
    }

    /// The folder the picker gave: used, and the wizard shows it with what went wrong.
    fn picked_workspace(&mut self) {
        let Some(dir) = self.workspace_picker.poll() else {
            return;
        };
        let shown = dir.display().to_string();
        let result = self.use_workspace(dir);
        match self.desktop.wizard.as_mut() {
            Some(w) => {
                w.workspace = shown;
                w.workspace_error = result.err();
            }
            None => {
                if let Err(e) = result {
                    self.desktop.notify(e);
                }
            }
        }
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
        self.auto_graph = false;
    }

    /// Shows a graph loaded for the project: replaced by the next one, unlike one opened by
    /// hand.
    fn show_project_graph(&mut self, ctx: &egui::Context, input: Input) {
        self.set_input(ctx, input);
        self.auto_graph = true;
        // set_input cleared the selection: an agent's focus is applied again.
        self.center_pending |= self.focus.is_some();
    }

    /// The graph of the open project on the 3D view: the newest `surgeon_graph` run that
    /// wrote one, else the architecture read from the model's headers (no SVD), so that the
    /// network is there as soon as the project is open. A graph opened by hand stays until
    /// a new graph run of the project ends.
    fn follow_project_graph(&mut self, ctx: &egui::Context) {
        let project = self.agents.project.clone();
        let newest = {
            let (ledger, mcp) = self.run_lists();
            results::newest_graph(ledger.iter().chain(mcp))
        };
        if project != self.graph_for {
            self.graph_for = project.clone();
            self.skeleton = None;
            self.focus = None;
            self.center_pending = false;
            if self.auto_graph {
                self.timeline = Timeline::default();
                self.feed = None;
                self.auto_graph = false;
            }
            let free = self.timeline.is_empty() && self.feed.is_none();
            if free {
                self.stream_error = None;
            }
            self.graph_run = newest.as_ref().filter(|_| !free).map(|(id, _)| id.clone());
            let model = project
                .as_deref()
                .map(ProjectPaths::of)
                .and_then(|p| p.copy.or(p.origin));
            if let (true, None, Some(model)) = (free, &newest, model) {
                let wake = ctx.clone();
                self.skeleton = Some(Task::spawn(
                    move || wake.request_repaint(),
                    move |_| weight_surgeon::graph::skeleton(&model).map_err(|e| e.to_string()),
                ));
            }
        }
        if let Some((id, path)) = newest.filter(|(id, _)| self.graph_run.as_ref() != Some(id)) {
            self.graph_run = Some(id);
            self.skeleton = None;
            match source::open(&path) {
                Ok(input) => self.show_project_graph(ctx, input),
                Err(e) => self.stream_error = Some(e),
            }
        }
        if let Some(done) = self.skeleton.as_ref().and_then(Task::poll) {
            self.skeleton = None;
            let free = self.auto_graph || (self.timeline.is_empty() && self.feed.is_none());
            match done {
                Ok(report) if free => self.show_project_graph(ctx, Input::Report(Box::new(report))),
                Ok(_) => {}
                Err(e) => self.stream_error = Some(e),
            }
        }
    }

    /// What `show_in_3d` asked: the node selected now, the camera on it as soon as the graph
    /// holding it is on screen.
    pub fn show_focus(&mut self, target: String) {
        self.selected = Some(target.clone());
        self.focus = Some(target);
        self.center_pending = true;
    }

    /// Selects the focused node, opens its layer in the tree and centers the camera on it.
    fn center_on_focus(&mut self) {
        if !self.center_pending {
            return;
        }
        let Some(step) = self.timeline.current() else {
            return;
        };
        let Some(&i) = self.focus.as_ref().and_then(|f| step.index.get(f)) else {
            return;
        };
        let node = &step.report.nodes[i];
        self.camera.target = match self.layout {
            LayoutMode::Column => step.scene.column[i],
            LayoutMode::Constellation => step.scene.helix[i],
        };
        self.open_layers.insert(node.layer);
        self.selected = Some(node.id.clone());
        self.center_pending = false;
    }

    /// The backends (mistral.rs found, the hipfire daemon answering) can take seconds: probe
    /// off the UI thread. An unset daemon address is tried at hipfire's default.
    fn spawn_probe(&mut self, ctx: &egui::Context) {
        let (tx, probe) = std::sync::mpsc::channel();
        let mut user = self.dialog.saved().clone();
        user.hipfire_url
            .get_or_insert_with(backend_choice::hipfire_url);
        let wake = ctx.clone();
        std::thread::spawn(move || {
            let report = backend::detect::detect_hardware_and_backends(&user);
            let up = report.hipfire.up;
            let _ = tx.send((report, up));
            wake.request_repaint();
        });
        self.probe = Some(probe);
    }

    /// Brings the daemon of this window in line with the last probe: started when the machine
    /// or the saved backend calls for hipfire and nothing answers, stopped once native is
    /// pinned. Without a binary nothing happens and the fallback engine keeps running.
    fn sync_hipfire(&mut self, ctx: &egui::Context) {
        let Some(report) = &self.hardware else {
            return;
        };
        let saved = self.dialog.saved();
        let auto = self.desktop.settings.backend_auto;
        if !backend_choice::wants_hipfire(report.target, saved.backend, auto) {
            self.hipfire.stop();
            return;
        }
        let Some(bin) = report
            .hipfire
            .binary
            .as_ref()
            .filter(|_| !report.hipfire.up)
        else {
            return;
        };
        let url = saved
            .hipfire_url
            .clone()
            .unwrap_or_else(backend_choice::hipfire_url);
        match hipfire::loopback_port(&url) {
            Ok(port) => {
                let wake = ctx.clone();
                self.hipfire
                    .start(bin, port, move || wake.request_repaint());
            }
            Err(e) => logging::warn(&e),
        }
    }

    /// Starts the agent on `mission`. A project with a source model and no working copy gets
    /// one first, and the session starts when it is made: the agent never has to ask for it.
    pub fn start_session(&mut self, mission: Mission) {
        let paths = self.agents.project.as_deref().map(ProjectPaths::of);
        if !paths.is_some_and(|p| p.copy.is_none() && p.origin.is_some()) {
            self.agents.begin(mission);
            return;
        }
        match self.studio.submit_copy() {
            Some(run) => self.pending_start = Some((mission, run)),
            None => self.agents.message = self.studio.message.take(),
        }
    }

    fn start_after_copy(&mut self, ended: &[crate::studio::jobs::Ended]) {
        let Some((mission, run)) = self.pending_start.take() else {
            return;
        };
        if self.studio.manager.is_none() {
            return;
        }
        match ended.iter().find(|e| e.id == run) {
            Some(e) if e.verdict == ledger::Verdict::Ok => self.agents.begin(mission),
            Some(e) => {
                let error = e.envelope["error"]["message"]
                    .as_str()
                    .map_or_else(|| format!("{:?}", e.verdict), str::to_owned);
                self.agents.message =
                    Some((i18n::tf("form.copy.failed", &[("error", &error)]), true));
            }
            None => self.pending_start = Some((mission, run)),
        }
    }

    /// Looks at the backends again (the hipfire daemon started, mistral.rs installed).
    pub fn reprobe(&mut self, ctx: &egui::Context) {
        if self.probe.is_none() {
            self.spawn_probe(ctx);
        }
    }

    /// The first launch, and every probe while the choice is automatic: the backend the
    /// hardware calls for among the Rust engines (`backend_choice`), written to the user
    /// `config.toml`. A backend saved by hand in the dialog ends the automatic choice.
    fn auto_backend(&mut self) {
        let (Some(report), Some(system)) = (&self.hardware, &self.desktop.system) else {
            return;
        };
        let saved = self.dialog.saved();
        // A disabled user file (WEIGHT_SURGEON_NO_USER_CONFIG) cannot hold the choice.
        let automatic =
            backend_choice::automatic(saved.backend, self.desktop.settings.backend_auto);
        if !automatic || !self.dialog.writable() {
            return;
        }
        let vendor = backend_choice::main_vendor(system.gpus.iter().map(|g| g.name.as_str()));
        let mistral = report.mistralrs.as_ref().map(|m| m.binary.clone());
        let target = backend_choice::target(vendor);
        let backend = backend_choice::effective(target, self.hipfire_up, mistral.is_some());
        let cfg = backend_choice::config(saved, backend, vendor, mistral);
        if cfg != *saved {
            if let Err(e) = self.dialog.save_config(cfg) {
                self.desktop.notify(e);
                return;
            }
        }
        if !self.desktop.settings.backend_auto {
            self.desktop.settings.backend_auto = true;
            self.desktop.save_settings();
        }
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
            Item::Folder(dir) => self.use_workspace(dir)?,
            Item::Unknown(path) => {
                return Err(i18n::tf("open.unknown", &[("path", &path.display())]))
            }
        }
        Ok(())
    }

    /// The project's ledger (studio jobs) and the runs CLI agents made through MCP
    /// (`chat/mcp_runs.jsonl`), each read again only when its file changed.
    fn run_lists(&mut self) -> (&[ledger::Entry], &[ledger::Entry]) {
        let ledger = self
            .studio
            .manager
            .as_ref()
            .map(|m| m.project.ledger_path());
        let mcp = self
            .agents
            .project
            .as_deref()
            .map(|p| Dirs::of(p).mcp_runs());
        (
            cached(&mut self.ledger_cache, ledger, |p| {
                ledger::read(p).unwrap_or_default()
            }),
            cached(&mut self.mcp_cache, mcp, evidence::mcp_runs),
        )
    }

    /// The project's ledger.
    pub fn ledger(&mut self) -> &[ledger::Entry] {
        self.run_lists().0
    }

    /// The ledger and the MCP runs, as one list a finding can cite.
    pub fn runs(&mut self) -> Vec<ledger::Entry> {
        let (ledger, mcp) = self.run_lists();
        ledger.iter().chain(mcp).cloned().collect()
    }

    /// The measuring runs of the project for the Results panel, parsed again only when one
    /// of the files changed.
    pub fn results(&mut self) -> &[Run] {
        self.run_lists();
        let stamp = |c: &RunCache| c.as_ref().map(|(p, m, _)| (p.clone(), *m));
        let stamp = [stamp(&self.ledger_cache), stamp(&self.mcp_cache)];
        if self.results_cache.as_ref().is_none_or(|(s, _)| *s != stamp) {
            let entries = [&self.ledger_cache, &self.mcp_cache]
                .into_iter()
                .flatten()
                .flat_map(|(_, _, e)| e);
            self.results_cache = Some((stamp, results::runs(entries)));
        }
        self.results_cache.as_ref().map_or(&[], |(_, r)| r)
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
        if let Some((report, up)) = self.probe.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.dialog.prefill(&report);
            self.hardware = Some(report);
            self.hipfire_up = up;
            self.probe = None;
            self.sync_hipfire(ctx);
        }
        // Kept for later while a probe runs: it may have looked before the port opened.
        if self.probe.is_none() && self.hipfire.poll() {
            self.spawn_probe(ctx);
        }
        self.auto_backend();
        if self.dialog.take_saved() {
            // Saved by hand: the user's choice from now on.
            self.desktop.settings.backend_auto = false;
            self.desktop.save_settings();
            self.sync_hipfire(ctx);
        }
        self.picked_workspace();
        let ended = self.studio.tick(ctx);
        self.agents.tick(&mut self.studio, &ended);
        self.start_after_copy(&ended);
        self.follow_project_graph(ctx);
        if let Some(target) = self.agents.take_focus() {
            self.show_focus(target);
        }
        self.center_on_focus();
        if self.agents.busy() {
            ctx.request_repaint_after(crate::studio::bus::FRAME);
        }
        if let Some(replay) = self.replay.as_mut().filter(|r| r.playing) {
            replay.advance(Duration::from_secs_f32(ctx.input(|i| i.stable_dt)));
            ctx.request_repaint();
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

    pub fn backend_label(&self) -> String {
        let saved: &BackendConfig = self.dialog.saved();
        let kind = saved.backend.unwrap_or(Backend::Native);
        backend::describe(
            kind.label(),
            None,
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

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        self.hipfire.stop();
    }
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
    use std::sync::Arc;

    use super::*;

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
    fn tabs_round_trip_through_their_screen_names() {
        for tab in Tab::ALL {
            assert_eq!(Tab::from_name(tab.name()), Some(tab));
        }
        assert_eq!(Tab::from_name("home"), None);
        assert_eq!(Tab::from_name("agents"), Some(Tab::Agent));
        for section in Section::ALL {
            let name = format!("{section:?}").to_lowercase();
            assert_eq!(Section::from_name(&name), Some(section));
        }
        assert_eq!(Section::from_name("machine"), Some(Section::Engine));
    }

    /// A window on a new project `p` of `source`, opened and polled once; never the user's
    /// settings nor connections.
    fn app_on(dir: &Path, source: &Path) -> (egui::Context, SurgeonApp, PathBuf) {
        let workspace = dir.join("projects");
        std::fs::create_dir_all(&workspace).unwrap();
        let project =
            crate::studio::project::Project::create(&workspace, "p", Some(source)).unwrap();
        let env = crate::actions::Env {
            components: dir.join("components"),
            config: None,
            settings: dir.join("settings.json"),
        };
        let desktop = Desktop::detached(None, Default::default(), env);
        let ctx = egui::Context::default();
        let mut app = SurgeonApp::with_shell(&ctx, None, desktop, workspace);
        app.dialog.read_only();
        app.agents = Hub::new(
            dir.join("agents.json"),
            Box::new(crate::agent::vault::SessionStore::default()),
            std::sync::Arc::new(|| {}),
        );
        app.studio.open_project(&ctx, &project.dir);
        app.poll(&ctx);
        assert_eq!(app.agents.project.as_deref(), Some(project.dir.as_path()));
        (ctx, app, project.dir)
    }

    /// Polls until `done` or 30 s.
    fn poll_until(app: &mut SurgeonApp, ctx: &egui::Context, done: impl Fn(&SurgeonApp) -> bool) {
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while !done(app) && std::time::Instant::now() < deadline {
            app.poll(ctx);
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(done(app), "timed out");
    }

    /// A two-layer llama-shaped model of zeros: every attention and MLP matrix, [4, 4].
    fn write_model(dir: &Path) {
        let names: Vec<String> = (0..2)
            .flat_map(|l| {
                ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]
                    .into_iter()
                    .chain(["self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj"])
                    .chain(["mlp.down_proj"])
                    .map(move |m| format!("model.layers.{l}.{m}.weight"))
            })
            .collect();
        let size = 4 * 4 * 4;
        let header: serde_json::Map<String, serde_json::Value> = names
            .iter()
            .enumerate()
            .map(|(i, n)| {
                let at = i * size;
                let info = serde_json::json!({"dtype": "F32", "shape": [4, 4],
                                              "data_offsets": [at, at + size]});
                (n.clone(), info)
            })
            .collect();
        let mut header = serde_json::to_vec(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(vec![0u8; names.len() * size]);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    #[test]
    fn an_open_project_shows_its_architecture_and_follows_show_in_3d_and_graph_runs() {
        use crate::agent::mock::{call, say, Mock};
        use crate::agent::session::{Conversation, Session, Settings};

        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("model");
        write_model(&source);
        let (ctx, mut app, project) = app_on(dir.path(), &source);

        // The architecture comes at once, with no graph run: every matrix, unmeasured.
        poll_until(&mut app, &ctx, |a| !a.timeline.is_empty());
        let report = app.timeline.current().unwrap().report.clone();
        assert_eq!(report.parameters.rank, 0);
        assert_eq!(report.nodes.len(), 14);
        assert!(!report.edges.is_empty());
        assert!(app.selected.is_none());

        // The agent shows a tensor by name: selected by its graph id, camera on it, its
        // layer open in the tree. The view is not switched for the user.
        let tensor = "model.layers.1.mlp.down_proj.weight";
        let mock = Arc::new(Mock::new(vec![
            call(vec![(
                "s",
                "show_in_3d",
                serde_json::json!({"target": tensor}),
            )]),
            say("Voilà la projection de sortie du MLP de la couche 1."),
        ]));
        let dirs = Dirs::of(&project);
        let conv = Conversation {
            id: Session::new_id(&dirs),
            ..Conversation::default()
        };
        let mut session = Session::new(conv, mock, Settings::default(), Vec::new(), dirs, {
            Arc::new(|| {})
        });
        session.consented = true;
        session.send("montre-moi la couche 1").unwrap();
        app.agents.session = Some(session);
        poll_until(&mut app, &ctx, |a| a.focus.is_some());
        app.poll(&ctx);
        let step = app.timeline.current().unwrap();
        let i = step.index["L1.mlp_out"];
        let (node, centre) = (&report.nodes[i], step.scene.column[i]);
        assert_eq!(node.tensor, tensor);
        assert_eq!(app.selected.as_deref(), Some("L1.mlp_out"));
        assert_eq!(app.camera.target, centre);
        assert!(app.open_layers.contains(&Some(1)));
        assert_eq!(app.tab, Tab::Agent);

        // A surgeon_graph run of a CLI agent ends: its graph replaces the architecture, and
        // the agent's node stays selected and centered.
        let mut measured = report.clone();
        measured.parameters.rank = 8;
        let file = project.join("out").join("graph.json");
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, serde_json::to_vec(&measured).unwrap()).unwrap();
        let line = serde_json::json!({"run_id": "mcp-1", "tool": "surgeon_graph", "ok": true,
            "exit_code": 0, "data": {"outputs": [file]}, "started_unix": 1, "finished_unix": 2});
        std::fs::write(Dirs::of(&project).mcp_runs(), format!("{line}\n")).unwrap();
        app.camera = Orbit::default();
        app.poll(&ctx);
        app.poll(&ctx);
        assert_eq!(app.timeline.current().unwrap().report.parameters.rank, 8);
        assert_eq!(app.selected.as_deref(), Some("L1.mlp_out"));
        assert_eq!(app.camera.target, centre);
    }

    #[test]
    fn a_graph_opened_by_hand_is_not_replaced_by_the_project_architecture() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("model");
        write_model(&source);
        let (ctx, mut app, _) = app_on(dir.path(), &source);
        assert!(app.skeleton.is_some(), "the architecture is being read");
        // A graph opened before the architecture arrives keeps the view.
        let by_hand = crate::ui::views::tests::report();
        app.set_input(&ctx, Input::Report(Box::new(by_hand.clone())));
        poll_until(&mut app, &ctx, |a| a.skeleton.is_none());
        assert_eq!(app.timeline.current().unwrap().report, by_hand);
    }

    #[test]
    fn results_are_read_from_the_project_ledger_and_the_mcp_runs() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("model");
        write_model(&source);
        let (_ctx, mut app, project) = app_on(dir.path(), &source);
        assert!(app.results().is_empty());

        let ledger_path = app.studio.manager.as_ref().unwrap().project.ledger_path();
        let item = |id: &str, ok: bool| {
            serde_json::json!({"id": id, "gold": "6", "extracted": if ok { "6" } else { "5" },
                               "correct": ok, "output_excerpt": ""})
        };
        let report = serde_json::json!({
            "dataset": "builtin:math", "n": 1, "correct": 1, "accuracy": 1.0,
            "ci95_low": 0.2, "ci95_high": 1.0, "items": [item("math-01", true)],
            "baseline": {"n": 1, "accuracy": 0.0, "ci95_low": 0.0, "ci95_high": 0.8,
                         "items": [item("math-01", false)]},
            "paired": {"gained": ["math-01"], "lost": []},
        });
        let studio_run = ledger::Entry {
            run_id: "r1".into(),
            tool: "surgeon_bench".into(),
            label: "eval".into(),
            params: serde_json::json!({"mode": "eval"}),
            resumed_from: None,
            started_unix: 100,
            finished_unix: 160,
            hashes: Default::default(),
            versions: Default::default(),
            verdict: ledger::Verdict::Ok,
            exit_code: 0,
            result: report,
            error: None,
        };
        ledger::append(&ledger_path, &studio_run).unwrap();
        let line = serde_json::json!({"run_id": "mcp-7", "tool": "surgeon_extract", "ok": false,
            "exit_code": 1, "data": null, "started_unix": 200, "finished_unix": 205});
        std::fs::write(Dirs::of(&project).mcp_runs(), format!("{line}\n")).unwrap();

        let runs = app.results().to_vec();
        let ids: Vec<&str> = runs.iter().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, ["mcp-7", "r1"], "newest first, both sources");
        assert_eq!(runs[0].verdict, ledger::Verdict::Failed);
        let bench = runs[1].bench.as_ref().unwrap();
        assert_eq!(runs[1].seconds, 60);
        assert_eq!(bench.dataset.as_deref(), Some("builtin:math"));
        assert_eq!((bench.after.rate, bench.before.unwrap().rate), (1.0, 0.0));
        assert_eq!((bench.gained, bench.lost), (1, 0));
        let first = &bench.items[0];
        assert!(first.question.as_deref().unwrap().starts_with("A baker"));
        assert_eq!(first.before.as_ref().unwrap().text, "5");
    }

    #[test]
    fn starting_without_a_working_copy_makes_one_then_starts() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("model");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("model.safetensors"), b"weights").unwrap();
        let (ctx, mut app, project_dir) = app_on(dir.path(), &source);
        app.agents.input = "go".into();

        app.start_session(Mission::Advance);
        assert!(app.pending_start.is_some(), "the copy comes first");
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        while app.pending_start.is_some() && std::time::Instant::now() < deadline {
            app.poll(&ctx);
            std::thread::sleep(Duration::from_millis(20));
        }
        assert!(app.pending_start.is_none(), "the copy ended");
        let paths = ProjectPaths::of(&project_dir);
        assert!(paths
            .copy
            .as_ref()
            .is_some_and(|c| c.join("model.safetensors").is_file()));
        // The session was started: with no connection it says so and keeps the text.
        assert_eq!(app.agents.mission, Some(Mission::Advance));
        assert_eq!(app.agents.input, "go");
        assert!(app.agents.message.as_ref().is_some_and(|(_, error)| *error));

        // A copy exists now: the next start does not make another.
        app.start_session(Mission::Advance);
        assert!(app.pending_start.is_none());
    }

    /// A job of the open project still writes there: neither another project nor another
    /// workspace replaces the studio until it ends; then every switch goes through.
    #[test]
    fn no_project_or_workspace_switch_while_a_job_runs() {
        use crate::studio::jobs::{Budgets, JobSpec};

        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("model");
        write_model(&source);
        let (ctx, mut app, project) = app_on(dir.path(), &source);
        let workspace = app.studio.workspace.clone();
        let second = crate::studio::project::Project::create(&workspace, "q", None)
            .unwrap()
            .dir;
        // Outside the app's data folder (`dir`), which a workspace may not be in.
        let elsewhere = tempfile::tempdir().unwrap();
        let other = elsewhere.path().join("other");
        let job = JobSpec {
            tool: "surgeon_check".into(),
            label: "check".into(),
            args: serde_json::json!({}),
            priority: 0,
            budgets: Budgets::default(),
            resumed_from: None,
            read_only: false,
        };
        // Queued and not started (no frame ran): the manager is busy all the same.
        let id = app.studio.manager.as_mut().unwrap().submit(job).unwrap();
        let open = |app: &SurgeonApp| app.studio.manager.as_ref().map(|m| m.project.dir.clone());

        let refused = app.use_workspace(other.clone()).unwrap_err();
        assert!(refused.contains("job"), "{refused}");
        let refused = app
            .open_item(&ctx, Item::Folder(other.clone()))
            .unwrap_err();
        assert!(refused.contains("job"), "{refused}");
        app.studio.open_project(&ctx, &second);
        let (message, error) = app.studio.message.clone().unwrap();
        assert!(error && message.contains("job"), "{message}");
        assert_eq!(open(&app), Some(project.clone()), "the same project");
        assert_eq!(app.studio.workspace, workspace, "the same workspace");
        assert_ne!(app.desktop.settings.workspace.as_ref(), Some(&other));

        app.studio.manager.as_mut().unwrap().cancel(&id).unwrap();
        app.studio.open_project(&ctx, &second);
        assert_eq!(open(&app), Some(second));
        app.open_item(&ctx, Item::Folder(other.clone())).unwrap();
        assert_eq!(app.studio.workspace, other);
        let third = elsewhere.path().join("third");
        app.use_workspace(third.clone()).unwrap();
        assert_eq!(app.studio.workspace, third);
    }
}
