//! The desktop services around the studio: paths handed over by a second launch or dropped on
//! the window, the first-run wizard, the machine probe, the manual update check and the
//! diagnostics export. It owns its own state and hands the app a list of [`Event`]s; the
//! Diagnostics view (`ui::views::diagnostics`) draws it.

mod wizard;

use std::path::PathBuf;

use eframe::egui;
use weight_surgeon::backend::config::BackendConfig;

use crate::actions::{self, Env};
use crate::cli::{self, Item};
use crate::diag::{self, Fix, Row, SystemReport};
use crate::instance::Inbox;
use crate::settings::{Lang, Settings};
use crate::task::Task;
use crate::ui::components;
use crate::ui::i18n::t;
use crate::update::{self, Update};
use wizard::{Context as WizardContext, Outcome, Step, Wizard};

/// What the shell asks the app to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// Open these items (arguments of a second launch, drops, the palette).
    Open(Vec<Item>),
    /// The trusted toolchain or the user config changed: reload the backend dialog.
    BackendChanged,
}

pub enum UpdateState {
    Idle,
    Checking(Task<Result<Option<Update>, String>>),
    UpToDate,
    Available(Update),
    Downloading(Update, Task<Result<PathBuf, String>>),
    Downloaded(PathBuf),
    Failed(String),
}

pub struct Desktop {
    pub settings: Settings,
    env: Env,
    pub system: Option<SystemReport>,
    system_task: Option<Task<SystemReport>>,
    wizard: Option<Wizard>,
    pub update: UpdateState,
    /// The last error of an open or a hand-over, for the status line.
    pub message: Option<String>,
    /// Items to hand to the app on the next frame (the palette).
    pub pending: Vec<Item>,
    inbox: Option<Inbox>,
    /// The last diagnostics export, also copied to the clipboard.
    pub export: Option<String>,
}

impl Desktop {
    /// Loads the settings from disk; `capture`: offline rendering (no probe, no wizard
    /// popping up on its own).
    pub fn new(ctx: &egui::Context, inbox: Option<Inbox>, capture: bool) -> Self {
        let env = Env::current();
        let mut settings = Settings::load(&env.settings).unwrap_or_else(|e| {
            crate::logging::warn(&format!("settings ignorés : {e}"));
            Settings::default()
        });
        if settings.language.is_none() {
            settings.language = Some(Lang::system());
        }
        let mut desktop = Self::detached(inbox, settings, env);
        if !capture {
            desktop.refresh_system(ctx);
            if !desktop.settings.wizard_done {
                desktop.wizard = Some(Wizard::new(&desktop.settings, None));
            }
        }
        desktop
    }

    /// No disk read, no probe, no wizard: the state the caller gives (tests, captures).
    pub fn detached(inbox: Option<Inbox>, settings: Settings, env: Env) -> Self {
        Self {
            wizard: None,
            system: None,
            system_task: None,
            update: UpdateState::Idle,
            message: None,
            pending: Vec::new(),
            inbox,
            export: None,
            settings,
            env,
        }
    }

    pub fn notify(&mut self, message: String) {
        self.message = Some(message);
    }

    /// A dropped or opened folder becomes the workspace folder.
    pub fn use_workspace(&mut self, dir: &std::path::Path) -> Result<(), String> {
        actions::set_workspace(&self.env, dir)?;
        self.settings.workspace = Some(dir.to_path_buf());
        Ok(())
    }

    pub fn lang(&self) -> Lang {
        self.settings.lang()
    }

    /// Switches the language and keeps it for the next launch.
    pub fn set_lang(&mut self, lang: Lang) {
        self.settings.language = Some(lang);
        self.save_settings();
    }

    pub fn save_settings(&mut self) {
        if let Err(e) = self.settings.save(&self.env.settings) {
            self.message = Some(e);
        }
    }

    /// Opens the wizard from the start (palette, Diagnostics).
    pub fn open_wizard(&mut self) {
        self.wizard = Some(Wizard::new(&self.settings, self.system.as_ref()));
    }

    /// The wizard on the step that repairs `fix`; `force` reinstalls.
    pub fn open_fix(&mut self, fix: Fix, force: bool) {
        let step = match fix {
            Fix::LlamaCpp => Step::Engine,
            Fix::SampleModel => Step::Sample,
            Fix::Workspace => Step::Workspace,
            Fix::None => Step::Summary,
        };
        self.wizard = Some(Wizard::at(
            &self.settings,
            self.system.as_ref(),
            step,
            force,
        ));
    }

    pub fn wizard_open(&self) -> bool {
        self.wizard.is_some()
    }

    /// Capture helper: the wizard on a given step with a given machine report.
    pub fn capture_wizard(&mut self, system: SystemReport) {
        self.system = Some(system);
        self.wizard = Some(Wizard::at(
            &self.settings,
            self.system.as_ref(),
            Step::System,
            false,
        ));
    }

    pub fn capture_system(&mut self, system: SystemReport) {
        self.system = Some(system);
    }

    pub fn probing(&self) -> bool {
        self.system_task.is_some()
    }

    /// Probes the machine again, off the UI thread.
    pub fn refresh_system(&mut self, ctx: &egui::Context) {
        if self.system_task.is_some() {
            return;
        }
        let dir = crate::paths::data_dir();
        let wake = ctx.clone();
        self.system_task = Some(Task::spawn(
            move || wake.request_repaint(),
            move |_| diag::probe(&dir),
        ));
    }

    /// The dependency rows: components, toolchain, workspace, machine.
    pub fn rows(&self, backend: &BackendConfig) -> Vec<Row> {
        let latest = None; // the releases API is only asked on an explicit click
        diag::dependencies(
            &self.env.components,
            backend,
            self.settings.workspace.as_deref(),
            self.system.as_ref(),
            latest,
            self.lang(),
        )
    }

    /// The diagnostics report, without personal path or secret, copied to the clipboard.
    pub fn export_diagnostics(&mut self, ctx: &egui::Context, rows: &[Row]) {
        let text = diag::export(
            self.system.as_ref(),
            rows,
            &crate::paths::logs_dir(),
            self.lang(),
        );
        ctx.copy_text(text.clone());
        self.export = Some(text);
    }

    /// One HTTPS request to the releases API; nothing is downloaded.
    pub fn check_update(&mut self, ctx: &egui::Context) {
        if matches!(
            self.update,
            UpdateState::Checking(_) | UpdateState::Downloading(..)
        ) {
            return;
        }
        let current = env!("CARGO_PKG_VERSION").to_owned();
        let wake = ctx.clone();
        self.update = UpdateState::Checking(Task::spawn(
            move || wake.request_repaint(),
            move |_| update::check(&actions::production_fetcher(), update::LATEST_URL, &current),
        ));
    }

    /// Downloads the available update, after the user's click.
    pub fn download_update(&mut self, ctx: &egui::Context) {
        let UpdateState::Available(u) = &self.update else {
            return;
        };
        let u = u.clone();
        let dir = crate::paths::data_dir().join("updates");
        let wake = ctx.clone();
        let job = u.clone();
        let task = Task::spawn(
            move || wake.request_repaint(),
            move |h| {
                update::download(
                    &actions::production_fetcher(),
                    &job,
                    &dir,
                    &h.cancel,
                    |d, t| h.report("update", d, t),
                )
            },
        );
        self.update = UpdateState::Downloading(u, task);
    }

    fn poll(&mut self) {
        if let Some(system) = self.system_task.as_ref().and_then(Task::poll) {
            self.system = Some(system);
            self.system_task = None;
        }
        let next = match &self.update {
            UpdateState::Checking(task) => task.poll().map(|r| match r {
                Ok(Some(u)) => UpdateState::Available(u),
                Ok(None) => UpdateState::UpToDate,
                Err(e) => UpdateState::Failed(e),
            }),
            UpdateState::Downloading(_, task) => task.poll().map(|r| match r {
                Ok(path) => UpdateState::Downloaded(path),
                Err(e) => UpdateState::Failed(e),
            }),
            _ => None,
        };
        if let Some(next) = next {
            self.update = next;
        }
    }

    /// Runs the services for one frame (hand-overs, drops, the wizard window, the drop
    /// hint) and returns what the app must do.
    pub fn ui(&mut self, ui: &mut egui::Ui) -> Vec<Event> {
        let ctx = ui.ctx().clone();
        let mut events = Vec::new();
        self.poll();
        // Paths from a second launch and from a drop on the window.
        let mut items: Vec<Item> = self
            .inbox
            .iter()
            .flat_map(Inbox::drain)
            .map(|p| cli::classify(&p))
            .collect();
        if self.inbox.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(250));
            if !items.is_empty() {
                crate::logging::info(&format!(
                    "second launch handed over {} item(s)",
                    items.len()
                ));
                ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
            }
        }
        items.append(&mut self.pending);
        items.extend(ctx.input(|i| {
            i.raw
                .dropped_files
                .iter()
                .map(|f| cli::classify(egui::DroppedFile::path(f.as_ref())))
                .collect::<Vec<_>>()
        }));
        if !items.is_empty() {
            events.push(Event::Open(items));
        }
        events.extend(self.show_wizard(&ctx));
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            components::drop_hint(&ctx, t("drop.hint"));
        }
        events
    }

    fn show_wizard(&mut self, ctx: &egui::Context) -> Option<Event> {
        let wizard = self.wizard.as_mut()?;
        let mut cx = WizardContext {
            settings: &mut self.settings,
            env: &self.env,
            system: self.system.as_ref(),
        };
        match wizard.show(ctx, &mut cx) {
            Outcome::Open => None,
            Outcome::BackendChanged => Some(Event::BackendChanged),
            Outcome::Closed(finished) => {
                self.wizard = None;
                // Closing it early is a choice too: it stays one command away in the palette
                // and in Diagnostics, but it does not pop up again by itself.
                self.settings.wizard_done = true;
                let _ = self.settings.save(&self.env.settings);
                finished.then_some(Event::BackendChanged)
            }
        }
    }
}
