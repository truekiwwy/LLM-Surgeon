//! The desktop services around the studio: paths handed over by a second launch or dropped on
//! the window, the first-run wizard's state and the installs it starts ([`setup`]), the
//! machine probe and the engine's machine check (`surgeon doctor`), the manual update check
//! and the diagnostics export. It owns its own state
//! and hands the app a list of [`Event`]s; `ui::wizard` and the System view draw it.

pub mod setup;

use std::path::PathBuf;

use eframe::egui;
use weight_surgeon::backend::config::BackendConfig;
use weight_surgeon::doctor::{self, DoctorReport, DoctorRequest};
use weight_surgeon::hardware::SystemProbe;

use crate::actions::{self, Env};
use crate::cli::{self, Item};
use crate::diag::{self, Fix, Row, SystemReport};
use crate::instance::Inbox;
use crate::settings::{Lang, Settings};
use crate::task::Task;
use crate::ui::components;
use crate::ui::i18n::t;
use crate::ui::wizard::{Step, Wizard};
use crate::update::{self, Update};
use setup::{Pick, Setup};

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
    /// The engine's machine check (`surgeon doctor`), run when Settings > Engine first shows.
    pub doctor: Option<DoctorReport>,
    doctor_task: Option<Task<DoctorReport>>,
    /// The first-run wizard, open over the whole body.
    pub wizard: Option<Wizard>,
    /// The optional components being planned or installed.
    pub setup: Setup,
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
                desktop.open_wizard();
            }
        }
        desktop
    }

    /// No disk read, no probe, no wizard: the state the caller gives (tests, captures).
    pub fn detached(inbox: Option<Inbox>, settings: Settings, env: Env) -> Self {
        Self {
            wizard: None,
            setup: Setup::default(),
            system: None,
            system_task: None,
            doctor: None,
            doctor_task: None,
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

    pub fn env(&self) -> &Env {
        &self.env
    }

    /// Opens the wizard from its first step (first launch, palette, System).
    pub fn open_wizard(&mut self) {
        self.wizard = Some(Wizard::new(&self.settings));
    }

    /// The panel that repairs `fix`, alone: the components with the piece picked, or the
    /// workspace. `force` reinstalls.
    pub fn open_fix(&mut self, fix: Fix, force: bool) {
        let (step, pick) = match fix {
            Fix::MistralRs => (
                Step::Components,
                Pick {
                    engine: true,
                    sample: false,
                },
            ),
            Fix::Hipfire => (Step::Components, Pick::default()),
            Fix::SampleModel => (
                Step::Components,
                Pick {
                    engine: false,
                    sample: true,
                },
            ),
            Fix::Workspace => (Step::Workspace, Pick::default()),
            Fix::None => return,
        };
        self.setup.force = force;
        self.wizard = Some(Wizard::at(&self.settings, step, pick));
    }

    /// The wizard ends, finished or passed: it does not open by itself again (it stays in
    /// System and in the palette).
    pub fn close_wizard(&mut self) {
        let walked = self.wizard.take().is_some_and(|w| !w.single);
        if walked {
            self.settings.wizard_done = true;
            self.save_settings();
        }
        self.setup.force = false;
    }

    /// Capture helper: the wizard on `step` with a given machine report.
    pub fn capture_wizard(&mut self, system: SystemReport, step: Step) {
        self.setup = Setup::default();
        self.system = Some(system);
        let mut wizard = Wizard::new(&self.settings);
        wizard.step = step;
        self.wizard = Some(wizard);
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

    pub fn doctoring(&self) -> bool {
        self.doctor_task.is_some()
    }

    /// `surgeon doctor` off the UI thread, the same checks as the CLI and `surgeon_doctor`: RAM
    /// and the dtype `model` gets, cores, the reference cache, the hipfire daemon. `clean`
    /// removes the unreadable and obsolete cache entries first (`ref_cache::purge`).
    pub fn refresh_doctor(&mut self, ctx: &egui::Context, model: Option<PathBuf>, clean: bool) {
        if self.doctor_task.is_some() {
            return;
        }
        let wake = ctx.clone();
        self.doctor_task = Some(Task::spawn(
            move || wake.request_repaint(),
            move |_| {
                let req = DoctorRequest {
                    model,
                    clean_cache: clean,
                    ..DoctorRequest::default()
                };
                let user = weight_surgeon::backend::load_user_config().unwrap_or_default();
                doctor::run(&req, &SystemProbe::detect(), &user)
            },
        ));
    }

    /// The dependency rows: components, toolchain, workspace, machine, with their severity
    /// under `uses`.
    pub fn rows(&self, backend: &BackendConfig, uses: diag::Uses) -> Vec<Row> {
        let latest = None; // the releases API is only asked on an explicit click
        diag::dependencies(
            &self.env.components,
            backend,
            self.settings.workspace.as_deref(),
            latest,
            uses,
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
        if self.settings.local_only {
            self.update = UpdateState::Failed(t("update.local_only").into());
            return;
        }
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
        if let Some(report) = self.doctor_task.as_ref().and_then(Task::poll) {
            self.doctor = Some(report);
            self.doctor_task = None;
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

    /// Runs the services for one frame (hand-overs, drops, the installs, the drop hint) and
    /// returns what the app must do.
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
        if self.setup.poll(&mut self.settings, &self.env) {
            events.push(Event::BackendChanged);
        }
        if self.setup.installing() || self.setup.fetching_plan() {
            ctx.request_repaint_after(std::time::Duration::from_millis(150));
        }
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            components::drop_hint(&ctx, t("drop.hint"));
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The engine's own check, off the UI thread, read only (no purge asked).
    #[test]
    fn the_machine_check_runs_in_the_background_and_lands_in_the_desktop() {
        let dir = tempfile::tempdir().unwrap();
        let env = Env {
            components: dir.path().join("components"),
            config: None,
            settings: dir.path().join("settings.json"),
        };
        let mut desktop = Desktop::detached(None, Settings::default(), env);
        let ctx = egui::Context::default();
        desktop.refresh_doctor(&ctx, None, false);
        assert!(desktop.doctoring());
        desktop.refresh_doctor(&ctx, None, true);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        while desktop.doctor.is_none() && std::time::Instant::now() < deadline {
            desktop.poll();
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let report = desktop.doctor.as_ref().expect("a report");
        assert!(!desktop.doctoring());
        let names: Vec<&str> = report.checks.iter().map(|c| c.name).collect();
        assert_eq!(&names[..3], ["memory", "cpu", "cache"]);
        assert_eq!(
            report.cache.removed, 0,
            "the second call, a purge, was ignored"
        );
    }
}
