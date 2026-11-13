//! The desktop shell around the studio: the home screen, the Diagnostics page, the first-run
//! wizard and the manual update check. It owns its own state and hands the app a list of
//! [`Event`]s; it never reaches into the studio's.

mod wizard;

use std::path::PathBuf;

use eframe::egui::{self, Align, Layout, Panel, RichText};
use weight_surgeon::backend::config::BackendConfig;

use crate::actions::{self, Env};
use crate::cli::{self, Item};
use crate::diag::{self, Fix, Row, Status, SystemReport};
use crate::instance::Inbox;
use crate::settings::{Lang, Settings};
use crate::task::Task;
use crate::theme::{self, mono};
use crate::update::{self, Update};
use wizard::{Context as WizardContext, Outcome, Step, Wizard};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    Home,
    /// The studio itself (canvas, 3D graph, editor): drawn by the app.
    Workbench,
    Diagnostics,
}

/// What the shell asks the app to do.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// Open these items (arguments of a second launch, drops, the home field).
    Open(Vec<Item>),
    /// The trusted toolchain or the user config changed: reload the backend dialog.
    BackendChanged,
}

enum UpdateState {
    Idle,
    Checking(Task<Result<Option<Update>, String>>),
    UpToDate,
    Available(Update),
    Downloading(Update, Task<Result<PathBuf, String>>),
    Downloaded(PathBuf),
    Failed(String),
}

pub struct Desktop {
    pub screen: Screen,
    pub settings: Settings,
    env: Env,
    system: Option<SystemReport>,
    system_task: Option<Task<SystemReport>>,
    wizard: Option<Wizard>,
    update: UpdateState,
    open_field: String,
    message: Option<String>,
    /// Items to hand to the app on the next frame (the home field).
    pending: Vec<Item>,
    inbox: Option<Inbox>,
    export: Option<String>,
}

impl Desktop {
    /// Loads the settings from disk; `capture`: offline rendering (no probe, no wizard
    /// popping up on its own).
    pub fn new(ctx: &egui::Context, inbox: Option<Inbox>, start: Screen, capture: bool) -> Self {
        let env = Env::current();
        let mut settings = Settings::load(&env.settings).unwrap_or_else(|e| {
            crate::logging::warn(&format!("settings ignorés : {e}"));
            Settings::default()
        });
        if settings.language.is_none() {
            settings.language = Some(Lang::system());
        }
        let mut desktop = Self::detached(inbox, start, settings, env);
        if !capture {
            desktop.refresh_system(ctx);
            if !desktop.settings.wizard_done {
                desktop.wizard = Some(Wizard::new(&desktop.settings, None));
            }
        }
        desktop
    }

    /// No disk read, no probe, no wizard: the state the caller gives (tests, captures).
    pub fn detached(inbox: Option<Inbox>, start: Screen, settings: Settings, env: Env) -> Self {
        Self {
            screen: start,
            wizard: None,
            system: None,
            system_task: None,
            update: UpdateState::Idle,
            open_field: String::new(),
            message: None,
            pending: Vec::new(),
            inbox,
            export: None,
            settings,
            env,
        }
    }

    /// An error to show on the home screen.
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

    /// Opens the wizard from the start (Settings / Home button).
    pub fn open_wizard(&mut self) {
        self.wizard = Some(Wizard::new(&self.settings, self.system.as_ref()));
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

    fn refresh_system(&mut self, ctx: &egui::Context) {
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

    /// The three-way switch shown in every top bar.
    pub fn nav(&mut self, ui: &mut egui::Ui) {
        let l = self.lang();
        ui.selectable_value(&mut self.screen, Screen::Home, l.pick("Accueil", "Home"));
        ui.selectable_value(
            &mut self.screen,
            Screen::Workbench,
            l.pick("Atelier", "Studio"),
        );
        ui.selectable_value(&mut self.screen, Screen::Diagnostics, "Diagnostics");
    }

    fn rows(&self, backend: &BackendConfig) -> Vec<Row> {
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

    /// Paints the shell for one frame and returns what the app must do.
    pub fn ui(&mut self, ui: &mut egui::Ui, backend: &BackendConfig) -> Vec<Event> {
        let ctx = ui.ctx().clone();
        let mut events = Vec::new();
        if let Some(system) = self.system_task.as_ref().and_then(Task::poll) {
            self.system = Some(system);
            self.system_task = None;
        }
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
        self.poll_update();

        match self.screen {
            Screen::Home => self.screen_frame(ui, backend, Self::home),
            Screen::Diagnostics => self.screen_frame(ui, backend, Self::diagnostics),
            Screen::Workbench => {}
        }
        events.extend(self.show_wizard(&ctx));
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let layer = egui::LayerId::new(egui::Order::Foreground, egui::Id::new("drop-hint"));
            let rect = ctx.content_rect();
            let painter = ctx.layer_painter(layer);
            painter.rect_filled(rect, 0.0, egui::Color32::from_black_alpha(170));
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                self.lang().pick("Déposez pour ouvrir", "Drop to open"),
                egui::FontId::proportional(26.0),
                theme::SELECTION,
            );
        }
        events
    }

    fn screen_frame(
        &mut self,
        ui: &mut egui::Ui,
        backend: &BackendConfig,
        body: fn(&mut Self, &mut egui::Ui, &BackendConfig),
    ) {
        Panel::top("desktop-nav").exact_size(30.0).show(ui, |ui| {
            ui.horizontal_centered(|ui| {
                self.nav(ui);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(mono(format!("v{}", env!("CARGO_PKG_VERSION"))).color(theme::DIM));
                });
            });
        });
        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical()
                .auto_shrink([false; 2])
                .show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.allocate_ui(
                            egui::vec2(680.0_f32.min(ui.available_width()), 0.0),
                            |ui| {
                                ui.with_layout(Layout::top_down(Align::Min), |ui| {
                                    body(self, ui, backend)
                                });
                            },
                        );
                    });
                });
        });
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
                // Closing it early is a choice too: it stays one click away on the home
                // screen and in Diagnostics, but it does not pop up again by itself.
                self.settings.wizard_done = true;
                let _ = self.settings.save(&self.env.settings);
                finished.then_some(Event::BackendChanged)
            }
        }
    }

    fn home(&mut self, ui: &mut egui::Ui, backend: &BackendConfig) {
        let l = self.lang();
        ui.add_space(28.0);
        ui.label(RichText::new("LLM Surgery Studio").size(30.0).strong());
        ui.label(
            RichText::new(l.pick(
                "Inspecter, patcher et évaluer les poids de vos LLM locaux.",
                "Inspect, patch and evaluate the weights of your local LLMs.",
            ))
            .color(theme::DIM),
        );
        ui.add_space(18.0);

        card(ui, |ui| {
            ui.label(RichText::new(l.pick("Ouvrir", "Open")).strong());
            ui.label(l.pick(
                "Un graph.json, un flux .ndjson, un workspace .surgeon-studio.json ou un dossier. Vous pouvez aussi les glisser-déposer sur la fenêtre.",
                "A graph.json, an .ndjson stream, a .surgeon-studio.json workspace or a folder. You can also drop them on the window.",
            ));
            ui.horizontal(|ui| {
                ui.add(
                    egui::TextEdit::singleline(&mut self.open_field)
                        .hint_text(l.pick("chemin du fichier ou du dossier", "file or folder path"))
                        .desired_width(420.0),
                );
                if ui.button(l.pick("Ouvrir", "Open")).clicked()
                    && !self.open_field.trim().is_empty()
                {
                    let item = cli::classify(std::path::Path::new(self.open_field.trim()));
                    self.pending.push(item);
                }
            });
            if ui
                .button(l.pick("Nouveau workspace (Studio)", "New workspace (Studio)"))
                .clicked()
            {
                self.screen = Screen::Workbench;
            }
        });

        card(ui, |ui| {
            ui.label(RichText::new(l.pick("Configuration", "Setup")).strong());
            for row in self.rows(backend) {
                ui.horizontal(|ui| {
                    status_chip(ui, row.status, l);
                    ui.label(row.name);
                });
            }
            ui.horizontal(|ui| {
                if ui
                    .button(l.pick("Assistant de configuration", "Setup wizard"))
                    .clicked()
                {
                    self.open_wizard();
                }
                if ui.button("Diagnostics").clicked() {
                    self.screen = Screen::Diagnostics;
                }
            });
        });

        card(ui, |ui| {
            ui.label(
                RichText::new(l.pick("Langue et mises à jour", "Language and updates")).strong(),
            );
            ui.horizontal(|ui| {
                for (value, name) in [(Lang::Fr, "Français"), (Lang::En, "English")] {
                    if ui.selectable_label(self.lang() == value, name).clicked() {
                        self.settings.language = Some(value);
                        let _ = self.settings.save(&self.env.settings);
                    }
                }
            });
            self.update_ui(ui);
        });
        if let Some(m) = self.message.as_ref().filter(|m| !m.is_empty()) {
            ui.colored_label(theme::BAD, m);
        }
    }

    fn update_ui(&mut self, ui: &mut egui::Ui) {
        let l = self.lang();
        let ctx = ui.ctx().clone();
        match &self.update {
            UpdateState::Idle | UpdateState::UpToDate | UpdateState::Failed(_) => {
                if ui
                    .button(l.pick("Rechercher une mise à jour", "Check for updates"))
                    .on_hover_text(l.pick(
                        "Une requête HTTPS vers l'API des releases du projet. Rien n'est téléchargé sans votre confirmation.",
                        "One HTTPS request to the project's releases API. Nothing is downloaded without your confirmation.",
                    ))
                    .clicked()
                {
                    let current = env!("CARGO_PKG_VERSION").to_owned();
                    let wake = ctx.clone();
                    self.update = UpdateState::Checking(Task::spawn(
                        move || wake.request_repaint(),
                        move |_| update::check(&actions::production_fetcher(), update::LATEST_URL, &current),
                    ));
                }
                match &self.update {
                    UpdateState::UpToDate => {
                        ui.label(l.pick("Vous avez la dernière version.", "You are up to date."));
                    }
                    UpdateState::Failed(e) => {
                        ui.colored_label(theme::BAD, e);
                    }
                    _ => {}
                }
            }
            UpdateState::Checking(_) => {
                ui.label(l.pick("Recherche…", "Checking…"));
            }
            UpdateState::Available(u) => {
                let u = u.clone();
                ui.label(
                    RichText::new(format!(
                        "{} {}",
                        l.pick("Version disponible :", "New version:"),
                        u.version
                    ))
                    .color(theme::GOOD),
                );
                ui.label(
                    mono(format!(
                        "{}  {:.1} MiB",
                        u.name,
                        u.size as f64 / (1 << 20) as f64
                    ))
                    .size(11.0),
                );
                ui.label(mono(u.url.clone()).size(10.5));
                ui.label(
                    mono(match &u.sha256 {
                        Some(h) => format!("SHA-256 {h}"),
                        None => l.pick("SHA-256 non publié", "SHA-256 not published").into(),
                    })
                    .size(10.5),
                );
                if ui
                    .button(l.pick("Télécharger (je confirme)", "Download (I confirm)"))
                    .clicked()
                {
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
            }
            UpdateState::Downloading(u, task) => {
                let p = task.progress();
                ui.label(mono(&u.name));
                ui.add(match p.fraction() {
                    Some(f) => egui::ProgressBar::new(f).show_percentage(),
                    None => egui::ProgressBar::new(0.0).animate(true),
                });
                if ui.button(l.pick("Annuler", "Cancel")).clicked() {
                    task.cancel();
                }
            }
            UpdateState::Downloaded(path) => {
                ui.colored_label(theme::GOOD, l.pick(
                    "Téléchargé et vérifié. Fermez l'application puis lancez le programme d'installation vous-même :",
                    "Downloaded and verified. Close the app, then run the installer yourself:",
                ));
                ui.label(mono(path.display().to_string()).size(11.0));
            }
        }
    }

    fn poll_update(&mut self) {
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

    fn diagnostics(&mut self, ui: &mut egui::Ui, backend: &BackendConfig) {
        let l = self.lang();
        ui.add_space(14.0);
        ui.horizontal(|ui| {
            ui.heading("Diagnostics");
            if ui.button(l.pick("Actualiser", "Refresh")).clicked() {
                let ctx = ui.ctx().clone();
                self.refresh_system(&ctx);
            }
            if self.system_task.is_some() {
                ui.add(egui::Spinner::new());
            }
        });
        ui.add_space(6.0);
        if let Some(system) = &self.system {
            wizard::system_card(ui, system, l);
        }
        ui.add_space(8.0);
        let rows = self.rows(backend);
        let mut fix: Option<(Fix, bool)> = None;
        card(ui, |ui| {
            for row in &rows {
                ui.horizontal(|ui| {
                    status_chip(ui, row.status, l);
                    ui.vertical(|ui| {
                        // A fixed width: a Grid column here collapses to a few letters.
                        ui.set_width((ui.available_width() - 190.0).max(200.0));
                        ui.label(RichText::new(row.name).strong());
                        ui.label(RichText::new(&row.detail).small().color(theme::DIM));
                    });
                    if row.fix != Fix::None {
                        if ui.button(l.pick("Réparer", "Repair")).clicked() {
                            fix = Some((row.fix, false));
                        }
                        if ui.button(l.pick("Réinstaller", "Reinstall")).clicked() {
                            fix = Some((row.fix, true));
                        }
                    }
                });
                ui.add_space(4.0);
            }
        });
        if let Some((what, force)) = fix {
            let step = match what {
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
        ui.add_space(8.0);
        card(ui, |ui| {
            ui.label(RichText::new(l.pick("Export de diagnostics", "Diagnostics export")).strong());
            ui.label(l.pick(
                "Sans chemin personnel ni secret : le dossier utilisateur et le nom de session sont remplacés, les clés et jetons masqués.",
                "No personal path or secret: the home folder and user name are replaced, keys and tokens masked.",
            ));
            if ui
                .button(l.pick("Générer et copier", "Generate and copy"))
                .clicked()
            {
                let text = diag::export(self.system.as_ref(), &rows, &crate::paths::logs_dir(), l);
                ui.ctx().copy_text(text.clone());
                self.export = Some(text);
            }
            if let Some(text) = &self.export {
                ui.add(
                    egui::TextEdit::multiline(&mut text.as_str())
                        .font(egui::TextStyle::Monospace)
                        .desired_rows(8)
                        .desired_width(f32::INFINITY),
                );
            }
        });
    }
}

fn card(ui: &mut egui::Ui, body: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::group(ui.style())
        .inner_margin(12.0)
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            body(ui);
        });
    ui.add_space(8.0);
}

fn status_chip(ui: &mut egui::Ui, status: Status, l: Lang) {
    let color = match status {
        Status::Ok => theme::GOOD,
        Status::Missing => theme::BAD,
        Status::Outdated => theme::WARN,
    };
    ui.label(mono(format!("[{:^9}]", diag::status_word(status, l))).color(color));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::SurgeonApp;
    use crate::raster;

    fn env(dir: &std::path::Path) -> Env {
        Env {
            components: dir.join("data").join("components"),
            config: None,
            settings: dir.join("data").join("settings.json"),
        }
    }

    /// The strings the whole window paints on `screen`, through the real app.
    fn texts(screen: Screen, lang: Lang, dir: &std::path::Path) -> Vec<String> {
        let ctx = egui::Context::default();
        let settings = Settings {
            language: Some(lang),
            wizard_done: true,
            ..Default::default()
        };
        let desktop = Desktop::detached(None, screen, settings, env(dir));
        let mut app = SurgeonApp::with_shell(&ctx, None, desktop, dir.to_path_buf());
        raster::render(&ctx, [1280, 800], 1.0, |ui| app.draw(ui)).texts
    }

    #[test]
    fn launching_without_argument_shows_the_home_screen_not_a_usage_line() {
        let dir = tempfile::tempdir().unwrap();
        // The version is not part of the structure: a release bump must not break the golden.
        let texts: Vec<String> = texts(Screen::Home, Lang::Fr, dir.path())
            .iter()
            .map(|t| t.replace(env!("CARGO_PKG_VERSION"), "{version}"))
            .collect();
        let golden = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/home.fr.golden.txt"
        );
        let actual = texts.join("\n") + "\n";
        if std::env::var_os("UPDATE_GOLDEN").is_some() {
            std::fs::write(golden, &actual).unwrap();
        }
        // The fixture may be checked out with CRLF on Windows.
        let expected = std::fs::read_to_string(golden)
            .unwrap()
            .replace("\r\n", "\n");
        assert_eq!(
            actual, expected,
            "home structure changed; UPDATE_GOLDEN=1 to accept"
        );
        assert!(
            !actual.contains("graph.json | graph.ndjson"),
            "no usage message"
        );
    }

    #[test]
    fn the_language_switches_the_screens() {
        let dir = tempfile::tempdir().unwrap();
        let en = texts(Screen::Home, Lang::En, dir.path());
        assert!(en.iter().any(|t| t == "Setup"), "{en:?}");
        let diagnostics = texts(Screen::Diagnostics, Lang::En, dir.path());
        assert!(
            diagnostics.iter().any(|t| t == "Diagnostics export"),
            "{diagnostics:?}"
        );
        assert!(diagnostics.iter().any(|t| t == "Repair"), "{diagnostics:?}");
    }
}
