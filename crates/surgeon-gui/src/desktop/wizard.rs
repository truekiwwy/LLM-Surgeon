//! The first-run wizard (replayable from the home screen and from Diagnostics): language,
//! system diagnostic, llama.cpp, workspace, agent, sample model, tour, summary. Every
//! download needs the user to tick a consent box that sits next to the URL, size and hash;
//! offline mode replaces them with manual instructions.

use std::path::Path;

use eframe::egui::{self, Align, Layout, ProgressBar, RichText};

use crate::actions::{self, Env};
use crate::diag::SystemReport;
use crate::install::{self, InstallError, Stage};
use crate::releases::{Plan, Target, Variant};
use crate::sample_model::{self, QWEN};
use crate::settings::{Lang, Settings};
use crate::task::Task;
use crate::theme::{self, mono};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Step {
    Language,
    System,
    Engine,
    Workspace,
    Agent,
    Sample,
    Tour,
    Summary,
}

const STEPS: [Step; 8] = [
    Step::Language,
    Step::System,
    Step::Engine,
    Step::Workspace,
    Step::Agent,
    Step::Sample,
    Step::Tour,
    Step::Summary,
];

impl Step {
    fn index(self) -> usize {
        STEPS.iter().position(|s| *s == self).unwrap_or(0)
    }

    fn title(self, l: Lang) -> &'static str {
        match self {
            Self::Language => l.pick("Langue", "Language"),
            Self::System => l.pick("Votre machine", "Your machine"),
            Self::Engine => l.pick("Moteur llama.cpp", "llama.cpp engine"),
            Self::Workspace => l.pick("Espace de travail", "Workspace"),
            Self::Agent => l.pick("Agent", "Agent"),
            Self::Sample => l.pick("Modèle d'exemple", "Sample model"),
            Self::Tour => l.pick("Visite guidée", "Guided tour"),
            Self::Summary => l.pick("Récapitulatif", "Summary"),
        }
    }
}

/// What the engine step ended with, kept as plain text for the summary.
#[derive(Debug, Clone)]
pub struct EngineSummary {
    pub tag: String,
    pub variant: String,
    pub hash_line: String,
    pub smoke: String,
    pub already_there: bool,
    pub register_error: Option<String>,
}

type PlanResult = Result<Plan, String>;
type EngineResult = Result<EngineSummary, String>;

pub struct Wizard {
    pub step: Step,
    variant: Variant,
    pub offline: bool,
    /// Replace the installed build (Diagnostics > Reinstall) instead of keeping it.
    force: bool,
    /// The user picked a build: the machine diagnostic, arriving later, no longer decides.
    variant_touched: bool,
    plan_task: Option<Task<PlanResult>>,
    plan: Option<PlanResult>,
    engine_consent: bool,
    engine_task: Option<Task<EngineResult>>,
    engine: Option<EngineResult>,
    workspace: String,
    workspace_result: Option<Result<(), String>>,
    sample_consent: bool,
    sample_task: Option<Task<Result<(), String>>>,
    sample: Option<Result<(), String>>,
    copied: bool,
}

pub enum Outcome {
    Open,
    /// The wizard ended; `true` when the user went through to the summary.
    Closed(bool),
    /// The trusted toolchain changed: the backend dialog must be reloaded.
    BackendChanged,
}

/// What the wizard borrows from the app each frame.
pub struct Context<'a> {
    pub settings: &'a mut Settings,
    pub env: &'a Env,
    pub system: Option<&'a SystemReport>,
}

impl Wizard {
    pub fn new(settings: &Settings, system: Option<&SystemReport>) -> Self {
        let gpu_ready = system.is_some_and(|s| s.vulkan == Some(true) && !s.gpus.is_empty());
        Self {
            step: Step::Language,
            variant: if gpu_ready {
                Variant::Vulkan
            } else {
                Variant::Cpu
            },
            offline: false,
            force: false,
            variant_touched: false,
            plan_task: None,
            plan: None,
            engine_consent: false,
            engine_task: None,
            engine: None,
            workspace: settings
                .workspace
                .clone()
                .unwrap_or_else(crate::paths::default_workspace)
                .display()
                .to_string(),
            workspace_result: None,
            sample_consent: false,
            sample_task: None,
            sample: None,
            copied: false,
        }
    }

    /// Opened from Diagnostics on one step; `force` replaces what is installed.
    pub fn at(settings: &Settings, system: Option<&SystemReport>, step: Step, force: bool) -> Self {
        Self {
            step,
            force,
            ..Self::new(settings, system)
        }
    }

    /// A download may start only with the consent box ticked, a plan on screen, and online.
    pub fn engine_download_allowed(&self) -> bool {
        !self.offline
            && self.engine_consent
            && matches!(self.plan, Some(Ok(_)))
            && self.engine_task.is_none()
    }

    pub fn sample_download_allowed(&self) -> bool {
        !self.offline && self.sample_consent && self.sample_task.is_none()
    }

    fn busy(&self) -> bool {
        self.plan_task.is_some() || self.engine_task.is_some() || self.sample_task.is_some()
    }

    fn poll(&mut self, cx: &mut Context<'_>) -> bool {
        let mut backend_changed = false;
        if let Some(done) = self.plan_task.as_ref().and_then(Task::poll) {
            self.plan = Some(done);
            self.plan_task = None;
        }
        if let Some(done) = self.engine_task.as_ref().and_then(Task::poll) {
            backend_changed = done.is_ok();
            self.engine = Some(done);
            self.engine_task = None;
            // The pins the task saved are on disk: re-read them so a later save keeps them.
            if let Ok(fresh) = Settings::load(&cx.env.settings) {
                cx.settings.pins = fresh.pins;
            }
        }
        if let Some(done) = self.sample_task.as_ref().and_then(Task::poll) {
            self.sample = Some(done);
            self.sample_task = None;
        }
        backend_changed
    }

    fn fetch_plan(&mut self, ctx: &egui::Context, lang: Lang) {
        let variant = self.variant;
        let wake = ctx.clone();
        self.plan = None;
        self.engine_consent = false;
        self.plan_task = Some(Task::spawn(
            move || wake.request_repaint(),
            move |_| {
                actions::plan_engine(
                    &actions::production_fetcher(),
                    install::RELEASES_URL,
                    variant,
                )
                .map_err(|e| e.message(lang))
            },
        ));
    }

    fn start_engine(&mut self, ctx: &egui::Context, env: &Env, lang: Lang) {
        let Some(Ok(plan)) = self.plan.clone() else {
            return;
        };
        if !self.engine_download_allowed() {
            return;
        }
        let (env, force, wake) = (env.clone(), self.force, ctx.clone());
        self.engine = None;
        self.engine_task = Some(Task::spawn(
            move || wake.request_repaint(),
            move |handle| run_engine(&plan, &env, force, lang, handle),
        ));
    }

    fn start_sample(&mut self, ctx: &egui::Context, workspace: &Path) {
        if !self.sample_download_allowed() {
            return;
        }
        let dest = QWEN.dir(workspace);
        let wake = ctx.clone();
        self.sample = None;
        self.sample_task = Some(Task::spawn(
            move || wake.request_repaint(),
            move |handle| {
                let fetcher = actions::production_fetcher();
                let names: Vec<&str> = QWEN.files.iter().map(|f| f.name).collect();
                sample_model::install(
                    &fetcher,
                    &QWEN,
                    &|f| QWEN.url(f),
                    &dest,
                    &handle.cancel,
                    |index, done| handle.report(names[index], done, Some(QWEN.files[index].size)),
                )
                .map_err(|e| e.to_string())
            },
        ));
    }

    pub fn show(&mut self, ctx: &egui::Context, cx: &mut Context<'_>) -> Outcome {
        let backend_changed = self.poll(cx);
        if !self.variant_touched && self.plan.is_none() && self.engine_task.is_none() {
            self.variant = default_variant(cx.system);
        }
        let lang = cx.settings.lang();
        let mut outcome = Outcome::Open;
        egui::Modal::new(egui::Id::new("first-run-wizard")).show(ctx, |ui| {
            ui.set_width(620.0);
            self.header(ui, lang);
            ui.add_space(6.0);
            egui::ScrollArea::vertical()
                .max_height(430.0)
                .show(ui, |ui| match self.step {
                    Step::Language => self.language(ui, cx),
                    Step::System => self.system(ui, cx, lang),
                    Step::Engine => self.engine(ui, ctx, cx, lang),
                    Step::Workspace => self.workspace(ui, cx, lang),
                    Step::Agent => self.agent(ui, ctx, lang),
                    Step::Sample => self.sample(ui, ctx, cx, lang),
                    Step::Tour => tour(ui, lang),
                    Step::Summary => self.summary(ui, cx, lang),
                });
            ui.separator();
            if let Some(closed) = self.footer(ui, lang) {
                outcome = Outcome::Closed(closed);
            }
        });
        if self.busy() {
            ctx.request_repaint_after(std::time::Duration::from_millis(150));
        }
        match outcome {
            Outcome::Open if backend_changed => Outcome::BackendChanged,
            other => other,
        }
    }

    fn header(&self, ui: &mut egui::Ui, l: Lang) {
        ui.horizontal(|ui| {
            ui.heading(self.step.title(l));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                ui.label(
                    mono(format!("{}/{}", self.step.index() + 1, STEPS.len())).color(theme::DIM),
                );
            });
        });
        ui.add(
            ProgressBar::new((self.step.index() + 1) as f32 / STEPS.len() as f32)
                .desired_height(4.0),
        );
    }

    /// Back / skip / next. `Some(finished)` when the wizard closes.
    fn footer(&mut self, ui: &mut egui::Ui, l: Lang) -> Option<bool> {
        let mut closed = None;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    self.step != Step::Language && !self.busy(),
                    egui::Button::new(l.pick("Retour", "Back")),
                )
                .clicked()
            {
                self.step = STEPS[self.step.index() - 1];
            }
            if ui
                .button(l.pick("Fermer", "Close"))
                .on_hover_text(l.pick(
                    "Ferme l'assistant ; il restera disponible dans l'accueil et dans Diagnostics.",
                    "Closes the wizard; it stays available from Home and Diagnostics.",
                ))
                .clicked()
                && !self.busy()
            {
                closed = Some(false);
            }
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                let last = self.step == Step::Summary;
                let label = match (last, self.skippable()) {
                    (true, _) => l.pick("Terminer", "Finish"),
                    (false, true) => l.pick("Ignorer", "Skip"),
                    (false, false) => l.pick("Suivant", "Next"),
                };
                if ui
                    .add_enabled(!self.busy(), egui::Button::new(label))
                    .clicked()
                {
                    match last {
                        true => closed = Some(true),
                        false => self.step = STEPS[self.step.index() + 1],
                    }
                }
            });
        });
        closed
    }

    /// The step has an optional action the user has not done: the button reads Skip.
    fn skippable(&self) -> bool {
        match self.step {
            Step::Engine => !matches!(self.engine, Some(Ok(_))),
            Step::Sample => !matches!(self.sample, Some(Ok(_))),
            Step::Agent => true,
            _ => false,
        }
    }

    fn language(&mut self, ui: &mut egui::Ui, cx: &mut Context<'_>) {
        let l = cx.settings.lang();
        ui.label(l.pick(
            "Bienvenue. Cet assistant vérifie votre machine et installe les composants du moteur. Rien n'est téléchargé sans votre accord explicite, et rien n'est envoyé : aucune télémétrie.",
            "Welcome. This wizard checks your machine and installs the engine components. Nothing is downloaded without your explicit consent, and nothing is sent anywhere: no telemetry.",
        ));
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            for (value, name) in [(Lang::Fr, "Français"), (Lang::En, "English")] {
                if ui
                    .selectable_label(cx.settings.lang() == value, name)
                    .clicked()
                {
                    cx.settings.language = Some(value);
                    let _ = cx.settings.save(&cx.env.settings);
                }
            }
        });
    }

    fn system(&mut self, ui: &mut egui::Ui, cx: &Context<'_>, l: Lang) {
        let Some(s) = cx.system else {
            ui.horizontal(|ui| {
                ui.label(l.pick("Diagnostic en cours…", "Diagnosing…"));
            });
            return;
        };
        system_card(ui, s, l);
        ui.add_space(6.0);
        ui.checkbox(
            &mut self.offline,
            l.pick(
                "Mode hors ligne : ne rien télécharger, afficher les instructions manuelles",
                "Offline mode: download nothing, show manual instructions",
            ),
        );
    }

    fn engine(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, cx: &mut Context<'_>, l: Lang) {
        let target = Target::current();
        if !target.supported() {
            ui.colored_label(theme::WARN, l.pick(
                "L'installation automatique de llama.cpp n'est proposée que sous Windows. Installez-le à la main (voir ci-dessous) puis indiquez son dossier dans Backend…",
                "Automatic llama.cpp installation is only offered on Windows. Install it by hand (see below) and set its folder in Backend…",
            ));
            manual_engine(ui, cx.env, l);
            return;
        }
        ui.label(l.pick(
            "llama.cpp exécute les modèles sur le GPU (Vulkan) ou le CPU. Il sera enregistré comme votre toolchain de confiance : seul votre choix, ici, détermine quel binaire est lancé, jamais un agent.",
            "llama.cpp runs models on the GPU (Vulkan) or the CPU. It will be recorded as your trusted toolchain: only your choice here decides which binary runs, never an agent.",
        ));
        ui.add_space(6.0);
        let locked = self.engine_task.is_some() || self.plan_task.is_some();
        ui.add_enabled_ui(!locked, |ui| {
            let before = self.variant;
            ui.radio_value(
                &mut self.variant,
                Variant::Vulkan,
                l.pick(
                    "GPU Vulkan (recommandé : Radeon, Intel, NVIDIA)",
                    "Vulkan GPU (recommended: Radeon, Intel, NVIDIA)",
                ),
            );
            ui.radio_value(
                &mut self.variant,
                Variant::Cpu,
                l.pick("CPU seulement", "CPU only"),
            );
            ui.radio_value(
                &mut self.variant,
                Variant::Hip,
                l.pick("AMD HIP / ROCm (optionnel)", "AMD HIP / ROCm (optional)"),
            );
            if before != self.variant {
                self.variant_touched = true;
                self.plan = None;
                self.engine_consent = false;
            }
        });
        if cx.system.is_some_and(|s| s.vulkan == Some(false)) && self.variant == Variant::Vulkan {
            ui.colored_label(theme::WARN, l.pick(
                "Vulkan est absent de cette machine : le build Vulkan ne démarrera pas. Choisissez CPU ou installez le pilote GPU.",
                "Vulkan is missing on this machine: the Vulkan build will not start. Choose CPU or install the GPU driver.",
            ));
        }
        ui.add_space(6.0);
        if self.offline {
            manual_engine(ui, cx.env, l);
            return;
        }
        if self.plan_task.is_some() {
            ui.horizontal(|ui| {
                ui.label(l.pick(
                    "Interrogation de api.github.com…",
                    "Querying api.github.com…",
                ));
            });
        } else if ui
            .button(l.pick(
                "Rechercher la version (api.github.com)",
                "Find the release (api.github.com)",
            ))
            .on_hover_text(l.pick(
                "Une seule requête HTTPS en lecture vers l'API des releases de ggml-org/llama.cpp.",
                "A single read-only HTTPS request to the ggml-org/llama.cpp releases API.",
            ))
            .clicked()
        {
            self.fetch_plan(ctx, l);
        }
        match &self.plan {
            Some(Err(e)) => {
                ui.colored_label(theme::BAD, e);
                ui.label(l.pick(
                    "Hors ligne ? Cochez le mode hors ligne à l'étape précédente pour les instructions manuelles.",
                    "Offline? Tick offline mode on the previous step for manual instructions.",
                ));
                manual_engine(ui, cx.env, l);
            }
            Some(Ok(plan)) => {
                let plan = plan.clone();
                plan_card(ui, &plan, cx.settings.pins.get(&plan.name), l);
                ui.checkbox(
                    &mut self.engine_consent,
                    l.pick("J'autorise ce téléchargement", "I allow this download"),
                );
                if self.force {
                    ui.colored_label(
                        theme::WARN,
                        l.pick(
                            "Réinstallation : la version déjà installée sera remplacée.",
                            "Reinstall: the installed build will be replaced.",
                        ),
                    );
                }
                let can = self.engine_download_allowed();
                if ui
                    .add_enabled(
                        can,
                        egui::Button::new(
                            l.pick("Télécharger et installer", "Download and install"),
                        ),
                    )
                    .clicked()
                {
                    self.start_engine(ctx, cx.env, l);
                }
            }
            None => {}
        }
        if let Some(task) = &self.engine_task {
            let p = task.progress();
            ui.label(mono(p.label.clone()));
            ui.add(progress_bar(&p));
            if ui.button(l.pick("Annuler", "Cancel")).clicked() {
                task.cancel();
            }
        }
        match &self.engine {
            Some(Ok(done)) => engine_done(ui, done, l),
            Some(Err(e)) => {
                ui.colored_label(theme::BAD, e);
            }
            None => {}
        }
    }

    fn workspace(&mut self, ui: &mut egui::Ui, cx: &mut Context<'_>, l: Lang) {
        ui.label(l.pick(
            "Le dossier où vivent vos modèles, patchs et workspaces. Choisissez un dossier de travail à vous, pas le dossier d'installation.",
            "The folder holding your models, patches and workspaces. Pick a work folder of your own, not the install folder.",
        ));
        ui.add_space(6.0);
        ui.add(egui::TextEdit::singleline(&mut self.workspace).desired_width(f32::INFINITY));
        if ui
            .button(l.pick("Créer / utiliser ce dossier", "Create / use this folder"))
            .clicked()
        {
            let result = actions::set_workspace(cx.env, Path::new(self.workspace.trim()));
            if result.is_ok() {
                if let Ok(fresh) = Settings::load(&cx.env.settings) {
                    cx.settings.workspace = fresh.workspace;
                }
            }
            self.workspace_result = Some(result);
        }
        match &self.workspace_result {
            Some(Ok(())) => {
                ui.colored_label(
                    theme::GOOD,
                    l.pick(
                        "Dossier prêt et inscriptible.",
                        "Folder ready and writable.",
                    ),
                );
            }
            Some(Err(e)) => {
                ui.colored_label(theme::BAD, e);
            }
            None => {
                if cx.settings.workspace.is_some() {
                    ui.colored_label(
                        theme::GOOD,
                        l.pick("Déjà configuré.", "Already configured."),
                    );
                }
            }
        }
    }

    fn agent(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, l: Lang) {
        ui.label(l.pick(
            "Un agent (Claude Code, un orchestrateur...) pilote le moteur par le serveur MCP, confiné à vos dossiers et toujours avec sauvegarde. Cette étape est facultative.",
            "An agent (Claude Code, an orchestrator...) drives the engine through the MCP server, confined to your folders and always backed up. This step is optional.",
        ));
        ui.add_space(6.0);
        let command = format!(
            "claude mcp add surgeon -- surgeon mcp --allow-dir \"{}\"",
            self.workspace.trim()
        );
        ui.label(l.pick("Commande pour Claude Code :", "Command for Claude Code:"));
        ui.add(
            egui::TextEdit::multiline(&mut command.as_str())
                .font(egui::TextStyle::Monospace)
                .desired_rows(2)
                .desired_width(f32::INFINITY),
        );
        if ui.button(l.pick("Copier", "Copy")).clicked() {
            ctx.copy_text(command);
            self.copied = true;
        }
        if self.copied {
            ui.colored_label(theme::GOOD, l.pick("Copié.", "Copied."));
        }
        ui.add_space(6.0);
        ui.label(
            RichText::new(l.pick(
                "Un Hub d'agents dédié n'existe pas encore dans cette version ; le serveur MCP (docs/MCP.md) et le flux NDJSON affiché en direct dans le graphe sont les deux voies actuelles.",
                "A dedicated agent hub does not exist in this version yet; the MCP server (docs/MCP.md) and the NDJSON stream shown live in the graph are the two current routes.",
            ))
            .color(theme::DIM),
        );
    }

    fn sample(&mut self, ui: &mut egui::Ui, ctx: &egui::Context, cx: &Context<'_>, l: Lang) {
        let workspace = cx
            .settings
            .workspace
            .clone()
            .unwrap_or_else(|| self.workspace.trim().into());
        ui.label(l.pick(
            "Un petit modèle pour essayer le moteur tout de suite (facultatif).",
            "A small model to try the engine right away (optional).",
        ));
        ui.add_space(6.0);
        crate::ui::components::inset_frame().show(ui, |ui| {
            ui.label(RichText::new(QWEN.id).strong());
            ui.label(mono(format!(
                "{}  ·  {:.0} MiB  ·  {} {}",
                QWEN.repo,
                QWEN.total_size() as f64 / (1 << 20) as f64,
                l.pick("licence", "license"),
                QWEN.license
            )));
            ui.label(mono(format!(
                "{}  @ {}",
                "huggingface.co",
                &QWEN.revision[..12]
            )));
            egui::CollapsingHeader::new(
                l.pick("Fichiers et SHA-256 vérifiés", "Files and verified SHA-256"),
            )
            .show(ui, |ui| {
                for f in QWEN.files {
                    ui.label(
                        mono(format!("{:<24} {:>10} B  {}", f.name, f.size, f.sha256)).size(10.5),
                    );
                }
            });
        });
        if QWEN.is_installed(&workspace) {
            ui.colored_label(
                theme::GOOD,
                l.pick(
                    "Déjà installé dans l'espace de travail.",
                    "Already installed in the workspace.",
                ),
            );
        }
        if self.offline {
            ui.label(l.pick("Mode hors ligne : téléchargez les fichiers ci-dessus depuis la page du modèle sur huggingface.co (révision indiquée) et placez-les dans :", "Offline mode: download the files above from the model page on huggingface.co (revision shown) and place them in:"));
            ui.label(mono(QWEN.dir(&workspace).display().to_string()));
            return;
        }
        ui.checkbox(
            &mut self.sample_consent,
            l.pick(
                "J'autorise ces téléchargements (huggingface.co)",
                "I allow these downloads (huggingface.co)",
            ),
        );
        if ui
            .add_enabled(
                self.sample_download_allowed(),
                egui::Button::new(l.pick("Télécharger le modèle", "Download the model")),
            )
            .clicked()
        {
            self.start_sample(ctx, &workspace);
        }
        if let Some(task) = &self.sample_task {
            let p = task.progress();
            ui.label(mono(p.label.clone()));
            ui.add(progress_bar(&p));
            if ui.button(l.pick("Annuler", "Cancel")).clicked() {
                task.cancel();
            }
        }
        match &self.sample {
            Some(Ok(())) => {
                ui.colored_label(
                    theme::GOOD,
                    l.pick(
                        "Modèle installé, hash vérifiés.",
                        "Model installed, hashes verified.",
                    ),
                );
            }
            Some(Err(e)) => {
                ui.colored_label(theme::BAD, e);
            }
            None => {}
        }
    }

    fn summary(&mut self, ui: &mut egui::Ui, cx: &Context<'_>, l: Lang) {
        let ws = cx.settings.workspace.clone();
        let line = |ui: &mut egui::Ui, ok: bool, text: String| {
            let (mark, color) = if ok {
                ("OK", theme::GOOD)
            } else {
                ("--", theme::DIM)
            };
            ui.horizontal(|ui| {
                ui.label(mono(mark).color(color));
                ui.label(text);
            });
        };
        line(
            ui,
            true,
            format!(
                "{} : {}",
                l.pick("langue", "language"),
                cx.settings.lang().pick("Français", "English")
            ),
        );
        if let Some(s) = cx.system {
            line(
                ui,
                true,
                format!(
                    "{} : {} GPU, Vulkan {}",
                    l.pick("machine", "machine"),
                    s.gpus.len(),
                    match s.vulkan {
                        Some(true) => l.pick("présent", "present"),
                        Some(false) => l.pick("absent", "missing"),
                        None => "?",
                    }
                ),
            );
        }
        match &self.engine {
            Some(Ok(e)) => line(
                ui,
                true,
                format!(
                    "llama.cpp {} ({}){}",
                    e.tag,
                    e.variant,
                    if e.already_there {
                        l.pick(" - déjà installé", " - already installed")
                    } else {
                        ""
                    }
                ),
            ),
            _ => line(
                ui,
                false,
                l.pick(
                    "llama.cpp : non installé (Diagnostics > Réparer)",
                    "llama.cpp: not installed (Diagnostics > Repair)",
                )
                .into(),
            ),
        }
        line(
            ui,
            ws.is_some(),
            match &ws {
                Some(w) => format!(
                    "{} : {}",
                    l.pick("espace de travail", "workspace"),
                    w.display()
                ),
                None => l
                    .pick("espace de travail : non défini", "workspace: not set")
                    .into(),
            },
        );
        let sample = ws.as_deref().is_some_and(|w| QWEN.is_installed(w));
        line(
            ui,
            sample,
            format!(
                "{} : {}",
                l.pick("modèle d'exemple", "sample model"),
                if sample {
                    QWEN.id
                } else {
                    l.pick("non installé", "not installed")
                }
            ),
        );
        ui.add_space(6.0);
        ui.label(
            RichText::new(l.pick(
                "Rejouez cet assistant à tout moment depuis l'accueil ou la page Diagnostics.",
                "Replay this wizard any time from Home or Diagnostics.",
            ))
            .color(theme::DIM),
        );
    }
}

/// Vulkan when the machine has a GPU and the loader, else the CPU build.
fn default_variant(system: Option<&SystemReport>) -> Variant {
    match system.is_some_and(|s| s.vulkan == Some(true) && !s.gpus.is_empty()) {
        true => Variant::Vulkan,
        false => Variant::Cpu,
    }
}

/// Runs the engine install on the task thread and boils the outcome down to text.
fn run_engine(
    plan: &Plan,
    env: &Env,
    force: bool,
    lang: Lang,
    handle: &crate::task::Handle,
) -> EngineResult {
    let fetcher = actions::production_fetcher();
    if force {
        if let Some(existing) = install::find_installed(&env.components, plan) {
            install::forget(&existing);
        }
    }
    let outcome = actions::ensure_engine(
        &fetcher,
        plan,
        env,
        &handle.cancel,
        &install::smoke_test,
        |stage| match stage {
            Stage::Download { done, total } => {
                handle.report(lang.pick("téléchargement", "downloading"), done, total)
            }
            Stage::Verify => handle.report(
                lang.pick("vérification SHA-256", "verifying SHA-256"),
                0,
                None,
            ),
            Stage::Extract => handle.report(lang.pick("extraction", "extracting"), 0, None),
            Stage::Smoke => handle.report(lang.pick("test de fumée", "smoke test"), 0, None),
        },
    )
    .map_err(|e: InstallError| e.message(lang))?;
    let m = &outcome.installed.manifest;
    let hash_line = match m.hash_source {
        install::HashSource::Published => format!(
            "SHA-256 {} ({})",
            m.sha256,
            lang.pick("publié par la release", "published by the release")
        ),
        install::HashSource::Pinned => format!(
            "SHA-256 {} ({})",
            m.sha256,
            lang.pick(
                "calculé ici, épinglé au premier usage : la release n'en publie pas",
                "computed here, pinned at first use: the release publishes none"
            )
        ),
    };
    Ok(EngineSummary {
        tag: m.tag.clone(),
        variant: m.variant.clone(),
        hash_line,
        smoke: outcome.smoke,
        already_there: outcome.already_there,
        register_error: outcome.register_error,
    })
}

fn progress_bar(p: &crate::task::Progress) -> ProgressBar {
    match p.fraction() {
        Some(f) => ProgressBar::new(f).show_percentage(),
        // Size unknown: an empty bar, not a looping shimmer (no animation without a reason).
        None => ProgressBar::new(0.0),
    }
}

fn plan_card(ui: &mut egui::Ui, plan: &Plan, pinned: Option<&String>, l: Lang) {
    crate::ui::components::inset_frame().show(ui, |ui| {
        ui.label(RichText::new(format!("llama.cpp {} - {}", plan.tag, plan.variant)).strong());
        ui.label(mono(&plan.name));
        ui.label(mono(format!("URL      {}", plan.url)).size(11.0));
        ui.label(mono(format!("{}  {:.1} MiB", l.pick("Taille  ", "Size    "), plan.size as f64 / (1 << 20) as f64)));
        match (&plan.sha256, pinned) {
            (Some(h), _) => ui.label(mono(format!("SHA-256  {h}")).size(11.0)),
            (None, Some(p)) => ui.label(mono(format!("SHA-256  {}", l.pick("non publié ; épinglé précédemment", "not published; pinned earlier")) + &format!(" {p}")).size(11.0)),
            (None, None) => ui.colored_label(theme::WARN, l.pick(
                "La release ne publie pas de hash : il sera calculé, affiché, puis épinglé à la première utilisation (rien d'indépendant pour le confirmer).",
                "The release publishes no hash: it will be computed, shown, then pinned at first use (nothing independent to confirm it).",
            )),
        };
    });
}

fn engine_done(ui: &mut egui::Ui, done: &EngineSummary, l: Lang) {
    ui.colored_label(
        theme::GOOD,
        format!(
            "{} {} ({})",
            l.pick("Installé :", "Installed:"),
            done.tag,
            done.variant
        ),
    );
    ui.label(mono(&done.hash_line).size(11.0));
    ui.label(
        mono(format!(
            "{} {}",
            l.pick("Test de fumée :", "Smoke test:"),
            done.smoke
        ))
        .size(11.0),
    );
    match &done.register_error {
        None => {
            ui.label(l.pick(
                "Enregistré comme toolchain de confiance (llama_cpp_dir dans votre config.toml).",
                "Recorded as the trusted toolchain (llama_cpp_dir in your config.toml).",
            ));
        }
        Some(e) => {
            ui.colored_label(
                theme::WARN,
                format!(
                    "{} {e}",
                    l.pick(
                        "Installé mais non enregistré :",
                        "Installed but not recorded:"
                    )
                ),
            );
        }
    }
}

fn manual_engine(ui: &mut egui::Ui, env: &Env, l: Lang) {
    crate::ui::components::inset_frame().show(ui, |ui| {
        ui.label(RichText::new(l.pick("Installation manuelle", "Manual install")).strong());
        ui.label(l.pick(
            "1. Ouvrez https://github.com/ggml-org/llama.cpp/releases et prenez l'archive llama-<tag>-bin-win-vulkan-x64.zip (ou -cpu-x64).\n2. Comparez son SHA-256 à celui affiché sur la page de la release.\n3. Décompressez-la dans le dossier ci-dessous.\n4. Dans Backend…, indiquez ce dossier comme dossier llama.cpp.",
            "1. Open https://github.com/ggml-org/llama.cpp/releases and take llama-<tag>-bin-win-vulkan-x64.zip (or -cpu-x64).\n2. Compare its SHA-256 with the one shown on the release page.\n3. Unzip it into the folder below.\n4. In Backend…, set that folder as the llama.cpp folder.",
        ));
        ui.label(mono(install::llama_root(&env.components).display().to_string()));
    });
}

pub(super) fn system_card(ui: &mut egui::Ui, s: &SystemReport, l: Lang) {
    crate::ui::components::inset_frame().show(ui, |ui| {
        egui::Grid::new("system-grid").num_columns(2).spacing([14.0, 4.0]).show(ui, |ui| {
            ui.label(RichText::new("OS").color(theme::DIM));
            ui.label(mono(format!("{} {}", s.os, s.arch)));
            ui.end_row();
            ui.label(RichText::new("GPU").color(theme::DIM));
            ui.vertical(|ui| {
                if s.gpus.is_empty() {
                    ui.label(l.pick("aucun GPU détecté (moteur CPU)", "no GPU detected (CPU engine)"));
                }
                for g in &s.gpus {
                    ui.label(mono(format!(
                        "{} · {} · {}",
                        g.name,
                        g.vram_mib.map_or("VRAM ?".into(), |v| format!("{v} MiB")),
                        g.driver.as_deref().map(|d| format!("{} {d}", l.pick("pilote", "driver"))).unwrap_or_default()
                    )));
                }
            });
            ui.end_row();
            ui.label(RichText::new("Vulkan").color(theme::DIM));
            match s.vulkan {
                Some(true) => ui.colored_label(theme::GOOD, l.pick("présent", "present")),
                Some(false) => ui.colored_label(theme::WARN, l.pick(
                    "absent : pilote GPU à installer. Le build GPU de llama.cpp ne démarrera pas (build CPU ou moteurs natifs possibles) ; l'application reste utilisable.",
                    "missing: install the GPU driver. The llama.cpp GPU build will not start (CPU build or native engines work); the app stays usable.",
                )),
                None => ui.label("?"),
            };
            ui.end_row();
            ui.label(RichText::new("RAM").color(theme::DIM));
            ui.label(mono(format!("{} MiB ({} {} MiB)", s.ram_total_mib, l.pick("libres", "free"), s.ram_free_mib)));
            ui.end_row();
            ui.label(RichText::new(l.pick("Disque", "Disk")).color(theme::DIM));
            match s.disk_free_mib {
                Some(d) if d < 3 * 1024 => ui.colored_label(theme::WARN, format!("{d} MiB {}", l.pick("libres : peu pour un modèle", "free: tight for a model"))),
                Some(d) => ui.label(mono(format!("{d} MiB {}", l.pick("libres", "free")))),
                None => ui.label("?"),
            };
            ui.end_row();
        });
        for note in &s.notes {
            ui.label(RichText::new(note).small().color(theme::DIM));
        }
    });
}

fn tour(ui: &mut egui::Ui, l: Lang) {
    ui.label(l.pick(
        "Les quatre endroits à connaître :",
        "The four places to know:",
    ));
    ui.add_space(4.0);
    for (title, text) in [
        (
            l.pick("Studio (Canvas)", "Studio (Canvas)"),
            l.pick("Des cartes reliées : Modèle, Extraction, Patch, Diff, Évaluation. Chaque carte lance une commande de surgeon, avec les mêmes sauvegardes et le même rollback que la CLI.", "Linked cards: Model, Extract, Patch, Diff, Eval. Each card runs a surgeon command, with the same backups and rollback as the CLI."),
        ),
        (
            l.pick("Graphe 3D", "3D graph"),
            l.pick("Ouvrez un graph.json (ou un flux .ndjson suivi en direct) pour voir les couches, leurs métriques et les diffs.", "Open a graph.json (or a live-followed .ndjson stream) to see layers, their metrics and diffs."),
        ),
        (
            l.pick("Éditeur JSON", "JSON editor"),
            l.pick("Écrivez un patch.json avec validation en direct, lancez un essai à blanc (Ctrl+Entrée), puis appliquez.", "Write a patch.json with live validation, run a dry run (Ctrl+Enter), then apply."),
        ),
        (
            l.pick("Diagnostics", "Diagnostics"),
            l.pick("L'état de chaque dépendance, avec Réparer et Réinstaller, et un export expurgé à joindre à un rapport.", "The state of every dependency, with Repair and Reinstall, and a redacted export to attach to a report."),
        ),
    ] {
        crate::ui::components::inset_frame().show(ui, |ui| {
            ui.label(RichText::new(title).strong());
            ui.label(text);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wizard() -> Wizard {
        Wizard::new(&Settings::default(), None)
    }

    #[test]
    fn no_download_starts_without_consent_a_plan_and_a_connection() {
        let mut w = wizard();
        assert!(!w.engine_download_allowed(), "nothing yet");
        w.plan = Some(Ok(Plan {
            tag: "b1".into(),
            name: "x.zip".into(),
            url: "https://github.com/x".into(),
            size: 1,
            sha256: None,
            variant: "cpu",
        }));
        assert!(!w.engine_download_allowed(), "a plan alone is not consent");
        w.engine_consent = true;
        assert!(w.engine_download_allowed());
        w.offline = true;
        assert!(!w.engine_download_allowed(), "offline mode never downloads");
        assert!(!w.sample_download_allowed());
        w.offline = false;
        assert!(
            !w.sample_download_allowed(),
            "the sample model has its own consent"
        );
        w.sample_consent = true;
        assert!(w.sample_download_allowed());
    }

    #[test]
    fn the_default_engine_follows_the_machine() {
        let with_gpu = SystemReport {
            vulkan: Some(true),
            gpus: vec![crate::diag::Gpu::default()],
            ..Default::default()
        };
        assert_eq!(
            Wizard::new(&Settings::default(), Some(&with_gpu)).variant,
            Variant::Vulkan
        );
        let no_vulkan = SystemReport {
            vulkan: Some(false),
            gpus: vec![crate::diag::Gpu::default()],
            ..Default::default()
        };
        assert_eq!(
            Wizard::new(&Settings::default(), Some(&no_vulkan)).variant,
            Variant::Cpu
        );
        assert_eq!(wizard().variant, Variant::Cpu);
    }

    #[test]
    fn steps_are_ordered_and_replayable_at_a_step() {
        assert!(Step::Language < Step::Summary);
        let w = Wizard::at(&Settings::default(), None, Step::Engine, true);
        assert_eq!(w.step, Step::Engine);
        assert!(w.force);
    }
}
