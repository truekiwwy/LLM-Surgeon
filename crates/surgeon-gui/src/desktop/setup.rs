//! The optional components the wizard and the settings install: mistral.rs (its official CPU
//! build, or on Windows with an NVIDIA GPU a CUDA build from source), hipfire in WSL2 for an
//! AMD GPU, and the sample model. Owned by the desktop, so an install goes on after the wizard
//! closes (its progress shows in the status bar). Nothing is downloaded without the user's
//! consent next to the URL, size and hash, and nothing at all in local-only mode.

use std::path::{Path, PathBuf};

use eframe::egui;

use crate::actions::{self, Env};
use crate::install::{self, InstallError, Stage};
use crate::releases::Plan;
use crate::sample_model::{self, QWEN};
use crate::settings::{Lang, Settings};
use crate::task::{Progress, Task};
use crate::ui::i18n::{self, t, tf};

/// What the engine install ended with, as text for the wizard and System.
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

/// The components picked for one install.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Pick {
    pub engine: bool,
    pub sample: bool,
}

#[derive(Default)]
pub struct Setup {
    /// Replace the installed build (System > Reinstall) instead of keeping it.
    pub force: bool,
    plan_task: Option<Task<PlanResult>>,
    pub plan: Option<PlanResult>,
    engine_task: Option<Task<EngineResult>>,
    pub engine: Option<EngineResult>,
    sample_task: Option<Task<Result<(), String>>>,
    pub sample: Option<Result<(), String>>,
    hipfire_task: Option<Task<Result<(), String>>>,
    pub hipfire: Option<Result<(), String>>,
}

impl Setup {
    pub fn fetching_plan(&self) -> bool {
        self.plan_task.is_some()
    }

    pub fn installing(&self) -> bool {
        self.engine_task.is_some() || self.sample_task.is_some() || self.hipfire_task.is_some()
    }

    /// The progress of the running installs, engine first.
    pub fn progress(&self) -> Vec<Progress> {
        self.engine_task
            .iter()
            .map(Task::progress)
            .chain(self.sample_task.iter().map(Task::progress))
            .chain(self.hipfire_task.iter().map(Task::progress))
            .collect()
    }

    pub fn cancel(&self) {
        for task in self.engine_task.iter() {
            task.cancel();
        }
        for task in self.sample_task.iter().chain(self.hipfire_task.iter()) {
            task.cancel();
        }
    }

    /// Installs and starts hipfire in WSL2 (`hipfire_setup`) for `gpu` (its name decides the
    /// workarounds), with the user's consent and outside local-only mode, one install at a time.
    pub fn install_hipfire(
        &mut self,
        ctx: &egui::Context,
        consent: bool,
        local_only: bool,
        gpu: String,
    ) {
        if !consent || local_only || self.installing() {
            return;
        }
        let (lang, wake) = (i18n::lang(), ctx.clone());
        self.hipfire = None;
        self.hipfire_task = Some(Task::spawn(
            move || wake.request_repaint(),
            move |handle| {
                i18n::set_lang(lang);
                crate::hipfire_setup::run(handle, &gpu)
            },
        ));
    }

    /// Stops the WSL daemon and starts it again (`hipfire_setup::restart`): nothing is
    /// downloaded, so no consent; one task at a time, like an install.
    pub fn restart_hipfire(&mut self, ctx: &egui::Context, gpu: String) {
        if self.installing() {
            return;
        }
        let (lang, wake) = (i18n::lang(), ctx.clone());
        self.hipfire = None;
        self.hipfire_task = Some(Task::spawn(
            move || wake.request_repaint(),
            move |handle| {
                i18n::set_lang(lang);
                crate::hipfire_setup::restart(handle, &gpu)
            },
        ));
    }

    /// Whether hipfire is being installed or restarted.
    pub fn hipfire_busy(&self) -> bool {
        self.hipfire_task.is_some()
    }

    /// A download may start only with the consent box ticked, outside local-only mode, with
    /// something picked, the engine's plan on screen, and no install running.
    pub fn download_allowed(&self, pick: Pick, consent: bool, local_only: bool) -> bool {
        consent
            && !local_only
            && (pick.engine || pick.sample)
            && (!pick.engine || matches!(self.plan, Some(Ok(_))))
            && !self.installing()
    }

    /// One read-only HTTPS request to the mistral.rs releases API.
    pub fn fetch_plan(&mut self, ctx: &egui::Context, local_only: bool) {
        if local_only || self.plan_task.is_some() {
            return;
        }
        let (lang, wake) = (i18n::lang(), ctx.clone());
        self.plan = None;
        self.plan_task = Some(Task::spawn(
            move || wake.request_repaint(),
            move |_| {
                actions::plan_engine(&actions::production_fetcher(), install::RELEASES_URL)
                    .map_err(|e| e.message(lang))
            },
        ));
    }

    /// Starts the picked downloads, when [`Self::download_allowed`] says so.
    pub fn install(
        &mut self,
        ctx: &egui::Context,
        env: &Env,
        workspace: &Path,
        pick: Pick,
        consent: bool,
        local_only: bool,
    ) {
        if !self.download_allowed(pick, consent, local_only) {
            return;
        }
        let lang = i18n::lang();
        if let (true, Some(Ok(plan))) = (pick.engine, self.plan.clone()) {
            let (env, force, wake) = (env.clone(), self.force, ctx.clone());
            self.engine = None;
            self.engine_task = Some(Task::spawn(
                move || wake.request_repaint(),
                move |handle| run_engine(&plan, &env, force, lang, handle),
            ));
        }
        if pick.sample {
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
                        |index, done| {
                            handle.report(names[index], done, Some(QWEN.files[index].size))
                        },
                    )
                    .map_err(|e| e.to_string())
                },
            ));
        }
    }

    /// Builds mistral.rs with CUDA from the tag of the plan on screen (`cargo install`), with
    /// the same consent as a download (the sources come from the network).
    pub fn build_cuda(&mut self, ctx: &egui::Context, env: &Env, consent: bool, local_only: bool) {
        let engine = Pick {
            engine: true,
            sample: false,
        };
        let Some(Ok(plan)) = self.plan.clone() else {
            return;
        };
        if !self.download_allowed(engine, consent, local_only) {
            return;
        }
        let (env, lang, wake) = (env.clone(), i18n::lang(), ctx.clone());
        self.engine = None;
        self.engine_task = Some(Task::spawn(
            move || wake.request_repaint(),
            move |handle| run_cuda_build(&plan.tag, &env, lang, handle),
        ));
    }

    /// Collects what ended. `true` when mistral.rs was installed or hipfire answers: the
    /// backends changed and must be probed again.
    pub fn poll(&mut self, settings: &mut Settings, env: &Env) -> bool {
        if let Some(done) = self.plan_task.as_ref().and_then(Task::poll) {
            self.plan = Some(done);
            self.plan_task = None;
        }
        let mut engine_done = false;
        if let Some(done) = self.engine_task.as_ref().and_then(Task::poll) {
            engine_done = done.is_ok();
            self.engine = Some(done);
            self.engine_task = None;
            // The pins the task saved are on disk: re-read them so a later save keeps them.
            if let Ok(fresh) = Settings::load(&env.settings) {
                settings.pins = fresh.pins;
            }
        }
        if let Some(done) = self.sample_task.as_ref().and_then(Task::poll) {
            self.sample = Some(done);
            self.sample_task = None;
        }
        // The daemon answers: the next probe sees it and the automatic choice takes it.
        let mut hipfire_up = false;
        if let Some(done) = self.hipfire_task.as_ref().and_then(Task::poll) {
            hipfire_up = done.is_ok();
            self.hipfire = Some(done);
            self.hipfire_task = None;
        }
        engine_done || hipfire_up
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
    // The language is per thread: this one speaks the interface's.
    i18n::set_lang(lang);
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
            Stage::Download { done, total } => handle.report(t("setup.download"), done, total),
            Stage::Verify => handle.report(t("setup.verify"), 0, None),
            Stage::Extract => handle.report(t("setup.extract"), 0, None),
            Stage::Smoke => handle.report(t("setup.smoke"), 0, None),
        },
    )
    .map_err(|e: InstallError| e.message(lang))?;
    let m = &outcome.installed.manifest;
    let source = match m.hash_source {
        install::HashSource::Published => t("setup.hash.published"),
        install::HashSource::Pinned => t("setup.hash.pinned"),
    };
    Ok(EngineSummary {
        tag: m.tag.clone(),
        variant: m.variant.clone(),
        hash_line: tf("setup.hash", &[("sha", &m.sha256), ("source", &source)]),
        smoke: outcome.smoke,
        already_there: outcome.already_there,
        register_error: outcome.register_error,
    })
}

/// `cargo` and the CUDA compiler on the PATH: what a CUDA build of mistral.rs needs.
pub fn cuda_toolchain() -> Result<(), String> {
    let path = std::env::var_os("PATH");
    for (tool, what) in [
        ("cargo", "Rust (rustup.rs)"),
        ("nvcc", "le CUDA Toolkit NVIDIA"),
    ] {
        if crate::agent::runtime::find_on_path(tool, path.as_deref()).is_none() {
            return Err(format!("{tool} introuvable : installez {what}"));
        }
    }
    Ok(())
}

/// The folder a CUDA build of `tag` installs into (`bin/mistralrs` under it).
pub fn cuda_root(components: &Path, tag: &str) -> PathBuf {
    install::mistral_root(components).join(format!("{tag}-cuda"))
}

/// `cargo install` of mistral.rs with CUDA, from the release tag, into the components
/// folder; then smoke-tested and recorded like a downloaded build.
fn run_cuda_build(tag: &str, env: &Env, lang: Lang, handle: &crate::task::Handle) -> EngineResult {
    i18n::set_lang(lang);
    cuda_toolchain()?;
    let root = cuda_root(&env.components, tag);
    let mut cargo = crate::platform::command("cargo");
    cargo
        .args([
            "install",
            "--locked",
            "--git",
            "https://github.com/EricLBuehler/mistral.rs",
        ])
        .args([
            "--tag",
            tag,
            "mistralrs-cli",
            "--features",
            "cuda",
            "--root",
        ])
        .arg(&root);
    crate::task::run_streamed(cargo, handle).map_err(|last| format!("cargo install : {last}"))?;
    let bin = root.join("bin").join(install::server_name());
    handle.report(t("setup.smoke"), 0, None);
    let smoke = install::smoke_test(&bin)?;
    let register_error = match &env.config {
        Some(path) => weight_surgeon::backend::config::load_from(path)
            .and_then(|mut cfg| {
                cfg.mistralrs_bin = Some(bin.clone());
                weight_surgeon::backend::config::save_to(path, &cfg)
            })
            .err(),
        None => Some("configuration utilisateur désactivée (WEIGHT_SURGEON_NO_USER_CONFIG)".into()),
    };
    Ok(EngineSummary {
        tag: tag.to_owned(),
        variant: "cuda".into(),
        hash_line: tf("setup.built", &[("tag", &tag)]),
        smoke,
        already_there: false,
        register_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> Plan {
        Plan {
            tag: "b1".into(),
            name: "x.zip".into(),
            url: "https://github.com/x".into(),
            size: 1,
            sha256: None,
            variant: "cpu",
        }
    }

    #[test]
    fn a_cuda_build_lands_next_to_the_downloaded_builds() {
        let root = cuda_root(Path::new("C:/data/components"), "v0.9.5");
        assert_eq!(root, Path::new("C:/data/components/mistral.rs/v0.9.5-cuda"));
    }

    #[test]
    fn no_download_starts_without_consent_a_pick_a_plan_and_the_network() {
        let mut s = Setup::default();
        let engine = Pick {
            engine: true,
            sample: false,
        };
        let sample = Pick {
            engine: false,
            sample: true,
        };
        assert!(
            !s.download_allowed(Pick::default(), true, false),
            "nothing picked"
        );
        assert!(!s.download_allowed(engine, true, false), "no plan yet");
        assert!(
            s.download_allowed(sample, true, false),
            "the sample needs no plan"
        );
        assert!(!s.download_allowed(sample, false, false), "no consent");
        s.plan = Some(Ok(plan()));
        assert!(s.download_allowed(engine, true, false));
        assert!(
            !s.download_allowed(engine, true, true),
            "local only never downloads"
        );
    }
}
