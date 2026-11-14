//! The optional components the wizard and System install: the llama.cpp release plan, the
//! engine and the sample model. Owned by the desktop, so an install goes on after the wizard
//! closes (its progress shows in the status bar). Nothing is downloaded without the user's
//! consent next to the URL, size and hash, and nothing at all in local-only mode.

use std::path::Path;

use eframe::egui;

use crate::actions::{self, Env};
use crate::diag::SystemReport;
use crate::install::{self, InstallError, Stage};
use crate::releases::{Plan, Variant};
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

pub struct Setup {
    pub variant: Variant,
    /// Replace the installed build (System > Reinstall) instead of keeping it.
    pub force: bool,
    plan_task: Option<Task<PlanResult>>,
    pub plan: Option<PlanResult>,
    engine_task: Option<Task<EngineResult>>,
    pub engine: Option<EngineResult>,
    sample_task: Option<Task<Result<(), String>>>,
    pub sample: Option<Result<(), String>>,
}

/// Vulkan when the machine has a GPU and the loader, else the CPU build.
pub fn default_variant(system: Option<&SystemReport>) -> Variant {
    match system.is_some_and(|s| s.vulkan == Some(true) && !s.gpus.is_empty()) {
        true => Variant::Vulkan,
        false => Variant::Cpu,
    }
}

impl Setup {
    pub fn new(system: Option<&SystemReport>) -> Self {
        Self {
            variant: default_variant(system),
            force: false,
            plan_task: None,
            plan: None,
            engine_task: None,
            engine: None,
            sample_task: None,
            sample: None,
        }
    }

    pub fn fetching_plan(&self) -> bool {
        self.plan_task.is_some()
    }

    pub fn installing(&self) -> bool {
        self.engine_task.is_some() || self.sample_task.is_some()
    }

    /// The progress of the running installs, engine first.
    pub fn progress(&self) -> Vec<Progress> {
        self.engine_task
            .iter()
            .map(Task::progress)
            .chain(self.sample_task.iter().map(Task::progress))
            .collect()
    }

    pub fn cancel(&self) {
        for task in self.engine_task.iter() {
            task.cancel();
        }
        for task in self.sample_task.iter() {
            task.cancel();
        }
    }

    /// Another build: the plan on screen no longer applies.
    pub fn set_variant(&mut self, variant: Variant) {
        if variant != self.variant && self.plan_task.is_none() {
            self.variant = variant;
            self.plan = None;
        }
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

    /// One read-only HTTPS request to the llama.cpp releases API.
    pub fn fetch_plan(&mut self, ctx: &egui::Context, local_only: bool) {
        if local_only || self.plan_task.is_some() {
            return;
        }
        let (variant, lang, wake) = (self.variant, i18n::lang(), ctx.clone());
        self.plan = None;
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

    /// Collects what ended. `true` when the engine was installed: the trusted toolchain
    /// changed and the backend dialog must be reloaded.
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
        engine_done
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
    fn no_download_starts_without_consent_a_pick_a_plan_and_the_network() {
        let mut s = Setup::new(None);
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
        s.set_variant(Variant::Vulkan);
        assert!(s.plan.is_none(), "another build drops the plan");
    }

    #[test]
    fn the_default_engine_follows_the_machine() {
        let gpu = || vec![crate::diag::Gpu::default()];
        let with_vulkan = SystemReport {
            vulkan: Some(true),
            gpus: gpu(),
            ..Default::default()
        };
        assert_eq!(default_variant(Some(&with_vulkan)), Variant::Vulkan);
        let without = SystemReport {
            vulkan: Some(false),
            gpus: gpu(),
            ..Default::default()
        };
        assert_eq!(default_variant(Some(&without)), Variant::Cpu);
        assert_eq!(default_variant(None), Variant::Cpu);
    }
}
