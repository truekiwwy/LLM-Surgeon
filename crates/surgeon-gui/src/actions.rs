//! What the wizard and the Diagnostics buttons actually do, without any UI: so that the same
//! code serves the first run, Repair and Reinstall, and is tested against the fake releases
//! server. Every function is safe to repeat.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use weight_surgeon::backend::config;

use crate::install::{self, InstallError, Installed, Stage};
use crate::net::{Fetcher, Policy};
use crate::releases::{Plan, Target, Variant};
use crate::settings::Settings;

/// The folders an action writes to; tests point them at a temp dir.
#[derive(Debug, Clone)]
pub struct Env {
    pub components: PathBuf,
    /// The engine's user `config.toml` (where the trusted llama.cpp is recorded), `None` when
    /// the environment disables it.
    pub config: Option<PathBuf>,
    pub settings: PathBuf,
}

impl Env {
    pub fn current() -> Self {
        Self {
            components: crate::paths::components_dir(),
            config: config::user_config_path(),
            settings: crate::paths::settings_path(),
        }
    }
}

pub struct EngineOutcome {
    pub installed: Installed,
    /// Nothing was downloaded: the build was already there and starts.
    pub already_there: bool,
    /// `Some` when the trust record could not be written (the build is installed anyway).
    pub register_error: Option<String>,
    pub smoke: String,
}

/// The release the wizard would download, for the consent screen. Network: one API call.
pub fn plan_engine(
    fetcher: &Fetcher,
    api_url: &str,
    variant: Variant,
) -> Result<Plan, InstallError> {
    install::fetch_plan(fetcher, api_url, variant, Target::current())
}

/// Installs `plan` (or finds it already installed and healthy), pins its hash when the
/// release published none, and records it as the trusted toolchain. Repair is this call; a
/// build that fails its smoke test is reinstalled.
pub fn ensure_engine(
    fetcher: &Fetcher,
    plan: &Plan,
    env: &Env,
    cancel: &AtomicBool,
    smoke: &dyn Fn(&Path) -> Result<String, String>,
    report: impl FnMut(Stage),
) -> Result<EngineOutcome, InstallError> {
    let mut settings = Settings::load(&env.settings).map_err(InstallError::Io)?;
    let mut already_there = false;
    // The last successful smoke line, whichever call (existing build or fresh install) made it.
    let line = std::cell::RefCell::new(None::<String>);
    let probe = |server: &Path| {
        let result = smoke(server);
        if let Ok(text) = &result {
            *line.borrow_mut() = Some(text.clone());
        }
        result
    };
    if let Some(existing) = install::find_installed(&env.components, plan) {
        match probe(&existing.server) {
            Ok(_) => already_there = true,
            // Broken on disk (antivirus, deleted DLL): forget it so the install lands clean.
            Err(_) => install::forget(&existing),
        }
    }
    let installed = install::install(
        fetcher,
        plan,
        &env.components,
        &mut settings.pins,
        cancel,
        &probe,
        report,
    )?;
    let smoke = line.take().unwrap_or_default();
    settings.save(&env.settings).map_err(InstallError::Io)?;
    let register_error = match &env.config {
        Some(path) => install::register_trusted(path, &installed).err(),
        None => Some("configuration utilisateur désactivée (WEIGHT_SURGEON_NO_USER_CONFIG)".into()),
    };
    install::collect_garbage(&env.components);
    Ok(EngineOutcome {
        installed,
        already_there,
        register_error,
        smoke,
    })
}

/// Creates the workspace folder, checks it is writable and remembers it. Idempotent.
pub fn set_workspace(env: &Env, dir: &Path) -> Result<(), String> {
    if dir.as_os_str().is_empty() {
        return Err("chemin vide".into());
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{} : {e}", dir.display()))?;
    let probe = dir.join(".surgeon-write-test");
    std::fs::write(&probe, b"")
        .map_err(|e| format!("{} n'est pas inscriptible : {e}", dir.display()))?;
    let _ = std::fs::remove_file(probe);
    let mut settings = Settings::load(&env.settings)?;
    settings.workspace = Some(dir.to_path_buf());
    settings.save(&env.settings)
}

pub fn production_fetcher() -> Fetcher {
    Fetcher::new(Policy::production())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::testkit::{archive, FakeReleases};

    fn env(dir: &Path) -> Env {
        Env {
            components: dir.join("components"),
            config: Some(dir.join("config.toml")),
            settings: dir.join("settings.json"),
        }
    }

    #[test]
    fn the_engine_flow_is_idempotent_and_records_the_trust() {
        let dir = tempfile::tempdir().unwrap();
        let env = env(dir.path());
        let name = "llama-b7-bin-win-vulkan-x64.zip";
        let fake = FakeReleases::serve(name, archive(&[("llama-server.exe", b"MZ")]), true);
        let fetcher = Fetcher::new(Policy::loopback());
        let target = Target {
            os: "win",
            arch: "x64",
        };
        let plan = install::fetch_plan(&fetcher, &fake.api_url(), Variant::Vulkan, target).unwrap();
        let ok = |_: &Path| -> Result<String, String> { Ok("version: b7".into()) };
        let cancel = AtomicBool::new(false);

        let first = ensure_engine(&fetcher, &plan, &env, &cancel, &ok, |_| {}).unwrap();
        assert!(!first.already_there && first.register_error.is_none());
        let saved = config::load_from(env.config.as_ref().unwrap()).unwrap();
        assert_eq!(
            saved.llama_cpp_dir.as_deref(),
            Some(first.installed.dir.as_path())
        );
        let before = (fake.archive_hits(), std::fs::read(&env.settings).unwrap());

        let second = ensure_engine(&fetcher, &plan, &env, &cancel, &ok, |_| {}).unwrap();
        assert!(
            second.already_there,
            "the rerun finds it and downloads nothing"
        );
        assert_eq!(second.installed, first.installed);
        assert_eq!(
            (fake.archive_hits(), std::fs::read(&env.settings).unwrap()),
            before
        );
    }

    #[test]
    fn a_broken_install_is_repaired_by_the_same_call() {
        let dir = tempfile::tempdir().unwrap();
        let env = env(dir.path());
        let name = "llama-b7-bin-win-cpu-x64.zip";
        let fake = FakeReleases::serve(name, archive(&[("llama-server.exe", b"MZ")]), true);
        let fetcher = Fetcher::new(Policy::loopback());
        let target = Target {
            os: "win",
            arch: "x64",
        };
        let plan = install::fetch_plan(&fetcher, &fake.api_url(), Variant::Cpu, target).unwrap();
        let cancel = AtomicBool::new(false);
        let good = |_: &Path| -> Result<String, String> { Ok("version: b7".into()) };
        let first = ensure_engine(&fetcher, &plan, &env, &cancel, &good, |_| {}).unwrap();
        // The antivirus ate the binary.
        std::fs::remove_file(&first.installed.server).unwrap();
        let repaired = ensure_engine(&fetcher, &plan, &env, &cancel, &good, |_| {}).unwrap();
        assert!(repaired.installed.server.is_file());
        assert_eq!(
            fake.archive_hits(),
            2,
            "a second download only because the file was gone"
        );
    }

    #[test]
    fn the_workspace_is_created_checked_and_remembered() {
        let dir = tempfile::tempdir().unwrap();
        let env = env(dir.path());
        let ws = dir.path().join("Mes modèles é à").join("espace de travail");
        set_workspace(&env, &ws).unwrap();
        set_workspace(&env, &ws).unwrap();
        assert!(ws.is_dir());
        assert_eq!(Settings::load(&env.settings).unwrap().workspace, Some(ws));
        assert!(set_workspace(&env, Path::new("")).is_err());
    }
}
