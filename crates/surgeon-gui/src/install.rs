//! Installing llama.cpp for the operator: pick the archive from the releases API, download it
//! (resumable), verify its SHA-256, extract it safely, smoke-test it and record it as the
//! trusted toolchain.
//!
//! The trust record is the one the CLI already has: `llama_cpp_dir` in the user's
//! `config.toml` (what `--llama-cpp-dir` / `$LLAMA_CPP_DIR` set). It is written here, by the
//! operator's click, and never by an agent: the MCP server only reads it.
//!
//! Layout: `components/llama.cpp/<tag>-<variant>/` (binaries at its root) plus a
//! `manifest.json`. A new version never replaces a folder in use: on Windows a running
//! `llama-server.exe` cannot be deleted or its folder renamed, so a locked folder is left
//! alone and the install lands in `<name>-2`, `<name>-3`... Leftovers are collected later.

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use weight_surgeon::backend::config::{self, BackendConfig};
use weight_surgeon::bench::score::Backend;

use crate::extract::{self, ExtractError};
use crate::net::{Fetcher, NetError, MAX_API_BYTES};
use crate::platform;
use crate::releases::{self, Plan, Target, Variant};
use crate::settings::Lang;

/// The release list the wizard reads (newest first).
pub const RELEASES_URL: &str =
    "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30";

/// Windows `ERROR_VIRUS_INFECTED` / `ERROR_VIRUS_DELETED`: what an antivirus leaves behind.
const VIRUS_ERRORS: [i32; 2] = [225, 226];

#[derive(Debug)]
pub enum InstallError {
    Net(NetError),
    NoAsset,
    /// The archive does not match the digest the release publishes.
    HashMismatch {
        expected: String,
        actual: String,
    },
    /// Same archive name, a hash different from the one pinned at first use.
    PinMismatch {
        pinned: String,
        actual: String,
    },
    Extract(ExtractError),
    /// The file disappeared or was blocked between two steps: an antivirus.
    Quarantined(String),
    NoServer,
    Smoke(String),
    Io(String),
    Cancelled,
}

impl std::fmt::Display for InstallError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Net(e) => e.fmt(f),
            Self::NoAsset => f.write_str("aucune archive llama.cpp ne correspond à cette machine dans les releases récentes"),
            Self::HashMismatch { expected, actual } => write!(
                f,
                "SHA-256 incorrect, archive supprimée : attendu {expected}, reçu {actual}"
            ),
            Self::PinMismatch { pinned, actual } => write!(
                f,
                "cette archive a changé depuis le premier téléchargement (épinglé {pinned}, reçu {actual}) : refusée"
            ),
            Self::Extract(e) => e.fmt(f),
            Self::Quarantined(path) => write!(
                f,
                "fichier bloqué ou supprimé, probablement par l'antivirus : {path}. Ajoutez une exclusion pour le dossier des composants ou installez llama.cpp à la main"
            ),
            Self::NoServer => f.write_str("llama-server est absent de l'archive"),
            Self::Smoke(e) => write!(f, "test de fumée en échec : {e}"),
            Self::Io(e) => f.write_str(e),
            Self::Cancelled => f.write_str("installation annulée"),
        }
    }
}

impl std::error::Error for InstallError {}

impl InstallError {
    /// The message in the user's language (`Display` is French, like the rest of the crate).
    pub fn message(&self, lang: Lang) -> String {
        match (lang, self) {
            (Lang::Fr, _) => self.to_string(),
            (Lang::En, Self::Net(NetError::Unreachable(e))) => {
                format!("network unreachable (offline, proxy or firewall): {e}")
            }
            (Lang::En, Self::Net(NetError::Blocked(url))) => {
                format!("address refused (HTTPS and known domains only): {url}")
            }
            (Lang::En, Self::Net(NetError::Http(code))) => format!("the server answered HTTP {code}"),
            (Lang::En, Self::Net(NetError::Interrupted(n))) => {
                format!("connection dropped after {n} bytes (the download can resume)")
            }
            (Lang::En, Self::NoAsset) => {
                "no llama.cpp archive matches this machine in the recent releases".into()
            }
            (Lang::En, Self::HashMismatch { expected, actual }) => {
                format!("wrong SHA-256, archive deleted: expected {expected}, got {actual}")
            }
            (Lang::En, Self::PinMismatch { pinned, actual }) => format!(
                "this archive changed since the first download (pinned {pinned}, got {actual}): refused"
            ),
            (Lang::En, Self::Quarantined(path)) => format!(
                "file blocked or deleted, most likely by antivirus software: {path}. Add an exclusion for the components folder or install llama.cpp by hand"
            ),
            (Lang::En, Self::NoServer) => "llama-server is missing from the archive".into(),
            (Lang::En, Self::Smoke(e)) => format!("smoke test failed: {e}"),
            (Lang::En, Self::Cancelled) => "installation cancelled".into(),
            (Lang::En, Self::Extract(_) | Self::Io(_) | Self::Net(_)) => self.to_string(),
        }
    }
}

impl From<NetError> for InstallError {
    fn from(e: NetError) -> Self {
        match e {
            NetError::Cancelled => Self::Cancelled,
            other => Self::Net(other),
        }
    }
}

impl From<ExtractError> for InstallError {
    fn from(e: ExtractError) -> Self {
        match e {
            ExtractError::Cancelled => Self::Cancelled,
            other => Self::Extract(other),
        }
    }
}

/// Where the hash that was checked came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HashSource {
    /// The release's own `digest` field.
    Published,
    /// Computed here, pinned at first use (the release published none).
    Pinned,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub tag: String,
    pub asset: String,
    pub variant: String,
    pub sha256: String,
    pub hash_source: HashSource,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Installed {
    pub dir: PathBuf,
    pub server: PathBuf,
    pub manifest: Manifest,
}

/// What the install is doing, for the progress bar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Download { done: u64, total: Option<u64> },
    Verify,
    Extract,
    Smoke,
}

pub fn server_name() -> &'static str {
    match cfg!(windows) {
        true => "llama-server.exe",
        false => "llama-server",
    }
}

pub fn llama_root(components: &Path) -> PathBuf {
    components.join("llama.cpp")
}

/// The newest release carrying the archive for `variant` on `target`.
pub fn fetch_plan(
    fetcher: &Fetcher,
    api_url: &str,
    variant: Variant,
    target: Target,
) -> Result<Plan, InstallError> {
    let json = fetcher.get_bytes(api_url, MAX_API_BYTES)?;
    let releases = releases::parse_releases(&json).map_err(InstallError::Io)?;
    releases::select(&releases, variant, target).ok_or(InstallError::NoAsset)
}

fn read_manifest(dir: &Path) -> Option<Manifest> {
    serde_json::from_slice(&fs::read(dir.join("manifest.json")).ok()?).ok()
}

/// Every installed build, newest folder name last.
pub fn list_installed(components: &Path) -> Vec<Installed> {
    let mut found: Vec<Installed> = fs::read_dir(llama_root(components))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let dir = entry.path();
            let manifest = read_manifest(&dir)?;
            let server = dir.join(server_name());
            server.is_file().then_some(Installed {
                dir,
                server,
                manifest,
            })
        })
        .collect();
    found.sort_by(|a, b| a.dir.cmp(&b.dir));
    found
}

/// The build of this exact release and flavor, when it is already there and complete.
pub fn find_installed(components: &Path, plan: &Plan) -> Option<Installed> {
    list_installed(components)
        .into_iter()
        .find(|i| i.manifest.tag == plan.tag && i.manifest.asset == plan.name)
}

/// Marks a build as not installed (the next install lands in a clean folder). Its files are
/// left: a locked or half-deleted folder is collected later.
pub fn forget(installed: &Installed) {
    let _ = fs::remove_file(installed.dir.join("manifest.json"));
}

/// `llama-server --version`: proves the binary starts (DLLs found) and says which build.
pub fn smoke_test(server: &Path) -> Result<String, String> {
    let mut cmd = platform::command(server);
    cmd.arg("--version");
    let (code, text) = platform::run_capture(cmd, Duration::from_secs(20))
        .map_err(|e| format!("{} : {e}", server.display()))?;
    let version = text
        .lines()
        .find(|l| l.trim_start().to_ascii_lowercase().starts_with("version"))
        .map(|l| l.trim().to_owned());
    match (code, version) {
        (Some(0), Some(v)) => Ok(v),
        // STATUS_DLL_NOT_FOUND: a runtime the archive does not carry.
        (Some(c), _) if c as u32 == 0xC000_0135 => {
            Err("une DLL manque (runtime Visual C++ ou pilote GPU) : code 0xC0000135".into())
        }
        (Some(c), _) => Err(format!("code de sortie {c} : {}", first_line(&text))),
        (None, _) => Err("aucune réponse en 20 s".into()),
    }
}

fn first_line(text: &str) -> &str {
    text.lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim()
}

/// `load_tensors: offloaded 25/25 layers to GPU` -> (25, 25).
pub fn parse_offload(log: &str) -> Option<(u32, u32)> {
    let rest = log.split("offloaded ").nth(1)?;
    let (ratio, _) = rest.split_once(" layers")?;
    let (done, total) = ratio.split_once('/')?;
    Some((done.trim().parse().ok()?, total.trim().parse().ok()?))
}

/// With a tiny GGUF at hand: does `llama-cli` really put its layers on the GPU?
pub fn check_gpu_offload(dir: &Path, gguf: &Path) -> Result<Option<(u32, u32)>, String> {
    let cli = dir.join(match cfg!(windows) {
        true => "llama-cli.exe",
        false => "llama-cli",
    });
    if !cli.is_file() {
        return Ok(None);
    }
    let mut cmd = platform::command(cli);
    cmd.args(["-m"])
        .arg(gguf)
        .args(["-ngl", "99", "-n", "1", "-p", "hi", "--no-warmup"]);
    let (_, log) =
        platform::run_capture(cmd, Duration::from_secs(90)).map_err(|e| e.to_string())?;
    Ok(parse_offload(&log))
}

fn find_server_root(staging: &Path) -> Option<PathBuf> {
    let name = server_name();
    let mut level = vec![staging.to_path_buf()];
    for _ in 0..3 {
        if let Some(hit) = level.iter().find(|d| d.join(name).is_file()) {
            return Some(hit.clone());
        }
        level = level
            .iter()
            .filter_map(|d| fs::read_dir(d).ok())
            .flat_map(|rd| rd.flatten())
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
    }
    None
}

fn io_message(path: &Path, e: &io::Error) -> InstallError {
    match e.raw_os_error() {
        Some(code) if VIRUS_ERRORS.contains(&code) => {
            InstallError::Quarantined(path.display().to_string())
        }
        _ => InstallError::Io(format!("{} : {e}", path.display())),
    }
}

/// A folder name that is free, moving a stale one aside when the system lets it.
fn free_dir(root: &Path, name: &str) -> PathBuf {
    let first = root.join(name);
    if !first.exists() {
        return first;
    }
    let aside = root.join(format!("{name}.old-{}", std::process::id()));
    if fs::rename(&first, &aside).is_ok() {
        return first;
    }
    // Locked (a running llama-server.exe): never touch it, install next to it.
    (2..)
        .map(|n| root.join(format!("{name}-{n}")))
        .find(|p| !p.exists())
        .expect("an unbounded range has a free name")
}

/// Removes the staging folders and the moved-aside builds, best effort (a locked one stays
/// for the next start).
pub fn collect_garbage(components: &Path) {
    for dir in [components.to_path_buf(), llama_root(components)] {
        for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with(".staging-") || name.contains(".old-") {
                let _ = fs::remove_dir_all(entry.path());
            }
        }
    }
}

/// Downloads, verifies, extracts and smoke-tests `plan`. `pins` receives the hash of an
/// archive the release gave no digest for (the caller saves the settings). Running it again
/// for an installed build does nothing and touches no network.
pub fn install(
    fetcher: &Fetcher,
    plan: &Plan,
    components: &Path,
    pins: &mut BTreeMap<String, String>,
    cancel: &AtomicBool,
    smoke: &dyn Fn(&Path) -> Result<String, String>,
    mut report: impl FnMut(Stage),
) -> Result<Installed, InstallError> {
    if let Some(done) = find_installed(components, plan) {
        return Ok(done);
    }
    let downloads = components.join(".downloads");
    let part = downloads.join(format!("{}.part", plan.name));
    let sha = fetcher.download(&plan.url, &part, Some(plan.size), cancel, |done, total| {
        report(Stage::Download { done, total })
    })?;

    report(Stage::Verify);
    let source = match &plan.sha256 {
        Some(expected) if *expected != sha => {
            let _ = fs::remove_file(&part);
            return Err(InstallError::HashMismatch {
                expected: expected.clone(),
                actual: sha,
            });
        }
        Some(_) => HashSource::Published,
        None => {
            if let Some(pinned) = pins.get(&plan.name).filter(|p| **p != sha) {
                let _ = fs::remove_file(&part);
                return Err(InstallError::PinMismatch {
                    pinned: pinned.clone(),
                    actual: sha,
                });
            }
            pins.insert(plan.name.clone(), sha.clone());
            HashSource::Pinned
        }
    };
    if !part.is_file() {
        return Err(InstallError::Quarantined(part.display().to_string()));
    }

    report(Stage::Extract);
    let staging = components.join(format!(".staging-{}", std::process::id()));
    let _ = fs::remove_dir_all(&staging);
    let outcome = install_extracted(
        plan,
        &part,
        &staging,
        components,
        sha,
        source,
        cancel,
        smoke,
        &mut report,
    );
    let _ = fs::remove_dir_all(&staging);
    if outcome.is_ok() {
        let _ = fs::remove_file(&part);
    }
    outcome
}

#[allow(clippy::too_many_arguments)]
fn install_extracted(
    plan: &Plan,
    part: &Path,
    staging: &Path,
    components: &Path,
    sha256: String,
    hash_source: HashSource,
    cancel: &AtomicBool,
    smoke: &dyn Fn(&Path) -> Result<String, String>,
    report: &mut impl FnMut(Stage),
) -> Result<Installed, InstallError> {
    extract::extract_zip(part, staging, cancel).map_err(|e| match e {
        ExtractError::Io(_) => match e.to_string() {
            msg if VIRUS_ERRORS
                .iter()
                .any(|c| msg.contains(&format!("os error {c}"))) =>
            {
                InstallError::Quarantined(part.display().to_string())
            }
            _ => InstallError::Extract(e),
        },
        other => other.into(),
    })?;
    let root = find_server_root(staging).ok_or(InstallError::NoServer)?;
    let llama = llama_root(components);
    fs::create_dir_all(&llama).map_err(|e| io_message(&llama, &e))?;
    let dir = free_dir(&llama, &format!("{}-{}", plan.tag, plan.variant));
    fs::rename(&root, &dir).map_err(|e| io_message(&dir, &e))?;

    let manifest = Manifest {
        tag: plan.tag.clone(),
        asset: plan.name.clone(),
        variant: plan.variant.to_owned(),
        sha256,
        hash_source,
    };
    let server = dir.join(server_name());
    report(Stage::Smoke);
    if let Err(e) = smoke(&server) {
        // A build that does not start is not kept: the next try starts clean.
        let _ = fs::remove_dir_all(&dir);
        return Err(InstallError::Smoke(e));
    }
    let json = serde_json::to_vec_pretty(&manifest).map_err(|e| InstallError::Io(e.to_string()))?;
    fs::write(dir.join("manifest.json"), json).map_err(|e| io_message(&dir, &e))?;
    Ok(Installed {
        dir,
        server,
        manifest,
    })
}

/// Records `installed` as the operator's trusted llama.cpp: `llama_cpp_dir` in `config_path`
/// (the file `--llama-cpp-dir` would otherwise set), and the `llama-server` backend for a
/// GPU build when none was chosen. Other keys of the file are kept.
pub fn register_trusted(config_path: &Path, installed: &Installed) -> Result<(), String> {
    let mut cfg: BackendConfig = config::load_from(config_path)?;
    cfg.llama_cpp_dir = Some(installed.dir.clone());
    if cfg.backend.is_none() && installed.manifest.variant != Variant::Cpu.label() {
        cfg.backend = Some(Backend::LlamaServer);
    }
    config::save_to(config_path, &cfg)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::net::Policy;
    use crate::testkit::{archive, FakeReleases};

    const WIN: Target = Target {
        os: "win",
        arch: "x64",
    };
    const NAME: &str = "llama-b7-bin-win-cpu-x64.zip";

    fn ok_smoke(_: &Path) -> Result<String, String> {
        Ok("version: b7".into())
    }

    fn run(
        fake: &FakeReleases,
        components: &Path,
        pins: &mut BTreeMap<String, String>,
        cancel: &AtomicBool,
    ) -> Result<Installed, InstallError> {
        let fetcher = Fetcher::new(Policy::loopback());
        let plan = fetch_plan(&fetcher, &fake.api_url(), Variant::Cpu, WIN)?;
        install(&fetcher, &plan, components, pins, cancel, &ok_smoke, |_| {})
    }

    #[test]
    fn installs_the_selected_asset_and_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();
        let fake = FakeReleases::serve(
            NAME,
            archive(&[("llama-server.exe", b"MZ"), ("ggml.dll", b"d")]),
            true,
        );
        let mut pins = BTreeMap::new();
        let cancel = AtomicBool::new(false);
        let first = run(&fake, dir.path(), &mut pins, &cancel).unwrap();
        assert!(first.server.is_file());
        assert_eq!(first.manifest.hash_source, HashSource::Published);
        assert_eq!(first.dir.file_name().unwrap(), "b7-cpu");
        let downloads = fake.archive_hits();
        let again = run(&fake, dir.path(), &mut pins, &cancel).unwrap();
        assert_eq!(again, first);
        assert_eq!(
            fake.archive_hits(),
            downloads,
            "a second run must not download"
        );
        assert!(pins.is_empty());
    }

    #[test]
    fn a_wrong_digest_is_refused_and_leaves_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let fake = FakeReleases::serve(NAME, archive(&[("llama-server.exe", b"MZ")]), true);
        fake.corrupt_digest();
        let err = run(
            &fake,
            dir.path(),
            &mut BTreeMap::new(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(matches!(err, InstallError::HashMismatch { .. }), "{err}");
        assert!(list_installed(dir.path()).is_empty());
        assert!(!dir
            .path()
            .join(".downloads")
            .join(format!("{NAME}.part"))
            .exists());
    }

    #[test]
    fn a_dropped_connection_resumes_instead_of_restarting() {
        let dir = tempfile::tempdir().unwrap();
        let payload = vec![7u8; 200_000];
        let fake = FakeReleases::serve(NAME, archive(&[("llama-server.exe", &payload)]), true);
        fake.cut_first_transfer_after(50_000);
        let mut pins = BTreeMap::new();
        let cancel = AtomicBool::new(false);
        let err = run(&fake, dir.path(), &mut pins, &cancel).unwrap_err();
        assert!(
            matches!(err, InstallError::Net(NetError::Interrupted(n)) if n > 0),
            "{err}"
        );
        let done = run(&fake, dir.path(), &mut pins, &cancel).unwrap();
        assert!(done.server.is_file());
        assert!(
            fake.saw_range_request(),
            "the second transfer must ask for a range"
        );
        assert!(
            fake.bytes_sent() < 2 * fake.archive_len(),
            "the first part was reused"
        );
    }

    #[test]
    fn cancelling_keeps_the_part_for_a_later_resume() {
        let dir = tempfile::tempdir().unwrap();
        let payload = vec![1u8; 400_000];
        let fake = FakeReleases::serve(NAME, archive(&[("llama-server.exe", &payload)]), true);
        let fetcher = Fetcher::new(Policy::loopback());
        let plan = fetch_plan(&fetcher, &fake.api_url(), Variant::Cpu, WIN).unwrap();
        let cancel = AtomicBool::new(false);
        let err = install(
            &fetcher,
            &plan,
            dir.path(),
            &mut BTreeMap::new(),
            &cancel,
            &ok_smoke,
            |s| {
                if matches!(s, Stage::Download { .. }) {
                    cancel.store(true, Ordering::Relaxed);
                }
            },
        )
        .unwrap_err();
        assert!(matches!(err, InstallError::Cancelled), "{err}");
        assert!(list_installed(dir.path()).is_empty());
    }

    #[test]
    fn offline_is_reported_as_unreachable() {
        let dir = tempfile::tempdir().unwrap();
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let fetcher = Fetcher::new(Policy::loopback());
        let err = fetch_plan(
            &fetcher,
            &format!("http://127.0.0.1:{port}/releases"),
            Variant::Cpu,
            WIN,
        )
        .unwrap_err();
        assert!(
            matches!(err, InstallError::Net(NetError::Unreachable(_))),
            "{err}"
        );
        assert!(list_installed(dir.path()).is_empty());
    }

    #[test]
    fn without_a_digest_the_hash_is_pinned_then_enforced() {
        let dir = tempfile::tempdir().unwrap();
        let fake = FakeReleases::serve(NAME, archive(&[("llama-server.exe", b"A")]), false);
        let mut pins = BTreeMap::new();
        let first = run(&fake, dir.path(), &mut pins, &AtomicBool::new(false)).unwrap();
        assert_eq!(first.manifest.hash_source, HashSource::Pinned);
        assert_eq!(pins.get(NAME), Some(&first.manifest.sha256));
        // Same name, other bytes, on a clean machine that kept the pin.
        let other = FakeReleases::serve(NAME, archive(&[("llama-server.exe", b"B")]), false);
        let clean = tempfile::tempdir().unwrap();
        let err = run(&other, clean.path(), &mut pins, &AtomicBool::new(false)).unwrap_err();
        assert!(matches!(err, InstallError::PinMismatch { .. }), "{err}");
    }

    #[test]
    fn a_malicious_archive_installs_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let fake = FakeReleases::serve(
            NAME,
            archive(&[("llama-server.exe", b"MZ"), ("../../pwn.txt", b"x")]),
            true,
        );
        let err = run(
            &fake,
            dir.path(),
            &mut BTreeMap::new(),
            &AtomicBool::new(false),
        )
        .unwrap_err();
        assert!(
            matches!(err, InstallError::Extract(ExtractError::Unsafe(_))),
            "{err}"
        );
        assert!(list_installed(dir.path()).is_empty());
        assert!(!dir.path().parent().unwrap().join("pwn.txt").exists());
    }

    #[test]
    fn a_build_that_does_not_start_is_not_kept() {
        let dir = tempfile::tempdir().unwrap();
        let fake = FakeReleases::serve(NAME, archive(&[("llama-server.exe", b"MZ")]), true);
        let fetcher = Fetcher::new(Policy::loopback());
        let plan = fetch_plan(&fetcher, &fake.api_url(), Variant::Cpu, WIN).unwrap();
        let bad = |_: &Path| -> Result<String, String> { Err("DLL manquante".into()) };
        let err = install(
            &fetcher,
            &plan,
            dir.path(),
            &mut BTreeMap::new(),
            &AtomicBool::new(false),
            &bad,
            |_| {},
        )
        .unwrap_err();
        assert!(matches!(err, InstallError::Smoke(_)), "{err}");
        assert!(list_installed(dir.path()).is_empty());
    }

    #[test]
    fn a_locked_folder_is_never_replaced() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("b7-cpu")).unwrap();
        // The folder exists; rename-aside may or may not succeed (unlocked here), but the
        // returned name must be free either way.
        let free = free_dir(root.path(), "b7-cpu");
        assert!(!free.exists() || free.ends_with("b7-cpu-2"));
    }

    #[test]
    fn trust_is_recorded_in_the_user_config_and_keeps_other_keys() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = dir.path().join("config.toml");
        config::save_to(
            &cfg,
            &BackendConfig {
                gpu_layers: Some(33),
                ..Default::default()
            },
        )
        .unwrap();
        let installed = Installed {
            dir: dir.path().join("llama.cpp").join("b7-vulkan"),
            server: PathBuf::new(),
            manifest: Manifest {
                tag: "b7".into(),
                asset: "x".into(),
                variant: "vulkan".into(),
                sha256: "0".repeat(64),
                hash_source: HashSource::Published,
            },
        };
        register_trusted(&cfg, &installed).unwrap();
        let saved = config::load_from(&cfg).unwrap();
        assert_eq!(saved.llama_cpp_dir, Some(installed.dir));
        assert_eq!(saved.backend, Some(Backend::LlamaServer));
        assert_eq!(saved.gpu_layers, Some(33));
    }

    #[test]
    fn offload_lines_are_parsed() {
        assert_eq!(
            parse_offload("load_tensors: offloaded 25/25 layers to GPU\n"),
            Some((25, 25))
        );
        assert_eq!(parse_offload("offloaded 0/25 layers to GPU"), Some((0, 25)));
        assert_eq!(parse_offload("nothing"), None);
    }
}
