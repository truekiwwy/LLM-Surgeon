//! hipfire (<https://github.com/warpfront/hipfire>): a Rust inference engine for AMD GPUs on
//! HIP / ROCm, run as a daemon (`hipfire serve`) that the engine talks to and never starts
//! (the desktop app starts its own, for as long as its window is open).
//!
//! What its README documents (read 2026-11-13): binary `hipfire` (`pull`, `run`, `chat`,
//! `serve`, `bench`, `quantize`...); a daemon on `127.0.0.1:11435` with the OpenAI-style
//! `/v1/chat/completions` and `/v1/models`; models named by tag (`qwen3:8b`...) or made with
//! `hipfire quantize`; Linux with ROCm 6, native Windows and WSL2 "untested on GPU
//! hardware". surgeon allows it on every OS and checks what the daemon answers:
//!
//! ```text
//! hipfire_url (loopback only) | hipfire binary found: 127.0.0.1:11435 | neither: error
//! POST /v1/completions {} -> 404/405 -> POST /v1/chat/completions {} -> 404/405 -> error
//! ```
//!
//! `/v1/completions` keeps the bench's raw prompt. With the chat endpoint only, hipfire's own
//! chat template goes on top of the bench's and no seed is documented: the run still goes,
//! with a warning (`surgeon backend show`, bench warnings). The `model` field of a request is
//! the name of the model given to the bench (its folder, or its file without extension): the
//! daemon must know a model by that name.
//!
//! The daemon answers from the quantized file it registered under that name and never reads
//! the files it is given, GGUF included: a patched working copy would be measured with the
//! old weights. [`validate_hipfire_target`] is the one rule every front-end applies (bench,
//! pipeline, MCP); [`is_working_copy`] tells a working copy from the original.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use super::config::BackendConfig;
use super::detect::HipfireDetection;
use crate::eval::server::{serves, Dialect};
use crate::eval::{find_binary, EvalError};

/// The daemon's documented port.
pub const DEFAULT_PORT: u16 = 11435;
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const ALTERNATIVE: &str = "--backend mistralrs ou native";
pub const CHAT_ONLY: &str = "hipfire n'expose que /v1/chat/completions, sans graine documentée : \
    son gabarit de chat s'ajoute à celui du bench, les résultats ne sont ni reproductibles ni \
    comparables aux autres backends";

/// `given` (the binary, or a folder holding it), then `exe_dir`, then the folders of `path`.
fn find(given: Option<&Path>, exe_dir: Option<&Path>, path: Option<OsString>) -> Option<PathBuf> {
    if let Some(file) = given.filter(|p| p.is_file()) {
        return Some(file.to_path_buf());
    }
    let dirs: Vec<PathBuf> = given
        .into_iter()
        .chain(exe_dir)
        .map(Path::to_path_buf)
        .collect();
    find_binary(&["hipfire"], &dirs, path)
}

/// The hipfire binary: `hipfire_bin`, the folder of this executable, then the `PATH`.
fn binary(config: &BackendConfig) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok();
    find(
        config.hipfire_bin.as_deref(),
        exe.as_deref().and_then(Path::parent),
        std::env::var_os("PATH"),
    )
}

/// `http://127.0.0.1:<port>` or `http://localhost:<port>`, a trailing `/` allowed: the
/// client only talks to loopback.
pub fn loopback_port(url: &str) -> Result<u16, String> {
    let bad = || {
        format!("hipfire_url « {url} » : http://127.0.0.1:<port> attendu (boucle locale seulement)")
    };
    let rest = url.trim().strip_prefix("http://").ok_or_else(bad)?;
    let rest = rest.strip_suffix('/').unwrap_or(rest);
    let (host, port) = rest.rsplit_once(':').ok_or_else(bad)?;
    if !matches!(host, "127.0.0.1" | "localhost") {
        return Err(bad());
    }
    port.parse().ok().filter(|p| *p != 0).ok_or_else(bad)
}

/// The daemon's port: `hipfire_url`, else the documented one when hipfire is installed.
pub fn port(config: &BackendConfig) -> Result<u16, String> {
    match &config.hipfire_url {
        Some(url) => loopback_port(url),
        None if binary(config).is_some() => Ok(DEFAULT_PORT),
        None => Err(format!(
            "hipfire introuvable : donnez --hipfire-bin ou --hipfire-url (hipfire_bin / \
             hipfire_url dans la configuration) ou mettez-le dans le PATH. Sinon : \
             {ALTERNATIVE}."
        )),
    }
}

/// A daemon that answers, and how to ask it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Endpoint {
    pub port: u16,
    pub dialect: Dialect,
}

impl Endpoint {
    pub fn warning(&self) -> Option<&'static str> {
        (self.dialect == Dialect::Chat).then_some(CHAT_ONLY)
    }
}

/// The endpoint the daemon serves, the raw completions first.
pub fn probe(config: &BackendConfig) -> Result<Endpoint, String> {
    let port = port(config)?;
    for dialect in [Dialect::Completions, Dialect::Chat] {
        match serves(port, dialect, PROBE_TIMEOUT) {
            Ok(true) => return Ok(Endpoint { port, dialect }),
            Ok(false) => {}
            Err(EvalError::Timeout { .. }) => {
                return Err(format!("hipfire (127.0.0.1:{port}) ne répond pas"))
            }
            Err(_) => {
                return Err(format!(
                    "hipfire injoignable sur 127.0.0.1:{port} : lancez `hipfire serve` (ou \
                     corrigez hipfire_url). Sinon : {ALTERNATIVE}."
                ))
            }
        }
    }
    Err(format!(
        "hipfire (127.0.0.1:{port}) n'expose ni /v1/completions ni /v1/chat/completions. \
         Sinon : {ALTERNATIVE}."
    ))
}

/// The `model` field of the requests: the model's folder name, or its file name without
/// the extension.
pub fn model_name(model: &Path) -> String {
    let name = match model.is_file() {
        true => model.file_stem(),
        false => model.file_name(),
    };
    name.unwrap_or_default().to_string_lossy().into_owned()
}

/// The manifest at the root of a surgeon-gui project (its `copies/`, `runs/`, `artifacts/`).
const PROJECT_FILE: &str = "project.surgeon.json";
/// The binding of a pipeline job directory (its `work/` holds the patched candidate).
const JOB_FILE: &str = "job.json";

/// What [`validate_hipfire_target`] lets through.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HipfireGuardDecision {
    /// A working copy let through by `--allow-stale-hipfire`: for the report and the logs.
    pub warning: Option<String>,
}

/// Whether hipfire may measure `model`. The original is served from the daemon's own
/// quantized copy of it; a working copy (`is_working_copy`, GGUF included) is refused, since
/// the daemon would answer with the weights registered under its name, unless `allow_stale`
/// (the operator requantized and registered it): it then passes with a warning.
pub fn validate_hipfire_target(
    model: &Path,
    is_working_copy: bool,
    allow_stale: bool,
) -> Result<HipfireGuardDecision, String> {
    if !is_working_copy {
        return Ok(HipfireGuardDecision::default());
    }
    let reason = format!(
        "{} est une copie de travail : hipfire sert le cache quantifié enregistré sous « {} », \
         pas ces poids",
        model.display(),
        model_name(model)
    );
    match allow_stale {
        true => Ok(HipfireGuardDecision {
            warning: Some(format!(
                "--allow-stale-hipfire : {reason} ; la mesure ne vaut que si cette copie a été \
                 requantifiée et enregistrée sous ce nom"
            )),
        }),
        false => Err(format!(
            "{reason}. Mesurez-la avec --backend native ou mistralrs, ou requantifiez-la, \
             enregistrez-la dans hipfire et passez --allow-stale-hipfire"
        )),
    }
}

/// Whether `model` holds weights surgery wrote, which hipfire's cache does not have: a path
/// inside a surgeon-gui project that is not the source its manifest records (`copies/`, run
/// outputs), the `work/` of a pipeline job, or a folder beside the backups `surgeon apply`
/// leaves (`<file>.bak-<secs>-<n>`).
pub fn is_working_copy(model: &Path) -> bool {
    let Ok(model) = std::path::absolute(model) else {
        return false;
    };
    let folder = match model.is_dir() {
        true => model.as_path(),
        false => model.parent().unwrap_or(&model),
    };
    let patched = std::fs::read_dir(folder)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| crate::patch::is_backup_name(&e.file_name().to_string_lossy()));
    patched
        || model.ancestors().skip(1).any(|dir| {
            (dir.join(JOB_FILE).is_file() && model.starts_with(dir.join("work")))
                || (dir.join(PROJECT_FILE).is_file() && !is_project_source(dir, &model))
        })
}

/// Whether `model` is the source the manifest of `project` records. An unreadable manifest
/// says no: inside a project, what cannot be told from a copy is treated as one.
fn is_project_source(project: &Path, model: &Path) -> bool {
    let source = std::fs::read_to_string(project.join(PROJECT_FILE))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|m| m["source"]["path"].as_str().map(PathBuf::from));
    let canonical = |p: &Path| std::fs::canonicalize(p).ok();
    source.is_some_and(|s| canonical(&s).is_some() && canonical(&s) == canonical(model))
}

pub fn detect(config: &BackendConfig) -> HipfireDetection {
    let binary = binary(config);
    let probed = probe(config);
    let up = probed.is_ok();
    let note = match probed {
        Ok(e) => match e.warning() {
            Some(w) => format!("démon sur 127.0.0.1:{} ; {w}", e.port),
            None => format!("démon sur 127.0.0.1:{} (/v1/completions)", e.port),
        },
        Err(e) => e,
    };
    HipfireDetection { binary, up, note }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_a_loopback_url_is_accepted() {
        for (url, port) in [
            ("http://127.0.0.1:11435", 11435),
            ("http://localhost:8080/", 8080),
            (" http://127.0.0.1:1 ", 1),
        ] {
            assert_eq!(loopback_port(url), Ok(port), "{url}");
        }
        for url in [
            "https://127.0.0.1:11435",
            "http://10.0.0.2:11435",
            "http://127.0.0.1",
            "http://127.0.0.1:0",
            "http://127.0.0.1:x",
            "http://evil.example:80",
        ] {
            assert!(
                loopback_port(url).unwrap_err().contains("boucle locale"),
                "{url}"
            );
        }
    }

    #[test]
    fn the_url_wins_and_nothing_is_an_actionable_error() {
        let config = BackendConfig {
            hipfire_url: Some("http://127.0.0.1:4321".into()),
            ..Default::default()
        };
        assert_eq!(port(&config), Ok(4321));
        let none = BackendConfig {
            hipfire_bin: Some("/nowhere/hipfire".into()),
            ..Default::default()
        };
        // Unless hipfire is in this machine's PATH.
        if binary(&none).is_none() {
            let err = port(&none).unwrap_err();
            assert!(
                err.contains("introuvable") && err.contains("--backend mistralrs"),
                "{err}"
            );
        }
    }

    #[test]
    fn an_installed_binary_means_the_default_port() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("hipfire");
        std::fs::write(&bin, b"").unwrap();
        let config = BackendConfig {
            hipfire_bin: Some(bin.clone()),
            ..Default::default()
        };
        assert_eq!(binary(&config), Some(bin));
        assert_eq!(port(&config), Ok(DEFAULT_PORT));
    }

    #[test]
    fn the_binary_is_searched_in_the_config_then_beside_the_app_then_on_the_path() {
        let root = tempfile::tempdir().unwrap();
        let name = format!("hipfire{}", std::env::consts::EXE_SUFFIX);
        let [given, exe_dir, on_path, empty] = ["given", "exe", "path", "empty"].map(|d| {
            let dir = root.path().join(d);
            std::fs::create_dir(&dir).unwrap();
            dir
        });
        for dir in [&given, &exe_dir, &on_path] {
            std::fs::write(dir.join(&name), b"").unwrap();
        }
        let path = || Some(on_path.clone().into_os_string());
        let file = given.join(&name);
        assert_eq!(
            find(Some(&file), Some(&exe_dir), path()),
            Some(file.clone())
        );
        assert_eq!(find(Some(&given), Some(&exe_dir), path()), Some(file));
        assert_eq!(
            find(Some(&empty), Some(&exe_dir), path()),
            Some(exe_dir.join(&name)),
            "a hipfire_bin that holds nothing falls through"
        );
        assert_eq!(find(None, Some(&empty), path()), Some(on_path.join(&name)));
        assert_eq!(
            find(None, Some(&empty), None),
            None,
            "absent is not an error"
        );
    }

    #[test]
    fn a_daemon_that_is_not_there_says_how_to_start_it() {
        let closed = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let config = BackendConfig {
            hipfire_url: Some(format!("http://127.0.0.1:{closed}")),
            ..Default::default()
        };
        let err = probe(&config).unwrap_err();
        assert!(err.contains("hipfire serve"), "{err}");
        assert!(detect(&config).note.contains("hipfire serve"));
    }

    /// `<root>/source/` (a SafeTensors model) and a surgeon-gui project naming it, with a
    /// SafeTensors working copy and a GGUF one.
    fn studio(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
        let source = root.join("source");
        std::fs::create_dir_all(&source).unwrap();
        std::fs::write(source.join("model.safetensors"), b"weights").unwrap();
        let project = root.join("projects").join("p");
        std::fs::create_dir_all(project.join("copies").join("c1")).unwrap();
        let manifest = serde_json::json!({"schema_version": 1, "source": {"path": source}});
        std::fs::write(project.join(PROJECT_FILE), manifest.to_string()).unwrap();
        let copy = project.join("copies").join("c1");
        std::fs::write(copy.join("model.safetensors"), b"patched").unwrap();
        let gguf = project.join("copies").join("c2").join("m.gguf");
        std::fs::create_dir_all(gguf.parent().unwrap()).unwrap();
        std::fs::write(&gguf, b"GGUF").unwrap();
        (source, copy, gguf)
    }

    #[test]
    fn the_original_model_is_served_and_a_working_copy_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let (source, copy, gguf) = studio(dir.path());
        let original = validate_hipfire_target(&source, is_working_copy(&source), false);
        assert_eq!(original, Ok(HipfireGuardDecision::default()));
        assert!(!is_working_copy(&source.join("model.safetensors")));

        assert!(is_working_copy(&copy));
        let err = validate_hipfire_target(&copy, is_working_copy(&copy), false).unwrap_err();
        assert!(
            err.contains("copie de travail") && err.contains("« c1 »"),
            "{err}"
        );
        assert!(
            err.contains("--allow-stale-hipfire"),
            "the way out is named: {err}"
        );
        // A GGUF is served by name too: a GGUF copy is refused the same way.
        assert!(is_working_copy(&gguf));
        let err = validate_hipfire_target(&gguf, is_working_copy(&gguf), false).unwrap_err();
        assert!(err.contains("« m »"), "{err}");
    }

    #[test]
    fn an_explicit_override_lets_a_working_copy_through_with_a_warning() {
        let dir = tempfile::tempdir().unwrap();
        let (_, copy, gguf) = studio(dir.path());
        for model in [&copy, &gguf] {
            let decision = validate_hipfire_target(model, true, true).unwrap();
            let warning = decision.warning.unwrap();
            assert!(warning.starts_with("--allow-stale-hipfire"), "{warning}");
        }
        assert_eq!(
            validate_hipfire_target(&copy, false, true),
            Ok(HipfireGuardDecision::default()),
            "nothing to warn about on the original"
        );
    }

    #[test]
    fn working_copies_are_told_by_layout_manifest_and_backups() {
        let dir = tempfile::tempdir().unwrap();
        // A plain folder: the original.
        let plain = dir.path().join("plain");
        std::fs::create_dir_all(&plain).unwrap();
        std::fs::write(plain.join("model.safetensors"), b"w").unwrap();
        assert!(!is_working_copy(&plain));
        // Patched in place by `surgeon apply`: its backups sit beside the weights.
        std::fs::write(plain.join("model.safetensors.bak-1700000000-1"), b"w").unwrap();
        assert!(is_working_copy(&plain));
        assert!(is_working_copy(&plain.join("model.safetensors")));
        // The candidate of a pipeline job, not the job's other files.
        let job = dir.path().join("job");
        std::fs::create_dir_all(job.join("work").join("model")).unwrap();
        std::fs::write(job.join(JOB_FILE), b"{}").unwrap();
        assert!(is_working_copy(&job.join("work").join("model")));
        assert!(!is_working_copy(&job.join("checkpoints")));
        // Inside a project whose manifest cannot be read: treated as a copy.
        let broken = dir.path().join("broken");
        std::fs::create_dir_all(broken.join("runs")).unwrap();
        std::fs::write(broken.join(PROJECT_FILE), b"{").unwrap();
        assert!(is_working_copy(&broken.join("runs")));
    }

    #[test]
    fn the_model_field_is_the_model_name() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("qwen2.5-7b");
        std::fs::create_dir(&folder).unwrap();
        assert_eq!(model_name(&folder), "qwen2.5-7b");
        let file = dir.path().join("q.gguf");
        std::fs::write(&file, b"").unwrap();
        assert_eq!(model_name(&file), "q");
    }
}
