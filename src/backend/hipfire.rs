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
