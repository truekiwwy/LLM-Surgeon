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
//! `/v1/completions` keeps the bench's raw prompt. With the chat endpoint only, the turns of
//! the rendered prompt go as messages (`extract::chat_messages`), so hipfire's own template is
//! the only one, and decoding is greedy since no seed is documented: the run goes with a
//! warning (`surgeon backend show`, bench warnings), and sampling is refused
//! ([`check_sampling`]). The `model` field of a request is
//! the name of the model given to the bench (its folder, or its file without extension): the
//! daemon must know a model by that name.
//!
//! The daemon answers from the quantized file it registered under that name and never reads
//! the files it is given, GGUF included: a patched working copy would be measured with the
//! old weights. [`validate_hipfire_target`] is the one rule every front-end applies (bench,
//! pipeline, MCP); [`is_working_copy`] tells a working copy from the original. A copy is
//! measured once [`requantize`]d: registered as `ws-<copy>-<sha8>`, recorded in its
//! [`ALIAS_FILE`], and served under that alias while its SafeTensors hash to the same digest.
//!
//! ```text
//! original                         -> its own name
//! copy, alias file, same SHA-256   -> ws-<copy>-<sha8>
//! copy, no alias or other SHA-256  -> hipfire_requantize = auto (bench, pipeline): requantize
//!                                     --allow-stale-hipfire: its own name, with a warning
//!                                     else: refused
//! ```

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::config::{BackendConfig, Requantize};
use super::detect::HipfireDetection;
use crate::eval::server::{serves, Dialect};
use crate::eval::{find_binary, EvalError};

/// The daemon's documented port.
pub const DEFAULT_PORT: u16 = 11435;
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const ALTERNATIVE: &str = "--backend mistralrs ou native";
pub const CHAT_ONLY: &str = "hipfire n'expose que /v1/chat/completions : les tours du gabarit du \
    bench lui sont envoyés comme messages et c'est son propre gabarit qui les écrit, en décodage \
    glouton (aucune graine documentée) ; les scores restent à comparer sur ce seul backend";

/// hipfire generates greedily only: with no documented seed, a sampled run could not be
/// replayed, and its samples per prompt would all be the same greedy answer.
pub fn check_sampling(sampling: &crate::eval::sampling::SamplingParams) -> Result<(), String> {
    match sampling.is_greedy() {
        true => Ok(()),
        false => Err(
            "hipfire : décodage glouton seulement (aucune graine documentée), \
                      retirez --temperature ou mesurez sur --backend native ou mistralrs"
                .into(),
        ),
    }
}

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HipfireGuardDecision {
    /// The `model` field of the requests: the model's name, or the alias its requantized
    /// working copy is registered under.
    pub model: String,
    /// A working copy let through by `--allow-stale-hipfire`: for the report and the logs.
    pub warning: Option<String>,
}

/// Whether hipfire may measure `model`, and under which name. The original is served from
/// the daemon's own quantized copy of it. A working copy (`is_working_copy`) is served under
/// the alias of its [`ALIAS_FILE`] when its SafeTensors still hash to the recorded digest
/// ([`registered_alias`]); else it is refused, since the daemon would answer with other
/// weights, unless `allow_stale`: it then passes under its own name, with a warning.
pub fn validate_hipfire_target(
    model: &Path,
    is_working_copy: bool,
    allow_stale: bool,
) -> Result<HipfireGuardDecision, String> {
    let served = |model: String, warning| Ok(HipfireGuardDecision { model, warning });
    if !is_working_copy {
        return served(model_name(model), None);
    }
    let stale = match registered_alias(model) {
        Ok(record) => return served(record.alias, None),
        Err(why) => why,
    };
    let reason = format!(
        "{} est une copie de travail ({stale}) : hipfire sert le cache quantifié enregistré \
         sous « {} », pas ces poids",
        model.display(),
        model_name(model)
    );
    match allow_stale {
        true => served(
            model_name(model),
            Some(format!(
                "--allow-stale-hipfire : {reason} ; la mesure ne vaut que si cette copie a été \
                 requantifiée et enregistrée sous ce nom"
            )),
        ),
        false => Err(format!(
            "{reason}. Requantifiez-la avant la mesure (--hipfire-requantize, ou \
             hipfire_requantize = \"auto\" dans config.toml), mesurez-la avec --backend native \
             ou mistralrs, ou passez --allow-stale-hipfire"
        )),
    }
}

/// [`validate_hipfire_target`] for a bench or a pipeline about to measure `model`: with
/// `hipfire_requantize = "auto"`, a working copy with no alias for its current weights is
/// [`requantize`]d first, then served under its new alias.
pub fn prepare_target(
    model: &Path,
    config: &BackendConfig,
    allow_stale: bool,
) -> Result<HipfireGuardDecision, String> {
    let copy = is_working_copy(model);
    if copy && config.hipfire_requantize == Some(Requantize::Auto) {
        let record = match registered_alias(model) {
            Ok(record) => record,
            Err(why) => {
                log(&format!("{} : {why} ; requantification", model.display()));
                requantize(model, config)?
            }
        };
        return Ok(HipfireGuardDecision {
            model: record.alias,
            warning: None,
        });
    }
    validate_hipfire_target(model, copy, allow_stale)
}

// ------------------------------------------------------------------ requantized copies

/// What [`requantize`] leaves in the folder of a working copy.
pub const ALIAS_FILE: &str = ".hipfire-alias.json";
/// The format copies are quantized to: MQ4 changed answers of Qwen2.5-0.5B on this setup
/// (17 + 25 gave 32), which would hide what the patch did.
const QUANT_FORMAT: &str = "q8f16";
/// The distribution hipfire runs in on Windows (the one surgeon-gui installs).
pub const WSL_DISTRO: &str = "Ubuntu-24.04";

pub fn wsl_vm_error(output: &str) -> Option<&'static str> {
    [
        "HCS_E_SERVICE_NOT_AVAILABLE",
        "WSL_E_WSL2_NEEDED",
        "CreateVm",
    ]
    .into_iter()
    .find(|code| output.contains(code))
}

/// The content of [`ALIAS_FILE`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AliasRecord {
    /// [`alias_for`] the weights below.
    pub alias: String,
    /// [`weights_sha256`] of the copy when it was quantized.
    pub weights_sha256: String,
    /// RFC 3339, UTC.
    pub created_at: String,
}

/// `ws-<copy>-<sha8>`: the copy's folder name (anything but ASCII letters, digits, `_` and
/// `-` as `-`, so the alias is a bare TOML key and a literal shell word) and the first 8 hex
/// digits of its weights' digest. The same weights give the same alias.
pub fn alias_for(copy: &str, weights_sha256: &str) -> String {
    let name: String = copy
        .chars()
        .map(
            |c| match c.is_ascii_alphanumeric() || matches!(c, '_' | '-') {
                true => c,
                false => '-',
            },
        )
        .collect();
    let sha8 = weights_sha256.get(..8).unwrap_or(weights_sha256);
    format!("ws-{name}-{sha8}")
}

/// The SafeTensors folder of `model` (the folder, or the folder of one of its files),
/// absolute. `hipfire quantize` no longer takes a GGUF.
fn weights_dir(model: &Path) -> Result<PathBuf, String> {
    let model = std::path::absolute(model).map_err(|e| format!("{} : {e}", model.display()))?;
    if model.is_dir() {
        return Ok(model);
    }
    let safetensors = model
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    match (safetensors, model.parent()) {
        (true, Some(dir)) => Ok(dir.to_path_buf()),
        _ => Err(format!(
            "{} n'est pas un dossier SafeTensors, seul format que hipfire requantifie",
            model.display()
        )),
    }
}

/// SHA-256 of the `*.safetensors` files of `dir` (names and digests, in name order): the
/// weights a requantization reads. The backups `apply` leaves beside them do not count.
pub fn weights_sha256(dir: &Path) -> Result<String, String> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{} : {e}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
        })
        .collect();
    if files.is_empty() {
        return Err(format!("{} : aucun fichier .safetensors", dir.display()));
    }
    files.sort();
    let mut manifest = String::new();
    for file in &files {
        let digest = crate::bench::sha256_file(file).map_err(|e| e.to_string())?;
        let name = file.file_name().unwrap_or_default().to_string_lossy();
        manifest.push_str(&format!("{name}\0{digest}\n"));
    }
    Ok(crate::bench::sha256_hex(manifest.as_bytes()))
}

/// The record of a working copy whose SafeTensors still hash to the digest it recorded, else
/// why it cannot be served by alias (none, unreadable, weights changed since, a GGUF).
pub fn registered_alias(model: &Path) -> Result<AliasRecord, String> {
    let dir = weights_dir(model)?;
    let path = dir.join(ALIAS_FILE);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            return Err(format!("aucun alias hipfire : {ALIAS_FILE} absent"))
        }
        Err(e) => return Err(format!("{} : {e}", path.display())),
    };
    let record: AliasRecord =
        serde_json::from_str(&text).map_err(|e| format!("{} illisible : {e}", path.display()))?;
    let current = weights_sha256(&dir)?;
    match current == record.weights_sha256 {
        true => Ok(record),
        false => Err(format!(
            "poids modifiés depuis l'alias {} : SHA-256 {} enregistré, {} actuel",
            record.alias, record.weights_sha256, current
        )),
    }
}

/// Registers the SafeTensors working copy `model` in hipfire under [`alias_for`] its
/// weights, then writes its [`ALIAS_FILE`]. hipfire's output goes to stderr. On Windows
/// hipfire runs in WSL ([`WSL_DISTRO`]) as the distribution's default user: its catalog must
/// be the one the daemon reads (`start-hipfire.ps1` starts it as that user).
pub fn requantize(model: &Path, config: &BackendConfig) -> Result<AliasRecord, String> {
    let dir = weights_dir(model)?;
    let sha = weights_sha256(&dir)?;
    let name = dir.file_name().unwrap_or_default().to_string_lossy();
    let alias = alias_for(&name, &sha);
    log(&format!("{} : SHA-256 {sha}, alias {alias}", dir.display()));
    let hipfire = match (cfg!(windows), binary(config)) {
        (false, Some(bin)) => shell_quote(&bin.to_string_lossy()),
        _ => "hipfire".into(),
    };
    let script = quantize_script(&alias, &source_word(&dir, cfg!(windows)), &hipfire);
    let status = shell(&script)
        .stdin(std::process::Stdio::null())
        .stdout(std::io::stderr())
        .status()
        .map_err(|e| format!("hipfire quantize : lancement impossible : {e}"))?;
    if !status.success() {
        return Err(format!(
            "hipfire quantize a échoué ({status}) : alias {alias} non enregistré"
        ));
    }
    if weights_sha256(&dir)? != sha {
        return Err(format!(
            "{} : poids modifiés pendant la quantification, alias {alias} écarté",
            dir.display()
        ));
    }
    let record = AliasRecord {
        alias,
        weights_sha256: sha,
        created_at: crate::pipeline::job::now_rfc3339(),
    };
    let path = dir.join(ALIAS_FILE);
    let json = serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?;
    std::fs::write(&path, json + "\n").map_err(|e| format!("{} : {e}", path.display()))?;
    log(&format!(
        "{} écrit : servie sous {}",
        path.display(),
        record.alias
    ));
    Ok(record)
}

/// The steps of a requantization, on stderr: stdout may be a `--json` envelope.
fn log(line: &str) {
    eprintln!("surgeon hipfire : {line}");
}

/// What registers `alias` for the SafeTensors folder `source` (a shell word) in the hipfire
/// home of the user it runs as (`$HIPFIRE_HOME`, else `~/.hipfire`, as hipfire 0.4.6 resolves
/// it): nothing when the catalog `models.toml` already holds the alias (same weights), else
/// `hipfire quantize` straight into the models folder, so `--install` copies nothing and
/// `--register` writes the catalog. The catalog layout and the models folder are those of
/// hipfire 0.4.6, to check after an upgrade.
fn quantize_script(alias: &str, source: &str, hipfire: &str) -> String {
    format!(
        r#"set -euo pipefail
export PATH="$HOME/.hipfire/bin:$PATH"
home="${{HIPFIRE_HOME:-$HOME/.hipfire}}"
if [ -f "$home/models.toml" ] && grep -Eq '^"?{alias}"? *=' "$home/models.toml"; then
  echo "alias {alias} déjà dans $home/models.toml : pas de requantification"
  exit 0
fi
src={source}
echo "hipfire quantize $src ({QUANT_FORMAT}) -> $home/models, alias {alias}"
{hipfire} quantize "$src" --format {QUANT_FORMAT} --stem {alias} --output-dir "$home/models" --install --register {alias}
"#
    )
}

/// `dir` as a shell word: through `wslpath` when the script runs in WSL.
fn source_word(dir: &Path, wsl: bool) -> String {
    let quoted = shell_quote(&dir.to_string_lossy());
    match wsl {
        true => format!("\"$(wslpath -a {quoted})\""),
        false => quoted,
    }
}

fn shell_quote(text: &str) -> String {
    format!("'{}'", text.replace('\'', r"'\''"))
}

/// `bash -lc <script>`, in WSL on Windows. `-e` hands the script to bash as is: `--` would
/// run it through another shell first, which strips its quotes.
fn shell(script: &str) -> std::process::Command {
    let (program, prefix): (&str, &[&str]) = match cfg!(windows) {
        true => ("wsl.exe", &["-d", WSL_DISTRO, "-e", "bash"]),
        false => ("bash", &[]),
    };
    let mut cmd = std::process::Command::new(program);
    cmd.args(prefix).args(["-lc", script]).env("WSL_UTF8", "1");
    cmd
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
    fn a_wsl_vm_that_cannot_start_is_recognised() {
        let hcs = "Le service de calcul hôte n'est pas disponible.\n\
                   Code d'erreur : Wsl/Service/CreateInstance/CreateVm/HCS/HCS_E_SERVICE_NOT_AVAILABLE";
        assert_eq!(wsl_vm_error(hcs), Some("HCS_E_SERVICE_NOT_AVAILABLE"));
        assert_eq!(
            wsl_vm_error("Error code: Wsl/WSL_E_WSL2_NEEDED"),
            Some("WSL_E_WSL2_NEEDED")
        );
        assert_eq!(
            wsl_vm_error("Wsl/Service/CreateInstance/CreateVm/E_FAIL"),
            Some("CreateVm")
        );
        assert_eq!(wsl_vm_error("Wsl/Service/WSL_E_DISTRO_NOT_FOUND"), None);
    }

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
        assert_eq!(original, Ok(served("source")));
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
            Ok(served("c1")),
            "nothing to warn about on the original"
        );
    }

    fn served(model: &str) -> HipfireGuardDecision {
        HipfireGuardDecision {
            model: model.into(),
            warning: None,
        }
    }

    /// Records `copy` as requantized under its alias, as [`requantize`] does.
    fn record(copy: &Path) -> AliasRecord {
        let sha = weights_sha256(copy).unwrap();
        let record = AliasRecord {
            alias: alias_for("c1", &sha),
            weights_sha256: sha,
            created_at: "2026-11-20T10:00:00Z".into(),
        };
        let json = serde_json::to_string(&record).unwrap();
        std::fs::write(copy.join(ALIAS_FILE), json).unwrap();
        record
    }

    #[test]
    fn the_alias_is_the_copy_name_and_the_first_digits_of_its_weights() {
        let sha = "0a1b2c3d4e5f".repeat(5) + "6789";
        assert_eq!(alias_for("c1", &sha), "ws-c1-0a1b2c3d");
        assert_eq!(
            alias_for("Qwen2.5 copie é", &sha),
            "ws-Qwen2-5-copie---0a1b2c3d",
            "a bare TOML key and a literal shell word"
        );

        let dir = tempfile::tempdir().unwrap();
        let copy = dir.path();
        std::fs::write(copy.join("model-00001.safetensors"), b"a").unwrap();
        std::fs::write(copy.join("model-00002.safetensors"), b"b").unwrap();
        let first = weights_sha256(copy).unwrap();
        assert_eq!(first.len(), 64);
        // Backups, the alias file and the other files are not the weights.
        std::fs::write(copy.join("model-00001.safetensors.bak-1700000000-1"), b"x").unwrap();
        std::fs::write(copy.join(ALIAS_FILE), b"{}").unwrap();
        std::fs::write(copy.join("config.json"), b"{}").unwrap();
        assert_eq!(
            weights_sha256(copy).unwrap(),
            first,
            "same weights, same digest"
        );
        // The content, and which shard holds it, both count.
        std::fs::write(copy.join("model-00002.safetensors"), b"c").unwrap();
        assert_ne!(weights_sha256(copy).unwrap(), first);
        std::fs::write(copy.join("model-00001.safetensors"), b"b").unwrap();
        std::fs::write(copy.join("model-00002.safetensors"), b"a").unwrap();
        assert_ne!(weights_sha256(copy).unwrap(), first);

        let empty = tempfile::tempdir().unwrap();
        assert!(weights_sha256(empty.path())
            .unwrap_err()
            .contains("aucun fichier .safetensors"));
    }

    #[test]
    fn a_requantized_copy_is_served_by_its_alias_until_a_shard_changes() {
        let dir = tempfile::tempdir().unwrap();
        let (_, copy, gguf) = studio(dir.path());
        let alias = record(&copy).alias;
        assert!(alias.starts_with("ws-c1-") && alias.len() == "ws-c1-".len() + 8);
        assert_eq!(
            validate_hipfire_target(&copy, true, false),
            Ok(served(&alias))
        );
        // Given by one of its files, the same folder.
        let weights = copy.join("model.safetensors");
        assert_eq!(
            validate_hipfire_target(&weights, true, false),
            Ok(served(&alias))
        );

        // One shard rewritten after the quantization: refused at once, the digests shown.
        std::fs::write(&weights, b"patched again").unwrap();
        let err = validate_hipfire_target(&copy, true, false).unwrap_err();
        assert!(
            err.contains(&format!("poids modifiés depuis l'alias {alias}")),
            "{err}"
        );
        assert!(err.contains("--hipfire-requantize"), "the way out: {err}");
        // --allow-stale-hipfire still lets it through, under its own name.
        let stale = validate_hipfire_target(&copy, true, true).unwrap();
        assert_eq!(stale.model, "c1");
        assert!(stale.warning.unwrap().contains("poids modifiés"));

        // No alias file, or an unreadable one.
        std::fs::remove_file(copy.join(ALIAS_FILE)).unwrap();
        let err = validate_hipfire_target(&copy, true, false).unwrap_err();
        assert!(err.contains("aucun alias hipfire"), "{err}");
        std::fs::write(copy.join(ALIAS_FILE), b"{").unwrap();
        let err = validate_hipfire_target(&copy, true, false).unwrap_err();
        assert!(err.contains("illisible"), "{err}");
        // A GGUF copy is never requantized: refused, whatever lies beside it.
        std::fs::write(gguf.with_file_name(ALIAS_FILE), b"{}").unwrap();
        let err = validate_hipfire_target(&gguf, true, false).unwrap_err();
        assert!(err.contains("dossier SafeTensors"), "{err}");
    }

    #[test]
    fn requantization_is_opt_in_and_skipped_for_a_valid_alias() {
        let dir = tempfile::tempdir().unwrap();
        let (source, copy, _) = studio(dir.path());
        // `off` (or unset): the guard as is, nothing runs.
        for mode in [None, Some(Requantize::Off)] {
            let config = BackendConfig {
                hipfire_requantize: mode,
                ..Default::default()
            };
            let err = prepare_target(&copy, &config, false).unwrap_err();
            assert!(err.contains("aucun alias hipfire"), "{err}");
            assert_eq!(
                prepare_target(&source, &config, false),
                Ok(served("source"))
            );
        }
        // `auto` with an alias for these weights: served at once, hipfire never called (the
        // binary does not exist).
        let alias = record(&copy).alias;
        let auto = BackendConfig {
            hipfire_requantize: Some(Requantize::Auto),
            hipfire_bin: Some(dir.path().join("no-hipfire")),
            ..Default::default()
        };
        assert_eq!(prepare_target(&copy, &auto, false), Ok(served(&alias)));
        assert_eq!(prepare_target(&source, &auto, false), Ok(served("source")));
    }

    #[test]
    fn the_quantize_script_registers_the_alias_once_and_reaches_the_copy() {
        let script = quantize_script("ws-c1-0a1b2c3d", "'/m/c1'", "hipfire");
        assert!(script.starts_with("set -euo pipefail"), "{script}");
        assert!(
            script.contains(r#"grep -Eq '^"?ws-c1-0a1b2c3d"? *=' "$home/models.toml""#),
            "an alias already in the catalog is not quantized again: {script}"
        );
        assert!(script.contains(
            "hipfire quantize \"$src\" --format q8f16 --stem ws-c1-0a1b2c3d \
             --output-dir \"$home/models\" --install --register ws-c1-0a1b2c3d"
        ));
        assert!(script.contains("home=\"${HIPFIRE_HOME:-$HOME/.hipfire}\""));
        // Windows paths go through wslpath, quotes kept literal.
        let windows = Path::new(r"C:\models\it's a copy");
        assert_eq!(
            source_word(windows, true),
            r#""$(wslpath -a 'C:\models\it'\''s a copy')""#
        );
        assert_eq!(source_word(Path::new("/m/c1"), false), "'/m/c1'");
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
    fn only_greedy_decoding_runs_on_hipfire() {
        use crate::eval::sampling::SamplingParams;
        assert_eq!(check_sampling(&SamplingParams::default()), Ok(()));
        let drawn = SamplingParams {
            temperature: 0.7,
            ..SamplingParams::default()
        };
        assert!(check_sampling(&drawn).unwrap_err().contains("glouton"));
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
