//! The backend settings, in layers, and the persistent user file they can come from.
//!
//! ```text
//! command line flag  >  pipeline.json "backend"  >  user config.toml  >  built-in default
//! ```
//!
//! The user file is a flat TOML subset (`key = "string"` or `key = 123`, `#` comments),
//! read and written here without a TOML crate. It lives in `%APPDATA%\weight-surgeon` on
//! Windows and `$XDG_CONFIG_HOME` (else `~/.config`) elsewhere. A GUI reads and writes the
//! same file, or the same struct as JSON.

use std::path::{Path, PathBuf};

use clap::ValueEnum;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::bench::score::Backend;
use crate::eval::WeightType;

/// When set (and not `0`), the default location of the user file is ignored: the test
/// harness sets it so that no test depends on the machine's own configuration.
pub const NO_USER_CONFIG_ENV: &str = "WEIGHT_SURGEON_NO_USER_CONFIG";
/// A user file at this exact path, whatever the platform (GUI, scripts, tests).
pub const CONFIG_PATH_ENV: &str = "WEIGHT_SURGEON_CONFIG";

/// Which engine generates, and how. Every field is optional: a layer only says what it
/// knows, [`resolve`] stacks the layers.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BackendConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        description = "native (CPU, Rust pur), llama-server (llama.cpp : GPU Vulkan, CUDA, Metal) ou hipfire (AMD HIP, Linux)."
    )]
    pub backend: Option<Backend>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "llama-server : couches déchargées sur le GPU (-ngl) ; 99 : toutes.")]
    pub gpu_layers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "llama-server : requêtes simultanées (slots).")]
    pub batch_size: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Dossier de llama.cpp (llama-server, convertisseur éventuel).")]
    pub llama_cpp_dir: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(description = "Binaire hipfire.")]
    pub hipfire_bin: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(
        description = "Type des matrices du GGUF temporaire : f16 (défaut) ou f32 (sans perte depuis bf16)."
    )]
    pub gguf_dtype: Option<WeightType>,
}

impl BackendConfig {
    /// `self` where it says something, `base` elsewhere.
    pub fn over(self, base: Self) -> Self {
        Self {
            backend: self.backend.or(base.backend),
            gpu_layers: self.gpu_layers.or(base.gpu_layers),
            batch_size: self.batch_size.or(base.batch_size),
            llama_cpp_dir: self.llama_cpp_dir.or(base.llama_cpp_dir),
            hipfire_bin: self.hipfire_bin.or(base.hipfire_bin),
            gguf_dtype: self.gguf_dtype.or(base.gguf_dtype),
        }
    }

    /// What a pipeline job records: `None` for the CPU backend, so that a job written before
    /// backends existed still binds to the same configuration.
    pub fn for_job(self) -> Option<Self> {
        (self.backend.unwrap_or(Backend::Native) != Backend::Native).then_some(self)
    }
}

/// `top` (flags, pipeline.json) over the user file `user`. The GPU knobs of the user file
/// (`gpu_layers`, `batch_size`) only apply when the resulting backend is a GPU one: they
/// must not change how the native engine batches.
pub fn resolve(top: BackendConfig, user: &BackendConfig) -> BackendConfig {
    let backend = top.backend.or(user.backend);
    let gpu = matches!(backend, Some(Backend::LlamaServer | Backend::Hipfire));
    let user = BackendConfig {
        gpu_layers: user.gpu_layers.filter(|_| gpu),
        batch_size: user.batch_size.filter(|_| gpu),
        ..user.clone()
    };
    top.over(user)
}

// ------------------------------------------------------------------ user file

/// Where the user file is: `$WEIGHT_SURGEON_CONFIG`, else the platform's config folder
/// (`None` when disabled by [`NO_USER_CONFIG_ENV`] or when no home folder is known).
pub fn user_config_path() -> Option<PathBuf> {
    let env = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty());
    if let Some(path) = env(CONFIG_PATH_ENV) {
        return Some(PathBuf::from(path));
    }
    if env(NO_USER_CONFIG_ENV).is_some_and(|v| v != "0") {
        return None;
    }
    let base = match cfg!(windows) {
        true => env("APPDATA").map(PathBuf::from),
        false => env("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| env("HOME").map(|h| PathBuf::from(h).join(".config"))),
    };
    base.map(|b| b.join("weight-surgeon").join("config.toml"))
}

/// The user's configuration: empty when there is none (or it is disabled).
pub fn load_user_config() -> Result<BackendConfig, String> {
    match user_config_path() {
        Some(path) => load_from(&path),
        None => Ok(BackendConfig::default()),
    }
}

/// A user file; absent, it is the empty configuration.
pub fn load_from(path: &Path) -> Result<BackendConfig, String> {
    match std::fs::read_to_string(path) {
        Ok(text) => parse_toml(&text).map_err(|e| format!("{} : {e}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(BackendConfig::default()),
        Err(e) => Err(format!("{} : {e}", path.display())),
    }
}

/// Writes the user file, creating its folder.
pub fn save_to(path: &Path, config: &BackendConfig) -> Result<(), String> {
    let fail = |e: std::io::Error| format!("{} : {e}", path.display());
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(fail)?;
    }
    std::fs::write(path, to_toml(config)).map_err(fail)
}

enum Value {
    Text(String),
    Number(u64),
}

/// `"..."` (escapes \\ \" \n \t) or `'...'` (verbatim), then an optional `# comment`; else a
/// bare unsigned integer.
fn parse_value(raw: &str) -> Result<Value, String> {
    let raw = raw.trim();
    let Some(quote) = raw.chars().next().filter(|c| matches!(c, '"' | '\'')) else {
        let digits = raw.split('#').next().unwrap_or_default().trim();
        return digits
            .parse()
            .map(Value::Number)
            .map_err(|_| format!("valeur « {raw} » : texte entre guillemets ou entier attendu"));
    };
    let mut text = String::new();
    let mut chars = raw[1..].chars();
    while let Some(c) = chars.next() {
        match (c, quote) {
            (c, q) if c == q => {
                let rest = chars.as_str().trim();
                return match rest.is_empty() || rest.starts_with('#') {
                    true => Ok(Value::Text(text)),
                    false => Err(format!("texte après la valeur : « {rest} »")),
                };
            }
            ('\\', '"') => text.push(match chars.next() {
                Some('n') => '\n',
                Some('t') => '\t',
                Some(e @ ('\\' | '"')) => e,
                other => return Err(format!("échappement \\{} inconnu", other.unwrap_or(' '))),
            }),
            (c, _) => text.push(c),
        }
    }
    Err("guillemet fermant absent".into())
}

pub fn parse_toml(text: &str) -> Result<BackendConfig, String> {
    let mut config = BackendConfig::default();
    for (index, line) in text.trim_start_matches('\u{feff}').lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let at = |e: String| format!("ligne {} : {e}", index + 1);
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| at("« clé = valeur » attendu".into()))?;
        let value = parse_value(value).map_err(at)?;
        let text = |v: Value| match v {
            Value::Text(t) => Ok(t),
            Value::Number(_) => Err(at(format!("{} attend un texte", key.trim()))),
        };
        let number = |v: Value| match v {
            Value::Number(n) => Ok(n),
            Value::Text(_) => Err(at(format!("{} attend un entier", key.trim()))),
        };
        match key.trim() {
            "default_backend" | "backend" => {
                let name = text(value)?;
                config.backend = Some(
                    Backend::from_str(&name, true)
                        .map_err(|_| at(format!("backend « {name} » inconnu")))?,
                );
            }
            "gguf_dtype" => {
                let name = text(value)?;
                config.gguf_dtype = Some(
                    WeightType::from_str(&name, true)
                        .map_err(|_| at(format!("gguf_dtype « {name} » : f16 ou f32")))?,
                );
            }
            "llama_cpp_dir" => config.llama_cpp_dir = Some(text(value)?.into()),
            "hipfire_bin" => config.hipfire_bin = Some(text(value)?.into()),
            "gpu_layers" => {
                config.gpu_layers = Some(
                    u32::try_from(number(value)?)
                        .map_err(|_| at("gpu_layers trop grand".into()))?,
                )
            }
            "batch_size" => {
                config.batch_size = Some(
                    usize::try_from(number(value)?)
                        .map_err(|_| at("batch_size trop grand".into()))?,
                )
            }
            other => return Err(at(format!("clé inconnue « {other} »"))),
        }
    }
    Ok(config)
}

fn quoted(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

pub fn to_toml(config: &BackendConfig) -> String {
    let mut out =
        String::from("# weight-surgeon: backend d'inférence (voir `surgeon backend show`)\n");
    let mut put = |key: &str, value: String| out.push_str(&format!("{key} = {value}\n"));
    if let Some(b) = config.backend {
        put("default_backend", quoted(b.label()));
    }
    if let Some(dir) = &config.llama_cpp_dir {
        put("llama_cpp_dir", quoted(&dir.to_string_lossy()));
    }
    if let Some(bin) = &config.hipfire_bin {
        put("hipfire_bin", quoted(&bin.to_string_lossy()));
    }
    if let Some(n) = config.gpu_layers {
        put("gpu_layers", n.to_string());
    }
    if let Some(n) = config.batch_size {
        put("batch_size", n.to_string());
    }
    if let Some(d) = config.gguf_dtype {
        put("gguf_dtype", quoted(&format!("{d:?}").to_lowercase()));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full() -> BackendConfig {
        BackendConfig {
            backend: Some(Backend::LlamaServer),
            gpu_layers: Some(99),
            batch_size: Some(6),
            llama_cpp_dir: Some(r"C:\Users\kiwwy\tools\llama-vulkan".into()),
            hipfire_bin: Some("/opt/hipfire/hipfire".into()),
            gguf_dtype: Some(WeightType::F32),
        }
    }

    #[test]
    fn toml_round_trip_keeps_windows_paths() {
        let text = to_toml(&full());
        assert!(text.contains(r#"llama_cpp_dir = "C:\\Users\\kiwwy\\tools\\llama-vulkan""#));
        assert_eq!(parse_toml(&text).unwrap(), full());
    }

    #[test]
    fn json_round_trip_for_a_gui() {
        let json = serde_json::to_string(&full()).unwrap();
        assert!(json.contains(r#""backend":"llama-server""#), "{json}");
        assert_eq!(
            serde_json::from_str::<BackendConfig>(&json).unwrap(),
            full()
        );
        // Unset fields are absent, and an unknown one is an error.
        assert_eq!(
            serde_json::to_string(&BackendConfig::default()).unwrap(),
            "{}"
        );
        assert!(serde_json::from_str::<BackendConfig>(r#"{"backnd":"native"}"#).is_err());
    }

    #[test]
    fn the_three_backends_parse_from_a_file() {
        for (name, backend) in [
            ("native", Backend::Native),
            ("llama-server", Backend::LlamaServer),
            ("hipfire", Backend::Hipfire),
        ] {
            let cfg = parse_toml(&format!("default_backend = \"{name}\"")).unwrap();
            assert_eq!(cfg.backend, Some(backend));
        }
    }

    #[test]
    fn toml_subset_comments_literals_and_errors() {
        let cfg = parse_toml(
            "\u{feff}# comment\n\nllama_cpp_dir = 'C:\\llama'   # verbatim\ngpu_layers = 99 # all\n",
        )
        .unwrap();
        assert_eq!(cfg.llama_cpp_dir, Some(PathBuf::from(r"C:\llama")));
        assert_eq!(cfg.gpu_layers, Some(99));
        for (bad, why) in [
            ("default_backend = \"cuda\"", "backend « cuda » inconnu"),
            ("gpu_layers = \"many\"", "attend un entier"),
            ("llama_cpp_dir = 3", "attend un texte"),
            ("nope = 1", "clé inconnue"),
            ("[section]", "« clé = valeur »"),
            ("llama_cpp_dir = \"open", "guillemet fermant"),
        ] {
            let err = parse_toml(bad).unwrap_err();
            assert!(
                err.starts_with("ligne 1") && err.contains(why),
                "{bad}: {err}"
            );
        }
    }

    #[test]
    fn layers_stack_and_gpu_knobs_do_not_leak_into_the_cpu_backend() {
        let user = full();
        let flag = BackendConfig {
            gpu_layers: Some(10),
            ..Default::default()
        };
        // A flag beats the file; the rest comes from the file.
        let eff = resolve(flag.clone(), &user);
        assert_eq!(
            (eff.backend, eff.gpu_layers, eff.batch_size),
            (Some(Backend::LlamaServer), Some(10), Some(6))
        );
        // The user's GPU knobs vanish under an explicit native backend, a flag's stay.
        let native = BackendConfig {
            backend: Some(Backend::Native),
            ..flag
        };
        let eff = resolve(native, &user);
        assert_eq!((eff.gpu_layers, eff.batch_size), (Some(10), None));
        assert_eq!(eff.llama_cpp_dir, user.llama_cpp_dir);
    }

    #[test]
    fn a_job_records_only_a_gpu_backend() {
        assert_eq!(BackendConfig::default().for_job(), None);
        let native = BackendConfig {
            backend: Some(Backend::Native),
            ..Default::default()
        };
        assert_eq!(native.for_job(), None);
        assert!(full().for_job().is_some());
    }

    #[test]
    fn save_and_load_through_the_file_system() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("config.toml");
        assert_eq!(load_from(&path).unwrap(), BackendConfig::default());
        save_to(&path, &full()).unwrap();
        assert_eq!(load_from(&path).unwrap(), full());
        std::fs::write(&path, "gpu_layers = x").unwrap();
        assert!(load_from(&path).unwrap_err().contains("config.toml"));
    }
}
