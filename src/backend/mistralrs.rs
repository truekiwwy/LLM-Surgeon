//! mistral.rs (<https://github.com/EricLBuehler/mistral.rs>): a Rust inference server that
//! loads the SafeTensors directory as is (no GGUF conversion) or a GGUF, on the GPU its
//! build targets (CUDA, Metal...). Its docs (read 2026-11-13) give two command lines:
//!
//! ```text
//! mistralrs        [--seed S] serve -p PORT --host 127.0.0.1 --max-seqs N -m DIR
//! mistralrs-server --port PORT --serve-ip 127.0.0.1 --seed S --max-seqs N plain -m DIR
//! a GGUF:          ... serve|gguf ... -m PARENT_DIR -f FILE.gguf
//! ```
//!
//! The form comes from the binary's `--help` (a `serve` subcommand, else `plain` / `gguf`)
//! and each flag from the help of the level that lists it: nothing is guessed. The server is
//! bound to loopback and `--cpu` is never passed. Requests go to `/v1/completions` (the raw
//! prompt: the bench's template stays the only one), seeded and greedy by default.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;

use super::config::BackendConfig;
use crate::eval::server::ServerOptions;
use crate::eval::server::{capture, listed, resolve_flags, Dialect, Launch, ServerGenerator};
use crate::eval::{find_binary, EvalError};

/// A folder holding the mistral.rs binary, like `LLAMA_CPP_DIR` for llama.cpp.
pub const DIR_ENV: &str = "MISTRALRS_DIR";
/// The legacy server first: `mistralrs` is also the name of its interactive CLI.
const NAMES: [&str; 2] = ["mistralrs-server", "mistralrs"];

/// The mistral.rs binary: `mistralrs_bin` when given (and then only it), else
/// `mistralrs-server` or `mistralrs` in `mistralrs_dir`, `$MISTRALRS_DIR`, then the PATH.
pub fn locate(config: &BackendConfig) -> Option<PathBuf> {
    if let Some(bin) = &config.mistralrs_bin {
        return bin.is_file().then(|| bin.clone());
    }
    let mut dirs: Vec<PathBuf> = config.mistralrs_dir.iter().cloned().collect();
    dirs.extend(std::env::var_os(DIR_ENV).map(PathBuf::from));
    // Folders before the PATH whatever the name (find_binary goes name by name).
    find_binary(&NAMES, &dirs, None).or_else(|| find_binary(&NAMES, &[], std::env::var_os("PATH")))
}

pub fn not_found(config: &BackendConfig) -> String {
    match &config.mistralrs_bin {
        Some(bin) => format!(
            "mistralrs_bin « {} » : fichier introuvable. Sinon : --backend llama-server ou native.",
            bin.display()
        ),
        None => format!(
            "mistral.rs introuvable : donnez --mistralrs-bin ou --mistralrs-dir (mistralrs_bin \
             / mistralrs_dir dans la configuration, ou ${DIR_ENV}) ou mettez mistralrs-server \
             dans le PATH. Sinon : --backend llama-server ou native."
        ),
    }
}

/// Starts mistral.rs on `model` (a SafeTensors directory, or a `.gguf` file) and waits until
/// it is healthy; `seed` is its global seed (each request carries one too).
pub fn start(
    binary: &Path,
    model: &Path,
    seed: u32,
    opts: &ServerOptions,
    cancel: &AtomicBool,
) -> crate::eval::Result<ServerGenerator> {
    let top = capture(binary, &["--help"])?;
    let gguf = model
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    let sub = match (listed(&top, "serve"), gguf) {
        (true, _) => "serve",
        (false, true) => "gguf",
        (false, false) => "plain",
    };
    if !listed(&top, sub) {
        return Err(EvalError::InvalidConfig(format!(
            "{} : ni `serve` ni `{sub}` dans son --help (version de mistral.rs non reconnue)",
            binary.display()
        )));
    }
    let local = capture(binary, &[sub, "--help"])?;
    let mut needed = vec!["port", "host", "seed", "max-seqs", "model"];
    needed.extend(gguf.then_some("gguf-file"));
    // A flag the subcommand lists goes after it, a global one before.
    let (mut before, mut after) = (Vec::new(), Vec::new());
    for name in needed {
        match resolve_flags(&local, &[name]) {
            Ok(flag) => after.extend(flag),
            Err(_) => before.extend(resolve_flags(&top, &[name])?),
        }
    }
    let dir = match gguf {
        true => model
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
        false => model,
    };
    let value = |name: &str, port: u16| -> OsString {
        match name {
            "port" => port.to_string().into(),
            "host" => "127.0.0.1".into(),
            "seed" => seed.to_string().into(),
            "max-seqs" => opts.slots.to_string().into(),
            "model" => dir.into(),
            _ => model.file_name().unwrap_or_default().into(),
        }
    };
    let mut command = |port: u16, _api_key: &str| {
        let mut cmd = Command::new(binary);
        for (name, flag) in &before {
            cmd.arg(flag).arg(value(name, port));
        }
        cmd.arg(sub);
        for (name, flag) in &after {
            cmd.arg(flag).arg(value(name, port));
        }
        cmd
    };
    let launch = Launch {
        name: "mistralrs",
        binary,
        dialect: Dialect::Completions,
        // mistral.rs's name for the model it serves when it serves one.
        model: "default".into(),
        command: &mut command,
    };
    ServerGenerator::spawn(launch, opts, cancel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mistralrs_bin_is_the_only_candidate_when_given() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("mistralrs-server");
        let config = BackendConfig {
            mistralrs_bin: Some(bin.clone()),
            mistralrs_dir: Some(dir.path().into()),
            ..Default::default()
        };
        assert_eq!(locate(&config), None, "absent: never another one");
        assert!(not_found(&config).contains("mistralrs_bin"));
        std::fs::write(&bin, b"").unwrap();
        assert_eq!(locate(&config), Some(bin));
    }

    #[test]
    fn a_folder_is_searched_for_both_names() {
        let dir = tempfile::tempdir().unwrap();
        let name = format!("mistralrs{}", std::env::consts::EXE_SUFFIX);
        std::fs::write(dir.path().join(&name), b"").unwrap();
        let config = BackendConfig {
            mistralrs_dir: Some(dir.path().into()),
            ..Default::default()
        };
        assert_eq!(locate(&config), Some(dir.path().join(name)));
    }
}
