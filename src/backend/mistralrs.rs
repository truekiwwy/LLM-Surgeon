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
        let mut cmd = crate::process::command(binary);
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

/// Whether this build runs mistral.rs on a GPU in its own process (`--features cuda|metal`).
pub const EMBEDDED_GPU: bool = cfg!(any(feature = "cuda", feature = "metal"));

/// Whether this build holds mistral.rs at all (`--features mistralrs`, on the CPU without
/// `cuda` or `metal`).
pub const EMBEDDED: bool = cfg!(feature = "mistralrs");

/// Whether `--backend mistralrs` can run here: built in, or a binary found.
pub fn available(config: &BackendConfig) -> bool {
    EMBEDDED || locate(config).is_some()
}

/// mistral.rs inside this process, on the GPU of its build: the SafeTensors folder given to
/// the bench is loaded as is, from that exact path. The SDK documents no seed, so decoding is
/// greedy only, like hipfire.
#[cfg(feature = "mistralrs")]
pub mod embedded {
    use std::ops::ControlFlow;
    use std::path::Path;
    use std::time::Instant;

    use mistralrs::blocking::BlockingModel;
    use mistralrs::{
        NormalRequest, Request, RequestMessage, Response, SamplingParams, TextModelBuilder,
    };

    use crate::eval::{Decoding, EvalError, GenerationOutcome, Result, TextGenerator};

    pub struct EmbeddedGenerator {
        model: BlockingModel,
    }

    fn failed(what: &str, e: impl std::fmt::Display) -> EvalError {
        EvalError::InvalidConfig(format!("mistral.rs embarqué : {what} : {e}"))
    }

    impl EmbeddedGenerator {
        /// Loads the SafeTensors folder `model`. Must not run inside a tokio runtime.
        pub fn load(model: &Path) -> Result<Self> {
            if !model.is_dir() {
                return Err(failed(
                    &model.display().to_string(),
                    "dossier SafeTensors attendu (un GGUF passe par le binaire mistralrs-server)",
                ));
            }
            let path = std::path::absolute(model).map_err(|e| failed("chemin", e))?;
            let model = BlockingModel::from_builder(TextModelBuilder::new(path.to_string_lossy()))
                .map_err(|e| failed("chargement", e))?;
            Ok(Self { model })
        }
    }

    impl TextGenerator for EmbeddedGenerator {
        fn generate(
            &mut self,
            prompt: &str,
            max_new_tokens: u32,
            on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
        ) -> Result<GenerationOutcome> {
            let started = Instant::now();
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            let sampling = SamplingParams {
                max_len: Some(max_new_tokens as usize),
                ..SamplingParams::deterministic()
            };
            let request = NormalRequest::new_simple(
                RequestMessage::Completion {
                    text: prompt.to_owned(),
                    echo_prompt: false,
                    best_of: None,
                },
                sampling,
                tx,
                0,
                None,
                None,
            );
            let sender = self
                .model
                .inner()
                .inner()
                .get_sender(None)
                .map_err(|e| failed("moteur", e))?;
            sender
                .blocking_send(Request::Normal(Box::new(request)))
                .map_err(|e| failed("envoi", e))?;
            let choice = match rx.blocking_recv() {
                Some(Response::CompletionDone(done)) => done.choices.into_iter().next(),
                Some(Response::InternalError(e) | Response::ValidationError(e)) => {
                    return Err(failed("génération", e))
                }
                Some(Response::CompletionModelError(e, _)) => return Err(failed("génération", e)),
                Some(_) => return Err(failed("génération", "réponse inattendue")),
                None => return Err(failed("génération", "canal fermé")),
            };
            let choice = choice.ok_or_else(|| failed("génération", "aucun choix"))?;
            let stopped_early = on_chunk(&choice.text).is_break();
            Ok(GenerationOutcome {
                truncated: choice.finish_reason == "length",
                text: choice.text,
                stopped_early,
                duration_ms: started.elapsed().as_millis(),
            })
        }

        fn configure(&mut self, decoding: &Decoding) -> bool {
            // Sampled runs are refused up front by `generator`: greedy is all there is.
            decoding.sampling.is_greedy()
        }
    }
}

/// The generator of `--backend mistralrs` on `model`: in this process when built for a GPU,
/// else the binary found, else in this process on the CPU (`--features mistralrs` alone).
/// The embedded engine decodes greedily only (its SDK takes no seed).
#[cfg_attr(not(feature = "mistralrs"), allow(unused_variables))]
pub fn generator(
    model: &Path,
    config: &BackendConfig,
    seed: u32,
    opts: &ServerOptions,
    greedy: bool,
    cancel: &AtomicBool,
) -> crate::eval::Result<Box<dyn crate::eval::TextGenerator>> {
    let binary = locate(config);
    #[cfg(feature = "mistralrs")]
    if model.is_dir() && (EMBEDDED_GPU || binary.is_none()) {
        if !greedy {
            return Err(EvalError::InvalidConfig(
                "mistral.rs embarqué : décodage glouton seulement (aucune graine dans son SDK), \
                 retirez --temperature"
                    .into(),
            ));
        }
        return Ok(Box::new(embedded::EmbeddedGenerator::load(model)?));
    }
    match binary {
        Some(binary) => Ok(Box::new(start(&binary, model, seed, opts, cancel)?)),
        None => Err(EvalError::BinaryNotFound {
            name: "mistralrs".into(),
            hint: not_found(config),
        }),
    }
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
