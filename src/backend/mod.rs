//! Which engine generates the answers of `bench`, `pipeline` and the MCP tools, and the
//! persistent user configuration that remembers the choice (docs/BACKENDS.md).
//!
//! The engines already share one interface, [`TextGenerator`](crate::eval::TextGenerator):
//! `native` (CPU, pure Rust), `llama-server` (llama.cpp: Vulkan, CUDA, Metal...) and
//! `hipfire` (AMD HIP, Linux, optional). This module adds the common settings
//! ([`BackendConfig`]), their layering, the machine probe and the report labels.
//!
//! ```text
//! flags > pipeline.json > user config.toml > default        (config::resolve)
//! apply_user_config: fills the optional flags of the bench commands from the user file
//! detect_hardware_and_backends: llama-server + --list-devices + hipfire -> recommendation
//! ```

pub mod config;
pub mod detect;
pub mod hipfire;

pub use config::{
    load_from, load_user_config, resolve, save_to, user_config_path, BackendConfig,
    CONFIG_PATH_ENV, NO_USER_CONFIG_ENV,
};
pub use detect::{detect_hardware_and_backends, BackendReport, GpuDevice};

use std::path::{Path, PathBuf};

use crate::bench::score::Backend;
use crate::cli::{BenchCommand, Commands, GenerationArgs, PipelineCommand};

/// The GPU API (`Vulkan`, `CUDA`...) the `llama-server` of `dir` sees first, for the report
/// labels; `None` when it cannot be asked or has no GPU.
pub fn gpu_api(dir: Option<&Path>) -> Option<&'static str> {
    let server = crate::eval::server::locate_server(dir).ok()?;
    let devices = detect::list_devices(&server).ok()?;
    Some(match devices.first()?.api() {
        "Vulkan" => "Vulkan",
        "CUDA" => "CUDA",
        "MTL" | "Metal" => "Metal",
        "HIP" | "ROCm" => "HIP",
        "SYCL" => "SYCL",
        _ => "GPU",
    })
}

/// The backend as a report shows it: `native-cpu`, `llama-server (Vulkan, ngl=99, batch=6)`,
/// `hipfire`, or the plain label of the llama.cpp process backends.
pub fn describe(
    label: &str,
    gpu_api: Option<&str>,
    gpu_layers: Option<u32>,
    batch_size: usize,
) -> String {
    match label {
        "native" => "native-cpu".into(),
        "llama-server" => {
            let ngl = gpu_layers.map_or("auto".into(), |n| n.to_string());
            let api = gpu_api.map(|a| format!("{a}, ")).unwrap_or_default();
            format!("llama-server ({api}ngl={ngl}, batch={batch_size})")
        }
        other => other.into(),
    }
}

/// [`describe`] of a resolved configuration (the backend defaults to native).
pub fn describe_config(config: &BackendConfig) -> String {
    let backend = config.backend.unwrap_or(Backend::Native);
    let api = match backend {
        Backend::LlamaServer => gpu_api(config.llama_cpp_dir.as_deref()),
        _ => None,
    };
    describe(
        backend.label(),
        api,
        config.gpu_layers,
        config.batch_size.unwrap_or(1),
    )
}

// ------------------------------------------------------------ command line layer

fn fill_dir(slot: &mut Option<PathBuf>, user: &BackendConfig) {
    if slot.is_none() {
        slot.clone_from(&user.llama_cpp_dir);
    }
}

/// One generating command: its flags over the user file, written back into the flags so
/// that nothing downstream reads the file. `default` is the command's historical backend.
fn generation(
    backend: &mut Option<Backend>,
    default: Backend,
    dir: &mut Option<PathBuf>,
    g: &mut GenerationArgs,
    user: &BackendConfig,
) -> Result<(), String> {
    let top = BackendConfig {
        backend: *backend,
        gpu_layers: g.gpu_layers,
        batch_size: g.batch_size,
        llama_cpp_dir: dir.clone(),
        hipfire_bin: g.hipfire_bin.clone(),
        gguf_dtype: g.gguf_dtype,
    };
    let effective = resolve(top, user);
    let chosen = effective.backend.unwrap_or(default);
    *backend = Some(chosen);
    g.gpu_layers = effective.gpu_layers;
    g.batch_size = effective.batch_size;
    g.hipfire_bin.clone_from(&effective.hipfire_bin);
    g.gguf_dtype = effective.gguf_dtype;
    *dir = effective.llama_cpp_dir.clone();
    match chosen {
        Backend::Hipfire => Err(hipfire::unavailable(&effective)),
        _ => Ok(()),
    }
}

/// Whether the command reads the backend settings (the others never fail on a broken file).
fn uses_backend(command: &Commands) -> bool {
    match command {
        Commands::Eval(_) => true,
        Commands::Bench(b) => matches!(
            b.command,
            BenchCommand::Ppl(_)
                | BenchCommand::Eval(_)
                | BenchCommand::Refusal(_)
                | BenchCommand::JudgeEval(_)
        ),
        Commands::Pipeline(p) => matches!(
            p.command,
            PipelineCommand::Plan(_) | PipelineCommand::Run(_)
        ),
        _ => false,
    }
}

/// Fills what the command line left out from the user configuration: the backend, the
/// llama.cpp folder, `--gpu-layers`, `--batch-size`, the GGUF type. The pipeline resolves its
/// own settings later (its `pipeline.json` sits between the flags and the file).
pub fn apply_user_config(command: &mut Commands) -> Result<(), String> {
    if !uses_backend(command) {
        return Ok(());
    }
    apply_with(command, &load_user_config()?)
}

pub fn apply_with(command: &mut Commands, user: &BackendConfig) -> Result<(), String> {
    match command {
        Commands::Eval(a) => fill_dir(&mut a.llama_cpp_dir, user),
        Commands::Bench(b) => match &mut b.command {
            BenchCommand::Ppl(a) => fill_dir(&mut a.llama_cpp_dir, user),
            BenchCommand::JudgeEval(a) => fill_dir(&mut a.judge_args.llama_cpp_dir, user),
            BenchCommand::Eval(a) => {
                let (backend, dir, g) = (&mut a.backend, &mut a.llama_cpp_dir, &mut a.generation);
                generation(backend, Backend::Llama, dir, g, user)?;
            }
            BenchCommand::Refusal(a) => {
                let (backend, dir) = (&mut a.backend, &mut a.judge_args.llama_cpp_dir);
                generation(backend, Backend::Native, dir, &mut a.generation, user)?;
            }
            _ => {}
        },
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::Cli;
    use clap::Parser;

    fn parse(args: &[&str]) -> Commands {
        Cli::try_parse_from(std::iter::once("surgeon").chain(args.iter().copied()))
            .unwrap()
            .command
    }

    fn gpu_user() -> BackendConfig {
        BackendConfig {
            backend: Some(Backend::LlamaServer),
            gpu_layers: Some(99),
            batch_size: Some(6),
            llama_cpp_dir: Some("llama-dir".into()),
            ..Default::default()
        }
    }

    const EVAL: [&str; 7] = [
        "bench",
        "eval",
        "--model",
        "m",
        "--dataset",
        "d.jsonl",
        "--limit",
    ];

    #[test]
    fn the_user_file_fills_what_the_flags_left_out() {
        let mut cmd = parse(&[EVAL.as_slice(), &["30"]].concat());
        apply_with(&mut cmd, &gpu_user()).unwrap();
        let Commands::Bench(b) = cmd else { panic!() };
        let BenchCommand::Eval(a) = b.command else {
            panic!()
        };
        assert_eq!(a.backend, Some(Backend::LlamaServer));
        assert_eq!(a.generation.gpu_layers, Some(99));
        assert_eq!(a.generation.batch_size, Some(6));
        assert_eq!(a.llama_cpp_dir, Some("llama-dir".into()));
        assert_eq!(a.generation.options().batch_size, 6);
    }

    #[test]
    fn flags_win_and_the_default_stays_historical_without_a_file() {
        let mut cmd = parse(
            &[
                EVAL.as_slice(),
                &["30", "--backend", "native", "--batch-size", "2"],
            ]
            .concat(),
        );
        apply_with(&mut cmd, &gpu_user()).unwrap();
        let Commands::Bench(b) = cmd else { panic!() };
        let BenchCommand::Eval(a) = b.command else {
            panic!()
        };
        // Native: the user's -ngl does not apply, the flag's batch does.
        assert_eq!(a.backend, Some(Backend::Native));
        assert_eq!(
            (a.generation.gpu_layers, a.generation.batch_size),
            (None, Some(2))
        );

        let mut cmd = parse(&[EVAL.as_slice(), &["30"]].concat());
        apply_with(&mut cmd, &BackendConfig::default()).unwrap();
        let Commands::Bench(b) = cmd else { panic!() };
        let BenchCommand::Eval(a) = b.command else {
            panic!()
        };
        assert_eq!(
            a.backend,
            Some(Backend::Llama),
            "bench eval keeps its old default"
        );
        assert_eq!(
            a.generation.options(),
            crate::bench::generation::GenOptions::default()
        );

        let mut cmd = parse(&["bench", "refusal", "--model", "m", "--prompts", "p"]);
        apply_with(&mut cmd, &BackendConfig::default()).unwrap();
        let Commands::Bench(b) = cmd else { panic!() };
        let BenchCommand::Refusal(a) = b.command else {
            panic!()
        };
        assert_eq!(
            a.backend,
            Some(Backend::Native),
            "bench refusal keeps its old default"
        );
    }

    #[test]
    fn hipfire_is_refused_with_an_actionable_message() {
        let mut cmd = parse(&[EVAL.as_slice(), &["30", "--backend", "hipfire"]].concat());
        let err = apply_with(&mut cmd, &BackendConfig::default()).unwrap_err();
        assert!(err.contains("llama-server"), "{err}");
    }

    #[test]
    fn labels_show_the_real_backend() {
        assert_eq!(describe("native", None, None, 1), "native-cpu");
        assert_eq!(
            describe("llama-server", Some("Vulkan"), Some(99), 6),
            "llama-server (Vulkan, ngl=99, batch=6)"
        );
        assert_eq!(
            describe("llama-server", None, None, 1),
            "llama-server (ngl=auto, batch=1)"
        );
        assert_eq!(describe("hipfire", None, None, 1), "hipfire");
        assert_eq!(
            describe("llama-completion", None, None, 1),
            "llama-completion"
        );
    }
}
