//! Command line definition only, no business logic.
//!
//! `main.rs` parses, dispatches, then maps errors to an exit code:
//!
//! ```text
//! let cli = Cli::parse();
//! let code = match cli.command {
//!     Commands::Check(args)   => /* hardware::can_load_model(&args.model) */,
//!     Commands::Inspect(args) => /* engine::read_index + find_mlp_tensors */,
//!     Commands::Apply(args)   => /* load_patch -> validate_patch -> BackupGuard -> apply_patch */,
//!     Commands::Eval(args)    => /* eval module */,
//! };
//! std::process::exit(code);
//! ```
//!
//! Exit codes (`impl From<&PatchError> for i32`, constants in `patch::exit_code`):
//! - 0: success
//! - 2: patch unreadable, malformed or invalid, nothing touched
//! - 3: execution error, original intact (never touched or restored from backup)
//! - 4: CRITICAL, rollback failed or model left partially patched without backup

use std::path::PathBuf;

use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};

use crate::extract::{ChatTemplate, DTypeChoice, Pooling, Site};
pub use crate::patch::exit_code;
use crate::patch::{LayerSelection, Projection};

#[derive(Debug, Clone, Parser)]
#[command(
    name = "surgeon",
    version,
    about = "Chirurgie de poids SafeTensors pilotée par patch.json",
    styles = crate::ui::clap_styles()
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
    /// Machine-readable JSON output (for an AI agent).
    #[arg(long, global = true)]
    pub json: bool,
    /// -v, -vv, -vvv
    #[arg(short, long, global = true, action = ArgAction::Count)]
    pub verbose: u8,
}

#[derive(Debug, Clone, Subcommand)]
pub enum Commands {
    /// Check that the machine can load the model.
    Check(CheckArgs),
    /// List the MLP tensors of a model.
    Inspect(InspectArgs),
    /// Apply a patch.json to a model.
    Apply(ApplyArgs),
    /// Evaluate a model on prompts.
    Eval(EvalArgs),
    /// Extract a steering vector from a Qwen2 layer (pure Rust, CPU).
    Extract(ExtractArgs),
    /// Terminal dashboard: MLP weight inspector and steering vector viewer.
    Tui(TuiArgs),
    /// Print the JSON Schema of patch.json, or the MCP tool definitions (raw JSON).
    Schema(SchemaArgs),
    /// Run the MCP (Model Context Protocol) server on stdio.
    Mcp(McpArgs),
    /// Measure a surgery: logits shift, perplexity delta, Pass@1, logits dumps.
    Bench(BenchArgs),
}

/// Exit codes: 0 ok (warnings included), 1 insufficient resources, llama.cpp binary not
/// found or cancelled, 2 invalid arguments or files, 8 bench failed (process, unreadable
/// output, incompatible shapes, generation), 9 threshold exceeded with `--fail-on-warn`,
/// 70 internal.
#[derive(Debug, Clone, Args)]
pub struct BenchArgs {
    #[command(subcommand)]
    pub command: BenchCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub enum BenchCommand {
    /// Compare two logits dumps: KL(before‖after), cosine, top-1 / top-k shifts.
    Logits(BenchLogitsArgs),
    /// Perplexity of a GGUF (or SafeTensors, converted) with llama-perplexity.
    Ppl(BenchPplArgs),
    /// Exact-match Pass@1 on a GSM8K / MATH style dataset (greedy, llama.cpp).
    Eval(BenchEvalArgs),
    /// Write the logits of a Qwen2 model on prompts (pure Rust, CPU).
    Dump(BenchDumpArgs),
}

#[derive(Debug, Clone, Args)]
pub struct BenchLogitsArgs {
    /// Reference dump (.safetensors, .npy, .pt/.pth).
    #[arg(long)]
    pub before: PathBuf,
    /// Dump to compare, same shape.
    #[arg(long)]
    pub after: PathBuf,
    /// Tensor to read (default: `logits`).
    #[arg(long)]
    pub tensor: Option<String>,
    /// Size of the compared top-k sets.
    #[arg(long, default_value_t = crate::bench::DEFAULT_TOP_K)]
    pub top_k: usize,
    /// Prompts listed in worst_prompts (needs prompt_offsets).
    #[arg(long, default_value_t = crate::bench::DEFAULT_WORST)]
    pub worst: usize,
    /// Warn when the mean KL is strictly above this value.
    #[arg(long)]
    pub warn_kl: Option<f64>,
    /// Warn when the top-1 shift rate (0..1) is strictly above this value.
    #[arg(long)]
    pub warn_top1_shift: Option<f64>,
    /// Exit with code 9 when a threshold is exceeded.
    #[arg(long)]
    pub fail_on_warn: bool,
}

#[derive(Debug, Clone, Args)]
pub struct BenchPplArgs {
    /// .gguf, or a SafeTensors directory converted to a temporary GGUF.
    #[arg(long)]
    pub model: PathBuf,
    /// Text file evaluated by llama-perplexity.
    #[arg(long)]
    pub text: PathBuf,
    /// Reference model, measured first.
    #[arg(long, conflicts_with = "baseline_ppl")]
    pub baseline_model: Option<PathBuf>,
    /// Reference perplexity, already known.
    #[arg(long)]
    pub baseline_ppl: Option<f64>,
    /// Uncertainty (±) of --baseline-ppl.
    #[arg(long, requires = "baseline_ppl")]
    pub baseline_ppl_err: Option<f64>,
    #[arg(long, default_value_t = crate::bench::DEFAULT_CTX)]
    pub ctx: u32,
    #[arg(long)]
    pub chunks: Option<u32>,
    #[arg(long)]
    pub threads: Option<u32>,
    /// Layers offloaded to the GPU (0: CPU only, reproducible).
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: u32,
    #[arg(long, env = "LLAMA_CPP_DIR")]
    pub llama_cpp_dir: Option<PathBuf>,
    /// Warn when the rounded delta is strictly above this value.
    #[arg(long, default_value_t = crate::bench::DEFAULT_WARN_DELTA)]
    pub warn_delta: f64,
    /// Per llama-perplexity run.
    #[arg(long)]
    pub timeout_secs: Option<u64>,
    /// Extra llama-perplexity flag from a whitelist: --batch-size=N, --ubatch-size=N,
    /// --ppl-stride=N, --seed=N, -b=N, -ub=N, --no-mmap, --mlock. Repeatable.
    #[arg(long = "llama-arg", allow_hyphen_values = true)]
    pub llama_args: Vec<String>,
    /// Exit with code 9 when the delta exceeds --warn-delta.
    #[arg(long)]
    pub fail_on_warn: bool,
}

#[derive(Debug, Clone, Args)]
pub struct BenchEvalArgs {
    /// .gguf, or a SafeTensors directory converted to a temporary GGUF.
    #[arg(long)]
    pub model: PathBuf,
    /// JSON array (or JSONL) of {question|problem, answer|solution, id?}.
    #[arg(long)]
    pub dataset: PathBuf,
    /// Reference model on the same questions (paired comparison).
    #[arg(long)]
    pub baseline_model: Option<PathBuf>,
    /// First N entries, in file order.
    #[arg(long, default_value_t = crate::bench::DEFAULT_LIMIT)]
    pub limit: usize,
    #[arg(long, default_value_t = crate::bench::DEFAULT_MAX_NEW_TOKENS)]
    pub max_new_tokens: u32,
    /// Template with {question}, or @FILE.
    #[arg(long)]
    pub prompt_template: Option<String>,
    #[arg(long, value_enum, default_value_t = crate::bench::dataset::DatasetFormat::Auto)]
    pub format: crate::bench::dataset::DatasetFormat,
    #[arg(long, env = "LLAMA_CPP_DIR")]
    pub llama_cpp_dir: Option<PathBuf>,
    /// Per question; a timeout counts the question wrong.
    #[arg(long)]
    pub timeout_secs: Option<u64>,
    /// Warn when the accuracy (0..1) is strictly below this value.
    #[arg(long)]
    pub min_accuracy: Option<f64>,
    /// Exit with code 9 when the accuracy is below --min-accuracy.
    #[arg(long)]
    pub fail_on_warn: bool,
}

#[derive(Debug, Clone, Args)]
pub struct BenchDumpArgs {
    /// Qwen2 directory (config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    pub model: PathBuf,
    /// JSON array of strings or of {"text": ...}, JSONL, or plain text.
    #[arg(long)]
    pub prompts: PathBuf,
    /// Output .safetensors (tensors `logits` and `prompt_offsets`).
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, value_enum, default_value_t = DTypeChoice::Auto)]
    pub dtype: DTypeChoice,
    #[arg(long, value_enum, default_value_t = crate::bench::dump::OutDtype::F32)]
    pub out_dtype: crate::bench::dump::OutDtype,
    /// Prompts are truncated on the right beyond this many tokens.
    #[arg(long, default_value_t = crate::bench::DEFAULT_DUMP_MAX_TOKENS)]
    pub max_tokens: usize,
    /// Refuses the dump, before any computation, above this many positions.
    #[arg(long, default_value_t = crate::bench::DEFAULT_MAX_TOTAL_POSITIONS)]
    pub max_total_positions: usize,
    /// Replace an existing --out.
    #[arg(long)]
    pub overwrite: bool,
}

#[derive(Debug, Clone, Args)]
pub struct CheckArgs {
    #[arg(long)]
    pub model: PathBuf,
    /// RAM safety margin in GiB (default: hardware policy).
    #[arg(long)]
    pub policy_ram_margin_gib: Option<f64>,
}

#[derive(Debug, Clone, Args)]
pub struct InspectArgs {
    #[arg(long)]
    pub model: PathBuf,
    /// "all", "10-20" or "3,5,7".
    #[arg(long, default_value = "all")]
    pub layers: LayerSelection,
    /// gate, up or down; repeatable. None = all three.
    #[arg(long = "proj")]
    pub projections: Vec<Projection>,
    #[arg(long)]
    pub show_stats: bool,
}

#[derive(Debug, Clone, Args)]
pub struct ApplyArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub patch: PathBuf,
    #[arg(long)]
    pub dry_run: bool,
    /// Refused by validation when the patch sets `backup: true`.
    #[arg(long)]
    pub no_backup: bool,
    /// Default: next to the model.
    #[arg(long)]
    pub backup_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct EvalArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub prompts: PathBuf,
    #[arg(long, default_value_t = 256)]
    pub max_new_tokens: u32,
    #[arg(long)]
    pub export_gguf: Option<PathBuf>,
    #[arg(long, default_value = "Q4_K_M")]
    pub quant: String,
    #[arg(long, env = "LLAMA_CPP_DIR")]
    pub llama_cpp_dir: Option<PathBuf>,
    #[arg(long)]
    pub allow_degraded: bool,
}

/// Exit codes: 0 ok, 1 insufficient resources (or cancelled), 2 invalid arguments, prompt
/// files or existing `--out`, 7 extraction failed, 70 internal.
#[derive(Debug, Clone, Args)]
pub struct ExtractArgs {
    /// Directory with config.json, tokenizer.json and the *.safetensors (or an index).
    #[arg(long)]
    pub model: PathBuf,
    /// JSON array of strings or of {"text": ...}, JSONL, or plain text.
    #[arg(long)]
    pub pos_prompts: PathBuf,
    #[arg(long)]
    pub neg_prompts: PathBuf,
    /// 0-based layer; must be < num_hidden_layers.
    #[arg(long, default_value_t = 20)]
    pub layer: usize,
    /// Output .safetensors (one 1D F32 tensor named `direction`).
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, value_enum, default_value_t = Pooling::Last)]
    pub pooling: Pooling,
    #[arg(long, value_enum, default_value_t = Site::ResidPost)]
    pub site: Site,
    #[arg(long, value_enum, default_value_t = ChatTemplate::NoTemplate)]
    pub chat_template: ChatTemplate,
    /// Prompts are truncated on the right beyond this many tokens.
    #[arg(long, default_value_t = 512)]
    pub max_tokens: usize,
    #[arg(long, value_enum, default_value_t = DTypeChoice::Auto)]
    pub dtype: DTypeChoice,
    /// Replace an existing --out.
    #[arg(long)]
    pub overwrite: bool,
}

/// Exit codes: 0 normal exit, 1 not an interactive terminal, 2 invalid arguments or files
/// (`--json` included: the TUI has no JSON output), 70 internal.
#[derive(Debug, Clone, Args)]
pub struct TuiArgs {
    /// Model directory or SafeTensors file (shards and index supported).
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Direction file (.safetensors) produced by `surgeon extract`.
    #[arg(long)]
    pub vector: Option<PathBuf>,
    /// Print one frame as plain text on stdout and exit (no interactive terminal needed).
    #[arg(long)]
    pub snapshot: bool,
    /// Frame size for --snapshot, COLSxROWS.
    #[arg(long, default_value = "100x32", value_parser = crate::tui::parse_size)]
    pub size: (u16, u16),
}

#[derive(Debug, Clone, Args)]
pub struct SchemaArgs {
    #[arg(long, value_enum, default_value = "patch")]
    pub kind: SchemaKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SchemaKind {
    /// JSON Schema (draft-07) of patch.json.
    Patch,
    /// The MCP tool definitions, exactly as `tools/list` returns them.
    Tools,
}

#[derive(Debug, Clone, Args)]
pub struct McpArgs {
    /// Allowed directory, repeatable. Default: $SURGEON_MCP_ALLOW_DIRS (system path
    /// separator), else the current directory.
    #[arg(long)]
    pub allow_dir: Vec<PathBuf>,
    /// Refuse surgeon_apply and GGUF exports.
    #[arg(long)]
    pub read_only: bool,
    /// Honor `backup: false` in patches (a backup is imposed otherwise).
    #[arg(long)]
    pub allow_no_backup: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::{parse_patch, validate_patch, PatchError, ValidationOptions};
    use clap::CommandFactory;

    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("weight-surgeon").chain(args.iter().copied()))
    }

    #[test]
    fn command_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn check_and_global_flags_anywhere() {
        let cli = parse(&[
            "--json",
            "check",
            "--model",
            "m.safetensors",
            "-vv",
            "--policy-ram-margin-gib",
            "1.5",
        ])
        .unwrap();
        assert!(cli.json);
        assert_eq!(cli.verbose, 2);
        let Commands::Check(args) = cli.command else {
            panic!()
        };
        assert_eq!(args.model, PathBuf::from("m.safetensors"));
        assert_eq!(args.policy_ram_margin_gib, Some(1.5));
    }

    #[test]
    fn inspect_parses_layers_and_repeated_projections() {
        let cli = parse(&[
            "inspect",
            "--model",
            "m",
            "--layers",
            "10-20",
            "--proj",
            "gate",
            "--proj",
            "down_proj",
            "--show-stats",
        ])
        .unwrap();
        let Commands::Inspect(args) = cli.command else {
            panic!()
        };
        assert_eq!(
            args.layers,
            LayerSelection::Range {
                start: 10,
                end_inclusive: 20
            }
        );
        assert_eq!(args.projections, [Projection::Gate, Projection::Down]);
        assert!(args.show_stats);

        let Commands::Inspect(args) = parse(&["inspect", "--model", "m"]).unwrap().command else {
            panic!()
        };
        assert_eq!(args.layers, LayerSelection::All);
        assert!(args.projections.is_empty());

        assert!(parse(&["inspect", "--model", "m", "--layers", "20-10"]).is_err());
        assert!(parse(&["inspect", "--model", "m", "--proj", "q_proj"]).is_err());
    }

    #[test]
    fn apply_and_eval_defaults() {
        let cli = parse(&[
            "apply",
            "--model",
            "m",
            "--patch",
            "p.json",
            "--dry-run",
            "--backup-dir",
            "bk",
        ])
        .unwrap();
        let Commands::Apply(args) = cli.command else {
            panic!()
        };
        assert!(args.dry_run && !args.no_backup);
        assert_eq!(args.backup_dir, Some(PathBuf::from("bk")));

        let cli = parse(&[
            "eval",
            "--model",
            "m",
            "--prompts",
            "p.txt",
            "--llama-cpp-dir",
            "llama",
            "--allow-degraded",
        ])
        .unwrap();
        let Commands::Eval(args) = cli.command else {
            panic!()
        };
        assert_eq!(args.max_new_tokens, 256);
        assert_eq!(args.quant, "Q4_K_M");
        assert_eq!(args.export_gguf, None);
        assert_eq!(args.llama_cpp_dir, Some(PathBuf::from("llama")));
        assert!(args.allow_degraded);
    }

    #[test]
    fn missing_required_argument_is_rejected() {
        let err = parse(&["apply", "--model", "m"]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
        assert!(parse(&[]).is_err());
        assert!(parse(&[
            "eval",
            "--model",
            "m",
            "--max-new-tokens",
            "-1",
            "--prompts",
            "p"
        ])
        .is_err());
    }

    #[test]
    fn bench_subcommands_defaults_requirements_and_conflicts() {
        let bench = |args: &[&str]| -> BenchCommand {
            let mut argv = vec!["bench"];
            argv.extend_from_slice(args);
            match parse(&argv).unwrap().command {
                Commands::Bench(b) => b.command,
                other => panic!("{other:?}"),
            }
        };
        let BenchCommand::Logits(l) = bench(&["logits", "--before", "a", "--after", "b"]) else {
            panic!()
        };
        assert_eq!(
            (l.top_k, l.worst, l.warn_kl, l.fail_on_warn),
            (5, 5, None, false)
        );
        let BenchCommand::Ppl(p) = bench(&["ppl", "--model", "m.gguf", "--text", "t.txt"]) else {
            panic!()
        };
        assert_eq!((p.ctx, p.n_gpu_layers, p.warn_delta), (512, 0, 0.2));
        assert!(p.baseline_model.is_none() && p.llama_args.is_empty());
        let BenchCommand::Ppl(p) = bench(&[
            "ppl",
            "--model",
            "m",
            "--text",
            "t",
            "--llama-arg",
            "--batch-size=64",
        ]) else {
            panic!()
        };
        assert_eq!(p.llama_args, ["--batch-size=64"]);
        let BenchCommand::Eval(e) = bench(&["eval", "--model", "m", "--dataset", "d.json"]) else {
            panic!()
        };
        assert_eq!((e.limit, e.max_new_tokens), (50, 512));
        assert_eq!(e.format, crate::bench::dataset::DatasetFormat::Auto);
        let BenchCommand::Dump(d) =
            bench(&["dump", "--model", "m", "--prompts", "p", "--out", "o"])
        else {
            panic!()
        };
        assert_eq!((d.max_tokens, d.max_total_positions), (256, 4096));
        assert_eq!(d.out_dtype, crate::bench::dump::OutDtype::F32);

        for bad in [
            &["bench", "logits", "--before", "a"][..],
            &["bench", "ppl", "--model", "m"],
            &["bench", "dump", "--model", "m", "--prompts", "p"],
            &[
                "bench",
                "ppl",
                "--model",
                "m",
                "--text",
                "t",
                "--baseline-model",
                "b",
                "--baseline-ppl",
                "7",
            ],
            &[
                "bench",
                "ppl",
                "--model",
                "m",
                "--text",
                "t",
                "--baseline-ppl-err",
                "0.1",
            ],
            &[
                "bench",
                "eval",
                "--model",
                "m",
                "--dataset",
                "d",
                "--format",
                "mmlu",
            ],
            &["bench"],
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }

    /// `--no-backup` parses, but validation refuses it against `backup: true`.
    #[test]
    fn no_backup_conflicts_with_patch_imposing_backup() {
        let Commands::Apply(args) =
            parse(&["apply", "--model", "m", "--patch", "p.json", "--no-backup"])
                .unwrap()
                .command
        else {
            panic!()
        };
        let patch = parse_patch(
            r#"{"schema_version":1,"backup":true,"operations":[{"id":"a","layers":"all",
                "projections":["gate_proj"],"operation":"add","vector_file":"v.json","alpha":1}]}"#,
        )
        .unwrap();
        let opts = ValidationOptions {
            check_files: false,
            no_backup: args.no_backup,
        };
        let err = validate_patch(&patch, std::path::Path::new("."), &opts).unwrap_err();
        assert!(
            matches!(&err, PatchError::Invalid(issues) if issues[0].path == "backup"),
            "{err:?}"
        );
        assert_eq!(i32::from(&err), exit_code::INVALID_PATCH);
    }
}
