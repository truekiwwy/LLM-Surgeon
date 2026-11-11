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

use clap::builder::{PossibleValue, PossibleValuesParser};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};

use crate::extract::{ChatTemplate, DTypeChoice, Method, MethodOptions, Pooling, Site};
pub use crate::patch::exit_code;
use crate::patch::{LayerSelection, Projection};

#[derive(Debug, Clone, Parser)]
#[command(
    name = "surgeon",
    version,
    about = "Surgery Engine : chirurgie des poids de LLM locaux, en terminal ou depuis un agent IA",
    long_about = "surgeon est le front-end CLI du Surgery Engine : inspecter, modifier \
(patch.json, .surgery, LoRA), évaluer et mesurer les poids de modèles de langage locaux \
(SafeTensors, GGUF), en Rust pur sur CPU.\n\n\
Le même moteur est appelable depuis un agent IA : `surgeon mcp` (extension MCP, outils \
typés, dossiers autorisés, sauvegarde imposée) et `--json` (enveloppe stable). Les méthodes \
d'extraction disponibles : `surgeon plugins list`.",
    after_help = "Architecture et points d'extension : docs/ARCHITECTURE.md, docs/PLUGINS.md ; \
serveur MCP : docs/MCP.md.",
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
    /// Extract a steering vector from a Qwen2, Llama/Mistral or Qwen3 layer (pure Rust, CPU).
    Extract(ExtractCommand),
    /// Terminal dashboard: MLP weight inspector and steering vector viewer.
    Tui(TuiArgs),
    /// Print the JSON Schema of patch.json, or the MCP tool definitions (raw JSON).
    Schema(SchemaArgs),
    /// Run the MCP (Model Context Protocol) server on stdio.
    Mcp(McpArgs),
    /// Measure a surgery: logits shift, perplexity delta, Pass@1, logits dumps.
    Bench(BenchArgs),
    /// Fit per-layer steering coefficients by ridge regression on measured responses.
    Autotune(AutotuneArgs),
    /// Low-rank differential patch (.surgery) between two checkpoints.
    Diff(crate::surgery::cli_args::DiffArgs),
    /// Merge or extract a LoRA adapter.
    Lora(crate::lora::cli_args::LoraArgs),
    /// Extension points of the engine and their registered implementations.
    Plugins(PluginsArgs),
    /// Built-in model architectures, and the one detected for a model.
    Arch(ArchArgs),
}

/// `surgeon arch`: the architecture descriptions (`src/arch`). Not exposed over MCP:
/// `surgeon_inspect` reports the detected architecture.
#[derive(Debug, Clone, Args)]
pub struct ArchArgs {
    #[command(subcommand)]
    pub command: ArchCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub enum ArchCommand {
    /// Built-in architectures: model types, GGUF names, project_out exactness, forward.
    List,
    /// The architecture of a model (config.json, GGUF metadata or --arch-spec) and the
    /// roles of its tensors.
    Detect(ArchDetectArgs),
}

#[derive(Debug, Clone, Args)]
pub struct ArchDetectArgs {
    #[arg(long)]
    pub model: PathBuf,
    /// Architecture spec (JSON, `surgeon schema --kind arch`) instead of detection.
    #[arg(long)]
    pub arch_spec: Option<PathBuf>,
}

/// `surgeon plugins`: always exit code 0. Not exposed over MCP.
#[derive(Debug, Clone, Args)]
pub struct PluginsArgs {
    #[command(subcommand)]
    pub command: PluginsCommand,
}

#[derive(Debug, Clone, Copy, Subcommand)]
pub enum PluginsCommand {
    /// Extension points and implementations: identifier, title, origin.
    List,
}

/// `surgeon extract`: the prompt 8 options plus the direction method (two structs because
/// `ExtractArgs` is built field by field by callers).
#[derive(Debug, Clone, Args)]
pub struct ExtractCommand {
    #[command(flatten)]
    pub args: ExtractArgs,
    #[command(flatten)]
    pub method: ExtractMethodArgs,
}

#[derive(Debug, Clone, Args)]
pub struct ExtractMethodArgs {
    /// Direction method: an identifier of the extraction registry (`surgeon plugins list`).
    #[arg(long, default_value = crate::plugin::DEFAULT_METHOD, value_parser = method_ids())]
    pub method: String,
    /// svd: center the paired deltas (for pairs of arbitrary order).
    #[arg(long)]
    pub svd_center: bool,
    /// leace: shrinkage λ in [0, 1] of Σ toward tr(Σ)/d·I (0: pseudo-inverse).
    #[arg(long, default_value_t = crate::extract::DEFAULT_LEACE_SHRINKAGE)]
    pub leace_shrinkage: f64,
    /// leace with shrinkage 0: relative threshold of the kept eigenvalues of Σ.
    #[arg(long, default_value_t = crate::extract::DEFAULT_LEACE_RCOND)]
    pub leace_rcond: f64,
    /// Held-out fraction (by index) for out-of-sample diagnostics (10+ prompts per set).
    #[arg(long, default_value_t = 0.0)]
    pub holdout_frac: f64,
    /// Directions of the subspace: svd keeps that many singular vectors (default 1),
    /// leace with --labels-file at most that many (default: all).
    #[arg(long, value_parser = clap::value_parser!(u64).range(1..=crate::engine::DEFAULT_MAX_RANK as u64))]
    pub rank: Option<u64>,
    /// leace: JSON array with one label per prompt, positives then negatives (a class
    /// index, or an array of m numbers).
    #[arg(long)]
    pub labels_file: Option<PathBuf>,
}

impl ExtractMethodArgs {
    /// The built-in options; `method` maps the identifier, always a built-in one since the
    /// parser only accepts those.
    pub fn options(&self) -> MethodOptions {
        MethodOptions {
            method: Method::from_id(&self.method).unwrap_or_default(),
            svd_center: self.svd_center,
            leace_shrinkage: self.leace_shrinkage,
            leace_rcond: self.leace_rcond,
            holdout_frac: self.holdout_frac,
        }
    }
}

/// `--method` accepts the identifiers of the built-in registry, in its listing order (the
/// possible values and the error message are those of the former enum).
fn method_ids() -> PossibleValuesParser {
    let registry = crate::plugin::Registry::with_builtin();
    PossibleValuesParser::new(
        registry
            .list()
            .map(|(m, _)| PossibleValue::new(m.id()).help(m.describe().summary)),
    )
}

/// Exit codes: 0 ok, 1 insufficient resources or cancelled, 2 invalid arguments or files
/// (existing --out included), 7 autotune failed (singular or ill-conditioned system, model
/// error), 70 internal.
#[derive(Debug, Clone, Args)]
pub struct AutotuneArgs {
    /// Qwen2, Llama/Mistral or Qwen3 directory (config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    pub model: PathBuf,
    /// Direction file from `surgeon extract` (tensor `direction` and its metadata).
    #[arg(long)]
    pub vector: PathBuf,
    /// JSON array (or JSONL) of strings or of {"text", "target"?}.
    #[arg(long)]
    pub eval_dataset: PathBuf,
    /// Candidate layers "A-B" (default: 0 up to the readout layer).
    #[arg(long, value_parser = crate::patch::autotune::parse_layer_range)]
    pub target_layer_range: Option<(usize, usize)>,
    /// Layer whose activation is read (default: the vector's layer).
    #[arg(long)]
    pub readout_layer: Option<usize>,
    #[arg(long, default_value = "down_proj")]
    pub proj: Projection,
    #[arg(long, value_enum, default_value_t = crate::patch::autotune::Objective::Suppress)]
    pub objective: crate::patch::autotune::Objective,
    /// Target projection (required by --objective target, overrides the vector's means).
    #[arg(long)]
    pub target_projection: Option<f64>,
    /// Ridge λ > 0, or auto (LOOCV over 1e-6..1e2).
    #[arg(long, default_value = "1e-3")]
    pub l2_reg: crate::patch::autotune::L2Reg,
    /// Finite-difference step δ of the response measure.
    #[arg(long, default_value_t = 0.05)]
    pub probe_alpha: f64,
    #[arg(long, default_value_t = 1.0)]
    pub alpha_max: f64,
    /// Coefficients smaller than this are dropped.
    #[arg(long, default_value_t = 1e-4)]
    pub min_alpha: f64,
    /// Refuses the run, before any computation, above this many forwards.
    #[arg(long, default_value_t = 2000)]
    pub max_forwards: usize,
    #[arg(long, default_value_t = 256)]
    pub max_tokens: usize,
    #[arg(long, value_enum, default_value_t = DTypeChoice::Auto)]
    pub dtype: DTypeChoice,
    /// Override of the vector's site.
    #[arg(long, value_enum)]
    pub site: Option<Site>,
    /// Override of the vector's pooling.
    #[arg(long, value_enum)]
    pub pooling: Option<Pooling>,
    /// Override of the vector's chat template.
    #[arg(long, value_enum)]
    pub chat_template: Option<ChatTemplate>,
    /// patch.json to write (validated first); without it the patch is only in the report.
    #[arg(long)]
    pub out: Option<PathBuf>,
    #[arg(long)]
    pub overwrite: bool,
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
    /// Exact-match Pass@1 on a GSM8K / MATH style dataset (greedy; llama.cpp or native).
    Eval(BenchEvalArgs),
    /// Write the logits of a Qwen2, Llama/Mistral or Qwen3 model on prompts (pure Rust, CPU).
    Dump(BenchDumpArgs),
    /// Refusal rate of a Qwen2, Llama/Mistral or Qwen3 model on prompts (greedy, pure Rust, CPU).
    Refusal(BenchRefusalArgs),
    /// Sample the answers of a `bench refusal` run for human labelling.
    JudgeSample(BenchJudgeSampleArgs),
    /// Agreement of a judge model with human labels; writes the calibration file.
    JudgeEval(BenchJudgeEvalArgs),
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
    /// Generator: llama.cpp on a GGUF, or pure Rust on a Qwen2, Llama/Mistral or Qwen3 SafeTensors directory.
    #[arg(long, value_enum, default_value_t = crate::bench::score::Backend::Llama)]
    pub backend: crate::bench::score::Backend,
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
    /// Qwen2, Llama/Mistral or Qwen3 directory (config.json, tokenizer.json, *.safetensors).
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
pub struct BenchRefusalArgs {
    /// Qwen2, Llama/Mistral or Qwen3 directory (config.json, tokenizer.json, *.safetensors).
    #[arg(long)]
    pub model: PathBuf,
    /// JSON array of strings or of {"text": ...}, JSONL, or plain text.
    #[arg(long)]
    pub prompts: PathBuf,
    /// Reference model on the same prompts (paired comparison, Newcombe interval).
    #[arg(long)]
    pub baseline_model: Option<PathBuf>,
    #[arg(long, default_value_t = crate::bench::DEFAULT_REFUSAL_MAX_NEW_TOKENS)]
    pub max_new_tokens: u32,
    /// Wrapping of each prompt before generation.
    #[arg(long, value_enum, default_value_t = ChatTemplate::Chatml)]
    pub chat_template: ChatTemplate,
    /// Verdict: rules (refusal markers), llm (markers and a judge model) or both (plus
    /// their agreement).
    #[arg(long, value_enum, default_value_t = crate::bench::judge::JudgeMode::Rules)]
    pub judge: crate::bench::judge::JudgeMode,
    #[command(flatten)]
    pub markers: MarkerArgs,
    #[command(flatten)]
    pub judge_args: JudgeArgs,
    /// Calibration file written by `bench judge-eval --out`; without one the judge is
    /// reported as not calibrated.
    #[arg(long)]
    pub judge_calibration: Option<PathBuf>,
    /// Human labels the calibration must hold.
    #[arg(long, default_value_t = crate::bench::DEFAULT_MIN_CALIBRATION_N)]
    pub min_calibration_n: usize,
    /// Cohen's kappa (judge against human) the calibration must reach.
    #[arg(long, default_value_t = crate::bench::DEFAULT_MIN_KAPPA, allow_negative_numbers = true)]
    pub min_kappa: f64,
    /// Warn when more judge verdicts than this are unparsable.
    #[arg(long)]
    pub max_unparsable: Option<usize>,
    /// Exit with code 9 when --max-unparsable is exceeded.
    #[arg(long)]
    pub fail_on_warn: bool,
}

/// Refusal markers of the rule-based verdict.
#[derive(Debug, Clone, Args)]
pub struct MarkerArgs {
    /// Marker languages on top of English: fr, es, de or all, comma-separated.
    #[arg(long, value_delimiter = ',')]
    pub marker_langs: Vec<String>,
    /// JSON object {"<lang>": ["marker", ...]}: adds languages, replaces built-in lists.
    #[arg(long)]
    pub markers_file: Option<PathBuf>,
}

/// The judge model of `bench refusal --judge llm|both` and of `bench judge-eval`.
#[derive(Debug, Clone, Args)]
pub struct JudgeArgs {
    /// GGUF of the judge, run by llama-completion (greedy, grammar-constrained).
    #[arg(long)]
    pub judge_model: Option<PathBuf>,
    /// Question asked: refusal (does the answer refuse?) or harm (is it harmful?).
    #[arg(long, value_enum, default_value_t = crate::bench::judge::Role::Refusal)]
    pub judge_role: crate::bench::judge::Role,
    /// Rubric file with {prompt} and {response}; default: the built-in rubric of the role.
    #[arg(long)]
    pub judge_template: Option<PathBuf>,
    /// JSONL cache of verdicts, read then appended.
    #[arg(long)]
    pub judge_cache: Option<PathBuf>,
    /// Count `partial` answers as refusals in the binary rate.
    #[arg(long)]
    pub partial_as_refusal: bool,
    #[arg(long, env = "LLAMA_CPP_DIR")]
    pub llama_cpp_dir: Option<PathBuf>,
    /// Per judge verdict.
    #[arg(long)]
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Clone, Args)]
pub struct BenchJudgeSampleArgs {
    /// `--json` output of a `bench refusal` run.
    #[arg(long)]
    pub responses: PathBuf,
    /// Prompts file of that run (its report only keeps their hash).
    #[arg(long)]
    pub prompts: PathBuf,
    /// Sample size.
    #[arg(long, default_value_t = crate::bench::DEFAULT_SAMPLE_N)]
    pub n: usize,
    #[arg(long, default_value_t = u64::from(crate::bench::score::SEED))]
    pub seed: u64,
    /// Sample file to label: no model name, no condition.
    #[arg(long)]
    pub out: PathBuf,
    /// Replace an existing --out.
    #[arg(long)]
    pub overwrite: bool,
}

#[derive(Debug, Clone, Args)]
pub struct BenchJudgeEvalArgs {
    /// Labelled sample: the file of `bench judge-sample` with every "label" filled.
    #[arg(long)]
    pub labels: PathBuf,
    #[command(flatten)]
    pub judge_args: JudgeArgs,
    #[command(flatten)]
    pub markers: MarkerArgs,
    /// Calibration file to write, read by `bench refusal --judge-calibration`.
    #[arg(long)]
    pub out: Option<PathBuf>,
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
    /// gate, up, down, o_proj or embed_tokens; repeatable. None = gate, up and down.
    #[arg(long = "proj")]
    pub projections: Vec<Projection>,
    #[arg(long)]
    pub show_stats: bool,
    /// Architecture spec (JSON, `surgeon schema --kind arch`) instead of detection.
    #[arg(long)]
    pub arch_spec: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct ApplyArgs {
    #[arg(long)]
    pub model: PathBuf,
    #[arg(long)]
    pub patch: PathBuf,
    /// GGUF and chunked-write options.
    #[command(flatten)]
    pub gguf: crate::patch::gguf::GgufApplyArgs,
    #[arg(long)]
    pub dry_run: bool,
    /// Refused by validation when the patch sets `backup: true`.
    #[arg(long)]
    pub no_backup: bool,
    /// Default: next to the model.
    #[arg(long)]
    pub backup_dir: Option<PathBuf>,
    #[command(flatten)]
    pub surgery: crate::surgery::cli_args::SurgeryApplyArgs,
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
    #[command(flatten)]
    pub preview: crate::tui::preview_args::PreviewArgs,
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
    /// JSON Schema (draft-07) of the .surgery header.
    Surgery,
    /// JSON Schema (draft-07) of an architecture spec (--arch-spec).
    Arch,
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

    #[test]
    fn extract_methods_and_autotune_parse_with_defaults() {
        let base = [
            "extract",
            "--model",
            "m",
            "--pos-prompts",
            "p",
            "--neg-prompts",
            "n",
            "--out",
            "o.safetensors",
        ];
        let Commands::Extract(cmd) = parse(&base).unwrap().command else {
            panic!()
        };
        assert_eq!(
            cmd.method.options(),
            crate::extract::MethodOptions::default()
        );
        assert_eq!(cmd.args.layer, 20, "prompt 8 options unchanged");
        let mut svd = base.to_vec();
        svd.extend([
            "--method",
            "leace",
            "--leace-shrinkage",
            "0",
            "--holdout-frac",
            "0.5",
        ]);
        let Commands::Extract(cmd) = parse(&svd).unwrap().command else {
            panic!()
        };
        let opts = cmd.method.options();
        assert_eq!(opts.method, crate::extract::Method::Leace);
        assert_eq!((opts.leace_shrinkage, opts.holdout_frac), (0.0, 0.5));
        let mut bad = base.to_vec();
        bad.extend(["--method", "pca"]);
        assert!(parse(&bad).is_err());

        let tune = [
            "autotune",
            "--model",
            "m",
            "--vector",
            "v.safetensors",
            "--eval-dataset",
            "e.json",
        ];
        let Commands::Autotune(a) = parse(&tune).unwrap().command else {
            panic!()
        };
        assert_eq!(a.proj, Projection::Down);
        assert_eq!(a.l2_reg, crate::patch::autotune::L2Reg::Fixed(1e-3));
        assert_eq!(
            (a.probe_alpha, a.alpha_max, a.max_forwards),
            (0.05, 1.0, 2000)
        );
        assert_eq!(a.target_layer_range, None);
        let mut more = tune.to_vec();
        more.extend([
            "--target-layer-range",
            "3-7",
            "--l2-reg",
            "auto",
            "--objective",
            "amplify",
            "--proj",
            "up",
        ]);
        let Commands::Autotune(a) = parse(&more).unwrap().command else {
            panic!()
        };
        assert_eq!(a.target_layer_range, Some((3, 7)));
        assert_eq!(a.l2_reg, crate::patch::autotune::L2Reg::Auto);
        assert_eq!(a.proj, Projection::Up);
        for wrong in [
            ["--target-layer-range", "7-3"],
            ["--l2-reg", "-1"],
            ["--objective", "boost"],
        ] {
            let mut args = tune.to_vec();
            args.extend(wrong);
            assert!(parse(&args).is_err(), "{wrong:?}");
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
