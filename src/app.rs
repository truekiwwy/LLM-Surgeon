//! Orchestration of the CLI: wires `hardware`, `engine`, `patch` and `eval` together.
//! `main.rs` only parses, builds [`real_deps`], calls [`run_with`] and exits.
//!
//! Call graph (graphify, checked by hand). The four modules never import each other;
//! only `cli` (-> patch types) and this module cross module boundaries:
//!
//! ```text
//! main -> install_panic_hook, Ui::detect, parse_cli_with, real_deps -> run_with
//!      -> catch_unwind(dispatch)
//! run_with -> schema (raw JSON) | mcp (stdio server, see `mcp/mod.rs`) | dispatch
//! dispatch -> check   -> hardware::evaluate_budget
//!          -> inspect -> hardware::compute_model_size (shards) -> ReadOnlyModel::open
//!                     -> arch::resolve -> engine::find_by_roles -> tensor_stats (mmap view, streamed)
//!          -> apply   -> patch::load_patch -> validate_patch -> hardware::evaluate_budget
//!                     -> EngineBackend::open (WritableModel per shard)
//!                     -> apply_patch(dry run: preflight, no write)
//!                     -> BackupSet::create_many_in (touched shards only)
//!                     -> apply_patch -> EngineBackend::shift_tensor -> WritableModel::shift_tensor
//!                        ; error | NaN | Ctrl-C -> RollbackHandle::rollback ; ok -> flush -> commit
//!          -> extract -> load_prompt_texts -> core_extract_request -> extract::run_extraction
//!          -> eval    -> load_prompts -> export_gguf (temporary f16, unless .gguf)
//!                     -> generator_factory -> run_eval -> export_if_healthy
//!          -> extract -> core_extract_with(args, method id, params) -> core_extract_method
//!                     -> plugin::Registry::with_builtin().get(id) -> extract::run_extraction_plugin
//!          -> plugins -> plugins_list -> plugin::Registry::with_builtin().catalog()
//!          -> autotune -> autotune_cli -> core_autotune -> patch::autotune::run
//!                      -> linalg::cholesky (ridge, LOOCV) -> validate_patch -> write_atomic
//!          -> bench   -> bench_cli (args -> *Request) -> core_bench_{logits,ppl,eval,dump}
//!                     -> bench::{logits_metrics, ppl_runner, score, dump}::run (see `bench/mod.rs`)
//! Every human byte goes through `ui::*` (output audit in `ui.rs`); `--json` builds
//! `Ui::off()` for both streams, so no rendering reaches stdout or stderr.
//! emit -> JSON envelope | human text (write errors ignored: closed pipe never panics)
//! ```
//!
//! Exit codes: 0 ok, 1 general error, 2 invalid CLI/patch/input, 3 execution failed and
//! model intact (rolled back or never touched), 4 CRITICAL (rollback failed, or model left
//! modified without backup), 5 eval degraded/broken, 6 GGUF export failed, 7 extraction
//! failed, 8 bench failed, 9 bench threshold exceeded under `--fail-on-warn`, 70 internal.
//!
//! Deviations from the assembly spec, adapted to the real signatures:
//! - One `EngineBackend` owns 1..N `WritableModel`s: a single file is a 1-shard model, so a
//!   separate `MultiShardBackend` would be a second code path doing the same thing.
//! - The RAM budget is reported as a warning by `apply`, not enforced: `WritableModel` is a
//!   memory map that only pages in the targeted tensors, so refusing a model larger than
//!   RAM would block patches that work.
//! - `apply` opens the shards and runs a dry-run preflight *before* the backup, so a patch
//!   with no target or an unreadable vector fails without copying gigabytes.
//! - Ctrl-C is cooperative: the flag is checked before every shift (rollback), between
//!   prompts and chunks (eval) and between tensors (inspect). It never kills mid-write.

use std::any::Any;
use std::cell::RefCell;
use std::ffi::OsString;
use std::fmt::Display;
use std::io::Write;
use std::ops::ControlFlow;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use clap::Parser;
use serde::Serialize;
use serde_json::{json, Value};

use crate::arch;
use crate::bench::calibration::{JudgeEvalRequest, SampleRequest};
use crate::bench::dump::DumpRequest;
use crate::bench::judge::{JudgeMode, JudgeSetup, RefusalJudging};
use crate::bench::logits_metrics::LogitsRequest;
use crate::bench::markers::Markers;
use crate::bench::native::NativeGenerator;
use crate::bench::ppl_runner::{PplBaseline, PplRequest};
use crate::bench::prompt::Template;
use crate::bench::refusal::RefusalRequest;
use crate::bench::report::{self as bench_report, BenchRun};
use crate::bench::score::{Backend, EvalRequest};
use crate::bench::{self, BenchError};
use crate::cli::{
    ApplyArgs, AutotuneArgs, BackendCommand, BenchArgs, BenchCommand, CheckArgs, Cli, Commands,
    EvalArgs, ExtractArgs, ExtractCommand, InspectArgs, JudgeArgs, MarkerArgs,
};
use crate::engine::stats::TensorStats;
use crate::engine::{
    self, BroadcastAxis, DType, EngineError, LayerSpec, ReadOnlyModel, ShiftOp, ShiftStats,
    WritableModel,
};
use crate::eval::{
    ensure_gguf, export_if_healthy, is_gguf, load_prompts, locate_binary, run_eval, CommandRunner,
    EvalConfig, EvalError, EvalPrompt, EvalReport, EvalVerdict, GenerationOutcome,
    GgufExportConfig, LlamaCliGenerator, PromptStatus, SystemCommandRunner, TextGenerator,
    ALLOWED_QUANTS, LLAMA_DIR_REQUIRED,
};
use crate::extract::{
    self, ExtractError, ExtractEvent, ExtractObserver, ExtractReport, ExtractRequest,
    MethodDiagnostics, PromptSet,
};
use crate::hardware::{
    self, BudgetPolicy, HardwareBudget, HardwareError, HardwareProbe, SystemProbe, Verdict,
};
use crate::kernel::{self, IoConfig, IoMeter, KernelChoice, Level};
use crate::patch::autotune::{self, AutotuneError, AutotuneEvent, AutotuneRequest, AutotuneRun};
use crate::patch::gguf::{self, GgufApplyArgs, GgufApplyError, Requant};
use crate::patch::{
    self, apply_patch, apply_patch_with, load_patch, untie, validate_patch, ApplyOptions,
    ApplyReport, Axis, BackupSet, LayerSelection, OpStats, OperationKind, PatchBackend, PatchError,
    PatchFile, Projection, RollbackHandle, ValidationIssue, ValidationOptions,
};
use crate::plugin::{unknown_method_message, Diagnostics, Registry, DEFAULT_METHOD};
use crate::ui::{self, ColorMode, Ui, UiConfig};

pub mod exit {
    use crate::patch::exit_code;

    pub const OK: i32 = exit_code::OK;
    pub const GENERAL: i32 = 1;
    pub const INVALID_INPUT: i32 = exit_code::INVALID_PATCH;
    pub const EXECUTION_FAILED: i32 = exit_code::EXECUTION_FAILED;
    pub const CRITICAL: i32 = exit_code::CRITICAL;
    pub const EVAL_UNHEALTHY: i32 = 5;
    pub const EXPORT_FAILED: i32 = 6;
    /// `extract`: config, tokenizer or weights invalid, degenerate direction, NaN/Inf.
    pub const EXTRACT_FAILED: i32 = 7;
    /// `bench`: process failed, unreadable output, incompatible shapes, generation failed.
    pub const BENCH_FAILED: i32 = 8;
    /// `bench --fail-on-warn`: a threshold was exceeded (the report is still in `data`).
    pub const THRESHOLD_EXCEEDED: i32 = 9;
    pub const INTERNAL: i32 = 70;
}

const GIB: f64 = (1u64 << 30) as f64;
/// Per prompt; generous because CPU-only generation of 256 tokens can be slow.
const GENERATION_TIMEOUT: Duration = Duration::from_secs(600);
const INTERRUPTED: &str = "interrompu (Ctrl-C)";

pub type GeneratorFactory =
    Box<dyn Fn(&Path, &EvalArgs) -> std::result::Result<Box<dyn TextGenerator>, EvalError>>;

pub struct AppDeps {
    pub probe: Box<dyn HardwareProbe>,
    pub runner: Box<dyn CommandRunner>,
    /// Builds a generator for a GGUF file.
    pub generator_factory: GeneratorFactory,
    /// Set by the Ctrl-C handler.
    pub interrupted: Arc<AtomicBool>,
    /// Color modes; `run_with` replaces both with `Off` under `--json`.
    pub ui: UiConfig,
}

/// Real hardware, real processes, and a Ctrl-C handler that only raises `interrupted`.
pub fn real_deps() -> AppDeps {
    let interrupted = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&interrupted);
    if let Err(e) = ctrlc::set_handler(move || flag.store(true, Ordering::SeqCst)) {
        let _ = writeln!(
            std::io::stderr(),
            "avertissement : Ctrl-C non intercepté ({e}), une interruption couperait net"
        );
    }
    AppDeps {
        probe: Box::new(SystemProbe::detect()),
        runner: Box::new(SystemCommandRunner),
        generator_factory: Box::new(real_generator),
        interrupted,
        ui: Ui::detect(false),
    }
}

/// Prefers `llama-completion` (raw completion in recent llama.cpp), else `llama-cli`.
/// `-no-cnv` disables chat mode on both; very old builds without it will reject it.
fn real_generator(
    gguf: &Path,
    args: &EvalArgs,
) -> std::result::Result<Box<dyn TextGenerator>, EvalError> {
    let dir = args.llama_cpp_dir.as_deref();
    let binary =
        locate_binary("llama-completion", dir).or_else(|_| locate_binary("llama-cli", dir))?;
    Ok(Box::new(LlamaCliGenerator {
        llama_cli_path: binary,
        model_path: gguf.to_path_buf(),
        extra_args: vec!["-no-cnv".into()],
        timeout: GENERATION_TIMEOUT,
        sampling: None,
    }))
}

// ---------------------------------------------------------------- panic handling

static CATCHING: AtomicBool = AtomicBool::new(false);
static LAST_PANIC: Mutex<Option<String>> = Mutex::new(None);

/// Inside `run_with` the panic text is kept for the error envelope (code 70); elsewhere
/// it is printed. Either way the user gets one readable message, never a raw backtrace.
pub fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let text = info.to_string();
        if CATCHING.load(Ordering::SeqCst) {
            if let Ok(mut last) = LAST_PANIC.lock() {
                *last = Some(text);
            }
        } else {
            let _ = writeln!(std::io::stderr(), "erreur interne : {text}");
        }
    }));
}

pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> String {
    LAST_PANIC
        .lock()
        .ok()
        .and_then(|mut last| last.take())
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_owned()))
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panic sans message".into())
}

// ---------------------------------------------------------------- entry points

/// Writers and terminal styles of one run. Under `--json` both `Ui`s are `Off`.
struct Io<'a> {
    out: &'a mut dyn Write,
    err: &'a mut dyn Write,
    ui: Ui,
    err_ui: Ui,
    err_is_tty: bool,
}

impl<'a> Io<'a> {
    fn new(out: &'a mut dyn Write, err: &'a mut dyn Write, cfg: &UiConfig, json: bool) -> Self {
        let mode = |m: ColorMode| if json { Ui::off() } else { Ui::new(m) };
        Self {
            out,
            err,
            ui: mode(cfg.out),
            err_ui: mode(cfg.err),
            err_is_tty: cfg.err_is_tty && !json,
        }
    }

    /// One line on stdout; nothing at all in `Off` mode.
    fn line(&mut self, line: &str) {
        self.ui.line(self.out, line);
    }
}

const PLAIN_UI: UiConfig = UiConfig {
    out: ColorMode::Plain,
    err: ColorMode::Plain,
    err_is_tty: false,
};

/// Parses the arguments. `Err(code)` means the help, the version or a usage error has
/// already been written (as a JSON envelope when `--json` is present).
pub fn parse_cli<I, T>(args: I, out: &mut dyn Write, err: &mut dyn Write) -> Result<Cli, i32>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    parse_cli_with(args, &PLAIN_UI, false, out, err)
}

/// [`parse_cli`] with the terminal identity: the banner before the help when `banner`
/// (main passes "stdout is a TTY and no --json"), and clap's styled help when `ui.out`
/// has colors.
pub fn parse_cli_with<I, T>(
    args: I,
    ui: &UiConfig,
    banner: bool,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> Result<Cli, i32>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString>,
{
    use clap::error::ErrorKind;

    let args: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let json = args.iter().skip(1).any(|a| a == "--json");
    let e = match Cli::try_parse_from(&args) {
        Ok(cli) => return Ok(cli),
        Err(e) => e,
    };
    let mut io = Io::new(out, err, ui, json);
    let help = matches!(
        e.kind(),
        ErrorKind::DisplayHelp | ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
    );
    if help || e.exit_code() == 0 {
        if help && banner {
            let _ = ui::print_banner(&io.ui, io.out);
        }
        let _ = match io.ui.mode() {
            ColorMode::Ansi256 | ColorMode::TrueColor => write!(io.out, "{}", e.render().ansi()),
            _ => write!(io.out, "{e}"),
        };
        return Err(e.exit_code());
    }
    let failure = Outcome::fail(exit::INVALID_INPUT, e.to_string().trim_end());
    Err(emit(&mut io, json, None, failure))
}

pub fn run_with(mut cli: Cli, deps: &AppDeps, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    // Raw outputs: no envelope, no UI, whatever `--json` says. Routed before the panic
    // catcher so that, in `mcp` mode, the hook reports on stderr and never on stdout.
    match &cli.command {
        Commands::Schema(args) => return crate::mcp::write_schema(args.kind, out),
        Commands::Mcp(args) => return crate::mcp::run_stdio(args, deps, out, err),
        // The TUI owns the terminal and has no envelope; it restores the terminal and
        // catches its own panics (code 70).
        Commands::Tui(args) => return crate::tui::run_cli(args, cli.json, deps, out, err),
        _ => {}
    }
    let mut io = Io::new(out, err, &deps.ui, cli.json);
    // The user file fills the flags the command line left out; a broken file or an unusable
    // backend stops here, before any work.
    if let Err(message) = crate::backend::apply_user_config(&mut cli.command) {
        let failure = Outcome::fail(exit::INVALID_INPUT, message).with_kind("invalid_input");
        return emit(&mut io, cli.json, Some(command_name(&cli.command)), failure);
    }
    CATCHING.store(true, Ordering::SeqCst);
    // A panic unwinds through BackupSet::drop, which rolls back an uncommitted patch.
    let outcome = catch_unwind(AssertUnwindSafe(|| dispatch(&cli.command, deps, &mut io)))
        .unwrap_or_else(|payload| {
            Outcome::fail(
                exit::INTERNAL,
                format!("erreur interne : {}", panic_message(payload.as_ref())),
            )
            .with_kind("internal")
        });
    CATCHING.store(false, Ordering::SeqCst);
    emit(&mut io, cli.json, Some(command_name(&cli.command)), outcome)
}

fn command_name(command: &Commands) -> &'static str {
    match command {
        Commands::Check(_) => "check",
        Commands::Inspect(_) => "inspect",
        Commands::Apply(_) => "apply",
        Commands::Eval(_) => "eval",
        Commands::Extract(_) => "extract",
        Commands::Schema(_) => "schema",
        Commands::Tui(_) => "tui",
        Commands::Mcp(_) => "mcp",
        Commands::Autotune(_) => "autotune",
        Commands::Diff(_) => "diff",
        Commands::Lora(args) => match args.command {
            crate::lora::cli_args::LoraCommand::Merge(_) => "lora merge",
            crate::lora::cli_args::LoraCommand::Extract(_) => "lora extract",
        },
        Commands::Plugins(_) => "plugins list",
        Commands::Graph(_) => "graph",
        Commands::Arch(args) => match args.command {
            crate::cli::ArchCommand::List => "arch list",
            crate::cli::ArchCommand::Detect(_) => "arch detect",
        },
        Commands::Backend(args) => match args.command {
            BackendCommand::Detect { .. } => "backend detect",
            BackendCommand::Show => "backend show",
        },
        Commands::Pipeline(args) => match args.command {
            crate::cli::PipelineCommand::Plan(_) => "pipeline plan",
            crate::cli::PipelineCommand::Run(_) => "pipeline run",
            crate::cli::PipelineCommand::Status(_) => "pipeline status",
            crate::cli::PipelineCommand::Cancel(_) => "pipeline cancel",
            crate::cli::PipelineCommand::Report(_) => "pipeline report",
            crate::cli::PipelineCommand::Promote(_) => "pipeline promote",
        },
        Commands::Bench(args) => match args.command {
            BenchCommand::Logits(_) => "bench logits",
            BenchCommand::Ppl(_) => "bench ppl",
            BenchCommand::Eval(_) => "bench eval",
            BenchCommand::Dump(_) => "bench dump",
            BenchCommand::Refusal(_) => "bench refusal",
            BenchCommand::JudgeSample(_) => "bench judge-sample",
            BenchCommand::JudgeEval(_) => "bench judge-eval",
        },
    }
}

/// `surgeon extract`: the method parameters of the command line (the labels file is read
/// here, before anything is loaded), then the extraction.
fn extract_cli(cmd: &ExtractCommand, deps: &AppDeps, io: &mut Io) -> CoreResult {
    let labels = match &cmd.method.labels_file {
        Some(path) => Some(extract::load_labels(path).map_err(|message| {
            let message = format!("--labels-file {} : {message}", path.display());
            Outcome::fail(exit::INVALID_INPUT, message).with_kind("invalid_input")
        })?),
        None => None,
    };
    let rank = cmd.method.rank.map(|r| r as usize);
    let ui = io.ui;
    core_extract_with(
        &cmd.args,
        &cmd.method.method,
        &extract::params_with_subspace(&cmd.method.options(), rank, labels),
        deps,
        &ui,
        &mut CliSink::new(io),
    )
}

fn dispatch(command: &Commands, deps: &AppDeps, io: &mut Io) -> Outcome {
    let result = match command {
        Commands::Check(args) => core_check(args, deps, &io.ui),
        Commands::Inspect(args) => core_inspect(args, deps, &io.ui),
        Commands::Apply(args) => apply(args, deps, io),
        Commands::Eval(args) => eval(args, deps, &io.ui),
        Commands::Extract(cmd) => extract_cli(cmd, deps, io),
        Commands::Plugins(_) => plugins_list(&io.ui),
        Commands::Arch(args) => match &args.command {
            crate::cli::ArchCommand::List => arch_list(&io.ui),
            crate::cli::ArchCommand::Detect(args) => arch_detect(args, &io.ui),
        },
        Commands::Autotune(args) => {
            let ui = io.ui;
            autotune_cli(args, deps, &ui, &mut CliSink::new(io))
        }
        Commands::Bench(args) => bench_cli(args, deps, io),
        Commands::Diff(args) => crate::surgery::core_diff(args, deps, &mut CliSink::new(io)),
        Commands::Lora(args) => crate::lora::core_lora(args, deps, &mut CliSink::new(io)),
        Commands::Pipeline(args) => pipeline_cli(&args.command, deps),
        Commands::Graph(args) => crate::graph::core_graph(args, deps, &mut CliSink::new(io)),
        Commands::Backend(args) => backend_cli(&args.command, io),
        // Routed by run_with before dispatch: they have no envelope.
        Commands::Schema(_) | Commands::Mcp(_) | Commands::Tui(_) => Err(Outcome::fail(
            exit::INTERNAL,
            "commande sans enveloppe JSON",
        )),
    };
    result.unwrap_or_else(|failure| failure)
}

/// `surgeon pipeline`: the CLI side of `pipeline::front`; the `--allow-*` options exist
/// here only, never over MCP.
fn pipeline_cli(command: &crate::cli::PipelineCommand, deps: &AppDeps) -> CoreResult {
    use crate::cli::PipelineCommand as C;
    use crate::pipeline::front;
    use crate::pipeline::run::RunOptions;
    match command {
        C::Plan(a) => {
            let options = RunOptions {
                allow_disable_safety: a.allow_disable_safety,
                allow_test_reuse: false,
            };
            front::core_plan(
                &a.config,
                &a.job_dir,
                &options,
                a.backend.config(),
                !a.no_calibration,
                true,
                deps,
            )
        }
        C::Run(a) => {
            let options = RunOptions {
                allow_disable_safety: a.allow_disable_safety,
                allow_test_reuse: a.allow_test_reuse,
            };
            match a.detach {
                true => front::core_detach(
                    a.config.as_deref(),
                    &a.job_dir,
                    &options,
                    &a.backend.config(),
                ),
                false => front::core_run(
                    a.config.as_deref(),
                    &a.job_dir,
                    options,
                    a.backend.config(),
                    a.watch_stdin,
                    deps,
                ),
            }
        }
        C::Status(a) => front::core_status(&a.job_dir),
        C::Cancel(a) => front::core_cancel(&a.job_dir),
        C::Report(a) => front::core_report(&a.job_dir),
        C::Promote(a) => front::core_promote(&a.job_dir, deps),
    }
}

// ---------------------------------------------------------------- output

/// Result of one command core, successful or not. Cores return [`CoreResult`] so `?`
/// works through the `From` conversions below; [`envelope_of`] is the only JSON form,
/// shared by the CLI `--json` output and the MCP tool results.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub exit_code: i32,
    pub data: Value,
    pub warnings: Vec<String>,
    /// Boxed: keeps `Result<Outcome, Outcome>` small on the success path.
    pub error: Option<Box<ErrorBody>>,
    /// Terminal rendering (empty when built with `Ui::off()`), never part of the envelope.
    human: String,
}

pub type CoreResult = Result<Outcome, Outcome>;

#[derive(Debug, Clone, Serialize)]
pub struct ErrorBody {
    pub message: String,
    /// Stable machine-readable category (`invalid_patch`, `path_not_allowed`...).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub issues: Vec<ValidationIssue>,
}

/// `{"ok","command","exit_code","data","error","warnings"}`, in that order.
#[derive(Debug, Serialize)]
pub struct Envelope<'a> {
    pub ok: bool,
    pub command: Option<&'a str>,
    pub exit_code: i32,
    pub data: &'a Value,
    pub error: Option<&'a ErrorBody>,
    pub warnings: &'a [String],
}

pub fn envelope_of<'a>(command: Option<&'a str>, outcome: &'a Outcome) -> Envelope<'a> {
    Envelope {
        ok: outcome.is_ok(),
        command,
        exit_code: outcome.exit_code,
        data: &outcome.data,
        error: outcome.error.as_deref(),
        warnings: &outcome.warnings,
    }
}

impl Outcome {
    pub fn ok(data: Value, human: String) -> Self {
        Self {
            exit_code: exit::OK,
            data,
            warnings: Vec::new(),
            error: None,
            human,
        }
    }

    pub fn fail(code: i32, message: impl Display) -> Self {
        Self {
            exit_code: code,
            data: Value::Null,
            warnings: Vec::new(),
            error: Some(Box::new(ErrorBody {
                message: message.to_string(),
                kind: None,
                issues: Vec::new(),
            })),
            human: String::new(),
        }
    }

    /// Sets `error.kind`; no effect on a success.
    pub fn with_kind(mut self, kind: &str) -> Self {
        if let Some(error) = self.error.as_mut() {
            error.kind = Some(kind.to_owned());
        }
        self
    }

    fn with_data(mut self, data: Value, human: String) -> Self {
        self.data = data;
        self.human = human;
        self
    }

    pub fn is_ok(&self) -> bool {
        self.exit_code == exit::OK
    }
}

impl From<PatchError> for Outcome {
    fn from(e: PatchError) -> Self {
        let code = i32::from(&e);
        let kind = match (&e, code) {
            (
                PatchError::Invalid(_)
                | PatchError::InvalidJson { .. }
                | PatchError::PatchUnreadable { .. },
                _,
            ) => "invalid_patch",
            (_, exit::CRITICAL) => "critical",
            _ => "execution_failed",
        };
        let mut outcome = Outcome::fail(code, &e).with_kind(kind);
        if let (PatchError::Invalid(issues), Some(body)) = (&e, outcome.error.as_mut()) {
            body.issues = issues.clone();
        }
        outcome
    }
}

impl From<HardwareError> for Outcome {
    fn from(e: HardwareError) -> Self {
        let kind = match &e {
            HardwareError::PathNotFound(_) => "model_not_found",
            HardwareError::InsufficientResources { .. } => "insufficient_resources",
            _ => "model_error",
        };
        Outcome::fail(exit::GENERAL, e).with_kind(kind)
    }
}

impl From<EngineError> for Outcome {
    fn from(e: EngineError) -> Self {
        Outcome::fail(exit::GENERAL, e).with_kind("model_error")
    }
}

/// Write errors are ignored on purpose: a closed pipe must not turn into a panic, and the
/// exit code still reports the real outcome.
fn emit(io: &mut Io, json: bool, command: Option<&str>, outcome: Outcome) -> i32 {
    if json {
        let _ = serde_json::to_writer(&mut *io.out, &envelope_of(command, &outcome));
        let _ = writeln!(io.out);
    } else {
        let _ = write!(io.out, "{}", outcome.human);
        if let Some(e) = &outcome.error {
            let text = error_text(io, &e.message);
            let _ = writeln!(io.err, "{text}");
        }
    }
    let _ = io.out.flush();
    outcome.exit_code
}

/// `[ERREUR]` badge on an interactive stderr. A redirected stderr keeps the stable
/// `erreur : <message>` line that scripts (and the prompt 5 tests) rely on.
fn error_text(io: &Io, message: &str) -> String {
    let ui = &io.err_ui;
    if !io.err_is_tty || ui.is_off() {
        return format!("erreur : {message}");
    }
    let mut lines = message.lines();
    let first = lines.next().unwrap_or_default();
    std::iter::once(format!(
        "{} {}",
        ui::badge(ui, "ERREUR"),
        ui::metric(ui, first)
    ))
    .chain(lines.map(|l| ui::dim(ui, l)))
    .collect::<Vec<_>>()
    .join("\n")
}

fn to_value(value: &impl Serialize) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

/// `86.0 MiB`, `14.96 GiB`: binary units.
fn size(bytes: u64) -> String {
    let b = bytes as f64;
    match bytes {
        0..=1023 => format!("{bytes} B"),
        1024..=1_048_575 => format!("{:.1} KiB", b / 1024.0),
        1_048_576..=1_073_741_823 => format!("{:.1} MiB", b / 1_048_576.0),
        _ => format!("{:.2} GiB", b / GIB),
    }
}

// ---------------------------------------------------------------- check

/// Hardware budget; human text only when `ui` is not `Off`.
pub fn core_check(args: &CheckArgs, deps: &AppDeps, ui: &Ui) -> CoreResult {
    let mut policy = BudgetPolicy::default();
    if let Some(margin) = args.policy_ram_margin_gib {
        if !(margin.is_finite() && margin >= 0.0) {
            return Err(Outcome::fail(
                exit::INVALID_INPUT,
                "--policy-ram-margin-gib doit être un nombre positif",
            )
            .with_kind("invalid_input"));
        }
        policy.ram_safety_margin_bytes = (margin * GIB) as u64;
    }
    let budget = hardware::evaluate_budget(&args.model, deps.probe.as_ref(), &policy)?;
    let human = budget_text(ui, &args.model, &budget);
    match &budget.verdict {
        Verdict::Insufficient { reason } => Err(Outcome::fail(
            exit::GENERAL,
            format!("ressources insuffisantes : {reason}"),
        )
        .with_kind("insufficient_resources")
        .with_data(to_value(&budget), human)),
        _ => Ok(Outcome::ok(to_value(&budget), human)),
    }
}

fn budget_text(ui: &Ui, model: &Path, b: &HardwareBudget) -> String {
    let gpu = b.best_gpu.as_ref().map_or("aucun".to_owned(), |g| {
        format!(
            "{} · {} libres / {}",
            g.name,
            size(g.free_vram_bytes),
            size(g.total_vram_bytes)
        )
    });
    let need = format!(
        "RAM {} · VRAM {}",
        size(b.required_ram_bytes),
        b.required_vram_bytes.map_or("n/a".to_owned(), size)
    );
    let verdict = match &b.verdict {
        Verdict::FitsInVram => ui::metric(ui, "FITS IN VRAM"),
        Verdict::FitsInRamOnly => ui::metric(ui, "FITS IN RAM ONLY"),
        Verdict::Insufficient { reason } => {
            format!("{} {}", ui::metric(ui, "INSUFFICIENT"), ui::dim(ui, reason))
        }
    };
    let mut lines = vec![
        format!(
            "{} {}",
            ui::badge(ui, "CHECK"),
            ui::metric(ui, "weight-surgeon")
        ),
        ui::separator(ui),
        ui::field(ui, "modèle", &ui::path(ui, model)),
        ui::key_val(ui, "taille", &size(b.model_bytes)),
        ui::key_val(
            ui,
            "RAM",
            &format!("{} libres", size(b.available_ram_bytes)),
        ),
        ui::key_val(ui, "GPU", &gpu),
        ui::key_val(ui, "besoin", &need),
        format!("{} {verdict}", ui::badge(ui, "VERDICT")),
    ];
    lines.extend(b.warnings.iter().map(|w| warning_line(ui, w)));
    lines.join("\n") + "\n"
}

fn warning_line(ui: &Ui, warning: &str) -> String {
    format!("  {}", ui::dim(ui, &format!("! {warning}")))
}

// ---------------------------------------------------------------- inspect

#[derive(Debug, Clone, Serialize)]
struct InspectedTensor {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    shard: PathBuf,
    stats: Option<TensorStats>,
}

/// MLP tensor topology (and streamed statistics); human text only when `ui` is not `Off`.
pub fn core_inspect(args: &InspectArgs, deps: &AppDeps, ui: &Ui) -> CoreResult {
    if gguf::is_gguf(&args.model) {
        let spec = match &args.arch_spec {
            Some(path) => Some(arch::load_spec(path).map_err(arch_failure)?),
            None => None,
        };
        let shown = gguf::report::inspect(
            &args.model,
            &args.layers,
            &args.projections,
            args.show_stats,
            spec.as_ref(),
            ui,
        );
        return shown
            .map(|(data, human)| Outcome::ok(data, human))
            .map_err(|e| Outcome::fail(exit::GENERAL, e).with_kind("invalid_gguf"));
    }
    let shards = hardware::compute_model_size(&args.model)?.shard_files;
    let resolution =
        arch::resolve(&args.model, None, args.arch_spec.as_deref()).map_err(arch_failure)?;
    let layers = LayerSpec::from(&args.layers);
    let projs = match args.projections.is_empty() {
        true => vec![Projection::Gate, Projection::Up, Projection::Down],
        false => args.projections.clone(),
    };
    let sharded = shards.len() > 1;
    let mut tensors = Vec::new();
    let mut rows = Vec::new();
    let mut all_names = Vec::new();
    for shard in &shards {
        let model = ReadOnlyModel::open(shard)?;
        all_names.extend(model.index().tensors.iter().map(|t| t.name.clone()));
        for info in find_targets(model.index(), &resolution.spec, &layers, &projs) {
            if deps.interrupted.load(Ordering::SeqCst) {
                return Err(Outcome::fail(exit::GENERAL, INTERRUPTED).with_kind("cancelled"));
            }
            let stats = match args.show_stats {
                // Unsupported dtypes have no statistics (shown as "—").
                true => {
                    let view = model.tensor(&info.name)?;
                    let never = AtomicBool::new(false);
                    engine::stats::tensor_stats(view.bytes, &info.dtype, &info.shape, 0, &never)
                        .ok()
                }
                false => None,
            };
            let mut row = vec![
                ui::accent(ui, &layer_label(&info.name)),
                ui::metric(ui, projection_label(&info.name)),
                ui::metric(ui, info.dtype.to_string().to_lowercase()),
                ui::metric(ui, shape_label(&info.shape)),
                ui::metric(ui, size((info.byte_end - info.byte_start) as u64)),
            ];
            if args.show_stats {
                row.extend(stats_cells(ui, stats.as_ref()));
            }
            if sharded {
                let file = shard.file_name().map(|f| f.to_string_lossy());
                row.push(ui::dim(ui, &file.unwrap_or_default()));
            }
            rows.push(row);
            tensors.push(InspectedTensor {
                name: info.name.clone(),
                dtype: info.dtype.clone(),
                shape: info.shape.clone(),
                shard: shard.clone(),
                stats,
            });
        }
    }

    let mut banner = Vec::new();
    let _ = ui::print_banner(ui, &mut banner);
    let mut human = String::from_utf8_lossy(&banner).into_owned();
    let report = arch_report(&resolution, &args.model, &all_names);
    human.push_str(&format!(
        "{} {} tenseurs sélectionnés\n",
        ui::badge(ui, "INSPECT"),
        ui::metric(ui, rows.len()),
    ));
    // Nothing detected: the historical naming, and the output it always had.
    if report.detected {
        human.push_str(&format!(
            "{} {}\n",
            ui::badge(ui, "ARCH"),
            ui::text(ui, &report.summary())
        ));
    }
    human.push_str(&format!("{}\n", ui::separator(ui)));
    if rows.is_empty() {
        human.push_str(&ui::dim(
            ui,
            "aucun tenseur MLP ne correspond à la sélection",
        ));
        human.push('\n');
    } else {
        let mut headers = vec!["COUCHE", "PROJ", "DTYPE", "SHAPE", "TAILLE"];
        if args.show_stats {
            headers.extend(["MIN", "MAX", "MOYENNE", "L2"]);
        }
        if sharded {
            headers.push("SHARD");
        }
        human.push_str(&ui::table(ui, &headers, &rows));
    }
    Ok(Outcome::ok(
        json!({ "shards": shards, "tensors": tensors, "arch": report }),
        human,
    ))
}

/// `arch::Report` of a SafeTensors model: every tensor name, tied rule from config.json.
fn arch_report(resolution: &arch::Resolution, model: &Path, names: &[String]) -> arch::Report {
    let tied = arch::read_config(model).map(|c| resolution.spec.tied_in_config(&c));
    arch::Report::new(
        resolution,
        names.iter().map(String::as_str),
        arch::Format::SafeTensors,
        tied,
    )
}

fn arch_failure(e: arch::ArchError) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, e.to_string()).with_kind(e.kind())
}

/// `[L12]`, or `[—]` for a tensor outside numbered layers.
fn layer_label(name: &str) -> String {
    engine::parse_layer_index(name).map_or("[—]".to_owned(), |l| format!("[L{l}]"))
}

fn projection_label(name: &str) -> &str {
    name.split('.')
        .find(|part| part.ends_with("_proj") || *part == "embed_tokens")
        .unwrap_or("—")
}

fn shape_label(shape: &[usize]) -> String {
    shape
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(" × ")
}

fn stats_cells(ui: &Ui, stats: Option<&TensorStats>) -> Vec<String> {
    let Some(s) = stats else {
        // Unsupported dtype: no decoding, so no statistics.
        return vec![ui::dim(ui, "—"); 4];
    };
    let mut l2 = ui::metric(ui, format!("{:.4}", s.l2));
    if s.non_finite > 0 {
        l2.push_str(&ui::dim(ui, &format!(" !{} NaN/Inf", s.non_finite)));
    }
    vec![
        ui::metric(ui, format!("{:.4}", s.min)),
        ui::metric(ui, format!("{:.4}", s.max)),
        ui::metric(ui, format!("{:.4}", s.mean)),
        l2,
    ]
}

// ---------------------------------------------------------------- apply

impl From<&LayerSelection> for LayerSpec {
    fn from(selection: &LayerSelection) -> Self {
        match selection {
            LayerSelection::All => LayerSpec::All,
            LayerSelection::List(layers) => LayerSpec::List(layers.clone()),
            LayerSelection::Range {
                start,
                end_inclusive,
            } => LayerSpec::Range {
                start: *start,
                end_inclusive: *end_inclusive,
            },
        }
    }
}

/// Tensors of one shard matching a selection, by the roles of `spec`: the embeddings first
/// (they belong to no layer, so `layers` does not filter them), then layer by layer the MLP
/// projections (gate, up, down) and `o_proj`.
fn find_targets<'a>(
    index: &'a engine::ModelIndex,
    spec: &arch::ArchSpec,
    layers: &LayerSpec,
    projections: &[Projection],
) -> Vec<&'a engine::TensorInfo> {
    let roles: Vec<arch::Role> = projections.iter().map(|p| p.role()).collect();
    engine::find_by_roles(index, spec, &roles, layers)
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

/// `project_out` is not a shift: it goes through `PatchBackend::project_out_tensor`.
fn shift_op(op: OperationKind) -> Option<ShiftOp> {
    match op {
        OperationKind::Subtract => Some(ShiftOp::Subtract),
        OperationKind::Add => Some(ShiftOp::Add),
        OperationKind::ProjectOut => None,
    }
}

impl From<Axis> for BroadcastAxis {
    fn from(axis: Axis) -> Self {
        match axis {
            Axis::Rows => BroadcastAxis::Rows,
            Axis::Cols => BroadcastAxis::Cols,
        }
    }
}

impl From<ShiftStats> for OpStats {
    fn from(s: ShiftStats) -> Self {
        OpStats {
            max_abs_delta: s.max_abs_delta,
            elements_modified: s.elements_modified,
            nan_or_inf_produced: s.nan_or_inf_produced,
        }
    }
}

fn backend_error(e: EngineError) -> PatchError {
    PatchError::Backend(e.to_string())
}

/// `PatchBackend` over the engine, for a single file or every shard of a sharded model.
pub struct EngineBackend {
    shards: Vec<(PathBuf, WritableModel)>,
    interrupted: Arc<AtomicBool>,
    spec: arch::ArchSpec,
}

impl EngineBackend {
    pub fn open(shard_paths: &[PathBuf], interrupted: Arc<AtomicBool>) -> engine::Result<Self> {
        let shards = shard_paths
            .iter()
            .map(|p| Ok((p.clone(), WritableModel::open(p)?)))
            .collect::<engine::Result<_>>()?;
        Ok(Self {
            shards,
            interrupted,
            spec: arch::legacy().clone(),
        })
    }

    /// Selects and orients tensors by `spec` (the default naming otherwise).
    pub fn with_arch(mut self, spec: arch::ArchSpec) -> Self {
        self.spec = spec;
        self
    }

    /// Writer biases (`.bias` next to a selected writer): they are not projected.
    pub fn writer_biases<'n>(&self, names: impl IntoIterator<Item = &'n str>) -> Vec<String> {
        names
            .into_iter()
            .filter(|n| {
                self.spec
                    .role_of(n, arch::Format::SafeTensors)
                    .is_some_and(|r| r.role.is_writer())
            })
            .filter_map(|n| n.strip_suffix(".weight").map(|stem| format!("{stem}.bias")))
            .filter(|bias| self.shard_of(bias).is_some())
            .collect()
    }

    fn layout_of(&self, name: &str) -> Option<arch::Layout> {
        self.spec
            .role_of(name, arch::Format::SafeTensors)
            .map(|r| r.layout)
    }

    fn shard_of(&self, name: &str) -> Option<usize> {
        self.shards
            .iter()
            .position(|(_, model)| model.index().get(name).is_ok())
    }

    /// Shards holding at least one of `names`: only those need a backup.
    pub fn shards_holding<'n>(&self, names: impl IntoIterator<Item = &'n str>) -> Vec<PathBuf> {
        let mut indices: Vec<usize> = names.into_iter().filter_map(|n| self.shard_of(n)).collect();
        indices.sort_unstable();
        indices.dedup();
        indices
            .into_iter()
            .map(|i| self.shards[i].0.clone())
            .collect()
    }

    /// Chunking / prefetch of every shard; the Ctrl-C flag is also checked between chunks.
    pub fn set_io(&mut self, io: IoConfig) {
        for (_, model) in &mut self.shards {
            model.set_io(io, Some(Arc::clone(&self.interrupted)));
        }
    }

    pub fn io_counters(&self) -> kernel::IoCounters {
        let mut total = kernel::IoCounters::default();
        for (_, model) in &self.shards {
            total.merge(model.io_counters());
        }
        total
    }
}

impl PatchBackend for EngineBackend {
    fn resolve_targets(
        &self,
        layers: &LayerSelection,
        projections: &[Projection],
    ) -> patch::Result<Vec<String>> {
        let layers = LayerSpec::from(layers);
        let mut names: Vec<String> = self
            .shards
            .iter()
            .flat_map(|(_, model)| {
                find_targets(model.index(), &self.spec, &layers, projections)
                    .into_iter()
                    .map(|t| t.name.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        let layer = |name: &str| {
            self.spec
                .role_of(name, arch::Format::SafeTensors)
                .and_then(|r| r.layer)
        };
        names.sort_by(|a, b| (layer(a), a).cmp(&(layer(b), b)));
        // A tensor listed by two shards must still be shifted once.
        names.dedup();
        Ok(names)
    }

    fn load_vector(&self, file: &Path, tensor: Option<&str>) -> patch::Result<Vec<f32>> {
        engine::load_direction_vector(file, tensor).map_err(backend_error)
    }

    fn shift_tensor(
        &mut self,
        tensor_name: &str,
        direction: &[f32],
        alpha: f32,
        op: OperationKind,
        axis: Axis,
    ) -> patch::Result<OpStats> {
        // Checked before each tensor: apply_patch turns this error into a rollback.
        if self.interrupted.load(Ordering::SeqCst) {
            return Err(PatchError::Backend(INTERRUPTED.into()));
        }
        let shard = self
            .shard_of(tensor_name)
            .ok_or_else(|| backend_error(EngineError::TensorNotFound(tensor_name.into())))?;
        let op = shift_op(op).ok_or_else(|| {
            PatchError::Backend("project_out passe par project_out_tensor".into())
        })?;
        if self.layout_of(tensor_name) == Some(arch::Layout::ExpertsOutIn) {
            return Err(PatchError::Backend(format!(
                "{tensor_name} : experts fusionnés [experts, out, in] : seul project_out les                  modifie (add/subtract n'a pas d'axe défini sur un tenseur 3D)"
            )));
        }
        self.shards[shard]
            .1
            .shift_tensor(tensor_name, direction, alpha, op, axis.into())
            .map(OpStats::from)
            .map_err(backend_error)
    }

    fn project_out_tensor(
        &mut self,
        tensor_name: &str,
        a: &[f32],
        b: &[f32],
        alpha: f32,
    ) -> patch::Result<OpStats> {
        self.project_out_subspace(tensor_name, a, b, 1, alpha)
    }

    fn load_directions(
        &self,
        file: &Path,
        tensor: Option<&str>,
    ) -> patch::Result<(Option<usize>, Vec<f32>)> {
        engine::load_direction_matrix(file, tensor).map_err(backend_error)
    }

    fn project_out_subspace(
        &mut self,
        tensor_name: &str,
        a: &[f32],
        b: &[f32],
        k: usize,
        alpha: f32,
    ) -> patch::Result<OpStats> {
        if self.interrupted.load(Ordering::SeqCst) {
            return Err(PatchError::Backend(INTERRUPTED.into()));
        }
        let shard = self
            .shard_of(tensor_name)
            .ok_or_else(|| backend_error(EngineError::TensorNotFound(tensor_name.into())))?;
        // The embeddings hold the residual stream in their rows, the other writers in
        // their columns (per expert for fused experts): same eraser, the side the
        // architecture gives.
        let projection = engine::SubspaceProjection { a, b, alpha, k };
        let layout = self.layout_of(tensor_name);
        let model = &mut self.shards[shard].1;
        match layout {
            Some(arch::Layout::ExpertsOutIn) => model.project_out_experts(tensor_name, &projection),
            layout => model.project_out_subspace(
                tensor_name,
                &projection,
                layout == Some(arch::Layout::Rows),
            ),
        }
        .map(OpStats::from)
        .map_err(backend_error)
    }

    fn flush(&mut self) -> patch::Result<()> {
        self.shards
            .iter()
            .try_for_each(|(_, model)| model.flush())
            .map_err(backend_error)
    }
}

/// Progress of a command core. The CLI renders it as the terminal UI, the MCP server as
/// `notifications/progress`, tests ignore it with [`NullSink`].
pub trait ProgressSink {
    fn event(&mut self, event: ProgressEvent);
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProgressEvent {
    ApplyStart {
        patch: PathBuf,
        operations: usize,
    },
    Warn(String),
    DryRun {
        op_id: String,
        tensors: Vec<String>,
        vector_len: usize,
        /// `project_out` with `[k, len]` tensors: k.
        rank: Option<usize>,
    },
    Backup {
        path: PathBuf,
    },
    /// One tensor shifted: `done/total` within the operation, `overall_*` across the patch.
    Shift {
        op_id: String,
        tensor: String,
        done: usize,
        total: usize,
        overall_done: usize,
        overall_total: usize,
    },
    OpDone {
        op_id: String,
        max_delta: f32,
        tensors: usize,
    },
    /// A `project_out` operation finished (same fields as `OpDone`).
    ProjectDone {
        op_id: String,
        max_delta: f32,
        tensors: usize,
    },
    Commit {
        seconds: f64,
    },
    Rollback(RollbackOutcome),
    EvalPrompt {
        id: String,
        done: usize,
        total: usize,
    },
    /// Extraction planned: resources checked, dtype chosen, weights about to load.
    ExtractStart {
        model: PathBuf,
        layer: usize,
        site: String,
        pooling: String,
        dtype: String,
        weight_bytes: u64,
        n_pos: usize,
        n_neg: usize,
    },
    ExtractLoading {
        done: usize,
        total: usize,
    },
    ExtractPrompt {
        set: PromptSet,
        done: usize,
        total: usize,
    },
    ExtractWriting,
    /// `bench dump`: prompts forwarded and written.
    BenchDumpPrompt {
        done: usize,
        total: usize,
    },
    /// `bench logits`: blocks of positions compared.
    BenchLogitsBlock {
        done: usize,
        total: usize,
    },
    /// `bench ppl`: chunks evaluated by llama-perplexity (total when announced).
    BenchPplChunk {
        done: usize,
        total: Option<usize>,
    },
    /// `bench eval`: questions answered (per model when a baseline runs first).
    BenchEvalQuestion {
        done: usize,
        total: usize,
    },
    /// `bench refusal`: prompts answered (per model when a baseline runs first).
    BenchRefusalPrompt {
        done: usize,
        total: usize,
    },
    /// `bench refusal --judge llm|both`, `bench judge-eval`: answers judged.
    BenchJudgeVerdict {
        done: usize,
        total: usize,
    },
    /// `autotune`: forwards done out of the planned total.
    AutotuneForward {
        done: usize,
        total: usize,
    },
    /// `autotune`: responses measured, ridge being solved.
    AutotuneSolve,
    /// `graph`: the model is open, the SVD loop is about to start.
    GraphStart {
        model: PathBuf,
        matrices: usize,
        rank: usize,
    },
    /// `graph`: one matrix analysed (`done/total`).
    GraphNode {
        id: String,
        done: usize,
        total: usize,
    },
    /// `graph`: every writer/reader pair scored, edges selected.
    GraphEdges {
        nodes: usize,
        edges: usize,
    },
    /// `graph`: output files written.
    GraphWrite {
        files: usize,
    },
    /// `graph`: done.
    GraphDone {
        nodes: usize,
        edges: usize,
        seconds: f64,
    },
    /// `.surgery` and LoRA engines (diff, apply, merge, extract).
    Surgery(crate::surgery::SurgeryEvent),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RollbackOutcome {
    Restored,
    /// The backups to restore by hand.
    Failed {
        backups: Vec<PathBuf>,
    },
    /// No backup was taken and the model was left partially modified.
    NoBackup,
}

pub struct NullSink;

impl ProgressSink for NullSink {
    fn event(&mut self, _: ProgressEvent) {}
}

/// Progress decorator: `apply_patch` has no hook, so the events are derived from the
/// backend calls. Every target is resolved (preflight) before the first shift, which gives
/// each operation its total.
struct ObservedBackend<'a, B: PatchBackend> {
    inner: &'a mut B,
    sink: &'a mut dyn ProgressSink,
    op_ids: Vec<String>,
    /// Per operation: `project_out` (shown as `[PROJECT]`) rather than a shift.
    projects: Vec<bool>,
    totals: RefCell<Vec<usize>>,
    op: usize,
    done: usize,
    overall_done: usize,
    max_delta: f32,
}

const BAR_WIDTH: usize = 20;

impl<'a, B: PatchBackend> ObservedBackend<'a, B> {
    fn new(
        inner: &'a mut B,
        sink: &'a mut dyn ProgressSink,
        ops: Vec<(String, OperationKind)>,
    ) -> Self {
        let projects = ops
            .iter()
            .map(|(_, kind)| *kind == OperationKind::ProjectOut)
            .collect();
        Self {
            inner,
            sink,
            op_ids: ops.into_iter().map(|(id, _)| id).collect(),
            projects,
            totals: RefCell::new(Vec::new()),
            op: 0,
            done: 0,
            overall_done: 0,
            max_delta: 0.0,
        }
    }

    fn op_id(&self) -> String {
        self.op_ids.get(self.op).cloned().unwrap_or_default()
    }

    fn total(&self) -> usize {
        self.totals.borrow().get(self.op).copied().unwrap_or(0)
    }

    /// One tensor done: progress, then the end of its operation.
    fn track(&mut self, tensor_name: &str, stats: OpStats) -> OpStats {
        self.done += 1;
        self.overall_done += 1;
        self.max_delta = self.max_delta.max(stats.max_abs_delta);
        let overall_total = self.totals.borrow().iter().sum();
        self.sink.event(ProgressEvent::Shift {
            op_id: self.op_id(),
            tensor: tensor_name.to_owned(),
            done: self.done,
            total: self.total(),
            overall_done: self.overall_done,
            overall_total,
        });
        if self.done >= self.total() {
            let (op_id, max_delta, tensors) = (self.op_id(), self.max_delta, self.done);
            self.sink
                .event(match self.projects.get(self.op).copied().unwrap_or(false) {
                    true => ProgressEvent::ProjectDone {
                        op_id,
                        max_delta,
                        tensors,
                    },
                    false => ProgressEvent::OpDone {
                        op_id,
                        max_delta,
                        tensors,
                    },
                });
            self.op += 1;
            self.done = 0;
            self.max_delta = 0.0;
        }
        stats
    }
}

impl<B: PatchBackend> PatchBackend for ObservedBackend<'_, B> {
    fn resolve_targets(
        &self,
        layers: &LayerSelection,
        projections: &[Projection],
    ) -> patch::Result<Vec<String>> {
        let targets = self.inner.resolve_targets(layers, projections)?;
        self.totals.borrow_mut().push(targets.len());
        Ok(targets)
    }

    fn load_vector(&self, file: &Path, tensor: Option<&str>) -> patch::Result<Vec<f32>> {
        self.inner.load_vector(file, tensor)
    }

    fn shift_tensor(
        &mut self,
        tensor_name: &str,
        direction: &[f32],
        alpha: f32,
        op: OperationKind,
        axis: Axis,
    ) -> patch::Result<OpStats> {
        let stats = self
            .inner
            .shift_tensor(tensor_name, direction, alpha, op, axis)?;
        Ok(self.track(tensor_name, stats))
    }

    fn project_out_tensor(
        &mut self,
        tensor_name: &str,
        a: &[f32],
        b: &[f32],
        alpha: f32,
    ) -> patch::Result<OpStats> {
        let stats = self.inner.project_out_tensor(tensor_name, a, b, alpha)?;
        Ok(self.track(tensor_name, stats))
    }

    fn load_directions(
        &self,
        file: &Path,
        tensor: Option<&str>,
    ) -> patch::Result<(Option<usize>, Vec<f32>)> {
        self.inner.load_directions(file, tensor)
    }

    fn project_out_subspace(
        &mut self,
        tensor_name: &str,
        a: &[f32],
        b: &[f32],
        k: usize,
        alpha: f32,
    ) -> patch::Result<OpStats> {
        let stats = self
            .inner
            .project_out_subspace(tensor_name, a, b, k, alpha)?;
        Ok(self.track(tensor_name, stats))
    }

    fn flush(&mut self) -> patch::Result<()> {
        self.inner.flush()
    }
}

/// Renders apply events as the terminal UI: one line per step on stdout, the in-place bar
/// on an interactive stderr only (a redirected stderr never gets `\r`).
struct CliSink<'a, 'io> {
    io: &'a mut Io<'io>,
    /// Visible width of the progress line currently drawn on stderr (0: none).
    shown: usize,
    extract: ExtractView,
}

/// What the extraction lines need between events.
#[derive(Default)]
struct ExtractView {
    dtype: String,
    weight_bytes: u64,
    n_pos: usize,
    n_neg: usize,
    pos_done: usize,
    neg_done: usize,
}

const FORWARD_BAR_WIDTH: usize = 16;

impl<'a, 'io> CliSink<'a, 'io> {
    fn new(io: &'a mut Io<'io>) -> Self {
        Self {
            io,
            shown: 0,
            extract: ExtractView::default(),
        }
    }

    fn progress(&mut self, op_id: &str, tensor: &str, done: usize, total: usize) {
        let ui = self.io.err_ui;
        let line = format!(
            "{} {} · {} · {}  {}",
            ui::badge(&ui, "SHIFT"),
            ui::metric(&ui, op_id),
            ui::metric(&ui, projection_label(tensor)),
            ui::metric(&ui, format!("{done}/{total}")),
            ui::progress_bar(&ui, done, total, BAR_WIDTH)
        );
        self.draw(&line);
    }

    /// `[FORWARD]  pos 8/16  ████░░░░  ·  neg 0/16`; without bar once every prompt ran.
    fn forward_line(&self, ui: &Ui, bar: bool) -> String {
        let e = &self.extract;
        let pos = ui::metric(ui, format!("pos {}/{}", e.pos_done, e.n_pos));
        let neg = ui::metric(ui, format!("neg {}/{}", e.neg_done, e.n_neg));
        let badge = ui::badge(ui, "FORWARD");
        match bar {
            true => {
                let bar = ui::progress_bar(
                    ui,
                    e.pos_done + e.neg_done,
                    e.n_pos + e.n_neg,
                    FORWARD_BAR_WIDTH,
                );
                format!("{badge} {pos}  {bar}  ·  {neg}")
            }
            false => format!("{badge} {pos} · {neg}"),
        }
    }

    /// `[BENCH]  ppl 12/40  ████░░░░`, in place on an interactive stderr only.
    fn bench_progress(&mut self, label: &str, done: usize, total: Option<usize>) {
        let ui = self.io.err_ui;
        let line = match total {
            Some(total) => format!(
                "{} {} {}  {}",
                ui::badge(&ui, "BENCH"),
                ui::metric(&ui, label),
                ui::metric(&ui, format!("{done}/{total}")),
                ui::progress_bar(&ui, done, total, BAR_WIDTH)
            ),
            None => format!(
                "{} {} {}",
                ui::badge(&ui, "BENCH"),
                ui::metric(&ui, label),
                ui::metric(&ui, done)
            ),
        };
        self.draw(&line);
    }

    /// In-place line on an interactive stderr only.
    fn draw(&mut self, line: &str) {
        if !self.io.err_is_tty || self.io.err_ui.is_off() {
            return;
        }
        let width = ui::visible_width(line);
        let blank = " ".repeat(self.shown.saturating_sub(width));
        let _ = write!(self.io.err, "\r{line}{blank}");
        let _ = self.io.err.flush();
        self.shown = width;
    }

    fn clear_progress(&mut self) {
        if self.shown > 0 {
            let _ = write!(self.io.err, "\r{}\r", " ".repeat(self.shown));
            let _ = self.io.err.flush();
            self.shown = 0;
        }
    }
}

impl Drop for CliSink<'_, '_> {
    fn drop(&mut self) {
        self.clear_progress();
    }
}

impl ProgressSink for CliSink<'_, '_> {
    fn event(&mut self, event: ProgressEvent) {
        let ui = self.io.ui;
        if let ProgressEvent::Shift {
            op_id,
            tensor,
            done,
            total,
            ..
        } = &event
        {
            return self.progress(op_id, tensor, *done, *total);
        }
        if let ProgressEvent::GraphNode { id, done, total } = &event {
            return self.progress("graph", id, *done, *total);
        }
        if let ProgressEvent::Surgery(ev) = &event {
            if let Some((label, name, done, total)) = crate::surgery::events::counter(ev) {
                return self.progress(label, name, done, total);
            }
            self.clear_progress();
            if let Some(line) = crate::surgery::events::render(&ui, ev) {
                self.io.line(&line);
            }
            return;
        }
        let bench = match &event {
            ProgressEvent::BenchDumpPrompt { done, total } => Some(("dump", *done, Some(*total))),
            ProgressEvent::BenchLogitsBlock { done, total } => {
                Some(("logits", *done, Some(*total)))
            }
            ProgressEvent::BenchPplChunk { done, total } => Some(("ppl", *done, *total)),
            ProgressEvent::BenchEvalQuestion { done, total } => Some(("eval", *done, Some(*total))),
            ProgressEvent::BenchRefusalPrompt { done, total } => {
                Some(("refusal", *done, Some(*total)))
            }
            ProgressEvent::BenchJudgeVerdict { done, total } => Some(("juge", *done, Some(*total))),
            ProgressEvent::AutotuneForward { done, total } => {
                Some(("autotune", *done, Some(*total)))
            }
            _ => None,
        };
        if let Some((label, done, total)) = bench {
            return self.bench_progress(label, done, total);
        }
        match &event {
            ProgressEvent::ExtractPrompt { set, done, .. } => {
                match set {
                    PromptSet::Pos => self.extract.pos_done = *done,
                    PromptSet::Neg => self.extract.neg_done = *done,
                }
                if !(*set == PromptSet::Neg && *done >= self.extract.n_neg) {
                    let err_ui = self.io.err_ui;
                    let line = self.forward_line(&err_ui, true);
                    return self.draw(&line);
                }
            }
            ProgressEvent::ExtractLoading { done, total } if done < total => return,
            _ => {}
        }
        self.clear_progress();
        match event {
            ProgressEvent::ApplyStart { patch, operations } => self.io.line(&format!(
                "{} {} · {} opérations",
                ui::badge(&ui, "APPLY"),
                ui::path(&ui, &patch),
                ui::metric(&ui, operations)
            )),
            ProgressEvent::Warn(warning) => self.io.line(&warning_line(&ui, &warning)),
            ProgressEvent::DryRun {
                op_id,
                tensors,
                vector_len,
                rank,
            } => {
                let what = match rank {
                    Some(k) => format!("{k} directions de"),
                    None => "vecteur de".to_owned(),
                };
                self.io.line(&format!(
                    "{} {} · {} tenseurs · {what} {}",
                    ui::badge(&ui, "DRY-RUN"),
                    ui::metric(&ui, &op_id),
                    ui::metric(&ui, tensors.len()),
                    ui::metric(&ui, vector_len)
                ));
                for t in &tensors {
                    self.io.line(&format!(
                        "{} {}",
                        ui::text(&ui, "          "),
                        ui::dim(&ui, t)
                    ));
                }
            }
            // BackupGuard::create re-hashes the copy against the original before returning.
            ProgressEvent::Backup { path } => self.io.line(&format!(
                "{} {}  {}",
                ui::badge(&ui, "BACKUP"),
                ui::path(&ui, &path),
                ui::metric(&ui, "✓ sha256 vérifié")
            )),
            ProgressEvent::OpDone {
                op_id,
                max_delta,
                tensors,
            } => self.io.line(&format!(
                "{} {} terminée · Δmax {} · {} tenseurs",
                ui::badge(&ui, "SHIFT"),
                ui::metric(&ui, op_id),
                ui::metric(&ui, format!("{max_delta:.4}")),
                ui::metric(&ui, tensors)
            )),
            ProgressEvent::ProjectDone {
                op_id,
                max_delta,
                tensors,
            } => self.io.line(&format!(
                "{} {} terminée · effacement a(bᵀW) · Δmax {} · {} tenseurs",
                ui::badge(&ui, "PROJECT"),
                ui::metric(&ui, op_id),
                ui::metric(&ui, format!("{max_delta:.4}")),
                ui::metric(&ui, tensors)
            )),
            ProgressEvent::Commit { seconds } => self.io.line(&format!(
                "{} {} · {}",
                ui::badge(&ui, "COMMIT"),
                ui::metric(&ui, "modèle patché"),
                ui::dim(&ui, &format!("{seconds:.1} s"))
            )),
            ProgressEvent::Rollback(outcome) => self.io.line(&rollback_line(&ui, &outcome)),
            ProgressEvent::ExtractStart {
                model,
                layer,
                site,
                pooling,
                dtype,
                weight_bytes,
                n_pos,
                n_neg,
            } => {
                self.extract = ExtractView {
                    dtype,
                    weight_bytes,
                    n_pos,
                    n_neg,
                    ..ExtractView::default()
                };
                self.io.line(&format!(
                    "{} {} · couche {} · {} · pooling {}  {}",
                    ui::badge(&ui, "EXTRACT"),
                    ui::metric(&ui, "Qwen2"),
                    ui::metric(&ui, layer),
                    ui::metric(&ui, site),
                    ui::metric(&ui, pooling),
                    ui::path(&ui, &model)
                ));
            }
            ProgressEvent::ExtractLoading { done, total } => self.io.line(&format!(
                "{} poids chargés {} tenseurs · {} · {}",
                ui::badge(&ui, "LOAD"),
                ui::metric(&ui, format!("{done}/{total}")),
                ui::metric(&ui, &self.extract.dtype),
                ui::metric(&ui, size(self.extract.weight_bytes))
            )),
            ProgressEvent::ExtractPrompt { .. } => {
                let line = self.forward_line(&ui, false);
                self.io.line(&line);
            }
            ProgressEvent::GraphStart {
                model,
                matrices,
                rank,
            } => self.io.line(&format!(
                "{} {} · {} matrices · rang {}",
                ui::badge(&ui, "GRAPH"),
                ui::path(&ui, &model),
                ui::metric(&ui, matrices),
                ui::metric(&ui, rank)
            )),
            ProgressEvent::GraphEdges { nodes, edges } => self.io.line(&format!(
                "{} {} nœuds · {} arêtes",
                ui::badge(&ui, "EDGES"),
                ui::metric(&ui, nodes),
                ui::metric(&ui, edges)
            )),
            ProgressEvent::GraphWrite { files } => self.io.line(&format!(
                "{} {} fichiers écrits",
                ui::badge(&ui, "GRAPH"),
                ui::metric(&ui, files)
            )),
            ProgressEvent::GraphDone {
                nodes,
                edges,
                seconds,
            } => self.io.line(&format!(
                "{} {} nœuds · {} arêtes · {}",
                ui::badge(&ui, "GRAPH"),
                ui::metric(&ui, nodes),
                ui::metric(&ui, edges),
                ui::dim(&ui, &format!("{seconds:.1} s"))
            )),
            ProgressEvent::Shift { .. }
            | ProgressEvent::GraphNode { .. }
            | ProgressEvent::EvalPrompt { .. }
            | ProgressEvent::ExtractWriting
            | ProgressEvent::BenchDumpPrompt { .. }
            | ProgressEvent::BenchLogitsBlock { .. }
            | ProgressEvent::BenchPplChunk { .. }
            | ProgressEvent::BenchEvalQuestion { .. }
            | ProgressEvent::BenchRefusalPrompt { .. }
            | ProgressEvent::BenchJudgeVerdict { .. }
            | ProgressEvent::AutotuneForward { .. }
            | ProgressEvent::AutotuneSolve
            | ProgressEvent::Surgery(_) => {}
        }
    }
}

/// Never followed by a `[COMMIT]`: only emitted when `apply_patch` failed.
fn rollback_line(ui: &Ui, outcome: &RollbackOutcome) -> String {
    let badge = ui::badge(ui, "ROLLBACK");
    match outcome {
        RollbackOutcome::Restored => format!("{badge} {}", ui::metric(ui, "restauré ✓")),
        RollbackOutcome::Failed { backups } => {
            let files = backups
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "{badge} {} {}",
                ui::metric(ui, "ÉCHEC CRITIQUE ✗"),
                ui::dim(ui, &format!("restaurer manuellement depuis {files}"))
            )
        }
        RollbackOutcome::NoBackup => format!(
            "{badge} {} {}",
            ui::metric(ui, "IMPOSSIBLE ✗"),
            ui::dim(ui, "pas de backup : modèle partiellement modifié")
        ),
    }
}

fn rollback_outcome(e: &PatchError, backups: &[PathBuf]) -> Option<RollbackOutcome> {
    match e {
        PatchError::ExecutionFailed {
            rolled_back: true, ..
        } => Some(RollbackOutcome::Restored),
        PatchError::RollbackFailed { .. } => Some(RollbackOutcome::Failed {
            backups: backups.to_vec(),
        }),
        PatchError::ExecutionFailed {
            left_modified: true,
            ..
        } => Some(RollbackOutcome::NoBackup),
        _ => None,
    }
}

/// Input of [`core_apply`], filled by the CLI (`patch.json` on disk) or by the MCP server
/// (inline patch, sandboxed paths).
#[derive(Debug, Clone)]
pub struct ApplyRequest {
    pub model: PathBuf,
    pub patch: PatchFile,
    /// Shown by the `[APPLY]` step.
    pub patch_label: PathBuf,
    /// Relative `vector_file`s resolve against it.
    pub base_dir: PathBuf,
    pub dry_run: bool,
    pub no_backup: bool,
    pub backup_dir: Option<PathBuf>,
}

fn apply(args: &ApplyArgs, deps: &AppDeps, io: &mut Io) -> CoreResult {
    // A `.surgery` file (SURG magic) is handled by its own engine; patch.json falls through.
    if let Some(outcome) = crate::surgery::try_apply(args, deps, &mut CliSink::new(io))? {
        return Ok(outcome);
    }
    let request = ApplyRequest {
        model: args.model.clone(),
        patch: load_patch(&args.patch)?,
        patch_label: args.patch.clone(),
        // Relative vector paths resolve against the patch directory ("" for "patch.json").
        base_dir: args.patch.parent().unwrap_or(Path::new("")).to_path_buf(),
        dry_run: args.dry_run,
        no_backup: args.no_backup,
        backup_dir: args.backup_dir.clone(),
    };
    let ui = io.ui;
    core_apply_routed(&request, &args.gguf, deps, &ui, &mut CliSink::new(io))
}

/// validate -> budget (warning only) -> open shards -> dry-run preflight -> backup of the
/// touched shards -> apply_patch (rollback on error, NaN, Ctrl-C or cancellation).
pub fn core_apply(
    request: &ApplyRequest,
    deps: &AppDeps,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let limits = ApplyOptions::default();
    let arch = ArchOptions::default();
    core_apply_io(request, IoConfig::default(), limits, &arch, deps, sink)
}

/// The architecture options of `apply` (`--arch-spec`, `--accept-approximate`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ArchOptions {
    pub spec: Option<PathBuf>,
    pub accept_approximate: bool,
}

/// The gate of a weight ablation ([`arch::gate`]): `Ok(Some(..))`, the `data.arch` object
/// and the warning, when the patch holds a `project_out`.
pub fn arch_gate(
    patch: &patch::ValidatedPatch,
    report: &arch::Report,
    accept: bool,
) -> Result<Option<(Value, Option<String>)>, Outcome> {
    let project_out = patch
        .operations
        .iter()
        .any(|op| op.kind == OperationKind::ProjectOut);
    let warning = arch::gate(report, project_out, accept).map_err(|refusal| {
        Outcome::fail(exit::INVALID_INPUT, refusal.message).with_kind(refusal.kind)
    })?;
    if !project_out {
        return Ok(None);
    }
    let mut data = json!(report);
    data["accepted_approximate"] = json!(warning.is_some());
    Ok(Some((data, warning)))
}

/// [`core_apply`] with explicit chunking / prefetch; a real run reports `data.io`.
fn core_apply_io(
    request: &ApplyRequest,
    io: IoConfig,
    limits: ApplyOptions,
    arch_opts: &ArchOptions,
    deps: &AppDeps,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let opts = ValidationOptions {
        no_backup: request.no_backup,
        ..Default::default()
    };
    let patch = validate_patch(&request.patch, &request.base_dir, &opts)?;
    sink.event(ProgressEvent::ApplyStart {
        patch: request.patch_label.clone(),
        operations: patch.operations.len(),
    });

    // Absolute, so backups next to a relative `--model` get a printable location.
    let model = std::path::absolute(&request.model).map_err(|e| {
        Outcome::fail(exit::GENERAL, format!("{} : {e}", request.model.display()))
            .with_kind("model_error")
    })?;
    let shards = hardware::compute_model_size(&model)?.shard_files;
    let budget = hardware::evaluate_budget(&model, deps.probe.as_ref(), &BudgetPolicy::default())?;
    let mut warnings: Vec<String> = patch.warnings.iter().map(ToString::to_string).collect();
    if let Verdict::Insufficient { reason } = &budget.verdict {
        warnings.push(format!(
            "le modèle ne tiendrait pas en mémoire ({reason}) ; sans effet sur apply, qui ne \
             charge que les tenseurs ciblés"
        ));
    }
    // The architecture decides the targets and their orientation, and gates an
    // approximate ablation before anything is read for writing.
    let resolution =
        arch::resolve(&model, None, arch_opts.spec.as_deref()).map_err(arch_failure)?;
    let mut names = Vec::new();
    for shard in &shards {
        let model = ReadOnlyModel::open(shard)?;
        names.extend(model.index().tensors.iter().map(|t| t.name.clone()));
    }
    let report = arch_report(&resolution, &model, &names);
    let arch_data = match arch_gate(&patch, &report, arch_opts.accept_approximate)? {
        Some((data, warning)) => {
            warnings.extend(warning);
            Some(data)
        }
        None => None,
    };
    for w in &warnings {
        sink.event(ProgressEvent::Warn(w.clone()));
    }

    // Preflight on its own mapping, released before any file is replaced or restored.
    let (preview, mut touched) = {
        let mut backend = EngineBackend::open(&shards, Arc::clone(&deps.interrupted))?
            .with_arch(resolution.spec.clone());
        let preview = apply_patch_with(&patch, &mut backend, None, true, &limits)?;
        let touched = backend.shards_holding(
            preview
                .operations
                .iter()
                .flat_map(|op| op.tensors.iter().map(|t| t.name.as_str())),
        );
        (preview, touched)
    };
    // What only the vectors can tell: a project_out that is not a projector.
    for warning in &preview.warnings {
        sink.event(ProgressEvent::Warn(warning.clone()));
        warnings.push(warning.clone());
    }
    // Patching tied embeddings would also deform the output projection: detach it first.
    let embed = preview
        .operations
        .iter()
        .flat_map(|op| &op.tensors)
        .find(|t| engine::is_embed_tokens(&t.name));
    let untie = match embed {
        Some(embed) => untie::plan(&shards, &embed.name)?,
        None => None,
    };
    if let Some(plan) = &untie {
        let what = match plan.copy_embeddings {
            true => format!(
                "embed_tokens est partagé avec lm_head : {} (copie non patchée des plongements) \
                 est ajouté à {} et",
                untie::LM_HEAD,
                plan.shard.display()
            ),
            false => format!(
                "{} existe mais la config lie encore les plongements :",
                untie::LM_HEAD
            ),
        };
        let warning = format!(
            "{what} tie_word_embeddings passe à false dans {}, pour que la projection de sortie \
             reste intacte",
            plan.config.display()
        );
        sink.event(ProgressEvent::Warn(warning.clone()));
        warnings.push(warning);
    }
    let with_untie = |mut outcome: Outcome| {
        if let Some(plan) = &untie {
            outcome.data["untie_lm_head"] = json!(plan);
        }
        if let Some(arch) = &arch_data {
            outcome.data["arch"] = arch.clone();
        }
        outcome
    };
    if request.dry_run {
        for op in &preview.operations {
            sink.event(ProgressEvent::DryRun {
                op_id: op.id.clone(),
                tensors: op.tensors.iter().map(|t| t.name.clone()).collect(),
                vector_len: op.vector_len,
                rank: op.rank,
            });
        }
        return Ok(with_untie(apply_outcome(&preview, &[], &warnings)));
    }
    if deps.interrupted.load(Ordering::SeqCst) {
        return Err(PatchError::ExecutionFailed {
            op_id: "preflight".into(),
            cause: format!("{INTERRUPTED} avant toute écriture"),
            rolled_back: false,
            left_modified: false,
        }
        .into());
    }

    for file in untie.iter().flat_map(untie::UntiePlan::files) {
        if !touched.contains(&file) {
            touched.push(file);
        }
    }
    let mut backup = match patch.backup {
        true => Some(BackupSet::create_many_in(
            &touched,
            request.backup_dir.as_deref(),
        )?),
        false => None,
    };
    let backup_files: Vec<PathBuf> = backup
        .iter()
        .flat_map(|b| b.guards())
        .map(|g| g.backup_path().to_path_buf())
        .collect();
    for file in &backup_files {
        sink.event(ProgressEvent::Backup { path: file.clone() });
    }

    let started = Instant::now();
    // Declared after `backup`, so dropped before it: a restore on unwind can rename.
    let prepared = untie
        .as_ref()
        .map_or(Ok(()), untie::UntiePlan::execute)
        .and_then(|()| {
            EngineBackend::open(&shards, Arc::clone(&deps.interrupted))
                .map(|b| b.with_arch(resolution.spec.clone()))
                .map_err(backend_error)
        });
    let mut backend = match prepared {
        Ok(backend) => backend,
        Err(e) => {
            let e = restored(UNTIE_OP, e.to_string(), backup.as_mut());
            if let Some(outcome) = rollback_outcome(&e, &backup_files) {
                sink.event(ProgressEvent::Rollback(outcome));
            }
            return Err(e.into());
        }
    };
    backend.set_io(io);
    let ops = patch
        .operations
        .iter()
        .map(|op| (op.id.clone(), op.kind))
        .collect();
    // Once the shard has grown, restoring it means shrinking it, which Windows refuses
    // while the backend maps it: the rollback is then recorded and done after the drop.
    let mut deferred = untie
        .as_ref()
        .zip(backup.as_ref())
        .map(|(_, b)| DeferredRollback {
            backup_path: b.backup_path(),
            requested: false,
        });
    let guard = match &mut deferred {
        Some(deferred) => Some(deferred as &mut dyn RollbackHandle),
        None => backup.as_mut().map(|b| b as &mut dyn RollbackHandle),
    };
    let kernel = kernel::from_env();
    if let Some(w) = &kernel.warning {
        sink.event(ProgressEvent::Warn(w.clone()));
    }
    let meter = IoMeter::start();
    let result = apply_patch_with(
        &patch,
        &mut ObservedBackend::new(&mut backend, sink, ops),
        guard,
        false,
        &limits,
    );
    match result {
        Ok(report) => {
            if let (Some(_), Some(backup)) = (&deferred, backup.as_mut()) {
                backup.commit_all();
            }
            sink.event(ProgressEvent::Commit {
                seconds: started.elapsed().as_secs_f64(),
            });
            let io_report = meter.finish(kernel, io, backend.io_counters());
            let mut outcome = with_untie(apply_outcome(&report, &backup_files, &warnings));
            outcome.data["io"] = json!(io_report);
            Ok(outcome)
        }
        Err(e) => {
            let e = match (deferred, e) {
                (
                    Some(DeferredRollback {
                        requested: true, ..
                    }),
                    PatchError::ExecutionFailed { op_id, cause, .. },
                ) => {
                    drop(backend);
                    restored(&op_id, cause, backup.as_mut())
                }
                (_, e) => e,
            };
            if let Some(outcome) = rollback_outcome(&e, &backup_files) {
                sink.event(ProgressEvent::Rollback(outcome));
            }
            Err(e.into())
        }
    }
}

const UNTIE_OP: &str = "untie_lm_head";

/// Stands in for the `BackupSet` in `apply_patch` after an untie; see `core_apply_io`.
struct DeferredRollback {
    backup_path: Option<PathBuf>,
    requested: bool,
}

impl RollbackHandle for DeferredRollback {
    fn rollback(&mut self) -> patch::Result<()> {
        self.requested = true;
        Ok(())
    }
    fn commit(&mut self) {}
    fn backup_path(&self) -> Option<PathBuf> {
        self.backup_path.clone()
    }
}

/// Restores the backups after a failure outside `apply_patch`'s own rollback, with the
/// same three outcomes: restored, rollback failed (critical), or no backup to restore.
fn restored(op_id: &str, cause: String, backup: Option<&mut BackupSet>) -> PatchError {
    let op_id = op_id.to_owned();
    match backup.map(BackupSet::rollback_all) {
        Some(Ok(())) => PatchError::ExecutionFailed {
            op_id,
            cause,
            rolled_back: true,
            left_modified: false,
        },
        Some(Err(e)) => PatchError::RollbackFailed {
            op_id,
            cause: format!("{cause} ; rollback : {e}"),
        },
        None => PatchError::ExecutionFailed {
            op_id,
            cause: format!("{cause} (pas de backup)"),
            rolled_back: false,
            left_modified: true,
        },
    }
}

/// Backups are kept after success: they are the manual undo. The human text was streamed
/// through the sink, so the outcome only carries the JSON data.
fn apply_outcome(report: &ApplyReport, backups: &[PathBuf], warnings: &[String]) -> Outcome {
    let mut outcome = Outcome::ok(
        json!({ "report": report, "backups": backups, "warnings": warnings }),
        String::new(),
    );
    outcome.warnings = warnings.to_vec();
    outcome
}

// ---------------------------------------------------------------- apply: chunked I/O and GGUF

/// `apply` routing on the file's magic: a GGUF only with `--gguf` (re-rounding is lossy,
/// never a surprise), SafeTensors with the chunking options.
pub fn core_apply_routed(
    request: &ApplyRequest,
    opts: &GgufApplyArgs,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let usage = |msg: String| Outcome::fail(exit::INVALID_INPUT, msg).with_kind("invalid_input");
    match (opts.gguf, gguf::is_gguf(&request.model)) {
        (true, true) => core_apply_gguf(request, opts, deps, ui, sink),
        (false, true) => Err(usage(format!(
            "{} est un fichier GGUF : relancez avec --gguf (re-quantification en place, avec \
             perte ; voir --requant)",
            request.model.display()
        ))),
        (true, false) => Err(usage(format!(
            "--gguf : {} n'est pas un fichier GGUF (magie « GGUF » absente)",
            request.model.display()
        ))),
        (false, false)
            if opts.force_no_backup
                || opts.no_verify
                || opts.requant != Requant::KeepScales
                || opts.allow_tied_embedding
                || opts.refine_passes != gguf::project::DEFAULT_REFINE_PASSES
                || opts.residual_target != gguf::project::DEFAULT_RESIDUAL_TARGET =>
        {
            Err(usage(
                "--requant, --no-verify, --force-no-backup, --allow-tied-embedding, \
                 --refine-passes et --residual-target ne s'appliquent qu'avec --gguf"
                    .into(),
            ))
        }
        (false, false) => {
            let limits = ApplyOptions {
                max_rank: opts.max_rank,
            };
            core_apply_io(request, opts.io(), limits, &opts.arch_options(), deps, sink)
        }
    }
}

/// `apply --gguf`: validation, then `patch::gguf::apply_gguf` (backup, crash marker,
/// chunked write, read-back check, rollback). `data.gguf` holds the per-tensor fidelity.
fn core_apply_gguf(
    request: &ApplyRequest,
    opts: &GgufApplyArgs,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    if request.no_backup {
        return Err(Outcome::fail(
            exit::INVALID_INPUT,
            "--no-backup est refusé avec --gguf : la copie de sauvegarde est obligatoire \
             (--force-no-backup pour s'en passer, un échec laisserait alors le fichier modifié)",
        )
        .with_kind("invalid_input"));
    }
    let validation = ValidationOptions {
        no_backup: opts.force_no_backup,
        ..Default::default()
    };
    let patch = validate_patch(&request.patch, &request.base_dir, &validation)?;
    sink.event(ProgressEvent::ApplyStart {
        patch: request.patch_label.clone(),
        operations: patch.operations.len(),
    });
    let model = std::path::absolute(&request.model).map_err(|e| {
        Outcome::fail(exit::GENERAL, format!("{} : {e}", request.model.display()))
            .with_kind("model_error")
    })?;
    let mut warnings: Vec<String> = patch.warnings.iter().map(ToString::to_string).collect();
    if opts.force_no_backup && !request.dry_run {
        warnings.push("--force-no-backup : aucune copie, un échec laisse le GGUF modifié".into());
    }
    for w in &warnings {
        sink.event(ProgressEvent::Warn(w.clone()));
    }

    let ops: Vec<(String, OperationKind)> = patch
        .operations
        .iter()
        .map(|op| (op.id.clone(), op.kind))
        .collect();
    let mut options = opts.options(request.backup_dir.clone());
    if let Some(path) = &opts.arch_spec {
        options.arch_spec = Some(arch::load_spec(path).map_err(arch_failure)?);
    }
    let dry_run = request.dry_run;
    let started = Instant::now();
    let meter = IoMeter::start();
    let result = gguf::apply_gguf(
        &model,
        &patch,
        &options,
        dry_run,
        Arc::clone(&deps.interrupted),
        &mut |backend, guard| {
            if let Some(path) = guard.as_ref().and_then(|g| g.backup_path()) {
                sink.event(ProgressEvent::Backup { path });
            }
            match dry_run {
                true => apply_patch(&patch, backend, None, false),
                false => apply_patch(
                    &patch,
                    &mut ObservedBackend::new(backend, &mut *sink, ops.clone()),
                    guard,
                    false,
                ),
            }
        },
    );
    let run = match result {
        Ok(run) => run,
        Err(GgufApplyError::Patch(e)) => return Err(e.into()),
        Err(e) => {
            let code = match &e {
                GgufApplyError::Usage(_) | GgufApplyError::Mapping(_) | GgufApplyError::Arch(_) => {
                    exit::INVALID_INPUT
                }
                _ => exit::GENERAL,
            };
            let kind = e.kind().unwrap_or("execution_failed");
            return Err(Outcome::fail(code, e).with_kind(kind));
        }
    };
    for t in &run.tensors {
        if let Some(w) = &t.warning {
            warnings.push(format!("{} : {w}", t.name));
            sink.event(ProgressEvent::Warn(format!("{} : {w}", t.name)));
        }
    }
    match dry_run {
        true => {
            for op in &run.report.operations {
                sink.event(ProgressEvent::DryRun {
                    op_id: op.id.clone(),
                    tensors: op.tensors.iter().map(|t| t.name.clone()).collect(),
                    vector_len: op.vector_len,
                    rank: op.rank,
                });
            }
        }
        false => sink.event(ProgressEvent::Commit {
            seconds: started.elapsed().as_secs_f64(),
        }),
    }
    if let Some(w) = &run.arch_warning {
        warnings.push(w.clone());
        sink.event(ProgressEvent::Warn(w.clone()));
    }
    let backups: Vec<PathBuf> = run.backup.iter().cloned().collect();
    let mut outcome = apply_outcome(&run.report, &backups, &warnings);
    outcome.data["gguf"] = json!(run);
    if patch
        .operations
        .iter()
        .any(|op| op.kind == OperationKind::ProjectOut)
    {
        outcome.data["arch"] = json!(run.arch);
        outcome.data["arch"]["accepted_approximate"] = json!(run.arch_warning.is_some());
    }
    if !dry_run {
        // Dequantize / requantize is scalar code (rows in parallel), not the SIMD kernels.
        let kernel = KernelChoice {
            level: Level::Scalar,
            requested: None,
            warning: None,
        };
        outcome.data["io"] = json!(meter.finish(kernel, options.io, run.io));
    }
    outcome.human = gguf::report::apply_text(ui, &model, &run, dry_run, opts.no_verify);
    Ok(outcome)
}

// ---------------------------------------------------------------- eval

/// Stops the generation at the next chunk once Ctrl-C (or an MCP cancellation) raised the
/// flag, and reports one progress event per prompt (`run_eval` calls `generate` once per
/// prompt, in order).
struct Interruptible<'a> {
    inner: Box<dyn TextGenerator>,
    flag: &'a AtomicBool,
    sink: &'a mut dyn ProgressSink,
    ids: Vec<String>,
    done: usize,
}

impl TextGenerator for Interruptible<'_> {
    fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: u32,
        on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> crate::eval::Result<GenerationOutcome> {
        let flag = self.flag;
        let result = match flag.load(Ordering::SeqCst) {
            true => Err(EvalError::Generator(INTERRUPTED.into())),
            false => self
                .inner
                .generate(
                    prompt,
                    max_new_tokens,
                    &mut |chunk| match flag.load(Ordering::SeqCst) {
                        true => ControlFlow::Break(()),
                        false => on_chunk(chunk),
                    },
                ),
        };
        self.done += 1;
        self.sink.event(ProgressEvent::EvalPrompt {
            id: self.ids.get(self.done - 1).cloned().unwrap_or_default(),
            done: self.done,
            total: self.ids.len(),
        });
        result
    }
}

fn eval(args: &EvalArgs, deps: &AppDeps, ui: &Ui) -> CoreResult {
    let prompts = load_prompts(&args.prompts)
        .map_err(|e| Outcome::fail(exit::INVALID_INPUT, e).with_kind("invalid_input"))?;
    core_eval(args, prompts, deps, ui, &mut NullSink)
}

/// Converts to a temporary GGUF unless the model already is one, evaluates the prompts,
/// then exports when healthy. `args.prompts` is ignored: the prompts come loaded.
pub fn core_eval(
    args: &EvalArgs,
    prompts: Vec<EvalPrompt>,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let invalid =
        |message: String| Outcome::fail(exit::INVALID_INPUT, message).with_kind("invalid_input");
    if !args.model.exists() {
        return Err(
            invalid(format!("modèle introuvable : {}", args.model.display()))
                .with_kind("model_not_found"),
        );
    }
    if prompts.is_empty() {
        return Err(invalid("aucun prompt".into()));
    }
    // Checked now so a typo does not waste a whole evaluation before the export.
    let quant = args.quant.trim().to_ascii_uppercase();
    if args.export_gguf.is_some() && !ALLOWED_QUANTS.contains(&quant.as_str()) {
        return Err(invalid(
            EvalError::InvalidQuant(args.quant.clone()).to_string(),
        ));
    }
    if is_gguf(&args.model) && args.export_gguf.is_some() {
        return Err(invalid(
            "--export-gguf convertit depuis SafeTensors : impossible depuis un .gguf".into(),
        ));
    }
    let model_dir = crate::eval::model_dir(&args.model);
    let llama_dir = || {
        args.llama_cpp_dir
            .clone()
            .ok_or_else(|| invalid(LLAMA_DIR_REQUIRED.into()))
    };
    let export_cfg =
        |llama_cpp_dir: PathBuf, output_gguf: PathBuf, quant: Option<String>| GgufExportConfig {
            llama_cpp_dir,
            model_dir: model_dir.clone(),
            output_gguf,
            quant,
            intermediate_type: "f16".into(),
            python_bin: if cfg!(windows) { "python" } else { "python3" }.into(),
            keep_intermediate: false,
        };
    let final_llama_dir = match args.export_gguf {
        Some(_) => Some(llama_dir()?),
        None => None,
    };

    let gguf = ensure_gguf(
        &args.model,
        args.llama_cpp_dir.as_deref(),
        "eval",
        deps.runner.as_ref(),
    )
    .map_err(|e| match e {
        EvalError::InvalidConfig(m) if m == LLAMA_DIR_REQUIRED => invalid(m),
        e => Outcome::fail(
            exit::GENERAL,
            format!("conversion GGUF pour l'évaluation : {e}"),
        )
        .with_kind("eval_failed"),
    })?;

    let generator = (deps.generator_factory)(gguf.path(), args)
        .map_err(|e| Outcome::fail(exit::GENERAL, e).with_kind("eval_failed"))?;
    let mut generator = Interruptible {
        inner: generator,
        flag: &deps.interrupted,
        sink,
        ids: prompts.iter().map(|p| p.id.clone()).collect(),
        done: 0,
    };
    let cfg = EvalConfig {
        max_new_tokens: args.max_new_tokens,
        ..Default::default()
    };
    let report = run_eval(&mut generator, &prompts, &cfg)
        .map_err(|e| Outcome::fail(exit::GENERAL, e).with_kind("eval_failed"))?;
    drop(generator);
    // Freed before the final conversion, which needs the disk space again.
    drop(gguf);

    let human = eval_text(ui, &report, args.max_new_tokens);
    if deps.interrupted.load(Ordering::SeqCst) {
        return Err(Outcome::fail(exit::GENERAL, INTERRUPTED)
            .with_kind("cancelled")
            .with_data(json!({ "report": report, "export": null }), human));
    }
    let export = match (&args.export_gguf, final_llama_dir) {
        (Some(output), Some(dir)) => {
            let cfg = export_cfg(dir, output.clone(), Some(quant.clone()));
            export_if_healthy(&report, &cfg, deps.runner.as_ref(), args.allow_degraded).map_err(
                |e| {
                    Outcome::fail(exit::EXPORT_FAILED, format!("export GGUF : {e}"))
                        .with_kind("export_failed")
                        .with_data(json!({ "report": report, "export": null }), human.clone())
                },
            )?
        }
        _ => None,
    };

    let mut human = human;
    match (&export, &args.export_gguf) {
        (Some(e), _) => human.push_str(&format!(
            "{} {} · {} · {}  {}\n",
            ui::badge(ui, "EXPORT"),
            ui::path(ui, &e.final_gguf),
            ui::metric(ui, &quant),
            ui::metric(ui, size(e.final_size_bytes)),
            ui::metric(ui, "✓ magic GGUF")
        )),
        (None, Some(_)) => human.push_str(&format!(
            "{} {}\n",
            ui::badge(ui, "EXPORT"),
            ui::dim(
                ui,
                &format!("ignoré : verdict {}", verdict_label(report.verdict))
            )
        )),
        (None, None) => {}
    }
    let data = json!({ "report": report, "export": export });
    match (report.verdict, args.allow_degraded) {
        (EvalVerdict::Healthy, _) | (EvalVerdict::Degraded, true) => Ok(Outcome::ok(data, human)),
        (verdict, _) => {
            let skipped = match args.export_gguf.is_some() {
                true => ", export non effectué",
                false => "",
            };
            Err(Outcome::fail(
                exit::EVAL_UNHEALTHY,
                format!("modèle {verdict:?}{skipped}").to_lowercase(),
            )
            .with_kind("eval_unhealthy")
            .with_data(data, human))
        }
    }
}

fn verdict_label(verdict: EvalVerdict) -> &'static str {
    match verdict {
        EvalVerdict::Healthy => "HEALTHY",
        EvalVerdict::Degraded => "DEGRADED",
        EvalVerdict::Broken => "BROKEN",
    }
}

const EXCERPT_CHARS: usize = 32;

fn eval_text(ui: &Ui, report: &EvalReport, max_new_tokens: u32) -> String {
    let total = report.results.len();
    let mut loops = 0;
    let rows: Vec<Vec<String>> = report
        .results
        .iter()
        .map(|r| {
            let (status, reason) = match &r.status {
                PromptStatus::Pass => ("PASS", "—".to_owned()),
                PromptStatus::LoopDetected { reason } => {
                    loops += 1;
                    ("LOOP", format!("{reason:?}"))
                }
                PromptStatus::ExpectationFailed { .. } => ("FAIL", "attente".to_owned()),
                PromptStatus::GeneratorError { .. } => ("FAIL", "erreur".to_owned()),
            };
            // One line per prompt: newlines and runs of spaces collapse.
            let flat = r
                .output_excerpt
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let cut = crate::eval::truncate_chars(&flat, EXCERPT_CHARS);
            let ellipsis = if cut.len() < flat.len() { "…" } else { "" };
            vec![
                ui::metric(ui, &r.id),
                ui::metric(ui, status),
                ui::metric(ui, reason),
                ui::dim(ui, &format!("« {cut}{ellipsis} »")),
            ]
        })
        .collect();
    format!(
        "{} {} prompts · max {} tokens\n{}\n{}{} {} · {}/{total} PASS · {loops} boucle(s)  {}\n",
        ui::badge(ui, "EVAL"),
        ui::metric(ui, total),
        ui::metric(ui, max_new_tokens),
        ui::separator(ui),
        ui::table(ui, &["PROMPT", "STATUT", "BOUCLE", "SORTIE"], &rows),
        ui::badge(ui, "VERDICT"),
        ui::metric(ui, verdict_label(report.verdict)),
        ui::metric(ui, report.passed),
        ui::progress_bar(ui, report.passed, total, BAR_WIDTH)
    )
}

// ---------------------------------------------------------------- extract

/// CLI entry: reads the two prompt files, then [`core_extract_request`] (`mean_diff`).
pub fn core_extract(
    args: &ExtractArgs,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    core_extract_with(args, DEFAULT_METHOD, &Value::Null, deps, ui, sink)
}

/// [`core_extract`] with a method of the built-in registry and its `params`.
pub fn core_extract_with(
    args: &ExtractArgs,
    method: &str,
    params: &Value,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let load = |set: &str, path: &Path| {
        extract::load_prompt_texts(path).map_err(|message| {
            extract_failure(&ExtractError::Prompts {
                set: set.into(),
                message,
            })
        })
    };
    let request = ExtractRequest {
        model_dir: args.model.clone(),
        pos_prompts: load("pos", &args.pos_prompts)?,
        neg_prompts: load("neg", &args.neg_prompts)?,
        layer: args.layer,
        out: args.out.clone(),
        pooling: args.pooling,
        site: args.site,
        chat_template: args.chat_template,
        max_tokens: args.max_tokens,
        dtype: args.dtype,
        overwrite: args.overwrite,
    };
    core_extract_method(&request, method, params, deps, ui, sink)
}

/// Shared by the CLI and the MCP tool. Cancellation (`deps.interrupted`) is honored
/// between two tensors and between two prompts; nothing is written then.
pub fn core_extract_request(
    request: &ExtractRequest,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    core_extract_method(request, DEFAULT_METHOD, &Value::Null, deps, ui, sink)
}

/// [`core_extract_request`] with a method of the built-in registry (`svd`, `leace`) and
/// its `params`. An unknown identifier is an invalid input (code 2) naming the known ones.
pub fn core_extract_method(
    request: &ExtractRequest,
    method: &str,
    params: &Value,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let registry = Registry::with_builtin();
    let Some(method) = registry.get(method) else {
        let message = unknown_method_message(method, &registry.ids());
        return Err(extract_failure(&ExtractError::InvalidArgument(message)));
    };
    let mut observer = SinkObserver {
        sink,
        interrupted: &deps.interrupted,
        request,
    };
    let report = extract::run_extraction_plugin(
        request,
        method,
        params,
        deps.probe.as_ref(),
        &mut observer,
        &deps.interrupted,
    )
    .map_err(|e| extract_failure(&e))?;
    let mut outcome = Outcome::ok(to_value(&report), extract_text(ui, &report));
    outcome.warnings = report.warnings.clone();
    Ok(outcome)
}

fn extract_failure(e: &ExtractError) -> Outcome {
    let code = match e {
        ExtractError::InvalidArgument(_)
        | ExtractError::Prompts { .. }
        | ExtractError::OutputExists(_)
        | ExtractError::UnpairedPrompts { .. } => exit::INVALID_INPUT,
        ExtractError::InsufficientResources(_) | ExtractError::Cancelled => exit::GENERAL,
        _ => exit::EXTRACT_FAILED,
    };
    Outcome::fail(code, e).with_kind(e.kind())
}

/// Adapts the extraction progress to a [`ProgressSink`] and turns the interruption flag
/// into a `Break`.
struct SinkObserver<'a> {
    sink: &'a mut dyn ProgressSink,
    interrupted: &'a AtomicBool,
    request: &'a ExtractRequest,
}

impl ExtractObserver for SinkObserver<'_> {
    fn on_event(&mut self, event: ExtractEvent) -> ControlFlow<()> {
        let progress = match event {
            ExtractEvent::Planned {
                dtype_used,
                weight_bytes,
                ..
            } => ProgressEvent::ExtractStart {
                model: self.request.model_dir.clone(),
                layer: self.request.layer,
                site: self.request.site.as_str().into(),
                pooling: self.request.pooling.as_str().into(),
                dtype: dtype_used,
                weight_bytes,
                n_pos: self.request.pos_prompts.len(),
                n_neg: self.request.neg_prompts.len(),
            },
            ExtractEvent::LoadingWeights {
                done_tensors,
                total_tensors,
            } => ProgressEvent::ExtractLoading {
                done: done_tensors,
                total: total_tensors,
            },
            ExtractEvent::Prompt { set, done, total } => {
                ProgressEvent::ExtractPrompt { set, done, total }
            }
            ExtractEvent::Writing => ProgressEvent::ExtractWriting,
        };
        self.sink.event(progress);
        match self.interrupted.load(Ordering::SeqCst) {
            true => ControlFlow::Break(()),
            false => ControlFlow::Continue(()),
        }
    }
}

/// `[VERDICT]` and `[WRITE]`, then the warnings; the earlier steps were streamed.
fn extract_text(ui: &Ui, report: &ExtractReport) -> String {
    let s = &report.stats;
    let d_prime = s.d_prime.map_or("∞".to_owned(), |d| format!("{d:.2}"));
    let mut text = format!(
        "{} d′ {} · accuracy {} · ‖Δ‖ {}\n{} {}\n",
        ui::badge(ui, "VERDICT"),
        ui::metric(ui, d_prime),
        ui::metric(ui, format!("{:.0}%", s.threshold_accuracy * 100.0)),
        ui::metric(ui, format!("{:.3}", s.raw_diff_norm)),
        ui::badge(ui, "WRITE"),
        ui::path(ui, &report.out_path),
    );
    if let Some(diag) = &report.method_diagnostics {
        text.push_str(&method_line(ui, diag));
    }
    for path in report.leace_path.iter().chain(&report.sidecar_paths) {
        text.push_str(&format!(
            "{} {}
",
            ui::badge(ui, "WRITE"),
            ui::path(ui, path)
        ));
    }
    for w in &report.warnings {
        text.push_str(&warning_line(ui, w));
        text.push('\n');
    }
    text
}

// ---------------------------------------------------------------- plugins

/// `surgeon plugins list`: the catalog of the built-in registry, the same in `--json`.
fn plugins_list(ui: &Ui) -> CoreResult {
    let catalog = Registry::with_builtin().catalog();
    let mut text = String::new();
    for point in &catalog.extension_points {
        text.push_str(&format!(
            "{} {} · {} · API {}\n",
            ui::badge(ui, "EXTENSION"),
            point.id,
            point.rust_trait,
            catalog.api_version
        ));
        for m in &point.implementations {
            text.push_str(&format!(
                "  {} {:<10} {}\n",
                ui::metric(ui, format!("{:<10}", m.id)),
                m.origin.as_str(),
                m.info.title
            ));
        }
    }
    Ok(Outcome::ok(to_value(&catalog), text))
}

/// `surgeon arch list`: every built-in spec, with its classification and how it was
/// checked.
fn arch_list(ui: &Ui) -> CoreResult {
    let mut text = String::new();
    let mut rows = Vec::new();
    for spec in arch::builtins() {
        let exactness = spec.exactness();
        text.push_str(&format!(
            "{} {:<10} {} · GGUF {} · project_out {} · forward {}
",
            ui::badge(ui, "ARCH"),
            spec.id,
            ui::metric(ui, spec.hf_model_types.join(", ")),
            ui::metric(ui, spec.gguf_arch.join(", ")),
            ui::metric(ui, exactness.label()),
            ui::metric(
                ui,
                if arch::FORWARD_IDS.contains(&spec.id.as_str()) {
                    "oui"
                } else {
                    "non"
                }
            ),
        ));
        if let Some(reason) = exactness.reason() {
            text.push_str(&format!(
                "  {}
",
                ui::dim(ui, reason)
            ));
        }
        rows.push(json!({
            "id": spec.id,
            "hf_model_types": spec.hf_model_types,
            "hf_architectures": spec.hf_architectures,
            "gguf_arch": spec.gguf_arch,
            "exactness": exactness,
            "forward": arch::FORWARD_IDS.contains(&spec.id.as_str()),
            "verification": arch::verification(&spec.id),
            "spec": spec,
        }));
    }
    Ok(Outcome::ok(json!({ "architectures": rows }), text))
}

/// `surgeon arch detect`: the architecture of a SafeTensors model or a GGUF file, with the
/// roles of its tensors.
fn arch_detect(args: &crate::cli::ArchDetectArgs, ui: &Ui) -> CoreResult {
    let report = match gguf::is_gguf(&args.model) {
        true => {
            let fail = |e: &dyn std::fmt::Display| {
                Outcome::fail(exit::GENERAL, format!("{} : {e}", args.model.display()))
                    .with_kind("invalid_gguf")
            };
            let file = std::fs::File::open(&args.model).map_err(|e| fail(&e))?;
            let map = kernel::map_read(&file).map_err(|e| fail(&e))?;
            let parsed = gguf::parse(&map).map_err(|e| fail(&e))?;
            let resolution = arch::resolve(
                &args.model,
                Some(parsed.architecture()),
                args.arch_spec.as_deref(),
            )
            .map_err(arch_failure)?;
            arch::Report::new(
                &resolution,
                parsed.tensors.iter().map(|t| t.name.as_str()),
                arch::Format::Gguf,
                Some(gguf::mapping::tied_embeddings(&parsed)),
            )
        }
        false => {
            let shards = hardware::compute_model_size(&args.model)?.shard_files;
            let resolution = arch::resolve(&args.model, None, args.arch_spec.as_deref())
                .map_err(arch_failure)?;
            let mut names = Vec::new();
            for shard in &shards {
                let model = ReadOnlyModel::open(shard)?;
                names.extend(model.index().tensors.iter().map(|t| t.name.clone()));
            }
            arch_report(&resolution, &args.model, &names)
        }
    };
    let text = format!(
        "{} {}
",
        ui::badge(ui, "ARCH"),
        ui::text(ui, &report.summary())
    );
    Ok(Outcome::ok(json!(report), text))
}

// ---------------------------------------------------------------- bench

/// Usage and file errors are code 2; resources, missing binaries and cancellations 1;
/// everything that failed while running the bench 8.
fn bench_failure(e: &BenchError) -> Outcome {
    let code = match e {
        BenchError::InvalidArgument(_)
        | BenchError::Input { .. }
        | BenchError::OutputExists(_)
        | BenchError::Model(_) => exit::INVALID_INPUT,
        BenchError::InsufficientResources(_)
        | BenchError::BinaryNotFound(_)
        | BenchError::Cancelled => exit::GENERAL,
        _ => exit::BENCH_FAILED,
    };
    Outcome::fail(code, e).with_kind(e.kind())
}

/// Warnings never change the exit code, except threshold ones under `--fail-on-warn`
/// (code 9, the report still in `data`).
fn bench_outcome<R: Serialize>(run: &BenchRun<R>, fail_on_warn: bool, human: String) -> CoreResult {
    let data = to_value(&run.report);
    if fail_on_warn && !run.threshold_warnings.is_empty() {
        let mut failed = Outcome::fail(
            exit::THRESHOLD_EXCEEDED,
            format!("seuil dépassé : {}", run.threshold_warnings.join(" ; ")),
        )
        .with_kind("threshold_exceeded")
        .with_data(data, human);
        failed.warnings = run.warnings.clone();
        return Err(failed);
    }
    let mut outcome = Outcome::ok(data, human);
    outcome.warnings = run.warnings.clone();
    Ok(outcome)
}

/// The options with the GPU API of the llama-server, for the report label only.
fn labelled(
    mut generation: bench::generation::GenOptions,
    backend: Backend,
    llama_cpp_dir: Option<&Path>,
) -> bench::generation::GenOptions {
    if backend == Backend::LlamaServer {
        generation.gpu_api = crate::backend::gpu_api(llama_cpp_dir);
    }
    generation
}

/// `surgeon backend detect|show`: the machine's engines and the user configuration.
fn backend_cli(command: &BackendCommand, io: &mut Io) -> CoreResult {
    use crate::backend::{
        detect_hardware_and_backends, load_user_config, save_to, user_config_path,
    };
    let ui = io.ui;
    let invalid = |m: String| Outcome::fail(exit::INVALID_INPUT, m).with_kind("invalid_input");
    let user = load_user_config().map_err(invalid)?;
    let path = user_config_path();
    match command {
        BackendCommand::Detect { save } => {
            let report = detect_hardware_and_backends(&user);
            let saved = match (save, &path) {
                (true, Some(p)) => {
                    save_to(p, &report.recommended).map_err(invalid)?;
                    Some(p.clone())
                }
                (true, None) => {
                    return Err(invalid(format!(
                        "aucun dossier de configuration (désactivé par {})",
                        crate::backend::NO_USER_CONFIG_ENV
                    )))
                }
                _ => None,
            };
            let mut text = String::new();
            match &report.llama {
                Some(l) => {
                    text.push_str(&format!(
                        "{} llama-server {}\n",
                        ui::badge(&ui, "BACKEND"),
                        l.server.display()
                    ));
                    for d in &l.devices {
                        text.push_str(&format!(
                            "  {} {} ({} Mio)\n",
                            ui::metric(&ui, format!("{:<8}", d.id)),
                            d.name,
                            d.vram_mib.map_or("?".into(), |v| v.to_string())
                        ));
                    }
                    if let Some(e) = &l.error {
                        text.push_str(&format!("  ! --list-devices : {e}\n"));
                    }
                }
                None => text.push_str(&format!(
                    "{} llama-server introuvable\n",
                    ui::badge(&ui, "BACKEND")
                )),
            }
            text.push_str(&format!(
                "{} hipfire : {}\n",
                ui::badge(&ui, "BACKEND"),
                report.hipfire.note
            ));
            text.push_str(&format!(
                "{} recommandé : {} ({})\n",
                ui::badge(&ui, "BACKEND"),
                crate::backend::describe_config(&report.recommended),
                report.reason
            ));
            if let Some(p) = &saved {
                text.push_str(&format!("  configuration écrite : {}\n", p.display()));
            }
            Ok(Outcome::ok(json!({"report": report, "saved": saved}), text))
        }
        BackendCommand::Show => {
            let text = format!(
                "{} {} · {}\n",
                ui::badge(&ui, "BACKEND"),
                path.as_ref()
                    .map_or("(désactivée)".into(), |p| p.display().to_string()),
                crate::backend::describe_config(&user)
            );
            Ok(Outcome::ok(
                json!({
                    "path": path,
                    "exists": path.as_ref().is_some_and(|p| p.exists()),
                    "config": user,
                    "effective": crate::backend::describe_config(&user),
                }),
                text,
            ))
        }
    }
}

fn bench_cli(args: &BenchArgs, deps: &AppDeps, io: &mut Io) -> CoreResult {
    let ui = io.ui;
    let mut sink = CliSink::new(io);
    let invalid = |e: BenchError| bench_failure(&e);
    let timeout = |secs: Option<u64>| match secs {
        Some(0) => Err(invalid(BenchError::InvalidArgument(
            "--timeout-secs doit être >= 1".into(),
        ))),
        other => Ok(other.map(Duration::from_secs)),
    };
    match &args.command {
        BenchCommand::Logits(a) => {
            let req = LogitsRequest {
                before: a.before.clone(),
                after: a.after.clone(),
                tensor: a.tensor.clone(),
                top_k: a.top_k,
                worst: a.worst,
                warn_kl: a.warn_kl,
                warn_top1_shift: a.warn_top1_shift,
            };
            core_bench_logits(&req, a.fail_on_warn, deps, &ui, &mut sink)
        }
        BenchCommand::Ppl(a) => {
            let baseline = match (&a.baseline_model, a.baseline_ppl) {
                (Some(model), _) => Some(PplBaseline::Model(model.clone())),
                (None, Some(ppl)) => Some(PplBaseline::Value {
                    ppl,
                    err: a.baseline_ppl_err,
                }),
                (None, None) => None,
            };
            let req = PplRequest {
                model: a.model.clone(),
                text: a.text.clone(),
                baseline,
                ctx: a.ctx,
                chunks: a.chunks,
                threads: a.threads,
                n_gpu_layers: a.n_gpu_layers,
                llama_cpp_dir: a.llama_cpp_dir.clone(),
                warn_delta: a.warn_delta,
                timeout: timeout(a.timeout_secs)?,
                llama_args: bench::ppl_runner::validate_llama_args(&a.llama_args)
                    .map_err(invalid)?,
            };
            core_bench_ppl(&req, a.fail_on_warn, deps, &ui, &mut sink)
        }
        BenchCommand::Eval(a) => {
            let eval_backend = a.backend.unwrap_or_default();
            let req = EvalRequest {
                model: a.model.clone(),
                dataset: a.dataset.clone(),
                backend: eval_backend,
                baseline_model: a.baseline_model.clone(),
                limit: a.limit,
                max_new_tokens: a.max_new_tokens,
                template: Template::from_cli(a.prompt_template.as_deref()).map_err(invalid)?,
                format: a.format,
                llama_cpp_dir: a.llama_cpp_dir.clone(),
                timeout: timeout(a.timeout_secs)?,
                min_accuracy: a.min_accuracy,
                generation: labelled(
                    a.generation.options(),
                    eval_backend,
                    a.llama_cpp_dir.as_deref(),
                ),
            };
            core_bench_eval(&req, a.fail_on_warn, deps, &ui, &mut sink)
        }
        BenchCommand::Dump(a) => {
            let prompts = extract::load_prompt_texts(&a.prompts)
                .map_err(|m| invalid(BenchError::input(&a.prompts, m)))?;
            let req = DumpRequest {
                model: a.model.clone(),
                prompts,
                out: a.out.clone(),
                dtype: a.dtype,
                out_dtype: a.out_dtype,
                max_tokens: a.max_tokens,
                max_total_positions: a.max_total_positions,
                overwrite: a.overwrite,
            };
            core_bench_dump(&req, deps, &ui, &mut sink)
        }
        BenchCommand::Refusal(a) => {
            let refusal_backend = a.backend.unwrap_or(Backend::Native);
            let prompts = extract::load_prompt_texts(&a.prompts)
                .map_err(|m| invalid(BenchError::input(&a.prompts, m)))?;
            let req = RefusalRequest {
                model: a.model.clone(),
                prompts,
                baseline_model: a.baseline_model.clone(),
                max_new_tokens: a.max_new_tokens,
                chat_template: a.chat_template,
                markers: markers(&a.markers).map_err(invalid)?,
                generation: labelled(
                    a.generation.options(),
                    refusal_backend,
                    a.judge_args.llama_cpp_dir.as_deref(),
                ),
                backend: refusal_backend.label().into(),
            };
            let judging = match a.judge {
                JudgeMode::Rules if a.judge_args.judge_model.is_some() => {
                    return Err(invalid(BenchError::InvalidArgument(
                        "--judge-model demande --judge llm ou --judge both".into(),
                    )))
                }
                JudgeMode::Rules => None,
                mode => Some(RefusalJudging {
                    setup: judge_setup(&a.judge_args, timeout(a.judge_args.timeout_secs)?)
                        .map_err(invalid)?,
                    compare_rules: mode == JudgeMode::Both,
                    max_unparsable: a.max_unparsable,
                    calibration: a.judge_calibration.clone(),
                    min_calibration_n: a.min_calibration_n,
                    min_kappa: a.min_kappa,
                }),
            };
            let llama_cpp_dir = a.judge_args.llama_cpp_dir.as_deref();
            core_bench_refusal(
                &req,
                refusal_backend,
                judging.as_ref(),
                llama_cpp_dir,
                a.fail_on_warn,
                deps,
                &ui,
                &mut sink,
            )
        }
        BenchCommand::JudgeSample(a) => {
            let prompts = extract::load_prompt_texts(&a.prompts)
                .map_err(|m| invalid(BenchError::input(&a.prompts, m)))?;
            let req = SampleRequest {
                responses: a.responses.clone(),
                prompts,
                n: a.n,
                seed: a.seed,
                out: a.out.clone(),
                overwrite: a.overwrite,
            };
            let run = bench::calibration::sample(&req).map_err(invalid)?;
            let text = bench::calibration::sample_text(&ui, &run);
            bench_outcome(&run, false, text)
        }
        BenchCommand::JudgeEval(a) => {
            let req = JudgeEvalRequest {
                labels: a.labels.clone(),
                setup: judge_setup(&a.judge_args, timeout(a.judge_args.timeout_secs)?)
                    .map_err(invalid)?,
                markers: markers(&a.markers).map_err(invalid)?,
                out: a.out.clone(),
                overwrite: a.overwrite,
            };
            let llama_cpp_dir = a.judge_args.llama_cpp_dir.as_deref();
            core_bench_judge_eval(&req, llama_cpp_dir, deps, &ui, &mut sink)
        }
    }
}

fn markers(args: &MarkerArgs) -> bench::Result<Markers> {
    Markers::new(&args.marker_langs, args.markers_file.as_deref())
}

fn judge_setup(args: &JudgeArgs, timeout: Option<Duration>) -> bench::Result<JudgeSetup> {
    let model = args.judge_model.clone().ok_or_else(|| {
        BenchError::InvalidArgument("--judge-model <GGUF> est requis pour un juge LLM".into())
    })?;
    Ok(JudgeSetup {
        model,
        role: args.judge_role,
        template: args.judge_template.clone(),
        cache: args.judge_cache.clone(),
        partial_as_refusal: args.partial_as_refusal,
        timeout,
    })
}

/// The llama.cpp generator of a judge GGUF, from `deps.generator_factory`.
fn judge_generator(
    gguf: &Path,
    llama_cpp_dir: Option<&Path>,
    deps: &AppDeps,
) -> bench::Result<Box<dyn TextGenerator>> {
    let args = EvalArgs {
        model: gguf.to_path_buf(),
        prompts: PathBuf::new(),
        max_new_tokens: 0,
        export_gguf: None,
        quant: "Q4_K_M".into(),
        llama_cpp_dir: llama_cpp_dir.map(Path::to_path_buf),
        allow_degraded: false,
    };
    (deps.generator_factory)(gguf, &args).map_err(BenchError::from)
}

/// Shared by the CLI and the MCP tool; cancellation checked before each block.
pub fn core_bench_logits(
    req: &LogitsRequest,
    fail_on_warn: bool,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let run = bench::logits_metrics::run(
        req,
        deps.probe.as_ref(),
        &deps.interrupted,
        &mut |done, total| sink.event(ProgressEvent::BenchLogitsBlock { done, total }),
    )
    .map_err(|e| bench_failure(&e))?;
    bench_outcome(&run, fail_on_warn, bench_report::logits_text(ui, &run))
}

/// Shared by the CLI and the MCP tool; cancellation kills llama-perplexity.
pub fn core_bench_ppl(
    req: &PplRequest,
    fail_on_warn: bool,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let run = bench::ppl_runner::run(
        req,
        deps.runner.as_ref(),
        &deps.interrupted,
        &mut |done, total| sink.event(ProgressEvent::BenchPplChunk { done, total }),
    )
    .map_err(|e| bench_failure(&e))?;
    bench_outcome(&run, fail_on_warn, bench_report::ppl_text(ui, &run))
}

/// The pure Rust generator on a Qwen2 directory; its loading notes (bf16 fallback) go to
/// the sink as warnings.
fn native_generator(
    model: &Path,
    deps: &AppDeps,
    sink: &RefCell<&mut dyn ProgressSink>,
) -> bench::Result<NativeGenerator> {
    let generator = NativeGenerator::load(
        model,
        extract::DTypeChoice::Auto,
        deps.probe.as_ref(),
        Arc::clone(&deps.interrupted),
    )?;
    for warning in generator.warnings() {
        sink.borrow_mut()
            .event(ProgressEvent::Warn(warning.clone()));
    }
    Ok(generator)
}

/// Shared by the CLI and the MCP tool. Generators come from `deps.generator_factory` (or
/// the native backend), set to greedy seeded decoding; cancellation stops the running
/// generation.
pub fn core_bench_eval(
    req: &EvalRequest,
    fail_on_warn: bool,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let sink = RefCell::new(sink);
    req.generation.validate().map_err(|e| bench_failure(&e))?;
    let pool = generation_pool(&req.generation, req.backend, &sink)?;
    let dir = req.llama_cpp_dir.as_deref();
    let mut make = |gguf: &Path| {
        let tag = "bench-eval";
        bench_generator(
            gguf,
            req.backend,
            &req.generation,
            &pool,
            dir,
            tag,
            deps,
            &sink,
        )
    };
    let run = bench::score::run(
        req,
        deps.runner.as_ref(),
        &mut make,
        &deps.interrupted,
        &mut |done, total| {
            sink.borrow_mut()
                .event(ProgressEvent::BenchEvalQuestion { done, total })
        },
    )
    .map_err(|e| bench_failure(&e))?;
    bench_outcome(&run, fail_on_warn, bench_report::eval_text(ui, &run))
}

/// A bench generator for `backend`: the native one on a SafeTensors folder (batches,
/// sampling), `llama-server` or `llama-completion` / `llama-cli` on a GGUF. A SafeTensors
/// model given to a llama backend is converted to a temporary GGUF, kept as long as the
/// generator.
#[allow(clippy::too_many_arguments)]
pub(crate) fn bench_generator(
    model: &Path,
    backend: Backend,
    generation: &bench::generation::GenOptions,
    pool: &Option<Arc<rayon::ThreadPool>>,
    llama_cpp_dir: Option<&Path>,
    tag: &str,
    deps: &AppDeps,
    sink: &RefCell<&mut dyn ProgressSink>,
) -> bench::Result<Box<dyn TextGenerator>> {
    if backend == Backend::Native {
        let mut native = native_generator(model, deps, sink)?;
        native.set_batching(generation.batch_size, generation.jobs, pool.clone());
        return Ok(Box::new(native));
    }
    if backend == Backend::Hipfire {
        return Err(BenchError::InvalidArgument(
            crate::backend::hipfire::unavailable(&crate::backend::BackendConfig::default()),
        ));
    }
    let gguf = crate::eval::ensure_gguf_as(
        model,
        llama_cpp_dir,
        tag,
        deps.runner.as_ref(),
        generation.gguf_dtype,
    )
    .map_err(|e| BenchError::Conversion(e.to_string()))?;
    let inner: Box<dyn TextGenerator> = match backend {
        Backend::LlamaServer => {
            let binary = crate::eval::server::locate_server(llama_cpp_dir)?;
            let server = crate::eval::server::LlamaServerGenerator::start(
                &binary,
                gguf.path(),
                &generation.server(),
                &deps.interrupted,
            )?;
            Box::new(server)
        }
        _ => {
            let args = EvalArgs {
                model: gguf.path().to_path_buf(),
                prompts: PathBuf::new(),
                max_new_tokens: 0,
                export_gguf: None,
                quant: "Q4_K_M".into(),
                llama_cpp_dir: llama_cpp_dir.map(Path::to_path_buf),
                allow_degraded: false,
            };
            (deps.generator_factory)(gguf.path(), &args)?
        }
    };
    Ok(Box::new(WithGguf { inner, _gguf: gguf }))
}

/// A generator and the temporary GGUF it runs on, deleted when both are dropped.
struct WithGguf {
    inner: Box<dyn TextGenerator>,
    _gguf: crate::eval::GgufModel,
}

impl TextGenerator for WithGguf {
    fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: u32,
        on_chunk: &mut dyn FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> crate::eval::Result<crate::eval::GenerationOutcome> {
        self.inner.generate(prompt, max_new_tokens, on_chunk)
    }
    fn configure(&mut self, decoding: &crate::eval::Decoding) -> bool {
        self.inner.configure(decoding)
    }
    fn constrain(&mut self, gbnf: &str) -> bool {
        self.inner.constrain(gbnf)
    }
    fn generate_job(
        &mut self,
        job: &crate::eval::GenJob,
        max_new_tokens: u32,
        on_chunk: &mut dyn FnMut(&str) -> std::ops::ControlFlow<()>,
    ) -> crate::eval::Result<crate::eval::GenerationOutcome> {
        self.inner.generate_job(job, max_new_tokens, on_chunk)
    }
    fn generate_many(
        &mut self,
        jobs: &[crate::eval::GenJob],
        max_new_tokens: u32,
        cancel: &AtomicBool,
        on_chunk: &mut dyn FnMut(usize, &str) -> std::ops::ControlFlow<()>,
        on_done: &mut dyn FnMut(usize),
    ) -> crate::eval::Result<Vec<crate::eval::Result<crate::eval::GenerationOutcome>>> {
        self.inner
            .generate_many(jobs, max_new_tokens, cancel, on_chunk, on_done)
    }
}

/// The rayon pool of a native generation, built only for `--threads` (otherwise the global
/// pool, sized by `RAYON_NUM_THREADS` or the cores, is that one budget), and its
/// over-subscription warnings sent to the sink. A llama backend runs its own threads.
fn generation_pool(
    generation: &bench::generation::GenOptions,
    backend: Backend,
    sink: &RefCell<&mut dyn ProgressSink>,
) -> std::result::Result<Option<Arc<rayon::ThreadPool>>, Outcome> {
    // The default options run as before: no process scan, no new warning.
    if backend != Backend::Native || generation.is_historical(true) {
        return Ok(None);
    }
    let (threads, warnings) = generation.thread_budget(other_surgeon_processes());
    for w in warnings {
        sink.borrow_mut().event(ProgressEvent::Warn(w));
    }
    if generation.threads.is_none() {
        return Ok(None);
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .map(|pool| Some(Arc::new(pool)))
        .map_err(|e| Outcome::fail(exit::GENERAL, format!("pool de threads : {e}")))
}

/// Other running `surgeon` processes (they compete for cores and memory bandwidth).
fn other_surgeon_processes() -> usize {
    let Ok(mut sys) = sysinfo::System::new() else {
        return 0;
    };
    sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    let me = std::process::id();
    sys.processes()
        .values()
        .filter(|p| p.pid().as_u32() != me)
        .filter(|p| {
            let name = p.name().to_string_lossy().to_ascii_lowercase();
            name == "surgeon" || name == "surgeon.exe"
        })
        .count()
}

/// Shared by the CLI and the MCP tool: native greedy generation, refusal markers, paired
/// comparison against a baseline; with `judging`, a llama.cpp judge then labels every
/// answer. Nothing is written but the verdict cache of the judge, when one is given.
#[allow(clippy::too_many_arguments)]
pub fn core_bench_refusal(
    req: &RefusalRequest,
    backend: Backend,
    judging: Option<&RefusalJudging>,
    llama_cpp_dir: Option<&Path>,
    fail_on_warn: bool,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let sink = RefCell::new(sink);
    // Before the generation: a wrong judge file must not cost a full run.
    let judging = judging
        .map(bench::judge::prepare)
        .transpose()
        .map_err(|e| bench_failure(&e))?;
    req.generation.validate().map_err(|e| bench_failure(&e))?;
    let pool = generation_pool(&req.generation, backend, &sink)?;
    let mut make = |model: &Path| {
        let (tag, generation) = ("bench-refusal", &req.generation);
        bench_generator(
            model,
            backend,
            generation,
            &pool,
            llama_cpp_dir,
            tag,
            deps,
            &sink,
        )
    };
    let mut run = bench::refusal::run(req, &mut make, &deps.interrupted, &mut |done, total| {
        sink.borrow_mut()
            .event(ProgressEvent::BenchRefusalPrompt { done, total })
    })
    .map_err(|e| bench_failure(&e))?;
    if let Some(judging) = judging {
        bench::judge::annotate(
            &mut run,
            &req.prompts,
            judging,
            &mut |gguf| judge_generator(gguf, llama_cpp_dir, deps),
            &deps.interrupted,
            &mut |done, total| {
                sink.borrow_mut()
                    .event(ProgressEvent::BenchJudgeVerdict { done, total })
            },
        )
        .map_err(|e| bench_failure(&e))?;
    }
    bench_outcome(&run, fail_on_warn, bench_report::refusal_text(ui, &run))
}

/// `bench judge-eval`: a llama.cpp judge against human labels. Writes the calibration
/// file when the request has one.
pub fn core_bench_judge_eval(
    req: &JudgeEvalRequest,
    llama_cpp_dir: Option<&Path>,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let run = bench::calibration::eval(
        req,
        &mut |gguf| judge_generator(gguf, llama_cpp_dir, deps),
        &deps.interrupted,
        &mut |done, total| sink.event(ProgressEvent::BenchJudgeVerdict { done, total }),
    )
    .map_err(|e| bench_failure(&e))?;
    bench_outcome(&run, false, bench::calibration::eval_text(ui, &run))
}

/// Shared by the CLI and the MCP tool; nothing is written unless the whole dump succeeds.
pub fn core_bench_dump(
    req: &DumpRequest,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let run = bench::dump::run(
        req,
        deps.probe.as_ref(),
        &deps.interrupted,
        &mut |done, total| sink.event(ProgressEvent::BenchDumpPrompt { done, total }),
    )
    .map_err(|e| bench_failure(&e))?;
    bench_outcome(&run, false, bench_report::dump_text(ui, &run))
}

/// `[METHOD]` line of an svd / leace extraction (mean_diff output is unchanged).
fn method_line(ui: &Ui, diag: &Diagnostics) -> String {
    let opt = |v: Option<f64>| v.map_or("—".to_owned(), |x| format!("{x:.3}"));
    // Nothing more without a subspace: the line of one direction is what it was.
    let subspace = |s: &Option<extract::SubspaceDiagnostics>| match s {
        Some(s) => format!(
            " · sous-espace de rang {} · ‖BᵀA−I‖ {}",
            ui::metric(ui, s.rank),
            ui::metric(ui, format!("{:.1e}", s.gram_bta_error))
        ),
        None => String::new(),
    };
    let body = match diag {
        // Registered methods never run here: the front-end only knows the built-in registry.
        Diagnostics::Json(_) => return String::new(),
        Diagnostics::Builtin(MethodDiagnostics::Svd(d)) => format!(
            "svd · {} paires · σ₁²/Σσ² {} · cos(μ₊−μ₋) {} · split-half {}{}",
            ui::metric(ui, d.n_pairs),
            ui::metric(ui, format!("{:.3}", d.explained_variance_ratio)),
            ui::metric(ui, opt(d.cosine_with_mean_diff)),
            ui::metric(ui, opt(d.split_half_cosine)),
            subspace(&d.subspace)
        ),
        Diagnostics::Builtin(MethodDiagnostics::Leace(d)) => format!(
            "leace · rang {} · λ {} · κ {} · cov(x,z) {} → {} · hors-échantillon {}{}",
            ui::metric(ui, d.rank),
            ui::metric(ui, d.shrinkage),
            ui::metric(ui, format!("{:.3e}", d.kappa)),
            ui::metric(ui, format!("{:.3e}", d.cross_cov_before)),
            ui::metric(ui, format!("{:.3e}", d.cross_cov_after)),
            ui::metric(ui, opt(d.holdout_cross_cov_ratio)),
            subspace(&d.subspace)
        ),
    };
    format!("{} {body}\n", ui::badge(ui, "METHOD"))
}

// ---------------------------------------------------------------- autotune

fn autotune_failure(e: &AutotuneError) -> Outcome {
    let code = match e {
        AutotuneError::InvalidArgument(_)
        | AutotuneError::Input { .. }
        | AutotuneError::OutputExists(_) => exit::INVALID_INPUT,
        AutotuneError::InsufficientResources(_) | AutotuneError::Cancelled => exit::GENERAL,
        AutotuneError::Model(_)
        | AutotuneError::Solve(_)
        | AutotuneError::Compute(_)
        | AutotuneError::Io { .. } => exit::EXTRACT_FAILED,
    };
    Outcome::fail(code, e).with_kind(e.kind())
}

fn autotune_cli(
    args: &AutotuneArgs,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let request = AutotuneRequest {
        model: args.model.clone(),
        vector: args.vector.clone(),
        eval_dataset: args.eval_dataset.clone(),
        layer_range: args.target_layer_range,
        readout_layer: args.readout_layer,
        proj: args.proj,
        objective: args.objective,
        target_projection: args.target_projection,
        l2_reg: args.l2_reg,
        probe_alpha: args.probe_alpha,
        alpha_max: args.alpha_max,
        min_alpha: args.min_alpha,
        max_forwards: args.max_forwards,
        max_tokens: args.max_tokens,
        dtype: args.dtype,
        site: args.site,
        pooling: args.pooling,
        chat_template: args.chat_template,
        out: args.out.clone(),
        overwrite: args.overwrite,
    };
    core_autotune(&request, deps, ui, sink)
}

/// Shared by the CLI and the MCP tool; cancellation is checked before every forward and
/// nothing is written then.
pub fn core_autotune(
    request: &AutotuneRequest,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let run = autotune::run(
        request,
        deps.probe.as_ref(),
        &deps.interrupted,
        &mut |event| match event {
            AutotuneEvent::Planned { .. } => {}
            AutotuneEvent::Forward { done, total } => {
                sink.event(ProgressEvent::AutotuneForward { done, total })
            }
            AutotuneEvent::Solve => sink.event(ProgressEvent::AutotuneSolve),
        },
    )
    .map_err(|e| autotune_failure(&e))?;
    let mut outcome = Outcome::ok(to_value(&run.report), autotune_text(ui, &run));
    outcome.warnings = run.warnings.clone();
    Ok(outcome)
}

fn autotune_text(ui: &Ui, run: &AutotuneRun) -> String {
    let r = &run.report;
    let opt = |v: Option<f64>| v.map_or("—".to_owned(), |x| format!("{x:.4}"));
    let alphas: Vec<String> = r
        .alphas
        .iter()
        .map(|a| match a.dropped {
            true => format!("L{} —", a.layer),
            false => format!("L{} {:+.4}", a.layer, a.alpha),
        })
        .collect();
    let worst_linearity = r
        .linearity
        .iter()
        .map(|c| c.relative_deviation)
        .fold(0.0f64, f64::max);
    let mut lines = vec![
        format!(
            "{} lecture L{} · {} · {} · {} prompts · {} couches · {} forwards",
            ui::badge(ui, "AUTOTUNE"),
            ui::metric(ui, r.readout_layer),
            ui::metric(ui, r.site.as_str()),
            ui::metric(ui, r.pooling.as_str()),
            ui::metric(ui, r.n_prompts),
            ui::metric(ui, r.layers.len()),
            ui::metric(ui, r.forwards)
        ),
        format!(
            "{} δ {} · s⁰ moyen {} → cible {} · linéarité (écart max) {}",
            ui::badge(ui, "RESPONSE"),
            ui::metric(ui, r.probe_alpha),
            ui::metric(ui, format!("{:.4}", r.baseline_mean)),
            ui::metric(ui, format!("{:.4}", r.target_mean)),
            ui::metric(ui, format!("{:.1} %", worst_linearity * 100.0))
        ),
        format!(
            "{} λ {} · R² {} · RMSE {} · LOO {} · κ {} · ‖α‖ {}",
            ui::badge(ui, "RIDGE"),
            ui::metric(ui, format!("{:.1e}", r.ridge.lambda)),
            ui::metric(ui, opt(r.ridge.r2)),
            ui::metric(ui, format!("{:.4}", r.ridge.rmse)),
            ui::metric(ui, opt(r.ridge.loocv_rmse)),
            ui::metric(ui, format!("{:.2e}", r.ridge.cond_estimate)),
            ui::metric(ui, format!("{:.4}", r.ridge.alpha_norm))
        ),
        format!("  {}", ui::dim(ui, &format!("α : {}", alphas.join(" · ")))),
        format!(
            "{} décalage prévu {} · mesuré {} · écart à la cible {} → {}",
            ui::badge(ui, "VERDICT"),
            ui::metric(ui, format!("{:+.4}", r.validation.predicted_mean_shift)),
            ui::metric(ui, format!("{:+.4}", r.validation.actual_mean_shift)),
            ui::metric(ui, format!("{:.4}", r.validation.target_rmse_before)),
            ui::metric(ui, format!("{:.4}", r.validation.target_rmse_after))
        ),
    ];
    lines.push(match (&r.out, &r.patch) {
        (Some(out), _) => format!("{} {}", ui::badge(ui, "WRITE"), ui::path(ui, out)),
        (None, Some(_)) => format!(
            "  {}",
            ui::dim(ui, "patch dans data.patch (--out pour l'écrire)")
        ),
        (None, None) => format!("  {}", ui::dim(ui, "aucun patch produit")),
    });
    lines.extend(run.warnings.iter().map(|w| warning_line(ui, w)));
    lines.push(format!("  {}", ui::dim(ui, &r.recommendation)));
    lines.join("\n") + "\n"
}
