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
//!                     -> engine::find_mlp_tensors -> tensor_stats (mmap view, streamed)
//!          -> apply   -> patch::load_patch -> validate_patch -> hardware::evaluate_budget
//!                     -> EngineBackend::open (WritableModel per shard)
//!                     -> apply_patch(dry run: preflight, no write)
//!                     -> BackupSet::create_many_in (touched shards only)
//!                     -> apply_patch -> EngineBackend::shift_tensor -> WritableModel::shift_tensor
//!                        ; error | NaN | Ctrl-C -> RollbackHandle::rollback ; ok -> flush -> commit
//!          -> extract -> load_prompt_texts -> core_extract_request -> extract::run_extraction
//!          -> eval    -> load_prompts -> export_gguf (temporary f16, unless .gguf)
//!                     -> generator_factory -> run_eval -> export_if_healthy
//! Every human byte goes through `ui::*` (output audit in `ui.rs`); `--json` builds
//! `Ui::off()` for both streams, so no rendering reaches stdout or stderr.
//! emit -> JSON envelope | human text (write errors ignored: closed pipe never panics)
//! ```
//!
//! Exit codes: 0 ok, 1 general error, 2 invalid CLI/patch/input, 3 execution failed and
//! model intact (rolled back or never touched), 4 CRITICAL (rollback failed, or model left
//! modified without backup), 5 eval degraded/broken, 6 GGUF export failed, 7 extraction
//! failed, 70 internal.
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

use crate::cli::{ApplyArgs, CheckArgs, Cli, Commands, EvalArgs, ExtractArgs, InspectArgs};
use crate::engine::stats::TensorStats;
use crate::engine::{
    self, BroadcastAxis, DType, EngineError, LayerSpec, MlpProj, ReadOnlyModel, ShiftOp,
    ShiftStats, WritableModel,
};
use crate::eval::{
    export_gguf, export_if_healthy, load_prompts, locate_binary, run_eval, CommandRunner,
    EvalConfig, EvalError, EvalPrompt, EvalReport, EvalVerdict, GenerationOutcome,
    GgufExportConfig, LlamaCliGenerator, PromptStatus, SystemCommandRunner, TextGenerator,
    ALLOWED_QUANTS,
};
use crate::extract::{
    self, ExtractError, ExtractEvent, ExtractObserver, ExtractReport, ExtractRequest, PromptSet,
};
use crate::hardware::{
    self, BudgetPolicy, HardwareBudget, HardwareError, HardwareProbe, SystemProbe, Verdict,
};
use crate::patch::{
    self, apply_patch, load_patch, validate_patch, ApplyReport, Axis, BackupSet, LayerSelection,
    OpStats, OperationKind, PatchBackend, PatchError, PatchFile, Projection, RollbackHandle,
    ValidationIssue, ValidationOptions,
};
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

pub fn run_with(cli: Cli, deps: &AppDeps, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
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
    }
}

fn dispatch(command: &Commands, deps: &AppDeps, io: &mut Io) -> Outcome {
    let result = match command {
        Commands::Check(args) => core_check(args, deps, &io.ui),
        Commands::Inspect(args) => core_inspect(args, deps, &io.ui),
        Commands::Apply(args) => apply(args, deps, io),
        Commands::Eval(args) => eval(args, deps, &io.ui),
        Commands::Extract(args) => {
            let ui = io.ui;
            core_extract(args, deps, &ui, &mut CliSink::new(io))
        }
        // Routed by run_with before dispatch: they have no envelope.
        Commands::Schema(_) | Commands::Mcp(_) | Commands::Tui(_) => Err(Outcome::fail(
            exit::INTERNAL,
            "commande sans enveloppe JSON",
        )),
    };
    result.unwrap_or_else(|failure| failure)
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
    fn ok(data: Value, human: String) -> Self {
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
    let shards = hardware::compute_model_size(&args.model)?.shard_files;
    let layers = LayerSpec::from(&args.layers);
    let projs: Vec<MlpProj> = match args.projections.is_empty() {
        true => vec![MlpProj::Gate, MlpProj::Up, MlpProj::Down],
        false => args
            .projections
            .iter()
            .copied()
            .map(MlpProj::from)
            .collect(),
    };
    let sharded = shards.len() > 1;
    let mut tensors = Vec::new();
    let mut rows = Vec::new();
    for shard in &shards {
        let model = ReadOnlyModel::open(shard)?;
        for info in engine::find_mlp_tensors(model.index(), &layers, &projs) {
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
    human.push_str(&format!(
        "{} {} tenseurs sélectionnés\n{}\n",
        ui::badge(ui, "INSPECT"),
        ui::metric(ui, rows.len()),
        ui::separator(ui)
    ));
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
        json!({ "shards": shards, "tensors": tensors }),
        human,
    ))
}

/// `[L12]`, or `[—]` for a tensor outside numbered layers.
fn layer_label(name: &str) -> String {
    engine::parse_layer_index(name).map_or("[—]".to_owned(), |l| format!("[L{l}]"))
}

fn projection_label(name: &str) -> &str {
    name.split('.')
        .find(|part| part.ends_with("_proj"))
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

impl From<Projection> for MlpProj {
    fn from(p: Projection) -> Self {
        match p {
            Projection::Gate => MlpProj::Gate,
            Projection::Up => MlpProj::Up,
            Projection::Down => MlpProj::Down,
        }
    }
}

impl From<OperationKind> for ShiftOp {
    fn from(op: OperationKind) -> Self {
        match op {
            OperationKind::Subtract => ShiftOp::Subtract,
            OperationKind::Add => ShiftOp::Add,
        }
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
        })
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
}

impl PatchBackend for EngineBackend {
    fn resolve_targets(
        &self,
        layers: &LayerSelection,
        projections: &[Projection],
    ) -> patch::Result<Vec<String>> {
        let spec = LayerSpec::from(layers);
        let projs: Vec<MlpProj> = projections.iter().copied().map(MlpProj::from).collect();
        let mut names: Vec<String> = self
            .shards
            .iter()
            .flat_map(|(_, model)| {
                engine::find_mlp_tensors(model.index(), &spec, &projs)
                    .into_iter()
                    .map(|t| t.name.clone())
                    .collect::<Vec<_>>()
            })
            .collect();
        names.sort_by(|a, b| {
            (engine::parse_layer_index(a), a).cmp(&(engine::parse_layer_index(b), b))
        });
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
        self.shards[shard]
            .1
            .shift_tensor(tensor_name, direction, alpha, op.into(), axis.into())
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
    totals: RefCell<Vec<usize>>,
    op: usize,
    done: usize,
    overall_done: usize,
    max_delta: f32,
}

const BAR_WIDTH: usize = 20;

impl<'a, B: PatchBackend> ObservedBackend<'a, B> {
    fn new(inner: &'a mut B, sink: &'a mut dyn ProgressSink, op_ids: Vec<String>) -> Self {
        Self {
            inner,
            sink,
            op_ids,
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
            self.sink.event(ProgressEvent::OpDone {
                op_id: self.op_id(),
                max_delta: self.max_delta,
                tensors: self.done,
            });
            self.op += 1;
            self.done = 0;
            self.max_delta = 0.0;
        }
        Ok(stats)
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
            } => {
                self.io.line(&format!(
                    "{} {} · {} tenseurs · vecteur de {}",
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
            ProgressEvent::Shift { .. }
            | ProgressEvent::EvalPrompt { .. }
            | ProgressEvent::ExtractWriting => {}
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
    core_apply(&request, deps, &mut CliSink::new(io))
}

/// validate -> budget (warning only) -> open shards -> dry-run preflight -> backup of the
/// touched shards -> apply_patch (rollback on error, NaN, Ctrl-C or cancellation).
pub fn core_apply(
    request: &ApplyRequest,
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
    for w in &warnings {
        sink.event(ProgressEvent::Warn(w.clone()));
    }

    let mut backend = EngineBackend::open(&shards, Arc::clone(&deps.interrupted))?;
    let preview = apply_patch(&patch, &mut backend, None, true)?;
    if request.dry_run {
        for op in &preview.operations {
            sink.event(ProgressEvent::DryRun {
                op_id: op.id.clone(),
                tensors: op.tensors.iter().map(|t| t.name.clone()).collect(),
                vector_len: op.vector_len,
            });
        }
        return Ok(apply_outcome(&preview, &[], &warnings));
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

    let touched = backend.shards_holding(
        preview
            .operations
            .iter()
            .flat_map(|op| op.tensors.iter().map(|t| t.name.as_str())),
    );
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
    let op_ids = patch.operations.iter().map(|op| op.id.clone()).collect();
    let guard = backup.as_mut().map(|b| b as &mut dyn RollbackHandle);
    let result = apply_patch(
        &patch,
        &mut ObservedBackend::new(&mut backend, sink, op_ids),
        guard,
        false,
    );
    match result {
        Ok(report) => {
            sink.event(ProgressEvent::Commit {
                seconds: started.elapsed().as_secs_f64(),
            });
            Ok(apply_outcome(&report, &backup_files, &warnings))
        }
        Err(e) => {
            if let Some(outcome) = rollback_outcome(&e, &backup_files) {
                sink.event(ProgressEvent::Rollback(outcome));
            }
            Err(e.into())
        }
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

/// Deletes the temporary GGUF on every path, including errors and panics.
struct TempFile(PathBuf);

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
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
    let is_gguf = args
        .model
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
    if is_gguf && args.export_gguf.is_some() {
        return Err(invalid(
            "--export-gguf convertit depuis SafeTensors : impossible depuis un .gguf".into(),
        ));
    }
    let model_dir = match args.model.is_dir() {
        true => args.model.clone(),
        false => args
            .model
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf),
    };
    let llama_dir = || {
        args.llama_cpp_dir.clone().ok_or_else(|| {
            invalid("--llama-cpp-dir (ou LLAMA_CPP_DIR) requis pour convertir en GGUF".into())
        })
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

    let mut temp = None;
    let gguf = match is_gguf {
        true => args.model.clone(),
        false => {
            // Next to the model: an f16 copy can be tens of GB, often too big for /tmp.
            let file = TempFile(
                model_dir.join(format!(".weight-surgeon-eval-{}.gguf", std::process::id())),
            );
            let cfg = export_cfg(llama_dir()?, file.0.clone(), None);
            export_gguf(&cfg, deps.runner.as_ref()).map_err(|e| {
                Outcome::fail(
                    exit::GENERAL,
                    format!("conversion GGUF pour l'évaluation : {e}"),
                )
                .with_kind("eval_failed")
            })?;
            let path = file.0.clone();
            temp = Some(file);
            path
        }
    };

    let generator = (deps.generator_factory)(&gguf, args)
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
    drop(temp);

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

/// CLI entry: reads the two prompt files, then [`core_extract_request`].
pub fn core_extract(
    args: &ExtractArgs,
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
    core_extract_request(&request, deps, ui, sink)
}

/// Shared by the CLI and the MCP tool. Cancellation (`deps.interrupted`) is honored
/// between two tensors and between two prompts; nothing is written then.
pub fn core_extract_request(
    request: &ExtractRequest,
    deps: &AppDeps,
    ui: &Ui,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let mut observer = SinkObserver {
        sink,
        interrupted: &deps.interrupted,
        request,
    };
    let report = extract::run_extraction(request, deps.probe.as_ref(), &mut observer)
        .map_err(|e| extract_failure(&e))?;
    let mut outcome = Outcome::ok(to_value(&report), extract_text(ui, &report));
    outcome.warnings = report.warnings.clone();
    Ok(outcome)
}

fn extract_failure(e: &ExtractError) -> Outcome {
    let code = match e {
        ExtractError::InvalidArgument(_)
        | ExtractError::Prompts { .. }
        | ExtractError::OutputExists(_) => exit::INVALID_INPUT,
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
    for w in &report.warnings {
        text.push_str(&warning_line(ui, w));
        text.push('\n');
    }
    text
}
