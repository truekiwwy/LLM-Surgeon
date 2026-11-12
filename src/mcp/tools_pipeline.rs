//! MCP tool `surgeon_pipeline`: a flat, typed `action` over one job directory. `start`
//! launches the worker (`surgeon pipeline run --watch-stdin`) as a child of the server and
//! returns at once; the server holds it and stops it when it shuts down (stdin closed: the
//! worker cancels cleanly; killed if it does not within the grace period). One job at a
//! time. Every path goes through the sandbox, every file the configuration names included;
//! no free argument reaches the worker, and the `--allow-*` options of the CLI do not
//! exist here: the safety gate cannot be disabled and a Test Set cannot be reused.

use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};
use crate::pipeline::config::inputs;
use crate::pipeline::front::{self, failure};
use crate::pipeline::job::{self, Job};
use crate::pipeline::run::{self, RunOptions};

pub const TOOL_PIPELINE: &str = "surgeon_pipeline";
/// Time a worker gets to cancel cleanly when the server shuts down.
const GRACE: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PipelineAction {
    /// Estimations (disque, durée, puissance) ; écrit plan.json sauf --read-only.
    Plan,
    /// Lance (ou reprend) le job en arrière-plan ; renvoie job_id tout de suite.
    Start,
    /// Étape, avancement, temps restant, portes Dev déjà évaluées.
    Status,
    /// Demande l'arrêt ; la copie de travail est supprimée.
    Cancel,
    /// Rapport final (agrégats seulement pour le Test).
    Report,
    /// Écrit le patch retenu (promoted/patch.json) après un GO ; ne l'applique jamais.
    PromotePatch,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_pipeline : une action sur un dossier de job, dans un dossier autorisé."
)]
pub struct PipelineToolArgs {
    pub action: PipelineAction,
    #[schemars(description = "Dossier du job (créé au besoin), dans un dossier autorisé.")]
    pub job_dir: PathBuf,
    #[serde(default)]
    #[schemars(
        description = "plan, start : pipeline.json (surgeon schema --kind pipeline). Absent pour start : reprise du job existant. Tous les fichiers qu'il nomme doivent être dans un dossier autorisé."
    )]
    pub config: Option<PathBuf>,
}

/// The worker started by this server, if any.
struct Running {
    child: Child,
    /// Held open: its end tells the worker the server is gone.
    stdin: Option<ChildStdin>,
    job_dir: PathBuf,
}

/// The server's workers: at most one, stopped when the server ends.
#[derive(Default)]
pub struct PipelineJobs {
    running: Mutex<Option<Running>>,
}

impl PipelineJobs {
    /// The job of the worker started here, while it is alive (a finished one is reaped).
    fn live(&self) -> Option<PathBuf> {
        let mut slot = self.running.lock().ok()?;
        let alive = slot
            .as_mut()
            .is_some_and(|r| matches!(r.child.try_wait(), Ok(None)));
        if !alive {
            *slot = None;
        }
        slot.as_ref().map(|r| r.job_dir.clone())
    }

    fn busy(&self) -> bool {
        self.running.is_poisoned() || self.live().is_some()
    }

    fn hold(&self, child: Child, stdin: Option<ChildStdin>, job_dir: &Path) {
        if let Ok(mut slot) = self.running.lock() {
            *slot = Some(Running {
                child,
                stdin,
                job_dir: job_dir.to_path_buf(),
            });
        }
    }

    /// Closes the worker's stdin: it cancels itself, even before it has taken its lock.
    fn release(&self) {
        if let Some(r) = self.running.lock().ok().as_mut().and_then(|s| s.as_mut()) {
            r.stdin = None;
        }
    }
}

impl Drop for PipelineJobs {
    fn drop(&mut self) {
        let Some(mut r) = self.running.get_mut().ok().and_then(Option::take) else {
            return;
        };
        drop(r.stdin.take());
        let started = Instant::now();
        while matches!(r.child.try_wait(), Ok(None)) && started.elapsed() < GRACE {
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = r.child.kill();
        let _ = r.child.wait();
    }
}

fn schema_value<T: JsonSchema>() -> Value {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    let schema = settings.into_generator().into_root_schema_for::<T>();
    serde_json::to_value(schema).unwrap_or(Value::Null)
}

pub(super) fn definition() -> Value {
    let annotations = json!({
        "readOnlyHint": false, "destructiveHint": false, "idempotentHint": false,
        "title": "Décision atomique sur un patch", "openWorldHint": false,
    });
    json!({
        "name": TOOL_PIPELINE,
        "description": "Orchestre la décision sur un patch sans jamais modifier le modèle d'origine : candidats d'alpha appliqués à une copie de travail, piliers (safety, unblocking, gsm8k, kl, logits) mesurés sur leurs jeux Dev dans l'ordre de coût avec arrêt à la première porte non GO (non_inferiority, superiority, max_drift ; intervalles appariés de Newcombe, bootstrap sur les prompts), choix du candidat sur le Dev seulement, puis UNE consultation du Test Set (registre append-only ; agrégats seulement, aucun item de test exposé). action : plan (estimations disque, durée, puissance), start (lance le travailleur et renvoie job_id tout de suite ; un job à la fois ; reprise si le job existe), status (étape, avancement, temps restant, portes Dev), cancel, report (verdict GO / NO_GO / INCONCLUSIVE), promote_patch (écrit promoted/patch.json après un GO ; l'appliquer reste une décision séparée via surgeon_apply). La porte safety est obligatoire.",
        "inputSchema": schema_value::<PipelineToolArgs>(),
        "annotations": annotations,
    })
}

fn invalid(kind: &str, message: impl std::fmt::Display) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, message).with_kind(kind)
}

fn sandboxed(ctx: &ToolContext, argument: &str, path: &Path) -> Result<PathBuf, Outcome> {
    ctx.cfg
        .sandbox
        .resolve(argument, path, ctx.cfg.sandbox.base())
        .map_err(|e| invalid("path_not_allowed", e))
}

fn read_only(ctx: &ToolContext, action: &str) -> Result<(), Outcome> {
    match ctx.cfg.read_only {
        true => Err(invalid(
            "read_only",
            format!("action {action} refusée : serveur en --read-only"),
        )),
        false => Ok(()),
    }
}

/// The configuration, with every file it names checked against the sandbox.
fn checked_config(ctx: &ToolContext, path: &Path) -> Result<PathBuf, Outcome> {
    let path = sandboxed(ctx, "config", path)?;
    let config = run::load_config(&path, &RunOptions::default()).map_err(failure)?;
    for (field, file) in inputs(&config) {
        sandboxed(ctx, &format!("config.{field}"), &file)?;
    }
    Ok(path)
}

pub(super) fn command_name(arguments: &Value) -> &'static str {
    match arguments.get("action").and_then(Value::as_str) {
        Some("plan") => "pipeline plan",
        Some("start") => "pipeline start",
        Some("status") => "pipeline status",
        Some("cancel") => "pipeline cancel",
        Some("report") => "pipeline report",
        Some("promote_patch") => "pipeline promote",
        _ => "pipeline",
    }
}

pub(super) fn call(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: PipelineToolArgs = serde_json::from_value(arguments)
        .map_err(|e| invalid("invalid_arguments", format!("arguments invalides : {e}")))?;
    let job_dir = sandboxed(ctx, "job_dir", &args.job_dir)?;
    match args.action {
        PipelineAction::Plan => {
            let Some(config) = &args.config else {
                return Err(invalid("invalid_arguments", "config requis pour plan"));
            };
            let config = checked_config(ctx, config)?;
            let options = RunOptions::default();
            front::core_plan(
                &config,
                &job_dir,
                &options,
                true,
                !ctx.cfg.read_only,
                ctx.deps,
            )
        }
        PipelineAction::Start => {
            read_only(ctx, "start")?;
            start(ctx, args.config.as_deref(), &job_dir)
        }
        PipelineAction::Status => status(ctx, &job_dir),
        PipelineAction::Cancel => {
            read_only(ctx, "cancel")?;
            match ctx.pipeline.live().as_deref() == Some(job_dir.as_path()) {
                true => {
                    ctx.pipeline.release();
                    status(ctx, &job_dir)
                }
                false => front::core_cancel(&job_dir),
            }
        }
        PipelineAction::Report => front::core_report(&job_dir),
        PipelineAction::PromotePatch => {
            read_only(ctx, "promote_patch")?;
            front::core_promote(&job_dir, ctx.deps)
        }
    }
}

/// The job's status. The lock file only shows a worker some time after `start` returned:
/// the server's own live worker counts from the start.
fn status(ctx: &ToolContext, job_dir: &Path) -> CoreResult {
    let mut outcome = front::core_status(job_dir)?;
    if ctx.pipeline.live().as_deref() == Some(job_dir) {
        outcome.data["worker_running"] = json!(true);
        outcome.data["resumable"] = json!(false);
    }
    Ok(outcome)
}

fn start(ctx: &ToolContext, config: Option<&Path>, job_dir: &Path) -> CoreResult {
    if ctx.pipeline.busy() {
        return Err(failure(crate::pipeline::PipelineError::Busy(
            "un job tourne déjà sur ce serveur (un seul à la fois)".into(),
        )));
    }
    let job = Job::new(job_dir);
    if let Some(pid) = job::holder(&job) {
        return Err(failure(crate::pipeline::PipelineError::Busy(format!(
            "un travailleur (pid {pid}) exécute déjà ce job"
        ))));
    }
    let config = config.map(|c| checked_config(ctx, c)).transpose()?;
    let loaded = config
        .as_deref()
        .map(|c| run::load_config(c, &RunOptions::default()))
        .transpose()
        .map_err(failure)?;
    // Bound here so that a mismatch fails now, not in the worker's log.
    let file = run::bind(&job, loaded, RunOptions::default()).map_err(failure)?;
    if config.is_none() {
        for (field, path) in inputs(&file.config) {
            sandboxed(ctx, &format!("config.{field}"), &path)?;
        }
    }
    if file.options != RunOptions::default() {
        return Err(invalid(
            "invalid_arguments",
            "ce job a été lancé avec des options de la ligne de commande (--allow-*) : \
             reprenez-le en ligne de commande",
        ));
    }
    let mut cmd = front::worker_command(config.as_deref(), job_dir, &RunOptions::default(), true)
        .map_err(failure)?;
    let log = job.path("worker.log");
    let out = std::fs::File::create(&log).map_err(|e| failure(job::io_error(&log, e)))?;
    let err = out
        .try_clone()
        .map_err(|e| failure(job::io_error(&log, e)))?;
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(out)
        .stderr(err)
        .spawn()
        .map_err(|e| {
            failure(crate::pipeline::PipelineError::Io(format!(
                "travailleur : {e}"
            )))
        })?;
    let pid = child.id();
    let stdin = child.stdin.take();
    ctx.pipeline.hold(child, stdin, job_dir);
    Ok(Outcome::ok(
        json!({"job_id": file.job_id, "job_dir": job_dir, "pid": pid, "started": true}),
        String::new(),
    ))
}
