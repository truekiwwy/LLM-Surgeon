//! The pipeline actions as command cores (`Outcome` and its exit codes), shared by the CLI
//! (`surgeon pipeline ...`) and the MCP tool `surgeon_pipeline`.

use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::json;

use super::job::{self, Job};
use super::run::{self, Ended, RunOptions};
use super::{plan, PipelineError};
use crate::app::{exit, AppDeps, CoreResult, Outcome};

/// The JSON spelling of a unit enum (GO, done...).
fn name<T: serde::Serialize>(value: &T) -> String {
    serde_json::to_value(value)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

pub fn failure(e: PipelineError) -> Outcome {
    let (code, kind) = match &e {
        PipelineError::Config(_) => (exit::INVALID_INPUT, "invalid_input"),
        PipelineError::Io(_) => (exit::GENERAL, "io"),
        PipelineError::Busy(_) => (exit::GENERAL, "busy"),
        PipelineError::Bench(_) => (exit::BENCH_FAILED, "bench_failed"),
        PipelineError::Apply(_) => (exit::EXECUTION_FAILED, "apply_failed"),
        PipelineError::TestSet(_) => (exit::GENERAL, "test_set_already_used"),
        PipelineError::Refused(_) => (exit::GENERAL, "not_promotable"),
        PipelineError::Cancelled => (exit::GENERAL, "cancelled"),
    };
    Outcome::fail(code, e.to_string()).with_kind(kind)
}

/// `plan`: estimates and power check; `plan.json` is written unless `write` is false
/// (MCP `--read-only`).
pub fn core_plan(
    config_path: &Path,
    job_dir: &Path,
    options: &RunOptions,
    calibrate: bool,
    write: bool,
    deps: &AppDeps,
) -> CoreResult {
    let config = run::load_config(config_path, options).map_err(failure)?;
    let canonical = serde_json::to_vec(&config).unwrap_or_default();
    let job_id = job::sha256_bytes(&canonical)[..16].to_string();
    let plan = plan::plan(&config, &job_id, deps, calibrate).map_err(failure)?;
    if write {
        std::fs::create_dir_all(job_dir).map_err(|e| failure(job::io_error(job_dir, e)))?;
        job::write_json(&Job::new(job_dir).path("plan.json"), &plan).map_err(failure)?;
    }
    let human = format!(
        "[PLAN]     job {} · {} candidat(s) · {} pilier(s) · disque {} Mio · durée estimée {}\n{}",
        plan.job_id,
        plan.candidates,
        plan.pillars.len(),
        plan.disk.total_bytes / (1 << 20),
        plan.estimated_secs
            .map_or_else(|| "non calibrée".into(), |s| format!("{s} s")),
        plan.warnings
            .iter()
            .map(|w| format!("  ! {w}\n"))
            .collect::<String>()
    );
    let mut outcome = Outcome::ok(json!(plan), human);
    outcome.warnings = plan.warnings.clone();
    Ok(outcome)
}

/// `run`: the worker, in this process. `watch_stdin` stops it when stdin closes (the MCP
/// server that started it is gone).
pub fn core_run(
    config_path: Option<&Path>,
    job_dir: &Path,
    options: RunOptions,
    watch_stdin: bool,
    deps: &AppDeps,
) -> CoreResult {
    let job = Job::new(job_dir);
    let config = config_path
        .map(|p| run::load_config(p, &options))
        .transpose()
        .map_err(failure)?;
    let file = run::bind(&job, config, options).map_err(failure)?;
    let stop = super::watch(&job, deps.interrupted.clone(), watch_stdin);
    let ended = run::run(&job, file, deps);
    stop.store(true, std::sync::atomic::Ordering::SeqCst);
    let (results, budget) = match ended.map_err(failure)? {
        Ended::Done(r) => (r, false),
        Ended::BudgetExceeded(r) => (r, true),
    };
    let human = format!(
        "[PIPELINE] {} · {}{}\n",
        name(&results.verdict),
        results.verdict_reason,
        if budget { " (budget épuisé)" } else { "" }
    );
    Ok(Outcome::ok(json!(results), human))
}

/// `run --detach` (CLI only): the worker in a background process, its output in
/// `<job>/worker.log`; returns at once.
pub fn core_detach(config_path: Option<&Path>, job_dir: &Path, options: &RunOptions) -> CoreResult {
    let job = Job::new(job_dir);
    if let Some(pid) = job::holder(&job) {
        return Err(failure(PipelineError::Busy(format!(
            "un travailleur (pid {pid}) exécute déjà ce job"
        ))));
    }
    std::fs::create_dir_all(job_dir).map_err(|e| failure(job::io_error(job_dir, e)))?;
    let mut child = worker_command(config_path, job_dir, options, false).map_err(failure)?;
    let log = job.path("worker.log");
    let file = std::fs::File::create(&log).map_err(|e| failure(job::io_error(&log, e)))?;
    let err = file
        .try_clone()
        .map_err(|e| failure(job::io_error(&log, e)))?;
    let spawned = child
        .stdin(Stdio::null())
        .stdout(file)
        .stderr(err)
        .spawn()
        .map_err(|e| failure(PipelineError::Io(format!("travailleur : {e}"))))?;
    Ok(Outcome::ok(
        json!({"job_dir": job_dir, "pid": spawned.id(), "log": log}),
        format!("[PIPELINE] travailleur détaché (pid {})\n", spawned.id()),
    ))
}

/// The worker command line: this executable, `pipeline run`, the structured options only.
pub fn worker_command(
    config_path: Option<&Path>,
    job_dir: &Path,
    options: &RunOptions,
    watch_stdin: bool,
) -> super::job::Result<Command> {
    let exe = std::env::current_exe()
        .map_err(|e| PipelineError::Io(format!("exécutable courant : {e}")))?;
    let mut cmd = Command::new(exe);
    cmd.args(["pipeline", "run", "--job-dir"]).arg(job_dir);
    if let Some(config) = config_path {
        cmd.arg("--config").arg(config);
    }
    if options.allow_disable_safety {
        cmd.arg("--allow-disable-safety");
    }
    if options.allow_test_reuse {
        cmd.arg("--allow-test-reuse");
    }
    if watch_stdin {
        cmd.arg("--watch-stdin");
    }
    Ok(cmd)
}

pub fn core_status(job_dir: &Path) -> CoreResult {
    let view = super::status(&Job::new(job_dir)).map_err(failure)?;
    let human = match &view.state {
        Some(s) => format!(
            "[STATUT]   {} · {} {} · {}/{} unités · reste {}\n",
            name(&s.status),
            s.stage,
            s.step,
            s.units_done,
            s.units_total,
            s.eta_secs.map_or_else(|| "?".into(), |e| format!("{e} s"))
        ),
        None => "[STATUT]   aucun lancement\n".into(),
    };
    Ok(Outcome::ok(json!(view), human))
}

pub fn core_cancel(job_dir: &Path) -> CoreResult {
    let view = super::cancel(&Job::new(job_dir)).map_err(failure)?;
    Ok(Outcome::ok(
        json!(view),
        "[PIPELINE] annulation demandée\n".into(),
    ))
}

pub fn core_report(job_dir: &Path) -> CoreResult {
    let report = super::report(&Job::new(job_dir)).map_err(failure)?;
    let human = format!(
        "[PIPELINE] {} · {}\n",
        report["results"]["verdict"].as_str().unwrap_or("?"),
        report["results"]["verdict_reason"].as_str().unwrap_or("")
    );
    Ok(Outcome::ok(report, human))
}

pub fn core_promote(job_dir: &Path, deps: &AppDeps) -> CoreResult {
    let path = super::promote(&Job::new(job_dir), deps).map_err(failure)?;
    Ok(Outcome::ok(
        json!({"patch": path, "applied": false}),
        format!(
            "[PIPELINE] patch promu : {} (non appliqué ; surgeon apply le fera sur décision)\n",
            path.display()
        ),
    ))
}
