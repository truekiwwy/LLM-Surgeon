//! `surgeon pipeline` and the MCP tool `surgeon_pipeline`: one atomic, resumable decision
//! on a patch. A configuration (`config`) names the model, the patch template and its
//! candidates, the pillars (Dev and Test sets, metric, gate); the worker (`run`) measures
//! each candidate on the Dev sets in order of cost and stops at the first gate that is not
//! GO, selects on the Dev only, consults the Test Set once for the selected candidate, and
//! writes a report. The original model is never written: every patch goes to a working
//! copy in the job directory. See `docs/PIPELINE.md`.
//!
//! ```text
//! config (pipeline.json, schemars) -> plan (disk, time, power) -> run (worker, lock)
//!   static_checks (check, dry-run per candidate)
//!   per candidate: working copy -> apply -> pillars in order (checkpoints) -> gates (stats)
//!   selection (Dev) -> registry (test_access.json) -> Test (private/) -> results, report
//! status / cancel / report / promote read or write the job directory only
//! ```

#![forbid(unsafe_code)]

pub mod config;
pub mod front;
pub mod job;
pub mod measure;
pub mod plan;
pub mod run;
pub mod stats;

use std::io::Read;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::app::AppDeps;
use job::{read_json, read_json_opt, write_json, Job, State};
use run::{JobFile, Results};
use stats::Verdict;

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("configuration invalide : {}", .0.join(" ; "))]
    Config(Vec<String>),
    #[error("{0}")]
    Io(String),
    #[error("{0}")]
    Busy(String),
    #[error("{0}")]
    Bench(String),
    #[error("application du patch : {0}")]
    Apply(String),
    #[error("{0}")]
    TestSet(String),
    #[error("{0}")]
    Refused(String),
    #[error("annulé")]
    Cancelled,
}

/// Raises `interrupted` when `<job>/cancel` appears, or, with `watch_stdin`, when stdin
/// reaches its end (the MCP server that started this worker is gone). Stops with the
/// returned flag.
pub fn watch(job: &Job, interrupted: Arc<AtomicBool>, watch_stdin: bool) -> Arc<AtomicBool> {
    let stop = Arc::new(AtomicBool::new(false));
    {
        let (job, stop, interrupted) = (job.clone(), Arc::clone(&stop), Arc::clone(&interrupted));
        std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                if job.cancel_requested() {
                    interrupted.store(true, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        });
    }
    if watch_stdin {
        // Detached: it may stay blocked on a read until the process ends.
        std::thread::spawn(move || {
            let mut buf = [0u8; 256];
            let mut stdin = std::io::stdin();
            while matches!(stdin.read(&mut buf), Ok(n) if n > 0) {}
            interrupted.store(true, Ordering::SeqCst);
        });
    }
    stop
}

/// What `status` returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct StatusView {
    pub job_dir: PathBuf,
    /// null avant le premier lancement.
    pub state: Option<State>,
    /// Un travailleur tient le verrou du job.
    pub worker_running: bool,
    /// results.json est écrit : report peut être lu.
    pub report_available: bool,
    /// Le job s'est arrêté sans finir (arrêt brutal) : relancer start (ou run) le reprend.
    pub resumable: bool,
}

pub fn status(job: &Job) -> job::Result<StatusView> {
    let state: Option<State> = read_json_opt(&job.path("state.json"))?;
    let worker_running = job::holder(job).is_some();
    let report_available = job.path("results.json").exists();
    let resumable = !worker_running
        && !report_available
        && state
            .as_ref()
            .is_some_and(|s| !matches!(s.status, job::Status::Planned));
    Ok(StatusView {
        job_dir: job.dir.clone(),
        state,
        worker_running,
        report_available,
        resumable,
    })
}

/// Asks the worker to stop (it cleans its working copy); without a worker, removes what a
/// crash may have left there.
pub fn cancel(job: &Job) -> job::Result<StatusView> {
    if job::holder(job).is_some() {
        job::write_atomic(&job.path("cancel"), b"cancel\n")?;
    } else {
        let _ = std::fs::remove_dir_all(job.work());
    }
    status(job)
}

/// `<job>/report.json`: aggregates only (Test items stay in `private/`).
pub fn report(job: &Job) -> job::Result<Value> {
    read_json(&job.path("report.json"))
}

/// Writes `<job>/promoted/patch.json`, the selected candidate's patch, validated by a dry
/// run on the original model. Only after a GO; never applied: applying it stays a separate
/// decision (`surgeon apply`).
pub fn promote(job: &Job, deps: &AppDeps) -> job::Result<PathBuf> {
    let file: JobFile = read_json(&job.path("job.json"))?;
    let results: Results = run::results(job)?;
    let index = match (results.verdict, results.selected) {
        (Verdict::Go, Some(index)) => index,
        (verdict, _) => {
            return Err(PipelineError::Refused(format!(
                "verdict {verdict:?} : seul un GO confirmé sur le Test se promeut ({})",
                results.verdict_reason
            )))
        }
    };
    let config = &file.config;
    let value = config.candidates.values()[index];
    let mut patch = config.candidates.patch(&config.patch, value);
    patch.backup = true;
    patch.description = Some(format!(
        "pipeline {} : candidat {index} (valeur {value}), verdict GO",
        file.job_id
    ));
    let request = crate::app::ApplyRequest {
        model: config.model.clone(),
        patch: patch.clone(),
        patch_label: PathBuf::from("promoted/patch.json"),
        base_dir: config.model.clone(),
        dry_run: true,
        no_backup: false,
        backup_dir: None,
    };
    crate::app::core_apply(&request, deps, &mut crate::app::NullSink)
        .map_err(|o| PipelineError::Apply(o.error.map_or_else(String::new, |e| e.message)))?;
    let dir = job.path("promoted");
    std::fs::create_dir_all(&dir).map_err(|e| job::io_error(&dir, e))?;
    let path = dir.join("patch.json");
    write_json(&path, &patch)?;
    Ok(path)
}
