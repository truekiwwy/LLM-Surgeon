//! The job manager of a project: a priority queue, at most one heavy job at a time, budgets,
//! cooperative cancellation, and recovery after a crash. A job is one engine tool call made
//! in process (`exec::EngineExecutor`), or a working copy. Each run lives in
//! `runs/<id>/`: `params.json` (what was asked), `status.json`, `events.ndjson`, and `out/`
//! (renamed from `out.partial/` only when the run succeeds: no partial output survives).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use weight_surgeon::events::SurgeonEvent;

use super::bus::{self, Feed, Recorder, RunView, EVENTS_FILE};
use super::ledger::{self, Entry, Hashes, Verdict, Versions};
use super::project::{now_unix, write_atomic, Project, ProjectState, SourceModel, Transition};

/// The working-copy job (not an engine tool).
pub const COPY_TOOL: &str = "studio_copy";
/// Replaced in every string argument by the run's output folder.
pub const OUT: &str = "{out}";
const PARAMS: &str = "params.json";
const STATUS: &str = "status.json";
const LOCK: &str = "project.lock";

/// Tools that rewrite their `model` in place: only a working copy may be given.
const IN_PLACE: [&str; 3] = ["surgeon_apply", "surgeon_lora", "surgeon_restore"];
/// Arguments a tool writes to: never inside the source model.
const WRITE_KEYS: [&str; 7] = [
    "out",
    "out_dir",
    "backup_dir",
    "export_gguf",
    "judge_cache",
    "cache_dir",
    "job_dir",
];

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobSpec {
    pub tool: String,
    pub label: String,
    pub args: Value,
    #[serde(default)]
    pub priority: i32,
    #[serde(default)]
    pub budgets: Budgets,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumed_from: Option<String>,
    /// The engine runs it as `surgeon mcp --read-only` (an agent in "Ask" mode).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub read_only: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Budgets {
    /// Cancelled (and the project flagged) once it runs longer.
    #[serde(default)]
    pub max_wall_time_secs: Option<u64>,
    /// `surgeon_pipeline`: refuses a pipeline.json whose `budgets.max_candidates` is higher.
    #[serde(default)]
    pub max_candidates: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Queued,
    Running,
    Finished,
    Failed,
    Cancelled,
    Interrupted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StatusFile {
    pub status: RunStatus,
    #[serde(default)]
    pub started_unix: Option<u64>,
    #[serde(default)]
    pub finished_unix: Option<u64>,
    #[serde(default)]
    pub exit_code: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// The run that resumed this interrupted one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumed_by: Option<String>,
}

impl StatusFile {
    fn new(status: RunStatus) -> Self {
        Self {
            status,
            started_unix: None,
            finished_unix: None,
            exit_code: None,
            note: None,
            resumed_by: None,
        }
    }
}

/// Analysis tools that read a little and finish fast, and previews (estimate, dry run) that
/// write no model; everything else takes the heavy slot. A pipeline `plan` stays heavy: it
/// may calibrate by running the model.
pub fn is_heavy(spec: &JobSpec) -> bool {
    let light_tool = matches!(
        spec.tool.as_str(),
        "surgeon_check" | "surgeon_inspect" | "surgeon_backends" | "surgeon_files"
    );
    let light_preview =
        spec.tool != "surgeon_pipeline" && super::estimate::is_preview(&spec.tool, &spec.args);
    !(light_tool || light_preview)
}

/// What a job returned: the engine's JSON envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub exit_code: i32,
    pub envelope: Value,
    pub cancelled: bool,
}

impl Outcome {
    pub fn failed(message: impl Into<String>) -> Self {
        let message = message.into();
        Self {
            exit_code: 1,
            envelope: serde_json::json!({ "ok": false, "exit_code": 1,
                "error": { "message": message, "kind": "studio" } }),
            cancelled: false,
        }
    }

    fn terminal(&self) -> SurgeonEvent {
        match (self.cancelled, self.exit_code) {
            (true, _) => SurgeonEvent::Cancelled,
            (false, 0) => SurgeonEvent::Finished { exit_code: 0 },
            (false, code) => SurgeonEvent::Failed {
                kind: self.envelope["error"]["kind"]
                    .as_str()
                    .unwrap_or("error")
                    .into(),
                message: self.envelope["error"]["message"]
                    .as_str()
                    .unwrap_or_default()
                    .into(),
                exit_code: code,
            },
        }
    }
}

pub struct RunContext {
    pub project_dir: PathBuf,
    pub copies_dir: PathBuf,
    pub source: Option<SourceModel>,
    pub cancel: Arc<AtomicBool>,
}

/// Runs one job on the worker thread; progress goes to `rec`. The manager writes the first
/// and last event lines.
pub trait Executor: Send + Sync {
    fn run(&self, spec: &JobSpec, args: &Value, ctx: &RunContext, rec: &mut Recorder) -> Outcome;
}

pub struct Queued {
    pub id: String,
    pub spec: JobSpec,
    seq: u64,
}

pub struct Running {
    pub id: String,
    pub spec: JobSpec,
    pub view: RunView,
    pub started: Instant,
    started_unix: u64,
    feed: Feed,
    cancel: Arc<AtomicBool>,
    /// Why the manager cancelled it (a budget), if it did.
    pub stopped_by: Option<String>,
    done: Receiver<(Outcome, Hashes)>,
}

impl Running {
    pub fn cancelling(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

/// A run that just ended.
#[derive(Debug, Clone)]
pub struct Ended {
    pub id: String,
    pub spec: JobSpec,
    pub verdict: Verdict,
    pub view: RunView,
    pub envelope: Value,
}

/// A run found on disk.
#[derive(Debug, Clone)]
pub struct RunRecord {
    pub id: String,
    pub spec: JobSpec,
    pub status: StatusFile,
}

pub struct Manager {
    pub project: Project,
    exec: Arc<dyn Executor>,
    wake: Arc<dyn Fn() + Send + Sync>,
    pub queue: Vec<Queued>,
    pub running: Vec<Running>,
    seq: u64,
    /// The worst end since the project last became `Running`.
    worst: Option<Transition>,
    /// Shared with every job thread: the lock outlives a manager dropped mid-job.
    lock: Arc<ProjectLock>,
}

impl Manager {
    /// Opens a project for jobs. Runs left `queued` or `running` by a studio that died are
    /// marked `interrupted` (and logged), and the project becomes `Interrupted`.
    pub fn open(
        dir: &Path,
        exec: Arc<dyn Executor>,
        wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<Self, String> {
        let lock = ProjectLock::acquire(dir)?;
        let mut project = Project::open(dir)?;
        let mut crashed = project.manifest.state == ProjectState::Running;
        for run in history(&project) {
            if !matches!(run.status.status, RunStatus::Queued | RunStatus::Running) {
                continue;
            }
            crashed = true;
            let mut status = run.status.clone();
            status.status = RunStatus::Interrupted;
            status.note = Some("le studio s'est arrêté pendant ce run".into());
            write_status(&project.runs_dir().join(&run.id), &status)?;
            let _ = std::fs::remove_dir_all(project.runs_dir().join(&run.id).join("out.partial"));
            ledger::append(
                &project.ledger_path(),
                &entry(
                    &run.id,
                    &run.spec,
                    &status,
                    Verdict::Interrupted,
                    Hashes::default(),
                    Value::Null,
                ),
            )?;
        }
        if crashed {
            project.manifest.state = ProjectState::Running;
            project.transition(Transition::Crash)?;
        }
        Ok(Self {
            project,
            exec,
            wake,
            queue: Vec::new(),
            running: Vec::new(),
            seq: 0,
            worst: None,
            lock: Arc::new(lock),
        })
    }

    /// Every run of the project, oldest first.
    pub fn history(&self) -> Vec<RunRecord> {
        history(&self.project)
    }

    /// Interrupted runs nobody resumed yet.
    pub fn resumable(&self) -> Vec<RunRecord> {
        self.history()
            .into_iter()
            .filter(|r| r.status.status == RunStatus::Interrupted && r.status.resumed_by.is_none())
            .collect()
    }

    pub fn submit(&mut self, spec: JobSpec) -> Result<String, String> {
        self.check(&spec)?;
        let id = new_run_id(&self.project.runs_dir())?;
        let dir = self.project.runs_dir().join(&id);
        let params = serde_json::to_vec_pretty(&spec).map_err(|e| e.to_string())?;
        write_atomic(&dir.join(PARAMS), &params)?;
        write_status(&dir, &StatusFile::new(RunStatus::Queued))?;
        self.seq += 1;
        self.queue.push(Queued {
            id: id.clone(),
            spec,
            seq: self.seq,
        });
        Ok(id)
    }

    /// Submits an interrupted run again, as a new run that points back to it.
    pub fn resume(&mut self, id: &str) -> Result<String, String> {
        let dir = self.project.runs_dir().join(id);
        let mut status = read_status(&dir)?;
        if status.status != RunStatus::Interrupted || status.resumed_by.is_some() {
            return Err(format!("{id} n'est pas un run interrompu à reprendre"));
        }
        let mut spec = read_params(&dir)?;
        spec.resumed_from = Some(id.into());
        let new = self.submit(spec)?;
        status.resumed_by = Some(new.clone());
        write_status(&dir, &status)?;
        Ok(new)
    }

    /// Dismisses an interruption or a failure: the project is `Ready` again.
    pub fn acknowledge(&mut self) -> Result<(), String> {
        self.project.transition(Transition::Acknowledge)
    }

    /// A queued run is dropped at once; a running one is asked to stop (the engine checks
    /// the flag between tensors, prompts and chunks, and rolls back).
    pub fn cancel(&mut self, id: &str) -> Result<(), String> {
        if let Some(i) = self.queue.iter().position(|q| q.id == id) {
            let q = self.queue.remove(i);
            let mut status = StatusFile::new(RunStatus::Cancelled);
            status.finished_unix = Some(now_unix());
            write_status(&self.project.runs_dir().join(id), &status)?;
            let e = entry(
                id,
                &q.spec,
                &status,
                Verdict::Cancelled,
                Hashes::default(),
                Value::Null,
            );
            return ledger::append(&self.project.ledger_path(), &e);
        }
        match self.running.iter().find(|r| r.id == id) {
            Some(r) => {
                r.cancel.store(true, Ordering::SeqCst);
                Ok(())
            }
            None => Err(format!("{id} : ni en file ni en cours")),
        }
    }

    pub fn busy(&self) -> bool {
        !self.running.is_empty() || !self.queue.is_empty()
    }

    /// Ticks until nothing is queued or running, without a window (self-test, tests).
    pub fn wait(&mut self, timeout: Duration) -> Result<Vec<Ended>, String> {
        let deadline = Instant::now() + timeout;
        let mut ended = Vec::new();
        while self.busy() {
            if Instant::now() > deadline {
                return Err(format!(
                    "jobs encore en cours après {} s",
                    timeout.as_secs()
                ));
            }
            ended.extend(self.tick(Instant::now()));
            std::thread::sleep(Duration::from_millis(2));
        }
        ended.extend(self.tick(Instant::now()));
        Ok(ended)
    }

    /// Folds the event streams (at most 60 Hz), enforces the budgets, closes finished runs
    /// and starts what the queue allows. Call it every frame.
    pub fn tick(&mut self, now: Instant) -> Vec<Ended> {
        let mut ended = Vec::new();
        let mut i = 0;
        while i < self.running.len() {
            let run = &mut self.running[i];
            run.feed.drain(now, &mut run.view);
            if let Some(max) = run.spec.budgets.max_wall_time_secs {
                if run.stopped_by.is_none()
                    && now.duration_since(run.started) > Duration::from_secs(max)
                {
                    run.stopped_by = Some(format!("budget max_wall_time ({max} s) dépassé"));
                    run.cancel.store(true, Ordering::SeqCst);
                }
            }
            let result = match run.done.try_recv() {
                Ok(result) => Some(result),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some((
                    Outcome::failed("le job s'est arrêté sans résultat (panique)"),
                    Hashes::default(),
                )),
            };
            match result {
                Some((outcome, hashes)) => {
                    let run = self.running.remove(i);
                    ended.push(self.close(run, outcome, hashes));
                }
                None => i += 1,
            }
        }
        if self.running.is_empty() {
            if let Some(t) = self.worst.take() {
                let _ = self.project.transition(t);
            }
        }
        while let Some(next) = self.next_startable() {
            let q = self.queue.remove(next);
            self.start(q);
        }
        ended
    }

    fn next_startable(&self) -> Option<usize> {
        let heavy_running = self.running.iter().any(|r| is_heavy(&r.spec));
        self.queue
            .iter()
            .enumerate()
            .filter(|(_, q)| !(heavy_running && is_heavy(&q.spec)))
            .min_by_key(|(_, q)| (-q.spec.priority, q.seq))
            .map(|(i, _)| i)
    }

    /// Why a job may not be submitted, if it may not: unknown tool, an in-place tool on
    /// anything but a working copy, an output inside the source model, a pipeline budget.
    pub fn check(&self, spec: &JobSpec) -> Result<(), String> {
        if spec.tool != COPY_TOOL
            && !weight_surgeon::mcp::tool_definitions()
                .as_array()
                .is_some_and(|tools| tools.iter().any(|t| t["name"] == spec.tool.as_str()))
        {
            return Err(format!("outil inconnu : {}", spec.tool));
        }
        let args = &spec.args;
        let source = self
            .project
            .manifest
            .source
            .as_ref()
            .map(|s| s.path.clone());
        let resolve = |v: &Value| v.as_str().map(|s| absolute(&self.project.dir, s));
        let in_place = IN_PLACE.contains(&spec.tool.as_str())
            && args["dry_run"] != true
            && (spec.tool != "surgeon_lora" || args["mode"] == "merge");
        if in_place {
            let copies = canonical(&self.project.copies_dir());
            let model = resolve(&args["model"]).map(|m| canonical(&m));
            if !model.is_some_and(|m| m.starts_with(&copies) && m != copies) {
                return Err(format!(
                    "{} écrit le modèle en place : choisissez une copie de travail (copies/), \
                     jamais le modèle d'origine",
                    spec.tool
                ));
            }
        }
        if let Some(source) = &source {
            for key in WRITE_KEYS {
                if let Some(path) = resolve(&args[key]) {
                    if canonical(&path).starts_with(source) {
                        return Err(format!("{key} : écriture dans le modèle d'origine refusée"));
                    }
                }
            }
        }
        if let (Some(max), "surgeon_pipeline") = (spec.budgets.max_candidates, spec.tool.as_str()) {
            if let Some(config) = resolve(&args["config"]) {
                let text = std::fs::read_to_string(&config)
                    .map_err(|e| format!("{} : {e}", config.display()))?;
                let config: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
                match config["budgets"]["max_candidates"].as_u64() {
                    Some(n) if n <= u64::from(max) => {}
                    n => {
                        return Err(format!(
                            "pipeline.json budgets.max_candidates = {n:?} : le budget du job \
                             est {max}"
                        ))
                    }
                }
            }
        }
        Ok(())
    }

    fn start(&mut self, q: Queued) {
        let dir = self.project.runs_dir().join(&q.id);
        let out_partial = dir.join("out.partial");
        let started_unix = now_unix();
        let prepared = std::fs::create_dir_all(&out_partial)
            .map_err(|e| e.to_string())
            .and_then(|()| bus::channel(&dir.join(EVENTS_FILE)));
        let (mut rec, feed) = match prepared {
            Ok(pair) => pair,
            Err(e) => {
                let mut status = StatusFile::new(RunStatus::Failed);
                status.note = Some(e.clone());
                let _ = write_status(&dir, &status);
                return;
            }
        };
        let mut status = StatusFile::new(RunStatus::Running);
        status.started_unix = Some(started_unix);
        let _ = write_status(&dir, &status);
        if self.project.manifest.state != ProjectState::Running {
            let _ = self.project.transition(Transition::Start);
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let ctx = RunContext {
            project_dir: self.project.dir.clone(),
            copies_dir: self.project.copies_dir(),
            source: self.project.manifest.source.clone(),
            cancel: Arc::clone(&cancel),
        };
        let args = resolve_out(&q.spec.args, &out_partial);
        let (tx, done) = mpsc::channel();
        let exec = Arc::clone(&self.exec);
        let wake = Arc::clone(&self.wake);
        let spec = q.spec.clone();
        // The thread is detached: it keeps the project locked until it stops writing there.
        let lock = Arc::clone(&self.lock);
        std::thread::spawn(move || {
            let _lock = lock;
            let source = ctx
                .source
                .as_ref()
                .map(|s| (s.path.as_path(), s.sha256.as_str()));
            let hashes = ledger::hash_inputs(&args, &ctx.project_dir, source);
            rec.emit(SurgeonEvent::JobStarted {
                command: spec.tool.clone(),
            });
            let outcome = exec.run(&spec, &args, &ctx, &mut rec);
            rec.emit(outcome.terminal());
            if let Some(e) = rec.take_error() {
                eprintln!("surgeon-gui : events.ndjson incomplet : {e}");
            }
            let _ = tx.send((outcome, hashes));
            wake();
        });
        self.running.push(Running {
            id: q.id,
            spec: q.spec,
            view: RunView::default(),
            started: Instant::now(),
            started_unix,
            feed,
            cancel,
            stopped_by: None,
            done,
        });
    }

    fn close(&mut self, run: Running, mut outcome: Outcome, hashes: Hashes) -> Ended {
        let dir = self.project.runs_dir().join(&run.id);
        let partial = dir.join("out.partial");
        // The file is the truth; the live view may have skipped progress lines.
        let view = RunView::replay(&run.feed.file).unwrap_or(run.view);
        let cancelled = outcome.cancelled || run.cancel.load(Ordering::SeqCst);
        let verdict = match (cancelled, outcome.exit_code) {
            (true, _) => Verdict::Cancelled,
            (false, 0) => Verdict::Ok,
            _ => Verdict::Failed,
        };
        if verdict == Verdict::Ok {
            let out = dir.join("out");
            if let Err(e) = std::fs::rename(&partial, &out) {
                outcome = Outcome::failed(format!("{} : {e}", out.display()));
            } else {
                outcome.envelope = repoint(&outcome.envelope, &partial, &out);
            }
        } else {
            let _ = std::fs::remove_dir_all(&partial);
        }
        let verdict = if outcome.exit_code != 0 && verdict == Verdict::Ok {
            Verdict::Failed
        } else {
            verdict
        };
        let mut status = StatusFile::new(match verdict {
            Verdict::Ok => RunStatus::Finished,
            Verdict::Cancelled => RunStatus::Cancelled,
            _ => RunStatus::Failed,
        });
        status.started_unix = Some(run.started_unix);
        status.finished_unix = Some(now_unix());
        status.exit_code = Some(outcome.exit_code);
        status.note = run.stopped_by.clone();
        let _ = write_status(&dir, &status);
        let data = outcome.envelope["data"].clone();
        let mut e = entry(&run.id, &run.spec, &status, verdict, hashes, data);
        e.error = outcome.envelope["error"]["message"]
            .as_str()
            .map(String::from)
            .or(run.stopped_by.clone());
        if let Err(err) = ledger::append(&self.project.ledger_path(), &e) {
            eprintln!("surgeon-gui : {err}");
        }
        let t = match (verdict, &run.stopped_by) {
            (Verdict::Ok, _) => Transition::Succeed,
            (Verdict::Cancelled, None) => Transition::Cancel,
            _ => Transition::Fail,
        };
        let rank = |t: &Transition| match t {
            Transition::Fail => 2,
            Transition::Cancel => 1,
            _ => 0,
        };
        if self.worst.as_ref().is_none_or(|w| rank(&t) > rank(w)) {
            self.worst = Some(t);
        }
        Ended {
            id: run.id,
            spec: run.spec,
            verdict,
            view,
            envelope: outcome.envelope,
        }
    }
}

fn entry(
    id: &str,
    spec: &JobSpec,
    status: &StatusFile,
    verdict: Verdict,
    hashes: Hashes,
    result: Value,
) -> Entry {
    let finished = status.finished_unix.unwrap_or_else(now_unix);
    Entry {
        run_id: id.into(),
        tool: spec.tool.clone(),
        label: spec.label.clone(),
        params: spec.args.clone(),
        resumed_from: spec.resumed_from.clone(),
        started_unix: status.started_unix.unwrap_or(finished),
        finished_unix: finished,
        hashes,
        versions: Versions::default(),
        verdict,
        exit_code: status.exit_code.unwrap_or(-1),
        result,
        error: status.note.clone(),
    }
}

fn history(project: &Project) -> Vec<RunRecord> {
    let Ok(entries) = std::fs::read_dir(project.runs_dir()) else {
        return Vec::new();
    };
    let mut runs: Vec<RunRecord> = entries
        .flatten()
        .filter_map(|e| {
            let dir = e.path();
            Some(RunRecord {
                id: e.file_name().to_str()?.to_owned(),
                spec: read_params(&dir).ok()?,
                status: read_status(&dir).ok()?,
            })
        })
        .collect();
    runs.sort_by(|a, b| a.id.cmp(&b.id));
    runs
}

fn read_params(dir: &Path) -> Result<JobSpec, String> {
    let text = std::fs::read_to_string(dir.join(PARAMS)).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| format!("{} : {e}", dir.display()))
}

fn read_status(dir: &Path) -> Result<StatusFile, String> {
    let text = std::fs::read_to_string(dir.join(STATUS)).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| format!("{} : {e}", dir.display()))
}

fn write_status(dir: &Path, status: &StatusFile) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(status).map_err(|e| e.to_string())?;
    write_atomic(&dir.join(STATUS), &bytes)
}

/// `r<unix seconds>-<n>`: sorts in start order, created atomically.
fn new_run_id(runs: &Path) -> Result<String, String> {
    let stamp = now_unix();
    for n in 1..10_000 {
        let id = format!("r{stamp}-{n:03}");
        match std::fs::create_dir(runs.join(&id)) {
            Ok(()) => return Ok(id),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(format!("{} : {e}", runs.display())),
        }
    }
    Err("trop de runs dans la même seconde".into())
}

fn absolute(base: &Path, path: &str) -> PathBuf {
    let p = Path::new(path);
    match p.is_absolute() {
        true => p.to_path_buf(),
        false => base.join(p),
    }
}

/// The canonical path, or the canonical parent joined with the name when it does not exist
/// yet (an output about to be written).
fn canonical(path: &Path) -> PathBuf {
    if let Ok(c) = std::fs::canonicalize(path) {
        return c;
    }
    match (path.parent(), path.file_name()) {
        (Some(parent), Some(name)) if !parent.as_os_str().is_empty() => {
            canonical(parent).join(name)
        }
        _ => path.to_path_buf(),
    }
}

fn resolve_out(args: &Value, out: &Path) -> Value {
    let out = out.display().to_string();
    map_strings(args, &|s| s.replace(OUT, &out))
}

/// The envelope with the paths under `from` moved under `to`.
fn repoint(envelope: &Value, from: &Path, to: &Path) -> Value {
    let (from, to) = (from.display().to_string(), to.display().to_string());
    map_strings(envelope, &|s| s.replace(&from, &to))
}

fn map_strings(value: &Value, f: &dyn Fn(&str) -> String) -> Value {
    match value {
        Value::String(s) => Value::String(f(s)),
        Value::Array(items) => Value::Array(items.iter().map(|v| map_strings(v, f)).collect()),
        Value::Object(map) => Value::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), map_strings(v, f)))
                .collect(),
        ),
        other => other.clone(),
    }
}

/// `project.lock` holds the pid of the studio that has the project open. A lock whose
/// process is gone (or is this one) is stale and taken over.
struct ProjectLock(PathBuf);

impl ProjectLock {
    fn acquire(dir: &Path) -> Result<Self, String> {
        let path = dir.join(LOCK);
        let own = std::process::id();
        if let Some(pid) = std::fs::read_to_string(&path)
            .ok()
            .and_then(|t| t.trim().parse::<u32>().ok())
        {
            if pid != own && process_alive(pid) {
                return Err(format!(
                    "{} est déjà ouvert par le processus {pid}",
                    dir.display()
                ));
            }
        }
        write_atomic(&path, own.to_string().as_bytes())?;
        Ok(Self(path))
    }
}

impl Drop for ProjectLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn process_alive(pid: u32) -> bool {
    use sysinfo::{Pid, ProcessesToUpdate, System};
    let pid = Pid::from_u32(pid);
    // Unknown (no process table): assume alive, the user can delete a stale lock.
    let Ok(mut system) = System::new() else {
        return true;
    };
    system.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
    system.process(pid).is_some()
}

#[cfg(test)]
mod tests;
