//! The job directory: every file the pipeline writes lives there (apart from the shared
//! Test Set registry), each one written whole (temporary file, then rename), so a crash
//! never leaves a partial result and a resume finds only complete checkpoints.
//!
//! ```text
//! <job>/config.json          resolved configuration (a job is bound to it)
//! <job>/plan.json            estimates written before the run
//! <job>/state.json           stage, progress, ETA, Dev gates so far (read by `status`)
//! <job>/lock                 pid of the running worker (one worker per job)
//! <job>/cancel               written by `cancel`; the worker stops at its next check
//! <job>/checkpoints/*.json   Dev measurements, one file per (model, pillar)
//! <job>/private/             Test measurements per item; never served by MCP
//! <job>/work/                working copy and dumps, removed at the end
//! <job>/results.json         deterministic results (identical after a resume)
//! <job>/report.json          results + manifest + timings
//! <job>/promoted/patch.json  written by `promote`, never applied
//! ```

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::stats::Verdict;
use super::PipelineError;

/// Marker that makes the MCP sandbox refuse every path under its directory.
pub const PRIVATE_MARKER: &str = ".surgeon-private";

pub type Result<T> = std::result::Result<T, PipelineError>;

pub fn io_error(path: &Path, e: impl std::fmt::Display) -> PipelineError {
    PipelineError::Io(format!("{} : {e}", path.display()))
}

/// The paths of one job directory.
#[derive(Debug, Clone)]
pub struct Job {
    pub dir: PathBuf,
}

impl Job {
    pub fn new(dir: &Path) -> Self {
        Self {
            dir: dir.to_path_buf(),
        }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.join(name)
    }

    pub fn checkpoint(&self, name: &str) -> PathBuf {
        self.dir.join("checkpoints").join(format!("{name}.json"))
    }

    pub fn private(&self, name: &str) -> PathBuf {
        self.dir.join("private").join(format!("{name}.json"))
    }

    pub fn work(&self) -> PathBuf {
        self.dir.join("work")
    }

    /// Creates the subdirectories, and the marker of `private/`.
    pub fn prepare(&self) -> Result<()> {
        for sub in ["checkpoints", "private", "work"] {
            let dir = self.dir.join(sub);
            fs::create_dir_all(&dir).map_err(|e| io_error(&dir, e))?;
        }
        let marker = self.dir.join("private").join(PRIVATE_MARKER);
        if !marker.exists() {
            write_atomic(&marker, b"test items: never served by MCP\n")?;
        }
        Ok(())
    }

    pub fn cancel_requested(&self) -> bool {
        self.path("cancel").exists()
    }
}

/// Writes `bytes` to `path` through a temporary file in the same directory, then renames:
/// readers see the old file or the new one, never a partial one.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    {
        let mut file = fs::File::create(&tmp).map_err(|e| io_error(&tmp, e))?;
        file.write_all(bytes).map_err(|e| io_error(&tmp, e))?;
        file.sync_all().map_err(|e| io_error(&tmp, e))?;
    }
    fs::rename(&tmp, path).map_err(|e| io_error(path, e))
}

/// Pretty JSON with a trailing newline: the same value always gives the same bytes.
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let mut text =
        serde_json::to_string_pretty(value).map_err(|e| PipelineError::Io(e.to_string()))?;
    text.push('\n');
    write_atomic(path, text.as_bytes())
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).map_err(|e| io_error(path, e))?;
    serde_json::from_str(&text).map_err(|e| io_error(path, e))
}

pub fn read_json_opt<T: DeserializeOwned>(path: &Path) -> Result<Option<T>> {
    match path.exists() {
        true => read_json(path).map(Some),
        false => Ok(None),
    }
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_file(path: &Path) -> Result<String> {
    crate::bench::sha256_file(path).map_err(|e| PipelineError::Io(e.to_string()))
}

/// SHA-256 of a model directory: of every regular file at its top level, names and
/// contents, in name order.
pub fn sha256_dir(dir: &Path) -> Result<String> {
    let mut hasher = Sha256::new();
    for (name, path) in files(dir)? {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(sha256_file(&path)?.as_bytes());
        hasher.update(b"\n");
    }
    Ok(hex(&hasher.finalize()))
}

/// Regular files at the top level of `dir`, by name.
pub fn files(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| io_error(dir, e))? {
        let entry = entry.map_err(|e| io_error(dir, e))?;
        if entry.file_type().map_err(|e| io_error(dir, e))?.is_file() {
            out.push((
                entry.file_name().to_string_lossy().into_owned(),
                entry.path(),
            ));
        }
    }
    out.sort();
    Ok(out)
}

pub fn dir_size(dir: &Path) -> Result<u64> {
    let mut total = 0;
    for (_, path) in files(dir)? {
        total += fs::metadata(&path).map_err(|e| io_error(&path, e))?.len();
    }
    Ok(total)
}

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

pub fn now_rfc3339() -> String {
    crate::surgery::diff::rfc3339_utc(now_secs())
}

// ------------------------------------------------------------------ lock

/// `<job>/lock` holding the worker's pid; removed on drop.
pub struct Lock {
    path: PathBuf,
}

impl Lock {
    /// Takes the lock, or reports the live worker that holds it. A lock left by a dead
    /// process (a crash) is taken over: that is a resume.
    pub fn acquire(job: &Job) -> Result<Self> {
        let path = job.path("lock");
        for _ in 0..2 {
            match fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
            {
                Ok(mut file) => {
                    let _ = write!(file, "{}", std::process::id());
                    let _ = file.sync_all();
                    return Ok(Self { path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if let Some(pid) = holder(job) {
                        return Err(PipelineError::Busy(format!(
                            "un travailleur (pid {pid}) exécute déjà ce job"
                        )));
                    }
                    let _ = fs::remove_file(&path);
                }
                Err(e) => return Err(io_error(&path, e)),
            }
        }
        Err(PipelineError::Busy("verrou du job inaccessible".into()))
    }
}

impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

/// The pid in `<job>/lock` when that process is still running.
pub fn holder(job: &Job) -> Option<u32> {
    let pid: u32 = fs::read_to_string(job.path("lock"))
        .ok()?
        .trim()
        .parse()
        .ok()?;
    let mut sys = sysinfo::System::new().ok()?;
    let target = sysinfo::Pid::from_u32(pid);
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[target]), true);
    sys.process(target).map(|_| pid)
}

// ------------------------------------------------------------------ state

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Planned,
    Running,
    Done,
    Cancelled,
    Failed,
    BudgetExceeded,
}

/// One Dev gate already decided.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DevGate {
    pub candidate: usize,
    pub pillar: String,
    pub verdict: Verdict,
}

/// `<job>/state.json`, what `status` returns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct State {
    pub job_id: String,
    pub status: Status,
    /// Étape en cours (static_checks, baseline_dev, candidate_dev, test, report).
    pub stage: String,
    /// Détail de l'étape (candidat, pilier).
    pub step: String,
    /// Unités de travail faites et prévues (tokens générés ou évalués, estimés).
    pub units_done: u64,
    pub units_total: u64,
    /// Secondes de travail cumulées sur toutes les reprises.
    pub elapsed_secs: u64,
    /// Estimation du temps restant ; null avant la première unité.
    pub eta_secs: Option<u64>,
    pub dev_gates: Vec<DevGate>,
    pub error: Option<String>,
    pub updated_at: String,
}

// ------------------------------------------------------------------ Test Set registry

/// One consultation of a Test Set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TestAccess {
    pub job_id: String,
    /// Dossier du job : une reprise est le même job dans le même dossier ; la même
    /// configuration dans un autre dossier est une seconde consultation.
    pub job_dir: PathBuf,
    /// SHA-256 du patch candidat évalué.
    pub candidate_sha256: String,
    /// SHA-256 des jeux Test consultés, dans l'ordre des piliers.
    pub datasets_sha256: Vec<String>,
    pub date: String,
    /// true quand l'accès réutilise un Test Set déjà consulté (--allow-test-reuse).
    pub reuse: bool,
}

pub fn read_registry(path: &Path) -> Result<Vec<TestAccess>> {
    Ok(read_json_opt(path)?.unwrap_or_default())
}

/// Appends `access` (the file is rewritten whole with the earlier entries first: nothing
/// is ever removed).
pub fn append_registry(path: &Path, access: TestAccess) -> Result<()> {
    let mut entries = read_registry(path)?;
    entries.push(access);
    write_json(path, &entries)
}
