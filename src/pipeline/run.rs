//! The worker: static checks, then per candidate (in list order) a working copy, the patch,
//! and the pillars in configuration order, stopping at the first gate that is not GO; the
//! selection on the Dev results only; one consultation of the Test Set by the selected
//! candidate; the report. Every measurement is a checkpoint: a resumed worker reuses them
//! and, the decoding being greedy, writes the same `results.json`, byte for byte.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::config::{self, PillarConfig, PillarKind, PipelineConfig, WorkingCopy};
use super::job::{
    self, append_registry, io_error, read_json, read_json_opt, read_registry, sha256_bytes,
    sha256_dir, sha256_file, write_json, DevGate, Job, Lock, Result, State, Status, TestAccess,
};
use super::measure::{self, Measurement, Outcomes};
use super::stats::{binary_gate, drift_gate, GateResult, PairedCounts, Verdict};
use super::PipelineError;
use crate::app::{core_apply, core_check, AppDeps, ApplyRequest, NullSink};
use crate::backend::BackendConfig;
use crate::bench::score::Backend;
use crate::cli::CheckArgs;
use crate::patch::PatchFile;

/// Options of one run, kept in `<job>/job.json` so that a resume keeps them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunOptions {
    /// CLI only: a configuration without a safety pillar is accepted, and the report says so.
    pub allow_disable_safety: bool,
    /// CLI only: a Test Set already consulted may be consulted again; the report is then
    /// marked as contaminated.
    pub allow_test_reuse: bool,
}

/// `<job>/job.json`: the resolved configuration the job is bound to, and its options.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobFile {
    pub job_id: String,
    pub config_sha256: String,
    pub config: PipelineConfig,
    pub options: RunOptions,
}

/// The configuration's backend: flags over `pipeline.json` over the user file, checked, and
/// stored in the config so that the job records the engine it was measured with.
pub fn resolve_backend(config: &mut PipelineConfig, flags: BackendConfig) -> Result<()> {
    let user = crate::backend::load_user_config().map_err(|e| PipelineError::Config(vec![e]))?;
    let top = flags.over(config.backend.take().unwrap_or_default());
    let effective = crate::backend::resolve(top, &user);
    match effective.backend {
        Some(Backend::Llama | Backend::LlamaCompletion) => {
            return Err(PipelineError::Config(vec![
                "backend : native, mistralrs, llama-server ou hipfire".into(),
            ]))
        }
        // Before any work: an engine that cannot be found stops the job now.
        Some(Backend::Hipfire) => {
            crate::backend::hipfire::port(&effective)
                .map_err(|e| PipelineError::Config(vec![e]))?;
        }
        Some(Backend::MistralRs) if crate::backend::mistralrs::locate(&effective).is_none() => {
            return Err(PipelineError::Config(vec![
                crate::backend::mistralrs::not_found(&effective),
            ]))
        }
        _ => {}
    }
    config.backend = effective.for_job();
    Ok(())
}

/// Reads, resolves and validates a configuration file.
pub fn load_config(path: &Path, options: &RunOptions) -> Result<PipelineConfig> {
    let text = fs::read_to_string(path).map_err(|e| io_error(path, e))?;
    let mut config: PipelineConfig = serde_json::from_str(text.trim_start_matches('\u{feff}'))
        .map_err(|e| PipelineError::Config(vec![format!("{} : {e}", path.display())]))?;
    // Absolute: the patch is applied with the working copy as base directory.
    let base = std::path::absolute(path)
        .map_err(|e| io_error(path, e))?
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_default();
    config::resolve(&mut config, &base);
    config::validate(&config, options.allow_disable_safety).map_err(PipelineError::Config)?;
    Ok(config)
}

/// The job bound to `config`: written on first use, checked against it afterwards.
pub fn bind(job: &Job, config: Option<PipelineConfig>, options: RunOptions) -> Result<JobFile> {
    let path = job.path("job.json");
    let existing: Option<JobFile> = read_json_opt(&path)?;
    match (existing, config) {
        (Some(file), None) => Ok(file),
        (Some(file), Some(config)) => {
            if file.config != config {
                return Err(PipelineError::Config(vec![format!(
                    "{} est lié à une autre configuration (job {})",
                    job.dir.display(),
                    file.job_id
                )]));
            }
            if file.options != options {
                return Err(PipelineError::Config(vec![
                    "options différentes de celles du job (--allow-*) : reprenez avec les mêmes"
                        .into(),
                ]));
            }
            Ok(file)
        }
        (None, Some(config)) => {
            fs::create_dir_all(&job.dir).map_err(|e| io_error(&job.dir, e))?;
            let canonical = serde_json::to_vec(&config).unwrap_or_default();
            let config_sha256 = sha256_bytes(&canonical);
            let file = JobFile {
                job_id: config_sha256[..16].to_string(),
                config_sha256,
                config,
                options,
            };
            write_json(&path, &file)?;
            Ok(file)
        }
        (None, None) => Err(PipelineError::Config(vec![format!(
            "{} : aucun job (job.json absent) ; donnez --config",
            job.dir.display()
        )])),
    }
}

// ------------------------------------------------------------------ results

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelCheck {
    pub path: PathBuf,
    /// SHA-256 du dossier du modèle d'origine (noms et contenus des fichiers) avant le job.
    pub sha256_before: String,
    /// Le même après le job.
    pub sha256_after: String,
    pub unchanged: bool,
}

/// One pillar of one candidate on one split: aggregates only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PillarResult {
    pub pillar: String,
    pub kind: PillarKind,
    pub metric: String,
    /// dev ou test.
    pub split: String,
    pub n: usize,
    /// Taux de succès avant et après (piliers binaires) ; null pour une dérive.
    pub rate_before: Option<f64>,
    pub rate_after: Option<f64>,
    pub counts: Option<PairedCounts>,
    /// Générations coupées par max_new_tokens avant et après (gsm8k) ; null ailleurs.
    #[serde(default)]
    pub truncated_before: Option<usize>,
    #[serde(default)]
    pub truncated_after: Option<usize>,
    /// Générations coupées par le détecteur de boucles (gsm8k, comptées fausses).
    #[serde(default)]
    pub loops_before: Option<usize>,
    #[serde(default)]
    pub loops_after: Option<usize>,
    /// Sorties sans réponse extractible (gsm8k, comptées fausses).
    #[serde(default)]
    pub no_answer_before: Option<usize>,
    #[serde(default)]
    pub no_answer_after: Option<usize>,
    pub gate: GateResult,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CandidateResult {
    pub index: usize,
    /// alpha (candidates.alphas) ou facteur (candidates.scales).
    pub value: f32,
    pub patch_sha256: String,
    /// Échec des contrôles statiques (dry-run du patch sur le modèle d'origine) ; null sinon.
    pub static_error: Option<String>,
    /// Piliers Dev évalués, dans l'ordre, jusqu'à la première porte non GO.
    pub dev: Vec<PillarResult>,
    /// Pilier où l'évaluation s'est arrêtée ; null si toutes les portes Dev sont GO.
    pub stopped_at: Option<String>,
    pub passed_dev: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TestResult {
    pub candidate: usize,
    /// Agrégats seulement : les réponses par item sont dans private/, jamais servies par MCP.
    pub pillars: Vec<PillarResult>,
    pub verdict: Verdict,
}

/// `<job>/results.json`: nothing in it depends on time, so a resume rewrites it identically.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Results {
    /// Le moteur réel des piliers binaires : `native-cpu`, `llama-server (Vulkan, ngl=99, batch=1)`.
    #[serde(default)]
    pub backend: String,
    pub job_id: String,
    pub config_sha256: String,
    /// GO seulement si le candidat retenu sur le Dev passe toutes les portes sur le Test.
    pub verdict: Verdict,
    pub verdict_reason: String,
    pub model: ModelCheck,
    /// --allow-disable-safety (ligne de commande seulement) a été utilisé.
    pub safety_disabled: bool,
    /// Le Test Set avait déjà été consulté (--allow-test-reuse) : le verdict n'est pas une
    /// confirmation indépendante.
    pub test_contaminated: bool,
    pub budget_exceeded: bool,
    pub candidates: Vec<CandidateResult>,
    /// Candidat retenu sur le Dev ; null si aucun ne passe.
    pub selected: Option<usize>,
    pub test: Option<TestResult>,
    pub notes: Vec<String>,
}

const NOTES: [&str; 3] = [
    "Les chiffres Dev ont servi à choisir le candidat : ce sont des chiffres de sélection, pas \
     de généralisation.",
    "La confirmation sur le Test, une seule fois et pour le seul candidat retenu, tient lieu de \
     correction de multiplicité.",
    "Tests appariés (mêmes prompts avant et après) : Newcombe apparié pour les piliers \
     binaires, bootstrap sur les prompts pour les dérives.",
];

// ------------------------------------------------------------------ worker

struct Worker<'a> {
    job: Job,
    file: JobFile,
    deps: &'a AppDeps,
    state: State,
    session_start: Instant,
    elapsed_before: u64,
    last_write: Instant,
    /// Dataset hashes by path.
    hashes: HashMap<PathBuf, String>,
    /// Measurements and dumps whose units are counted (each once per session).
    counted: HashSet<PathBuf>,
}

impl Worker<'_> {
    fn config(&self) -> &PipelineConfig {
        &self.file.config
    }

    fn cancel(&self) -> &AtomicBool {
        &self.deps.interrupted
    }

    fn elapsed(&self) -> u64 {
        self.elapsed_before + self.session_start.elapsed().as_secs()
    }

    fn over_budget(&self) -> bool {
        self.elapsed() >= self.config().budgets.max_wall_time_secs
    }

    fn save_state(&mut self, force: bool) -> Result<()> {
        if !force && self.last_write.elapsed().as_millis() < 500 {
            return Ok(());
        }
        self.state.elapsed_secs = self.elapsed();
        self.state.eta_secs = (self.state.units_done > 0).then(|| {
            let rate = self.state.elapsed_secs as f64 / self.state.units_done as f64;
            (rate * self.state.units_total.saturating_sub(self.state.units_done) as f64) as u64
        });
        self.state.updated_at = job::now_rfc3339();
        self.last_write = Instant::now();
        write_json(&self.job.path("state.json"), &self.state)
    }

    fn step(&mut self, stage: &str, step: String) -> Result<()> {
        if self.cancel().load(Ordering::SeqCst) || self.job.cancel_requested() {
            return Err(PipelineError::Cancelled);
        }
        self.state.stage = stage.into();
        self.state.step = step;
        self.save_state(true)
    }

    /// Adds the units of `what` the first time it is seen.
    fn count(&mut self, what: &Path, units: u64) {
        if self.counted.insert(what.to_path_buf()) {
            self.state.units_done += units;
        }
    }

    fn hash(&mut self, path: &Path) -> Result<String> {
        if let Some(h) = self.hashes.get(path) {
            return Ok(h.clone());
        }
        let h = sha256_file(path)?;
        self.hashes.insert(path.to_path_buf(), h.clone());
        Ok(h)
    }

    /// Units of one measurement (tokens generated or evaluated, upper estimate).
    fn units(&self, pillar: &PillarConfig, n: usize) -> u64 {
        let per_item = match pillar.kind {
            PillarKind::Kl | PillarKind::Logits => pillar
                .max_tokens
                .map_or(crate::bench::DEFAULT_DUMP_MAX_TOKENS as u64, u64::from),
            PillarKind::Gsm8k => pillar
                .max_new_tokens
                .map_or(u64::from(crate::bench::DEFAULT_MAX_NEW_TOKENS), u64::from),
            _ => pillar.max_new_tokens.map_or(
                u64::from(crate::bench::DEFAULT_REFUSAL_MAX_NEW_TOKENS),
                u64::from,
            ),
        };
        n as u64 * per_item
    }

    /// A binary measurement of `model` on `pillar`'s `split`, from its checkpoint when it
    /// exists.
    fn binary(
        &mut self,
        pillar: &PillarConfig,
        dataset: &Path,
        model: &Path,
        checkpoint: &Path,
    ) -> Result<Measurement> {
        let sha = self.hash(dataset)?;
        if let Some(m) = read_json_opt::<Measurement>(checkpoint)? {
            if m.dataset_sha256 == sha {
                let units = self.units(pillar, outcome_len(&m.outcomes));
                self.count(checkpoint, units);
                return Ok(m);
            }
        }
        let n = measure::count_items(pillar.kind, dataset, pillar.limit)?;
        let per_item = self.units(pillar, 1);
        let start = self.state.units_done;
        let deps = self.deps;
        let backend = self.file.config.backend.clone();
        let (outcomes, items) = {
            let state = std::cell::RefCell::new(&mut *self);
            measure::binary(
                pillar,
                dataset,
                model,
                backend.as_ref(),
                deps,
                &deps.interrupted,
                &mut |done, _| {
                    let mut w = state.borrow_mut();
                    w.state.units_done = start + done as u64 * per_item;
                    let _ = w.save_state(false);
                },
            )?
        };
        self.state.units_done = start + self.units(pillar, n);
        self.counted.insert(checkpoint.to_path_buf());
        let m = Measurement {
            pillar: pillar.id.clone(),
            dataset_sha256: sha,
            outcomes: Outcomes::Binary(outcomes),
            items,
        };
        write_json(checkpoint, &m)?;
        Ok(m)
    }

    /// The drift of `model` against the base model on `pillar`'s `split`.
    fn drift(
        &mut self,
        pillar: &PillarConfig,
        dataset: &Path,
        model: &Path,
        label: &str,
        checkpoint: &Path,
    ) -> Result<Measurement> {
        let sha = self.hash(dataset)?;
        if let Some(m) = read_json_opt::<Measurement>(checkpoint)? {
            if m.dataset_sha256 == sha {
                let units = self.units(pillar, outcome_len(&m.outcomes));
                self.count(checkpoint, units);
                return Ok(m);
            }
        }
        let work = self.job.work();
        let base_dump = measure::dump_path(&work, "base", &sha, pillar.max_tokens);
        let cand_dump = measure::dump_path(&work, label, &sha, pillar.max_tokens);
        let dumps = work.join("dumps");
        fs::create_dir_all(&dumps).map_err(|e| io_error(&dumps, e))?;
        let n = measure::count_items(pillar.kind, dataset, None)?;
        for (dump, source) in [
            (&base_dump, &self.config().model.clone()),
            (&cand_dump, &model.to_path_buf()),
        ] {
            if !dump.exists() {
                let deps = self.deps;
                measure::dump(
                    dataset,
                    pillar.max_tokens,
                    source,
                    dump,
                    deps,
                    &deps.interrupted,
                    &mut |_, _| {},
                )?;
            }
            let units = self.units(pillar, n);
            self.count(dump, units);
            self.save_state(false)?;
        }
        let (values, items) = measure::drift(
            pillar.kind,
            &base_dump,
            &cand_dump,
            self.deps,
            self.cancel(),
        )?;
        let _ = fs::remove_file(&cand_dump);
        let m = Measurement {
            pillar: pillar.id.clone(),
            dataset_sha256: sha,
            outcomes: Outcomes::Drift(values),
            items,
        };
        write_json(checkpoint, &m)?;
        Ok(m)
    }

    /// The pillar's gate for `candidate` against the base model on `split`.
    fn evaluate(
        &mut self,
        pillar: &PillarConfig,
        split: &str,
        candidate_label: &str,
        model: Option<&Path>,
        store: &dyn Fn(&Job, &str) -> PathBuf,
    ) -> Result<PillarResult> {
        let dataset = match split {
            "dev" => pillar.dataset_dev.clone(),
            _ => pillar.dataset_test.clone(),
        };
        let checkpoint_name = |who: &str| format!("{who}_{split}_{}", pillar.id);
        let cand_checkpoint = store(&self.job, &checkpoint_name(candidate_label));
        let model = match model {
            Some(m) => m.to_path_buf(),
            None => PathBuf::new(),
        };
        let seed = self.config().seed;
        let result = if pillar.kind.is_binary() {
            let base_model = self.config().model.clone();
            let base_checkpoint = store(&self.job, &checkpoint_name("base"));
            let before = self.binary(pillar, &dataset, &base_model, &base_checkpoint)?;
            let after = self.binary(pillar, &dataset, &model, &cand_checkpoint)?;
            let (Outcomes::Binary(b), Outcomes::Binary(a)) = (&before.outcomes, &after.outcomes)
            else {
                return Err(PipelineError::Io("point de reprise incohérent".into()));
            };
            let counts = PairedCounts::from_pairs(b, a);
            let gate = pillar.gate.binary().ok_or_else(|| {
                PipelineError::Config(vec![format!("pilier {} : porte non binaire", pillar.id)])
            })?;
            let rate = |v: &[bool]| v.iter().filter(|x| **x).count() as f64 / v.len().max(1) as f64;
            // How many GSM8K items of a measurement carry `field == value`.
            let items_with = |m: &Measurement, field: &str, value: Value| {
                let count = |items: &Vec<Value>| items.iter().filter(|i| i[field] == value).count();
                (pillar.kind == PillarKind::Gsm8k).then(|| m.items.as_array().map_or(0, count))
            };
            PillarResult {
                pillar: pillar.id.clone(),
                kind: pillar.kind,
                metric: pillar.kind.metric().into(),
                split: split.into(),
                n: counts.n(),
                rate_before: Some(rate(b)),
                rate_after: Some(rate(a)),
                counts: Some(counts),
                truncated_before: items_with(&before, "truncated", json!(true)),
                truncated_after: items_with(&after, "truncated", json!(true)),
                loops_before: items_with(&before, "reason", json!("loop")),
                loops_after: items_with(&after, "reason", json!("loop")),
                no_answer_before: items_with(&before, "reason", json!("no_answer")),
                no_answer_after: items_with(&after, "reason", json!("no_answer")),
                gate: binary_gate(&counts, gate),
            }
        } else {
            let m = self.drift(pillar, &dataset, &model, candidate_label, &cand_checkpoint)?;
            let Outcomes::Drift(values) = &m.outcomes else {
                return Err(PipelineError::Io("point de reprise incohérent".into()));
            };
            let values: Vec<f64> = values.iter().map(|v| v.unwrap_or(f64::INFINITY)).collect();
            let config::Gate::MaxDrift { value, confidence } = pillar.gate else {
                return Err(PipelineError::Config(vec![format!(
                    "pilier {} : porte max_drift attendue",
                    pillar.id
                )]));
            };
            PillarResult {
                pillar: pillar.id.clone(),
                kind: pillar.kind,
                metric: pillar.kind.metric().into(),
                split: split.into(),
                n: values.len(),
                rate_before: None,
                rate_after: None,
                counts: None,
                truncated_before: None,
                truncated_after: None,
                loops_before: None,
                loops_after: None,
                no_answer_before: None,
                no_answer_after: None,
                gate: drift_gate(&values, value, confidence, seed),
            }
        };
        Ok(result)
    }
}

fn outcome_len(o: &Outcomes) -> usize {
    match o {
        Outcomes::Binary(v) => v.len(),
        Outcomes::Drift(v) => v.len(),
    }
}

// ------------------------------------------------------------------ working copy

fn copy_model(from: &Path, to: &Path) -> Result<()> {
    if to.exists() {
        fs::remove_dir_all(to).map_err(|e| io_error(to, e))?;
    }
    fs::create_dir_all(to).map_err(|e| io_error(to, e))?;
    for (name, path) in job::files(from)? {
        fs::copy(&path, to.join(&name)).map_err(|e| io_error(&path, e))?;
    }
    Ok(())
}

/// Applies `patch` to the working copy; with `backups`, keeps a verified copy of every file
/// it touches there and returns the (backup, original) pairs.
fn apply(
    patch: &PatchFile,
    model: &Path,
    backups: Option<&Path>,
    deps: &AppDeps,
) -> Result<Vec<(PathBuf, PathBuf)>> {
    let request = ApplyRequest {
        model: model.to_path_buf(),
        patch: PatchFile {
            backup: backups.is_some(),
            ..patch.clone()
        },
        patch_label: PathBuf::from("pipeline"),
        base_dir: model.to_path_buf(),
        dry_run: false,
        no_backup: backups.is_none(),
        backup_dir: backups.map(Path::to_path_buf),
    };
    if let Some(dir) = backups {
        fs::create_dir_all(dir).map_err(|e| io_error(dir, e))?;
    }
    let outcome = core_apply(&request, deps, &mut NullSink)
        .map_err(|o| PipelineError::Apply(o.error.map_or_else(String::new, |e| e.message)))?;
    let files: Vec<PathBuf> =
        serde_json::from_value(outcome.data["backups"].clone()).unwrap_or_default();
    Ok(files
        .into_iter()
        .filter_map(|backup| {
            let name = backup.file_name()?.to_string_lossy().into_owned();
            let original = name.rfind(".bak-").map(|at| model.join(&name[..at]))?;
            Some((backup, original))
        })
        .collect())
}

/// Puts every backed up file back, then removes the backups.
fn restore(pairs: &[(PathBuf, PathBuf)]) -> Result<()> {
    for (backup, original) in pairs {
        fs::copy(backup, original).map_err(|e| io_error(original, e))?;
        let _ = fs::remove_file(backup);
    }
    Ok(())
}

/// Dry run of `patch` on the original model: validation and target resolution, nothing
/// written.
fn dry_run(patch: &PatchFile, model: &Path, deps: &AppDeps) -> std::result::Result<(), String> {
    let request = ApplyRequest {
        model: model.to_path_buf(),
        // As in `apply`: the template's `backup` (true by default) is about the user's
        // model; here nothing is written, and the run patches a disposable working copy.
        patch: PatchFile {
            backup: false,
            ..patch.clone()
        },
        patch_label: PathBuf::from("pipeline"),
        base_dir: model.to_path_buf(),
        dry_run: true,
        no_backup: true,
        backup_dir: None,
    };
    core_apply(&request, deps, &mut NullSink)
        .map(|_| ())
        .map_err(|o| o.error.map_or_else(String::new, |e| e.message))
}

// ------------------------------------------------------------------ entry point

/// What ended the run.
pub enum Ended {
    Done(Box<Results>),
    BudgetExceeded(Box<Results>),
}

/// Runs (or resumes) the job. Cancellation (the `cancel` file, or `deps.interrupted`)
/// removes the working copy and leaves only complete checkpoints.
pub fn run(job: &Job, file: JobFile, deps: &AppDeps) -> Result<Ended> {
    let _lock = Lock::acquire(job)?;
    job.prepare()?;
    let _ = fs::remove_file(job.path("cancel"));
    if let Some(results) = read_json_opt::<Results>(&job.path("results.json"))? {
        return Ok(Ended::Done(Box::new(results)));
    }
    let previous: Option<State> = read_json_opt(&job.path("state.json"))?;
    let elapsed_before = previous.as_ref().map_or(0, |s| s.elapsed_secs);
    let units_total = previous.as_ref().map_or(0, |s| s.units_total);
    let mut worker = Worker {
        state: State {
            job_id: file.job_id.clone(),
            status: Status::Running,
            stage: "static_checks".into(),
            step: String::new(),
            units_done: 0,
            units_total,
            elapsed_secs: elapsed_before,
            eta_secs: None,
            dev_gates: Vec::new(),
            error: None,
            updated_at: job::now_rfc3339(),
        },
        job: job.clone(),
        file,
        deps,
        session_start: Instant::now(),
        elapsed_before,
        last_write: Instant::now(),
        hashes: HashMap::new(),
        counted: HashSet::new(),
    };
    // `apply` and `check` report an interruption as a failure of their own: once a
    // cancellation was requested, whatever error follows is that cancellation.
    let outcome = execute(&mut worker).map_err(|e| {
        match worker.cancel().load(Ordering::SeqCst) || job.cancel_requested() {
            true => PipelineError::Cancelled,
            false => e,
        }
    });
    let work = job.work();
    let _ = fs::remove_dir_all(&work);
    let _ = fs::remove_file(job.path("cancel"));
    match &outcome {
        Ok(Ended::Done(_)) => worker.state.status = Status::Done,
        Ok(Ended::BudgetExceeded(_)) => worker.state.status = Status::BudgetExceeded,
        Err(PipelineError::Cancelled) => worker.state.status = Status::Cancelled,
        Err(e) => {
            worker.state.status = Status::Failed;
            worker.state.error = Some(e.to_string());
        }
    }
    worker.state.stage = "end".into();
    worker.state.eta_secs = Some(0);
    worker.save_state(true)?;
    outcome
}

fn execute(w: &mut Worker) -> Result<Ended> {
    let config = w.config().clone();
    w.step("static_checks", "modèle d'origine".into())?;
    let model_before = sha256_dir(&config.model)?;
    let check = core_check(
        &CheckArgs {
            model: config.model.clone(),
            policy_ram_margin_gib: None,
        },
        w.deps,
        &crate::ui::Ui::off(),
    );
    if let Err(o) = check {
        return Err(PipelineError::Apply(format!(
            "check : {}",
            o.error.map_or_else(String::new, |e| e.message)
        )));
    }
    for p in &config.pillars {
        let (dev, test) = (w.hash(&p.dataset_dev)?, w.hash(&p.dataset_test)?);
        if dev == test {
            return Err(PipelineError::Config(vec![format!(
                "pilier {} : dataset_dev et dataset_test ont le même contenu",
                p.id
            )]));
        }
    }
    let patches: Vec<PatchFile> = config
        .candidates
        .values()
        .iter()
        .map(|v| config.candidates.patch(&config.patch, *v))
        .collect();
    if w.state.units_total == 0 {
        w.state.units_total = super::plan::units_total(&config)?;
    }

    let mut candidates = Vec::new();
    let mut budget_exceeded = false;
    let work_model = w.job.work().join("model");
    let backups = w.job.work().join("backups");
    let mut copied = false;
    for (index, patch) in patches.iter().enumerate() {
        let value = config.candidates.values()[index];
        let patch_sha256 = sha256_bytes(&serde_json::to_vec(patch).unwrap_or_default());
        let label = format!("cand{index}");
        let mut result = CandidateResult {
            index,
            value,
            patch_sha256,
            static_error: None,
            dev: Vec::new(),
            stopped_at: None,
            passed_dev: false,
        };
        if w.over_budget() {
            budget_exceeded = true;
            result.stopped_at = Some("budget".into());
            candidates.push(result);
            continue;
        }
        w.step("static_checks", format!("dry-run du candidat {index}"))?;
        if let Err(e) = dry_run(patch, &config.model, w.deps) {
            result.static_error = Some(e);
            result.stopped_at = Some("static_checks".into());
            candidates.push(result);
            continue;
        }
        // The patched copy is only made when a measurement is missing.
        let mut applied: Option<Vec<(PathBuf, PathBuf)>> = None;
        let mut stopped = None;
        for pillar in &config.pillars {
            if w.over_budget() {
                budget_exceeded = true;
                stopped = Some("budget".to_string());
                break;
            }
            w.step(
                "candidate_dev",
                format!("candidat {index}, pilier {}", pillar.id),
            )?;
            let needs_model = !w
                .job
                .checkpoint(&format!("{label}_dev_{}", pillar.id))
                .exists();
            if needs_model && applied.is_none() {
                w.step("working_copy", format!("candidat {index}"))?;
                let keep = config.working_copy == WorkingCopy::RestoreFromBackup;
                if !(keep && copied) {
                    copy_model(&config.model, &work_model)?;
                    copied = true;
                }
                applied = Some(apply(
                    patch,
                    &work_model,
                    keep.then_some(backups.as_path()),
                    w.deps,
                )?);
            }
            let r = w.evaluate(
                pillar,
                "dev",
                &label,
                Some(&work_model),
                &|job: &Job, name: &str| job.checkpoint(name),
            )?;
            w.state.dev_gates.push(DevGate {
                candidate: index,
                pillar: pillar.id.clone(),
                verdict: r.gate.verdict,
            });
            let verdict = r.gate.verdict;
            result.dev.push(r);
            if verdict != Verdict::Go {
                stopped = Some(pillar.id.clone());
                break;
            }
        }
        if let Some(pairs) = applied {
            match config.working_copy {
                WorkingCopy::RestoreFromBackup => restore(&pairs)?,
                WorkingCopy::Copy => {
                    let _ = fs::remove_dir_all(&work_model);
                }
            }
        }
        result.passed_dev = stopped.is_none() && result.dev.len() == config.pillars.len();
        result.stopped_at = stopped;
        // Units of the pillars skipped by the early stop are no longer expected.
        if !result.passed_dev {
            let done: usize = result.dev.len();
            for pillar in &config.pillars[done..] {
                let n = measure::count_items(pillar.kind, &pillar.dataset_dev, pillar.limit)?;
                let skipped = w.units(pillar, n);
                w.state.units_total = w.state.units_total.saturating_sub(skipped);
            }
        }
        candidates.push(result);
    }
    let _ = fs::remove_dir_all(&work_model);

    // Selection on the Dev only.
    let selection = &config.selection.pillar;
    let selected = candidates
        .iter()
        .filter(|c| c.passed_dev)
        .max_by(|a, b| {
            let effect = |c: &CandidateResult| {
                c.dev
                    .iter()
                    .find(|r| &r.pillar == selection)
                    .map_or(f64::MIN, |r| r.gate.estimate)
            };
            effect(a)
                .total_cmp(&effect(b))
                .then(b.value.abs().total_cmp(&a.value.abs()))
                .then(b.index.cmp(&a.index))
        })
        .map(|c| c.index);

    let mut test = None;
    let mut contaminated = false;
    if let (Some(index), false) = (selected, budget_exceeded) {
        w.step("test", format!("candidat {index}"))?;
        let datasets: Vec<String> = config
            .pillars
            .iter()
            .map(|p| w.hash(&p.dataset_test))
            .collect::<Result<_>>()?;
        let registry = config.test_registry.clone().unwrap_or_default();
        let entries = read_registry(&registry)?;
        let candidate_sha256 = candidates[index].patch_sha256.clone();
        let job_dir = std::path::absolute(&w.job.dir).map_err(|e| io_error(&w.job.dir, e))?;
        let ours = entries.iter().find(|e| {
            e.job_id == w.file.job_id
                && e.job_dir == job_dir
                && e.candidate_sha256 == candidate_sha256
        });
        match ours {
            // A resume of this very consultation.
            Some(entry) => contaminated = entry.reuse,
            None => {
                let seen = entries
                    .iter()
                    .any(|e| e.datasets_sha256.iter().any(|d| datasets.contains(d)));
                if seen && !w.file.options.allow_test_reuse {
                    return Err(PipelineError::TestSet(
                        "Test Set déjà consulté (registre des accès) : seconde consultation \
                         refusée ; --allow-test-reuse (ligne de commande) la permet et marque le \
                         rapport comme contaminé"
                            .into(),
                    ));
                }
                contaminated = seen;
                append_registry(
                    &registry,
                    TestAccess {
                        job_id: w.file.job_id.clone(),
                        job_dir,
                        candidate_sha256,
                        datasets_sha256: datasets,
                        date: job::now_rfc3339(),
                        reuse: seen,
                    },
                )?;
            }
        }
        let label = format!("cand{index}");
        let needs_model = config
            .pillars
            .iter()
            .any(|p| !w.job.private(&format!("{label}_test_{}", p.id)).exists());
        if needs_model {
            copy_model(&config.model, &work_model)?;
            apply(&patches[index], &work_model, None, w.deps)?;
        }
        let mut pillars = Vec::new();
        for pillar in &config.pillars {
            w.step("test", format!("candidat {index}, pilier {}", pillar.id))?;
            pillars.push(w.evaluate(
                pillar,
                "test",
                &label,
                Some(&work_model),
                &|job: &Job, name: &str| job.private(name),
            )?);
        }
        let _ = fs::remove_dir_all(&work_model);
        let verdict = combine(pillars.iter().map(|p| p.gate.verdict));
        test = Some(TestResult {
            candidate: index,
            pillars,
            verdict,
        });
    }

    w.step("report", String::new())?;
    // Estimates: generation may stop early, so the count may fall short of the plan.
    w.state.units_done = w.state.units_total;
    let model_after = sha256_dir(&config.model)?;
    let (verdict, reason) = match (&test, selected, budget_exceeded) {
        (_, _, true) => (
            Verdict::Inconclusive,
            "budget de temps épuisé avant la fin".to_string(),
        ),
        (Some(t), _, _) if contaminated && t.verdict == Verdict::Go => (
            Verdict::Inconclusive,
            "Test Set contaminé (déjà consulté) : pas de confirmation indépendante".into(),
        ),
        (Some(t), _, _) => (
            t.verdict,
            match t.verdict {
                Verdict::Go => "le candidat retenu passe toutes les portes sur le Test".into(),
                Verdict::NoGo => "au moins une porte échoue sur le Test".into(),
                Verdict::Inconclusive => "au moins une porte du Test n'est pas concluante".into(),
            },
        ),
        (None, None, _) => {
            let any_inconclusive = candidates.iter().any(|c| {
                c.dev
                    .last()
                    .is_some_and(|r| r.gate.verdict == Verdict::Inconclusive)
            });
            match any_inconclusive {
                _ if candidates.iter().all(|c| c.static_error.is_some()) => (
                    Verdict::NoGo,
                    "aucun candidat ne passe les contrôles statiques (dry-run du patch sur le \
                     modèle d'origine, voir candidates[].static_error) : le Dev n'a pas été mesuré"
                        .into(),
                ),
                true => (
                    Verdict::Inconclusive,
                    "aucun candidat ne passe le Dev ; au moins une porte non concluante".into(),
                ),
                false => (Verdict::NoGo, "aucun candidat ne passe le Dev".into()),
            }
        }
        (None, Some(_), _) => (Verdict::Inconclusive, "Test non évalué".into()),
    };
    let results = Results {
        backend: crate::backend::describe_config(&config.backend.clone().unwrap_or_default()),
        job_id: w.file.job_id.clone(),
        config_sha256: w.file.config_sha256.clone(),
        verdict: if model_after == model_before {
            verdict
        } else {
            Verdict::NoGo
        },
        verdict_reason: if model_after == model_before {
            reason
        } else {
            "le modèle d'origine a changé pendant le job".into()
        },
        model: ModelCheck {
            path: config.model.clone(),
            unchanged: model_after == model_before,
            sha256_before: model_before,
            sha256_after: model_after,
        },
        safety_disabled: !config.pillars.iter().any(|p| p.kind == PillarKind::Safety),
        test_contaminated: contaminated,
        budget_exceeded,
        candidates,
        selected,
        test,
        notes: NOTES.iter().map(|s| s.to_string()).collect(),
    };
    write_json(&w.job.path("results.json"), &results)?;
    write_json(&w.job.path("report.json"), &report(w, &results)?)?;
    Ok(match budget_exceeded {
        true => Ended::BudgetExceeded(Box::new(results)),
        false => Ended::Done(Box::new(results)),
    })
}

/// GO if every gate is GO; NO_GO if any is NO_GO; INCONCLUSIVE otherwise.
pub fn combine(verdicts: impl Iterator<Item = Verdict>) -> Verdict {
    let mut out = Verdict::Go;
    for v in verdicts {
        match v {
            Verdict::NoGo => return Verdict::NoGo,
            Verdict::Inconclusive => out = Verdict::Inconclusive,
            Verdict::Go => {}
        }
    }
    out
}

/// `<job>/report.json`: the results, the reproducibility manifest and the timings.
fn report(w: &mut Worker, results: &Results) -> Result<Value> {
    let config = w.config().clone();
    let mut datasets = Vec::new();
    for p in &config.pillars {
        for (split, path) in [("dev", &p.dataset_dev), ("test", &p.dataset_test)] {
            datasets.push(json!({
                "pillar": p.id, "split": split, "path": path, "sha256": w.hash(path)?,
            }));
        }
    }
    let mut vectors = Vec::new();
    for op in &config.patch.operations {
        vectors.push(json!({"path": op.vector_file, "sha256": w.hash(&op.vector_file)?}));
    }
    Ok(json!({
        "results": results,
        "manifest": {
            "surgeon_version": env!("CARGO_PKG_VERSION"),
            "config_sha256": w.file.config_sha256,
            "config": config,
            "model_sha256": results.model.sha256_before,
            "datasets": datasets,
            "vectors": vectors,
            "backend": results.backend,
            "llama_cpp": match config.backend.as_ref().and_then(|b| b.llama_cpp_dir.as_ref()) {
                Some(dir) => json!(dir),
                None => json!("non utilisé (backend natif, décodage glouton)"),
            },
            "seeds": {"bootstrap": config.seed},
            "options": w.file.options,
        },
        "timing": {
            "elapsed_secs": w.elapsed(),
            "finished_at": job::now_rfc3339(),
        },
    }))
}

/// Reads the results of a finished job.
pub fn results(job: &Job) -> Result<Results> {
    read_json(&job.path("results.json"))
}
