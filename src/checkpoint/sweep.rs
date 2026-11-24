use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::Serialize;

use super::pareto::{self, Point};
use super::{CheckpointError, Result, Store};
use crate::app::{core_apply, AppDeps, ApplyRequest, NullSink, ProgressEvent, ProgressSink};
use crate::bench::generation::GenOptions;
use crate::bench::ref_cache::RefCache;
use crate::patch::{Axis, LayersField, Operation, OperationKind, PatchFile, Projection};
use crate::pipeline::config::{Gate, PillarConfig, PillarKind};
use crate::pipeline::{measure, PipelineError};

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Candidate {
    pub vector: PathBuf,
    pub layers: String,
    pub alpha: f32,
}

impl Candidate {
    pub fn label(&self) -> String {
        let stem = self
            .vector
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        format!("{stem} · L{} · α {}", self.layers, self.alpha)
    }
}

pub fn grid(vectors: &[PathBuf], windows: &[String], alphas: &[f32]) -> Vec<Candidate> {
    vectors
        .iter()
        .flat_map(|vector| {
            windows.iter().flat_map(move |layers| {
                alphas.iter().map(move |&alpha| Candidate {
                    vector: vector.clone(),
                    layers: layers.clone(),
                    alpha,
                })
            })
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Score {
    pub target: f64,
    pub drift: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    Kept,
    DriftExceeded,
    Dominated,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Evaluated {
    pub index: usize,
    #[serde(flatten)]
    pub candidate: Candidate,
    pub target: f64,
    pub gain: f64,
    pub drift: f64,
    pub decision: Decision,
    pub checkpoint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SweepReport {
    pub model: PathBuf,
    pub base_checkpoint: String,
    pub base_target: f64,
    pub max_drift: f64,
    pub points: Vec<Evaluated>,
    pub front: Vec<usize>,
    pub best: Option<usize>,
    pub restored: String,
}

pub trait Lab {
    fn check(&mut self, model: &Path, candidate: &Candidate) -> Result<()>;
    fn apply(&mut self, model: &Path, candidate: &Candidate) -> Result<()>;
    fn measure(&mut self, model: &Path, base: bool) -> Result<Score>;
    fn record(&mut self, _point: &Evaluated, _total: usize) {}
}

fn metrics(score: Score, gain: f64, alpha: Option<f32>) -> BTreeMap<String, f64> {
    let mut m = BTreeMap::from([
        ("target".to_owned(), score.target),
        ("gain".to_owned(), gain),
        ("drift".to_owned(), score.drift),
    ]);
    if let Some(alpha) = alpha {
        m.insert("alpha".to_owned(), f64::from(alpha));
    }
    m
}

pub fn run(
    store: &Store,
    model: &Path,
    grid: &[Candidate],
    max_drift: f64,
    lab: &mut dyn Lab,
    cancel: &AtomicBool,
) -> Result<SweepReport> {
    if grid.is_empty() {
        return Err(CheckpointError::Invalid("grille vide".into()));
    }
    if !(max_drift.is_finite() && max_drift >= 0.0) {
        return Err(CheckpointError::Invalid(format!(
            "max_drift = {max_drift} : un nombre positif ou nul"
        )));
    }
    for candidate in grid {
        lab.check(model, candidate)?;
    }
    let base_score = lab.measure(model, true)?;
    let base_score = Score {
        drift: 0.0,
        ..base_score
    };
    let base = store.create(model, "base", metrics(base_score, 0.0, None))?;
    let mut seen = vec![Point {
        gain: 0.0,
        drift: 0.0,
    }];
    let mut points = Vec::with_capacity(grid.len());
    for (index, candidate) in grid.iter().enumerate() {
        let mut step = || -> Result<Evaluated> {
            if cancel.load(Ordering::SeqCst) {
                return Err(CheckpointError::Cancelled);
            }
            lab.apply(model, candidate)?;
            let score = lab.measure(model, false)?;
            let point = Point {
                gain: score.target - base_score.target,
                drift: score.drift,
            };
            let decision = match point.drift <= max_drift {
                false => Decision::DriftExceeded,
                true if pareto::improves(&seen, point) => Decision::Kept,
                true => Decision::Dominated,
            };
            let checkpoint = match decision {
                Decision::Kept => {
                    let m = metrics(score, point.gain, Some(candidate.alpha));
                    seen.push(point);
                    Some(store.create(model, &candidate.label(), m)?.id)
                }
                _ => None,
            };
            Ok(Evaluated {
                index,
                candidate: candidate.clone(),
                target: score.target,
                gain: point.gain,
                drift: point.drift,
                decision,
                checkpoint,
            })
        };
        let evaluated = step();
        match (evaluated, store.restore(&base.id, model)) {
            (Ok(e), Ok(_)) => {
                lab.record(&e, grid.len());
                points.push(e);
            }
            (Err(e), Ok(_)) | (Ok(_), Err(e)) => return Err(e),
            (Err(e), Err(r)) => {
                return Err(CheckpointError::Failed(format!(
                    "{e} ; rollback vers {} impossible : {r}",
                    base.id
                )))
            }
        }
    }
    let kept: Vec<usize> = points
        .iter()
        .filter(|p| p.decision == Decision::Kept)
        .map(|p| p.index)
        .collect();
    let kept_points: Vec<Point> = kept
        .iter()
        .map(|&i| Point {
            gain: points[i].gain,
            drift: points[i].drift,
        })
        .collect();
    let front_local = pareto::front(&kept_points);
    let best = pareto::best(&kept_points, &front_local).map(|i| kept[i]);
    let front = front_local.iter().map(|&i| kept[i]).collect();
    let restored = match best.and_then(|i| points[i].checkpoint.clone()) {
        Some(id) => {
            store.restore(&id, model)?;
            id
        }
        None => base.id.clone(),
    };
    store.check_model(&restored, model)?;
    Ok(SweepReport {
        model: model.to_path_buf(),
        base_checkpoint: base.id,
        base_target: base_score.target,
        max_drift,
        points,
        front,
        best,
        restored,
    })
}

pub struct Target {
    pub kind: PillarKind,
    pub dataset: PathBuf,
    pub max_new_tokens: Option<u32>,
    pub limit: Option<u32>,
}

pub struct Drift {
    pub kind: PillarKind,
    pub dataset: PathBuf,
    pub max_tokens: Option<u32>,
}

pub struct EngineLab<'a> {
    pub deps: &'a AppDeps,
    pub sink: &'a mut dyn ProgressSink,
    pub operation: OperationKind,
    pub projections: Vec<Projection>,
    pub vector_tensor: Option<String>,
    pub target: Target,
    pub drift: Drift,
    pub scratch: PathBuf,
}

fn measure_error(e: PipelineError) -> CheckpointError {
    match e {
        PipelineError::Cancelled => CheckpointError::Cancelled,
        other => CheckpointError::Failed(other.to_string()),
    }
}

impl EngineLab<'_> {
    pub fn validate(&self) -> Result<()> {
        if !self.target.kind.is_binary() {
            return Err(CheckpointError::Invalid(
                "target.kind : safety, unblocking ou gsm8k".into(),
            ));
        }
        if self.drift.kind.is_binary() {
            return Err(CheckpointError::Invalid("drift.kind : kl ou logits".into()));
        }
        Ok(())
    }

    fn patch(&self, candidate: &Candidate) -> PatchFile {
        let json = candidate
            .vector
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("json"));
        let tensor = match json {
            true => None,
            false => Some(
                self.vector_tensor
                    .clone()
                    .unwrap_or_else(|| "direction".into()),
            ),
        };
        let project_out = self.operation == OperationKind::ProjectOut;
        PatchFile {
            schema_version: crate::patch::SUPPORTED_SCHEMA_VERSION,
            description: Some(format!("surgeon_sweep : {}", candidate.label())),
            backup: false,
            operations: vec![Operation {
                id: "sweep".into(),
                layers: LayersField::Spec(candidate.layers.clone()),
                projections: self.projections.clone(),
                operation: self.operation,
                vector_file: candidate.vector.clone(),
                vector_tensor: tensor.clone(),
                vector_tensor_b: tensor.filter(|_| project_out),
                rank: None,
                allow_non_projector: false,
                alpha: candidate.alpha,
                axis: Axis::default(),
            }],
            composition: None,
        }
    }

    fn core_apply(&self, model: &Path, candidate: &Candidate, dry_run: bool) -> Result<()> {
        let request = ApplyRequest {
            model: model.to_path_buf(),
            patch: self.patch(candidate),
            patch_label: PathBuf::from("sweep"),
            base_dir: model.to_path_buf(),
            dry_run,
            no_backup: true,
            backup_dir: None,
        };
        core_apply(&request, self.deps, &mut NullSink)
            .map(|_| ())
            .map_err(|o| {
                let message = o.error.map_or_else(String::new, |e| e.message);
                CheckpointError::Failed(format!("{} : {message}", candidate.label()))
            })
    }

    fn target_rate(&self, model: &Path, base: bool) -> Result<f64> {
        let pillar = PillarConfig {
            id: "target".into(),
            kind: self.target.kind,
            dataset_dev: self.target.dataset.clone(),
            dataset_test: self.target.dataset.clone(),
            gate: Gate::Superiority {
                min_gain: 0.0,
                confidence: 0.95,
            },
            max_new_tokens: self.target.max_new_tokens,
            chat_template: None,
            prompt_template: None,
            format: None,
            limit: self.target.limit,
            max_tokens: None,
        };
        let cache = base.then(RefCache::from_env).flatten();
        let (success, _) = measure::binary(
            &pillar,
            &self.target.dataset,
            model,
            None,
            GenOptions::default(),
            cache.as_ref(),
            self.deps,
            &self.deps.interrupted,
            &mut |_, _| {},
        )
        .map_err(measure_error)?;
        if success.is_empty() {
            return Err(CheckpointError::Invalid(format!(
                "{} : aucun élément mesuré",
                self.target.dataset.display()
            )));
        }
        Ok(success.iter().filter(|s| **s).count() as f64 / success.len() as f64)
    }

    fn dump(&self, model: &Path, name: &str) -> Result<PathBuf> {
        let out = self.scratch.join(name);
        measure::dump(
            &self.drift.dataset,
            self.drift.max_tokens,
            model,
            &out,
            self.deps,
            &self.deps.interrupted,
            &mut |_, _| {},
        )
        .map_err(measure_error)?;
        Ok(out)
    }
}

impl Lab for EngineLab<'_> {
    fn check(&mut self, model: &Path, candidate: &Candidate) -> Result<()> {
        self.core_apply(model, candidate, true)
    }

    fn apply(&mut self, model: &Path, candidate: &Candidate) -> Result<()> {
        let applied = self.core_apply(model, candidate, false);
        if let Some(cache) = crate::extract::weight_cache::global() {
            cache.revalidate();
        }
        applied
    }

    fn measure(&mut self, model: &Path, base: bool) -> Result<Score> {
        let target = self.target_rate(model, base)?;
        if base {
            self.dump(model, "base.safetensors")?;
            return Ok(Score { target, drift: 0.0 });
        }
        let after = self.dump(model, "candidate.safetensors")?;
        let before = self.scratch.join("base.safetensors");
        let (values, _) = measure::drift(
            self.drift.kind,
            &before,
            &after,
            self.deps,
            &self.deps.interrupted,
        )
        .map_err(measure_error)?;
        let drift = match values.iter().copied().collect::<Option<Vec<f64>>>() {
            Some(v) if !v.is_empty() => v.iter().sum::<f64>() / v.len() as f64,
            Some(_) => {
                return Err(CheckpointError::Invalid(format!(
                    "{} : aucun prompt de dérive",
                    self.drift.dataset.display()
                )))
            }
            None => f64::INFINITY,
        };
        Ok(Score { target, drift })
    }

    fn record(&mut self, point: &Evaluated, total: usize) {
        self.sink.event(ProgressEvent::Info(format!(
            "sweep {}/{total} · {} · gain {:+.4} · dérive {:.4} · {:?}",
            point.index + 1,
            point.candidate.label(),
            point.gain,
            point.drift,
            point.decision,
        )));
    }
}
