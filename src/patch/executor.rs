use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::error::{PatchError, Result};
use super::schema::{Axis, LayerSelection, OperationKind, Projection};
use super::validate::{ValidatedOperation, ValidatedPatch, VectorTensor, DEFAULT_MAX_RANK};
use crate::engine::SubspaceProjection;

/// `‖BᵀA − I‖_F` above this at full strength: `a` and `b` are not a projector.
pub const PROJECTOR_TOLERANCE: f64 = 1e-3;

/// What the executor needs from the tensor engine. The integration implements it on top of
/// `engine` and converts its errors with `PatchError::Backend(e.to_string())`.
pub trait PatchBackend {
    fn resolve_targets(
        &self,
        layers: &LayerSelection,
        projections: &[Projection],
    ) -> Result<Vec<String>>;
    fn load_vector(&self, file: &Path, tensor: Option<&str>) -> Result<Vec<f32>>;
    fn shift_tensor(
        &mut self,
        tensor_name: &str,
        direction: &[f32],
        alpha: f32,
        op: OperationKind,
        axis: Axis,
    ) -> Result<OpStats>;
    /// `W ← W − α a (bᵀ W)` (`project_out`). Backends without it refuse the operation.
    fn project_out_tensor(
        &mut self,
        tensor_name: &str,
        _a: &[f32],
        _b: &[f32],
        _alpha: f32,
    ) -> Result<OpStats> {
        Err(PatchError::Backend(format!(
            "{tensor_name} : project_out non supporté par ce backend"
        )))
    }
    /// The directions of a `project_out` tensor: `Some(k)` and `k × len` values for a
    /// `[k, len]` tensor (one direction per row), `None` for the historical 1D form. The
    /// default only knows 1D vectors.
    fn load_directions(
        &self,
        file: &Path,
        tensor: Option<&str>,
    ) -> Result<(Option<usize>, Vec<f32>)> {
        Ok((None, self.load_vector(file, tensor)?))
    }
    /// `W ← W − α A (Bᵀ W)` in one pass, `a` and `b` holding `k` directions each. The
    /// default sends one direction to `project_out_tensor` and refuses more.
    fn project_out_subspace(
        &mut self,
        tensor_name: &str,
        a: &[f32],
        b: &[f32],
        k: usize,
        alpha: f32,
    ) -> Result<OpStats> {
        match k {
            1 => self.project_out_tensor(tensor_name, a, b, alpha),
            _ => Err(PatchError::Backend(format!(
                "{tensor_name} : project_out de rang {k} non supporté par ce backend"
            ))),
        }
    }
    fn flush(&mut self) -> Result<()>;
}

/// Implemented by `BackupGuard` and `BackupSet`.
pub trait RollbackHandle {
    fn rollback(&mut self) -> Result<()>;
    fn commit(&mut self);
    /// Reported in `ApplyReport::backup_path`.
    fn backup_path(&self) -> Option<PathBuf> {
        None
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct OpStats {
    pub max_abs_delta: f32,
    pub elements_modified: u64,
    pub nan_or_inf_produced: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorReport {
    pub name: String,
    /// `None` in dry run.
    pub stats: Option<OpStats>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OperationReport {
    pub id: String,
    /// Length of one direction.
    pub vector_len: usize,
    /// `project_out` with `[k, len]` tensors: the rank k. Absent for the 1D form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
    pub tensors: Vec<TensorReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApplyReport {
    pub operations: Vec<OperationReport>,
    pub backup_path: Option<PathBuf>,
    pub dry_run: bool,
    /// Always `false` in an `Ok` report: a rollback is reported through
    /// `PatchError::ExecutionFailed { rolled_back: true, .. }`.
    pub rolled_back: bool,
    /// Found while loading the vectors (a `project_out` that is not a projector).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub warnings: Vec<String>,
}

/// Limits of a run that the patch file does not carry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ApplyOptions {
    /// Largest `project_out` rank accepted (`--max-rank`).
    pub max_rank: usize,
}

impl Default for ApplyOptions {
    fn default() -> Self {
        Self {
            max_rank: DEFAULT_MAX_RANK,
        }
    }
}

/// The read side of a checked `project_out`: `k` directions, as long as those of `a`.
struct Subspace {
    b: Vec<f32>,
    k: usize,
    /// `[k, len]` tensors rather than the 1D form.
    ranked: bool,
}

/// Runs a validated patch.
///
/// 1. Preflight, no write: resolve every target and load every vector. A failure here
///    leaves the model untouched (`rolled_back: false, left_modified: false`).
/// 2. Unless `dry_run`: shift tensors in patch order. Any error, or any NaN/Inf produced,
///    triggers `guard.rollback()`; a failed rollback is `RollbackFailed` (critical).
/// 3. `flush()` then `guard.commit()`.
///
/// In dry run nothing is written; a guard, if given, is committed since there is nothing
/// to undo (better: do not create one).
pub fn apply_patch(
    patch: &ValidatedPatch,
    backend: &mut dyn PatchBackend,
    guard: Option<&mut dyn RollbackHandle>,
    dry_run: bool,
) -> Result<ApplyReport> {
    apply_patch_with(patch, backend, guard, dry_run, &ApplyOptions::default())
}

/// The preflight of one `project_out`: both tensors, their rank against the declared one
/// and the limit, and whether they form a projector. Nothing is written.
fn load_subspace(
    backend: &dyn PatchBackend,
    op: &ValidatedOperation,
    names: (&str, &str),
    opts: &ApplyOptions,
    warnings: &mut Vec<String>,
) -> std::result::Result<(Vec<f32>, Subspace), String> {
    let (rank, allow_non_projector) = match &op.vector_tensor {
        Some(VectorTensor::Subspace {
            rank,
            allow_non_projector,
            ..
        }) => (*rank, *allow_non_projector),
        _ => (None, false),
    };
    let load = |name: &str| {
        backend
            .load_directions(&op.vector_file, Some(name))
            .map_err(|e| e.to_string())
    };
    let ((ka, a), (kb, b)) = (load(names.0)?, load(names.1)?);
    if b.len() != a.len() {
        return Err(format!(
            "a et b de tailles différentes ({} et {})",
            a.len(),
            b.len()
        ));
    }
    let (k, kb) = (ka.unwrap_or(1), kb.unwrap_or(1));
    if k != kb || k == 0 {
        return Err(format!("a et b de rangs différents ou nuls ({k} et {kb})"));
    }
    if let Some(declared) = rank.filter(|declared| *declared != k) {
        return Err(format!(
            "rank = {declared} mais les tenseurs a et b ont {k} direction(s)"
        ));
    }
    if k > opts.max_rank {
        return Err(format!("rang {k} > plafond {} (--max-rank)", opts.max_rank));
    }
    let gram = SubspaceProjection {
        a: &a,
        b: &b,
        alpha: op.alpha,
        k,
    }
    .gram_error();
    let ranked = ka.is_some();
    // Not `gram > tolerance`: an undefined error is not a projector either.
    if op.alpha == 1.0 && !gram.le(&PROJECTOR_TOLERANCE) {
        let message = format!(
            "‖BᵀA − I‖ = {gram:.3e} > {PROJECTOR_TOLERANCE:e} : a et b ne forment pas un \
             projecteur, alpha = 1 retire autre chose que le sous-espace lu par b"
        );
        // The 1D form was always accepted: it keeps working, with the warning.
        if ranked && !allow_non_projector {
            return Err(format!(
                "{message} (alpha < 1, ou allow_non_projector: true pour passer outre)"
            ));
        }
        warnings.push(format!("opération {} : {message}", op.id));
    }
    Ok((a, Subspace { b, k, ranked }))
}

/// [`apply_patch`] with explicit limits.
pub fn apply_patch_with(
    patch: &ValidatedPatch,
    backend: &mut dyn PatchBackend,
    guard: Option<&mut dyn RollbackHandle>,
    dry_run: bool,
    opts: &ApplyOptions,
) -> Result<ApplyReport> {
    let mut warnings = Vec::new();
    let mut plans = Vec::with_capacity(patch.operations.len());
    for op in &patch.operations {
        let untouched = |cause: String| PatchError::ExecutionFailed {
            op_id: op.id.clone(),
            cause,
            rolled_back: false,
            left_modified: false,
        };
        let targets = backend
            .resolve_targets(&op.layers, &op.projections)
            .map_err(|e| untouched(e.to_string()))?;
        if targets.is_empty() {
            return Err(untouched(format!(
                "aucun tenseur ne correspond à {:?} / {:?}",
                op.layers, op.projections
            )));
        }
        let tensors = op.vector_tensor.as_ref();
        let (vector, read) = match (&op.kind, tensors.and_then(|t| Some((t, t.read()?)))) {
            (OperationKind::ProjectOut, Some((a, b))) => {
                let names = (a.direction(), b);
                let (a, subspace) =
                    load_subspace(&*backend, op, names, opts, &mut warnings).map_err(untouched)?;
                (a, Some(subspace))
            }
            (OperationKind::ProjectOut, None) => {
                return Err(untouched("project_out sans vecteur b".into()));
            }
            _ => {
                let vector = backend
                    .load_vector(&op.vector_file, tensors.map(VectorTensor::direction))
                    .map_err(|e| untouched(e.to_string()))?;
                (vector, None)
            }
        };
        plans.push((op, targets, vector, read));
    }

    let mut report = ApplyReport {
        operations: Vec::with_capacity(plans.len()),
        backup_path: guard.as_ref().and_then(|g| g.backup_path()),
        dry_run,
        rolled_back: false,
        warnings,
    };
    let rank = |read: &Option<Subspace>| read.as_ref().filter(|s| s.ranked).map(|s| s.k);
    let direction_len =
        |vector: &[f32], read: &Option<Subspace>| vector.len() / read.as_ref().map_or(1, |s| s.k);

    if dry_run {
        report.operations = plans
            .into_iter()
            .map(|(op, targets, vector, read)| OperationReport {
                id: op.id.clone(),
                vector_len: direction_len(&vector, &read),
                rank: rank(&read),
                tensors: targets
                    .into_iter()
                    .map(|name| TensorReport { name, stats: None })
                    .collect(),
            })
            .collect();
        if let Some(g) = guard {
            g.commit();
        }
        return Ok(report);
    }

    for (op, targets, vector, read) in plans {
        let mut tensors = Vec::with_capacity(targets.len());
        for name in targets {
            let shifted = match &read {
                Some(s) => backend.project_out_subspace(&name, &vector, &s.b, s.k, op.alpha),
                None => backend.shift_tensor(&name, &vector, op.alpha, op.kind, op.axis),
            }
            .and_then(|stats| match stats.nan_or_inf_produced {
                0 => Ok(stats),
                n => Err(PatchError::Backend(format!(
                    "{n} valeurs NaN/Inf produites"
                ))),
            });
            match shifted {
                Ok(stats) => tensors.push(TensorReport {
                    name,
                    stats: Some(stats),
                }),
                Err(e) => return Err(abort(guard, &op.id, format!("{name} : {e}"))),
            }
        }
        report.operations.push(OperationReport {
            id: op.id.clone(),
            vector_len: direction_len(&vector, &read),
            rank: rank(&read),
            tensors,
        });
    }

    if let Err(e) = backend.flush() {
        return Err(abort(guard, "flush", e.to_string()));
    }
    if let Some(g) = guard {
        g.commit();
    }
    Ok(report)
}

/// The model may already be modified: restore it if we can, and say so either way.
fn abort(guard: Option<&mut dyn RollbackHandle>, op_id: &str, cause: String) -> PatchError {
    let op_id = op_id.to_owned();
    match guard {
        Some(g) => match g.rollback() {
            Ok(()) => PatchError::ExecutionFailed {
                op_id,
                cause,
                rolled_back: true,
                left_modified: false,
            },
            Err(e) => PatchError::RollbackFailed {
                op_id,
                cause: format!("{cause} ; rollback : {e}"),
            },
        },
        None => PatchError::ExecutionFailed {
            op_id,
            cause: format!("{cause} (pas de backup)"),
            rolled_back: false,
            left_modified: true,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::backup::BackupGuard;
    use crate::patch::error::exit_code;
    use crate::patch::validate::{ValidatedOperation, VectorTensor};
    use std::fs;

    /// Writes one byte per shifted tensor into a real file, so rollbacks are observable.
    struct MockBackend {
        file: PathBuf,
        targets_per_op: usize,
        fail_on_call: Option<usize>,
        nan_on_call: Option<usize>,
        calls: Vec<String>,
        flushed: bool,
    }

    impl MockBackend {
        fn new(file: &Path) -> Self {
            Self {
                file: file.to_path_buf(),
                targets_per_op: 2,
                fail_on_call: None,
                nan_on_call: None,
                calls: Vec::new(),
                flushed: false,
            }
        }
    }

    impl PatchBackend for MockBackend {
        fn resolve_targets(
            &self,
            layers: &LayerSelection,
            projections: &[Projection],
        ) -> Result<Vec<String>> {
            Ok((0..self.targets_per_op)
                .map(|i| format!("{layers:?}.{:?}.{i}", projections[0]))
                .collect())
        }
        fn load_vector(&self, _file: &Path, _tensor: Option<&str>) -> Result<Vec<f32>> {
            Ok(vec![1.0; 4])
        }
        fn shift_tensor(
            &mut self,
            name: &str,
            _: &[f32],
            _: f32,
            _: OperationKind,
            _: Axis,
        ) -> Result<OpStats> {
            let call = self.calls.len();
            self.calls.push(name.to_owned());
            if self.fail_on_call == Some(call) {
                return Err(PatchError::Backend("disque plein".into()));
            }
            let mut bytes = fs::read(&self.file).unwrap();
            bytes[call] = 0xFF;
            fs::write(&self.file, bytes).unwrap();
            Ok(OpStats {
                max_abs_delta: 0.5,
                elements_modified: 16,
                nan_or_inf_produced: if self.nan_on_call == Some(call) { 3 } else { 0 },
            })
        }
        fn flush(&mut self) -> Result<()> {
            self.flushed = true;
            Ok(())
        }
    }

    fn patch(ids: &[&str]) -> ValidatedPatch {
        ValidatedPatch {
            description: None,
            backup: true,
            warnings: Vec::new(),
            operations: ids
                .iter()
                .enumerate()
                .map(|(i, id)| ValidatedOperation {
                    id: (*id).to_owned(),
                    layers: LayerSelection::List(vec![i]),
                    projections: vec![Projection::Down],
                    kind: OperationKind::Subtract,
                    vector_file: PathBuf::from("v.json"),
                    vector_tensor: None,
                    alpha: 0.5,
                    axis: Axis::Cols,
                })
                .collect(),
        }
    }

    fn model(dir: &Path) -> PathBuf {
        let p = dir.join("model.safetensors");
        fs::write(&p, [0u8; 16]).unwrap();
        p
    }

    #[test]
    fn success_runs_in_order_flushes_commits_and_reports_stably() {
        let dir = tempfile::tempdir().unwrap();
        let file = model(dir.path());
        let mut backend = MockBackend::new(&file);
        let mut guard = BackupGuard::create(&file, None).unwrap();
        let report = apply_patch(
            &patch(&["first", "second"]),
            &mut backend,
            Some(&mut guard),
            false,
        )
        .unwrap();
        drop(guard); // committed: must not restore
        assert!(backend.flushed);
        assert_eq!(
            backend.calls,
            [
                "List([0]).Down.0",
                "List([0]).Down.1",
                "List([1]).Down.0",
                "List([1]).Down.1"
            ]
        );
        assert_eq!(fs::read(&file).unwrap()[..4], [0xFF; 4]);

        let json = serde_json::to_value(&report).unwrap();
        let keys: Vec<&str> = json
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            ["backup_path", "dry_run", "operations", "rolled_back"]
        );
        assert_eq!(json["operations"][0]["id"], "first");
        assert_eq!(
            json["operations"][1]["tensors"][0]["stats"]["elements_modified"],
            16
        );
        assert!(json["backup_path"].as_str().unwrap().contains(".bak-"));
        assert_eq!(json["dry_run"], false);
    }

    #[test]
    fn failure_on_second_op_restores_the_original() {
        let dir = tempfile::tempdir().unwrap();
        let file = model(dir.path());
        let original = fs::read(&file).unwrap();
        let mut backend = MockBackend::new(&file);
        backend.fail_on_call = Some(2); // first op (2 tensors) succeeds, second fails
        let mut guard = BackupGuard::create(&file, None).unwrap();
        let err =
            apply_patch(&patch(&["a", "b"]), &mut backend, Some(&mut guard), false).unwrap_err();
        assert!(
            matches!(&err, PatchError::ExecutionFailed { op_id, rolled_back: true, left_modified: false, .. } if op_id == "b"),
            "{err:?}"
        );
        assert_eq!(i32::from(&err), exit_code::EXECUTION_FAILED);
        assert!(!backend.flushed);
        assert_eq!(fs::read(&file).unwrap(), original);
    }

    #[test]
    fn nan_or_inf_triggers_rollback() {
        let dir = tempfile::tempdir().unwrap();
        let file = model(dir.path());
        let original = fs::read(&file).unwrap();
        let mut backend = MockBackend::new(&file);
        backend.nan_on_call = Some(1);
        let mut guard = BackupGuard::create(&file, None).unwrap();
        let err = apply_patch(&patch(&["a"]), &mut backend, Some(&mut guard), false).unwrap_err();
        assert!(
            matches!(&err, PatchError::ExecutionFailed { rolled_back: true, cause, .. } if cause.contains("3 valeurs NaN/Inf")),
            "{err:?}"
        );
        assert_eq!(fs::read(&file).unwrap(), original);
    }

    #[test]
    fn failed_rollback_is_critical() {
        let dir = tempfile::tempdir().unwrap();
        let file = model(dir.path());
        let mut backend = MockBackend::new(&file);
        backend.fail_on_call = Some(1);
        let mut guard = BackupGuard::create(&file, None).unwrap();
        fs::remove_file(guard.backup_path()).unwrap();
        let err = apply_patch(&patch(&["a"]), &mut backend, Some(&mut guard), false).unwrap_err();
        assert!(matches!(err, PatchError::RollbackFailed { .. }), "{err:?}");
        assert_eq!(i32::from(&err), exit_code::CRITICAL);
        guard.commit(); // the backup is gone, skip the drop-time retry
    }

    #[test]
    fn failure_without_backup_is_reported_as_left_modified() {
        let dir = tempfile::tempdir().unwrap();
        let file = model(dir.path());
        let mut backend = MockBackend::new(&file);
        backend.fail_on_call = Some(1);
        let err = apply_patch(&patch(&["a"]), &mut backend, None, false).unwrap_err();
        assert!(matches!(
            err,
            PatchError::ExecutionFailed {
                left_modified: true,
                ..
            }
        ));
        assert_eq!(i32::from(&err), exit_code::CRITICAL);
    }

    #[test]
    fn dry_run_never_shifts() {
        let dir = tempfile::tempdir().unwrap();
        let file = model(dir.path());
        let original = fs::read(&file).unwrap();
        let mut backend = MockBackend::new(&file);
        let report = apply_patch(&patch(&["a", "b"]), &mut backend, None, true).unwrap();
        assert!(backend.calls.is_empty());
        assert!(!backend.flushed);
        assert!(report.dry_run);
        assert_eq!(report.operations[1].tensors.len(), 2);
        assert_eq!(report.operations[0].tensors[0].stats, None);
        assert_eq!(fs::read(&file).unwrap(), original);
    }

    #[test]
    fn operation_without_target_fails_before_any_write() {
        let dir = tempfile::tempdir().unwrap();
        let file = model(dir.path());
        let mut backend = MockBackend::new(&file);
        backend.targets_per_op = 0;
        let err = apply_patch(&patch(&["a"]), &mut backend, None, false).unwrap_err();
        assert!(matches!(
            err,
            PatchError::ExecutionFailed {
                rolled_back: false,
                left_modified: false,
                ..
            }
        ));
        assert!(backend.calls.is_empty());
    }

    /// project_out through a backend that does not support it, and one that produces NaN.
    struct ProjectBackend {
        file: PathBuf,
        nan: bool,
        loaded: Vec<Option<String>>,
    }

    impl PatchBackend for ProjectBackend {
        fn resolve_targets(&self, _: &LayerSelection, _: &[Projection]) -> Result<Vec<String>> {
            Ok(vec!["t0".into(), "t1".into()])
        }
        fn load_vector(&self, _: &Path, tensor: Option<&str>) -> Result<Vec<f32>> {
            Ok(match tensor {
                Some("b") => vec![0.5; 4],
                _ => vec![1.0; 4],
            })
        }
        fn shift_tensor(
            &mut self,
            _: &str,
            _: &[f32],
            _: f32,
            _: OperationKind,
            _: Axis,
        ) -> Result<OpStats> {
            Err(PatchError::Backend("shift inattendu".into()))
        }
        fn project_out_tensor(
            &mut self,
            name: &str,
            a: &[f32],
            b: &[f32],
            _: f32,
        ) -> Result<OpStats> {
            assert_eq!((a[0], b[0]), (1.0, 0.5), "a and b from their own tensors");
            self.loaded.push(Some(name.to_owned()));
            fs::write(&self.file, [0xFF; 16]).unwrap();
            Ok(OpStats {
                max_abs_delta: 0.1,
                elements_modified: 4,
                nan_or_inf_produced: u64::from(self.nan),
            })
        }
        fn flush(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn project_patch() -> ValidatedPatch {
        let mut p = patch(&["erase"]);
        p.operations[0].kind = OperationKind::ProjectOut;
        p.operations[0].vector_tensor = Some(VectorTensor::Pair {
            a: "a".into(),
            b: "b".into(),
        });
        p
    }

    #[test]
    fn project_out_runs_through_its_own_backend_method_and_rolls_back_on_nan() {
        let dir = tempfile::tempdir().unwrap();
        let file = model(dir.path());
        let mut ok = ProjectBackend {
            file: file.clone(),
            nan: false,
            loaded: Vec::new(),
        };
        let report = apply_patch(&project_patch(), &mut ok, None, false).unwrap();
        assert_eq!(report.operations[0].tensors.len(), 2);
        assert_eq!(ok.loaded.len(), 2);

        fs::write(&file, [0u8; 16]).unwrap();
        let mut bad = ProjectBackend {
            file: file.clone(),
            nan: true,
            loaded: Vec::new(),
        };
        let mut guard = BackupGuard::create(&file, None).unwrap();
        let err = apply_patch(&project_patch(), &mut bad, Some(&mut guard), false).unwrap_err();
        assert!(
            matches!(
                err,
                PatchError::ExecutionFailed {
                    rolled_back: true,
                    ..
                }
            ),
            "{err:?}"
        );
        assert_eq!(fs::read(&file).unwrap(), [0u8; 16], "restored");

        // The prompt 2 mock has no project_out: refused at the first tensor, rolled back.
        let mut legacy = MockBackend::new(&file);
        let mut guard = BackupGuard::create(&file, None).unwrap();
        let err = apply_patch(&project_patch(), &mut legacy, Some(&mut guard), false).unwrap_err();
        assert!(
            err.to_string().contains("project_out non supporté"),
            "{err}"
        );
        assert!(matches!(
            err,
            PatchError::ExecutionFailed {
                rolled_back: true,
                ..
            }
        ));
    }

    /// `[k, len]` tensors by name, and a rank-k `project_out` that records what it gets.
    struct Ranked {
        tensors: Vec<(&'static str, Option<usize>, Vec<f32>)>,
        calls: Vec<(String, usize, usize)>,
        nan: bool,
    }

    impl Ranked {
        fn new(tensors: Vec<(&'static str, Option<usize>, Vec<f32>)>) -> Self {
            Self {
                tensors,
                calls: Vec::new(),
                nan: false,
            }
        }
    }

    impl PatchBackend for Ranked {
        fn resolve_targets(&self, _: &LayerSelection, _: &[Projection]) -> Result<Vec<String>> {
            Ok(vec!["t0".into()])
        }
        fn load_vector(&self, _: &Path, _: Option<&str>) -> Result<Vec<f32>> {
            Err(PatchError::Backend("vecteur 1D inattendu".into()))
        }
        fn load_directions(
            &self,
            _: &Path,
            tensor: Option<&str>,
        ) -> Result<(Option<usize>, Vec<f32>)> {
            self.tensors
                .iter()
                .find(|t| Some(t.0) == tensor)
                .map(|t| (t.1, t.2.clone()))
                .ok_or_else(|| PatchError::Backend(format!("tenseur absent : {tensor:?}")))
        }
        fn shift_tensor(
            &mut self,
            _: &str,
            _: &[f32],
            _: f32,
            _: OperationKind,
            _: Axis,
        ) -> Result<OpStats> {
            Err(PatchError::Backend("shift inattendu".into()))
        }
        fn project_out_subspace(
            &mut self,
            name: &str,
            a: &[f32],
            b: &[f32],
            k: usize,
            _: f32,
        ) -> Result<OpStats> {
            assert_eq!(a.len(), b.len());
            self.calls.push((name.to_owned(), k, a.len()));
            Ok(OpStats {
                max_abs_delta: 0.1,
                elements_modified: 4,
                nan_or_inf_produced: u64::from(self.nan),
            })
        }
        fn flush(&mut self) -> Result<()> {
            Ok(())
        }
    }

    fn subspace_patch(
        alpha: f32,
        rank: Option<usize>,
        allow_non_projector: bool,
    ) -> ValidatedPatch {
        let mut p = project_patch();
        p.operations[0].alpha = alpha;
        p.operations[0].vector_tensor = Some(VectorTensor::Subspace {
            a: "a".into(),
            b: "b".into(),
            rank,
            allow_non_projector,
        });
        p
    }

    /// Two orthonormal directions of length 3, `[2, 3]`.
    const E12: [f32; 6] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

    fn orthonormal_pair() -> Vec<(&'static str, Option<usize>, Vec<f32>)> {
        vec![("a", Some(2), E12.to_vec()), ("b", Some(2), E12.to_vec())]
    }

    fn untouched_cause(err: PatchError) -> String {
        match err {
            PatchError::ExecutionFailed {
                cause,
                rolled_back: false,
                left_modified: false,
                ..
            } => cause,
            other => panic!("expected a preflight failure, got {other:?}"),
        }
    }

    #[test]
    fn rank_k_project_out_goes_to_the_backend_in_one_call_per_tensor() {
        // A plain a/b pair (no new field), the declared rank, and a dry run.
        for (patch, dry_run) in [
            (project_patch(), false),
            (subspace_patch(1.0, Some(2), false), false),
            (subspace_patch(1.0, None, false), true),
        ] {
            let mut backend = Ranked::new(orthonormal_pair());
            let report = apply_patch(&patch, &mut backend, None, dry_run).unwrap();
            let op = &report.operations[0];
            assert_eq!((op.vector_len, op.rank), (3, Some(2)));
            assert!(report.warnings.is_empty());
            match dry_run {
                true => assert!(backend.calls.is_empty() && op.tensors[0].stats.is_none()),
                false => assert_eq!(backend.calls, [("t0".to_owned(), 2, 6)]),
            }
            let json = serde_json::to_value(&report).unwrap();
            assert_eq!(json["operations"][0]["rank"], 2);
            assert!(json.get("warnings").is_none(), "absent when empty");
        }
    }

    #[test]
    fn the_rank_is_checked_against_the_tensors_before_any_write() {
        let run = |patch: &ValidatedPatch, tensors, max_rank: usize| {
            let mut backend = Ranked::new(tensors);
            let options = ApplyOptions { max_rank };
            let result = apply_patch_with(patch, &mut backend, None, false, &options);
            assert!(
                result.is_ok() || backend.calls.is_empty(),
                "nothing written"
            );
            result
        };
        // The declared rank is not the one of the tensors.
        let declared = subspace_patch(1.0, Some(3), false);
        let cause = untouched_cause(run(&declared, orthonormal_pair(), 64).unwrap_err());
        assert!(
            cause.contains("rank = 3") && cause.contains("2 direction"),
            "{cause}"
        );
        // Above the limit, and exactly at it.
        let cause = untouched_cause(run(&project_patch(), orthonormal_pair(), 1).unwrap_err());
        assert!(cause.contains("rang 2 > plafond 1"), "{cause}");
        assert!(run(&project_patch(), orthonormal_pair(), 2).is_ok());
        // a and b of different ranks, although they hold as many values.
        let mixed = vec![("a", Some(2), E12.to_vec()), ("b", None, E12.to_vec())];
        let cause = untouched_cause(run(&project_patch(), mixed, 64).unwrap_err());
        assert!(cause.contains("rangs différents"), "{cause}");
        let short = vec![
            ("a", Some(2), E12.to_vec()),
            ("b", Some(1), E12[..3].to_vec()),
        ];
        let cause = untouched_cause(run(&project_patch(), short, 64).unwrap_err());
        assert!(cause.contains("tailles différentes"), "{cause}");
    }

    /// `Ranked` without its rank-k kernel: only the trait defaults.
    struct OneDirection(Ranked);

    impl PatchBackend for OneDirection {
        fn resolve_targets(&self, l: &LayerSelection, p: &[Projection]) -> Result<Vec<String>> {
            self.0.resolve_targets(l, p)
        }
        fn load_vector(&self, f: &Path, t: Option<&str>) -> Result<Vec<f32>> {
            self.0.load_vector(f, t)
        }
        fn load_directions(&self, f: &Path, t: Option<&str>) -> Result<(Option<usize>, Vec<f32>)> {
            self.0.load_directions(f, t)
        }
        fn shift_tensor(
            &mut self,
            n: &str,
            d: &[f32],
            a: f32,
            o: OperationKind,
            x: Axis,
        ) -> Result<OpStats> {
            self.0.shift_tensor(n, d, a, o, x)
        }
        fn flush(&mut self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn a_backend_without_a_rank_k_kernel_refuses_more_than_one_direction() {
        let mut one = OneDirection(Ranked::new(orthonormal_pair()));
        let err = apply_patch(&project_patch(), &mut one, None, false).unwrap_err();
        assert!(err.to_string().contains("rang 2 non supporté"), "{err}");
    }

    #[test]
    fn full_strength_needs_a_projector_unless_explicitly_allowed() {
        // a = 2 b: BᵀA = 2 I, ‖BᵀA − I‖ = √2.
        let doubled: Vec<f32> = E12.iter().map(|x| 2.0 * x).collect();
        let run = |patch: &ValidatedPatch| {
            let tensors = vec![
                ("a", Some(2), doubled.clone()),
                ("b", Some(2), E12.to_vec()),
            ];
            let mut backend = Ranked::new(tensors);
            apply_patch(patch, &mut backend, None, false).map(|report| (report, backend.calls))
        };
        let cause = untouched_cause(run(&subspace_patch(1.0, None, false)).unwrap_err());
        assert!(
            cause.contains("projecteur") && cause.contains("allow_non_projector"),
            "{cause}"
        );
        assert!(cause.contains("1.414e0"), "the measured error: {cause}");
        // Allowed on request, with a warning that names the operation.
        let (report, calls) = run(&subspace_patch(1.0, None, true)).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(report.warnings.len(), 1);
        assert!(report.warnings[0].contains("erase") && report.warnings[0].contains("projecteur"));
        // Below full strength nothing claims to be a projection: no check, no warning.
        let (report, _) = run(&subspace_patch(0.5, None, false)).unwrap();
        assert!(report.warnings.is_empty());
        // The 1D form was never checked: it still applies, now with the warning.
        let one_d = vec![
            ("a", None, vec![2.0, 0.0, 0.0]),
            ("b", None, vec![1.0, 0.0, 0.0]),
        ];
        let mut backend = Ranked::new(one_d);
        let patch = subspace_patch(1.0, None, false);
        let report = apply_patch(&patch, &mut backend, None, false).unwrap();
        assert_eq!((report.operations[0].rank, backend.calls[0].1), (None, 1));
        assert_eq!(report.warnings.len(), 1);
    }

    #[test]
    fn rank_k_nan_or_inf_rolls_back() {
        let dir = tempfile::tempdir().unwrap();
        let file = model(dir.path());
        let mut backend = Ranked::new(orthonormal_pair());
        backend.nan = true;
        let mut guard = BackupGuard::create(&file, None).unwrap();
        let err = apply_patch(&project_patch(), &mut backend, Some(&mut guard), false).unwrap_err();
        assert!(
            matches!(
                err,
                PatchError::ExecutionFailed {
                    rolled_back: true,
                    ..
                }
            ),
            "{err:?}"
        );
    }
}
