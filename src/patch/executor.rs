use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::error::{PatchError, Result};
use super::schema::{Axis, LayerSelection, OperationKind, Projection};
use super::validate::{ValidatedPatch, VectorTensor};

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
    pub vector_len: usize,
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
        let vector = backend
            .load_vector(
                &op.vector_file,
                op.vector_tensor.as_ref().map(VectorTensor::direction),
            )
            .map_err(|e| untouched(e.to_string()))?;
        let read = match (&op.kind, &op.vector_tensor) {
            (OperationKind::ProjectOut, Some(VectorTensor::Pair { b, .. })) => {
                let b = backend
                    .load_vector(&op.vector_file, Some(b))
                    .map_err(|e| untouched(e.to_string()))?;
                if b.len() != vector.len() {
                    return Err(untouched(format!(
                        "a et b de tailles différentes ({} et {})",
                        vector.len(),
                        b.len()
                    )));
                }
                Some(b)
            }
            (OperationKind::ProjectOut, _) => {
                return Err(untouched("project_out sans vecteur b".into()));
            }
            _ => None,
        };
        plans.push((op, targets, vector, read));
    }

    let mut report = ApplyReport {
        operations: Vec::with_capacity(plans.len()),
        backup_path: guard.as_ref().and_then(|g| g.backup_path()),
        dry_run,
        rolled_back: false,
    };

    if dry_run {
        report.operations = plans
            .into_iter()
            .map(|(op, targets, vector, _)| OperationReport {
                id: op.id.clone(),
                vector_len: vector.len(),
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
                Some(b) => backend.project_out_tensor(&name, &vector, b, op.alpha),
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
            vector_len: vector.len(),
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
}
