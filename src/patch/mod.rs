//! `patch.json` pipeline: parse, validate, back up, apply through an abstract backend.
//!
//! No dependency on `engine`, `hardware` or `eval`: the integration implements
//! [`PatchBackend`] on top of the engine and converts its errors with
//! [`PatchError::Backend`]. One exception, [`autotune`], which drives the extraction model
//! to produce a `PatchFile` (validated by `validate_patch` before it is written).
//!
//! Call graph, extracted with graphify (AST) and checked by hand. Module edges form a DAG:
//!
//! ```text
//! modules:  schema -> error ; validate -> schema, error ; executor -> validate, schema, error
//!           backup -> executor (RollbackHandle), error
//!
//! load_patch -> parse_patch -> serde_json (deny_unknown_fields)
//! validate_patch -> LayersField::to_selection -> LayerSelection::from_str
//! apply_patch -> PatchBackend::resolve_targets, load_vector      (preflight, no write)
//!             -> PatchBackend::shift_tensor (in order)           (unless dry_run)
//!             -> abort -> RollbackHandle::rollback               (error or NaN/Inf)
//!             -> PatchBackend::flush -> RollbackHandle::commit
//! BackupGuard::create   -> sha256_file, create_unique, copy, sha256_file (verify)
//! BackupGuard::rollback -> sha256_file(backup), restore (rename | in-place), sha256_file
//! BackupGuard::drop     -> rollback (state Created only)
//! BackupSet::{create_many, rollback_all, commit_all} -> BackupGuard::*
//! untie::plan -> safetensors headers, config.json            (read only)
//! UntiePlan::execute -> append_copy (lm_head.weight), write_json (config, index)
//! ```
//!
//! Guarantee: whatever happens in `apply_patch`, the original is either intact, restored
//! (`rolled_back: true`), or the error says otherwise (`RollbackFailed`, or
//! `left_modified: true` when no backup was taken). See [`exit_code`].

pub mod autotune;
mod backup;
mod error;
mod executor;
pub mod gguf;
mod schema;
pub mod untie;
mod validate;

pub use backup::{BackupGuard, BackupSet};
pub use error::{exit_code, PatchError, Result, ValidationIssue};
pub use executor::{
    apply_patch, ApplyReport, OpStats, OperationReport, PatchBackend, RollbackHandle, TensorReport,
};
pub use schema::{
    load_patch, parse_patch, Axis, LayerRange, LayerSelection, LayersField, Operation,
    OperationKind, PatchFile, Projection,
};
pub use validate::{
    validate_patch, ValidatedOperation, ValidatedPatch, ValidationOptions, VectorTensor,
    MAX_ABS_ALPHA, SUPPORTED_SCHEMA_VERSION, WARN_ABS_ALPHA,
};
