//! `surgeon apply --patch x.surgery` (frozen API, body by the `lowrank` agent).

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde::Serialize;

use super::cli_args::VerifyBase;
use super::error::{Result, SurgeryError};
use super::events::SurgeryObserver;
use crate::app::{AppDeps, Outcome, ProgressSink};
use crate::cli::ApplyArgs;
use crate::patch::RollbackHandle;

#[derive(Debug, Clone)]
pub struct SurgeryApplyRequest {
    /// Shards of the target model (one file for a single SafeTensors).
    pub model_shards: Vec<PathBuf>,
    pub patch: PathBuf,
    pub scale: f64,
    pub verify_base: VerifyBase,
    pub dry_run: bool,
    /// Raised by Ctrl-C or an MCP cancellation (`AppDeps::interrupted`).
    pub cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorApplyReport {
    pub name: String,
    pub kind: String,
    pub rank: usize,
    pub elements_modified: u64,
    pub max_abs_delta: f32,
    pub base_hash_ok: Option<bool>,
}

/// Serialized under `data.surgery` of the envelope.
#[derive(Debug, Clone, Serialize, Default)]
pub struct SurgeryApplyReport {
    pub patch: String,
    pub scale: f64,
    pub dry_run: bool,
    pub tensors: Vec<TensorApplyReport>,
    pub warnings: Vec<String>,
}

/// Verifies the patch (global hash first), checks every tensor (presence, shape, dtype,
/// base hash per `verify_base`), then writes. Any error, NaN/Inf or cancellation calls
/// `guard.rollback()` before returning; success calls `guard.commit()`.
pub fn run_apply(
    _req: &SurgeryApplyRequest,
    _guard: &mut dyn RollbackHandle,
    _obs: &mut dyn SurgeryObserver,
) -> Result<SurgeryApplyReport> {
    Err(SurgeryError::NotImplemented("run_apply"))
}

/// CLI entry called by `app::apply` before `load_patch`: `Ok(None)` when `--patch` is not
/// a `.surgery` file (detected by magic), so the `patch.json` path runs unchanged. Builds
/// the backups of the touched shards, runs [`run_apply`], and shapes the `Outcome`
/// (`data.surgery`, `data.backups`).
pub fn try_apply(
    _args: &ApplyArgs,
    _deps: &AppDeps,
    _sink: &mut dyn ProgressSink,
) -> std::result::Result<Option<Outcome>, Outcome> {
    Ok(None)
}
