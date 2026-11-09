//! `surgeon lora merge` (frozen API, body by the `lora` agent, wave 2).

use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde::Serialize;

use super::cli_args::LoraMergeArgs;
use crate::app::{AppDeps, CoreResult, ProgressSink};
use crate::patch::RollbackHandle;
use crate::surgery::error::{Result, SurgeryError};
use crate::surgery::events::SurgeryObserver;

#[derive(Debug, Clone)]
pub struct MergeRequest {
    pub model_shards: Vec<PathBuf>,
    pub lora: PathBuf,
    pub alpha: Option<f64>,
    pub adapter_config: Option<PathBuf>,
    pub scale: f64,
    pub dry_run: bool,
    pub cancel: Arc<AtomicBool>,
}

/// Serialized under `data.lora` of the envelope.
#[derive(Debug, Clone, Serialize, Default)]
pub struct MergeReport {
    pub rank: usize,
    pub lora_alpha: f64,
    /// Effective multiplier: scale * lora_alpha / r (or / sqrt(r) with rsLoRA).
    pub effective_scale: f64,
    pub tensors: usize,
    pub max_abs_delta: f32,
    pub dry_run: bool,
    pub warnings: Vec<String>,
}

/// Any error, NaN/Inf or cancellation calls `guard.rollback()`; success calls `commit()`.
pub fn run_merge(
    _req: &MergeRequest,
    _guard: &mut dyn RollbackHandle,
    _obs: &mut dyn SurgeryObserver,
) -> Result<MergeReport> {
    Err(SurgeryError::NotImplemented("lora merge"))
}

pub fn core_merge(
    _args: &LoraMergeArgs,
    _deps: &AppDeps,
    _sink: &mut dyn ProgressSink,
) -> CoreResult {
    Err(SurgeryError::NotImplemented("lora merge").into())
}
