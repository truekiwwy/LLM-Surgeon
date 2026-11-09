//! `surgeon diff` (frozen API, body by the `diff` agent, wave 2).

use std::sync::atomic::AtomicBool;

use serde::Serialize;

use super::cli_args::DiffArgs;
use super::error::{Result, SurgeryError};
use super::events::SurgeryObserver;
use crate::app::{AppDeps, CoreResult, ProgressSink};

#[derive(Debug, Clone, Serialize)]
pub struct DiffTensorReport {
    pub name: String,
    pub kind: String,
    pub rank: usize,
    pub rel_error: f64,
    pub energy: f64,
}

/// Serialized under `data.surgery` of the envelope.
#[derive(Debug, Clone, Serialize, Default)]
pub struct DiffReport {
    pub out: String,
    pub bytes: u64,
    pub sha256: String,
    pub rank: usize,
    pub factor_dtype: String,
    pub tensors: Vec<DiffTensorReport>,
    pub skipped_identical: usize,
    pub skipped_filtered: usize,
    pub warnings: Vec<String>,
}

/// Engine: paths and options are already validated by the caller (CLI or MCP sandbox).
pub fn run_diff(
    _args: &DiffArgs,
    _obs: &mut dyn SurgeryObserver,
    _cancel: &AtomicBool,
) -> Result<DiffReport> {
    Err(SurgeryError::NotImplemented("diff"))
}

/// Shared by the CLI and the `surgeon_diff` MCP tool.
pub fn core_diff(_args: &DiffArgs, _deps: &AppDeps, _sink: &mut dyn ProgressSink) -> CoreResult {
    Err(SurgeryError::NotImplemented("diff").into())
}
