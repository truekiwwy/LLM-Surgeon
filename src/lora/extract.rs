//! `surgeon lora extract` (frozen API, body by the `lora` agent, wave 2).

use std::sync::atomic::AtomicBool;

use serde::Serialize;

use super::cli_args::LoraExtractArgs;
use crate::app::{AppDeps, CoreResult, ProgressSink};
use crate::surgery::error::{Result, SurgeryError};
use crate::surgery::events::SurgeryObserver;

/// Serialized under `data.lora` of the envelope.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ExtractReport {
    pub out: String,
    pub config: String,
    pub rank: usize,
    pub dtype: String,
    pub tensors: usize,
    pub mean_rel_error: f64,
    pub warnings: Vec<String>,
}

pub fn run_extract(
    _args: &LoraExtractArgs,
    _obs: &mut dyn SurgeryObserver,
    _cancel: &AtomicBool,
) -> Result<ExtractReport> {
    Err(SurgeryError::NotImplemented("lora extract"))
}

pub fn core_extract(
    _args: &LoraExtractArgs,
    _deps: &AppDeps,
    _sink: &mut dyn ProgressSink,
) -> CoreResult {
    Err(SurgeryError::NotImplemented("lora extract").into())
}
