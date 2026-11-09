//! Deterministic `.surgery` writer (frozen API, body written by the `format` agent).

use std::path::Path;

use serde::Serialize;

use super::error::{Result, SurgeryError};
use super::format::{Compression, FactorDtype, LowRankFactors};

/// Header fields known before the first tensor.
#[derive(Debug, Clone)]
pub struct HeaderInit {
    pub created_with: String,
    pub base_name: String,
    pub base_tensor_count: usize,
    pub tuned_name: String,
    pub rank: usize,
    pub factor_dtype: FactorDtype,
    pub compression: Compression,
    /// RFC 3339 string; `None` keeps the file reproducible.
    pub timestamp: Option<String>,
}

/// Per-tensor metadata supplied by the caller.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub target_dtype: String,
    pub base_sha256: String,
    pub rel_error: f64,
    pub energy: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WriteSummary {
    pub bytes: u64,
    pub sha256: String,
    pub tensors: usize,
}

pub struct SurgeryWriter {
    _private: (),
}

impl SurgeryWriter {
    /// Fails when `path` exists and `overwrite` is false. Nothing is visible at `path`
    /// until `finish` succeeds (temporary file next to it, then rename).
    pub fn create(_path: &Path, _init: HeaderInit, _overwrite: bool) -> Result<Self> {
        Err(SurgeryError::NotImplemented("SurgeryWriter::create"))
    }

    /// `rank` is the stored rank of `factors` (`s.len()`). Returns
    /// `SurgeryError::FactorOverflow` when a factor does not fit `factor_dtype`.
    pub fn add_lowrank(&mut self, _meta: TensorMeta, _factors: &LowRankFactors) -> Result<()> {
        Err(SurgeryError::NotImplemented("SurgeryWriter::add_lowrank"))
    }

    /// Exact f32 delta of a 1D tensor (or of a tensor kept whole by `--on-high-error full`).
    pub fn add_full_delta(&mut self, _meta: TensorMeta, _delta: &[f32]) -> Result<()> {
        Err(SurgeryError::NotImplemented(
            "SurgeryWriter::add_full_delta",
        ))
    }

    pub fn finish(self) -> Result<WriteSummary> {
        Err(SurgeryError::NotImplemented("SurgeryWriter::finish"))
    }
}
