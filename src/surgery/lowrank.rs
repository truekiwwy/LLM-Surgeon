//! In-place `W <- W + U_scaled V` on a row-major tensor (frozen API, body by the `lowrank`
//! agent). Shared kernel of `.surgery` apply and `lora merge`.

use std::sync::atomic::AtomicBool;

use serde::Serialize;

use super::error::{Result, SurgeryError};
use crate::engine::DType;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct LowRankStats {
    /// Elements whose stored bits changed.
    pub elements_modified: u64,
    /// Largest |new - old| over finite results, after rounding to the dtype.
    pub max_abs_delta: f32,
    /// Finite weights that became NaN or Inf (the caller rolls back when non-zero).
    pub nan_or_inf_produced: u64,
}

/// `bytes` is a `[rows, cols]` tensor of `dtype`; `u_scaled` is [rows x r] with the scale
/// and singular values already folded in, `v` is [r x cols]. For row `i`:
/// `delta[i, :] = sum_k u_scaled[i, k] * v[k, :]` accumulated in f32 in fixed `k` order,
/// then `W[i, j] <- round_to_nearest_even_dtype(W[i, j] + delta[i, j])`. Rayon over row
/// blocks; identical bytes whatever the thread count. Nothing is written on `Err`
/// (validation) or when `cancel` is raised before the first block.
#[allow(clippy::too_many_arguments)]
pub fn apply_lowrank_rows(
    _bytes: &mut [u8],
    _dtype: DType,
    _rows: usize,
    _cols: usize,
    _u_scaled: &[f32],
    _v: &[f32],
    _r: usize,
    _cancel: &AtomicBool,
) -> Result<LowRankStats> {
    Err(SurgeryError::NotImplemented("apply_lowrank_rows"))
}
