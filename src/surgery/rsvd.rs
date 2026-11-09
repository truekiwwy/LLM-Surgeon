//! Randomized SVD (Halko, Martinsson, Tropp) of a matrix read by row blocks
//! (frozen API, body by the `rsvd` agent). Shared by `diff` and `lora extract`.

use std::sync::atomic::AtomicBool;

use super::error::{Result, SurgeryError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RsvdParams {
    pub rank: usize,
    pub oversample: usize,
    pub power_iters: usize,
    pub seed: u64,
}

impl Default for RsvdParams {
    fn default() -> Self {
        Self {
            rank: 16,
            oversample: 8,
            power_iters: 3,
            seed: 0,
        }
    }
}

/// `A ~ U diag(s) V`, `u` row-major [m x r], `v` row-major [r x n].
#[derive(Debug, Clone, PartialEq)]
pub struct Rsvd {
    pub m: usize,
    pub n: usize,
    pub r: usize,
    pub u: Vec<f32>,
    pub s: Vec<f32>,
    pub v: Vec<f32>,
    /// `||A - U S V||_F / ||A||_F`, exact (computed by blocks), 0 for a null matrix.
    pub rel_error: f64,
    /// `sum(s^2) / ||A||_F^2`.
    pub energy: f64,
}

/// A matrix that can be streamed by rows (e.g. `tuned - base` computed on the fly).
pub trait RowSource: Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// Fills `out` with consecutive rows starting at `start` (`out.len()` is a multiple of
    /// `cols`); returns the number of rows written.
    fn read_rows(&self, start: usize, out: &mut [f32]) -> Result<usize>;
}

/// Deterministic for a given `(A, params)`, whatever the rayon thread count.
pub fn rsvd(_a: &dyn RowSource, _p: &RsvdParams, _cancel: &AtomicBool) -> Result<Rsvd> {
    Err(SurgeryError::NotImplemented("rsvd"))
}
