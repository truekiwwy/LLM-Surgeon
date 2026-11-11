//! `project_out` on a block-quantized tensor: the SafeTensors eraser `x ↦ x − αA(Bᵀx)` on
//! the dequantized weights, then requantized, with a measured residual and an optional
//! refinement.
//!
//! Requantization adds noise, and the part of that noise read by `B` brings back some of
//! the erased direction. Per tensor, with `W_exact` the float result of the SafeTensors
//! engine on `deq(W)`:
//!
//! ```text
//! writer (ffn_down, attn_output)   R(X) = BᵀX               [k, cols]
//! writer_local (token_embd)        R(X) = X B               [rows, k], row by row
//! residual_fraction         = ‖R(deq W')‖ / ‖R(W)‖           (≈ 1 − α for a projector)
//! requant_residual_fraction = ‖R(deq W' − W_exact)‖ / ‖R(W)‖ (the requantization part)
//! ```
//!
//! Refinement (`--refine-passes N`): pass 1 requantizes `W_exact`; pass n + 1 requantizes
//! `W_exact − A·C` with `C += R(deq W'_n − W_exact)` (an integral correction: the error read
//! by `B` is pushed back through `A`, so `Bᵀ(W_exact − A·C) = Bᵀ W_exact − C` when `BᵀA = I`).
//! Every pass starts again from the original bytes, so keep-scales stays on the original
//! grid. Passes stop once `requant_residual_fraction <= --residual-target`; the pass with
//! the smallest residual is the one written. The correction is far below a quantization
//! step: it only moves elements near a rounding boundary, so it lowers the residual
//! statistically, never to zero.
//!
//! Nothing is written while measuring: every pass is a read-only sweep (pass 1 of a
//! writer, `R(W)`, is one more), then one write pass recomputes the chosen bytes, which
//! are a pure function of the original row and the plan. Rows go in fixed blocks of 64,
//! partial sums added in block order (as `engine::project`), so every number is
//! bit-identical whatever the thread count or `--prefetch-chunk-mib`.

use rayon::prelude::*;

use super::mapping::Layout;
use super::quant::Requant;
use super::verify::{self, Fidelity, PatchedRow, RowError};
use crate::engine::{accumulate_r, local_deltas, row_deltas, SubspaceProjection};

/// Default of `--refine-passes`: one quantization, no refinement.
pub const DEFAULT_REFINE_PASSES: usize = 1;
pub const MAX_REFINE_PASSES: usize = 5;
/// Default of `--residual-target`: at most 5 % of `‖R(W)‖` brought back by the
/// requantization. Q8_0 usually meets it in one pass, 4-bit types often do not.
pub const DEFAULT_RESIDUAL_TARGET: f64 = 0.05;

const ROW_BLOCK: usize = 64;
/// Bytes of block accumulators alive at once.
const WAVE_BYTES: usize = 64 << 20;

/// One `project_out` on one tensor.
#[derive(Debug, Clone, Copy)]
pub struct Projector<'a> {
    pub p: SubspaceProjection<'a>,
    /// `token_embd`: directions along the columns, each row projected alone.
    pub local: bool,
    pub layout: Layout,
    pub mode: Requant,
}

/// What the measuring sweeps found and what the write pass replays.
#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    /// Writer: `R(W)` rounded to f32, `[k, cols]`. Empty for the embeddings.
    r: Vec<f32>,
    /// The chosen correction (writer `[k, cols]`, embeddings `[rows, k]`); empty: none.
    correction: Vec<f32>,
    pub fidelity: Fidelity,
    pub residual_fraction: Option<f64>,
    pub requant_residual_fraction: Option<f64>,
    /// `requant_residual_fraction` after each pass run.
    pub history: Vec<Option<f64>>,
}

impl Plan {
    pub fn passes(&self) -> usize {
        self.history.len()
    }
}

/// One sweep folded in block order.
#[derive(Debug, Clone, Default)]
struct Sweep {
    fidelity: Fidelity,
    /// `R(deq W' − W_exact)`, the shape of the correction.
    err: Vec<f64>,
    /// Writer only: `R(deq W')`.
    after: Vec<f64>,
    after_sq: f64,
    before_sq: f64,
}

impl Sweep {
    fn err_sq(&self) -> f64 {
        self.err.iter().map(|e| e * e).sum()
    }
}

fn f64_dot(x: &[f32], y: &[f32]) -> f64 {
    x.iter()
        .zip(y)
        .map(|(a, b)| f64::from(*a) * f64::from(*b))
        .sum()
}

impl Projector<'_> {
    fn k(&self) -> usize {
        self.p.k
    }

    /// Rows of 64 per block, blocks per wave so that `per_block` f64 per block fit.
    fn wave_bytes(&self, per_block: usize) -> usize {
        let row_bytes = self.layout.row_bytes();
        let blocks = (WAVE_BYTES / (per_block * 8).max(1)).max(1);
        blocks * ROW_BLOCK * row_bytes
    }

    /// Patches row `i` (its original bytes in `row`) with the correction `corr`.
    fn row(
        &self,
        i: usize,
        row: &mut [u8],
        r: &[f32],
        corr: &[f32],
    ) -> Result<PatchedRow, RowError> {
        let p = &self.p;
        let deltas = |before: &[f32], out: &mut [f32]| match self.local {
            true => local_deltas(p, before, out),
            false => row_deltas(p, i, r, out),
        };
        let correct = |target: &mut [f32]| {
            if corr.is_empty() {
                return;
            }
            let mut c = vec![0.0f32; target.len()];
            match self.local {
                // −Σ_t (1 · a[t][i]) C[t][j]: the eraser at full strength on C.
                false => row_deltas(&SubspaceProjection { alpha: 1.0, ..*p }, i, corr, &mut c),
                // −Σ_t C[i][t] a_t[j], in the order of t.
                true => {
                    let k = self.k();
                    for (t, a_t) in p.a.chunks_exact(target.len()).enumerate() {
                        let coef = corr[i * k + t];
                        c.iter_mut().zip(a_t).for_each(|(c, a)| *c -= coef * a);
                    }
                }
            }
            target.iter_mut().zip(&c).for_each(|(x, c)| *x += c);
        };
        verify::patch_row_with(&self.layout, self.mode, row, deltas, correct)
    }

    /// `R(W) = BᵀW` of a writer in f64, `[k, cols]`.
    fn read_r(&self, bytes: &[u8], cancelled: &dyn Fn() -> bool) -> Result<Vec<f64>, String> {
        let (k, cols, row_bytes) = (self.k(), self.layout.cols, self.layout.row_bytes());
        let mut r = vec![0.0f64; k * cols];
        let wave = self.wave_bytes(k * cols);
        for (w, chunk) in bytes.chunks(wave).enumerate() {
            if cancelled() {
                return Err(INTERRUPTED.into());
            }
            let first = w * wave / row_bytes;
            let partials: Vec<Vec<f64>> = chunk
                .par_chunks(ROW_BLOCK * row_bytes)
                .enumerate()
                .map(|(blk, rows)| {
                    let mut partial = vec![0.0f64; k * cols];
                    for (o, row) in rows.chunks_exact(row_bytes).enumerate() {
                        let x = verify::decode_checked(&self.layout, row)?;
                        accumulate_r(&self.p, first + blk * ROW_BLOCK + o, &x, cols, &mut partial);
                    }
                    Ok(partial)
                })
                .collect::<Result<_, RowError>>()
                .map_err(|e| e.to_string())?;
            for partial in &partials {
                r.iter_mut().zip(partial).for_each(|(s, x)| *s += x);
            }
        }
        Ok(r)
    }

    /// One read-only pass: every row requantized in a copy, the sums folded in block order.
    fn sweep(
        &self,
        bytes: &[u8],
        r: &[f32],
        corr: &[f32],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Sweep, String> {
        let (k, cols, row_bytes) = (self.k(), self.layout.cols, self.layout.row_bytes());
        let len = self.p.direction_len();
        let per_block = match self.local {
            true => ROW_BLOCK * k,
            false => 2 * k * cols,
        };
        let wave = self.wave_bytes(per_block);
        let mut total = Sweep {
            err: match self.local {
                true => Vec::with_capacity(self.layout.rows * k),
                false => vec![0.0; k * cols],
            },
            after: vec![0.0; if self.local { 0 } else { k * cols }],
            ..Sweep::default()
        };
        for (w, chunk) in bytes.chunks(wave).enumerate() {
            if cancelled() {
                return Err(INTERRUPTED.into());
            }
            let first = w * wave / row_bytes;
            let parts: Vec<Sweep> = chunk
                .par_chunks(ROW_BLOCK * row_bytes)
                .enumerate()
                .map(|(blk, rows)| {
                    let mut part = Sweep {
                        err: vec![0.0; if self.local { 0 } else { k * cols }],
                        after: vec![0.0; if self.local { 0 } else { k * cols }],
                        ..Sweep::default()
                    };
                    for (o, row) in rows.chunks_exact(row_bytes).enumerate() {
                        let i = first + blk * ROW_BLOCK + o;
                        let mut copy = row.to_vec();
                        let done = self.row(i, &mut copy, r, corr)?;
                        part.fidelity.merge(&done.fidelity);
                        let diff: Vec<f64> = done
                            .after
                            .iter()
                            .zip(&done.exact)
                            .map(|(a, x)| f64::from(*a) - f64::from(*x))
                            .collect();
                        if self.local {
                            for b in self.p.b.chunks_exact(len) {
                                let read_after = f64_dot(&done.after, b);
                                let read_before = f64_dot(&done.before, b);
                                part.after_sq += read_after * read_after;
                                part.before_sq += read_before * read_before;
                                part.err
                                    .push(diff.iter().zip(b).map(|(d, b)| d * f64::from(*b)).sum());
                            }
                        } else {
                            accumulate_r(&self.p, i, &done.after, cols, &mut part.after);
                            for t in 0..k {
                                let bt = f64::from(self.p.b[t * len + i]);
                                let e = &mut part.err[t * cols..(t + 1) * cols];
                                e.iter_mut().zip(&diff).for_each(|(s, d)| *s += bt * d);
                            }
                        }
                    }
                    Ok(part)
                })
                .collect::<Result<_, RowError>>()
                .map_err(|e| e.to_string())?;
            for part in parts {
                total.fidelity.merge(&part.fidelity);
                total.after_sq += part.after_sq;
                total.before_sq += part.before_sq;
                match self.local {
                    true => total.err.extend(part.err),
                    false => {
                        total
                            .err
                            .iter_mut()
                            .zip(&part.err)
                            .for_each(|(s, x)| *s += x);
                        total
                            .after
                            .iter_mut()
                            .zip(&part.after)
                            .for_each(|(s, x)| *s += x);
                    }
                }
            }
        }
        if !self.local {
            total.after_sq = total.after.iter().map(|x| x * x).sum();
        }
        Ok(total)
    }

    /// Measures up to `passes` quantizations of the tensor (`bytes`, its original data)
    /// and keeps the one with the smallest requantization residual. Writes nothing.
    pub fn plan(
        &self,
        bytes: &[u8],
        passes: usize,
        target: f64,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Plan, String> {
        let (r64, r_sq) = match self.local {
            true => (Vec::new(), None),
            false => {
                let r = self.read_r(bytes, cancelled)?;
                let sq = r.iter().map(|x| x * x).sum::<f64>();
                (r, Some(sq))
            }
        };
        let r: Vec<f32> = r64.iter().map(|&x| x as f32).collect();
        let mut correction = vec![
            0.0f64;
            match self.local {
                true => self.layout.rows * self.k(),
                false => self.k() * self.layout.cols,
            }
        ];
        let mut best: Option<Plan> = None;
        let mut history = Vec::new();
        for pass in 1..=passes.max(1) {
            let corr: Vec<f32> = match pass {
                1 => Vec::new(),
                _ => correction.iter().map(|&x| x as f32).collect(),
            };
            let sweep = self.sweep(bytes, &r, &corr, cancelled)?;
            let norm = r_sq.unwrap_or(sweep.before_sq).sqrt();
            let fraction = |sq: f64| (norm > 0.0).then(|| sq.sqrt() / norm);
            let requant = fraction(sweep.err_sq());
            history.push(requant);
            let better = match (&best, requant) {
                (None, _) => true,
                (Some(b), Some(x)) => b.requant_residual_fraction.is_some_and(|y| x < y),
                (Some(_), None) => false,
            };
            if better {
                best = Some(Plan {
                    r: r.clone(),
                    correction: corr,
                    fidelity: sweep.fidelity,
                    residual_fraction: fraction(sweep.after_sq),
                    requant_residual_fraction: requant,
                    history: Vec::new(),
                });
            }
            if requant.is_none_or(|x| x <= target) {
                break;
            }
            correction
                .iter_mut()
                .zip(&sweep.err)
                .for_each(|(c, e)| *c += e);
        }
        // `passes.max(1)`: the loop ran at least once.
        best.map(|plan| Plan { history, ..plan })
            .ok_or_else(|| "project_out : aucune passe mesurée".to_owned())
    }

    /// The write pass: rows `first_row..` of `chunk` (their original bytes) receive the
    /// planned bytes.
    pub fn write_rows(
        &self,
        chunk: &mut [u8],
        first_row: usize,
        plan: &Plan,
    ) -> Result<(), RowError> {
        chunk
            .par_chunks_mut(self.layout.row_bytes())
            .enumerate()
            .try_for_each(|(o, row)| {
                self.row(first_row + o, row, &plan.r, &plan.correction)
                    .map(|_| ())
            })
    }
}

pub(super) const INTERRUPTED: &str = "interrompu (Ctrl-C ou annulation)";
