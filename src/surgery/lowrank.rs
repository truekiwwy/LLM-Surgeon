//! In-place `W <- W + U_scaled V` on a row-major tensor. Shared kernel of `.surgery` apply
//! and `lora merge`.
//!
//! Rounding is the one of `engine::apply_vector_shift`: `half::{bf16,f16}::from_f32`
//! (round to nearest, ties to even), little-endian bytes, no pointer casts.

use std::sync::atomic::{AtomicBool, Ordering};

use half::{bf16, f16};
use rayon::prelude::*;
use serde::Serialize;

use super::error::{Result, SurgeryError};
use crate::engine::DType;

/// Elements per rayon task: amortises scheduling and bounds the cancellation latency.
const BLOCK_ELEMS: usize = 1 << 14;

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct LowRankStats {
    /// Elements whose stored bits changed.
    pub elements_modified: u64,
    /// Largest |new - old| over finite results, after rounding to the dtype.
    pub max_abs_delta: f32,
    /// Finite weights that became NaN or Inf (the caller rolls back when non-zero).
    pub nan_or_inf_produced: u64,
}

impl LowRankStats {
    pub fn merge(self, other: Self) -> Self {
        Self {
            elements_modified: self.elements_modified + other.elements_modified,
            max_abs_delta: self.max_abs_delta.max(other.max_abs_delta),
            nan_or_inf_produced: self.nan_or_inf_produced + other.nan_or_inf_produced,
        }
    }
}

/// `bytes` is a `[rows, cols]` tensor of `dtype`; `u_scaled` is [rows x r] with the scale
/// and singular values already folded in, `v` is [r x cols]. For row `i`:
/// `delta[i, :] = sum_k u_scaled[i, k] * v[k, :]` accumulated in f32 in fixed `k` order,
/// then `W[i, j] <- round_to_nearest_even_dtype(W[i, j] + delta[i, j])`. Rayon over row
/// blocks; identical bytes whatever the thread count. Nothing is written on `Err`
/// (validation) or when `cancel` is raised before the first block; a cancellation raised
/// later returns `Cancelled` with the finished blocks written (the caller rolls back).
/// A weight whose delta is exactly zero is left untouched (so `-0.0` stays `-0.0`).
#[allow(clippy::too_many_arguments)]
pub fn apply_lowrank_rows(
    bytes: &mut [u8],
    dtype: DType,
    rows: usize,
    cols: usize,
    u_scaled: &[f32],
    v: &[f32],
    r: usize,
    cancel: &AtomicBool,
) -> Result<LowRankStats> {
    let elem = check_layout(bytes.len(), &dtype, rows, cols)?;
    if rows.checked_mul(r) != Some(u_scaled.len()) {
        return Err(SurgeryError::Usage(format!(
            "u_scaled : {} valeurs, {rows} x {r} attendues",
            u_scaled.len()
        )));
    }
    if r.checked_mul(cols) != Some(v.len()) {
        return Err(SurgeryError::Usage(format!(
            "v : {} valeurs, {r} x {cols} attendues",
            v.len()
        )));
    }
    check_finite("u_scaled", u_scaled)?;
    check_finite("v", v)?;
    let row_delta = |i: usize, out: &mut [f32]| {
        out.fill(0.0);
        for (k, &a) in u_scaled[i * r..(i + 1) * r].iter().enumerate() {
            for (d, x) in out.iter_mut().zip(&v[k * cols..(k + 1) * cols]) {
                *d += a * x;
            }
        }
    };
    run(bytes, &dtype, elem, cols, cancel, row_delta)
}

/// `W <- RNE(W + scale * delta)` for a dense `[rows, cols]` delta (1D tensors: `rows = 1`).
/// Same guarantees as [`apply_lowrank_rows`].
pub fn apply_full_delta(
    bytes: &mut [u8],
    dtype: DType,
    rows: usize,
    cols: usize,
    delta: &[f32],
    scale: f32,
    cancel: &AtomicBool,
) -> Result<LowRankStats> {
    let elem = check_layout(bytes.len(), &dtype, rows, cols)?;
    if rows.checked_mul(cols) != Some(delta.len()) {
        return Err(SurgeryError::Usage(format!(
            "delta : {} valeurs, {rows} x {cols} attendues",
            delta.len()
        )));
    }
    if !scale.is_finite() {
        return Err(SurgeryError::Usage(format!("scale non fini : {scale}")));
    }
    check_finite("delta", delta)?;
    let row_delta = |i: usize, out: &mut [f32]| {
        for (d, x) in out.iter_mut().zip(&delta[i * cols..(i + 1) * cols]) {
            *d = scale * x;
        }
    };
    run(bytes, &dtype, elem, cols, cancel, row_delta)
}

/// Returns the element size once dtype and byte length are consistent with the shape.
fn check_layout(len: usize, dtype: &DType, rows: usize, cols: usize) -> Result<usize> {
    let elem = dtype.size_of().ok_or_else(|| {
        SurgeryError::Usage(format!(
            "dtype {dtype} non supporté (BF16, F16, F32 seulement)"
        ))
    })?;
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(elem))
        .ok_or_else(|| SurgeryError::Usage(format!("forme {rows} x {cols} trop grande")))?;
    if len != expected {
        return Err(SurgeryError::Usage(format!(
            "tenseur {rows} x {cols} {dtype} : {expected} octets attendus, {len} fournis"
        )));
    }
    Ok(elem)
}

fn check_finite(what: &str, values: &[f32]) -> Result<()> {
    match values.iter().position(|x| !x.is_finite()) {
        Some(i) => Err(SurgeryError::Usage(format!(
            "{what}[{i}] n'est pas fini ({})",
            values[i]
        ))),
        None => Ok(()),
    }
}

fn run(
    bytes: &mut [u8],
    dtype: &DType,
    elem: usize,
    cols: usize,
    cancel: &AtomicBool,
    row_delta: impl Fn(usize, &mut [f32]) + Sync,
) -> Result<LowRankStats> {
    // `par_chunks_mut(0)` panics, and an empty tensor has nothing to do.
    if bytes.is_empty() {
        return Ok(LowRankStats::default());
    }
    let rows_per_block = (BLOCK_ELEMS / cols).max(1);
    let block_bytes = rows_per_block * cols * elem;
    match dtype {
        DType::BF16 => blocks::<2>(
            bytes,
            cols,
            block_bytes,
            cancel,
            &row_delta,
            bf16_load,
            bf16_store,
        ),
        DType::F16 => blocks::<2>(
            bytes,
            cols,
            block_bytes,
            cancel,
            &row_delta,
            f16_load,
            f16_store,
        ),
        DType::F32 => blocks::<4>(
            bytes,
            cols,
            block_bytes,
            cancel,
            &row_delta,
            f32::from_le_bytes,
            f32::to_le_bytes,
        ),
        DType::Other(name) => Err(SurgeryError::Usage(format!("dtype {name} non supporté"))),
    }
}

fn bf16_load(b: [u8; 2]) -> f32 {
    bf16::from_le_bytes(b).to_f32()
}
fn bf16_store(x: f32) -> [u8; 2] {
    bf16::from_f32(x).to_le_bytes()
}
fn f16_load(b: [u8; 2]) -> f32 {
    f16::from_le_bytes(b).to_f32()
}
fn f16_store(x: f32) -> [u8; 2] {
    f16::from_f32(x).to_le_bytes()
}

/// Every element depends only on itself and its row's delta, and the stats reduction (sum,
/// max) is exact: the result does not depend on how rayon splits or orders the blocks.
fn blocks<const N: usize>(
    bytes: &mut [u8],
    cols: usize,
    block_bytes: usize,
    cancel: &AtomicBool,
    row_delta: &(impl Fn(usize, &mut [f32]) + Sync),
    load: impl Fn([u8; N]) -> f32 + Sync,
    store: impl Fn(f32) -> [u8; N] + Sync,
) -> Result<LowRankStats> {
    let skipped = AtomicBool::new(false);
    let rows_per_block = block_bytes / (cols * N);
    let stats = bytes
        .par_chunks_mut(block_bytes)
        .enumerate()
        .map(|(b, block)| {
            if cancel.load(Ordering::SeqCst) {
                skipped.store(true, Ordering::SeqCst);
                return LowRankStats::default();
            }
            let mut delta = vec![0f32; cols];
            let mut stats = LowRankStats::default();
            for (local, row) in block.chunks_exact_mut(cols * N).enumerate() {
                row_delta(b * rows_per_block + local, &mut delta);
                let (cells, _) = row.as_chunks_mut::<N>();
                stats = stats.merge(add_row(cells, &delta, &load, &store));
            }
            stats
        })
        .reduce(LowRankStats::default, LowRankStats::merge);
    match skipped.load(Ordering::SeqCst) {
        true => Err(SurgeryError::Cancelled),
        false => Ok(stats),
    }
}

fn add_row<const N: usize>(
    cells: &mut [[u8; N]],
    delta: &[f32],
    load: &impl Fn([u8; N]) -> f32,
    store: &impl Fn(f32) -> [u8; N],
) -> LowRankStats {
    let mut stats = LowRankStats::default();
    for (cell, &d) in cells.iter_mut().zip(delta) {
        if d == 0.0 {
            continue;
        }
        let old = load(*cell);
        let new_bits = store(old + d);
        if new_bits != *cell {
            let new = load(new_bits);
            stats.elements_modified += 1;
            if new.is_finite() {
                stats.max_abs_delta = stats.max_abs_delta.max((new - old).abs());
            } else if old.is_finite() {
                stats.nan_or_inf_produced += 1;
            }
            *cell = new_bits;
        }
    }
    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic values in [-1, 1) with few mantissa bits.
    fn lcg(seed: u64, n: usize) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as i32 % 1024) as f32 / 512.0 - 1.0
            })
            .collect()
    }

    fn encode(dtype: &DType, w: &[f32]) -> Vec<u8> {
        match dtype {
            DType::BF16 => w.iter().flat_map(|x| bf16_store(*x)).collect(),
            DType::F16 => w.iter().flat_map(|x| f16_store(*x)).collect(),
            _ => w.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    fn decode(dtype: &DType, b: &[u8]) -> Vec<f32> {
        match dtype {
            DType::BF16 => b.chunks(2).map(|c| bf16_load([c[0], c[1]])).collect(),
            DType::F16 => b.chunks(2).map(|c| f16_load([c[0], c[1]])).collect(),
            _ => b
                .chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }

    fn never() -> AtomicBool {
        AtomicBool::new(false)
    }

    #[test]
    fn matches_f64_reference_within_half_an_ulp() {
        let (rows, cols, r) = (37, 53, 4);
        let u = lcg(1, rows * r);
        let v = lcg(2, r * cols);
        let w = lcg(3, rows * cols);
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let mut bytes = encode(&dtype, &w);
            let old = decode(&dtype, &bytes);
            let stats = apply_lowrank_rows(
                &mut bytes,
                dtype.clone(),
                rows,
                cols,
                &u,
                &v,
                r,
                &never(),
            )
            .unwrap();
            let new = decode(&dtype, &bytes);
            let mut modified = 0;
            for i in 0..rows {
                for j in 0..cols {
                    let delta: f64 = (0..r)
                        .map(|k| f64::from(u[i * r + k]) * f64::from(v[k * cols + j]))
                        .sum();
                    let exact = f64::from(old[i * cols + j]) + delta;
                    let got = f64::from(new[i * cols + j]);
                    // RNE: at most half an ulp of the dtype (2^-8 relative for bf16, 2^-11
                    // for f16), plus a hair for the f32 accumulation.
                    let rel = match dtype {
                        DType::BF16 => 2f64.powi(-8),
                        DType::F16 => 2f64.powi(-11),
                        _ => 2f64.powi(-22),
                    };
                    let tol = rel * exact.abs().max(1e-3) * 1.01;
                    assert!(
                        (got - exact).abs() <= tol,
                        "{dtype} [{i},{j}] {got} vs {exact}"
                    );
                    modified += u64::from(new[i * cols + j] != old[i * cols + j]);
                }
            }
            assert_eq!(stats.elements_modified, modified);
            assert_eq!(stats.nan_or_inf_produced, 0);
            assert!(stats.max_abs_delta > 0.0);
        }
    }

    #[test]
    fn exact_values_and_ties_round_to_even() {
        // bf16 has 8 significant bits: 256 + 1 is a tie between 256 and 258 (even mantissa
        // wins -> 256), 258 + 1 is a tie between 258 and 260 (-> 260).
        let mut bytes = encode(&DType::BF16, &[256.0, 258.0, 1.0, 0.5]);
        let stats = apply_lowrank_rows(
            &mut bytes,
            DType::BF16,
            2,
            2,
            &[1.0, 0.5],
            &[1.0, 1.0],
            1,
            &never(),
        )
        .unwrap();
        assert_eq!(decode(&DType::BF16, &bytes), [256.0, 260.0, 1.5, 1.0]);
        assert_eq!(stats.elements_modified, 3);
        assert_eq!(stats.max_abs_delta, 2.0);
    }

    #[test]
    fn identical_bytes_with_1_2_and_8_threads() {
        let (rows, cols, r) = (301, 97, 7);
        let (u, v, w) = (lcg(4, rows * r), lcg(5, r * cols), lcg(6, rows * cols));
        let run_with = |threads: usize| {
            let mut bytes = encode(&DType::BF16, &w);
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let stats = pool
                .install(|| {
                    apply_lowrank_rows(&mut bytes, DType::BF16, rows, cols, &u, &v, r, &never())
                })
                .unwrap();
            (bytes, stats)
        };
        let reference = run_with(1);
        assert_eq!(run_with(2), reference);
        assert_eq!(run_with(8), reference);
    }

    #[test]
    fn zero_delta_leaves_negative_zero_untouched() {
        let mut bytes = encode(&DType::F32, &[-0.0, 1.0]);
        let before = bytes.clone();
        let stats = apply_lowrank_rows(
            &mut bytes,
            DType::F32,
            1,
            2,
            &[0.0],
            &[3.0, 4.0],
            1,
            &never(),
        )
        .unwrap();
        assert_eq!((bytes, stats), (before, LowRankStats::default()));
    }

    #[test]
    fn overflow_is_counted_not_hidden() {
        let mut bytes = encode(&DType::F16, &[60000.0, 1.0]);
        let stats = apply_lowrank_rows(
            &mut bytes,
            DType::F16,
            1,
            2,
            &[1.0],
            &[60000.0, 1.0],
            1,
            &never(),
        )
        .unwrap();
        assert_eq!(stats.nan_or_inf_produced, 1);
        assert_eq!(stats.elements_modified, 2);
    }

    #[test]
    fn invalid_inputs_fail_before_any_write() {
        let original = encode(&DType::F32, &[1.0; 6]);
        let mut bytes = original.clone();
        let call = |bytes: &mut [u8], dtype: DType, u: &[f32], v: &[f32]| {
            apply_lowrank_rows(bytes, dtype, 2, 3, u, v, 1, &never())
        };
        let ok_u = [1.0, 1.0];
        let ok_v = [1.0, 1.0, 1.0];
        assert!(call(&mut bytes, DType::F32, &ok_u[..1], &ok_v).is_err());
        assert!(call(&mut bytes, DType::F32, &ok_u, &ok_v[..2]).is_err());
        assert!(call(&mut bytes, DType::F32, &[1.0, f32::NAN], &ok_v).is_err());
        assert!(call(&mut bytes, DType::F32, &ok_u, &[1.0, f32::INFINITY, 1.0]).is_err());
        assert!(call(&mut bytes, DType::Other("I8".into()), &ok_u, &ok_v).is_err());
        assert!(call(&mut bytes[..20], DType::F32, &ok_u, &ok_v).is_err());
        assert_eq!(bytes, original);
    }

    #[test]
    fn cancellation_before_start_writes_nothing() {
        let original = encode(&DType::F32, &[1.0; 6]);
        let mut bytes = original.clone();
        let cancel = AtomicBool::new(true);
        let res = apply_lowrank_rows(
            &mut bytes,
            DType::F32,
            2,
            3,
            &[1.0; 2],
            &[1.0; 3],
            1,
            &cancel,
        );
        assert!(matches!(res, Err(SurgeryError::Cancelled)));
        assert_eq!(bytes, original);
    }

    #[test]
    fn full_delta_scales_and_handles_1d() {
        let mut bytes = encode(&DType::BF16, &[1.0, 2.0, 3.0]);
        let stats = apply_full_delta(
            &mut bytes,
            DType::BF16,
            1,
            3,
            &[1.0, 0.0, -2.0],
            0.5,
            &never(),
        )
        .unwrap();
        assert_eq!(decode(&DType::BF16, &bytes), [1.5, 2.0, 2.0]);
        assert_eq!(stats.elements_modified, 2);
        assert!(
            apply_full_delta(&mut bytes, DType::BF16, 1, 3, &[0.0; 2], 1.0, &never()).is_err()
        );
    }
}
