//! `project_out`: `W' = W − α · a (bᵀ W)` on a row-major `[out, in]` tensor, the weight form
//! of the LEACE eraser for a matrix that writes into the residual stream (`down_proj`,
//! `[hidden, intermediate]`): every output `W x` becomes `r(W x)` around a zero mean, i.e.
//! `(I − α a bᵀ) W x`. `a` and `b` have the length of the rows (`out`).
//!
//! Two parallel passes, every check before the first write:
//! 1. `r_j = Σ_i b_i W_ij` in f64, by blocks of 64 rows: each block sums its rows in order,
//!    then the block partials are added in block order. The block size does not depend on
//!    the thread count, so `r` is bit-identical with any number of threads.
//! 2. `W_ij ← W_ij − (α a_i) r_j` in f32, rounded to nearest even into bf16/f16/f32 exactly
//!    as `apply_vector_shift` stores, with the same `ShiftStats`.

use rayon::prelude::*;

use super::error::{EngineError, Result};
use super::index::DType;
use super::shift::{bf16_load, bf16_store, f16_load, f16_store, shift_cells, ShiftStats};

const ROW_BLOCK: usize = 64;

pub fn project_out_tensor(
    tensor_bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    a: &[f32],
    b: &[f32],
    alpha: f32,
) -> Result<ShiftStats> {
    let Some(cols) = check(tensor_bytes, &dtype, shape, a, b, alpha, true)? else {
        return Ok(ShiftStats::default());
    };
    Ok(match dtype {
        DType::BF16 => project(tensor_bytes, cols, a, b, alpha, bf16_load, bf16_store),
        DType::F16 => project(tensor_bytes, cols, a, b, alpha, f16_load, f16_store),
        DType::F32 => project(
            tensor_bytes,
            cols,
            a,
            b,
            alpha,
            f32::from_le_bytes,
            f32::to_le_bytes,
        ),
        DType::Other(name) => return Err(EngineError::UnsupportedDtype(name)),
    })
}

/// `W' = W − α · (W b) aᵀ` for a matrix whose rows live in the residual stream (the input
/// embeddings, `[vocab, hidden]`): every row `x` becomes `x − α a (bᵀ x)`. `a` and `b` have
/// the length of the columns. Rows are independent (`bᵀ x` in f64, in column order), so the
/// result is bit-identical with any number of threads; same checks and stores as
/// [`project_out_tensor`].
pub fn project_out_embedding(
    tensor_bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    a: &[f32],
    b: &[f32],
    alpha: f32,
) -> Result<ShiftStats> {
    let Some(cols) = check(tensor_bytes, &dtype, shape, a, b, alpha, false)? else {
        return Ok(ShiftStats::default());
    };
    Ok(match dtype {
        DType::BF16 => project_rows(tensor_bytes, cols, a, b, alpha, bf16_load, bf16_store),
        DType::F16 => project_rows(tensor_bytes, cols, a, b, alpha, f16_load, f16_store),
        DType::F32 => project_rows(
            tensor_bytes,
            cols,
            a,
            b,
            alpha,
            f32::from_le_bytes,
            f32::to_le_bytes,
        ),
        DType::Other(name) => return Err(EngineError::UnsupportedDtype(name)),
    })
}

/// Every check of both orientations, before the first write. `a` and `b` index the rows
/// when `along_rows`, the columns otherwise. `Ok(None)`: nothing to do; else the columns.
fn check(
    tensor_bytes: &[u8],
    dtype: &DType,
    shape: &[usize],
    a: &[f32],
    b: &[f32],
    alpha: f32,
    along_rows: bool,
) -> Result<Option<usize>> {
    let elem = dtype
        .size_of()
        .ok_or_else(|| EngineError::UnsupportedDtype(dtype.to_string()))?;
    let invalid_shape = |reason: &str| EngineError::InvalidShape {
        shape: shape.to_vec(),
        reason: reason.to_owned(),
    };
    let &[rows, cols] = shape else {
        return Err(invalid_shape("tenseur 2D attendu"));
    };
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(elem))
        .ok_or_else(|| invalid_shape("dépassement de capacité"))?;
    if tensor_bytes.len() != expected {
        return Err(EngineError::ByteLengthMismatch {
            expected,
            got: tensor_bytes.len(),
        });
    }
    let len = if along_rows { rows } else { cols };
    for v in [a, b] {
        if v.len() != len {
            return Err(EngineError::DirectionLengthMismatch {
                expected: len,
                got: v.len(),
            });
        }
    }
    if !(alpha.is_finite() && (0.0..=1.0).contains(&alpha)) {
        return Err(EngineError::InvalidAlpha(alpha));
    }
    if let Some(index) = a.iter().chain(b).position(|v| !v.is_finite()) {
        return Err(EngineError::NonFiniteDirection { index: index % len });
    }
    Ok((alpha != 0.0 && !tensor_bytes.is_empty()).then_some(cols))
}

fn project_rows<const N: usize, L, S>(
    bytes: &mut [u8],
    cols: usize,
    a: &[f32],
    b: &[f32],
    alpha: f32,
    load: L,
    store: S,
) -> ShiftStats
where
    L: Fn([u8; N]) -> f32 + Sync,
    S: Fn(f32) -> [u8; N] + Sync,
{
    let row_bytes = cols * N;
    bytes
        .par_chunks_mut(row_bytes)
        .with_min_len((64 * 1024 / row_bytes).max(1))
        .map(|row| {
            let (cells, _) = row.as_chunks_mut::<N>();
            let read: f64 = cells
                .iter()
                .zip(b)
                .map(|(cell, bj)| f64::from(*bj) * f64::from(load(*cell)))
                .sum();
            let coef = alpha * read as f32;
            shift_cells(cells, a.iter().map(|aj| -(coef * aj)), &load, &store)
        })
        .reduce(ShiftStats::default, merge)
}

fn project<const N: usize, L, S>(
    bytes: &mut [u8],
    cols: usize,
    a: &[f32],
    b: &[f32],
    alpha: f32,
    load: L,
    store: S,
) -> ShiftStats
where
    L: Fn([u8; N]) -> f32 + Sync,
    S: Fn(f32) -> [u8; N] + Sync,
{
    let row_bytes = cols * N;
    let partials: Vec<Vec<f64>> = bytes
        .par_chunks(row_bytes * ROW_BLOCK)
        .enumerate()
        .map(|(block, chunk)| {
            let mut acc = vec![0.0f64; cols];
            for (offset, row) in chunk.chunks_exact(row_bytes).enumerate() {
                let bi = f64::from(b[block * ROW_BLOCK + offset]);
                let (cells, _) = row.as_chunks::<N>();
                for (s, cell) in acc.iter_mut().zip(cells) {
                    *s += bi * f64::from(load(*cell));
                }
            }
            acc
        })
        .collect();
    let mut r = vec![0.0f64; cols];
    for partial in &partials {
        r.iter_mut().zip(partial).for_each(|(s, p)| *s += p);
    }
    let r32: Vec<f32> = r.iter().map(|&x| x as f32).collect();
    bytes
        .par_chunks_mut(row_bytes)
        .with_min_len((64 * 1024 / row_bytes).max(1))
        .enumerate()
        .map(|(i, row)| {
            let coef = alpha * a[i];
            let (cells, _) = row.as_chunks_mut::<N>();
            shift_cells(cells, r32.iter().map(|rj| -(coef * rj)), &load, &store)
        })
        .reduce(ShiftStats::default, merge)
}

fn merge(x: ShiftStats, y: ShiftStats) -> ShiftStats {
    ShiftStats {
        elements_modified: x.elements_modified + y.elements_modified,
        max_abs_delta: x.max_abs_delta.max(y.max_abs_delta),
        nan_or_inf_produced: x.nan_or_inf_produced + y.nan_or_inf_produced,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(values: &[f32], dtype: &DType) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| match dtype {
                DType::BF16 => half::bf16::from_f32(v).to_le_bytes().to_vec(),
                DType::F16 => half::f16::from_f32(v).to_le_bytes().to_vec(),
                _ => v.to_le_bytes().to_vec(),
            })
            .collect()
    }

    fn decode(bytes: &[u8], dtype: &DType) -> Vec<f32> {
        match dtype {
            DType::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::BF16 => bytes
                .chunks_exact(2)
                .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
            _ => bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect(),
        }
    }

    /// Dyadic values (multiples of 1/8 up to ±4): every f64 sum is exact, so the reference
    /// is independent of the summation order and the comparison can be bit for bit.
    fn weights(rows: usize, cols: usize) -> Vec<f32> {
        (0..rows * cols)
            .map(|k| ((k * 37 % 65) as f32 - 32.0) / 8.0)
            .collect()
    }

    /// Plain loops: r in f64 column by column, then the same f32 update and RNE store.
    fn reference(
        w: &[f32],
        rows: usize,
        cols: usize,
        a: &[f32],
        b: &[f32],
        alpha: f32,
        dtype: &DType,
    ) -> Vec<u8> {
        let stored = decode(&encode(w, dtype), dtype);
        let mut out = Vec::new();
        let r: Vec<f32> = (0..cols)
            .map(|j| {
                (0..rows)
                    .map(|i| f64::from(b[i]) * f64::from(stored[i * cols + j]))
                    .sum::<f64>() as f32
            })
            .collect();
        for i in 0..rows {
            for j in 0..cols {
                let new = stored[i * cols + j] + -(alpha * a[i] * r[j]);
                out.push(new);
            }
        }
        encode(&out, dtype)
    }

    #[test]
    fn matches_an_independent_reference_bit_for_bit_in_every_dtype() {
        let (rows, cols) = (130, 7); // three row blocks, the last one partial
        let w = weights(rows, cols);
        let a: Vec<f32> = (0..rows).map(|i| ((i % 5) as f32 - 2.0) / 4.0).collect();
        let b: Vec<f32> = (0..rows).map(|i| ((i % 3) as f32 - 1.0) / 2.0).collect();
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let mut bytes = encode(&w, &dtype);
            let stats =
                project_out_tensor(&mut bytes, dtype.clone(), &[rows, cols], &a, &b, 0.5).unwrap();
            assert_eq!(
                bytes,
                reference(&w, rows, cols, &a, &b, 0.5, &dtype),
                "{dtype}"
            );
            assert!(stats.elements_modified > 0 && stats.nan_or_inf_produced == 0);
        }
    }

    #[test]
    fn identical_with_any_thread_count() {
        let (rows, cols) = (300, 33);
        let w: Vec<f32> = (0..rows * cols)
            .map(|k| ((k as f32) * 0.618).sin())
            .collect();
        let a: Vec<f32> = (0..rows).map(|i| ((i as f32) * 0.3).cos()).collect();
        let b: Vec<f32> = (0..rows).map(|i| ((i as f32) * 0.7).sin() / 10.0).collect();
        let runs: Vec<Vec<u8>> = [1, 2, 8]
            .iter()
            .map(|&t| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap()
                    .install(|| {
                        let mut bytes = encode(&w, &DType::F32);
                        project_out_tensor(&mut bytes, DType::F32, &[rows, cols], &a, &b, 1.0)
                            .unwrap();
                        bytes
                    })
            })
            .collect();
        assert!(runs.windows(2).all(|p| p[0] == p[1]));
    }

    #[test]
    fn full_strength_erases_the_read_direction_and_is_idempotent() {
        let (rows, cols) = (8, 5);
        let w: Vec<f32> = (0..rows * cols).map(|k| ((k as f32) * 1.3).sin()).collect();
        let mut a = vec![0.0f32; rows];
        let mut b = vec![0.0f32; rows];
        (a[0], a[1], b[0]) = (1.0, 0.5, 1.0); // bᵀa = 1 exactly
        let mut bytes = encode(&w, &DType::F32);
        project_out_tensor(&mut bytes, DType::F32, &[rows, cols], &a, &b, 1.0).unwrap();
        let once = decode(&bytes, &DType::F32);
        for j in 0..cols {
            let read: f32 = (0..rows).map(|i| b[i] * once[i * cols + j]).sum();
            assert!(read.abs() < 1e-6, "bᵀW' = {read}");
        }
        project_out_tensor(&mut bytes, DType::F32, &[rows, cols], &a, &b, 1.0).unwrap();
        let twice = decode(&bytes, &DType::F32);
        for (x, y) in once.iter().zip(&twice) {
            let ulps = (x.to_bits() as i64 - y.to_bits() as i64).abs();
            assert!(ulps <= 1 || (x - y).abs() < 1e-7, "{x} vs {y}");
        }
        let mut untouched = encode(&w, &DType::F32);
        let stats =
            project_out_tensor(&mut untouched, DType::F32, &[rows, cols], &a, &b, 0.0).unwrap();
        assert_eq!(untouched, encode(&w, &DType::F32));
        assert_eq!(stats, ShiftStats::default());
    }

    #[test]
    fn embedding_side_matches_a_plain_reference_and_erases_the_read_direction() {
        let (rows, cols) = (70, 6);
        let w = weights(rows, cols);
        let a: Vec<f32> = (0..cols).map(|j| ((j % 5) as f32 - 2.0) / 4.0).collect();
        let b: Vec<f32> = (0..cols).map(|j| ((j % 3) as f32 - 1.0) / 2.0).collect();
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let stored = decode(&encode(&w, &dtype), &dtype);
            let mut expected = Vec::new();
            for row in stored.chunks_exact(cols) {
                let read = row
                    .iter()
                    .zip(&b)
                    .map(|(x, bj)| f64::from(*bj) * f64::from(*x))
                    .sum::<f64>() as f32;
                expected.extend(row.iter().zip(&a).map(|(x, aj)| x + -(0.5 * read * aj)));
            }
            let mut bytes = encode(&w, &dtype);
            let stats =
                project_out_embedding(&mut bytes, dtype.clone(), &[rows, cols], &a, &b, 0.5)
                    .unwrap();
            assert_eq!(bytes, encode(&expected, &dtype), "{dtype}");
            assert!(stats.elements_modified > 0 && stats.nan_or_inf_produced == 0);
        }

        // a = b = unit vector: every row loses its component along it, the rest is kept.
        let mut v = vec![0.0f32; cols];
        (v[1], v[4]) = (0.6, 0.8);
        let mut bytes = encode(&w, &DType::F32);
        project_out_embedding(&mut bytes, DType::F32, &[rows, cols], &v, &v, 1.0).unwrap();
        for (new, old) in decode(&bytes, &DType::F32)
            .chunks_exact(cols)
            .zip(w.chunks_exact(cols))
        {
            let read: f32 = new.iter().zip(&v).map(|(x, vj)| x * vj).sum();
            assert!(read.abs() < 1e-5, "W'v = {read}");
            assert_eq!((new[0], new[5]), (old[0], old[5]), "orthogonal columns");
        }

        // The vectors index the columns here, the rows in `project_out_tensor`.
        let mut bytes = encode(&w, &DType::F32);
        let rows_len = vec![0.5f32; rows];
        assert!(matches!(
            project_out_embedding(
                &mut bytes,
                DType::F32,
                &[rows, cols],
                &rows_len,
                &rows_len,
                1.0
            ),
            Err(EngineError::DirectionLengthMismatch { expected: 6, .. })
        ));
        assert_eq!(bytes, encode(&w, &DType::F32));
    }

    #[test]
    fn every_check_runs_before_the_first_write() {
        let mut bytes = encode(&[1.0; 6], &DType::F32);
        let original = bytes.clone();
        let ok = [1.0f32, 0.0];
        for (shape, a, b, alpha) in [
            (vec![2, 3], &ok[..], &[1.0f32][..], 0.5),
            (vec![2, 3], &[1.0, 2.0, 3.0][..], &ok[..], 0.5),
            (vec![2, 3], &ok[..], &ok[..], 1.5),
            (vec![2, 3], &ok[..], &ok[..], -0.1),
            (vec![2, 3], &[f32::NAN, 0.0][..], &ok[..], 0.5),
            (vec![6], &ok[..], &ok[..], 0.5),
            (vec![3, 3], &ok[..], &ok[..], 0.5),
        ] {
            assert!(project_out_tensor(&mut bytes, DType::F32, &shape, a, b, alpha).is_err());
            assert_eq!(bytes, original);
        }
        assert!(project_out_tensor(
            &mut bytes,
            DType::Other("I8".into()),
            &[2, 3],
            &ok,
            &ok,
            0.5
        )
        .is_err());
    }
}
