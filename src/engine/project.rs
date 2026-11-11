//! `project_out`: `W' = W − α · A (Bᵀ W)` on a row-major `[out, in]` tensor, the weight form
//! of an eraser for a matrix that writes into the residual stream (`down_proj`,
//! `[hidden, intermediate]`): every output `W x` becomes `(I − α A Bᵀ) W x`. `A` and `B`
//! hold `k` directions of the length of the rows (`out`), one per row of `[k, out]`. With
//! `k = 1` this is the rank-1 form `W − α · a (bᵀ W)`, bit for bit what it always was; with
//! `Bᵀ A = I_k` it is a projector (orthogonal when `A = B`, the LEACE eraser otherwise).
//!
//! The `k` directions go in ONE pass over the weights, so each element is rounded once,
//! whatever `k`: `k` successive rank-1 operations would round `k` times, in an order that
//! changes the result.
//!
//! Two parallel passes, every check before the first write:
//! 1. `R = Bᵀ W` (`[k, in]`) in f64, by blocks of 64 rows: each block sums its rows in
//!    order, then the block partials are added in block order. The block size does not
//!    depend on the thread count, so `R` is bit-identical with any number of threads.
//!    Blocks run in waves, which bounds the memory of the partials and changes nothing.
//! 2. `W_ij ← W_ij − Σ_t (α A_ti) R_tj`, the sum in f32 in the order of `t`, rounded to
//!    nearest even into bf16/f16/f32 exactly as `apply_vector_shift` stores, with the same
//!    `ShiftStats`.
//!
//! [`accumulate_r`] and [`project_rows_inplace`] are the two passes on decoded rows, for a
//! container that is not a plain array of floats.

use rayon::prelude::*;

use super::error::{EngineError, Result};
use super::index::DType;
use super::shift::{bf16_load, bf16_store, f16_load, f16_store, shift_cells, ShiftStats};

/// Largest rank a front-end accepts unless told otherwise (`surgeon apply --max-rank`).
pub const DEFAULT_MAX_RANK: usize = 64;

const ROW_BLOCK: usize = 64;
/// Bytes of block partials (`[k, in]` f64 each) alive at once in pass 1.
const WAVE_BYTES: usize = 64 << 20;

/// A rank-k eraser `x ↦ x − α A (Bᵀ x)`: `a` and `b` hold `k` directions, one per row of
/// `[k, len]`, row-major. `b` reads, `a` writes.
#[derive(Debug, Clone, Copy)]
pub struct SubspaceProjection<'a> {
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub alpha: f32,
    pub k: usize,
}

impl<'a> SubspaceProjection<'a> {
    /// The historical form: one direction, `a` and `b` of the length of the space.
    pub fn rank_one(a: &'a [f32], b: &'a [f32], alpha: f32) -> Self {
        Self { a, b, alpha, k: 1 }
    }

    /// Length of one direction (0 when `k` is 0).
    pub fn direction_len(&self) -> usize {
        self.a.len().checked_div(self.k).unwrap_or(0)
    }

    /// `‖Bᵀ A − I_k‖_F` in f64. Zero means a projector: applied twice at full strength,
    /// the eraser removes nothing more. Infinite when `a` and `b` are not both `[k, len]`.
    pub fn gram_error(&self) -> f64 {
        let len = self.direction_len();
        if len == 0 || self.a.len() != self.k * len || self.b.len() != self.a.len() {
            return f64::INFINITY;
        }
        let mut sum = 0.0f64;
        for (s, b) in self.b.chunks_exact(len).enumerate() {
            for (t, a) in self.a.chunks_exact(len).enumerate() {
                let gram: f64 = b
                    .iter()
                    .zip(a)
                    .map(|(x, y)| f64::from(*x) * f64::from(*y))
                    .sum();
                let off = gram - f64::from(u8::from(s == t));
                sum += off * off;
            }
        }
        sum.sqrt()
    }
}

/// `W' = W − α · a (bᵀ W)`: [`project_out_subspace`] with one direction.
pub fn project_out_tensor(
    tensor_bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    a: &[f32],
    b: &[f32],
    alpha: f32,
) -> Result<ShiftStats> {
    let p = SubspaceProjection::rank_one(a, b, alpha);
    project_out_subspace(tensor_bytes, dtype, shape, &p)
}

/// `W' = W − α · (W b) aᵀ`: [`project_out_subspace_embedding`] with one direction.
pub fn project_out_embedding(
    tensor_bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    a: &[f32],
    b: &[f32],
    alpha: f32,
) -> Result<ShiftStats> {
    let p = SubspaceProjection::rank_one(a, b, alpha);
    project_out_subspace_embedding(tensor_bytes, dtype, shape, &p)
}

/// `W' = W − α · A (Bᵀ W)` in one pass; the directions have the length of the rows.
pub fn project_out_subspace(
    tensor_bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    p: &SubspaceProjection<'_>,
) -> Result<ShiftStats> {
    run(tensor_bytes, dtype, shape, p, true)
}

/// `W' = W − α · (W Bᵀ) A` for a matrix whose rows live in the residual stream (the input
/// embeddings, `[vocab, hidden]`): every row `x` becomes `x − α Σ_t (x · b_t) a_t`. The
/// directions have the length of the columns. Rows are independent (`x · b_t` in f64, in
/// column order), so the result is bit-identical with any number of threads; same checks
/// and stores as [`project_out_subspace`], one pass.
pub fn project_out_subspace_embedding(
    tensor_bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    p: &SubspaceProjection<'_>,
) -> Result<ShiftStats> {
    run(tensor_bytes, dtype, shape, p, false)
}

fn run(
    bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    p: &SubspaceProjection<'_>,
    along_rows: bool,
) -> Result<ShiftStats> {
    let Some(cols) = check(bytes, &dtype, shape, p, along_rows)? else {
        return Ok(ShiftStats::default());
    };
    fn go<const N: usize, L, S>(
        bytes: &mut [u8],
        cols: usize,
        p: &SubspaceProjection<'_>,
        along_rows: bool,
        load: L,
        store: S,
    ) -> ShiftStats
    where
        L: Fn([u8; N]) -> f32 + Sync,
        S: Fn(f32) -> [u8; N] + Sync,
    {
        let wave_blocks = (WAVE_BYTES / (p.k * cols * 8)).max(1);
        match along_rows {
            true => project(bytes, cols, p, wave_blocks, load, store),
            false => project_local(bytes, cols, p, load, store),
        }
    }
    Ok(match dtype {
        DType::BF16 => go(bytes, cols, p, along_rows, bf16_load, bf16_store),
        DType::F16 => go(bytes, cols, p, along_rows, f16_load, f16_store),
        DType::F32 => go(
            bytes,
            cols,
            p,
            along_rows,
            f32::from_le_bytes,
            f32::to_le_bytes,
        ),
        DType::Other(name) => return Err(EngineError::UnsupportedDtype(name)),
    })
}

/// Every check of both orientations, before the first write. The directions index the rows
/// when `along_rows`, the columns otherwise. `Ok(None)`: nothing to do; else the columns.
fn check(
    tensor_bytes: &[u8],
    dtype: &DType,
    shape: &[usize],
    p: &SubspaceProjection<'_>,
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
    let total =
        p.k.checked_mul(len)
            .filter(|_| p.k > 0)
            .ok_or_else(|| invalid_shape("project_out : au moins une direction (rang ≥ 1)"))?;
    for v in [p.a, p.b] {
        if v.len() != total {
            return Err(EngineError::DirectionLengthMismatch {
                expected: total,
                got: v.len(),
            });
        }
    }
    if !(p.alpha.is_finite() && (0.0..=1.0).contains(&p.alpha)) {
        return Err(EngineError::InvalidAlpha(p.alpha));
    }
    if let Some(index) = p.a.iter().chain(p.b).position(|v| !v.is_finite()) {
        return Err(EngineError::NonFiniteDirection { index: index % len });
    }
    Ok((p.alpha != 0.0 && !tensor_bytes.is_empty()).then_some(cols))
}

/// Pass 1 on decoded rows: `r[t][j] += Σ_i b[t][first_row + i] · rows[i][j]` in f64, the
/// rows in order. `rows` holds whole rows of `cols` values, the first one being row
/// `first_row` of the matrix; `r` is `[k, cols]`, row-major. For a result that does not
/// depend on the threads, accumulate fixed blocks of rows and add them in block order.
pub fn accumulate_r(
    p: &SubspaceProjection<'_>,
    first_row: usize,
    rows: &[f32],
    cols: usize,
    r: &mut [f64],
) {
    if cols == 0 {
        return;
    }
    let len = p.direction_len();
    for (i, row) in rows.chunks_exact(cols).enumerate() {
        for (t, r_t) in r.chunks_exact_mut(cols).enumerate() {
            let b = f64::from(p.b[t * len + first_row + i]);
            for (sum, x) in r_t.iter_mut().zip(row) {
                *sum += b * f64::from(*x);
            }
        }
    }
}

/// Pass 2 on decoded rows: `rows[i][j] −= Σ_t (α a[t][first_row + i]) · r[t][j]`, the sum
/// in f32 in the order of `t`. `r` is `R` of pass 1 rounded to f32, `[k, cols]`. Storing
/// each value once is the single rounding of the operation.
pub fn project_rows_inplace(
    p: &SubspaceProjection<'_>,
    first_row: usize,
    rows: &mut [f32],
    cols: usize,
    r: &[f32],
) {
    if cols == 0 {
        return;
    }
    let mut deltas = vec![0.0f32; cols];
    for (i, row) in rows.chunks_exact_mut(cols).enumerate() {
        row_deltas(p, first_row + i, r, &mut deltas);
        row.iter_mut().zip(&deltas).for_each(|(x, d)| *x += d);
    }
}

/// `deltas[j] = −Σ_t (α a[t][row]) · r[t][j]`. The first term starts the sum (not `0 +`):
/// with one direction this is `−(α a[row] · r[j])`, the historical expression. `r` is `R`
/// rounded to f32, `[k, deltas.len()]`: pass 2 is `row[j] + deltas[j]`.
pub fn row_deltas(p: &SubspaceProjection<'_>, row: usize, r: &[f32], deltas: &mut [f32]) {
    let len = p.direction_len();
    for (t, r_t) in r.chunks_exact(deltas.len()).enumerate() {
        let coef = p.alpha * p.a[t * len + row];
        match t {
            0 => deltas.iter_mut().zip(r_t).for_each(|(d, x)| *d = coef * x),
            _ => deltas.iter_mut().zip(r_t).for_each(|(d, x)| *d += coef * x),
        }
    }
    deltas.iter_mut().for_each(|d| *d = -*d);
}

/// `deltas[j] = −Σ_t (α (x · b_t)) · a_t[j]` for one row `x` of an embedding matrix: the
/// projected row is `x[j] + deltas[j]`.
pub fn local_deltas(p: &SubspaceProjection<'_>, x: &[f32], deltas: &mut [f32]) {
    let cols = x.len();
    let directions = p.a.chunks_exact(cols).zip(p.b.chunks_exact(cols));
    for (t, (a_t, b_t)) in directions.enumerate() {
        let read: f64 = x
            .iter()
            .zip(b_t)
            .map(|(x, b)| f64::from(*b) * f64::from(*x))
            .sum();
        let coef = p.alpha * read as f32;
        match t {
            0 => deltas.iter_mut().zip(a_t).for_each(|(d, a)| *d = coef * a),
            _ => deltas.iter_mut().zip(a_t).for_each(|(d, a)| *d += coef * a),
        }
    }
    deltas.iter_mut().for_each(|d| *d = -*d);
}

fn project_local<const N: usize, L, S>(
    bytes: &mut [u8],
    cols: usize,
    p: &SubspaceProjection<'_>,
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
        .map_init(
            || (vec![0.0f32; cols], vec![0.0f32; cols]),
            |(decoded, deltas), row| {
                let (cells, _) = row.as_chunks_mut::<N>();
                for (x, cell) in decoded.iter_mut().zip(cells.iter()) {
                    *x = load(*cell);
                }
                local_deltas(p, decoded, deltas);
                shift_cells(cells, deltas.iter().copied(), &load, &store)
            },
        )
        .reduce(ShiftStats::default, merge)
}

fn project<const N: usize, L, S>(
    bytes: &mut [u8],
    cols: usize,
    p: &SubspaceProjection<'_>,
    wave_blocks: usize,
    load: L,
    store: S,
) -> ShiftStats
where
    L: Fn([u8; N]) -> f32 + Sync,
    S: Fn(f32) -> [u8; N] + Sync,
{
    let row_bytes = cols * N;
    let block_bytes = row_bytes * ROW_BLOCK;
    let mut r = vec![0.0f64; p.k * cols];
    for (wave, chunk) in bytes.chunks(block_bytes * wave_blocks).enumerate() {
        let partials: Vec<Vec<f64>> = chunk
            .par_chunks(block_bytes)
            .enumerate()
            .map(|(block, rows)| {
                let first_row = (wave * wave_blocks + block) * ROW_BLOCK;
                let mut partial = vec![0.0f64; p.k * cols];
                let mut decoded = vec![0.0f32; cols];
                for (offset, row) in rows.chunks_exact(row_bytes).enumerate() {
                    let (cells, _) = row.as_chunks::<N>();
                    for (x, cell) in decoded.iter_mut().zip(cells) {
                        *x = load(*cell);
                    }
                    accumulate_r(p, first_row + offset, &decoded, cols, &mut partial);
                }
                partial
            })
            .collect();
        // Block order, whatever the wave size: the same additions in the same order.
        for partial in &partials {
            r.iter_mut().zip(partial).for_each(|(sum, x)| *sum += x);
        }
    }
    let r32: Vec<f32> = r.iter().map(|&x| x as f32).collect();
    bytes
        .par_chunks_mut(row_bytes)
        .with_min_len((64 * 1024 / row_bytes).max(1))
        .enumerate()
        .map_init(
            || vec![0.0f32; cols],
            |deltas, (i, row)| {
                row_deltas(p, i, &r32, deltas);
                let (cells, _) = row.as_chunks_mut::<N>();
                shift_cells(cells, deltas.iter().copied(), &load, &store)
            },
        )
        .reduce(ShiftStats::default, merge)
}

/// Fused experts `[E, out, in]` (a MoE `experts.down_proj`): [`project_out_subspace`] on
/// each expert's `[out, in]` matrix, in expert order. Every expert has the same shape,
/// dtype and directions, so the checks that pass for expert 0 pass for all: a refusal
/// happens before the first write.
pub fn project_out_experts(
    tensor_bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    p: &SubspaceProjection<'_>,
) -> Result<ShiftStats> {
    let &[experts, rows, cols] = shape else {
        return Err(EngineError::InvalidShape {
            shape: shape.to_vec(),
            reason: "experts fusionnés : tenseur 3D [experts, out, in] attendu".into(),
        });
    };
    if experts == 0 || !tensor_bytes.len().is_multiple_of(experts) {
        return Err(EngineError::InvalidShape {
            shape: shape.to_vec(),
            reason: "taille non divisible par le nombre d'experts".into(),
        });
    }
    let per_expert = tensor_bytes.len() / experts;
    tensor_bytes
        .chunks_mut(per_expert)
        .try_fold(ShiftStats::default(), |total, expert| {
            let stats = project_out_subspace(expert, dtype.clone(), &[rows, cols], p)?;
            Ok(merge(total, stats))
        })
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

    // ------------------------------------------------------------------ rank k

    /// The rank-1 kernel as it was before rank k, kept verbatim: the reference of k = 1.
    fn legacy_project<const N: usize, L, S>(
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
            .enumerate()
            .map(|(i, row)| {
                let coef = alpha * a[i];
                let (cells, _) = row.as_chunks_mut::<N>();
                shift_cells(cells, r32.iter().map(|rj| -(coef * rj)), &load, &store)
            })
            .reduce(ShiftStats::default, merge)
    }

    /// Same for the embedding side.
    fn legacy_project_rows<const N: usize, L, S>(
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
        bytes
            .par_chunks_mut(cols * N)
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

    fn legacy(
        bytes: &mut [u8],
        cols: usize,
        a: &[f32],
        b: &[f32],
        dtype: &DType,
        embedding: bool,
    ) -> ShiftStats {
        match (dtype, embedding) {
            (DType::BF16, false) => legacy_project(bytes, cols, a, b, 0.75, bf16_load, bf16_store),
            (DType::F16, false) => legacy_project(bytes, cols, a, b, 0.75, f16_load, f16_store),
            (_, false) => legacy_project(
                bytes,
                cols,
                a,
                b,
                0.75,
                f32::from_le_bytes,
                f32::to_le_bytes,
            ),
            (DType::BF16, true) => {
                legacy_project_rows(bytes, cols, a, b, 0.75, bf16_load, bf16_store)
            }
            (DType::F16, true) => legacy_project_rows(bytes, cols, a, b, 0.75, f16_load, f16_store),
            (_, true) => legacy_project_rows(
                bytes,
                cols,
                a,
                b,
                0.75,
                f32::from_le_bytes,
                f32::to_le_bytes,
            ),
        }
    }

    /// Values with no exact binary form: every rounding of the kernel is exercised.
    fn wave(len: usize, step: f32, phase: f32) -> Vec<f32> {
        (0..len).map(|k| (k as f32 * step + phase).sin()).collect()
    }

    /// `k` orthonormal directions of length `len`, `[k, len]` row-major (Gram-Schmidt in f64).
    fn orthonormal(k: usize, len: usize) -> Vec<f32> {
        let mut rows: Vec<Vec<f64>> = Vec::new();
        for t in 0..k {
            let mut v: Vec<f64> = (0..len)
                .map(|i| ((i * (t + 2)) as f64 * 0.37 + t as f64).cos())
                .collect();
            for u in &rows {
                let along: f64 = v.iter().zip(u).map(|(x, y)| x * y).sum();
                v.iter_mut().zip(u).for_each(|(x, y)| *x -= along * y);
            }
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            rows.push(v.into_iter().map(|x| x / norm).collect());
        }
        rows.into_iter().flatten().map(|x| x as f32).collect()
    }

    const DTYPES: [DType; 3] = [DType::F32, DType::BF16, DType::F16];

    #[test]
    fn rank_one_is_bit_identical_to_the_previous_kernel() {
        let (rows, cols) = (200, 37); // four row blocks, the last one partial
        let w = wave(rows * cols, 0.618, 0.3);
        for dtype in DTYPES {
            for embedding in [false, true] {
                let len = if embedding { cols } else { rows };
                let (a, b) = (wave(len, 0.3, 1.0), wave(len, 0.7, 2.0));
                let mut expected = encode(&w, &dtype);
                let expected_stats = legacy(&mut expected, cols, &a, &b, &dtype, embedding);
                let p = SubspaceProjection {
                    a: &a,
                    b: &b,
                    alpha: 0.75,
                    k: 1,
                };
                let shape = [rows, cols];
                // The historical entry points, and the rank-k ones with k = 1.
                for subspace in [false, true] {
                    let mut bytes = encode(&w, &dtype);
                    let stats = match (embedding, subspace) {
                        (false, false) => {
                            project_out_tensor(&mut bytes, dtype.clone(), &shape, &a, &b, 0.75)
                        }
                        (true, false) => {
                            project_out_embedding(&mut bytes, dtype.clone(), &shape, &a, &b, 0.75)
                        }
                        (false, true) => {
                            project_out_subspace(&mut bytes, dtype.clone(), &shape, &p)
                        }
                        (true, true) => {
                            project_out_subspace_embedding(&mut bytes, dtype.clone(), &shape, &p)
                        }
                    }
                    .unwrap();
                    assert_eq!(bytes, expected, "{dtype} embedding={embedding}");
                    assert_eq!(stats, expected_stats, "{dtype} embedding={embedding}");
                }
            }
        }
    }

    /// `max |Vᵀ W|` over the k directions and the columns (rows of `w` live in V's space).
    fn max_read(v: &[f32], k: usize, w: &[f32], rows: usize, cols: usize) -> f64 {
        let mut worst = 0.0f64;
        for t in 0..k {
            for j in 0..cols {
                let read: f64 = (0..rows)
                    .map(|i| f64::from(v[t * rows + i]) * f64::from(w[i * cols + j]))
                    .sum();
                worst = worst.max(read.abs());
            }
        }
        worst
    }

    /// One unit in the last place of `x` in `dtype`.
    fn ulp(x: f32, dtype: &DType) -> f32 {
        let mantissa_bits = match dtype {
            DType::F32 => 23,
            DType::F16 => 10,
            _ => 7,
        };
        let exponent = x.abs().max(f32::MIN_POSITIVE).log2().floor() as i32;
        2f32.powi(exponent - mantissa_bits)
    }

    #[test]
    fn an_orthonormal_subspace_is_erased_in_one_pass_and_the_pass_is_idempotent() {
        let (rows, cols, k) = (96, 40, 3);
        let v = orthonormal(k, rows);
        // Noise plus a strong component along each direction of V.
        let mut w = wave(rows * cols, 0.618, 0.3);
        for (index, x) in w.iter_mut().enumerate() {
            let (i, j) = (index / cols, index % cols);
            *x += (0..k)
                .map(|t| 3.0 * v[t * rows + i] * ((j + t) as f32).cos())
                .sum::<f32>();
        }
        let p = SubspaceProjection {
            a: &v,
            b: &v,
            alpha: 1.0,
            k,
        };
        assert!(p.gram_error() < 1e-6, "{}", p.gram_error());
        // What one rounding per element leaves of Vᵀ W: about √rows half-ULPs of |w| ≤ 1.
        for (dtype, tolerance) in [(DType::F32, 1e-5), (DType::F16, 1e-2), (DType::BF16, 5e-2)] {
            let mut bytes = encode(&w, &dtype);
            let before = max_read(&v, k, &decode(&bytes, &dtype), rows, cols);
            assert!(before > 2.0, "the test matrix reads on V: {before}");
            project_out_subspace(&mut bytes, dtype.clone(), &[rows, cols], &p).unwrap();
            let once = decode(&bytes, &dtype);
            let after = max_read(&v, k, &once, rows, cols);
            assert!(after < tolerance, "{dtype}: VᵀW' = {after}");

            // Applied again, nothing is left to remove: no element moves by more than one
            // ULP of the matrix scale (the first pass already rounded each element once).
            project_out_subspace(&mut bytes, dtype.clone(), &[rows, cols], &p).unwrap();
            let twice = decode(&bytes, &dtype);
            let scale = ulp(1.0, &dtype);
            for (x, y) in once.iter().zip(&twice) {
                assert!((x - y).abs() <= scale, "{dtype}: {x} then {y}");
            }
        }
    }

    /// Root mean square distance to `W − A (Bᵀ W)` computed in f64 on the stored weights.
    fn rms_error(
        result: &[f32],
        stored: &[f32],
        rows: usize,
        cols: usize,
        a: &[f32],
        b: &[f32],
    ) -> f64 {
        let k = a.len() / rows;
        let mut sum = 0.0f64;
        let mut r = vec![0.0f64; k * cols];
        for t in 0..k {
            for j in 0..cols {
                r[t * cols + j] = (0..rows)
                    .map(|i| f64::from(b[t * rows + i]) * f64::from(stored[i * cols + j]))
                    .sum();
            }
        }
        for i in 0..rows {
            for j in 0..cols {
                let removed: f64 = (0..k)
                    .map(|t| f64::from(a[t * rows + i]) * r[t * cols + j])
                    .sum();
                let exact = f64::from(stored[i * cols + j]) - removed;
                sum += (f64::from(result[i * cols + j]) - exact).powi(2);
            }
        }
        (sum / (rows * cols) as f64).sqrt()
    }

    #[test]
    fn one_atomic_pass_is_at_least_as_accurate_as_k_rank_one_passes() {
        let (rows, cols, k) = (64, 48, 4);
        let v = orthonormal(k, rows);
        let w = wave(rows * cols, 0.618, 0.3);
        for dtype in DTYPES {
            let stored = decode(&encode(&w, &dtype), &dtype);
            let mut atomic = encode(&w, &dtype);
            let p = SubspaceProjection {
                a: &v,
                b: &v,
                alpha: 1.0,
                k,
            };
            project_out_subspace(&mut atomic, dtype.clone(), &[rows, cols], &p).unwrap();
            // k successive rank-1 operations: k roundings per element, in the order of t.
            let mut sequential = encode(&w, &dtype);
            for t in 0..k {
                let direction = &v[t * rows..(t + 1) * rows];
                project_out_tensor(
                    &mut sequential,
                    dtype.clone(),
                    &[rows, cols],
                    direction,
                    direction,
                    1.0,
                )
                .unwrap();
            }
            let error =
                |bytes: &[u8]| rms_error(&decode(bytes, &dtype), &stored, rows, cols, &v, &v);
            let (one_pass, k_passes) = (error(&atomic), error(&sequential));
            assert!(
                one_pass <= k_passes,
                "{dtype}: atomic {one_pass:e} > sequential {k_passes:e}"
            );
            assert_ne!(
                atomic, sequential,
                "{dtype}: the two paths round differently"
            );
        }
    }

    #[test]
    fn rank_k_is_identical_with_any_thread_count_and_any_wave_partition() {
        let (rows, cols, k) = (300, 33, 4); // five row blocks
        let w = wave(rows * cols, 0.618, 0.0);
        let (a, b) = (wave(k * rows, 0.3, 1.0), wave(k * rows, 0.07, 2.0));
        let p = SubspaceProjection {
            a: &a,
            b: &b,
            alpha: 0.5,
            k,
        };
        let run = |threads: usize, wave_blocks: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut bytes = encode(&w, &DType::F32);
                    project(
                        &mut bytes,
                        cols,
                        &p,
                        wave_blocks,
                        f32::from_le_bytes,
                        f32::to_le_bytes,
                    );
                    bytes
                })
        };
        let reference = run(1, 1);
        for (threads, wave_blocks) in [(2, 1), (8, 1), (1, 2), (8, 3), (2, 5), (8, 64)] {
            assert_eq!(
                run(threads, wave_blocks),
                reference,
                "{threads} threads, {wave_blocks} blocks"
            );
        }
        let mut public = encode(&w, &DType::F32);
        project_out_subspace(&mut public, DType::F32, &[rows, cols], &p).unwrap();
        assert_eq!(public, reference);

        // The embedding side has no reduction across rows: threads cannot change it.
        let (a, b) = (wave(k * cols, 0.3, 1.0), wave(k * cols, 0.07, 2.0));
        let p = SubspaceProjection {
            a: &a,
            b: &b,
            alpha: 0.5,
            k,
        };
        let runs: Vec<Vec<u8>> = [1, 2, 8]
            .iter()
            .map(|&threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        let mut bytes = encode(&w, &DType::BF16);
                        project_out_subspace_embedding(&mut bytes, DType::BF16, &[rows, cols], &p)
                            .unwrap();
                        bytes
                    })
            })
            .collect();
        assert!(runs.windows(2).all(|pair| pair[0] == pair[1]));
    }

    #[test]
    fn rank_k_matches_plain_loops_bit_for_bit_on_both_sides() {
        let (rows, cols, k) = (130, 6, 3);
        let w = weights(rows, cols);
        // Dyadic directions too: every f64 sum is exact, whatever its order.
        let dyadic = |len: usize, m: usize| -> Vec<f32> {
            (0..k * len)
                // A multiplicative hash: no period shared with the weights or the rows.
                .map(|i| (((i * 2_654_435_761) >> 8) % m) as f32 / 4.0 - (m / 2) as f32 / 4.0)
                .collect()
        };
        for dtype in DTYPES {
            let stored = decode(&encode(&w, &dtype), &dtype);

            // Matrix side: directions index the rows.
            let (a, b) = (dyadic(rows, 5), dyadic(rows, 3));
            let mut expected = Vec::new();
            let r: Vec<f32> = (0..k * cols)
                .map(|tj| {
                    let (t, j) = (tj / cols, tj % cols);
                    (0..rows)
                        .map(|i| f64::from(b[t * rows + i]) * f64::from(stored[i * cols + j]))
                        .sum::<f64>() as f32
                })
                .collect();
            for i in 0..rows {
                for j in 0..cols {
                    let mut sum = 0.5 * a[i] * r[j];
                    for t in 1..k {
                        sum += 0.5 * a[t * rows + i] * r[t * cols + j];
                    }
                    expected.push(stored[i * cols + j] + -sum);
                }
            }
            let p = SubspaceProjection {
                a: &a,
                b: &b,
                alpha: 0.5,
                k,
            };
            let mut bytes = encode(&w, &dtype);
            let stats = project_out_subspace(&mut bytes, dtype.clone(), &[rows, cols], &p).unwrap();
            assert_eq!(bytes, encode(&expected, &dtype), "{dtype}");
            assert!(
                stats.elements_modified > 0 && stats.nan_or_inf_produced == 0,
                "{dtype}: {stats:?}"
            );

            // Embedding side: directions index the columns, each row on its own.
            let (a, b) = (dyadic(cols, 5), dyadic(cols, 3));
            let mut expected = Vec::new();
            for row in stored.chunks_exact(cols) {
                let coefs: Vec<f32> = (0..k)
                    .map(|t| {
                        let read = row
                            .iter()
                            .zip(&b[t * cols..])
                            .map(|(x, bj)| f64::from(*bj) * f64::from(*x))
                            .sum::<f64>() as f32;
                        0.5 * read
                    })
                    .collect();
                for (j, x) in row.iter().enumerate() {
                    let mut sum = coefs[0] * a[j];
                    for t in 1..k {
                        sum += coefs[t] * a[t * cols + j];
                    }
                    expected.push(x + -sum);
                }
            }
            let p = SubspaceProjection {
                a: &a,
                b: &b,
                alpha: 0.5,
                k,
            };
            let mut bytes = encode(&w, &dtype);
            project_out_subspace_embedding(&mut bytes, dtype.clone(), &[rows, cols], &p).unwrap();
            assert_eq!(bytes, encode(&expected, &dtype), "{dtype} embedding");
        }

        // An orthonormal subspace leaves no component of any embedding row along it.
        let v = orthonormal(2, cols);
        let p = SubspaceProjection {
            a: &v,
            b: &v,
            alpha: 1.0,
            k: 2,
        };
        let mut bytes = encode(&w, &DType::F32);
        project_out_subspace_embedding(&mut bytes, DType::F32, &[rows, cols], &p).unwrap();
        for row in decode(&bytes, &DType::F32).chunks_exact(cols) {
            for direction in v.chunks_exact(cols) {
                let read: f32 = row.iter().zip(direction).map(|(x, d)| x * d).sum();
                assert!(read.abs() < 1e-5, "W'v = {read}");
            }
        }
    }

    #[test]
    fn decoded_row_helpers_reproduce_the_byte_kernel() {
        let (rows, cols, k) = (150, 9, 3);
        let w = wave(rows * cols, 0.618, 0.3);
        let (a, b) = (wave(k * rows, 0.3, 1.0), wave(k * rows, 0.07, 2.0));
        let p = SubspaceProjection {
            a: &a,
            b: &b,
            alpha: 0.5,
            k,
        };
        let mut bytes = encode(&w, &DType::F32);
        project_out_subspace(&mut bytes, DType::F32, &[rows, cols], &p).unwrap();

        // What another container (a quantized file) would do: decode whole row blocks,
        // accumulate block by block, merge in block order, then update the decoded rows.
        let mut r = vec![0.0f64; k * cols];
        for (block, chunk) in w.chunks(ROW_BLOCK * cols).enumerate() {
            let mut partial = vec![0.0f64; k * cols];
            accumulate_r(&p, block * ROW_BLOCK, chunk, cols, &mut partial);
            r.iter_mut().zip(&partial).for_each(|(s, x)| *s += x);
        }
        let r32: Vec<f32> = r.iter().map(|&x| x as f32).collect();
        let mut decoded = w.clone();
        let (head, tail) = decoded.split_at_mut(70 * cols);
        project_rows_inplace(&p, 0, head, cols, &r32);
        project_rows_inplace(&p, 70, tail, cols, &r32);
        assert_eq!(encode(&decoded, &DType::F32), bytes);
    }

    #[test]
    fn gram_error_tells_a_projector_from_the_rest() {
        let v = orthonormal(3, 16);
        let projector = SubspaceProjection {
            a: &v,
            b: &v,
            alpha: 1.0,
            k: 3,
        };
        assert!(projector.gram_error() < 1e-6);
        // Twice as long: BᵀA = 2 I, so ‖BᵀA − I‖_F = √3.
        let doubled: Vec<f32> = v.iter().map(|x| 2.0 * x).collect();
        let scaled = SubspaceProjection {
            a: &doubled,
            b: &v,
            alpha: 1.0,
            k: 3,
        };
        assert!((scaled.gram_error() - 3f64.sqrt()).abs() < 1e-5);
        // Oblique but still a projector: a = b + something b never reads.
        let (b, mut a) = (v[..16].to_vec(), v[..16].to_vec());
        a.iter_mut()
            .zip(&v[16..32])
            .for_each(|(x, y)| *x += 3.0 * y);
        assert!(SubspaceProjection::rank_one(&a, &b, 1.0).gram_error() < 1e-5);
        assert_eq!(projector.direction_len(), 16);
    }

    #[test]
    fn every_rank_k_check_runs_before_the_first_write() {
        let mut bytes = encode(&[1.0; 12], &DType::F32);
        let original = bytes.clone();
        let two = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0]; // k = 2 on three rows
        let cases: [(&[f32], &[f32], usize, f32); 6] = [
            (&two, &two, 0, 0.5),           // no direction
            (&two, &two, 3, 0.5),           // six values are not three directions of three
            (&two, &two[..3], 2, 0.5),      // a and b of different ranks
            (&two[..4], &two[..4], 2, 0.5), // directions of the wrong length
            (&two, &two, 2, 1.5),           // alpha out of [0, 1]
            (&[1.0, 0.0, 0.0, 0.0, f32::NAN, 0.0], &two, 2, 0.5),
        ];
        for (a, b, k, alpha) in cases {
            let p = SubspaceProjection { a, b, alpha, k };
            assert!(
                project_out_subspace(&mut bytes, DType::F32, &[3, 4], &p).is_err(),
                "k = {k}, alpha = {alpha}"
            );
            assert_eq!(bytes, original);
        }
        // Overflow is counted, as for a shift: the executor turns it into a rollback.
        let mut huge = encode(&[f32::MAX; 12], &DType::F32);
        let (a, b) = ([-1.0f32, -1.0, -1.0, 1.0, 1.0, 1.0], [1.0f32; 6]);
        let p = SubspaceProjection {
            a: &a,
            b: &b,
            alpha: 1.0,
            k: 2,
        };
        let stats = project_out_subspace(&mut huge, DType::F32, &[3, 4], &p).unwrap();
        assert!(stats.nan_or_inf_produced > 0);
    }

    /// Throughput of the atomic pass, for the README. Release only:
    /// `cargo test --release --lib rank_k_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn rank_k_throughput() {
        let best_ms = |run: &mut dyn FnMut()| {
            (0..3)
                .map(|_| {
                    let started = std::time::Instant::now();
                    run();
                    started.elapsed().as_secs_f64() * 1e3
                })
                .fold(f64::INFINITY, f64::min)
        };
        // down_proj of Qwen2.5-0.5B, then a slice of its embeddings.
        for (rows, cols, embedding) in [(896, 4864, false), (32768, 896, true)] {
            let len = if embedding { cols } else { rows };
            let w = wave(rows * cols, 0.618, 0.3);
            for dtype in [DType::BF16, DType::F32] {
                let original = encode(&w, &dtype);
                let mib = original.len() as f64 / (1u64 << 20) as f64;
                for k in [1usize, 4, 16, 64] {
                    let v = orthonormal(k, len);
                    let p = SubspaceProjection {
                        a: &v,
                        b: &v,
                        alpha: 1.0,
                        k,
                    };
                    let mut bytes = original.clone();
                    let atomic = best_ms(&mut || {
                        bytes.copy_from_slice(&original);
                        let run = match embedding {
                            true => project_out_subspace_embedding,
                            false => project_out_subspace,
                        };
                        run(&mut bytes, dtype.clone(), &[rows, cols], &p).unwrap();
                    });
                    let sequential = best_ms(&mut || {
                        bytes.copy_from_slice(&original);
                        for direction in v.chunks_exact(len) {
                            let run = match embedding {
                                true => project_out_embedding,
                                false => project_out_tensor,
                            };
                            let shape = [rows, cols];
                            run(&mut bytes, dtype.clone(), &shape, direction, direction, 1.0)
                                .unwrap();
                        }
                    });
                    println!(
                        "[{rows}, {cols}] {dtype} k={k}: atomic {atomic:.1} ms ({:.0} MiB/s), \
                         {k} rank-1 passes {sequential:.1} ms",
                        mib / (atomic / 1e3)
                    );
                }
            }
        }
    }
}
