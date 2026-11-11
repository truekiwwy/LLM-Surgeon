//! Per-tensor static metrics, computed off the UI thread on a deterministic
//! strided sample (the sample seed and strides are fixed, so two runs of the
//! same file always display the same numbers).
//!
//! ```text
//! compute(dtype, shape, bytes) -> Stats {
//!   l2 (scaled back to the full tensor), min/max, mean, |mean|,
//!   excess kurtosis (outlier weight), r_eff ≈ participation ratio of the
//!   squared singular values estimated through a seeded Rademacher projection
//!   of a sampled rows × cols matrix (k = 16), and the L2 profile of the
//!   sampled rows (for the sparkline).
//! }
//! ```
//!
//! Everything is approximate by design and flagged as such in the UI
//! (`sampled` badge): it must stay instant even on a 70 B file mapped from a
//! 40 GB checkpoint.

use crate::model::dtype::DType;

/// Row cap of the sampled matrix.
const MAX_ROWS: usize = 64;
/// Column cap of the sampled matrix.
const MAX_COLS: usize = 256;
/// Basis size of the random projection used for `r_eff`.
pub const BASIS: usize = 16;

/// Metrics of one tensor.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Stats {
    /// Stored elements.
    pub numel: u64,
    /// Decoded elements actually observed.
    pub sampled: u64,
    /// L2 norm scaled back to the whole tensor.
    pub l2: f64,
    pub min: f32,
    pub max: f32,
    pub mean: f64,
    pub abs_mean: f64,
    /// Excess kurtosis: 0 for a normal sample, large when outliers dominate.
    pub kurtosis: f64,
    /// Estimated effective rank (participation ratio, in `[1, BASIS]`).
    pub r_eff: f64,
    /// L2 of each sampled row, for the magnitude sparkline.
    pub row_profile: Vec<f64>,
    /// False when the dtype cannot be decoded or the bytes are unreadable.
    pub available: bool,
}

/// Computes the metrics of one tensor from its raw bytes.
pub fn compute(dtype: &DType, shape: &[u64], bytes: &[u8]) -> Stats {
    let numel: u64 = shape.iter().product();
    if numel == 0 || !dtype.decodable() || bytes.is_empty() {
        return Stats {
            numel,
            available: false,
            ..Stats::default()
        };
    }
    let cols_u = shape.last().copied().unwrap_or(numel).max(1);
    let cols_u = cols_u.min(1 << 32) as usize;
    let rows_u = (numel / u64::try_from(cols_u).unwrap_or(1)).max(1);
    let (rows_s, row_stride) = stride_for(rows_u, MAX_ROWS);
    let (cols_s, col_stride) = stride_for(cols_u as u64, MAX_COLS);

    let mut matrix = Vec::with_capacity(rows_s * cols_s);
    let mut scratch = Vec::new();
    for r in 0..rows_s {
        // Element offset of the row, strided deterministically.
        let row = r * row_stride as usize * cols_u;
        decode_range(
            dtype,
            bytes,
            row,
            cols_u,
            cols_s,
            col_stride as usize,
            &mut matrix,
            &mut scratch,
        );
    }

    scalar_stats(&matrix, numel, rows_u, rows_s, cols_u as u64, cols_s)
}

fn stride_for(count: u64, cap: usize) -> (usize, u64) {
    let count = count.max(1);
    if count <= cap as u64 {
        (count as usize, 1)
    } else {
        (cap, count / cap as u64)
    }
}

/// Decodes `cols_s` strided columns of one row into `out`.
#[allow(clippy::too_many_arguments)]
fn decode_range(
    dtype: &DType,
    bytes: &[u8],
    row_start: usize,
    cols: usize,
    cols_s: usize,
    col_stride: usize,
    out: &mut Vec<f32>,
    scratch: &mut Vec<f32>,
) {
    if matches!(dtype, DType::F32 | DType::F16 | DType::Bf16) {
        let width = if *dtype == DType::F32 { 4 } else { 2 };
        for c in 0..cols_s {
            let index = row_start + c * col_stride;
            if index >= row_start + cols {
                break;
            }
            let at = (index as u64 * width as u64) as usize;
            out.push(read_element(dtype, bytes, at));
        }
        return;
    }
    let Some((elems, _)) = dtype.block() else {
        return;
    };
    let (elems, cols, col_stride) = (elems as usize, cols, col_stride);
    // Decode the blocks covered by this row once, then stride.
    let block = row_start / elems;
    let end = (row_start + cols - 1) / elems;
    let covered = (end - block + 1) * elems;
    let first = block * elems;
    dtype.decode_range(bytes, (block * elems) as u64, covered as u64, scratch);
    // `scratch` starts at element `first`: the row's element `row_start + index`
    // sits at relative position `row_start - first + index`.
    let base = row_start - first;
    for c in 0..cols_s {
        let index = c * col_stride;
        if index >= cols {
            break;
        }
        out.push(scratch.get(base + index).copied().unwrap_or(0.0));
    }
}

fn read_element(dtype: &DType, bytes: &[u8], at: usize) -> f32 {
    match dtype {
        DType::F32 if at + 4 <= bytes.len() => {
            f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        }
        DType::F16 if at + 2 <= bytes.len() => {
            half::f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32()
        }
        DType::Bf16 if at + 2 <= bytes.len() => {
            half::bf16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32()
        }
        _ => 0.0,
    }
}

fn scalar_stats(
    matrix: &[f32],
    numel: u64,
    rows: u64,
    rows_s: usize,
    cols: u64,
    cols_s: usize,
) -> Stats {
    let n = matrix.len();
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut sum_q = 0.0f64;
    for &v in matrix {
        if v.is_nan() {
            continue;
        }
        min = min.min(v);
        max = max.max(v);
        let a = f64::from(v);
        sum += a;
        sum_abs += a.abs();
        sum_sq += a * a;
        sum_q += a * a * a * a;
    }
    if n == 0 || min > max {
        return Stats {
            numel,
            sampled: 0,
            available: false,
            ..Stats::default()
        };
    }
    let n = n as f64;
    let mean = sum / n;
    let m2 = sum_sq / n;
    let m4 = sum_q / n;
    let kurtosis = if m2 > 0.0 { m4 / (m2 * m2) - 3.0 } else { 0.0 };
    // The sample covers a known fraction of the tensor: scale the energy back.
    let fraction = (rows_s as f64 / rows as f64) * (cols_s as f64 / cols as f64);
    let fraction = fraction.clamp(f64::EPSILON, 1.0);
    let l2 = (sum_sq / fraction).sqrt();

    let row_profile = matrix
        .chunks(cols_s.max(1))
        .map(|row| {
            row.iter()
                .map(|v| f64::from(*v) * f64::from(*v))
                .sum::<f64>()
                .sqrt()
        })
        .collect();

    let r_eff = effective_rank(matrix, cols_s);
    Stats {
        numel,
        sampled: matrix.len() as u64,
        l2,
        min,
        max,
        mean,
        abs_mean: sum_abs / n,
        kurtosis,
        r_eff,
        row_profile,
        available: true,
    }
}

/// Participation ratio of the squared singular values of the sampled matrix,
/// through a seeded Rademacher projection: `r_eff = (Σλ)² / Σλ²` where the `λ`
/// are the eigenvalues of the `k × k` Gram matrix of `X·R`.
fn effective_rank(matrix: &[f32], cols: usize) -> f64 {
    let rows = matrix.len() / cols.max(1);
    if rows < 2 || cols < 2 {
        return 1.0;
    }
    // Seeded xorshift*: identical projections for identical inputs. The
    // projection matrix is fixed and shared by every row: Y = X·R.
    let mut state = 0x243F_6A88_85A3_08D3u64;
    let mut projection = vec![0.0f64; cols * BASIS];
    for value in &mut projection {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        *value = f64::from(((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 63) as i8) * 2 - 1);
    }
    let mut y = vec![0.0f64; rows * BASIS];
    for (r, row) in matrix.chunks(cols.max(1)).enumerate() {
        for (k, y_cell) in y[r * BASIS..(r + 1) * BASIS].iter_mut().enumerate() {
            let mut acc = 0.0f64;
            for (c, &v) in row.iter().enumerate() {
                if c >= cols {
                    break;
                }
                acc += f64::from(v) * projection[c * BASIS + k];
            }
            *y_cell = acc;
        }
    }
    let mut gram = [[0.0f64; BASIS]; BASIS];
    for k in 0..BASIS {
        for j in 0..=k {
            let dot = (0..rows).map(|r| y[r * BASIS + k] * y[r * BASIS + j]).sum();
            gram[k][j] = dot;
            gram[j][k] = dot;
        }
    }
    let eigenvalues = jacobi_eigenvalues(&mut gram);
    let total: f64 = eigenvalues.iter().map(|l| l.max(0.0)).sum();
    let squares: f64 = eigenvalues.iter().map(|l| l.max(0.0) * l.max(0.0)).sum();
    if squares <= 0.0 {
        1.0
    } else {
        (total * total / squares).clamp(1.0, BASIS as f64)
    }
}

/// Eigenvalues of a small symmetric matrix, cyclic Jacobi, deterministic.
/// The rotations are inherently index-matrix work: range loops read best here.
#[allow(clippy::needless_range_loop)]
fn jacobi_eigenvalues(a: &mut [[f64; BASIS]; BASIS]) -> Vec<f64> {
    for _ in 0..40 {
        let mut off = 0.0f64;
        for p in 0..BASIS {
            for q in (p + 1)..BASIS {
                off += a[p][q] * a[p][q];
            }
        }
        if off < 1e-16 {
            break;
        }
        for p in 0..BASIS {
            for q in (p + 1)..BASIS {
                let apq = a[p][q];
                if apq.abs() < 1e-15 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                // theta == 0 means equal diagonals: the optimal rotation is 45°.
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = c * t;
                let (app, aqq) = (a[p][p], a[q][q]);
                a[p][p] = app - t * apq;
                a[q][q] = aqq + t * apq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                for k in 0..BASIS {
                    if k == p || k == q {
                        continue;
                    }
                    let (akp, akq) = (a[k][p], a[k][q]);
                    a[k][p] = c * akp - s * akq;
                    a[p][k] = a[k][p];
                    a[k][q] = s * akp + c * akq;
                    a[q][k] = a[k][q];
                }
            }
        }
    }
    (0..BASIS).map(|i| a[i][i]).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn scalar_stats_match_a_reference_on_a_small_matrix() {
        // 4 × 8 known matrix.
        let mut values = Vec::new();
        for r in 0..4u64 {
            for c in 0..8u64 {
                values.push(((r * 8 + c) as f32 * 0.25 - 4.0).tanh());
            }
        }
        let bytes = f32_bytes(&values);
        let stats = compute(&DType::F32, &[4, 8], &bytes);
        assert!(stats.available);
        assert_eq!(stats.numel, 32);
        assert_eq!(stats.sampled, 32);
        let reference_l2 = values
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            .sqrt();
        assert!((stats.l2 - reference_l2).abs() < 1e-9);
        let reference_mean = values.iter().map(|v| f64::from(*v)).sum::<f64>() / 32.0;
        assert!((stats.mean - reference_mean).abs() < 1e-9);
        assert_eq!(
            stats.min,
            values.iter().copied().fold(f32::INFINITY, f32::min)
        );
        assert_eq!(
            stats.max,
            values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        );
        assert_eq!(stats.row_profile.len(), 4);
    }

    #[test]
    fn large_tensors_are_sampled_deterministically() {
        let values: Vec<f32> = (0..128 * 512)
            .map(|i| ((i as f64 * 0.01).sin() * 0.5) as f32)
            .collect();
        let bytes = f32_bytes(&values);
        let a = compute(&DType::F32, &[128, 512], &bytes);
        let b = compute(&DType::F32, &[128, 512], &bytes);
        assert_eq!(a, b, "two runs must agree bit for bit");
        assert_eq!(a.sampled, 64 * 256);
        assert!(a.l2 > 0.0);
    }

    #[test]
    fn r_eff_sits_between_1_and_the_basis() {
        // Rank-1 matrix: every row is the same vector.
        let base: Vec<f32> = (0..32).map(|i| (i as f32).sin()).collect();
        let mut values = Vec::new();
        for _ in 0..16 {
            values.extend_from_slice(&base);
        }
        let stats = compute(&DType::F32, &[16, 32], &f32_bytes(&values));
        assert!(
            (stats.r_eff - 1.0).abs() < 0.2,
            "rank 1, got {}",
            stats.r_eff
        );

        // Full-rank noise: far above 1.
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut noise = Vec::new();
        for _ in 0..16 * 32 {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            noise.push(((state >> 11) as f64 / (1u64 << 53) as f64 - 0.5) as f32);
        }
        let stats = compute(&DType::F32, &[16, 32], &f32_bytes(&noise));
        assert!(stats.r_eff > 4.0, "full rank, got {}", stats.r_eff);
    }

    #[test]
    fn unsupported_and_empty_tensors_are_reported_unavailable() {
        let stats = compute(&DType::Other("IQ2_XXS".into()), &[2, 2], &[0, 1, 2, 3]);
        assert!(!stats.available);
        let stats = compute(&DType::F32, &[0], &[]);
        assert!(!stats.available);
    }

    #[test]
    fn q8_0_metrics_match_a_reference() {
        // Two blocks, scale 0.5, qs 1..32: exact values.
        let mut bytes = Vec::new();
        for _ in 0..2 {
            bytes.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
            for i in 0..32u8 {
                bytes.push(i + 1);
            }
        }
        let stats = compute(&DType::Q8_0, &[2, 32], &bytes);
        assert!(stats.available);
        // Every decoded value is 0.5 * (i+1) for i in 0..32, per row.
        assert_eq!(stats.min, 0.5);
        assert_eq!(stats.max, 16.0);
        let reference_l2: f64 = (0..2)
            .map(|_| (1..=32).map(|i| (0.5 * f64::from(i)).powi(2)).sum::<f64>())
            .sum::<f64>()
            .sqrt();
        assert!((stats.l2 - reference_l2).abs() < 1e-9);
    }
}
