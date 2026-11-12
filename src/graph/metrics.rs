//! Per-matrix spectral metrics and streaming statistics for `surgeon graph`.
//!
//! Every matrix is read through [`NodeSource`], so the values used here are exactly the
//! values `rsvd` sees (Gemma `1 + gamma` folded, non-finite entries replaced by zero after
//! being counted). The randomized SVD gives a range certificate
//! `range_energy = ||Q^T A||_F^2 / ||A||_F^2`; the residual energy of that range is
//! `1 - range_energy`, and every interval below is derived from it without assuming that
//! the singular values are exact.
//!
//! Statistics (mean, M2, M3, M4, kurtosis) use Pébay's one-pass update, in row-major order,
//! so they do not depend on the rayon thread count. Full-spectrum diagnostics are opt-in:
//! the existing Gram SVD is quadratic in memory and refuses both dimensions above 1024,
//! so they are skipped (with a warning) outside those limits.

use std::sync::atomic::AtomicBool;

use crate::linalg::gram::thin_svd;
use crate::linalg::jacobi::MAX_N;
use crate::surgery::rsvd::{rsvd, Rsvd, RsvdParams};

use super::error::{self, GraphError, Result};
use super::model::{Counters, ModelData, NodeSource, TensorRef};
use super::GraphRole;

/// Above this many elements the full-spectrum matrix is not materialised (f32 -> f64).
pub const FULL_SPECTRUM_PRODUCT_CAP: usize = 4_000_000;

/// The residual-stream direction basis of a node, row-major `rows x cols`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Basis {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

impl Basis {
    pub fn dot(&self, row: usize, vector: &[f32]) -> f64 {
        let start = row * self.cols;
        self.values[start..start + self.cols]
            .iter()
            .zip(vector)
            .map(|(&a, &b)| f64::from(a) * f64::from(b))
            .sum()
    }

    /// Fraction of `vector`'s energy that lies in this subspace.
    pub fn captured_energy(&self, vector: &[f32]) -> f64 {
        let norm2: f64 = vector.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
        if norm2 == 0.0 {
            return 0.0;
        }
        let captured: f64 = (0..self.rows)
            .map(|row| {
                let dot = self.dot(row, vector);
                dot * dot
            })
            .sum();
        (captured / norm2).clamp(0.0, 1.0)
    }

    fn cross(&self, other: &Basis) -> Vec<f64> {
        let rows = self.rows.min(other.rows);
        let mut out = vec![0.0; rows * rows];
        for i in 0..rows {
            for j in 0..rows {
                let mut sum = 0.0;
                let (a, b) = (
                    &self.values[i * self.cols..(i + 1) * self.cols],
                    &other.values[j * other.cols..(j + 1) * other.cols],
                );
                for (&x, &y) in a.iter().zip(b) {
                    sum += f64::from(x) * f64::from(y);
                }
                out[i * rows + j] = sum;
            }
        }
        out
    }
}

/// Alignment between a writer's write basis and a reader's read basis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Similarity {
    /// `||B_w B_r^T||_F^2 / r`, chance-corrected below.
    pub alignment: f64,
    /// `r / hidden`: the alignment of two unrelated random subspaces.
    pub chance: f64,
    /// `(alignment - chance) / (1 - chance)`.
    pub alignment_norm: f64,
    /// Energy-weighted overlap, scaled by `hidden` so 1 is chance level.
    pub coupling: f64,
    /// Cosine of the smallest principal angle (`sigma_max(B_w B_r^T)`).
    pub cos_theta: f64,
    pub rank_used: usize,
}

/// `B_w` rows are write directions, `B_r` rows are read directions, both in `R^hidden`.
pub fn similarity(
    write: &Basis,
    read: &Basis,
    hidden: usize,
    write_energy: &[f32],
    read_energy: &[f32],
) -> Similarity {
    let rank_used = write.rows.min(read.rows);
    if rank_used == 0 || hidden == 0 {
        return Similarity {
            alignment: 0.0,
            chance: 0.0,
            alignment_norm: 0.0,
            coupling: 0.0,
            cos_theta: 0.0,
            rank_used: 0,
        };
    }
    let cross = write.cross(read);
    let mut sum = 0.0;
    for value in &cross {
        sum += value * value;
    }
    let alignment = sum / rank_used as f64;
    let chance = rank_used as f64 / hidden as f64;
    let alignment_norm = if chance < 1.0 {
        (alignment - chance) / (1.0 - chance)
    } else {
        0.0
    };
    let mut coupling_raw = 0.0;
    for i in 0..rank_used {
        let wi = write_energy.get(i).copied().unwrap_or(0.0) as f64;
        for j in 0..rank_used {
            let rj = read_energy.get(j).copied().unwrap_or(0.0) as f64;
            let value = cross[i * rank_used + j];
            coupling_raw += wi * rj * value * value;
        }
    }
    let coupling = (coupling_raw * hidden as f64).clamp(0.0, hidden as f64);
    let cos_theta = sigma_max_power(&cross, rank_used);
    Similarity {
        alignment,
        chance,
        alignment_norm,
        coupling,
        cos_theta,
        rank_used,
    }
}

/// Largest singular value of a small square matrix, by deterministic power iteration.
fn sigma_max_power(matrix: &[f64], n: usize) -> f64 {
    let mut vector = vec![1.0 / (n as f64).sqrt(); n];
    let mut rayleigh = 0.0;
    for _ in 0..16 {
        // y = M v, then z = M^T y = M^T M v.
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                y[i] += matrix[i * n + j] * vector[j];
            }
        }
        let mut z = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                z[i] += matrix[j * n + i] * y[j];
            }
        }
        let norm = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return 0.0;
        }
        for (v, zi) in vector.iter_mut().zip(&z) {
            *v = zi / norm;
        }
        rayleigh = vector.iter().zip(&z).map(|(&v, &zi)| v * zi).sum::<f64>();
    }
    rayleigh.max(0.0).sqrt().clamp(0.0, 1.0)
}

/// One-pass moments. The `M*` fields are sums of powers of deviations.
#[derive(Debug, Clone, Copy, Default)]
struct Moments {
    n: u64,
    mean: f64,
    m2: f64,
    m3: f64,
    m4: f64,
}

impl Moments {
    fn push(&mut self, x: f64) {
        let n1 = self.n as f64;
        self.n += 1;
        let n = self.n as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * n1;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0) + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
        self.mean += delta_n;
    }

    /// Population kurtosis (normal distribution: 3).
    fn kurtosis(&self) -> f64 {
        if self.n < 2 || self.m2 <= 0.0 {
            0.0
        } else {
            self.n as f64 * self.m4 / (self.m2 * self.m2)
        }
    }
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct ChannelStats {
    pub count: usize,
    /// Median of the per-channel L2 norms along the residual axis.
    pub norm_median: f64,
    pub norm_max: f64,
    pub norm_argmax: usize,
    /// Largest per-channel kurtosis (0 when a channel has fewer than 4 values).
    pub kurtosis_max: f64,
    /// Channels whose kurtosis exceeds `--kurtosis-threshold`.
    pub outliers: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct NodeMetrics {
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    pub frobenius_norm: f64,
    /// `sum(sigma^2) / ||A||_F^2`: energy captured by the rank-r approximation.
    pub energy: f64,
    pub residual_energy: f64,
    /// `||Q^T A||_F^2 / ||A||_F^2`: the range certificate.
    pub range_energy: f64,
    pub rel_error: f64,
    pub sigma1: Option<f64>,
    pub sigma1_lower: Option<f64>,
    pub sigma1_upper: Option<f64>,
    /// `||A||_F^2 / sigma1^2` (stable rank), with the certified interval.
    pub d_spec: Option<f64>,
    pub d_spec_lower: Option<f64>,
    pub d_spec_upper: Option<f64>,
    /// Entropy effective rank `exp(H(sigma^2))`, certified from the tail energy.
    pub erank_energy_lower: f64,
    pub erank_energy_upper: f64,
    /// `exp(H(sigma / ||sigma||_1))`: only with a full spectrum.
    pub erank_l1: Option<f64>,
    pub full_spectrum: bool,
    /// Population kurtosis of every matrix entry (normal: 3).
    pub kurtosis: f64,
    pub kurtosis_excess: f64,
    pub channels: ChannelStats,
    /// Fraction of the optional `--vector` captured by this node's basis.
    pub e_write: Option<f64>,
    pub e_read: Option<f64>,
    pub non_finite: u64,
    pub singular_values: Vec<f32>,
}

impl NodeMetrics {
    pub fn is_outlier(&self, threshold: f64) -> bool {
        self.kurtosis > threshold || !self.channels.outliers.is_empty()
    }
}

/// Everything the graph needs from one node after the SVD.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NodeAnalysis {
    pub metrics: NodeMetrics,
    pub basis: Basis,
    pub energy_weights: Vec<f32>,
}

pub struct AnalyzeOptions<'a> {
    pub params: RsvdParams,
    pub full_spectrum_max_dim: usize,
    pub kurtosis_threshold: f64,
    pub vector: Option<&'a [f32]>,
    pub cancel: &'a AtomicBool,
    pub external: Option<&'a AtomicBool>,
}

/// Runs the randomized SVD, the streaming statistics and the optional full spectrum.
pub fn analyze(
    model: &ModelData,
    tensor: &TensorRef,
    hidden: usize,
    options: &AnalyzeOptions<'_>,
) -> Result<NodeAnalysis> {
    let gamma = model.gamma_for(tensor);
    let source = NodeSource::new(
        model,
        tensor,
        gamma.as_deref(),
        options.cancel,
        options.external,
    );
    let counters = source.pre_scan()?;
    let svd = rsvd(&source, &options.params, options.cancel).map_err(error::surgery)?;
    let (basis, energy_weights) = basis_of(tensor.role, &svd)?;
    let stats = streaming_stats(&source, tensor, options.kurtosis_threshold)?;
    let full = full_spectrum(&source, tensor, options, hidden, &svd)?;
    let metrics = build_metrics(&svd, tensor, counters, stats, full, &basis, options.vector)?;
    Ok(NodeAnalysis {
        metrics,
        basis,
        energy_weights,
    })
}

fn basis_of(role: GraphRole, svd: &Rsvd) -> Result<(Basis, Vec<f32>)> {
    let values = match role {
        GraphRole::Embed => svd.v.clone(),
        GraphRole::AttnOut | GraphRole::MlpOut => {
            // `u` is m x r (row-major): transpose to r rows of length m.
            let mut out = vec![0.0f32; svd.r * svd.m];
            for row in 0..svd.m {
                for column in 0..svd.r {
                    out[column * svd.m + row] = svd.u[row * svd.r + column];
                }
            }
            out
        }
        _ => svd.v.clone(),
    };
    let cols = match role {
        GraphRole::Embed => svd.n,
        GraphRole::AttnOut | GraphRole::MlpOut => svd.m,
        _ => svd.n,
    };
    let total: f64 = svd.s.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    let weights = match total > 0.0 {
        true => svd
            .s
            .iter()
            .map(|&x| (f64::from(x) * f64::from(x) / total) as f32)
            .collect(),
        false => vec![0.0; svd.r],
    };
    Ok((
        Basis {
            rows: svd.r,
            cols,
            values,
        },
        weights,
    ))
}

struct StreamingStats {
    global: Moments,
    channels: ChannelStats,
}

fn streaming_stats(
    source: &NodeSource<'_>,
    tensor: &TensorRef,
    threshold: f64,
) -> Result<StreamingStats> {
    let (m, n) = (tensor.m(), tensor.n());
    // Writers store residual channels as rows; readers (and embeddings) as columns.
    let channels_are_columns = !matches!(tensor.role, GraphRole::AttnOut | GraphRole::MlpOut);
    let channel_count = match channels_are_columns {
        true => n,
        false => m,
    };
    let mut global = Moments::default();
    let mut rows = vec![0.0f32; n];
    let mut norms = vec![0.0f64; channel_count];
    let mut channel_moments = match channels_are_columns {
        true => vec![Moments::default(); channel_count],
        false => Vec::new(),
    };
    let mut channel_kurtosis = vec![0.0f64; channel_count];
    if channels_are_columns {
        for row in 0..m {
            source.read_sanitized_row(row, &mut rows)?;
            for (channel, &value) in rows.iter().enumerate() {
                let value = f64::from(value);
                global.push(value);
                channel_moments[channel].push(value);
                norms[channel] += value * value;
            }
        }
        for (channel, moments) in channel_moments.iter().enumerate() {
            channel_kurtosis[channel] = moments.kurtosis();
        }
    } else {
        for row in 0..m {
            source.read_sanitized_row(row, &mut rows)?;
            let mut local = Moments::default();
            let mut norm2 = 0.0;
            for &value in &rows {
                let value = f64::from(value);
                global.push(value);
                local.push(value);
                norm2 += value * value;
            }
            norms[row] = norm2;
            channel_kurtosis[row] = local.kurtosis();
        }
    }
    for norm in &mut norms {
        *norm = norm.sqrt();
    }
    let mut sorted = norms.clone();
    sorted.sort_by(f64::total_cmp);
    let norm_median = median(&sorted);
    let norm_argmax = norms
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
        .map(|(index, _)| index)
        .unwrap_or(0);
    let norm_max = norms.get(norm_argmax).copied().unwrap_or(0.0);
    let kurtosis_max = channel_kurtosis.iter().copied().fold(0.0f64, f64::max);
    let outliers = channel_kurtosis
        .iter()
        .enumerate()
        .filter(|(_, kurtosis)| **kurtosis > threshold)
        .map(|(channel, _)| channel)
        .collect();
    Ok(StreamingStats {
        global,
        channels: ChannelStats {
            count: channel_count,
            norm_median,
            norm_max,
            norm_argmax,
            kurtosis_max,
            outliers,
        },
    })
}

fn median(sorted: &[f64]) -> f64 {
    if sorted.is_empty() {
        0.0
    } else if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    }
}

#[derive(Debug, Clone, Copy)]
struct FullSpectrum {
    sigma: f64,
    erank_l1: f64,
    erank_energy: f64,
}

fn full_spectrum(
    source: &NodeSource<'_>,
    tensor: &TensorRef,
    options: &AnalyzeOptions<'_>,
    _hidden: usize,
    svd: &Rsvd,
) -> Result<Option<FullSpectrum>> {
    let (m, n) = (tensor.m(), tensor.n());
    let smallest = m.min(n);
    if options.full_spectrum_max_dim == 0
        || smallest > options.full_spectrum_max_dim
        || smallest > MAX_N
        || m.saturating_mul(n) > FULL_SPECTRUM_PRODUCT_CAP
    {
        return Ok(None);
    }
    let mut matrix = vec![0.0f64; m * n];
    let mut row = vec![0.0f32; n];
    for index in 0..m {
        source.read_sanitized_row(index, &mut row)?;
        for (target, &value) in matrix[index * n..(index + 1) * n].iter_mut().zip(&row) {
            *target = f64::from(value);
        }
    }
    let svd_full = thin_svd(&matrix, m, n)?;
    let sigma = svd_full.sigma;
    let sum: f64 = sigma.iter().sum();
    let erank_l1 = match sum > 0.0 {
        true => {
            let entropy: f64 = sigma
                .iter()
                .filter(|&&s| s > 0.0)
                .map(|&s| {
                    let p = s / sum;
                    -p * p.ln()
                })
                .sum();
            entropy.exp()
        }
        false => 0.0,
    };
    let sum_sq: f64 = sigma.iter().map(|s| s * s).sum();
    let erank_energy = match sum_sq > 0.0 {
        true => {
            let entropy: f64 = sigma
                .iter()
                .filter(|&&s| s > 0.0)
                .map(|&s| {
                    let p = s * s / sum_sq;
                    -p * p.ln()
                })
                .sum();
            entropy.exp()
        }
        false => 0.0,
    };
    let sigma1 = sigma.first().copied().unwrap_or(0.0);
    // A full spectrum certifies the first singular value exactly; guard against a matrix
    // whose randomized estimate was below the exact one.
    let sigma1 = if sigma1.is_finite() {
        sigma1.max(f64::from(svd.s.first().copied().unwrap_or(0.0)))
    } else {
        f64::from(svd.s.first().copied().unwrap_or(0.0))
    };
    Ok(Some(FullSpectrum {
        sigma: sigma1,
        erank_l1,
        erank_energy,
    }))
}

fn build_metrics(
    svd: &Rsvd,
    tensor: &TensorRef,
    counters: Counters,
    stats: StreamingStats,
    full: Option<FullSpectrum>,
    basis: &Basis,
    vector: Option<&[f32]>,
) -> Result<NodeMetrics> {
    let rank = svd.r;
    let total_dim = svd.m.min(svd.n);
    let tail = (1.0 - svd.range_energy).clamp(0.0, 1.0);
    let tail_norm = svd.frob_norm * tail.sqrt();
    let sigma1 = svd.s.first().copied().map(f64::from).filter(|s| *s > 0.0);
    let (sigma1, sigma1_lower, sigma1_upper, d_spec, d_spec_lower, d_spec_upper) = match full {
        Some(full) if full.sigma > 0.0 => {
            let d = svd.frob_norm * svd.frob_norm / (full.sigma * full.sigma);
            (
                Some(full.sigma),
                Some(full.sigma),
                Some(full.sigma),
                Some(d),
                Some(d),
                Some(d),
            )
        }
        _ => match sigma1 {
            Some(s1) => {
                let upper = s1 + tail_norm;
                let frob2 = svd.frob_norm * svd.frob_norm;
                let d = frob2 / (s1 * s1);
                let lower = frob2 / (upper * upper);
                (
                    Some(s1),
                    Some(s1),
                    Some(upper),
                    Some(d),
                    Some(lower),
                    Some(d),
                )
            }
            None => (None, None, None, None, None, None),
        },
    };
    let (erank_lower, erank_upper) = match full {
        Some(full) if full.erank_energy > 0.0 => (full.erank_energy, full.erank_energy),
        _ => energy_entropy_bounds(&svd.s, svd.energy, total_dim),
    };
    let e_write = match (vector, tensor.role) {
        (Some(vector), role) if role.is_writer() => Some(basis.captured_energy(vector)),
        _ => None,
    };
    let e_read = match (vector, tensor.role) {
        (Some(vector), role) if role.is_reader() => Some(basis.captured_energy(vector)),
        _ => None,
    };
    Ok(NodeMetrics {
        rows: svd.m,
        cols: svd.n,
        rank,
        frobenius_norm: svd.frob_norm,
        energy: svd.energy,
        residual_energy: (1.0 - svd.energy).clamp(0.0, 1.0),
        range_energy: svd.range_energy,
        rel_error: svd.rel_error,
        sigma1,
        sigma1_lower,
        sigma1_upper,
        d_spec,
        d_spec_lower,
        d_spec_upper,
        erank_energy_lower: erank_lower,
        erank_energy_upper: erank_upper,
        erank_l1: full.map(|f| f.erank_l1),
        full_spectrum: full.is_some(),
        kurtosis: stats.global.kurtosis(),
        kurtosis_excess: stats.global.kurtosis() - 3.0,
        channels: stats.channels,
        e_write,
        e_read,
        non_finite: counters.total(),
        singular_values: svd.s.clone(),
    })
}

/// Energy-entropy effective rank of `s`, with the unseen tail packed (lower bound) or
/// spread uniformly (upper bound) over `total_dim - rank` singular values.
pub fn energy_entropy_bounds(s: &[f32], captured_energy: f64, total_dim: usize) -> (f64, f64) {
    let sum: f64 = s.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
    if sum <= 0.0 || total_dim == 0 {
        return (0.0, 0.0);
    }
    let rank = s.len().min(total_dim);
    let entropy: f64 = s
        .iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| {
            let p = f64::from(x) * f64::from(x) / sum;
            -p * p.ln()
        })
        .sum();
    let energy = captured_energy.clamp(0.0, 1.0);
    if energy == 0.0 {
        return (0.0, 0.0);
    }
    let tail = 1.0 - energy;
    let captured = energy * entropy - energy * energy.ln();
    let (lower_h, upper_h) = if tail <= 0.0 || rank >= total_dim {
        (captured, captured)
    } else {
        let d = (total_dim - rank) as f64;
        (
            captured - tail * tail.ln(),
            captured - tail * tail.ln() + tail * d.ln(),
        )
    };
    (
        lower_h.exp().clamp(0.0, total_dim as f64),
        upper_h.exp().clamp(0.0, total_dim as f64),
    )
}

/// `(sum(sigma)^2 / sum(sigma^2))`, the participation-ratio effective rank.
pub fn participation_rank(s: &[f32]) -> f64 {
    let sum: f64 = s.iter().map(|&x| f64::from(x)).sum();
    let sum_sq: f64 = s.iter().map(|&x| f64::from(x) * f64::from(x)).sum();
    if sum_sq > 0.0 {
        sum * sum / sum_sq
    } else {
        0.0
    }
}

/// Rejects a vector whose length is not the model's hidden size.
pub fn check_vector(vector: &[f32], hidden: usize) -> Result<()> {
    match vector.len() == hidden {
        true => Ok(()),
        false => Err(GraphError::Usage(format!(
            "--vector : {} valeurs pour hidden = {hidden}",
            vector.len()
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_bounds_are_exact_on_a_full_spectrum() {
        let s = [3.0f32, 2.0, 1.0, 0.0];
        let (lower, upper) = energy_entropy_bounds(&s, 1.0, 4);
        assert!((lower - upper).abs() < 1e-12);
        assert!(lower > 1.0 && lower <= 4.0);
    }

    #[test]
    fn entropy_bounds_frame_the_packed_and_spread_tails() {
        let s = [10.0f32, 1.0];
        let (lower, upper) = energy_entropy_bounds(&s, 1.0 - 0.05, 64);
        assert!(lower <= upper, "{lower} > {upper}");
        assert!(upper <= 64.0);
    }

    #[test]
    fn participation_rank_lies_between_one_and_the_rank() {
        assert!((participation_rank(&[1.0f32, 0.0, 0.0]) - 1.0).abs() < 1e-12);
        assert!((participation_rank(&[1.0f32, 1.0, 1.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn identical_subspaces_align_and_orthogonal_ones_do_not() {
        let write = Basis {
            rows: 2,
            cols: 4,
            values: vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        };
        let read = write.clone();
        let same = similarity(&write, &read, 4, &[0.5, 0.5], &[0.5, 0.5]);
        assert!((same.alignment - 1.0).abs() < 1e-9);
        assert!((same.cos_theta - 1.0).abs() < 1e-9);
        assert!(same.alignment_norm > 0.9);
        let orthogonal = Basis {
            rows: 2,
            cols: 4,
            values: vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        };
        let none = similarity(&write, &orthogonal, 4, &[0.5, 0.5], &[0.5, 0.5]);
        assert!(none.alignment < 1e-9);
        assert!(none.cos_theta < 1e-9);
        assert!((none.chance - 0.5).abs() < 1e-12);
        assert!(none.alignment_norm < 0.0);
    }

    #[test]
    fn moment_accumulation_matches_direct_kurtosis() {
        let values = [1.0, 2.0, 3.0, 4.0, 10.0, -2.0];
        let mut moments = Moments::default();
        for value in values {
            moments.push(value);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let m2: f64 = values.iter().map(|x| (x - mean).powi(2)).sum();
        let m4: f64 = values.iter().map(|x| (x - mean).powi(4)).sum();
        let expected = values.len() as f64 * m4 / (m2 * m2);
        assert!((moments.kurtosis() - expected).abs() < 1e-10);
    }
}
