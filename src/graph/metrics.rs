//! Per-matrix spectral metrics and streaming statistics for `surgeon graph`.
//!
//! Every matrix is read through [`NodeSource`], so the values used here are exactly the
//! values `rsvd` sees (Gemma `1 + gamma` folded, non-finite entries replaced by zero after
//! being counted). The randomized SVD gives a range certificate
//! `range_energy = ||Q^T A||_F^2 / ||A||_F^2`; the residual energy of that range is
//! `1 - range_energy`, and every interval below is derived from it without assuming that
//! the singular values are exact.
//!
//! One linear pass reads every entry once for the norm, the counts, the moments and the
//! optional difference to a baseline tensor. Moments are exact two-pass sums per block of
//! [`BLOCK_ROWS`] rows, merged with Pébay's pairwise formulas in block order, so they do not
//! depend on the rayon thread count. The exact spectrum is only computed when the smallest
//! dimension is within 4 of `rank + oversample`, where the randomized range is nearly the
//! whole space anyway.

use std::sync::atomic::AtomicBool;

use rayon::prelude::*;

use crate::linalg::jacobi::MAX_N;
use crate::surgery::rsvd::{rsvd, Rsvd, RsvdParams};

use super::error::{self, GraphError, Result};
use super::model::{Counters, ModelData, NodeSource, TensorRef};
use super::GraphRole;

const BLOCK_ROWS: usize = 64;
/// Blocks read at once: bounds the per-block channel moments held in memory.
const WAVE_BLOCKS: usize = 64;
/// The exact spectrum is computed when `min(m, n) <= rank + oversample + EXACT_MARGIN`.
pub const EXACT_MARGIN: usize = 4;

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
    fn of(values: &[f32]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let mean = values.iter().map(|&x| f64::from(x)).sum::<f64>() / values.len() as f64;
        let mut out = Self {
            n: values.len() as u64,
            mean,
            ..Self::default()
        };
        for &x in values {
            out.add_centered(f64::from(x) - mean);
        }
        out
    }

    fn add_centered(&mut self, d: f64) {
        let d2 = d * d;
        self.m2 += d2;
        self.m3 += d2 * d;
        self.m4 += d2 * d2;
    }

    /// Moments of each column of a row-major block.
    fn columns(block: &[f32], cols: usize) -> Vec<Self> {
        let rows = block.len() / cols;
        let mut out = vec![Self::default(); cols];
        if rows == 0 {
            return out;
        }
        for row in block.chunks_exact(cols) {
            for (acc, &x) in out.iter_mut().zip(row) {
                acc.mean += f64::from(x);
            }
        }
        for acc in &mut out {
            acc.n = rows as u64;
            acc.mean /= rows as f64;
        }
        for row in block.chunks_exact(cols) {
            for (acc, &x) in out.iter_mut().zip(row) {
                acc.add_centered(f64::from(x) - acc.mean);
            }
        }
        out
    }

    /// Pébay's pairwise update: the moments of both samples together.
    fn merge(&mut self, other: &Self) {
        if other.n == 0 {
            return;
        }
        if self.n == 0 {
            *self = *other;
            return;
        }
        let (na, nb) = (self.n as f64, other.n as f64);
        let n = na + nb;
        let d = other.mean - self.mean;
        let (d2, d3, d4) = (d * d, d * d * d, d * d * d * d);
        let m4 = self.m4
            + other.m4
            + d4 * na * nb * (na * na - na * nb + nb * nb) / (n * n * n)
            + 6.0 * d2 * (na * na * other.m2 + nb * nb * self.m2) / (n * n)
            + 4.0 * d * (na * other.m3 - nb * self.m3) / n;
        let m3 = self.m3
            + other.m3
            + d3 * na * nb * (na - nb) / (n * n)
            + 3.0 * d * (na * other.m2 - nb * self.m2) / n;
        self.m2 += other.m2 + d2 * na * nb / n;
        self.m3 = m3;
        self.m4 = m4;
        self.mean += d * nb / n;
        self.n += other.n;
    }

    fn sum_squares(&self) -> f64 {
        self.m2 + self.n as f64 * self.mean * self.mean
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
    /// Exact zeros among the finite entries.
    #[serde(default)]
    pub zeros: u64,
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

impl NodeAnalysis {
    /// A node with its shape only: zero metrics and an empty basis, whose edges align at 0.
    pub fn unmeasured(rows: usize, cols: usize) -> Self {
        Self {
            metrics: NodeMetrics {
                rows,
                cols,
                rank: 0,
                frobenius_norm: 0.0,
                energy: 0.0,
                residual_energy: 0.0,
                range_energy: 0.0,
                rel_error: 0.0,
                sigma1: None,
                sigma1_lower: None,
                sigma1_upper: None,
                d_spec: None,
                d_spec_lower: None,
                d_spec_upper: None,
                erank_energy_lower: 0.0,
                erank_energy_upper: 0.0,
                erank_l1: None,
                full_spectrum: false,
                kurtosis: 0.0,
                kurtosis_excess: 0.0,
                channels: ChannelStats {
                    count: 0,
                    norm_median: 0.0,
                    norm_max: 0.0,
                    norm_argmax: 0,
                    kurtosis_max: 0.0,
                    outliers: Vec::new(),
                },
                e_write: None,
                e_read: None,
                non_finite: 0,
                zeros: 0,
                singular_values: Vec::new(),
            },
            basis: Basis {
                rows: 0,
                cols: 0,
                values: Vec::new(),
            },
            energy_weights: Vec::new(),
        }
    }
}

pub struct AnalyzeOptions<'a> {
    pub params: RsvdParams,
    pub kurtosis_threshold: f64,
    pub vector: Option<&'a [f32]>,
    pub cancel: &'a AtomicBool,
    pub external: Option<&'a AtomicBool>,
}

/// The linear pass and the randomized SVD of one node. With `peer` (same shape, see
/// [`same_weights`]), the pass also returns `||W - W_peer||_F`.
pub fn analyze(
    model: &ModelData,
    tensor: &TensorRef,
    options: &AnalyzeOptions<'_>,
    peer: Option<(&ModelData, &TensorRef)>,
) -> Result<(NodeAnalysis, Option<f64>)> {
    let gamma = model.gamma_for(tensor);
    let source = NodeSource::new(
        model,
        tensor,
        gamma.as_deref(),
        options.cancel,
        options.external,
    );
    let peer_gamma = peer.and_then(|(peer_model, peer_tensor)| peer_model.gamma_for(peer_tensor));
    let peer_source = peer.map(|(peer_model, peer_tensor)| {
        NodeSource::new(
            peer_model,
            peer_tensor,
            peer_gamma.as_deref(),
            options.cancel,
            options.external,
        )
    });
    let pass = linear_pass(
        &source,
        tensor,
        options.kurtosis_threshold,
        peer_source.as_ref(),
    )?;
    let diff = pass.diff2.map(f64::sqrt);
    let svd = rsvd(&source, &options.params, options.cancel).map_err(error::surgery)?;
    let (basis, energy_weights) = basis_of(tensor.role, &svd)?;
    let full = exact_spectrum(&source, tensor, &options.params, options.cancel, &svd)?;
    let metrics = build_metrics(&svd, tensor, pass, full, &basis, options.vector)?;
    let analysis = NodeAnalysis {
        metrics,
        basis,
        energy_weights,
    };
    Ok((analysis, diff))
}

/// `Some(true)`: same shape, dtype, Gemma factor and stored bytes. `Some(false)`: same shape
/// only. `None`: the shapes differ, so the two matrices cannot be subtracted.
pub fn same_weights(
    model: &ModelData,
    tensor: &TensorRef,
    peer_model: &ModelData,
    peer_tensor: &TensorRef,
) -> Result<Option<bool>> {
    if tensor.shape != peer_tensor.shape {
        return Ok(None);
    }
    Ok(Some(
        tensor.dtype == peer_tensor.dtype
            && model.gamma_for(tensor) == peer_model.gamma_for(peer_tensor)
            && model.raw_bytes(tensor)? == peer_model.raw_bytes(peer_tensor)?,
    ))
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

struct LinearPass {
    counters: Counters,
    global: Moments,
    channels: ChannelStats,
    /// `||W - W_peer||_F^2`, with a peer only.
    diff2: Option<f64>,
}

struct BlockPass {
    counters: Counters,
    global: Moments,
    /// One entry per column, or one per row of the block for writers.
    channels: Vec<Moments>,
    diff2: f64,
}

fn linear_pass(
    source: &NodeSource<'_>,
    tensor: &TensorRef,
    threshold: f64,
    peer: Option<&NodeSource<'_>>,
) -> Result<LinearPass> {
    let (m, n) = (tensor.m(), tensor.n());
    // Writers store residual channels as rows; readers (and embeddings) as columns.
    let by_columns = !matches!(tensor.role, GraphRole::AttnOut | GraphRole::MlpOut);
    let mut counters = Counters::default();
    let mut global = Moments::default();
    let mut channels = match by_columns {
        true => vec![Moments::default(); n],
        false => Vec::with_capacity(m),
    };
    let mut diff2 = 0.0;
    let blocks = m.div_ceil(BLOCK_ROWS);
    for first in (0..blocks).step_by(WAVE_BLOCKS) {
        let last = (first + WAVE_BLOCKS).min(blocks);
        let parts = (first..last)
            .into_par_iter()
            .map(|b| {
                let start = b * BLOCK_ROWS;
                block_pass(
                    source,
                    peer,
                    start..(start + BLOCK_ROWS).min(m),
                    n,
                    by_columns,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        for part in parts {
            counters.nan += part.counters.nan;
            counters.inf += part.counters.inf;
            counters.zeros += part.counters.zeros;
            global.merge(&part.global);
            diff2 += part.diff2;
            match by_columns {
                true => {
                    for (acc, block) in channels.iter_mut().zip(&part.channels) {
                        acc.merge(block);
                    }
                }
                false => channels.extend(part.channels),
            }
        }
    }
    Ok(LinearPass {
        counters,
        global,
        channels: channel_stats(&channels, threshold),
        diff2: peer.map(|_| diff2),
    })
}

fn block_pass(
    source: &NodeSource<'_>,
    peer: Option<&NodeSource<'_>>,
    rows: std::ops::Range<usize>,
    n: usize,
    by_columns: bool,
) -> Result<BlockPass> {
    let mut counters = Counters::default();
    let mut block = vec![0.0f32; rows.len() * n];
    for (row, out) in rows.clone().zip(block.chunks_exact_mut(n)) {
        source.read_counted_row(row, out, &mut counters)?;
    }
    let mut diff2 = 0.0;
    if let Some(peer) = peer {
        let mut other = vec![0.0f32; n];
        let mut ignored = Counters::default();
        for (row, mine) in rows.zip(block.chunks_exact(n)) {
            peer.read_counted_row(row, &mut other, &mut ignored)?;
            diff2 += mine
                .iter()
                .zip(&other)
                .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                .sum::<f64>();
        }
    }
    let channels = match by_columns {
        true => Moments::columns(&block, n),
        false => block.chunks_exact(n).map(Moments::of).collect(),
    };
    Ok(BlockPass {
        counters,
        global: Moments::of(&block),
        channels,
        diff2,
    })
}

fn channel_stats(channels: &[Moments], threshold: f64) -> ChannelStats {
    let norms: Vec<f64> = channels.iter().map(|c| c.sum_squares().sqrt()).collect();
    let kurtosis: Vec<f64> = channels.iter().map(Moments::kurtosis).collect();
    let mut sorted = norms.clone();
    sorted.sort_by(f64::total_cmp);
    let norm_argmax = norms
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
        .map(|(index, _)| index)
        .unwrap_or(0);
    ChannelStats {
        count: channels.len(),
        norm_median: median(&sorted),
        norm_max: norms.get(norm_argmax).copied().unwrap_or(0.0),
        norm_argmax,
        kurtosis_max: kurtosis.iter().copied().fold(0.0f64, f64::max),
        outliers: kurtosis
            .iter()
            .enumerate()
            .filter(|(_, k)| **k > threshold)
            .map(|(channel, _)| channel)
            .collect(),
    }
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

/// Every singular value, from a randomized SVD whose range is the whole smaller side.
fn exact_spectrum(
    source: &NodeSource<'_>,
    tensor: &TensorRef,
    params: &RsvdParams,
    cancel: &AtomicBool,
    svd: &Rsvd,
) -> Result<Option<FullSpectrum>> {
    let smallest = tensor.m().min(tensor.n());
    let reach = params
        .rank
        .saturating_add(params.oversample)
        .saturating_add(EXACT_MARGIN)
        .min(MAX_N);
    if smallest > reach {
        return Ok(None);
    }
    let all = RsvdParams {
        rank: smallest,
        oversample: 0,
        ..*params
    };
    let sigma: Vec<f64> = rsvd(source, &all, cancel)
        .map_err(error::surgery)?
        .s
        .iter()
        .map(|&s| f64::from(s))
        .collect();
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
    pass: LinearPass,
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
        frobenius_norm: pass.global.sum_squares().sqrt(),
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
        kurtosis: pass.global.kurtosis(),
        kurtosis_excess: pass.global.kurtosis() - 3.0,
        channels: pass.channels,
        e_write,
        e_read,
        non_finite: pass.counters.total(),
        zeros: pass.counters.zeros,
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
    fn merged_block_moments_match_direct_kurtosis() {
        let values = [1.0f32, 2.0, 3.0, 4.0, 10.0, -2.0, 0.5];
        let direct = |xs: &[f32]| {
            let n = xs.len() as f64;
            let mean = xs.iter().map(|&x| f64::from(x)).sum::<f64>() / n;
            let m2: f64 = xs.iter().map(|&x| (f64::from(x) - mean).powi(2)).sum();
            let m4: f64 = xs.iter().map(|&x| (f64::from(x) - mean).powi(4)).sum();
            n * m4 / (m2 * m2)
        };
        for split in 0..=values.len() {
            let mut merged = Moments::of(&values[..split]);
            merged.merge(&Moments::of(&values[split..]));
            assert!(
                (merged.kurtosis() - direct(&values)).abs() < 1e-10,
                "{split}"
            );
            let sum_squares: f64 = values.iter().map(|&x| f64::from(x).powi(2)).sum();
            assert!((merged.sum_squares() - sum_squares).abs() < 1e-10);
        }
    }

    #[test]
    fn column_moments_match_each_column_alone() {
        let block = [1.0f32, -4.0, 2.0, 0.0, 3.0, 8.0, 7.0, 1.0];
        let columns = Moments::columns(&block, 2);
        for (j, column) in columns.iter().enumerate() {
            let alone: Vec<f32> = block.iter().skip(j).step_by(2).copied().collect();
            let expected = Moments::of(&alone);
            assert_eq!(column.n, expected.n);
            assert!((column.m2 - expected.m2).abs() < 1e-12);
            assert!((column.m4 - expected.m4).abs() < 1e-12);
        }
    }

    /// `H D`: a normalized Sylvester Hadamard matrix times a scattered diagonal, so the
    /// singular values are exactly `spectrum` while every row is dense on the diagonal's
    /// columns. The randomized SVD's Gaussian test matrix is rotation invariant, so its
    /// accuracy here is the one it has on `U diag(spectrum) V^T` for any orthogonal `U`, `V`.
    struct KnownSpectrum {
        m: usize,
        n: usize,
        spectrum: Vec<f32>,
    }

    impl KnownSpectrum {
        fn column(&self, t: usize) -> usize {
            t * 2749 % self.n
        }
    }

    impl crate::surgery::rsvd::RowSource for KnownSpectrum {
        fn rows(&self) -> usize {
            self.m
        }
        fn cols(&self) -> usize {
            self.n
        }
        fn read_rows(&self, start: usize, out: &mut [f32]) -> crate::surgery::Result<usize> {
            let rows = (out.len() / self.n).min(self.m - start);
            let scale = 1.0 / (self.m as f32).sqrt();
            for (i, row) in out.chunks_exact_mut(self.n).take(rows).enumerate() {
                row.fill(0.0);
                for (t, &s) in self.spectrum.iter().enumerate() {
                    let sign = match ((start + i) & t).count_ones() % 2 {
                        0 => 1.0,
                        _ => -1.0,
                    };
                    row[self.column(t)] = sign * s * scale;
                }
            }
            Ok(rows)
        }
    }

    struct Measured {
        elapsed: std::time::Duration,
        sigma1_ratio: f64,
        energy_ratio: f64,
        energy_error: f64,
    }

    fn measure(spectrum: Vec<f32>) -> Measured {
        let source = KnownSpectrum {
            m: 2048,
            n: 5632,
            spectrum,
        };
        let params = RsvdParams {
            rank: super::super::DEFAULT_RANK,
            oversample: super::super::DEFAULT_OVERSAMPLE,
            power_iters: super::super::DEFAULT_POWER_ITERS,
            seed: 0,
        };
        let started = std::time::Instant::now();
        let svd = rsvd(&source, &params, &AtomicBool::new(false)).unwrap();
        let elapsed = started.elapsed();
        let exact: Vec<f64> = source.spectrum.iter().map(|&s| f64::from(s)).collect();
        let total: f64 = exact.iter().map(|s| s * s).sum();
        let exact_energy = exact[..params.rank].iter().map(|s| s * s).sum::<f64>() / total;
        let measured = Measured {
            elapsed,
            sigma1_ratio: f64::from(svd.s[0]) / exact[0],
            energy_ratio: svd.energy / exact_energy,
            energy_error: (svd.energy - exact_energy).abs(),
        };
        println!(
            "2048x5632 rank {} k {} q {}: {:.2?}, sigma1 {:.6}, energy {:.6} (abs error {:.2e})",
            params.rank,
            params.rank + params.oversample,
            params.power_iters,
            measured.elapsed,
            measured.sigma1_ratio,
            measured.energy_ratio,
            measured.energy_error
        );
        measured
    }

    /// `cargo test --release --lib spectral_block_benchmark -- --nocapture`
    #[test]
    #[cfg_attr(debug_assertions, ignore = "timing: run with --release")]
    fn spectral_block_benchmark_2048x5632() {
        let decaying = measure((0..2048).map(|t| (1.0 + t as f32).recip()).collect());
        assert!((decaying.sigma1_ratio - 1.0).abs() < 1e-3);
        assert!((decaying.energy_ratio - 1.0).abs() < 1e-3);
        assert!(decaying.elapsed.as_millis() < 500, "{:?}", decaying.elapsed);

        // A flat bulk under a few spikes, as in trained weights: sigma1 stays exact, the
        // rank-r energy is underestimated by a few percent of itself with k = r + 8, q = 2.
        let flat = measure(
            (0..2048)
                .map(|t| match t < 8 {
                    true => 4.0 / (1.0 + t as f32),
                    false => 1.0 - 0.5 * t as f32 / 2048.0,
                })
                .collect(),
        );
        assert!((flat.sigma1_ratio - 1.0).abs() < 1e-3);
        assert!(flat.energy_error < 5e-3, "{}", flat.energy_error);
        assert!(flat.elapsed.as_millis() < 500, "{:?}", flat.elapsed);
    }
}
