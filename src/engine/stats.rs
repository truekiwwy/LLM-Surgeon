//! Statistics of a tensor straight from its bytes (a memory-mapped view): NaN, +Inf and
//! -Inf are counted apart and never enter the moments nor the histogram. Shared by
//! `inspect --show-stats` and the TUI.
//!
//! Two parallel passes over blocks of the bytes: (1) counts, min/max and Welford moments
//! per block, merged with Chan's formula in block order (deterministic for a given block
//! size); (2) a histogram of the finite values per block, summed. Everything is f64.
//! Elements are decoded with `from_le_bytes` (never a pointer cast). The cancellation flag
//! is checked at every block.

use std::sync::atomic::{AtomicBool, Ordering};

use half::{bf16, f16};
use rayon::prelude::*;
use serde::Serialize;

use super::index::DType;

/// Elements per parallel block, which is also the cancellation granularity.
pub const DEFAULT_BLOCK: usize = 1 << 16;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum StatsError {
    #[error("calcul annulé")]
    Cancelled,
    #[error("dtype {0} non supporté pour les statistiques")]
    UnsupportedDtype(String),
    #[error(
        "{bytes} octets ne correspondent pas à la forme {shape:?} ({element} octets par élément)"
    )]
    Length {
        bytes: usize,
        shape: Vec<usize>,
        element: usize,
    },
}

pub type Result<T> = std::result::Result<T, StatsError>;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Histogram {
    pub min: f64,
    pub max: f64,
    /// One bin when `min == max`; they sum to the number of finite values.
    pub counts: Vec<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TensorStats {
    /// Finite values: the only ones in min/max/mean/std/l2 and the histogram.
    pub count: u64,
    /// `nan + pos_inf + neg_inf`.
    pub non_finite: u64,
    pub nan: u64,
    pub pos_inf: u64,
    pub neg_inf: u64,
    /// Exact zeros, `-0.0` included.
    pub zeros: u64,
    pub elements: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    /// Population standard deviation (divides by the number of finite values).
    pub std: f64,
    pub l2: f64,
    /// `None` when no bins were asked or there is no finite value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub histogram: Option<Histogram>,
}

impl TensorStats {
    /// Share of exact zeros among all elements.
    pub fn sparsity(&self) -> f64 {
        match self.elements {
            0 => 0.0,
            n => self.zeros as f64 / n as f64,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct Moments {
    n: u64,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
    sum_sq: f64,
    nan: u64,
    pos_inf: u64,
    neg_inf: u64,
    zeros: u64,
}

impl Moments {
    const EMPTY: Self = Self {
        n: 0,
        mean: 0.0,
        m2: 0.0,
        min: f64::INFINITY,
        max: f64::NEG_INFINITY,
        sum_sq: 0.0,
        nan: 0,
        pos_inf: 0,
        neg_inf: 0,
        zeros: 0,
    };

    fn push(&mut self, x: f64) {
        if x.is_nan() {
            self.nan += 1;
        } else if x == f64::INFINITY {
            self.pos_inf += 1;
        } else if x == f64::NEG_INFINITY {
            self.neg_inf += 1;
        } else {
            self.n += 1;
            let delta = x - self.mean;
            self.mean += delta / self.n as f64;
            self.m2 += delta * (x - self.mean);
            self.min = self.min.min(x);
            self.max = self.max.max(x);
            self.sum_sq += x * x;
            self.zeros += u64::from(x == 0.0);
        }
    }

    /// Chan, Golub & LeVeque: exact merge of two Welford accumulators.
    fn merge(self, other: Self) -> Self {
        let n = self.n + other.n;
        let (mean, m2) = match (self.n, other.n) {
            (0, _) => (other.mean, other.m2),
            (_, 0) => (self.mean, self.m2),
            (na, nb) => {
                let delta = other.mean - self.mean;
                let (na, nb, nt) = (na as f64, nb as f64, n as f64);
                (
                    self.mean + delta * nb / nt,
                    self.m2 + other.m2 + delta * delta * na * nb / nt,
                )
            }
        };
        Self {
            n,
            mean,
            m2,
            min: self.min.min(other.min),
            max: self.max.max(other.max),
            sum_sq: self.sum_sq + other.sum_sq,
            nan: self.nan + other.nan,
            pos_inf: self.pos_inf + other.pos_inf,
            neg_inf: self.neg_inf + other.neg_inf,
            zeros: self.zeros + other.zeros,
        }
    }
}

type Decode = fn(&[u8]) -> f64;

fn decoder(dtype: &DType) -> Result<(usize, Decode)> {
    match dtype {
        DType::F32 => Ok((4, |b| {
            f64::from(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        })),
        DType::F16 => Ok((2, |b| {
            f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f64()
        })),
        DType::BF16 => Ok((2, |b| {
            bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f64()
        })),
        DType::Other(name) => Err(StatsError::UnsupportedDtype(name.clone())),
    }
}

pub fn tensor_stats(
    bytes: &[u8],
    dtype: &DType,
    shape: &[usize],
    bins: usize,
    cancel: &AtomicBool,
) -> Result<TensorStats> {
    tensor_stats_with_block(bytes, dtype, shape, bins, DEFAULT_BLOCK, cancel)
}

/// [`tensor_stats`] with an explicit block size (tests compare several).
pub fn tensor_stats_with_block(
    bytes: &[u8],
    dtype: &DType,
    shape: &[usize],
    bins: usize,
    block: usize,
    cancel: &AtomicBool,
) -> Result<TensorStats> {
    let (size, decode) = decoder(dtype)?;
    let elements: usize = shape.iter().product();
    if bytes.len() != elements * size {
        return Err(StatsError::Length {
            bytes: bytes.len(),
            shape: shape.to_vec(),
            element: size,
        });
    }
    let chunk = block.max(1) * size;
    let cancelled = || cancel.load(Ordering::Relaxed);

    let partials: Vec<Option<Moments>> = bytes
        .par_chunks(chunk)
        .map(|part| {
            if cancelled() {
                return None;
            }
            let mut m = Moments::EMPTY;
            part.chunks_exact(size).for_each(|b| m.push(decode(b)));
            Some(m)
        })
        .collect();
    if cancelled() {
        return Err(StatsError::Cancelled);
    }
    let m = partials
        .into_iter()
        .try_fold(Moments::EMPTY, |acc, p| p.map(|p| acc.merge(p)))
        .ok_or(StatsError::Cancelled)?;

    let histogram = match (bins, m.n) {
        (0, _) | (_, 0) => None,
        // A constant tensor: one bin, no division by a zero range.
        _ if m.min == m.max => Some(Histogram {
            min: m.min,
            max: m.max,
            counts: vec![m.n],
        }),
        _ => Some(histogram(
            bytes, decode, size, chunk, bins, m.min, m.max, cancel,
        )?),
    };

    let (min, max) = match m.n {
        0 => (0.0, 0.0),
        _ => (m.min, m.max),
    };
    Ok(TensorStats {
        count: m.n,
        non_finite: m.nan + m.pos_inf + m.neg_inf,
        nan: m.nan,
        pos_inf: m.pos_inf,
        neg_inf: m.neg_inf,
        zeros: m.zeros,
        elements: elements as u64,
        min,
        max,
        mean: m.mean,
        std: match m.n {
            0 => 0.0,
            n => (m.m2 / n as f64).max(0.0).sqrt(),
        },
        l2: m.sum_sq.sqrt(),
        histogram,
    })
}

#[allow(clippy::too_many_arguments)]
fn histogram(
    bytes: &[u8],
    decode: Decode,
    size: usize,
    chunk: usize,
    bins: usize,
    min: f64,
    max: f64,
    cancel: &AtomicBool,
) -> Result<Histogram> {
    let width = max - min;
    let locals: Vec<Option<Vec<u64>>> = bytes
        .par_chunks(chunk)
        .map(|part| {
            if cancel.load(Ordering::Relaxed) {
                return None;
            }
            let mut counts = vec![0u64; bins];
            for b in part.chunks_exact(size) {
                let x = decode(b);
                if x.is_finite() {
                    let bin = ((x - min) / width * bins as f64) as usize;
                    counts[bin.min(bins - 1)] += 1;
                }
            }
            Some(counts)
        })
        .collect();
    if cancel.load(Ordering::Relaxed) {
        return Err(StatsError::Cancelled);
    }
    let mut counts = vec![0u64; bins];
    for local in locals {
        let local = local.ok_or(StatsError::Cancelled)?;
        counts.iter_mut().zip(local).for_each(|(c, l)| *c += l);
    }
    Ok(Histogram { min, max, counts })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn no_cancel() -> AtomicBool {
        AtomicBool::new(false)
    }

    /// Independent two-pass reference over finite values, in f64.
    fn reference(values: &[f64]) -> (f64, f64, f64, f64, f64) {
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        let n = finite.len() as f64;
        let mean = finite.iter().sum::<f64>() / n;
        let var = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
        let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (
            min,
            max,
            mean,
            var.sqrt(),
            finite.iter().map(|v| v * v).sum::<f64>().sqrt(),
        )
    }

    #[test]
    fn mixed_values_against_reference_for_every_dtype() {
        let values: Vec<f32> = vec![
            1.5,
            -2.25,
            0.0,
            -0.0,
            f32::NAN,
            3.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            0.125,
        ];
        let as_f64: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
        let (min, max, mean, std, l2) = reference(&as_f64);
        let encodings: [(DType, Vec<u8>); 3] = [
            (DType::F32, f32_bytes(&values)),
            (
                DType::BF16,
                values
                    .iter()
                    .flat_map(|v| bf16::from_f32(*v).to_le_bytes())
                    .collect(),
            ),
            (
                DType::F16,
                values
                    .iter()
                    .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                    .collect(),
            ),
        ];
        for (dtype, bytes) in encodings {
            let s = tensor_stats(&bytes, &dtype, &[2, 5], 4, &no_cancel()).unwrap();
            assert_eq!(
                (s.count, s.nan, s.pos_inf, s.neg_inf, s.non_finite),
                (6, 2, 1, 1, 4),
                "{dtype:?}"
            );
            assert_eq!((s.zeros, s.elements), (2, 10), "-0.0 is an exact zero");
            assert!((s.sparsity() - 0.2).abs() < 1e-12);
            // These values are exact in bf16 and f16 too.
            assert_eq!((s.min, s.max), (min, max));
            assert!(
                (s.mean - mean).abs() < 1e-12
                    && (s.std - std).abs() < 1e-12
                    && (s.l2 - l2).abs() < 1e-12
            );
            let h = s.histogram.unwrap();
            assert_eq!(
                h.counts.iter().sum::<u64>(),
                s.count,
                "bins hold finite values only"
            );
            assert_eq!((h.min, h.max), (min, max));
        }
    }

    #[test]
    fn degenerate_tensors() {
        let constant =
            tensor_stats(&f32_bytes(&[2.0; 7]), &DType::F32, &[7], 64, &no_cancel()).unwrap();
        assert_eq!(
            constant.histogram.unwrap().counts,
            [7],
            "min == max: one bin"
        );
        assert_eq!((constant.std, constant.mean), (0.0, 2.0));
        let empty = tensor_stats(&[], &DType::F32, &[0, 4], 64, &no_cancel()).unwrap();
        assert_eq!(
            (empty.count, empty.elements, empty.histogram.clone()),
            (0, 0, None)
        );
        assert_eq!(empty.sparsity(), 0.0);
        let all_nan = tensor_stats(
            &f32_bytes(&[f32::NAN; 3]),
            &DType::F32,
            &[3],
            64,
            &no_cancel(),
        )
        .unwrap();
        assert_eq!(
            (
                all_nan.count,
                all_nan.nan,
                all_nan.histogram.clone(),
                all_nan.min
            ),
            (0, 3, None, 0.0)
        );
        let extreme = tensor_stats(
            &f32_bytes(&[f32::MAX, f32::MIN, 0.0]),
            &DType::F32,
            &[3],
            8,
            &no_cancel(),
        )
        .unwrap();
        assert_eq!(
            (extreme.max, extreme.min),
            (f64::from(f32::MAX), f64::from(f32::MIN))
        );
        assert!(extreme.mean.is_finite() && extreme.std.is_finite());
        assert_eq!(extreme.histogram.unwrap().counts.iter().sum::<u64>(), 3);
        assert!(matches!(
            tensor_stats(&[0; 3], &DType::F32, &[1], 4, &no_cancel()),
            Err(StatsError::Length { .. })
        ));
        assert!(matches!(
            tensor_stats(&[0; 4], &DType::Other("I8".into()), &[4], 4, &no_cancel()),
            Err(StatsError::UnsupportedDtype(_))
        ));
    }

    #[test]
    fn block_size_does_not_change_the_result() {
        let values: Vec<f32> = (0..10_007)
            .map(|i| ((i * 7919 % 1000) as f32 - 500.0) / 37.0)
            .collect();
        let bytes = f32_bytes(&values);
        let whole = tensor_stats_with_block(
            &bytes,
            &DType::F32,
            &[10_007],
            32,
            usize::MAX / 8,
            &no_cancel(),
        )
        .unwrap();
        for block in [1, 7, 64, 1000, 4096] {
            let s =
                tensor_stats_with_block(&bytes, &DType::F32, &[10_007], 32, block, &no_cancel())
                    .unwrap();
            assert_eq!(
                (s.count, s.min, s.max, s.zeros),
                (whole.count, whole.min, whole.max, whole.zeros)
            );
            assert_eq!(s.histogram, whole.histogram, "block {block}");
            assert!(
                (s.mean - whole.mean).abs() < 1e-12 && (s.std - whole.std).abs() < 1e-9,
                "block {block}"
            );
            let again =
                tensor_stats_with_block(&bytes, &DType::F32, &[10_007], 32, block, &no_cancel())
                    .unwrap();
            assert_eq!(s, again, "deterministic for a given block size");
        }
    }

    #[test]
    fn cancellation_returns_no_partial_result() {
        let bytes = f32_bytes(&vec![1.0; 100_000]);
        let flag = AtomicBool::new(true);
        assert_eq!(
            tensor_stats_with_block(&bytes, &DType::F32, &[100_000], 64, 1024, &flag),
            Err(StatsError::Cancelled)
        );
    }
}
