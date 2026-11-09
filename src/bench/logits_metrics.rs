//! Distribution shift between two logits dumps, all in f64.
//!
//! Per position, with `p = softmax(before)` and `q = softmax(after)` computed in log space
//! (log-sum-exp after subtracting the max, so ±1e4 logits never overflow):
//!
//! - `KL(p‖q) = Σ p_i (log p_i − log q_i)`: the "before" distribution is the reference.
//!   Terms with `p_i = 0` (logit −∞, or underflow) are skipped. A logit finite before and
//!   −∞ after (`p_i > 0`, `q_i = 0`) makes the KL infinite at that position: it is counted
//!   in `kl_infinite_positions` and kept out of the finite mean, max and p95. Rounding can
//!   produce −1e-17; the KL is clamped at 0 (identical rows give exactly 0).
//! - cosine of the probability vectors (and, labelled apart, of the raw logits over the
//!   entries finite on both sides; undefined when a norm is zero).
//! - top-1: argmax, ties broken by the lowest index. top-k: the set of the k largest
//!   (value desc, index asc); changed when the sets differ, overlap `|A∩B| / k`.
//!
//! Determinism: each position is computed by one thread with sequential sums; positions
//! of a block run in parallel and are collected in order (`map_init` + `collect`), then
//! reduced sequentially in position order. The result is bit-identical whatever the
//! number of threads. Memory: two f64 rows per worker thread plus 8 bytes per position
//! (the finite KLs kept for the exact p95), independent of `positions × vocab`.

use std::cmp::Ordering as CmpOrdering;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use super::error::{BenchError, Result};
use super::logits_io::{check_pair, open_dump, shared_offsets, LogitsDump, Rows};
use super::report::{BenchLogitsReport, BenchRun, DumpInfo, LogitsTiming, PromptKl, Verdict};
use super::{finite, Neumaier};
use crate::hardware::HardwareProbe;

/// Positions per parallel block: the cancellation and progress granularity.
pub const BLOCK_POSITIONS: usize = 64;
/// Stop scanning once this many non-finite positions are known.
const MAX_REPORTED_BAD: usize = 5;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Kl {
    Finite(f64),
    Infinite,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PositionMetrics {
    pub kl: Kl,
    pub cosine_probs: f64,
    pub cosine_logits: Option<f64>,
    pub top1_changed: bool,
    pub topk_changed: bool,
    pub topk_overlap: f64,
}

/// Which dump has a NaN, +Inf or all -Inf row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BadSide {
    Before,
    After,
}

/// Log-sum-exp of a row; `None` for NaN, +Inf or a row with no finite value.
fn log_sum_exp(x: &[f64]) -> Option<f64> {
    let mut max = f64::NEG_INFINITY;
    for &v in x {
        if v.is_nan() || v == f64::INFINITY {
            return None;
        }
        max = max.max(v);
    }
    if max == f64::NEG_INFINITY {
        return None;
    }
    let mut sum = Neumaier::default();
    for &v in x {
        sum.add((v - max).exp());
    }
    Some(max + sum.total().ln())
}

/// Indices of the `k` largest values, ties to the lowest index, sorted ascending.
pub fn top_k(x: &[f64], k: usize) -> Vec<usize> {
    let mut best: Vec<(f64, usize)> = Vec::with_capacity(k + 1);
    for (i, &v) in x.iter().enumerate() {
        // Scanned in index order: an equal value never beats an earlier one.
        if best.len() == k && best.last().is_some_and(|&(w, _)| v <= w) {
            continue;
        }
        let at = best.partition_point(|&(w, _)| w >= v);
        best.insert(at, (v, i));
        best.truncate(k);
    }
    let mut indices: Vec<usize> = best.into_iter().map(|(_, i)| i).collect();
    indices.sort_unstable();
    indices
}

fn argmax(x: &[f64]) -> usize {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best
}

fn cosine(dot: f64, na: f64, nb: f64) -> Option<f64> {
    let denom = (na * nb).sqrt();
    (denom > 0.0).then(|| (dot / denom).clamp(-1.0, 1.0))
}

/// Metrics of one position from its two decoded rows.
pub fn position_metrics(
    a: &[f64],
    b: &[f64],
    k: usize,
) -> std::result::Result<PositionMetrics, BadSide> {
    let lse_a = log_sum_exp(a).ok_or(BadSide::Before)?;
    let lse_b = log_sum_exp(b).ok_or(BadSide::After)?;
    let (mut kl, mut infinite) = (Neumaier::default(), false);
    let (mut dot_p, mut np, mut nq) = (
        Neumaier::default(),
        Neumaier::default(),
        Neumaier::default(),
    );
    let (mut dot_l, mut na, mut nb) = (
        Neumaier::default(),
        Neumaier::default(),
        Neumaier::default(),
    );
    for (&x, &y) in a.iter().zip(b) {
        let (log_p, log_q) = (x - lse_a, y - lse_b);
        let (p, q) = (log_p.exp(), log_q.exp());
        if x != f64::NEG_INFINITY {
            match y == f64::NEG_INFINITY {
                true => infinite = true,
                false => kl.add(p * (log_p - log_q)),
            }
        }
        dot_p.add(p * q);
        np.add(p * p);
        nq.add(q * q);
        if x.is_finite() && y.is_finite() {
            dot_l.add(x * y);
            na.add(x * x);
            nb.add(y * y);
        }
    }
    let (set_a, set_b) = (top_k(a, k), top_k(b, k));
    let common = set_a
        .iter()
        .filter(|i| set_b.binary_search(i).is_ok())
        .count();
    Ok(PositionMetrics {
        kl: match infinite {
            true => Kl::Infinite,
            false => Kl::Finite(kl.total().max(0.0)),
        },
        cosine_probs: cosine(dot_p.total(), np.total(), nq.total()).unwrap_or(0.0),
        cosine_logits: cosine(dot_l.total(), na.total(), nb.total()),
        top1_changed: argmax(a) != argmax(b),
        topk_changed: set_a != set_b,
        topk_overlap: common as f64 / k as f64,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct PromptShift {
    pub index: usize,
    pub start: usize,
    pub end: usize,
    pub kl_mean: Option<f64>,
    pub infinite_positions: usize,
}

/// Aggregates in position order; every field is free of NaN/Inf (`None` instead).
#[derive(Debug, Clone, PartialEq)]
pub struct Comparison {
    pub positions: usize,
    pub vocab: usize,
    pub k: usize,
    pub kl_mean: Option<f64>,
    pub kl_max: Option<f64>,
    pub kl_max_position: Option<usize>,
    pub kl_p95: Option<f64>,
    pub kl_infinite_positions: usize,
    pub cosine_probs_mean: f64,
    pub cosine_logits_mean: Option<f64>,
    pub cosine_logits_positions: usize,
    pub top1_shift_rate: f64,
    pub topk_set_change_rate: f64,
    pub topk_overlap_mean: f64,
    pub per_prompt: Vec<PromptShift>,
}

#[derive(Default)]
struct Accumulator {
    kl_finite: Vec<f64>,
    kl_sum: Neumaier,
    kl_max: Option<(f64, usize)>,
    kl_infinite: usize,
    cos_p: Neumaier,
    cos_l: Neumaier,
    cos_l_count: usize,
    top1: usize,
    topk: usize,
    overlap: Neumaier,
    prompt_kl: Vec<(Neumaier, usize, usize)>,
}

impl Accumulator {
    fn push(&mut self, position: usize, prompt: Option<usize>, m: &PositionMetrics) {
        match m.kl {
            Kl::Finite(kl) => {
                self.kl_finite.push(kl);
                self.kl_sum.add(kl);
                // Strict: the first position wins a tie.
                if self.kl_max.is_none_or(|(max, _)| kl > max) {
                    self.kl_max = Some((kl, position));
                }
            }
            Kl::Infinite => self.kl_infinite += 1,
        }
        if let Some(entry) = prompt.and_then(|p| self.prompt_kl.get_mut(p)) {
            match m.kl {
                Kl::Finite(kl) => {
                    entry.0.add(kl);
                    entry.1 += 1;
                }
                Kl::Infinite => entry.2 += 1,
            }
        }
        self.cos_p.add(m.cosine_probs);
        if let Some(c) = m.cosine_logits {
            self.cos_l.add(c);
            self.cos_l_count += 1;
        }
        self.top1 += usize::from(m.top1_changed);
        self.topk += usize::from(m.topk_changed);
        self.overlap.add(m.topk_overlap);
    }
}

/// Nearest-rank 95th percentile of the finite KLs.
fn p95(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(CmpOrdering::Equal));
    let rank = (0.95 * values.len() as f64).ceil() as usize;
    values.get(rank.saturating_sub(1)).copied()
}

/// Compares two dumps checked by `check_pair`. `on_block(done, total)` reports progress;
/// `cancel` is checked before each block.
pub fn compare(
    before: &LogitsDump,
    after: &LogitsDump,
    k: usize,
    offsets: Option<&[usize]>,
    cancel: &AtomicBool,
    on_block: &mut dyn FnMut(usize, usize),
) -> Result<Comparison> {
    let (positions, vocab) = (before.positions, before.vocab);
    if k == 0 || k > vocab {
        return Err(BenchError::InvalidArgument(format!(
            "--top-k {k} : doit être entre 1 et la taille du vocabulaire ({vocab})"
        )));
    }
    let (rows_a, rows_b) = (before.rows()?, after.rows()?);
    compare_rows(
        rows_a, rows_b, positions, vocab, k, offsets, cancel, on_block,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_rows(
    rows_a: Rows<'_>,
    rows_b: Rows<'_>,
    positions: usize,
    vocab: usize,
    k: usize,
    offsets: Option<&[usize]>,
    cancel: &AtomicBool,
    on_block: &mut dyn FnMut(usize, usize),
) -> Result<Comparison> {
    let blocks = positions.div_ceil(BLOCK_POSITIONS);
    let mut acc = Accumulator {
        prompt_kl: vec![
            (Neumaier::default(), 0, 0);
            offsets.map_or(0, |o| o.len().saturating_sub(1))
        ],
        ..Accumulator::default()
    };
    let mut bad: Vec<(usize, BadSide)> = Vec::new();
    let mut prompt = 0;
    for block in 0..blocks {
        if cancel.load(Ordering::SeqCst) {
            return Err(BenchError::Cancelled);
        }
        let start = block * BLOCK_POSITIONS;
        let end = (start + BLOCK_POSITIONS).min(positions);
        let results: Vec<_> = (start..end)
            .into_par_iter()
            .map_init(
                || (vec![0.0; vocab], vec![0.0; vocab]),
                |(a, b), i| {
                    rows_a.decode(i, a);
                    rows_b.decode(i, b);
                    position_metrics(a, b, k)
                },
            )
            .collect();
        for (i, result) in (start..end).zip(results) {
            let current = offsets.map(|o| {
                while prompt + 1 < o.len() - 1 && i >= o[prompt + 1] {
                    prompt += 1;
                }
                prompt
            });
            match result {
                Ok(m) => acc.push(i, current, &m),
                Err(side) => bad.push((i, side)),
            }
        }
        on_block(block + 1, blocks);
        if bad.len() >= MAX_REPORTED_BAD {
            break;
        }
    }
    if let Some(&(_, side)) = bad.first() {
        let side_bad: Vec<usize> = bad
            .iter()
            .filter(|(_, s)| *s == side)
            .map(|(i, _)| *i)
            .take(MAX_REPORTED_BAD)
            .collect();
        return Err(BenchError::NonFiniteLogits {
            side: match side {
                BadSide::Before => "avant",
                BadSide::After => "après",
            },
            positions: side_bad,
        });
    }

    let n = positions as f64;
    let finite_count = acc.kl_finite.len();
    let per_prompt = offsets
        .map(|o| {
            acc.prompt_kl
                .iter()
                .enumerate()
                .map(|(index, (sum, count, infinite))| PromptShift {
                    index,
                    start: o[index],
                    end: o[index + 1],
                    kl_mean: (*count > 0)
                        .then(|| sum.total() / *count as f64)
                        .and_then(finite),
                    infinite_positions: *infinite,
                })
                .collect()
        })
        .unwrap_or_default();
    Ok(Comparison {
        positions,
        vocab,
        k,
        kl_mean: (finite_count > 0)
            .then(|| acc.kl_sum.total() / finite_count as f64)
            .and_then(finite),
        kl_max: acc.kl_max.and_then(|(v, _)| finite(v)),
        kl_max_position: acc.kl_max.map(|(_, p)| p),
        kl_p95: p95(&mut acc.kl_finite).and_then(finite),
        kl_infinite_positions: acc.kl_infinite,
        cosine_probs_mean: acc.cos_p.total() / n,
        cosine_logits_mean: (acc.cos_l_count > 0)
            .then(|| acc.cos_l.total() / acc.cos_l_count as f64)
            .and_then(finite),
        cosine_logits_positions: acc.cos_l_count,
        top1_shift_rate: acc.top1 as f64 / n,
        topk_set_change_rate: acc.topk as f64 / n,
        topk_overlap_mean: acc.overlap.total() / n,
        per_prompt,
    })
}

/// The `n` prompts with the highest finite mean KL (ties: lowest index); prompts with an
/// infinite KL position come first.
pub fn worst_prompts(per_prompt: &[PromptShift], n: usize) -> Vec<PromptShift> {
    let mut ranked: Vec<&PromptShift> = per_prompt.iter().collect();
    ranked.sort_by(|a, b| {
        (b.infinite_positions > 0)
            .cmp(&(a.infinite_positions > 0))
            .then(
                b.kl_mean
                    .unwrap_or(f64::NEG_INFINITY)
                    .partial_cmp(&a.kl_mean.unwrap_or(f64::NEG_INFINITY))
                    .unwrap_or(CmpOrdering::Equal),
            )
            .then(a.index.cmp(&b.index))
    });
    ranked.into_iter().take(n).cloned().collect()
}

#[derive(Debug, Clone)]
pub struct LogitsRequest {
    pub before: PathBuf,
    pub after: PathBuf,
    pub tensor: Option<String>,
    pub top_k: usize,
    pub worst: usize,
    pub warn_kl: Option<f64>,
    pub warn_top1_shift: Option<f64>,
}

fn dump_info(d: &LogitsDump) -> DumpInfo {
    DumpInfo {
        path: d.path.clone(),
        format: d.format,
        tensor: d.tensor.clone(),
        dtype: d.dtype,
        shape: d.shape.clone(),
        prompts_sha256: d.meta("prompts_sha256").map(str::to_owned),
        tokenizer_sha256: d.meta("tokenizer_sha256").map(str::to_owned),
    }
}

fn prompt_kl(p: &PromptShift) -> PromptKl {
    PromptKl {
        index: p.index,
        start: p.start,
        end: p.end,
        kl_mean: p.kl_mean,
        infinite_positions: p.infinite_positions,
    }
}

/// Opens both dumps (RAM checked by `probe` for `.npy`/`.pt`), checks them, compares,
/// then applies the thresholds (strictly exceeded; an infinite KL exceeds any `warn_kl`).
pub fn run(
    req: &LogitsRequest,
    probe: &dyn HardwareProbe,
    cancel: &AtomicBool,
    on_block: &mut dyn FnMut(usize, usize),
) -> Result<BenchRun<BenchLogitsReport>> {
    for (name, value) in [
        ("--warn-kl", req.warn_kl),
        ("--warn-top1-shift", req.warn_top1_shift),
    ] {
        if value.is_some_and(|v| !(v.is_finite() && v >= 0.0)) {
            return Err(BenchError::InvalidArgument(format!(
                "{name} doit être un nombre positif"
            )));
        }
    }
    let started = Instant::now();
    let before = open_dump(&req.before, req.tensor.as_deref(), probe)?;
    let after = open_dump(&req.after, req.tensor.as_deref(), probe)?;
    let mut warnings = check_pair(&before, &after)?;
    let offsets = shared_offsets(&before, &after);
    let cmp = compare(&before, &after, req.top_k, offsets, cancel, on_block)?;

    let mut threshold_warnings = Vec::new();
    if let Some(t) = req.warn_kl {
        match (cmp.kl_infinite_positions, cmp.kl_mean) {
            (n, _) if n > 0 => threshold_warnings.push(format!(
                "KL infinie sur {n} position(s), au-delà de tout seuil ({t})"
            )),
            (_, Some(mean)) if mean > t => {
                threshold_warnings.push(format!("KL moyenne {mean:.4} > {t}"))
            }
            _ => {}
        }
    }
    if let Some(t) = req.warn_top1_shift {
        if cmp.top1_shift_rate > t {
            threshold_warnings.push(format!(
                "top-1 changé sur {:.1} % des positions > {:.1} %",
                cmp.top1_shift_rate * 100.0,
                t * 100.0
            ));
        }
    }
    warnings.extend(threshold_warnings.iter().cloned());
    let verdict = match (
        req.warn_kl.or(req.warn_top1_shift),
        threshold_warnings.is_empty(),
    ) {
        (None, _) => Verdict::None,
        (Some(_), true) => Verdict::Ok,
        (Some(_), false) => Verdict::Warn,
    };
    let report = BenchLogitsReport {
        before: dump_info(&before),
        after: dump_info(&after),
        positions: cmp.positions,
        vocab_size: cmp.vocab,
        kl_direction: "KL(before‖after)".into(),
        kl_mean: cmp.kl_mean,
        kl_max: cmp.kl_max,
        kl_max_position: cmp.kl_max_position,
        kl_p95: cmp.kl_p95,
        kl_infinite_positions: cmp.kl_infinite_positions,
        cosine_probs_mean: cmp.cosine_probs_mean,
        cosine_logits_mean: cmp.cosine_logits_mean,
        cosine_logits_positions: cmp.cosine_logits_positions,
        top1_shift_rate: cmp.top1_shift_rate,
        k: cmp.k,
        topk_set_change_rate: cmp.topk_set_change_rate,
        topk_overlap_mean: cmp.topk_overlap_mean,
        prompts: offsets.map(|o| o.len() - 1),
        worst_prompts: worst_prompts(&cmp.per_prompt, req.worst)
            .iter()
            .map(prompt_kl)
            .collect(),
        per_prompt: cmp.per_prompt.iter().map(prompt_kl).collect(),
        warn_kl: req.warn_kl,
        warn_top1_shift: req.warn_top1_shift,
        verdict,
        timing_ms: LogitsTiming {
            total: started.elapsed().as_millis() as u64,
        },
    };
    Ok(BenchRun {
        report,
        warnings,
        threshold_warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(a: &[f64], b: &[f64], k: usize) -> PositionMetrics {
        position_metrics(a, b, k).unwrap()
    }

    fn kl(m: &PositionMetrics) -> f64 {
        match m.kl {
            Kl::Finite(v) => v,
            Kl::Infinite => f64::INFINITY,
        }
    }

    #[test]
    fn analytic_kl_and_identity() {
        let m = metrics(&[0.0, 0.0], &[3f64.ln(), 0.0], 1);
        let expected = 0.5 * (2.0f64 / 3.0).ln() + 0.5 * 2f64.ln();
        assert!((kl(&m) - expected).abs() < 1e-12, "{}", kl(&m));
        assert!((kl(&m) - 0.143_841_036_225_890_3).abs() < 1e-12);

        let row = [1.5, -2.25, 0.125, 7.0, -0.5];
        let same = metrics(&row, &row, 2);
        assert_eq!(same.kl, Kl::Finite(0.0));
        assert_eq!(same.cosine_probs, 1.0);
        assert!(!same.top1_changed && !same.topk_changed);
        assert_eq!(same.topk_overlap, 1.0);
    }

    #[test]
    fn invariant_to_a_constant_shift_and_stable_on_huge_logits() {
        let a = [0.3, -1.7, 2.9, 0.0, 1.1];
        let b = [0.1, -1.2, 2.5, 0.4, 1.0];
        let base = metrics(&a, &b, 3);
        for c in [1000.0, -1000.0] {
            let shifted: Vec<f64> = b.iter().map(|v| v + c).collect();
            let m = metrics(&a, &shifted, 3);
            assert!((kl(&m) - kl(&base)).abs() < 1e-9);
            assert!((m.cosine_probs - base.cosine_probs).abs() < 1e-12);
            assert_eq!(
                (m.top1_changed, m.topk_changed, m.topk_overlap),
                (base.top1_changed, base.topk_changed, base.topk_overlap)
            );
        }
        let m = metrics(&[1e4, -1e4, 0.0], &[-1e4, 1e4, 0.0], 1);
        assert!(kl(&m).is_finite() && kl(&m) > 1e4, "{}", kl(&m));
        assert!(m.top1_changed);
    }

    #[test]
    fn masking_and_infinite_kl() {
        let ninf = f64::NEG_INFINITY;
        // Masked on both sides: the entry has no mass and no effect.
        let m = metrics(&[0.0, 0.0, ninf], &[0.0, 0.0, ninf], 1);
        assert_eq!(m.kl, Kl::Finite(0.0));
        // Masked before only: q has mass where p has none, KL(p‖q) stays finite.
        let m = metrics(&[0.0, ninf], &[0.0, 0.0], 1);
        assert!((kl(&m) - 2f64.ln()).abs() < 1e-12);
        // p_i > 0 and q_i = 0: infinite.
        let m = metrics(&[0.0, 0.0], &[0.0, ninf], 1);
        assert_eq!(m.kl, Kl::Infinite);
        assert_eq!(
            position_metrics(&[f64::NAN, 0.0], &[0.0, 0.0], 1),
            Err(BadSide::Before)
        );
        assert_eq!(
            position_metrics(&[0.0, 0.0], &[f64::INFINITY, 0.0], 1),
            Err(BadSide::After)
        );
        assert_eq!(
            position_metrics(&[ninf, ninf], &[0.0, 0.0], 1),
            Err(BadSide::Before)
        );
    }

    #[test]
    fn top_k_sets_ties_and_overlap() {
        assert_eq!(top_k(&[1.0, 3.0, 3.0, 2.0], 2), [1, 2]);
        assert_eq!(
            top_k(&[5.0, 5.0, 5.0], 2),
            [0, 1],
            "ties to the lowest index"
        );
        assert_eq!(argmax(&[2.0, 7.0, 7.0]), 1);
        let a = [9.0, 8.0, 7.0, 6.0, 5.0, 1.0, 0.0];
        // Permutation inside the top-5: same set, argmax changed.
        let permuted = [5.0, 8.0, 7.0, 6.0, 9.0, 1.0, 0.0];
        let m = metrics(&a, &permuted, 5);
        assert!(m.top1_changed && !m.topk_changed);
        assert_eq!(m.topk_overlap, 1.0);
        // One token replaced: set changed, overlap 4/5, argmax kept.
        let replaced = [9.0, 8.0, 7.0, 6.0, 0.0, 1.0, 5.5];
        let m = metrics(&a, &replaced, 5);
        assert!(!m.top1_changed && m.topk_changed);
        assert!((m.topk_overlap - 0.8).abs() < 1e-15);
        // k = 1 equals the top-1 test.
        let m = metrics(&a, &permuted, 1);
        assert!(m.topk_changed && m.topk_overlap == 0.0);
    }

    /// Deterministic pseudo-random logits in [-8, 8], with exact ties now and then.
    fn random_rows(positions: usize, vocab: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..positions * vocab)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let v = ((state >> 40) as f32 / (1u64 << 24) as f32) * 16.0 - 8.0;
                if state.is_multiple_of(17) {
                    1.0
                } else {
                    v
                }
            })
            .collect()
    }

    fn compare_f32(a: &[f32], b: &[f32], positions: usize, vocab: usize, k: usize) -> Comparison {
        let never = AtomicBool::new(false);
        compare_rows(
            Rows::F32(a),
            Rows::F32(b),
            positions,
            vocab,
            k,
            Some(&[0, 50, 50, positions]),
            &never,
            &mut |_, _| {},
        )
        .unwrap()
    }

    #[test]
    fn identical_whatever_the_thread_count() {
        let (positions, vocab, k) = (300, 97, 5);
        let a = random_rows(positions, vocab, 1);
        let b = random_rows(positions, vocab, 2);
        let runs: Vec<Comparison> = [1, 2, 8]
            .iter()
            .map(|&threads| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| compare_f32(&a, &b, positions, vocab, k))
            })
            .collect();
        for run in &runs[1..] {
            assert_eq!(run, &runs[0], "bit-identical across thread counts");
            assert_eq!(
                run.kl_mean.unwrap().to_bits(),
                runs[0].kl_mean.unwrap().to_bits()
            );
        }
        let again = compare_f32(&a, &b, positions, vocab, k);
        assert_eq!(again, runs[0]);
        assert_eq!(runs[0].per_prompt.len(), 3);
        assert_eq!(runs[0].per_prompt[1].kl_mean, None, "empty prompt");
    }

    /// Plain loops, no log-sum-exp helper, no compensated sum, sort-based top-k.
    #[test]
    fn parallel_result_equals_an_independent_sequential_reference() {
        let (positions, vocab, k) = (120, 31, 3);
        let a = random_rows(positions, vocab, 7);
        let b = random_rows(positions, vocab, 8);
        let got = compare_f32(&a, &b, positions, vocab, k);

        let softmax = |row: &[f32]| -> Vec<f64> {
            let max = row.iter().fold(f64::MIN, |m, &v| m.max(f64::from(v)));
            let exps: Vec<f64> = row.iter().map(|&v| (f64::from(v) - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            exps.iter().map(|e| e / sum).collect()
        };
        let topk = |row: &[f32]| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..row.len()).collect();
            idx.sort_by(|&i, &j| row[j].partial_cmp(&row[i]).unwrap().then(i.cmp(&j)));
            let mut set = idx[..k].to_vec();
            set.sort_unstable();
            set
        };
        let (mut kl_sum, mut kl_max, mut top1, mut topk_changed, mut cos_sum) =
            (0.0, 0.0f64, 0, 0, 0.0);
        for pos in 0..positions {
            let ra = &a[pos * vocab..(pos + 1) * vocab];
            let rb = &b[pos * vocab..(pos + 1) * vocab];
            let (p, q) = (softmax(ra), softmax(rb));
            let kl: f64 = p.iter().zip(&q).map(|(p, q)| p * (p / q).ln()).sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            let (sa, sb) = (topk(ra), topk(rb));
            let first_max = |r: &[f32]| (0..vocab).fold(0, |m, i| if r[i] > r[m] { i } else { m });
            top1 += usize::from(first_max(ra) != first_max(rb));
            topk_changed += usize::from(sa != sb);
            let dot: f64 = p.iter().zip(&q).map(|(x, y)| x * y).sum();
            let n = |v: &[f64]| v.iter().map(|x| x * x).sum::<f64>().sqrt();
            cos_sum += dot / (n(&p) * n(&q));
        }
        let n = positions as f64;
        assert!((got.kl_mean.unwrap() - kl_sum / n).abs() < 1e-12);
        assert!((got.kl_max.unwrap() - kl_max).abs() < 1e-12);
        assert_eq!(got.top1_shift_rate, top1 as f64 / n);
        assert_eq!(got.topk_set_change_rate, topk_changed as f64 / n);
        assert!((got.cosine_probs_mean - cos_sum / n).abs() < 1e-12);
    }

    #[test]
    fn non_finite_rows_are_reported_with_their_positions_and_cancel_stops() {
        let (positions, vocab) = (10, 4);
        let mut a = random_rows(positions, vocab, 3);
        let b = random_rows(positions, vocab, 4);
        for pos in [2, 7] {
            a[pos * vocab + 1] = f32::NAN;
        }
        let never = AtomicBool::new(false);
        let err = compare_rows(
            Rows::F32(&a),
            Rows::F32(&b),
            positions,
            vocab,
            1,
            None,
            &never,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(
            matches!(&err, BenchError::NonFiniteLogits { side: "avant", positions } if positions == &[2, 7]),
            "{err:?}"
        );
        let stop = AtomicBool::new(true);
        let ok = random_rows(positions, vocab, 3);
        let err = compare_rows(
            Rows::F32(&ok),
            Rows::F32(&b),
            positions,
            vocab,
            1,
            None,
            &stop,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(matches!(err, BenchError::Cancelled));
    }

    #[test]
    fn worst_prompts_order() {
        let p = |index, kl_mean, infinite_positions| PromptShift {
            index,
            start: 0,
            end: 1,
            kl_mean,
            infinite_positions,
        };
        let all = [
            p(0, Some(0.1), 0),
            p(1, Some(0.5), 0),
            p(2, Some(0.5), 0),
            p(3, None, 1),
        ];
        let worst: Vec<usize> = worst_prompts(&all, 3).iter().map(|w| w.index).collect();
        assert_eq!(worst, [3, 1, 2]);
        assert_eq!(p95(&mut []), None);
        let mut v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(p95(&mut v), Some(95.0));
    }
}
