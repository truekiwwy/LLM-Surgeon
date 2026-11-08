//! v = normalize(mean(pos) - mean(neg)), accumulated in f64, and how well v separates the
//! two prompt sets.

use serde::Serialize;

use super::error::{ExtractError, Result};

/// Below this norm the two means coincide: writing a direction would be noise.
pub const MIN_DIFF_NORM: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct SeparationStats {
    /// ‖mean(pos) - mean(neg)‖₂ before normalization.
    pub raw_diff_norm: f64,
    /// cos(mean(pos), mean(neg)); 0 when one mean is the zero vector.
    pub cosine_means: f64,
    /// Mean projection of each set on v.
    pub pos_mean_proj: f64,
    pub neg_mean_proj: f64,
    /// (pos_mean_proj - neg_mean_proj) / pooled standard deviation of the projections;
    /// `None` when both sets have zero spread (perfect, degenerate-free separation).
    pub d_prime: Option<f64>,
    /// Best accuracy of a single threshold on the projections (pos above, neg below).
    pub threshold_accuracy: f64,
    /// That threshold: a prompt projecting above it is classified positive.
    pub threshold: f64,
}

/// Everything the extraction computes: the stored direction, its diagnostics, and the
/// projection of every prompt on it (written to the companion `.stats.json`).
#[derive(Debug, Clone, PartialEq)]
pub struct Separation {
    pub direction: Vec<f32>,
    pub stats: SeparationStats,
    pub pos_projections: Vec<f64>,
    pub neg_projections: Vec<f64>,
}

fn mean(vectors: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut sums = vec![0f64; dim];
    for v in vectors {
        for (s, x) in sums.iter_mut().zip(v) {
            *s += x;
        }
    }
    sums.iter().map(|s| s / vectors.len() as f64).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn variance(values: &[f64], mean: f64) -> f64 {
    match values.len() {
        0 | 1 => 0.0,
        n => values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64,
    }
}

/// Unit direction (f32, as stored) and the diagnostics. Both sets must be non-empty and
/// share one dimension.
pub fn separation(pos: &[Vec<f64>], neg: &[Vec<f64>]) -> Result<(Vec<f32>, SeparationStats)> {
    separation_with_projections(pos, neg).map(|s| (s.direction, s.stats))
}

/// [`separation`] plus the per-prompt projections.
pub fn separation_with_projections(pos: &[Vec<f64>], neg: &[Vec<f64>]) -> Result<Separation> {
    let dim = pos.first().map(Vec::len).unwrap_or(0);
    if pos.is_empty() || neg.is_empty() || pos.iter().chain(neg).any(|v| v.len() != dim) || dim == 0
    {
        return Err(ExtractError::Compute(
            "ensembles d'activations vides ou de tailles différentes".into(),
        ));
    }
    let (mean_pos, mean_neg) = (mean(pos, dim), mean(neg, dim));
    let diff: Vec<f64> = mean_pos.iter().zip(&mean_neg).map(|(p, n)| p - n).collect();
    let norm = dot(&diff, &diff).sqrt();
    if !norm.is_finite() || norm < MIN_DIFF_NORM {
        return Err(ExtractError::DegenerateDirection { norm });
    }
    let unit: Vec<f64> = diff.iter().map(|d| d / norm).collect();

    let (norm_pos, norm_neg) = (
        dot(&mean_pos, &mean_pos).sqrt(),
        dot(&mean_neg, &mean_neg).sqrt(),
    );
    let cosine_means = match norm_pos * norm_neg {
        n if n > 0.0 => dot(&mean_pos, &mean_neg) / n,
        _ => 0.0,
    };
    let proj_pos: Vec<f64> = pos.iter().map(|v| dot(v, &unit)).collect();
    let proj_neg: Vec<f64> = neg.iter().map(|v| dot(v, &unit)).collect();
    let pos_mean_proj = proj_pos.iter().sum::<f64>() / proj_pos.len() as f64;
    let neg_mean_proj = proj_neg.iter().sum::<f64>() / proj_neg.len() as f64;
    let pooled_sd =
        ((variance(&proj_pos, pos_mean_proj) + variance(&proj_neg, neg_mean_proj)) / 2.0).sqrt();
    let d_prime = (pooled_sd > MIN_DIFF_NORM).then(|| (pos_mean_proj - neg_mean_proj) / pooled_sd);

    let (threshold_accuracy, threshold) = best_threshold(&proj_pos, &proj_neg);
    let stats = SeparationStats {
        raw_diff_norm: norm,
        cosine_means,
        pos_mean_proj,
        neg_mean_proj,
        d_prime,
        threshold_accuracy,
        threshold,
    };
    Ok(Separation {
        direction: unit.iter().map(|&u| u as f32).collect(),
        stats,
        pos_projections: proj_pos,
        neg_projections: proj_neg,
    })
}

/// Sweeps every cut between sorted projections: classify `> cut` as positive.
/// Accuracy only (kept for the prompt 8 unit tests).
#[cfg(test)]
fn threshold_accuracy(pos: &[f64], neg: &[f64]) -> f64 {
    best_threshold(pos, neg).0
}

/// (best accuracy, its threshold). The threshold sits halfway between the two values the
/// best cut separates; below every value when classifying all as positive is best.
fn best_threshold(pos: &[f64], neg: &[f64]) -> (f64, f64) {
    let mut all: Vec<(f64, bool)> = pos
        .iter()
        .map(|&p| (p, true))
        .chain(neg.iter().map(|&n| (n, false)))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total = all.len();
    // Cut below everything: every positive right, every negative wrong.
    let mut correct = pos.len();
    let mut best = correct;
    let mut threshold = all.first().map_or(0.0, |f| f.0 - 1.0);
    let mut i = 0;
    while i < total {
        // Move every element with the same value at once: a cut cannot split ties.
        let value = all[i].0;
        while i < total && all[i].0 == value {
            if all[i].1 {
                correct -= 1;
            } else {
                correct += 1;
            }
            i += 1;
        }
        if correct > best {
            best = correct;
            threshold = all.get(i).map_or(value, |next| (value + next.0) / 2.0);
        }
    }
    (best as f64 / total.max(1) as f64, threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_matches_an_independent_reference() {
        let pos = vec![vec![2.0, 0.0, 1.0], vec![4.0, 2.0, 1.0]];
        let neg = vec![
            vec![0.0, 0.0, 1.0],
            vec![0.0, 2.0, 1.0],
            vec![0.0, 1.0, 1.0],
        ];
        // mean(pos) = (3, 1, 1); mean(neg) = (0, 1, 1); d = (3, 0, 0); v = (1, 0, 0).
        let (v, stats) = separation(&pos, &neg).unwrap();
        assert_eq!(v, [1.0, 0.0, 0.0]);
        assert!((stats.raw_diff_norm - 3.0).abs() < 1e-12);
        let cos = (0.0 + 1.0 + 1.0) / (11f64.sqrt() * 2f64.sqrt());
        assert!((stats.cosine_means - cos).abs() < 1e-12);
        assert_eq!((stats.pos_mean_proj, stats.neg_mean_proj), (3.0, 0.0));
        // Projections: pos {2, 4} (var 2), neg {0, 0, 0} (var 0): d' = 3 / sqrt(1).
        assert!((stats.d_prime.unwrap() - 3.0).abs() < 1e-12);
        assert_eq!(stats.threshold_accuracy, 1.0);
    }

    #[test]
    fn unit_norm_and_overlap_accuracy() {
        let pos: Vec<Vec<f64>> = (0..8)
            .map(|i| vec![1.0 + i as f64 * 0.1, 0.3, -0.2])
            .collect();
        let neg: Vec<Vec<f64>> = (0..8)
            .map(|i| vec![1.5 - i as f64 * 0.1, 0.1, 0.4])
            .collect();
        let (v, stats) = separation(&pos, &neg).unwrap();
        let norm: f64 = v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!(stats.threshold_accuracy > 0.5 && stats.threshold_accuracy <= 1.0);
        assert_eq!(
            threshold_accuracy(&[1.0, 2.0], &[3.0, 4.0]),
            0.5,
            "inverted sets: best cut is trivial"
        );
        assert_eq!(
            threshold_accuracy(&[1.0], &[1.0]),
            0.5,
            "ties cannot be split"
        );
    }

    #[test]
    fn identical_sets_are_degenerate() {
        let same = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(matches!(
            separation(&same, &same),
            Err(ExtractError::DegenerateDirection { .. })
        ));
        assert!(separation(&[], &same).is_err());
        assert!(separation(&same, &[vec![1.0]]).is_err());
    }
}
