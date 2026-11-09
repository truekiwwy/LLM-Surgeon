//! Contrastive PCA on paired differences: `ΔX_i = x⁺_i − x⁻_i` (prompts paired by index),
//! direction = first right singular vector of ΔX, i.e. the dominant eigenvector of
//! `(1/N) ΔXᵀΔX`.
//!
//! The matrix is NOT centered by default. Centering would subtract the mean delta, which is
//! `μ₊ − μ₋`, the very axis of the concept: on pairs whose delta always points the same way
//! the centered PCA keeps only the spread around it. `--svd-center` is for pairs whose
//! order (and so the sign of the delta) is arbitrary, where the mean delta is noise.
//!
//! Sign: `v` is oriented so that `⟨v, μ₊ − μ₋⟩ ≥ 0`; when that product is exactly zero, the
//! first non-zero component is made positive.

use schemars::JsonSchema;
use serde::Serialize;

use super::error::{ExtractError, Result};
use crate::linalg::gram::thin_svd;

/// Fewer pairs than this give an unstable direction.
pub const MIN_RECOMMENDED_PAIRS: usize = 8;
/// σ₂/σ₁ above this: the first direction is barely dominant.
pub const UNSTABLE_SIGMA_RATIO: f64 = 0.9;

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct SvdDiagnostics {
    /// Paires utilisées.
    pub n_pairs: usize,
    /// Matrice des deltas centrée (--svd-center).
    pub centered: bool,
    /// Rang numérique de ΔX.
    pub rank: usize,
    /// Premières valeurs singulières de ΔX (5 au plus).
    pub sigma: Vec<f64>,
    /// σ₁² / Σσ² : part de l'énergie des deltas portée par la direction.
    pub explained_variance_ratio: f64,
    /// Cosinus avec μ₊ − μ₋ ; null si les moyennes coïncident.
    pub cosine_with_mean_diff: Option<f64>,
    /// Cosinus entre les directions des paires d'indices pairs et impairs (stabilité) ; null sous 2 paires.
    pub split_half_cosine: Option<f64>,
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Flips `v` to the orientation of `reference` (see the module documentation).
pub(crate) fn orient(v: &mut [f64], reference: &[f64]) {
    let along = dot(v, reference);
    let flip = match along == 0.0 {
        true => v.iter().find(|x| **x != 0.0).is_some_and(|x| *x < 0.0),
        false => along < 0.0,
    };
    if flip {
        v.iter_mut().for_each(|x| *x = -*x);
    }
}

fn mean_rows(rows: &[Vec<f64>], d: usize) -> Vec<f64> {
    let mut m = vec![0.0; d];
    for r in rows {
        m.iter_mut().zip(r).for_each(|(a, b)| *a += b);
    }
    m.iter_mut().for_each(|a| *a /= rows.len() as f64);
    m
}

/// (direction, singular values, rank) of the deltas, `None` at rank 0.
type Leading = Option<(Vec<f64>, Vec<f64>, usize)>;

/// First right singular vector of the (optionally centered) deltas, oriented along
/// `mean_diff`, with the singular values; `None` when the deltas have rank 0.
fn leading(deltas: &[Vec<f64>], center: bool, mean_diff: &[f64]) -> Result<Leading> {
    let (n, d) = (deltas.len(), mean_diff.len());
    let offset = match center {
        true => mean_rows(deltas, d),
        false => vec![0.0; d],
    };
    let flat: Vec<f64> = deltas
        .iter()
        .flat_map(|r| r.iter().zip(&offset).map(|(x, o)| x - o))
        .collect();
    let svd = thin_svd(&flat, n, d).map_err(|e| ExtractError::Compute(e.to_string()))?;
    if svd.rank == 0 {
        return Ok(None);
    }
    let mut v = svd.vector(0).to_vec();
    orient(&mut v, mean_diff);
    Ok(Some((v, svd.sigma, svd.rank)))
}

/// Unit direction, diagnostics and warnings. `pos[i]` is paired with `neg[i]`.
pub fn svd_direction(
    pos: &[Vec<f64>],
    neg: &[Vec<f64>],
    center: bool,
) -> Result<(Vec<f64>, SvdDiagnostics, Vec<String>)> {
    if pos.len() != neg.len() {
        return Err(ExtractError::UnpairedPrompts {
            pos: pos.len(),
            neg: neg.len(),
        });
    }
    let d = pos.first().map_or(0, Vec::len);
    let deltas: Vec<Vec<f64>> = pos
        .iter()
        .zip(neg)
        .map(|(p, q)| p.iter().zip(q).map(|(a, b)| a - b).collect())
        .collect();
    let mean_diff = mean_rows(&deltas, d);
    let (v, sigma, rank) = leading(&deltas, center, &mean_diff)?
        .ok_or(ExtractError::DegenerateDirection { norm: 0.0 })?;

    let energy: f64 = sigma.iter().map(|s| s * s).sum();
    let md_norm = dot(&mean_diff, &mean_diff).sqrt();
    let half = |parity: usize| -> Vec<Vec<f64>> {
        deltas.iter().skip(parity).step_by(2).cloned().collect()
    };
    let split_half_cosine = match deltas.len() >= 2 {
        false => None,
        true => match (
            leading(&half(0), center, &mean_diff)?,
            leading(&half(1), center, &mean_diff)?,
        ) {
            (Some((a, ..)), Some((b, ..))) => Some(dot(&a, &b).clamp(-1.0, 1.0)),
            _ => None,
        },
    };
    let mut warnings = Vec::new();
    if deltas.len() < MIN_RECOMMENDED_PAIRS {
        warnings.push(format!(
            "seulement {} paires (au moins {MIN_RECOMMENDED_PAIRS} recommandées) : direction SVD peu fiable",
            deltas.len()
        ));
    }
    if let [s1, s2, ..] = sigma[..] {
        if s1 > 0.0 && s2 / s1 > UNSTABLE_SIGMA_RATIO {
            warnings.push(format!(
                "σ₂/σ₁ = {:.3} > {UNSTABLE_SIGMA_RATIO} : la première direction domine à peine (instable)",
                s2 / s1
            ));
        }
    }
    let diagnostics = SvdDiagnostics {
        n_pairs: deltas.len(),
        centered: center,
        rank,
        sigma: sigma.iter().take(5).copied().collect(),
        explained_variance_ratio: match energy > 0.0 {
            true => sigma[0] * sigma[0] / energy,
            false => 0.0,
        },
        cosine_with_mean_diff: (md_norm > 0.0)
            .then(|| (dot(&v, &mean_diff) / md_norm).clamp(-1.0, 1.0)),
        split_half_cosine,
    };
    Ok((v, diagnostics, warnings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Lcg;

    fn unit(d: usize, axis: usize) -> Vec<f64> {
        (0..d).map(|j| f64::from(u8::from(j == axis))).collect()
    }

    /// pos_i = base_i + c_i t + noise, neg_i = base_i + noise: rank-1 concept t planted with
    /// per-pair strength c_i ∈ [1, 2].
    fn pairs(
        n: usize,
        d: usize,
        noise: f64,
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Lcg(seed);
        let t = unit(d, 3);
        let (mut pos, mut neg) = (Vec::new(), Vec::new());
        for _ in 0..n {
            let base: Vec<f64> = (0..d).map(|_| rng.next() * 5.0).collect();
            let c = 1.5 + 0.5 * rng.next();
            pos.push(
                base.iter()
                    .zip(&t)
                    .map(|(b, t)| b + c * t + noise * rng.next())
                    .collect(),
            );
            neg.push(base.iter().map(|b| b + noise * rng.next()).collect());
        }
        (pos, neg, t)
    }

    #[test]
    fn recovers_a_planted_rank_one_concept() {
        let (d, n, noise) = (40, 64, 0.1);
        let (pos, neg, t) = pairs(n, d, noise, 1);
        let (v, diag, warnings) = svd_direction(&pos, &neg, false).unwrap();
        // Delta noise per coordinate: difference of two U(-0.1, 0.1), variance 2·0.1²/3; with
        // signal strength ≥ 1 per pair, the error angle is about sqrt(d·var / Σc²) ≈ 0.012:
        // a cosine above 0.99 leaves a wide margin.
        assert!(dot(&v, &t) > 0.99, "{}", dot(&v, &t));
        assert!(diag.explained_variance_ratio > 0.9);
        assert!(diag.cosine_with_mean_diff.unwrap() > 0.99);
        let split = diag.split_half_cosine.unwrap();
        assert!((-1.0..=1.0).contains(&split) && split > 0.99);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert!((dot(&v, &v) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sign_follows_the_mean_difference() {
        let (pos, neg, t) = pairs(10, 12, 0.05, 2);
        let (v, ..) = svd_direction(&pos, &neg, false).unwrap();
        assert!(dot(&v, &t) > 0.0);
        let (w, ..) = svd_direction(&neg, &pos, false).unwrap();
        assert!(dot(&w, &t) < 0.0, "swapping the sets flips the direction");
        let mut z = vec![0.0, -2.0, 1.0];
        orient(&mut z, &[1.0, 0.0, 0.0]);
        assert_eq!(
            z,
            [0.0, 2.0, -1.0],
            "zero product: first non-zero component positive"
        );
    }

    #[test]
    fn centering_loses_a_constant_delta_axis() {
        // Every delta = t + small noise orthogonal to t: the axis is the mean delta itself.
        let mut rng = Lcg(3);
        let d = 10;
        let t = unit(d, 0);
        let (mut pos, mut neg) = (Vec::new(), Vec::new());
        for _ in 0..20 {
            let base: Vec<f64> = (0..d).map(|_| rng.next()).collect();
            let wobble: Vec<f64> = (0..d)
                .map(|j| if j == 0 { 0.0 } else { 0.05 * rng.next() })
                .collect();
            neg.push(base.clone());
            pos.push(
                base.iter()
                    .zip(&t)
                    .zip(&wobble)
                    .map(|((b, t), w)| b + t + w)
                    .collect(),
            );
        }
        let (plain, ..) = svd_direction(&pos, &neg, false).unwrap();
        let (centered, ..) = svd_direction(&pos, &neg, true).unwrap();
        assert!(dot(&plain, &t) > 0.99);
        assert!(dot(&centered, &t).abs() < 0.1, "{}", dot(&centered, &t));
    }

    #[test]
    fn unpaired_small_and_degenerate_inputs() {
        let (pos, neg, _) = pairs(5, 6, 0.1, 4);
        assert!(matches!(
            svd_direction(&pos, &neg[..4], false),
            Err(ExtractError::UnpairedPrompts { pos: 5, neg: 4 })
        ));
        let (v, diag, warnings) = svd_direction(&pos[..1], &neg[..1], false).unwrap();
        assert_eq!(diag.n_pairs, 1);
        assert_eq!(diag.split_half_cosine, None);
        assert!((dot(&v, &v) - 1.0).abs() < 1e-12);
        assert!(warnings.iter().any(|w| w.contains("paires")));
        assert!(matches!(
            svd_direction(&pos[..1], &neg[..1], true),
            Err(ExtractError::DegenerateDirection { .. })
        ));
        assert!(matches!(
            svd_direction(&pos, &pos, false),
            Err(ExtractError::DegenerateDirection { .. })
        ));
    }

    #[test]
    fn noisy_isotropic_deltas_warn_about_instability() {
        let mut rng = Lcg(5);
        // 12 random pairs in d = 400: the singular values concentrate (Marchenko-Pastur, ratio
        // 12/400), so σ₂/σ₁ is close to 1.
        let d = 400;
        let pos: Vec<Vec<f64>> = (0..12)
            .map(|_| (0..d).map(|_| rng.next()).collect())
            .collect();
        let neg: Vec<Vec<f64>> = (0..12)
            .map(|_| (0..d).map(|_| rng.next()).collect())
            .collect();
        let (_, diag, warnings) = svd_direction(&pos, &neg, true).unwrap();
        assert!(diag.sigma.len() == 5 && diag.sigma.windows(2).all(|w| w[0] >= w[1]));
        assert!(warnings.iter().any(|w| w.contains("σ₂/σ₁")), "{warnings:?}");
    }
}
