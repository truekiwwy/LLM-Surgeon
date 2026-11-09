//! LEACE (LEAst-squares Concept Erasure), binary concept, as a rank-1 AFFINE eraser:
//!
//! ```text
//! X: every pooled activation (pos then neg), z ∈ {0, 1}; μ = mean(X); X̃ = X − μ
//! Σ = X̃ᵀX̃ / n ; c = μ₊ − μ₋ ; Σ_λ = (1 − λ) Σ + λ s I with s = tr(Σ) / d
//! w = Σ_λ^{-1/2} c ; u = w / ‖w‖ ; a = Σ_λ^{1/2} u (∥ c) ; b = Σ_λ^{-1/2} u (∥ Σ_λ⁻¹ c)
//! r(x) = x − a bᵀ (x − μ), with bᵀa = 1
//! ```
//!
//! `Σ^{-1/2} μ_Δ` alone (the naive formula) is only the whitened direction `w`: the eraser
//! needs both the write direction `a` and the read direction `b`.
//!
//! `Σ_λ^p y` is computed without any d×d matrix, from the thin SVD of `X̃/√n`
//! (`Σ = V diag(σ²) Vᵀ`): `V diag(((1−λ)σ² + λs)^p) Vᵀ y + (λs)^p (y − V Vᵀ y)`. With
//! `λ = 0` this is the pseudo-inverse restricted to the data subspace: eigenvalues below
//! `rcond · λ_max` are dropped and the complement term vanishes.
//!
//! Honest limits: `cov(r(x), z) = p(1−p)(c − a bᵀ c) = 0` holds EXACTLY in sample for any
//! positive definite `Σ_λ` (since `a bᵀ c = c`). The guarantee that no linear predictor
//! beats a constant is a population statement: with n < d a classifier can still separate
//! the erased sample, and only the held-out diagnostic says whether the erasure
//! generalizes. Erasing the concept from the whole residual stream would need the eraser
//! on every write into it; a weight patch of one layer does not give that guarantee.

use schemars::JsonSchema;
use serde::Serialize;

use super::error::{ExtractError, Result};
use super::svd::dot;
use crate::linalg::gram::thin_svd;

#[derive(Debug, Clone, PartialEq)]
pub struct LeaceFit {
    pub a: Vec<f64>,
    pub b: Vec<f64>,
    pub mean: Vec<f64>,
    pub u_whitened: Vec<f64>,
    /// Kept eigen-directions of Σ.
    pub rank: usize,
    /// λ_max(Σ_λ) / λ_min(Σ_λ) (on the kept subspace when λ = 0).
    pub kappa: f64,
    pub shrinkage: f64,
    pub b_dot_a: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct LeaceDiagnostics {
    /// λ du rétrécissement Σ_λ = (1−λ)Σ + λ·tr(Σ)/d·I (0 : pseudo-inverse sur le sous-espace).
    pub shrinkage: f64,
    /// Seuil relatif des valeurs propres de Σ conservées quand λ = 0.
    pub rcond: f64,
    /// Rang effectif retenu.
    pub rank: usize,
    /// Conditionnement de Σ_λ (sur le sous-espace retenu si λ = 0).
    pub kappa: f64,
    /// bᵀa (1 par construction).
    pub b_dot_a: f64,
    /// ‖cov(x, z)‖ sur l'échantillon d'ajustement.
    pub cross_cov_before: f64,
    /// ‖cov(r(x), z)‖ après effacement, même échantillon (≈ 0 par construction).
    pub cross_cov_after: f64,
    /// ‖moyenne(r(x)|pos) − moyenne(r(x)|neg)‖ après effacement.
    pub class_mean_gap_after: f64,
    /// cross_cov_after / cross_cov_before en échantillon.
    pub in_sample_cross_cov_ratio: f64,
    /// Même ratio sur la part retenue, l'effaceur étant ajusté sur le reste ; null sans --holdout-frac.
    pub holdout_cross_cov_ratio: Option<f64>,
    pub n_pos: usize,
    pub n_neg: usize,
}

fn mean_of<'a>(rows: impl Iterator<Item = &'a Vec<f64>>, d: usize) -> Vec<f64> {
    let mut m = vec![0.0; d];
    let mut count = 0usize;
    for r in rows {
        m.iter_mut().zip(r).for_each(|(a, b)| *a += b);
        count += 1;
    }
    m.iter_mut().for_each(|a| *a /= count.max(1) as f64);
    m
}

pub fn fit(pos: &[Vec<f64>], neg: &[Vec<f64>], shrinkage: f64, rcond: f64) -> Result<LeaceFit> {
    let d = pos.first().map_or(0, Vec::len);
    let n = pos.len() + neg.len();
    if pos.is_empty() || neg.is_empty() || d == 0 {
        return Err(ExtractError::Compute(
            "ensembles d'activations vides".into(),
        ));
    }
    if !(0.0..=1.0).contains(&shrinkage) || !(rcond > 0.0 && rcond < 1.0) {
        return Err(ExtractError::InvalidArgument(
            "--leace-shrinkage doit être dans [0, 1] et --leace-rcond dans ]0, 1[".into(),
        ));
    }
    let all = || pos.iter().chain(neg);
    let mean = mean_of(all(), d);
    let c: Vec<f64> = mean_of(pos.iter(), d)
        .iter()
        .zip(mean_of(neg.iter(), d))
        .map(|(p, q)| p - q)
        .collect();
    let c_norm = dot(&c, &c).sqrt();
    if c_norm.is_nan() || c_norm <= super::MIN_DIFF_NORM {
        return Err(ExtractError::DegenerateDirection { norm: c_norm });
    }
    let scale = 1.0 / (n as f64).sqrt();
    let centered: Vec<f64> = all()
        .flat_map(|x| x.iter().zip(&mean).map(move |(v, m)| (v - m) * scale))
        .collect();
    let svd = thin_svd(&centered, n, d).map_err(|e| ExtractError::Compute(e.to_string()))?;
    let eig: Vec<f64> = svd.sigma.iter().map(|s| s * s).collect();
    let e_max = eig.first().copied().unwrap_or(0.0);
    let s = eig.iter().sum::<f64>() / d as f64;
    let lam = shrinkage;
    let rank = match lam > 0.0 {
        true => svd.rank,
        false => eig[..svd.rank]
            .iter()
            .take_while(|&&e| e > rcond * e_max)
            .count(),
    };
    let floor = lam * s;
    let power = |p: f64, y: &[f64]| -> Vec<f64> {
        let coeffs = svd.project(y);
        let scaled: Vec<f64> = coeffs
            .iter()
            .zip(&eig)
            .take(rank)
            .map(|(cf, e)| cf * ((1.0 - lam) * e + floor).powf(p))
            .collect();
        let mut out = svd.combine(&scaled);
        if lam > 0.0 {
            // Complement of the kept subspace: eigenvalue λ s.
            let inside = svd.combine(&coeffs[..rank]);
            let f = floor.powf(p);
            out.iter_mut()
                .zip(y.iter().zip(&inside))
                .for_each(|(o, (yi, vi))| *o += f * (yi - vi));
        }
        out
    };
    let w = power(-0.5, &c);
    let w_norm = dot(&w, &w).sqrt();
    if !(w_norm > 0.0 && w_norm.is_finite()) {
        return Err(ExtractError::DegenerateDirection { norm: w_norm });
    }
    let u: Vec<f64> = w.iter().map(|x| x / w_norm).collect();
    let a = power(0.5, &u);
    let b = power(-0.5, &u);
    let kept = &eig[..rank];
    let top = (1.0 - lam) * e_max + floor;
    let bottom = match (lam > 0.0, rank < d) {
        (true, true) => floor,
        _ => (1.0 - lam) * kept.last().copied().unwrap_or(0.0) + floor,
    };
    Ok(LeaceFit {
        b_dot_a: dot(&b, &a),
        a,
        b,
        mean,
        u_whitened: u,
        rank,
        kappa: match bottom > 0.0 {
            true => top / bottom,
            false => f64::INFINITY,
        },
        shrinkage: lam,
    })
}

impl LeaceFit {
    /// `r(x) = x − a bᵀ(x − μ)`.
    pub fn erase(&self, x: &[f64]) -> Vec<f64> {
        let t: f64 = self
            .b
            .iter()
            .zip(x.iter().zip(&self.mean))
            .map(|(b, (x, m))| b * (x - m))
            .sum();
        x.iter().zip(&self.a).map(|(x, a)| x - a * t).collect()
    }
}

/// `‖(1/n) Σ (x_i − x̄)(z_i − z̄)‖` for z = 1 on `pos`, 0 on `neg`.
pub fn cross_cov_norm(pos: &[Vec<f64>], neg: &[Vec<f64>]) -> f64 {
    let d = pos.first().or(neg.first()).map_or(0, Vec::len);
    let n = (pos.len() + neg.len()) as f64;
    let z_bar = pos.len() as f64 / n;
    let x_bar = mean_of(pos.iter().chain(neg), d);
    let mut cov = vec![0.0; d];
    for (rows, z) in [(pos, 1.0), (neg, 0.0)] {
        for x in rows {
            for ((c, xi), m) in cov.iter_mut().zip(x).zip(&x_bar) {
                *c += (xi - m) * (z - z_bar);
            }
        }
    }
    dot(&cov, &cov).sqrt() / n
}

/// Erases both sets with `fit` and measures what is left of the class signal.
pub fn erased_stats(fit: &LeaceFit, pos: &[Vec<f64>], neg: &[Vec<f64>]) -> (f64, f64, f64) {
    let rp: Vec<Vec<f64>> = pos.iter().map(|x| fit.erase(x)).collect();
    let rn: Vec<Vec<f64>> = neg.iter().map(|x| fit.erase(x)).collect();
    let d = fit.mean.len();
    let gap: Vec<f64> = mean_of(rp.iter(), d)
        .iter()
        .zip(mean_of(rn.iter(), d))
        .map(|(p, q)| p - q)
        .collect();
    (
        cross_cov_norm(pos, neg),
        cross_cov_norm(&rp, &rn),
        dot(&gap, &gap).sqrt(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Lcg;

    /// Two Gaussian-like classes with a shared ANISOTROPIC covariance (per-axis scales
    /// 0.2..3 after a fixed rotation-free mixing) and class means apart along `shift`.
    fn classes(n_per: usize, d: usize, seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>) {
        let mut rng = Lcg(seed);
        let scales: Vec<f64> = (0..d)
            .map(|j| 0.2 + 2.8 * j as f64 / d.max(2) as f64)
            .collect();
        let shift: Vec<f64> = (0..d)
            .map(|j| if j % 3 == 0 { 1.0 } else { 0.25 })
            .collect();
        let sample = |rng: &mut Lcg, offset: f64| -> Vec<f64> {
            // Sum of 3 uniforms: close to Gaussian, deterministic.
            let raw: Vec<f64> = (0..d)
                .map(|j| scales[j] * (rng.next() + rng.next() + rng.next()))
                .collect();
            // Mix neighbouring axes so the covariance is not diagonal.
            (0..d)
                .map(|j| raw[j] + 0.5 * raw[(j + 1) % d] + offset * shift[j])
                .collect()
        };
        let pos = (0..n_per).map(|_| sample(&mut rng, 0.5)).collect();
        let neg = (0..n_per).map(|_| sample(&mut rng, -0.5)).collect();
        (pos, neg, shift)
    }

    fn cosine(a: &[f64], b: &[f64]) -> f64 {
        dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt())
    }

    #[test]
    fn many_samples_erase_the_linear_signal_exactly_in_sample() {
        let (pos, neg, _) = classes(1000, 8, 1);
        let fit = fit(&pos, &neg, 0.0, 1e-10).unwrap();
        assert_eq!(fit.rank, 8);
        assert!((fit.b_dot_a - 1.0).abs() < 1e-12);
        let (before, after, gap) = erased_stats(&fit, &pos, &neg);
        assert!(after <= 1e-9 * before, "{after} vs {before}");
        assert!(gap < 1e-9, "{gap}");
        // r is idempotent.
        let x = &pos[3];
        let once = fit.erase(x);
        let twice = fit.erase(&once);
        assert!(once.iter().zip(&twice).all(|(a, b)| (a - b).abs() < 1e-12));
        // a ∥ μ₊ − μ₋.
        let c: Vec<f64> = mean_of(pos.iter(), 8)
            .iter()
            .zip(mean_of(neg.iter(), 8))
            .map(|(p, q)| p - q)
            .collect();
        assert!(cosine(&fit.a, &c) > 1.0 - 1e-12);
        // Least squares of z on [1, r(x)]: every slope ≈ 0 after erasure.
        let rows: Vec<Vec<f64>> = pos.iter().chain(&neg).map(|x| fit.erase(x)).collect();
        let z: Vec<f64> = (0..rows.len())
            .map(|i| f64::from(u8::from(i < pos.len())))
            .collect();
        let slopes = least_squares_slopes(&rows, &z);
        assert!(slopes.iter().all(|s| s.abs() < 1e-8), "{slopes:?}");
    }

    /// Slopes of the OLS fit of z on [1, x] (normal equations, centered).
    fn least_squares_slopes(rows: &[Vec<f64>], z: &[f64]) -> Vec<f64> {
        let d = rows[0].len();
        let n = rows.len() as f64;
        let xm = mean_of(rows.iter(), d);
        let zm = z.iter().sum::<f64>() / n;
        let mut a = vec![0.0; d * d];
        let mut b = vec![0.0; d];
        for (x, zi) in rows.iter().zip(z) {
            for i in 0..d {
                b[i] += (x[i] - xm[i]) * (zi - zm);
                for j in 0..d {
                    a[i * d + j] += (x[i] - xm[i]) * (x[j] - xm[j]);
                }
            }
        }
        // The erased data has rank d − 1 (null direction a): a small ridge keeps κ < 1e12.
        for i in 0..d {
            a[i * d + i] += 1e-3;
        }
        crate::linalg::cholesky::Cholesky::factor(&a, d)
            .unwrap()
            .solve(&b)
    }

    #[test]
    fn few_samples_need_shrinkage_and_overfit_out_of_sample() {
        let (pos, neg, _) = classes(20, 60, 2);
        let fit_shrunk = fit(&pos[..10], &neg[..10], 0.1, 1e-10).unwrap();
        assert!(fit_shrunk.kappa.is_finite() && fit_shrunk.kappa > 1.0);
        assert!((fit_shrunk.b_dot_a - 1.0).abs() < 1e-10);
        let (before, after, _) = erased_stats(&fit_shrunk, &pos[..10], &neg[..10]);
        let in_sample = after / before;
        assert!(in_sample < 1e-10, "{in_sample}");
        let (hb, ha, _) = erased_stats(&fit_shrunk, &pos[10..], &neg[10..]);
        let held_out = ha / hb;
        assert!(
            held_out > in_sample,
            "n < d overfits: {held_out} vs {in_sample}"
        );

        let pinv = fit(&pos[..10], &neg[..10], 0.0, 1e-10).unwrap();
        assert!(pinv.rank <= 19, "rank {}", pinv.rank);
        assert!((pinv.b_dot_a - 1.0).abs() < 1e-8);
        let (b2, a2, _) = erased_stats(&pinv, &pos[..10], &neg[..10]);
        assert!(a2 / b2 < 1e-8);
    }

    #[test]
    fn invalid_and_degenerate_inputs() {
        let (pos, neg, _) = classes(5, 4, 3);
        assert!(matches!(
            fit(&pos, &neg, 1.5, 1e-10),
            Err(ExtractError::InvalidArgument(_))
        ));
        assert!(matches!(
            fit(&pos, &neg, 0.1, 0.0),
            Err(ExtractError::InvalidArgument(_))
        ));
        assert!(matches!(
            fit(&pos, &pos, 0.1, 1e-10),
            Err(ExtractError::DegenerateDirection { .. })
        ));
        let full = fit(&pos, &neg, 1.0, 1e-10).unwrap();
        assert!(
            cosine(&full.a, &full.b) > 1.0 - 1e-12,
            "λ = 1: whitening is the identity"
        );
    }
}
