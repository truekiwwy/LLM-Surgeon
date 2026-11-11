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
//! Several labels (`z ∈ R^m`, [`fit_multi`]): the same eraser with a subspace instead of
//! a direction.
//!
//! ```text
//! Σ_XZ = X̃ᵀZ̃ / n (d × m) ; M = Σ_λ^{-1/2} Σ_XZ = U S Vᵀ (thin) ; r = #{s_i > rcond · s_1}
//! A = Σ_λ^{1/2} U_r ; B = Σ_λ^{-1/2} U_r (BᵀA = I_r) ; r(x) = x − A Bᵀ (x − μ)
//! ```
//!
//! `A Bᵀ Σ_XZ = Σ_λ^{1/2} U_r U_rᵀ M = Σ_XZ` when no singular value is cut, so the cross
//! covariance with every label vanishes in sample. `U` comes from the m × m Gram matrix of
//! the columns of `M`: still no d×d matrix.
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
use crate::linalg::gram::{thin_svd, ThinSvd};

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
    /// Effaceur de plusieurs labels (--labels-file) : diagnostics par composante ; absent
    /// pour le concept binaire.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subspace: Option<super::subspace::SubspaceDiagnostics>,
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
    let whitener = Whitener::new(&centered, n, d, shrinkage, rcond)?;
    let w = whitener.power(-0.5, &c);
    let w_norm = dot(&w, &w).sqrt();
    if !(w_norm > 0.0 && w_norm.is_finite()) {
        return Err(ExtractError::DegenerateDirection { norm: w_norm });
    }
    let u: Vec<f64> = w.iter().map(|x| x / w_norm).collect();
    let a = whitener.power(0.5, &u);
    let b = whitener.power(-0.5, &u);
    Ok(LeaceFit {
        b_dot_a: dot(&b, &a),
        a,
        b,
        mean,
        u_whitened: u,
        rank: whitener.rank,
        kappa: whitener.kappa(),
        shrinkage,
    })
}

/// `Σ_λ^p y` without any d×d matrix (see the module documentation).
struct Whitener {
    svd: ThinSvd,
    eig: Vec<f64>,
    /// Kept eigen-directions of Σ.
    rank: usize,
    lam: f64,
    floor: f64,
    d: usize,
}

impl Whitener {
    /// `centered`: the rows of `X̃ / √n`, n × d.
    fn new(centered: &[f64], n: usize, d: usize, shrinkage: f64, rcond: f64) -> Result<Self> {
        let svd = thin_svd(centered, n, d).map_err(|e| ExtractError::Compute(e.to_string()))?;
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
        Ok(Self {
            svd,
            eig,
            rank,
            lam,
            floor: lam * s,
            d,
        })
    }

    fn power(&self, p: f64, y: &[f64]) -> Vec<f64> {
        let (lam, floor, rank) = (self.lam, self.floor, self.rank);
        let coeffs = self.svd.project(y);
        let scaled: Vec<f64> = coeffs
            .iter()
            .zip(&self.eig)
            .take(rank)
            .map(|(cf, e)| cf * ((1.0 - lam) * e + floor).powf(p))
            .collect();
        let mut out = self.svd.combine(&scaled);
        if lam > 0.0 {
            // Complement of the kept subspace: eigenvalue λ s.
            let inside = self.svd.combine(&coeffs[..rank]);
            let f = floor.powf(p);
            out.iter_mut()
                .zip(y.iter().zip(&inside))
                .for_each(|(o, (yi, vi))| *o += f * (yi - vi));
        }
        out
    }

    /// λ_max(Σ_λ) / λ_min(Σ_λ) (on the kept subspace when λ = 0).
    fn kappa(&self) -> f64 {
        let e_max = self.eig.first().copied().unwrap_or(0.0);
        let kept = &self.eig[..self.rank];
        let top = (1.0 - self.lam) * e_max + self.floor;
        let bottom = match (self.lam > 0.0, self.rank < self.d) {
            (true, true) => self.floor,
            _ => (1.0 - self.lam) * kept.last().copied().unwrap_or(0.0) + self.floor,
        };
        match bottom > 0.0 {
            true => top / bottom,
            false => f64::INFINITY,
        }
    }
}

/// The LEACE eraser of several labels: `r` write and read directions, `BᵀA = I_r`.
#[derive(Debug, Clone, PartialEq)]
pub struct LeaceSubspace {
    /// Write directions `Σ_λ^{1/2} u_t`, one per row.
    pub a: Vec<Vec<f64>>,
    /// Read directions `Σ_λ^{-1/2} u_t`, one per row.
    pub b: Vec<Vec<f64>>,
    pub mean: Vec<f64>,
    /// Left singular vectors `u_t` of `Σ_λ^{-1/2} Σ_XZ`, orthonormal.
    pub u_whitened: Vec<Vec<f64>>,
    /// Every singular value of `Σ_λ^{-1/2} Σ_XZ`, decreasing.
    pub sigma: Vec<f64>,
    /// Kept eigen-directions of Σ.
    pub cov_rank: usize,
    pub kappa: f64,
    pub shrinkage: f64,
}

/// `(1/n) Σ_i (x_i − x̄)(z_il − z̄_l)`: one vector of size d per label.
pub fn cross_cov(x: &[&[f64]], z: &[&[f64]]) -> Vec<Vec<f64>> {
    let (n, d) = (x.len(), x.first().map_or(0, |r| r.len()));
    let m = z.first().map_or(0, |r| r.len());
    let mean = |rows: &[&[f64]], size: usize| {
        let mut sum = vec![0.0; size];
        for row in rows {
            sum.iter_mut().zip(*row).for_each(|(s, v)| *s += v);
        }
        sum.iter_mut().for_each(|s| *s /= n.max(1) as f64);
        sum
    };
    let (x_bar, z_bar) = (mean(x, d), mean(z, m));
    let mut cov = vec![vec![0.0; d]; m];
    for (row, labels) in x.iter().zip(z) {
        for ((c, zl), zm) in cov.iter_mut().zip(*labels).zip(&z_bar) {
            let weight = zl - zm;
            for ((cj, xj), mj) in c.iter_mut().zip(*row).zip(&x_bar) {
                *cj += (xj - mj) * weight;
            }
        }
    }
    for c in &mut cov {
        c.iter_mut().for_each(|v| *v /= n.max(1) as f64);
    }
    cov
}

/// Frobenius norm of a cross covariance.
pub fn frobenius(cov: &[Vec<f64>]) -> f64 {
    cov.iter().map(|c| dot(c, c)).sum::<f64>().sqrt()
}

/// The largest component of `v` in absolute value is made positive (the first one on a
/// tie): a sign that does not depend on rounding noise.
fn fix_sign(v: &mut [f64]) {
    let mut largest = 0.0f64;
    for x in v.iter() {
        if x.abs() > largest.abs() {
            largest = *x;
        }
    }
    if largest < 0.0 {
        v.iter_mut().for_each(|x| *x = -*x);
    }
}

/// LEACE for labels `z ∈ R^m` (one row of `z` per row of `x`): at most `max_rank`
/// directions, fewer when `Σ_λ^{-1/2} Σ_XZ` has a lower rank.
pub fn fit_multi(
    x: &[&[f64]],
    z: &[&[f64]],
    shrinkage: f64,
    rcond: f64,
    max_rank: Option<usize>,
) -> Result<LeaceSubspace> {
    let (n, d) = (x.len(), x.first().map_or(0, |r| r.len()));
    let m = z.first().map_or(0, |r| r.len());
    if n < 2 || d == 0 || m == 0 || z.len() != n || z.iter().any(|r| r.len() != m) {
        return Err(ExtractError::Compute(format!(
            "labels incohérents : {n} activations, {} lignes de labels de taille {m}",
            z.len()
        )));
    }
    if !(0.0..=1.0).contains(&shrinkage) || !(rcond > 0.0 && rcond < 1.0) {
        return Err(ExtractError::InvalidArgument(
            "--leace-shrinkage doit être dans [0, 1] et --leace-rcond dans ]0, 1[".into(),
        ));
    }
    let mut mean = vec![0.0; d];
    for row in x {
        mean.iter_mut().zip(*row).for_each(|(s, v)| *s += v);
    }
    mean.iter_mut().for_each(|s| *s /= n as f64);
    let scale = 1.0 / (n as f64).sqrt();
    let centered: Vec<f64> = x
        .iter()
        .flat_map(|row| row.iter().zip(&mean).map(move |(v, mu)| (v - mu) * scale))
        .collect();
    let whitener = Whitener::new(&centered, n, d, shrinkage, rcond)?;
    // The columns of M = Σ_λ^{-1/2} Σ_XZ, as the m rows of Mᵀ: the right singular vectors
    // of Mᵀ are the left singular vectors of M.
    let columns: Vec<f64> = cross_cov(x, z)
        .iter()
        .flat_map(|c| whitener.power(-0.5, c))
        .collect();
    let svd = thin_svd(&columns, m, d).map_err(|e| ExtractError::Compute(e.to_string()))?;
    let s1 = svd.sigma.first().copied().unwrap_or(0.0);
    let numerical = svd.sigma[..svd.rank]
        .iter()
        .take_while(|s| **s > rcond * s1)
        .count();
    let rank = max_rank.map_or(numerical, |k| numerical.min(k));
    if rank == 0 || !s1.is_finite() {
        return Err(ExtractError::DegenerateDirection { norm: s1 });
    }
    let u_whitened: Vec<Vec<f64>> = (0..rank)
        .map(|t| {
            let mut u = svd.vector(t).to_vec();
            fix_sign(&mut u);
            u
        })
        .collect();
    Ok(LeaceSubspace {
        a: u_whitened.iter().map(|u| whitener.power(0.5, u)).collect(),
        b: u_whitened.iter().map(|u| whitener.power(-0.5, u)).collect(),
        mean,
        u_whitened,
        sigma: svd.sigma,
        cov_rank: whitener.rank,
        kappa: whitener.kappa(),
        shrinkage,
    })
}

impl LeaceSubspace {
    /// `r(x) = x − A Bᵀ (x − μ)`.
    pub fn erase(&self, x: &[f64]) -> Vec<f64> {
        let mut out = x.to_vec();
        for (a, b) in self.a.iter().zip(&self.b) {
            let read: f64 = b
                .iter()
                .zip(x.iter().zip(&self.mean))
                .map(|(b, (x, m))| b * (x - m))
                .sum();
            out.iter_mut().zip(a).for_each(|(o, a)| *o -= a * read);
        }
        out
    }
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

    // ------------------------------------------------------------------ several labels

    fn refs(rows: &[Vec<f64>]) -> Vec<&[f64]> {
        rows.iter().map(Vec::as_slice).collect()
    }

    /// `n` continuous label rows of size `m`.
    fn random_labels(n: usize, m: usize, seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| (0..m).map(|_| rng.next()).collect())
            .collect()
    }

    /// One row per label row: anisotropic noise plus `z_l` times a fixed pattern per label.
    fn labelled(d: usize, z: &[Vec<f64>], seed: u64) -> Vec<Vec<f64>> {
        let mut rng = Lcg(seed);
        z.iter()
            .map(|labels| {
                (0..d)
                    .map(|j| {
                        let noise = (0.3 + j as f64 / d as f64) * (rng.next() + rng.next());
                        let signal: f64 = labels
                            .iter()
                            .enumerate()
                            .map(|(l, z)| z * (((j * (l + 2)) % 7) as f64 - 3.0) / 3.0)
                            .sum();
                        noise + signal
                    })
                    .collect()
            })
            .collect()
    }

    fn gram_error(fit: &LeaceSubspace) -> f64 {
        let mut worst = 0.0f64;
        for (s, b) in fit.b.iter().enumerate() {
            for (t, a) in fit.a.iter().enumerate() {
                worst = worst.max((dot(b, a) - f64::from(u8::from(s == t))).abs());
            }
        }
        worst
    }

    fn erased_cross_cov(fit: &LeaceSubspace, x: &[Vec<f64>], z: &[Vec<f64>]) -> f64 {
        let erased: Vec<Vec<f64>> = x.iter().map(|row| fit.erase(row)).collect();
        frobenius(&cross_cov(&refs(&erased), &refs(z)))
    }

    #[test]
    fn several_labels_are_erased_at_once_by_a_projector() {
        let (n, d, m) = (80, 12, 3);
        let z = random_labels(n, m, 5);
        let x = labelled(d, &z, 9);
        let before = frobenius(&cross_cov(&refs(&x), &refs(&z)));
        assert!(before > 0.1, "the labels are in the activations: {before}");
        for shrinkage in [0.1, 0.0] {
            let fit = fit_multi(&refs(&x), &refs(&z), shrinkage, 1e-10, None).unwrap();
            assert_eq!((fit.a.len(), fit.b.len(), fit.sigma.len()), (m, m, m));
            assert!(fit.sigma.windows(2).all(|s| s[0] >= s[1]));
            assert!(gram_error(&fit) < 1e-9, "BᵀA = I: {}", gram_error(&fit));
            let after = erased_cross_cov(&fit, &x, &z);
            assert!(
                after < 1e-9 * before,
                "λ = {shrinkage}: {after:e} of {before:e}"
            );
            // Same inputs, same bits.
            let again = fit_multi(&refs(&x), &refs(&z), shrinkage, 1e-10, None).unwrap();
            assert_eq!(fit, again);
        }
    }

    #[test]
    fn one_hot_classes_lose_one_rank_and_a_capped_rank_erases_less() {
        let (n, d, classes) = (80, 12, 4);
        let z: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..classes)
                    .map(|c| f64::from(u8::from(i % classes == c)))
                    .collect()
            })
            .collect();
        let x = labelled(d, &z, 3);
        let before = frobenius(&cross_cov(&refs(&x), &refs(&z)));
        // Centered one-hot labels sum to zero: m classes, m − 1 directions.
        let full = fit_multi(&refs(&x), &refs(&z), 0.1, 1e-10, None).unwrap();
        assert_eq!(full.a.len(), classes - 1);
        assert!(erased_cross_cov(&full, &x, &z) < 1e-9 * before);
        assert!(gram_error(&full) < 1e-9);

        let capped = fit_multi(&refs(&x), &refs(&z), 0.1, 1e-10, Some(1)).unwrap();
        assert_eq!(capped.a.len(), 1);
        assert_eq!(capped.u_whitened[0], full.u_whitened[0], "the leading one");
        let left = erased_cross_cov(&capped, &x, &z);
        assert!(
            left < before && left > 1e-3 * before,
            "{left:e} of {before:e}"
        );
        // Asking for more than there is gives what there is.
        let more = fit_multi(&refs(&x), &refs(&z), 0.1, 1e-10, Some(9)).unwrap();
        assert_eq!(more.a.len(), classes - 1);
    }

    #[test]
    fn one_binary_label_gives_the_eraser_of_the_binary_fit() {
        let (pos, neg, _) = classes(30, 10, 21);
        let binary = fit(&pos, &neg, 0.1, 1e-10).unwrap();
        let x: Vec<Vec<f64>> = pos.iter().chain(&neg).cloned().collect();
        let z: Vec<Vec<f64>> = (0..x.len())
            .map(|i| vec![f64::from(u8::from(i < pos.len()))])
            .collect();
        let multi = fit_multi(&refs(&x), &refs(&z), 0.1, 1e-10, None).unwrap();
        assert_eq!(multi.a.len(), 1);
        assert_eq!(multi.mean, binary.mean);
        // The same operator a bᵀ (the sign of u is free, the product is not).
        for i in 0..10 {
            for j in 0..10 {
                let (m, b) = (multi.a[0][i] * multi.b[0][j], binary.a[i] * binary.b[j]);
                assert!((m - b).abs() < 1e-9, "a bᵀ[{i}][{j}]: {m} vs {b}");
            }
        }
    }

    #[test]
    fn a_wide_model_with_few_prompts_is_fitted_quickly_without_any_d_by_d_matrix() {
        let (n, d, m) = (64, 8192, 3);
        let z = random_labels(n, m, 2);
        let x = labelled(d, &z, 4);
        let started = std::time::Instant::now();
        let fit = fit_multi(&refs(&x), &refs(&z), 0.1, 1e-10, None).unwrap();
        let elapsed = started.elapsed();
        assert!(elapsed.as_secs_f64() < 5.0, "{elapsed:?}");
        assert_eq!(fit.a.len(), m);
        assert!(gram_error(&fit) < 1e-8, "{}", gram_error(&fit));
        let before = frobenius(&cross_cov(&refs(&x), &refs(&z)));
        assert!(erased_cross_cov(&fit, &x, &z) < 1e-8 * before);
    }

    #[test]
    fn inconsistent_labels_are_refused() {
        let x = vec![vec![1.0, 2.0], vec![0.0, 1.0], vec![2.0, 0.0]];
        let short = vec![vec![1.0], vec![0.0]];
        assert!(fit_multi(&refs(&x), &refs(&short), 0.1, 1e-10, None).is_err());
        let ragged = vec![vec![1.0], vec![0.0, 1.0], vec![0.0]];
        assert!(fit_multi(&refs(&x), &refs(&ragged), 0.1, 1e-10, None).is_err());
        // A constant label carries nothing to erase.
        let constant = vec![vec![1.0], vec![1.0], vec![1.0]];
        assert!(matches!(
            fit_multi(&refs(&x), &refs(&constant), 0.1, 1e-10, None),
            Err(ExtractError::DegenerateDirection { .. })
        ));
    }
}
