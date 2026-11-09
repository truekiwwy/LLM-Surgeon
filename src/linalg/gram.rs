//! Thin SVD `X = U Σ Vᵀ` of a row-major N×d matrix through its smaller Gram matrix.
//!
//! - N ≤ d (a few dozen prompts, d in the thousands): Jacobi on `G = X Xᵀ` (N×N), then
//!   `v_k = Xᵀ u_k / σ_k`. No d×d matrix is ever formed.
//! - d < N (many samples in a small space): Jacobi on `C = XᵀX` (d×d), the smaller side.
//!   Both sides above 1024 are refused (Jacobi limit).
//!
//! Rank threshold: the Gram matrix squares the singular values, so its eigenvalues carry
//! an absolute error of about `max(N,d)·ε·σ_max²`. A singular value is kept when
//! `σ_k² > max(N,d)·ε·σ_max²`, i.e. `σ_k > sqrt(max(N,d)·ε)·σ_max`. (A threshold of
//! `max(N,d)·ε·σ_max` on σ itself, as for a direct SVD, would keep pure rounding noise
//! here.) The kept right vectors are re-normalized and re-orthogonalized once (modified
//! Gram-Schmidt in decreasing σ order), which removes the loss of orthogonality the Gram
//! route causes on the smaller kept values.
//!
//! Parallelism: every output entry (a Gram entry, a coordinate of v_k) is one sequential
//! sum over a fixed index order; rayon only splits independent outputs, so the result is
//! identical with any number of threads.

use rayon::prelude::*;

use super::jacobi::symmetric_eigen;
use super::{check_finite, dot, norm, LinalgError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct ThinSvd {
    pub n: usize,
    pub d: usize,
    /// Every singular value, decreasing (`min(N, d)` of them, tiny ones included).
    pub sigma: Vec<f64>,
    /// Number of singular values above the rank threshold.
    pub rank: usize,
    /// Right singular vectors of the kept values: `rank` rows of length d.
    pub v: Vec<f64>,
}

impl ThinSvd {
    pub fn vector(&self, k: usize) -> &[f64] {
        &self.v[k * self.d..(k + 1) * self.d]
    }

    /// `Vᵀ y` (length `rank`).
    pub fn project(&self, y: &[f64]) -> Vec<f64> {
        (0..self.rank).map(|k| dot(self.vector(k), y)).collect()
    }

    /// `V c` (length d) for coefficients `c` of length `rank`.
    pub fn combine(&self, c: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.d];
        for (k, ck) in c.iter().enumerate().take(self.rank) {
            for (o, vk) in out.iter_mut().zip(self.vector(k)) {
                *o += ck * vk;
            }
        }
        out
    }
}

/// Relative eigenvalue threshold of the Gram route (see the module documentation).
pub fn gram_rank_tolerance(n: usize, d: usize) -> f64 {
    n.max(d) as f64 * f64::EPSILON
}

pub fn thin_svd(x: &[f64], n: usize, d: usize) -> Result<ThinSvd> {
    if x.len() != n * d || n == 0 || d == 0 {
        return Err(LinalgError::Shape(format!(
            "{} valeurs pour une matrice {n}×{d}",
            x.len()
        )));
    }
    check_finite(x)?;
    let row = |i: usize| &x[i * d..(i + 1) * d];
    let wide = n <= d;
    let m = n.min(d);
    let gram: Vec<f64> = match wide {
        // G_ij = <x_i, x_j>, upper triangle in parallel, mirrored.
        true => (0..n)
            .into_par_iter()
            .flat_map_iter(|i| (0..n).map(move |j| (i, j)))
            .map(|(i, j)| match j >= i {
                true => dot(row(i), row(j)),
                false => f64::NAN,
            })
            .collect(),
        // C_jk = Σ_i x_ij x_ik.
        false => (0..d)
            .into_par_iter()
            .flat_map_iter(|j| (0..d).map(move |k| (j, k)))
            .map(|(j, k)| match k >= j {
                true => (0..n).map(|i| x[i * d + j] * x[i * d + k]).sum(),
                false => f64::NAN,
            })
            .collect(),
    };
    let mut gram = gram;
    for i in 0..m {
        for j in 0..i {
            gram[i * m + j] = gram[j * m + i];
        }
    }
    let eigen = symmetric_eigen(&gram, m)?;
    let lambda_max = eigen.values.first().copied().unwrap_or(0.0).max(0.0);
    let sigma: Vec<f64> = eigen.values.iter().map(|l| l.max(0.0).sqrt()).collect();
    let tolerance = gram_rank_tolerance(n, d) * lambda_max;
    let rank = eigen
        .values
        .iter()
        .take_while(|&&l| lambda_max > 0.0 && l > tolerance)
        .count();

    let mut v: Vec<f64> = match wide {
        true => (0..rank)
            .into_par_iter()
            .flat_map_iter(|k| {
                let u = eigen.vector(k);
                let mut out = vec![0.0; d];
                for (i, ui) in u.iter().enumerate() {
                    for (o, xij) in out.iter_mut().zip(row(i)) {
                        *o += ui * xij;
                    }
                }
                let s = sigma[k];
                out.into_iter().map(move |o| o / s)
            })
            .collect(),
        false => (0..rank).flat_map(|k| eigen.vector(k)).collect(),
    };
    for k in 0..rank {
        let (done, rest) = v.split_at_mut(k * d);
        let vk = &mut rest[..d];
        for j in 0..k {
            let vj = &done[j * d..(j + 1) * d];
            let c = dot(vk, vj);
            for (a, b) in vk.iter_mut().zip(vj) {
                *a -= c * b;
            }
        }
        let len = norm(vk);
        if len > 0.0 {
            vk.iter_mut().for_each(|a| *a /= len);
        }
    }
    Ok(ThinSvd {
        n,
        d,
        sigma,
        rank,
        v,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::jacobi::symmetric_eigen;
    use crate::linalg::power::top_right_singular;
    use crate::linalg::Lcg;

    /// X = Σ_k s_k a_k b_kᵀ with orthonormal a (N), b (d): known singular values.
    fn planted(n: usize, d: usize, values: &[f64], seed: u64) -> Vec<f64> {
        let mut rng = Lcg(seed);
        let orthonormal = |rng: &mut Lcg, len: usize| {
            let mut basis: Vec<Vec<f64>> = Vec::new();
            for _ in 0..values.len() {
                let mut v: Vec<f64> = (0..len).map(|_| rng.next()).collect();
                for b in &basis {
                    let c = dot(&v, b);
                    v.iter_mut().zip(b).for_each(|(x, y)| *x -= c * y);
                }
                let l = norm(&v);
                v.iter_mut().for_each(|x| *x /= l);
                basis.push(v);
            }
            basis
        };
        let (a, b) = (orthonormal(&mut rng, n), orthonormal(&mut rng, d));
        let mut x = vec![0.0; n * d];
        for (k, s) in values.iter().enumerate() {
            for i in 0..n {
                for j in 0..d {
                    x[i * d + j] += s * a[k][i] * b[k][j];
                }
            }
        }
        x
    }

    /// ‖P_a − P_b‖ entrywise max for the projectors on two sets of orthonormal rows.
    fn subspace_gap(a: &[f64], b: &[f64], d: usize, k: usize) -> f64 {
        let proj = |m: &[f64], i: usize, j: usize| -> f64 {
            (0..k).map(|r| m[r * d + i] * m[r * d + j]).sum()
        };
        let mut worst: f64 = 0.0;
        for i in 0..d {
            for j in 0..d {
                worst = worst.max((proj(a, i, j) - proj(b, i, j)).abs());
            }
        }
        worst
    }

    #[test]
    fn wide_and_tall_recover_planted_singular_values() {
        let values = [5.0, 3.0, 1.0];
        for (n, d) in [(6, 40), (40, 6)] {
            let x = planted(n, d, &values, 11);
            let s = thin_svd(&x, n, d).unwrap();
            assert_eq!(s.rank, 3, "n={n} d={d}");
            for (got, want) in s.sigma.iter().zip(values) {
                assert!((got - want).abs() < 1e-10, "{got} vs {want}");
            }
            // X v_k = σ_k u_k: ‖X v_k‖ = σ_k.
            for (k, want) in values.iter().enumerate() {
                let xv: Vec<f64> = (0..n)
                    .map(|i| dot(&x[i * d..(i + 1) * d], s.vector(k)))
                    .collect();
                assert!((norm(&xv) - want).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn matches_a_direct_eigen_decomposition_and_power_iteration() {
        let (n, d) = (8, 30);
        let x = planted(n, d, &[9.0, 4.0, 2.0, 0.5], 5);
        let s = thin_svd(&x, n, d).unwrap();
        let mut c = vec![0.0; d * d];
        for j in 0..d {
            for k in 0..d {
                c[j * d + k] = (0..n).map(|i| x[i * d + j] * x[i * d + k]).sum();
            }
        }
        let direct = symmetric_eigen(&c, d).unwrap();
        let direct_v: Vec<f64> = (0..4).flat_map(|k| direct.vector(k)).collect();
        assert!(subspace_gap(&s.v, &direct_v, d, 4) < 1e-8);
        let (power_sigma, power_v) = top_right_singular(&x, n, d, 4, 1e-14, 100_000).unwrap();
        assert!(subspace_gap(&s.v, &power_v, d, 4) < 1e-8);
        for k in 0..4 {
            assert!((power_sigma[k] - s.sigma[k]).abs() < 1e-8);
            let one = |m: &[f64]| m[k * d..(k + 1) * d].to_vec();
            assert!(
                subspace_gap(&one(&s.v), &one(&power_v), d, 1) < 1e-8,
                "vector {k}"
            );
        }
    }

    #[test]
    fn rank_deficiency_orthogonality_and_thread_invariance() {
        let (n, d) = (10, 50);
        let x = planted(n, d, &[3.0, 2.0], 21);
        let s = thin_svd(&x, n, d).unwrap();
        assert_eq!(s.rank, 2);
        assert!(s.sigma[2..].iter().all(|&v| v < 1e-6));
        let g = dot(s.vector(0), s.vector(1));
        assert!(g.abs() < 1e-14 && (norm(s.vector(1)) - 1.0).abs() < 1e-14);

        let y: Vec<f64> = Lcg(4).matrix(n, d);
        let runs: Vec<ThinSvd> = [1, 2, 8]
            .iter()
            .map(|&t| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(t)
                    .build()
                    .unwrap()
                    .install(|| thin_svd(&y, n, d).unwrap())
            })
            .collect();
        assert!(runs.windows(2).all(|w| w[0] == w[1]), "bit-identical");
        assert!(thin_svd(&[0.0; 12], 3, 4).unwrap().rank == 0);
        assert!(thin_svd(&[1.0], 1, 1).unwrap().sigma == [1.0]);
        assert!(matches!(
            thin_svd(&[1.0; 5], 2, 3),
            Err(LinalgError::Shape(_))
        ));
    }

    #[test]
    fn large_d_never_forms_a_d_by_d_matrix() {
        let (n, d) = (64, 8192);
        let x = Lcg(77).matrix(n, d);
        let started = std::time::Instant::now();
        let s = thin_svd(&x, n, d).unwrap();
        assert_eq!(s.rank, n);
        // A d×d Gram would be 512 MiB and 8192 Jacobi rows: far beyond this budget.
        assert!(started.elapsed() < std::time::Duration::from_secs(60));
    }
}
