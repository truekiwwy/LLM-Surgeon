//! Cholesky factorization `A = L Lᵀ` of a symmetric positive definite matrix, solves, and
//! a condition estimate.
//!
//! κ estimate: `(max L_ii / min L_ii)²`. The diagonal of the triangular L holds its
//! eigenvalues, so this ratio is a LOWER bound of `κ₂(L)` and its square a lower bound of
//! `κ₂(A) = κ₂(L)²`: cheap and never a false alarm, but it can miss part of the
//! ill-conditioning. Above [`KAPPA_LIMIT`] (1e12, where f64 solves keep about 4 correct
//! digits) the factorization is refused with `IllConditioned`.

use super::{check_finite, LinalgError, Result};

pub const KAPPA_LIMIT: f64 = 1e12;

#[derive(Debug, Clone, PartialEq)]
pub struct Cholesky {
    pub n: usize,
    /// Row-major lower triangle (upper part zero).
    l: Vec<f64>,
    pub kappa: f64,
}

impl Cholesky {
    /// Factors `a` (row-major N×N, only the lower triangle is read).
    pub fn factor(a: &[f64], n: usize) -> Result<Self> {
        if a.len() != n * n || n == 0 {
            return Err(LinalgError::Shape(format!(
                "{} valeurs pour une matrice {n}×{n}",
                a.len()
            )));
        }
        check_finite(a)?;
        let mut l = vec![0.0; n * n];
        for j in 0..n {
            let mut diag = a[j * n + j];
            for k in 0..j {
                diag -= l[j * n + k] * l[j * n + k];
            }
            if diag.is_nan() || diag <= 0.0 {
                return Err(LinalgError::NotPositiveDefinite {
                    index: j,
                    pivot: diag,
                });
            }
            let ljj = diag.sqrt();
            l[j * n + j] = ljj;
            for i in j + 1..n {
                let mut s = a[i * n + j];
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = s / ljj;
            }
        }
        let diag = (0..n).map(|i| l[i * n + i]);
        let (lo, hi) = diag.fold((f64::INFINITY, 0.0f64), |(lo, hi), x| {
            (lo.min(x), hi.max(x))
        });
        let kappa = (hi / lo).powi(2);
        if kappa.is_nan() || kappa > KAPPA_LIMIT {
            return Err(LinalgError::IllConditioned {
                kappa,
                limit: KAPPA_LIMIT,
            });
        }
        Ok(Self { n, l, kappa })
    }

    /// Solves `A x = b` by forward then backward substitution.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let (n, l) = (self.n, &self.l);
        let mut y = b.to_vec();
        for i in 0..n {
            for k in 0..i {
                y[i] -= l[i * n + k] * y[k];
            }
            y[i] /= l[i * n + i];
        }
        for i in (0..n).rev() {
            for k in i + 1..n {
                y[i] -= l[k * n + i] * y[k];
            }
            y[i] /= l[i * n + i];
        }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Lcg;

    #[test]
    fn solves_a_known_system() {
        // A = [[4, 2, 0], [2, 5, 1], [0, 1, 3]], x = [1, -2, 3] -> b = A x.
        let a = [4.0, 2.0, 0.0, 2.0, 5.0, 1.0, 0.0, 1.0, 3.0];
        let x = [1.0, -2.0, 3.0];
        let b: Vec<f64> = (0..3)
            .map(|i| (0..3).map(|j| a[i * 3 + j] * x[j]).sum())
            .collect();
        let c = Cholesky::factor(&a, 3).unwrap();
        for (got, want) in c.solve(&b).iter().zip(x) {
            assert!((got - want).abs() < 1e-14);
        }
        assert!(c.kappa >= 1.0 && c.kappa < 10.0);

        let n = 12;
        let m = Lcg(3).matrix(n, n);
        let mut spd = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                spd[i * n + j] = (0..n).map(|k| m[i * n + k] * m[j * n + k]).sum::<f64>()
                    + f64::from(u8::from(i == j));
            }
        }
        let x: Vec<f64> = (0..n).map(|i| i as f64 - 5.5).collect();
        let b: Vec<f64> = (0..n)
            .map(|i| (0..n).map(|j| spd[i * n + j] * x[j]).sum())
            .collect();
        let c = Cholesky::factor(&spd, n).unwrap();
        for (got, want) in c.solve(&b).iter().zip(&x) {
            assert!((got - want).abs() < 1e-10);
        }
    }

    #[test]
    fn indefinite_and_ill_conditioned_matrices_are_refused() {
        assert!(matches!(
            Cholesky::factor(&[1.0, 2.0, 2.0, 1.0], 2),
            Err(LinalgError::NotPositiveDefinite { index: 1, .. })
        ));
        assert!(matches!(
            Cholesky::factor(&[0.0], 1),
            Err(LinalgError::NotPositiveDefinite { index: 0, .. })
        ));
        // Hilbert 12×12: κ₂ ≈ 1.7e16.
        let n = 12;
        let hilbert: Vec<f64> = (0..n * n)
            .map(|k| 1.0 / ((k / n + k % n + 1) as f64))
            .collect();
        match Cholesky::factor(&hilbert, n) {
            Err(LinalgError::IllConditioned { kappa, .. }) => assert!(kappa > KAPPA_LIMIT),
            Err(LinalgError::NotPositiveDefinite { .. }) => {}
            other => panic!("{other:?}"),
        }
        let small = 4;
        let h4: Vec<f64> = (0..small * small)
            .map(|k| 1.0 / ((k / small + k % small + 1) as f64))
            .collect();
        let c = Cholesky::factor(&h4, small).unwrap();
        assert!(
            c.kappa > 100.0,
            "the estimate sees part of κ₂ ≈ 1.6e4: {}",
            c.kappa
        );
        assert!(matches!(
            Cholesky::factor(&[1.0, 2.0], 2),
            Err(LinalgError::Shape(_))
        ));
    }
}
