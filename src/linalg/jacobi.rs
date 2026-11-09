//! Eigen-decomposition of a symmetric matrix by cyclic Jacobi.
//!
//! Each sweep visits every pair `p < q` in the same row-major order and zeroes `a_pq` with
//! a rotation computed the stable way (Rutishauser: `t = sign(θ)/(|θ| + sqrt(θ²+1))`, the
//! smaller root, and `τ = s/(1+c)` updates). Stops when `‖offdiag‖_F ≤ ε·‖A‖_F`
//! (ε = f64::EPSILON), after at most 60 sweeps; entries negligible next to both diagonal
//! terms are set to zero after the fourth sweep, as in Numerical Recipes, so that the
//! relative criterion is reached instead of stalling on rounding noise. Eigenvalues come
//! out sorted in decreasing order (ties keep their index order). Two buffers of N×N are
//! allocated once; the sweeps allocate nothing.

use super::{check_finite, LinalgError, Result};

pub const MAX_N: usize = 1024;
pub const MAX_SWEEPS: usize = 60;

#[derive(Debug, Clone, PartialEq)]
pub struct Eigen {
    pub n: usize,
    /// Decreasing.
    pub values: Vec<f64>,
    /// Row-major N×N; column k is the unit eigenvector of `values[k]`.
    pub vectors: Vec<f64>,
    pub sweeps: usize,
}

impl Eigen {
    /// Column `k` of `vectors`.
    pub fn vector(&self, k: usize) -> Vec<f64> {
        (0..self.n).map(|r| self.vectors[r * self.n + k]).collect()
    }
}

fn offdiag_norm(a: &[f64], n: usize) -> f64 {
    let mut sum = 0.0;
    for p in 0..n {
        for q in 0..n {
            if p != q {
                sum += a[p * n + q] * a[p * n + q];
            }
        }
    }
    sum.sqrt()
}

/// `a` is row-major N×N; it is symmetrized (`(A + Aᵀ)/2`) so a slightly asymmetric input
/// from rounding is accepted.
pub fn symmetric_eigen(a: &[f64], n: usize) -> Result<Eigen> {
    if a.len() != n * n {
        return Err(LinalgError::Shape(format!(
            "{} valeurs pour une matrice {n}×{n}",
            a.len()
        )));
    }
    if n > MAX_N {
        return Err(LinalgError::TooLarge { n, max: MAX_N });
    }
    check_finite(a)?;
    let mut m = vec![0.0; n * n];
    for p in 0..n {
        for q in 0..n {
            m[p * n + q] = 0.5 * (a[p * n + q] + a[q * n + p]);
        }
    }
    let mut v = vec![0.0; n * n];
    for p in 0..n {
        v[p * n + p] = 1.0;
    }
    let fro = m.iter().map(|x| x * x).sum::<f64>().sqrt();
    let tolerance = f64::EPSILON * fro;

    let mut sweeps = 0;
    loop {
        let off = offdiag_norm(&m, n);
        if off <= tolerance {
            break;
        }
        if sweeps == MAX_SWEEPS {
            return Err(LinalgError::NoConvergence {
                sweeps,
                residual: off / fro,
            });
        }
        sweeps += 1;
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let (app, aqq) = (m[p * n + p], m[q * n + q]);
                let g = 100.0 * apq.abs();
                if sweeps > 4 && app.abs() + g == app.abs() && aqq.abs() + g == aqq.abs() {
                    m[p * n + q] = 0.0;
                    m[q * n + p] = 0.0;
                    continue;
                }
                let theta = (aqq - app) / (2.0 * apq);
                // θ² would overflow beyond ~1e154; 1/(2θ) is then the exact limit of t.
                let t = match theta.abs() > 1e150 {
                    true => 1.0 / (2.0 * theta),
                    false => theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt()),
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);
                m[p * n + p] = app - t * apq;
                m[q * n + q] = aqq + t * apq;
                m[p * n + q] = 0.0;
                m[q * n + p] = 0.0;
                for r in 0..n {
                    if r != p && r != q {
                        let (arp, arq) = (m[r * n + p], m[r * n + q]);
                        let new_rp = arp - s * (arq + tau * arp);
                        let new_rq = arq + s * (arp - tau * arq);
                        m[r * n + p] = new_rp;
                        m[p * n + r] = new_rp;
                        m[r * n + q] = new_rq;
                        m[q * n + r] = new_rq;
                    }
                    let (vrp, vrq) = (v[r * n + p], v[r * n + q]);
                    v[r * n + p] = vrp - s * (vrq + tau * vrp);
                    v[r * n + q] = vrq + s * (vrp - tau * vrq);
                }
            }
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    // Stable sort: equal eigenvalues keep their index order.
    order.sort_by(|&i, &j| m[j * n + j].total_cmp(&m[i * n + i]));
    let values = order.iter().map(|&i| m[i * n + i]).collect();
    let mut vectors = vec![0.0; n * n];
    for (k, &i) in order.iter().enumerate() {
        for r in 0..n {
            vectors[r * n + k] = v[r * n + i];
        }
    }
    Ok(Eigen {
        n,
        values,
        vectors,
        sweeps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Lcg;

    fn reconstruction_error(a: &[f64], e: &Eigen) -> f64 {
        let n = e.n;
        let mut worst: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let r: f64 = (0..n)
                    .map(|k| e.vectors[i * n + k] * e.values[k] * e.vectors[j * n + k])
                    .sum();
                worst = worst.max((r - 0.5 * (a[i * n + j] + a[j * n + i])).abs());
            }
        }
        worst
    }

    fn orthogonality_error(e: &Eigen) -> f64 {
        let n = e.n;
        let mut worst: f64 = 0.0;
        for k in 0..n {
            for l in 0..n {
                let d: f64 = (0..n)
                    .map(|r| e.vectors[r * n + k] * e.vectors[r * n + l])
                    .sum();
                worst = worst.max((d - f64::from(u8::from(k == l))).abs());
            }
        }
        worst
    }

    fn symmetric(n: usize, seed: u64) -> Vec<f64> {
        let r = Lcg(seed).matrix(n, n);
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                a[i * n + j] = r[i * n + j] + r[j * n + i];
            }
        }
        a
    }

    #[test]
    fn diagonal_and_analytic_two_by_two() {
        let e = symmetric_eigen(&[3.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 7.0], 3).unwrap();
        assert_eq!(e.values, [7.0, 3.0, -1.0]);
        assert_eq!(e.sweeps, 0);
        // [[2, 1], [1, 2]]: eigenvalues 3 and 1, eigenvectors (1, 1)/√2 and (1, -1)/√2.
        let e = symmetric_eigen(&[2.0, 1.0, 1.0, 2.0], 2).unwrap();
        assert!((e.values[0] - 3.0).abs() < 1e-15 && (e.values[1] - 1.0).abs() < 1e-15);
        let v0 = e.vector(0);
        assert!((v0[0].abs() - 0.5f64.sqrt()).abs() < 1e-15 && v0[0] * v0[1] > 0.0);
    }

    #[test]
    fn random_symmetric_matrices_reconstruct_sorted_and_orthonormal() {
        for (n, seed) in [(5, 1), (17, 2), (64, 3)] {
            let a = symmetric(n, seed);
            let e = symmetric_eigen(&a, n).unwrap();
            assert!(reconstruction_error(&a, &e) < 1e-10, "n={n}");
            assert!(orthogonality_error(&e) < 1e-12, "n={n}");
            assert!(e.values.windows(2).all(|w| w[0] >= w[1]));
            assert!(e.sweeps <= MAX_SWEEPS);
        }
    }

    #[test]
    fn repeated_zero_rank_deficient_and_tiny_cases() {
        // Eigenvalue 2 repeated three times (2·I plus a rank-1 term).
        let n = 4;
        let u = [0.5, 0.5, 0.5, 0.5];
        let a: Vec<f64> = (0..n * n)
            .map(|k| f64::from(u8::from(k / n == k % n)) * 2.0 + 4.0 * u[k / n] * u[k % n])
            .collect();
        let e = symmetric_eigen(&a, n).unwrap();
        assert!((e.values[0] - 6.0).abs() < 1e-14);
        assert!(e.values[1..].iter().all(|v| (v - 2.0).abs() < 1e-14));
        assert!(reconstruction_error(&a, &e) < 1e-13 && orthogonality_error(&e) < 1e-13);

        let zero = symmetric_eigen(&[0.0; 9], 3).unwrap();
        assert_eq!(zero.values, [0.0; 3]);
        assert_eq!(orthogonality_error(&zero), 0.0);

        let one = symmetric_eigen(&[-2.5], 1).unwrap();
        assert_eq!((one.values[0], one.vectors[0]), (-2.5, 1.0));

        // Rank 2 in dimension 6: four eigenvalues at zero.
        let x = Lcg(9).matrix(2, 6);
        let mut a = vec![0.0; 36];
        for i in 0..6 {
            for j in 0..6 {
                a[i * 6 + j] = x[i] * x[j] + x[6 + i] * x[6 + j];
            }
        }
        let e = symmetric_eigen(&a, 6).unwrap();
        assert!(
            e.values[2..].iter().all(|v| v.abs() < 1e-14),
            "{:?}",
            e.values
        );
    }

    #[test]
    fn invalid_inputs_are_typed_errors() {
        assert!(matches!(
            symmetric_eigen(&[1.0, 2.0, 3.0], 2),
            Err(LinalgError::Shape(_))
        ));
        assert!(matches!(
            symmetric_eigen(&[f64::NAN, 0.0, 0.0, 1.0], 2),
            Err(LinalgError::NonFinite)
        ));
        let n = MAX_N + 1;
        assert!(matches!(
            symmetric_eigen(&vec![0.0; n * n], n),
            Err(LinalgError::TooLarge { .. })
        ));
    }
}
