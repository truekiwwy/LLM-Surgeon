//! Orthonormalization helpers of the randomized SVD: modified Gram-Schmidt applied twice
//! ("twice is enough", Giraud et al.), f64, sequential and therefore bit-reproducible.
//!
//! A column that collapses during the projection (its norm falls below [`DEPENDENT_REL_TOL`]
//! of its original norm, or is zero) is dependent on the previous ones, e.g. `A * Omega` of an
//! exactly low-rank `A`. Normalizing the rounding noise would give an arbitrary direction, so
//! it is replaced by the canonical vector that has the largest component orthogonal to the
//! previous columns: the basis stays orthonormal and the choice is deterministic.

use super::{LinalgError, Result};

/// Relative norm below which a column is considered dependent on the previous ones.
pub const DEPENDENT_REL_TOL: f64 = 1e-10;

/// Orthonormalizes the columns of the row-major `rows x cols` matrix `a` in place (needs
/// `cols <= rows`). Returns how many columns were dependent and had to be replaced.
pub fn orthonormalize_columns(a: &mut [f64], rows: usize, cols: usize) -> Result<usize> {
    if a.len() != rows * cols || cols > rows {
        return Err(LinalgError::Shape(format!(
            "{} valeurs pour {cols} colonnes orthonormales de longueur {rows}",
            a.len()
        )));
    }
    let mut replaced = 0;
    for j in 0..cols {
        let before = column_norm(a, rows, cols, j);
        if !before.is_finite() {
            return Err(LinalgError::NonFinite);
        }
        project_out_twice(a, rows, cols, j);
        let after = column_norm(a, rows, cols, j);
        if after > DEPENDENT_REL_TOL * before && after > 0.0 {
            scale_column(a, rows, cols, j, 1.0 / after);
            continue;
        }
        replaced += 1;
        let pivot = best_canonical_row(a, rows, cols, j);
        for r in 0..rows {
            a[r * cols + j] = f64::from(u8::from(r == pivot));
        }
        project_out_twice(a, rows, cols, j);
        let norm = column_norm(a, rows, cols, j);
        scale_column(a, rows, cols, j, 1.0 / norm);
    }
    Ok(replaced)
}

fn column_norm(a: &[f64], rows: usize, cols: usize, j: usize) -> f64 {
    (0..rows)
        .map(|r| a[r * cols + j] * a[r * cols + j])
        .sum::<f64>()
        .sqrt()
}

fn scale_column(a: &mut [f64], rows: usize, cols: usize, j: usize, factor: f64) {
    for r in 0..rows {
        a[r * cols + j] *= factor;
    }
}

/// Subtracts from column `j` its components on the columns `0..j`, twice.
fn project_out_twice(a: &mut [f64], rows: usize, cols: usize, j: usize) {
    for _ in 0..2 {
        for i in 0..j {
            let d: f64 = (0..rows).map(|r| a[r * cols + i] * a[r * cols + j]).sum();
            for r in 0..rows {
                a[r * cols + j] -= d * a[r * cols + i];
            }
        }
    }
}

/// Row `c` maximizing `1 - sum_{i<j} a[c,i]^2`, the squared residual of `e_c` after the
/// projection on the first `j` columns (first one on ties).
fn best_canonical_row(a: &[f64], rows: usize, cols: usize, j: usize) -> usize {
    let mut best = (0, f64::NEG_INFINITY);
    for r in 0..rows {
        let residual = 1.0 - (0..j).map(|i| a[r * cols + i].powi(2)).sum::<f64>();
        if residual > best.1 {
            best = (r, residual);
        }
    }
    best.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Lcg;

    fn max_gram_deviation(a: &[f64], rows: usize, cols: usize) -> f64 {
        let mut worst = 0.0f64;
        for i in 0..cols {
            for j in 0..cols {
                let d: f64 = (0..rows).map(|r| a[r * cols + i] * a[r * cols + j]).sum();
                worst = worst.max((d - f64::from(u8::from(i == j))).abs());
            }
        }
        worst
    }

    #[test]
    fn random_matrix_becomes_orthonormal() {
        let (rows, cols) = (40, 9);
        let mut a = Lcg(3).matrix(rows, cols);
        assert_eq!(orthonormalize_columns(&mut a, rows, cols), Ok(0));
        assert!(max_gram_deviation(&a, rows, cols) < 1e-14);
    }

    #[test]
    fn ill_conditioned_columns_stay_orthonormal() {
        let (rows, cols) = (30, 6);
        let mut rng = Lcg(5);
        let base: Vec<f64> = (0..rows).map(|_| rng.next()).collect();
        let mut a = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                a[r * cols + c] = base[r] + 1e-7 * rng.next();
            }
        }
        orthonormalize_columns(&mut a, rows, cols).unwrap();
        assert!(max_gram_deviation(&a, rows, cols) < 1e-12);
    }

    #[test]
    fn dependent_and_zero_columns_are_replaced() {
        let (rows, cols) = (12, 4);
        let mut rng = Lcg(9);
        let mut a = vec![0.0; rows * cols];
        for r in 0..rows {
            let x = rng.next();
            a[r * cols] = x;
            a[r * cols + 1] = 2.0 * x;
            a[r * cols + 3] = rng.next();
        }
        // columns 1 (multiple of 0) and 2 (zero) are dependent
        assert_eq!(orthonormalize_columns(&mut a, rows, cols), Ok(2));
        assert!(max_gram_deviation(&a, rows, cols) < 1e-12);
        let mut zero = vec![0.0; rows * cols];
        assert_eq!(orthonormalize_columns(&mut zero, rows, cols), Ok(cols));
        assert!(max_gram_deviation(&zero, rows, cols) < 1e-12);
    }

    #[test]
    fn square_matrix_is_supported_and_run_is_reproducible() {
        let mut a = Lcg(11).matrix(7, 7);
        let mut b = a.clone();
        orthonormalize_columns(&mut a, 7, 7).unwrap();
        orthonormalize_columns(&mut b, 7, 7).unwrap();
        assert_eq!(a, b);
        assert!(max_gram_deviation(&a, 7, 7) < 1e-13);
    }

    #[test]
    fn bad_shapes_and_non_finite_are_rejected() {
        let mut a = vec![1.0; 6];
        assert!(matches!(
            orthonormalize_columns(&mut a, 2, 3),
            Err(LinalgError::Shape(_))
        ));
        assert!(matches!(
            orthonormalize_columns(&mut a, 3, 3),
            Err(LinalgError::Shape(_))
        ));
        let mut nan = vec![1.0, f64::NAN, 0.0, 1.0];
        assert_eq!(
            orthonormalize_columns(&mut nan, 2, 2),
            Err(LinalgError::NonFinite)
        );
    }
}
