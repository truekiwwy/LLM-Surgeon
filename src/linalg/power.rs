//! Power iteration with deflation on the implicit operator `XᵀX` (matvec `Xᵀ(X v)`), the
//! independent cross-check of `gram` in the tests: no Gram matrix, no Jacobi.

use super::{check_finite, dot, norm, LinalgError, Result};

/// The `k` leading right singular vectors of the row-major N×d `x` (rows of length d) and
/// their singular values. Each vector is iterated until it moves by less than `tol` in
/// norm (the Rayleigh quotient converges twice as fast as the vector, so it would stop too
/// early), orthogonalized against the previous ones at every step.
pub fn top_right_singular(
    x: &[f64],
    n: usize,
    d: usize,
    k: usize,
    tol: f64,
    max_iter: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if x.len() != n * d || k > n.min(d) {
        return Err(LinalgError::Shape(format!(
            "{k} vecteurs demandés pour une matrice {n}×{d}"
        )));
    }
    check_finite(x)?;
    let apply = |v: &[f64]| -> Vec<f64> {
        let xv: Vec<f64> = (0..n).map(|i| dot(&x[i * d..(i + 1) * d], v)).collect();
        let mut out = vec![0.0; d];
        for (i, s) in xv.iter().enumerate() {
            for (o, xij) in out.iter_mut().zip(&x[i * d..(i + 1) * d]) {
                *o += s * xij;
            }
        }
        out
    };
    let deflate = |v: &mut Vec<f64>, found: &[f64]| {
        for prev in found.chunks_exact(d) {
            let c = dot(v, prev);
            v.iter_mut().zip(prev).for_each(|(a, b)| *a -= c * b);
        }
    };
    let (mut sigmas, mut vectors) = (Vec::with_capacity(k), Vec::with_capacity(k * d));
    for index in 0..k {
        // Deterministic start with no symmetry that could vanish on a singular subspace.
        let mut v: Vec<f64> = (0..d)
            .map(|j| 1.0 + ((j * 7919 + index * 104_729) % 1000) as f64 / 1000.0)
            .collect();
        deflate(&mut v, &vectors);
        let len = norm(&v);
        v.iter_mut().for_each(|a| *a /= len);
        let mut lambda = 0.0;
        for _ in 0..max_iter {
            let mut w = apply(&v);
            deflate(&mut w, &vectors);
            let next = dot(&w, &v);
            let len = norm(&w);
            if len == 0.0 {
                break;
            }
            w.iter_mut().for_each(|a| *a /= len);
            // XᵀX is positive semi-definite: no sign flip between iterates.
            let moved = w
                .iter()
                .zip(&v)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f64>()
                .sqrt();
            let settled = moved <= tol;
            lambda = next;
            v = w;
            if settled {
                break;
            }
        }
        sigmas.push(lambda.max(0.0).sqrt());
        vectors.extend_from_slice(&v);
    }
    Ok((sigmas, vectors))
}
