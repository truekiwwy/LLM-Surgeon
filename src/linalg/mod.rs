//! Dense linear algebra in pure Rust, f64 everywhere: no BLAS, no LAPACK, no C.
//!
//! - [`jacobi`]: eigen-decomposition of a symmetric N×N matrix (cyclic Jacobi, N ≤ 1024).
//! - [`gram`]: thin SVD of a tall or wide X through its smaller Gram matrix (never d×d
//!   when d is the large side).
//! - [`power`]: power iteration with deflation on the implicit `XᵀX`, an independent
//!   cross-check of `gram` in the tests.
//! - [`cholesky`]: SPD factorization, solves, and the condition estimate κ (the "cond"
//!   part of the map below lives here: κ is read off the diagonal of L).
//!
//! Determinism: every reduction runs in a fixed order (parallel loops only split
//! independent outputs, each summed sequentially), so results are bit-identical whatever
//! the number of threads.
//!
//! (a) Modules and consumers (graphify AST, checked by hand):
//!
//! ```text
//! jacobi <- gram <- extract::svd (contrastive PCA on paired deltas)
//!                <- extract::leace (Σ_λ^p y through the thin SVD of X̃/√n)
//! power  (tests only: cross-check of gram)
//! cholesky (factor, solve, kappa) <- patch::autotune (ridge, LOOCV hat matrix)
//! engine::project_out (applies the LEACE a, b to down_proj weights; no linalg needed)
//! ```

pub mod cholesky;
pub mod gram;
pub mod jacobi;
pub mod power;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum LinalgError {
    #[error("dimension {n} au-delà de la limite {max} de Jacobi")]
    TooLarge { n: usize, max: usize },

    #[error("dimensions incohérentes : {0}")]
    Shape(String),

    #[error("valeur non finie (NaN ou Inf) dans la matrice d'entrée")]
    NonFinite,

    #[error(
        "Jacobi n'a pas convergé en {sweeps} balayages (hors-diagonale relative {residual:e})"
    )]
    NoConvergence { sweeps: usize, residual: f64 },

    #[error("matrice non définie positive (pivot {index} = {pivot:e})")]
    NotPositiveDefinite { index: usize, pivot: f64 },

    #[error("matrice mal conditionnée : κ ≈ {kappa:e} > {limit:e}")]
    IllConditioned { kappa: f64, limit: f64 },
}

pub type Result<T> = std::result::Result<T, LinalgError>;

/// Sequential dot product: the summation order is the index order.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub(crate) fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

pub(crate) fn check_finite(values: &[f64]) -> Result<()> {
    match values.iter().all(|v| v.is_finite()) {
        true => Ok(()),
        false => Err(LinalgError::NonFinite),
    }
}

/// Deterministic pseudo-random numbers in [-1, 1) for the tests (no `rand` crate).
#[cfg(test)]
pub(crate) struct Lcg(pub u64);

#[cfg(test)]
impl Lcg {
    pub(crate) fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }

    pub(crate) fn matrix(&mut self, rows: usize, cols: usize) -> Vec<f64> {
        (0..rows * cols).map(|_| self.next()).collect()
    }
}
