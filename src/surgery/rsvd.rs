//! Randomized SVD (Halko, Martinsson, Tropp) of a matrix read by row blocks
//! (frozen API). Shared by `diff` and `lora extract`.
//!
//! Algorithm for `A` (m x n), target rank `r`, `k = r + oversample` (clamped to `min(m, n)`):
//! `Q = orth(A Omega)`, then `power_iters` times `Z = orth(A^T Q)`, `Q = orth(A Z)`, then
//! `B^T = A^T Q` (n x k), the k x k eigenproblem `B B^T = U_B S^2 U_B^T` (cyclic Jacobi),
//! `U = Q U_B`, `V = (B^T U_B)^T / sigma`, truncated to `r`. `orth` is modified Gram-Schmidt
//! applied twice ([`crate::linalg::qr`]).
//!
//! Streaming: `A` is never held in memory. Every product walks `A` in blocks of
//! [`BLOCK_ROWS`] rows, [`WAVE_BLOCKS`] blocks at a time (read and computed in parallel with
//! rayon). Block and wave sizes are constants, so the partition, and with it every
//! floating point reduction order, does not depend on the thread count: row-wise products
//! are independent per row, and the per-block partial sums (`A^T Q`, squared norms) are
//! added in block order after each wave. All accumulations are f64.
//!
//! PRNG: `Omega` (n x k, row-major) is filled with standard normal variates from xoshiro256**
//! (Blackman, Vigna) whose 256-bit state is four consecutive outputs of splitmix64 seeded with
//! `seed`; Box-Muller turns pairs of uniforms into pairs of normals. The result is therefore
//! a pure function of `(A, params)` for one build of the program (`ln`/`cos` come from the
//! platform libm, bit-identical results across platforms are not promised).

use std::sync::atomic::{AtomicBool, Ordering};

use rayon::prelude::*;

use super::error::{Result, SurgeryError};
use crate::linalg::jacobi::{symmetric_eigen, MAX_N};
use crate::linalg::qr::orthonormalize_columns;
use crate::linalg::LinalgError;

/// Rows per block: fixed so that the reduction order never depends on the thread count.
const BLOCK_ROWS: usize = 64;
/// Blocks processed at once; bounds the memory of the per-block partial sums.
const WAVE_BLOCKS: usize = 16;
/// Columns handled together in the cache-friendly kernels.
const COL_TILE: usize = 256;
/// A singular value below this fraction of the largest one is treated as zero (its row of
/// `V` is set to zero instead of dividing rounding noise by a tiny sigma).
const SIGMA_FLOOR: f64 = 1e-10;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RsvdParams {
    pub rank: usize,
    pub oversample: usize,
    pub power_iters: usize,
    pub seed: u64,
}

impl Default for RsvdParams {
    fn default() -> Self {
        Self {
            rank: 16,
            oversample: 8,
            power_iters: 3,
            seed: 0,
        }
    }
}

/// `A ~ U diag(s) V`, `u` row-major [m x r], `v` row-major [r x n].
#[derive(Debug, Clone, PartialEq)]
pub struct Rsvd {
    pub m: usize,
    pub n: usize,
    pub r: usize,
    pub u: Vec<f32>,
    pub s: Vec<f32>,
    pub v: Vec<f32>,
    /// `||A - U S V||_F / ||A||_F`, exact (computed by blocks), 0 for a null matrix.
    pub rel_error: f64,
    /// `sum(s^2) / ||A||_F^2`.
    pub energy: f64,
}

/// A matrix that can be streamed by rows (e.g. `tuned - base` computed on the fly).
pub trait RowSource: Sync {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    /// Fills `out` with consecutive rows starting at `start` (`out.len()` is a multiple of
    /// `cols`); returns the number of rows written.
    fn read_rows(&self, start: usize, out: &mut [f32]) -> Result<usize>;
}

/// Deterministic for a given `(A, params)`, whatever the rayon thread count.
///
/// `rank` is clamped to `min(m, n)`; the effective rank is `Rsvd::r`. A null matrix gives
/// zero factors. `rank == 0` or an empty matrix is a usage error, a non finite entry a
/// failure.
pub fn rsvd(a: &dyn RowSource, p: &RsvdParams, cancel: &AtomicBool) -> Result<Rsvd> {
    let (m, n) = (a.rows(), a.cols());
    if p.rank == 0 {
        return Err(SurgeryError::Usage("rsvd : rank must be at least 1".into()));
    }
    if m == 0 || n == 0 {
        return Err(SurgeryError::Usage(format!("rsvd : empty matrix {m}x{n}")));
    }
    let r = p.rank.min(m).min(n);
    let k = r.saturating_add(p.oversample).min(m).min(n);
    if k > MAX_N {
        return Err(SurgeryError::Usage(format!(
            "rsvd : rank + oversample = {k} exceeds {MAX_N}"
        )));
    }
    let src = Streamer { a, m, n, cancel };

    let omega = gaussian_matrix(n, k, p.seed);
    let (mut q, frob2) = src.mul(&omega, k)?;
    if frob2 == 0.0 {
        return Ok(Rsvd {
            m,
            n,
            r,
            u: vec![0.0; m * r],
            s: vec![0.0; r],
            v: vec![0.0; r * n],
            rel_error: 0.0,
            energy: 0.0,
        });
    }
    orthonormalize(&mut q, m, k)?;
    for _ in 0..p.power_iters {
        let mut z = src.transpose_mul(&q, k)?;
        orthonormalize(&mut z, n, k)?;
        (q, _) = src.mul(&z, k)?;
        orthonormalize(&mut q, m, k)?;
    }
    // z = B^T (n x k); G = B B^T = z^T z, summed over j in index order.
    let z = src.transpose_mul(&q, k)?;
    let mut gram = vec![0.0; k * k];
    for zrow in z.chunks_exact(k) {
        for i in 0..k {
            for j in i..k {
                gram[i * k + j] += zrow[i] * zrow[j];
            }
        }
    }
    for i in 0..k {
        for j in 0..i {
            gram[i * k + j] = gram[j * k + i];
        }
    }
    let eigen = symmetric_eigen(&gram, k).map_err(numeric)?;

    // Rows of V before normalization are B^T u_t; their norm is the singular value (more
    // accurate for the small ones than sqrt of the eigenvalue).
    let mut parts: Vec<Component> = (0..r)
        .map(|t| {
            let ub = eigen.vector(t);
            let v: Vec<f64> = z
                .chunks_exact(k)
                .map(|zrow| zrow.iter().zip(&ub).map(|(a, b)| a * b).sum())
                .collect();
            let sigma = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            Component { ub, v, sigma }
        })
        .collect();
    if parts.iter().any(|c| !c.sigma.is_finite()) {
        return Err(SurgeryError::Failed(
            "rsvd : non finite singular value".into(),
        ));
    }
    parts.sort_by(|x, y| y.sigma.total_cmp(&x.sigma));
    let floor = SIGMA_FLOOR * parts[0].sigma;

    let mut u = vec![0.0; m * r];
    for (qrow, urow) in q.chunks_exact(k).zip(u.chunks_exact_mut(r)) {
        for (uv, c) in urow.iter_mut().zip(&parts) {
            *uv = qrow.iter().zip(&c.ub).map(|(a, b)| a * b).sum();
        }
    }
    let mut s = vec![0.0f32; r];
    let mut v = vec![0.0f32; r * n];
    for (t, c) in parts.iter().enumerate() {
        let live = c.sigma > floor;
        let flip = if live {
            c.v[largest_abs(c.v.iter().copied())] < 0.0
        } else {
            column_flip(&u, r, t)
        };
        let sign = if flip { -1.0 } else { 1.0 };
        if flip {
            for i in 0..m {
                u[i * r + t] = -u[i * r + t];
            }
        }
        if live {
            if c.sigma > f64::from(f32::MAX) {
                return Err(SurgeryError::Failed(
                    "rsvd : valeur singulière hors de la plage f32".into(),
                ));
            }
            s[t] = c.sigma as f32;
            for (dst, &x) in v[t * n..(t + 1) * n].iter_mut().zip(&c.v) {
                *dst = (sign * x / c.sigma) as f32;
            }
        }
    }
    let u: Vec<f32> = u.into_iter().map(|x| x as f32).collect();

    let residual2 = src.residual_sq(&u, &s, &v, r)?;
    let captured2: f64 = s.iter().map(|&x| f64::from(x).powi(2)).sum();
    Ok(Rsvd {
        m,
        n,
        r,
        u,
        s,
        v,
        rel_error: (residual2 / frob2).sqrt(),
        energy: (captured2 / frob2).min(1.0),
    })
}

struct Component {
    /// Eigenvector of `B B^T` (length k).
    ub: Vec<f64>,
    /// `B^T ub`, length n, norm `sigma`.
    v: Vec<f64>,
    sigma: f64,
}

fn numeric(e: LinalgError) -> SurgeryError {
    SurgeryError::Failed(format!("rsvd : {e}"))
}

fn orthonormalize(a: &mut [f64], rows: usize, cols: usize) -> Result<()> {
    orthonormalize_columns(a, rows, cols).map_err(numeric)?;
    Ok(())
}

/// Index of the first entry of maximal absolute value.
fn largest_abs(values: impl Iterator<Item = f64>) -> usize {
    let mut best = (0, -1.0);
    for (i, x) in values.enumerate() {
        if x.abs() > best.1 {
            best = (i, x.abs());
        }
    }
    best.0
}

/// Sign rule applied to a column of `U` (stride `r`) when its `V` row is null.
fn column_flip(u: &[f64], r: usize, t: usize) -> bool {
    let i = largest_abs(u.iter().skip(t).step_by(r).copied());
    u[i * r + t] < 0.0
}

/// Reads `A` block by block and runs the products of the algorithm over it.
struct Streamer<'a> {
    a: &'a dyn RowSource,
    m: usize,
    n: usize,
    cancel: &'a AtomicBool,
}

impl Streamer<'_> {
    /// Rows of block `b`, checked finite. Sources may return fewer rows than asked.
    fn read_block(&self, b: usize) -> Result<Vec<f32>> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(SurgeryError::Cancelled);
        }
        let row0 = b * BLOCK_ROWS;
        let rows = BLOCK_ROWS.min(self.m - row0);
        let mut buf = vec![0.0f32; rows * self.n];
        let mut filled = 0;
        while filled < rows {
            let got = self
                .a
                .read_rows(row0 + filled, &mut buf[filled * self.n..])?;
            if got == 0 || got > rows - filled {
                return Err(SurgeryError::Failed(format!(
                    "rsvd : source returned {got} rows at row {}",
                    row0 + filled
                )));
            }
            filled += got;
        }
        if !buf.iter().all(|x| x.is_finite()) {
            return Err(SurgeryError::Failed(format!(
                "rsvd : non finite value in rows {row0}..{}",
                row0 + rows
            )));
        }
        Ok(buf)
    }

    /// Runs `work(first_row, rows)` on every block, a wave of blocks in parallel, and feeds
    /// the results to `sink` in block order.
    fn run<T: Send>(
        &self,
        work: impl Fn(usize, &[f32]) -> T + Sync,
        mut sink: impl FnMut(T),
    ) -> Result<()> {
        let blocks = self.m.div_ceil(BLOCK_ROWS);
        for first in (0..blocks).step_by(WAVE_BLOCKS) {
            let last = (first + WAVE_BLOCKS).min(blocks);
            let parts = (first..last)
                .into_par_iter()
                .map(|b| Ok(work(b * BLOCK_ROWS, &self.read_block(b)?)))
                .collect::<Result<Vec<T>>>()?;
            parts.into_iter().for_each(&mut sink);
        }
        Ok(())
    }

    /// `A W` (m x width) for `w` of shape n x width, and `||A||_F^2`.
    fn mul(&self, w: &[f64], width: usize) -> Result<(Vec<f64>, f64)> {
        let n = self.n;
        let mut out = Vec::with_capacity(self.m * width);
        let mut frob2 = 0.0;
        self.run(
            |_, data| {
                let mut y = vec![0.0; data.len() / n * width];
                let mut sq = 0.0;
                for (row, yrow) in data.chunks_exact(n).zip(y.chunks_exact_mut(width)) {
                    for (&x, wrow) in row.iter().zip(w.chunks_exact(width)) {
                        let x = f64::from(x);
                        sq += x * x;
                        for (acc, wv) in yrow.iter_mut().zip(wrow) {
                            *acc += x * wv;
                        }
                    }
                }
                (y, sq)
            },
            |(y, sq)| {
                out.extend_from_slice(&y);
                frob2 += sq;
            },
        )?;
        Ok((out, frob2))
    }

    /// `A^T Q` (n x width) for `q` of shape m x width.
    fn transpose_mul(&self, q: &[f64], width: usize) -> Result<Vec<f64>> {
        let n = self.n;
        let mut z = vec![0.0; n * width];
        self.run(
            |row0, data| {
                let mut part = vec![0.0; n * width];
                for j0 in (0..n).step_by(COL_TILE) {
                    let j1 = (j0 + COL_TILE).min(n);
                    for (i, row) in data.chunks_exact(n).enumerate() {
                        let qrow = &q[(row0 + i) * width..(row0 + i + 1) * width];
                        let zrows = part[j0 * width..j1 * width].chunks_exact_mut(width);
                        for (zrow, &x) in zrows.zip(&row[j0..j1]) {
                            let x = f64::from(x);
                            for (zv, qv) in zrow.iter_mut().zip(qrow) {
                                *zv += x * qv;
                            }
                        }
                    }
                }
                part
            },
            |part| {
                for (zv, pv) in z.iter_mut().zip(&part) {
                    *zv += pv;
                }
            },
        )?;
        Ok(z)
    }

    /// `||A - U diag(s) V||_F^2` with the factors as delivered (f32).
    fn residual_sq(&self, u: &[f32], s: &[f32], v: &[f32], r: usize) -> Result<f64> {
        let n = self.n;
        let mut total = 0.0;
        self.run(
            |row0, data| {
                let mut sum = 0.0;
                let mut res = [0.0f64; COL_TILE];
                for (i, row) in data.chunks_exact(n).enumerate() {
                    let urow = &u[(row0 + i) * r..(row0 + i + 1) * r];
                    for j0 in (0..n).step_by(COL_TILE) {
                        let j1 = (j0 + COL_TILE).min(n);
                        let res = &mut res[..j1 - j0];
                        for (rv, &x) in res.iter_mut().zip(&row[j0..j1]) {
                            *rv = f64::from(x);
                        }
                        for (t, (&ut, &st)) in urow.iter().zip(s).enumerate() {
                            let c = f64::from(ut) * f64::from(st);
                            for (rv, &vv) in res.iter_mut().zip(&v[t * n + j0..t * n + j1]) {
                                *rv -= c * f64::from(vv);
                            }
                        }
                        sum += res.iter().map(|x| x * x).sum::<f64>();
                    }
                }
                sum
            },
            |sum| total += sum,
        )?;
        Ok(total)
    }
}

/// xoshiro256** seeded through splitmix64.
struct Xoshiro256([u64; 4]);

impl Xoshiro256 {
    fn new(seed: u64) -> Self {
        let mut sm = seed;
        let mut next = || {
            sm = sm.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = sm;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self([next(), next(), next(), next()])
    }

    fn next_u64(&mut self) -> u64 {
        let s = &mut self.0;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Two independent standard normal variates (Box-Muller).
    fn next_gaussian_pair(&mut self) -> (f64, f64) {
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        let u1 = ((self.next_u64() >> 11) + 1) as f64 * SCALE; // (0, 1], keeps ln finite
        let u2 = (self.next_u64() >> 11) as f64 * SCALE;
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f64::consts::TAU * u2;
        (radius * angle.cos(), radius * angle.sin())
    }
}

/// `rows x cols` standard normal matrix, row-major, filled in index order.
fn gaussian_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f64> {
    let mut rng = Xoshiro256::new(seed);
    let len = rows * cols;
    let mut out = Vec::with_capacity(len + 1);
    while out.len() < len {
        let (a, b) = rng.next_gaussian_pair();
        out.push(a);
        out.push(b);
    }
    out.truncate(len);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Lcg;
    use std::sync::atomic::AtomicUsize;

    /// In-memory matrix served `chunk` rows at a time at most.
    struct MemSource {
        m: usize,
        n: usize,
        data: Vec<f32>,
        chunk: usize,
    }

    impl MemSource {
        fn new(m: usize, n: usize, data: Vec<f32>) -> Self {
            Self {
                m,
                n,
                data,
                chunk: usize::MAX,
            }
        }
        fn chunked(mut self, chunk: usize) -> Self {
            self.chunk = chunk;
            self
        }
    }

    impl RowSource for MemSource {
        fn rows(&self) -> usize {
            self.m
        }
        fn cols(&self) -> usize {
            self.n
        }
        fn read_rows(&self, start: usize, out: &mut [f32]) -> Result<usize> {
            let rows = (out.len() / self.n).min(self.m - start).min(self.chunk);
            out[..rows * self.n]
                .copy_from_slice(&self.data[start * self.n..(start + rows) * self.n]);
            Ok(rows)
        }
    }

    fn run(a: &MemSource, rank: usize) -> Rsvd {
        let p = RsvdParams {
            rank,
            ..RsvdParams::default()
        };
        rsvd(a, &p, &AtomicBool::new(false)).unwrap()
    }

    /// Sum of `r` outer products with decreasing weights (exact rank `r`).
    fn exact_rank(m: usize, n: usize, r: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg(seed);
        let mut a = vec![0.0f64; m * n];
        for t in 0..r {
            let w = 4.0 / (t + 1) as f64;
            let u: Vec<f64> = (0..m).map(|_| rng.next()).collect();
            let v: Vec<f64> = (0..n).map(|_| rng.next()).collect();
            for i in 0..m {
                for j in 0..n {
                    a[i * n + j] += w * u[i] * v[j];
                }
            }
        }
        a.into_iter().map(|x| x as f32).collect()
    }

    fn random(m: usize, n: usize, seed: u64) -> Vec<f32> {
        Lcg(seed)
            .matrix(m, n)
            .into_iter()
            .map(|x| x as f32)
            .collect()
    }

    /// Singular values of `a` (descending) through the eigenvalues of `A^T A` in f64.
    fn reference_sigma(a: &[f32], m: usize, n: usize) -> Vec<f64> {
        let mut g = vec![0.0f64; n * n];
        for row in a.chunks_exact(n) {
            for i in 0..n {
                for j in 0..n {
                    g[i * n + j] += f64::from(row[i]) * f64::from(row[j]);
                }
            }
        }
        let e = symmetric_eigen(&g, n).unwrap();
        assert_eq!(m * n, a.len());
        e.values.iter().map(|&l| l.max(0.0).sqrt()).collect()
    }

    fn gram_deviation(x: &[f32], rows: usize, cols: usize, by_columns: bool) -> f64 {
        let at = |i: usize, j: usize| f64::from(x[i * cols + j]);
        let (count, len) = if by_columns {
            (cols, rows)
        } else {
            (rows, cols)
        };
        let mut worst = 0.0f64;
        for a in 0..count {
            for b in 0..count {
                let d: f64 = (0..len)
                    .map(|c| {
                        if by_columns {
                            at(c, a) * at(c, b)
                        } else {
                            at(a, c) * at(b, c)
                        }
                    })
                    .sum();
                worst = worst.max((d - f64::from(u8::from(a == b))).abs());
            }
        }
        worst
    }

    fn projector(u: &[f32], m: usize, r: usize) -> Vec<f64> {
        let mut p = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                p[i * m + j] = (0..r)
                    .map(|t| f64::from(u[i * r + t]) * f64::from(u[j * r + t]))
                    .sum();
            }
        }
        p
    }

    #[test]
    fn exact_low_rank_matrix_is_recovered() {
        let a = MemSource::new(90, 70, exact_rank(90, 70, 6, 1));
        let out = run(&a, 6);
        assert_eq!(out.r, 6);
        assert!(out.rel_error < 1e-5, "rel_error {}", out.rel_error);
        assert!(out.energy > 1.0 - 1e-5 && out.energy <= 1.0);
    }

    #[test]
    fn full_matrix_error_is_near_the_eckart_young_optimum() {
        let (m, n, r) = (64, 48, 8);
        let data = random(m, n, 7);
        let sigma = reference_sigma(&data, m, n);
        let total: f64 = sigma.iter().map(|x| x * x).sum();
        let tail: f64 = sigma[r..].iter().map(|x| x * x).sum();
        let optimum = (tail / total).sqrt();
        let out = run(&MemSource::new(m, n, data), r);
        assert!(out.rel_error >= optimum * (1.0 - 1e-4));
        assert!(
            out.rel_error <= 1.2 * optimum,
            "{} vs optimum {optimum}",
            out.rel_error
        );
        for (got, want) in out.s.iter().zip(&sigma) {
            assert!((f64::from(*got) / want - 1.0).abs() < 0.2);
        }
    }

    #[test]
    fn factors_are_orthonormal_sorted_and_sign_normalized() {
        let (m, n, r) = (64, 48, 8);
        let out = run(&MemSource::new(m, n, random(m, n, 3)), r);
        assert!(gram_deviation(&out.u, m, r, true) < 1e-5);
        assert!(gram_deviation(&out.v, r, n, false) < 1e-5);
        assert!(out.s.windows(2).all(|w| w[0] >= w[1]));
        assert!(out.s.iter().all(|&x| x > 0.0));
        for row in out.v.chunks_exact(n) {
            let big = largest_abs(row.iter().map(|&x| f64::from(x)));
            assert!(row[big] > 0.0);
        }
    }

    #[test]
    fn result_is_bitwise_identical_across_threads_and_runs() {
        // 1100 rows = 18 blocks: more than one wave.
        let a = MemSource::new(1100, 40, exact_rank(1100, 40, 9, 5));
        let p = RsvdParams {
            rank: 5,
            ..RsvdParams::default()
        };
        let go = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| rsvd(&a, &p, &AtomicBool::new(false)).unwrap())
        };
        let one = go(1);
        assert_eq!(one, go(1));
        assert_eq!(one, go(2));
        assert_eq!(one, go(8));
        assert!(one.rel_error > 0.0);
    }

    #[test]
    fn different_seeds_find_the_same_subspace() {
        let (m, n, r) = (60, 50, 5);
        let a = MemSource::new(m, n, exact_rank(m, n, r, 2));
        let with = |seed| {
            let p = RsvdParams {
                rank: r,
                seed,
                ..RsvdParams::default()
            };
            rsvd(&a, &p, &AtomicBool::new(false)).unwrap()
        };
        let (x, y) = (with(1), with(99));
        let (px, py) = (projector(&x.u, m, r), projector(&y.u, m, r));
        let diff: f64 = px.iter().zip(&py).map(|(a, b)| (a - b).powi(2)).sum();
        assert!(diff.sqrt() < 1e-4, "projector gap {}", diff.sqrt());
    }

    #[test]
    fn odd_sized_source_reads_give_the_same_bits() {
        let (m, n) = (150, 33);
        let data = exact_rank(m, n, 7, 4);
        let whole = run(&MemSource::new(m, n, data.clone()), 4);
        for chunk in [1, 7, 13, 64, 65] {
            assert_eq!(
                whole,
                run(&MemSource::new(m, n, data.clone()).chunked(chunk), 4)
            );
        }
    }

    #[test]
    fn null_matrix_gives_zero_factors() {
        let out = run(&MemSource::new(10, 6, vec![0.0; 60]), 3);
        assert_eq!((out.r, out.rel_error, out.energy), (3, 0.0, 0.0));
        assert!(out.u.iter().chain(&out.s).chain(&out.v).all(|&x| x == 0.0));
    }

    #[test]
    fn single_row_and_single_column_matrices() {
        let row = MemSource::new(1, 5, vec![3.0, -4.0, 0.0, 0.0, 12.0]);
        let out = run(&row, 3);
        assert_eq!(out.r, 1);
        assert!((out.s[0] - 13.0).abs() < 1e-4);
        assert!(out.rel_error < 1e-6);
        let col = MemSource::new(4, 1, vec![1.0, 2.0, 2.0, 0.0]);
        let out = run(&col, 2);
        assert_eq!(out.r, 1);
        assert!((out.s[0] - 3.0).abs() < 1e-5);
        assert!(out.rel_error < 1e-6);
    }

    #[test]
    fn rank_above_min_dimension_is_clamped() {
        let (m, n) = (7, 4);
        let out = run(&MemSource::new(m, n, random(m, n, 6)), 50);
        assert_eq!(out.r, 4);
        assert_eq!((out.u.len(), out.s.len(), out.v.len()), (m * 4, 4, 4 * n));
        assert!(out.rel_error < 1e-5, "rel_error {}", out.rel_error);
        assert!(gram_deviation(&out.v, 4, n, false) < 1e-5);
    }

    #[test]
    fn rank_above_actual_rank_stays_finite_and_exact() {
        let a = MemSource::new(40, 30, exact_rank(40, 30, 2, 8));
        let out = run(&a, 10);
        assert!(out.rel_error < 1e-5);
        assert!(out.u.iter().chain(&out.v).all(|x| x.is_finite()));
    }

    #[test]
    fn bad_inputs_are_rejected() {
        let ok = MemSource::new(3, 3, vec![1.0; 9]);
        let params = |rank| RsvdParams {
            rank,
            ..RsvdParams::default()
        };
        let never = AtomicBool::new(false);
        assert!(matches!(
            rsvd(&ok, &params(0), &never),
            Err(SurgeryError::Usage(_))
        ));
        let empty = MemSource::new(0, 3, vec![]);
        assert!(matches!(
            rsvd(&empty, &params(1), &never),
            Err(SurgeryError::Usage(_))
        ));
        let mut data = vec![1.0; 9];
        data[4] = f32::NAN;
        assert!(matches!(
            rsvd(&MemSource::new(3, 3, data), &params(1), &never),
            Err(SurgeryError::Failed(_))
        ));
        let mut data = vec![1.0; 9];
        data[8] = f32::INFINITY;
        assert!(matches!(
            rsvd(&MemSource::new(3, 3, data), &params(1), &never),
            Err(SurgeryError::Failed(_))
        ));
    }

    #[test]
    fn cancellation_is_honoured_before_and_during_the_run() {
        let a = MemSource::new(200, 20, random(200, 20, 1));
        let p = RsvdParams::default();
        assert!(matches!(
            rsvd(&a, &p, &AtomicBool::new(true)),
            Err(SurgeryError::Cancelled)
        ));

        /// Raises the flag after a few reads, as a Ctrl-C would.
        struct Trip<'a> {
            inner: &'a MemSource,
            flag: &'a AtomicBool,
            reads: AtomicUsize,
        }
        impl RowSource for Trip<'_> {
            fn rows(&self) -> usize {
                self.inner.rows()
            }
            fn cols(&self) -> usize {
                self.inner.cols()
            }
            fn read_rows(&self, start: usize, out: &mut [f32]) -> Result<usize> {
                if self.reads.fetch_add(1, Ordering::Relaxed) == 5 {
                    self.flag.store(true, Ordering::Relaxed);
                }
                self.inner.read_rows(start, out)
            }
        }
        let flag = AtomicBool::new(false);
        let trip = Trip {
            inner: &a,
            flag: &flag,
            reads: AtomicUsize::new(0),
        };
        assert!(matches!(
            rsvd(&trip, &p, &flag),
            Err(SurgeryError::Cancelled)
        ));
    }

    #[test]
    fn misbehaving_source_is_an_error() {
        struct Empty;
        impl RowSource for Empty {
            fn rows(&self) -> usize {
                4
            }
            fn cols(&self) -> usize {
                2
            }
            fn read_rows(&self, _: usize, _: &mut [f32]) -> Result<usize> {
                Ok(0)
            }
        }
        assert!(matches!(
            rsvd(&Empty, &RsvdParams::default(), &AtomicBool::new(false)),
            Err(SurgeryError::Failed(_))
        ));
    }

    #[test]
    fn prng_is_reproducible_and_roughly_standard_normal() {
        let a = gaussian_matrix(200, 25, 42);
        assert_eq!(a, gaussian_matrix(200, 25, 42));
        assert_ne!(a, gaussian_matrix(200, 25, 43));
        let mean = a.iter().sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(mean.abs() < 0.1 && (var - 1.0).abs() < 0.1, "{mean} {var}");
        assert_eq!(gaussian_matrix(3, 3, 0).len(), 9);
    }

    /// Synthetic delta computed row by row: rank-16 signal plus noise, nothing stored.
    struct Synthetic {
        m: usize,
        n: usize,
        left: Vec<f32>,
        right: Vec<f32>,
    }

    impl RowSource for Synthetic {
        fn rows(&self) -> usize {
            self.m
        }
        fn cols(&self) -> usize {
            self.n
        }
        fn read_rows(&self, start: usize, out: &mut [f32]) -> Result<usize> {
            let rows = (out.len() / self.n).min(self.m - start);
            for (i, row) in out.chunks_exact_mut(self.n).take(rows).enumerate() {
                let l = &self.left[(start + i) * 16..(start + i + 1) * 16];
                for (j, x) in row.iter_mut().enumerate() {
                    let r = &self.right[j * 16..(j + 1) * 16];
                    *x = l.iter().zip(r).map(|(a, b)| a * b).sum::<f32>() * 1e-3;
                }
            }
            Ok(rows)
        }
    }

    /// Timing reference: `cargo test --release rsvd_benchmark -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark"]
    fn rsvd_benchmark_1024x2752() {
        let (m, n) = (1024, 2752);
        let gen = |len, seed| -> Vec<f32> {
            Lcg(seed)
                .matrix(len, 1)
                .into_iter()
                .map(|x| x as f32)
                .collect()
        };
        let src = Synthetic {
            m,
            n,
            left: gen(m * 16, 1),
            right: gen(n * 16, 2),
        };
        let start = std::time::Instant::now();
        let out = rsvd(&src, &RsvdParams::default(), &AtomicBool::new(false)).unwrap();
        println!(
            "rsvd {m}x{n} rank 16: {:.2?}, rel_error {:.2e}",
            start.elapsed(),
            out.rel_error
        );
        assert!(out.rel_error < 1e-5);
    }
}
