//! Compares the three `surgeon extract --method` estimators on SYNTHETIC data with a known
//! planted direction. Deterministic (fixed seed); only the timings vary between runs.
//!
//! Model, for pair i (pos/neg share a "syntax" component g_i, as two prompts that differ by
//! the concept only):
//!
//! ```text
//! x⁻_i = g_i + e⁻_i,  x⁺_i = g_i + c_i t + e⁺_i
//! g_i: shared, large on the first 10 % of the axes (scale 3), small elsewhere (0.3)
//! e:   independent, anisotropic per axis (scale 0.2 .. 2.0)
//! c_i: concept strength, uniform in [0.5 s, 1.5 s], s = 0.5·√d ("weak": 0.15·√d), so the
//!      signal keeps the same footing against the per-axis noise whatever d
//! ```
//!
//! Measures, per method: cosine of the written direction with the true t, split-half
//! stability (directions fitted on even and odd pairs), and for LEACE the cosine of the read
//! direction b with the optimal one W⁻¹t (W: the within-class covariance, diagonal here).
//!
//! `cargo run --release --example bench_directions`

use std::time::Instant;

use weight_surgeon::extract::{leace_fit, separation_with_projections, svd_direction};

struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Sum of three centered uniforms, rescaled to unit variance: close to N(0, 1).
    fn normal(&mut self) -> f64 {
        (self.uniform() + self.uniform() + self.uniform() - 1.5) * 2.0
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn cosine(a: &[f64], b: &[f64]) -> f64 {
    dot(a, b) / (dot(a, a).sqrt() * dot(b, b).sqrt())
}

struct Data {
    pos: Vec<Vec<f64>>,
    neg: Vec<Vec<f64>>,
    truth: Vec<f64>,
    /// W⁻¹ t for the diagonal within-class covariance.
    lda: Vec<f64>,
}

fn generate(pairs: usize, d: usize, strength: f64, seed: u64) -> Data {
    let mut rng = Lcg(seed);
    let mut truth: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
    let n = dot(&truth, &truth).sqrt();
    truth.iter_mut().for_each(|x| *x /= n);
    let g_scale: Vec<f64> = (0..d).map(|j| if j < d / 10 { 3.0 } else { 0.3 }).collect();
    let e_scale: Vec<f64> = (0..d)
        .map(|j| 0.2 + 1.8 * ((j * 7919) % d) as f64 / d as f64)
        .collect();
    let (mut pos, mut neg) = (Vec::new(), Vec::new());
    for _ in 0..pairs {
        let g: Vec<f64> = g_scale.iter().map(|s| s * rng.normal()).collect();
        let c = strength * (0.5 + rng.uniform());
        pos.push(
            (0..d)
                .map(|j| g[j] + c * truth[j] + e_scale[j] * rng.normal())
                .collect(),
        );
        neg.push((0..d).map(|j| g[j] + e_scale[j] * rng.normal()).collect());
    }
    // Within-class covariance: g and e are both inside each class (diagonal).
    let lda = (0..d)
        .map(|j| truth[j] / (g_scale[j].powi(2) + e_scale[j].powi(2)))
        .collect();
    Data {
        pos,
        neg,
        truth,
        lda,
    }
}

fn halves(rows: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    (
        rows.iter().step_by(2).cloned().collect(),
        rows.iter().skip(1).step_by(2).cloned().collect(),
    )
}

#[derive(Clone, Copy)]
enum Method {
    MeanDiff,
    Svd,
    Leace,
}

impl Method {
    fn name(self) -> &'static str {
        match self {
            Self::MeanDiff => "mean_diff",
            Self::Svd => "svd",
            Self::Leace => "leace",
        }
    }

    /// (written direction, LEACE read direction b).
    fn fit(self, pos: &[Vec<f64>], neg: &[Vec<f64>]) -> (Vec<f64>, Option<Vec<f64>>) {
        match self {
            Self::MeanDiff => {
                let s = separation_with_projections(pos, neg).expect("mean_diff");
                (s.direction.iter().map(|&x| f64::from(x)).collect(), None)
            }
            Self::Svd => (svd_direction(pos, neg, false).expect("svd").0, None),
            Self::Leace => {
                let fit = leace_fit(pos, neg, 0.1, 1e-10).expect("leace");
                (fit.a, Some(fit.b))
            }
        }
    }
}

fn main() {
    println!("| scénario | paires N | d | méthode | cos(v, t) | split-half | cos(b, W⁻¹t) | temps (ms) |");
    println!("|---|---:|---:|---|---:|---:|---:|---:|");
    let scenarios = [
        ("standard", 8, 64, 0.5),
        ("standard", 32, 64, 0.5),
        ("standard", 128, 64, 0.5),
        ("standard", 32, 1024, 0.5),
        ("standard", 128, 1024, 0.5),
        ("faible", 32, 1024, 0.15),
        ("faible", 128, 1024, 0.15),
    ];
    for (k, (label, pairs, d, strength)) in scenarios.into_iter().enumerate() {
        let data = generate(pairs, d, strength * (d as f64).sqrt(), 2026 + k as u64);
        let (pos_a, pos_b) = halves(&data.pos);
        let (neg_a, neg_b) = halves(&data.neg);
        for method in [Method::MeanDiff, Method::Svd, Method::Leace] {
            let started = Instant::now();
            let (v, b) = method.fit(&data.pos, &data.neg);
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            let (va, _) = method.fit(&pos_a, &neg_a);
            let (vb, _) = method.fit(&pos_b, &neg_b);
            let read = b.map_or("—".to_owned(), |b| {
                format!("{:.3}", cosine(&b, &data.lda))
            });
            println!(
                "| {label} | {pairs} | {d} | {} | {:.3} | {:.3} | {read} | {ms:.1} |",
                method.name(),
                cosine(&v, &data.truth),
                cosine(&va, &vb),
            );
        }
    }
}
