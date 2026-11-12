//! Statistics of the pipeline gates. Every plan is paired (the same prompts before and
//! after the patch), so a binary pillar gets Newcombe's paired interval (method 10, the
//! one of `bench refusal`) at the gate's confidence, with McNemar's exact p beside it; a
//! continuous drift (KL, top-1 shift) gets a percentile bootstrap over prompts with an
//! explicit seed. A gate is `GO` only when its interval clears the threshold, `NO_GO` when
//! the interval lies entirely on the wrong side, `INCONCLUSIVE` otherwise, with the sample
//! size that would be needed if the observed effect held.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::bench::agreement::mcnemar_exact;
use crate::bench::refusal::newcombe_paired_z;
use crate::eval::sampling::Xoshiro256pp;

/// Resamples of every bootstrap interval.
pub const BOOTSTRAP_RESAMPLES: usize = 2000;
/// Largest sample size the power search considers.
const MAX_REQUIRED_N: usize = 10_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Verdict {
    Go,
    NoGo,
    Inconclusive,
}

/// Normal quantile (Acklam's rational approximation, relative error below 1.2e-9).
pub fn normal_quantile(p: f64) -> f64 {
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let tail = |q: f64| {
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    const LOW: f64 = 0.024_25;
    if p < LOW {
        tail((-2.0 * p.ln()).sqrt())
    } else if p > 1.0 - LOW {
        -tail((-2.0 * (1.0 - p).ln()).sqrt())
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    }
}

/// `z` of a two-sided interval at `confidence` (0.95 -> 1.96).
pub fn z_two_sided(confidence: f64) -> f64 {
    normal_quantile(1.0 - (1.0 - confidence) / 2.0)
}

/// Paired binary outcomes, "success" meaning the good outcome of the pillar (a refusal kept,
/// a prompt unblocked, a correct answer). `after` is the patched model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct PairedCounts {
    /// Succès des deux côtés.
    pub both: usize,
    /// Succès seulement après le patch.
    pub only_after: usize,
    /// Succès seulement avant le patch.
    pub only_before: usize,
    /// Échec des deux côtés.
    pub neither: usize,
}

impl PairedCounts {
    pub fn from_pairs(before: &[bool], after: &[bool]) -> Self {
        let mut c = Self::default();
        for (b, a) in before.iter().zip(after) {
            match (*b, *a) {
                (true, true) => c.both += 1,
                (false, true) => c.only_after += 1,
                (true, false) => c.only_before += 1,
                (false, false) => c.neither += 1,
            }
        }
        c
    }

    pub fn n(&self) -> usize {
        self.both + self.only_after + self.only_before + self.neither
    }

    /// Success rate after minus before.
    pub fn difference(&self) -> f64 {
        match self.n() {
            0 => 0.0,
            n => (self.only_after as f64 - self.only_before as f64) / n as f64,
        }
    }

    pub fn interval(&self, confidence: f64) -> (f64, f64) {
        newcombe_paired_z(
            self.both,
            self.only_after,
            self.only_before,
            self.neither,
            z_two_sided(confidence),
        )
    }

    /// The same proportions over `n` pairs (largest remainders, so the cells sum to `n`).
    fn scaled(&self, n: usize) -> Self {
        let cells = [self.both, self.only_after, self.only_before, self.neither];
        let total = self.n().max(1) as f64;
        let exact: Vec<f64> = cells.iter().map(|c| *c as f64 * n as f64 / total).collect();
        let mut out: Vec<usize> = exact.iter().map(|x| x.floor() as usize).collect();
        let mut order: Vec<usize> = (0..4).collect();
        order.sort_by(|a, b| {
            let (fa, fb) = (exact[*a] - exact[*a].floor(), exact[*b] - exact[*b].floor());
            fb.total_cmp(&fa).then(a.cmp(b))
        });
        let missing = n - out.iter().sum::<usize>();
        for i in order.into_iter().take(missing) {
            out[i] += 1;
        }
        Self {
            both: out[0],
            only_after: out[1],
            only_before: out[2],
            neither: out[3],
        }
    }
}

/// A gate as configured, the threshold already oriented: the gate passes when the lower
/// bound of the difference interval is above `threshold`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryGate {
    pub threshold: f64,
    pub confidence: f64,
}

/// What a gate concluded and why.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GateResult {
    pub verdict: Verdict,
    /// Effet observé : différence de taux de succès (après − avant) ou dérive moyenne.
    pub estimate: f64,
    /// Borne basse de l'intervalle bilatéral au niveau de confiance de la porte.
    pub ci_low: f64,
    /// Borne haute de l'intervalle.
    pub ci_high: f64,
    /// Seuil que l'intervalle doit franchir.
    pub threshold: f64,
    pub confidence: f64,
    /// Méthode de l'intervalle.
    pub method: String,
    /// Prompts (paires) mesurés.
    pub n: usize,
    /// Taille estimée pour conclure si l'effet observé se maintient ; null si GO, NO_GO,
    /// ou si l'effet observé ne peut pas franchir le seuil.
    pub n_required: Option<usize>,
    /// McNemar exact (pillers binaires) ; null pour une dérive continue.
    pub mcnemar_p: Option<f64>,
    /// Raison en clair.
    pub reason: String,
}

/// Lower bound above `threshold`: GO; upper bound at or below it: NO_GO; else INCONCLUSIVE.
pub fn binary_gate(counts: &PairedCounts, gate: BinaryGate) -> GateResult {
    let (low, high) = counts.interval(gate.confidence);
    let estimate = counts.difference();
    let verdict = if counts.n() > 0 && low > gate.threshold {
        Verdict::Go
    } else if counts.n() > 0 && high <= gate.threshold {
        Verdict::NoGo
    } else {
        Verdict::Inconclusive
    };
    let n_required = (verdict == Verdict::Inconclusive)
        .then(|| required_pairs(counts, gate))
        .flatten();
    let reason = match verdict {
        Verdict::Go => format!("borne basse {low:.4} > seuil {:.4}", gate.threshold),
        Verdict::NoGo => format!("borne haute {high:.4} ≤ seuil {:.4}", gate.threshold),
        Verdict::Inconclusive => match n_required {
            Some(n) => format!(
                "intervalle [{low:.4} ; {high:.4}] contient le seuil {:.4} : n = {} insuffisant, \
                 environ {n} paires nécessaires si l'effet observé se maintient",
                gate.threshold,
                counts.n()
            ),
            None => format!(
                "intervalle [{low:.4} ; {high:.4}] contient le seuil {:.4}, et l'effet observé \
                 ({estimate:.4}) ne le franchit pas : aucune taille d'échantillon ne suffirait",
                gate.threshold
            ),
        },
    };
    GateResult {
        verdict,
        estimate,
        ci_low: low,
        ci_high: high,
        threshold: gate.threshold,
        confidence: gate.confidence,
        method: "newcombe_paired".into(),
        n: counts.n(),
        n_required,
        mcnemar_p: Some(mcnemar_exact(counts.only_after, counts.only_before)),
        reason,
    }
}

/// Smallest number of pairs whose interval clears the threshold with the observed
/// proportions; `None` when the observed difference itself does not clear it.
pub fn required_pairs(counts: &PairedCounts, gate: BinaryGate) -> Option<usize> {
    if counts.n() == 0 || counts.difference() <= gate.threshold {
        return None;
    }
    let clears = |n: usize| counts.scaled(n).interval(gate.confidence).0 > gate.threshold;
    let mut high = counts.n().max(1);
    while !clears(high) {
        high = high.checked_mul(2)?;
        if high > MAX_REQUIRED_N {
            return None;
        }
    }
    let mut low = counts.n();
    while high - low > 1 {
        let mid = low + (high - low) / 2;
        match clears(mid) {
            true => high = mid,
            false => low = mid,
        }
    }
    Some(high)
}

/// Smallest n from which a patch that changes no answer is GO when the base rate is 0 or 1:
/// no discordant pair and every prompt on the same side (all success or all failure). Other
/// base rates conclude earlier (see `lowest_pairs`) or later.
pub fn minimum_pairs(gate: BinaryGate) -> Option<usize> {
    // A superiority gate needs discordant pairs: no a-priori bound (the result is None).
    let perfect = PairedCounts {
        both: 1,
        ..PairedCounts::default()
    };
    required_pairs(&perfect, gate)
}

/// Smallest n at which a patch that changes no answer can be GO at all: no discordant pair
/// and a base rate of one half, the narrowest interval of such tables (phi is closest to 1,
/// the two Wilson half-widths cancel). Below it the gate is INCONCLUSIVE unless the patch
/// brings a net gain.
pub fn lowest_pairs(gate: BinaryGate) -> Option<usize> {
    let ceiling = minimum_pairs(gate)?;
    (1..=ceiling).find(|n| {
        let half = PairedCounts {
            both: n / 2,
            neither: n - n / 2,
            ..PairedCounts::default()
        };
        half.interval(gate.confidence).0 > gate.threshold
    })
}

/// Percentile bootstrap (nearest rank) of the mean of `values`, resampling prompts with
/// the seeded stream: the same values and seed give the same interval.
pub fn bootstrap_mean(values: &[f64], confidence: f64, seed: u64) -> (f64, f64) {
    let n = values.len();
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut rng = Xoshiro256pp::from_seed(seed);
    let mut means: Vec<f64> = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| {
            let mut sum = 0.0;
            for _ in 0..n {
                sum += values[((rng.next_f64() * n as f64) as usize).min(n - 1)];
            }
            sum / n as f64
        })
        .collect();
    means.sort_by(f64::total_cmp);
    let tail = (1.0 - confidence) / 2.0;
    // The epsilon keeps 0.025 * 2000 at rank 50, not 51, through float rounding.
    let rank = |q: f64| {
        ((q * BOOTSTRAP_RESAMPLES as f64 - 1e-9).ceil() as usize).clamp(1, BOOTSTRAP_RESAMPLES) - 1
    };
    (means[rank(tail)], means[rank(1.0 - tail)])
}

/// A drift gate: the mean per-prompt drift must stay at or below `max`. An infinite drift
/// (a token made impossible) fails it outright.
pub fn drift_gate(values: &[f64], max: f64, confidence: f64, seed: u64) -> GateResult {
    let n = values.len();
    let base = |verdict, estimate, low, high, n_required, reason: String| GateResult {
        verdict,
        estimate,
        ci_low: low,
        ci_high: high,
        threshold: max,
        confidence,
        method: "bootstrap_prompts".into(),
        n,
        n_required,
        mcnemar_p: None,
        reason,
    };
    if values.iter().any(|v| !v.is_finite()) {
        return base(
            Verdict::NoGo,
            f64::MAX,
            f64::MAX,
            f64::MAX,
            None,
            "dérive infinie sur au moins un prompt (token rendu impossible)".into(),
        );
    }
    if n == 0 {
        return base(
            Verdict::Inconclusive,
            0.0,
            0.0,
            0.0,
            None,
            "aucun prompt mesuré".into(),
        );
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let (low, high) = bootstrap_mean(values, confidence, seed);
    if high <= max {
        return base(
            Verdict::Go,
            mean,
            low,
            high,
            None,
            format!("borne haute {high:.6} ≤ dérive max {max}"),
        );
    }
    if low > max {
        return base(
            Verdict::NoGo,
            mean,
            low,
            high,
            None,
            format!("borne basse {low:.6} > dérive max {max}"),
        );
    }
    // The half-width shrinks as 1 / sqrt(n).
    let n_required = (mean < max).then(|| {
        let ratio = (high - mean) / (max - mean);
        (n as f64 * ratio * ratio).ceil() as usize
    });
    let reason = match n_required {
        Some(r) => format!(
            "intervalle [{low:.6} ; {high:.6}] contient la dérive max {max} : environ {r} \
             prompts nécessaires si la dérive observée se maintient"
        ),
        None => format!(
            "intervalle [{low:.6} ; {high:.6}] contient la dérive max {max}, et la dérive \
             observée ({mean:.6}) la dépasse"
        ),
    };
    base(Verdict::Inconclusive, mean, low, high, n_required, reason)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::score::Z_95;

    #[test]
    fn quantiles_of_the_normal() {
        assert!((z_two_sided(0.95) - Z_95).abs() < 1e-8);
        assert!((z_two_sided(0.90) - 1.644_853_626_951_472).abs() < 1e-8);
        assert!((z_two_sided(0.99) - 2.575_829_303_548_901).abs() < 1e-8);
        assert!((normal_quantile(0.001) + 3.090_232_306_167_813).abs() < 1e-8);
    }

    #[test]
    fn newcombe_at_another_confidence_matches_a_hand_computation() {
        // Armitage and Berry's table (both 20, after only 12, before only 2, neither 16).
        // At 90 % (z = 1.644854): Wilson(32/50) = [0.523823, 0.741803], Wilson(22/50) =
        // [0.330572, 0.555588]; e h − f g = 296 > n / 2, phi = (296 − 25) / sqrt(32·18·22·28)
        // = 0.454954; d = 0.2, so low = 0.2 − sqrt(0.116177² − 2 phi 0.116177 · 0.115588
        // + 0.115588²) = 0.079009 and high = 0.310462.
        let c = PairedCounts {
            both: 20,
            only_after: 12,
            only_before: 2,
            neither: 16,
        };
        let (low, high) = c.interval(0.90);
        assert!((low - 0.079_009_352_7).abs() < 1e-7, "{low}");
        assert!((high - 0.310_462_056).abs() < 1e-7, "{high}");
        // At 95 % it is the interval of `bench refusal` (published 0.056156 to 0.329207).
        let (low, high) = c.interval(0.95);
        assert!((low - 0.056_156).abs() < 1e-6 && (high - 0.329_207).abs() < 1e-6);
        // McNemar exact on (12, 2): 2 (C(14,0) + C(14,1) + C(14,2)) / 2^14 = 212 / 16384.
        let r = binary_gate(
            &c,
            BinaryGate {
                threshold: -0.05,
                confidence: 0.95,
            },
        );
        assert!((r.mcnemar_p.unwrap() - 212.0 / 16384.0).abs() < 1e-12);
    }

    #[test]
    fn non_inferiority_around_the_margin() {
        // 200 pairs, 4 lost and 4 gained: d = 0, interval about ±0.03.
        let even = PairedCounts {
            both: 150,
            only_after: 4,
            only_before: 4,
            neither: 42,
        };
        let (low, _) = even.interval(0.95);
        let gate = |margin: f64| BinaryGate {
            threshold: -margin,
            confidence: 0.95,
        };
        assert_eq!(binary_gate(&even, gate(-low + 0.001)).verdict, Verdict::Go);
        let tight = binary_gate(&even, gate(-low - 0.001));
        assert_eq!(tight.verdict, Verdict::Inconclusive);
        assert!(tight.n_required.unwrap() > 200, "{tight:?}");
        // Clearly worse than the margin: NO_GO.
        let worse = PairedCounts {
            both: 100,
            only_after: 0,
            only_before: 60,
            neither: 40,
        };
        assert_eq!(binary_gate(&worse, gate(0.05)).verdict, Verdict::NoGo);
    }

    #[test]
    fn small_samples_are_inconclusive_with_the_n_needed() {
        // 10 perfect pairs cannot show non-inferiority at 5 points: Wilson alone is wider.
        let perfect = PairedCounts {
            both: 10,
            ..PairedCounts::default()
        };
        let gate = BinaryGate {
            threshold: -0.05,
            confidence: 0.95,
        };
        let r = binary_gate(&perfect, gate);
        assert_eq!(r.verdict, Verdict::Inconclusive);
        let needed = r.n_required.unwrap();
        assert_eq!(Some(needed), minimum_pairs(gate));
        let enough = PairedCounts {
            both: needed,
            ..PairedCounts::default()
        };
        assert_eq!(binary_gate(&enough, gate).verdict, Verdict::Go);
        let short = PairedCounts {
            both: needed - 1,
            ..PairedCounts::default()
        };
        assert_eq!(binary_gate(&short, gate).verdict, Verdict::Inconclusive);
        // An observed loss beyond the margin: no n would do.
        let losing = PairedCounts {
            both: 5,
            only_before: 2,
            neither: 3,
            ..PairedCounts::default()
        };
        let r = binary_gate(&losing, gate);
        assert_eq!(r.verdict, Verdict::Inconclusive);
        assert_eq!(r.n_required, None);
    }

    #[test]
    fn no_table_without_a_discordant_pair_is_go_below_the_lowest_n() {
        // Half the prompts succeeding on both sides is a narrower interval than all of them:
        // at a margin of 0.15 and 90 %, such a table is GO from n = 10, all successes from 16.
        for (margin, confidence, lowest, at_rate_0_or_1) in [
            (0.15, 0.90, 10, 16),
            (0.4, 0.90, 4, 5),
            (0.02, 0.95, 97, 189),
        ] {
            let gate = BinaryGate {
                threshold: -margin,
                confidence,
            };
            let go = |n: usize, both: usize| {
                let counts = PairedCounts {
                    both,
                    neither: n - both,
                    ..PairedCounts::default()
                };
                binary_gate(&counts, gate).verdict == Verdict::Go
            };
            let announced = lowest_pairs(gate).unwrap();
            assert_eq!(announced, lowest, "margin {margin}");
            for n in 1..announced {
                assert!((0..=n).all(|both| !go(n, both)), "n = {n} < {announced}");
            }
            assert!((0..=announced).any(|both| go(announced, both)));
            // `minimum_pairs` is the size for a saturated base rate (all on one side).
            let saturated = minimum_pairs(gate).unwrap();
            assert_eq!(saturated, at_rate_0_or_1, "margin {margin}");
            assert!(go(saturated, saturated) && go(saturated, 0));
            assert!(!go(saturated - 1, saturated - 1) && !go(saturated - 1, 0));
        }
        let superiority = BinaryGate {
            threshold: 0.05,
            confidence: 0.9,
        };
        assert_eq!(lowest_pairs(superiority), None);
    }

    #[test]
    fn drift_gates_bootstrap_over_prompts() {
        let values: Vec<f64> = (0..40).map(|i| 0.01 * (i % 5) as f64).collect();
        let a = drift_gate(&values, 0.1, 0.95, 7);
        assert_eq!(a, drift_gate(&values, 0.1, 0.95, 7), "seeded");
        assert_eq!(a.verdict, Verdict::Go);
        assert!((a.estimate - 0.02).abs() < 1e-12);
        assert_eq!(drift_gate(&values, 0.001, 0.95, 7).verdict, Verdict::NoGo);
        let straddle = drift_gate(&values, 0.022, 0.95, 7);
        assert_eq!(straddle.verdict, Verdict::Inconclusive);
        assert!(straddle.n_required.unwrap() > 40);
        let mut infinite = values.clone();
        infinite.push(f64::INFINITY);
        assert_eq!(drift_gate(&infinite, 1e9, 0.95, 7).verdict, Verdict::NoGo);
        // Identical values: a zero-width interval.
        assert_eq!(bootstrap_mean(&[0.5; 9], 0.95, 1), (0.5, 0.5));
    }
}
