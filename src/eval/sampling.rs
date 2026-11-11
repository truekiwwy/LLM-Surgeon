//! Reproducible sampling of the native generator.
//!
//! The chain, in the order of llama.cpp's default sampler chain (without claiming its
//! bits):
//!
//! ```text
//! logits -> repetition penalty (every token already in the sequence, prompt included:
//!           l / p when l > 0, l * p otherwise)
//!        -> T = 0: argmax (the smallest index on ties), stop; the historical greedy path
//!        -> sort by logit (stable: the smaller index first on ties)
//!        -> top-k (k > 0)  -> softmax at T = 1, f64
//!        -> top-p: the shortest prefix whose cumulated probability reaches p
//!        -> min-p: keep p_i >= min_p * p_max
//!        -> temperature: softmax of l / T over what is left, f64
//!        -> draw: u in [0, 1) from the sequence's own stream, first index whose
//!           cumulated probability exceeds u
//! ```
//!
//! Randomness: xoshiro256++ seeded by SplitMix64, one stream per sequence derived from
//! `(seed, sequence_id)` and consumed once per sampled token. A sequence therefore draws
//! the same numbers whatever else is decoded with it (batch composition, order, jobs);
//! its tokens are the same as long as its logits are.

use serde::{Deserialize, Serialize};

/// Sampling of one sequence. The default is greedy decoding.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, schemars::JsonSchema)]
pub struct SamplingParams {
    /// 0: greedy.
    #[serde(serialize_with = "short_f32")]
    pub temperature: f32,
    /// 0: off.
    pub top_k: usize,
    /// 1: off.
    #[serde(serialize_with = "short_f32")]
    pub top_p: f32,
    /// 0: off.
    #[serde(serialize_with = "short_f32")]
    pub min_p: f32,
    /// 1: off.
    #[serde(serialize_with = "short_f32")]
    pub repetition_penalty: f32,
    pub seed: u64,
}

/// The shortest decimal of the f32 (0.9), not of its f64 widening (0.8999999761581421),
/// which is what a JSON value of an f32 would print.
fn short_f32<S: serde::Serializer>(x: &f32, s: S) -> std::result::Result<S::Ok, S::Error> {
    s.serialize_f64(x.to_string().parse().unwrap_or(f64::from(*x)))
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 42,
        }
    }
}

impl SamplingParams {
    pub fn is_greedy(&self) -> bool {
        self.temperature == 0.0
    }

    /// Every bound checked, with the parameter named.
    pub fn validate(&self) -> Result<(), String> {
        let bad = |what: &str| Err(format!("paramètre de sampling invalide : {what}"));
        if !(self.temperature.is_finite() && self.temperature >= 0.0) {
            return bad("temperature doit être finie et >= 0");
        }
        if !(self.top_p.is_finite() && self.top_p > 0.0 && self.top_p <= 1.0) {
            return bad("top_p doit être dans ]0, 1]");
        }
        if !(self.min_p.is_finite() && (0.0..1.0).contains(&self.min_p)) {
            return bad("min_p doit être dans [0, 1[");
        }
        if !(self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0) {
            return bad("repetition_penalty doit être finie et > 0");
        }
        Ok(())
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++ (Blackman and Vigna, 2019).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xoshiro256pp {
    s: [u64; 4],
}

impl Xoshiro256pp {
    /// The state from four SplitMix64 outputs, as the authors recommend.
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        Self {
            s: [(); 4].map(|()| splitmix64(&mut sm)),
        }
    }

    /// The stream of sequence `sequence_id` under `seed`.
    pub fn for_sequence(seed: u64, sequence_id: u64) -> Self {
        let mut sm = seed;
        let mixed = splitmix64(&mut sm) ^ sequence_id.wrapping_mul(0xD1B5_4A32_D192_ED03);
        Self::from_seed(mixed)
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)`, 53 bits.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// The 32-bit seed of sequence `sequence_id` for a llama.cpp backend (one seed per
/// request): the first output of the sequence's stream.
pub fn sequence_seed(seed: u64, sequence_id: u64) -> u32 {
    (Xoshiro256pp::for_sequence(seed, sequence_id).next_u64() >> 32) as u32
}

/// Index of the largest value, the smallest index on ties; `None` when empty or a value
/// is not finite.
pub fn argmax(logits: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &logit) in logits.iter().enumerate() {
        if !logit.is_finite() {
            return None;
        }
        if best.is_none_or(|(_, top)| logit > top) {
            best = Some((i, logit));
        }
    }
    best.map(|(i, _)| i)
}

/// `logits` with the repetition penalty applied to every token of `history`.
fn penalized(logits: &[f32], penalty: f32, history: &[u32]) -> Vec<f32> {
    let mut out = logits.to_vec();
    if penalty == 1.0 {
        return out;
    }
    let mut seen = vec![false; out.len()];
    for &id in history {
        let Some(slot) = seen.get_mut(id as usize) else {
            continue;
        };
        if std::mem::replace(slot, true) {
            continue;
        }
        let l = &mut out[id as usize];
        *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
    }
    out
}

/// `exp(l_i - max) / sum` in f64, in the order of `logits`.
fn softmax(logits: impl Iterator<Item = f64> + Clone) -> Vec<f64> {
    let max = logits.clone().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Candidates after penalty, top-k, top-p and min-p, with their probabilities at the
/// requested temperature (sorted by logit, ties by index). The draw is [`sample`].
pub fn distribution(logits: &[f32], params: &SamplingParams, history: &[u32]) -> Vec<(u32, f64)> {
    let logits = penalized(logits, params.repetition_penalty, history);
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    if params.top_k > 0 {
        order.truncate(params.top_k);
    }
    let probs = softmax(order.iter().map(|&i| f64::from(logits[i])));
    let mut keep = order.len();
    if params.top_p < 1.0 {
        let mut cumulated = 0.0;
        for (n, p) in probs.iter().enumerate() {
            cumulated += p;
            if cumulated >= f64::from(params.top_p) {
                keep = n + 1;
                break;
            }
        }
    }
    if params.min_p > 0.0 {
        let floor = f64::from(params.min_p) * probs[0];
        keep = keep.min(probs.iter().take_while(|p| **p >= floor).count().max(1));
    }
    order.truncate(keep);
    let t = f64::from(params.temperature);
    let scaled = softmax(order.iter().map(|&i| f64::from(logits[i]) / t));
    order.iter().map(|&i| i as u32).zip(scaled).collect()
}

/// The next token of a sequence: greedy at `T = 0`, else one draw from `rng`.
pub fn sample(
    logits: &[f32],
    params: &SamplingParams,
    history: &[u32],
    rng: &mut Xoshiro256pp,
) -> Option<u32> {
    if logits.iter().any(|l| !l.is_finite()) || logits.is_empty() {
        return None;
    }
    if params.is_greedy() {
        return argmax(&penalized(logits, params.repetition_penalty, history)).map(|i| i as u32);
    }
    let candidates = distribution(logits, params, history);
    let u = rng.next_f64();
    let mut cumulated = 0.0;
    for (id, p) in &candidates {
        cumulated += p;
        if u < cumulated {
            return Some(*id);
        }
    }
    // Rounding left the sum a hair under 1 and u above it: the last candidate.
    candidates.last().map(|(id, _)| *id)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(t: f32) -> SamplingParams {
        SamplingParams {
            temperature: t,
            ..SamplingParams::default()
        }
    }

    #[test]
    fn xoshiro_matches_the_reference_sequence() {
        // Reference: the C implementation seeded with s = {1, 2, 3, 4}.
        let mut r = Xoshiro256pp { s: [1, 2, 3, 4] };
        let first: Vec<u64> = (0..4).map(|_| r.next_u64()).collect();
        assert_eq!(
            first,
            [41943041, 58720359, 3588806011781223, 3591011842654386]
        );
        // SplitMix64 from 0: the published first output.
        let mut sm = 0;
        assert_eq!(splitmix64(&mut sm), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn streams_depend_on_seed_and_sequence_only() {
        let a: Vec<u64> = (0..3)
            .map({
                let mut r = Xoshiro256pp::for_sequence(7, 3);
                move |_| r.next_u64()
            })
            .collect();
        let mut again = Xoshiro256pp::for_sequence(7, 3);
        assert_eq!(a, (0..3).map(|_| again.next_u64()).collect::<Vec<_>>());
        assert_ne!(
            Xoshiro256pp::for_sequence(7, 4),
            Xoshiro256pp::for_sequence(7, 3)
        );
        assert_ne!(
            Xoshiro256pp::for_sequence(8, 3),
            Xoshiro256pp::for_sequence(7, 3)
        );
    }

    #[test]
    fn zero_temperature_is_the_historical_argmax() {
        let logits = [0.5, 2.0, 2.0, -1.0];
        let mut rng = Xoshiro256pp::from_seed(1);
        assert_eq!(sample(&logits, &params(0.0), &[], &mut rng), Some(1));
        assert_eq!(rng, Xoshiro256pp::from_seed(1), "greedy draws nothing");
        // A penalty on token 1 (positive logit divided) hands the argmax to token 2.
        let penalized = SamplingParams {
            repetition_penalty: 2.0,
            ..params(0.0)
        };
        assert_eq!(sample(&logits, &penalized, &[1, 1], &mut rng), Some(2));
        assert_eq!(sample(&[f32::NAN], &params(1.0), &[], &mut rng), None);
    }

    #[test]
    fn filters_keep_exactly_the_reference_sets() {
        // Probabilities at T = 1 of ln([4, 3, 2, 1]) are [0.4, 0.3, 0.2, 0.1].
        let logits: Vec<f32> = [4.0f32, 3.0, 2.0, 1.0].iter().map(|x| x.ln()).collect();
        let ids = |p: SamplingParams| -> Vec<u32> {
            distribution(&logits, &p, &[])
                .iter()
                .map(|(i, _)| *i)
                .collect()
        };
        assert_eq!(ids(params(1.0)), [0, 1, 2, 3]);
        assert_eq!(
            ids(SamplingParams {
                top_k: 2,
                ..params(1.0)
            }),
            [0, 1]
        );
        // 0.4 + 0.3 = 0.7 reaches 0.69: two tokens; 0.71 needs a third.
        assert_eq!(
            ids(SamplingParams {
                top_p: 0.69,
                ..params(1.0)
            }),
            [0, 1]
        );
        assert_eq!(
            ids(SamplingParams {
                top_p: 0.71,
                ..params(1.0)
            }),
            [0, 1, 2]
        );
        // min_p 0.49: p >= 0.196 keeps three; 0.51 (p >= 0.204) keeps two. Away from the
        // boundaries: 0.2 against 0.5 * 0.4 is not exact in floating point.
        assert_eq!(
            ids(SamplingParams {
                min_p: 0.49,
                ..params(1.0)
            }),
            [0, 1, 2]
        );
        assert_eq!(
            ids(SamplingParams {
                min_p: 0.51,
                ..params(1.0)
            }),
            [0, 1]
        );
        // Temperature only reshapes what is left: T = 0.5 squares the odds.
        let d = distribution(
            &logits,
            &SamplingParams {
                top_k: 2,
                ..params(0.5)
            },
            &[],
        );
        // The logits are ln() rounded to f32: about 1e-8 off.
        assert!((d[0].1 - 16.0 / 25.0).abs() < 1e-6, "{d:?}");
        assert!((d[0].1 + d[1].1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn frequencies_follow_the_exact_probabilities() {
        // 20 000 draws from [0.4, 0.3, 0.2, 0.1] at a fixed seed: every frequency within
        // 0.012 of its probability (more than 4 standard deviations), checked once and
        // deterministic since.
        let logits: Vec<f32> = [4.0f32, 3.0, 2.0, 1.0].iter().map(|x| x.ln()).collect();
        let mut rng = Xoshiro256pp::for_sequence(42, 0);
        let mut counts = [0usize; 4];
        let n = 20_000;
        for _ in 0..n {
            counts[sample(&logits, &params(1.0), &[], &mut rng).unwrap() as usize] += 1;
        }
        for (count, p) in counts.iter().zip([0.4, 0.3, 0.2, 0.1]) {
            let f = *count as f64 / n as f64;
            assert!((f - p).abs() < 0.012, "{counts:?}");
        }
    }

    #[test]
    fn invalid_parameters_are_named() {
        for (p, what) in [
            (
                SamplingParams {
                    temperature: -1.0,
                    ..params(1.0)
                },
                "temperature",
            ),
            (
                SamplingParams {
                    top_p: 0.0,
                    ..params(1.0)
                },
                "top_p",
            ),
            (
                SamplingParams {
                    min_p: 1.0,
                    ..params(1.0)
                },
                "min_p",
            ),
            (
                SamplingParams {
                    repetition_penalty: 0.0,
                    ..params(1.0)
                },
                "repetition_penalty",
            ),
        ] {
            assert!(p.validate().unwrap_err().contains(what));
        }
        assert!(SamplingParams::default().validate().is_ok());
    }
}
