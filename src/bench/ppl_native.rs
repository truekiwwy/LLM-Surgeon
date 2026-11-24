use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use rayon::prelude::*;

use super::error::{BenchError, Result};
use super::native::NativeGenerator;
use super::ppl_parse::PplEstimate;
use super::Neumaier;
use crate::extract::tokenize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Window {
    pub begin: usize,
    pub end: usize,
    pub scored_from: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowPlan {
    pub ctx: usize,
    pub stride: usize,
    pub max_windows: Option<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    pub estimate: PplEstimate,
    pub scored: usize,
}

pub fn windows(tokens: usize, ctx: usize, stride: usize) -> Vec<Window> {
    let mut plan = Vec::new();
    let (mut begin, mut scored_to) = (0, 1);
    while scored_to < tokens && ctx > 1 && stride > 0 {
        let end = (begin + ctx).min(tokens);
        plan.push(Window {
            begin,
            end,
            scored_from: scored_to.max(begin + 1),
        });
        scored_to = end;
        begin += stride;
    }
    plan
}

pub fn token_nll(logits: &[f32], target: usize) -> Option<f64> {
    let max = logits
        .iter()
        .map(|&x| f64::from(x))
        .fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = logits.iter().map(|&x| (f64::from(x) - max).exp()).sum();
    let nll = max + sum.ln() - f64::from(*logits.get(target)?);
    nll.is_finite().then_some(nll)
}

pub fn estimate(nll: &[f64]) -> Option<PplEstimate> {
    let (mut sum, mut squares) = (Neumaier::default(), Neumaier::default());
    for &x in nll {
        sum.add(x);
        squares.add(x * x);
    }
    let n = nll.len() as f64;
    let mean = sum.total() / n;
    let ppl = mean.exp();
    let variance = squares.total() / n - mean * mean;
    let err = (nll.len() > 1 && variance > 0.0).then(|| ppl * (variance / (n - 1.0)).sqrt());
    (!nll.is_empty() && ppl.is_finite()).then_some(PplEstimate { ppl, err })
}

pub fn measure(
    generator: &NativeGenerator,
    text: &str,
    plan: WindowPlan,
    deadline: Option<Instant>,
    cancel: &AtomicBool,
    on_window: &mut dyn FnMut(usize, Option<usize>),
) -> Result<Measured> {
    let started = Instant::now();
    let ids = tokenize(generator.tokenizer(), text, usize::MAX)?.ids;
    let mut windows = windows(ids.len(), plan.ctx, plan.stride);
    if let Some(max) = plan.max_windows {
        windows.truncate(max);
    }
    if windows.is_empty() {
        return Err(BenchError::InvalidArgument(format!(
            "texte de {} token(s) : il en faut au moins 2",
            ids.len()
        )));
    }
    let mut nll = Vec::new();
    for (done, w) in windows.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            return Err(BenchError::Cancelled);
        }
        if deadline.is_some_and(|d| Instant::now() > d) {
            return Err(BenchError::Timeout {
                after_ms: started.elapsed().as_millis(),
            });
        }
        let logits = generator.teacher_logits(&ids[w.begin..w.end])?;
        let vocab = logits.len() / (w.end - w.begin);
        let scored: Vec<Option<f64>> = (w.scored_from..w.end)
            .into_par_iter()
            .map(|p| {
                let row = p - 1 - w.begin;
                token_nll(&logits[row * vocab..(row + 1) * vocab], ids[p] as usize)
            })
            .collect();
        for (position, value) in (w.scored_from..w.end).zip(scored) {
            nll.push(value.ok_or_else(|| {
                BenchError::Compute(format!(
                    "log-probabilité non finie au token {position} : modèle numériquement cassé"
                ))
            })?);
        }
        on_window(done + 1, Some(windows.len()));
    }
    let estimate =
        estimate(&nll).ok_or_else(|| BenchError::Compute("perplexité non finie".into()))?;
    Ok(Measured {
        estimate,
        scored: nll.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scored(plan: &[Window]) -> Vec<usize> {
        plan.iter().flat_map(|w| w.scored_from..w.end).collect()
    }

    #[test]
    fn overlapping_windows_score_every_token_once_with_its_context() {
        let plan = windows(10, 4, 2);
        assert_eq!(
            plan,
            [
                Window {
                    begin: 0,
                    end: 4,
                    scored_from: 1
                },
                Window {
                    begin: 2,
                    end: 6,
                    scored_from: 4
                },
                Window {
                    begin: 4,
                    end: 8,
                    scored_from: 6
                },
                Window {
                    begin: 6,
                    end: 10,
                    scored_from: 8
                },
            ]
        );
        assert_eq!(scored(&plan), (1..10).collect::<Vec<_>>());
        for w in &plan {
            assert!(w.end - w.begin <= 4 && w.scored_from > w.begin);
        }
    }

    #[test]
    fn disjoint_windows_and_degenerate_inputs() {
        let plan = windows(9, 4, 4);
        assert_eq!(scored(&plan), [1, 2, 3, 5, 6, 7]);
        assert_eq!(windows(3, 512, 256).len(), 1);
        assert_eq!(scored(&windows(3, 512, 256)), [1, 2]);
        assert!(windows(1, 512, 256).is_empty());
        assert!(windows(0, 512, 256).is_empty());
        assert!(windows(10, 1, 1).is_empty());
        assert!(windows(10, 4, 0).is_empty());
    }

    #[test]
    fn uniform_logits_have_the_vocabulary_as_perplexity() {
        let vocab = 1000;
        let logits = vec![3.25f32; vocab];
        let nll = token_nll(&logits, 7).unwrap();
        assert!((nll - (vocab as f64).ln()).abs() < 1e-12, "{nll}");
        let estimate = estimate(&[nll; 50]).unwrap();
        assert!(
            (estimate.ppl - vocab as f64).abs() < 1e-9,
            "{}",
            estimate.ppl
        );
        assert_eq!(estimate.err, None, "no spread, no uncertainty");
    }

    #[test]
    fn known_distribution_gives_the_exact_geometric_mean() {
        let logits = [0.5f32.ln(), 0.25f32.ln(), 0.25f32.ln()];
        let a = token_nll(&logits, 0).unwrap();
        let b = token_nll(&logits, 1).unwrap();
        assert!((a - 2f64.ln()).abs() < 1e-6 && (b - 4f64.ln()).abs() < 1e-6);
        let estimate = estimate(&[a, b]).unwrap();
        assert!(
            (estimate.ppl - 8f64.sqrt()).abs() < 1e-5,
            "{}",
            estimate.ppl
        );
        let mean = (a + b) / 2.0;
        let variance = (a * a + b * b) / 2.0 - mean * mean;
        let expected = estimate.ppl * variance.sqrt();
        assert!((estimate.err.unwrap() - expected).abs() < 1e-9);
    }

    #[test]
    fn shifted_logits_do_not_change_the_log_probability() {
        let logits = [1.0f32, 2.0, 3.0, 4.0];
        let shifted: Vec<f32> = logits.iter().map(|x| x + 80.0).collect();
        let (a, b) = (
            token_nll(&logits, 2).unwrap(),
            token_nll(&shifted, 2).unwrap(),
        );
        assert!((a - b).abs() < 1e-5, "{a} {b}");
    }

    #[test]
    fn non_finite_inputs_are_refused() {
        assert_eq!(token_nll(&[1.0, f32::NAN], 0), None);
        assert_eq!(token_nll(&[f32::NEG_INFINITY; 3], 0), None);
        assert_eq!(token_nll(&[1.0, f32::NEG_INFINITY], 1), None);
        assert_eq!(token_nll(&[1.0, 2.0], 5), None);
        assert_eq!(estimate(&[]), None);
    }
}
