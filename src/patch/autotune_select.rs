//! Scale selection of `surgeon autotune --refusal-prompts`: after the ridge, the fitted
//! coefficients are tried at several scales c·α, applied in memory, and the smallest scale
//! whose generated answers reach the target refusal rate is written. Generating every
//! candidate is the cost, so it runs in two stages:
//!
//! 1. Screening: the first-token probe (`bench refusal-probe`, one forward pass per prompt)
//!    scores every scale. A scale passes when its probed refusal rate reaches the target
//!    and its mean refusal probability drops below the base model's. The finalists are the
//!    `top_k` smallest passing scales, completed by the lowest probed probabilities when
//!    fewer pass; every other scale is rejected, with its reason.
//! 2. Proof: each finalist is generated (40 tokens, refusal markers, plus the retention set
//!    when one is given). The probe never decides alone.
//!
//! Without the prefilter every candidate is generated: the reference the screening saves
//! generations against. `direction-projection` is never the screen: it over-triggers on
//! sensitive legitimate prompts.

use std::path::PathBuf;
use std::time::Instant;

use schemars::JsonSchema;
use serde::Serialize;

use super::autotune::{AutotuneEvent, Result};
use crate::bench::ref_cache::RefCache;

/// Candidate scales c of the fitted α.
pub const DEFAULT_SCALES: [f64; 6] = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5];
pub const DEFAULT_TARGET_REFUSAL_RATE: f64 = 0.1;
/// Finalists the probe passes to the generation.
pub const DEFAULT_TOP_K: usize = 2;
/// Questions of the retention set generated per finalist.
pub const DEFAULT_RETENTION_LIMIT: usize = 20;
/// Largest scale accepted (the bounds of the fit still apply on top).
pub const MAX_SCALE: f64 = 4.0;

/// `--refusal-prompts` and the options of the selection.
#[derive(Debug, Clone)]
pub struct SelectionRequest {
    /// Prompts the patched model should stop refusing: a file or `builtin:refusals`.
    pub prompts: PathBuf,
    pub target_refusal_rate: f64,
    pub scales: Vec<f64>,
    pub top_k: usize,
    /// false (`--no-probe-prefilter`): every candidate generated.
    pub prefilter: bool,
    /// Capability set generated on the finalists (`bench eval`, native), and its size.
    pub retention: Option<(PathBuf, usize)>,
    /// Where the base model's answers are kept between runs, shared with `bench`.
    pub ref_cache: Option<RefCache>,
}

/// The stage a candidate was measured by.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    Probe,
    Generation,
}

impl Stage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Probe => "probe",
            Self::Generation => "generation",
        }
    }
}

/// The generated answers of one candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct Generated {
    /// Per refusal prompt: refused according to the markers.
    pub refused: Vec<bool>,
    /// Exact-match accuracy on the retention set, when one is configured.
    pub retention: Option<f64>,
    /// The base model only: `hit` or `miss` of the reference cache.
    pub cache: Option<String>,
}

/// The model patched with `scale`·α (0: the base model), measured.
pub trait ScaleEvaluator {
    /// First-token refusal probability of every refusal prompt.
    fn probe(&mut self, scale: f64) -> Result<Vec<f64>>;
    /// Every refusal prompt generated and judged.
    fn generate(&mut self, scale: f64) -> Result<Generated>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct SelectionOptions {
    pub scales: Vec<f64>,
    pub target_refusal_rate: f64,
    pub top_k: usize,
    /// false: no probe, every candidate generated.
    pub prefilter: bool,
    /// Probability above which a probed prompt counts as refused.
    pub threshold: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct CandidateScore {
    /// c : le patch écrit vaut c·α sur chaque couche (0 : le modèle de base).
    pub scale: f64,
    /// Étape 1 : part des prompts dont la probabilité de refus sondée dépasse le seuil.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub probe_refusal_rate: Option<f64>,
    /// Étape 1 : probabilité moyenne que la réponse s'ouvre sur un refus.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_refusal_prob: Option<f64>,
    /// Écartée à l'étape 1, et pourquoi.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rejected: Option<String>,
    /// Étape 2 : taux de refus des réponses générées (marqueurs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_refusal_rate: Option<f64>,
    /// Étape 2 : exactitude sur le jeu de rétention, s'il y en a un.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retention_accuracy: Option<f64>,
}

impl CandidateScore {
    fn new(scale: f64) -> Self {
        Self {
            scale,
            probe_refusal_rate: None,
            mean_refusal_prob: None,
            rejected: None,
            generated_refusal_rate: None,
            retention_accuracy: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct SelectionTiming {
    /// Étape 1 (sondes), base comprise.
    pub probe: u64,
    /// Étape 2 (générations), base comprise.
    pub generation: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct ScaleSelection {
    /// false : --no-probe-prefilter, chaque candidat généré.
    pub prefilter: bool,
    pub target_refusal_rate: f64,
    pub top_k: usize,
    pub probe_threshold: f64,
    /// Prompts de refus.
    pub n_prompts: usize,
    /// Le modèle de base (échelle 0).
    pub baseline: CandidateScore,
    /// Réponses de la base lues dans le cache des références (hit) ou générées (miss).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_cache: Option<String>,
    /// Chaque échelle candidate, dans l'ordre croissant.
    pub candidates: Vec<CandidateScore>,
    /// Candidats évalués par la sonde.
    pub candidates_probed: usize,
    /// Candidats générés en entier, la base non comptée.
    pub candidates_generated: usize,
    /// Générations complètes évitées par la sonde (candidats − générés).
    pub generations_saved: usize,
    /// Plus petite échelle dont les réponses générées atteignent la cible ; null sinon.
    pub selected_scale: Option<f64>,
    pub timing_ms: SelectionTiming,
}

fn rate(flags: impl Iterator<Item = bool>) -> f64 {
    let (mut hits, mut n) = (0usize, 0usize);
    for f in flags {
        hits += usize::from(f);
        n += 1;
    }
    hits as f64 / n.max(1) as f64
}

/// The two stages over `opts.scales` (sorted, deduplicated).
pub fn select_scale(
    eval: &mut dyn ScaleEvaluator,
    opts: &SelectionOptions,
    on_event: &mut dyn FnMut(AutotuneEvent),
) -> Result<ScaleSelection> {
    let target = opts.target_refusal_rate;
    let mut scales = opts.scales.clone();
    scales.sort_by(f64::total_cmp);
    scales.dedup();
    let mut candidates: Vec<CandidateScore> =
        scales.iter().map(|&s| CandidateScore::new(s)).collect();
    let mut baseline = CandidateScore::new(0.0);
    let mut n_prompts = 0;

    let started = Instant::now();
    let finalists: Vec<usize> = match opts.prefilter {
        false => (0..candidates.len()).collect(),
        true => {
            let mut probed = |c: &mut CandidateScore| -> Result<()> {
                let probs = eval.probe(c.scale)?;
                n_prompts = probs.len();
                let r = rate(probs.iter().map(|p| *p > opts.threshold));
                c.probe_refusal_rate = Some(r);
                c.mean_refusal_prob = Some(probs.iter().sum::<f64>() / probs.len().max(1) as f64);
                Ok(())
            };
            probed(&mut baseline)?;
            for c in &mut candidates {
                probed(c)?;
                on_event(AutotuneEvent::Scale {
                    scale: c.scale,
                    stage: Stage::Probe,
                    refusal_rate: c.probe_refusal_rate.unwrap_or_default(),
                });
            }
            let base_prob = baseline.mean_refusal_prob.unwrap_or(f64::INFINITY);
            let passes = |c: &CandidateScore| {
                c.probe_refusal_rate.is_some_and(|r| r <= target)
                    && c.mean_refusal_prob.is_some_and(|p| p < base_prob)
            };
            let mut order: Vec<usize> = (0..candidates.len()).collect();
            order.sort_by(|&a, &b| {
                let (ca, cb) = (&candidates[a], &candidates[b]);
                match (passes(ca), passes(cb)) {
                    (true, true) => ca.scale.total_cmp(&cb.scale),
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    (false, false) => ca
                        .mean_refusal_prob
                        .unwrap_or_default()
                        .total_cmp(&cb.mean_refusal_prob.unwrap_or_default()),
                }
            });
            let k = opts.top_k.min(order.len());
            for &i in &order[k..] {
                let c = &candidates[i];
                let (r, p) = (
                    c.probe_refusal_rate.unwrap_or_default(),
                    c.mean_refusal_prob.unwrap_or_default(),
                );
                candidates[i].rejected = Some(match () {
                    _ if r > target => {
                        format!("taux de refus sondé {r:.3} > cible {target:.3}")
                    }
                    _ if p >= base_prob => format!(
                        "probabilité de refus sondée {p:.3}, sans baisse face à la base ({base_prob:.3})"
                    ),
                    _ => format!("hors des {k} meilleures (plus petite échelle d'abord)"),
                });
            }
            let mut kept = order[..k].to_vec();
            kept.sort_unstable();
            kept
        }
    };
    let probe_ms = started.elapsed().as_millis() as u64;

    let started = Instant::now();
    let base = eval.generate(0.0)?;
    n_prompts = base.refused.len();
    baseline.generated_refusal_rate = Some(rate(base.refused.iter().copied()));
    baseline.retention_accuracy = base.retention;
    for &i in &finalists {
        let got = eval.generate(candidates[i].scale)?;
        let c = &mut candidates[i];
        c.generated_refusal_rate = Some(rate(got.refused.iter().copied()));
        c.retention_accuracy = got.retention;
        on_event(AutotuneEvent::Scale {
            scale: c.scale,
            stage: Stage::Generation,
            refusal_rate: c.generated_refusal_rate.unwrap_or_default(),
        });
    }
    // Finalists are in increasing scale: the first that reaches the target is the smallest.
    let selected_scale = finalists
        .iter()
        .map(|&i| &candidates[i])
        .find(|c| c.generated_refusal_rate.is_some_and(|r| r <= target))
        .map(|c| c.scale);
    let probed = if opts.prefilter { candidates.len() } else { 0 };
    let generated = finalists.len();
    let saved = candidates.len() - generated;
    on_event(AutotuneEvent::Selected {
        probed,
        generated,
        saved,
    });
    Ok(ScaleSelection {
        prefilter: opts.prefilter,
        target_refusal_rate: target,
        top_k: opts.top_k,
        probe_threshold: opts.threshold,
        n_prompts,
        baseline,
        baseline_cache: base.cache,
        candidates,
        candidates_probed: probed,
        candidates_generated: generated,
        generations_saved: saved,
        selected_scale,
        timing_ms: SelectionTiming {
            probe: probe_ms,
            generation: started.elapsed().as_millis() as u64,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model whose refusals fall linearly with the scale: generated refusals reach 0.1
    /// at c = 0.8, the probe sees slightly more (`probe_bias`).
    struct Simulated {
        prompts: usize,
        probe_bias: f64,
        probes: Vec<f64>,
        generations: Vec<f64>,
    }

    impl Simulated {
        fn new(probe_bias: f64) -> Self {
            Self {
                prompts: 20,
                probe_bias,
                probes: Vec::new(),
                generations: Vec::new(),
            }
        }

        /// Refusal probability of prompt i at scale c, spread over the prompts so that the
        /// share above 0.5 is the generated share plus `bias`.
        fn prob(&self, i: usize, scale: f64, bias: f64) -> f64 {
            let spread = (i as f64 + 0.5) / self.prompts as f64;
            (0.5 + (0.9 - scale + bias) - spread).clamp(0.0, 1.0)
        }
    }

    impl ScaleEvaluator for Simulated {
        fn probe(&mut self, scale: f64) -> Result<Vec<f64>> {
            self.probes.push(scale);
            Ok((0..self.prompts)
                .map(|i| self.prob(i, scale, self.probe_bias))
                .collect())
        }

        fn generate(&mut self, scale: f64) -> Result<Generated> {
            self.generations.push(scale);
            let refused_share = (0.9 - scale).clamp(0.0, 1.0);
            let refused = (0..self.prompts)
                .map(|i| (i as f64) < refused_share * self.prompts as f64)
                .collect();
            Ok(Generated {
                refused,
                retention: Some(1.0 - scale / 4.0),
                cache: (scale == 0.0).then(|| "hit".into()),
            })
        }
    }

    fn options(prefilter: bool) -> SelectionOptions {
        SelectionOptions {
            scales: DEFAULT_SCALES.to_vec(),
            target_refusal_rate: 0.1,
            top_k: 2,
            prefilter,
            threshold: 0.5,
        }
    }

    fn run(model: &mut Simulated, opts: &SelectionOptions) -> (ScaleSelection, Vec<String>) {
        let mut events = Vec::new();
        let s = select_scale(model, opts, &mut |e| {
            events.push(match e {
                AutotuneEvent::Scale { scale, stage, .. } => format!("{}:{scale}", stage.as_str()),
                AutotuneEvent::Selected { saved, .. } => format!("saved:{saved}"),
                _ => String::new(),
            })
        })
        .unwrap();
        (s, events)
    }

    #[test]
    fn the_probe_rejects_weak_scales_and_saves_their_generation() {
        let mut model = Simulated::new(0.05);
        let (s, events) = run(&mut model, &options(true));
        // (a) Every scale is probed once, the base too; the weak ones stop at stage 1.
        assert_eq!(model.probes, [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]);
        for c in &s.candidates[..3] {
            let reason = c.rejected.as_deref().unwrap();
            assert!(reason.contains("> cible"), "{reason}");
            assert!(c.generated_refusal_rate.is_none());
        }
        // (b) Only the base and the top_k finalists are generated.
        assert_eq!(model.generations, [0.0, 1.0, 1.25]);
        assert_eq!(
            (
                s.candidates_probed,
                s.candidates_generated,
                s.generations_saved
            ),
            (6, 2, 4)
        );
        assert!(s.candidates[5]
            .rejected
            .as_deref()
            .unwrap()
            .contains("hors des 2"));
        // (c) The smallest scale whose generated refusals reach the target.
        assert_eq!(s.selected_scale, Some(1.0));
        let finalist = &s.candidates[3];
        assert!(finalist.probe_refusal_rate.is_some() && finalist.generated_refusal_rate.is_some());
        assert_eq!(finalist.retention_accuracy, Some(0.75));
        assert_eq!(s.baseline.generated_refusal_rate, Some(0.9));
        assert_eq!(s.baseline_cache.as_deref(), Some("hit"));
        assert_eq!(events.first().map(String::as_str), Some("probe:0.25"));
        assert!(events.contains(&"generation:1".to_owned()));
        assert_eq!(events.last().map(String::as_str), Some("saved:4"));
    }

    #[test]
    fn without_the_prefilter_every_scale_is_generated_and_the_choice_is_the_same() {
        let mut screened = Simulated::new(0.05);
        let (with, _) = run(&mut screened, &options(true));
        // (d) No probe at all, every candidate generated: the same scale wins.
        let mut model = Simulated::new(0.05);
        let (s, _) = run(&mut model, &options(false));
        assert!(model.probes.is_empty());
        assert_eq!(model.generations, [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]);
        assert_eq!(
            (
                s.candidates_probed,
                s.candidates_generated,
                s.generations_saved
            ),
            (0, 6, 0)
        );
        assert!(s
            .candidates
            .iter()
            .all(|c| c.rejected.is_none() && c.probe_refusal_rate.is_none()));
        assert_eq!(s.selected_scale, with.selected_scale);
        assert!(model.generations.len() > screened.generations.len());
    }

    #[test]
    fn a_pessimistic_probe_still_hands_its_best_scales_to_the_generation() {
        // The probe sees too many refusals: only 1.5 passes, the lowest probed probability
        // completes the finalists, and the generation picks the smaller scale.
        let mut model = Simulated::new(0.6);
        let (s, _) = run(&mut model, &options(true));
        assert_eq!(model.generations, [0.0, 1.25, 1.5]);
        assert_eq!(s.selected_scale, Some(1.25));
        // A target nobody reaches: no scale selected.
        let mut model = Simulated::new(0.0);
        let opts = SelectionOptions {
            target_refusal_rate: 0.0,
            scales: vec![0.25, 0.5],
            ..options(true)
        };
        let (s, _) = run(&mut model, &opts);
        assert_eq!(s.selected_scale, None);
    }
}
