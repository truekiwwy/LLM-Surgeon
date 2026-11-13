//! Generation options shared by `bench eval` and `bench refusal`, and the statistics of
//! several samples per prompt.
//!
//! N samples of one prompt are not independent observations: the unit is the prompt. With
//! `samples > 1` the report gives the rate of each prompt, their mean, a 95 % percentile
//! bootstrap interval that resamples PROMPTS (fixed seed, [`BOOTSTRAP_RESAMPLES`]), and the
//! dispersion within prompts (mean of `p (1 - p)`) and between them (variance of the
//! per-prompt rates). The historical Wilson interval over items stays in `score`, valid only
//! for one sample per prompt.

use schemars::JsonSchema;
use serde::Serialize;

use super::error::{BenchError, Result};
use super::score::SEED;
use crate::eval::sampling::{SamplingParams, Xoshiro256pp};
use crate::eval::{Decoding, GenJob, WeightType};

pub const BOOTSTRAP_RESAMPLES: usize = 2000;
/// Seed of the bootstrap (not the sampling seed).
pub const BOOTSTRAP_SEED: u64 = 20_241_111;
pub const MAX_SAMPLES: usize = 1000;
pub const MAX_BATCH: usize = 256;
pub const MAX_JOBS: usize = 64;

/// How the benches generate: sampling, samples per prompt, native batching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GenOptions {
    pub sampling: SamplingParams,
    pub samples: usize,
    /// Native backend: sequences decoded together; llama-server: concurrent requests.
    pub batch_size: usize,
    /// Native backend: batches decoded in parallel (one rayon pool for all).
    pub jobs: usize,
    /// Native backend: threads of that pool (`None`: RAYON_NUM_THREADS, else every core);
    /// llama-server: its `-t`.
    pub threads: Option<usize>,
    /// llama-server: `-ngl`.
    pub gpu_layers: Option<u32>,
    /// llama-server: `-c`.
    pub ctx_size: Option<u32>,
    /// Matrix type of the temporary GGUF a llama backend converts a SafeTensors model to.
    pub gguf_dtype: WeightType,
    /// The GPU API of the llama-server (`Vulkan`...), only to label the report.
    pub gpu_api: Option<&'static str>,
}

impl Default for GenOptions {
    fn default() -> Self {
        Self {
            sampling: SamplingParams::default(),
            samples: 1,
            batch_size: 1,
            jobs: 1,
            threads: None,
            gpu_layers: None,
            ctx_size: None,
            gguf_dtype: WeightType::default(),
            gpu_api: None,
        }
    }
}

impl GenOptions {
    pub fn validate(&self) -> Result<()> {
        let bad = |m: String| Err(BenchError::InvalidArgument(m));
        self.sampling
            .validate()
            .map_err(BenchError::InvalidArgument)?;
        if !(1..=MAX_SAMPLES).contains(&self.samples) {
            return bad(format!("--samples doit être entre 1 et {MAX_SAMPLES}"));
        }
        if self.samples > 1 && self.sampling.is_greedy() {
            return bad(
                "--samples > 1 demande --temperature > 0 (en glouton, les échantillons d'un \
                 prompt sont identiques)"
                    .into(),
            );
        }
        if !(1..=MAX_BATCH).contains(&self.batch_size) {
            return bad(format!("--batch-size doit être entre 1 et {MAX_BATCH}"));
        }
        if !(1..=MAX_JOBS).contains(&self.jobs) {
            return bad(format!("--jobs doit être entre 1 et {MAX_JOBS}"));
        }
        if self.threads == Some(0) {
            return bad("--threads doit être >= 1".into());
        }
        Ok(())
    }

    /// The llama-server options: one slot per concurrent request.
    pub fn server(&self) -> crate::eval::server::ServerOptions {
        crate::eval::server::ServerOptions {
            gpu_layers: self.gpu_layers,
            ctx_size: self.ctx_size,
            slots: self.batch_size,
            threads: self.threads,
            ..Default::default()
        }
    }

    /// Threads of the pool: `--threads`, else RAYON_NUM_THREADS, else every core; and the
    /// warnings of an over-subscribed machine (more threads than cores, other surgeon
    /// processes sharing cores and memory bandwidth).
    pub fn thread_budget(&self, other_processes: usize) -> (usize, Vec<String>) {
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        let env = std::env::var("RAYON_NUM_THREADS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok());
        let threads = self.threads.or(env).unwrap_or(cores).max(1);
        let mut warnings = Vec::new();
        if let (Some(t), Some(e)) = (self.threads, env) {
            if t != e {
                warnings.push(format!("--threads {t} remplace RAYON_NUM_THREADS={e}"));
            }
        }
        if threads > cores {
            warnings.push(format!(
                "{threads} threads pour {cores} cœurs logiques : sur-souscription"
            ));
        }
        if other_processes > 0 {
            warnings.push(format!(
                "{other_processes} autre(s) processus surgeon en cours : ils se partagent les \
                 cœurs et la bande passante mémoire de cette machine"
            ));
        }
        (threads, warnings)
    }

    pub fn decoding(&self, timeout: Option<std::time::Duration>) -> Decoding {
        Decoding {
            seed: SEED,
            timeout,
            sampling: self.sampling,
        }
    }

    /// Every (prompt, sample), prompt by prompt; the id of a sample's stream depends only
    /// on its prompt index and sample index.
    pub fn jobs<'a>(&self, prompts: impl Iterator<Item = String> + 'a) -> Vec<GenJob> {
        let samples = self.samples;
        prompts
            .enumerate()
            .flat_map(|(p, prompt)| {
                (0..samples).map(move |s| GenJob {
                    prompt: prompt.clone(),
                    sequence_id: GenJob::id(p, s),
                })
            })
            .collect()
    }
}

/// Statistics of several samples per prompt (see the module doc).
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct SampledScore {
    /// Échantillons par prompt.
    pub samples_per_prompt: usize,
    /// Taux de chaque prompt, dans l'ordre du fichier.
    pub per_prompt: Vec<f64>,
    /// Moyenne des taux par prompt.
    pub mean: f64,
    /// Borne basse de l'intervalle bootstrap à 95 % (rééchantillonnage des prompts).
    pub ci95_low: f64,
    /// Borne haute de l'intervalle bootstrap à 95 % (rééchantillonnage des prompts).
    pub ci95_high: f64,
    /// Moyenne de p (1 - p) sur les prompts : la dispersion entre échantillons d'un prompt.
    pub within_prompt_variance: f64,
    /// Variance des taux par prompt (population).
    pub between_prompt_variance: f64,
    pub bootstrap_resamples: usize,
    pub bootstrap_seed: u64,
}

/// Percentile interval (2.5 %, 97.5 %, nearest rank) of a statistic of resampled prompts.
fn bootstrap(n: usize, statistic: impl Fn(&[usize]) -> f64) -> (f64, f64) {
    if n == 0 {
        return (0.0, 0.0);
    }
    let mut rng = Xoshiro256pp::from_seed(BOOTSTRAP_SEED);
    let mut draws = vec![0usize; n];
    let mut values: Vec<f64> = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| {
            draws
                .iter_mut()
                .for_each(|d| *d = ((rng.next_f64() * n as f64) as usize).min(n - 1));
            statistic(&draws)
        })
        .collect();
    values.sort_by(f64::total_cmp);
    let at = |q: f64| {
        values[((q * BOOTSTRAP_RESAMPLES as f64).ceil() as usize).clamp(1, values.len()) - 1]
    };
    (at(0.025), at(0.975))
}

fn mean(values: &[f64]) -> f64 {
    match values.len() {
        0 => 0.0,
        n => values.iter().sum::<f64>() / n as f64,
    }
}

/// `hits[p][s]`: whether sample `s` of prompt `p` counts (refusal, correct answer).
pub fn sampled_score(hits: &[Vec<bool>]) -> SampledScore {
    let per_prompt: Vec<f64> = hits
        .iter()
        .map(|h| h.iter().filter(|x| **x).count() as f64 / h.len().max(1) as f64)
        .collect();
    let m = mean(&per_prompt);
    let (ci95_low, ci95_high) = bootstrap(per_prompt.len(), |draws| {
        draws.iter().map(|&i| per_prompt[i]).sum::<f64>() / draws.len() as f64
    });
    SampledScore {
        samples_per_prompt: hits.first().map_or(0, Vec::len),
        mean: m,
        ci95_low,
        ci95_high,
        within_prompt_variance: mean(&per_prompt.iter().map(|p| p * (1.0 - p)).collect::<Vec<_>>()),
        between_prompt_variance: mean(
            &per_prompt
                .iter()
                .map(|p| (p - m).powi(2))
                .collect::<Vec<_>>(),
        ),
        per_prompt,
        bootstrap_resamples: BOOTSTRAP_RESAMPLES,
        bootstrap_seed: BOOTSTRAP_SEED,
    }
}

/// `generation` of the eval and refusal reports, present when the generation is not the
/// historical one (greedy, one sample, no batching, default backend).
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct GenerationReport {
    /// native, llama, llama-completion, llama-server, mistralrs ou hipfire.
    pub backend: String,
    /// Le backend réel : `native-cpu`, `llama-server (Vulkan, ngl=99, batch=6)`...
    pub backend_detail: String,
    pub sampling: SamplingParams,
    pub samples: usize,
    pub batch_size: usize,
    pub jobs: usize,
    /// Statistiques par prompt du modèle évalué, si samples > 1.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampled: Option<SampledScore>,
    /// Statistiques par prompt de la référence, si samples > 1 et --baseline-model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_sampled: Option<SampledScore>,
    /// Différence appariée par prompt, si samples > 1 et --baseline-model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<SampledDelta>,
}

impl GenOptions {
    /// Whether the run differs from the historical one with this backend label.
    pub fn is_historical(&self, backend_is_default: bool) -> bool {
        backend_is_default && *self == GenOptions::default()
    }

    /// The report block, with the per-prompt statistics of `samples > 1`.
    pub fn report(
        &self,
        backend: &str,
        main: Option<&[Vec<bool>]>,
        baseline: Option<&[Vec<bool>]>,
    ) -> GenerationReport {
        let sampled = main.filter(|_| self.samples > 1).map(sampled_score);
        let baseline_sampled = baseline.filter(|_| self.samples > 1).map(sampled_score);
        let delta = match (&sampled, &baseline_sampled) {
            (Some(m), Some(b)) => Some(sampled_delta(m, b)),
            _ => None,
        };
        GenerationReport {
            backend: backend.to_owned(),
            backend_detail: crate::backend::describe(
                backend,
                self.gpu_api,
                self.gpu_layers,
                self.batch_size,
            ),
            sampling: self.sampling,
            samples: self.samples,
            batch_size: self.batch_size,
            jobs: self.jobs,
            sampled,
            baseline_sampled,
            delta,
        }
    }
}

/// `hits[p][s]` from items laid out prompt by prompt, `samples` each.
pub fn by_prompt(hits: impl Iterator<Item = bool>, samples: usize) -> Vec<Vec<bool>> {
    let all: Vec<bool> = hits.collect();
    all.chunks(samples.max(1)).map(<[bool]>::to_vec).collect()
}

/// Paired difference of two models on the same prompts, several samples each.
#[derive(Debug, Clone, PartialEq, Serialize, JsonSchema)]
pub struct SampledDelta {
    /// Moyenne des différences de taux par prompt (évalué − référence).
    pub delta: f64,
    /// Intervalle bootstrap à 95 % des prompts appariés.
    pub ci95_low: f64,
    pub ci95_high: f64,
}

pub fn sampled_delta(main: &SampledScore, baseline: &SampledScore) -> SampledDelta {
    let diffs: Vec<f64> = main
        .per_prompt
        .iter()
        .zip(&baseline.per_prompt)
        .map(|(a, b)| a - b)
        .collect();
    let (ci95_low, ci95_high) = bootstrap(diffs.len(), |draws| {
        draws.iter().map(|&i| diffs[i]).sum::<f64>() / draws.len() as f64
    });
    SampledDelta {
        delta: mean(&diffs),
        ci95_low,
        ci95_high,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_prompt_rates_and_dispersions() {
        let hits = vec![
            vec![true, true, false, false],
            vec![true, true, true, true],
            vec![false, false, false, false],
        ];
        let s = sampled_score(&hits);
        assert_eq!(s.per_prompt, [0.5, 1.0, 0.0]);
        assert!((s.mean - 0.5).abs() < 1e-12);
        assert!((s.within_prompt_variance - 0.25 / 3.0).abs() < 1e-12);
        assert!((s.between_prompt_variance - (0.0 + 0.25 + 0.25) / 3.0).abs() < 1e-12);
        assert!(s.ci95_low <= s.mean && s.mean <= s.ci95_high);
        assert_eq!(sampled_score(&hits), s, "fixed bootstrap seed");
    }

    #[test]
    fn bootstrap_matches_the_normal_approximation_on_many_prompts() {
        // 400 prompts, half at 0 and half at 1: the mean of resampled prompts has a
        // standard error of 0.5 / 20 = 0.025, so the interval is about 0.5 ± 0.049.
        let hits: Vec<Vec<bool>> = (0..400).map(|i| vec![i % 2 == 0; 3]).collect();
        let s = sampled_score(&hits);
        assert!((s.ci95_low - 0.451).abs() < 0.01, "{}", s.ci95_low);
        assert!((s.ci95_high - 0.549).abs() < 0.01, "{}", s.ci95_high);
        // Identical prompts: no spread at all.
        let same = sampled_score(&vec![vec![true, false]; 10]);
        assert_eq!((same.ci95_low, same.ci95_high), (0.5, 0.5));
        let d = sampled_delta(&s, &s);
        assert_eq!((d.delta, d.ci95_low, d.ci95_high), (0.0, 0.0, 0.0));
    }

    #[test]
    fn bootstrap_matches_the_exact_resampling_distribution() {
        // 5 prompts: the 5^5 equally likely resamples, enumerated, give the exact
        // percentiles; the 2000 seeded draws land within one step (0.05) of them.
        let hits = [0, 0, 1, 2, 4].map(|k| (0..4).map(|s| s < k).collect::<Vec<bool>>());
        let rates: Vec<f64> = hits
            .iter()
            .map(|h| h.iter().filter(|x| **x).count() as f64 / 4.0)
            .collect();
        let mut exact: Vec<f64> = (0..5usize.pow(5))
            .map(|mut code| {
                (0..5)
                    .map(|_| {
                        let r = rates[code % 5];
                        code /= 5;
                        r
                    })
                    .sum::<f64>()
                    / 5.0
            })
            .collect();
        exact.sort_by(f64::total_cmp);
        let at = |q: f64| exact[(q * exact.len() as f64).ceil() as usize - 1];
        let s = sampled_score(&hits);
        assert!(
            (s.ci95_low - at(0.025)).abs() <= 0.05 + 1e-12,
            "{} vs {}",
            s.ci95_low,
            at(0.025)
        );
        assert!(
            (s.ci95_high - at(0.975)).abs() <= 0.05 + 1e-12,
            "{} vs {}",
            s.ci95_high,
            at(0.975)
        );
        assert!((s.mean - 0.35).abs() < 1e-12);
    }

    #[test]
    fn thread_budget_warns_of_over_subscription() {
        let cores = std::thread::available_parallelism().map_or(1, usize::from);
        let fits = GenOptions {
            threads: Some(1),
            ..GenOptions::default()
        };
        let (threads, warnings) = fits.thread_budget(0);
        assert_eq!(threads, 1);
        assert!(
            warnings.iter().all(|w| !w.contains("sur-souscription")),
            "{warnings:?}"
        );
        let (_, warnings) = fits.thread_budget(2);
        assert!(warnings
            .iter()
            .any(|w| w.starts_with("2 autre(s) processus surgeon")));
        let too_many = GenOptions {
            threads: Some(cores + 1),
            ..GenOptions::default()
        };
        let (threads, warnings) = too_many.thread_budget(0);
        assert_eq!(threads, cores + 1);
        assert!(
            warnings.iter().any(|w| w.contains("sur-souscription")),
            "{warnings:?}"
        );
    }

    #[test]
    fn options_are_checked_and_jobs_keep_their_ids() {
        let sampled = GenOptions {
            sampling: SamplingParams {
                temperature: 0.7,
                ..SamplingParams::default()
            },
            samples: 3,
            ..GenOptions::default()
        };
        sampled.validate().unwrap();
        let jobs = sampled.jobs(["a", "b"].iter().map(|s| (*s).to_owned()));
        assert_eq!(jobs.len(), 6);
        assert_eq!(jobs[4].sequence_id, GenJob::id(1, 1));
        assert!(GenOptions {
            samples: 3,
            ..GenOptions::default()
        }
        .validate()
        .is_err());
        assert!(GenOptions {
            batch_size: 0,
            ..GenOptions::default()
        }
        .validate()
        .is_err());
        assert!(GenOptions::default().validate().is_ok());
    }
}
