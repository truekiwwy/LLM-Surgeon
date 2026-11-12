//! `plan`: what a run will cost and whether its gates can conclude, before anything is
//! written to a model or generated at length. Disk: the working copy, the logits dumps and,
//! for `restore-from-backup`, the backups. Time: work units (tokens generated or evaluated,
//! an upper estimate since generation may stop early) over a measured decoding rate. Power:
//! the smallest Dev and Test sizes at which a non-inferiority gate can be GO at all.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::config::{Gate, PillarKind, PipelineConfig, WorkingCopy};
use super::job::{dir_size, sha256_file, Result};
use super::measure;
use super::stats::{lowest_pairs, minimum_pairs};
use super::PipelineError;
use crate::app::AppDeps;
use crate::bench::native::NativeGenerator;
use crate::eval::TextGenerator;
use crate::extract::{load_prompt_texts, load_tokenizer, tokenize, DTypeChoice};

/// Tokens decoded by the calibration.
const CALIBRATION_TOKENS: u32 = 8;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PillarPlan {
    pub id: String,
    pub kind: PillarKind,
    pub metric: String,
    pub gate: Gate,
    pub n_dev: usize,
    pub n_test: usize,
    /// Non-infériorité : n à partir duquel un patch qui ne change aucune réponse est GO quand
    /// le taux de base vaut 0 ou 1 (tous les prompts du même côté) ; null pour les autres portes.
    pub min_n_for_go: Option<usize>,
    /// Non-infériorité : plus petit n auquel un patch qui ne change aucune réponse peut être
    /// GO (taux de base de 1/2, le cas le plus favorable) ; null pour les autres portes.
    pub min_n_possible: Option<usize>,
    /// n_dev < min_n_possible : la porte Dev sera INCONCLUSIVE, sauf gain net du patch.
    pub dev_underpowered: bool,
    pub test_underpowered: bool,
    /// Coût relatif : 0 dérive (un forward par prompt), 1 GSM8K, 2 refus.
    pub cost_rank: u8,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct DiskPlan {
    pub model_bytes: u64,
    /// Copie de travail (une à la fois).
    pub working_copy_bytes: u64,
    /// Dumps de logits présents en même temps, au plus : un dump de référence par jeu Dev ou
    /// Test des piliers kl et logits (gardés jusqu'à la fin) et un dump candidat.
    pub dumps_bytes: u64,
    /// Sauvegardes des fichiers touchés (restore-from-backup), au plus le modèle.
    pub backups_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Calibration {
    pub load_secs: f64,
    pub tokens_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Plan {
    pub job_id: String,
    pub candidates: usize,
    pub pillars: Vec<PillarPlan>,
    pub disk: DiskPlan,
    /// Unités de travail (tokens générés ou évalués) si aucun candidat ne s'arrête tôt.
    pub units_total: u64,
    /// Chargements du modèle prévus (un par mesure).
    pub model_loads: u64,
    /// null si la calibration n'a pas été faite.
    pub calibration: Option<Calibration>,
    /// Estimation haute de la durée, en secondes ; null sans calibration.
    pub estimated_secs: Option<u64>,
    pub max_wall_time_secs: u64,
    pub warnings: Vec<String>,
}

/// Units of the whole run if no candidate stops early: every candidate on every Dev set,
/// the base model once per set, then the Test sets (base and selected candidate).
pub fn units_total(config: &PipelineConfig) -> Result<u64> {
    let candidates = config.candidates.values().len() as u64;
    let mut total = 0u64;
    for p in &config.pillars {
        let per = |dataset: &Path| -> Result<u64> {
            let n = measure::count_items(p.kind, dataset, p.limit)? as u64;
            let per_item = match p.kind {
                PillarKind::Kl | PillarKind::Logits => p
                    .max_tokens
                    .map_or(crate::bench::DEFAULT_DUMP_MAX_TOKENS as u64, u64::from),
                PillarKind::Gsm8k => p
                    .max_new_tokens
                    .map_or(u64::from(crate::bench::DEFAULT_MAX_NEW_TOKENS), u64::from),
                _ => p.max_new_tokens.map_or(
                    u64::from(crate::bench::DEFAULT_REFUSAL_MAX_NEW_TOKENS),
                    u64::from,
                ),
            };
            Ok(n * per_item)
        };
        // Per candidate one measurement (a dump for a drift); the base model once.
        total += per(&p.dataset_dev)? * (candidates + 1);
        total += 2 * per(&p.dataset_test)?;
    }
    Ok(total)
}

fn vocab_size(model: &Path) -> u64 {
    std::fs::read_to_string(model.join("config.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
        .and_then(|v| v["vocab_size"].as_u64())
        .unwrap_or(0)
}

/// Logits bytes (f32) of the dump of `dataset`: one row per token of each prompt, cut at
/// `max_tokens`, as `bench dump` writes it.
fn dump_bytes(
    tokenizer: &tokenizers::Tokenizer,
    dataset: &Path,
    max_tokens: usize,
    vocab: u64,
) -> Result<u64> {
    let failed =
        |e: &dyn std::fmt::Display| PipelineError::Bench(format!("{} : {e}", dataset.display()));
    let mut positions = 0u64;
    for prompt in load_prompt_texts(dataset).map_err(|e| failed(&e))? {
        let tokens = tokenize(tokenizer, &prompt, max_tokens).map_err(|e| failed(&e))?;
        positions += tokens.ids.len() as u64;
    }
    Ok(positions * vocab * 4)
}

/// Loads the model and decodes a few tokens: the rate the estimate rests on.
fn calibrate(config: &PipelineConfig, deps: &AppDeps) -> Result<Calibration> {
    let started = Instant::now();
    let mut generator = NativeGenerator::load(
        &config.model,
        DTypeChoice::Auto,
        deps.probe.as_ref(),
        Arc::new(AtomicBool::new(false)),
    )
    .map_err(measure::bench_error)?;
    let load_secs = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let mut tokens = 0u32;
    generator
        .generate("Hello", CALIBRATION_TOKENS, &mut |_| {
            tokens += 1;
            std::ops::ControlFlow::Continue(())
        })
        .map_err(|e| PipelineError::Bench(e.to_string()))?;
    let secs = started.elapsed().as_secs_f64().max(1e-6);
    Ok(Calibration {
        load_secs,
        tokens_per_second: f64::from(tokens.max(1)) / secs,
    })
}

pub fn plan(
    config: &PipelineConfig,
    job_id: &str,
    deps: &AppDeps,
    calibrate_rate: bool,
) -> Result<Plan> {
    let mut warnings = Vec::new();
    let mut pillars = Vec::new();
    let mut previous_rank = 0;
    let candidates = config.candidates.values().len();
    // Base dumps by (set, max_tokens): pillars on the same set share them.
    let mut base_dumps = BTreeMap::new();
    let mut model_loads = 0u64;
    let vocab = vocab_size(&config.model);
    for p in &config.pillars {
        let n_dev = measure::count_items(p.kind, &p.dataset_dev, p.limit)?;
        let n_test = measure::count_items(p.kind, &p.dataset_test, p.limit)?;
        let (sha_dev, sha_test) = (sha256_file(&p.dataset_dev)?, sha256_file(&p.dataset_test)?);
        if sha_dev == sha_test {
            return Err(PipelineError::Config(vec![format!(
                "pilier {} : dataset_dev et dataset_test ont le même contenu",
                p.id
            )]));
        }
        let min_n_for_go = match p.gate {
            Gate::NonInferiority { .. } => p.gate.binary().and_then(minimum_pairs),
            _ => None,
        };
        let min_n_possible = match p.gate {
            Gate::NonInferiority { .. } => p.gate.binary().and_then(lowest_pairs),
            _ => None,
        };
        let short = |n: usize| min_n_possible.is_some_and(|m| n < m);
        let rate_dependent = |n: usize| !short(n) && min_n_for_go.is_some_and(|m| n < m);
        if short(n_dev) || short(n_test) {
            warnings.push(format!(
                "pilier {} : {} prompts Dev, {} Test, il en faut au moins {} pour qu'une \
                 non-infériorité à cette marge puisse être GO sans gain net du patch : verdict \
                 INCONCLUSIVE garanti pour un patch qui ne change aucune réponse",
                p.id,
                n_dev,
                n_test,
                min_n_possible.unwrap_or_default()
            ));
        } else if rate_dependent(n_dev) || rate_dependent(n_test) {
            warnings.push(format!(
                "pilier {} : {} prompts Dev, {} Test, en dessous de {} : un patch qui ne change \
                 aucune réponse est GO ou INCONCLUSIVE selon le taux de base (GO à 1/2, pas à 0 \
                 ni à 1)",
                p.id,
                n_dev,
                n_test,
                min_n_for_go.unwrap_or_default()
            ));
        }
        if p.kind.cost_rank() < previous_rank {
            warnings.push(format!(
                "pilier {} : moins coûteux qu'un pilier précédent ; l'ordre du tableau est \
                 l'ordre d'évaluation",
                p.id
            ));
        }
        previous_rank = previous_rank.max(p.kind.cost_rank());
        if !p.kind.is_binary() {
            let tokens = p
                .max_tokens
                .map_or(crate::bench::DEFAULT_DUMP_MAX_TOKENS, |t| t as usize);
            let tokenizer =
                load_tokenizer(&config.model).map_err(|e| PipelineError::Bench(e.to_string()))?;
            for (sha, dataset) in [(sha_dev, &p.dataset_dev), (sha_test, &p.dataset_test)] {
                let bytes = dump_bytes(&tokenizer, dataset, tokens, vocab)?;
                base_dumps.insert((sha, tokens), bytes);
            }
            model_loads += 2 * (candidates as u64 + 1);
        } else {
            model_loads += candidates as u64 + 1 + 2;
        }
        pillars.push(PillarPlan {
            id: p.id.clone(),
            kind: p.kind,
            metric: p.kind.metric().into(),
            gate: p.gate,
            n_dev,
            n_test,
            min_n_for_go,
            min_n_possible,
            dev_underpowered: short(n_dev),
            test_underpowered: short(n_test),
            cost_rank: p.kind.cost_rank(),
        });
    }
    // Every base dump stays until the end; one candidate dump at a time, at most the largest.
    let dumps_bytes = base_dumps.values().sum::<u64>() + base_dumps.values().max().unwrap_or(&0);
    let model_bytes = dir_size(&config.model)?;
    let backups_bytes = match config.working_copy {
        WorkingCopy::RestoreFromBackup => model_bytes,
        WorkingCopy::Copy => 0,
    };
    let disk = DiskPlan {
        model_bytes,
        working_copy_bytes: model_bytes,
        dumps_bytes,
        backups_bytes,
        total_bytes: model_bytes + dumps_bytes + backups_bytes,
    };
    let units_total = units_total(config)?;
    let calibration = match calibrate_rate {
        true => Some(calibrate(config, deps)?),
        false => None,
    };
    let estimated_secs = calibration.as_ref().map(|c| {
        (units_total as f64 / c.tokens_per_second + model_loads as f64 * c.load_secs).ceil() as u64
    });
    if let Some(secs) = estimated_secs {
        if secs > config.budgets.max_wall_time_secs {
            warnings.push(format!(
                "durée estimée {secs} s au-delà du budget de {} s : le job s'arrêtera avant la fin \
                 si aucun candidat ne s'arrête tôt",
                config.budgets.max_wall_time_secs
            ));
        }
    }
    Ok(Plan {
        job_id: job_id.into(),
        candidates,
        pillars,
        disk,
        units_total,
        model_loads,
        calibration,
        estimated_secs,
        max_wall_time_secs: config.budgets.max_wall_time_secs,
        warnings,
    })
}
