//! One pillar measured on one model directory, through the same library functions as
//! `bench refusal`, `bench eval --backend native`, `bench dump` and `bench logits`, greedy
//! and native (no llama.cpp): the outcomes are deterministic on a given machine.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::config::{PillarConfig, PillarKind};
use super::job::Result;
use super::PipelineError;
use crate::app::AppDeps;
use crate::bench::dataset::DatasetFormat;
use crate::bench::dump::{DumpRequest, OutDtype};
use crate::bench::generation::GenOptions;
use crate::bench::logits_io::{check_pair, open_dump, shared_offsets};
use crate::bench::logits_metrics::compare;
use crate::bench::markers::Markers;
use crate::bench::native::NativeGenerator;
use crate::bench::prompt::Template;
use crate::bench::refusal::RefusalRequest;
use crate::bench::score::{Backend, EvalRequest};
use crate::bench::{self, BenchError};
use crate::eval::TextGenerator;
use crate::extract::{load_prompt_texts, ChatTemplate, DTypeChoice};

/// Per-item outcomes of one model on one pillar. A drift is `None` where it is infinite.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcomes {
    Binary(Vec<bool>),
    Drift(Vec<Option<f64>>),
}

/// A checkpoint: what a model did on a pillar's set, item by item.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
    pub pillar: String,
    pub dataset_sha256: String,
    pub outcomes: Outcomes,
    /// Item details (answers, per-prompt drifts): Dev checkpoints keep them, Test ones live
    /// in `private/` only.
    pub items: Value,
}

pub(super) fn bench_error(e: BenchError) -> PipelineError {
    match e {
        BenchError::Cancelled => PipelineError::Cancelled,
        other => PipelineError::Bench(other.to_string()),
    }
}

/// Prompts of a refusal or drift set, or the items of a GSM8K set: their count.
pub fn count_items(kind: PillarKind, dataset: &Path, limit: Option<u32>) -> Result<usize> {
    match kind {
        PillarKind::Gsm8k => {
            let limit = limit.map_or(usize::MAX, |l| l as usize);
            bench::dataset::load(dataset, DatasetFormat::Auto, limit)
                .map(|items| items.len())
                .map_err(bench_error)
        }
        _ => load_prompt_texts(dataset)
            .map(|p| p.len())
            .map_err(|e| PipelineError::Bench(format!("{} : {e}", dataset.display()))),
    }
}

fn native(model: &Path, deps: &AppDeps) -> bench::Result<Box<dyn TextGenerator>> {
    let generator = NativeGenerator::load(
        model,
        DTypeChoice::Auto,
        deps.probe.as_ref(),
        Arc::clone(&deps.interrupted),
    )?;
    Ok(Box::new(generator))
}

/// A binary pillar (safety, unblocking, gsm8k) of `model` on `dataset`: success per item.
pub fn binary(
    pillar: &PillarConfig,
    dataset: &Path,
    model: &Path,
    deps: &AppDeps,
    cancel: &AtomicBool,
    on_item: &mut dyn FnMut(usize, usize),
) -> Result<(Vec<bool>, Value)> {
    match pillar.kind {
        PillarKind::Safety | PillarKind::Unblocking => {
            let prompts = load_prompt_texts(dataset)
                .map_err(|e| PipelineError::Bench(format!("{} : {e}", dataset.display())))?;
            let req = RefusalRequest {
                model: model.to_path_buf(),
                prompts,
                baseline_model: None,
                max_new_tokens: pillar
                    .max_new_tokens
                    .unwrap_or(bench::DEFAULT_REFUSAL_MAX_NEW_TOKENS),
                chat_template: pillar.chat_template.unwrap_or(ChatTemplate::Chatml),
                markers: Markers::default(),
                generation: GenOptions::default(),
                backend: "native".into(),
            };
            let mut make = |m: &Path| native(m, deps);
            let run = bench::refusal::run(&req, &mut make, cancel, on_item).map_err(bench_error)?;
            let keep = pillar.kind == PillarKind::Safety;
            let success = run.report.items.iter().map(|i| i.refused == keep).collect();
            Ok((success, json!(run.report.items)))
        }
        PillarKind::Gsm8k => {
            let template = match &pillar.prompt_template {
                Some(path) => Template::from_file(path).map_err(bench_error)?,
                None => Template::default(),
            };
            let req = EvalRequest {
                model: model.to_path_buf(),
                dataset: dataset.to_path_buf(),
                backend: Backend::Native,
                baseline_model: None,
                limit: pillar.limit.map_or(usize::MAX, |l| l as usize),
                max_new_tokens: pillar
                    .max_new_tokens
                    .unwrap_or(bench::DEFAULT_MAX_NEW_TOKENS),
                template,
                format: pillar.format.unwrap_or_default(),
                llama_cpp_dir: None,
                timeout: None,
                min_accuracy: None,
                generation: GenOptions::default(),
            };
            let mut make = |m: &Path| native(m, deps);
            let run = bench::score::run(&req, deps.runner.as_ref(), &mut make, cancel, on_item)
                .map_err(bench_error)?;
            let success = run.report.items.iter().map(|i| i.correct).collect();
            Ok((success, json!(run.report.items)))
        }
        PillarKind::Kl | PillarKind::Logits => Err(PipelineError::Bench(format!(
            "pilier {} : pas une mesure binaire",
            pillar.id
        ))),
    }
}

/// The logits of `model` on the prompts of `dataset`, written to `out` (complete or absent).
pub fn dump(
    dataset: &Path,
    max_tokens: Option<u32>,
    model: &Path,
    out: &Path,
    deps: &AppDeps,
    cancel: &AtomicBool,
    on_prompt: &mut dyn FnMut(usize, usize),
) -> Result<()> {
    let prompts = load_prompt_texts(dataset)
        .map_err(|e| PipelineError::Bench(format!("{} : {e}", dataset.display())))?;
    let req = DumpRequest {
        model: model.to_path_buf(),
        prompts,
        out: out.to_path_buf(),
        dtype: DTypeChoice::Auto,
        out_dtype: OutDtype::F32,
        max_tokens: max_tokens.map_or(bench::DEFAULT_DUMP_MAX_TOKENS, |t| t as usize),
        max_total_positions: bench::DEFAULT_MAX_TOTAL_POSITIONS,
        overwrite: true,
    };
    bench::dump::run(&req, deps.probe.as_ref(), cancel, on_prompt).map_err(bench_error)?;
    Ok(())
}

/// Where the dump of a model state on a set lives.
pub fn dump_path(
    work: &Path,
    state: &str,
    dataset_sha256: &str,
    max_tokens: Option<u32>,
) -> PathBuf {
    let tokens = max_tokens.map_or(bench::DEFAULT_DUMP_MAX_TOKENS, |t| t as usize);
    work.join("dumps").join(format!(
        "{state}-{}-{tokens}.safetensors",
        &dataset_sha256[..16]
    ))
}

/// The per-prompt drift between two dumps: KL(before‖after) mean, or the share of
/// positions whose argmax changed.
pub fn drift(
    kind: PillarKind,
    before: &Path,
    after: &Path,
    deps: &AppDeps,
    cancel: &AtomicBool,
) -> Result<(Vec<Option<f64>>, Value)> {
    let probe = deps.probe.as_ref();
    let a = open_dump(before, None, probe).map_err(bench_error)?;
    let b = open_dump(after, None, probe).map_err(bench_error)?;
    check_pair(&a, &b).map_err(bench_error)?;
    let offsets = shared_offsets(&a, &b).ok_or_else(|| {
        PipelineError::Bench("dumps sans prompt_offsets communs : pas de dérive par prompt".into())
    })?;
    let cmp = compare(&a, &b, 1, Some(offsets), cancel, &mut |_, _| {}).map_err(bench_error)?;
    let values: Vec<Option<f64>> = cmp
        .per_prompt
        .iter()
        .map(|p| match kind {
            PillarKind::Logits => Some(p.top1_shifts as f64 / (p.end - p.start).max(1) as f64),
            _ if p.infinite_positions > 0 => None,
            _ => Some(p.kl_mean.unwrap_or(0.0)),
        })
        .collect();
    let items = json!(cmp
        .per_prompt
        .iter()
        .zip(&values)
        .map(|(p, v)| json!({"index": p.index, "positions": p.end - p.start, "drift": v}))
        .collect::<Vec<_>>());
    Ok((values, items))
}
