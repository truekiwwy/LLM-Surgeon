//! `surgeon extract`: steering vector `v = normalize(mean(pos) - mean(neg))` from the
//! activations of a Qwen2 layer, with a native pure-Rust forward pass (candle on CPU, the
//! `tokenizers` crate on its fancy-regex engine; no C/C++, no Python, no GPU).
//!
//! Exit codes (mapped by `app::core_extract`): 0 ok; 1 insufficient resources (or
//! cancelled, `error.kind = "cancelled"`); 2 invalid arguments or prompt files, or `--out`
//! already exists; 7 extraction failed (config, tokenizer, weights, unsupported
//! architecture, degenerate direction, NaN/Inf); 70 internal. Nothing is written unless
//! the whole run succeeds.
//!
//! (a) Modules (graphify, checked by hand):
//!
//! ```text
//! error <- every module
//! config -> error ; weights -> config, engine::ReadOnlyModel, hardware::compute_model_size
//! hooked_qwen2 -> config, weights ; prompts -> eval::load_prompts, tokenizers
//! capture -> error ; stats -> error ; output -> safetensors (re-read check)
//! mod (run_extraction) -> every module above
//! ```
//!
//! Tensors read for target layer L (nothing else, never `lm_head` nor `model.norm`):
//!
//! ```text
//! model.embed_tokens.weight
//! model.layers.{0..=L}.self_attn.{q,k,v}_proj.{weight,bias}, self_attn.o_proj.weight
//! model.layers.{0..=L}.mlp.{gate,up,down}_proj.weight
//! model.layers.{0..=L}.{input_layernorm,post_attention_layernorm}.weight
//! ```
//!
//! (b) Call graph shared by the CLI and the MCP server:
//!
//! ```text
//! run_with -> dispatch -> app::core_extract (prompt files) --+
//! mcp::tools::extract (sandbox, inline or file prompts) -----+-> app::core_extract_request
//!   -> extract::run_extraction(req, deps.probe, SinkObserver: ProgressSink + interrupted)
//!      -> output::check_output -> config::load_config -> SafetensorsDir::open (engine maps)
//!      -> weights::check_tensors -> plan_memory -> choose_dtype (BudgetPolicy, HardwareProbe)
//!      -> prompts::load_tokenizer -> tokenize -> weights::load_tensors (candle)
//!      -> HookedQwen2::from_tensors -> forward_capture per prompt -> capture::pool
//!      -> stats::separation -> output::encode -> output::write_atomic
//! ```

mod capture;
mod config;
mod error;
mod hooked_qwen2;
mod output;
mod prompts;
mod stats;
mod weights;

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

pub use capture::{pool, Pooling};
pub use config::{load_config, parse_config, Qwen2Config};
pub use error::{ExtractError, Result};
pub use hooked_qwen2::{HookedQwen2, Site};
pub use output::{
    check_output, companion_path, encode, write_atomic, write_pair_atomic, TENSOR_NAME,
};
pub use prompts::{
    check_texts, load_prompt_texts, load_tokenizer, tokenize, ChatTemplate, Tokenized,
};
pub use stats::{
    separation, separation_with_projections, Separation, SeparationStats, MIN_DIFF_NORM,
};
pub use weights::{
    check_tensors, load_tensors, required_tensors, RawTensor, SafetensorsDir, TensorSource,
    TensorTotals,
};

use crate::hardware::{BudgetPolicy, HardwareProbe};

/// Fewer prompts per set make the mean, and so the direction, noisy.
pub const MIN_RECOMMENDED_PROMPTS: usize = 8;

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum DTypeChoice {
    /// f32 when it fits in RAM, else bf16.
    #[default]
    Auto,
    F32,
    Bf16,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PromptSet {
    Pos,
    Neg,
}

impl PromptSet {
    pub fn as_str(self) -> &'static str {
        match self {
            PromptSet::Pos => "pos",
            PromptSet::Neg => "neg",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExtractRequest {
    pub model_dir: PathBuf,
    pub pos_prompts: Vec<String>,
    pub neg_prompts: Vec<String>,
    pub layer: usize,
    pub out: PathBuf,
    pub pooling: Pooling,
    pub site: Site,
    pub chat_template: ChatTemplate,
    pub max_tokens: usize,
    pub dtype: DTypeChoice,
    pub overwrite: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExtractReport {
    pub out_path: PathBuf,
    /// Companion `<out>.stats.json` (per-prompt projections, for `surgeon tui`).
    pub stats_path: PathBuf,
    pub hidden_size: usize,
    pub layer: usize,
    pub n_pos: usize,
    pub n_neg: usize,
    pub dtype_used: String,
    pub stats: SeparationStats,
    pub warnings: Vec<String>,
    pub duration_ms: u128,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExtractEvent {
    /// Resources checked, dtype chosen; weights about to load.
    Planned {
        dtype_used: String,
        weight_bytes: u64,
        tensors: usize,
    },
    LoadingWeights {
        done_tensors: usize,
        total_tensors: usize,
    },
    Prompt {
        set: PromptSet,
        done: usize,
        total: usize,
    },
    Writing,
}

/// Progress and cancellation: `Break` stops the run (checked between tensors and between
/// prompts) and nothing is written.
pub trait ExtractObserver {
    fn on_event(&mut self, event: ExtractEvent) -> ControlFlow<()>;
}

/// Bytes to hold the minimal weights, plus activations, per compute dtype.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryPlan {
    pub f32_bytes: u64,
    pub bf16_bytes: u64,
}

/// Weights of layers `0..=L` in the target dtype, the largest tensor once more in its
/// source dtype (converted one at a time), and the activations of the longest prompt:
/// q/k/v/attention output/residuals (~8 x hidden), the MLP (3 x intermediate) and the
/// attention scores (heads x seq x seq) per token.
pub fn plan_memory(cfg: &Qwen2Config, totals: TensorTotals, max_tokens: usize) -> MemoryPlan {
    let seq = max_tokens as u64;
    let per_token = 8 * cfg.hidden_size as u64 + 3 * cfg.intermediate_size as u64;
    let activations = seq * per_token + cfg.num_attention_heads as u64 * seq * seq;
    let bytes = |size: u64| (totals.elements + activations) * size + totals.largest_source_bytes;
    MemoryPlan {
        f32_bytes: bytes(4),
        bf16_bytes: bytes(2),
    }
}

/// `hardware`'s RAM rule (`evaluate_with_size` is crate-internal to `hardware` and not
/// re-exported): bytes x overhead factor (at least 1), plus the safety margin.
pub fn required_ram(bytes: u64, policy: &BudgetPolicy) -> u64 {
    let factor = match policy.ram_overhead_factor.is_finite() {
        true => policy.ram_overhead_factor.max(1.0),
        false => 1.0,
    };
    ((bytes as f64 * factor).ceil() as u64).saturating_add(policy.ram_safety_margin_bytes)
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
}

/// CPU only: RAM decides. `auto` prefers f32 and falls back to bf16.
fn choose_dtype(
    choice: DTypeChoice,
    plan: MemoryPlan,
    probe: &dyn HardwareProbe,
    warnings: &mut Vec<String>,
) -> Result<DType> {
    let policy = BudgetPolicy::default();
    let available = probe
        .system_ram()
        .map_err(|e| ExtractError::InsufficientResources(e.to_string()))?
        .available_bytes;
    let fits = |bytes: u64| required_ram(bytes, &policy) <= available;
    let refuse = |bytes: u64, dtype: &str| {
        ExtractError::InsufficientResources(format!(
            "{dtype} : besoin de {} (marge incluse), disponible {}",
            gib(required_ram(bytes, &policy)),
            gib(available)
        ))
    };
    match choice {
        DTypeChoice::F32 if fits(plan.f32_bytes) => Ok(DType::F32),
        DTypeChoice::F32 => Err(refuse(plan.f32_bytes, "f32")),
        DTypeChoice::Bf16 if fits(plan.bf16_bytes) => Ok(DType::BF16),
        DTypeChoice::Bf16 => Err(refuse(plan.bf16_bytes, "bf16")),
        DTypeChoice::Auto if fits(plan.f32_bytes) => Ok(DType::F32),
        DTypeChoice::Auto if fits(plan.bf16_bytes) => {
            warnings.push("f32 ne tient pas en RAM : calcul en bf16 (précision réduite)".into());
            Ok(DType::BF16)
        }
        DTypeChoice::Auto => Err(refuse(plan.bf16_bytes, "bf16")),
    }
}

fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "bf16",
        _ => "f32",
    }
}

/// Opens the model directory and runs [`run_extraction_with`].
pub fn run_extraction(
    req: &ExtractRequest,
    probe: &dyn HardwareProbe,
    observer: &mut dyn ExtractObserver,
) -> Result<ExtractReport> {
    validate_request(req)?;
    let source = SafetensorsDir::open(&req.model_dir)?;
    run_extraction_with(req, &source, probe, observer)
}

fn validate_request(req: &ExtractRequest) -> Result<()> {
    for (set, prompts) in [
        (PromptSet::Pos, &req.pos_prompts),
        (PromptSet::Neg, &req.neg_prompts),
    ] {
        check_texts(prompts.clone()).map_err(|message| ExtractError::Prompts {
            set: set.as_str().into(),
            message,
        })?;
    }
    if req.max_tokens == 0 {
        return Err(ExtractError::InvalidArgument(
            "--max-tokens doit être >= 1".into(),
        ));
    }
    check_output(&req.out, req.overwrite)
}

/// validate -> config -> tensor index check -> resources -> tokenizer -> minimal weights
/// -> forward pos then neg -> statistics -> atomic write. Each step fails before the next
/// one touches anything heavier.
pub fn run_extraction_with(
    req: &ExtractRequest,
    source: &dyn TensorSource,
    probe: &dyn HardwareProbe,
    observer: &mut dyn ExtractObserver,
) -> Result<ExtractReport> {
    let started = Instant::now();
    validate_request(req)?;
    let cfg = load_config(&req.model_dir)?;
    if req.layer >= cfg.num_hidden_layers {
        return Err(ExtractError::InvalidArgument(format!(
            "couche {} hors bornes : le modèle a num_hidden_layers = {} (couches 0 à {})",
            req.layer,
            cfg.num_hidden_layers,
            cfg.num_hidden_layers - 1
        )));
    }
    let required = required_tensors(&cfg, req.layer);
    let totals = check_tensors(source, &required)?;
    let mut warnings = Vec::new();
    let dtype = choose_dtype(
        req.dtype,
        plan_memory(&cfg, totals, req.max_tokens),
        probe,
        &mut warnings,
    )?;
    let tensor_bytes = totals.elements * dtype.size_in_bytes() as u64;
    let dtype_used = dtype_name(dtype).to_owned();
    if observer
        .on_event(ExtractEvent::Planned {
            dtype_used: dtype_used.clone(),
            weight_bytes: tensor_bytes,
            tensors: required.len(),
        })
        .is_break()
    {
        return Err(ExtractError::Cancelled);
    }

    let tokenizer = load_tokenizer(&req.model_dir)?;
    let mut truncated = 0;
    let mut encode_set = |set: PromptSet, prompts: &[String]| -> Result<Vec<Vec<u32>>> {
        prompts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let t = tokenize(&tokenizer, &req.chat_template.apply(p), req.max_tokens)?;
                if t.ids.is_empty() {
                    return Err(ExtractError::Prompts {
                        set: set.as_str().into(),
                        message: format!("prompt {i} : aucun token"),
                    });
                }
                truncated += usize::from(t.truncated);
                Ok(t.ids)
            })
            .collect()
    };
    let pos_ids = encode_set(PromptSet::Pos, &req.pos_prompts)?;
    let neg_ids = encode_set(PromptSet::Neg, &req.neg_prompts)?;

    let device = Device::Cpu;
    let tensors = load_tensors(source, &required, dtype, &device, &mut |done, total| {
        observer.on_event(ExtractEvent::LoadingWeights {
            done_tensors: done,
            total_tensors: total,
        })
    })?;
    let model = HookedQwen2::from_tensors(tensors, &cfg, req.layer, dtype, &device)?;

    let mut activations = |set: PromptSet, ids: &[Vec<u32>]| -> Result<Vec<Vec<f64>>> {
        let mut pooled = Vec::with_capacity(ids.len());
        for (i, prompt_ids) in ids.iter().enumerate() {
            let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
            let activation = model.forward_capture(&input, req.site)?;
            pooled.push(pool(
                &activation,
                req.pooling,
                &format!("{}[{i}]", set.as_str()),
            )?);
            let event = ExtractEvent::Prompt {
                set,
                done: i + 1,
                total: ids.len(),
            };
            if observer.on_event(event).is_break() {
                return Err(ExtractError::Cancelled);
            }
        }
        Ok(pooled)
    };
    let pos = activations(PromptSet::Pos, &pos_ids)?;
    let neg = activations(PromptSet::Neg, &neg_ids)?;
    drop(model);

    let Separation {
        direction,
        stats,
        pos_projections,
        neg_projections,
    } = separation_with_projections(&pos, &neg)?;
    for (set, n) in [("positifs", pos.len()), ("négatifs", neg.len())] {
        if n < MIN_RECOMMENDED_PROMPTS {
            warnings.push(format!(
                "seulement {n} prompts {set} (au moins {MIN_RECOMMENDED_PROMPTS} recommandés)"
            ));
        }
    }
    if truncated > 0 {
        warnings.push(format!(
            "{truncated} prompt(s) tronqué(s) à {} tokens",
            req.max_tokens
        ));
    }
    if stats.d_prime.is_some_and(|d| d < 1.0) {
        warnings.push("séparation faible (d′ < 1) : direction peu fiable".into());
    }

    if observer.on_event(ExtractEvent::Writing).is_break() {
        return Err(ExtractError::Cancelled);
    }
    let metadata = file_metadata(req, &cfg, &dtype_used, pos.len(), neg.len(), &stats);
    let companion = Companion {
        schema: COMPANION_SCHEMA,
        pos_projections: &pos_projections,
        neg_projections: &neg_projections,
        threshold: stats.threshold,
        d_prime: stats.d_prime,
        threshold_accuracy: stats.threshold_accuracy,
        cos_mean_pos_neg: stats.cosine_means,
        raw_diff_norm: stats.raw_diff_norm,
        layer: req.layer,
        site: req.site.as_str(),
        pooling: req.pooling.as_str(),
        hidden_size: cfg.hidden_size,
    };
    let mut companion_bytes =
        serde_json::to_vec_pretty(&companion).map_err(|e| ExtractError::Compute(e.to_string()))?;
    companion_bytes.push(b'\n');
    write_pair_atomic(
        &req.out,
        &encode(&direction, &metadata)?,
        &companion_bytes,
        req.overwrite,
    )?;

    Ok(ExtractReport {
        out_path: req.out.clone(),
        stats_path: companion_path(&req.out),
        hidden_size: cfg.hidden_size,
        layer: req.layer,
        n_pos: pos.len(),
        n_neg: neg.len(),
        dtype_used,
        stats,
        warnings,
        duration_ms: started.elapsed().as_millis(),
    })
}

fn file_metadata(
    req: &ExtractRequest,
    cfg: &Qwen2Config,
    dtype_used: &str,
    n_pos: usize,
    n_neg: usize,
    stats: &SeparationStats,
) -> BTreeMap<String, String> {
    let model = model_name(&req.model_dir);
    [
        ("model", model),
        ("model_type", cfg.model_type.clone()),
        ("layer", req.layer.to_string()),
        ("site", req.site.as_str().into()),
        ("pooling", req.pooling.as_str().into()),
        ("chat_template", req.chat_template.as_str().into()),
        ("n_pos", n_pos.to_string()),
        ("n_neg", n_neg.to_string()),
        ("raw_diff_norm", stats.raw_diff_norm.to_string()),
        (
            "d_prime",
            stats.d_prime.map_or("inf".to_owned(), |d| d.to_string()),
        ),
        ("threshold_accuracy", stats.threshold_accuracy.to_string()),
        ("pos_mean_proj", stats.pos_mean_proj.to_string()),
        ("neg_mean_proj", stats.neg_mean_proj.to_string()),
        ("cos_mean_pos_neg", stats.cosine_means.to_string()),
        ("dtype_used", dtype_used.into()),
        ("surgeon_version", env!("CARGO_PKG_VERSION").into()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect()
}

/// Version of the `<out>.stats.json` layout read by `surgeon tui`.
pub const COMPANION_SCHEMA: u32 = 1;

/// `<out>.stats.json`. `d_prime` is `null` when both sets have zero spread (JSON has no
/// infinity).
#[derive(Serialize)]
struct Companion<'a> {
    schema: u32,
    pos_projections: &'a [f64],
    neg_projections: &'a [f64],
    threshold: f64,
    d_prime: Option<f64>,
    threshold_accuracy: f64,
    cos_mean_pos_neg: f64,
    raw_diff_norm: f64,
    layer: usize,
    site: &'a str,
    pooling: &'a str,
    hidden_size: usize,
}

/// Name of the model directory (absolute form first, so `.` still gets a name).
pub fn model_name(model_dir: &Path) -> String {
    std::path::absolute(model_dir)
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| model_dir.display().to_string())
}
