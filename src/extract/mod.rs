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
//! decoder -> config, weights ; forward -> decoder ; prompts -> eval::load_prompts, tokenizers
//! capture -> error ; stats -> error ; output -> safetensors (re-read check)
//! method (MethodOptions, holdout_mask, compute) -> svd, leace
//! svd -> linalg::gram (contrastive PCA on paired deltas)
//! leace -> linalg::gram (Σ_λ^p y through the thin SVD of X̃/√n), svd::dot
//! mod (run_extraction*) -> every module above, plugin::ExtractionMethod
//! plugin::Builtin (mean_diff, svd, leace) -> stats::mean_diff_direction, method::compute
//! consumers: patch::autotune (forward::build, weights, prompts), bench::dump (forward_logits)
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
//! run_with -> dispatch -> app::core_extract_with (args, method id, params) --------+
//! mcp::tools::extract (sandbox, inline or file prompts, method id, params) --------+
//!   -> app::core_extract_method(req, id, params) -> Registry::with_builtin().get(id)
//!   -> extract::run_extraction_plugin(req, method, params, deps.probe, SinkObserver,
//!                                     deps.interrupted)
//!      -> output::check_output -> validate_method (params, pairing, declared sidecars)
//!      -> config::load_config -> SafetensorsDir::open (engine maps)
//!      -> weights::check_tensors -> plan_memory -> choose_dtype (BudgetPolicy, HardwareProbe)
//!      -> prompts::load_tokenizer -> tokenize -> weights::load_tensors (candle)
//!      -> forward::build (ResidualForward) -> forward_capture per prompt -> capture::pool
//!      -> method.extract(ctx, ActivationSet) -> check_direction -> stats::separation_along
//!      -> output::encode (+ encode_tensors per sidecar) -> write_pair_atomic | write_files_atomic
//! autotune (patch::autotune) -> required_tensors, load_tensors, forward::build
//!   per probe, forward_capture, pool ; engine::apply_vector_shift on the stored bytes
//! ```
//!
//! (c) Output tensors and sidecars:
//!
//! ```text
//! <out>                     direction (1D f32, L2 norm 1)  + __metadata__ (layer, site,
//!                           pooling, stats; svd/leace add method and its parameters)
//! <out>.stats.json          per-prompt projections on direction, d′, accuracy, threshold;
//!                           method_diagnostics for svd/leace (absent for mean_diff)
//! <out>.leace.safetensors   a, b, mean, u_whitened (1D f32, size d) + shrinkage, rank,
//!                           kappa, bTa, n_pos, n_neg, layer, site, pooling (leace only) ;
//!                           with --labels-file and k > 1 directions: a, b, u_whitened are
//!                           [k, d] (one direction per row) + subspace_rank
//! <out>.subspace.safetensors  a = b = V_k, [k, d] (svd --rank k > 1 only)
//! consumers: patch vector_file (direction) ; project_out (a = vector_tensor,
//!            b = vector_tensor_b, 1D or [k, d]) ; autotune --vector (direction + metadata
//!            + stats.json)
//! ```

mod capture;
mod config;
mod decoder;
mod error;
pub mod forward;
pub mod generate;
mod leace;
mod method;
mod output;
mod prompts;
mod stats;
mod subspace;
mod svd;
mod weights;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub use capture::{pool, Pooling};
pub use config::{
    load_config, parse_config, DecoderConfig, Llama3Scaling, Qwen2Config, Variant,
    SUPPORTED_MODEL_TYPES,
};
pub use decoder::{HookedDecoder, HookedQwen2, KvCache, Site};
pub use error::{ExtractError, Result};
pub use leace::{cross_cov_norm, erased_stats, fit as leace_fit, LeaceDiagnostics, LeaceFit};
pub(crate) use method::compute_subspace as compute_method;
pub use method::{
    check_subspace, holdout_mask, params_with_subspace, Method, MethodDiagnostics, MethodOptions,
    MethodOutput, SubspaceFit, DEFAULT_LEACE_RCOND, DEFAULT_LEACE_SHRINKAGE, MIN_HOLDOUT_PER_CLASS,
};
pub use output::{
    check_output, companion_path, encode, encode_shaped, encode_tensors, ensure_parent, leace_path,
    sidecar_path, write_atomic, write_files_atomic, write_pair_atomic, TENSOR_NAME,
};
pub use prompts::{
    chat_messages, check_texts, load_prompt_texts, load_tokenizer, model_folder, tokenize,
    ChatTemplate, PromptFormat, Tokenized,
};
pub use stats::{
    mean_diff_direction, separation, separation_along, separation_with_projections, Separation,
    SeparationStats, MIN_DIFF_NORM,
};
pub use subspace::{load_labels, parse_labels, SubspaceDiagnostics, SubspaceOptions};
pub use svd::{svd_direction, svd_subspace, SvdDiagnostics};
pub use weights::{
    check_tensors, load_tensors, required_tensors, required_tensors_full, RawTensor,
    SafetensorsDir, TensorSource, TensorTotals,
};

use crate::hardware::{BudgetPolicy, HardwareProbe};
use crate::plugin::{
    is_valid_id, ActivationSet, Builtin, Diagnostics, ExtractContext, ExtractionMethod, MethodInfo,
    PluginError,
};

/// Fewer prompts per set make the mean, and so the direction, noisy.
pub const MIN_RECOMMENDED_PROMPTS: usize = 8;
/// Defaults of the CLI and of the MCP tool.
pub const DEFAULT_LAYER: usize = 20;
pub const DEFAULT_MAX_TOKENS: usize = 512;

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

/// `stats` always measures the written direction on the prompts it was fit on; with
/// `holdout_frac`, the method is fit again on the kept prompts and measured on the held-out
/// ones, which says whether the separation generalizes.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct HoldoutReport {
    /// Fraction asked for (0: none).
    pub frac: f64,
    /// What `stats` was measured on: `"all_prompts"`, always.
    pub stats_basis: &'static str,
    pub n_train_pos: usize,
    pub n_train_neg: usize,
    pub n_holdout_pos: usize,
    pub n_holdout_neg: usize,
    /// The same statistics, out of sample; null without a hold-out.
    pub stats: Option<SeparationStats>,
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
    /// Separation of the written direction, fit and measured on every prompt.
    pub stats: SeparationStats,
    /// Projection of each prompt on the written direction, as in `<out>.stats.json`.
    pub pos_projections: Vec<f64>,
    pub neg_projections: Vec<f64>,
    /// The train / held-out split and the out-of-sample separation.
    pub holdout: HoldoutReport,
    pub warnings: Vec<String>,
    pub duration_ms: u128,
    /// Direction method (prompt 11, additive): its registry identifier.
    pub method: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub method_diagnostics: Option<Diagnostics>,
    /// `<out>.leace.safetensors` (a, b, mean, u_whitened) for `--method leace`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub leace_path: Option<PathBuf>,
    /// The other sidecars, `<out>.<suffix>.safetensors`, of a registered method
    /// (`leace_path` keeps its own key).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sidecar_paths: Vec<PathBuf>,
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
    /// A decision worth a log line (the dtype, the free RAM it was taken on).
    Note(String),
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

/// What [`dtype_decision`] chose, and on which numbers, for the logs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtypeDecision {
    pub choice: DTypeChoice,
    pub dtype: DType,
    /// Free RAM when it was decided.
    pub available_bytes: u64,
    /// What the chosen dtype needs, margin included.
    pub required_bytes: u64,
    /// `auto` fell back to bf16: what f32 needed, margin included.
    pub f32_required_bytes: Option<u64>,
}

impl DtypeDecision {
    /// The log line of every load: dtype, how it was chosen, free RAM, need.
    pub fn note(&self) -> String {
        let how = match self.choice {
            DTypeChoice::Auto => "auto",
            _ => "demandé",
        };
        format!(
            "dtype {} ({how}) : RAM libre {}, besoin {} marge incluse",
            dtype_name(self.dtype),
            gib(self.available_bytes),
            gib(self.required_bytes)
        )
    }

    /// The warning of a fallback to bf16, with the numbers that caused it.
    pub fn fallback(&self) -> Option<String> {
        let f32_needs = self.f32_required_bytes?;
        Some(format!(
            "repli en bf16 : f32 demande {} marge incluse, RAM libre {} ; bf16 est environ 2,5 \
             fois plus lent sur CPU (forward de Qwen2.5-0.5B : 5,9 s en f32, 14,4 s en bf16). \
             Un autre processus occupe-t-il la mémoire ? Libérez-la puis relancez pour revenir \
             en f32",
            gib(f32_needs),
            gib(self.available_bytes)
        ))
    }
}

/// CPU only: RAM decides. `auto` prefers f32 and falls back to bf16 when f32 does not fit the
/// RAM free at that moment; [`DtypeDecision::fallback`] says so with the numbers.
pub fn dtype_decision(
    choice: DTypeChoice,
    plan: MemoryPlan,
    probe: &dyn HardwareProbe,
) -> Result<DtypeDecision> {
    let policy = BudgetPolicy::default();
    let available = probe
        .system_ram()
        .map_err(|e| ExtractError::InsufficientResources(e.to_string()))?
        .available_bytes;
    let need = |bytes: u64| required_ram(bytes, &policy);
    let fits = |bytes: u64| need(bytes) <= available;
    let refuse = |bytes: u64, dtype: &str| {
        ExtractError::InsufficientResources(format!(
            "{dtype} : besoin de {} (marge incluse), disponible {}",
            gib(need(bytes)),
            gib(available)
        ))
    };
    let decided = |dtype: DType, bytes: u64, fell_back: bool| DtypeDecision {
        choice,
        dtype,
        available_bytes: available,
        required_bytes: need(bytes),
        f32_required_bytes: fell_back.then(|| need(plan.f32_bytes)),
    };
    match choice {
        DTypeChoice::F32 if fits(plan.f32_bytes) => Ok(decided(DType::F32, plan.f32_bytes, false)),
        DTypeChoice::F32 => Err(refuse(plan.f32_bytes, "f32")),
        DTypeChoice::Bf16 if fits(plan.bf16_bytes) => {
            Ok(decided(DType::BF16, plan.bf16_bytes, false))
        }
        DTypeChoice::Bf16 => Err(refuse(plan.bf16_bytes, "bf16")),
        DTypeChoice::Auto if fits(plan.f32_bytes) => Ok(decided(DType::F32, plan.f32_bytes, false)),
        DTypeChoice::Auto if fits(plan.bf16_bytes) => {
            Ok(decided(DType::BF16, plan.bf16_bytes, true))
        }
        DTypeChoice::Auto => Err(refuse(plan.bf16_bytes, "bf16")),
    }
}

/// [`dtype_decision`], its fallback warning added to `warnings`.
pub(crate) fn choose_dtype(
    choice: DTypeChoice,
    plan: MemoryPlan,
    probe: &dyn HardwareProbe,
    warnings: &mut Vec<String>,
) -> Result<DType> {
    let decision = dtype_decision(choice, plan, probe)?;
    warnings.extend(decision.fallback());
    Ok(decision.dtype)
}

pub(crate) fn dtype_name(dtype: DType) -> &'static str {
    match dtype {
        DType::BF16 => "bf16",
        _ => "f32",
    }
}

/// Opens the model directory and runs `mean_diff`.
pub fn run_extraction(
    req: &ExtractRequest,
    probe: &dyn HardwareProbe,
    observer: &mut dyn ExtractObserver,
) -> Result<ExtractReport> {
    let never = AtomicBool::new(false);
    let mean_diff = Builtin(Method::MeanDiff);
    run_extraction_plugin(req, &mean_diff, &Value::Null, probe, observer, &never)
}

/// [`run_extraction`] with any method of the extension API and its `params` (`null`: the
/// method's defaults). `cancel` is handed to the method and checked after it; the observer
/// still stops the run between tensors and between prompts.
pub fn run_extraction_plugin(
    req: &ExtractRequest,
    method: &dyn ExtractionMethod,
    params: &Value,
    probe: &dyn HardwareProbe,
    observer: &mut dyn ExtractObserver,
    cancel: &AtomicBool,
) -> Result<ExtractReport> {
    validate_request(req)?;
    validate_method(req, method, params)?;
    let source = SafetensorsDir::open(&req.model_dir)?;
    run_pipeline(req, method, params, &source, probe, observer, cancel)
}

/// Checked before any computation: the parameters, pairing, the declared sidecars.
fn validate_method(
    req: &ExtractRequest,
    method: &dyn ExtractionMethod,
    params: &Value,
) -> Result<MethodInfo> {
    method.validate(params)?;
    let info = method.describe();
    if info.requires_paired && req.pos_prompts.len() != req.neg_prompts.len() {
        return Err(ExtractError::UnpairedPrompts {
            pos: req.pos_prompts.len(),
            neg: req.neg_prompts.len(),
        });
    }
    for suffix in &info.sidecars {
        if !is_valid_id(suffix) {
            let message = format!("suffixe de fichier annexe invalide : {suffix:?}");
            return Err(PluginError::InvalidOutput(message).into());
        }
        let path = sidecar_path(&req.out, suffix);
        if path.exists() && !req.overwrite {
            return Err(ExtractError::OutputExists(path));
        }
    }
    Ok(info)
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

/// `mean_diff` on any tensor source.
pub fn run_extraction_with(
    req: &ExtractRequest,
    source: &dyn TensorSource,
    probe: &dyn HardwareProbe,
    observer: &mut dyn ExtractObserver,
) -> Result<ExtractReport> {
    let never = AtomicBool::new(false);
    let mean_diff = Builtin(Method::MeanDiff);
    run_pipeline(
        req,
        &mean_diff,
        &Value::Null,
        source,
        probe,
        observer,
        &never,
    )
}

/// validate -> config -> tensor index check -> resources -> tokenizer -> minimal weights
/// -> forward pos then neg -> method -> statistics -> atomic write. Each step fails before
/// the next one touches anything heavier.
fn run_pipeline(
    req: &ExtractRequest,
    method: &dyn ExtractionMethod,
    params: &Value,
    source: &dyn TensorSource,
    probe: &dyn HardwareProbe,
    observer: &mut dyn ExtractObserver,
    cancel: &AtomicBool,
) -> Result<ExtractReport> {
    let started = Instant::now();
    validate_request(req)?;
    let info = validate_method(req, method, params)?;
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
    let decision = dtype_decision(req.dtype, plan_memory(&cfg, totals, req.max_tokens), probe)?;
    warnings.extend(decision.fallback());
    let dtype = decision.dtype;
    let tensor_bytes = totals.elements * dtype.size_in_bytes() as u64;
    let dtype_used = dtype_name(dtype).to_owned();
    let planned = ExtractEvent::Planned {
        dtype_used: dtype_used.clone(),
        weight_bytes: tensor_bytes,
        tensors: required.len(),
    };
    for event in [planned, ExtractEvent::Note(decision.note())] {
        if observer.on_event(event).is_break() {
            return Err(ExtractError::Cancelled);
        }
    }

    let tokenizer = load_tokenizer(&req.model_dir)?;
    let format = req
        .chat_template
        .resolve(&req.model_dir, ChatTemplate::NoTemplate);
    let mut truncated = 0;
    let mut encode_set = |set: PromptSet, prompts: &[String]| -> Result<Vec<Vec<u32>>> {
        prompts
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let t = tokenize(&tokenizer, &format.apply(p), req.max_tokens)?;
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
    let model = forward::build(
        &cfg,
        tensors,
        forward::Depth::UpTo(req.layer),
        dtype,
        &device,
    )?;

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

    let ctx = ExtractContext {
        hidden_size: cfg.hidden_size,
        layer: req.layer,
        site: req.site,
        pooling: req.pooling,
        params,
    };
    let acts = ActivationSet {
        pos: &pos,
        neg: &neg,
    };
    let out = method.extract(&ctx, &acts, cancel)?;
    if cancel.load(Ordering::SeqCst) {
        return Err(ExtractError::Cancelled);
    }
    check_direction(&out.direction, cfg.hidden_size)?;
    let separation = separation_along(&pos, &neg, &out.direction)?;
    let Separation {
        direction,
        stats,
        pos_projections,
        neg_projections,
    } = separation;
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
    warnings.extend(out.warnings);
    let holdout = holdout_report(method, &ctx, &pos, &neg, cancel, &mut warnings);

    if observer.on_event(ExtractEvent::Writing).is_break() {
        return Err(ExtractError::Cancelled);
    }
    let mut metadata = file_metadata(
        req,
        format.template,
        &cfg,
        &dtype_used,
        pos.len(),
        neg.len(),
        &stats,
    );
    // Autotune replays the exact wrapping of the extraction from these two keys.
    if let Some(system) = &format.system {
        metadata.insert("chat_system".into(), system.clone());
    }
    let method_meta = method_metadata(method.id(), &out.metadata)?;
    merge_metadata(&mut metadata, &method_meta)?;
    let method_diagnostics = out.diagnostics.map(|d| match d {
        Diagnostics::Json(mut map) => {
            map.insert("method".into(), method.id().into());
            Diagnostics::Json(map)
        }
        builtin => builtin,
    });
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
        method_diagnostics: method_diagnostics.as_ref(),
    };
    let mut companion_bytes =
        serde_json::to_vec_pretty(&companion).map_err(|e| ExtractError::Compute(e.to_string()))?;
    companion_bytes.push(b'\n');
    let main_bytes = encode(&direction, &metadata)?;
    let provenance: BTreeMap<String, String> = [
        ("n_pos", pos.len().to_string()),
        ("n_neg", neg.len().to_string()),
        ("layer", req.layer.to_string()),
        ("site", req.site.as_str().into()),
        ("pooling", req.pooling.as_str().into()),
        ("model", model_name(&req.model_dir)),
        ("surgeon_version", env!("CARGO_PKG_VERSION").into()),
    ]
    .map(|(k, v)| (k.to_owned(), v))
    .into();
    let mut sidecars: Vec<(&str, PathBuf, Vec<u8>)> = Vec::new();
    for sidecar in &out.sidecars {
        let header = sidecar_header(&info, sidecar, &method_meta, &provenance, &sidecars)?;
        // A tensor listed in `sidecar_rows` holds that many directions, one per row.
        let rows_of = |name: &String| {
            let key = (sidecar.suffix.to_owned(), name.clone());
            out.sidecar_rows.get(&key).copied().unwrap_or(0)
        };
        let tensors: Vec<(&str, &[f32], usize)> = sidecar
            .tensors
            .iter()
            .map(|(name, values)| (name.as_str(), values.as_slice(), rows_of(name)))
            .collect();
        if let Some((name, values, rows)) = tensors
            .iter()
            .find(|(_, values, rows)| *rows > 0 && values.len() != rows * cfg.hidden_size)
        {
            return Err(invalid_output(format!(
                "tenseur {name} : {} valeurs pour {rows} directions de taille {}",
                values.len(),
                cfg.hidden_size
            )));
        }
        let bytes = encode_shaped(&tensors, &header)?;
        sidecars.push((
            sidecar.suffix,
            sidecar_path(&req.out, sidecar.suffix),
            bytes,
        ));
    }
    match sidecars.is_empty() {
        true => write_pair_atomic(&req.out, &main_bytes, &companion_bytes, req.overwrite)?,
        false => {
            let companion = companion_path(&req.out);
            let mut files: Vec<(&Path, &[u8])> =
                vec![(&req.out, &main_bytes), (&companion, &companion_bytes)];
            files.extend(sidecars.iter().map(|(_, p, b)| (p.as_path(), b.as_slice())));
            write_files_atomic(&files, req.overwrite)?;
        }
    }
    let leace_out = sidecars
        .iter()
        .find(|s| s.0 == "leace")
        .map(|s| s.1.clone());
    let sidecar_paths = sidecars.into_iter().filter(|s| s.0 != "leace").map(|s| s.1);

    Ok(ExtractReport {
        out_path: req.out.clone(),
        stats_path: companion_path(&req.out),
        hidden_size: cfg.hidden_size,
        layer: req.layer,
        n_pos: pos.len(),
        n_neg: neg.len(),
        dtype_used,
        stats,
        pos_projections,
        neg_projections,
        holdout,
        warnings,
        duration_ms: started.elapsed().as_millis(),
        method: method.id(),
        method_diagnostics,
        leace_path: leace_out,
        sidecar_paths: sidecar_paths.collect(),
    })
}

/// The split of `holdout_frac` (from the method parameters), and the method fit again on
/// the kept prompts then measured on the held-out ones. A refit the method refuses (labels
/// given per prompt, say) leaves `stats` null with a warning.
fn holdout_report(
    method: &dyn ExtractionMethod,
    ctx: &ExtractContext,
    pos: &[Vec<f64>],
    neg: &[Vec<f64>],
    cancel: &AtomicBool,
    warnings: &mut Vec<String>,
) -> HoldoutReport {
    let frac = ctx
        .params
        .get("holdout_frac")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let active = frac > 0.0 && pos.len().min(neg.len()) >= MIN_HOLDOUT_PER_CLASS;
    let ignored = format!(
        "--holdout-frac ignoré : au moins {MIN_HOLDOUT_PER_CLASS} prompts par classe requis"
    );
    if frac > 0.0 && !active && !warnings.contains(&ignored) {
        warnings.push(ignored);
    }
    let split = |rows: &[Vec<f64>]| {
        let mask = holdout_mask(rows.len(), if active { frac } else { 0.0 });
        let (held, train): (Vec<_>, Vec<_>) = rows.iter().zip(mask).partition(|(_, h)| *h);
        let rows = |v: Vec<(&Vec<f64>, bool)>| v.into_iter().map(|(r, _)| r.clone()).collect();
        (rows(train), rows(held))
    };
    let ((pos_train, pos_held), (neg_train, neg_held)): ((Vec<_>, Vec<_>), (Vec<_>, Vec<_>)) =
        (split(pos), split(neg));
    let stats = active
        .then(|| -> Result<SeparationStats> {
            // The refit must not hold out again inside its own training part.
            let mut params = ctx.params.clone();
            if let Some(p) = params.as_object_mut() {
                p.insert("holdout_frac".into(), 0.0.into());
            }
            let train_ctx = ExtractContext {
                params: &params,
                ..*ctx
            };
            let acts = ActivationSet {
                pos: &pos_train,
                neg: &neg_train,
            };
            let fit = method.extract(&train_ctx, &acts, cancel)?;
            Ok(separation_along(&pos_held, &neg_held, &fit.direction)?.stats)
        })
        .and_then(|r| {
            r.map_err(|e| warnings.push(format!("hold-out non mesuré : {e}")))
                .ok()
        });
    HoldoutReport {
        frac,
        stats_basis: "all_prompts",
        n_train_pos: pos_train.len(),
        n_train_neg: neg_train.len(),
        n_holdout_pos: pos_held.len(),
        n_holdout_neg: neg_held.len(),
        stats,
    }
}

fn invalid_output(message: String) -> ExtractError {
    PluginError::InvalidOutput(message).into()
}

/// `hidden_size` finite values of L2 norm 1 (within 1e-6).
fn check_direction(direction: &[f64], hidden_size: usize) -> Result<()> {
    if direction.len() != hidden_size {
        let size = direction.len();
        return Err(invalid_output(format!(
            "direction de taille {size}, attendu {hidden_size}"
        )));
    }
    let norm = direction.iter().map(|x| x * x).sum::<f64>().sqrt();
    match direction.iter().all(|x| x.is_finite()) && (norm - 1.0).abs() <= 1e-6 {
        true => Ok(()),
        false => Err(invalid_output(format!(
            "direction non unitaire ou non finie (‖v‖ = {norm})"
        ))),
    }
}

/// The method's header entries plus `method = <id>`, except for `mean_diff`: its files stay
/// byte-identical to prompt 8 (no `method` key means mean_diff).
fn method_metadata(
    id: &'static str,
    extra: &BTreeMap<String, String>,
) -> Result<BTreeMap<String, String>> {
    if extra.contains_key("method") {
        return Err(invalid_output(
            "la clé `method` est écrite par le moteur".into(),
        ));
    }
    let mut meta = extra.clone();
    if id != Method::MeanDiff.as_str() {
        meta.insert("method".into(), id.into());
    }
    Ok(meta)
}

/// Adds `from` to `into`, refusing a key already there.
fn merge_metadata(
    into: &mut BTreeMap<String, String>,
    from: &BTreeMap<String, String>,
) -> Result<()> {
    for (key, value) in from {
        if into.insert(key.clone(), value.clone()).is_some() {
            return Err(invalid_output(format!(
                "clé de métadonnées déjà écrite : {key}"
            )));
        }
    }
    Ok(())
}

/// Header of one sidecar: method metadata, provenance, then its own entries. The suffix
/// must be declared and appear once; tensor names must be unique and non-empty.
fn sidecar_header(
    info: &MethodInfo,
    sidecar: &crate::plugin::Sidecar,
    method_meta: &BTreeMap<String, String>,
    provenance: &BTreeMap<String, String>,
    done: &[(&str, PathBuf, Vec<u8>)],
) -> Result<BTreeMap<String, String>> {
    let suffix = sidecar.suffix;
    if !info.sidecars.contains(&suffix) || done.iter().any(|d| d.0 == suffix) {
        return Err(invalid_output(format!(
            "fichier annexe non déclaré ou en double : {suffix}"
        )));
    }
    let mut names = BTreeSet::new();
    let bad_name = |n: &str| n.is_empty() || n == "__metadata__";
    if sidecar
        .tensors
        .iter()
        .any(|(n, _)| bad_name(n) || !names.insert(n.as_str()))
    {
        return Err(invalid_output(format!(
            "noms de tenseurs vides, réservés ou en double dans le fichier annexe {suffix}"
        )));
    }
    let mut header = method_meta.clone();
    merge_metadata(&mut header, provenance)?;
    merge_metadata(&mut header, &sidecar.metadata)?;
    Ok(header)
}

fn file_metadata(
    req: &ExtractRequest,
    template: ChatTemplate,
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
        ("chat_template", template.as_str().into()),
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
    /// svd / leace only: absent for mean_diff, so its companion is unchanged.
    #[serde(skip_serializing_if = "Option::is_none")]
    method_diagnostics: Option<&'a Diagnostics>,
}

/// Name of the model directory (absolute form first, so `.` still gets a name).
pub fn model_name(model_dir: &Path) -> String {
    std::path::absolute(model_dir)
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_else(|| model_dir.display().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::{GpuInfo, MemoryInfo};

    /// A machine with this much free RAM.
    struct Free(u64);

    impl HardwareProbe for Free {
        fn system_ram(&self) -> crate::hardware::Result<MemoryInfo> {
            Ok(MemoryInfo {
                total_bytes: 64 << 30,
                available_bytes: self.0,
            })
        }
        fn gpus(&self) -> crate::hardware::Result<Vec<GpuInfo>> {
            Ok(Vec::new())
        }
    }

    const GIB: u64 = 1 << 30;

    #[test]
    fn every_dtype_decision_is_logged_and_a_fallback_says_why() {
        let plan = MemoryPlan {
            f32_bytes: 4 * GIB,
            bf16_bytes: 2 * GIB,
        };
        let policy = BudgetPolicy::default();
        let f32_need = required_ram(plan.f32_bytes, &policy);
        let bf16_need = required_ram(plan.bf16_bytes, &policy);

        // Room for f32: f32, an info line, no warning.
        let roomy = dtype_decision(DTypeChoice::Auto, plan, &Free(f32_need)).unwrap();
        assert_eq!(roomy.dtype, DType::F32);
        assert_eq!(roomy.fallback(), None);
        let note = roomy.note();
        assert!(note.starts_with("dtype f32 (auto) : RAM libre"), "{note}");
        assert!(note.contains(&gib(f32_need)), "the need: {note}");

        // Free RAM short of f32 at that moment: bf16, and the warning gives both numbers.
        let short = dtype_decision(DTypeChoice::Auto, plan, &Free(f32_need - 1)).unwrap();
        assert_eq!(short.dtype, DType::BF16);
        assert_eq!(short.required_bytes, bf16_need);
        let warning = short.fallback().unwrap();
        assert!(warning.starts_with("repli en bf16"), "{warning}");
        assert!(warning.contains(&gib(f32_need)) && warning.contains(&gib(f32_need - 1)));
        assert!(warning.contains("2,5 fois plus lent"), "{warning}");
        let mut warnings = Vec::new();
        let dtype = choose_dtype(DTypeChoice::Auto, plan, &Free(f32_need - 1), &mut warnings);
        assert_eq!((dtype.unwrap(), warnings), (DType::BF16, vec![warning]));

        // An explicit choice is never a fallback; nothing that fits is an error.
        let asked = dtype_decision(DTypeChoice::Bf16, plan, &Free(64 * GIB)).unwrap();
        assert!(asked.note().contains("(demandé)") && asked.fallback().is_none());
        assert!(dtype_decision(DTypeChoice::F32, plan, &Free(f32_need - 1)).is_err());
        assert!(dtype_decision(DTypeChoice::Auto, plan, &Free(bf16_need - 1)).is_err());
    }
}
