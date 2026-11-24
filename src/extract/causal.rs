use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};

use super::forward::{Depth, ResidualForward};
use super::{
    check_texts, load_config, open_session, resolve_layers, ChatTemplate, DTypeChoice,
    ExtractError, ExtractEvent, ExtractObserver, HookPoint, ModelInput, Result, SafetensorsDir,
    TensorSource,
};
use crate::hardware::HardwareProbe;
use crate::patch::LayerSelection;

pub const MIN_LOG_PROB_GAP: f64 = 1e-2;
pub const DEFAULT_WINDOW_THRESHOLD: f64 = 0.5;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Component {
    #[default]
    Resid,
    Attn,
    Mlp,
}

impl Component {
    pub fn as_str(self) -> &'static str {
        match self {
            Component::Resid => "resid",
            Component::Attn => "attn",
            Component::Mlp => "mlp",
        }
    }

    fn hook_point(self) -> HookPoint {
        match self {
            Component::Resid => HookPoint::ResidPost,
            Component::Attn => HookPoint::AttnOut,
            Component::Mlp => HookPoint::MlpOut,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TraceRequest {
    pub model_dir: PathBuf,
    pub clean_prompts: Vec<String>,
    pub corrupted_prompts: Vec<String>,
    pub component: Component,
    pub layers: LayerSelection,
    pub target: Option<String>,
    pub window_threshold: f64,
    pub chat_template: ChatTemplate,
    pub max_tokens: usize,
    pub dtype: DTypeChoice,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LayerEffect {
    pub layer: usize,
    pub score: f64,
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CausalWindow {
    pub start: usize,
    pub end: usize,
    pub layers: String,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TraceReport {
    pub component: Component,
    pub target: Option<String>,
    pub n_pairs: usize,
    pub pairs_used: usize,
    pub clean_log_prob: f64,
    pub corrupted_log_prob: f64,
    pub layers: Vec<LayerEffect>,
    pub ranking: Vec<usize>,
    pub peak_layer: usize,
    pub window: CausalWindow,
    pub window_threshold: f64,
    pub dtype_used: String,
    pub forwards: usize,
    pub warnings: Vec<String>,
    pub duration_ms: u128,
}

pub fn log_prob(logits: &[f32], token: usize) -> f64 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let sum: f64 = logits.iter().map(|&x| (x as f64 - max).exp()).sum();
    logits[token] as f64 - max - sum.ln()
}

pub fn argmax(logits: &[f32]) -> usize {
    (0..logits.len())
        .reduce(|best, i| match logits[i] > logits[best] {
            true => i,
            false => best,
        })
        .unwrap_or(0)
}

pub fn ranking(effects: &[LayerEffect]) -> Vec<usize> {
    let mut order: Vec<&LayerEffect> = effects.iter().collect();
    order.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.layer.cmp(&b.layer)));
    order.into_iter().map(|e| e.layer).collect()
}

pub fn causal_window(effects: &[LayerEffect], threshold: f64) -> Option<CausalWindow> {
    let peak =
        (0..effects.len()).reduce(|best, i| match effects[i].score > effects[best].score {
            true => i,
            false => best,
        })?;
    let cut = threshold * effects[peak].score;
    let (mut lo, mut hi) = (peak, peak);
    while lo > 0 && effects[lo - 1].score >= cut {
        lo -= 1;
    }
    while hi + 1 < effects.len() && effects[hi + 1].score >= cut {
        hi += 1;
    }
    let (start, end) = (effects[lo].layer, effects[hi].layer);
    let layers = match start == end {
        true => start.to_string(),
        false => format!("{start}-{end}"),
    };
    Some(CausalWindow { start, end, layers })
}

struct PairTrace {
    clean: f64,
    corrupted: f64,
    patched: Vec<f64>,
}

fn last_row(t: &Tensor) -> Result<Tensor> {
    let seq = t.dim(1)?;
    Ok(t.narrow(1, seq - 1, 1)?.contiguous()?)
}

fn replace_last(t: &Tensor, row: &Tensor) -> Result<Tensor> {
    match t.dim(1)? {
        1 => Ok(row.clone()),
        seq => Ok(Tensor::cat(&[&t.narrow(1, 0, seq - 1)?, row], 1)?),
    }
}

fn ids_tensor(ids: &[u32]) -> Result<Tensor> {
    Ok(Tensor::new(ids, &Device::Cpu)?.unsqueeze(0)?)
}

fn trace_pair(
    model: &dyn ResidualForward,
    clean: &[u32],
    corrupted: &[u32],
    layers: &[usize],
    point: HookPoint,
    target: Option<usize>,
    tick: &mut dyn FnMut() -> Result<()>,
) -> Result<PairTrace> {
    let mut saved: Vec<Option<Tensor>> = vec![None; layers.len()];
    let hidden = model.forward_hooked(&ids_tensor(clean)?, &mut |layer, at, t| {
        if let (true, Some(k)) = (at == point, layers.iter().position(|&l| l == layer)) {
            saved[k] = Some(last_row(&t)?);
        }
        Ok(t)
    })?;
    tick()?;
    let clean_logits = model.logits_last(&hidden)?;
    let token = target.unwrap_or_else(|| argmax(&clean_logits));
    let corrupted_ids = ids_tensor(corrupted)?;
    let hidden = model.forward_hooked(&corrupted_ids, &mut |_, _, t| Ok(t))?;
    tick()?;
    let mut trace = PairTrace {
        clean: log_prob(&clean_logits, token),
        corrupted: log_prob(&model.logits_last(&hidden)?, token),
        patched: Vec::with_capacity(layers.len()),
    };
    if (trace.clean - trace.corrupted).abs() < MIN_LOG_PROB_GAP {
        return Ok(trace);
    }
    for (&layer, row) in layers.iter().zip(&saved) {
        let row = row
            .as_ref()
            .ok_or_else(|| ExtractError::Compute(format!("couche {layer} absente")))?;
        let hidden = model.forward_hooked(&corrupted_ids, &mut |l, at, t| match l == layer
            && at == point
        {
            true => replace_last(&t, row),
            false => Ok(t),
        })?;
        trace
            .patched
            .push(log_prob(&model.logits_last(&hidden)?, token));
        tick()?;
    }
    Ok(trace)
}

fn effects(layers: &[usize], restorations: &[Vec<f64>]) -> Vec<LayerEffect> {
    layers
        .iter()
        .enumerate()
        .map(|(k, &layer)| {
            let values: Vec<f64> = restorations.iter().map(|r| r[k]).collect();
            LayerEffect {
                layer,
                score: values.iter().sum::<f64>() / values.len().max(1) as f64,
                min: values.iter().copied().fold(f64::INFINITY, f64::min),
                max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            }
        })
        .collect()
}

fn validate(req: &TraceRequest) -> Result<()> {
    for (set, prompts) in [
        ("clean", &req.clean_prompts),
        ("corrupted", &req.corrupted_prompts),
    ] {
        check_texts(prompts.clone()).map_err(|message| ExtractError::Prompts {
            set: set.into(),
            message,
        })?;
    }
    if req.clean_prompts.len() != req.corrupted_prompts.len() {
        return Err(ExtractError::InvalidArgument(format!(
            "prompts non appariés : {} propres, {} corrompus ; il en faut autant, appariés par index",
            req.clean_prompts.len(),
            req.corrupted_prompts.len()
        )));
    }
    if req.max_tokens == 0 {
        return Err(ExtractError::InvalidArgument(
            "max_tokens doit être >= 1".into(),
        ));
    }
    if !(req.window_threshold > 0.0 && req.window_threshold <= 1.0) {
        return Err(ExtractError::InvalidArgument(format!(
            "window_threshold doit être dans ]0, 1], reçu {}",
            req.window_threshold
        )));
    }
    if req.target.as_deref().is_some_and(str::is_empty) {
        return Err(ExtractError::InvalidArgument("target vide".into()));
    }
    Ok(())
}

pub fn run_trace(
    req: &TraceRequest,
    probe: &dyn HardwareProbe,
    observer: &mut dyn ExtractObserver,
    cancel: &AtomicBool,
) -> Result<TraceReport> {
    validate(req)?;
    let source = SafetensorsDir::open(&req.model_dir)?;
    trace(req, &source, probe, observer, cancel)
}

fn trace(
    req: &TraceRequest,
    source: &dyn TensorSource,
    probe: &dyn HardwareProbe,
    observer: &mut dyn ExtractObserver,
    cancel: &AtomicBool,
) -> Result<TraceReport> {
    let started = Instant::now();
    let cfg = load_config(&req.model_dir)?;
    let layers = resolve_layers(&req.layers, cfg.num_hidden_layers)?;
    let input = ModelInput {
        model_dir: &req.model_dir,
        chat_template: req.chat_template,
        max_tokens: req.max_tokens,
        dtype: req.dtype,
    };
    let mut session = open_session(input, &cfg, Depth::Full, source, probe, observer)?;
    let clean = session.encode("clean", &req.clean_prompts)?;
    let corrupted = session.encode("corrupted", &req.corrupted_prompts)?;
    let target = match &req.target {
        None => None,
        Some(text) => {
            let encoding = session
                .tokenizer
                .encode(text.as_str(), false)
                .map_err(|e| ExtractError::Tokenizer(e.to_string()))?;
            let first = encoding.get_ids().first().copied().ok_or_else(|| {
                ExtractError::InvalidArgument(format!("target « {text} » : aucun token"))
            })?;
            Some(first as usize)
        }
    };
    let model = session.load(&cfg, source, observer)?;
    let point = req.component.hook_point();
    let total = clean.len() * (2 + layers.len());
    let mut done = 0;
    let mut tick = || -> Result<()> {
        done += 1;
        let event = ExtractEvent::Forward { done, total };
        match observer.on_event(event).is_break() || cancel.load(Ordering::SeqCst) {
            true => Err(ExtractError::Cancelled),
            false => Ok(()),
        }
    };
    let mut warnings = std::mem::take(&mut session.warnings);
    let mut restorations = Vec::new();
    let (mut clean_sum, mut corrupted_sum) = (0.0, 0.0);
    for (pair, (c, x)) in clean.iter().zip(&corrupted).enumerate() {
        let traced = trace_pair(model.as_ref(), c, x, &layers, point, target, &mut tick)?;
        let gap = traced.clean - traced.corrupted;
        if traced.patched.is_empty() {
            warnings.push(format!(
                "paire {pair} ignorée : log-prob de la cible identique en propre et en corrompu (écart {gap:.2e})"
            ));
            continue;
        }
        restorations.push(
            traced
                .patched
                .iter()
                .map(|p| (p - traced.corrupted) / gap)
                .collect::<Vec<f64>>(),
        );
        clean_sum += traced.clean;
        corrupted_sum += traced.corrupted;
    }
    let used = restorations.len();
    if used == 0 {
        return Err(ExtractError::InvalidArgument(
            "aucune paire ne change la cible entre prompt propre et corrompu : rien à tracer"
                .into(),
        ));
    }
    if session.truncated > 0 {
        warnings.push(format!(
            "{} prompt(s) tronqué(s) à {} tokens",
            session.truncated, req.max_tokens
        ));
    }
    let effects = effects(&layers, &restorations);
    let window = causal_window(&effects, req.window_threshold)
        .ok_or_else(|| ExtractError::InvalidArgument("aucune couche sélectionnée".into()))?;
    let ranking = ranking(&effects);
    Ok(TraceReport {
        component: req.component,
        target: req.target.clone(),
        n_pairs: clean.len(),
        pairs_used: used,
        clean_log_prob: clean_sum / used as f64,
        corrupted_log_prob: corrupted_sum / used as f64,
        peak_layer: ranking[0],
        ranking,
        layers: effects,
        window,
        window_threshold: req.window_threshold,
        dtype_used: session.dtype_used,
        forwards: done,
        warnings,
        duration_ms: started.elapsed().as_millis(),
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use candle_core::DType;

    use super::*;
    use crate::extract::{parse_config, required_tensors_full, HookedDecoder};

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self, scale: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        }
    }

    fn tiny_model() -> HookedDecoder {
        let cfg = parse_config(
            r#"{"model_type":"qwen2","vocab_size":32,"hidden_size":16,"intermediate_size":32,
            "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
            "max_position_embeddings":64,"rms_norm_eps":1e-6,"rope_theta":10000.0,
            "tie_word_embeddings":false,"sliding_window":64,"hidden_act":"silu"}"#,
        )
        .unwrap();
        let mut rng = Lcg(3);
        let tensors: HashMap<String, Tensor> = required_tensors_full(&cfg)
            .into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                let values: Vec<f32> = (0..n)
                    .map(|_| match name.contains("norm") {
                        true => 1.0 + rng.next(0.2),
                        false => rng.next(0.5),
                    })
                    .collect();
                (name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap())
            })
            .collect();
        HookedDecoder::from_tensors_full(tensors, &cfg, DType::F32, &Device::Cpu).unwrap()
    }

    fn effect(layer: usize, score: f64) -> LayerEffect {
        LayerEffect {
            layer,
            score,
            min: score,
            max: score,
        }
    }

    #[test]
    fn restoring_the_last_residual_restores_the_clean_output_and_ranks_first() {
        let model = tiny_model();
        let layers = [0, 1, 2, 3];
        let mut forwards = 0;
        let mut tick = || -> Result<()> {
            forwards += 1;
            Ok(())
        };
        let clean = [3, 9, 17, 5];
        let corrupted = [3, 22, 11, 5];
        let point = Component::Resid.hook_point();
        let traced =
            trace_pair(&model, &clean, &corrupted, &layers, point, None, &mut tick).unwrap();
        assert_eq!(forwards, 2 + layers.len());
        let gap = traced.clean - traced.corrupted;
        assert!(gap.abs() >= MIN_LOG_PROB_GAP, "gap {gap}");
        let scores: Vec<f64> = traced
            .patched
            .iter()
            .map(|p| (p - traced.corrupted) / gap)
            .collect();
        assert!((scores[3] - 1.0).abs() < 1e-4, "{scores:?}");
        assert!(scores.iter().all(|s| s.is_finite()));
        let effects = effects(&layers, &[scores.clone(), scores]);
        assert_eq!(ranking(&effects)[0], 3);
    }

    #[test]
    fn patching_attention_or_mlp_keeps_shapes_and_identical_prompts_are_skipped() {
        let model = tiny_model();
        let mut tick = || -> Result<()> { Ok(()) };
        for component in [Component::Attn, Component::Mlp] {
            let traced = trace_pair(
                &model,
                &[1, 2, 3],
                &[1, 7],
                &[1, 2],
                component.hook_point(),
                Some(4),
                &mut tick,
            )
            .unwrap();
            assert!(traced.patched.iter().all(|p| p.is_finite()));
        }
        let same = trace_pair(
            &model,
            &[1, 2, 3],
            &[1, 2, 3],
            &[0, 1],
            HookPoint::ResidPost,
            None,
            &mut tick,
        )
        .unwrap();
        assert!(same.patched.is_empty());
    }

    #[test]
    fn the_window_grows_around_the_peak_while_scores_stay_above_the_cut() {
        let effects = [
            effect(4, 0.1),
            effect(5, 0.6),
            effect(6, 0.9),
            effect(7, 0.5),
            effect(8, 0.2),
            effect(9, 0.7),
        ];
        let window = causal_window(&effects, 0.5).unwrap();
        assert_eq!((window.start, window.end), (5, 7));
        assert_eq!(window.layers, "5-7");
        assert_eq!(causal_window(&effects, 1.0).unwrap().layers, "6");
        assert_eq!(ranking(&effects), [6, 9, 5, 7, 8, 4]);
        assert!(causal_window(&[], 0.5).is_none());
    }

    #[test]
    fn log_prob_is_a_log_softmax() {
        let logits = [1.0f32, 2.0, 3.0];
        let total: f64 = (0..3).map(|t| log_prob(&logits, t).exp()).sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!((log_prob(&[0.0, 0.0], 1) - 0.5f64.ln()).abs() < 1e-12);
        assert_eq!(argmax(&logits), 2);
        assert_eq!(argmax(&[1.0, 1.0]), 0);
    }
}
