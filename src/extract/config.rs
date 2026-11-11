//! `config.json` of a dense decoder with a native forward: Qwen2 / Qwen2.5, Llama 2/3,
//! Mistral and Qwen3 (the ids of `arch::FORWARD_IDS`). They share the HF layout
//! (`model.layers.{i}`, pre-norm RMSNorm, rotate-half RoPE, GQA, SwiGLU) and differ by:
//!
//! ```text
//!            q/k/v bias       o_proj bias      MLP bias   q_norm/k_norm   head_dim
//! qwen2      always           never            never      no              hidden / heads
//! llama      attention_bias   attention_bias   mlp_bias   no              head_dim or hidden / heads
//! mistral    never            never            never      no              head_dim or hidden / heads
//! qwen3      attention_bias   attention_bias   never      yes (head_dim)  head_dim or hidden / heads
//! ```
//!
//! (transformers `modeling_{qwen2,llama,mistral,qwen3}.py`, read on 2026-11-11.) RoPE
//! scaling: none, or Llama 3's `rope_type: "llama3"` (as candle's `llama.rs`); any other
//! type is refused rather than run wrong. MoE variants and other architectures are
//! `unsupported_architecture`: their weights can still be patched (`arch`), not run here.

use std::path::Path;

use candle_nn::Activation;
use serde::Deserialize;

use super::error::{ExtractError, Result};

/// `model_type` values with a forward.
pub const SUPPORTED_MODEL_TYPES: [&str; 4] = ["qwen2", "llama", "mistral", "qwen3"];

/// Llama 3 frequency scaling (`rope_scaling` with `rope_type: "llama3"`).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct Llama3Scaling {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DecoderConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Missing in some exports: no window then, as if it spanned every position.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// candle defaults to GELU; these checkpoints use SiLU.
    #[serde(default = "default_act")]
    pub hidden_act: Activation,
    /// Llama, Mistral, Qwen3: set when it differs from `hidden_size / num_attention_heads`.
    #[serde(default)]
    pub head_dim: Option<usize>,
    /// Llama, Qwen3: biases on q/k/v and o_proj.
    #[serde(default)]
    pub attention_bias: bool,
    /// Llama: biases on the three MLP projections.
    #[serde(default)]
    pub mlp_bias: bool,
    /// Parsed from `rope_scaling` by [`parse_config`].
    #[serde(skip)]
    pub rope_llama3: Option<Llama3Scaling>,
}

/// The historical name, kept for the library API: every decoder of [`SUPPORTED_MODEL_TYPES`].
pub type Qwen2Config = DecoderConfig;

/// What a forward builds, from the model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variant {
    pub qkv_bias: bool,
    pub o_bias: bool,
    pub mlp_bias: bool,
    pub qk_norm: bool,
}

fn default_act() -> Activation {
    Activation::Silu
}

impl DecoderConfig {
    pub fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }

    /// candle's reference applies the window whatever `use_sliding_window` says; with
    /// prompts capped by `--max-tokens` it never bites on real Qwen2 configs.
    pub fn sliding_window(&self) -> usize {
        self.sliding_window.unwrap_or(self.max_position_embeddings)
    }

    pub fn variant(&self) -> Variant {
        let bias = self.attention_bias;
        match self.model_type.as_str() {
            "qwen2" => Variant {
                qkv_bias: true,
                o_bias: false,
                mlp_bias: false,
                qk_norm: false,
            },
            "qwen3" => Variant {
                qkv_bias: bias,
                o_bias: bias,
                mlp_bias: false,
                qk_norm: true,
            },
            "mistral" => Variant {
                qkv_bias: false,
                o_bias: false,
                mlp_bias: false,
                qk_norm: false,
            },
            // llama
            _ => Variant {
                qkv_bias: bias,
                o_bias: bias,
                mlp_bias: self.mlp_bias,
                qk_norm: false,
            },
        }
    }
}

pub fn load_config(model_dir: &Path) -> Result<DecoderConfig> {
    let path = model_dir.join("config.json");
    let json = std::fs::read_to_string(&path)
        .map_err(|e| ExtractError::Config(format!("{} : {e}", path.display())))?;
    parse_config(&json)
}

/// `rope_scaling`: absent, null, `default`, or `llama3`.
fn parse_rope_scaling(value: Option<&serde_json::Value>) -> Result<Option<Llama3Scaling>> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let kind = value
        .get("rope_type")
        .or_else(|| value.get("type"))
        .and_then(|v| v.as_str())
        .unwrap_or("default");
    match kind {
        "default" => Ok(None),
        "llama3" => serde_json::from_value(value.clone())
            .map(Some)
            .map_err(|e| ExtractError::Config(format!("rope_scaling llama3 : {e}"))),
        other => Err(ExtractError::Config(format!(
            "rope_scaling « {other} » non supporté (seuls default et llama3)"
        ))),
    }
}

/// The architecture is checked first, so an unsupported config reports its type rather
/// than a missing field.
pub fn parse_config(json: &str) -> Result<DecoderConfig> {
    let value: serde_json::Value =
        serde_json::from_str(json).map_err(|e| ExtractError::Config(e.to_string()))?;
    match value.get("model_type").and_then(|v| v.as_str()) {
        Some(t) if SUPPORTED_MODEL_TYPES.contains(&t) => {}
        Some(other) => {
            return Err(ExtractError::UnsupportedArchitecture {
                model_type: other.to_owned(),
            })
        }
        None => return Err(ExtractError::Config("champ model_type manquant".into())),
    }
    let mut cfg: DecoderConfig =
        serde_json::from_value(value.clone()).map_err(|e| ExtractError::Config(e.to_string()))?;
    cfg.rope_llama3 = parse_rope_scaling(value.get("rope_scaling"))?;
    let invalid = |message: &str| Err(ExtractError::Config(message.to_owned()));
    if cfg.num_hidden_layers == 0 || cfg.hidden_size == 0 || cfg.vocab_size == 0 {
        return invalid("num_hidden_layers, hidden_size et vocab_size doivent être > 0");
    }
    if cfg.num_attention_heads == 0
        || (cfg.head_dim.is_none() && !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads))
    {
        return invalid("hidden_size doit être un multiple de num_attention_heads");
    }
    if cfg.num_key_value_heads == 0
        || !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
    {
        return invalid("num_attention_heads doit être un multiple de num_key_value_heads");
    }
    if cfg.head_dim() == 0 || !cfg.head_dim().is_multiple_of(2) {
        return invalid("la dimension de tête doit être paire (RoPE)");
    }
    if !(cfg.rms_norm_eps.is_finite() && cfg.rms_norm_eps > 0.0)
        || !(cfg.rope_theta.is_finite() && cfg.rope_theta > 0.0)
    {
        return invalid("rms_norm_eps et rope_theta doivent être finis et positifs");
    }
    if let Some(s) = cfg.rope_llama3 {
        let finite = [s.factor, s.low_freq_factor, s.high_freq_factor]
            .iter()
            .all(|x| x.is_finite() && *x > 0.0);
        if !finite
            || s.high_freq_factor <= s.low_freq_factor
            || s.original_max_position_embeddings == 0
        {
            return invalid(
                "rope_scaling llama3 : facteurs positifs, high_freq_factor > low_freq_factor",
            );
        }
    }
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    const QWEN2: &str = r#"{"model_type":"qwen2","vocab_size":64,"hidden_size":16,
        "intermediate_size":32,"num_hidden_layers":4,"num_attention_heads":2,
        "num_key_value_heads":2,"max_position_embeddings":128,"rms_norm_eps":1e-6,
        "rope_theta":10000.0,"tie_word_embeddings":false,"hidden_act":"silu"}"#;

    #[test]
    fn qwen2_parses_with_defaults() {
        let cfg = parse_config(QWEN2).unwrap();
        assert_eq!(
            (cfg.hidden_size, cfg.head_dim(), cfg.sliding_window()),
            (16, 8, 128)
        );
        assert_eq!(cfg.hidden_act, Activation::Silu);
        assert!(cfg.variant().qkv_bias && !cfg.variant().o_bias);
        let no_act = parse_config(&QWEN2.replace(r#","hidden_act":"silu""#, "")).unwrap();
        assert_eq!(no_act.hidden_act, Activation::Silu);
    }

    #[test]
    fn llama_mistral_and_qwen3_variants() {
        let llama = parse_config(&QWEN2.replace("qwen2", "llama")).unwrap();
        assert_eq!(
            llama.variant(),
            Variant {
                qkv_bias: false,
                o_bias: false,
                mlp_bias: false,
                qk_norm: false
            }
        );
        let qwen3 = parse_config(
            &QWEN2
                .replace("qwen2", "qwen3")
                .replace(r#""rms_norm_eps""#, r#""head_dim":12,"rms_norm_eps""#),
        )
        .unwrap();
        assert_eq!(qwen3.head_dim(), 12);
        assert!(qwen3.variant().qk_norm && !qwen3.variant().qkv_bias);
        let scaled = QWEN2.replace("qwen2", "llama").replace(
            r#""rope_theta":10000.0"#,
            r#""rope_theta":10000.0,"rope_scaling":{"rope_type":"llama3","factor":8.0,
               "low_freq_factor":1.0,"high_freq_factor":4.0,
               "original_max_position_embeddings":64}"#,
        );
        assert_eq!(
            parse_config(&scaled).unwrap().rope_llama3.unwrap().factor,
            8.0
        );
        let yarn = scaled.replace("llama3", "yarn");
        assert!(parse_config(&yarn)
            .unwrap_err()
            .to_string()
            .contains("yarn"));
    }

    #[test]
    fn other_architectures_and_broken_configs_are_refused() {
        let gemma = parse_config(&QWEN2.replace("qwen2", "gemma2")).unwrap_err();
        assert!(
            matches!(&gemma, ExtractError::UnsupportedArchitecture { model_type } if model_type == "gemma2")
        );
        assert!(gemma.to_string().contains("gemma2") && gemma.to_string().contains("Qwen2"));
        let moe = parse_config(&QWEN2.replace("qwen2", "qwen3_moe")).unwrap_err();
        assert!(matches!(moe, ExtractError::UnsupportedArchitecture { .. }));
        let missing = parse_config(&QWEN2.replace(r#""hidden_size":16,"#, "")).unwrap_err();
        assert!(missing.to_string().contains("hidden_size"), "{missing}");
        assert!(matches!(
            parse_config(r#"{"vocab_size":1}"#),
            Err(ExtractError::Config(_))
        ));
        assert!(parse_config(
            &QWEN2.replace(r#""num_key_value_heads":2"#, r#""num_key_value_heads":3"#)
        )
        .is_err());
        assert!(parse_config("{oops").is_err());
    }
}
