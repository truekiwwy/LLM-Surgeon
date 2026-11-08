//! `config.json` of a Qwen2 / Qwen2.5 checkpoint.

use std::path::Path;

use candle_nn::Activation;
use serde::Deserialize;

use super::error::{ExtractError, Result};

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Qwen2Config {
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
    /// candle defaults to GELU; Qwen2 checkpoints use SiLU.
    #[serde(default = "default_act")]
    pub hidden_act: Activation,
}

fn default_act() -> Activation {
    Activation::Silu
}

impl Qwen2Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// candle's reference applies the window whatever `use_sliding_window` says; with
    /// prompts capped by `--max-tokens` it never bites on real Qwen2 configs.
    pub fn sliding_window(&self) -> usize {
        self.sliding_window.unwrap_or(self.max_position_embeddings)
    }
}

pub fn load_config(model_dir: &Path) -> Result<Qwen2Config> {
    let path = model_dir.join("config.json");
    let json = std::fs::read_to_string(&path)
        .map_err(|e| ExtractError::Config(format!("{} : {e}", path.display())))?;
    parse_config(&json)
}

/// The architecture is checked first, so a Qwen3 config reports its type rather than a
/// missing field.
pub fn parse_config(json: &str) -> Result<Qwen2Config> {
    #[derive(Deserialize)]
    struct Probe {
        model_type: Option<String>,
    }
    let probe: Probe =
        serde_json::from_str(json).map_err(|e| ExtractError::Config(e.to_string()))?;
    match probe.model_type.as_deref() {
        Some("qwen2") => {}
        Some(other) => {
            return Err(ExtractError::UnsupportedArchitecture {
                model_type: other.to_owned(),
            })
        }
        None => return Err(ExtractError::Config("champ model_type manquant".into())),
    }
    let cfg: Qwen2Config =
        serde_json::from_str(json).map_err(|e| ExtractError::Config(e.to_string()))?;
    let invalid = |message: &str| Err(ExtractError::Config(message.to_owned()));
    if cfg.num_hidden_layers == 0 || cfg.hidden_size == 0 || cfg.vocab_size == 0 {
        return invalid("num_hidden_layers, hidden_size et vocab_size doivent être > 0");
    }
    if cfg.num_attention_heads == 0 || !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) {
        return invalid("hidden_size doit être un multiple de num_attention_heads");
    }
    if cfg.num_key_value_heads == 0
        || !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
    {
        return invalid("num_attention_heads doit être un multiple de num_key_value_heads");
    }
    if !cfg.head_dim().is_multiple_of(2) {
        return invalid("la dimension de tête doit être paire (RoPE)");
    }
    if !(cfg.rms_norm_eps.is_finite() && cfg.rms_norm_eps > 0.0)
        || !(cfg.rope_theta.is_finite() && cfg.rope_theta > 0.0)
    {
        return invalid("rms_norm_eps et rope_theta doivent être finis et positifs");
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
        let no_act = parse_config(&QWEN2.replace(r#","hidden_act":"silu""#, "")).unwrap();
        assert_eq!(no_act.hidden_act, Activation::Silu);
    }

    #[test]
    fn other_architectures_and_broken_configs_are_refused() {
        let qwen3 = parse_config(&QWEN2.replace("qwen2", "qwen3")).unwrap_err();
        assert!(
            matches!(&qwen3, ExtractError::UnsupportedArchitecture { model_type } if model_type == "qwen3")
        );
        assert!(qwen3.to_string().contains("qwen3") && qwen3.to_string().contains("Qwen2"));
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
