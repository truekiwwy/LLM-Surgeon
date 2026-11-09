//! `surgeon tui --preview ...` options (flattened into `TuiArgs`).

use clap::{Args, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum PreviewTemplate {
    None,
    Chatml,
}

#[derive(Debug, Clone, Args)]
pub struct PreviewArgs {
    /// Third tab: live generation with activation steering (needs --model and --vector).
    #[arg(long)]
    pub preview: bool,
    #[arg(long)]
    pub preview_prompt: Option<String>,
    /// Initial relative alpha (alpha_abs = alpha * mean residual norm at the layer).
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true)]
    pub preview_alpha: f64,
    #[arg(long, default_value_t = 30)]
    pub preview_tokens: usize,
    /// Layer of the injection (default: the vector's layer).
    #[arg(long)]
    pub preview_layer: Option<usize>,
    #[arg(long, value_enum, default_value_t = PreviewTemplate::Chatml)]
    pub preview_template: PreviewTemplate,
    /// Slider bounds MIN:MAX.
    #[arg(long, default_value = "-2:2", allow_hyphen_values = true, value_parser = parse_alpha_range)]
    pub alpha_range: (f64, f64),
}

/// `-2:2` -> `(-2.0, 2.0)`; finite bounds with min < max.
pub fn parse_alpha_range(value: &str) -> Result<(f64, f64), String> {
    let (lo, hi) = value
        .split_once(':')
        .ok_or_else(|| format!("MIN:MAX attendu, reçu {value:?}"))?;
    let parse = |s: &str| {
        s.trim()
            .parse::<f64>()
            .ok()
            .filter(|x| x.is_finite())
            .ok_or_else(|| format!("borne invalide {s:?}"))
    };
    let (lo, hi) = (parse(lo)?, parse(hi)?);
    match lo < hi {
        true => Ok((lo, hi)),
        false => Err(format!("MIN doit être inférieur à MAX ({lo} >= {hi})")),
    }
}
