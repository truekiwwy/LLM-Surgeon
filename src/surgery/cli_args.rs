//! clap argument structs of `surgeon diff` and of the `.surgery` options of `surgeon apply`.

use std::path::PathBuf;

use clap::{Args, ValueEnum};

use super::format::FactorDtype;

/// What `apply` does when the current base tensor differs from the one the patch was
/// computed against (`base_sha256`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum VerifyBase {
    /// Error, nothing written.
    Strict,
    /// Warning, then apply anyway.
    Warn,
    /// No check, no hashing.
    Off,
}

/// `diff` behavior on a tensor whose reconstruction error exceeds `--max-rel-error`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum OnHighError {
    Warn,
    /// Leave the tensor out of the patch.
    Skip,
    /// Store the exact delta (`full_delta`) instead of a low-rank approximation.
    Full,
}

/// `.surgery` options of `surgeon apply` (the patch is detected by its `SURG` magic).
#[derive(Debug, Clone, Args)]
pub struct SurgeryApplyArgs {
    /// .surgery only: multiplier of the stored delta, `W <- W + scale * delta`.
    #[arg(long, default_value_t = 1.0, allow_hyphen_values = true)]
    pub scale: f64,
    /// .surgery only: check each base tensor against the hash stored in the patch.
    #[arg(long, value_enum, default_value_t = VerifyBase::Warn)]
    pub verify_base: VerifyBase,
}

/// Exit codes: 0 ok, 1 cancelled or unreadable model, 2 invalid arguments or checkpoints
/// (existing `--out` included), 3 execution failed, 70 internal.
#[derive(Debug, Clone, Args)]
pub struct DiffArgs {
    /// Base checkpoint: .safetensors file or sharded model directory.
    #[arg(long)]
    pub base: PathBuf,
    /// Fine-tuned checkpoint, same tensor names, shapes and dtypes.
    #[arg(long)]
    pub tuned: PathBuf,
    /// .surgery file to write.
    #[arg(long)]
    pub out: PathBuf,
    /// Rank of the low-rank approximation of each 2D delta.
    #[arg(long, default_value_t = 16)]
    pub rank: usize,
    /// Storage dtype of the factors (falls back to bf16 with a warning on f16 overflow).
    #[arg(long, value_enum, default_value_t = FactorDtype::F16)]
    pub factor_dtype: FactorDtype,
    /// Only tensors whose name matches this regex.
    #[arg(long)]
    pub include: Option<String>,
    /// Skip tensors whose name matches this regex.
    #[arg(long)]
    pub exclude: Option<String>,
    /// Also diff embed_tokens and lm_head (excluded by default: huge).
    #[arg(long)]
    pub include_embeddings: bool,
    /// Relative reconstruction error above which --on-high-error applies.
    #[arg(long)]
    pub max_rel_error: Option<f64>,
    #[arg(long, value_enum, default_value_t = OnHighError::Warn)]
    pub on_high_error: OnHighError,
    /// Extra random directions of the randomized SVD.
    #[arg(long, default_value_t = 8)]
    pub oversample: usize,
    #[arg(long, default_value_t = 3)]
    pub power_iters: usize,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Record the creation time in the header (output is reproducible without it).
    #[arg(long)]
    pub timestamp: bool,
    /// Replace an existing --out.
    #[arg(long)]
    pub overwrite: bool,
}
