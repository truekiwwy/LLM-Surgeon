//! clap argument structs of `surgeon lora merge|extract`.

use std::path::PathBuf;

use clap::{Args, Subcommand};

use crate::surgery::format::FactorDtype;

/// Exit codes: 0 ok, 1 cancelled or unreadable model, 2 invalid arguments, adapter or
/// config (missing `lora_alpha` included), 3 execution failed (rolled back), 70 internal.
#[derive(Debug, Clone, Args)]
pub struct LoraArgs {
    #[command(subcommand)]
    pub command: LoraCommand,
}

#[derive(Debug, Clone, Subcommand)]
pub enum LoraCommand {
    /// Merge a PEFT LoRA adapter into the weights, in place: W <- W + scale * (alpha/r) B A.
    Merge(LoraMergeArgs),
    /// Extract a rank-r LoRA adapter from the difference between two checkpoints.
    Extract(LoraExtractArgs),
}

#[derive(Debug, Clone, Args)]
pub struct LoraMergeArgs {
    /// Model to modify: .safetensors file or sharded directory.
    #[arg(long)]
    pub model: PathBuf,
    /// adapter_model.safetensors (PEFT keys).
    #[arg(long)]
    pub lora: PathBuf,
    /// lora_alpha; replaces the value of adapter_config.json. One of the two is required.
    #[arg(long, allow_hyphen_values = true)]
    pub alpha: Option<f64>,
    /// adapter_config.json (default: the one next to --lora).
    #[arg(long)]
    pub adapter_config: Option<PathBuf>,
    /// Extra multiplier of the delta.
    #[arg(long, default_value_t = 1.0, allow_hyphen_values = true)]
    pub scale: f64,
    #[arg(long)]
    pub dry_run: bool,
    /// Do not back up the touched shards (they are restored on error otherwise).
    #[arg(long)]
    pub no_backup: bool,
    /// Default: next to the model.
    #[arg(long)]
    pub backup_dir: Option<PathBuf>,
}

#[derive(Debug, Clone, Args)]
pub struct LoraExtractArgs {
    /// Base checkpoint.
    #[arg(long)]
    pub base: PathBuf,
    /// Modified checkpoint (same tensor names, shapes and dtypes).
    #[arg(long)]
    pub patched: PathBuf,
    /// adapter_model.safetensors to write (adapter_config.json is written next to it).
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 16)]
    pub rank: usize,
    #[arg(long, value_enum, default_value_t = FactorDtype::F32)]
    pub dtype: FactorDtype,
    /// Only 2D tensors whose name matches this regex (default: attention and MLP projections).
    #[arg(long)]
    pub target_modules: Option<String>,
    #[arg(long, default_value_t = 0)]
    pub seed: u64,
    /// Replace an existing --out.
    #[arg(long)]
    pub overwrite: bool,
}
