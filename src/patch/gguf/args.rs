//! `surgeon apply` options added by the GGUF patch, the chunked writer and the architecture
//! descriptions (flattened into `ApplyArgs`).

use clap::Args;

use super::backend::GgufOptions;
use super::project::{DEFAULT_REFINE_PASSES, DEFAULT_RESIDUAL_TARGET, MAX_REFINE_PASSES};
use super::quant::Requant;
use crate::kernel::IoConfig;

#[derive(Debug, Clone, PartialEq, Args)]
pub struct GgufApplyArgs {
    /// Patch a quantized .gguf in place (lossy: values are re-rounded to the file's grid).
    #[arg(long)]
    pub gguf: bool,
    /// GGUF re-quantization: keep-scales (default) or recompute.
    #[arg(long, value_enum, default_value_t = Requant::KeepScales)]
    pub requant: Requant,
    /// GGUF: skip the read-back check after writing.
    #[arg(long)]
    pub no_verify: bool,
    /// GGUF: patch without a backup copy (a failure leaves the file modified).
    #[arg(long)]
    pub force_no_backup: bool,
    /// Size of the write chunks in MiB (prefetch window and flush granularity).
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(u64).range(1..=65536))]
    pub prefetch_chunk_mib: u64,
    /// No OS prefetch hint nor intermediate flush (for comparison).
    #[arg(long)]
    pub no_prefetch: bool,
    /// project_out: largest rank of a subspace (rows of the a and b tensors).
    #[arg(long, default_value_t = crate::patch::DEFAULT_MAX_RANK)]
    pub max_rank: usize,
    /// GGUF: patch token_embd even when it is also the output head (no output.weight).
    #[arg(long)]
    pub allow_tied_embedding: bool,
    /// GGUF project_out: quantization passes measured (1 = no refinement, max 5).
    #[arg(long, default_value_t = DEFAULT_REFINE_PASSES, value_parser = clap::builder::RangedU64ValueParser::<usize>::new().range(1..=MAX_REFINE_PASSES as u64))]
    pub refine_passes: usize,
    /// GGUF project_out: stop refining once the requantization residual is at most this share.
    #[arg(long, default_value_t = DEFAULT_RESIDUAL_TARGET)]
    pub residual_target: f64,
    /// Architecture spec (JSON, `surgeon schema --kind arch`) instead of detection.
    #[arg(long)]
    pub arch_spec: Option<std::path::PathBuf>,
    /// Accept a project_out the architecture makes approximate (post-sublayer norms...).
    #[arg(long)]
    pub accept_approximate: bool,
}

/// The CLI defaults (the MCP server builds its options from it).
impl Default for GgufApplyArgs {
    fn default() -> Self {
        Self {
            gguf: false,
            requant: Requant::KeepScales,
            no_verify: false,
            force_no_backup: false,
            prefetch_chunk_mib: 64,
            no_prefetch: false,
            max_rank: crate::patch::DEFAULT_MAX_RANK,
            allow_tied_embedding: false,
            refine_passes: DEFAULT_REFINE_PASSES,
            residual_target: DEFAULT_RESIDUAL_TARGET,
            arch_spec: None,
            accept_approximate: false,
        }
    }
}

impl GgufApplyArgs {
    pub fn io(&self) -> IoConfig {
        IoConfig {
            chunk_bytes: usize::try_from(self.prefetch_chunk_mib << 20).unwrap_or(usize::MAX),
            prefetch: !self.no_prefetch,
        }
    }

    pub fn arch_options(&self) -> crate::app::ArchOptions {
        crate::app::ArchOptions {
            spec: self.arch_spec.clone(),
            accept_approximate: self.accept_approximate,
        }
    }

    pub fn options(&self, backup_dir: Option<std::path::PathBuf>) -> GgufOptions {
        GgufOptions {
            requant: self.requant,
            verify: !self.no_verify,
            force_no_backup: self.force_no_backup,
            io: self.io(),
            backup_dir,
            allow_tied_embedding: self.allow_tied_embedding,
            refine_passes: self.refine_passes,
            residual_target: self.residual_target,
            // Loaded (and sandboxed) by the caller: options never read files.
            arch_spec: None,
            accept_approximate: self.accept_approximate,
        }
    }
}
