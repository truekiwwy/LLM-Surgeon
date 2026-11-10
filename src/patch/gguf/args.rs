//! `surgeon apply` options added by the GGUF patch and the chunked writer (flattened into
//! `ApplyArgs`).

use clap::Args;

use super::backend::GgufOptions;
use super::quant::Requant;
use crate::kernel::IoConfig;

#[derive(Debug, Clone, PartialEq, Eq, Args)]
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

    pub fn options(&self, backup_dir: Option<std::path::PathBuf>) -> GgufOptions {
        GgufOptions {
            requant: self.requant,
            verify: !self.no_verify,
            force_no_backup: self.force_no_backup,
            io: self.io(),
            backup_dir,
        }
    }
}
