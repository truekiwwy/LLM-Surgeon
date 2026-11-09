//! Kernel level selection (run-time CPU detection, `SURGEON_KERNEL` override) and the rayon
//! split of a tensor into blocks of rows.

use std::fmt;

use rayon::prelude::*;
use serde::Serialize;
use thiserror::Error;

use super::simd::{self, Deltas, Elem, RowDeltas, ShiftStats};

/// Environment override: `scalar`, `avx2`, `avx512` or `neon`.
pub const ENV_VAR: &str = "SURGEON_KERNEL";

/// Rows handed to one rayon task: about half of a typical 512 KiB - 2 MiB L2, so the
/// block stays in cache between the load and the store of each element.
pub const DEFAULT_BLOCK_BYTES: usize = 256 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Scalar,
    Avx2,
    Avx512,
    Neon,
}

impl Level {
    pub const ALL: [Level; 4] = [Level::Scalar, Level::Avx2, Level::Avx512, Level::Neon];

    pub fn name(self) -> &'static str {
        match self {
            Level::Scalar => "scalar",
            Level::Avx2 => "avx2",
            Level::Avx512 => "avx512",
            Level::Neon => "neon",
        }
    }

    pub fn parse(s: &str) -> Option<Level> {
        Level::ALL
            .into_iter()
            .find(|l| l.name().eq_ignore_ascii_case(s.trim()))
    }

    /// Run-time detection. A level compiled out for this architecture is never available,
    /// so an unavailable level can never execute an illegal instruction.
    pub fn is_available(self) -> bool {
        match self {
            Level::Scalar => true,
            #[cfg(target_arch = "x86_64")]
            Level::Avx2 => {
                std::arch::is_x86_feature_detected!("avx2")
                    && std::arch::is_x86_feature_detected!("f16c")
            }
            #[cfg(target_arch = "x86_64")]
            Level::Avx512 => std::arch::is_x86_feature_detected!("avx512f"),
            #[cfg(target_arch = "aarch64")]
            Level::Neon => std::arch::is_aarch64_feature_detected!("neon"),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    /// The widest available level.
    pub fn best() -> Level {
        [Level::Avx512, Level::Avx2, Level::Neon]
            .into_iter()
            .find(|l| l.is_available())
            .unwrap_or(Level::Scalar)
    }
}

impl fmt::Display for Level {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// The level actually used, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KernelChoice {
    pub level: Level,
    /// Raw value of `SURGEON_KERNEL` (or of the explicit request), if any.
    pub requested: Option<String>,
    /// Set when the request was unknown or unavailable and the best level was used instead.
    pub warning: Option<String>,
}

/// Resolves a requested level: unknown or unavailable falls back to the best available one,
/// with a warning, never an error.
pub fn choose(requested: Option<&str>) -> KernelChoice {
    let best = Level::best();
    let Some(raw) = requested.map(str::trim).filter(|s| !s.is_empty()) else {
        return KernelChoice {
            level: best,
            requested: None,
            warning: None,
        };
    };
    let (level, warning) = match Level::parse(raw) {
        Some(level) if level.is_available() => (level, None),
        Some(level) => (
            best,
            Some(format!(
                "noyau {level} indisponible sur ce processeur, repli sur {best}"
            )),
        ),
        None => (
            best,
            Some(format!(
                "{ENV_VAR}={raw} inconnu (attendu scalar, avx2, avx512 ou neon), repli sur {best}"
            )),
        ),
    };
    KernelChoice {
        level,
        requested: Some(raw.to_owned()),
        warning,
    }
}

/// [`choose`] applied to `SURGEON_KERNEL`.
pub fn from_env() -> KernelChoice {
    choose(std::env::var(ENV_VAR).ok().as_deref())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelConfig {
    pub level: Level,
    /// Bytes of rows per rayon task (at least one row).
    pub block_bytes: usize,
}

impl KernelConfig {
    pub fn new(level: Level) -> Self {
        Self {
            level,
            block_bytes: DEFAULT_BLOCK_BYTES,
        }
    }

    /// Level from `SURGEON_KERNEL`, default block size.
    pub fn from_env() -> Self {
        Self::new(from_env().level)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum KernelError {
    #[error("{bytes} octets ne forment pas des lignes entières de {cols} éléments")]
    NotWholeRows { bytes: usize, cols: usize },
    #[error("{got} deltas pour {expected} attendus")]
    DeltaLength { expected: usize, got: usize },
}

/// Shifts every row of a row-major `[rows, cols]` slice with the configured level (the
/// scalar kernel when that level is not available on this CPU).
///
/// One rayon task per block of `block_bytes` of whole rows; rows never share bytes and the
/// stats reduction is exact, so the result does not depend on the thread count.
pub fn shift_rows(
    bytes: &mut [u8],
    elem: Elem,
    cols: usize,
    deltas: Deltas<'_>,
    cfg: &KernelConfig,
) -> Result<ShiftStats, KernelError> {
    let row_bytes = cols * elem.size();
    if bytes.is_empty() {
        return Ok(ShiftStats::default());
    }
    if row_bytes == 0 || !bytes.len().is_multiple_of(row_bytes) {
        return Err(KernelError::NotWholeRows {
            bytes: bytes.len(),
            cols,
        });
    }
    let rows = bytes.len() / row_bytes;
    let (expected, got) = match deltas {
        Deltas::PerCol(d) => (cols, d.len()),
        Deltas::PerRow(d) => (rows, d.len()),
    };
    if expected != got {
        return Err(KernelError::DeltaLength { expected, got });
    }
    let level = match cfg.level.is_available() {
        true => cfg.level,
        false => Level::Scalar,
    };
    let rows_per_task = (cfg.block_bytes / row_bytes).max(1);
    Ok(bytes
        .par_chunks_mut(rows_per_task * row_bytes)
        .enumerate()
        .map(|(task, block)| {
            block
                .chunks_mut(row_bytes)
                .enumerate()
                .fold(ShiftStats::default(), |acc, (k, row)| {
                    let d = match deltas {
                        Deltas::PerCol(d) => RowDeltas::Each(d),
                        // In range: rows == d.len(), checked above.
                        Deltas::PerRow(d) => RowDeltas::Splat(d[task * rows_per_task + k]),
                    };
                    acc.merge(shift_row(level, row, elem, d))
                })
        })
        .reduce(ShiftStats::default, ShiftStats::merge))
}

/// The scalar reference over a whole slice: [`shift_rows`] at level `Scalar`.
pub fn shift_scalar(
    bytes: &mut [u8],
    elem: Elem,
    cols: usize,
    deltas: Deltas<'_>,
) -> Result<ShiftStats, KernelError> {
    shift_rows(bytes, elem, cols, deltas, &KernelConfig::new(Level::Scalar))
}

/// `level` must come from [`Level::is_available`] (checked by [`shift_rows`]).
fn shift_row(level: Level, row: &mut [u8], elem: Elem, d: RowDeltas<'_>) -> ShiftStats {
    match level {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: `shift_rows` only passes a level for which `is_available` detected
        // avx512f at run time, the only feature `row_avx512` enables.
        Level::Avx512 => unsafe { simd::x86::row_avx512(row, elem, d) },
        #[cfg(target_arch = "x86_64")]
        // SAFETY: same, `is_available` detected avx2 and f16c.
        Level::Avx2 => unsafe { simd::x86::row_avx2(row, elem, d) },
        #[cfg(target_arch = "aarch64")]
        // SAFETY: `is_available` detected neon at run time.
        Level::Neon => unsafe { simd::neon::row_neon(row, elem, d) },
        _ => simd::shift_row_scalar(row, elem, d),
    }
}
