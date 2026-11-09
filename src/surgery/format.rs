//! `.surgery` v1 header types (frozen contract, see `docs/SPECIFICATION.md`).
//!
//! Layout, little-endian: `"SURG"` | version u16 | flags u16 | reserved u32 | header_len u64
//! | canonical JSON header | blobs | sha256 of every preceding byte (32 B) | `"SURG"`.

use clap::ValueEnum;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const MAGIC: [u8; 4] = *b"SURG";
pub const FORMAT_VERSION: u16 = 1;

/// Typed reader/writer failures; the reader never panics on hostile input.
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("{0}")]
    Invalid(String),
    #[error("pas encore implémenté")]
    NotImplemented,
}

/// Storage dtype of the factors `U`, `S`, `V` (also the `lora extract --dtype` values).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema, ValueEnum, Default,
)]
#[serde(rename_all = "lowercase")]
pub enum FactorDtype {
    #[default]
    F16,
    Bf16,
    F32,
}

impl FactorDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
            Self::F32 => "f32",
        }
    }

    pub fn size_of(self) -> usize {
        match self {
            Self::F16 | Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum CompressionAlgo {
    Deflate,
    None,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Compression {
    pub algo: CompressionAlgo,
    pub level: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TensorKind {
    Lowrank,
    FullDelta,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct BaseInfo {
    pub name: String,
    pub tensor_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct TunedInfo {
    pub name: String,
}

/// Present only with `diff --timestamp`; keeps default output reproducible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct Provenance {
    pub timestamp: String,
}

/// Location of one blob inside the file (offset from the start of the file).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct BlobRef {
    pub offset: u64,
    /// Stored (possibly deflated) length.
    pub comp_len: u64,
    /// Length after decompression: must equal `shape x dtype x rank` of its role.
    pub raw_len: u64,
    pub sha256: String,
}

/// `U` [m x r], `S` [r], `V` [r x n] for `lowrank`; `delta` [shape] (f32) for `full_delta`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
pub struct Blobs {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub u: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub s: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub v: Option<BlobRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delta: Option<BlobRef>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub kind: TensorKind,
    /// 0 for `full_delta`.
    pub rank: usize,
    /// dtype of the target tensor: `BF16`, `F16` or `F32`.
    pub target_dtype: String,
    /// Hex SHA-256 of the base tensor bytes the delta was computed against.
    pub base_sha256: String,
    pub rel_error: f64,
    pub energy: f64,
    pub blobs: Blobs,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SurgeryHeader {
    pub format_version: u16,
    pub created_with: String,
    pub base: BaseInfo,
    pub tuned: TunedInfo,
    pub rank: usize,
    pub factor_dtype: FactorDtype,
    pub compression: Compression,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<Provenance>,
    pub tensors: Vec<TensorEntry>,
}

/// In-memory factors of one tensor, f32: `u` row-major [m x r], `s` [r], `v` row-major
/// [r x n]. `W + U diag(S) V` is the reconstructed delta.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LowRankFactors {
    pub u: Vec<f32>,
    pub s: Vec<f32>,
    pub v: Vec<f32>,
}
