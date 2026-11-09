//! `.surgery` v1 header types and structural validation (see `docs/SPECIFICATION.md`).
//!
//! Layout, little-endian: `"SURG"` | version u16 | flags u16 | reserved u32 | header_len u64
//! | canonical JSON header | blobs | sha256 of every preceding byte (32 B) | `"SURG"`.

use std::collections::HashSet;

use clap::ValueEnum;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use thiserror::Error;

pub const MAGIC: [u8; 4] = *b"SURG";
pub const FORMAT_VERSION: u16 = 1;

/// magic + version + flags + reserved + header_len.
pub const PREAMBLE_LEN: u64 = 20;
/// sha256 + magic.
pub const FOOTER_LEN: u64 = 36;
/// Caps applied to untrusted values before any allocation derived from them.
pub const MAX_HEADER_LEN: u64 = 64 << 20;
pub const MAX_BLOB_RAW_LEN: u64 = 2 << 30;
pub const MAX_RANK: usize = 1 << 16;
pub const MAX_NAME_LEN: usize = 1024;
pub const MAX_DIMS: usize = 8;
pub const MAX_DEFLATE_LEVEL: u8 = 10;
/// A deflate stream cannot expand by more than ~1032:1 (258 bytes per 2 bits).
pub const DEFLATE_MAX_RATIO: u64 = 1032;

const TARGET_DTYPES: [&str; 3] = ["BF16", "F16", "F32"];

/// Typed reader/writer failures; the reader never panics on hostile input.
#[derive(Debug, Error)]
pub enum FormatError {
    #[error("{0}")]
    Invalid(String),
    #[error("pas encore implémenté")]
    NotImplemented,
    #[error("magie absente ou incorrecte (attendu \"SURG\")")]
    BadMagic,
    #[error("version {found} non supportée (ce lecteur gère la version {supported})")]
    UnsupportedVersion { found: u16, supported: u16 },
    #[error("flags ou champ réservé non nuls (flags={flags}, reserved={reserved})")]
    UnsupportedFlags { flags: u16, reserved: u32 },
    #[error("fichier tronqué : {0}")]
    Truncated(String),
    #[error("header_len incohérent : déclaré {declared}, disponible {available}")]
    HeaderLenMismatch { declared: u64, available: u64 },
    #[error("en-tête invalide : {0}")]
    InvalidHeader(String),
    #[error("{tensor}/{blob} : blob hors du fichier ({detail})")]
    OffsetOutOfBounds {
        tensor: String,
        blob: &'static str,
        detail: String,
    },
    #[error("blobs qui se chevauchent : {first} et {second}")]
    BlobOverlap { first: String, second: String },
    #[error("{tensor}/{blob} : raw_len déclaré {declared}, attendu {expected}")]
    RawLenMismatch {
        tensor: String,
        blob: &'static str,
        declared: u64,
        expected: u64,
    },
    #[error("{tensor}/{blob} : décompression au-delà de la taille déclarée ({limit} octets)")]
    DecompressionBomb {
        tensor: String,
        blob: &'static str,
        limit: u64,
    },
    #[error("{tensor}/{blob} : décompression impossible ({reason})")]
    DecompressionFailed {
        tensor: String,
        blob: &'static str,
        reason: String,
    },
    #[error("{tensor}/{blob} : SHA-256 du blob incorrect")]
    BlobHashMismatch { tensor: String, blob: &'static str },
    #[error("SHA-256 global incorrect : fichier altéré")]
    GlobalHashMismatch,
    #[error("{tensor}/{blob} : valeur non finie dans les données")]
    NonFiniteValue { tensor: String, blob: &'static str },
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

/// `deflate` is a raw deflate stream (no zlib header), level 0..=10; `none` requires level 0.
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
    #[schemars(range(min = 1))]
    pub comp_len: u64,
    /// Length after decompression: must equal `shape x dtype x rank` of its role.
    #[schemars(range(min = 1))]
    pub raw_len: u64,
    /// Lowercase hex SHA-256 of the RAW (decompressed) bytes.
    #[schemars(regex(pattern = "^[0-9a-f]{64}$"))]
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
    #[schemars(length(min = 1, max = 1024))]
    pub name: String,
    #[schemars(length(min = 1, max = 8))]
    pub shape: Vec<usize>,
    pub kind: TensorKind,
    /// 0 for `full_delta`.
    pub rank: usize,
    /// dtype of the target tensor: `BF16`, `F16` or `F32`.
    #[schemars(regex(pattern = "^(BF16|F16|F32)$"))]
    pub target_dtype: String,
    /// Hex SHA-256 of the base tensor bytes the delta was computed against.
    #[schemars(regex(pattern = "^[0-9a-f]{64}$"))]
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

/// Role of a blob inside a tensor entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlobRole {
    U,
    S,
    V,
    Delta,
}

impl BlobRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::U => "u",
            Self::S => "s",
            Self::V => "v",
            Self::Delta => "delta",
        }
    }
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn sha256_hex(data: &[u8]) -> String {
    hex(&Sha256::digest(data))
}

pub fn is_hex_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn invalid(msg: impl Into<String>) -> FormatError {
    FormatError::InvalidHeader(msg.into())
}

impl TensorEntry {
    /// Blobs of this entry in storage order (u, s, v, delta).
    pub fn blob_list(&self) -> Vec<(BlobRole, &BlobRef)> {
        [
            (BlobRole::U, &self.blobs.u),
            (BlobRole::S, &self.blobs.s),
            (BlobRole::V, &self.blobs.v),
            (BlobRole::Delta, &self.blobs.delta),
        ]
        .into_iter()
        .filter_map(|(role, blob)| blob.as_ref().map(|b| (role, b)))
        .collect()
    }

    pub fn blob(&self, role: BlobRole) -> Option<&BlobRef> {
        match role {
            BlobRole::U => self.blobs.u.as_ref(),
            BlobRole::S => self.blobs.s.as_ref(),
            BlobRole::V => self.blobs.v.as_ref(),
            BlobRole::Delta => self.blobs.delta.as_ref(),
        }
    }

    /// Product of the dimensions, `None` on overflow.
    pub fn numel(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(u64::try_from(d).ok()?))
    }

    /// Exact raw byte length a blob of `role` must have (`None` on overflow or if the shape does
    /// not fit the role).
    pub fn expected_raw_len(&self, role: BlobRole, dtype: FactorDtype) -> Option<u64> {
        let sz = dtype.size_of() as u64;
        let r = u64::try_from(self.rank).ok()?;
        match role {
            BlobRole::Delta => self.numel()?.checked_mul(4),
            BlobRole::S => r.checked_mul(sz),
            BlobRole::U | BlobRole::V => {
                let [m, n] = self.shape[..] else { return None };
                let dim = u64::try_from(if role == BlobRole::U { m } else { n }).ok()?;
                dim.checked_mul(r)?.checked_mul(sz)
            }
        }
    }

    /// Everything except the blobs: name, shape, dtype, hash, kind/rank consistency.
    pub fn validate_fields(&self, header_rank: usize) -> Result<(), FormatError> {
        let name = &self.name;
        if name.is_empty() || name.len() > MAX_NAME_LEN || name.chars().any(char::is_control) {
            return Err(invalid(format!(
                "nom de tenseur invalide ({} octets)",
                name.len()
            )));
        }
        if self.shape.is_empty() || self.shape.len() > MAX_DIMS || self.shape.contains(&0) {
            return Err(invalid(format!("{name} : forme invalide {:?}", self.shape)));
        }
        if self.numel().is_none() {
            return Err(invalid(format!("{name} : forme trop grande")));
        }
        if !TARGET_DTYPES.contains(&self.target_dtype.as_str()) {
            return Err(invalid(format!(
                "{name} : target_dtype inconnu {:?}",
                self.target_dtype
            )));
        }
        if !is_hex_sha256(&self.base_sha256) {
            return Err(invalid(format!("{name} : base_sha256 invalide")));
        }
        for (label, v) in [("rel_error", self.rel_error), ("energy", self.energy)] {
            if !v.is_finite() || v < 0.0 {
                return Err(invalid(format!("{name} : {label} invalide ({v})")));
            }
        }
        match self.kind {
            TensorKind::Lowrank => {
                if self.shape.len() != 2 {
                    return Err(invalid(format!("{name} : lowrank exige une forme 2D")));
                }
                if self.rank == 0 || self.rank > header_rank {
                    return Err(invalid(format!(
                        "{name} : rang {} hors de 1..={header_rank}",
                        self.rank
                    )));
                }
            }
            TensorKind::FullDelta if self.rank != 0 => {
                return Err(invalid(format!("{name} : full_delta exige rank = 0")));
            }
            TensorKind::FullDelta => {}
        }
        Ok(())
    }

    /// Blob set per kind, `raw_len` against shape x dtype x rank, plausibility of `comp_len`.
    pub fn validate_blobs(
        &self,
        dtype: FactorDtype,
        compression: Compression,
    ) -> Result<(), FormatError> {
        let required: &[BlobRole] = match self.kind {
            TensorKind::Lowrank => &[BlobRole::U, BlobRole::S, BlobRole::V],
            TensorKind::FullDelta => &[BlobRole::Delta],
        };
        let present = self.blob_list();
        if present.len() != required.len() || present.iter().any(|(r, _)| !required.contains(r)) {
            return Err(invalid(format!("{} : jeu de blobs incohérent", self.name)));
        }
        for (role, blob) in present {
            let expected = self
                .expected_raw_len(role, dtype)
                .ok_or_else(|| invalid(format!("{} : taille de blob en débordement", self.name)))?;
            if blob.raw_len != expected {
                return Err(FormatError::RawLenMismatch {
                    tensor: self.name.clone(),
                    blob: role.as_str(),
                    declared: blob.raw_len,
                    expected,
                });
            }
            let ctx = |msg: &str| invalid(format!("{}/{} : {msg}", self.name, role.as_str()));
            if expected > MAX_BLOB_RAW_LEN {
                return Err(ctx("blob plus grand que la limite de 2 GiB"));
            }
            if !is_hex_sha256(&blob.sha256) {
                return Err(ctx("sha256 invalide"));
            }
            let plausible = match compression.algo {
                CompressionAlgo::None => blob.comp_len == blob.raw_len,
                CompressionAlgo::Deflate => {
                    blob.comp_len >= 1
                        && blob.raw_len <= blob.comp_len.saturating_mul(DEFLATE_MAX_RATIO)
                }
            };
            if !plausible {
                return Err(ctx("comp_len incohérent avec raw_len"));
            }
        }
        Ok(())
    }
}

impl SurgeryHeader {
    /// Structural checks that need no file: version, rank, compression, per-tensor rules,
    /// unique names. Used by the reader (untrusted) and by the writer (self-check).
    pub fn validate(&self) -> Result<(), FormatError> {
        if self.format_version != FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion {
                found: self.format_version,
                supported: FORMAT_VERSION,
            });
        }
        if self.rank == 0 || self.rank > MAX_RANK {
            return Err(invalid(format!(
                "rank {} hors de 1..={MAX_RANK}",
                self.rank
            )));
        }
        let level_ok = match self.compression.algo {
            CompressionAlgo::None => self.compression.level == 0,
            CompressionAlgo::Deflate => self.compression.level <= MAX_DEFLATE_LEVEL,
        };
        if !level_ok {
            return Err(invalid(format!(
                "niveau de compression {} invalide",
                self.compression.level
            )));
        }
        let mut seen = HashSet::new();
        for t in &self.tensors {
            t.validate_fields(self.rank)?;
            if !seen.insert(t.name.as_str()) {
                return Err(invalid(format!("tenseur en double : {}", t.name)));
            }
            t.validate_blobs(self.factor_dtype, self.compression)?;
        }
        Ok(())
    }

    /// Canonical JSON: keys sorted by byte order at every level, no whitespace, so equal headers
    /// give equal bytes whatever the map implementation of `serde_json`.
    pub fn to_canonical_json(&self) -> Result<Vec<u8>, FormatError> {
        let value = serde_json::to_value(self).map_err(|e| invalid(e.to_string()))?;
        let mut out = Vec::new();
        write_canonical(&value, &mut out).map_err(|e| invalid(e.to_string()))?;
        Ok(out)
    }
}

fn write_canonical(value: &Value, out: &mut Vec<u8>) -> Result<(), serde_json::Error> {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<_> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            out.push(b'{');
            for (i, (key, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                serde_json::to_writer(&mut *out, key)?;
                out.push(b':');
                write_canonical(v, out)?;
            }
            out.push(b'}');
        }
        Value::Array(items) => {
            out.push(b'[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(v, out)?;
            }
            out.push(b']');
        }
        scalar => serde_json::to_writer(&mut *out, scalar)?,
    }
    Ok(())
}
