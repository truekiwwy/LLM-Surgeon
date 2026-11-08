use std::path::Path;

use safetensors::{Dtype, SafeTensorError, SafeTensors};
use serde::Serialize;

use super::error::{EngineError, Result};

/// Size of the little-endian u64 header length that starts every safetensors file.
const LEN_PREFIX: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub enum DType {
    BF16,
    F16,
    F32,
    Other(String),
}

impl DType {
    /// Element size in bytes for the dtypes the engine can shift.
    pub fn size_of(&self) -> Option<usize> {
        match self {
            DType::BF16 | DType::F16 => Some(2),
            DType::F32 => Some(4),
            DType::Other(_) => None,
        }
    }
}

impl std::fmt::Display for DType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DType::BF16 => f.write_str("BF16"),
            DType::F16 => f.write_str("F16"),
            DType::F32 => f.write_str("F32"),
            DType::Other(name) => f.write_str(name),
        }
    }
}

impl From<Dtype> for DType {
    fn from(d: Dtype) -> Self {
        match d {
            Dtype::BF16 => DType::BF16,
            Dtype::F16 => DType::F16,
            Dtype::F32 => DType::F32,
            other => DType::Other(other.to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
    /// Absolute offsets in the file: `8 + header_len + data_offsets`.
    pub byte_start: usize,
    pub byte_end: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelIndex {
    /// Sorted by `byte_start`.
    pub tensors: Vec<TensorInfo>,
    pub header_len: usize,
}

impl ModelIndex {
    pub fn get(&self, name: &str) -> Result<&TensorInfo> {
        self.tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| EngineError::TensorNotFound(name.to_owned()))
    }
}

/// Parses and validates the header of a mapped file. `safetensors::read_metadata` already
/// checks contiguous non-overlapping offsets, shape x dtype sizes and full file coverage,
/// so after this every `byte_start..byte_end` is in bounds.
pub(crate) fn parse_index(path: &Path, buffer: &[u8]) -> Result<ModelIndex> {
    let (header_len, metadata) =
        SafeTensors::read_metadata(buffer).map_err(|e| map_err(path, buffer.len(), e))?;
    let base = LEN_PREFIX + header_len;
    let tensors = metadata
        .offset_keys()
        .into_iter()
        .filter_map(|name| {
            let info = metadata.info(&name)?;
            Some(TensorInfo {
                dtype: info.dtype.into(),
                shape: info.shape.clone(),
                byte_start: base + info.data_offsets.0,
                byte_end: base + info.data_offsets.1,
                name,
            })
        })
        .collect();
    Ok(ModelIndex {
        tensors,
        header_len,
    })
}

fn map_err(path: &Path, len: usize, err: SafeTensorError) -> EngineError {
    let path = path.to_path_buf();
    match err {
        SafeTensorError::HeaderTooSmall => EngineError::FileTooSmall {
            path,
            len: len as u64,
        },
        SafeTensorError::HeaderTooLarge | SafeTensorError::InvalidHeaderLength => {
            EngineError::InvalidHeaderLength { path }
        }
        SafeTensorError::InvalidOffset(tensor) => EngineError::InvalidOffsets { path, tensor },
        SafeTensorError::MetadataIncompleteBuffer => EngineError::DataSizeMismatch { path },
        SafeTensorError::TensorInvalidInfo
        | SafeTensorError::ValidationOverflow
        | SafeTensorError::MisalignedSlice
        | SafeTensorError::InvalidTensorView(..) => EngineError::TensorSizeMismatch { path },
        // Invalid UTF-8, invalid JSON, and anything a future safetensors version adds.
        other => EngineError::InvalidHeaderJson {
            path,
            message: other.to_string(),
        },
    }
}
