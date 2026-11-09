//! `.surgery` reader. The file is untrusted input (frozen API, body by the `format` agent).

use std::path::Path;

use super::error::{Result, SurgeryError};
use super::format::{LowRankFactors, SurgeryHeader, TensorEntry};

/// True when the file starts with the `SURG` magic (cheap: reads 4 bytes, never fails).
pub fn is_surgery_file(_path: &Path) -> bool {
    false
}

pub struct SurgeryReader {
    _private: (),
}

impl SurgeryReader {
    /// Validates magic, version, `header_len` against the real size, blob bounds and
    /// overlaps, declared lengths, and the global SHA-256 (streamed) before returning:
    /// a reader that exists is a reader on a file whose integrity was checked.
    pub fn open(_path: &Path) -> Result<Self> {
        Err(SurgeryError::NotImplemented("SurgeryReader::open"))
    }

    pub fn header(&self) -> &SurgeryHeader {
        unreachable!("SurgeryReader cannot be constructed in the skeleton")
    }

    /// Inflates (never beyond the declared `raw_len`), verifies the blob hash, converts to f32.
    pub fn read_lowrank(&mut self, _entry: &TensorEntry) -> Result<LowRankFactors> {
        Err(SurgeryError::NotImplemented("SurgeryReader::read_lowrank"))
    }

    pub fn read_full_delta(&mut self, _entry: &TensorEntry) -> Result<Vec<f32>> {
        Err(SurgeryError::NotImplemented(
            "SurgeryReader::read_full_delta",
        ))
    }
}
