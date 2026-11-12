//! Append-only JSONL cache of per-matrix analyses.
//!
//! The key is the complete model SHA-256 plus every parameter that can change the numbers
//! (tensor identity, shape, dtype, role, expert, rank, oversampling, power iterations,
//! seed, full-spectrum limit, kurtosis threshold and the optional vector digest). The value
//! is the rendered basis and the metrics, so a resumed run does not read the matrix again.
//! A cache record is appended as soon as a matrix finishes, which makes Ctrl-C leave a
//! usable partial cache.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::error::{GraphError, Result};
use super::metrics::{AnalyzeOptions, NodeAnalysis};
use super::model::{hex, TensorRef};
use super::output::{open_cache_append, prepare_cache_dir, remove_cache_file};
use super::GraphRole;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheKey {
    pub model_sha256: String,
    pub tensor: String,
    pub dtype: String,
    pub shape: [usize; 2],
    pub role: GraphRole,
    pub expert: Option<usize>,
    pub rank: usize,
    pub oversample: usize,
    pub power_iters: usize,
    pub seed: u64,
    pub full_spectrum_max_dim: usize,
    pub kurtosis_threshold: String,
    pub vector_sha256: Option<String>,
}

impl CacheKey {
    pub fn as_string(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| GraphError::Compute(format!("clé de cache : {e}")))
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheRecord {
    key: String,
    analysis: NodeAnalysis,
}

pub struct Cache {
    path: Option<PathBuf>,
    file: Option<File>,
    records: HashMap<String, NodeAnalysis>,
}

impl Cache {
    pub fn disabled() -> Self {
        Self {
            path: None,
            file: None,
            records: HashMap::new(),
        }
    }

    /// `resume` keeps existing records; otherwise the file is replaced.
    pub fn open(dir: &Path, model_sha256: &str, resume: bool) -> Result<Self> {
        prepare_cache_dir(dir)?;
        let path = dir.join(format!("graph-{model_sha256}.jsonl"));
        let mut records = HashMap::new();
        if resume {
            match File::open(&path) {
                Ok(file) => {
                    for line in BufReader::new(file).lines() {
                        let line = line.map_err(|e| GraphError::io(&path, &e))?;
                        if line.trim().is_empty() {
                            continue;
                        }
                        if let Ok(record) = serde_json::from_str::<CacheRecord>(&line) {
                            records.insert(record.key, record.analysis);
                        }
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(GraphError::io(&path, &error)),
            }
        } else {
            remove_cache_file(&path)?;
        }
        let file = open_cache_append(&path)?;
        Ok(Self {
            path: Some(path),
            file: Some(file),
            records,
        })
    }

    pub fn get(&self, key: &str) -> Option<&NodeAnalysis> {
        self.records.get(key)
    }

    pub fn put(&mut self, key: &str, analysis: &NodeAnalysis) -> Result<()> {
        let (Some(path), Some(file)) = (self.path.as_ref(), self.file.as_mut()) else {
            return Ok(());
        };
        let line = serde_json::to_string(&CacheRecord {
            key: key.to_owned(),
            analysis: analysis.clone(),
        })
        .map_err(|e| GraphError::Compute(format!("cache {} : {e}", path.display())))?;
        file.write_all(line.as_bytes())
            .and_then(|()| file.write_all(b"\n"))
            .and_then(|()| file.flush())
            .map_err(|e| GraphError::io(path, &e))?;
        self.records.insert(key.to_owned(), analysis.clone());
        Ok(())
    }
}

/// Digest of a `--vector` file's decoded values, for the cache key.
pub fn vector_digest(values: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for value in values {
        hasher.update(value.to_le_bytes());
    }
    hex(&hasher.finalize())
}

pub fn key_for(
    model_sha256: &str,
    tensor: &TensorRef,
    options: &AnalyzeOptions<'_>,
    vector_sha256: Option<String>,
) -> Result<String> {
    let params = &options.params;
    CacheKey {
        model_sha256: model_sha256.to_owned(),
        tensor: tensor.name.clone(),
        dtype: tensor.dtype.clone(),
        shape: tensor.shape,
        role: tensor.role,
        expert: tensor.expert,
        rank: params.rank,
        oversample: params.oversample,
        power_iters: params.power_iters,
        seed: params.seed,
        full_spectrum_max_dim: options.full_spectrum_max_dim,
        kurtosis_threshold: format!("{:?}", options.kurtosis_threshold),
        vector_sha256,
    }
    .as_string()
}

/// Reads a vector file: a JSON array, or raw little-endian f32 when the extension is `.bin`.
pub fn read_vector(path: &Path, hidden: usize) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).map_err(|e| GraphError::io(path, &e))?;
    let is_binary = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("bin") || e.eq_ignore_ascii_case("f32"));
    let values = if is_binary {
        if bytes.len() % 4 != 0 {
            return Err(GraphError::Usage(format!(
                "--vector {} : {} octets, multiple de 4 attendu",
                path.display(),
                bytes.len()
            )));
        }
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        let text = String::from_utf8(bytes).map_err(|e| {
            GraphError::Usage(format!(
                "--vector {} : UTF-8 invalide ({e})",
                path.display()
            ))
        })?;
        serde_json::from_str::<Vec<f32>>(&text).map_err(|e| {
            GraphError::Usage(format!(
                "--vector {} : tableau JSON de nombres attendu ({e})",
                path.display()
            ))
        })?
    };
    if values.len() != hidden {
        return Err(GraphError::Usage(format!(
            "--vector {} : {} valeurs pour hidden = {hidden}",
            path.display(),
            values.len()
        )));
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(GraphError::Usage(format!(
            "--vector {} : valeurs non finies",
            path.display()
        )));
    }
    Ok(values)
}

/// Useful in tests and the estimate path.
pub fn cache_path(dir: &Path, model_sha256: &str) -> PathBuf {
    dir.join(format!("graph-{model_sha256}.jsonl"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_f32_vectors_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v.bin");
        let values = [1.0f32, -2.0, 0.5, 3.5];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(&path, bytes).unwrap();
        assert_eq!(read_vector(&path, 4).unwrap(), values);
        assert!(read_vector(&path, 3).is_err());
    }
}
