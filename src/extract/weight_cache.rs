//! In-process cache of the tensors [`load_tensors`](super::weights::load_tensors) reads, for
//! the long-lived `surgeon mcp` server only (one-shot CLI runs read the disk as before).
//!
//! ```text
//! key      shard files, SHA-256 of their contents, dtype; CPU tensors (native backend)
//! lookup   same files and dtype, stamps (length, mtime) unchanged  -> hit, nothing hashed
//!          stamps changed -> SHA-256 again: same digest -> hit, else the entry is dropped
//! miss     the previous entry is dropped before reading: one model in RAM at a time
//! subset   tensors missing from a valid entry (a deeper layer, the head) are added to it
//! clear    after every MCP tool that writes weights (apply, restore, checkpoint...)
//! ```
//!
//! A cached tensor is shared with the models built from it (candle storage is reference
//! counted and never written in place), so a hit costs no copy.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::SystemTime;

use candle_core::{DType, Device, Tensor};

use super::error::{ExtractError, Result};
use super::weights::{read_tensors, TensorSource};

static ENABLED: AtomicBool = AtomicBool::new(false);
static GLOBAL: WeightCache = WeightCache::new();

/// Turns the process-wide cache on (`surgeon mcp`).
pub fn enable() {
    ENABLED.store(true, Ordering::SeqCst);
}

/// The process-wide cache, when enabled.
pub fn global() -> Option<&'static WeightCache> {
    ENABLED.load(Ordering::SeqCst).then_some(&GLOBAL)
}

/// Bytes held by the process-wide cache: freed before any other model is read, so a RAM
/// budget counts them as available.
pub fn reclaimable_bytes() -> u64 {
    global().map_or(0, WeightCache::resident_bytes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Stamp {
    len: u64,
    modified: Option<SystemTime>,
}

struct Entry {
    files: Vec<PathBuf>,
    stamps: Vec<Stamp>,
    sha256: String,
    dtype: DType,
    tensors: HashMap<String, Tensor>,
}

/// One model's tensors at one dtype, checked against its files on every lookup.
#[derive(Default)]
pub struct WeightCache {
    slot: Mutex<Option<Entry>>,
}

fn weights_error(file: &std::path::Path, e: impl std::fmt::Display) -> ExtractError {
    ExtractError::Weights(format!("{} : {e}", file.display()))
}

fn stamps(files: &[PathBuf]) -> Result<Vec<Stamp>> {
    files
        .iter()
        .map(|file| {
            let meta = std::fs::metadata(file).map_err(|e| weights_error(file, e))?;
            Ok(Stamp {
                len: meta.len(),
                modified: meta.modified().ok(),
            })
        })
        .collect()
}

/// SHA-256 of the shard paths and of their digests, in order.
fn sha256(files: &[PathBuf]) -> Result<String> {
    let mut manifest = String::new();
    for file in files {
        let digest = crate::bench::sha256_file(file).map_err(|e| weights_error(file, e))?;
        manifest.push_str(&format!("{}\0{digest}\n", file.display()));
    }
    Ok(crate::bench::sha256_hex(manifest.as_bytes()))
}

impl WeightCache {
    pub const fn new() -> Self {
        Self {
            slot: Mutex::new(None),
        }
    }

    fn slot(&self) -> MutexGuard<'_, Option<Entry>> {
        self.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Drops the cached model.
    pub fn clear(&self) {
        *self.slot() = None;
    }

    /// Bytes of the cached tensors.
    pub fn resident_bytes(&self) -> u64 {
        self.slot().as_ref().map_or(0, |entry| {
            entry
                .tensors
                .values()
                .map(|t| (t.elem_count() * t.dtype().size_in_bytes()) as u64)
                .sum()
        })
    }

    /// SHA-256 of the cached model's shard files, if any.
    pub fn sha256(&self) -> Option<String> {
        self.slot().as_ref().map(|entry| entry.sha256.clone())
    }

    /// The `required` CPU tensors of `src` (read from `files`) at `dtype`: from the cache
    /// while the files hash the same, else read, cached and returned.
    pub fn load(
        &self,
        files: &[PathBuf],
        src: &dyn TensorSource,
        required: &[(String, Vec<usize>)],
        dtype: DType,
        on_tensor: &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut slot = self.slot();
        let now = stamps(files)?;
        let mut digest = None;
        let kept = match slot.take() {
            Some(entry) if entry.files == files && entry.dtype == dtype => {
                match entry.stamps == now {
                    true => Some(entry),
                    false => {
                        let current = sha256(files)?;
                        let same = current == entry.sha256;
                        digest = Some(current);
                        same.then_some(Entry {
                            stamps: now.clone(),
                            ..entry
                        })
                    }
                }
            }
            _ => None,
        };
        let mut entry = match kept {
            Some(entry) => entry,
            None => Entry {
                files: files.to_vec(),
                stamps: now.clone(),
                sha256: match digest {
                    Some(digest) => digest,
                    None => sha256(files)?,
                },
                dtype,
                tensors: HashMap::new(),
            },
        };
        let missing: Vec<(String, Vec<usize>)> = required
            .iter()
            .filter(|(name, _)| !entry.tensors.contains_key(name))
            .cloned()
            .collect();
        match read_tensors(src, &missing, dtype, &Device::Cpu, on_tensor) {
            Ok(read) => entry.tensors.extend(read),
            Err(e) => {
                *slot = Some(entry);
                return Err(e);
            }
        }
        let tensors = required
            .iter()
            .map(|(name, expected)| {
                let tensor = &entry.tensors[name];
                match tensor.dims() == expected.as_slice() {
                    true => Ok((name.clone(), tensor.clone())),
                    false => Err(ExtractError::TensorShape {
                        name: name.clone(),
                        expected: expected.clone(),
                        found: tensor.dims().to_vec(),
                    }),
                }
            })
            .collect::<Result<HashMap<_, _>>>();
        if stamps(files)? == now {
            *slot = Some(entry);
        }
        tensors
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::weights::SafetensorsDir;

    fn write_model(path: &std::path::Path, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        let header = serde_json::json!({
            "w": {"dtype": "F32", "shape": [values.len()], "data_offsets": [0, bytes.len()]}
        })
        .to_string();
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&bytes);
        std::fs::write(path, file).unwrap();
    }

    fn load(cache: &WeightCache, dir: &std::path::Path, len: usize) -> Tensor {
        let src = SafetensorsDir::open(dir).unwrap();
        let files = src.shard_files().unwrap().to_vec();
        let required = [("w".to_owned(), vec![len])];
        let mut tensors = cache
            .load(&files, &src, &required, DType::F32, &mut |_, _| {
                ControlFlow::Continue(())
            })
            .unwrap();
        tensors.remove("w").unwrap()
    }

    #[test]
    fn two_loads_share_the_tensor_and_a_weight_change_reloads_it() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("model.safetensors");
        write_model(&shard, &[1.0, 2.0, 3.0]);
        let cache = WeightCache::new();
        let first = load(&cache, dir.path(), 3);
        let second = load(&cache, dir.path(), 3);
        assert_eq!(first.id(), second.id());
        assert_eq!(cache.resident_bytes(), 12);
        let digest = cache.sha256().unwrap();

        let tick = || std::thread::sleep(std::time::Duration::from_millis(50));
        tick();
        write_model(&shard, &[1.0, 2.0, 3.0]);
        assert_eq!(load(&cache, dir.path(), 3).id(), first.id());

        tick();
        write_model(&shard, &[1.0, 2.0, 4.0]);
        let changed = load(&cache, dir.path(), 3);
        assert_ne!(changed.id(), first.id());
        assert_eq!(changed.to_vec1::<f32>().unwrap(), [1.0, 2.0, 4.0]);
        assert_ne!(cache.sha256().unwrap(), digest);

        cache.clear();
        assert_eq!(cache.resident_bytes(), 0);
        assert_ne!(load(&cache, dir.path(), 3).id(), changed.id());
    }

    #[test]
    fn a_cached_tensor_of_another_shape_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        write_model(&dir.path().join("model.safetensors"), &[1.0, 2.0]);
        let cache = WeightCache::new();
        load(&cache, dir.path(), 2);
        let src = SafetensorsDir::open(dir.path()).unwrap();
        let files = src.shard_files().unwrap().to_vec();
        let wrong = [("w".to_owned(), vec![3])];
        let err = cache
            .load(&files, &src, &wrong, DType::F32, &mut |_, _| {
                ControlFlow::Continue(())
            })
            .unwrap_err();
        assert!(matches!(err, ExtractError::TensorShape { .. }), "{err}");
    }
}
