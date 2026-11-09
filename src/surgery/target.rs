//! Target model resolution shared by `.surgery` apply and `lora merge` (single file or
//! sharded directory: which shard holds which tensor, which shards to back up).
//!
//! Usage:
//! 1. [`resolve_shards`] turns `--model` (a `.safetensors` file or a model directory, with or
//!    without `model.safetensors.index.json`) into absolute shard paths.
//! 2. [`Target::open`] maps every shard read-write (`engine::WritableModel`).
//! 3. [`Target::info`] / [`Target::tensor_bytes_mut`] address a tensor by name, whatever its
//!    shard; [`Target::shards_holding`] lists the shards to hand to
//!    `patch::BackupSet::create_many_in`.
//! 4. Drop the `Target` BEFORE a rollback: the mappings flush on drop, and a restore must
//!    land after them. [`Target::flush`] makes the writes durable before a commit.

use std::path::{Path, PathBuf};

use super::error::{Result, SurgeryError};
use crate::engine::{EngineError, TensorInfo, WritableModel};
use crate::hardware;

/// Absolute shard files of `model`, in the order `hardware::compute_model_size` lists them
/// (absolute so backups next to a relative `--model` get a printable location).
pub fn resolve_shards(model: &Path) -> Result<Vec<PathBuf>> {
    let model = std::path::absolute(model).map_err(|e| SurgeryError::io(model, &e))?;
    match hardware::compute_model_size(&model) {
        Ok(size) => Ok(size.shard_files),
        Err(e) => Err(SurgeryError::Io {
            path: model.display().to_string(),
            message: e.to_string(),
        }),
    }
}

/// The shards of a model, mapped read-write.
pub struct Target {
    shards: Vec<(PathBuf, WritableModel)>,
}

impl Target {
    pub fn open(shards: &[PathBuf]) -> Result<Self> {
        let shards = shards
            .iter()
            .map(|p| Ok((p.clone(), WritableModel::open(p)?)))
            .collect::<Result<_>>()?;
        Ok(Self { shards })
    }

    /// Index of the first shard holding `name`.
    pub fn shard_of(&self, name: &str) -> Option<usize> {
        self.shards
            .iter()
            .position(|(_, model)| model.index().get(name).is_ok())
    }

    /// Dtype, shape and byte range of `name`, `None` when no shard has it.
    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        let shard = self.shard_of(name)?;
        self.shards[shard].1.index().get(name).ok()
    }

    /// Raw bytes of `name`, writable. Errors with `TensorNotFound` when no shard has it.
    pub fn tensor_bytes_mut(&mut self, name: &str) -> Result<&mut [u8]> {
        let shard = self
            .shard_of(name)
            .ok_or_else(|| SurgeryError::Engine(EngineError::TensorNotFound(name.to_owned())))?;
        Ok(self.shards[shard].1.tensor_bytes_mut(name)?)
    }

    /// Shards holding at least one of `names` (deduplicated, in shard order): only those
    /// need a backup.
    pub fn shards_holding<'n>(&self, names: impl IntoIterator<Item = &'n str>) -> Vec<PathBuf> {
        let mut indices: Vec<usize> = names.into_iter().filter_map(|n| self.shard_of(n)).collect();
        indices.sort_unstable();
        indices.dedup();
        indices
            .into_iter()
            .map(|i| self.shards[i].0.clone())
            .collect()
    }

    pub fn flush(&self) -> Result<()> {
        self.shards.iter().try_for_each(|(_, m)| Ok(m.flush()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal F32 safetensors file with 1x1 tensors.
    fn write_st(path: &Path, names: &[&str]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for name in names {
            let start = data.len();
            data.extend_from_slice(&1f32.to_le_bytes());
            header.insert(
                (*name).into(),
                serde_json::json!({"dtype": "F32", "shape": [1, 1], "data_offsets": [start, data.len()]}),
            );
        }
        let mut header = serde_json::to_string(&header).unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(' ');
        }
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        file.extend_from_slice(&data);
        std::fs::write(path, file).unwrap();
    }

    #[test]
    fn locates_tensors_across_shards() {
        let dir = tempfile::tempdir().unwrap();
        write_st(&dir.path().join("a.safetensors"), &["x", "y"]);
        write_st(&dir.path().join("b.safetensors"), &["z"]);
        let shards = resolve_shards(dir.path()).unwrap();
        assert_eq!(shards.len(), 2);
        let mut target = Target::open(&shards).unwrap();
        assert_eq!(target.shard_of("z"), Some(1));
        assert_eq!(target.info("y").map(|i| i.shape.clone()), Some(vec![1, 1]));
        assert!(target.info("nope").is_none());
        assert!(target.tensor_bytes_mut("nope").is_err());
        assert_eq!(target.tensor_bytes_mut("z").unwrap().len(), 4);
        assert_eq!(
            target.shards_holding(["z", "z", "nope"]),
            [shards[1].clone()]
        );
        assert_eq!(target.shards_holding(["z", "x"]), shards);
        assert!(resolve_shards(&dir.path().join("missing")).is_err());
    }
}
