//! SafeTensors reader: a single file, a checkpoint directory (every
//! `*.safetensors` shard) or a sharded `model.safetensors.index.json`.
//! Everything is mmap-backed: tensor bytes are never copied at load time.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use safetensors::tensor::Dtype as StDtype;
use safetensors::SafeTensors;

use crate::model::dtype::DType;
use crate::model::{Format, ModelError};

/// A raw entry before role classification.
pub type RawEntry = (String, Vec<u64>, DType, u64, PathBuf, u64);

/// Opens a checkpoint directory.
pub fn open_dir(dir: &Path) -> Result<(Vec<RawEntry>, Format), ModelError> {
    let index_path = dir.join("model.safetensors.index.json");
    let files: Vec<PathBuf> = if index_path.is_file() {
        let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let weight_map = json
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| ModelError::Json(serde::de::Error::custom("index sans weight_map")))?;
        let mut files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(str::to_owned)
            .collect();
        files.sort();
        files.dedup();
        files.into_iter().map(|f| dir.join(f)).collect()
    } else {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
            })
            .collect();
        files.sort();
        if files.is_empty() {
            return Err(ModelError::UnknownFormat(dir.display().to_string()));
        }
        files
    };
    let mut entries = Vec::new();
    for file in files {
        entries.extend(open_file(&file)?);
    }
    Ok((entries, Format::SafeTensors))
}

/// Opens one `.safetensors` file (mmap, header parsed in place).
pub fn open_file(path: &Path) -> Result<Vec<RawEntry>, ModelError> {
    let map = crate::model::map_file(path)?;
    let buffer = &map[..];
    let st = SafeTensors::deserialize(buffer)?;
    let base = map.as_ptr() as usize;
    let mut entries = Vec::new();
    for view in st.tensors() {
        let (name, info) = view;
        let offset = (info.data().as_ptr() as usize).saturating_sub(base) as u64;
        entries.push((
            name,
            info.shape().iter().map(|d| *d as u64).collect(),
            map_dtype(info.dtype()),
            info.data().len() as u64,
            path.to_path_buf(),
            offset,
        ));
    }
    Ok(entries)
}

/// Maps a safetensors dtype; unsupported ones keep their name for the UI.
fn map_dtype(dtype: StDtype) -> DType {
    match dtype {
        StDtype::F32 => DType::F32,
        StDtype::F16 => DType::F16,
        StDtype::BF16 => DType::Bf16,
        other => DType::Other(format!("{other:?}")),
    }
}

/// Groups entries by shard file, preserving the load order (for tests and the
/// `inspect`-style summary).
pub fn group_by_file(entries: &[RawEntry]) -> BTreeMap<PathBuf, Vec<String>> {
    let mut groups: BTreeMap<PathBuf, Vec<String>> = BTreeMap::new();
    for (name, .., file, _) in entries {
        groups.entry(file.clone()).or_default().push(name.clone());
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::Dtype;

    #[test]
    fn dtype_mapping_covers_the_floats() {
        assert_eq!(map_dtype(Dtype::F32), DType::F32);
        assert_eq!(map_dtype(Dtype::F16), DType::F16);
        assert_eq!(map_dtype(Dtype::BF16), DType::Bf16);
        assert!(matches!(map_dtype(Dtype::U8), DType::Other(_)));
    }
}
