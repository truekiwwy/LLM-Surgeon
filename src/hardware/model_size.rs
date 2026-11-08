use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::error::{HardwareError, Result};

const INDEX_FILE: &str = "model.safetensors.index.json";

#[derive(Debug, Clone, Serialize)]
pub struct ModelSize {
    pub total_bytes: u64,
    pub shard_files: Vec<PathBuf>,
}

#[derive(Deserialize)]
struct Index {
    weight_map: HashMap<String, String>,
}

/// Size on disk of a `.safetensors` file or of a model directory
/// (sharded via `model.safetensors.index.json`, else every top-level `*.safetensors`).
pub fn compute_model_size(model_path: &Path) -> Result<ModelSize> {
    let meta = fs::metadata(model_path).map_err(|e| match e.kind() {
        ErrorKind::NotFound => HardwareError::PathNotFound(model_path.to_path_buf()),
        _ => HardwareError::io(model_path, &e),
    })?;

    if meta.is_file() {
        if !is_safetensors(model_path) {
            return Err(HardwareError::InvalidExtension(model_path.to_path_buf()));
        }
        return Ok(ModelSize {
            total_bytes: meta.len(),
            shard_files: vec![model_path.to_path_buf()],
        });
    }

    let index = model_path.join(INDEX_FILE);
    let shard_files = if index.is_file() {
        shards_from_index(model_path, &index)?
    } else {
        list_safetensors(model_path)?
    };
    if shard_files.is_empty() {
        return Err(HardwareError::NoSafetensors(model_path.to_path_buf()));
    }

    let mut total_bytes = 0u64;
    for shard in &shard_files {
        let len = fs::metadata(shard)
            .map_err(|e| match e.kind() {
                ErrorKind::NotFound => HardwareError::MissingShard {
                    shard: shard.clone(),
                },
                _ => HardwareError::io(shard, &e),
            })?
            .len();
        total_bytes = total_bytes.saturating_add(len);
    }
    Ok(ModelSize {
        total_bytes,
        shard_files,
    })
}

fn is_safetensors(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
}

fn shards_from_index(dir: &Path, index: &Path) -> Result<Vec<PathBuf>> {
    let corrupt = |message: String| HardwareError::CorruptIndex {
        path: index.to_path_buf(),
        message,
    };
    let raw = fs::read_to_string(index).map_err(|e| HardwareError::io(index, &e))?;
    let parsed: Index = serde_json::from_str(&raw).map_err(|e| corrupt(e.to_string()))?;

    // BTreeSet: dedup + deterministic order. Shard names come from a downloaded file,
    // so only bare file names are accepted (no `../`, no absolute path).
    let names: BTreeSet<String> = parsed.weight_map.into_values().collect();
    names
        .into_iter()
        .map(|name| {
            if Path::new(&name).file_name() != Some(name.as_ref()) {
                return Err(corrupt(format!("nom de shard invalide : {name}")));
            }
            Ok(dir.join(name))
        })
        .collect()
}

fn list_safetensors(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| HardwareError::io(dir, &e))? {
        let path = entry.map_err(|e| HardwareError::io(dir, &e))?.path();
        if path.is_file() && is_safetensors(&path) {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_sized(path: &Path, len: u64) {
        fs::File::create(path).unwrap().set_len(len).unwrap();
    }

    #[test]
    fn single_file_exact_size() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("model.safetensors");
        write_sized(&file, 12_345);
        let size = compute_model_size(&file).unwrap();
        assert_eq!(size.total_bytes, 12_345);
        assert_eq!(size.shard_files, vec![file]);
    }

    #[test]
    fn sharded_index_sums_deduplicated_shards() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write_sized(&p.join("a.safetensors"), 100);
        write_sized(&p.join("b.safetensors"), 200);
        write_sized(&p.join("c.safetensors"), 300);
        // Not referenced by the index: must be ignored.
        write_sized(&p.join("stray.safetensors"), 9_999);
        fs::write(
            p.join(INDEX_FILE),
            r#"{"metadata":{},"weight_map":{
                "w1":"a.safetensors","w2":"a.safetensors",
                "w3":"b.safetensors","w4":"c.safetensors","w5":"c.safetensors"}}"#,
        )
        .unwrap();
        let size = compute_model_size(p).unwrap();
        assert_eq!(size.total_bytes, 600);
        assert_eq!(size.shard_files.len(), 3);
    }

    #[test]
    fn directory_without_index_sums_top_level_files() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write_sized(&p.join("x.safetensors"), 10);
        write_sized(&p.join("y.SafeTensors"), 20);
        write_sized(&p.join("notes.txt"), 1_000);
        fs::create_dir(p.join("sub")).unwrap();
        write_sized(&p.join("sub").join("z.safetensors"), 1_000);
        assert_eq!(compute_model_size(p).unwrap().total_bytes, 30);
    }

    #[test]
    fn missing_shard_is_typed_error() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        write_sized(&p.join("a.safetensors"), 1);
        fs::write(
            p.join(INDEX_FILE),
            r#"{"weight_map":{"w1":"a.safetensors","w2":"gone.safetensors"}}"#,
        )
        .unwrap();
        let err = compute_model_size(p).unwrap_err();
        assert!(
            matches!(&err, HardwareError::MissingShard { shard } if shard.ends_with("gone.safetensors")),
            "{err:?}"
        );
    }

    #[test]
    fn corrupt_index_and_path_traversal_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        fs::write(p.join(INDEX_FILE), "{ not json").unwrap();
        assert!(matches!(
            compute_model_size(p),
            Err(HardwareError::CorruptIndex { .. })
        ));
        fs::write(
            p.join(INDEX_FILE),
            r#"{"weight_map":{"w":"../evil.safetensors"}}"#,
        )
        .unwrap();
        assert!(matches!(
            compute_model_size(p),
            Err(HardwareError::CorruptIndex { .. })
        ));
    }

    #[test]
    fn distinct_errors_for_empty_dir_bad_extension_missing_path() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        assert!(matches!(
            compute_model_size(p),
            Err(HardwareError::NoSafetensors(_))
        ));

        let bin = p.join("model.bin");
        write_sized(&bin, 1);
        assert!(matches!(
            compute_model_size(&bin),
            Err(HardwareError::InvalidExtension(_))
        ));

        assert!(matches!(
            compute_model_size(&p.join("nope")),
            Err(HardwareError::PathNotFound(_))
        ));
    }
}
