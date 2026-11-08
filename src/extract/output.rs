//! The direction file: one 1D F32 tensor `direction` plus `__metadata__`, written
//! atomically (sibling temporary file, then rename).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::error::{ExtractError, Result};

pub const TENSOR_NAME: &str = "direction";

/// Checked before any computation: an existing `--out` needs `--overwrite`, its directory
/// must exist, and the engine only reads `.safetensors` directions.
pub fn check_output(out: &Path, overwrite: bool) -> Result<()> {
    let is_safetensors = out
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"));
    if !is_safetensors {
        return Err(ExtractError::InvalidArgument(format!(
            "{} : l'extension .safetensors est requise",
            out.display()
        )));
    }
    let parent = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    if !parent.is_dir() {
        return Err(ExtractError::InvalidArgument(format!(
            "dossier de sortie inexistant : {}",
            parent.display()
        )));
    }
    if out.exists() && !overwrite {
        return Err(ExtractError::OutputExists(out.to_path_buf()));
    }
    Ok(())
}

/// SafeTensors bytes with a sorted header: the same direction and metadata always give
/// the same file. (The `safetensors` serializer keeps metadata in a `HashMap`, whose order
/// changes from one run to the next.) The result is re-read by the `safetensors` crate.
pub fn encode(direction: &[f32], metadata: &BTreeMap<String, String>) -> Result<Vec<u8>> {
    let data: Vec<u8> = direction.iter().flat_map(|v| v.to_le_bytes()).collect();
    let header = serde_json::json!({
        "__metadata__": metadata,
        TENSOR_NAME: { "dtype": "F32", "shape": [direction.len()], "data_offsets": [0, data.len()] },
    });
    let mut header = header.to_string();
    // Data starts on an 8-byte boundary, as the safetensors writer does.
    while !(8 + header.len()).is_multiple_of(8) {
        header.push(' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend_from_slice(&data);

    let parsed = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| ExtractError::Compute(format!("fichier produit illisible : {e}")))?;
    let view = parsed
        .tensor(TENSOR_NAME)
        .map_err(|e| ExtractError::Compute(e.to_string()))?;
    if view.shape() != [direction.len()] || view.dtype() != safetensors::Dtype::F32 {
        return Err(ExtractError::Compute("tenseur direction incohérent".into()));
    }
    Ok(bytes)
}

/// Temporary sibling (same volume, so `rename` is atomic), synced, then renamed over
/// `out`. Any failure removes the temporary: no partial file is ever left behind.
pub fn write_atomic(out: &Path, bytes: &[u8], overwrite: bool) -> Result<()> {
    let io = |path: &Path, e: std::io::Error| ExtractError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    };
    let name = out
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "direction.safetensors".into());
    let temp: PathBuf = out.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let result = (|| {
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| io(&temp, e))?;
        file.write_all(bytes).map_err(|e| io(&temp, e))?;
        file.sync_all().map_err(|e| io(&temp, e))?;
        drop(file);
        // Re-checked at the last moment: the file may have appeared during the run.
        if out.exists() && !overwrite {
            return Err(ExtractError::OutputExists(out.to_path_buf()));
        }
        std::fs::rename(&temp, out).map_err(|e| io(out, e))
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&temp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta() -> BTreeMap<String, String> {
        [("layer", "2"), ("site", "resid_post"), ("model", "m")]
            .into_iter()
            .map(|(k, v)| (k.to_owned(), v.to_owned()))
            .collect()
    }

    #[test]
    fn encoded_file_is_deterministic_and_readable_by_the_engine() {
        let v = [0.6f32, -0.8, 0.0];
        let a = encode(&v, &meta()).unwrap();
        assert_eq!(a, encode(&v, &meta()).unwrap());
        assert_eq!((8 + u64::from_le_bytes(a[..8].try_into().unwrap())) % 8, 0);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("d.safetensors");
        write_atomic(&out, &a, false).unwrap();
        assert_eq!(
            crate::engine::load_direction_vector(&out, Some(TENSOR_NAME)).unwrap(),
            v
        );
        assert_eq!(crate::engine::load_direction_vector(&out, None).unwrap(), v);
        let (_, metadata) = safetensors::SafeTensors::read_metadata(&a).unwrap();
        assert_eq!(metadata.metadata().as_ref().unwrap()["layer"], "2");
    }

    #[test]
    fn existing_output_and_bad_targets_are_refused_without_residue() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("d.safetensors");
        std::fs::write(&out, "keep").unwrap();
        assert!(matches!(
            check_output(&out, false),
            Err(ExtractError::OutputExists(_))
        ));
        assert!(check_output(&out, true).is_ok());
        assert!(matches!(
            write_atomic(&out, b"new", false),
            Err(ExtractError::OutputExists(_))
        ));
        assert_eq!(std::fs::read(&out).unwrap(), b"keep");
        write_atomic(&out, b"new", true).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"new");
        let leftovers = std::fs::read_dir(dir.path()).unwrap().count();
        assert_eq!(leftovers, 1, "no temporary file left");
        assert!(check_output(&dir.path().join("absent/d.safetensors"), false).is_err());
        assert!(check_output(&dir.path().join("d.json"), false).is_err());
    }
}
