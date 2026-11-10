//! Detaching `lm_head` from the input embeddings before `embed_tokens` is patched.
//!
//! A checkpoint with tied embeddings stores `embed_tokens` only and the loader reuses it as
//! the output projection: patching it would also deform the logits. [`UntiePlan::execute`]
//! makes the head independent first:
//!
//! 1. the shard holding the embeddings is rewritten with `lm_head.weight`, a byte copy of
//!    the still unpatched embeddings, appended after the existing data (the other tensors
//!    keep their offsets; the temp file is re-read, then renamed over the shard);
//! 2. `config.json` gets `"tie_word_embeddings": false`;
//! 3. a `model.safetensors.index.json` gets the `weight_map` entry of the new tensor.
//!
//! Each file is replaced atomically. The caller backs up [`UntiePlan::files`] first, so a
//! failure between two steps is undone by the rollback of `apply`, and must not hold a
//! mapping of the shard while `execute` runs (Windows refuses the rename otherwise).

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use serde::Serialize;
use serde_json::{json, Map, Value};

use super::error::{PatchError, Result};

pub const LM_HEAD: &str = "lm_head.weight";
const CONFIG_FILE: &str = "config.json";
const INDEX_FILE: &str = "model.safetensors.index.json";
const TIE_KEY: &str = "tie_word_embeddings";
/// Limit of the safetensors format.
const MAX_HEADER_BYTES: u64 = 100_000_000;

/// What `apply` must do so that patching `embed_tokens` leaves the output projection alone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UntiePlan {
    /// Shard holding the embeddings.
    pub shard: PathBuf,
    pub embed_tensor: String,
    /// `lm_head.weight` is absent: a copy of the embeddings is appended to `shard`.
    pub copy_embeddings: bool,
    /// Its `tie_word_embeddings` becomes `false`.
    pub config: PathBuf,
    /// Index of a sharded model, completed when the tensor is added.
    pub index: Option<PathBuf>,
}

fn invalid(path: &Path, message: impl std::fmt::Display) -> PatchError {
    PatchError::Io {
        path: path.to_path_buf(),
        message: message.to_string(),
    }
}

/// Header length and JSON header of a safetensors file.
fn read_header(path: &Path) -> Result<(u64, Map<String, Value>)> {
    let mut file = File::open(path).map_err(|e| PatchError::io(path, &e))?;
    let file_len = file.metadata().map_err(|e| PatchError::io(path, &e))?.len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .map_err(|e| PatchError::io(path, &e))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > MAX_HEADER_BYTES || header_len + 8 > file_len {
        return Err(invalid(path, "en-tête safetensors invalide"));
    }
    let mut raw = vec![0u8; header_len as usize];
    file.read_exact(&mut raw)
        .map_err(|e| PatchError::io(path, &e))?;
    let header = serde_json::from_slice(&raw).map_err(|e| invalid(path, e))?;
    Ok((header_len, header))
}

fn read_json(path: &Path) -> Result<Map<String, Value>> {
    let raw = fs::read_to_string(path).map_err(|e| PatchError::io(path, &e))?;
    serde_json::from_str(raw.trim_start_matches('\u{feff}')).map_err(|e| invalid(path, e))
}

/// `None` when the head is already independent: `lm_head.weight` exists and the config
/// does not tie it. A tied model without `config.json` is refused: nothing could tell a
/// loader that the head is no longer the embeddings.
pub fn plan(shards: &[PathBuf], embed_tensor: &str) -> Result<Option<UntiePlan>> {
    let mut holder = None;
    let mut head_present = false;
    for shard in shards {
        let (_, header) = read_header(shard)?;
        head_present |= header.contains_key(LM_HEAD);
        if header.contains_key(embed_tensor) {
            holder.get_or_insert(shard);
        }
    }
    let shard = holder.ok_or_else(|| {
        PatchError::Backend(format!(
            "{embed_tensor} : tenseur introuvable dans le modèle"
        ))
    })?;
    let dir = shard.parent().unwrap_or(Path::new(""));
    let config = dir.join(CONFIG_FILE);
    if !config.is_file() {
        return match head_present {
            true => Ok(None),
            false => Err(PatchError::Backend(format!(
                "{embed_tensor} est partagé avec lm_head (aucun {LM_HEAD}) et {} est introuvable : \
                 détachement impossible, embed_tokens n'est pas patchable sur ce modèle",
                config.display()
            ))),
        };
    }
    // Absent means tied for a transformers loader: written explicitly either way.
    let tied = read_json(&config)?.get(TIE_KEY) != Some(&Value::Bool(false));
    if head_present && !tied {
        return Ok(None);
    }
    let index = dir.join(INDEX_FILE);
    Ok(Some(UntiePlan {
        shard: shard.clone(),
        embed_tensor: embed_tensor.to_owned(),
        copy_embeddings: !head_present,
        config,
        index: (!head_present && index.is_file()).then_some(index),
    }))
}

impl UntiePlan {
    /// Every file `execute` replaces: back them up first.
    pub fn files(&self) -> Vec<PathBuf> {
        let shard = self.copy_embeddings.then(|| self.shard.clone());
        shard
            .into_iter()
            .chain([self.config.clone()])
            .chain(self.index.clone())
            .collect()
    }

    pub fn execute(&self) -> Result<()> {
        if self.copy_embeddings {
            append_copy(&self.shard, &self.embed_tensor, LM_HEAD)?;
        }
        let mut config = read_json(&self.config)?;
        config.insert(TIE_KEY.into(), Value::Bool(false));
        write_json(&self.config, &config)?;
        if let Some(index) = &self.index {
            let mut json = read_json(index)?;
            let shard_name = self
                .shard
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            match json.get_mut("weight_map").and_then(Value::as_object_mut) {
                Some(map) => map.insert(LM_HEAD.into(), Value::String(shard_name)),
                None => return Err(invalid(index, "weight_map absent")),
            };
            write_json(index, &json)?;
        }
        Ok(())
    }
}

fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".ws-untie");
    PathBuf::from(name)
}

/// Writes through a sibling temp file, removed on any failure, then renames it over `path`.
fn replace_file(path: &Path, write: impl FnOnce(&mut File) -> io::Result<()>) -> Result<()> {
    let tmp = temp_path(path);
    let written = File::create(&tmp)
        .and_then(|mut file| {
            write(&mut file)?;
            file.sync_all()
        })
        .and_then(|()| fs::rename(&tmp, path));
    written.map_err(|e| {
        let _ = fs::remove_file(&tmp);
        PatchError::io(path, &e)
    })
}

fn write_json(path: &Path, json: &Map<String, Value>) -> Result<()> {
    let mut text = serde_json::to_string_pretty(json).map_err(|e| invalid(path, e))?;
    text.push('\n');
    replace_file(path, |file| file.write_all(text.as_bytes()))
}

/// Rewrites `shard` with `dst`, a byte copy of `src`, appended after the existing data.
fn append_copy(shard: &Path, src: &str, dst: &str) -> Result<()> {
    let (header_len, mut header) = read_header(shard)?;
    let data_start = 8 + header_len;
    let file_len = fs::metadata(shard)
        .map_err(|e| PatchError::io(shard, &e))?
        .len();
    let data_len = file_len - data_start;
    let entry = header
        .get(src)
        .cloned()
        .ok_or_else(|| invalid(shard, format!("{src} absent de l'en-tête")))?;
    let offset = |i: usize| entry["data_offsets"][i].as_u64();
    let (start, end) = match (offset(0), offset(1)) {
        (Some(start), Some(end)) if start <= end && end <= data_len => (start, end),
        _ => return Err(invalid(shard, format!("{src} : data_offsets invalides"))),
    };
    let mut copy = entry;
    copy["data_offsets"] = json!([data_len, data_len + (end - start)]);
    header.insert(dst.to_owned(), copy);
    let mut text = serde_json::to_string(&header).map_err(|e| invalid(shard, e))?;
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let expected_len = 8 + text.len() as u64 + data_len + (end - start);

    replace_file(shard, |out| {
        let mut source = File::open(shard)?;
        out.write_all(&(text.len() as u64).to_le_bytes())?;
        out.write_all(text.as_bytes())?;
        source.seek(SeekFrom::Start(data_start))?;
        io::copy(&mut Read::by_ref(&mut source).take(data_len), out)?;
        source.seek(SeekFrom::Start(data_start + start))?;
        io::copy(&mut source.take(end - start), out)?;
        // Checked before the rename: a short copy must never replace the shard.
        match out.metadata()?.len() == expected_len {
            true => Ok(()),
            false => Err(io::Error::other("copie incomplète du shard")),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ReadOnlyModel;

    /// F32 safetensors with the given tensors, in order.
    fn write_shard(path: &Path, tensors: &[(&str, Vec<usize>)]) {
        let mut header = Map::new();
        header.insert("__metadata__".into(), json!({"format": "pt"}));
        let mut data = Vec::new();
        for (i, (name, shape)) in tensors.iter().enumerate() {
            let start = data.len();
            let n: usize = shape.iter().product();
            data.extend((0..n).flat_map(|k| ((i * 100 + k) as f32).to_le_bytes()));
            header.insert(
                (*name).into(),
                json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
            );
        }
        let mut text = serde_json::to_string(&header).unwrap();
        while !text.len().is_multiple_of(8) {
            text.push(' ');
        }
        let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(text.as_bytes());
        bytes.extend_from_slice(&data);
        fs::write(path, bytes).unwrap();
    }

    const EMBED: &str = "model.embed_tokens.weight";

    fn tied_model(dir: &Path) -> PathBuf {
        let shard = dir.join("model.safetensors");
        write_shard(
            &shard,
            &[
                (EMBED, vec![5, 3]),
                ("model.layers.0.mlp.down_proj.weight", vec![3, 4]),
                ("model.norm.weight", vec![3]),
            ],
        );
        fs::write(
            dir.join(CONFIG_FILE),
            r#"{"model_type":"qwen2","tie_word_embeddings":true,"vocab_size":5}"#,
        )
        .unwrap();
        shard
    }

    fn tensor(shard: &Path, name: &str) -> Vec<u8> {
        ReadOnlyModel::open(shard)
            .unwrap()
            .tensor(name)
            .unwrap()
            .bytes
            .to_vec()
    }

    #[test]
    fn tied_model_gets_an_unpatched_head_and_an_untied_config() {
        let dir = tempfile::tempdir().unwrap();
        let shard = tied_model(dir.path());
        let before: Vec<Vec<u8>> = [EMBED, "model.layers.0.mlp.down_proj.weight"]
            .iter()
            .map(|n| tensor(&shard, n))
            .collect();

        let plan = plan(std::slice::from_ref(&shard), EMBED).unwrap().unwrap();
        assert!(plan.copy_embeddings && plan.index.is_none());
        assert_eq!(plan.files(), [shard.clone(), dir.path().join(CONFIG_FILE)]);
        plan.execute().unwrap();

        // The engine's strict parser accepts the rewritten shard.
        let model = ReadOnlyModel::open(&shard).unwrap();
        assert_eq!(model.index().tensors.len(), 4);
        assert_eq!(model.index().get(LM_HEAD).unwrap().shape, [5, 3]);
        assert_eq!(model.metadata()["format"], "pt");
        drop(model);
        assert_eq!(tensor(&shard, LM_HEAD), before[0]);
        assert_eq!(tensor(&shard, EMBED), before[0]);
        assert_eq!(
            tensor(&shard, "model.layers.0.mlp.down_proj.weight"),
            before[1]
        );
        let config = read_json(&dir.path().join(CONFIG_FILE)).unwrap();
        assert_eq!(config[TIE_KEY], false);
        assert_eq!(config["vocab_size"], 5, "other fields kept");
        assert!(!temp_path(&shard).exists());

        assert_eq!(super::plan(&[shard], EMBED).unwrap(), None, "idempotent");
    }

    #[test]
    fn existing_head_only_needs_the_config_and_a_sharded_index_is_completed() {
        let dir = tempfile::tempdir().unwrap();
        let shard = dir.path().join("model.safetensors");
        write_shard(&shard, &[(EMBED, vec![2, 2]), (LM_HEAD, vec![2, 2])]);
        fs::write(dir.path().join(CONFIG_FILE), "{}").unwrap();
        let bytes = fs::read(&shard).unwrap();
        let config_only = plan(std::slice::from_ref(&shard), EMBED).unwrap().unwrap();
        assert!(!config_only.copy_embeddings, "absent tie flag means tied");
        assert_eq!(config_only.files(), [dir.path().join(CONFIG_FILE)]);
        config_only.execute().unwrap();
        assert_eq!(fs::read(&shard).unwrap(), bytes, "shard untouched");
        assert_eq!(
            read_json(&dir.path().join(CONFIG_FILE)).unwrap()[TIE_KEY],
            false
        );

        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("model-00001-of-00002.safetensors");
        let second = dir.path().join("model-00002-of-00002.safetensors");
        write_shard(&first, &[("model.norm.weight", vec![2])]);
        write_shard(&second, &[(EMBED, vec![2, 2])]);
        fs::write(
            dir.path().join(CONFIG_FILE),
            r#"{"tie_word_embeddings":true}"#,
        )
        .unwrap();
        fs::write(
            dir.path().join(INDEX_FILE),
            r#"{"metadata":{},"weight_map":{"model.embed_tokens.weight":"model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();
        let sharded = plan(&[first.clone(), second.clone()], EMBED)
            .unwrap()
            .unwrap();
        assert_eq!(sharded.shard, second);
        assert_eq!(sharded.files().len(), 3);
        sharded.execute().unwrap();
        let index = read_json(&dir.path().join(INDEX_FILE)).unwrap();
        assert_eq!(
            index["weight_map"][LM_HEAD],
            "model-00002-of-00002.safetensors"
        );
        assert!(ReadOnlyModel::open(&second)
            .unwrap()
            .index()
            .get(LM_HEAD)
            .is_ok());
    }

    #[test]
    fn refusals_leave_every_file_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let shard = tied_model(dir.path());
        let bytes = fs::read(&shard).unwrap();
        fs::remove_file(dir.path().join(CONFIG_FILE)).unwrap();
        let err = plan(std::slice::from_ref(&shard), EMBED).unwrap_err();
        assert!(err.to_string().contains("détachement impossible"), "{err}");
        assert!(plan(std::slice::from_ref(&shard), "absent.weight").is_err());
        assert_eq!(fs::read(&shard).unwrap(), bytes);

        // An explicit head without config is already independent.
        write_shard(&shard, &[(EMBED, vec![2, 2]), (LM_HEAD, vec![2, 2])]);
        assert_eq!(plan(&[shard], EMBED).unwrap(), None);
    }
}
