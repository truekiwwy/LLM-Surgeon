//! Zero-copy inspection of a model directory or file: the tensor index and the
//! per-entry classification (role, layer, expert).
//!
//! ```text
//! ModelIndex::open(path)
//!   -> .safetensors file, a directory with *.safetensors (+ optional sharded
//!      index and config.json), or a .gguf file (mmap, header parsed, no copy)
//!   -> entries sorted by name, each with its Role / layer / expert, and the
//!      detected architecture family (config.json model_type or GGUF
//!      general.architecture) that drives the ablation exactness badge.
//! ```

pub mod dtype;
pub mod gguf;
pub mod metrics;
pub mod roles;
pub mod safetensors;

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use thiserror::Error;

use dtype::DType;
pub use roles::{Classified, Role};

/// The on-disk family a tensor name belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    SafeTensors,
    Gguf,
}

/// What a tensor is for, plus where its bytes live.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub name: String,
    /// Shape in row-major order (`shape.last()` indexes a row fastest),
    /// GGUF dims reversed to match the SafeTensors convention.
    pub shape: Vec<u64>,
    pub dtype: DType,
    /// Bytes of the tensor data on disk.
    pub byte_len: u64,
    /// File holding the data (the shard for a sharded checkpoint).
    pub file: PathBuf,
    /// Byte offset of the data inside `file`.
    pub offset: u64,
    pub role: Role,
    pub layer: Option<u32>,
    pub expert: Option<u32>,
    /// Fused MoE experts (`[experts, hidden, in]`) or a shared expert.
    pub moe: MoeKind,
}

impl TensorEntry {
    /// Number of stored elements.
    pub fn numel(&self) -> u64 {
        self.shape.iter().product()
    }

    /// The shape rendered `[a, b]`.
    pub fn shape_text(&self) -> String {
        let joined = self
            .shape
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!("[{joined}]")
    }

    /// The `L{layer}` prefix, or a blank when the tensor is outside the blocks.
    pub fn layer_text(&self) -> String {
        match self.layer {
            Some(l) => format!("L{l}"),
            None => "  ".to_owned(),
        }
    }
}

/// How the entry relates to a mixture-of-experts MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeKind {
    NotMoe,
    Dense,
    /// One row of a fused expert tensor.
    Fused,
    Shared,
    Router,
}

/// One loaded model: the entries and the architecture facts.
pub struct ModelIndex {
    pub root: PathBuf,
    pub format: Format,
    /// Architecture family (`llama`, `qwen3_moe`, ...), from config.json or GGUF
    /// metadata when available.
    pub family: Option<String>,
    /// Entries sorted by name.
    pub tensors: Vec<TensorEntry>,
    /// Highest block index seen, so the tree can label layers.
    pub num_layers: u32,
}

impl fmt::Debug for ModelIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModelIndex")
            .field("root", &self.root)
            .field("format", &self.format)
            .field("family", &self.family)
            .field("tensors", &self.tensors.len())
            .field("num_layers", &self.num_layers)
            .finish()
    }
}

/// Memory-maps a file read-only. The single `unsafe` of the crate lives here:
/// `Mmap::map` is unsound only when the file is truncated concurrently, which
/// the inspector never does (read-only use of model files, like the main
/// LLM-Surgeon engine).
#[allow(unsafe_code)]
pub(crate) fn map_file(path: &Path) -> Result<Mmap, ModelError> {
    let file = std::fs::File::open(path)?;
    // SAFETY: read-only mapping of an existing file; the inspector never
    // writes to nor truncates the files it inspects, and the Mmap keeps the
    // file handle alive for its own lifetime.
    unsafe { Mmap::map(&file) }.map_err(Into::into)
}

#[derive(Debug, Error)]
pub enum ModelError {
    #[error("chemin introuvable : {0}")]
    NotFound(String),
    #[error("format non reconnu (attendu .safetensors, un dossier de checkpoint ou .gguf) : {0}")]
    UnknownFormat(String),
    #[error("safetensors : {0}")]
    SafeTensors(#[from] ::safetensors::SafeTensorError),
    #[error("gguf : {0}")]
    Gguf(String),
    #[error("io : {0}")]
    Io(#[from] std::io::Error),
    #[error("json : {0}")]
    Json(#[from] serde_json::Error),
}

impl ModelIndex {
    /// Opens a `.safetensors` file, a checkpoint directory or a `.gguf` file.
    pub fn open(path: &Path) -> Result<ModelIndex, ModelError> {
        if path.is_dir() {
            let (tensors, format) = safetensors::open_dir(path)?;
            let family = read_config_family(path);
            Ok(Self::finish(path, format, family, tensors))
        } else if path.is_file() {
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase();
            if name.ends_with(".safetensors") {
                let tensors = safetensors::open_file(path)?;
                let family = read_config_family(path.parent().unwrap_or(Path::new(".")));
                Ok(Self::finish(path, Format::SafeTensors, family, tensors))
            } else if name.ends_with(".gguf") {
                let (tensors, family) = gguf::open_file(path)?;
                Ok(Self::finish(path, Format::Gguf, family, tensors))
            } else {
                Err(ModelError::UnknownFormat(path.display().to_string()))
            }
        } else {
            Err(ModelError::NotFound(path.display().to_string()))
        }
    }

    fn finish(
        root: &Path,
        format: Format,
        family: Option<String>,
        tensors: Vec<(String, Vec<u64>, DType, u64, PathBuf, u64)>,
    ) -> ModelIndex {
        let mut tensors: Vec<TensorEntry> = tensors
            .into_iter()
            .map(|(name, shape, dtype, byte_len, file, offset)| {
                let Classified {
                    role,
                    layer,
                    expert,
                    moe,
                } = roles::classify(&name, format);
                TensorEntry {
                    name,
                    shape,
                    dtype,
                    byte_len,
                    file,
                    offset,
                    role,
                    layer,
                    expert,
                    moe,
                }
            })
            .collect();
        tensors.sort_by(|a, b| a.name.cmp(&b.name));
        let num_layers = tensors.iter().filter_map(|t| t.layer).max().unwrap_or(0) + 1;
        ModelIndex {
            root: root.to_path_buf(),
            format,
            family,
            tensors,
            num_layers,
        }
    }

    /// Roles present in the model, in taxonomy order (for the filter cycle).
    pub fn roles_present(&self) -> Vec<Role> {
        let mut present: BTreeSet<Role> = self.tensors.iter().map(|t| t.role).collect();
        Role::ALL
            .iter()
            .copied()
            .filter(|r| present.remove(r))
            .collect()
    }

    /// First tensor index of a layer, in display order.
    pub fn first_of_layer(&self, layer: u32) -> Option<usize> {
        self.tensors.iter().position(|t| t.layer == Some(layer))
    }

    /// Exact tensor index by name.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }
}

/// The raw bytes of one entry, mmap-read from its file and clamped to the file
/// length (used by the metrics worker and by tests).
pub fn read_entry_bytes(entry: &TensorEntry) -> Result<Vec<u8>, std::io::Error> {
    if entry.byte_len == 0 {
        return Ok(Vec::new());
    }
    let map = map_file(&entry.file).map_err(|e| std::io::Error::other(e.to_string()))?;
    let from = entry.offset.min(map.len() as u64) as usize;
    let to = (entry.offset as usize)
        .saturating_add(entry.byte_len as usize)
        .min(map.len());
    Ok(map.get(from..to.max(from)).unwrap_or(&[]).to_vec())
}

/// Architecture family of a SafeTensors checkpoint, from its config.json when
/// present; tolerated missing or malformed.
fn read_config_family(dir: &Path) -> Option<String> {
    let config = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let json: serde_json::Value = serde_json::from_str(&config).ok()?;
    let model_type = json.get("model_type").and_then(|v| v.as_str());
    model_type.map_or_else(
        || {
            json.get("architectures")
                .and_then(|v| v.as_array())
                .and_then(|a| a.first())
                .and_then(|v| v.as_str())
                .map(family_from_hf_architecture)
        },
        |t| Some(t.to_owned()),
    )
}

/// `MistralForCausalLM` -> `mistral`; MoE families get their underscore back
/// (`Qwen3MoeForCausalLM` -> `qwen3_moe`) so they line up with the GGUF
/// `general.architecture` names used by the exactness table.
fn family_from_hf_architecture(arch: &str) -> String {
    let lowered = arch.to_ascii_lowercase();
    let stripped = lowered
        .strip_suffix("forcausallm")
        .or_else(|| lowered.strip_suffix("formaskedlm"))
        .unwrap_or(&lowered);
    match stripped.strip_suffix("moe") {
        Some(base) => format!("{base}_moe"),
        None => stripped.to_owned(),
    }
}

impl fmt::Display for Format {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Format::SafeTensors => f.write_str("safetensors"),
            Format::Gguf => f.write_str("gguf"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hf_architectures_map_to_families() {
        assert_eq!(family_from_hf_architecture("LlamaForCausalLM"), "llama");
        assert_eq!(
            family_from_hf_architecture("Qwen3MoeForCausalLM"),
            "qwen3_moe"
        );
        assert_eq!(
            family_from_hf_architecture("GPT2LMHeadModel"),
            "gpt2lmheadmodel"
        );
    }

    #[test]
    fn entries_expose_their_shape_and_numel() {
        let entry = TensorEntry {
            name: "model.layers.3.mlp.down_proj.weight".into(),
            shape: vec![1024, 4096],
            dtype: DType::F16,
            byte_len: 2 * 1024 * 4096,
            file: PathBuf::from("m.safetensors"),
            offset: 0,
            role: Role::MlpOut,
            layer: Some(3),
            expert: None,
            moe: MoeKind::NotMoe,
        };
        assert_eq!(entry.numel(), 1024 * 4096);
        assert_eq!(entry.shape_text(), "[1024, 4096]");
        assert_eq!(entry.layer_text(), "L3");
    }
}
