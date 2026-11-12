//! Model access for `surgeon graph`: SafeTensors and GGUF, streamed row by row.
//!
//! Nothing here materialises a matrix. A [`TensorRef`] is a row-addressable view of one
//! residual-stream matrix (a fused expert tensor is split into one view per expert), and
//! [`NodeSource`] implements [`crate::surgery::rsvd::RowSource`] over it. The same source
//! decodes GGUF blocks with [`crate::patch::gguf::quant::dequantize_row`], so graph metrics
//! are computed on exactly the values the patch engine would read.
//!
//! Every graph run hashes the complete model (SafeTensors shards in path order, or the
//! GGUF file) with SHA-256: the cache key and `graph.json` both carry that digest.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use sha2::{Digest, Sha256};

use crate::arch::{self, Format, Role};
use crate::engine::{DType, ReadOnlyModel};
use crate::patch::gguf;
use crate::patch::gguf::quant::{dequantize_row, Codec};
use crate::surgery::rsvd::RowSource;

use super::error::{GraphError, Result};

const HASH_BUFFER: usize = 1 << 20;

/// `canonicalize` returns verbatim `\\?\C:\...` paths on Windows; the report and the MCP
/// envelope are nicer, and external tools safer, with the plain short form.
fn plain_path(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        let plain = path
            .to_str()
            .and_then(|s| s.strip_prefix(r"\\?\"))
            .filter(|rest| {
                let b = rest.as_bytes();
                b.len() >= 3
                    && b[0].is_ascii_alphabetic()
                    && b[1] == b':'
                    && b[2] == b'\\'
                    && b.len() < 260
            });
        if let Some(plain) = plain {
            return PathBuf::from(plain);
        }
    }
    path
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFormat {
    SafeTensors,
    Gguf,
}

impl ModelFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SafeTensors => "safetensors",
            Self::Gguf => "gguf",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SafeDtype {
    F32,
    F16,
    Bf16,
    F64,
}

impl SafeDtype {
    fn from_dtype(dtype: &DType) -> Option<Self> {
        match dtype {
            DType::F32 => Some(Self::F32),
            DType::BF16 => Some(Self::Bf16),
            DType::F16 => Some(Self::F16),
            DType::Other(name) if name.eq_ignore_ascii_case("F64") => Some(Self::F64),
            _ => None,
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
            Self::F64 => 8,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F64 => "F64",
        }
    }
}

/// A row-addressable matrix, with the expert offset already folded into `row_offset`.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub name: String,
    pub role: super::GraphRole,
    pub layer: Option<usize>,
    pub expert: Option<usize>,
    pub dtype: String,
    /// `[rows, cols]` / `[out, in]`; the residual dimension is `rows` for writers
    /// (`embed` excepted, whose rows are tokens) and `cols` for readers.
    pub shape: [usize; 2],
    pub weight_bytes: u64,
    pub storage: Storage,
    /// Byte offset of the first row of this expert inside `storage`.
    pub row_offset: usize,
}

#[derive(Debug, Clone)]
pub enum Storage {
    Safe {
        shard: usize,
        /// Absolute byte offset of the tensor's first byte in the shard.
        start: usize,
        /// Bytes per row (`cols * dtype_size`).
        row_stride: usize,
        dtype: SafeDtype,
    },
    Gguf {
        /// Absolute byte offset of the tensor's first byte in the file.
        start: u64,
        row_bytes: usize,
        codec: Codec,
    },
}

impl TensorRef {
    pub fn m(&self) -> usize {
        self.shape[0]
    }

    pub fn n(&self) -> usize {
        self.shape[1]
    }

    pub fn is_writer(&self) -> bool {
        self.role.is_writer()
    }
}

#[derive(Debug)]
enum Store {
    Safe(Vec<ReadOnlyModel>),
    Gguf { map: memmap2::Mmap },
}

/// Everything a graph node needs from the model, plus the classification that drives the
/// causal order.
pub struct ModelData {
    pub path: PathBuf,
    pub format: ModelFormat,
    pub sha256: String,
    pub arch_id: String,
    pub arch_source: String,
    pub hidden: usize,
    pub layers: usize,
    pub parallel_blocks: bool,
    pub gemma: bool,
    pub tied_lm_head: bool,
    pub tensors: Vec<TensorRef>,
    pub skipped: Vec<SkippedTensor>,
    pub warnings: Vec<String>,
    store: Store,
    /// 1D norms, used only by the Gemma `1 + gamma` folding.
    norms: Vec<(String, Vec<f32>)>,
}

#[derive(
    Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize, schemars::JsonSchema,
)]
pub struct SkippedTensor {
    pub name: String,
    pub reason: String,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Counters {
    pub nan: u64,
    pub inf: u64,
}

impl Counters {
    pub fn total(self) -> u64 {
        self.nan + self.inf
    }
}

struct RawTensor {
    name: String,
    role: RawRole,
    layer: Option<usize>,
    expert: Option<usize>,
    dtype: String,
    /// `[rows, cols]` for 2D, `[experts, rows, cols]` for a fused expert tensor.
    shape: Vec<usize>,
    weight_bytes: u64,
    storage: Storage,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RawRole {
    Graph(super::GraphRole),
    Router,
    Ignored,
}

impl ModelData {
    pub fn open(path: &Path) -> Result<Self> {
        let canonical =
            plain_path(std::fs::canonicalize(path).map_err(|e| GraphError::io(path, &e))?);
        if canonical.is_dir() {
            Self::open_safe(&canonical)
        } else if gguf::is_gguf(&canonical) {
            Self::open_gguf(&canonical)
        } else {
            let extension = canonical
                .extension()
                .map(|e| e.to_string_lossy().to_ascii_lowercase())
                .unwrap_or_default();
            match extension.as_str() {
                "safetensors" | "json" => Self::open_safe(&canonical),
                _ => Err(GraphError::Model(format!(
                    "{} : format non reconnu (attendu .safetensors, dossier shardé ou .gguf)",
                    canonical.display()
                ))),
            }
        }
    }

    fn open_safe(path: &Path) -> Result<Self> {
        let size = crate::hardware::compute_model_size(path)
            .map_err(|e| GraphError::Model(e.to_string()))?;
        let resolution = arch::resolve(path, None, None)?;
        let mut shards = Vec::with_capacity(size.shard_files.len());
        let mut raw = Vec::new();
        for (index, shard) in size.shard_files.iter().enumerate() {
            let model = ReadOnlyModel::open(shard)?;
            for info in &model.index().tensors {
                raw.push(raw_safe(index, info, &resolution.spec));
            }
            shards.push(model);
        }
        let sha256 = hash_files(&size.shard_files)?;
        let store = Store::Safe(shards);
        Self::assemble(
            path.to_path_buf(),
            ModelFormat::SafeTensors,
            sha256,
            resolution,
            raw,
            store,
        )
    }

    fn open_gguf(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| GraphError::io(path, &e))?;
        let map = crate::kernel::map_read(&file).map_err(|e| GraphError::io(path, &e))?;
        let parsed = gguf::parse(&map)?;
        let resolution = arch::resolve(path, Some(parsed.architecture()), None)?;
        let raw = parsed
            .tensors
            .iter()
            .map(|entry| raw_gguf(entry, &resolution.spec))
            .collect::<Vec<_>>();
        let sha256 = hash_files(std::slice::from_ref(&path.to_path_buf()))?;
        let store = Store::Gguf { map };
        Self::assemble(
            path.to_path_buf(),
            ModelFormat::Gguf,
            sha256,
            resolution,
            raw,
            store,
        )
    }

    fn assemble(
        path: PathBuf,
        format: ModelFormat,
        sha256: String,
        resolution: arch::Resolution,
        raw: Vec<RawTensor>,
        store: Store,
    ) -> Result<Self> {
        let parallel_blocks = resolution.spec.parallel_block;
        let gemma = resolution.spec.id.to_ascii_lowercase().contains("gemma");
        let mut warnings = Vec::new();
        let mut skipped = Vec::new();

        let mut norms = Vec::new();
        if gemma {
            for raw in &raw {
                if raw.shape.len() == 1 && is_norm_name(&raw.name) {
                    if let Some(values) = read_raw_vector(&store, raw) {
                        norms.push((raw.name.clone(), values));
                    }
                }
            }
        }

        let hidden = raw
            .iter()
            .find(|t| t.role == RawRole::Graph(super::GraphRole::Embed))
            .and_then(|t| t.shape.last().copied())
            .or_else(|| {
                raw.iter()
                    .find(|t| t.role == RawRole::Graph(super::GraphRole::AttnOut))
                    .and_then(|t| t.shape.last().copied())
            })
            .ok_or_else(|| {
                GraphError::Model(
                    "aucun tenseur d'embedding ni de sortie d'attention reconnu".into(),
                )
            })?;

        let layers = raw
            .iter()
            .filter_map(|t| t.layer)
            .max()
            .map(|l| l + 1)
            .unwrap_or(0);

        let mut tensors = Vec::new();
        let mut has_lm_head = false;
        for raw in &raw {
            if unsupported_storage(raw) {
                skipped.push(SkippedTensor {
                    name: raw.name.clone(),
                    reason: "dtype non pris en charge (matrix metrics need f32/f16/bf16/f64 or a patchable GGUF block)".into(),
                });
                continue;
            }
            match raw.role {
                RawRole::Ignored => continue,
                RawRole::Router => {
                    skipped.push(SkippedTensor {
                        name: raw.name.clone(),
                        reason: "routeur MoE : hors du graphe (aucune direction résiduelle lue)"
                            .into(),
                    });
                    continue;
                }
                RawRole::Graph(role) => {
                    if role == super::GraphRole::LmHead {
                        has_lm_head = true;
                    }
                    if raw.shape.len() == 3 {
                        let experts = raw.shape[0];
                        let (rows, cols) = (raw.shape[1], raw.shape[2]);
                        let stride = match &raw.storage {
                            Storage::Safe { row_stride, .. } => *row_stride,
                            Storage::Gguf { row_bytes, .. } => *row_bytes,
                        };
                        let bytes_each = raw.weight_bytes / experts.max(1) as u64;
                        for expert in 0..experts {
                            push_ref(
                                &mut tensors,
                                raw,
                                RefSpec {
                                    role,
                                    shape: [rows, cols],
                                    weight_bytes: bytes_each,
                                    row_offset: expert * rows * stride,
                                    expert: Some(expert),
                                },
                                &mut warnings,
                                hidden,
                            );
                        }
                    } else if raw.shape.len() == 2 {
                        push_ref(
                            &mut tensors,
                            raw,
                            RefSpec {
                                role,
                                shape: [raw.shape[0], raw.shape[1]],
                                weight_bytes: raw.weight_bytes,
                                row_offset: 0,
                                expert: raw.expert,
                            },
                            &mut warnings,
                            hidden,
                        );
                    } else {
                        skipped.push(SkippedTensor {
                            name: raw.name.clone(),
                            reason: format!("{} dimensions : matrice 2D attendue", raw.shape.len()),
                        });
                    }
                }
            }
        }

        let embed_index = tensors
            .iter()
            .position(|t| t.role == super::GraphRole::Embed);
        let tied_lm_head = !has_lm_head;
        if let Some(index) = embed_index.filter(|_| tied_lm_head) {
            let mut tied = tensors[index].clone();
            tied.role = super::GraphRole::LmHead;
            tied.name = format!("{} (tête liée)", tied.name);
            tensors.push(tied);
            warnings.push(
                "lm_head.weight absent : tête liée aux embeddings, le nœud lm_head réutilise \
                 la matrice d'embedding"
                    .into(),
            );
        }

        tensors.sort_by(|a, b| {
            let seq = |t: &TensorRef| super::causal_seq(t.role, t.layer, layers, parallel_blocks);
            seq(a)
                .cmp(&seq(b))
                .then_with(|| a.layer.cmp(&b.layer))
                .then_with(|| a.expert.cmp(&b.expert))
                .then_with(|| a.name.cmp(&b.name))
        });

        if tensors.is_empty() {
            return Err(GraphError::Model(
                "aucun écrivain ni lecteur du flux résiduel reconnu".into(),
            ));
        }

        let arch_source = serde_json::to_value(resolution.source)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();

        Ok(Self {
            path,
            format,
            sha256,
            arch_id: resolution.spec.id.clone(),
            arch_source,
            hidden,
            layers,
            parallel_blocks,
            gemma,
            tied_lm_head,
            tensors,
            skipped,
            warnings,
            store,
            norms,
        })
    }

    /// The `1 + gamma` factor of a Gemma reader, or `None`.
    pub fn gamma_for(&self, tensor: &TensorRef) -> Option<Vec<f32>> {
        if !self.gemma {
            return None;
        }
        let layer = tensor.layer?;
        let attention = matches!(
            tensor.role,
            super::GraphRole::Q | super::GraphRole::K | super::GraphRole::V
        );
        let candidates: &[&str] = match attention {
            true => &[
                "input_layernorm.weight",
                "attn_norm.weight",
                "pre_attention_layernorm.weight",
            ],
            false => &[
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "ffn_norm.weight",
            ],
        };
        let suffix = candidates
            .iter()
            .find(|suffix| norm_matches(&tensor.name, layer, suffix))
            .copied()?;
        let name = norm_name(&tensor.name, layer, suffix)?;
        self.norm_named(&name)
            .map(|g| g.iter().map(|x| 1.0 + x).collect())
    }

    fn norm_named(&self, name: &str) -> Option<&[f32]> {
        self.norms
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, values)| values.as_slice())
    }

    /// Reads row `row` of `tensor` into `out` (length `n`), applying no transform.
    pub fn read_row(&self, tensor: &TensorRef, row: usize, out: &mut [f32]) -> Result<()> {
        if row >= tensor.m() || out.len() != tensor.n() {
            return Err(GraphError::Compute(format!(
                "{} : ligne {row} hors de {}×{}",
                tensor.name,
                tensor.m(),
                tensor.n()
            )));
        }
        match (&self.store, &tensor.storage) {
            (
                Store::Safe(shards),
                Storage::Safe {
                    shard,
                    start: _,
                    row_stride,
                    dtype,
                },
            ) => {
                let model = shards.get(*shard).ok_or_else(|| {
                    GraphError::Model(format!("{} : shard {shard} absent", tensor.name))
                })?;
                let view = model.tensor(&raw_name(&tensor.name))?;
                // `view.bytes` starts at this tensor's first byte: `start` is only metadata.
                let offset = tensor
                    .row_offset
                    .checked_add(row.checked_mul(*row_stride).ok_or_else(|| {
                        GraphError::Compute("offset de ligne hors limites".into())
                    })?)
                    .ok_or_else(|| GraphError::Compute("offset de ligne hors limites".into()))?;
                let end = offset
                    .checked_add(*row_stride)
                    .ok_or_else(|| GraphError::Compute("fin de ligne hors limites".into()))?;
                let bytes = view.bytes.get(offset..end).ok_or_else(|| {
                    GraphError::Model(format!("{} : ligne {row} hors du fichier", tensor.name))
                })?;
                decode_safe_row(*dtype, bytes, out);
                Ok(())
            }
            (
                Store::Gguf { map, .. },
                Storage::Gguf {
                    start,
                    row_bytes,
                    codec,
                },
            ) => {
                let offset = start
                    .checked_add(tensor.row_offset as u64)
                    .and_then(|o| o.checked_add((row * row_bytes) as u64))
                    .ok_or_else(|| GraphError::Compute("offset GGUF hors limites".into()))?;
                let offset = usize::try_from(offset)
                    .map_err(|_| GraphError::Compute("offset GGUF hors limites".into()))?;
                let end = offset
                    .checked_add(*row_bytes)
                    .ok_or_else(|| GraphError::Compute("fin de ligne GGUF hors limites".into()))?;
                let bytes = map.get(offset..end).ok_or_else(|| {
                    GraphError::Model(format!("{} : ligne {row} hors du fichier", tensor.name))
                })?;
                dequantize_row(*codec, bytes, out);
                Ok(())
            }
            _ => Err(GraphError::Compute(format!(
                "{} : stockage incompatible avec le format",
                tensor.name
            ))),
        }
    }
}

fn unsupported_storage(raw: &RawTensor) -> bool {
    match &raw.storage {
        Storage::Safe { row_stride, .. } => raw.shape.len() >= 2 && *row_stride == 0,
        Storage::Gguf { row_bytes, .. } => raw.shape.len() >= 2 && *row_bytes == 0,
    }
}

struct RefSpec {
    role: super::GraphRole,
    shape: [usize; 2],
    weight_bytes: u64,
    row_offset: usize,
    expert: Option<usize>,
}

fn push_ref(
    out: &mut Vec<TensorRef>,
    raw: &RawTensor,
    spec: RefSpec,
    warnings: &mut Vec<String>,
    hidden: usize,
) {
    let RefSpec {
        role,
        shape,
        weight_bytes,
        row_offset,
        expert,
    } = spec;
    let residual_dim = match role {
        super::GraphRole::Embed => shape[1],
        super::GraphRole::AttnOut | super::GraphRole::MlpOut => shape[0],
        _ => shape[1],
    };
    if residual_dim != hidden {
        warnings.push(format!(
            "{} : dimension résiduelle {residual_dim} ≠ hidden {hidden}, tenseur ignoré",
            raw.name
        ));
        return;
    }
    out.push(TensorRef {
        name: raw.name.clone(),
        role,
        layer: raw.layer,
        expert,
        dtype: raw.dtype.clone(),
        shape,
        weight_bytes,
        storage: raw.storage.clone(),
        row_offset,
    });
}

fn raw_safe(shard: usize, info: &crate::engine::TensorInfo, spec: &arch::ArchSpec) -> RawTensor {
    let dtype = SafeDtype::from_dtype(&info.dtype);
    let shape = info.shape.clone();
    let storage = match (dtype, shape.last().copied()) {
        (Some(dtype), Some(cols)) => Storage::Safe {
            shard,
            start: info.byte_start,
            row_stride: cols * dtype.bytes(),
            dtype,
        },
        _ => Storage::Safe {
            shard,
            start: info.byte_start,
            row_stride: 0,
            dtype: SafeDtype::F32,
        },
    };
    let classification = classify(&info.name, Format::SafeTensors, shape.len() == 3, spec);
    RawTensor {
        name: info.name.clone(),
        role: classification.0,
        layer: classification.1,
        expert: classification.2,
        dtype: info.dtype.to_string(),
        shape,
        weight_bytes: (info.byte_end - info.byte_start) as u64,
        storage,
    }
}

fn raw_gguf(entry: &crate::patch::gguf::TensorEntry, spec: &arch::ArchSpec) -> RawTensor {
    let codec = Codec::from_type(entry.ggml_type);
    // GGUF stores `ne[0]` contiguous: 2D is [in, out], 3D is [in, out, experts].
    let shape: Vec<usize> = match entry.ne.len() {
        0 | 1 => entry
            .ne
            .iter()
            .map(|&d| usize::try_from(d).unwrap_or(usize::MAX))
            .collect(),
        2 => entry
            .ne
            .iter()
            .rev()
            .map(|&d| usize::try_from(d).unwrap_or(usize::MAX))
            .collect(),
        3 => vec![
            usize::try_from(entry.ne[2]).unwrap_or(usize::MAX),
            usize::try_from(entry.ne[1]).unwrap_or(usize::MAX),
            usize::try_from(entry.ne[0]).unwrap_or(usize::MAX),
        ],
        _ => entry
            .ne
            .iter()
            .map(|&d| usize::try_from(d).unwrap_or(usize::MAX))
            .collect(),
    };
    let storage = match (codec, entry.ne.first().copied()) {
        (Some(codec), Some(in_features)) => Storage::Gguf {
            start: entry.start,
            row_bytes: codec.row_bytes(usize::try_from(in_features).unwrap_or(0)),
            codec,
        },
        _ => Storage::Gguf {
            start: entry.start,
            row_bytes: 0,
            codec: Codec::F32,
        },
    };
    let classification = classify(&entry.name, Format::Gguf, shape.len() == 3, spec);
    RawTensor {
        name: entry.name.clone(),
        role: classification.0,
        layer: classification.1,
        expert: classification.2,
        dtype: entry.ggml_type.name(),
        shape,
        weight_bytes: entry.size.unwrap_or(0),
        storage,
    }
}

/// `(role, layer, expert)` from the architecture spec plus the attention/head names the
/// spec does not model. Fused GGUF expert tensors keep their 3D shape and are split later.
fn classify(
    name: &str,
    format: Format,
    fused: bool,
    spec: &arch::ArchSpec,
) -> (RawRole, Option<usize>, Option<usize>) {
    let layer = layer_of(name);
    if let Some(role) = attention_role(name) {
        return (RawRole::Graph(role), layer, None);
    }
    if is_lm_head(name, format) {
        return (RawRole::Graph(super::GraphRole::LmHead), layer, None);
    }
    if fused {
        if let Some(role) = fused_expert_role(name, format) {
            return (RawRole::Graph(role), layer, None);
        }
    }
    // The spec is the authority for writers and dense MLP readers.
    match spec.role_of(name, format) {
        Some(role) => {
            let graph_role = match role.role {
                Role::Embed => super::GraphRole::Embed,
                Role::AttnOut => super::GraphRole::AttnOut,
                Role::MlpOut => super::GraphRole::MlpOut,
                Role::MlpGate => super::GraphRole::Gate,
                Role::MlpUp => super::GraphRole::Up,
                Role::Router => return (RawRole::Router, role.layer, role.expert),
            };
            (RawRole::Graph(graph_role), role.layer, role.expert)
        }
        None => (RawRole::Ignored, layer, None),
    }
}

fn attention_role(name: &str) -> Option<super::GraphRole> {
    let last = name
        .strip_suffix(".weight")
        .unwrap_or(name)
        .rsplit('.')
        .next()
        .unwrap_or_default();
    match last {
        "q_proj" | "attn_q" | "q" => Some(super::GraphRole::Q),
        "k_proj" | "attn_k" | "k" => Some(super::GraphRole::K),
        "v_proj" | "attn_v" | "v" => Some(super::GraphRole::V),
        _ => None,
    }
}

fn is_lm_head(name: &str, format: Format) -> bool {
    match format {
        Format::Gguf => name == "output.weight",
        Format::SafeTensors => name == "lm_head.weight" || name.ends_with(".lm_head.weight"),
    }
}

fn fused_expert_role(name: &str, format: Format) -> Option<super::GraphRole> {
    match format {
        // `blk.N.ffn_{gate,up,down}_exps.weight`: ne = [in, out, experts].
        Format::Gguf => {
            if name.ends_with(".ffn_gate_exps.weight") {
                Some(super::GraphRole::Gate)
            } else if name.ends_with(".ffn_up_exps.weight") {
                Some(super::GraphRole::Up)
            } else if name.ends_with(".ffn_down_exps.weight") {
                Some(super::GraphRole::MlpOut)
            } else {
                None
            }
        }
        Format::SafeTensors => {
            if name.contains(".experts.gate_up_proj") {
                None
            } else if name.contains(".experts.down_proj") {
                Some(super::GraphRole::MlpOut)
            } else {
                None
            }
        }
    }
}

fn layer_of(name: &str) -> Option<usize> {
    crate::engine::parse_layer_index(name).or_else(|| {
        let mut parts = name.split('.');
        while let Some(part) = parts.next() {
            if part == "blk" {
                return parts.next()?.parse().ok();
            }
        }
        None
    })
}

fn raw_name(name: &str) -> String {
    name.strip_suffix(" (tête liée)").unwrap_or(name).to_owned()
}

fn is_norm_name(name: &str) -> bool {
    name.ends_with("layernorm.weight")
        || name.ends_with("_norm.weight")
        || name.ends_with("ln_1.weight")
        || name.ends_with("ln_2.weight")
}

fn norm_matches(tensor: &str, layer: usize, suffix: &str) -> bool {
    norm_name(tensor, layer, suffix).is_some()
}

fn norm_name(tensor: &str, layer: usize, suffix: &str) -> Option<String> {
    // GGUF uses `blk.N.attn_norm.weight`; SafeTensors keeps the whole prefix.
    for marker in [format!("blk.{layer}."), format!("layers.{layer}.")] {
        if let Some(position) = tensor.find(&marker) {
            let start = position + marker.len();
            return Some(format!("{}{suffix}", &tensor[..start]));
        }
    }
    None
}

fn read_raw_vector(store: &Store, raw: &RawTensor) -> Option<Vec<f32>> {
    match (&store, &raw.storage) {
        (
            Store::Safe(shards),
            Storage::Safe {
                shard,
                start: _,
                dtype,
                ..
            },
        ) => {
            let model = shards.get(*shard)?;
            let view = model.tensor(&raw.name).ok()?;
            let len = raw.shape.first().copied()?;
            let bytes = view.bytes.get(0..len.checked_mul(dtype.bytes())?)?;
            let mut out = vec![0.0; len];
            decode_safe_row(*dtype, bytes, &mut out);
            Some(out)
        }
        (
            Store::Gguf { map, .. },
            Storage::Gguf {
                start,
                codec,
                row_bytes,
                ..
            },
        ) => {
            let len = raw.shape.first().copied()?;
            let bytes = map.get(*start as usize..(*start as usize).checked_add(*row_bytes)?)?;
            let mut out = vec![0.0; len];
            dequantize_row(*codec, bytes, &mut out);
            Some(out)
        }
        _ => None,
    }
}

fn decode_safe_row(dtype: SafeDtype, bytes: &[u8], out: &mut [f32]) {
    match dtype {
        SafeDtype::F32 => {
            for (value, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        SafeDtype::F16 => {
            for (value, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *value = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            }
        }
        SafeDtype::Bf16 => {
            for (value, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
                *value = bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            }
        }
        SafeDtype::F64 => {
            for (value, chunk) in out.iter_mut().zip(bytes.chunks_exact(8)) {
                let bits = u64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]);
                *value = f64::from_bits(bits) as f32;
            }
        }
    }
}

/// Streams one tensor for `rsvd`, applying the Gemma `1 + gamma` factor and replacing
/// non-finite decoded values (already counted by [`pre_scan`]) with zero.
pub struct NodeSource<'a> {
    model: &'a ModelData,
    tensor: &'a TensorRef,
    gamma: Option<&'a [f32]>,
    cancel: &'a std::sync::atomic::AtomicBool,
    /// The process-wide Ctrl-C flag, checked at every block boundary.
    external: Option<&'a std::sync::atomic::AtomicBool>,
}

impl<'a> NodeSource<'a> {
    pub fn new(
        model: &'a ModelData,
        tensor: &'a TensorRef,
        gamma: Option<&'a [f32]>,
        cancel: &'a std::sync::atomic::AtomicBool,
        external: Option<&'a std::sync::atomic::AtomicBool>,
    ) -> Self {
        Self {
            model,
            tensor,
            gamma,
            cancel,
            external,
        }
    }

    fn check_cancel(&self) -> Result<()> {
        use std::sync::atomic::Ordering;
        if self.cancel.load(Ordering::Relaxed)
            || self
                .external
                .is_some_and(|flag| flag.load(Ordering::Relaxed))
        {
            return Err(GraphError::Cancelled);
        }
        Ok(())
    }

    fn count_row(&self, row: &[f32], counters: &mut Counters) {
        for value in row {
            if value.is_nan() {
                counters.nan += 1;
            } else if value.is_infinite() {
                counters.inf += 1;
            }
        }
    }

    pub fn pre_scan(&self) -> Result<Counters> {
        let mut counters = Counters::default();
        let mut row = vec![0.0f32; self.tensor.n()];
        for index in 0..self.tensor.m() {
            self.check_cancel()?;
            self.model.read_row(self.tensor, index, &mut row)?;
            if let Some(gamma) = self.gamma {
                for (value, &g) in row.iter_mut().zip(gamma) {
                    *value *= g;
                }
            }
            self.count_row(&row, &mut counters);
        }
        Ok(counters)
    }

    /// One sanitised row: exactly the values `rsvd` sees.
    pub fn read_sanitized_row(&self, index: usize, out: &mut [f32]) -> Result<()> {
        self.check_cancel()?;
        self.model.read_row(self.tensor, index, out)?;
        if let Some(gamma) = self.gamma {
            for (value, &g) in out.iter_mut().zip(gamma) {
                *value *= g;
            }
        }
        for value in out.iter_mut() {
            if !value.is_finite() {
                *value = 0.0;
            }
        }
        Ok(())
    }
}

impl RowSource for NodeSource<'_> {
    fn rows(&self) -> usize {
        self.tensor.m()
    }

    fn cols(&self) -> usize {
        self.tensor.n()
    }

    fn read_rows(&self, start: usize, out: &mut [f32]) -> crate::surgery::Result<usize> {
        self.check_cancel()
            .map_err(|_| crate::surgery::SurgeryError::Cancelled)?;
        let n = self.tensor.n();
        if n == 0 || !out.len().is_multiple_of(n) {
            return Err(crate::surgery::SurgeryError::Usage(
                "graph : buffer de lignes incompatible".into(),
            ));
        }
        let rows = (out.len() / n).min(self.tensor.m().saturating_sub(start));
        for (index, chunk) in out[..rows * n].chunks_exact_mut(n).enumerate() {
            let row = start + index;
            self.model
                .read_row(self.tensor, row, chunk)
                .map_err(|e| crate::surgery::SurgeryError::Failed(e.to_string()))?;
            if let Some(gamma) = self.gamma {
                for (value, &g) in chunk.iter_mut().zip(gamma) {
                    *value *= g;
                }
            }
            for value in chunk.iter_mut() {
                if !value.is_finite() {
                    *value = 0.0;
                }
            }
        }
        Ok(rows)
    }
}

/// SHA-256 of the complete model: each file name and content, in path order.
pub fn hash_files(paths: &[PathBuf]) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; HASH_BUFFER];
    for path in paths {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        hasher.update(name.as_bytes());
        let mut file = File::open(path).map_err(|e| GraphError::io(path, &e))?;
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|e| GraphError::io(path, &e))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
    }
    Ok(hex(&hasher.finalize()))
}

pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

/// A single `(name, shape, values)` matrix for tests and the micro-benchmark.
pub struct MemorySource {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl RowSource for MemorySource {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn read_rows(&self, start: usize, out: &mut [f32]) -> crate::surgery::Result<usize> {
        if self.cols == 0 || !out.len().is_multiple_of(self.cols) {
            return Err(crate::surgery::SurgeryError::Usage(
                "graph : buffer de lignes incompatible".into(),
            ));
        }
        let rows = (out.len() / self.cols).min(self.rows.saturating_sub(start));
        out[..rows * self.cols]
            .copy_from_slice(&self.data[start * self.cols..(start + rows) * self.cols]);
        Ok(rows)
    }
}

/// A small source used by the estimate micro-benchmark and by unit tests.
pub fn memory_source(rows: usize, cols: usize, seed: u64) -> MemorySource {
    let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    let mut data = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        data.push(((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0);
    }
    MemorySource { data, rows, cols }
}

/// The number of BLAS-free tokens the estimate prints, kept here so tests can reuse it.
pub fn model_bytes(model: &ModelData) -> u64 {
    model.tensors.iter().map(|t| t.weight_bytes).sum::<u64>()
}

/// Convenience for callers that need the number of residual matrices.
pub fn matrix_count(model: &ModelData) -> usize {
    model.tensors.len()
}

/// Shared with `GraphRole::is_writer`; kept here to avoid a circular import in tests.
pub fn is_writer(role: super::GraphRole) -> bool {
    role.is_writer()
}

/// A cheap, deterministic CPU throughput probe: 3 passes of a 128×128 product.
pub fn micro_bench_seconds() -> f64 {
    let n = 128;
    let a = memory_source(n, n, 1);
    let b = memory_source(n, n, 2);
    let started = std::time::Instant::now();
    let mut acc = 0.0f64;
    for _ in 0..3 {
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f64;
                for k in 0..n {
                    sum += f64::from(a.data[i * n + k]) * f64::from(b.data[k * n + j]);
                }
                acc += sum;
            }
        }
    }
    // Keep the result observable without relying on optimisation behaviour.
    std::hint::black_box(acc);
    started.elapsed().as_secs_f64()
}
