//! Patch targets in a GGUF file and their orientation:
//!
//! ```text
//! gate_proj / up_proj / down_proj  blk.{i}.ffn_{gate,up,down}.weight
//! o_proj                           blk.{i}.attn_output.weight
//! embed_tokens                     token_embd.weight          (no layer: any selection)
//! ```
//!
//! Layout: `ne[0]` is the contiguous dimension (columns, `in_features`), `ne[1]` the rows
//! (`out_features`), so a GGUF `[ne0, ne1]` tensor is the SafeTensors `[out, in] =
//! [ne1, ne0]` matrix, row-major, and the patch axes keep their meaning: `rows` takes a v
//! of length `ne0` (gate, up, embeddings), `cols` a v of length `ne1` (down, o_proj).
//! Quantization blocks run along `ne[0]`, inside a row.
//!
//! `project_out` follows the SafeTensors engine: `ffn_down` (`ne = [inter, hidden]`) and
//! `attn_output` (`ne = [heads * head_dim, hidden]`) write the residual stream through
//! their rows, so the directions have length `ne1` and `W' = W − αA(BᵀW)` couples every
//! row (two passes); `token_embd` (`ne = [hidden, vocab]`) holds one residual vector per
//! row, so the directions have length `ne0` and each row is projected locally. llama.cpp
//! permutes `attn_q` / `attn_k` for RoPE, never `attn_output`.
//!
//! `output.weight` (the output head, a reader) is never a target. When it is absent the
//! head reuses `token_embd` (tied embeddings): see [`tied_embeddings`].

use serde::Serialize;

use super::parse::{GgufFile, TensorEntry};
use super::quant::Codec;
use crate::arch::{ArchSpec, Format, Layout as ArchLayout, Role, TensorRole};
use crate::patch::{LayerSelection, Projection};

/// Name of the GGUF input embeddings.
pub const TOKEN_EMBD: &str = "token_embd.weight";
/// Name of the GGUF output head; absent when it is tied to the input embeddings.
pub const OUTPUT: &str = "output.weight";

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MappingError {
    /// The machine-readable kind is `unsupported_quant_type`.
    #[error("unsupported_quant_type : {tensor} est en {ggml_type} (patchables : F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K)")]
    UnsupportedQuantType { tensor: String, ggml_type: String },
    #[error("{tensor} : tenseur 2D attendu, dimensions {ne:?}")]
    NotMatrix { tensor: String, ne: Vec<u64> },
    #[error("{tensor} : {cols} colonnes, pas un multiple du bloc de {block} éléments")]
    PartialBlock {
        tensor: String,
        cols: u64,
        block: usize,
    },
}

/// The GGUF name of a patch target (`layer` is ignored for `embed_tokens`).
pub fn tensor_name(layer: usize, projection: Projection) -> String {
    let proj = match projection {
        Projection::Gate => "gate",
        Projection::Up => "up",
        Projection::Down => "down",
        Projection::AttnOut => return format!("blk.{layer}.attn_output.weight"),
        Projection::EmbedTokens => return TOKEN_EMBD.to_owned(),
    };
    format!("blk.{layer}.ffn_{proj}.weight")
}

/// How a target meets the residual stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Orientation {
    /// gate, up: read the residual stream through their columns (`ne0 = hidden`).
    Reader,
    /// down, o_proj, experts: write it through their rows (`ne1 = hidden`); `project_out`
    /// couples the whole matrix (each expert's, for fused experts).
    Writer,
    /// token_embd: one residual vector per row (`ne0 = hidden`); `project_out` row by row.
    WriterLocal,
}

impl Orientation {
    pub fn of(role: Role) -> Self {
        match role {
            Role::Embed => Orientation::WriterLocal,
            Role::AttnOut | Role::MlpOut => Orientation::Writer,
            Role::MlpGate | Role::MlpUp | Role::Router => Orientation::Reader,
        }
    }
}

/// `true` when the file has input embeddings but no `output.weight`: llama.cpp then uses
/// `token_embd` as the output head too, and patching it changes both.
pub fn tied_embeddings(file: &GgufFile) -> bool {
    file.tensor(TOKEN_EMBD).is_some() && file.tensor(OUTPUT).is_none()
}

fn selected(layers: &LayerSelection, layer: Option<usize>) -> bool {
    let Some(layer) = layer else {
        // The embeddings sit outside the layers: selected whatever `layers` says, as in
        // the SafeTensors engine.
        return true;
    };
    match layers {
        LayerSelection::All => true,
        LayerSelection::List(list) => list.contains(&layer),
        LayerSelection::Range {
            start,
            end_inclusive,
        } => (*start..=*end_inclusive).contains(&layer),
    }
}

/// The patch projection of a role (`None`: the router, never a target).
fn projection_of(role: Role) -> Option<Projection> {
    [
        Projection::Gate,
        Projection::Up,
        Projection::Down,
        Projection::AttnOut,
        Projection::EmbedTokens,
    ]
    .into_iter()
    .find(|p| p.role() == role)
}

/// Patch targets of the file by the roles of `spec`, matching the selection (every
/// projection when `projections` is empty), sorted by layer (the embeddings first),
/// projection, expert, name.
pub fn targets_by<'f>(
    spec: &ArchSpec,
    file: &'f GgufFile,
    layers: &LayerSelection,
    projections: &[Projection],
) -> Vec<(TensorRole, Projection, &'f TensorEntry)> {
    let mut found: Vec<_> = file
        .tensors
        .iter()
        .filter_map(|t| {
            let role = spec.role_of(&t.name, Format::Gguf)?;
            Some((role, projection_of(role.role)?, t))
        })
        .filter(|(r, p, _)| {
            selected(layers, r.layer) && (projections.is_empty() || projections.contains(p))
        })
        .collect();
    found.sort_by(|(ra, pa, a), (rb, pb, b)| {
        (ra.layer, *pa, ra.expert, &a.name).cmp(&(rb.layer, *pb, rb.expert, &b.name))
    });
    found
}

/// [`targets_by`] with the default naming (`arch::legacy`), as before architectures were
/// described: `(layer, projection, tensor)`.
pub fn targets<'f>(
    file: &'f GgufFile,
    layers: &LayerSelection,
    projections: &[Projection],
) -> Vec<(Option<usize>, Projection, &'f TensorEntry)> {
    targets_by(crate::arch::legacy(), file, layers, projections)
        .into_iter()
        .map(|(r, p, t)| (r.layer, p, t))
        .collect()
}

/// How a target is laid out and decoded: `experts` stacked `[rows, cols]` matrices (1 for a
/// plain matrix; fused `ffn_down_exps` `ne = [inter, hidden, n_expert]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Layout {
    pub codec: Codec,
    pub rows: usize,
    pub cols: usize,
    #[serde(skip)]
    pub experts: usize,
}

impl Layout {
    pub fn row_bytes(&self) -> usize {
        self.codec.row_bytes(self.cols)
    }

    /// Bytes of one expert's matrix (the whole tensor when `experts == 1`).
    pub fn matrix_bytes(&self) -> usize {
        self.rows * self.row_bytes()
    }

    /// The `[rows, cols]` matrix of one expert.
    pub fn matrix(&self) -> Layout {
        Layout {
            experts: 1,
            ..*self
        }
    }
}

/// Checks that a target can be patched: 2D (trailing dimensions of 1 allowed), patchable
/// type, whole blocks per row.
pub fn layout(t: &TensorEntry) -> Result<Layout, MappingError> {
    layout_as(t, ArchLayout::OutIn)
}

/// [`layout`] for a role layout: `experts_out_in` also takes a 3D `[cols, rows, experts]`.
pub fn layout_as(t: &TensorEntry, role_layout: ArchLayout) -> Result<Layout, MappingError> {
    let codec =
        Codec::from_type(t.ggml_type).ok_or_else(|| MappingError::UnsupportedQuantType {
            tensor: t.name.clone(),
            ggml_type: t.ggml_type.name(),
        })?;
    let not_matrix = || MappingError::NotMatrix {
        tensor: t.name.clone(),
        ne: t.ne.clone(),
    };
    let (cols, rows, experts) = match (t.ne.as_slice(), role_layout) {
        ([cols, rows, experts], ArchLayout::ExpertsOutIn) if *experts > 0 => {
            (*cols, *rows, *experts)
        }
        ([cols, rows, rest @ ..], _) if rest.iter().all(|&d| d == 1) => (*cols, *rows, 1),
        _ => return Err(not_matrix()),
    };
    let block = codec.block_elems();
    if !cols.is_multiple_of(block as u64) {
        return Err(MappingError::PartialBlock {
            tensor: t.name.clone(),
            cols,
            block,
        });
    }
    Ok(Layout {
        codec,
        rows: usize::try_from(rows).map_err(|_| not_matrix())?,
        cols: usize::try_from(cols).map_err(|_| not_matrix())?,
        experts: usize::try_from(experts).map_err(|_| not_matrix())?,
    })
}
