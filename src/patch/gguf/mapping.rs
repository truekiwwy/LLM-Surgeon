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

/// A target name -> `(layer, projection)`, no layer for `token_embd.weight`.
pub fn parse_name(name: &str) -> Option<(Option<usize>, Projection)> {
    if name == TOKEN_EMBD {
        return Some((None, Projection::EmbedTokens));
    }
    let rest = name.strip_prefix("blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    let projection = match rest {
        "ffn_gate.weight" => Projection::Gate,
        "ffn_up.weight" => Projection::Up,
        "ffn_down.weight" => Projection::Down,
        "attn_output.weight" => Projection::AttnOut,
        _ => return None,
    };
    Some((Some(layer.parse().ok()?), projection))
}

/// How a target meets the residual stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Orientation {
    /// gate, up: read the residual stream through their columns (`ne0 = hidden`).
    Reader,
    /// down, o_proj: write it through their rows (`ne1 = hidden`); `project_out` couples
    /// the whole matrix.
    Writer,
    /// token_embd: one residual vector per row (`ne0 = hidden`); `project_out` row by row.
    WriterLocal,
}

impl Orientation {
    pub fn of(projection: Projection) -> Self {
        match projection {
            Projection::Gate | Projection::Up => Orientation::Reader,
            Projection::Down | Projection::AttnOut => Orientation::Writer,
            Projection::EmbedTokens => Orientation::WriterLocal,
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

/// Patch targets of the file matching the selection (every projection when
/// `projections` is empty), sorted by layer (the embeddings first) then projection.
pub fn targets<'f>(
    file: &'f GgufFile,
    layers: &LayerSelection,
    projections: &[Projection],
) -> Vec<(Option<usize>, Projection, &'f TensorEntry)> {
    let mut found: Vec<_> = file
        .tensors
        .iter()
        .filter_map(|t| parse_name(&t.name).map(|(l, p)| (l, p, t)))
        .filter(|(l, p, _)| {
            selected(layers, *l) && (projections.is_empty() || projections.contains(p))
        })
        .collect();
    found.sort_by_key(|(l, p, _)| (*l, *p));
    found
}

/// How a target is laid out and decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Layout {
    pub codec: Codec,
    pub rows: usize,
    pub cols: usize,
}

impl Layout {
    pub fn row_bytes(&self) -> usize {
        self.codec.row_bytes(self.cols)
    }
}

/// Checks that a target can be patched: 2D (trailing dimensions of 1 allowed), patchable
/// type, whole blocks per row.
pub fn layout(t: &TensorEntry) -> Result<Layout, MappingError> {
    let codec =
        Codec::from_type(t.ggml_type).ok_or_else(|| MappingError::UnsupportedQuantType {
            tensor: t.name.clone(),
            ggml_type: t.ggml_type.name(),
        })?;
    let not_matrix = || MappingError::NotMatrix {
        tensor: t.name.clone(),
        ne: t.ne.clone(),
    };
    let (cols, rows) = match t.ne.as_slice() {
        [cols, rows, rest @ ..] if rest.iter().all(|&d| d == 1) => (*cols, *rows),
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
    })
}
