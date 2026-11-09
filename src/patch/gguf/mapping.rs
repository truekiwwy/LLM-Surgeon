//! Patch targets in a GGUF file: `{gate,up,down}_proj` of layer `i` is
//! `blk.{i}.ffn_{gate,up,down}.weight`.
//!
//! Layout: `ne[0]` is the contiguous dimension (columns, `in_features`), `ne[1]` the rows
//! (`out_features`), so a GGUF `[ne0, ne1]` tensor is the SafeTensors `[out, in] =
//! [ne1, ne0]` matrix, row-major, and the patch axes keep their meaning: `rows` takes a v
//! of length `ne0` (gate, up), `cols` a v of length `ne1` (down). Quantization blocks run
//! along `ne[0]`, inside a row.

use serde::Serialize;

use super::parse::{GgufFile, TensorEntry};
use super::quant::Codec;
use crate::patch::{LayerSelection, Projection};

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum MappingError {
    /// The machine-readable kind is `unsupported_quant_type`.
    #[error("unsupported_quant_type : {tensor} est en {ggml_type} (patchables : F32, F16, BF16, Q8_0, Q4_K, Q6_K)")]
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

pub fn tensor_name(layer: usize, projection: Projection) -> String {
    let proj = match projection {
        Projection::Gate => "gate",
        Projection::Up => "up",
        Projection::Down => "down",
    };
    format!("blk.{layer}.ffn_{proj}.weight")
}

/// `blk.{i}.ffn_{gate,up,down}.weight` -> `(i, projection)`.
pub fn parse_name(name: &str) -> Option<(usize, Projection)> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, rest) = rest.split_once('.')?;
    let projection = match rest {
        "ffn_gate.weight" => Projection::Gate,
        "ffn_up.weight" => Projection::Up,
        "ffn_down.weight" => Projection::Down,
        _ => return None,
    };
    Some((layer.parse().ok()?, projection))
}

fn selected(layers: &LayerSelection, layer: usize) -> bool {
    match layers {
        LayerSelection::All => true,
        LayerSelection::List(list) => list.contains(&layer),
        LayerSelection::Range {
            start,
            end_inclusive,
        } => (*start..=*end_inclusive).contains(&layer),
    }
}

/// MLP tensors of the file matching the selection, sorted by layer then gate, up, down.
pub fn mlp_tensors<'f>(
    file: &'f GgufFile,
    layers: &LayerSelection,
    projections: &[Projection],
) -> Vec<(usize, Projection, &'f TensorEntry)> {
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
