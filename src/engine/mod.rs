//! SafeTensors engine: zero-copy inspection, MLP tensor selection and in-place vector
//! shift `W ∓= alpha * v` on bf16 / f16 / f32 weights.
//!
//! For the `main.rs` / patch assembler:
//! - [`read_index`] or [`ReadOnlyModel`] to inspect, [`find_mlp_tensors`] to pick targets,
//! - [`load_direction_vector`] for v, then [`WritableModel::shift_tensor`] (or the
//!   [`TensorShifter`] hook) and [`WritableModel::flush`].
//! - A rejected shift (bad dtype, shape, vector length, non-finite alpha or v) returns
//!   `Err` before any write: a tensor is never left half-modified.
//!
//! Tensor index, as the selector sees it (one file; a sharded model is one file per shard):
//!
//! ```text
//! name "model.layers.{i}.mlp.{proj}.weight"
//!   -> layer i                       (parse_layer_index, LayerSpec)
//!     -> projection Gate | Up | Down (MlpProj, sorted in that order)
//!       -> TensorInfo { dtype, shape [out, in], byte_start..byte_end (absolute) }
//! byte_start = 8 + header_len + data_offsets.0
//! ```
//!
//! Call graph, extracted with graphify (AST) and checked by hand. Module edges form a DAG
//! with no dependency on other project modules:
//!
//! ```text
//! modules:  mmap_file -> index, shift, error ; selector -> index ; shift -> index, error
//!           index -> error ; mod (TensorShifter) -> mmap_file, shift
//!
//! read_index              -> ReadOnlyModel::open -> open_checked, mmap, parse_index
//! parse_index             -> safetensors::read_metadata, map_err
//! WritableModel::open     -> open_checked, mmap_mut, parse_index
//! TensorShifter::shift    -> WritableModel::shift_tensor -> ModelIndex::get, apply_vector_shift
//! apply_vector_shift      -> (validation) -> shift_rows (rayon) -> shift_cells
//! load_direction_vector   -> ReadOnlyModel::open | serde_json, decode_f32
//! find_mlp_tensors        -> parse_mlp, LayerSpec::contains
//! ```
//!
//! Axis per projection, for Llama-style weights stored `[out_features, in_features]`
//! (`y = W x`) and a direction v living in the residual stream (`hidden_size`):
//! - `gate_proj`, `up_proj`: `[intermediate, hidden]`, v indexes the input -> `Rows`.
//! - `down_proj`: `[hidden, intermediate]`, v indexes the output -> `Cols`.
//!
//! The length check rejects a wrong axis whenever `intermediate != hidden`.

mod error;
mod index;
mod mmap_file;
mod selector;
mod shift;
pub mod stats;
#[cfg(test)]
mod test_support;

pub use error::{EngineError, Result};
pub use index::{DType, ModelIndex, TensorInfo};
pub use mmap_file::{load_direction_vector, read_index, ReadOnlyModel, TensorView, WritableModel};
pub use selector::{find_mlp_tensors, parse_layer_index, LayerSpec, MlpProj};
pub use shift::{apply_vector_shift, BroadcastAxis, ShiftOp, ShiftStats};

/// Hook for the patch module: anything that can shift a named tensor in place.
pub trait TensorShifter {
    fn shift(
        &mut self,
        tensor_name: &str,
        direction: &[f32],
        alpha: f32,
        op: ShiftOp,
        axis: BroadcastAxis,
    ) -> Result<ShiftStats>;
}

impl TensorShifter for WritableModel {
    fn shift(
        &mut self,
        tensor_name: &str,
        direction: &[f32],
        alpha: f32,
        op: ShiftOp,
        axis: BroadcastAxis,
    ) -> Result<ShiftStats> {
        self.shift_tensor(tensor_name, direction, alpha, op, axis)
    }
}
