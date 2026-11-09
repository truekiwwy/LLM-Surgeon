//! In-place patching of single-file GGUF models (`surgeon apply --gguf`).
//!
//! Exception to the `patch` rule of no engine dependency, like `autotune`: the vectors are
//! read with `engine::load_direction_vector`, and the bytes go through `kernel`
//! (`ChunkedPatcher`, mappings, the scalar conversions of the SafeTensors kernels).
//!
//! Call graph, extracted with graphify (AST) and checked by hand. Module edges form a DAG:
//!
//! ```text
//! modules:  backend -> mapping, parse, quant, verify, kernel ; verify -> mapping, quant
//!           mapping -> parse, quant ; quant -> parse, kernel::simd ; parse -> (none)
//!           args -> backend, quant, kernel ; report -> backend, mapping, parse, quant, ui
//! (graphify also reports edges to mcp, bench and tui: homonym artefacts, not calls)
//!
//! app::core_apply (--gguf) -> backend::apply_gguf
//!   -> parse::is_split_name, marker_path, kernel::map_read, parse::parse
//!   -> preflight -> mapping::{mlp_tensors, layout}, engine::load_direction_vector
//!   -> check_space (kernel::available_space) -> BackupGuard::create -> write_marker
//!   -> kernel::map_write -> run(apply_patch) -> GgufBackend (PatchBackend):
//!        resolve_targets -> mapping::mlp_tensors
//!        shift_tensor -> kernel::ChunkedPatcher::run -> patch_chunk (rayon, rows)
//!          -> verify::patch_row -> quant::{dequantize_row, requantize_row, scales_finite}
//!        flush -> MmapMut::flush, verify::verify_file (header SHA-256, tensor SHA-256,
//!                 finite scales and values, on a second mapping)
//!   -> RollbackHandle::{commit | rollback} (in apply_patch) -> remove the marker
//! ```
//!
//! Quantization blocks (bytes, elements, fields), see [`quant`] for the bit layout:
//!
//! ```text
//! type  id  elems  bytes  fields
//! F32    0      1      4  f32
//! F16    1      1      2  f16
//! BF16  30      1      2  bf16
//! Q8_0   8     32     34  d f16 | qs i8[32]
//! Q4_K  12    256    144  d f16 | dmin f16 | scales u8[12] (8 x 6-bit sc, 8 x 6-bit m) | qs u8[128]
//! Q6_K  14    256    210  ql u8[128] | qh u8[64] | scales i8[16] | d f16
//! ```
//!
//! A "Q4_K_M" file mixes types per tensor (`ffn_down` is often Q6_K): the type is checked
//! per target, untouched tensors may have any type.
//!
//! Limits: single-file GGUF (no split), little-endian, F32/F16/BF16/Q8_0/Q4_K/Q6_K targets,
//! no importance matrix, a shift smaller than half a quantization step is lost (reported by
//! `realized_over_intended` and the warning, never claimed preserved). Validate a patched
//! file with `surgeon bench ppl` against the original.

pub mod args;
pub mod backend;
pub mod mapping;
pub mod parse;
pub mod quant;
pub mod report;
pub mod verify;

pub use args::GgufApplyArgs;
pub use backend::{
    apply_gguf, marker_path, GgufApplyError, GgufBackend, GgufOptions, GgufRun, Marker,
};
pub use parse::{parse, GgmlType, GgufError, GgufFile, TensorEntry};
pub use quant::{Codec, Requant};
pub use verify::TensorFidelity;

/// `true` when the file starts with the GGUF magic.
pub fn is_gguf(path: &std::path::Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    std::fs::File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok_and(|()| magic == parse::MAGIC)
}
