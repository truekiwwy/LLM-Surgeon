//! In-place patching of single-file GGUF models (`surgeon apply --gguf`).
//!
//! Exception to the `patch` rule of no engine dependency, like `autotune`: the vectors are
//! read with `engine::load_direction_{vector,matrix}`, `project_out` reuses the eraser
//! kernels of `engine::project` (`accumulate_r`, `row_deltas`, `local_deltas`), and the
//! bytes go through `kernel` (`ChunkedPatcher`, mappings, the scalar conversions).
//!
//! Call graph, extracted with graphify (AST) and checked by hand. Module edges form a DAG:
//!
//! ```text
//! modules:  backend -> mapping, parse, project, quant, verify, kernel, engine
//!           project -> mapping, quant, verify, engine ; verify -> mapping, quant
//!           mapping -> parse, quant ; quant -> parse, kernel::simd ; parse -> (none)
//!           args -> backend, project, quant, kernel ; report -> backend, mapping, parse, quant, ui
//! (graphify also reports edges to mcp, bench and tui: homonym artefacts, not calls)
//!
//! app::core_apply (--gguf) -> backend::apply_gguf
//!   -> parse::is_split_name, marker_path, kernel::map_read, parse::parse
//!   -> arch::detect_gguf (or --arch-spec) -> arch::Report -> arch::gate (approximate / unsupported)
//!   -> preflight -> mapping::{targets_by, layout_as, tied_embeddings},
//!                   engine::load_direction_{vector,matrix}, throughput (timed sample)
//!   -> check_space (kernel::available_space) -> BackupGuard::create -> write_marker
//!   -> kernel::map_write -> run(apply_patch) -> GgufBackend (PatchBackend):
//!        resolve_targets -> mapping::targets_by (ArchSpec GGUF names)
//!        shift_tensor -> write_tensor (kernel::ChunkedPatcher::run) -> patch_chunk
//!          (rayon, rows) -> verify::patch_row -> quant::{dequantize_row, requantize_row}
//!        project_out_subspace -> project::Projector::plan (read-only: read_r, sweep x N)
//!          -> write_tensor -> Projector::write_rows -> verify::patch_row_with
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
//! Q4_0   2     32     18  d f16 | qs u8[16]
//! Q4_1   3     32     20  d f16 | m f16 | qs u8[16]
//! Q5_0   6     32     22  d f16 | qh u32 | qs u8[16]
//! Q5_1   7     32     24  d f16 | m f16 | qh u32 | qs u8[16]
//! Q8_0   8     32     34  d f16 | qs i8[32]
//! Q4_K  12    256    144  d f16 | dmin f16 | scales u8[12] (8 x 6-bit sc, 8 x 6-bit m) | qs u8[128]
//! Q5_K  13    256    176  d f16 | dmin f16 | scales u8[12] | qh u8[32] | qs u8[128]
//! Q6_K  14    256    210  ql u8[128] | qh u8[64] | scales i8[16] | d f16
//! ```
//!
//! A "Q4_K_M" file mixes types per tensor (`ffn_down` is often Q6_K, and a `hidden` that
//! is not a multiple of 256, 896 for Qwen2.5-0.5B, makes llama.cpp fall back to the
//! 32-element types for the tensors whose rows have that length): the type is checked per
//! target, for every target before the first write; untouched tensors may have any type.
//!
//! Targets: `ffn_{gate,up,down}`, `attn_output` (`o_proj`) and `token_embd`
//! (`embed_tokens`), see [`mapping`]; `output.weight` is never written, and a `token_embd`
//! that is also the output head (no `output.weight`) is refused unless
//! `--allow-tied-embedding`. `project_out` measures what the requantization brings back of
//! the erased direction (`residual_fraction`), see [`project`].
//!
//! Limits: single-file GGUF (no split), little-endian, F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/
//! Q8_0/Q4_K/Q5_K/Q6_K targets (IQ*, Q2_K, Q3_K, MXFP4... are `unsupported_quant_type`),
//! no importance matrix, a shift smaller than half a quantization step is lost (reported by
//! `realized_over_intended` and the warning, never claimed preserved); a GGUF ablation is
//! never complete, only measured. The codecs follow `ggml-quants.c` as read through
//! candle's Rust port: not compared against llama.cpp itself, and `q * d + m` (Q4_1,
//! Q5_1) is two roundings here where a C build that contracts it into an FMA rounds once.
//! Validate a patched file with `surgeon bench ppl` against the original.

pub mod args;
pub mod backend;
pub mod mapping;
pub mod parse;
pub mod project;
pub mod quant;
pub mod report;
pub mod verify;

pub use args::GgufApplyArgs;
pub use backend::{
    apply_gguf, marker_path, GgufApplyError, GgufBackend, GgufOptions, GgufRun, Marker, Preflight,
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
