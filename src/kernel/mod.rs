//! Low-level layer: SIMD shift kernels, chunked in-place patching of mapped files with OS
//! prefetch hints, and I/O measurements. No dependency on any other project module.
//!
//! The only place, with the engine's mmap, where `unsafe` is allowed: CPU intrinsics,
//! Windows FFI and Unix system calls. Each block is minimal, preceded by the bound checks it
//! relies on and documented by a `// SAFETY:` comment (enforced by an integration test).
//!
//! Call graph, extracted with graphify (AST) and checked by hand. Module edges form a DAG:
//!
//! ```text
//! modules:  dispatch -> simd ; metrics -> dispatch, mmap_io ; mmap_io, simd -> (std, OS)
//!
//! engine::apply_vector_shift  -> ShiftPlan::run -> dispatch::shift_rows
//! WritableModel::shift_tensor -> mmap_io::ChunkedPatcher::run -> ShiftPlan::run
//! dispatch::shift_rows (rayon, blocks of rows) -> shift_row
//!   -> simd::x86::row_avx512 | simd::x86::row_avx2 | simd::neon::row_neon
//!   -> simd::shift_row_scalar -> simd::shift_cells          (reference, and every tail)
//! ChunkedPatcher::run -> PageHints::{sequential, will_need, flush_async}
//!   OsHints: memmap2 advise_range (Unix) | PrefetchVirtualMemory (Windows) | flush_async_range
//! patch::gguf::backend -> ChunkedPatcher::run -> verify::patch_row
//!   -> quant::{dequantize_row, requantize_row}, `w + delta` (the single addition of
//!      simd::shift_cells; scalar, rows in parallel) and simd::{f16,bf16}_{load,store}
//! app::core_apply -> metrics::IoMeter::{start, finish} -> metrics::page_faults
//! ```
//!
//! Level selection: [`Level::best`] at run time (`is_x86_feature_detected!`,
//! `is_aarch64_feature_detected!`), overridden by `SURGEON_KERNEL=scalar|avx2|avx512|neon`
//! or by [`KernelConfig::level`]. An unknown or unavailable level falls back to the best
//! available one with a warning ([`KernelChoice::warning`]); a level that is not available is
//! never executed, so no illegal instruction can be reached.

#![deny(unsafe_op_in_unsafe_fn)]

pub mod dispatch;
pub mod metrics;
pub mod mmap_io;
pub mod simd;
#[cfg(test)]
mod tests;

pub use dispatch::{
    choose, from_env, shift_rows, shift_scalar, KernelChoice, KernelConfig, KernelError, Level,
};
pub use metrics::{page_faults, IoMeter, IoReport, PageFaults};
pub use mmap_io::{
    available_space, drop_file_cache, map_read, map_write, ChunkFailure, ChunkedPatcher, IoConfig,
    IoCounters, OsHints, PageHints,
};
pub use simd::{Deltas, Elem, RowDeltas, ShiftStats};
