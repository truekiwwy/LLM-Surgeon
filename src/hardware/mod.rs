//! Hardware budget: can this machine load a SafeTensors model?
//!
//! For the `main.rs` assembler:
//! - [`can_load_model`]`(path)`: real hardware + default policy; `Err(InsufficientResources)`
//!   when the model does not fit. Use it as a guard before loading.
//! - [`evaluate_budget`]`(path, probe, policy)`: same analysis, never `Err` on a bad verdict.
//!   [`HardwareBudget`] is `Serialize`, print it with `serde_json` for `--json`.
//! - [`SystemProbe::detect`] is the real [`HardwareProbe`]; a missing GPU, NVML or sysfs
//!   only adds entries to `warnings`, it never fails.
//!
//! Call graph, extracted with graphify (AST) then checked by hand. Module edges form a
//! DAG with no dependency on other project modules:
//!
//! ```text
//! modules:  budget -> model_size, probe, error
//!           model_size -> error
//!           probe -> error
//!
//! can_load_model      -> SystemProbe::detect, BudgetPolicy::default, can_load_model_with
//! can_load_model_with -> evaluate_budget  (Insufficient => Err(InsufficientResources))
//! evaluate_budget     -> compute_model_size, evaluate_with_size
//! evaluate_with_size  -> HardwareProbe::{system_ram, gpus, warnings}, scale
//!
//! compute_model_size  -> is_safetensors
//!                     -> shards_from_index (serde_json weight_map, dedup) | list_safetensors
//!                     -> fs::metadata per shard (missing => MissingShard)
//!
//! SystemProbe::detect -> detect_nvidia (not macOS), amd_gpus_in (Linux)
//! SystemProbe::gpus   -> system_ram (Apple Silicon unified pool only)
//! ```

mod budget;
mod error;
mod model_size;
mod probe;

pub use budget::{
    can_load_model, can_load_model_with, evaluate_budget, BudgetPolicy, HardwareBudget, Verdict,
};
pub use error::{HardwareError, Result};
pub use model_size::{compute_model_size, ModelSize};
pub use probe::{GpuInfo, GpuVendor, HardwareProbe, MemoryInfo, SystemProbe};
