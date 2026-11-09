//! Native LoRA merge and extract on SafeTensors (PEFT key layout).
//!
//! ```text
//! cli_args -> surgery::format (FactorDtype)
//! merge    -> keys, config, surgery::{lowrank, target, events, error}
//! extract  -> keys, config, surgery::{rsvd, events, error}
//! ```

pub mod cli_args;
pub mod config;
pub mod extract;
pub mod keys;
pub mod merge;

use cli_args::{LoraArgs, LoraCommand};

use crate::app::{AppDeps, CoreResult, ProgressSink};

/// Shared by the CLI and the `surgeon_lora` MCP tool.
pub fn core_lora(args: &LoraArgs, deps: &AppDeps, sink: &mut dyn ProgressSink) -> CoreResult {
    match &args.command {
        LoraCommand::Merge(a) => merge::core_merge(a, deps, sink),
        LoraCommand::Extract(a) => extract::core_extract(a, deps, sink),
    }
}
