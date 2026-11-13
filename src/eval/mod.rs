//! Fast non-regression check of a patched model, then GGUF export if it is healthy.
//!
//! No dependency on the other modules of the crate: generation and process execution go
//! through the `TextGenerator` and `CommandRunner` traits, mocked in tests.
//!
//! Graph (graphify):
//!
//! ```text
//! modules:  error <- every module
//!           loop_detector -> mod (truncate_chars)
//!           command -> error
//!           generator -> command (drain_tail, last_lines, spawn_error), error
//!           prompts -> error
//!           runner -> generator (TextGenerator), loop_detector, prompts, error
//!           gguf -> command (CommandRunner), generator (find_binary), runner (EvalReport),
//!                   safetensors_gguf (pure Rust conversion: Qwen2, Llama/Mistral,
//!                   Gemma/Gemma 2, Phi-3; no python)
//! run_eval -> LoopDetector::reset -> TextGenerator::generate -> on_chunk
//!          -> LoopDetector::push -> step -> check_marker | check_period | feed_word | feed_line
//!          -> ControlFlow::Break on Loop -> check_expectations -> verdict
//! LlamaCliGenerator::generate -> command (args) -> stream_child -> Command::spawn (no shell)
//!          -> reader thread -> sync_channel -> Utf8Decoder::push -> on_chunk
//!          ; Break | timeout | read error -> ChildGuard (kill + wait) ; exit != 0 -> CommandFailed
//! SystemCommandRunner::run_streaming -> spawn (no shell) -> 2 reader threads (LineSplitter)
//!          -> sync_channel -> on_line ; Break | cancel flag | timeout -> ChildGuard kill + wait
//! ensure_gguf -> .gguf as is | export_gguf (f16, no quant) -> GgufModel (deleted on drop)
//! export_if_healthy -> export_gguf (Healthy, or Degraded + allow_degraded)
//!          -> validate (quant whitelist, outtype, dirs, converter, find_binary(quantize))
//!          -> CommandRunner::run(python, convert) -> CommandRunner::run(llama-quantize)
//!             | fs::rename -> remove intermediate -> verify_gguf (non empty, "GGUF")
//! ```
//!
//! Wiring for the assembler. `llama-cli` reads GGUF only, so the patched model must be
//! converted once before it can be evaluated:
//!
//! ```text
//! let prompts = load_prompts(&args.prompts)?;
//! let f16 = export_gguf(&GgufExportConfig { quant: None, output_gguf: eval.f16.gguf, .. }, &SystemCommandRunner)?;
//! let mut gen = LlamaCliGenerator { llama_cli_path: locate_binary("llama-cli", dir)?, model_path: f16.final_gguf, .. };
//! let report = run_eval(&mut gen, &prompts, &EvalConfig::default())?;
//! export_if_healthy(&report, &GgufExportConfig { quant: Some(args.quant), .. }, &SystemCommandRunner, args.allow_degraded)?;
//! ```

mod command;
mod error;
mod generator;
mod gguf;
mod http;
mod loop_detector;
mod prompts;
mod runner;
mod safetensors_gguf;
pub mod sampling;
pub mod server;

pub(crate) use command::last_lines;
pub use command::{CommandOutput, CommandRunner, StreamControl, SystemCommandRunner};
pub use error::{EvalError, Result};
pub(crate) use generator::find_binary;
pub use generator::{
    generate_one_by_one, locate_binary, sampling_flags, Decoding, GenJob, GenerationOutcome,
    LlamaCliGenerator, TextGenerator,
};
pub use gguf::{
    ensure_gguf, ensure_gguf_as, export_gguf, export_if_healthy, is_gguf, model_dir,
    GgufExportConfig, GgufExportReport, GgufModel, StepReport, ALLOWED_QUANTS, LLAMA_DIR_REQUIRED,
};
pub use loop_detector::{LoopDetector, LoopDetectorConfig, LoopReason, LoopVerdict};
pub use prompts::{load_prompts, EvalPrompt};
pub use runner::{run_eval, EvalConfig, EvalReport, EvalVerdict, PromptResult, PromptStatus};
pub use safetensors_gguf::{convert_safetensors, WeightType};

/// Longest prefix of at most `max_chars` characters; never splits a multi-byte character.
pub(crate) fn truncate_chars(s: &str, max_chars: usize) -> &str {
    s.char_indices().nth(max_chars).map_or(s, |(i, _)| &s[..i])
}
