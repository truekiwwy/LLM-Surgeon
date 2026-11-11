//! `surgeon bench`: measures what a weight surgery changed. Five modes, pure Rust apart
//! from the llama.cpp binaries that `ppl` and `eval` (default backend) drive:
//!
//! - `dump`: full Qwen2 forward (candle, CPU) on prompts, logits written to a SafeTensors.
//! - `logits`: KL(before‖after), cosine, top-1 / top-k shifts between two dumps.
//! - `ppl`: perplexity of a GGUF through `llama-perplexity`, delta against a baseline.
//! - `eval`: exact-match Pass@1 on GSM8K / MATH style datasets through `llama-cli`, or
//!   through the native generator (`--backend native`, Qwen2 SafeTensors, CPU).
//! - `refusal`: refusal rate on prompts (native generator, canonical refusal markers),
//!   paired against a baseline with Newcombe's interval. Opt-in: markers in other
//!   languages, and a llama.cpp judge model that labels every answer.
//!
//! `judge-sample` and `judge-eval` are not modes: they calibrate that judge against human
//! labels, and are CLI only.
//!
//! Exit codes (mapped by `app`): 0 ok (warnings included); 1 insufficient resources,
//! llama.cpp binary not found, or cancelled; 2 invalid arguments or files; 8 bench
//! execution failed (process failed, unreadable output, incompatible shapes, generation
//! impossible); 9 threshold exceeded, only with `--fail-on-warn`; 70 internal.
//!
//! (a) Modules (graphify AST, checked by hand):
//!
//! ```text
//! error <- every module ; mod (Neumaier, sha256, finite) <- every module
//! logits_io -> logits_metrics -> report          (engine::ReadOnlyModel mmap, candle npy/pth)
//! ppl_runner -> ppl_parse -> report              (eval::CommandRunner::run_streaming)
//! dataset -> prompt -> score(generate) -> extract_answer -> score -> report
//! dump -> extract::forward::build -> ResidualForward::forward_logits   (SafetensorsDir)
//! native (eval::TextGenerator over extract::generate::DecoderState) <- score, refusal
//! refusal -> native -> score::wilson -> report
//! markers <- refusal, calibration                (English by default, fr/es/de or a file)
//! judge -> calibration::gate, agreement -> report  (eval::TextGenerator::constrain, GBNF)
//! calibration (sample, eval, gate) -> judge::Judge, agreement::Confusion
//! ```
//!
//! (b) Call graph shared by the CLI and the MCP server:
//!
//! ```text
//! run_with -> dispatch -> app::bench (BenchArgs -> *Request) --+
//! mcp::tools::bench (sandbox, mode/field validation) ----------+-> app::core_bench_{logits,ppl,eval,dump}
//!   logits -> logits_io::open_dump x2 (hardware probe for .npy/.pt) -> check_pair
//!          -> logits_metrics::compare (rayon per block, sequential Neumaier reduction)
//!   ppl    -> eval::ensure_gguf -> eval::locate_binary(llama-perplexity | perplexity)
//!          -> ppl_runner::run_one -> CommandRunner::run_streaming(StreamControl: timeout,
//!             interrupted) -> ppl_parse::{progress, parse_final} -> delta (rounded 1e-4)
//!   eval   -> dataset::load -> prompt::Template -> eval::ensure_gguf -> generator_factory
//!          -> TextGenerator::configure(Decoding) -> score::run_suite (LoopDetector per
//!             question) -> extract_answer -> score::{matches, wilson, paired}
//!   refusal -> judge::prepare (judge file, rubric, calibration::gate: before any generation)
//!          -> native::NativeGenerator::load -> refusal::run_prompts (ChatTemplate,
//!             markers::Markers) -> refusal::{score, paired, newcombe_paired}
//!          -> judge::annotate (generator_factory, verdict cache, parse_verdict)
//!             -> judge::{score, paired, agreement} (Wilson, McNemar exact, Cohen's kappa)
//!   dump   -> extract::{load_config, SafetensorsDir, check_tensors, choose_dtype}
//!          -> tokenize -> forward::build(Depth::Full) -> forward_logits per prompt
//!          -> streamed .tmp -> rename
//! every core -> report::*Report (Serialize + JsonSchema) -> Outcome.data / report::*_text
//! ```
//!
//! (c) Report schemas: `BenchLogitsReport`, `BenchPplReport`, `BenchEvalReport`,
//! `BenchRefusalReport` and `BenchDumpReport` (`report.rs`) are the `data` of the JSON envelope, identical for the
//! CLI `--json` and the MCP `structuredContent.data`; durations live in `timing_ms` only.

#![forbid(unsafe_code)]

pub mod agreement;
pub mod calibration;
pub mod dataset;
pub mod dump;
mod error;
pub mod extract_answer;
pub mod judge;
pub mod logits_io;
pub mod logits_metrics;
pub mod markers;
pub mod native;
pub mod ppl_parse;
pub mod ppl_runner;
pub mod prompt;
pub mod refusal;
pub mod report;
pub mod score;

use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

pub use error::{BenchError, Result};

/// Defaults shared by the CLI and the MCP tool.
pub const DEFAULT_TOP_K: usize = 5;
pub const DEFAULT_WORST: usize = 5;
pub const DEFAULT_CTX: u32 = 512;
pub const DEFAULT_WARN_DELTA: f64 = 0.2;
pub const DEFAULT_LIMIT: usize = 50;
pub const DEFAULT_MAX_NEW_TOKENS: u32 = 512;
pub const DEFAULT_DUMP_MAX_TOKENS: usize = 256;
pub const DEFAULT_MAX_TOTAL_POSITIONS: usize = 4096;
pub const DEFAULT_REFUSAL_MAX_NEW_TOKENS: u32 = 256;
/// Human labels a calibration file must hold before a judge is called calibrated.
pub const DEFAULT_MIN_CALIBRATION_N: usize = 100;
/// Lower end of the "substantial agreement" band of Landis and Koch (1977): a convention,
/// not a guarantee.
pub const DEFAULT_MIN_KAPPA: f64 = 0.6;
pub const DEFAULT_SAMPLE_N: usize = 100;
pub use report::{
    BenchDumpReport, BenchEvalReport, BenchLogitsReport, BenchPplReport, BenchRefusalReport,
    Verdict,
};

/// Compensated (Neumaier) sum: the result depends only on the order of the terms, which
/// every caller keeps fixed (position order), never on the number of threads.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct Neumaier {
    sum: f64,
    compensation: f64,
}

impl Neumaier {
    pub(crate) fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.compensation += (self.sum - t) + x;
        } else {
            self.compensation += (x - t) + self.sum;
        }
        self.sum = t;
    }

    pub(crate) fn total(&self) -> f64 {
        self.sum + self.compensation
    }
}

/// JSON has no NaN nor infinity: those become `null`.
pub(crate) fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path).map_err(|e| BenchError::input(path, e))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| BenchError::input(path, e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neumaier_recovers_what_naive_summation_loses() {
        let mut n = Neumaier::default();
        for x in [1.0, 1e100, 1.0, -1e100] {
            n.add(x);
        }
        assert_eq!(n.total(), 2.0);
        assert_eq!(finite(f64::NAN), None);
        assert_eq!(finite(f64::INFINITY), None);
        assert_eq!(finite(1.5), Some(1.5));
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
