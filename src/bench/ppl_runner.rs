//! `surgeon bench ppl`: perplexity of a GGUF through `llama-perplexity`, never through a
//! shell. Arguments: `-m <model> -f <text> -c <ctx> [--chunks N] [-t N] [-ngl N]`, paths
//! made absolute (so a path starting with `-` can never be read as an option), numbers
//! rendered by Rust. These flags are stable in llama.cpp since 2023 (`--chunks`, `-ngl`);
//! version-specific tuning goes through the CLI-only `--llama-arg` whitelist below.
//!
//! The delta `ppl_patched − ppl_baseline` is rounded to 4 decimals (what llama.cpp
//! prints) before any comparison: `7.4 − 7.2` is `0.20000000000000018` in f64 and must
//! not exceed a 0.2 threshold. CPU runs with the same text, context and threads are
//! reproducible; GPU offload (`-ngl`) can change the last digits between runs.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::error::{BenchError, Result};
use super::ppl_parse::{self, PplEstimate};
use super::report::{BenchPplReport, BenchRun, PplBaselineReport, PplDelta, PplTiming, Verdict};
use super::sha256_file;
use crate::eval::{
    ensure_gguf, last_lines, locate_binary, CommandRunner, EvalError, StreamControl,
    LLAMA_DIR_REQUIRED,
};

const STDERR_TAIL_LINES: usize = 20;

/// `--llama-arg` flags taking a number.
pub const NUMERIC_FLAGS: [&str; 6] = [
    "-b",
    "--batch-size",
    "-ub",
    "--ubatch-size",
    "--ppl-stride",
    "--seed",
];
/// `--llama-arg` flags without a value. Nothing that changes an output or log path.
pub const SWITCH_FLAGS: [&str; 2] = ["--no-mmap", "--mlock"];

#[derive(Debug, Clone, PartialEq)]
pub enum PplBaseline {
    /// Second `llama-perplexity` run on this model, before the patched one.
    Model(PathBuf),
    Value {
        ppl: f64,
        err: Option<f64>,
    },
}

#[derive(Debug, Clone)]
pub struct PplRequest {
    pub model: PathBuf,
    pub text: PathBuf,
    pub baseline: Option<PplBaseline>,
    pub ctx: u32,
    pub chunks: Option<u32>,
    pub threads: Option<u32>,
    pub n_gpu_layers: u32,
    pub llama_cpp_dir: Option<PathBuf>,
    pub warn_delta: f64,
    pub timeout: Option<Duration>,
    /// Validated by [`validate_llama_args`]; always empty from MCP.
    pub llama_args: Vec<String>,
}

/// `FLAG=VALUE` or `FLAG` from the whitelists, re-rendered as separate arguments with the
/// number parsed and printed by Rust.
pub fn validate_llama_args(raw: &[String]) -> Result<Vec<String>> {
    let mut out = Vec::new();
    for arg in raw {
        let (flag, value) = match arg.split_once('=') {
            Some((f, v)) => (f, Some(v)),
            None => (arg.as_str(), None),
        };
        match (
            NUMERIC_FLAGS.contains(&flag),
            SWITCH_FLAGS.contains(&flag),
            value,
        ) {
            (true, _, Some(v)) => {
                let n: u32 = v.trim().parse().map_err(|_| {
                    BenchError::InvalidArgument(format!("--llama-arg {arg} : entier attendu"))
                })?;
                out.extend([flag.to_owned(), n.to_string()]);
            }
            (false, true, None) => out.push(flag.to_owned()),
            _ => {
                return Err(BenchError::InvalidArgument(format!(
                    "--llama-arg {arg} refusé : autorisés {}=N, {}",
                    NUMERIC_FLAGS.join("=N, "),
                    SWITCH_FLAGS.join(", ")
                )))
            }
        }
    }
    Ok(out)
}

fn absolute_utf8(path: &Path) -> Result<String> {
    let abs = std::path::absolute(path).map_err(|e| BenchError::input(path, e))?;
    abs.to_str()
        .map(str::to_owned)
        .ok_or_else(|| BenchError::input(path, "chemin non UTF-8"))
}

/// The exact argument vector of one run.
pub fn build_args(model: &Path, req: &PplRequest) -> Result<Vec<String>> {
    let mut args = vec![
        "-m".to_owned(),
        absolute_utf8(model)?,
        "-f".to_owned(),
        absolute_utf8(&req.text)?,
        "-c".to_owned(),
        req.ctx.to_string(),
    ];
    if let Some(chunks) = req.chunks {
        args.extend(["--chunks".to_owned(), chunks.to_string()]);
    }
    if let Some(threads) = req.threads {
        args.extend(["-t".to_owned(), threads.to_string()]);
    }
    if req.n_gpu_layers > 0 {
        args.extend(["-ngl".to_owned(), req.n_gpu_layers.to_string()]);
    }
    args.extend(req.llama_args.iter().cloned());
    Ok(args)
}

fn validate(req: &PplRequest) -> Result<()> {
    let bad = |m: &str| Err(BenchError::InvalidArgument(m.to_owned()));
    if req.ctx == 0 {
        return bad("--ctx doit être >= 1");
    }
    if req.chunks == Some(0) || req.threads == Some(0) {
        return bad("--chunks et --threads doivent être >= 1");
    }
    if !(req.warn_delta.is_finite() && req.warn_delta >= 0.0) {
        return bad("--warn-delta doit être un nombre positif");
    }
    if let Some(PplBaseline::Value { ppl, err }) = &req.baseline {
        if !(ppl.is_finite() && *ppl > 0.0) {
            return bad("--baseline-ppl doit être un nombre > 0");
        }
        if err.is_some_and(|e| !(e.is_finite() && e >= 0.0)) {
            return bad("--baseline-ppl-err doit être un nombre positif");
        }
    }
    for model in std::iter::once(&req.model).chain(match &req.baseline {
        Some(PplBaseline::Model(m)) => Some(m),
        _ => None,
    }) {
        if !model.exists() {
            return Err(BenchError::input(model, "modèle introuvable"));
        }
    }
    if !req.text.is_file() {
        return Err(BenchError::input(&req.text, "fichier texte introuvable"));
    }
    Ok(())
}

/// Rounded to the 4 decimals llama.cpp prints.
pub fn round4(x: f64) -> f64 {
    (x * 1e4).round() / 1e4
}

pub fn delta(patched: PplEstimate, baseline: PplEstimate, warn_delta: f64) -> PplDelta {
    let delta = round4(patched.ppl - baseline.ppl);
    PplDelta {
        delta,
        delta_relative_pct: round4(delta / baseline.ppl * 100.0),
        within_noise: match (patched.err, baseline.err) {
            (Some(a), Some(b)) => Some(delta.abs() <= (a * a + b * b).sqrt()),
            _ => None,
        },
        improvement: delta < 0.0,
        warn_delta,
        exceeded: delta > warn_delta,
    }
}

/// One run: progress through `on_chunk(done, total)`; exit code, cancellation and
/// parsing checked in that order.
pub fn run_one(
    runner: &dyn CommandRunner,
    binary: &Path,
    args: &[String],
    control: StreamControl<'_>,
    on_chunk: &mut dyn FnMut(usize, Option<usize>),
) -> Result<PplEstimate> {
    let (mut total, mut done) = (None, 0);
    let output = runner.run_streaming(binary, args, None, control, &mut |line| {
        if let Some(t) = ppl_parse::total_chunks(line) {
            total = Some(t);
        }
        if let Some(d) = ppl_parse::progress(line).filter(|d| *d > done) {
            done = d;
            on_chunk(d, total);
        }
        ControlFlow::Continue(())
    })?;
    if control.cancel.is_some_and(|c| c.load(Ordering::SeqCst)) {
        return Err(BenchError::Cancelled);
    }
    let tail = last_lines(&output.stderr, STDERR_TAIL_LINES);
    if output.status_code != 0 {
        return Err(BenchError::ProcessFailed {
            step: "llama-perplexity".into(),
            code: output.status_code,
            stderr_tail: tail,
        });
    }
    ppl_parse::parse_final(&format!("{}\n{}", output.stdout, output.stderr)).map_err(|reason| {
        BenchError::PplParse {
            reason,
            stderr_tail: tail,
        }
    })
}

fn locate_perplexity(dir: Option<&Path>) -> Result<PathBuf> {
    // Renamed from `perplexity` to `llama-perplexity` in mid-2024.
    locate_binary("llama-perplexity", dir)
        .or_else(|_| locate_binary("perplexity", dir))
        .map_err(BenchError::from)
}

fn gguf_error(e: EvalError) -> BenchError {
    match e {
        EvalError::InvalidConfig(m) if m == LLAMA_DIR_REQUIRED => BenchError::InvalidArgument(m),
        other => BenchError::Conversion(other.to_string()),
    }
}

/// Baseline (if a model) then patched model; a SafeTensors model is converted to a
/// temporary GGUF first and that file is removed on every path.
pub fn run(
    req: &PplRequest,
    runner: &dyn CommandRunner,
    cancel: &AtomicBool,
    on_chunk: &mut dyn FnMut(usize, Option<usize>),
) -> Result<BenchRun<BenchPplReport>> {
    validate(req)?;
    let binary = locate_perplexity(req.llama_cpp_dir.as_deref())?;
    let text_sha256 = sha256_file(&req.text)?;
    let control = StreamControl {
        timeout: req.timeout,
        cancel: Some(cancel),
    };
    let mut measure = |model: &Path, tag: &str| -> Result<(PplEstimate, Vec<String>, u64)> {
        let started = Instant::now();
        let gguf =
            ensure_gguf(model, req.llama_cpp_dir.as_deref(), tag, runner).map_err(gguf_error)?;
        let args = build_args(gguf.path(), req)?;
        let estimate = run_one(runner, &binary, &args, control, on_chunk)?;
        Ok((estimate, args, started.elapsed().as_millis() as u64))
    };
    let (baseline, baseline_ms) = match &req.baseline {
        None => (None, None),
        Some(PplBaseline::Value { ppl, err }) => (
            Some((
                PplEstimate {
                    ppl: *ppl,
                    err: *err,
                },
                None,
            )),
            None,
        ),
        Some(PplBaseline::Model(model)) => {
            let (estimate, _, ms) = measure(model, "bench-ppl-base")?;
            (Some((estimate, Some(model.clone()))), Some(ms))
        }
    };
    let (patched, args, patched_ms) = measure(&req.model, "bench-ppl")?;
    let delta = baseline
        .as_ref()
        .map(|(b, _)| delta(patched, *b, req.warn_delta));
    let mut warnings = Vec::new();
    if req.n_gpu_layers > 0 {
        warnings.push(
            "-ngl > 0 : le calcul GPU peut faire varier les derniers chiffres d'un run à l'autre"
                .into(),
        );
    }
    let threshold_warnings: Vec<String> = delta
        .iter()
        .filter(|d| d.exceeded)
        .map(|d| format!("ΔPPL {:.4} > {}", d.delta, d.warn_delta))
        .collect();
    warnings.extend(threshold_warnings.iter().cloned());
    let report = BenchPplReport {
        model: req.model.clone(),
        text: req.text.clone(),
        text_sha256,
        ctx: req.ctx,
        chunks: req.chunks,
        threads: req.threads,
        n_gpu_layers: req.n_gpu_layers,
        binary,
        args,
        ppl: patched.ppl,
        ppl_err: patched.err,
        baseline: baseline.map(|(b, model)| PplBaselineReport {
            source: match model {
                Some(_) => "model".into(),
                None => "value".into(),
            },
            model,
            ppl: b.ppl,
            ppl_err: b.err,
        }),
        verdict: match &delta {
            None => Verdict::None,
            Some(d) if d.exceeded => Verdict::Warn,
            Some(_) => Verdict::Ok,
        },
        delta,
        timing_ms: PplTiming {
            patched: patched_ms,
            baseline: baseline_ms,
        },
    };
    Ok(BenchRun {
        report,
        warnings,
        threshold_warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn est(ppl: f64, err: Option<f64>) -> PplEstimate {
        PplEstimate { ppl, err }
    }

    #[test]
    fn delta_is_rounded_before_the_strict_comparison() {
        assert_eq!(7.4 - 7.2, 0.20000000000000018, "the trap");
        let d = delta(est(7.4, None), est(7.2, None), 0.2);
        assert_eq!(d.delta, 0.2);
        assert!(!d.exceeded, "equal is not above");
        assert!(delta(est(7.4001, None), est(7.2, None), 0.2).exceeded);
        let better = delta(est(7.0, None), est(7.2, None), 0.2);
        assert!(better.improvement && !better.exceeded);
        assert_eq!(better.delta, -0.2);
        let noisy = delta(est(7.25, Some(0.04)), est(7.2, Some(0.04)), 0.2);
        assert_eq!(noisy.within_noise, Some(true), "0.05 <= 0.0566");
        let clear = delta(est(7.3, Some(0.04)), est(7.2, Some(0.04)), 0.2);
        assert_eq!(clear.within_noise, Some(false));
        assert_eq!(
            delta(est(7.3, None), est(7.2, Some(0.04)), 0.2).within_noise,
            None
        );
        assert_eq!(clear.delta_relative_pct, 1.3889);
    }

    #[test]
    fn llama_args_whitelist() {
        let ok = validate_llama_args(&["--batch-size=512".into(), "--no-mmap".into()]).unwrap();
        assert_eq!(ok, ["--batch-size", "512", "--no-mmap"]);
        for bad in [
            "--log-file=x.txt",
            "-o=out",
            "--batch-size=12abc",
            "--batch-size",
            "--no-mmap=1",
            "--kl-divergence-base=f",
            "-m=other.gguf",
        ] {
            assert!(validate_llama_args(&[bad.into()]).is_err(), "{bad}");
        }
    }

    #[test]
    fn arguments_are_structured_and_absolute() {
        let req = PplRequest {
            model: "-evil.gguf".into(),
            text: "-text.txt".into(),
            baseline: None,
            ctx: 256,
            chunks: Some(4),
            threads: Some(2),
            n_gpu_layers: 0,
            llama_cpp_dir: None,
            warn_delta: 0.2,
            timeout: None,
            llama_args: vec!["--seed".into(), "7".into()],
        };
        let args = build_args(Path::new("-evil.gguf"), &req).unwrap();
        assert_eq!(args[0], "-m");
        assert!(Path::new(&args[1]).is_absolute() && !args[1].starts_with('-'));
        assert!(Path::new(&args[3]).is_absolute() && !args[3].starts_with('-'));
        assert_eq!(
            &args[4..],
            ["-c", "256", "--chunks", "4", "-t", "2", "--seed", "7"]
        );
        let gpu = PplRequest {
            n_gpu_layers: 99,
            chunks: None,
            threads: None,
            llama_args: vec![],
            ..req
        };
        assert_eq!(
            &build_args(Path::new("m.gguf"), &gpu).unwrap()[4..],
            ["-c", "256", "-ngl", "99"]
        );
    }
}
