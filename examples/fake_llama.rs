//! Stand-in for the llama.cpp binaries, to exercise `surgeon bench ppl` and `bench eval`
//! end to end (real processes, streaming, timeout, kill) without llama.cpp. Its role comes
//! from its file name: copy it as `llama-perplexity` and as `llama-completion`.
//!
//! - perplexity: prints the chunk progress on stderr like llama.cpp (`[1]6.1234,` without
//!   newline), then `Final estimate: PPL = X +/- 0.0400`, X derived from the model bytes.
//! - completion: answers `#### <sum of the integers of the problem>` on stdout; a model
//!   whose bytes contain `broken` answers one more.
//!
//! `FAKE_LLAMA_SLEEP_MS` slows every chunk down (timeout tests), `FAKE_LLAMA_FAIL=1`
//! exits with code 1 after a few log lines.
//!
//! ```text
//! cargo build --release --example fake_llama
//! cp target/release/examples/fake_llama llama/llama-perplexity
//! cp target/release/examples/fake_llama llama/llama-completion
//! ```

use std::io::Write;
use std::time::Duration;

fn arg_after(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let name = std::path::Path::new(&args[0])
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let model = arg_after(&args, "-m").unwrap_or_default();
    let bytes = std::fs::read(&model).unwrap_or_default();
    let pause = std::env::var("FAKE_LLAMA_SLEEP_MS")
        .ok()
        .and_then(|v| v.parse().ok())
        .map(Duration::from_millis);
    let mut err = std::io::stderr();
    if std::env::var("FAKE_LLAMA_FAIL").as_deref() == Ok("1") {
        for i in 1..=25 {
            let _ = writeln!(err, "llama_model_load: fake log line {i}");
        }
        let _ = writeln!(err, "error: failed to load model '{model}'");
        std::process::exit(1);
    }

    if name.contains("perplexity") {
        let hash = bytes
            .iter()
            .fold(17u64, |h, &b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
        let ppl = 6.0 + (hash % 2000) as f64 / 1000.0;
        let ctx = arg_after(&args, "-c").unwrap_or_else(|| "512".into());
        let _ = writeln!(
            err,
            "perplexity: calculating perplexity over 4 chunks, n_ctx={ctx}, batch_size=2048, n_seq=4"
        );
        for i in 1..=4 {
            if let Some(p) = pause {
                std::thread::sleep(p);
            }
            let _ = write!(err, "[{i}]{:.4},", ppl + (4 - i) as f64 * 0.1);
            let _ = err.flush();
        }
        let _ = writeln!(err, "\nFinal estimate: PPL = {ppl:.4} +/- 0.0400");
        return;
    }

    let prompt = arg_after(&args, "-p").unwrap_or_default();
    let problem = prompt
        .lines()
        .find_map(|l| l.strip_prefix("Problem: "))
        .unwrap_or(&prompt);
    let sum: i64 = problem
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|t| t.parse::<i64>().ok())
        .sum();
    let broken = bytes.windows(6).any(|w| w == b"broken");
    if let Some(p) = pause {
        std::thread::sleep(p);
    }
    let mut out = std::io::stdout();
    let _ = writeln!(
        out,
        "Adding the numbers step by step.\n#### {}",
        sum + i64::from(broken)
    );
}
