//! Native generation throughput (non-blocking: not part of the test suite).
//!
//! `cargo run --release --example bench_gen_scaling -- <qwen2 dir> [--tokens 32]
//!  [--max-batch 16] [--procs 1,2,3,6] [--skip batch,jobs,threads,procs]`
//!
//! Greedy decoding of `--tokens` new tokens per sequence after a 32-token prompt, no EOS,
//! so every run generates exactly the same number of tokens. Throughput counts the
//! generated tokens of all sequences over the wall time, prefill included.
//!
//! 1. batch: one batch of B sequences, B = 1, 2, 4, 8, 16 (weights read once per step for
//!    the whole batch).
//! 2. jobs: 8 sequences as 1 x 8, 2 x 4, 4 x 2 (`--jobs` workers x `--batch-size`), one
//!    shared rayon pool.
//! 3. threads: 8 sequences in one batch, in a pool of 1, 2, 4, 6 threads and every core.
//! 4. procs: N copies of this program at once (N from `--procs`), one sequence each, with
//!    `RAYON_NUM_THREADS` unset (every process takes every core) and set to cores / N.
//!
//! Prints Markdown tables; the numbers are this machine's, nothing is extrapolated.

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use weight_surgeon::extract::generate::{BatchJob, Batching, GenRequest, Qwen2State, Sequence};
use weight_surgeon::extract::DTypeChoice;
use weight_surgeon::hardware::SystemProbe;

const PROMPT_TOKENS: usize = 32;

struct Args {
    model: PathBuf,
    tokens: usize,
    max_batch: usize,
    procs: Vec<usize>,
    skip: Vec<String>,
    /// Child mode of `procs`: one sequence, prints its tokens per second.
    single: bool,
}

fn args() -> Args {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut a = Args {
        model: PathBuf::new(),
        tokens: 32,
        max_batch: 16,
        procs: vec![1, 2, 3, 6],
        skip: Vec::new(),
        single: false,
    };
    let mut i = 0;
    while i < argv.len() {
        let value = || argv.get(i + 1).cloned().expect("value missing");
        match argv[i].as_str() {
            "--tokens" => a.tokens = value().parse().expect("--tokens"),
            "--max-batch" => a.max_batch = value().parse().expect("--max-batch"),
            "--procs" => {
                a.procs = value()
                    .split(',')
                    .map(|n| n.parse().expect("--procs"))
                    .collect()
            }
            "--skip" => a.skip = value().split(',').map(String::from).collect(),
            "--single" => {
                a.single = true;
                i += 1;
                continue;
            }
            path => {
                a.model = PathBuf::from(path);
                i += 1;
                continue;
            }
        }
        i += 2;
    }
    assert!(
        a.model.is_dir(),
        "usage: bench_gen_scaling <qwen2 dir> [options]"
    );
    a
}

fn load(model: &Path) -> Qwen2State {
    Qwen2State::load(
        model,
        DTypeChoice::Auto,
        &SystemProbe::detect(),
        PROMPT_TOKENS + 512,
    )
    .expect("model")
}

fn jobs(n: usize, tokens: usize) -> Vec<BatchJob> {
    (0..n)
        .map(|s| BatchJob {
            request: GenRequest {
                // Distinct ordinary token ids per sequence.
                prompt_ids: (0..PROMPT_TOKENS)
                    .map(|i| 1000 + ((i * 7919 + s * 104_729) % 20_000) as u32)
                    .collect(),
                max_new_tokens: tokens,
                eos_ids: Vec::new(),
                steering: None,
            },
            sequence: Sequence::default(),
        })
        .collect()
}

/// Generated tokens per second of `jobs` decoded with `layout`.
fn measure(state: &mut Qwen2State, jobs: &[BatchJob], layout: Batching) -> f64 {
    let started = Instant::now();
    let stats = state
        .generate_batch(
            jobs,
            layout,
            &AtomicBool::new(false),
            &mut |_, _| std::ops::ControlFlow::Continue(()),
            &mut |_| {},
        )
        .expect("generation");
    let generated: usize = stats.iter().map(|s| s.generated).sum();
    generated as f64 / started.elapsed().as_secs_f64()
}

fn layout(batch_size: usize, workers: usize, pool: Option<&rayon::ThreadPool>) -> Batching<'_> {
    Batching {
        batch_size,
        workers,
        pool,
    }
}

fn main() {
    let a = args();
    let cores = std::thread::available_parallelism().map_or(1, usize::from);
    let mut state = load(&a.model);
    if a.single {
        let rate = measure(&mut state, &jobs(1, a.tokens), layout(1, 1, None));
        println!("{rate}");
        return;
    }
    let run = |part: &str| !a.skip.iter().any(|s| s == part);
    println!(
        "{} logical cores, RAYON_NUM_THREADS={}, {} new tokens per sequence, prompt {} tokens\n",
        cores,
        std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".into()),
        a.tokens,
        PROMPT_TOKENS
    );
    // Warm-up: page cache, allocator, rayon threads.
    measure(&mut state, &jobs(1, 4), layout(1, 1, None));

    if run("batch") {
        println!("| B | tokens/s | per sequence | vs B = 1 |\n|---|---|---|---|");
        let mut base = None;
        for b in [1, 2, 4, 8, 16].into_iter().filter(|b| *b <= a.max_batch) {
            let rate = measure(&mut state, &jobs(b, a.tokens), layout(b, 1, None));
            let base = *base.get_or_insert(rate);
            println!(
                "| {b} | {rate:.1} | {:.1} | {:.2}x |",
                rate / b as f64,
                rate / base
            );
        }
        println!();
    }
    if run("jobs") {
        println!("| --jobs x --batch-size | tokens/s |\n|---|---|");
        for workers in [1, 2, 4] {
            let rate = measure(
                &mut state,
                &jobs(8, a.tokens),
                layout(8 / workers, workers, None),
            );
            println!("| {workers} x {} | {rate:.1} |", 8 / workers);
        }
        println!();
    }
    if run("threads") {
        println!("| threads (B = 8) | tokens/s |\n|---|---|");
        let mut counts = vec![1, 2, 4, 6];
        counts.push(cores);
        counts.dedup();
        for threads in counts.into_iter().filter(|t| *t <= cores) {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("pool");
            let rate = measure(&mut state, &jobs(8, a.tokens), layout(8, 1, Some(&pool)));
            println!("| {threads} | {rate:.1} |");
        }
        println!();
    }
    if run("procs") {
        drop(state);
        let exe = std::env::current_exe().expect("exe");
        println!(
            "| processes | RAYON_NUM_THREADS | tokens/s each (mean) | total | vs 1 process |\n\
             |---|---|---|---|---|"
        );
        let mut base = None;
        for n in a.procs.iter().copied() {
            for threads in [None, Some((cores / n).max(1))] {
                let children: Vec<_> = (0..n)
                    .map(|_| {
                        let mut cmd = Command::new(&exe);
                        cmd.arg(&a.model)
                            .args(["--tokens", &a.tokens.to_string(), "--single"]);
                        match threads {
                            Some(t) => cmd.env("RAYON_NUM_THREADS", t.to_string()),
                            None => cmd.env_remove("RAYON_NUM_THREADS"),
                        };
                        cmd.stdout(Stdio::piped())
                            .stderr(Stdio::piped())
                            .spawn()
                            .expect("spawn")
                    })
                    .collect();
                let rates: Vec<f64> = children.into_iter().map(rate).collect();
                let total: f64 = rates.iter().sum();
                let base = *base.get_or_insert(total);
                println!(
                    "| {n} | {} | {:.1} | {total:.1} | {:.2}x |",
                    threads.map_or("unset".into(), |t| t.to_string()),
                    total / n as f64,
                    total / base
                );
                if n == 1 {
                    break; // cores / 1 is the unset case
                }
            }
        }
    }
}

/// Tokens per second printed by a `--single` child.
fn rate(child: std::process::Child) -> f64 {
    let out = child.wait_with_output().expect("child");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .expect("tokens/s")
}
