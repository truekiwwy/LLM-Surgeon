//! P17: the `llama-server` backend against `examples/fake_llama_server.rs`, a real process
//! on a real loopback socket: start (health, whitelisted flags, key, port retry), streaming,
//! sampling passed through, timeouts, cancellation, concurrency up to the slots, a dying
//! server, and the kill when the generator is dropped. No llama.cpp, no model, no GPU.

use std::net::TcpStream;
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

#[path = "support/fake_server.rs"]
mod support;

use support::{fake_server, server_name};
use weight_surgeon::eval::sampling::{sequence_seed, SamplingParams};
use weight_surgeon::eval::server::{ServerGenerator, ServerOptions};
use weight_surgeon::eval::{Decoding, EvalError, GenJob, TextGenerator};

/// A copy of the fake as `llama-server` in its own directory, with its script.
struct Fake {
    dir: tempfile::TempDir,
}

impl Fake {
    fn new(script: Value) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let fake = Self { dir };
        // A hard link, not a copy: a copy holds a write descriptor on the executable, and a
        // child forked meanwhile by another test thread inherits it until its own exec, so
        // executing the copy can fail with ETXTBSY on Linux (rust-lang/rust#114554).
        if std::fs::hard_link(fake_server(), fake.binary()).is_err() {
            std::fs::copy(fake_server(), fake.binary()).unwrap();
        }
        let mut script = script;
        script["args_out"] = json!(fake.path("args.txt"));
        script["log"] = json!(fake.path("requests.jsonl"));
        std::fs::write(fake.binary().with_extension("json"), script.to_string()).unwrap();
        std::fs::write(fake.path("model.gguf"), b"GGUF fake").unwrap();
        fake
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    fn binary(&self) -> PathBuf {
        self.path(&server_name())
    }

    /// Retried while the executable is busy (ETXTBSY, os error 26): only the copy fallback
    /// of `new` can meet it, for the time a concurrently forked child takes to exec.
    fn start(&self, opts: &ServerOptions) -> weight_surgeon::eval::Result<ServerGenerator> {
        let mut attempts = 0;
        loop {
            let started = ServerGenerator::start(
                &self.binary(),
                &self.path("model.gguf"),
                opts,
                &AtomicBool::new(false),
            );
            match started {
                Err(EvalError::Spawn { ref message, .. })
                    if message.contains("os error 26") && attempts < 50 =>
                {
                    attempts += 1;
                    std::thread::sleep(Duration::from_millis(20));
                }
                other => return other,
            }
        }
    }

    fn args(&self) -> Vec<String> {
        std::fs::read_to_string(self.path("args.txt"))
            .unwrap()
            .lines()
            .map(String::from)
            .collect()
    }

    fn requests(&self) -> Vec<Value> {
        std::fs::read_to_string(self.path("requests.jsonl"))
            .unwrap_or_default()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }
}

fn opts(slots: usize) -> ServerOptions {
    ServerOptions {
        slots,
        startup_timeout: Duration::from_secs(20),
        ..ServerOptions::default()
    }
}

fn job(prompt: &str, id: u64) -> GenJob {
    GenJob {
        prompt: prompt.into(),
        sequence_id: id,
    }
}

fn whole(g: &mut ServerGenerator, prompt: &str) -> weight_surgeon::eval::Result<String> {
    g.generate(prompt, 64, &mut |_| ControlFlow::Continue(()))
        .map(|o| o.text)
}

/// A raw request without the key: what any other local process would see.
fn status_without_key(port: u16, path: &str) -> String {
    use std::io::{Read, Write};
    let mut s = TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut reply = String::new();
    s.read_to_string(&mut reply).unwrap();
    reply.lines().next().unwrap_or_default().to_owned()
}

#[test]
fn start_waits_for_health_with_whitelisted_flags_and_a_private_key() {
    let fake = Fake::new(json!({"loading_ms": 400}));
    let started = Instant::now();
    let full = ServerOptions {
        gpu_layers: Some(99),
        ctx_size: Some(4096),
        threads: Some(3),
        ..opts(2)
    };
    let g = fake.start(&full).unwrap();
    assert!(
        started.elapsed() >= Duration::from_millis(400),
        "health polled"
    );
    let args = fake.args();
    let value = |flag: &str| {
        let at = args.iter().position(|a| a == flag).unwrap();
        args[at + 1].clone()
    };
    let flags: Vec<&str> = args.iter().step_by(2).map(String::as_str).collect();
    assert_eq!(
        flags,
        [
            "-m",
            "--host",
            "--port",
            "--api-key",
            "-np",
            "-c",
            "-ngl",
            "-t"
        ]
    );
    assert_eq!(value("--host"), "127.0.0.1");
    assert_eq!(value("--port"), g.port().to_string());
    assert_eq!((value("-np"), value("-c")), ("2".into(), "4096".into()));
    assert_eq!((value("-ngl"), value("-t")), ("99".into(), "3".into()));
    let key = value("--api-key");
    assert!(key.len() == 32 && key.chars().all(|c| c.is_ascii_hexdigit()));
    // The server's own report and its offload line (the way to check a GPU offload).
    assert_eq!(g.props().unwrap()["total_slots"], 2);
    assert!(g.stderr_tail().contains("offloaded 25/25 layers to GPU"));
    // Without the key, the API answers nothing.
    assert!(status_without_key(g.port(), "/props").contains("401"));
    // The build's defaults when nothing is asked: no -c, -ngl, -t.
    let fake = Fake::new(json!({}));
    let _g = fake.start(&opts(1)).unwrap();
    assert_eq!(fake.args().len(), 10);
}

#[test]
fn completions_stream_and_carry_the_sampling() {
    let fake = Fake::new(json!({}));
    let mut g = fake.start(&opts(1)).unwrap();
    let mut chunks = Vec::new();
    let out = g
        .generate("Problem: 2 apples and 40 pears", 64, &mut |c| {
            chunks.push(c.to_owned());
            ControlFlow::Continue(())
        })
        .unwrap();
    assert_eq!(out.text, "Adding the numbers step by step.\n#### 42");
    assert_eq!(chunks.len(), 7, "one event per word: {chunks:?}");
    assert!(!out.stopped_early && !out.truncated);
    // Greedy: the seed of the decoding, temperature 0.
    assert!(g.configure(&Decoding::greedy(1234, None)));
    whole(&mut g, "x").unwrap();
    let sampled = SamplingParams {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.9,
        min_p: 0.05,
        repetition_penalty: 1.1,
        seed: 7,
    };
    g.configure(&Decoding {
        sampling: sampled,
        ..Decoding::greedy(1234, None)
    });
    g.generate_job(&job("x", GenJob::id(3, 2)), 5, &mut |_| {
        ControlFlow::Continue(())
    })
    .unwrap();
    let requests = fake.requests();
    assert_eq!(requests[1]["seed"], 1234);
    assert_eq!(requests[1]["temperature"], 0.0);
    let last = &requests[2];
    assert_eq!(last["seed"], sequence_seed(7, GenJob::id(3, 2)));
    assert_eq!(last["n_predict"], 5);
    assert_eq!(last["top_k"], 40);
    assert!((last["temperature"].as_f64().unwrap() - 0.8).abs() < 1e-6);
    assert!((last["min_p"].as_f64().unwrap() - 0.05).abs() < 1e-6);
    // n_predict bounds the answer.
    let short = g
        .generate("Problem: 1", 3, &mut |_| ControlFlow::Continue(()))
        .unwrap();
    assert_eq!(short.text, "Adding the numbers ");
    assert!(
        short.truncated,
        "the budget ran out before the end of the answer"
    );
}

#[test]
fn break_stops_the_stream_and_timeouts_are_timeouts() {
    let fake = Fake::new(json!({"sleep_ms": 300}));
    let mut g = fake.start(&opts(1)).unwrap();
    let started = Instant::now();
    let out = g
        .generate("Problem: 1", 64, &mut |_| ControlFlow::Break(()))
        .unwrap();
    assert!(out.stopped_early);
    assert_eq!(out.text, "Adding ");
    assert!(
        started.elapsed() < Duration::from_millis(1500),
        "not waited to the end"
    );
    g.configure(&Decoding::greedy(1, Some(Duration::from_millis(100))));
    assert!(matches!(
        whole(&mut g, "Problem: 1"),
        Err(EvalError::Timeout { .. })
    ));
}

#[test]
fn many_jobs_run_concurrently_up_to_the_slots() {
    let fake = Fake::new(json!({"sleep_ms": 40}));
    let mut g = fake.start(&opts(3)).unwrap();
    let jobs: Vec<GenJob> = (1..=7)
        .map(|i| job(&format!("Problem: {i} and {i}"), i))
        .collect();
    let mut done = Vec::new();
    let mut texts = vec![String::new(); jobs.len()];
    let outcomes = g
        .generate_many(
            &jobs,
            64,
            &AtomicBool::new(false),
            &mut |j, c| {
                texts[j].push_str(c);
                ControlFlow::Continue(())
            },
            &mut |j| done.push(j),
        )
        .unwrap();
    assert_eq!(outcomes.len(), 7);
    for (i, o) in outcomes.iter().enumerate() {
        let expected = format!("Adding the numbers step by step.\n#### {}", 2 * (i + 1));
        assert_eq!(o.as_ref().unwrap().text, expected);
        assert_eq!(texts[i], expected, "chunks of job {i} in order");
    }
    done.sort_unstable();
    assert_eq!(done, (0..7).collect::<Vec<_>>());
    let props = g.props().unwrap();
    let concurrent = props["max_concurrent"].as_u64().unwrap();
    assert!((2..=3).contains(&concurrent), "{props}");
    // A raised cancel: no job starts.
    let cancelled = g
        .generate_many(
            &jobs,
            64,
            &AtomicBool::new(true),
            &mut |_, _| ControlFlow::Continue(()),
            &mut |_| {},
        )
        .unwrap();
    assert!(cancelled.is_empty());
}

#[test]
fn a_dying_server_is_an_error_with_its_stderr() {
    let fake = Fake::new(json!({"die_after": 1}));
    let mut g = fake.start(&opts(1)).unwrap();
    whole(&mut g, "Problem: 1").unwrap();
    let err = whole(&mut g, "Problem: 1").unwrap_err();
    assert!(matches!(err, EvalError::Generator(_)), "{err:?}");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !g.stderr_tail().contains("dying") && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(g.stderr_tail().contains("dying after 1"));
}

#[test]
fn dropping_the_generator_kills_the_server() {
    let fake = Fake::new(json!({}));
    let g = fake.start(&opts(1)).unwrap();
    let port = g.port();
    assert!(TcpStream::connect(("127.0.0.1", port)).is_ok());
    drop(g);
    // Killed and reaped by the drop: nothing listens any more.
    assert!(TcpStream::connect(("127.0.0.1", port)).is_err());
}

#[test]
fn start_failures() {
    // A port taken between the choice and the bind: another port, transparently.
    let fake = Fake::new(json!({"busy": "MARKER"}));
    let marker = fake.path("busy.marker");
    let script = json!({"busy": marker, "args_out": fake.path("args.txt")});
    std::fs::write(fake.binary().with_extension("json"), script.to_string()).unwrap();
    let g = fake.start(&opts(1)).unwrap();
    let first: u16 = std::fs::read_to_string(&marker).unwrap().parse().unwrap();
    assert_ne!(first, g.port());
    drop(g);
    // A server that exits: its stderr in the error.
    let fake = Fake::new(json!({"fail": true}));
    match fake.start(&opts(1)) {
        Err(EvalError::CommandFailed { stderr_tail, .. }) => {
            assert!(stderr_tail.contains("fake failure"), "{stderr_tail}")
        }
        other => panic!("{:?}", other.err()),
    }
    // A model that never finishes loading: the startup timeout, the child killed.
    let fake = Fake::new(json!({"loading_ms": 60_000}));
    let quick = ServerOptions {
        startup_timeout: Duration::from_millis(500),
        ..opts(1)
    };
    assert!(matches!(fake.start(&quick), Err(EvalError::Timeout { .. })));
    // A build whose --help lacks a needed flag: refused before any start.
    let fake = Fake::new(json!({"hide": ["-np", "--parallel"]}));
    match fake.start(&opts(2)) {
        Err(EvalError::InvalidConfig(m)) => assert!(m.contains("--parallel"), "{m}"),
        other => panic!("{:?}", other.err()),
    }
    assert!(!fake.path("args.txt").exists(), "never started");
    // Zero slots.
    assert!(matches!(
        Fake::new(json!({})).start(&opts(0)),
        Err(EvalError::InvalidConfig(_))
    ));
}
