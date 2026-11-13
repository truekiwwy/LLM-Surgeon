//! The `llama-server` backend: one local llama.cpp server for a whole bench, requests sent
//! concurrently up to its slots. Whether it uses a GPU (Vulkan, HIP, CUDA...) depends on
//! how llama.cpp was built and on `--gpu-layers`; surgeon builds nothing. GPU runs are not
//! reproducible bit for bit (reduction order of the kernels), CPU runs of the same build
//! usually are.
//!
//! ```text
//! start -> flags checked against `llama-server --help` (no free argument, ever)
//!       -> free loopback port, random API key -> spawn (ChildGuard: killed on drop)
//!       -> stderr tail kept -> GET /health until 200 (503 while loading), child alive
//!       -> retry on another port when the bind failed
//! generate_job  -> POST /completion {"stream": true, sampling...} -> SSE `data:` events
//!                  -> on_chunk(content); `Break` drops the connection (the slot stops)
//! generate_many -> `slots` workers, callbacks on the calling thread through a channel
//! ```

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::io::Read;
use std::net::TcpListener;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::command::{last_lines, spawn_error, ChildGuard};
use super::error::{EvalError, Result};
use super::generator::{Decoding, GenJob, GenerationOutcome, TextGenerator};
use super::http::{self, HttpError};
use super::sampling::{self, SamplingParams};

const STDERR_TAIL_BYTES: usize = 64 * 1024;
const PORT_ATTEMPTS: usize = 3;
const POLL: Duration = Duration::from_millis(100);
/// `--help` must answer within this delay.
const HELP_TIMEOUT: Duration = Duration::from_secs(30);

/// The structured options of the server; nothing else reaches its command line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServerOptions {
    /// `-ngl`: layers offloaded to the GPU (0: CPU only; `None`: the build's default).
    pub gpu_layers: Option<u32>,
    /// `-c`: context size of the server (shared by its slots).
    pub ctx_size: Option<u32>,
    /// `-np`: parallel slots, the most concurrent requests.
    pub slots: usize,
    /// `-t`: CPU threads.
    pub threads: Option<usize>,
    pub startup_timeout: Duration,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            gpu_layers: None,
            ctx_size: None,
            slots: 1,
            threads: None,
            startup_timeout: Duration::from_secs(300),
        }
    }
}

/// The flags surgeon may pass, each with the spellings llama.cpp has used: the first one the
/// installed `--help` lists is used, none listed is an error.
const WHITELIST: [(&str, &[&str]); 8] = [
    ("model", &["-m", "--model"]),
    ("host", &["--host"]),
    ("port", &["--port"]),
    ("api-key", &["--api-key"]),
    ("slots", &["-np", "--parallel"]),
    ("ctx", &["-c", "--ctx-size"]),
    ("gpu", &["-ngl", "--gpu-layers", "--n-gpu-layers"]),
    ("threads", &["-t", "--threads"]),
];

/// `true` when `flag` appears in `help` as a whole option (not inside a longer one).
fn listed(help: &str, flag: &str) -> bool {
    help.match_indices(flag).any(|(at, _)| {
        let before = help[..at].chars().next_back();
        let after = help[at + flag.len()..].chars().next();
        !before.is_some_and(|c| c.is_alphanumeric() || c == '-')
            && !after.is_some_and(|c| c.is_alphanumeric() || c == '-' || c == '_')
    })
}

/// The spelling of each whitelisted flag in this build, from its `--help`.
pub fn resolve_flags(help: &str, needed: &[&str]) -> Result<Vec<(&'static str, &'static str)>> {
    needed
        .iter()
        .map(|name| {
            let (key, spellings) = WHITELIST.iter().find(|(k, _)| k == name).ok_or_else(|| {
                EvalError::InvalidConfig(format!("option {name} hors liste blanche"))
            })?;
            spellings
                .iter()
                .find(|s| listed(help, s))
                .map(|s| (*key, *s))
                .ok_or_else(|| {
                    EvalError::InvalidConfig(format!(
                        "llama-server : aucune de {} dans son --help (build trop ancien ou \
                         différent)",
                        spellings.join(", ")
                    ))
                })
        })
        .collect()
}

/// 128 random bits as hex, from the OS-seeded keys of the standard hasher.
fn random_key() -> String {
    (0..2)
        .map(|i| {
            let mut h = RandomState::new().build_hasher();
            h.write_u64(i);
            format!("{:016x}", h.finish())
        })
        .collect()
}

/// A free loopback port (another process may take it before the server binds it: the
/// start retries on another one).
fn free_port() -> Result<u16> {
    TcpListener::bind(("127.0.0.1", 0))
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .map_err(|e| EvalError::InvalidConfig(format!("aucun port local libre : {e}")))
}

/// Output of `binary --help`, stdout and stderr together (llama.cpp prints usage on both).
fn help_text(binary: &Path) -> Result<String> {
    capture(binary, "--help")
}

/// Output of `binary <flag>` (stdout and stderr together), killed after [`HELP_TIMEOUT`].
pub(crate) fn capture(binary: &Path, flag: &str) -> Result<String> {
    let mut child = ChildGuard(
        Command::new(binary)
            .arg(flag)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| spawn_error(binary, e))?,
    );
    let mut readers = Vec::new();
    for pipe in [
        child
            .0
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
        child
            .0
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    ]
    .into_iter()
    .flatten()
    {
        readers.push(thread::spawn(move || {
            let mut text = String::new();
            let mut pipe = pipe;
            let _ = pipe.read_to_string(&mut text);
            text
        }));
    }
    let started = Instant::now();
    while child
        .0
        .try_wait()
        .map_err(|e| EvalError::Generator(e.to_string()))?
        .is_none()
    {
        if started.elapsed() > HELP_TIMEOUT {
            return Err(EvalError::Timeout {
                after_ms: started.elapsed().as_millis(),
            });
        }
        thread::sleep(POLL);
    }
    Ok(readers
        .into_iter()
        .filter_map(|r| r.join().ok())
        .collect::<Vec<_>>()
        .join("\n"))
}

/// A running `llama-server` and its client.
pub struct LlamaServerGenerator {
    // Dropped (killed and reaped) with the generator.
    _child: ChildGuard,
    port: u16,
    api_key: String,
    slots: usize,
    stderr: Arc<Mutex<Vec<u8>>>,
    timeout: Duration,
    seed: u32,
    sampling: SamplingParams,
}

impl LlamaServerGenerator {
    /// Starts the server on `model` (a GGUF) and waits until it is healthy.
    pub fn start(
        binary: &Path,
        model: &Path,
        opts: &ServerOptions,
        cancel: &AtomicBool,
    ) -> Result<Self> {
        if opts.slots == 0 {
            return Err(EvalError::InvalidConfig("--slots doit être >= 1".into()));
        }
        let mut needed = vec!["model", "host", "port", "api-key", "slots"];
        needed.extend(opts.ctx_size.map(|_| "ctx"));
        needed.extend(opts.gpu_layers.map(|_| "gpu"));
        needed.extend(opts.threads.map(|_| "threads"));
        let flags = resolve_flags(&help_text(binary)?, &needed)?;
        let flag = |key: &str| {
            flags
                .iter()
                .find(|(k, _)| *k == key)
                .map(|(_, f)| *f)
                .unwrap_or_default()
        };
        let mut last = None;
        for _ in 0..PORT_ATTEMPTS {
            let port = free_port()?;
            let api_key = random_key();
            let mut cmd = Command::new(binary);
            cmd.arg(flag("model"))
                .arg(model)
                .args([flag("host"), "127.0.0.1", flag("port")])
                .arg(port.to_string())
                .args([flag("api-key"), &api_key, flag("slots")])
                .arg(opts.slots.to_string());
            if let Some(c) = opts.ctx_size {
                cmd.args([flag("ctx"), &c.to_string()]);
            }
            if let Some(n) = opts.gpu_layers {
                cmd.args([flag("gpu"), &n.to_string()]);
            }
            if let Some(t) = opts.threads {
                cmd.args([flag("threads"), &t.to_string()]);
            }
            match Self::launch(cmd, binary, port, api_key, opts, cancel) {
                Ok(server) => return Ok(server),
                Err((e, port_taken)) if port_taken => last = Some(e),
                Err((e, _)) => return Err(e),
            }
        }
        Err(last
            .unwrap_or_else(|| EvalError::Generator("llama-server : démarrage impossible".into())))
    }

    /// Spawns and waits for health; `Err((_, true))` when the port was taken.
    fn launch(
        mut cmd: Command,
        binary: &Path,
        port: u16,
        api_key: String,
        opts: &ServerOptions,
        cancel: &AtomicBool,
    ) -> std::result::Result<Self, (EvalError, bool)> {
        let mut child = ChildGuard(
            cmd.stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| (spawn_error(binary, e), false))?,
        );
        let stderr = Arc::new(Mutex::new(Vec::new()));
        if let Some(mut pipe) = child.0.stderr.take() {
            let tail = Arc::clone(&stderr);
            thread::spawn(move || {
                let mut buf = [0u8; 4096];
                while let Ok(n) = pipe.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    let mut tail = tail.lock().unwrap_or_else(|e| e.into_inner());
                    tail.extend_from_slice(&buf[..n]);
                    let excess = tail.len().saturating_sub(STDERR_TAIL_BYTES);
                    tail.drain(..excess);
                }
            });
        }
        let started = Instant::now();
        let mut server = Self {
            _child: child,
            port,
            api_key,
            slots: opts.slots,
            stderr,
            timeout: Duration::from_secs(600),
            seed: 0,
            sampling: SamplingParams::default(),
        };
        loop {
            if cancel.load(Ordering::SeqCst) {
                return Err((
                    EvalError::Generator("interrompu pendant le démarrage de llama-server".into()),
                    false,
                ));
            }
            if let Ok(Some(status)) = server._child.0.try_wait() {
                let tail = server.stderr_tail();
                let taken = tail.contains("bind")
                    || tail.to_ascii_lowercase().contains("address already in use");
                return Err((
                    EvalError::CommandFailed {
                        step: "llama-server".into(),
                        code: status.code().unwrap_or(-1),
                        stderr_tail: tail,
                    },
                    taken,
                ));
            }
            match http::request(
                port,
                "GET",
                "/health",
                Some(&server.api_key),
                None,
                POLL * 10,
            ) {
                Ok((200, _)) => break,
                Ok(_)
                | Err(HttpError::Status { status: 503, .. })
                | Err(HttpError::Connect { .. })
                | Err(HttpError::Timeout) => {}
                Err(e) => {
                    return Err((
                        EvalError::Generator(format!("llama-server /health : {e}")),
                        false,
                    ))
                }
            }
            if started.elapsed() > opts.startup_timeout {
                return Err((
                    EvalError::Timeout {
                        after_ms: started.elapsed().as_millis(),
                    },
                    false,
                ));
            }
            thread::sleep(POLL);
        }
        Ok(server)
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// The last lines the server wrote on stderr (offload lines, errors).
    pub fn stderr_tail(&self) -> String {
        let tail = self.stderr.lock().unwrap_or_else(|e| e.into_inner());
        last_lines(&String::from_utf8_lossy(&tail), 40)
    }

    /// `GET /props`: the server's own view of its settings (slots, model, build).
    pub fn props(&self) -> Result<Value> {
        let (_, body) = http::request(
            self.port,
            "GET",
            "/props",
            Some(&self.api_key),
            None,
            self.timeout,
        )
        .map_err(|e| self.failed(e))?;
        serde_json::from_str(&body.text().map_err(|e| self.failed(e))?)
            .map_err(|e| EvalError::Generator(format!("llama-server /props : {e}")))
    }

    fn failed(&self, e: HttpError) -> EvalError {
        match e {
            HttpError::Timeout => EvalError::Timeout {
                after_ms: self.timeout.as_millis(),
            },
            other => {
                EvalError::Generator(format!("llama-server : {other}\n{}", self.stderr_tail()))
            }
        }
    }

    fn request_body(&self, job: &GenJob, max_new_tokens: u32) -> String {
        request_body(&self.sampling, self.seed, job, max_new_tokens)
    }
}

/// The `/completion` request: the sampling passed as is (llama.cpp applies it with its
/// own sampler chain, so the draws are not those of the native backend), the seed of the
/// decoding when greedy, else the seed of the job's stream.
fn request_body(p: &SamplingParams, seed: u32, job: &GenJob, max_new_tokens: u32) -> String {
    let seed = match p.is_greedy() {
        true => seed,
        false => sampling::sequence_seed(p.seed, job.sequence_id),
    };
    json!({
        "prompt": job.prompt,
        "n_predict": max_new_tokens,
        "stream": true,
        "temperature": p.temperature,
        "top_k": p.top_k,
        "top_p": p.top_p,
        "min_p": p.min_p,
        "repeat_penalty": p.repetition_penalty,
        "seed": seed,
    })
    .to_string()
}

/// One streamed completion: every `content` to `on_content` (`false` stops and drops the
/// connection). Returns the text, whether it was stopped, and whether the server ended it on
/// the token budget (`stop_type: "limit"`).
fn complete(
    port: u16,
    api_key: &str,
    body: &str,
    timeout: Duration,
    on_content: &mut dyn FnMut(&str) -> bool,
) -> std::result::Result<(String, bool, bool), HttpError> {
    let started = Instant::now();
    let (_, events) = http::request(
        port,
        "POST",
        "/completion",
        Some(api_key),
        Some(body),
        timeout,
    )?;
    let mut text = String::new();
    let mut stopped = false;
    let mut limit = false;
    events.events(&mut |data| {
        if started.elapsed() > timeout {
            return Err(HttpError::Timeout);
        }
        let event: Value = serde_json::from_str(data)
            .map_err(|e| HttpError::Protocol(format!("événement SSE : {e}")))?;
        let content = event
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !content.is_empty() {
            text.push_str(content);
            if !on_content(content) {
                stopped = true;
                return Ok(false);
            }
        }
        limit |= event.get("stop_type").and_then(Value::as_str) == Some("limit");
        Ok(!event.get("stop").and_then(Value::as_bool).unwrap_or(false))
    })?;
    Ok((text, stopped, limit))
}

enum Event {
    Chunk(usize, String),
    Done(usize, Result<GenerationOutcome>),
}

impl TextGenerator for LlamaServerGenerator {
    fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: u32,
        on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<GenerationOutcome> {
        let job = GenJob {
            prompt: prompt.to_owned(),
            sequence_id: 0,
        };
        self.generate_job(&job, max_new_tokens, on_chunk)
    }

    fn generate_job(
        &mut self,
        job: &GenJob,
        max_new_tokens: u32,
        on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<GenerationOutcome> {
        let started = Instant::now();
        let body = self.request_body(job, max_new_tokens);
        let (text, stopped, limit) =
            complete(self.port, &self.api_key, &body, self.timeout, &mut |c| {
                on_chunk(c).is_continue()
            })
            .map_err(|e| self.failed(e))?;
        Ok(GenerationOutcome {
            text,
            stopped_early: stopped,
            truncated: limit,
            duration_ms: started.elapsed().as_millis(),
        })
    }

    /// The sampling as request fields, the timeout per request.
    fn configure(&mut self, decoding: &Decoding) -> bool {
        self.seed = decoding.seed;
        self.sampling = decoding.sampling;
        if let Some(t) = decoding.timeout {
            self.timeout = t;
        }
        true
    }

    /// Up to `slots` requests at once; callbacks run on the calling thread.
    fn generate_many(
        &mut self,
        jobs: &[GenJob],
        max_new_tokens: u32,
        cancel: &AtomicBool,
        on_chunk: &mut dyn FnMut(usize, &str) -> ControlFlow<()>,
        on_done: &mut dyn FnMut(usize),
    ) -> Result<Vec<Result<GenerationOutcome>>> {
        let bodies: Vec<String> = jobs
            .iter()
            .map(|j| self.request_body(j, max_new_tokens))
            .collect();
        let stops: Vec<AtomicBool> = jobs.iter().map(|_| AtomicBool::new(false)).collect();
        let next = AtomicUsize::new(0);
        // Raised by a failed request (other than a timeout): no new request starts.
        let failed = AtomicBool::new(false);
        let mut outcomes: Vec<Option<Result<GenerationOutcome>>> =
            jobs.iter().map(|_| None).collect();
        let (port, key, timeout) = (self.port, self.api_key.as_str(), self.timeout);
        let (tx, rx) = mpsc::channel::<Event>();
        thread::scope(|scope| {
            for _ in 0..self.slots.min(jobs.len()) {
                let (tx, bodies, stops, next) = (tx.clone(), &bodies, &stops, &next);
                let failed = &failed;
                scope.spawn(move || loop {
                    if cancel.load(Ordering::SeqCst) || failed.load(Ordering::SeqCst) {
                        break;
                    }
                    let j = next.fetch_add(1, Ordering::SeqCst);
                    if j >= bodies.len() {
                        break;
                    }
                    let started = Instant::now();
                    let result = complete(port, key, &bodies[j], timeout, &mut |c| {
                        let _ = tx.send(Event::Chunk(j, c.to_owned()));
                        !stops[j].load(Ordering::SeqCst) && !cancel.load(Ordering::SeqCst)
                    })
                    .map(|(text, stopped, limit)| GenerationOutcome {
                        text,
                        stopped_early: stopped,
                        truncated: limit,
                        duration_ms: started.elapsed().as_millis(),
                    });
                    let result = result.map_err(|e| match e {
                        HttpError::Timeout => EvalError::Timeout {
                            after_ms: timeout.as_millis(),
                        },
                        other => {
                            failed.store(true, Ordering::SeqCst);
                            EvalError::Generator(format!("llama-server : {other}"))
                        }
                    });
                    let _ = tx.send(Event::Done(j, result));
                });
            }
            drop(tx);
            for event in rx {
                match event {
                    Event::Chunk(j, chunk) if !stops[j].load(Ordering::SeqCst) => {
                        if on_chunk(j, &chunk).is_break() {
                            stops[j].store(true, Ordering::SeqCst);
                        }
                    }
                    Event::Chunk(..) => {}
                    Event::Done(j, outcome) => {
                        outcomes[j] = Some(outcome);
                        on_done(j);
                    }
                }
            }
        });
        // In job order, up to the first job that did not run (cancellation).
        let mut ordered: Vec<Result<GenerationOutcome>> = Vec::with_capacity(jobs.len());
        for outcome in outcomes {
            match outcome {
                Some(Err(EvalError::Generator(m))) => {
                    return Err(EvalError::Generator(format!("{m}\n{}", self.stderr_tail())))
                }
                Some(o) => ordered.push(o),
                None => break,
            }
        }
        Ok(ordered)
    }
}

/// `llama-server` next to the other llama.cpp binaries.
pub fn locate_server(hint_dir: Option<&Path>) -> Result<PathBuf> {
    super::generator::locate_binary("llama-server", hint_dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HELP: &str = "\
-m,    --model FNAME          model path
-c,    --ctx-size N           size of the prompt context
-cd,   --ctx-size-draft N     draft context
-np,   --parallel N           number of parallel sequences
-ngl,  --gpu-layers, --n-gpu-layers N
--host HOST                   ip address to listen
--port PORT                   port to listen
--api-key KEY                 API key
-t,    --threads N            threads";

    #[test]
    fn flags_come_from_the_help_of_the_installed_build() {
        let all = [
            "model", "host", "port", "api-key", "slots", "ctx", "gpu", "threads",
        ];
        let flags = resolve_flags(HELP, &all).unwrap();
        assert_eq!(
            flags,
            [
                ("model", "-m"),
                ("host", "--host"),
                ("port", "--port"),
                ("api-key", "--api-key"),
                ("slots", "-np"),
                ("ctx", "-c"),
                ("gpu", "-ngl"),
                ("threads", "-t"),
            ]
        );
        // Only whole options: `-c` is not found inside `-cd` or `--ctx-size-draft`.
        assert!(!listed("-cd, --ctx-size-draft N", "-c"));
        assert!(!listed("-cd, --ctx-size-draft N", "--ctx-size"));
        assert!(listed(
            "-ngl, --gpu-layers, --n-gpu-layers N",
            "--gpu-layers"
        ));
        // An older build spelling a flag otherwise: the next spelling of the whitelist.
        let older = HELP.replace("-np,   --parallel", "--parallel");
        assert_eq!(
            resolve_flags(&older, &["slots"]).unwrap(),
            [("slots", "--parallel")]
        );
    }

    #[test]
    fn nothing_outside_the_whitelist() {
        for name in ["", "chat-template", "--model", "rpc", "../model"] {
            assert!(matches!(
                resolve_flags(HELP, &[name]),
                Err(EvalError::InvalidConfig(m)) if m.contains("hors liste blanche")
            ));
        }
        // A whitelisted option the build does not know is an error, never guessed.
        let without = HELP.replace("--api-key KEY", "");
        assert!(matches!(
            resolve_flags(&without, &["api-key"]),
            Err(EvalError::InvalidConfig(m)) if m.contains("--api-key")
        ));
    }

    #[test]
    fn keys_are_random_and_ports_free() {
        let (a, b) = (random_key(), random_key());
        assert_eq!(a.len(), 32);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
        assert_ne!(a, b);
        let port = free_port().unwrap();
        assert!(TcpListener::bind(("127.0.0.1", port)).is_ok());
    }

    #[test]
    fn sampling_reaches_the_request_as_is() {
        // The body only: no process is needed for it.
        let job = GenJob {
            prompt: "p".into(),
            sequence_id: GenJob::id(2, 1),
        };
        let fields = |sampling: SamplingParams, seed: u32| {
            serde_json::from_str::<Value>(&request_body(&sampling, seed, &job, 9)).unwrap()
        };
        let greedy = fields(SamplingParams::default(), 1234);
        assert_eq!(greedy["seed"], 1234);
        assert_eq!(greedy["temperature"], 0.0);
        assert_eq!(greedy["prompt"], "p");
        assert_eq!(greedy["n_predict"], 9);
        assert_eq!(greedy["stream"], true);
        let p = SamplingParams {
            temperature: 0.75,
            top_k: 40,
            top_p: 0.5,
            min_p: 0.25,
            repetition_penalty: 1.5,
            seed: 7,
        };
        let drawn = fields(p, 1234);
        assert_eq!(drawn["seed"], sampling::sequence_seed(7, job.sequence_id));
        assert_eq!(drawn["temperature"], 0.75);
        assert_eq!(drawn["top_k"], 40);
        assert_eq!(drawn["top_p"], 0.5);
        assert_eq!(drawn["min_p"], 0.25);
        assert_eq!(drawn["repeat_penalty"], 1.5);
    }
}
