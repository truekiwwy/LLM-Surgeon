//! Stand-in for the inference servers surgeon drives, to exercise its server backends end to
//! end (real process, real loopback sockets, real HTTP/1.1) without an engine or a model.
//! What it plays depends on the name it is copied as: `llama-server` (llama.cpp),
//! `mistralrs-server` / `mistralrs` (mistral.rs) or `hipfire` (the hipfire daemon). Its
//! script is the optional JSON file of the same name beside it (`llama-server.json`), so that
//! every copy has its own (tests run in parallel).
//!
//! - `--help`: the usage lines of the flags surgeon may pass; `"hide": ["-np", ...]`
//!   leaves some out, like an older build. mistral.rs: the `serve` subcommand and `--seed`,
//!   or with `"legacy": true` the old `mistralrs-server` (`--port --serve-ip --seed
//!   --max-seqs`, subcommands `plain` and `gguf`); `<subcommand> --help`: its own flags.
//! - `--list-devices` (llama.cpp): one fake Vulkan GPU.
//! - otherwise: accepts its whitelisted flags (and the mistral.rs subcommand), each flag with
//!   one value, and exits with code 2 on anything else. Writes its arguments, one per line,
//!   to `"args_out"` when given. llama.cpp prints an offload line on stderr (`offloaded
//!   N/25 layers to GPU`, N from `-ngl`).
//! - `GET /health`: 503 for `"loading_ms"`, then 200.
//! - `GET /props` (llama.cpp): `{"total_slots", "completions", "max_concurrent"}`.
//! - `POST /completion` (llama.cpp, Bearer key required, 401 otherwise), `POST
//!   /v1/completions` and `POST /v1/chat/completions` (mistral.rs and hipfire, no key;
//!   hipfire has the chat one only unless `"completions": true`): the answer of
//!   `examples/fake_llama.rs` (`#### <sum of the integers of the problem>`), or a refusal
//!   when the prompt contains `malware`; at most `n_predict` / `max_tokens` words.
//!   `"stream": true` sends one SSE event per word, each in its own HTTP chunk, then the
//!   end of the dialect (`"stop": true`, or a `finish_reason` and `[DONE]`); otherwise one
//!   JSON body with a Content-Length. Every request body is appended to the file `"log"`
//!   (one JSON per line) when given.
//!
//! Failure scripts: `"sleep_ms"` pauses before every event, `"die_after": N` exits after N
//! completions, `"busy": "<file>"` fails to bind (like a taken port) when the file does not
//! exist yet, then creates it, `"fail": true` exits at once with a load error.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const LLAMA_FLAGS: [&str; 13] = [
    "-m",
    "--model",
    "--host",
    "--port",
    "--api-key",
    "-np",
    "--parallel",
    "-c",
    "--ctx-size",
    "-ngl",
    "--gpu-layers",
    "-t",
    "--threads",
];

/// mistral.rs (both forms) and the hipfire daemon.
const OPENAI_FLAGS: [&str; 10] = [
    "--seed",
    "-p",
    "--port",
    "--host",
    "--serve-ip",
    "--max-seqs",
    "-m",
    "-f",
    "--model-id",
    "--quantized-file",
];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Llama,
    MistralRs,
    Hipfire,
}

/// The script beside the executable, `{}` when there is none.
fn script() -> Value {
    let exe = std::env::current_exe().unwrap();
    std::fs::read_to_string(exe.with_extension("json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_else(|| json!({}))
}

fn mode() -> Mode {
    let exe = std::env::current_exe().unwrap();
    let name = exe.file_stem().unwrap().to_string_lossy().to_string();
    match name {
        n if n.starts_with("mistralrs") => Mode::MistralRs,
        n if n.starts_with("hipfire") => Mode::Hipfire,
        _ => Mode::Llama,
    }
}

fn ms(script: &Value, key: &str) -> Option<Duration> {
    script[key].as_u64().map(Duration::from_millis)
}

struct State {
    mode: Mode,
    /// hipfire without `/v1/completions`.
    chat_only: bool,
    api_key: String,
    slots: usize,
    ready_at: Instant,
    sleep: Option<Duration>,
    die_after: Option<usize>,
    log: Option<Mutex<std::fs::File>>,
    completions: AtomicUsize,
    active: AtomicUsize,
    max_concurrent: AtomicUsize,
}

/// The `--help` lines of `args` (`--help` or `<subcommand> --help`), `None` for a run.
fn help(mode: Mode, args: &[String], script: &Value) -> Option<String> {
    let hidden: Vec<&str> = script["hide"]
        .as_array()
        .map(|h| h.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let lines = |flags: &[&str]| -> String {
        flags
            .iter()
            .filter(|f| !hidden.contains(f))
            .map(|f| format!("  {f:<24} N        (fake)\n"))
            .collect()
    };
    let legacy = script["legacy"] == true;
    let words: Vec<&str> = args.iter().map(String::as_str).collect();
    match (mode, words.as_slice()) {
        (Mode::Llama, ["--help"]) => Some(format!(
            "usage: llama-server [options]\n\n{}",
            lines(&LLAMA_FLAGS)
        )),
        (Mode::MistralRs, ["--help"]) if legacy => Some(format!(
            "Usage: mistralrs-server [OPTIONS] <COMMAND>\n\nCommands:\n  plain  a plain \
             model\n  gguf   a GGUF model\n\nOptions:\n{}",
            lines(&["--port", "--serve-ip", "--seed", "--max-seqs"])
        )),
        (Mode::MistralRs, ["--help"]) => Some(format!(
            "Usage: mistralrs [OPTIONS] <COMMAND>\n\nCommands:\n  serve  Start an \
             OpenAI-compatible HTTP server\n  run    Chat with a model\n\nOptions:\n{}",
            lines(&["--seed"])
        )),
        (Mode::MistralRs, ["serve", "--help"]) => Some(lines(&[
            "-p, --port",
            "--host",
            "--max-seqs",
            "-m, --model-id",
            "-f, --quantized-file",
            "--cpu",
        ])),
        (Mode::MistralRs, ["plain", "--help"]) => Some(lines(&["-m, --model-id"])),
        (Mode::MistralRs, ["gguf", "--help"]) => Some(lines(&[
            "-m, --quantized-model-id",
            "-f, --quantized-filename",
        ])),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let script = script();
    let mode = mode();
    if mode == Mode::Llama && args.iter().any(|a| a == "--list-devices") {
        println!(
            "Available devices:
  Vulkan0: Fake GPU (4096 MiB, 4000 MiB free)"
        );
        return;
    }
    if let Some(text) = help(mode, &args, &script) {
        print!("{text}");
        return;
    }
    let flags: &[&str] = match mode {
        Mode::Llama => &LLAMA_FLAGS,
        _ => &OPENAI_FLAGS,
    };
    let mut values = std::collections::HashMap::new();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        if mode == Mode::MistralRs && matches!(flag.as_str(), "serve" | "plain" | "gguf") {
            values.insert("subcommand".to_owned(), flag.clone());
            continue;
        }
        let Some(value) = it.next().filter(|_| flags.contains(&flag.as_str())) else {
            eprintln!("error: invalid argument: {flag}");
            std::process::exit(2);
        };
        values.insert(flag.trim_start_matches('-').to_owned(), value.clone());
    }
    if let Some(path) = script["args_out"].as_str() {
        std::fs::write(path, args.join("\n")).unwrap();
    }
    let get = |names: &[&str]| names.iter().find_map(|n| values.get(*n).cloned());
    if script["fail"] == true {
        eprintln!("llama_model_load: error loading model: fake failure");
        std::process::exit(1);
    }
    let port = get(&["port", "p"]).unwrap_or_default();
    let host = get(&["host", "serve-ip"]).unwrap_or_default();
    if let Some(marker) = script["busy"].as_str() {
        if std::fs::metadata(marker).is_err() {
            std::fs::write(marker, port.as_bytes()).unwrap();
            eprintln!("couldn't bind HTTP server socket, hostname: {host}, port: {port}");
            std::process::exit(1);
        }
    }
    let listener = match TcpListener::bind(format!("{host}:{port}")) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("couldn't bind HTTP server socket, hostname: {host}, port: {port}: {e}");
            std::process::exit(1);
        }
    };
    if mode == Mode::Llama {
        let ngl: u32 = get(&["ngl", "gpu-layers"]).map_or(0, |n| n.parse().unwrap_or(0));
        eprintln!("load_tensors: offloaded {}/25 layers to GPU", ngl.min(25));
    }
    let state = Arc::new(State {
        mode,
        chat_only: mode == Mode::Hipfire && script["completions"] != true,
        api_key: get(&["api-key"]).unwrap_or_default(),
        slots: get(&["np", "parallel", "max-seqs"]).map_or(1, |n| n.parse().unwrap_or(1)),
        ready_at: Instant::now() + ms(&script, "loading_ms").unwrap_or_default(),
        sleep: ms(&script, "sleep_ms"),
        die_after: script["die_after"].as_u64().map(|n| n as usize),
        log: script["log"].as_str().map(|p| {
            Mutex::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
                    .unwrap(),
            )
        }),
        completions: AtomicUsize::new(0),
        active: AtomicUsize::new(0),
        max_concurrent: AtomicUsize::new(0),
    });
    for stream in listener.incoming().flatten() {
        let state = Arc::clone(&state);
        std::thread::spawn(move || {
            let _ = serve(stream, &state);
        });
    }
}

fn respond(stream: &mut TcpStream, status: &str, body: &str) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n{body}",
        body.len()
    )
}

/// How a completion route is asked and answered.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Route {
    Llama,
    Completions,
    Chat,
}

fn serve(mut stream: TcpStream, state: &State) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let (mut length, mut auth) = (0usize, String::new());
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 || line.trim().is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            match name.trim().to_ascii_lowercase().as_str() {
                "content-length" => length = value.trim().parse().unwrap_or(0),
                "authorization" => auth = value.trim().to_owned(),
                _ => {}
            }
        }
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body)?;
    let mut parts = request_line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    if (method, path) == ("GET", "/health") {
        return match Instant::now() < state.ready_at {
            true => respond(
                &mut stream,
                "503 Service Unavailable",
                r#"{"error":{"message":"Loading model"}}"#,
            ),
            false => respond(&mut stream, "200 OK", r#"{"status":"ok"}"#),
        };
    }
    if state.mode == Mode::Llama && auth != format!("Bearer {}", state.api_key) {
        return respond(
            &mut stream,
            "401 Unauthorized",
            r#"{"error":{"message":"Invalid API Key"}}"#,
        );
    }
    let route = match (state.mode, method, path) {
        (Mode::Llama, "GET", "/props") => {
            let props = json!({
                "total_slots": state.slots,
                "completions": state.completions.load(Ordering::SeqCst),
                "max_concurrent": state.max_concurrent.load(Ordering::SeqCst),
            });
            return respond(&mut stream, "200 OK", &props.to_string());
        }
        (Mode::Llama, "POST", "/completion") => Route::Llama,
        (Mode::MistralRs | Mode::Hipfire, "POST", "/v1/completions") if !state.chat_only => {
            Route::Completions
        }
        (Mode::MistralRs | Mode::Hipfire, "POST", "/v1/chat/completions") => Route::Chat,
        _ => return respond(&mut stream, "404 Not Found", r#"{"error":"not found"}"#),
    };
    let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
    state.max_concurrent.fetch_max(active, Ordering::SeqCst);
    let result = complete(&mut stream, state, route, &body);
    state.active.fetch_sub(1, Ordering::SeqCst);
    let done = state.completions.fetch_add(1, Ordering::SeqCst) + 1;
    if state.die_after.is_some_and(|n| done >= n) {
        eprintln!("fake server: dying after {done} completion(s)");
        std::process::exit(3);
    }
    result
}

fn answer(prompt: &str) -> String {
    if prompt.contains("malware") {
        return "I'm sorry, but I cannot help with that.".into();
    }
    let problem = prompt
        .lines()
        .find_map(|l| l.strip_prefix("Problem: "))
        .unwrap_or(prompt);
    let sum: i64 = problem
        .split(|c: char| !c.is_ascii_digit())
        .filter_map(|t| t.parse::<i64>().ok())
        .sum();
    format!("Adding the numbers step by step.\n#### {sum}")
}

fn chunk(stream: &mut TcpStream, data: &str) -> std::io::Result<()> {
    write!(stream, "{:x}\r\n{data}\r\n", data.len())?;
    stream.flush()
}

/// One streamed event of `route` carrying `text`, and the end when `end` is set.
fn event(route: Route, text: &str, end: Option<&str>) -> Value {
    match route {
        Route::Llama => match end {
            None => json!({"content": text, "stop": false}),
            Some(stop_type) => json!({"content": text, "stop": true, "stop_type": stop_type}),
        },
        Route::Completions => {
            json!({"choices": [{"index": 0, "text": text, "finish_reason": end}]})
        }
        Route::Chat => {
            json!({"choices": [{"index": 0, "delta": {"content": text}, "finish_reason": end}]})
        }
    }
}

fn complete(
    stream: &mut TcpStream,
    state: &State,
    route: Route,
    body: &[u8],
) -> std::io::Result<()> {
    let request: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    if let Some(log) = &state.log {
        let mut log = log.lock().unwrap();
        writeln!(log, "{request}")?;
    }
    let (prompt, budget) = match route {
        Route::Llama => (&request["prompt"], &request["n_predict"]),
        Route::Completions => (&request["prompt"], &request["max_tokens"]),
        Route::Chat => (&request["messages"][0]["content"], &request["max_tokens"]),
    };
    let words = budget.as_u64().unwrap_or(u64::MAX) as usize;
    let answer = answer(prompt.as_str().unwrap_or_default());
    let pieces: Vec<&str> = answer.split_inclusive(' ').take(words).collect();
    let cut = pieces.concat().len() < answer.len();
    let end = match (route, cut) {
        (Route::Llama, true) => "limit",
        (Route::Llama, false) => "eos",
        (_, true) => "length",
        (_, false) => "stop",
    };
    if request["stream"].as_bool() != Some(true) {
        let reply = event(route, &pieces.concat(), Some(end));
        return respond(stream, "200 OK", &reply.to_string());
    }
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\
         Connection: close\r\n\r\n"
    )?;
    for piece in &pieces {
        if let Some(pause) = state.sleep {
            std::thread::sleep(pause);
        }
        chunk(stream, &format!("data: {}\n\n", event(route, piece, None)))?;
    }
    chunk(
        stream,
        &format!("data: {}\n\n", event(route, "", Some(end))),
    )?;
    if route != Route::Llama {
        chunk(stream, "data: [DONE]\n\n")?;
    }
    write!(stream, "0\r\n\r\n")?;
    stream.flush()
}
