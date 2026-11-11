//! Stand-in for `llama-server`, to exercise surgeon's server backend end to end (real
//! process, real loopback sockets, real HTTP/1.1) without llama.cpp or a model. Copy it as
//! `llama-server`; its script is the optional JSON file of the same name beside it
//! (`llama-server.json`), so that every copy has its own (tests run in parallel).
//!
//! - `--help`: the usage lines of the flags surgeon may pass; `"hide": ["-np", ...]`
//!   leaves some out, like an older build.
//! - otherwise: accepts `-m --host --port --api-key -np -c -ngl -t` (and their long
//!   spellings), each with one value, and exits with code 2 on anything else. Writes its
//!   arguments, one per line, to `"args_out"` when given. Prints an offload line on stderr
//!   like llama.cpp (`offloaded N/25 layers to GPU`, N from `-ngl`).
//! - `GET /health`: 503 for `"loading_ms"`, then 200.
//! - `GET /props`: `{"total_slots", "completions", "max_concurrent"}`.
//! - `POST /completion` (Bearer key required, 401 otherwise): the answer of
//!   `examples/fake_llama.rs` (`#### <sum of the integers of the problem>`), or a refusal
//!   when the prompt contains `malware`; at most `n_predict` words. `"stream": true` sends
//!   one SSE event per word, each in its own HTTP chunk, then a `"stop": true` event;
//!   otherwise one JSON body with a Content-Length. Every request body is appended to the
//!   file `"log"` (one JSON per line) when given.
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

const FLAGS: [&str; 13] = [
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

/// The script beside the executable, `{}` when there is none.
fn script() -> Value {
    let exe = std::env::current_exe().unwrap();
    std::fs::read_to_string(exe.with_extension("json"))
        .ok()
        .and_then(|text| serde_json::from_str(&text).ok())
        .unwrap_or_else(|| json!({}))
}

fn ms(script: &Value, key: &str) -> Option<Duration> {
    script[key].as_u64().map(Duration::from_millis)
}

struct State {
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let script = script();
    if args.iter().any(|a| a == "--help") {
        let hidden: Vec<&str> = script["hide"]
            .as_array()
            .map(|h| h.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        println!("usage: llama-server [options]\n");
        for flag in FLAGS.iter().filter(|f| !hidden.contains(f)) {
            println!("{flag:<24} N        (fake)");
        }
        return;
    }
    let mut values = std::collections::HashMap::new();
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let Some(value) = it.next().filter(|_| FLAGS.contains(&flag.as_str())) else {
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
    let port = get(&["port"]).unwrap_or_default();
    let host = get(&["host"]).unwrap_or_default();
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
    let ngl: u32 = get(&["ngl", "gpu-layers"]).map_or(0, |n| n.parse().unwrap_or(0));
    eprintln!("load_tensors: offloaded {}/25 layers to GPU", ngl.min(25));
    let state = Arc::new(State {
        api_key: get(&["api-key"]).unwrap_or_default(),
        slots: get(&["np", "parallel"]).map_or(1, |n| n.parse().unwrap_or(1)),
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
    if auth != format!("Bearer {}", state.api_key) {
        return respond(
            &mut stream,
            "401 Unauthorized",
            r#"{"error":{"message":"Invalid API Key"}}"#,
        );
    }
    match (method, path) {
        ("GET", "/props") => {
            let props = json!({
                "total_slots": state.slots,
                "completions": state.completions.load(Ordering::SeqCst),
                "max_concurrent": state.max_concurrent.load(Ordering::SeqCst),
            });
            respond(&mut stream, "200 OK", &props.to_string())
        }
        ("POST", "/completion") => {
            let active = state.active.fetch_add(1, Ordering::SeqCst) + 1;
            state.max_concurrent.fetch_max(active, Ordering::SeqCst);
            let result = complete(&mut stream, state, &body);
            state.active.fetch_sub(1, Ordering::SeqCst);
            let done = state.completions.fetch_add(1, Ordering::SeqCst) + 1;
            if state.die_after.is_some_and(|n| done >= n) {
                eprintln!("fake llama-server: dying after {done} completion(s)");
                std::process::exit(3);
            }
            result
        }
        _ => respond(&mut stream, "404 Not Found", r#"{"error":"not found"}"#),
    }
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

fn complete(stream: &mut TcpStream, state: &State, body: &[u8]) -> std::io::Result<()> {
    let request: Value = serde_json::from_slice(body).unwrap_or(Value::Null);
    if let Some(log) = &state.log {
        let mut log = log.lock().unwrap();
        writeln!(log, "{request}")?;
    }
    let prompt = request["prompt"].as_str().unwrap_or_default();
    let words = request["n_predict"].as_u64().unwrap_or(u64::MAX) as usize;
    let answer = answer(prompt);
    let pieces: Vec<&str> = answer.split_inclusive(' ').take(words).collect();
    if request["stream"].as_bool() != Some(true) {
        let reply = json!({"content": pieces.concat(), "stop": true});
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
        let event = json!({"content": piece, "stop": false});
        chunk(stream, &format!("data: {event}\n\n"))?;
    }
    let last = json!({"content": "", "stop": true, "tokens_predicted": pieces.len()});
    chunk(stream, &format!("data: {last}\n\n"))?;
    write!(stream, "0\r\n\r\n")?;
    stream.flush()
}
