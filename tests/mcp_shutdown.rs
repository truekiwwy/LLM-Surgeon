//! The end of stdin is the MCP client's shutdown (Claude Code closing the server): a running
//! native benchmark is cancelled at its next check and `surgeon mcp` exits at once, instead
//! of keeping the CPU and the RAM for the rest of the run.

#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

#[test]
fn closing_stdin_mid_benchmark_ends_the_server_within_two_seconds() {
    let dir = tempfile::tempdir().unwrap();
    fake_qwen2::write_fake_qwen2(dir.path()).unwrap();
    // Far more prompts than the mini model answers in two seconds.
    let prompts: Vec<String> = (0..5000)
        .map(|i| format!("le soleil est {i} et la pluie"))
        .collect();
    std::fs::write(dir.path().join("long.json"), json!(prompts).to_string()).unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_surgeon"))
        .args(["mcp", "--allow-dir", dir.path().to_str().unwrap()])
        .current_dir(dir.path())
        .env("NO_COLOR", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for line in [
        json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "shutdown", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "surgeon_bench", "_meta": {"progressToken": "p"},
            "arguments": {"mode": "refusal", "model": "qwen2-mini", "prompts": "long.json",
                "max_new_tokens": 256, "backend": "native"}}}),
    ] {
        writeln!(stdin, "{line}").unwrap();
    }
    stdin.flush().unwrap();
    let (tx, rx) = mpsc::channel();
    let stdout = child.stdout.take().unwrap();
    std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            let value: Value = serde_json::from_str(&line).expect("JSON-RPC on stdout");
            if tx.send(value).is_err() {
                break;
            }
        }
    });
    // The benchmark runs: its first progress notification came.
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let message = rx
            .recv_timeout(deadline.saturating_duration_since(Instant::now()))
            .expect("the benchmark never reported progress");
        if message["method"] == "notifications/progress" {
            break;
        }
        assert!(message["id"] != 1, "the benchmark ended first: {message}");
    }

    drop(stdin);
    let closed = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if closed.elapsed() > Duration::from_secs(10) {
            let _ = child.kill();
            panic!("surgeon mcp still running 10 s after stdin closed");
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let took = closed.elapsed();
    assert!(
        took < Duration::from_secs(2),
        "exited {took:?} after stdin closed"
    );
    assert_eq!(status.code(), Some(0), "an orderly shutdown");

    // The call was cancelled, not finished: its reply (still written) says so.
    let reply = rx
        .iter()
        .find(|m| m["id"] == 1)
        .expect("the cancelled call is answered");
    let envelope = &reply["result"]["structuredContent"];
    assert_eq!(envelope["error"]["kind"], "cancelled", "{reply}");
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(
        stderr.contains("stdin fermé pendant un outil"),
        "the shutdown is logged: {stderr}"
    );
}
