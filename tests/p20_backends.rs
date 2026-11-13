//! P20: the inference backends and the persistent user configuration, with the real binary
//! and `examples/fake_llama_server.rs` standing for a GPU llama.cpp, mistral.rs (both command
//! lines) and a hipfire daemon: the first-start probe (`backend detect`), the file it writes,
//! the flags that stay in charge, and what each engine is sent. Every run sets
//! `WEIGHT_SURGEON_CONFIG` itself and clears `LLAMA_CPP_DIR` / `MISTRALRS_DIR`: the harness
//! never reads the machine's own configuration.

#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;
#[path = "support/fake_server.rs"]
mod support;

use std::ffi::OsStr;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

struct Ws {
    dir: tempfile::TempDir,
}

impl Ws {
    /// The mini Qwen2, a "GGUF", a GSM8K file and a fake llama-server in `llama/`.
    fn new() -> Self {
        let ws = Self {
            dir: tempfile::tempdir().unwrap(),
        };
        fake_qwen2::write_fake_qwen2(ws.dir.path()).unwrap();
        std::fs::create_dir(ws.path("llama")).unwrap();
        std::fs::copy(
            support::fake_server(),
            ws.path("llama").join(support::server_name()),
        )
        .unwrap();
        std::fs::write(ws.path("m.gguf"), b"GGUF fake").unwrap();
        std::fs::write(
            ws.path("gsm.json"),
            r#####"[{"id":"q1","question":"1 and 2","answer":"#### 3"},
                {"id":"q2","question":"4 and 5","answer":"#### 9"}]"#####,
        )
        .unwrap();
        ws
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    /// The fake server copied as `<folder>/<name>`, with its script; returns the binary.
    fn engine(&self, folder: &str, name: &str, script: Value) -> PathBuf {
        std::fs::create_dir_all(self.path(folder)).unwrap();
        let bin = self
            .path(folder)
            .join(format!("{name}{}", std::env::consts::EXE_SUFFIX));
        std::fs::copy(support::fake_server(), &bin).unwrap();
        std::fs::write(bin.with_extension("json"), script.to_string()).unwrap();
        bin
    }

    /// The JSON lines a fake engine logged.
    fn requests(&self, log: &str) -> Vec<Value> {
        std::fs::read_to_string(self.path(log))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    fn config(&self) -> PathBuf {
        self.path("user").join("config.toml")
    }

    /// Exit code and envelope of `surgeon --json <args>`; the llama.cpp folder only comes
    /// from `llama_env`, the user file only from `config`.
    fn run(&self, args: &[&str], llama_env: bool) -> (i32, Value) {
        let llama = self.path("llama");
        let env: &[(&str, &OsStr)] = match llama_env {
            true => &[("LLAMA_CPP_DIR", llama.as_os_str())],
            false => &[],
        };
        self.run_env(args, env)
    }

    /// [`Ws::run`] with these variables on top of a clean environment.
    fn run_env(&self, args: &[&str], env: &[(&str, &OsStr)]) -> (i32, Value) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_surgeon"));
        cmd.current_dir(self.dir.path())
            .env("NO_COLOR", "1")
            .env("WEIGHT_SURGEON_CONFIG", self.config())
            .env_remove("LLAMA_CPP_DIR")
            .env_remove("MISTRALRS_DIR")
            .envs(env.iter().copied());
        let out = cmd.arg("--json").args(args).output().unwrap();
        let stdout = String::from_utf8(out.stdout).unwrap();
        let envelope = serde_json::from_str(&stdout)
            .unwrap_or_else(|e| panic!("{args:?}: not strict JSON ({e}): {stdout}"));
        (out.status.code().unwrap(), envelope)
    }

    fn eval(&self, extra: &[&str]) -> (i32, Value) {
        let mut args = vec!["bench", "eval", "--dataset", "gsm.json", "--limit", "2"];
        args.extend(extra);
        self.run(&args, false)
    }
}

#[test]
fn detect_recommends_the_gpu_backend_and_saves_what_bench_then_uses() {
    let ws = Ws::new();
    let (code, v) = ws.run(&["backend", "show"], false);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["exists"], false);
    assert_eq!(v["data"]["effective"], "native-cpu");

    let (code, v) = ws.run(&["backend", "detect", "--save"], true);
    assert_eq!(code, 0, "{v}");
    let report = &v["data"]["report"];
    assert_eq!(report["llama"]["devices"][0]["name"], "Fake GPU");
    assert_eq!(report["llama"]["devices"][0]["id"], "Vulkan0");
    assert_eq!(report["recommended"]["backend"], "llama-server");
    assert_eq!(report["recommended"]["gpu_layers"], 99);
    assert_eq!(report["hipfire"]["binary"], Value::Null);
    let text = std::fs::read_to_string(ws.config()).unwrap();
    assert!(
        text.contains("default_backend = \"llama-server\""),
        "{text}"
    );

    let (code, v) = ws.run(&["backend", "show"], false);
    assert_eq!(code, 0, "{v}");
    assert_eq!(
        v["data"]["effective"],
        "llama-server (Vulkan, ngl=99, batch=1)"
    );

    // No --backend, no --llama-cpp-dir, no LLAMA_CPP_DIR: the file decides.
    let (code, v) = ws.eval(&["--model", "m.gguf"]);
    assert_eq!(code, 0, "{v}");
    let generation = &v["data"]["generation"];
    assert_eq!(generation["backend"], "llama-server", "{v}");
    assert_eq!(
        generation["backend_detail"],
        "llama-server (Vulkan, ngl=99, batch=1)"
    );
    assert_eq!(v["data"]["correct"], 2);

    // A flag beats the file, and the native engine ignores the file's GPU knobs.
    std::fs::write(
        ws.config(),
        text.replace("batch_size = 1", "batch_size = 6"),
    )
    .unwrap();
    let (code, v) = ws.eval(&["--model", "m.gguf"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(
        v["data"]["generation"]["backend_detail"],
        "llama-server (Vulkan, ngl=99, batch=6)"
    );
    let (code, v) = ws.eval(&["--model", "qwen2-mini", "--backend", "native"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["generation"]["backend_detail"], "native-cpu");
    assert_eq!(v["data"]["generation"]["batch_size"], 1, "not the file's 6");
    let (code, v) = ws.eval(&[
        "--model",
        "m.gguf",
        "--gpu-layers",
        "7",
        "--batch-size",
        "2",
    ]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(
        v["data"]["generation"]["backend_detail"],
        "llama-server (Vulkan, ngl=7, batch=2)"
    );
}

#[test]
fn nothing_is_configured_by_default() {
    let ws = Ws::new();
    // Historical defaults: bench eval runs llama (not GPU), and says nothing more.
    let (code, v) = ws.run(&["backend", "detect"], false);
    assert_eq!(code, 0, "{v}");
    assert!(
        !ws.config().exists(),
        "detect without --save writes nothing"
    );
    let backend = &v["data"]["report"]["recommended"]["backend"];
    // mistral.rs only when this machine has it on its PATH.
    assert!(
        backend == "native" || backend == "llama-server" || backend == "mistralrs",
        "{v}"
    );
}

#[test]
fn hipfire_without_binary_or_daemon_names_the_way_out() {
    let ws = Ws::new();
    let missing = ws.path("nowhere").join("hipfire");
    let missing = missing.to_str().unwrap();
    let (code, v) = ws.eval(&[
        "--model",
        "qwen2-mini",
        "--backend",
        "hipfire",
        "--hipfire-bin",
        missing,
    ]);
    assert_eq!(code, 2, "{v}");
    let message = v["error"]["message"].as_str().unwrap();
    // Unless hipfire is on this machine's PATH: then its daemon is not running here.
    assert!(
        message.contains("introuvable") || message.contains("hipfire serve"),
        "{message}"
    );
    assert!(message.contains("--backend mistralrs"), "{message}");
    // A URL off loopback is refused before any request.
    let (code, v) = ws.eval(&[
        "--model",
        "qwen2-mini",
        "--backend",
        "hipfire",
        "--hipfire-url",
        "http://10.0.0.2:11435",
    ]);
    assert_eq!(code, 2, "{v}");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("boucle locale"));
}

#[test]
fn a_broken_user_file_stops_the_generating_commands_only() {
    let ws = Ws::new();
    std::fs::create_dir_all(ws.config().parent().unwrap()).unwrap();
    std::fs::write(ws.config(), "gpu_layers = lots\n").unwrap();
    let (code, v) = ws.eval(&["--model", "m.gguf"]);
    assert_eq!(code, 2, "{v}");
    assert_eq!(v["error"]["kind"], "invalid_input");
    let message = v["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("config.toml") && message.contains("ligne 1"),
        "{message}"
    );
    // A command that never generates does not read it.
    let (code, v) = ws.run(&["check", "--model", "qwen2-mini"], false);
    assert_eq!(code, 0, "{v}");
}

fn args_of(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(String::from)
        .collect()
}

/// The value after `flag` in a command line.
fn after<'a>(args: &'a [String], flag: &str) -> &'a str {
    let at = args.iter().position(|a| a == flag).unwrap();
    &args[at + 1]
}

#[test]
fn mistralrs_serves_the_safetensors_folder_as_is() {
    let ws = Ws::new();
    let script = json!({"args_out": ws.path("args.txt"), "log": ws.path("requests.jsonl")});
    let bin = ws.engine("mistral", "mistralrs", script);
    let (code, v) = ws.eval(&[
        "--model",
        "qwen2-mini",
        "--backend",
        "mistralrs",
        "--mistralrs-bin",
        bin.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["correct"], 2, "{v}");
    let generation = &v["data"]["generation"];
    assert_eq!(generation["backend"], "mistralrs");
    assert_eq!(generation["backend_detail"], "mistralrs (batch=1)");

    // The current command line: the global seed before `serve`, the folder itself, loopback.
    let args = args_of(&ws.path("args.txt"));
    let serve = args.iter().position(|a| a == "serve").unwrap();
    assert_eq!(&args[..serve], ["--seed", "42"]);
    assert_eq!(after(&args, "--host"), "127.0.0.1");
    assert_eq!(after(&args, "--max-seqs"), "1");
    assert!(after(&args, "-m").ends_with("qwen2-mini"), "{args:?}");
    assert!(!args.iter().any(|a| a == "--cpu" || a == "-f"), "{args:?}");
    let ggufs = std::fs::read_dir(ws.dir.path())
        .unwrap()
        .filter(|e| e.as_ref().unwrap().path().extension() == Some(OsStr::new("gguf")))
        .count();
    assert_eq!(ggufs, 1, "only the test's own m.gguf: nothing converted");

    // Raw prompts, greedy and seeded.
    let requests = ws.requests("requests.jsonl");
    assert_eq!(requests.len(), 2);
    for r in &requests {
        assert_eq!(r["model"], "default");
        assert_eq!(r["temperature"].as_f64(), Some(0.0));
        assert_eq!(r["top_p"].as_f64(), Some(1.0));
        assert_eq!(r["seed"], 42);
        assert_eq!(r["stream"], true);
        assert!(
            r["prompt"].is_string() && r.get("messages").is_none(),
            "{r}"
        );
    }
}

#[test]
fn the_legacy_mistralrs_server_takes_a_gguf_too() {
    let ws = Ws::new();
    let script = json!({"legacy": true, "args_out": ws.path("args.txt")});
    ws.engine("mistral", "mistralrs-server", script);
    let dir = ws.path("mistral");
    let (code, v) = ws.run_env(
        &[
            "bench",
            "eval",
            "--dataset",
            "gsm.json",
            "--limit",
            "2",
            "--model",
            "m.gguf",
            "--backend",
            "mistralrs",
        ],
        &[("MISTRALRS_DIR", dir.as_os_str())],
    );
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["correct"], 2, "{v}");
    let args = args_of(&ws.path("args.txt"));
    let gguf = args.iter().position(|a| a == "gguf").unwrap();
    // Global flags before the subcommand, its own after.
    for flag in ["--port", "--serve-ip", "--seed", "--max-seqs"] {
        assert!(args[..gguf].iter().any(|a| a == flag), "{flag} in {args:?}");
    }
    assert_eq!(after(&args, "--serve-ip"), "127.0.0.1");
    assert_eq!(after(&args, "-f"), "m.gguf");
    assert!(args[gguf..].iter().any(|a| a == "-m"), "{args:?}");
}

#[test]
fn detect_puts_mistralrs_first_and_bench_follows_the_file() {
    let ws = Ws::new();
    ws.engine("mistral", "mistralrs", json!({}));
    let mistral = ws.path("mistral");
    let llama = ws.path("llama");
    let env = [
        ("MISTRALRS_DIR", mistral.as_os_str()),
        ("LLAMA_CPP_DIR", llama.as_os_str()),
    ];
    let (code, v) = ws.run_env(&["backend", "detect", "--save"], &env);
    assert_eq!(code, 0, "{v}");
    let report = &v["data"]["report"];
    assert_eq!(report["recommended"]["backend"], "mistralrs");
    assert_eq!(
        report["llama"]["devices"][0]["name"], "Fake GPU",
        "still probed"
    );
    let text = std::fs::read_to_string(ws.config()).unwrap();
    assert!(text.contains("default_backend = \"mistralrs\""), "{text}");
    assert!(text.contains("mistralrs_bin = "), "{text}");
    assert!(
        text.contains("llama_cpp_dir = "),
        "kept for llama-server: {text}"
    );

    let (code, v) = ws.run(&["backend", "show"], false);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["effective"], "mistralrs (batch=1)");
    assert_eq!(v["warnings"], json!([]), "{v}");

    // No flag, no environment: the file decides, and a flag still wins over it.
    let (code, v) = ws.eval(&["--model", "qwen2-mini"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["generation"]["backend"], "mistralrs", "{v}");
    let (code, v) = ws.eval(&["--model", "m.gguf", "--backend", "llama-server"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["generation"]["backend"], "llama-server", "{v}");
}

/// A fake hipfire daemon on a free loopback port, killed on drop.
struct Daemon {
    child: Child,
    url: String,
}

impl Daemon {
    fn start(ws: &Ws, script: Value) -> Self {
        let bin = ws.engine("hip", "hipfire", script);
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let child = Command::new(bin)
            .args(["--port", &port.to_string(), "--host", "127.0.0.1"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let started = Instant::now();
        while std::net::TcpStream::connect(("127.0.0.1", port)).is_err() {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "daemon never listened"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        Self {
            child,
            url: format!("http://127.0.0.1:{port}"),
        }
    }

    /// A user file that picks hipfire on this daemon.
    fn configure(&self, ws: &Ws) {
        std::fs::create_dir_all(ws.config().parent().unwrap()).unwrap();
        let text = format!(
            "default_backend = \"hipfire\"\nhipfire_url = \"{}\"\n",
            self.url
        );
        std::fs::write(ws.config(), text).unwrap();
    }
}

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn hipfire_runs_on_every_os_and_warns_on_a_chat_only_daemon() {
    let ws = Ws::new();
    let daemon = Daemon::start(&ws, json!({"log": ws.path("hip.jsonl")}));
    let url = daemon.url.as_str();
    let (code, v) = ws.eval(&[
        "--model",
        "qwen2-mini",
        "--backend",
        "hipfire",
        "--hipfire-url",
        url,
    ]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["correct"], 2, "{v}");
    assert_eq!(v["data"]["generation"]["backend"], "hipfire");
    // The probe's empty request, then the two questions as chat messages.
    let requests = ws.requests("hip.jsonl");
    let asked: Vec<&Value> = requests
        .iter()
        .filter(|r| r.get("messages").is_some())
        .collect();
    assert_eq!(asked.len(), 2, "{requests:?}");
    assert_eq!(asked[0]["model"], "qwen2-mini");
    assert_eq!(asked[0]["seed"], 42);

    // `backend show` says it, without failing.
    daemon.configure(&ws);
    let (code, v) = ws.run(&["backend", "show"], false);
    assert_eq!(code, 0, "{v}");
    assert!(
        v["warnings"].to_string().contains("/v1/chat/completions"),
        "{v}"
    );
    drop(daemon);

    // A daemon with raw completions: no warning, the bench's prompt as is.
    let ws = Ws::new();
    let daemon = Daemon::start(
        &ws,
        json!({"completions": true, "log": ws.path("hip.jsonl")}),
    );
    daemon.configure(&ws);
    let (code, v) = ws.run(&["backend", "show"], false);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["warnings"], json!([]), "{v}");
    let (code, v) = ws.eval(&["--model", "qwen2-mini"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["correct"], 2, "{v}");
    let prompts = ws
        .requests("hip.jsonl")
        .iter()
        .filter(|r| r["prompt"].as_str().is_some_and(|p| !p.is_empty()))
        .count();
    assert_eq!(prompts, 2);
}
