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
fn detect_recommends_a_rust_engine_and_bench_follows_a_llama_file() {
    let ws = Ws::new();
    let (code, v) = ws.run(&["backend", "show"], false);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["exists"], false);
    assert_eq!(v["data"]["effective"], "native-cpu");

    // A llama.cpp build with a GPU is in reach: the probe neither looks for it nor picks it.
    let (code, v) = ws.run(&["backend", "detect"], true);
    assert_eq!(code, 0, "{v}");
    let report = &v["data"]["report"];
    assert!(report.get("llama").is_none(), "{report}");
    assert!(
        report["gpus"].is_array() && report["vendor"].is_string(),
        "{report}"
    );
    let backend = report["recommended"]["backend"].as_str().unwrap();
    assert!(
        ["native", "mistralrs", "hipfire"].contains(&backend),
        "{backend}"
    );
    assert_eq!(report["recommended"]["llama_cpp_dir"], Value::Null);
    assert!(!v.to_string().contains("llama-server"), "{v}");

    // llama-server stays a backend one names by hand.
    let llama = ws.path("llama").display().to_string().replace('\\', "/");
    let text = format!(
        "default_backend = \"llama-server\"\ngpu_layers = 99\nbatch_size = 1\nllama_cpp_dir = \"{llama}\"\ngguf_dtype = \"f16\"\n"
    );
    std::fs::create_dir_all(ws.config().parent().unwrap()).unwrap();
    std::fs::write(ws.config(), &text).unwrap();

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
    let report = &v["data"]["report"];
    let backend = &report["recommended"]["backend"];
    // mistral.rs only when this machine has it on its PATH, hipfire only on an AMD machine
    // whose daemon answers at its default address (what `detect` probes there).
    let amd_daemon = report["vendor"] == "amd" && report["hipfire"]["up"] == true;
    assert!(
        backend == "native"
            || backend == "llama-server"
            || backend == "mistralrs"
            || (amd_daemon && backend == "hipfire"),
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
    // Off AMD, or on AMD until its hipfire daemon answers.
    if report["recommended"]["backend"] == "hipfire" {
        return;
    }
    assert_eq!(report["recommended"]["backend"], "mistralrs");
    assert!(report.get("llama").is_none(), "{report}");
    let text = std::fs::read_to_string(ws.config()).unwrap();
    assert!(text.contains("default_backend = \"mistralrs\""), "{text}");
    assert!(text.contains("mistralrs_bin = "), "{text}");

    let (code, v) = ws.run(&["backend", "show"], false);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["effective"], "mistralrs (batch=1)");
    assert_eq!(v["warnings"], json!([]), "{v}");

    // No flag, no environment: the file decides, and a flag still wins over it.
    let (code, v) = ws.eval(&["--model", "qwen2-mini"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["generation"]["backend"], "mistralrs", "{v}");
    let (code, v) = ws.run(
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
            "llama-server",
        ],
        true,
    );
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

/// `copy/`: the mini model patched in place, the backup `surgeon apply` left beside it.
fn working_copy(ws: &Ws) {
    let copy = ws.path("copy");
    std::fs::create_dir_all(&copy).unwrap();
    for entry in std::fs::read_dir(ws.path("qwen2-mini")).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), copy.join(entry.file_name())).unwrap();
    }
    let weights = copy.join("model.safetensors");
    std::fs::copy(&weights, copy.join("model.safetensors.bak-1700000000-1")).unwrap();
}

/// The questions a fake daemon was asked (the probe's empty request is not one).
fn asked(ws: &Ws, log: &str) -> usize {
    match ws.path(log).exists() {
        true => ws
            .requests(log)
            .iter()
            .filter(|r| r["prompt"].as_str().is_some_and(|p| !p.is_empty()))
            .count(),
        false => 0,
    }
}

#[test]
fn the_cli_never_measures_a_working_copy_on_hipfire_unless_told_to() {
    let ws = Ws::new();
    let daemon = Daemon::start(
        &ws,
        json!({"completions": true, "log": ws.path("hip.jsonl")}),
    );
    working_copy(&ws);
    let hip = ["--backend", "hipfire", "--hipfire-url", daemon.url.as_str()];
    let (code, v) = ws.eval(&[&["--model", "copy"][..], &hip].concat());
    assert_eq!(code, 2, "{v}");
    let message = v["error"]["message"].as_str().unwrap();
    assert!(message.contains("copie de travail"), "{message}");
    assert_eq!(asked(&ws, "hip.jsonl"), 0, "refused before any question");
    // A working copy as the baseline is refused the same way, before the original runs.
    let baseline = ["--model", "qwen2-mini", "--baseline-model", "copy"];
    let (code, v) = ws.eval(&[&baseline[..], &hip].concat());
    assert_eq!(code, 2, "{v}");
    assert_eq!(asked(&ws, "hip.jsonl"), 0);

    // Told to (requantized and registered): measured, and the report says so.
    let allow = ["--model", "copy", "--allow-stale-hipfire"];
    let (code, v) = ws.eval(&[&allow[..], &hip].concat());
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["generation"]["backend"], "hipfire");
    assert!(
        v["warnings"].to_string().contains("--allow-stale-hipfire"),
        "{v}"
    );
    assert_eq!(asked(&ws, "hip.jsonl"), 2);
}

/// The models a fake daemon was asked about, question by question.
fn served_models(ws: &Ws, log: &str) -> Vec<String> {
    ws.requests(log)
        .iter()
        .filter(|r| r["prompt"].as_str().is_some_and(|p| !p.is_empty()))
        .map(|r| r["model"].as_str().unwrap().to_owned())
        .collect()
}

/// One byte of `copy/model.safetensors` changed, its size kept.
fn touch_weights(ws: &Ws) {
    let weights = ws.path("copy").join("model.safetensors");
    let mut bytes = std::fs::read(&weights).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    std::fs::write(&weights, bytes).unwrap();
}

#[test]
fn a_requantized_copy_is_measured_by_alias_next_to_its_original() {
    use weight_surgeon::backend::hipfire;
    let ws = Ws::new();
    let daemon = Daemon::start(
        &ws,
        json!({"completions": true, "log": ws.path("hip.jsonl")}),
    );
    working_copy(&ws);
    // What `--hipfire-requantize` leaves once hipfire registered the copy.
    let copy = ws.path("copy");
    let sha = hipfire::weights_sha256(&copy).unwrap();
    let alias = hipfire::alias_for("copy", &sha);
    let record =
        json!({"alias": alias, "weights_sha256": sha, "created_at": "2026-11-20T00:00:00Z"});
    std::fs::write(copy.join(hipfire::ALIAS_FILE), record.to_string()).unwrap();

    let hip = ["--backend", "hipfire", "--hipfire-url", daemon.url.as_str()];
    let pair = ["--model", "copy", "--baseline-model", "qwen2-mini"];
    let (code, v) = ws.eval(&[&pair[..], &hip].concat());
    assert_eq!(code, 0, "{v}");
    assert!(!v["warnings"].to_string().contains("hipfire"), "{v}");
    // The original first, then the copy under its alias: both on the same daemon.
    let original = "qwen2-mini".to_owned();
    assert_eq!(
        served_models(&ws, "hip.jsonl"),
        [original.clone(), original, alias.clone(), alias.clone()]
    );

    // A shard rewritten after the quantization: refused before any question.
    touch_weights(&ws);
    let (code, v) = ws.eval(&[&pair[..], &hip].concat());
    assert_eq!(code, 2, "{v}");
    let message = v["error"]["message"].as_str().unwrap();
    assert!(
        message.contains(&format!("poids modifiés depuis l'alias {alias}")),
        "{message}"
    );
    assert_eq!(served_models(&ws, "hip.jsonl").len(), 4);
}

/// `--hipfire-requantize` with a fake hipfire CLI that registers aliases as hipfire 0.4.6
/// writes them. Unix only: on Windows the requantization runs hipfire in WSL.
#[cfg(unix)]
#[test]
fn hipfire_requantize_registers_a_copy_once_and_again_when_its_weights_change() {
    use std::os::unix::fs::PermissionsExt;
    let ws = Ws::new();
    let daemon = Daemon::start(
        &ws,
        json!({"completions": true, "log": ws.path("hip.jsonl")}),
    );
    working_copy(&ws);
    let home = ws.path("hipfire-home");
    std::fs::create_dir_all(&home).unwrap();
    let cli = ws.path("hipfire-cli");
    std::fs::write(
        &cli,
        "#!/bin/sh\necho \"$@\" >> \"$HIPFIRE_HOME/calls\"\n\
         while [ \"$1\" != --register ]; do shift; done\n\
         printf '%s = \"%s.q8f16\"\\n' \"$2\" \"$2\" >> \"$HIPFIRE_HOME/models.toml\"\n",
    )
    .unwrap();
    std::fs::set_permissions(&cli, std::fs::Permissions::from_mode(0o755)).unwrap();
    let (url, cli) = (daemon.url.as_str(), cli.to_str().unwrap());
    let run = |extra: &[&str]| {
        let mut args = vec!["bench", "eval", "--dataset", "gsm.json", "--limit", "2"];
        args.extend([
            "--model",
            "copy",
            "--backend",
            "hipfire",
            "--hipfire-url",
            url,
        ]);
        args.extend(["--hipfire-bin", cli]);
        args.extend(extra);
        ws.run_env(&args, &[("HIPFIRE_HOME", home.as_os_str())])
    };
    let calls = || {
        std::fs::read_to_string(home.join("calls"))
            .unwrap_or_default()
            .lines()
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let recorded = || -> Value {
        let text = std::fs::read_to_string(ws.path("copy").join(".hipfire-alias.json"));
        serde_json::from_str(&text.unwrap()).unwrap()
    };

    let (code, v) = run(&["--hipfire-requantize"]);
    assert_eq!(code, 0, "{v}");
    let first = recorded();
    let alias = first["alias"].as_str().unwrap().to_owned();
    assert!(
        alias.starts_with("ws-copy-") && alias.len() == 16,
        "{alias}"
    );
    assert_eq!(first["weights_sha256"].as_str().unwrap()[..8], alias[8..]);
    assert_eq!(calls().len(), 1);
    let call = &calls()[0];
    assert!(call.starts_with("quantize "), "{call}");
    assert!(
        call.contains(&format!("--format q8f16 --stem {alias}")),
        "{call}"
    );
    assert!(
        call.contains(&format!("--install --register {alias}")),
        "{call}"
    );
    assert!(call.contains(ws.path("copy").to_str().unwrap()), "{call}");
    assert_eq!(
        served_models(&ws, "hip.jsonl"),
        [alias.clone(), alias.clone()]
    );

    // Same weights: served at once, hipfire not called again.
    let (code, v) = run(&["--hipfire-requantize"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(calls().len(), 1);

    // Weights changed: refused without the flag, requantized under a new alias with it.
    touch_weights(&ws);
    let (code, v) = run(&[]);
    assert_eq!(code, 2, "{v}");
    let (code, v) = run(&["--hipfire-requantize"]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(calls().len(), 2);
    let second = recorded()["alias"].as_str().unwrap().to_owned();
    assert_ne!(second, alias);
    assert_eq!(served_models(&ws, "hip.jsonl").last(), Some(&second));
}

#[test]
fn mcp_moves_a_working_copy_off_hipfire_and_says_why() {
    use std::io::{BufRead, BufReader, Read, Write};
    let ws = Ws::new();
    let daemon = Daemon::start(
        &ws,
        json!({"completions": true, "log": ws.path("hip.jsonl")}),
    );
    daemon.configure(&ws);
    working_copy(&ws);
    let mut child = Command::new(env!("CARGO_BIN_EXE_surgeon"))
        .args(["mcp", "--allow-dir", ws.dir.path().to_str().unwrap()])
        .current_dir(ws.dir.path())
        .env("WEIGHT_SURGEON_CONFIG", ws.config())
        .env_remove("LLAMA_CPP_DIR")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    for line in [
        json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "p20", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "surgeon_bench", "arguments": {"mode": "refusal", "model": "copy",
            "prompts": "builtin:refusals", "max_new_tokens": 2}}}),
    ] {
        writeln!(stdin, "{line}").unwrap();
    }
    stdin.flush().unwrap();
    let mut reply = Value::Null;
    for line in BufReader::new(child.stdout.take().unwrap()).lines() {
        let value: Value = serde_json::from_str(&line.unwrap()).unwrap();
        if value["id"] == 1 {
            reply = value;
            break;
        }
    }
    drop(stdin);
    child.wait().unwrap();
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    let result = &reply["result"]["structuredContent"];
    assert_eq!(result["ok"], true, "{reply}");
    let first = result["warnings"][0].as_str().unwrap();
    assert!(first.starts_with("backend hipfire écarté"), "{first}");
    assert!(first.contains("copy"), "the reason names the copy: {first}");
    assert!(
        stderr.contains("backend hipfire écarté"),
        "logged: {stderr}"
    );
    assert_eq!(asked(&ws, "hip.jsonl"), 0, "measured on native");
}

#[test]
fn hipfire_runs_on_every_os_and_warns_on_a_chat_only_daemon() {
    let ws = Ws::new();
    let daemon = Daemon::start(&ws, json!({"log": ws.path("hip.jsonl")}));
    let url = daemon.url.as_str();
    let hip = [
        "--model",
        "qwen2-mini",
        "--backend",
        "hipfire",
        "--hipfire-url",
        url,
    ];
    let (code, v) = ws.eval(&[&hip[..], &["--chat-template", "chatml"]].concat());
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["correct"], 2, "{v}");
    assert_eq!(v["data"]["generation"]["backend"], "hipfire");
    // The probe's empty request, then the two questions as chat messages: the bench's
    // ChatML turns unwrapped, so hipfire's template is the only one, greedy.
    let requests = ws.requests("hip.jsonl");
    let asked: Vec<&Value> = requests
        .iter()
        .filter(|r| r.get("messages").is_some())
        .collect();
    assert_eq!(asked.len(), 2, "{requests:?}");
    assert_eq!(asked[0]["model"], "qwen2-mini");
    assert_eq!(asked[0]["seed"], 42);
    assert_eq!(asked[0]["temperature"], 0.0);
    let messages = asked[0]["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1, "{messages:?}");
    assert_eq!(messages[0]["role"], "user");
    let content = messages[0]["content"].as_str().unwrap();
    assert!(
        content.contains("1 and 2") && !content.contains("<|im_"),
        "{content}"
    );

    // Sampling cannot be replayed without a seed: refused before any question.
    let sampled = ws.eval(&[&hip[..], &["--temperature", "0.7"]].concat());
    assert_eq!(sampled.0, 2, "{}", sampled.1);
    assert!(sampled.1.to_string().contains("glouton"), "{}", sampled.1);
    assert_eq!(ws.requests("hip.jsonl").len(), requests.len());

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
