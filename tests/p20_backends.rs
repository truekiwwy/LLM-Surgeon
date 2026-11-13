//! P20: the inference backends and the persistent user configuration, with the real binary
//! and `examples/fake_llama_server.rs` standing for a GPU llama.cpp: the first-start probe
//! (`backend detect`), the file it writes, and the flags that stay in charge. Every run sets
//! `WEIGHT_SURGEON_CONFIG` itself: the harness never reads the machine's own configuration.

#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;
#[path = "support/fake_server.rs"]
mod support;

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

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

    fn config(&self) -> PathBuf {
        self.path("user").join("config.toml")
    }

    /// Exit code and envelope of `surgeon --json <args>`; the llama.cpp folder only comes
    /// from `llama_env`, the user file only from `config`.
    fn run(&self, args: &[&str], llama_env: bool) -> (i32, Value) {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_surgeon"));
        cmd.current_dir(self.dir.path())
            .env("NO_COLOR", "1")
            .env("WEIGHT_SURGEON_CONFIG", self.config())
            .env_remove("LLAMA_CPP_DIR");
        if llama_env {
            cmd.env("LLAMA_CPP_DIR", self.path("llama"));
        }
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
    assert!(backend == "native" || backend == "llama-server", "{v}");
}

#[test]
fn hipfire_is_refused_with_a_way_out() {
    let ws = Ws::new();
    let (code, v) = ws.eval(&["--model", "qwen2-mini", "--backend", "hipfire"]);
    assert_eq!(code, 2, "{v}");
    let message = v["error"]["message"].as_str().unwrap();
    assert!(message.contains("--backend llama-server"), "{message}");
    assert!(
        message.contains("introuvable") || message.contains("non supporté"),
        "{message}"
    );
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
    let _: &Path = ws.dir.path();
}
