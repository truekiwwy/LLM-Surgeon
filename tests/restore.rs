//! The rollback cycle over MCP and the CLI: `surgeon_apply` on a working copy, a measured
//! regression (`surgeon_bench` dump + logits), then `surgeon_restore` from the `.bak-*` and
//! from an `--allow-read` origin, each time back to the initial SHA-256 byte for byte. Also
//! the sandbox (never a read-only root, `--read-only`) and the ambiguous detection.

#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use weight_surgeon::app::{self, AppDeps};
use weight_surgeon::eval::{CommandOutput, CommandRunner, EvalError};
use weight_surgeon::hardware::{GpuInfo, HardwareProbe, MemoryInfo};
use weight_surgeon::mcp::{self, McpConfig};

struct FakeProbe;

impl HardwareProbe for FakeProbe {
    fn system_ram(&self) -> weight_surgeon::hardware::Result<MemoryInfo> {
        Ok(MemoryInfo {
            total_bytes: 64 << 30,
            available_bytes: 64 << 30,
        })
    }
    fn gpus(&self) -> weight_surgeon::hardware::Result<Vec<GpuInfo>> {
        Ok(vec![])
    }
}

struct NoRunner;

impl CommandRunner for NoRunner {
    fn run(
        &self,
        _: &Path,
        _: &[String],
        _: Option<&Path>,
    ) -> weight_surgeon::eval::Result<CommandOutput> {
        Err(EvalError::Generator("unused".into()))
    }
}

fn deps() -> AppDeps {
    AppDeps {
        probe: Box::new(FakeProbe),
        runner: Box::new(NoRunner),
        generator_factory: Box::new(|_, _| Err(EvalError::Generator("unused".into()))),
        interrupted: Arc::new(AtomicBool::new(false)),
        ui: weight_surgeon::ui::UiConfig {
            out: weight_surgeon::ui::ColorMode::Plain,
            err: weight_surgeon::ui::ColorMode::Plain,
            err_is_tty: false,
        },
    }
}

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

/// SHA-256 of every model file (backups and hidden files left out).
fn model_hashes(dir: &Path) -> BTreeMap<String, String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            !name.contains(".bak-") && !name.starts_with('.')
        })
        .map(|e| {
            let digest = Sha256::digest(std::fs::read(e.path()).unwrap());
            let hex = digest.iter().map(|b| format!("{b:02x}")).collect();
            (e.file_name().to_string_lossy().into_owned(), hex)
        })
        .collect()
}

/// One `tools/call` through a real server session: `(isError, envelope)`.
fn one_call(cfg: &McpConfig, tool: &str, arguments: Value) -> (bool, Value) {
    let lines = [
        json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments}}),
    ]
    .map(|v| v.to_string());
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let input = (lines.join("\n") + "\n").into_bytes();
    let code = mcp::serve(
        std::io::Cursor::new(input),
        &mut out,
        &mut err,
        &deps(),
        cfg,
    );
    assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
    let reply = String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str::<Value>(l).unwrap())
        .find(|r| r["id"] == 1)
        .unwrap();
    let result = &reply["result"];
    (
        result["isError"].as_bool().unwrap(),
        result["structuredContent"].clone(),
    )
}

/// Every file restored, each final hash the source's, and the model back to `initial`.
fn assert_restored(env: &Value, model: &Path, initial: &BTreeMap<String, String>) {
    assert_eq!(env["data"]["identical"], true, "{env}");
    for file in env["data"]["files"].as_array().unwrap() {
        assert_eq!(file["sha256_after"], file["sha256_source"], "{file}");
    }
    assert_eq!(&model_hashes(model), initial, "byte for byte");
}

struct Workspace {
    _dir: tempfile::TempDir,
    root: PathBuf,
    origin: PathBuf,
    copy: PathBuf,
    cfg: McpConfig,
}

/// `<root>/origin/qwen2-mini` (read only, nested in the allowed root on purpose) and its
/// working copy `<root>/copy/qwen2-mini`, the prompts and a steering vector.
fn workspace(read_only: bool) -> Workspace {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    let origin = fake_qwen2::write_fake_qwen2(&root.join("origin")).unwrap();
    let copy = fake_qwen2::write_fake_qwen2(&root.join("copy")).unwrap();
    fake_qwen2::write_prompt_files(&root).unwrap();
    let vector = vec![1.0f32; fake_qwen2::HIDDEN];
    std::fs::write(root.join("v.json"), serde_json::to_string(&vector).unwrap()).unwrap();
    let cfg = McpConfig::new(std::slice::from_ref(&root), read_only, false)
        .unwrap()
        .with_read_roots(&[root.join("origin")])
        .unwrap();
    Workspace {
        _dir: dir,
        root,
        origin,
        copy,
        cfg,
    }
}

fn apply(w: &Workspace) -> Value {
    let patch = json!({"schema_version": 1, "backup": true, "operations": [{
        "id": "push", "layers": "all", "projections": ["down_proj"], "operation": "subtract",
        "vector_file": s(&w.root.join("v.json")), "alpha": 4.0, "axis": "cols"}]});
    let (is_error, env) = one_call(
        &w.cfg,
        "surgeon_apply",
        json!({"model": s(&w.copy), "patch": patch}),
    );
    assert!(!is_error, "{env}");
    env
}

#[test]
fn apply_regression_then_restore_from_backup_and_from_origin() {
    let w = workspace(false);
    let initial = model_hashes(&w.copy);
    assert_eq!(
        initial,
        model_hashes(&w.origin),
        "the copy starts as the origin"
    );

    let dump = |model: &Path, out: &str| {
        let out = w.root.join(out);
        let (is_error, env) = one_call(
            &w.cfg,
            "surgeon_bench",
            json!({"mode": "dump", "model": s(model), "prompts": s(&w.root.join("pos.json")), "out": s(&out)}),
        );
        assert!(!is_error, "{env}");
        out
    };
    let before = dump(&w.origin, "before.safetensors");
    let applied = apply(&w);
    assert_ne!(model_hashes(&w.copy), initial, "the apply changed the copy");

    // The counter-measure: the logits moved past the threshold.
    let after = dump(&w.copy, "after.safetensors");
    let (is_error, env) = one_call(
        &w.cfg,
        "surgeon_bench",
        json!({"mode": "logits", "before": s(&before), "after": s(&after), "warn_kl": 0.0}),
    );
    assert!(!is_error, "{env}");
    assert!(
        !env["warnings"].as_array().unwrap().is_empty(),
        "regression flagged: {env}"
    );

    // Rollback from the newest backup, found by itself.
    let (is_error, env) = one_call(&w.cfg, "surgeon_restore", json!({"model": s(&w.copy)}));
    assert!(!is_error, "{env}");
    assert_eq!(env["command"], "restore");
    assert_eq!(env["data"]["mode"], "backup");
    assert_restored(&env, &w.copy, &initial);
    let files = env["data"]["files"].as_array().unwrap();
    assert!(files
        .iter()
        .any(|f| f["rewritten"] == true && f["sha256_before"] != f["sha256_after"]));

    // A second apply leaves two backups of the same shard: detection refuses to guess.
    let second = apply(&w);
    let (is_error, env) = one_call(&w.cfg, "surgeon_restore", json!({"model": s(&w.copy)}));
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "ambiguous_backup", "{env}");

    // The backups named by the apply report.
    let backups = &second["data"]["backups"];
    assert_ne!(backups, &applied["data"]["backups"]);
    let (is_error, env) = one_call(
        &w.cfg,
        "surgeon_restore",
        json!({"model": s(&w.copy), "backup": backups}),
    );
    assert!(!is_error, "{env}");
    assert_restored(&env, &w.copy, &initial);

    // From the read-only origin, after a third apply.
    apply(&w);
    let origin_before = model_hashes(&w.origin);
    let (is_error, env) = one_call(
        &w.cfg,
        "surgeon_restore",
        json!({"model": s(&w.copy), "from_origin": s(&w.origin)}),
    );
    assert!(!is_error, "{env}");
    assert_eq!(env["data"]["mode"], "origin");
    assert_restored(&env, &w.copy, &initial);
    assert_eq!(model_hashes(&w.origin), origin_before, "origin only read");

    // Same through the CLI, already identical: nothing rewritten.
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let argv = [
        "surgeon",
        "--json",
        "restore",
        "--model",
        s(&w.copy),
        "--from-origin",
        s(&w.origin),
    ];
    let cli = app::parse_cli(argv, &mut out, &mut err).unwrap();
    let code = app::run_with(cli, &deps(), &mut out, &mut err);
    let env: Value = serde_json::from_slice(&out).unwrap();
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["command"], "restore");
    assert!(env["data"]["files"]
        .as_array()
        .unwrap()
        .iter()
        .all(|f| f["rewritten"] == false && f["sha256_before"] == f["sha256_source"]));
    assert_restored(&env, &w.copy, &initial);
}

#[test]
fn restore_never_writes_a_read_root_nor_under_read_only() {
    let w = workspace(false);
    let origin_before = model_hashes(&w.origin);
    // The origin is inside the allowed root, but it is a read root: never a destination.
    let (is_error, env) = one_call(
        &w.cfg,
        "surgeon_restore",
        json!({"model": s(&w.origin), "from_origin": s(&w.copy)}),
    );
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "path_not_allowed", "{env}");
    // Outside every root.
    let outside = tempfile::tempdir().unwrap();
    let (is_error, env) = one_call(
        &w.cfg,
        "surgeon_restore",
        json!({"model": s(&w.copy), "from_origin": s(outside.path())}),
    );
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "path_not_allowed", "{env}");
    // Both sources at once.
    let (is_error, env) = one_call(
        &w.cfg,
        "surgeon_restore",
        json!({"model": s(&w.copy), "from_origin": s(&w.origin), "backup_dir": s(&w.root)}),
    );
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "invalid_arguments", "{env}");
    // Nothing to restore from.
    let (is_error, env) = one_call(&w.cfg, "surgeon_restore", json!({"model": s(&w.copy)}));
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "backup_not_found", "{env}");
    assert_eq!(model_hashes(&w.origin), origin_before);

    let ro = workspace(true);
    let (is_error, env) = one_call(
        &ro.cfg,
        "surgeon_restore",
        json!({"model": s(&ro.copy), "from_origin": s(&ro.origin)}),
    );
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "read_only", "{env}");
}
