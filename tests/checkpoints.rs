#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use weight_surgeon::app::AppDeps;
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

fn hashes(dir: &Path) -> BTreeMap<String, String> {
    std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| e.file_type().unwrap().is_file())
        .filter(|e| !e.file_name().to_string_lossy().contains(".bak-"))
        .map(|e| {
            let digest = Sha256::digest(std::fs::read(e.path()).unwrap());
            let hex = digest.iter().map(|b| format!("{b:02x}")).collect();
            (e.file_name().to_string_lossy().into_owned(), hex)
        })
        .collect()
}

fn call(cfg: &McpConfig, tool: &str, arguments: Value) -> (bool, Value) {
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

fn ok(cfg: &McpConfig, tool: &str, arguments: Value) -> Value {
    let (is_error, env) = call(cfg, tool, arguments);
    assert!(!is_error, "{tool}: {env}");
    env["data"].clone()
}

struct Workspace {
    _dir: tempfile::TempDir,
    root: PathBuf,
    copy: PathBuf,
    cfg: McpConfig,
}

fn workspace(read_only: bool) -> Workspace {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().canonicalize().unwrap();
    fake_qwen2::write_fake_qwen2(&root.join("origin")).unwrap();
    let copy = fake_qwen2::write_fake_qwen2(&root.join("copies")).unwrap();
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
        copy,
        cfg,
    }
}

fn patch(w: &Workspace) -> Value {
    json!({"schema_version": 1, "backup": true, "operations": [{
        "id": "push", "layers": "all", "projections": ["down_proj"], "operation": "subtract",
        "vector_file": s(&w.root.join("v.json")), "alpha": 4.0, "axis": "cols"}]})
}

#[test]
fn apply_never_writes_through_a_checkpoint_and_restore_is_byte_for_byte() {
    let w = workspace(false);
    let initial = hashes(&w.copy);
    let created = ok(
        &w.cfg,
        "surgeon_checkpoint",
        json!({"action": "create", "model": s(&w.copy), "label": "before", "metrics": {"gain": 0.0}}),
    );
    let id = created["id"].as_str().unwrap().to_owned();
    let snapshot = w.root.join("work/checkpoints").join(&id).join("model");
    assert_eq!(hashes(&snapshot), initial);
    assert!(created["files"]
        .as_array()
        .unwrap()
        .iter()
        .all(|f| f["linked"] == true));

    ok(
        &w.cfg,
        "surgeon_apply",
        json!({"model": s(&w.copy), "patch": patch(&w)}),
    );
    assert_ne!(hashes(&w.copy), initial, "the apply changed the copy");
    assert_eq!(hashes(&snapshot), initial, "but never the checkpoint");

    let listing = ok(&w.cfg, "surgeon_checkpoint", json!({"action": "list"}));
    assert_eq!(listing["checkpoints"][0]["label"], "before");
    assert_eq!(listing["checkpoints"][0]["metrics"]["gain"], 0.0);
    assert!(listing["disk_bytes"].as_u64().unwrap() > 0);

    let restored = ok(
        &w.cfg,
        "surgeon_checkpoint",
        json!({"action": "restore", "model": s(&w.copy), "id": id}),
    );
    assert_eq!(restored["sha256"], created["sha256"]);
    assert_eq!(hashes(&w.copy), initial, "byte for byte");
    let backups = std::fs::read_dir(&w.copy)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".bak-")
        })
        .count();
    assert!(backups > 0, "the apply backups stay in the copy");
}

#[test]
fn the_store_is_closed_to_writes_and_the_origin_to_both_tools() {
    let w = workspace(false);
    let created = ok(
        &w.cfg,
        "surgeon_checkpoint",
        json!({"action": "create", "model": s(&w.copy)}),
    );
    let snapshot = w
        .root
        .join("work/checkpoints")
        .join(created["id"].as_str().unwrap())
        .join("model");
    let kind = |tool: &str, args: Value| {
        let (is_error, env) = call(&w.cfg, tool, args);
        assert!(is_error, "{tool}: {env}");
        env["error"]["kind"].as_str().unwrap().to_owned()
    };
    assert_eq!(
        kind(
            "surgeon_apply",
            json!({"model": s(&snapshot), "patch": patch(&w)})
        ),
        "path_not_allowed"
    );
    assert_eq!(
        kind(
            "surgeon_files",
            json!({"action": "write", "path": "checkpoints/notes.json", "content": "{}"})
        ),
        "path_not_allowed"
    );
    let origin = w.root.join("origin").join("qwen2-mini");
    for tool_args in [
        (
            "surgeon_checkpoint",
            json!({"action": "create", "model": s(&origin)}),
        ),
        (
            "surgeon_sweep",
            json!({"model": s(&origin), "vectors": [s(&w.root.join("v.json"))],
                "operation": "subtract", "alphas": [1.0], "layers": ["all"], "max_drift": 1.0,
                "target": {"kind": "unblocking", "dataset": s(&w.root.join("pos.json"))},
                "drift": {"dataset": s(&w.root.join("neg.json"))}}),
        ),
    ] {
        assert_eq!(kind(tool_args.0, tool_args.1), "path_not_allowed");
    }
    assert_eq!(
        kind(
            "surgeon_checkpoint",
            json!({"action": "restore", "model": s(&w.copy), "id": "9999"})
        ),
        "invalid_input"
    );

    let ro = workspace(true);
    let (is_error, env) = call(&ro.cfg, "surgeon_checkpoint", json!({"action": "list"}));
    assert!(!is_error, "{env}");
    let (is_error, env) = call(
        &ro.cfg,
        "surgeon_checkpoint",
        json!({"action": "create", "model": s(&ro.copy)}),
    );
    assert!(is_error && env["error"]["kind"] == "read_only", "{env}");
}

fn sweep_args(w: &Workspace, max_drift: f64) -> Value {
    json!({"model": s(&w.copy), "vectors": [s(&w.root.join("v.json"))],
        "operation": "subtract", "projections": ["down_proj"], "alphas": [0.5, 4.0],
        "layers": ["1-2", "all"], "max_drift": max_drift,
        "target": {"kind": "unblocking", "dataset": s(&w.root.join("pos.json")), "max_new_tokens": 4},
        "drift": {"kind": "kl", "dataset": s(&w.root.join("neg.json")), "max_tokens": 16}})
}

#[test]
fn a_sweep_over_budget_rolls_every_point_back_to_the_base() {
    let w = workspace(false);
    let initial = hashes(&w.copy);

    let report = ok(&w.cfg, "surgeon_sweep", sweep_args(&w, 0.0));

    let points = report["points"].as_array().unwrap();
    assert_eq!(points.len(), 4);
    for p in points {
        assert_eq!(p["decision"], "drift_exceeded", "{p}");
        assert!(p["drift"].as_f64().unwrap() > 0.0);
        assert!(p["checkpoint"].is_null());
    }
    assert_eq!(report["front"], json!([]));
    assert!(report["best"].is_null());
    assert_eq!(report["restored"], report["base_checkpoint"]);
    assert_eq!(hashes(&w.copy), initial, "back to the base byte for byte");
    let listing = ok(&w.cfg, "surgeon_checkpoint", json!({"action": "list"}));
    assert_eq!(
        listing["checkpoints"].as_array().unwrap().len(),
        1,
        "the base only"
    );
    let leftovers: Vec<_> = std::fs::read_dir(w.root.join("work"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(leftovers, ["checkpoints"], "no scratch dumps left");
}

#[test]
fn a_sweep_returns_a_sorted_front_and_ends_on_its_best_checkpoint() {
    let w = workspace(false);

    let report = ok(&w.cfg, "surgeon_sweep", sweep_args(&w, 1e9));

    let points = report["points"].as_array().unwrap();
    let front: Vec<usize> = serde_json::from_value(report["front"].clone()).unwrap();
    let drifts: Vec<f64> = front
        .iter()
        .map(|&i| points[i]["drift"].as_f64().unwrap())
        .collect();
    assert!(drifts.windows(2).all(|d| d[0] <= d[1]), "{drifts:?}");
    for &i in &front {
        assert_eq!(points[i]["decision"], "kept");
        assert!(points[i]["gain"].as_f64().unwrap() > 0.0, "{}", points[i]);
    }
    let listing = ok(&w.cfg, "surgeon_checkpoint", json!({"action": "list"}));
    let ids: Vec<String> = listing["checkpoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["id"].as_str().unwrap().to_owned())
        .collect();
    let kept = points.iter().filter(|p| p["decision"] == "kept").count();
    assert_eq!(ids.len(), 1 + kept);
    let restored = report["restored"].as_str().unwrap();
    match report["best"].as_u64() {
        Some(best) => {
            assert!(front.contains(&(best as usize)));
            assert_eq!(points[best as usize]["checkpoint"], restored);
        }
        None => assert_eq!(restored, report["base_checkpoint"]),
    }
    let snapshot = w.root.join("work/checkpoints").join(restored).join("model");
    assert_eq!(
        hashes(&w.copy),
        hashes(&snapshot),
        "the copy is the chosen checkpoint"
    );
    assert_eq!(
        listing["heads"]
            .as_object()
            .unwrap()
            .values()
            .next()
            .unwrap(),
        restored
    );
}
