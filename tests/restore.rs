//! The rollback cycle over MCP and the CLI: `surgeon_apply` on a working copy, a measured
//! regression (`surgeon_bench` dump + logits), then `surgeon_restore` from the `.bak-*` and
//! from an `--allow-read` origin, each time back to the initial SHA-256 byte for byte. Also
//! the sandbox (never a read-only root, `--read-only`), stacked applies undone one at a time,
//! and what an agent needs around them: working files and the model's chat format.

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

    // Applied again from the restored state: that state is already saved.
    let second = apply(&w);
    let backups = &second["data"]["backups"];
    assert_eq!(backups, &applied["data"]["backups"], "no second copy");

    // Stacked on it: a new checkpoint, and no argument goes back exactly one apply, as often
    // as it is asked.
    let patched = model_hashes(&w.copy);
    let third = apply(&w);
    assert_ne!(&third["data"]["backups"], backups);
    for _ in 0..2 {
        let (is_error, env) = one_call(&w.cfg, "surgeon_restore", json!({"model": s(&w.copy)}));
        assert!(!is_error, "{env}");
        assert_eq!(env["data"]["checkpoint"], "last_apply", "{env}");
        assert_restored(&env, &w.copy, &patched);
    }

    // The backups named by the apply report.
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
    // A dry run writes nothing: it stays available.
    let patch = json!({"schema_version": 1, "operations": [{
        "id": "push", "layers": "all", "projections": ["down_proj"], "operation": "subtract",
        "vector_file": s(&ro.root.join("v.json")), "alpha": 1.0, "axis": "cols"}]});
    let (is_error, env) = one_call(
        &ro.cfg,
        "surgeon_apply",
        json!({"model": s(&ro.copy), "patch": patch, "dry_run": true}),
    );
    assert!(!is_error, "{env}");
    let (is_error, env) = one_call(
        &ro.cfg,
        "surgeon_apply",
        json!({"model": s(&ro.copy), "patch": patch}),
    );
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "read_only", "{env}");
}

#[test]
fn working_files_are_written_under_work_and_feed_the_tools() {
    let w = workspace(false);
    let files = |args: Value| one_call(&w.cfg, "surgeon_files", args);
    let lines = ["joie soleil", "rire fete", "danse musique"]
        .map(|p| json!({ "text": p }).to_string())
        .join("\n");
    let (is_error, env) =
        files(json!({"action": "write", "path": "sets/pos.jsonl", "content": lines}));
    assert!(!is_error, "{env}");
    let written = PathBuf::from(env["data"]["path"].as_str().unwrap());
    let in_work = written.canonicalize().unwrap();
    assert!(in_work.starts_with(w.root.join("work")), "{env}");

    // The file is a dataset for the engine, and comes back as text.
    let (is_error, env) = one_call(
        &w.cfg,
        "surgeon_extract",
        json!({"model": s(&w.copy), "layer": 2, "pos_prompts_file": s(&written),
            "neg_prompts_file": s(&w.root.join("neg.json")),
            "out": s(&w.root.join("vectors/new/d.safetensors"))}),
    );
    assert!(!is_error, "the output folder is created: {env}");
    let (is_error, env) = files(json!({"action": "read", "path": s(&written)}));
    assert!(!is_error, "{env}");
    assert_eq!(env["data"]["content"], lines);

    // An existing file needs a decision; append and overwrite are that decision.
    let again = json!({"action": "write", "path": "sets/pos.jsonl", "content": "x"});
    let (is_error, env) = files(again.clone());
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "output_exists", "{env}");
    let mut append = again.clone();
    append["append"] = json!(true);
    assert!(!files(append).0);
    assert!(std::fs::read_to_string(&written).unwrap().ends_with('x'));
    let mut overwrite = again;
    overwrite["overwrite"] = json!(true);
    assert!(!files(overwrite).0);
    assert_eq!(std::fs::read_to_string(&written).unwrap(), "x");

    // Never outside work/, never anything but text files.
    let before = model_hashes(&w.copy);
    for (path, kind) in [
        (
            s(&w.copy.join("config.json")).to_owned(),
            "path_not_allowed",
        ),
        ("../project.surgeon.json".to_owned(), "path_not_allowed"),
        ("poids.safetensors".to_owned(), "invalid_arguments"),
    ] {
        let (is_error, env) = files(json!({"action": "write", "path": path, "content": "{}"}));
        assert!(is_error, "{path}");
        assert_eq!(env["error"]["kind"], kind, "{env}");
    }
    assert_eq!(model_hashes(&w.copy), before);

    // Weights are not text, and a read-only root can be listed to find models.
    let weights = w.copy.join("model.safetensors");
    let (is_error, env) = files(json!({"action": "read", "path": s(&weights)}));
    assert!(is_error, "{env}");
    let (is_error, env) = files(json!({"action": "list", "path": s(&w.root.join("origin"))}));
    assert!(!is_error, "{env}");
    let entries = env["data"]["entries"].as_array().unwrap();
    assert!(entries
        .iter()
        .any(|e| e["name"] == "qwen2-mini" && e["model"] == true));
    let outside = tempfile::tempdir().unwrap();
    let (is_error, env) = files(json!({"action": "list", "path": s(outside.path())}));
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "path_not_allowed", "{env}");

    // A read-only server reads but never writes.
    let ro = workspace(true);
    let (is_error, env) = one_call(
        &ro.cfg,
        "surgeon_files",
        json!({"action": "write", "path": "a.txt", "content": "x"}),
    );
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "read_only", "{env}");
    assert!(!one_call(&ro.cfg, "surgeon_files", json!({"action": "list"})).0);
}

#[test]
fn the_chat_format_of_the_model_reaches_extract_autotune_and_the_measures() {
    let w = workspace(false);
    let system = "You are helpful.";
    std::fs::write(
        w.copy.join("tokenizer_config.json"),
        json!({ "chat_template": format!(
            "{{%- if m %}}{{{{- '<|im_start|>system\\n{system}<|im_end|>\\n' }}}}{{%- endif %}}"
        )})
        .to_string(),
    )
    .unwrap();
    let dump = |extra: Value, out: &str| {
        let mut args = json!({"mode": "dump", "model": s(&w.copy),
            "prompts": s(&w.root.join("pos.json")), "out": s(&w.root.join(out))});
        args.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let (is_error, env) = one_call(&w.cfg, "surgeon_bench", args);
        assert!(!is_error, "{env}");
        env["data"].clone()
    };
    let native = dump(json!({}), "a.safetensors");
    assert_eq!(native["chat_template"], "chatml");
    // Disabled on request only.
    let raw = dump(json!({"chat_template": "none"}), "b.safetensors");
    assert!(raw.get("chat_template").is_none(), "{raw}");
    assert!(native["n_positions"].as_u64() > raw["n_positions"].as_u64());

    // The vector remembers the wrapping, system message included, and autotune replays it.
    let vector = w.root.join("d.safetensors");
    let (is_error, env) = one_call(
        &w.cfg,
        "surgeon_extract",
        json!({"model": s(&w.copy), "layer": 2, "pos_prompts_file": s(&w.root.join("pos.json")),
            "neg_prompts_file": s(&w.root.join("neg.json")), "out": s(&vector)}),
    );
    assert!(!is_error, "{env}");
    let (is_error, env) = one_call(
        &w.cfg,
        "surgeon_autotune",
        json!({"model": s(&w.copy), "vector": s(&vector), "target_layer_range": "1-2",
            "eval_dataset": s(&w.root.join("pos.json")), "max_delta_norm_ratio": 1e-4}),
    );
    assert!(!is_error, "{env}");
    assert_eq!(env["data"]["chat_template"], "chatml", "{env}");
    assert_eq!(env["data"]["chat_system"], system, "{env}");
    // The ratio bound holds on every kept layer.
    for layer in env["data"]["alphas"].as_array().unwrap() {
        let ratio = layer["delta_norm_ratio"].as_f64().unwrap();
        assert!(ratio <= 1e-4 * (1.0 + 1e-9), "{layer}");
    }
}
