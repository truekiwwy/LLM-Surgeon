//! MCP tools `surgeon_diff` and `surgeon_lora`: tools/list and schemas, sandbox, `--read-only`,
//! per-mode validation, then the end-to-end tests on the real diff and LoRA engines
//! (the progress test relies on `McpProgress` handling `SurgeryEvent`, wired in `server.rs`).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};
use weight_surgeon::app::{self, AppDeps, ProgressEvent, ProgressSink};
use weight_surgeon::cli::SchemaKind;
use weight_surgeon::eval::{CommandOutput, CommandRunner, EvalError};
use weight_surgeon::hardware::{GpuInfo, HardwareProbe, MemoryInfo};
use weight_surgeon::lora::cli_args::{LoraArgs, LoraCommand, LoraMergeArgs};
use weight_surgeon::mcp::{self, McpConfig};
use weight_surgeon::surgery::events::SurgeryEvent;

// ------------------------------------------------------------------ harness

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

fn rpc(id: i64, method: &str, params: Value) -> String {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string()
}

fn init() -> String {
    rpc(
        0,
        "initialize",
        json!({"protocolVersion": "2025-06-18", "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"}}),
    )
}

fn call(id: i64, tool: &str, arguments: Value) -> String {
    rpc(
        id,
        "tools/call",
        json!({"name": tool, "arguments": arguments}),
    )
}

fn call_with_token(id: i64, tool: &str, arguments: Value, token: &str) -> String {
    rpc(
        id,
        "tools/call",
        json!({"name": tool, "arguments": arguments, "_meta": {"progressToken": token}}),
    )
}

fn cfg(dir: &Path) -> McpConfig {
    McpConfig::new(&[dir.to_path_buf()], false, false).unwrap()
}

/// Every stdout line must be one JSON-RPC 2.0 message (nothing else on stdout).
fn session(deps: &AppDeps, cfg: &McpConfig, lines: &[String]) -> Vec<Value> {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let input = (lines.join("\n") + "\n").into_bytes();
    let code = mcp::serve(std::io::Cursor::new(input), &mut out, &mut err, deps, cfg);
    assert_eq!(code, 0);
    String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|line| {
            let v: Value = serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"));
            assert_eq!(v["jsonrpc"], "2.0");
            v
        })
        .collect()
}

/// `(isError, envelope)` of the reply to request `id`.
fn envelope(responses: &[Value], id: i64) -> (bool, Value) {
    let reply = responses
        .iter()
        .find(|r| r["id"] == id && r.get("result").is_some())
        .unwrap_or_else(|| panic!("no result for {id} in {responses:?}"));
    let result = &reply["result"];
    let text: Value = serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(text, result["structuredContent"]);
    (
        result["isError"].as_bool().unwrap(),
        result["structuredContent"].clone(),
    )
}

fn one_call(config: &McpConfig, tool: &str, arguments: Value) -> (bool, Value) {
    let responses = session(&deps(), config, &[init(), call(1, tool, arguments)]);
    envelope(&responses, 1)
}

fn tools_list(dir: &Path) -> Vec<Value> {
    let responses = session(
        &deps(),
        &cfg(dir),
        &[init(), rpc(1, "tools/list", json!({}))],
    );
    responses.iter().find(|r| r["id"] == 1).unwrap()["result"]["tools"]
        .as_array()
        .unwrap()
        .clone()
}

// ------------------------------------------------------------------ tools/list and schemas

#[test]
fn tools_list_has_ten_tools_and_the_two_p13_schemas_are_valid() {
    let dir = tempfile::tempdir().unwrap();
    let tools = tools_list(dir.path());
    assert_eq!(tools.len(), 10);
    let by_name = |name: &str| tools.iter().find(|t| t["name"] == name).unwrap();
    for name in ["surgeon_diff", "surgeon_lora"] {
        let tool = by_name(name);
        let schema = &tool["inputSchema"];
        assert_eq!(schema["type"], "object");
        assert!(jsonschema::draft7::meta::is_valid(schema), "{name}");
        assert!(tool["description"].as_str().unwrap().len() > 20);
        assert!(tool["annotations"]["readOnlyHint"].is_boolean());
        assert_eq!(tool["annotations"]["openWorldHint"], false);
        assert!(
            !schema.to_string().contains("$ref"),
            "{name}: subschemas are inlined"
        );
    }
    let keys = |name: &str| -> Vec<String> {
        by_name(name)["inputSchema"]["properties"]
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect()
    };
    let mut diff = keys("surgeon_diff");
    diff.sort();
    assert_eq!(
        diff,
        [
            "base",
            "exclude",
            "factor_dtype",
            "include",
            "include_embeddings",
            "max_rel_error",
            "on_high_error",
            "out",
            "overwrite",
            "rank",
            "seed",
            "tuned"
        ]
    );
    assert_eq!(
        by_name("surgeon_diff")["inputSchema"]["required"],
        json!(["base", "out", "tuned"])
    );
    assert_eq!(
        by_name("surgeon_diff")["inputSchema"]["properties"]["on_high_error"]["enum"],
        json!(["warn", "skip", "full"])
    );
    assert_eq!(
        by_name("surgeon_diff")["annotations"]["readOnlyHint"],
        false
    );
    assert_eq!(
        by_name("surgeon_diff")["annotations"]["destructiveHint"],
        false
    );

    let mut lora = keys("surgeon_lora");
    lora.sort();
    assert_eq!(
        lora,
        [
            "adapter_config",
            "alpha",
            "backup_dir",
            "base",
            "dry_run",
            "dtype",
            "lora",
            "mode",
            "model",
            "out",
            "overwrite",
            "patched",
            "rank",
            "scale",
            "seed",
            "target_modules"
        ]
    );
    let lora_schema = &by_name("surgeon_lora")["inputSchema"];
    assert_eq!(lora_schema["required"], json!(["mode"]));
    assert_eq!(
        lora_schema["properties"]["mode"]["enum"],
        json!(["merge", "extract"])
    );
    assert!(
        lora_schema.get("oneOf").is_none(),
        "flat schema, validated at runtime"
    );
    assert_eq!(
        by_name("surgeon_lora")["annotations"]["destructiveHint"],
        true
    );
}

#[test]
fn schema_kind_tools_equals_tools_list_for_the_new_tools() {
    let dir = tempfile::tempdir().unwrap();
    let exported: Value = serde_json::from_str(&mcp::render_schema(SchemaKind::Tools)).unwrap();
    assert_eq!(exported, Value::Array(tools_list(dir.path())));
    let names: Vec<&str> = exported
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        &names[7..],
        ["surgeon_diff", "surgeon_lora", "surgeon_pipeline"]
    );
}

// ------------------------------------------------------------------ sandbox, read-only

fn outside_path(outside: &Path, name: &str) -> String {
    outside.join(name).display().to_string()
}

#[test]
fn every_path_argument_is_sandboxed() {
    let allowed = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let out = |name: &str| outside_path(outside.path(), name);
    let inside = json!("ok.safetensors");
    let cases = [
        (
            "surgeon_diff",
            json!({"base": out("b"), "tuned": inside, "out": "o.surgery"}),
            "base",
        ),
        (
            "surgeon_diff",
            json!({"base": inside, "tuned": out("t"), "out": "o.surgery"}),
            "tuned",
        ),
        (
            "surgeon_diff",
            json!({"base": inside, "tuned": inside, "out": out("o.surgery")}),
            "out",
        ),
        (
            "surgeon_lora",
            json!({"mode": "merge", "model": out("m"), "lora": inside}),
            "model",
        ),
        (
            "surgeon_lora",
            json!({"mode": "merge", "model": inside, "lora": out("l")}),
            "lora",
        ),
        (
            "surgeon_lora",
            json!({"mode": "merge", "model": inside, "lora": inside, "adapter_config": out("c.json")}),
            "adapter_config",
        ),
        (
            "surgeon_lora",
            json!({"mode": "merge", "model": inside, "lora": inside, "backup_dir": out("bk")}),
            "backup_dir",
        ),
        (
            "surgeon_lora",
            json!({"mode": "extract", "base": out("b"), "patched": inside, "out": "a.safetensors"}),
            "base",
        ),
        (
            "surgeon_lora",
            json!({"mode": "extract", "base": inside, "patched": out("p"), "out": "a.safetensors"}),
            "patched",
        ),
        (
            "surgeon_lora",
            json!({"mode": "extract", "base": inside, "patched": inside, "out": out("a.safetensors")}),
            "out",
        ),
    ];
    let config = cfg(allowed.path());
    for (i, (tool, arguments, field)) in cases.into_iter().enumerate() {
        let (is_error, env) = one_call(&config, tool, arguments);
        assert!(is_error, "case {i}");
        assert_eq!(env["error"]["kind"], "path_not_allowed", "case {i}: {env}");
        let message = env["error"]["message"].as_str().unwrap();
        assert!(message.contains(field), "case {i}: {message}");
        assert!(
            !message.contains(&outside.path().display().to_string()),
            "case {i} leaks the path: {message}"
        );
    }
    assert!(std::fs::read_dir(outside.path()).unwrap().next().is_none());
}

#[test]
fn read_only_refuses_both_tools_before_touching_anything() {
    let dir = tempfile::tempdir().unwrap();
    let config = McpConfig::new(&[dir.path().to_path_buf()], true, false).unwrap();
    for (tool, arguments) in [
        (
            "surgeon_diff",
            json!({"base": "b.safetensors", "tuned": "t.safetensors", "out": "o.surgery"}),
        ),
        (
            "surgeon_lora",
            json!({"mode": "merge", "model": "m.safetensors", "lora": "l.safetensors"}),
        ),
        (
            "surgeon_lora",
            json!({"mode": "extract", "base": "b.safetensors", "patched": "p.safetensors", "out": "a.safetensors"}),
        ),
    ] {
        let (is_error, env) = one_call(&config, tool, arguments);
        assert!(is_error);
        assert_eq!(env["error"]["kind"], "read_only", "{tool}: {env}");
    }
    assert!(std::fs::read_dir(dir.path()).unwrap().next().is_none());
    // The read-only tools keep working.
    let responses = session(&deps(), &config, &[init(), rpc(1, "tools/list", json!({}))]);
    assert!(responses
        .iter()
        .any(|r| r["id"] == 1 && r.get("result").is_some()));
}

// ------------------------------------------------------------------ argument validation

fn issue_paths(env: &Value) -> Vec<String> {
    let mut paths: Vec<String> = env["error"]["issues"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|i| i["path"].as_str().unwrap().into())
                .collect()
        })
        .unwrap_or_default();
    paths.sort();
    paths
}

#[test]
fn lora_fields_of_the_other_mode_or_missing_ones_are_invalid_arguments() {
    let dir = tempfile::tempdir().unwrap();
    let config = cfg(dir.path());
    let cases = [
        (
            json!({"mode": "merge", "model": "m", "lora": "l", "out": "o"}),
            vec!["out"],
        ),
        (
            json!({"mode": "merge", "model": "m", "lora": "l", "rank": 4, "dtype": "f32", "overwrite": true}),
            vec!["dtype", "overwrite", "rank"],
        ),
        (json!({"mode": "merge", "model": "m"}), vec!["lora"]),
        (json!({"mode": "merge"}), vec!["lora", "model"]),
        (
            json!({"mode": "extract", "base": "b", "patched": "p"}),
            vec!["out"],
        ),
        (
            json!({"mode": "extract", "base": "b", "patched": "p", "out": "o", "model": "m", "dry_run": false}),
            vec!["dry_run", "model"],
        ),
        (
            json!({"mode": "extract", "base": "b", "patched": "p", "out": "o", "backup_dir": "x", "alpha": 1.0}),
            vec!["alpha", "backup_dir"],
        ),
    ];
    for (i, (arguments, expected)) in cases.into_iter().enumerate() {
        let (is_error, env) = one_call(&config, "surgeon_lora", arguments);
        assert!(is_error, "case {i}");
        assert_eq!(env["error"]["kind"], "invalid_arguments", "case {i}: {env}");
        assert_eq!(env["exit_code"], 2, "case {i}");
        assert_eq!(issue_paths(&env), expected, "case {i}: {env}");
    }
}

#[test]
fn malformed_arguments_are_invalid_arguments_with_the_command_name() {
    let dir = tempfile::tempdir().unwrap();
    let config = cfg(dir.path());
    let cases = [
        ("surgeon_diff", json!({"base": "b", "tuned": "t"}), "diff"),
        (
            "surgeon_diff",
            json!({"base": "b", "tuned": "t", "out": "o", "rank": "sixteen"}),
            "diff",
        ),
        (
            "surgeon_diff",
            json!({"base": "b", "tuned": "t", "out": "o", "timestamp": true}),
            "diff",
        ),
        (
            "surgeon_diff",
            json!({"base": "b", "tuned": "t", "out": "o", "on_high_error": "explode"}),
            "diff",
        ),
        (
            "surgeon_diff",
            json!({"base": "b", "tuned": "t", "out": "o", "factor_dtype": "f64"}),
            "diff",
        ),
        ("surgeon_lora", json!({}), "lora"),
        ("surgeon_lora", json!({"mode": "split"}), "lora"),
        (
            "surgeon_lora",
            json!({"mode": "merge", "model": "m", "lora": "l", "no_backup": true}),
            "lora merge",
        ),
        (
            "surgeon_lora",
            json!({"mode": "extract", "base": "b", "patched": "p", "out": "o", "rank": -1}),
            "lora extract",
        ),
    ];
    for (i, (tool, arguments, command)) in cases.into_iter().enumerate() {
        let (is_error, env) = one_call(&config, tool, arguments);
        assert!(is_error, "case {i}");
        assert_eq!(env["error"]["kind"], "invalid_arguments", "case {i}: {env}");
        assert_eq!(env["command"], command, "case {i}");
        assert_eq!(env["exit_code"], 2, "case {i}");
    }
}

#[test]
fn valid_arguments_reach_the_engine() {
    // Inputs missing inside the sandbox: whatever the engine answers, it is not the
    // wrapper refusing (sandbox, read-only) and the envelope names the right command.
    let dir = tempfile::tempdir().unwrap();
    let config = cfg(dir.path());
    let cases = [
        (
            "surgeon_diff",
            json!({"base": "b.safetensors", "tuned": "t.safetensors", "out": "o.surgery",
                   "rank": 4, "factor_dtype": "bf16", "include": "q_proj", "exclude": "mlp",
                   "include_embeddings": true, "max_rel_error": 0.5, "on_high_error": "full",
                   "seed": 7, "overwrite": true}),
            "diff",
        ),
        (
            "surgeon_lora",
            json!({"mode": "merge", "model": "m.safetensors", "lora": "l.safetensors",
                   "alpha": 8.0, "adapter_config": "c.json", "scale": 0.5, "dry_run": true,
                   "backup_dir": "bk"}),
            "lora merge",
        ),
        (
            "surgeon_lora",
            json!({"mode": "extract", "base": "b.safetensors", "patched": "p.safetensors",
                   "out": "adapter/adapter_model.safetensors", "rank": 4, "dtype": "bf16",
                   "target_modules": "q_proj", "seed": 1, "overwrite": true}),
            "lora extract",
        ),
    ];
    for (tool, arguments, command) in cases {
        let (is_error, env) = one_call(&config, tool, arguments);
        assert!(is_error, "{tool}: the inputs do not exist");
        assert_eq!(env["command"], command);
        let kind = env["error"]["kind"].as_str().unwrap_or_default();
        assert!(
            kind != "path_not_allowed" && kind != "read_only",
            "{tool}: {env}"
        );
    }
    assert!(
        std::fs::read_dir(dir.path()).unwrap().next().is_none(),
        "nothing written"
    );
}

// ------------------------------------------------------------------ fixtures for the e2e tests

const Q0: &str = "model.layers.0.self_attn.q_proj.weight";
const Q1: &str = "model.layers.1.self_attn.q_proj.weight";
const ROWS: usize = 16;
const COLS: usize = 12;

/// Deterministic values in eighths, so f32 arithmetic is easy to reason about.
fn eighths(seed: usize, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 37 + seed * 11) % 17) as f32 - 8.0) / 8.0)
        .collect()
}

fn write_st(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        values
            .iter()
            .for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
        header.insert(
            (*name).into(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = serde_json::to_string(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

/// F32 tensors of a SafeTensors file, read independently of the engine.
fn read_st(path: &Path) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path).unwrap();
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
    header
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| *name != "__metadata__")
        .map(|(name, info)| {
            let (start, end) = (
                info["data_offsets"][0].as_u64().unwrap() as usize,
                info["data_offsets"][1].as_u64().unwrap() as usize,
            );
            let shape = info["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|d| d.as_u64().unwrap() as usize)
                .collect();
            let values = bytes[8 + len + start..8 + len + end]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            (name.clone(), (shape, values))
        })
        .collect()
}

fn outer(u: &[f32], v: &[f32]) -> Vec<f32> {
    u.iter()
        .flat_map(|a| v.iter().map(move |b| a * b))
        .collect()
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

/// `base.safetensors` and `tuned.safetensors` (tuned = base + a rank 1 delta per tensor).
fn checkpoints(dir: &Path) -> HashMap<&'static str, Vec<f32>> {
    let base: Vec<(&str, Vec<f32>)> =
        vec![(Q0, eighths(1, ROWS * COLS)), (Q1, eighths(2, ROWS * COLS))];
    let delta = |seed: usize| outer(&eighths(seed, ROWS), &eighths(seed + 5, COLS));
    let tuned: Vec<(&str, Vec<f32>)> = base
        .iter()
        .zip([3, 4])
        .map(|((name, w), seed)| (*name, add(w, &delta(seed))))
        .collect();
    write_st(&dir.join("base.safetensors"), &with_shape(&base));
    write_st(&dir.join("tuned.safetensors"), &with_shape(&tuned));
    tuned.into_iter().collect()
}

/// PEFT adapter of rank 2 on both layers, `lora_alpha` 4 (effective scale 2).
fn adapter(dir: &Path) -> HashMap<&'static str, Vec<f32>> {
    let mut tensors = Vec::new();
    let mut expected_delta = HashMap::new();
    for (layer, (name, seed)) in [(Q0, 5), (Q1, 6)].into_iter().enumerate() {
        let a = eighths(seed, 2 * COLS);
        let b = eighths(seed + 1, ROWS * 2);
        let mut ba = vec![0.0f32; ROWS * COLS];
        for i in 0..ROWS {
            for j in 0..COLS {
                ba[i * COLS + j] =
                    (0..2).map(|k| b[i * 2 + k] * a[k * COLS + j]).sum::<f32>() * 2.0;
            }
        }
        expected_delta.insert(name, ba);
        let prefix = format!("base_model.model.model.layers.{layer}.self_attn.q_proj");
        tensors.push((format!("{prefix}.lora_A.weight"), vec![2, COLS], a));
        tensors.push((format!("{prefix}.lora_B.weight"), vec![ROWS, 2], b));
    }
    let refs: Vec<(&str, Vec<usize>, Vec<f32>)> = tensors
        .iter()
        .map(|(n, s, v)| (n.as_str(), s.clone(), v.clone()))
        .collect();
    write_st(&dir.join("adapter_model.safetensors"), &refs);
    std::fs::write(
        dir.join("adapter_config.json"),
        json!({"peft_type": "LORA", "r": 2, "lora_alpha": 4,
               "target_modules": ["q_proj"], "task_type": "CAUSAL_LM"})
        .to_string(),
    )
    .unwrap();
    expected_delta
}

fn backups_in(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.to_string_lossy().contains(".bak-"))
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn progress_of<'a>(responses: &'a [Value], token: &str) -> Vec<&'a Value> {
    responses
        .iter()
        .filter(|r| {
            r["method"] == "notifications/progress" && r["params"]["progressToken"] == token
        })
        .collect()
}

/// Notifications are monotonic, end at `total`, carry a message, and have no id.
fn assert_progress_shape(progress: &[&Value]) {
    assert!(!progress.is_empty(), "no notifications/progress");
    let mut last = 0;
    for p in progress {
        let done = p["params"]["progress"].as_u64().unwrap();
        assert!(done >= last && done <= p["params"]["total"].as_u64().unwrap());
        assert!(p["params"]["message"].is_string());
        assert!(p.get("id").is_none());
        last = done;
    }
    let tail = progress[progress.len() - 1];
    assert_eq!(tail["params"]["progress"], tail["params"]["total"]);
}

fn cli(deps: &AppDeps, args: &[&str]) -> (i32, Value) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let argv = ["weight-surgeon", "--json"]
        .into_iter()
        .chain(args.iter().copied());
    let code = match app::parse_cli(argv, &mut out, &mut err) {
        Ok(parsed) => app::run_with(parsed, deps, &mut out, &mut err),
        Err(code) => code,
    };
    let out = String::from_utf8(out).unwrap();
    (
        code,
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}")),
    )
}

// ------------------------------------------------------------------ e2e (need the engines)

#[test]
fn diff_then_cli_apply_reproduces_the_tuned_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let tuned = checkpoints(dir.path());
    let responses = session(
        &deps(),
        &cfg(dir.path()),
        &[
            init(),
            call(
                1,
                "surgeon_diff",
                json!({"base": "base.safetensors", "tuned": "tuned.safetensors",
                       "out": "d.surgery", "rank": 2, "factor_dtype": "f32"}),
            ),
        ],
    );
    let (is_error, env) = envelope(&responses, 1);
    assert!(!is_error, "{env}");
    assert_eq!(env["command"], "diff");
    assert!(env["data"]["surgery"].is_object(), "{env}");
    assert_eq!(
        &std::fs::read(dir.path().join("d.surgery")).unwrap()[..4],
        b"SURG"
    );

    // The patch is an approximation, exact here because the delta has rank 1 <= 2.
    let copy = dir.path().join("copy.safetensors");
    std::fs::copy(dir.path().join("base.safetensors"), &copy).unwrap();
    let (code, v) = cli(
        &deps(),
        &[
            "apply",
            "--model",
            copy.to_str().unwrap(),
            "--patch",
            dir.path().join("d.surgery").to_str().unwrap(),
            "--no-backup",
        ],
    );
    assert_eq!(code, 0, "{v}");
    let patched = read_st(&copy);
    for (name, expected) in &tuned {
        assert!(max_abs_diff(&patched[*name].1, expected) < 1e-4, "{name}");
    }

    // An existing out is refused without overwrite, then replaced with it.
    let again = |extra: Value| {
        let mut arguments =
            json!({"base": "base.safetensors", "tuned": "tuned.safetensors", "out": "d.surgery"});
        arguments
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        one_call(&cfg(dir.path()), "surgeon_diff", arguments)
    };
    let (is_error, env) = again(json!({}));
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "invalid_arguments", "{env}");
    assert!(!again(json!({"overwrite": true})).0);
}

#[test]
fn lora_merge_adds_scaled_delta_and_always_backs_up() {
    let dir = tempfile::tempdir().unwrap();
    checkpoints(dir.path());
    let delta = adapter(dir.path());
    let model = dir.path().join("base.safetensors");
    let original = std::fs::read(&model).unwrap();
    let before = read_st(&model);

    let (is_error, env) = one_call(
        &cfg(dir.path()),
        "surgeon_lora",
        json!({"mode": "merge", "model": "base.safetensors", "lora": "adapter_model.safetensors", "dry_run": true}),
    );
    assert!(!is_error, "{env}");
    assert_eq!(
        std::fs::read(&model).unwrap(),
        original,
        "dry run writes nothing"
    );
    assert!(backups_in(dir.path()).is_empty());

    // Even an operator-enabled --allow-no-backup does not let the tool skip the backup.
    let permissive = McpConfig::new(&[dir.path().to_path_buf()], false, true).unwrap();
    let (is_error, env) = one_call(
        &permissive,
        "surgeon_lora",
        json!({"mode": "merge", "model": "base.safetensors", "lora": "adapter_model.safetensors"}),
    );
    assert!(!is_error, "{env}");
    assert_eq!(env["command"], "lora merge");
    assert!(env["data"]["lora"].is_object(), "{env}");
    let backups = backups_in(dir.path());
    assert_eq!(backups.len(), 1, "{backups:?}");
    assert_eq!(std::fs::read(&backups[0]).unwrap(), original);
    let after = read_st(&model);
    for name in [Q0, Q1] {
        let expected = add(&before[name].1, &delta[name]);
        assert!(max_abs_diff(&after[name].1, &expected) < 1e-5, "{name}");
    }
}

#[test]
fn lora_extract_writes_a_peft_adapter_that_rebuilds_the_delta() {
    let dir = tempfile::tempdir().unwrap();
    let tuned = checkpoints(dir.path());
    let base = read_st(&dir.path().join("base.safetensors"));
    std::fs::create_dir(dir.path().join("out")).unwrap();
    let (is_error, env) = one_call(
        &cfg(dir.path()),
        "surgeon_lora",
        json!({"mode": "extract", "base": "base.safetensors", "patched": "tuned.safetensors",
               "out": "out/adapter_model.safetensors", "rank": 2}),
    );
    assert!(!is_error, "{env}");
    assert_eq!(env["command"], "lora extract");
    assert!(env["data"]["lora"].is_object(), "{env}");
    let config: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("out/adapter_config.json")).unwrap())
            .unwrap();
    assert_eq!(config["peft_type"], "LORA");
    assert_eq!(config["r"], 2);

    let adapter = read_st(&dir.path().join("out/adapter_model.safetensors"));
    for (layer, name) in [Q0, Q1].into_iter().enumerate() {
        let key = |side: &str| {
            adapter
                .iter()
                .find(|(k, _)| {
                    k.contains(&format!("layers.{layer}."))
                        && k.ends_with(&format!("lora_{side}.weight"))
                })
                .unwrap_or_else(|| panic!("no lora_{side} for layer {layer}"))
                .1
        };
        let (a, b) = (&key("A").1, &key("B").1);
        let alpha_over_r = config["lora_alpha"].as_f64().unwrap() as f32 / 2.0;
        let rebuilt: Vec<f32> = (0..ROWS * COLS)
            .map(|idx| {
                let (i, j) = (idx / COLS, idx % COLS);
                (0..2).map(|k| b[i * 2 + k] * a[k * COLS + j]).sum::<f32>() * alpha_over_r
            })
            .collect();
        let expected: Vec<f32> = tuned[name]
            .iter()
            .zip(&base[name].1)
            .map(|(t, w)| t - w)
            .collect();
        assert!(max_abs_diff(&rebuilt, &expected) < 1e-3, "{name}");
    }
}

#[test]
fn progress_notifications_follow_the_token_for_diff_and_merge() {
    let dir = tempfile::tempdir().unwrap();
    checkpoints(dir.path());
    adapter(dir.path());
    let responses = session(
        &deps(),
        &cfg(dir.path()),
        &[
            init(),
            call_with_token(
                1,
                "surgeon_diff",
                json!({"base": "base.safetensors", "tuned": "tuned.safetensors", "out": "d.surgery", "rank": 2}),
                "diff-tok",
            ),
            call_with_token(
                2,
                "surgeon_lora",
                json!({"mode": "merge", "model": "base.safetensors", "lora": "adapter_model.safetensors"}),
                "merge-tok",
            ),
            call(
                3,
                "surgeon_diff",
                json!({"base": "base.safetensors", "tuned": "tuned.safetensors", "out": "d2.surgery", "rank": 2}),
            ),
        ],
    );
    assert!(!envelope(&responses, 1).0);
    assert!(!envelope(&responses, 2).0);
    assert_progress_shape(&progress_of(&responses, "diff-tok"));
    assert_progress_shape(&progress_of(&responses, "merge-tok"));
    let all = responses
        .iter()
        .filter(|r| r["method"] == "notifications/progress")
        .count();
    assert_eq!(
        all,
        progress_of(&responses, "diff-tok").len() + progress_of(&responses, "merge-tok").len(),
        "none for the call without a token"
    );
}

#[test]
fn cancellation_queued_before_the_call_leaves_everything_untouched() {
    let dir = tempfile::tempdir().unwrap();
    checkpoints(dir.path());
    adapter(dir.path());
    let model = dir.path().join("base.safetensors");
    let original = std::fs::read(&model).unwrap();
    let cancel = |id: i64| {
        json!({"jsonrpc": "2.0", "method": "notifications/cancelled",
               "params": {"requestId": id, "reason": "test"}})
        .to_string()
    };
    let responses = session(
        &deps(),
        &cfg(dir.path()),
        &[
            init(),
            cancel(1),
            call(
                1,
                "surgeon_lora",
                json!({"mode": "merge", "model": "base.safetensors", "lora": "adapter_model.safetensors"}),
            ),
            cancel(2),
            call(
                2,
                "surgeon_diff",
                json!({"base": "base.safetensors", "tuned": "tuned.safetensors", "out": "d.surgery"}),
            ),
        ],
    );
    for id in [1, 2] {
        let (is_error, env) = envelope(&responses, id);
        assert!(is_error);
        assert_eq!(env["error"]["kind"], "cancelled", "{env}");
    }
    assert_eq!(std::fs::read(&model).unwrap(), original);
    assert!(
        !dir.path().join("d.surgery").exists(),
        "no partial .surgery"
    );
}

/// Raises the cancellation flag after the first merged tensor, as the reader thread of the
/// server does when `notifications/cancelled` arrives mid-call.
struct CancelAfterFirstTensor<'a> {
    flag: &'a AtomicBool,
    seen: usize,
}

impl ProgressSink for CancelAfterFirstTensor<'_> {
    fn event(&mut self, event: ProgressEvent) {
        if let ProgressEvent::Surgery(SurgeryEvent::Tensor { .. }) = event {
            self.seen += 1;
            self.flag.store(true, Ordering::SeqCst);
        }
    }
}

#[test]
fn cancellation_in_the_middle_of_a_merge_rolls_back_to_identical_bytes() {
    let dir = tempfile::tempdir().unwrap();
    checkpoints(dir.path());
    adapter(dir.path());
    let model = dir.path().join("base.safetensors");
    let original = std::fs::read(&model).unwrap();
    let deps = deps();
    let args = LoraArgs {
        command: LoraCommand::Merge(LoraMergeArgs {
            model: model.clone(),
            lora: dir.path().join("adapter_model.safetensors"),
            alpha: None,
            adapter_config: None,
            scale: 1.0,
            dry_run: false,
            no_backup: false,
            backup_dir: None,
        }),
    };
    let mut sink = CancelAfterFirstTensor {
        flag: &deps.interrupted,
        seen: 0,
    };
    let failure = weight_surgeon::lora::core_lora(&args, &deps, &mut sink).unwrap_err();
    assert!(!failure.is_ok());
    assert!(sink.seen >= 1, "the engine reports each tensor");
    assert_eq!(std::fs::read(&model).unwrap(), original, "rolled back");
}

fn with_shape<'a>(list: &[(&'a str, Vec<f32>)]) -> Vec<(&'a str, Vec<usize>, Vec<f32>)> {
    list.iter()
        .map(|(n, v)| (*n, vec![ROWS, COLS], v.clone()))
        .collect()
}
