//! `surgeon graph`: SafeTensors and GGUF, deterministic output, causal adjacent edges,
//! certified metrics consistency, cache/resume, cancellation, safe writes and the MCP
//! `surgeon_graph` contract (bounded inline summary, `out_dir`, `--read-only`).

#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

#[path = "support/gguf.rs"]
mod support;

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

use support::{gguf_bytes, tensor_data, Kv, Rng, F32, T};

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

fn run(deps: &AppDeps, args: &[&str]) -> (i32, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let argv = std::iter::once("weight-surgeon").chain(args.iter().copied());
    let code = match app::parse_cli(argv, &mut out, &mut err) {
        Ok(cli) => app::run_with(cli, deps, &mut out, &mut err),
        Err(code) => code,
    };
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

fn run_json(deps: &AppDeps, args: &[&str]) -> (i32, Value) {
    let mut argv = vec!["--json"];
    argv.extend_from_slice(args);
    let (code, out, err) = run(deps, &argv);
    let value: Value =
        serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: stdout={out:?} stderr={err:?}"));
    assert_eq!(value["exit_code"], code, "envelope and exit code agree");
    assert_eq!(value["ok"], code == 0);
    (code, value)
}

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn file_hashes(dir: &Path) -> BTreeMap<String, String> {
    let mut hashes = BTreeMap::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        hashes.insert(name, sha256(&std::fs::read(entry.path()).unwrap()));
    }
    hashes
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
}

fn model(dir: &Path) -> PathBuf {
    fake_qwen2::write_fake_qwen2(dir).unwrap()
}

fn graph_args<'a>(model_path: &'a str, out_dir: &'a str) -> Vec<&'a str> {
    vec![
        "graph",
        "--model",
        model_path,
        "--out-dir",
        out_dir,
        "--format",
        "json,dot,mmd,csv",
        "--rank",
        "8",
        "--oversample",
        "4",
        "--power-iters",
        "2",
        "--seed",
        "0",
    ]
}

// ------------------------------------------------------------------ MCP harness

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

fn session(deps: &AppDeps, cfg: &McpConfig, lines: &[String]) -> Vec<Value> {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let input = (lines.join("\n") + "\n").into_bytes();
    let code = mcp::serve(std::io::Cursor::new(input), &mut out, &mut err, deps, cfg);
    assert_eq!(code, 0, "{}", String::from_utf8_lossy(&err));
    String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|line| {
            let value: Value = serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"));
            assert_eq!(value["jsonrpc"], "2.0");
            value
        })
        .collect()
}

fn one_call(cfg: &McpConfig, tool: &str, arguments: Value) -> (bool, Value) {
    let responses = session(&deps(), cfg, &[init(), call(1, tool, arguments)]);
    let result = &responses
        .iter()
        .find(|r| r["id"] == 1 && r.get("result").is_some())
        .unwrap_or_else(|| panic!("no result in {responses:?}"))["result"];
    let text: Value = serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(text, result["structuredContent"]);
    (
        result["isError"].as_bool().unwrap(),
        result["structuredContent"].clone(),
    )
}

// ------------------------------------------------------------------ fixtures

/// One-layer qwen2 GGUF: 8 graph nodes and 9 adjacent edges.
fn write_gguf(dir: &Path) -> PathBuf {
    const HIDDEN: usize = 8;
    const INTER: usize = 16;
    const VOCAB: usize = 12;
    let mut rng = Rng(0x5eed);
    let mut tensors = vec![
        T {
            name: "token_embd.weight".into(),
            ne: vec![HIDDEN as u64, VOCAB as u64],
            ty: F32,
            data: tensor_data(F32, VOCAB, HIDDEN, 1.0, &mut rng),
        },
        T {
            name: "output.weight".into(),
            ne: vec![HIDDEN as u64, VOCAB as u64],
            ty: F32,
            data: tensor_data(F32, VOCAB, HIDDEN, 1.0, &mut rng),
        },
    ];
    for name in ["attn_q", "attn_k", "attn_v", "attn_output"] {
        tensors.push(T {
            name: format!("blk.0.{name}.weight"),
            ne: vec![HIDDEN as u64, HIDDEN as u64],
            ty: F32,
            data: tensor_data(F32, HIDDEN, HIDDEN, 0.05, &mut rng),
        });
    }
    for (name, rows, cols) in [
        ("ffn_gate", INTER, HIDDEN),
        ("ffn_up", INTER, HIDDEN),
        ("ffn_down", HIDDEN, INTER),
    ] {
        tensors.push(T {
            name: format!("blk.0.{name}.weight"),
            // GGUF ne = [in, out].
            ne: vec![cols as u64, rows as u64],
            ty: F32,
            data: tensor_data(F32, rows, cols, 0.05, &mut rng),
        });
    }
    let kv = vec![
        ("general.architecture", Kv::Str("qwen2")),
        ("general.name", Kv::Str("graph-gguf")),
        ("general.alignment", Kv::U32(32)),
        ("qwen2.block_count", Kv::U32(1)),
    ];
    let path = dir.join("tiny-qwen2.gguf");
    std::fs::write(&path, gguf_bytes(&kv, &tensors)).unwrap();
    path
}

/// One-layer qwen2moe GGUF with two routed experts and a router: fused 3D tensors become
/// one node per expert plus one structural edge per gate/up -> down pair.
fn write_moe_gguf(dir: &Path) -> PathBuf {
    const HIDDEN: usize = 8;
    const INTER: usize = 12;
    const EXPERTS: usize = 2;
    let mut rng = Rng(0xbeef);
    let mut tensors = vec![
        T {
            name: "token_embd.weight".into(),
            ne: vec![HIDDEN as u64, 16],
            ty: F32,
            data: tensor_data(F32, 16, HIDDEN, 1.0, &mut rng),
        },
        T {
            name: "output.weight".into(),
            ne: vec![HIDDEN as u64, 16],
            ty: F32,
            data: tensor_data(F32, 16, HIDDEN, 1.0, &mut rng),
        },
    ];
    for name in ["attn_q", "attn_k", "attn_v", "attn_output"] {
        tensors.push(T {
            name: format!("blk.0.{name}.weight"),
            ne: vec![HIDDEN as u64, HIDDEN as u64],
            ty: F32,
            data: tensor_data(F32, HIDDEN, HIDDEN, 0.05, &mut rng),
        });
    }
    for (name, out, inp) in [
        ("ffn_gate_exps", INTER, HIDDEN),
        ("ffn_up_exps", INTER, HIDDEN),
        ("ffn_down_exps", HIDDEN, INTER),
    ] {
        let mut data = Vec::new();
        for _ in 0..EXPERTS {
            data.extend(tensor_data(F32, out, inp, 0.05, &mut rng));
        }
        tensors.push(T {
            name: format!("blk.0.{name}.weight"),
            ne: vec![inp as u64, out as u64, EXPERTS as u64],
            ty: F32,
            data,
        });
    }
    tensors.push(T {
        name: "blk.0.ffn_gate_inp.weight".into(),
        ne: vec![HIDDEN as u64, EXPERTS as u64],
        ty: F32,
        data: tensor_data(F32, EXPERTS, HIDDEN, 0.1, &mut rng),
    });
    let kv = vec![
        ("general.architecture", Kv::Str("qwen2moe")),
        ("general.name", Kv::Str("graph-moe")),
        ("general.alignment", Kv::U32(32)),
        ("qwen2moe.block_count", Kv::U32(1)),
    ];
    let path = dir.join("tiny-moe.gguf");
    std::fs::write(&path, gguf_bytes(&kv, &tensors)).unwrap();
    path
}

// ------------------------------------------------------------------ pipeline

#[test]
fn graph_writes_every_format_deterministically_across_thread_counts() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = model(dir.path());
    let out = dir.path().join("out");
    let mut args = graph_args(s(model_path.as_path()), s(&out));
    args.extend(["--jobs", "1"]);
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["command"], "graph");
    assert_eq!(env["data"]["global"]["nodes"], 30);
    assert_eq!(env["data"]["global"]["edges"], 45);
    assert_eq!(env["data"]["model"]["layers"], 4);
    assert_eq!(env["data"]["model"]["hidden"], 16);
    assert_eq!(env["data"]["parameters"]["edge_mode"], "adjacent");
    assert_eq!(env["data"]["truncated"]["nodes"], false);
    assert_eq!(env["data"]["outputs"].as_array().unwrap().len(), 5);

    let report = read_json(&out.join("graph.json"));
    assert_eq!(report["schema_version"], 1);
    assert_eq!(report["tool"]["name"], "surgeon");
    assert_eq!(report["model"]["format"], "safetensors");
    assert_eq!(report["model"]["architecture"], "qwen2");
    assert_eq!(report["model"]["tensor_count"], 30);
    assert_eq!(report["global"]["nodes"], 30);
    assert_eq!(report["global"]["edges"], 45);
    assert_eq!(report["global"]["structural_edges"], 0);
    let layers = report["global"]["layers"].as_u64().unwrap() as i64;
    assert_eq!(
        report["edges"].as_array().unwrap().len() as i64,
        12 * layers - 3
    );

    for edge in report["edges"].as_array().unwrap() {
        let source_seq = edge["source_seq"].as_u64().unwrap();
        let target_seq = edge["target_seq"].as_u64().unwrap();
        assert!(source_seq < target_seq, "causal order violated: {edge}");
        assert_eq!(edge["structural"], false);
        let rank = edge["rank_used"].as_f64().unwrap();
        let hidden = report["global"]["hidden"].as_f64().unwrap();
        let chance = edge["chance"].as_f64().unwrap();
        assert!((chance - rank / hidden).abs() < 1e-12, "{edge}");
        let alignment = edge["alignment"].as_f64().unwrap();
        let normalized = edge["alignment_norm"].as_f64().unwrap();
        assert!((normalized - (alignment - chance) / (1.0 - chance)).abs() < 1e-12);
        assert_eq!(edge["weight"], edge["alignment"]);
        assert!((0.0..=1.0).contains(&edge["cos_theta"].as_f64().unwrap()));
    }
    for node in report["nodes"].as_array().unwrap() {
        assert!(node["metrics"]["erank_energy_lower"]
            .as_f64()
            .unwrap()
            .le(&node["metrics"]["erank_energy_upper"].as_f64().unwrap()));
        assert_eq!(node["metrics"]["full_spectrum"], true, "{node}");
    }
    assert!(report["global"]["outlier_channels"].is_array());

    let first = file_hashes(&out);
    assert_eq!(
        first.keys().cloned().collect::<Vec<_>>(),
        [
            "edges.csv",
            "graph.dot",
            "graph.json",
            "graph.mmd",
            "matrices.csv"
        ]
    );
    for (name, text) in [
        (
            "graph.dot",
            std::fs::read_to_string(out.join("graph.dot")).unwrap(),
        ),
        (
            "graph.mmd",
            std::fs::read_to_string(out.join("graph.mmd")).unwrap(),
        ),
        (
            "matrices.csv",
            std::fs::read_to_string(out.join("matrices.csv")).unwrap(),
        ),
        (
            "edges.csv",
            std::fs::read_to_string(out.join("edges.csv")).unwrap(),
        ),
    ] {
        assert!(!text.is_empty(), "{name}");
        for node in report["nodes"].as_array().unwrap() {
            let id = node["id"].as_str().unwrap();
            assert!(text.contains(id), "{name} misses {id}");
        }
    }
    let dot = std::fs::read_to_string(out.join("graph.dot")).unwrap();
    assert!(dot.contains("subgraph cluster_L3"));
    assert!(dot.contains("cluster_legend"));
    assert!(dot.contains("shape=octagon") || !dot.contains("outlier"));
    let mmd = std::fs::read_to_string(out.join("graph.mmd")).unwrap();
    assert!(mmd.contains("flowchart LR"));
    assert!(mmd.contains("{{"));
    assert!(mmd.contains("linkStyle 44"));
    assert!(mmd.contains("classDef writer"));

    // Same command, more threads: every file must be byte for byte identical.
    let mut again = graph_args(s(model_path.as_path()), s(&out));
    again.extend(["--jobs", "4", "--overwrite"]);
    let (code, env) = run_json(&deps(), &again);
    assert_eq!(code, 0, "{env}");
    assert_eq!(file_hashes(&out), first);
}

#[test]
fn vector_and_edge_modes_keep_the_math_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = model(dir.path());
    let vector = dir.path().join("vec.json");
    std::fs::write(&vector, serde_json::to_string(&[0.5f32; 16]).unwrap()).unwrap();

    let all_out = dir.path().join("all");
    let mut args = graph_args(s(model_path.as_path()), s(&all_out));
    args.extend([
        "--edges",
        "all",
        "--edge-weight",
        "coupling",
        "--vector",
        s(&vector),
    ]);
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    // Writers: embed, 4 attn_out, 4 mlp_out; readers: 4 x (q,k,v,gate,up) + lm_head.
    assert_eq!(env["data"]["global"]["edges"], 97);
    let report = read_json(&all_out.join("graph.json"));
    assert_eq!(report["parameters"]["edge_weight"], "coupling");
    assert!(report["parameters"]["vector"].is_string());
    assert!(report["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|node| node["metrics"]["e_write"].as_f64().is_some()));
    assert!(report["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|node| node["metrics"]["e_read"].as_f64().is_some()));
    for edge in report["edges"].as_array().unwrap() {
        assert_eq!(edge["weight"], edge["coupling"]);
        assert!(edge["source_seq"].as_u64().unwrap() < edge["target_seq"].as_u64().unwrap());
    }

    let topk_out = dir.path().join("topk");
    let mut args = graph_args(s(model_path.as_path()), s(&topk_out));
    args.extend(["--edges", "topk:1"]);
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    let topk = env["data"]["global"]["edges"].as_u64().unwrap();
    assert!(topk > 45 && topk <= 97, "{topk}");

    let filtered = dir.path().join("filtered");
    let mut args = graph_args(s(model_path.as_path()), s(&filtered));
    args.extend(["--min-alignment", "1.1"]);
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["global"]["edges"], 0);
    assert_eq!(env["data"]["global"]["nodes"], 30);

    let selection = dir.path().join("selection");
    let mut args = graph_args(s(model_path.as_path()), s(&selection));
    args.extend(["--layers", "0", "--matrices", "q,k,v,o,embed"]);
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["global"]["nodes"], 5);
    assert_eq!(env["data"]["global"]["edges"], 3);
    let report = read_json(&selection.join("graph.json"));
    let ids: Vec<&str> = report["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|n| n["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["embed", "L0.k", "L0.q", "L0.v", "L0.attn_out"]);
}

#[test]
fn gguf_dense_and_moe_models_are_read_with_structural_edges() {
    let dir = tempfile::tempdir().unwrap();
    let dense = write_gguf(dir.path());
    let out = dir.path().join("dense");
    let args = vec![
        "graph",
        "--model",
        s(&dense),
        "--out-dir",
        s(&out),
        "--format",
        "json",
        "--rank",
        "4",
        "--oversample",
        "2",
        "--power-iters",
        "2",
    ];
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["model"]["format"], "gguf");
    assert_eq!(env["data"]["global"]["nodes"], 9);
    assert_eq!(env["data"]["global"]["edges"], 9);
    let report = read_json(&out.join("graph.json"));
    assert_eq!(report["model"]["architecture"], "qwen2");
    assert!(report["model"]["skipped"].as_array().unwrap().is_empty());
    assert!(report["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .all(|n| n["dtype"] == "F32"));

    let moe = write_moe_gguf(dir.path());
    let out = dir.path().join("moe");
    let args = vec![
        "graph",
        "--model",
        s(&moe),
        "--out-dir",
        s(&out),
        "--format",
        "json",
        "--rank",
        "4",
        "--oversample",
        "2",
        "--power-iters",
        "2",
    ];
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["global"]["nodes"], 12);
    assert_eq!(env["data"]["global"]["structural_edges"], 4);
    assert_eq!(env["data"]["global"]["edges"], 18);
    let report = read_json(&out.join("graph.json"));
    assert_eq!(report["model"]["architecture"], "qwen2_moe");
    assert_eq!(report["model"]["skipped"].as_array().unwrap().len(), 1);
    assert!(report["model"]["skipped"][0]["name"]
        .as_str()
        .unwrap()
        .ends_with("ffn_gate_inp.weight"));
    for expert in 0..2 {
        for role in ["gate", "up"] {
            let source = format!("L0.{role}.e{expert}");
            let target = format!("L0.mlp_out.e{expert}");
            assert!(report["edges"].as_array().unwrap().iter().any(|edge| {
                edge["source"] == source && edge["target"] == target && edge["structural"] == true
            }));
        }
    }
    // Structural edges have no alignment and point forward in causal time.
    for edge in report["edges"].as_array().unwrap() {
        if edge["structural"] == true {
            assert_eq!(edge["alignment"], 0.0);
            assert!(edge["source_seq"].as_u64().unwrap() < edge["target_seq"].as_u64().unwrap());
        }
    }
}

#[test]
fn estimate_never_runs_the_svd_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = model(dir.path());
    let out = dir.path().join("estimate-out");
    let mut args = graph_args(s(model_path.as_path()), s(&out));
    args.push("--estimate");
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["estimate"], true);
    assert_eq!(env["data"]["nodes"], 30);
    assert!(env["data"]["passes_per_matrix"].as_u64().unwrap() >= 4);
    assert!(env["data"]["bytes_read"].as_u64().unwrap() > 0);
    assert!(
        !out.exists(),
        "--estimate must not create the output directory"
    );
}

#[test]
fn cache_resume_reuses_partial_records_and_plain_runs_invalidate_them() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = model(dir.path());
    let cache = dir.path().join("cache");
    let out1 = dir.path().join("out1");
    let mut args = graph_args(s(model_path.as_path()), s(&out1));
    args.extend(["--cache-dir", s(&cache)]);
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    let cache_file = std::fs::read_dir(&cache)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .unwrap();
    let lines: Vec<String> = std::fs::read_to_string(&cache_file)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect();
    assert_eq!(lines.len(), 30);

    // Poison one record: a resumed run must trust it, a plain run must recompute.
    let mut record: Value = serde_json::from_str(&lines[0]).unwrap();
    record["analysis"]["metrics"]["kurtosis"] = json!(123.456);
    std::fs::write(&cache_file, serde_json::to_string(&record).unwrap() + "\n").unwrap();

    let resumed = dir.path().join("resumed");
    let mut args = graph_args(s(model_path.as_path()), s(&resumed));
    args.extend(["--cache-dir", s(&cache), "--resume"]);
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    let report = read_json(&resumed.join("graph.json"));
    assert!(report["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .any(|node| { (node["metrics"]["kurtosis"].as_f64().unwrap() - 123.456).abs() < 1e-9 }));
    assert_eq!(
        std::fs::read_to_string(&cache_file)
            .unwrap()
            .lines()
            .count(),
        30
    );

    let fresh = dir.path().join("fresh");
    let mut args = graph_args(s(model_path.as_path()), s(&fresh));
    args.extend(["--cache-dir", s(&cache)]);
    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{env}");
    let report = read_json(&fresh.join("graph.json"));
    assert!(report["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .all(|node| (node["metrics"]["kurtosis"].as_f64().unwrap() - 123.456).abs() > 1e-6));
}

#[test]
fn cancellation_exits_1_and_leaves_no_outputs() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = model(dir.path());
    let out = dir.path().join("cancelled");
    let mut cancelled = deps();
    cancelled.interrupted = Arc::new(AtomicBool::new(true));
    let args = graph_args(s(model_path.as_path()), s(&out));
    let (code, env) = run_json(&cancelled, &args);
    assert_eq!(code, 1, "{env}");
    assert_eq!(env["error"]["kind"], "cancelled");
    assert!(!out.exists());
}

#[test]
fn existing_outputs_require_overwrite_and_bad_options_exit_2() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = model(dir.path());
    let out = dir.path().join("out");
    let args = graph_args(s(model_path.as_path()), s(&out));
    assert_eq!(run_json(&deps(), &args).0, 0);
    let before = read_json(&out.join("graph.json"));

    let (code, env) = run_json(&deps(), &args);
    assert_eq!(code, 2, "{env}");
    assert_eq!(env["error"]["kind"], "invalid_input");
    assert!(env["error"]["message"]
        .as_str()
        .unwrap()
        .contains("--overwrite"));
    assert_eq!(read_json(&out.join("graph.json")), before);

    let cache = dir.path().join("c");
    let cases: Vec<Vec<&str>> = vec![
        vec!["graph", "--model", s(&model_path), "--rank", "0"],
        vec!["graph", "--model", s(&model_path), "--resume"],
        vec![
            "graph",
            "--model",
            s(&model_path),
            "--cache-dir",
            s(&cache),
            "--no-cache",
        ],
        vec!["graph", "--model", s(&model_path), "--edges", "sideways"],
        vec![
            "graph",
            "--model",
            s(&model_path),
            "--layers",
            "99",
            "--matrices",
            "q",
        ],
    ];
    for args in cases {
        let (code, env) = run_json(&deps(), &args);
        assert_eq!(code, 2, "{args:?}: {env}");
    }
}

// ------------------------------------------------------------------ schema & MCP

#[test]
fn schema_kind_graph_is_the_report_schema() {
    let (code, out, err) = run(&deps(), &["schema", "--kind", "graph"]);
    assert_eq!(code, 0, "{err}");
    let schema: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(schema["title"], "GraphReport");
    let properties = schema["properties"].as_object().unwrap();
    for key in [
        "schema_version",
        "tool",
        "model",
        "parameters",
        "global",
        "nodes",
        "edges",
    ] {
        assert!(properties.contains_key(key), "{key}: {schema}");
    }
    let node = schema["definitions"]["NodeReport"]["properties"]
        .as_object()
        .unwrap();
    assert!(node.contains_key("metrics"));
    assert!(node.contains_key("seq"));
    let edge = schema["definitions"]["EdgeReport"]["properties"]
        .as_object()
        .unwrap();
    for key in [
        "alignment",
        "chance",
        "alignment_norm",
        "coupling",
        "cos_theta",
    ] {
        assert!(edge.contains_key(key), "{key}");
    }

    let (code, out, _) = run(&deps(), &["schema", "--kind", "tools"]);
    assert_eq!(code, 0);
    let tools: Value = serde_json::from_str(&out).unwrap();
    let list = tools.as_array().unwrap();
    assert_eq!(list.len(), 10);
    let graph = list.iter().find(|t| t["name"] == "surgeon_graph").unwrap();
    assert_eq!(graph["inputSchema"]["required"], json!(["model"]));
    assert!(graph["inputSchema"]["properties"]["out_dir"].is_object());
    assert_eq!(graph["annotations"]["readOnlyHint"], false);
    assert_eq!(graph["annotations"]["openWorldHint"], false);
}

#[test]
fn mcp_returns_a_bounded_summary_and_writes_only_with_out_dir() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = model(dir.path());
    let config = McpConfig::new(&[dir.path().to_path_buf()], false, false).unwrap();

    let (is_error, env) = one_call(
        &config,
        "surgeon_graph",
        json!({"model": s(&model_path), "rank": 8, "oversample": 4, "power_iters": 2}),
    );
    assert!(!is_error, "{env}");
    assert_eq!(env["ok"], true);
    assert_eq!(env["command"], "graph");
    assert!(
        env["data"].get("edges").is_none(),
        "full graph leaked inline"
    );
    assert_eq!(env["data"]["outputs"], json!([]));
    assert_eq!(env["data"]["global"]["nodes"], 30);
    assert!(env["data"]["nodes"].as_array().unwrap().len() <= 200);
    assert!(env["data"]["top_edges"].as_array().unwrap().len() <= 20);
    assert!(env["data"]["note"].as_str().unwrap().contains("graph.json"));
    assert!(std::fs::read_dir(dir.path()).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with("graph.")));

    let out = dir.path().join("mcp-out");
    let (is_error, env) = one_call(
        &config,
        "surgeon_graph",
        json!({"model": s(&model_path), "out_dir": s(&out), "format": ["json", "dot"], "rank": 8, "oversample": 4, "power_iters": 2}),
    );
    assert!(!is_error, "{env}");
    let outputs = env["data"]["outputs"].as_array().unwrap();
    assert_eq!(outputs.len(), 2);
    assert!(out.join("graph.json").is_file());
    assert!(out.join("graph.dot").is_file());

    let outside = tempfile::tempdir().unwrap();
    let denied = outside.path().join("out");
    let (is_error, env) = one_call(
        &config,
        "surgeon_graph",
        json!({"model": s(&model_path), "out_dir": s(&denied)}),
    );
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "path_not_allowed", "{env}");
    assert!(!denied.exists());
}

#[test]
fn mcp_read_only_refuses_out_dir_and_cache_dir_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let model_path = model(dir.path());
    let config = McpConfig::new(&[dir.path().to_path_buf()], true, false).unwrap();
    for arguments in [
        json!({"model": s(&model_path), "out_dir": s(&dir.path().join("out"))}),
        json!({"model": s(&model_path), "cache_dir": s(&dir.path().join("cache"))}),
    ] {
        let (is_error, env) = one_call(&config, "surgeon_graph", arguments);
        assert!(is_error);
        assert_eq!(env["error"]["kind"], "read_only", "{env}");
        assert_eq!(env["exit_code"], 2);
    }
    assert!(std::fs::read_dir(dir.path()).unwrap().all(|e| !e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with("graph.")));

    let (is_error, env) = one_call(
        &config,
        "surgeon_graph",
        json!({"model": s(&model_path), "rank": 4, "oversample": 2}),
    );
    assert!(!is_error, "{env}");
    assert_eq!(env["data"]["global"]["nodes"], 30);
    assert_eq!(env["data"]["outputs"], json!([]));
}
