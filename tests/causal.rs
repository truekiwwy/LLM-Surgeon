#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;
#[path = "support/mcp_client.rs"]
mod mcp_client;

use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{json, Value};

fn command(dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_surgeon"));
    cmd.current_dir(dir).env("NO_COLOR", "1");
    cmd
}

fn surgeon(dir: &Path, args: &[&str]) -> (i32, Value) {
    let output = command(dir).arg("--json").args(args).output().unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    let envelope = serde_json::from_str(&stdout)
        .unwrap_or_else(|e| panic!("{args:?}: not strict JSON ({e}): {stdout}"));
    (output.status.code().unwrap(), envelope)
}

fn workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fake_qwen2::write_fake_qwen2(dir.path()).unwrap();
    fake_qwen2::write_prompt_files(dir.path()).unwrap();
    dir
}

fn tensor_names(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap();
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
    let mut names: Vec<String> = header
        .as_object()
        .unwrap()
        .keys()
        .filter(|k| *k != "__metadata__")
        .cloned()
        .collect();
    names.sort();
    names
}

#[test]
fn a_layer_sweep_writes_every_layer_and_its_best_direction_feeds_compose() {
    let dir = workspace();
    let (code, v) = surgeon(
        dir.path(),
        &[
            "extract",
            "--model",
            "qwen2-mini",
            "--pos-prompts",
            "pos.json",
            "--neg-prompts",
            "neg.json",
            "--layers",
            "all",
            "--method",
            "svd",
            "--out",
            "work/sweep.safetensors",
        ],
    );
    assert_eq!(code, 0, "{v}");
    let data = &v["data"];
    let layers = data["layers"].as_array().unwrap();
    assert_eq!(layers.len(), fake_qwen2::LAYERS);
    for (i, layer) in layers.iter().enumerate() {
        assert_eq!(layer["layer"], i);
        let auc = layer["auc"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&auc), "{layer}");
        assert!(layer["margin"].as_f64().unwrap().is_finite());
    }
    let best = data["best_layer"].as_u64().unwrap() as usize;
    assert!(best < fake_qwen2::LAYERS);
    let file = dir.path().join("work/sweep.safetensors");
    let mut expected: Vec<String> = (0..fake_qwen2::LAYERS)
        .map(|l| format!("layers.{l}"))
        .chain(["direction".to_owned()])
        .collect();
    expected.sort();
    assert_eq!(tensor_names(&file), expected);
    let stats: Value = serde_json::from_str(
        &std::fs::read_to_string(dir.path().join("work/sweep.safetensors.stats.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(stats["layer"], best);
    assert_eq!(
        stats["layer_scores"].as_array().unwrap().len(),
        fake_qwen2::LAYERS
    );

    let (code, composed) = surgeon(
        dir.path(),
        &[
            "compose",
            "--ablate",
            "work/sweep.safetensors",
            "--out",
            "work/composed.safetensors",
        ],
    );
    assert_eq!(code, 0, "{composed}");

    let (code, refused) = surgeon(
        dir.path(),
        &[
            "extract",
            "--model",
            "qwen2-mini",
            "--pos-prompts",
            "pos.json",
            "--neg-prompts",
            "neg.json",
            "--layers",
            "9",
            "--out",
            "work/out_of_range.safetensors",
        ],
    );
    assert_eq!(code, 2, "{refused}");
    assert!(!dir.path().join("work/out_of_range.safetensors").exists());
}

#[test]
fn mcp_sweeps_and_traces_and_a_read_only_server_still_traces() {
    let dir = workspace();
    let read = |name: &str| -> Value {
        serde_json::from_str(&std::fs::read_to_string(dir.path().join(name)).unwrap()).unwrap()
    };
    let (pos, neg) = (read("pos.json"), read("neg.json"));
    let call = |id: u64, name: &str, arguments: Value| {
        json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments}})
    };
    let exchange = |read_only: bool, calls: Vec<Value>| -> Vec<Value> {
        let mut cmd = command(dir.path());
        cmd.args(["mcp", "--allow-dir", dir.path().to_str().unwrap()]);
        if read_only {
            cmd.arg("--read-only");
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut lines = vec![
            json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "causal", "version": "0"}}}),
            json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        ];
        lines.extend(calls);
        let (replies, _) = mcp_client::exchange(&mut child, &lines);
        replies
    };
    let result = |replies: &[Value], id: u64| -> Value {
        replies.iter().find(|r| r["id"] == id).unwrap()["result"]["structuredContent"].clone()
    };

    let trace_args = json!({"model": "qwen2-mini", "clean_prompts": pos,
        "corrupted_prompts": neg, "component": "mlp", "layers": "0-3"});
    let replies = exchange(
        false,
        vec![
            call(
                1,
                "surgeon_extract",
                json!({"model": "qwen2-mini", "pos_prompts_file": "pos.json",
                    "neg_prompts_file": "neg.json", "layers": "1-2",
                    "out": "work/mcp_sweep.safetensors"}),
            ),
            call(2, "surgeon_causal_trace", trace_args.clone()),
            call(
                3,
                "surgeon_causal_trace",
                json!({"model": "qwen2-mini", "clean_prompts": ["a"],
                    "corrupted_prompts": ["a", "b"]}),
            ),
        ],
    );
    let sweep = result(&replies, 1);
    assert_eq!(sweep["ok"], true, "{sweep}");
    let swept: Vec<u64> = sweep["data"]["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|l| l["layer"].as_u64().unwrap())
        .collect();
    assert_eq!(swept, [1, 2]);

    let trace = result(&replies, 2);
    assert_eq!(trace["ok"], true, "{trace}");
    let data = &trace["data"];
    assert_eq!(data["component"], "mlp");
    assert_eq!(data["layers"].as_array().unwrap().len(), 4);
    assert_eq!(data["ranking"].as_array().unwrap().len(), 4);
    assert_eq!(data["peak_layer"], data["ranking"][0]);
    let (start, end) = (
        data["window"]["start"].as_u64().unwrap(),
        data["window"]["end"].as_u64().unwrap(),
    );
    assert!(start <= end && end < 4);
    let peak = data["peak_layer"].as_u64().unwrap();
    assert!((start..=end).contains(&peak));
    let used = data["pairs_used"].as_u64().unwrap();
    assert!(used >= 1, "{data}");

    let unpaired = result(&replies, 3);
    assert_eq!(unpaired["ok"], false);
    assert_eq!(unpaired["error"]["kind"], "invalid_input");

    let replies = exchange(
        true,
        vec![
            call(1, "surgeon_causal_trace", trace_args),
            call(
                2,
                "surgeon_extract",
                json!({"model": "qwen2-mini", "pos_prompts_file": "pos.json",
                    "neg_prompts_file": "neg.json", "layers": "all",
                    "out": "work/denied.safetensors"}),
            ),
        ],
    );
    let again = result(&replies, 1);
    assert_eq!(again["data"]["layers"], data["layers"], "deterministic");
    assert_eq!(again["data"]["window"], data["window"]);
    assert_eq!(result(&replies, 2)["error"]["kind"], "read_only");
}
