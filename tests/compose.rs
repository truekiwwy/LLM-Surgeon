//! `surgeon compose` end to end with the real binary: a direction extracted from the mini
//! Qwen2 composed against a kept direction, its patch applied (dry run) to the model, the
//! refusals, and the same call through the MCP tool `surgeon_compose`.

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

/// The mini model, a refusal-like direction extracted from it (layer 1) and a kept
/// direction that overlaps it.
fn workspace() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fake_qwen2::write_fake_qwen2(dir.path()).unwrap();
    fake_qwen2::write_prompt_files(dir.path()).unwrap();
    let (code, extracted) = surgeon(
        dir.path(),
        &[
            "extract",
            "--model",
            "qwen2-mini",
            "--pos-prompts",
            "pos.json",
            "--neg-prompts",
            "neg.json",
            "--layer",
            "1",
            "--out",
            "refusal.safetensors",
        ],
    );
    assert_eq!(code, 0, "{extracted}");
    let keep: Vec<f64> = (0..fake_qwen2::HIDDEN)
        .map(|i| ((i * 7 % 5) as f64 - 2.0) / 3.0)
        .collect();
    std::fs::write(
        dir.path().join("math.json"),
        serde_json::to_string(&keep).unwrap(),
    )
    .unwrap();
    dir
}

#[test]
fn the_composed_direction_is_orthogonal_to_the_kept_one_and_its_patch_applies() {
    let dir = workspace();
    let (code, v) = surgeon(
        dir.path(),
        &[
            "compose",
            "--ablate",
            "refusal.safetensors:0.8",
            "--keep",
            "math.json",
            "--out",
            "composed.safetensors",
            "--patch-out",
            "patch.json",
        ],
    );
    assert_eq!(code, 0, "{v}");
    let c = &v["data"]["composition"];
    assert_eq!(c["mode"], "orthogonal_preserve");
    let diag = &c["diagnostics"][0];
    assert!(
        diag["cos_before"][0].as_f64().unwrap().abs() > 1e-3,
        "{diag}"
    );
    assert!(
        diag["cos_after"][0].as_f64().unwrap().abs() < 1e-6,
        "{diag}"
    );
    assert!(c["result_cos_keep"][0].as_f64().unwrap().abs() < 1e-6);
    let sources = c["sources"].as_array().unwrap();
    assert_eq!(sources.len(), 2);
    assert_eq!(sources[0]["role"], "ablate");
    assert_eq!(sources[0]["alpha"], 0.8);
    assert_eq!(sources[0]["sha256"].as_str().unwrap().len(), 64);
    assert_eq!(sources[0]["layer"], 1);

    // The recipe is in the patch, and the patch is one the engine applies to this model.
    let patch: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("patch.json")).unwrap())
            .unwrap();
    assert_eq!(patch["operations"][0]["operation"], "project_out");
    assert_eq!(
        patch["composition"]["output_sha256"],
        v["data"]["out_sha256"]
    );
    let (code, applied) = surgeon(
        dir.path(),
        &[
            "apply",
            "--model",
            "qwen2-mini",
            "--patch",
            "patch.json",
            "--dry-run",
        ],
    );
    assert_eq!(code, 0, "{applied}");

    // Existing outputs are kept unless asked.
    let again = [
        "compose",
        "--ablate",
        "refusal.safetensors",
        "--out",
        "composed.safetensors",
    ];
    let (code, refused) = surgeon(dir.path(), &again);
    assert_eq!(code, 2, "{refused}");
    assert!(refused["error"]["message"]
        .as_str()
        .unwrap()
        .contains("--overwrite"));
}

#[test]
fn collinear_null_and_mismatched_vectors_are_refused_by_name() {
    let dir = workspace();
    let refuse = |args: &[&str], kind: &str, name: &str| {
        let (code, v) = surgeon(dir.path(), args);
        assert_eq!(code, 2, "{args:?}: {v}");
        assert_eq!(v["error"]["kind"], kind, "{v}");
        let message = v["error"]["message"].as_str().unwrap();
        assert!(message.contains(name), "{message}");
        assert!(
            !dir.path().join("out.safetensors").exists(),
            "nothing written"
        );
    };
    std::fs::write(
        dir.path().join("zero.json"),
        "[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
    )
    .unwrap();
    std::fs::write(dir.path().join("short.json"), "[1, 2, 3]").unwrap();
    let base = ["compose", "--out", "out.safetensors"];
    refuse(
        &[
            &base[..],
            &["--ablate", "math.json:2", "--keep", "math.json"],
        ]
        .concat(),
        "collinear_vectors",
        "math.json",
    );
    refuse(
        &[&base[..], &["--ablate", "zero.json"]].concat(),
        "invalid_input",
        "zero.json",
    );
    refuse(
        &[
            &base[..],
            &["--ablate", "refusal.safetensors", "--keep", "short.json"],
        ]
        .concat(),
        "invalid_input",
        "short.json",
    );
}

#[test]
fn mcp_composes_the_same_way() {
    let dir = workspace();
    let mut child = command(dir.path())
        .args(["mcp", "--allow-dir", dir.path().to_str().unwrap()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let lines = [
        json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "compose", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        json!({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "surgeon_compose", "arguments": {
                "ablate": ["refusal.safetensors"], "keep": ["math.json"],
                "out": "mcp.safetensors", "patch_out": "mcp.json"}}}),
        json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
            "name": "surgeon_compose", "arguments": {
                "ablate": ["../outside.safetensors"], "out": "x.safetensors"}}}),
    ];
    let (replies, _) = mcp_client::exchange(&mut child, &lines);
    let reply = |id: u64| replies.iter().find(|r| r["id"] == id).unwrap()["result"].clone();
    let ok = reply(1);
    assert_eq!(ok["structuredContent"]["ok"], true, "{ok}");
    let c = &ok["structuredContent"]["data"]["composition"];
    assert!(
        c["result_cos_keep"][0].as_f64().unwrap().abs() < 1e-6,
        "{c}"
    );
    assert!(dir.path().join("mcp.json").is_file());
    let outside = reply(2);
    assert_eq!(outside["structuredContent"]["ok"], false, "{outside}");
    assert_eq!(
        outside["structuredContent"]["error"]["kind"],
        "path_not_allowed"
    );
}
