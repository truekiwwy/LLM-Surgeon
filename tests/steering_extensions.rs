//! End to end, with the real binary on the mini Qwen2 made tied: a directional ablation of
//! every residual-stream writer (`down_proj`, `o_proj`, `embed_tokens`) is simulated by a
//! dry-run, measured by the native generator, applied to a copy with `lm_head` detached,
//! and rolled back byte for byte when a later operation fails.

#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};
use weight_surgeon::engine::ReadOnlyModel;

const EMBED: &str = "model.embed_tokens.weight";
const LM_HEAD: &str = "lm_head.weight";
const ABLATION: &str = r#"{"schema_version":1,"operations":[{"id":"ablate","layers":"all",
    "projections":["down_proj","o_proj","embed_tokens"],"operation":"project_out",
    "vector_file":"direction.safetensors","vector_tensor":"direction",
    "vector_tensor_b":"direction","alpha":1.0}]}"#;

/// Exit code and JSON envelope of `surgeon --json <args>`, run in `dir`.
fn surgeon(dir: &Path, args: &[&str]) -> (i32, Value) {
    let output = Command::new(env!("CARGO_BIN_EXE_surgeon"))
        .arg("--json")
        .args(args)
        .current_dir(dir)
        .output()
        .unwrap();
    let envelope = serde_json::from_slice(&output.stdout).unwrap_or_else(|e| {
        panic!(
            "{args:?}: {e}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status.code().unwrap(), envelope)
}

/// Drops `lm_head.weight` (the last tensor of the mini model) and ties the config, as a
/// Qwen2.5-0.5B checkpoint is stored.
fn tie(model: &Path) {
    let shard = model.join("model.safetensors");
    let bytes = std::fs::read(&shard).unwrap();
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header: serde_json::Map<String, Value> =
        serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let head = header.remove(LM_HEAD).unwrap();
    let start = head["data_offsets"][0].as_u64().unwrap() as usize;
    let data = &bytes[8 + header_len..];
    assert_eq!(
        head["data_offsets"][1].as_u64().unwrap() as usize,
        data.len()
    );
    let mut text = serde_json::to_string(&header).unwrap();
    while !text.len().is_multiple_of(8) {
        text.push(' ');
    }
    let mut file = (text.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(text.as_bytes());
    file.extend_from_slice(&data[..start]);
    std::fs::write(&shard, file).unwrap();

    let config = model.join("config.json");
    let mut json: Value = serde_json::from_slice(&std::fs::read(&config).unwrap()).unwrap();
    json["tie_word_embeddings"] = json!(true);
    std::fs::write(config, json.to_string()).unwrap();
}

/// The tied mini model, a unit direction extracted from it, the patch and prompt files.
fn workspace() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let model = fake_qwen2::write_fake_qwen2(dir.path()).unwrap();
    fake_qwen2::write_prompt_files(dir.path()).unwrap();
    tie(&model);
    let (code, extract) = surgeon(
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
            "2",
            "--out",
            "direction.safetensors",
        ],
    );
    assert_eq!(code, 0, "{extract}");
    std::fs::write(dir.path().join("patch.json"), ABLATION).unwrap();
    (dir, model)
}

fn tensor(shard: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let model = ReadOnlyModel::open(shard).unwrap();
    let view = model.tensor(name).unwrap();
    let values = view
        .bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (view.info.shape.clone(), values)
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        std::fs::copy(entry.path(), to.join(entry.file_name())).unwrap();
    }
}

fn snapshot(model: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(model)
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| !e.file_name().to_string_lossy().contains(".bak-"))
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                std::fs::read(e.path()).unwrap(),
            )
        })
        .collect();
    files.sort();
    files
}

#[test]
fn dry_run_simulates_the_whole_ablation_without_writing() {
    let (dir, model) = workspace();
    let before = snapshot(&model);

    let (code, dry) = surgeon(
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
    assert_eq!(code, 0, "{dry}");
    let report = &dry["data"]["report"];
    assert_eq!(report["dry_run"], true);
    let targets: Vec<&str> = report["operations"][0]["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    // The embeddings first, then both writers of each of the 4 layers.
    assert_eq!(targets.len(), 1 + 2 * fake_qwen2::LAYERS);
    assert_eq!(targets[0], EMBED);
    assert_eq!(targets[1], "model.layers.0.mlp.down_proj.weight");
    assert_eq!(targets[2], "model.layers.0.self_attn.o_proj.weight");
    assert_eq!(report["operations"][0]["vector_len"], fake_qwen2::HIDDEN);
    let untie = &dry["data"]["untie_lm_head"];
    assert_eq!(untie["copy_embeddings"], true);
    assert!(untie["config"].as_str().unwrap().ends_with("config.json"));
    assert!(dry["warnings"][0].as_str().unwrap().contains(LM_HEAD));
    assert_eq!(snapshot(&model), before, "a dry-run writes nothing");

    // The native generator runs the tied model: the baseline of the simulation.
    let (code, refusal) = surgeon(
        dir.path(),
        &[
            "bench",
            "refusal",
            "--model",
            "qwen2-mini",
            "--prompts",
            "pos.json",
            "--max-new-tokens",
            "8",
        ],
    );
    assert_eq!(code, 0, "{refusal}");
    let data = &refusal["data"];
    assert_eq!(data["n"], 6);
    assert_eq!(data["chat_template"], "chatml");
    assert_eq!(data["items"].as_array().unwrap().len(), 6);
    assert!(data["items"][0]["output"].is_string());
    assert!(data["refused"].as_u64().unwrap() <= 6);
    assert_eq!(data["paired"], Value::Null);
    assert_eq!(snapshot(&model), before);

    // In a GGUF only the MLP is patchable: refused on the patch alone, the file unread.
    std::fs::write(dir.path().join("m.gguf"), b"GGUF\0\0\0\0").unwrap();
    std::fs::write(dir.path().join("v.json"), "[1.0]").unwrap();
    std::fs::write(
        dir.path().join("gguf.json"),
        r#"{"schema_version":1,"operations":[{"id":"attn","layers":"all",
            "projections":["o_proj"],"operation":"subtract","vector_file":"v.json","alpha":0.5}]}"#,
    )
    .unwrap();
    let (code, refused) = surgeon(
        dir.path(),
        &[
            "apply",
            "--gguf",
            "--model",
            "m.gguf",
            "--patch",
            "gguf.json",
            "--dry-run",
        ],
    );
    assert_eq!(code, 2, "{refused}");
    assert_eq!(refused["error"]["kind"], "invalid_input");
    assert!(refused["error"]["message"]
        .as_str()
        .unwrap()
        .contains("SafeTensors"));

    // o_proj and embed_tokens are listed on request.
    let (_, inspect) = surgeon(
        dir.path(),
        &[
            "inspect",
            "--model",
            "qwen2-mini",
            "--proj",
            "o_proj",
            "--proj",
            "embed_tokens",
        ],
    );
    assert_eq!(
        inspect["data"]["tensors"].as_array().unwrap().len(),
        1 + fake_qwen2::LAYERS
    );
}

#[test]
fn apply_detaches_the_head_erases_the_direction_and_measures_the_copy() {
    let (dir, model) = workspace();
    let patched = dir.path().join("qwen2-patched");
    copy_dir(&model, &patched);
    let original = model.join("model.safetensors");
    let shard = patched.join("model.safetensors");
    let (_, embed_before) = tensor(&original, EMBED);
    let (_, direction) = tensor(&dir.path().join("direction.safetensors"), "direction");

    let (code, applied) = surgeon(
        dir.path(),
        &["apply", "--model", "qwen2-patched", "--patch", "patch.json"],
    );
    assert_eq!(code, 0, "{applied}");
    assert_eq!(applied["data"]["untie_lm_head"]["copy_embeddings"], true);
    assert_eq!(
        applied["data"]["backups"].as_array().unwrap().len(),
        2,
        "the shard and config.json"
    );

    // The head is the unpatched embedding matrix, declared independent.
    let (shape, head) = tensor(&shard, LM_HEAD);
    assert_eq!(shape, [64, fake_qwen2::HIDDEN]);
    assert_eq!(head, embed_before);
    let config: Value =
        serde_json::from_slice(&std::fs::read(patched.join("config.json")).unwrap()).unwrap();
    assert_eq!(config["tie_word_embeddings"], false);
    assert_eq!(config["hidden_size"], fake_qwen2::HIDDEN);

    // Nothing writes along v any more: vᵀW = 0 for the layer writers, W v = 0 for the rows
    // of the embeddings; the readers of the MLP are untouched.
    let (shape, embed) = tensor(&shard, EMBED);
    for row in embed.chunks_exact(shape[1]) {
        let read: f32 = row.iter().zip(&direction).map(|(x, v)| x * v).sum();
        assert!(read.abs() < 1e-5, "W v = {read}");
    }
    assert_ne!(embed, embed_before);
    for layer in 0..fake_qwen2::LAYERS {
        for writer in ["mlp.down_proj", "self_attn.o_proj"] {
            let (shape, w) = tensor(&shard, &format!("model.layers.{layer}.{writer}.weight"));
            for col in 0..shape[1] {
                let read: f32 = (0..shape[0])
                    .map(|row| direction[row] * w[row * shape[1] + col])
                    .sum();
                assert!(read.abs() < 1e-5, "{writer} layer {layer}: vᵀW = {read}");
            }
        }
        let gate = format!("model.layers.{layer}.mlp.gate_proj.weight");
        assert_eq!(tensor(&shard, &gate), tensor(&original, &gate));
    }

    // Both models generate, pair by pair; the original was never touched.
    let (code, refusal) = surgeon(
        dir.path(),
        &[
            "bench",
            "refusal",
            "--model",
            "qwen2-patched",
            "--baseline-model",
            "qwen2-mini",
            "--prompts",
            "neg.json",
            "--max-new-tokens",
            "8",
        ],
    );
    assert_eq!(code, 0, "{refusal}");
    let data = &refusal["data"];
    assert_eq!(data["baseline"]["n"], 6);
    let paired = &data["paired"];
    assert_eq!(
        paired["both_refused"].as_u64().unwrap()
            + paired["both_complied"].as_u64().unwrap()
            + paired["unblocked"].as_array().unwrap().len() as u64
            + paired["newly_refused"].as_array().unwrap().len() as u64,
        6
    );
    assert!(paired["delta_ci95_low"].as_f64().unwrap() <= paired["delta_rate"].as_f64().unwrap());

    // Pass@1 without llama.cpp, on the patched copy against the original.
    std::fs::write(
        dir.path().join("gsm.json"),
        r#####"[{"id":"q1","question":"joie et soleil","answer":"#### 5"},
            {"id":"q2","question":"pluie et nuit","answer":"#### 7"}]"#####,
    )
    .unwrap();
    let (code, eval) = surgeon(
        dir.path(),
        &[
            "bench",
            "eval",
            "--backend",
            "native",
            "--model",
            "qwen2-patched",
            "--baseline-model",
            "qwen2-mini",
            "--dataset",
            "gsm.json",
            "--max-new-tokens",
            "8",
        ],
    );
    assert_eq!(code, 0, "{eval}");
    assert_eq!(eval["data"]["n"], 2);
    assert_eq!(eval["data"]["baseline"]["n"], 2);

    // A second apply finds the head already independent: no untie, same erased state.
    let (code, again) = surgeon(
        dir.path(),
        &[
            "apply",
            "--model",
            "qwen2-patched",
            "--patch",
            "patch.json",
            "--dry-run",
        ],
    );
    assert_eq!(code, 0, "{again}");
    assert_eq!(again["data"].get("untie_lm_head"), None);
}

#[test]
fn a_failure_after_the_untie_restores_every_file_byte_for_byte() {
    let (dir, model) = workspace();
    let before = snapshot(&model);
    // The second operation fails at write time: its vector has the wrong length.
    std::fs::write(dir.path().join("short.json"), "[1.0, 2.0, 3.0]").unwrap();
    let mut patch: Value = serde_json::from_str(ABLATION).unwrap();
    patch["operations"].as_array_mut().unwrap().push(json!({
        "id": "broken", "layers": "0", "projections": ["down_proj"],
        "operation": "subtract", "vector_file": "short.json", "alpha": 0.5
    }));
    std::fs::write(dir.path().join("broken.json"), patch.to_string()).unwrap();

    let (code, failed) = surgeon(
        dir.path(),
        &["apply", "--model", "qwen2-mini", "--patch", "broken.json"],
    );
    assert_eq!(code, 3, "execution failed, original intact: {failed}");
    assert_eq!(failed["ok"], false);
    let message = failed["error"]["message"].as_str().unwrap();
    assert!(message.contains("broken"), "{message}");
    assert!(message.contains("rollback : true"), "{message}");
    assert_eq!(
        snapshot(&model),
        before,
        "shard without lm_head and tied config are back"
    );
    assert!(ReadOnlyModel::open(&model.join("model.safetensors"))
        .unwrap()
        .index()
        .get(LM_HEAD)
        .is_err());
}
