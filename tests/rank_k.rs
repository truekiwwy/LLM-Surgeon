//! Rank-k subspaces, end to end with the real binary on the mini Qwen2: `extract --rank`
//! (svd) and `extract --labels-file` (leace) write `[k, hidden]` tensors, one `project_out`
//! erases them in a single pass from every residual-stream writer, and every check on the
//! rank and on the projector runs before the first write.

#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{json, Value};
use weight_surgeon::engine::{read_index, DType, ReadOnlyModel};

const HIDDEN: usize = fake_qwen2::HIDDEN;
const DOWN: &str = "model.layers.0.mlp.down_proj.weight";
const O_PROJ: &str = "model.layers.3.self_attn.o_proj.weight";
const EMBED: &str = "model.embed_tokens.weight";

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

/// The mini model and its prompt files.
fn workspace() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let model = fake_qwen2::write_fake_qwen2(dir.path()).unwrap();
    fake_qwen2::write_prompt_files(dir.path()).unwrap();
    (dir, model)
}

fn extract(dir: &Path, out: &str, extra: &[&str]) -> (i32, Value) {
    let mut args = vec![
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
        out,
    ];
    args.extend(extra);
    surgeon(dir, &args)
}

/// Shape and values of a tensor of a `.safetensors` file (f32 or bf16).
fn tensor(file: &Path, name: &str) -> (Vec<usize>, Vec<f32>) {
    let model = ReadOnlyModel::open(file).unwrap();
    let view = model.tensor(name).unwrap();
    let values = match view.info.dtype {
        DType::F32 => view
            .bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        DType::BF16 => view
            .bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()).to_f32())
            .collect(),
        ref other => panic!("{name}: dtype {other}"),
    };
    (view.info.shape.clone(), values)
}

/// A `.safetensors` file with the given tensors, in f32 or bf16.
fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)], bf16: bool) {
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for v in values {
            match bf16 {
                true => data.extend(half::bf16::from_f32(*v).to_le_bytes()),
                false => data.extend(v.to_le_bytes()),
            }
        }
        let dtype = if bf16 { "BF16" } else { "F32" };
        header.insert(
            name.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut text = Value::Object(header).to_string();
    while !(8 + text.len()).is_multiple_of(8) {
        text.push(' ');
    }
    let mut file = (text.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(text.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

/// Every file of the model directory, to prove that nothing was written.
fn snapshot(model: &Path) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(model)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_file())
        .map(|path| {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            (name, std::fs::read(&path).unwrap())
        })
        .collect();
    files.sort();
    files
}

/// A `project_out` of every residual-stream writer with the tensors `a` and `b` of `file`.
fn patch(file: &str, alpha: f64, extra: Value) -> String {
    let mut op = json!({"id": "erase", "layers": "all",
        "projections": ["down_proj", "o_proj", "embed_tokens"], "operation": "project_out",
        "vector_file": file, "vector_tensor": "a", "vector_tensor_b": "b", "alpha": alpha});
    for (key, value) in extra.as_object().unwrap() {
        op[key] = value.clone();
    }
    json!({"schema_version": 1, "operations": [op]}).to_string()
}

fn apply(dir: &Path, patch_text: &str, extra: &[&str]) -> (i32, Value) {
    std::fs::write(dir.join("patch.json"), patch_text).unwrap();
    let mut args = vec!["apply", "--model", "qwen2-mini", "--patch", "patch.json"];
    args.extend(extra);
    surgeon(dir, &args)
}

/// `max |v_t · column|` of a `[hidden, in]` writer, or `max |v_t · row|` of the embeddings.
fn max_read(v: &[f32], shape: &[usize], w: &[f32], rows_in_hidden: bool) -> f32 {
    let (rows, cols) = (shape[0], shape[1]);
    let mut worst = 0.0f32;
    for direction in v.chunks_exact(HIDDEN) {
        if rows_in_hidden {
            for j in 0..cols {
                let read: f32 = (0..rows).map(|i| direction[i] * w[i * cols + j]).sum();
                worst = worst.max(read.abs());
            }
        } else {
            for row in w.chunks_exact(cols) {
                let read: f32 = row.iter().zip(direction).map(|(x, d)| x * d).sum();
                worst = worst.max(read.abs());
            }
        }
    }
    worst
}

const SUBSPACE: &str = "v.safetensors.subspace.safetensors";

/// `extract --method svd --rank 3`, its envelope and the `[3, hidden]` directions.
fn svd_rank_three(dir: &Path) -> (Value, Vec<f32>) {
    let (code, envelope) = extract(dir, "v.safetensors", &["--method", "svd", "--rank", "3"]);
    assert_eq!(code, 0, "{envelope}");
    let (shape, v) = tensor(&dir.join(SUBSPACE), "a");
    assert_eq!(shape, [3, HIDDEN]);
    (envelope, v)
}

#[test]
fn svd_rank_k_is_extracted_then_erased_from_every_writer_in_one_pass() {
    let (dir, model) = workspace();
    let dir = dir.path();
    let (envelope, v) = svd_rank_three(dir);
    let data = &envelope["data"];
    assert!(
        data["sidecar_paths"][0]
            .as_str()
            .unwrap()
            .ends_with(SUBSPACE),
        "{data}"
    );
    let diagnostics = &data["method_diagnostics"]["subspace"];
    assert_eq!(diagnostics["rank"], 3);
    assert!(diagnostics["gram_BtA_error"].as_f64().unwrap() < 1e-5);
    for key in ["sigma_t", "explained_variance_cumulative", "d_prime"] {
        assert_eq!(diagnostics[key].as_array().unwrap().len(), 3, "{key}");
    }
    assert!(diagnostics["holdout_cross_cov_residual"].is_null());
    // Six pairs for three directions: said.
    assert!(envelope["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|w| w.as_str().unwrap().contains("n/4")));
    // a = b = V, and `direction` is its first row: the rank-1 output is still there.
    assert_eq!(
        tensor(&dir.join(SUBSPACE), "b"),
        (vec![3, HIDDEN], v.clone())
    );
    let (shape, direction) = tensor(&dir.join("v.safetensors"), "direction");
    assert_eq!((shape, &direction[..]), (vec![HIDDEN], &v[..HIDDEN]));
    let stats = std::fs::read_to_string(dir.join("v.safetensors.stats.json")).unwrap();
    assert!(stats.contains("\"subspace\"") && stats.contains("gram_BtA_error"));

    // Dry run: the rank is reported, nothing is written.
    let before = snapshot(&model);
    let erase = patch(SUBSPACE, 1.0, json!({"rank": 3}));
    let (code, dry) = apply(dir, &erase, &["--dry-run"]);
    assert_eq!(code, 0, "{dry}");
    let op = &dry["data"]["report"]["operations"][0];
    assert_eq!(
        (&op["rank"], &op["vector_len"]),
        (&json!(3), &json!(HIDDEN))
    );
    assert_eq!(
        op["tensors"].as_array().unwrap().len(),
        2 * fake_qwen2::LAYERS + 1
    );
    assert_eq!(snapshot(&model), before);

    // One pass: no component of any writer is left along the three directions.
    let shard = model.join("model.safetensors");
    let (shape, down) = tensor(&shard, DOWN);
    assert!(
        max_read(&v, &shape, &down, true) > 1e-3,
        "something to erase"
    );
    let (code, applied) = apply(dir, &erase, &[]);
    assert_eq!(code, 0, "{applied}");
    assert_eq!(applied["data"]["report"]["operations"][0]["rank"], 3);
    for (name, rows_in_hidden) in [(DOWN, true), (O_PROJ, true), (EMBED, false)] {
        let (shape, w) = tensor(&shard, name);
        let left = max_read(&v, &shape, &w, rows_in_hidden);
        assert!(left < 1e-5, "{name}: {left}");
    }

    // Without --rank, svd writes what it always wrote: no subspace file.
    let (code, plain) = extract(dir, "one.safetensors", &["--method", "svd"]);
    assert_eq!(code, 0, "{plain}");
    assert!(!dir.join("one.safetensors.subspace.safetensors").exists());
    assert!(plain["data"]["method_diagnostics"]
        .get("subspace")
        .is_none());
    assert!(plain["data"].get("sidecar_paths").is_none());
}

#[test]
fn leace_with_labels_writes_a_projector_that_project_out_takes_as_is() {
    let (dir, model) = workspace();
    let dir = dir.path();
    // Three classes over the twelve prompts (positives first): two directions.
    std::fs::write(dir.join("labels.json"), "[0,1,2,0,1,2,0,1,2,0,1,2]").unwrap();
    let labels = ["--method", "leace", "--labels-file", "labels.json"];
    let (code, envelope) = extract(dir, "e.safetensors", &labels);
    assert_eq!(code, 0, "{envelope}");
    let file = dir.join("e.safetensors.leace.safetensors");
    let reported = envelope["data"]["leace_path"].as_str().unwrap();
    assert!(
        reported.ends_with("e.safetensors.leace.safetensors"),
        "{reported}"
    );
    for (name, shape) in [
        ("a", vec![2, HIDDEN]),
        ("b", vec![2, HIDDEN]),
        ("u_whitened", vec![2, HIDDEN]),
        ("mean", vec![HIDDEN]),
    ] {
        assert_eq!(tensor(&file, name).0, shape, "{name}");
    }
    let diagnostics = &envelope["data"]["method_diagnostics"];
    assert!(diagnostics["in_sample_cross_cov_ratio"].as_f64().unwrap() < 1e-6);
    assert_eq!(diagnostics["subspace"]["rank"], 2);
    assert!(diagnostics["subspace"]["gram_BtA_error"].as_f64().unwrap() < 1e-4);
    // Same inputs, same bytes.
    let first = std::fs::read(&file).unwrap();
    let (code, _) = extract(
        dir,
        "e.safetensors",
        &[&labels[..], &["--overwrite"]].concat(),
    );
    assert_eq!(code, 0);
    assert_eq!(std::fs::read(&file).unwrap(), first);

    // The patch names a and b as before; the rank comes from the tensors.
    let (code, applied) = apply(
        dir,
        &patch("e.safetensors.leace.safetensors", 1.0, json!({})),
        &[],
    );
    assert_eq!(code, 0, "{applied}");
    assert_eq!(applied["data"]["report"]["operations"][0]["rank"], 2);
    assert!(!applied["warnings"].to_string().contains("projecteur"));
    let (_, b) = tensor(&file, "b");
    let (shape, down) = tensor(&model.join("model.safetensors"), DOWN);
    assert!(max_read(&b, &shape, &down, true) < 1e-4, "bᵀ W' = 0");

    // One direction at most on request: the historical 1D tensors.
    let one = [&labels[..], &["--rank", "1"]].concat();
    let (code, envelope) = extract(dir, "one.safetensors", &one);
    assert_eq!(code, 0, "{envelope}");
    let (shape, _) = tensor(&dir.join("one.safetensors.leace.safetensors"), "a");
    assert_eq!(shape, [HIDDEN]);

    // What a method cannot do is refused with code 2, and nothing is written.
    std::fs::write(dir.join("short.json"), "[0,1]").unwrap();
    std::fs::write(dir.join("object.json"), "{\"a\": 1}").unwrap();
    for (extra, why) in [
        (&["--method", "leace", "--rank", "2"][..], "--labels-file"),
        (&["--rank", "2"][..], "mean_diff"),
        (
            &["--method", "svd", "--labels-file", "labels.json"][..],
            "leace",
        ),
        (
            &["--method", "leace", "--labels-file", "short.json"][..],
            "2 lignes pour 12",
        ),
        (
            &["--method", "leace", "--labels-file", "object.json"][..],
            "tableau JSON",
        ),
        (
            &["--method", "leace", "--labels-file", "absent.json"][..],
            "absent.json",
        ),
    ] {
        let (code, refused) = extract(dir, "refused.safetensors", extra);
        assert_eq!(code, 2, "{extra:?}: {refused}");
        let message = refused["error"]["message"].as_str().unwrap();
        assert!(message.contains(why), "{extra:?}: {message}");
        assert!(!dir.join("refused.safetensors").exists(), "{extra:?}");
    }
}

#[test]
fn the_rank_and_the_projector_are_checked_before_any_write() {
    let (dir, model) = workspace();
    let dir = dir.path();
    let (_, v) = svd_rank_three(dir);
    let before = snapshot(&model);
    let refused = |patch_text: &str, extra: &[&str], why: &str| {
        let (code, envelope) = apply(dir, patch_text, extra);
        assert_ne!(code, 0, "{envelope}");
        let error = envelope["error"].to_string();
        assert!(error.contains(why), "{why}: {error}");
        assert_eq!(snapshot(&model), before, "{why}: nothing written");
        code
    };
    // The declared rank is not the one of the tensors; the limit is the operator's.
    assert_eq!(
        refused(&patch(SUBSPACE, 1.0, json!({"rank": 2})), &[], "rank = 2"),
        3
    );
    refused(
        &patch(SUBSPACE, 1.0, json!({})),
        &["--max-rank", "2"],
        "plafond 2",
    );
    let (code, _) = apply(
        dir,
        &patch(SUBSPACE, 1.0, json!({})),
        &["--max-rank", "3", "--dry-run"],
    );
    assert_eq!(code, 0);
    // Fields of another operation, and a rank that is not one: invalid patch (code 2).
    let subtract = json!({"schema_version": 1, "operations": [{"id": "s", "layers": "all",
        "projections": ["down_proj"], "operation": "subtract", "vector_file": "v.safetensors",
        "vector_tensor": "direction", "alpha": 0.5, "rank": 1}]});
    assert_eq!(refused(&subtract.to_string(), &[], "operations[0].rank"), 2);
    assert_eq!(
        refused(&patch(SUBSPACE, 1.0, json!({"rank": 0})), &[], "rank"),
        2
    );

    // a = 2 V, b = V: BᵀA = 2 I is not a projector.
    let doubled: Vec<f32> = v.iter().map(|x| 2.0 * x).collect();
    let tensors = [
        ("a".to_owned(), vec![3, HIDDEN], doubled),
        ("b".to_owned(), vec![3, HIDDEN], v.clone()),
    ];
    write_safetensors(&dir.join("twice.safetensors"), &tensors, false);
    refused(
        &patch("twice.safetensors", 1.0, json!({})),
        &[],
        "projecteur",
    );
    let allowed = patch(
        "twice.safetensors",
        1.0,
        json!({"allow_non_projector": true}),
    );
    let (code, envelope) = apply(dir, &allowed, &["--dry-run"]);
    assert_eq!(code, 0, "{envelope}");
    assert!(
        envelope["warnings"].to_string().contains("projecteur"),
        "{envelope}"
    );
    // Below full strength nothing claims to be a projection.
    let (code, envelope) = apply(
        dir,
        &patch("twice.safetensors", 0.5, json!({})),
        &["--dry-run"],
    );
    assert_eq!(code, 0, "{envelope}");
    assert!(!envelope["warnings"].to_string().contains("projecteur"));
    assert_eq!(snapshot(&model), before);
}

/// Rewrites the single-file f32 model as two bf16 shards with an index.
fn shard_as_bf16(model: &Path) {
    let single = model.join("model.safetensors");
    let tensors: Vec<(String, Vec<usize>, Vec<f32>)> = read_index(&single)
        .unwrap()
        .tensors
        .iter()
        .map(|info| {
            let (shape, values) = tensor(&single, &info.name);
            (info.name.clone(), shape, values)
        })
        .collect();
    std::fs::remove_file(&single).unwrap();
    let names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    let (first, second) = tensors.split_at(tensors.len() / 2);
    write_safetensors(&model.join(names[0]), first, true);
    write_safetensors(&model.join(names[1]), second, true);
    let weight_map: serde_json::Map<String, Value> = first
        .iter()
        .map(|t| (t.0.clone(), json!(names[0])))
        .chain(second.iter().map(|t| (t.0.clone(), json!(names[1]))))
        .collect();
    let index = json!({"weight_map": weight_map}).to_string();
    std::fs::write(model.join("model.safetensors.index.json"), index).unwrap();
}

#[test]
fn a_sharded_bf16_model_is_erased_across_shards_and_restored_on_nan() {
    let (dir, model) = workspace();
    let dir = dir.path();
    let (_, v) = svd_rank_three(dir);
    shard_as_bf16(&model);
    let before = snapshot(&model);
    let shard_of = |name: &str| {
        let holds = |file: &str| {
            let index = read_index(&model.join(file)).unwrap();
            index.tensors.iter().any(|t| t.name == name)
        };
        let names = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ];
        model.join(names.iter().find(|file| holds(file)).unwrap())
    };
    assert_ne!(
        shard_of(DOWN),
        shard_of(O_PROJ),
        "the writers span both shards"
    );

    // An overflow in the second operation: both shards come back byte for byte.
    let huge = [
        ("a".to_owned(), vec![2, HIDDEN], vec![1e30f32; 2 * HIDDEN]),
        ("b".to_owned(), vec![2, HIDDEN], vec![1e30f32; 2 * HIDDEN]),
    ];
    write_safetensors(&dir.join("huge.safetensors"), &huge, false);
    let mut both: Value = serde_json::from_str(&patch(SUBSPACE, 1.0, json!({}))).unwrap();
    let overflow: Value = serde_json::from_str(&patch("huge.safetensors", 0.5, json!({}))).unwrap();
    let mut second = overflow["operations"][0].clone();
    second["id"] = json!("overflow");
    both["operations"].as_array_mut().unwrap().push(second);
    let (code, failed) = apply(dir, &both.to_string(), &[]);
    assert_eq!(code, 3, "{failed}");
    assert!(failed["error"].to_string().contains("NaN/Inf"), "{failed}");
    let restored: Vec<_> = snapshot(&model)
        .into_iter()
        .filter(|(name, _)| before.iter().any(|(known, _)| known == name))
        .collect();
    assert_eq!(restored, before, "rolled back");

    // The erasure alone: one rounding per bf16 element, on both shards.
    let (code, applied) = apply(dir, &patch(SUBSPACE, 1.0, json!({})), &[]);
    assert_eq!(code, 0, "{applied}");
    assert_eq!(applied["data"]["backups"].as_array().unwrap().len(), 2);
    for (name, rows_in_hidden) in [(DOWN, true), (O_PROJ, true), (EMBED, false)] {
        let (shape, w) = tensor(&shard_of(name), name);
        let left = max_read(&v, &shape, &w, rows_in_hidden);
        assert!(left < 5e-2, "{name}: {left}");
    }
}

/// `structuredContent` of `tools/call` requests to `surgeon mcp --allow-dir <dir>`.
fn mcp(dir: &Path, calls: &[(&str, Value)]) -> Vec<Value> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_surgeon"))
        .args(["mcp", "--allow-dir", dir.to_str().unwrap()])
        .current_dir(dir)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut lines = vec![
        json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "rank-k", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    ];
    for (i, (tool, arguments)) in calls.iter().enumerate() {
        lines.push(
            json!({"jsonrpc": "2.0", "id": i + 1, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments}}),
        );
    }
    let mut stdin = child.stdin.take().unwrap();
    for line in &lines {
        writeln!(stdin, "{line}").unwrap();
    }
    drop(stdin);
    let replies: Vec<Value> = BufReader::new(child.stdout.take().unwrap())
        .lines()
        .map(|l| serde_json::from_str(&l.unwrap()).unwrap())
        .collect();
    child.wait().unwrap();
    (1..=calls.len())
        .map(|id| replies.iter().find(|r| r["id"] == id).unwrap().clone())
        .collect()
}

#[test]
fn mcp_tools_take_the_rank_and_keep_every_path_in_the_sandbox() {
    let (dir, model) = workspace();
    let dir = dir.path();
    let outside = tempfile::tempdir().unwrap();
    let foreign = outside.path().join("labels.json");
    std::fs::write(&foreign, "[0,1,2,0,1,2,0,1,2,0,1,2]").unwrap();
    std::fs::copy(&foreign, dir.join("labels.json")).unwrap();
    let extract = |extra: Value| {
        let mut args = json!({"model": "qwen2-mini", "pos_prompts_file": "pos.json",
            "neg_prompts_file": "neg.json", "layer": 2});
        for (key, value) in extra.as_object().unwrap() {
            args[key] = value.clone();
        }
        ("surgeon_extract", args)
    };
    let operation = |extra: Value| -> Value {
        let text = patch("m.safetensors.subspace.safetensors", 1.0, extra);
        serde_json::from_str(&text).unwrap()
    };
    let before = snapshot(&model);
    let replies = mcp(
        dir,
        &[
            extract(json!({"method": "svd", "rank": 2, "out": "m.safetensors"})),
            extract(
                json!({"method": "leace", "labels_file": "labels.json", "out": "l.safetensors"}),
            ),
            extract(
                json!({"method": "leace", "labels_file": foreign.to_str().unwrap(),
                "out": "x.safetensors"}),
            ),
            extract(json!({"method": "svd", "rank": 65, "out": "y.safetensors"})),
            extract(json!({"rank": 2, "out": "z.safetensors"})),
            (
                "surgeon_apply",
                json!({"model": "qwen2-mini", "dry_run": true,
                "patch": operation(json!({"rank": 2}))}),
            ),
            (
                "surgeon_apply",
                json!({"model": "qwen2-mini", "dry_run": true,
                "patch": operation(json!({"rank": 3}))}),
            ),
            (
                "surgeon_apply",
                json!({"model": "qwen2-mini", "dry_run": true,
                "patch": operation(json!({"max_rank": 128}))}),
            ),
        ],
    );
    let content = |i: usize| &replies[i]["result"]["structuredContent"];
    assert_eq!(content(0)["ok"], true, "{}", replies[0]);
    assert_eq!(
        content(0)["data"]["method_diagnostics"]["subspace"]["rank"],
        2
    );
    assert_eq!(
        tensor(&dir.join("m.safetensors.subspace.safetensors"), "a").0,
        [2, HIDDEN]
    );
    assert_eq!(content(1)["ok"], true, "{}", replies[1]);
    assert_eq!(
        tensor(&dir.join("l.safetensors.leace.safetensors"), "b").0,
        [2, HIDDEN]
    );
    // A labels file outside the allowed directories is never read.
    assert_eq!(
        content(2)["error"]["kind"],
        "path_not_allowed",
        "{}",
        replies[2]
    );
    // The schema bounds the rank; mean_diff has no subspace.
    assert!(
        replies[3].get("error").is_some() || content(3)["ok"] == false,
        "{}",
        replies[3]
    );
    assert_eq!(content(4)["ok"], false, "{}", replies[4]);
    assert!(content(4)["error"].to_string().contains("mean_diff"));
    for name in ["x", "y", "z"] {
        assert!(!dir.join(format!("{name}.safetensors")).exists(), "{name}");
    }
    // The patch carries the rank; an agent cannot raise the limit.
    assert_eq!(content(5)["ok"], true, "{}", replies[5]);
    assert_eq!(content(5)["data"]["report"]["operations"][0]["rank"], 2);
    assert_eq!(content(6)["ok"], false, "{}", replies[6]);
    assert!(content(6)["error"].to_string().contains("rank = 3"));
    assert!(
        replies[7].get("error").is_some() || content(7)["ok"] == false,
        "{}",
        replies[7]
    );
    assert_eq!(
        snapshot(&model),
        before,
        "dry runs and refusals write nothing"
    );
}
