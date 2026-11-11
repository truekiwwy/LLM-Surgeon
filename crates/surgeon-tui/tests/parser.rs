//! End-to-end parser tests: hand-built SafeTensors (single, sharded, with a
//! config.json) and GGUF files go through `ModelIndex::open` and must expose
//! the right names, dtypes, byte ranges, roles and architecture family.

use std::path::Path;

use surgeon_tui::model::dtype::DType;
use surgeon_tui::model::metrics;
use surgeon_tui::model::roles::{Exactness, MoeKind, Role};
use surgeon_tui::model::ModelIndex;

/// Builds a minimal safetensors file: u64 header length, JSON header, data.
fn build_safetensors(entries: &[(&str, &str, Vec<u64>, Vec<u8>)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in entries {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            (*name).to_owned(),
            serde_json::json!({
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [start, data.len()],
            }),
        );
    }
    let header_text = serde_json::Value::Object(header).to_string();
    let mut out = (header_text.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header_text.as_bytes());
    out.extend_from_slice(&data);
    out
}

fn f32_values(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[test]
fn a_single_safetensors_file_opens_with_roles_and_offsets() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("model.safetensors");
    let bytes = build_safetensors(&[
        ("model.layers.0.mlp.down_proj.weight", "F32", vec![2, 4], f32_values(&[0.0; 8])),
        ("model.layers.0.self_attn.o_proj.weight", "F32", vec![4, 4], f32_values(&[0.0; 16])),
        ("model.embed_tokens.weight", "F32", vec![4, 2], f32_values(&[0.0; 8])),
    ]);
    std::fs::write(&path, bytes).expect("write");

    let index = ModelIndex::open(&path).expect("open");
    assert_eq!(index.tensors.len(), 3);
    assert_eq!(index.format.to_string(), "safetensors");
    // Sorted by name.
    assert_eq!(index.tensors[0].name, "model.embed_tokens.weight");
    assert_eq!(index.tensors[0].role, Role::Embed);
    assert_eq!(index.tensors[1].name, "model.layers.0.mlp.down_proj.weight");
    assert_eq!(index.tensors[1].role, Role::MlpOut);
    assert_eq!(index.tensors[1].layer, Some(0));
    assert_eq!(index.tensors[2].role, Role::AttnOut);
    // Byte ranges follow the header offsets.
    assert_eq!(index.tensors[0].byte_len, 32);
    assert_eq!(index.tensors[1].byte_len, 32);
    // Metrics decode the mapped bytes correctly.
    let entry = &index.tensors[1];
    let stats = metrics::compute(&entry.dtype, &entry.shape, &surgeon_tui::model::read_entry_bytes(entry).expect("bytes"));
    assert!(stats.available);
    assert_eq!(stats.numel, 8);
}

#[test]
fn a_sharded_directory_and_its_config_are_honored() {
    let dir = tempfile::tempdir().expect("tempdir");
    let shard1 = build_safetensors(&[(
        "model.layers.0.mlp.gate_proj.weight",
        "F32",
        vec![2, 2],
        f32_values(&[0.0; 4]),
    )]);
    let shard2 = build_safetensors(&[(
        "model.layers.1.mlp.down_proj.weight",
        "F32",
        vec![2, 2],
        f32_values(&[0.0; 4]),
    )]);
    std::fs::write(dir.path().join("model-00001.safetensors"), shard1).expect("write");
    std::fs::write(dir.path().join("model-00002.safetensors"), shard2).expect("write");
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        r#"{"weight_map": {"model.layers.0.mlp.gate_proj.weight": "model-00001.safetensors", "model.layers.1.mlp.down_proj.weight": "model-00002.safetensors"}}"#,
    )
    .expect("write");
    std::fs::write(dir.path().join("config.json"), r#"{"model_type": "gemma3"}"#).expect("write");

    let index = ModelIndex::open(dir.path()).expect("open");
    assert_eq!(index.tensors.len(), 2);
    assert_eq!(index.family.as_deref(), Some("gemma3"));
    assert_eq!(
        Exactness::of_family(index.family.as_deref()),
        Exactness::Approximate
    );
    let gate = index
        .tensors
        .iter()
        .find(|t| t.name.ends_with("gate_proj.weight"))
        .expect("gate");
    assert_eq!(gate.role, Role::MlpGate);
    assert!(gate.file.ends_with("model-00001.safetensors"));
}

#[test]
fn an_empty_directory_is_a_typed_error() {
    let dir = tempfile::tempdir().expect("tempdir");
    let error = ModelIndex::open(dir.path()).expect_err("empty dir");
    assert!(matches!(error, surgeon_tui::model::ModelError::UnknownFormat(_)));
}

fn build_gguf() -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&2u64.to_le_bytes()); // tensors
    out.extend_from_slice(&1u64.to_le_bytes()); // kvs
    let mut kv = Vec::new();
    kv.extend_from_slice(&20u64.to_le_bytes());
    kv.extend_from_slice(b"general.architecture");
    kv.extend_from_slice(&8u32.to_le_bytes());
    kv.extend_from_slice(&8u64.to_le_bytes());
    kv.extend_from_slice(b"qwen3_moe");
    out.extend_from_slice(&kv);
    // token_embd.weight [8] F32 @ 0
    out.extend_from_slice(&17u64.to_le_bytes());
    out.extend_from_slice(b"token_embd.weight");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&8u64.to_le_bytes());
    out.extend_from_slice(&0i32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    // blk.3.ffn_down_exps.weight [64, 2] fastest-first, Q8_0, fused experts @ 32
    out.extend_from_slice(&26u64.to_le_bytes());
    out.extend_from_slice(b"blk.3.ffn_down_exps.weight");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&64u64.to_le_bytes());
    out.extend_from_slice(&2u64.to_le_bytes());
    out.extend_from_slice(&8i32.to_le_bytes());
    out.extend_from_slice(&32u64.to_le_bytes());
    while out.len() % 32 != 0 {
        out.push(0);
    }
    for v in [1.0f32, -1.0, 0.5, -0.5, 0.25, -0.25, 0.125, -0.125] {
        out.extend_from_slice(&v.to_le_bytes());
    }
    let mut block = Vec::new();
    block.extend_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
    block.extend_from_slice(&[1u8; 32]);
    out.extend_from_slice(&block);
    out.extend_from_slice(&block);
    out
}

#[test]
fn a_gguf_file_opens_with_family_and_fused_experts() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("m.gguf");
    std::fs::write(&path, build_gguf()).expect("write");

    let index = ModelIndex::open(&path).expect("open");
    assert_eq!(index.format.to_string(), "gguf");
    assert_eq!(index.family.as_deref(), Some("qwen3_moe"));
    assert_eq!(index.tensors.len(), 2);

    let fused = &index.tensors[1];
    assert_eq!(fused.name, "blk.3.ffn_down_exps.weight");
    assert_eq!(fused.role, Role::MlpOut);
    assert_eq!(fused.moe, MoeKind::Fused);
    assert_eq!(fused.layer, Some(3));
    assert_eq!(fused.dtype, DType::Q8_0);
    assert_eq!(fused.byte_len, 2 * 34);
    // Rows: dims written fastest-first [64, 2] are read as [2, 64].
    assert_eq!(fused.shape, vec![2, 64]);

    let embed = &index.tensors[0];
    assert_eq!(embed.role, Role::Embed);
    let stats = metrics::compute(&embed.dtype, &embed.shape, &surgeon_tui::model::read_entry_bytes(embed).expect("bytes"));
    assert_eq!(stats.min, -1.0);
    assert_eq!(stats.max, 1.0);
    // Exact ablation on qwen3 MoE writers.
    assert_eq!(
        Exactness::for_entry(index.family.as_deref(), fused),
        Some(Exactness::Exact)
    );
}

#[test]
fn a_missing_model_reports_a_clear_error() {
    let error = ModelIndex::open(Path::new("definitely/not/here.gguf")).expect_err("missing");
    assert!(matches!(error, surgeon_tui::model::ModelError::NotFound(_)));
}
