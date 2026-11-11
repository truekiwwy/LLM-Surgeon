//! P16: architecture descriptions (`src/arch`), the forward adapters (`extract::forward`)
//! and the exactness of a weight ablation per architecture.
//!
//! Every name fixture cites the file it was read from (2026-11-11). The exactness
//! classification is measured twice: on the real forward of the architectures that have
//! one (Llama, Qwen2, Qwen3: the direction must vanish from the residual stream after every
//! layer), and on a residual stack that reproduces each built-in's topology (norm kind,
//! post-sublayer norm, parallel block) for the others, which is all the classification
//! depends on. The latter is said "topology" in `arch list`, never "forward".

#[path = "support/gguf.rs"]
mod support;

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::{json, Value};

use support::{
    codec, decode_tensor, gguf_bytes, requant_keep_scales, sha, tensor_data, Kv, Rng, T,
};
use weight_surgeon::arch::{self, Exactness, Format, Layout, Role};
use weight_surgeon::engine::{self, DType as EDType, SubspaceProjection};
use weight_surgeon::extract::{parse_config, required_tensors_full, HookedDecoder, KvCache};
use weight_surgeon::patch::gguf;

// ------------------------------------------------------------------ helpers

fn normal(rng: &mut Rng, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| scale * rng.normal()).collect()
}

fn unit(rng: &mut Rng, n: usize) -> Vec<f32> {
    let v = normal(rng, n, 1.0);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn surgeon() -> Command {
    Command::new(env!("CARGO_BIN_EXE_surgeon"))
}

fn run_json(args: &[&str]) -> (i32, Value) {
    let out = surgeon().arg("--json").args(args).output().unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert_eq!(stdout.lines().count(), 1, "{stdout}{stderr}");
    (
        out.status.code().unwrap(),
        serde_json::from_str(&stdout).unwrap(),
    )
}

/// A single-file F32 SafeTensors model, tensors in the given order.
/// `(name, shape, values)` of an F32 tensor.
type Named = (&'static str, Vec<usize>, Vec<f32>);

fn write_safetensors(path: &Path, tensors: &[Named]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(
            (*name).to_owned(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = Value::Object(header).to_string();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header.as_bytes());
    file.extend(data);
    std::fs::write(path, file).unwrap();
}

/// The F32 values of a tensor of a single-file SafeTensors model.
fn read_tensor(path: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    st.tensor(name)
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn from_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// `a = b = dirs` (`[k, len]`) in `dirs.safetensors`, and the full ablation recipe.
fn write_ablation(dir: &Path, dirs: &[f32], k: usize, alpha: f32) -> PathBuf {
    let len = dirs.len() / k;
    write_safetensors(
        &dir.join("dirs.safetensors"),
        &[
            ("a", vec![k, len], dirs.to_vec()),
            ("b", vec![k, len], dirs.to_vec()),
        ],
    );
    let patch = dir.join("ablate.json");
    std::fs::write(
        &patch,
        json!({"schema_version": 1, "backup": true, "operations": [{
            "id": "ablate", "layers": "all",
            "projections": ["embed_tokens", "o_proj", "down_proj"],
            "operation": "project_out", "vector_file": "dirs.safetensors",
            "vector_tensor": "a", "vector_tensor_b": "b", "alpha": alpha}]})
        .to_string(),
    )
    .unwrap();
    patch
}

// ------------------------------------------------------------------ detection and names

/// `model.layers.0.*` of Qwen/Qwen3-30B-A3B, `model.safetensors.index.json`.
const QWEN3_MOE_NAMES: [&str; 13] = [
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.self_attn.q_norm.weight",
    "model.layers.0.self_attn.k_norm.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.mlp.gate.weight",
    "model.layers.0.mlp.experts.0.down_proj.weight",
    "model.layers.0.mlp.experts.127.down_proj.weight",
];

/// deepseek-ai/DeepSeek-V2-Lite, `model.safetensors.index.json` (layer 0 dense, layer 1
/// MoE, MLA attention).
const DEEPSEEK_NAMES: [&str; 10] = [
    "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.layers.1.self_attn.kv_b_proj.weight",
    "model.layers.1.self_attn.o_proj.weight",
    "model.layers.1.mlp.gate.weight",
    "model.layers.1.mlp.shared_experts.down_proj.weight",
    "model.layers.1.mlp.shared_experts.gate_proj.weight",
    "model.layers.1.mlp.experts.0.down_proj.weight",
    "model.layers.1.mlp.experts.0.up_proj.weight",
];

fn role(
    spec: &str,
    name: &str,
    format: Format,
) -> Option<(Role, Option<usize>, Option<usize>, Layout)> {
    arch::builtin(spec)
        .unwrap()
        .role_of(name, format)
        .map(|r| (r.role, r.layer, r.expert, r.layout))
}

#[test]
fn moe_names_resolve_per_expert_shared_fused_and_router() {
    use Layout::*;
    use Role::*;
    let st = Format::SafeTensors;
    assert_eq!(
        role("qwen3_moe", QWEN3_MOE_NAMES[0], st),
        Some((Embed, None, None, Rows))
    );
    assert_eq!(
        role("qwen3_moe", QWEN3_MOE_NAMES[1], st),
        None,
        "the head is a reader"
    );
    assert_eq!(
        role("qwen3_moe", QWEN3_MOE_NAMES[5], st),
        Some((AttnOut, Some(0), None, OutIn))
    );
    assert_eq!(role("qwen3_moe", QWEN3_MOE_NAMES[6], st), None);
    assert_eq!(
        role("qwen3_moe", QWEN3_MOE_NAMES[10], st),
        Some((Router, Some(0), None, OutIn))
    );
    assert_eq!(
        role("qwen3_moe", QWEN3_MOE_NAMES[12], st),
        Some((MlpOut, Some(0), Some(127), OutIn))
    );
    // transformers v5 fused parameter, [E, hidden, inter].
    assert_eq!(
        role("qwen3_moe", "model.layers.4.mlp.experts.down_proj", st),
        Some((MlpOut, Some(4), None, ExpertsOutIn))
    );
    let deepseek: Vec<_> = DEEPSEEK_NAMES
        .iter()
        .map(|n| role("deepseek", n, st))
        .collect();
    assert_eq!(deepseek[0], None, "MLA projections are readers");
    assert_eq!(
        deepseek[2],
        Some((MlpOut, Some(0), None, OutIn)),
        "dense first layer"
    );
    assert_eq!(deepseek[5], Some((Router, Some(1), None, OutIn)));
    assert_eq!(
        deepseek[6],
        Some((MlpOut, Some(1), None, OutIn)),
        "shared experts"
    );
    assert_eq!(deepseek[7], Some((MlpGate, Some(1), None, OutIn)));
    assert_eq!(deepseek[8], Some((MlpOut, Some(1), Some(0), OutIn)));
    assert_eq!(deepseek[9], Some((MlpUp, Some(1), Some(0), OutIn)));
}

#[test]
fn dense_names_of_each_family() {
    use Role::*;
    let st = Format::SafeTensors;
    // modeling_falcon.py: transformer.h.{i}.self_attention.dense, mlp.dense_4h_to_h.
    assert_eq!(
        role("falcon", "transformer.word_embeddings.weight", st).map(|r| r.0),
        Some(Embed)
    );
    assert_eq!(
        role("falcon", "transformer.h.31.self_attention.dense.weight", st).map(|r| (r.0, r.1)),
        Some((AttnOut, Some(31)))
    );
    assert_eq!(
        role("falcon", "transformer.h.0.mlp.dense_4h_to_h.weight", st).map(|r| r.0),
        Some(MlpOut)
    );
    assert_eq!(
        role("falcon", "transformer.h.0.mlp.dense_h_to_4h.weight", st).map(|r| r.0),
        Some(MlpUp)
    );
    // modeling_phi3.py: fused qkv_proj and gate_up_proj are not targets.
    assert_eq!(
        role("phi3", "model.layers.0.mlp.gate_up_proj.weight", st),
        None
    );
    assert_eq!(
        role("phi3", "model.layers.0.mlp.down_proj.weight", st).map(|r| r.0),
        Some(MlpOut)
    );
    // Gemma 3 multimodal checkpoints nest the text model (modeling_gemma3.py).
    assert_eq!(
        role(
            "gemma3",
            "model.language_model.layers.7.self_attn.o_proj.weight",
            st
        )
        .map(|r| r.1),
        Some(Some(7))
    );
    assert_eq!(
        role(
            "gemma2",
            "model.layers.0.post_feedforward_layernorm.weight",
            st
        ),
        None
    );
    // GGUF names are whole names (constants.py TENSOR_NAMES).
    let g = Format::Gguf;
    assert_eq!(
        role("qwen2", "blk.3.attn_output.weight", g).map(|r| (r.0, r.1)),
        Some((AttnOut, Some(3)))
    );
    assert_eq!(role("qwen2", "v.blk.3.attn_output.weight", g), None);
    assert_eq!(
        role("qwen3_moe", "blk.2.ffn_down_exps.weight", g),
        Some((MlpOut, Some(2), None, Layout::ExpertsOutIn))
    );
    assert_eq!(
        role("qwen2_moe", "blk.2.ffn_down_shexp.weight", g).map(|r| r.0),
        Some(MlpOut)
    );
    assert_eq!(
        role("deepseek", "blk.0.ffn_gate_inp.weight", g).map(|r| r.0),
        Some(Router)
    );
    assert_eq!(
        role("phi3", "blk.0.ffn_up.weight", g),
        None,
        "fused gate+up in llama.cpp"
    );
}

#[test]
fn detection_from_config_gguf_and_fallback() {
    let r = arch::detect_config(&json!({"model_type": "mistral"}));
    assert_eq!((r.spec.id.as_str(), r.detected()), ("llama", true));
    let r =
        arch::detect_config(&json!({"model_type": "x", "architectures": ["Gemma2ForCausalLM"]}));
    assert_eq!(
        (r.spec.id.as_str(), r.source),
        ("gemma2", arch::Source::ConfigArchitectures)
    );
    let r = arch::detect_config(&json!({"model_type": "gpt2"}));
    assert_eq!((r.spec.id.as_str(), r.detected()), (arch::LEGACY_ID, false));
    assert_eq!(r.declared.as_deref(), Some("gpt2"));
    assert_eq!(arch::detect_gguf(Some("command-r")).spec.id, "command-r");
    assert!(!arch::detect_gguf(Some("mamba")).detected());
    // Every built-in is valid against its own rules and the published schema.
    let schema = arch::schema();
    let validator = jsonschema::validator_for(&schema).unwrap();
    for spec in arch::builtins() {
        spec.validate().unwrap();
        assert!(validator.is_valid(&json!(spec)), "{}", spec.id);
        assert!(!spec.sources.is_empty(), "{} cites its sources", spec.id);
    }
    // The ids with a forward are the model types the forward accepts.
    for t in weight_surgeon::extract::SUPPORTED_MODEL_TYPES {
        let id = arch::detect_config(&json!({"model_type": t})).spec.id;
        assert!(arch::FORWARD_IDS.contains(&id.as_str()), "{t} -> {id}");
    }
}

#[test]
fn selector_results_are_unchanged_on_llama_and_qwen2_names() {
    // The historical functions now go through the default spec.
    let mut names: Vec<String> = vec!["model.embed_tokens.weight".into(), "lm_head.weight".into()];
    for l in (0..3).rev() {
        for t in [
            "mlp.down_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "self_attn.o_proj",
            "self_attn.q_proj",
        ] {
            names.push(format!("model.layers.{l}.{t}.weight"));
        }
        names.push(format!("model.layers.{l}.self_attn.q_proj.bias"));
    }
    names.push("model.language_model.layers.5.mlp.down_proj.weight".into());
    let index = engine::ModelIndex {
        header_len: 0,
        tensors: names
            .iter()
            .map(|name| engine::TensorInfo {
                name: name.clone(),
                dtype: EDType::BF16,
                shape: vec![1, 1],
                byte_start: 0,
                byte_end: 2,
            })
            .collect(),
    };
    let found =
        |v: Vec<&engine::TensorInfo>| v.into_iter().map(|t| t.name.clone()).collect::<Vec<_>>();
    assert_eq!(
        found(engine::find_mlp_tensors(
            &index,
            &engine::LayerSpec::List(vec![1, 5]),
            &[engine::MlpProj::Down, engine::MlpProj::Gate]
        )),
        [
            "model.layers.1.mlp.gate_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            "model.language_model.layers.5.mlp.down_proj.weight"
        ]
    );
    assert_eq!(
        found(engine::find_attn_out_tensors(
            &index,
            &engine::LayerSpec::All
        )),
        [
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.1.self_attn.o_proj.weight",
            "model.layers.2.self_attn.o_proj.weight"
        ]
    );
    assert_eq!(
        engine::find_embed_tokens(&index).map(|t| t.name.as_str()),
        Some("model.embed_tokens.weight")
    );
    // Qwen2's own spec selects the same tensors as the default one.
    let roles = [
        Role::Embed,
        Role::MlpGate,
        Role::MlpUp,
        Role::MlpOut,
        Role::AttnOut,
    ];
    let by = |id: &str| {
        engine::find_by_roles(
            &index,
            arch::builtin(id).unwrap(),
            &roles,
            &engine::LayerSpec::All,
        )
        .into_iter()
        .map(|(t, _)| t.name.clone())
        .collect::<Vec<_>>()
    };
    assert_eq!(by("qwen2"), by(arch::LEGACY_ID));
    assert_eq!(by("qwen2").len(), 1 + 3 * 4 + 1);
}

// ------------------------------------------------------------------ forward fidelity vs candle

const TINY: &str = r#"{"model_type":"llama","vocab_size":64,"hidden_size":32,
    "intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":4,
    "num_key_value_heads":2,"max_position_embeddings":128,"rms_norm_eps":1e-6,
    "rope_theta":10000.0,"tie_word_embeddings":false,"hidden_act":"silu",
    "max_window_layers":2,"use_sliding_window":false,"attention_bias":false,
    "sliding_window":null"#;

fn tiny(model_type: &str, extra: &str) -> String {
    format!(
        "{}{extra}}}",
        TINY.replace("\"llama\"", &format!("\"{model_type}\""))
    )
}

fn weights(cfg_json: &str, seed: u64) -> HashMap<String, Tensor> {
    let cfg = parse_config(cfg_json).unwrap();
    let mut rng = Rng(seed);
    required_tensors_full(&cfg)
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let values: Vec<f32> = match name.contains("norm") {
                true => normal(&mut rng, n, 0.1).iter().map(|x| 1.0 + x).collect(),
                false => normal(&mut rng, n, 0.08),
            };
            (name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap())
        })
        .collect()
}

const IDS: [u32; 7] = [3, 17, 42, 5, 63, 0, 29];

fn ours_last(cfg_json: &str, w: &HashMap<String, Tensor>) -> Vec<f32> {
    let cfg = parse_config(cfg_json).unwrap();
    let model =
        HookedDecoder::from_tensors_full(w.clone(), &cfg, DType::F32, &Device::Cpu).unwrap();
    let ids = Tensor::new(&IDS[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let logits = model.forward_logits(&ids).unwrap();
    logits.get(IDS.len() - 1).unwrap().to_vec1().unwrap()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn ids() -> Tensor {
    Tensor::new(&IDS[..], &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap()
}

#[test]
fn llama_forward_matches_candle_llama() {
    use candle_transformers::models::llama;
    for extra in [
        "",
        // Llama 3.1-style RoPE frequency scaling.
        r#","rope_scaling":{"rope_type":"llama3","factor":8.0,"low_freq_factor":1.0,
           "high_freq_factor":4.0,"original_max_position_embeddings":16}"#,
    ] {
        let json = tiny("llama", extra);
        let w = weights(&json, 11);
        let reference: llama::LlamaConfig = serde_json::from_str(&json).unwrap();
        let reference = reference.into_config(false);
        let vb = VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
        let model = llama::Llama::load(vb, &reference).unwrap();
        let mut cache = llama::Cache::new(false, DType::F32, &reference, &Device::Cpu).unwrap();
        let theirs: Vec<f32> = model
            .forward(&ids(), 0, &mut cache)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let d = max_diff(&ours_last(&json, &w), &theirs);
        assert!(d < 1e-4, "llama {extra}: {d}");
    }
}

#[test]
fn mistral_forward_matches_candle_mistral() {
    use candle_transformers::models::mistral;
    let json = tiny("mistral", r#","head_dim":8"#)
        .replace(r#""sliding_window":null"#, r#""sliding_window":4"#);
    let w = weights(&json, 12);
    let reference: mistral::Config = serde_json::from_str(&json).unwrap();
    let vb = VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
    let mut model = mistral::Model::new(&reference, vb).unwrap();
    let theirs: Vec<f32> = model
        .forward(&ids(), 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let d = max_diff(&ours_last(&json, &w), &theirs);
    assert!(d < 1e-4, "mistral: {d}");
}

#[test]
fn qwen3_forward_matches_candle_qwen3_with_a_wider_head_dim() {
    use candle_transformers::models::qwen3;
    // head_dim 12 != hidden / heads = 8: the attention width is heads * head_dim.
    let json = tiny("qwen3", r#","head_dim":12"#);
    let w = weights(&json, 13);
    assert!(w.contains_key("model.layers.0.self_attn.q_norm.weight"));
    let reference: qwen3::Config = serde_json::from_str(&json).unwrap();
    let vb = VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
    let mut model = qwen3::ModelForCausalLM::new(&reference, vb).unwrap();
    let theirs: Vec<f32> = model
        .forward(&ids(), 0)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let d = max_diff(&ours_last(&json, &w), &theirs);
    assert!(d < 1e-4, "qwen3: {d}");
}

#[test]
fn moe_and_unknown_types_have_no_forward() {
    for t in ["qwen3_moe", "qwen2_moe", "deepseek_v3", "gemma2", "falcon"] {
        let err = parse_config(&tiny(t, "")).unwrap_err();
        assert_eq!(err.kind(), "unsupported_architecture", "{t}");
    }
}

// ------------------------------------------------------------------ exactness: real forwards

/// `‖Bᵀ x‖ / ‖x‖` maximal over the positions of the residual stream after every layer.
fn residual_component(cfg_json: &str, w: HashMap<String, Tensor>, b: &[f32]) -> f64 {
    let cfg = parse_config(cfg_json).unwrap();
    let model = HookedDecoder::from_tensors_full(w, &cfg, DType::F32, &Device::Cpu).unwrap();
    let mut cache = KvCache::new(cfg.num_hidden_layers);
    let mut worst = 0.0f64;
    model
        .forward_cached(
            &ids(),
            &mut cache,
            cfg.num_hidden_layers,
            &mut |_, resid| {
                let rows: Vec<Vec<f32>> = resid.squeeze(0)?.to_vec2()?;
                for row in rows {
                    let dot: f64 = row
                        .iter()
                        .zip(b)
                        .map(|(x, y)| f64::from(*x) * f64::from(*y))
                        .sum();
                    let norm = row
                        .iter()
                        .map(|x| f64::from(*x).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    worst = worst.max(dot.abs() / norm);
                }
                Ok(resid)
            },
        )
        .unwrap();
    worst
}

/// `project_out` with `a = b` on every writer, through the engine kernels `apply` uses.
fn ablate(w: &mut HashMap<String, Tensor>, b: &[f32]) {
    let p = SubspaceProjection::rank_one(b, b, 1.0);
    for (name, tensor) in w.iter_mut() {
        let embed = name.ends_with("embed_tokens.weight");
        let writer = name.ends_with("o_proj.weight") || name.ends_with("down_proj.weight");
        if !(embed || writer) {
            continue;
        }
        let shape = tensor.dims().to_vec();
        let mut bytes = to_bytes(&tensor.flatten_all().unwrap().to_vec1::<f32>().unwrap());
        match embed {
            true => engine::project_out_subspace_embedding(&mut bytes, EDType::F32, &shape, &p),
            false => engine::project_out_subspace(&mut bytes, EDType::F32, &shape, &p),
        }
        .unwrap();
        *tensor = Tensor::from_vec(from_bytes(&bytes), shape, &Device::Cpu).unwrap();
    }
}

#[test]
fn ablation_is_exact_on_the_forwards_of_llama_qwen2_and_qwen3() {
    for (t, extra) in [("llama", ""), ("qwen2", ""), ("qwen3", r#","head_dim":12"#)] {
        let json = tiny(t, extra);
        let mut rng = Rng(21);
        let b = unit(&mut rng, 32);
        let mut w = weights(&json, 22);
        let before = residual_component(&json, w.clone(), &b);
        ablate(&mut w, &b);
        let after = residual_component(&json, w, &b);
        assert!(
            before > 1e-2,
            "{t}: the direction is present before ({before})"
        );
        assert!(after < 1e-5, "{t}: exact, {after} left");
        let id = arch::detect_config(&json!({"model_type": t})).spec.id;
        assert_eq!(arch::builtin(&id).unwrap().exactness(), Exactness::Exact);
        assert_eq!(arch::verification(&id), "forward");
    }
}

// ------------------------------------------------------------------ exactness: topologies

/// A residual stack with one built-in's topology: RMSNorm (Gemma's `(1 + w)` included in
/// the weights) or LayerNorm with bias on the read side, sequential or parallel attention
/// and MLP, optional learned norm on each sublayer output. f64, so an exact ablation leaves
/// rounding only.
fn simulate(spec: &arch::ArchSpec, ablated: bool) -> f64 {
    let (d, f, vocab, layers, seq) = (16, 32, 20, 3, 6);
    let mut rng = Rng(77);
    let mut mat = |rows: usize, cols: usize| -> Vec<Vec<f64>> {
        (0..rows)
            .map(|_| (0..cols).map(|_| 0.3 * f64::from(rng.normal())).collect())
            .collect()
    };
    let b: Vec<f64> = {
        let v: Vec<f64> = mat(1, d).remove(0);
        let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        v.iter().map(|x| x / n).collect()
    };
    let project_rows = |m: &mut Vec<Vec<f64>>| {
        for row in m.iter_mut() {
            let c: f64 = row.iter().zip(&b).map(|(x, y)| x * y).sum();
            row.iter_mut().zip(&b).for_each(|(x, y)| *x -= c * y);
        }
    };
    // W' = (I - b bᵀ) W for a [d, n] writer.
    let project_cols = |m: &mut Vec<Vec<f64>>| {
        let c: Vec<f64> = (0..m[0].len())
            .map(|j| m.iter().zip(&b).map(|(row, bi)| bi * row[j]).sum())
            .collect();
        for (row, bi) in m.iter_mut().zip(&b) {
            row.iter_mut().zip(&c).for_each(|(x, cj)| *x -= cj * bi);
        }
    };
    let mut embed = mat(vocab, d);
    let weights: Vec<[Vec<Vec<f64>>; 4]> = (0..layers)
        .map(|_| [mat(d, d), mat(d, d), mat(f, d), mat(d, f)])
        .collect();
    let gains: Vec<Vec<Vec<f64>>> = (0..layers).map(|_| mat(6, d)).collect();
    let mut weights = weights;
    if ablated {
        project_rows(&mut embed);
        for [_, wo, _, wd] in weights.iter_mut() {
            project_cols(wo);
            project_cols(wd);
        }
    }
    let matvec = |m: &Vec<Vec<f64>>, x: &[f64]| -> Vec<f64> {
        m.iter()
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    };
    let norm = |x: &[f64], gain: &[f64], bias: &[f64]| -> Vec<f64> {
        let n = x.len() as f64;
        let (centered, var) = match spec.norm_kind {
            arch::NormKind::RmsNorm => (x.to_vec(), x.iter().map(|v| v * v).sum::<f64>() / n),
            arch::NormKind::LayerNorm => {
                let mean = x.iter().sum::<f64>() / n;
                let c: Vec<f64> = x.iter().map(|v| v - mean).collect();
                let var = c.iter().map(|v| v * v).sum::<f64>() / n;
                (c, var)
            }
        };
        let bias_on = spec.norm_kind == arch::NormKind::LayerNorm;
        centered
            .iter()
            .zip(gain)
            .zip(bias)
            .map(|((v, g), b0)| {
                v / (var + 1e-6).sqrt() * (1.0 + g) + if bias_on { *b0 } else { 0.0 }
            })
            .collect()
    };
    let post = |y: Vec<f64>, gain: &[f64]| -> Vec<f64> {
        match spec.post_sublayer_norm {
            true => norm(&y, gain, &vec![0.0; y.len()]),
            false => y,
        }
    };
    let tokens = [3usize, 7, 11, 2, 19, 5];
    let mut xs: Vec<Vec<f64>> = tokens[..seq].iter().map(|&t| embed[t].clone()).collect();
    let mut worst = 0.0f64;
    for (l, [wv, wo, wu, wd]) in weights.iter().enumerate() {
        let g = &gains[l];
        let n1: Vec<Vec<f64>> = xs.iter().map(|x| norm(x, &g[0], &g[1])).collect();
        // Causal softmax attention over the normalized inputs.
        let attn: Vec<Vec<f64>> = (0..seq)
            .map(|t| {
                let scores: Vec<f64> = (0..=t)
                    .map(|s| {
                        n1[t].iter().zip(&n1[s]).map(|(a, b)| a * b).sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::MIN, f64::max);
                let e: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let z: f64 = e.iter().sum();
                let mut mixed = vec![0.0; d];
                for (s, w) in e.iter().enumerate() {
                    let v = matvec(wv, &n1[s]);
                    mixed.iter_mut().zip(&v).for_each(|(m, v)| *m += w / z * v);
                }
                post(matvec(wo, &mixed), &g[4])
            })
            .collect();
        let mlp = |x: &[f64]| -> Vec<f64> {
            let h: Vec<f64> = matvec(wu, x)
                .iter()
                .map(|v| v / (1.0 + (-v).exp()))
                .collect();
            post(matvec(wd, &h), &g[5])
        };
        for t in 0..seq {
            match spec.parallel_block {
                true => {
                    let m = mlp(&n1[t]);
                    for i in 0..d {
                        xs[t][i] += attn[t][i] + m[i];
                    }
                }
                false => {
                    for i in 0..d {
                        xs[t][i] += attn[t][i];
                    }
                    let m = mlp(&norm(&xs[t], &g[2], &g[3]));
                    for i in 0..d {
                        xs[t][i] += m[i];
                    }
                }
            }
            let dot: f64 = xs[t].iter().zip(&b).map(|(x, y)| x * y).sum();
            let n = xs[t].iter().map(|x| x * x).sum::<f64>().sqrt();
            worst = worst.max(dot.abs() / n);
        }
    }
    worst
}

#[test]
fn every_builtin_classification_agrees_with_its_topology_measured() {
    for spec in arch::builtins() {
        let before = simulate(spec, false);
        let after = simulate(spec, true);
        assert!(before > 1e-2, "{}: {before}", spec.id);
        match spec.exactness() {
            Exactness::Exact => assert!(after < 1e-12, "{} exact but {after} left", spec.id),
            Exactness::Approximate { .. } => {
                assert!(
                    after > 1e-3,
                    "{} approximate but only {after} left",
                    spec.id
                )
            }
            Exactness::Unsupported { .. } => panic!("{} has no unsupported built-in", spec.id),
        }
    }
    // The brief's hypothesis, checked: parallel blocks and LayerNorm stay exact (the norms
    // only read the stream), the post-sublayer norms of Gemma do not.
    assert_eq!(
        arch::builtin("falcon").unwrap().exactness(),
        Exactness::Exact
    );
    assert_eq!(
        arch::builtin("command-r").unwrap().exactness(),
        Exactness::Exact
    );
    assert_eq!(
        arch::builtin("gemma2").unwrap().exactness().label(),
        "approximate"
    );
    assert_eq!(arch::verification("gemma2"), "topology");
}

// ------------------------------------------------------------------ MoE: SafeTensors

const H: usize = 8;
const INTER: usize = 6;
const E: usize = 3;

/// A qwen3_moe-like file: fused experts in layer 0, per-expert tensors in layer 1, the
/// router of each layer.
fn moe_model(dir: &Path) -> (PathBuf, Vec<Named>) {
    std::fs::write(dir.join("config.json"), r#"{"model_type":"qwen3_moe"}"#).unwrap();
    let mut rng = Rng(31);
    let tensors: Vec<Named> = vec![
        (
            "model.embed_tokens.weight",
            vec![10, H],
            normal(&mut rng, 10 * H, 1.0),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![H, H],
            normal(&mut rng, H * H, 0.1),
        ),
        (
            "model.layers.0.mlp.gate.weight",
            vec![E, H],
            normal(&mut rng, E * H, 0.1),
        ),
        (
            "model.layers.0.mlp.experts.down_proj",
            vec![E, H, INTER],
            normal(&mut rng, E * H * INTER, 0.1),
        ),
        (
            "model.layers.1.self_attn.o_proj.weight",
            vec![H, H],
            normal(&mut rng, H * H, 0.1),
        ),
        (
            "model.layers.1.mlp.gate.weight",
            vec![E, H],
            normal(&mut rng, E * H, 0.1),
        ),
        (
            "model.layers.1.mlp.experts.0.down_proj.weight",
            vec![H, INTER],
            normal(&mut rng, H * INTER, 0.1),
        ),
        (
            "model.layers.1.mlp.experts.1.down_proj.weight",
            vec![H, INTER],
            normal(&mut rng, H * INTER, 0.1),
        ),
    ];
    let model = dir.join("model.safetensors");
    write_safetensors(&model, &tensors);
    (model, tensors)
}

#[test]
fn moe_ablation_patches_fused_and_per_expert_down_projections() {
    let dir = tempfile::tempdir().unwrap();
    let (model, tensors) = moe_model(dir.path());
    let mut rng = Rng(32);
    let dirs = [unit(&mut rng, H), unit(&mut rng, H)].concat();
    // Orthonormal pair: Gram-Schmidt on the second.
    let (b0, b1) = dirs.split_at(H);
    let c: f32 = b0.iter().zip(b1).map(|(x, y)| x * y).sum();
    let mut b1: Vec<f32> = b1.iter().zip(b0).map(|(y, x)| y - c * x).collect();
    let n = b1.iter().map(|x| x * x).sum::<f32>().sqrt();
    b1.iter_mut().for_each(|x| *x /= n);
    let dirs = [b0.to_vec(), b1].concat();
    let patch = write_ablation(dir.path(), &dirs, 2, 1.0);
    let (code, env) = run_json(&[
        "apply",
        "--model",
        model.to_str().unwrap(),
        "--patch",
        patch.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["arch"]["id"], "qwen3_moe");
    assert_eq!(env["data"]["arch"]["exactness"]["kind"], "exact");
    let p = SubspaceProjection {
        a: &dirs,
        b: &dirs,
        alpha: 1.0,
        k: 2,
    };
    for (name, shape, values) in &tensors {
        let got = read_tensor(&model, name);
        let mut bytes = to_bytes(values);
        match *name {
            "model.embed_tokens.weight" => {
                engine::project_out_subspace_embedding(&mut bytes, EDType::F32, shape, &p).unwrap();
            }
            n if n.ends_with("mlp.gate.weight") => {} // the router is never modified
            "model.layers.0.mlp.experts.down_proj" => {
                engine::project_out_experts(&mut bytes, EDType::F32, shape, &p).unwrap();
                // Each expert is its own [hidden, inter] eraser.
                let mut first = to_bytes(&values[..H * INTER]);
                engine::project_out_subspace(&mut first, EDType::F32, &[H, INTER], &p).unwrap();
                assert_eq!(&bytes[..first.len()], &first[..]);
            }
            _ => {
                engine::project_out_subspace(&mut bytes, EDType::F32, shape, &p).unwrap();
            }
        }
        assert_eq!(to_bytes(&got), bytes, "{name}");
    }
    // add / subtract has no axis on a 3D tensor: refused, model restored.
    let before = std::fs::read(&model).unwrap();
    std::fs::write(
        dir.path().join("v.json"),
        serde_json::to_string(&vec![1.0f32; H]).unwrap(),
    )
    .unwrap();
    let shift = dir.path().join("shift.json");
    std::fs::write(
        &shift,
        json!({"schema_version": 1, "operations": [{"id": "s", "layers": "0",
        "projections": ["down_proj"], "operation": "subtract", "vector_file": "v.json",
        "alpha": 0.1, "axis": "cols"}]})
        .to_string(),
    )
    .unwrap();
    let (code, env) = run_json(&[
        "apply",
        "--model",
        model.to_str().unwrap(),
        "--patch",
        shift.to_str().unwrap(),
    ]);
    assert_ne!(code, 0, "{env}");
    assert!(
        env["error"]["message"]
            .as_str()
            .unwrap()
            .contains("experts fusionnés"),
        "{env}"
    );
    assert_eq!(std::fs::read(&model).unwrap(), before);
}

// ------------------------------------------------------------------ MoE: GGUF ffn_down_exps

#[test]
fn gguf_fused_experts_equal_the_per_expert_reference_byte_for_byte() {
    const GH: usize = 32;
    const GI: usize = 64;
    let mut rng = Rng(41);
    let tensors = vec![
        T {
            name: "token_embd.weight".into(),
            ne: vec![GH as u64, 8],
            ty: 8,
            data: tensor_data(8, 8, GH, 1.0, &mut rng),
        },
        T {
            name: "output.weight".into(),
            ne: vec![GH as u64, 8],
            ty: 8,
            data: tensor_data(8, 8, GH, 1.0, &mut rng),
        },
        T {
            name: "blk.0.attn_output.weight".into(),
            ne: vec![GH as u64, GH as u64],
            ty: 8,
            data: tensor_data(8, GH, GH, 0.05, &mut rng),
        },
        T {
            name: "blk.0.ffn_gate_inp.weight".into(),
            ne: vec![GH as u64, E as u64],
            ty: 0,
            data: tensor_data(0, E, GH, 0.1, &mut rng),
        },
        T {
            name: "blk.0.ffn_down_exps.weight".into(),
            ne: vec![GI as u64, GH as u64, E as u64],
            ty: 8,
            data: tensor_data(8, E * GH, GI, 0.02, &mut rng),
        },
    ];
    let kv = [
        ("general.architecture", Kv::Str("qwen3moe")),
        ("general.alignment", Kv::U32(32)),
    ];
    let original = gguf_bytes(&kv, &tensors);
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("m.gguf");
    std::fs::write(&model, &original).unwrap();
    let b = unit(&mut rng, GH);
    let patch = write_ablation(dir.path(), &b, 1, 1.0);
    let (code, env) = run_json(&[
        "apply",
        "--gguf",
        "--model",
        model.to_str().unwrap(),
        "--patch",
        patch.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["arch"]["id"], "qwen3_moe");
    let exps = env["data"]["gguf"]["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "blk.0.ffn_down_exps.weight")
        .unwrap()
        .clone();
    assert!(exps["residual_fraction"].as_f64().unwrap() < 0.5, "{exps}");

    // Reference: per expert, requant_keep_scales(project(dequant(W_e))).
    let file = gguf::parse(&original).unwrap();
    let mut expected = original.clone();
    let p = SubspaceProjection::rank_one(&b, &b, 1.0);
    for (name, rows, cols, experts, local) in [
        ("token_embd.weight", 8, GH, 1, true),
        ("blk.0.attn_output.weight", GH, GH, 1, false),
        ("blk.0.ffn_down_exps.weight", GH, GI, E, false),
    ] {
        let t = file.tensor(name).unwrap();
        let c = codec(t.ggml_type.0);
        let matrix = c.row_bytes(cols) * rows;
        for e in 0..experts {
            let start = t.start as usize + e * matrix;
            let w = decode_tensor(t.ggml_type.0, &original[start..start + matrix]);
            let mut bytes = to_bytes(&w);
            match local {
                true => engine::project_out_subspace_embedding(
                    &mut bytes,
                    EDType::F32,
                    &[rows, cols],
                    &p,
                ),
                false => engine::project_out_subspace(&mut bytes, EDType::F32, &[rows, cols], &p),
            }
            .unwrap();
            let exact = from_bytes(&bytes);
            let blocks = expected[start..start + matrix].chunks_mut(c.block_bytes());
            for ((blk, before), after) in blocks
                .zip(w.chunks(c.block_elems()))
                .zip(exact.chunks(c.block_elems()))
            {
                requant_keep_scales(t.ggml_type.0, blk, before, after);
            }
        }
    }
    assert_eq!(sha(&std::fs::read(&model).unwrap()), sha(&expected));
}

// ------------------------------------------------------------------ the gate

/// A dense model with Llama names and the given config.
fn dense_model(dir: &Path, config: &str, with_o_bias: bool) -> PathBuf {
    std::fs::write(dir.join("config.json"), config).unwrap();
    let mut rng = Rng(51);
    let mut tensors = vec![
        (
            "model.embed_tokens.weight",
            vec![10, H],
            normal(&mut rng, 10 * H, 1.0),
        ),
        (
            "model.layers.0.self_attn.o_proj.weight",
            vec![H, H],
            normal(&mut rng, H * H, 0.1),
        ),
        (
            "model.layers.0.mlp.down_proj.weight",
            vec![H, INTER],
            normal(&mut rng, H * INTER, 0.1),
        ),
    ];
    if with_o_bias {
        tensors.push((
            "model.layers.0.self_attn.o_proj.bias",
            vec![H],
            normal(&mut rng, H, 0.1),
        ));
    }
    let model = dir.join("model.safetensors");
    write_safetensors(&model, &tensors);
    model
}

fn backups(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".bak-")
        })
        .count()
}

#[test]
fn approximate_architectures_need_explicit_acceptance() {
    for (config, bias, reason) in [
        (r#"{"model_type":"gemma2"}"#, false, "norme apprise"),
        (r#"{"model_type":"qwen2"}"#, true, "biais"),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let model = dense_model(dir.path(), config, bias);
        let patch = write_ablation(dir.path(), &unit(&mut Rng(52), H), 1, 1.0);
        let original = std::fs::read(&model).unwrap();
        let args = [
            "apply",
            "--model",
            model.to_str().unwrap(),
            "--patch",
            patch.to_str().unwrap(),
        ];
        let (code, env) = run_json(&args);
        assert_eq!(
            (code, env["error"]["kind"].as_str()),
            (2, Some("approximate_ablation")),
            "{env}"
        );
        assert!(
            env["error"]["message"].as_str().unwrap().contains(reason),
            "{env}"
        );
        assert!(env["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--accept-approximate"));
        assert_eq!(std::fs::read(&model).unwrap(), original, "nothing written");
        assert_eq!(backups(dir.path()), 0, "refused before the backup");

        let (code, env) = run_json(&[&args[..], &["--accept-approximate"]].concat());
        assert_eq!(code, 0, "{env}");
        assert_eq!(env["data"]["arch"]["accepted_approximate"], true);
        assert_eq!(env["data"]["arch"]["exactness"]["kind"], "approximate");
        assert!(env["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("approximative")));
        assert_ne!(std::fs::read(&model).unwrap(), original);
    }
    // A shift (steering) is not an ablation: no gate on a Gemma 2.
    let dir = tempfile::tempdir().unwrap();
    let model = dense_model(dir.path(), r#"{"model_type":"gemma2"}"#, false);
    std::fs::write(
        dir.path().join("v.json"),
        serde_json::to_string(&vec![1.0f32; H]).unwrap(),
    )
    .unwrap();
    let shift = dir.path().join("shift.json");
    std::fs::write(
        &shift,
        json!({"schema_version": 1, "operations": [{"id": "s", "layers": "0",
        "projections": ["down_proj"], "operation": "subtract", "vector_file": "v.json",
        "alpha": 0.1, "axis": "cols"}]})
        .to_string(),
    )
    .unwrap();
    let (code, env) = run_json(&[
        "apply",
        "--model",
        model.to_str().unwrap(),
        "--patch",
        shift.to_str().unwrap(),
    ]);
    assert_eq!(code, 0, "{env}");
    assert!(env["data"].get("arch").is_none(), "{env}");
}

#[test]
fn gguf_gemma2_ablation_is_gated_before_any_write() {
    let mut rng = Rng(53);
    let tensors = vec![
        T {
            name: "token_embd.weight".into(),
            ne: vec![32, 8],
            ty: 8,
            data: tensor_data(8, 8, 32, 1.0, &mut rng),
        },
        T {
            name: "blk.0.attn_output.weight".into(),
            ne: vec![32, 32],
            ty: 8,
            data: tensor_data(8, 32, 32, 0.05, &mut rng),
        },
    ];
    let kv = [
        ("general.architecture", Kv::Str("gemma2")),
        ("general.alignment", Kv::U32(32)),
    ];
    let original = gguf_bytes(&kv, &tensors);
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("g.gguf");
    std::fs::write(&model, &original).unwrap();
    let patch = write_ablation(dir.path(), &unit(&mut rng, 32), 1, 1.0);
    let args = [
        "apply",
        "--gguf",
        "--model",
        model.to_str().unwrap(),
        "--patch",
        patch.to_str().unwrap(),
    ];
    let (code, env) = run_json(&args);
    assert_eq!(
        (code, env["error"]["kind"].as_str()),
        (2, Some("approximate_ablation")),
        "{env}"
    );
    assert_eq!(std::fs::read(&model).unwrap(), original);
    assert_eq!(backups(dir.path()), 0);
    // Accepted, the tied embeddings still need their own consent (gemma2 GGUF: no output).
    let (code, env) = run_json(
        &[
            &args[..],
            &["--accept-approximate", "--allow-tied-embedding"],
        ]
        .concat(),
    );
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["arch"]["accepted_approximate"], true);
}

// ------------------------------------------------------------------ third-party specs

fn third_party() -> Value {
    json!({
        "id": "toy",
        "safetensors": {
            "embed": ["tok.weight"],
            "attn_out": ["blocks.{layer}.mixer.out.weight"],
            "mlp_out": ["blocks.{layer}.ff.w2.weight"]
        },
        "norm_kind": "rms_norm",
        "post_sublayer_norm": false,
        "tied_embeddings": {"config_key": "tie_word_embeddings", "default": false}
    })
}

#[test]
fn a_third_party_spec_drives_inspect_and_apply() {
    let dir = tempfile::tempdir().unwrap();
    let mut rng = Rng(61);
    let tensors = vec![
        ("tok.weight", vec![10, H], normal(&mut rng, 10 * H, 1.0)),
        (
            "blocks.0.mixer.out.weight",
            vec![H, H],
            normal(&mut rng, H * H, 0.1),
        ),
        (
            "blocks.0.ff.w2.weight",
            vec![H, INTER],
            normal(&mut rng, H * INTER, 0.1),
        ),
        (
            "blocks.0.ff.w1.weight",
            vec![INTER, H],
            normal(&mut rng, H * INTER, 0.1),
        ),
    ];
    let model = dir.path().join("model.safetensors");
    write_safetensors(&model, &tensors);
    let spec = dir.path().join("toy.json");
    std::fs::write(&spec, third_party().to_string()).unwrap();
    let m = model.to_str().unwrap();
    let s = spec.to_str().unwrap();
    let (code, env) = run_json(&[
        "inspect",
        "--model",
        m,
        "--arch-spec",
        s,
        "--proj",
        "down",
        "--proj",
        "o_proj",
        "--proj",
        "embed_tokens",
    ]);
    assert_eq!(code, 0, "{env}");
    let names: Vec<&str> = env["data"]["tensors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "tok.weight",
            "blocks.0.ff.w2.weight",
            "blocks.0.mixer.out.weight"
        ]
    );
    assert_eq!(env["data"]["arch"]["source"], "arch_spec_file");
    let (code, env) = run_json(&["arch", "detect", "--model", m, "--arch-spec", s]);
    assert_eq!(code, 0, "{env}");
    assert_eq!(
        env["data"]["roles"],
        json!({"embed": 1, "attn_out": 1, "mlp_out": 1})
    );
    // Without the spec nothing matches the default naming.
    let patch = write_ablation(dir.path(), &unit(&mut rng, H), 1, 1.0);
    let (code, _) = run_json(&["apply", "--model", m, "--patch", patch.to_str().unwrap()]);
    assert_ne!(code, 0);
    let (code, env) = run_json(&[
        "apply",
        "--model",
        m,
        "--patch",
        patch.to_str().unwrap(),
        "--arch-spec",
        s,
    ]);
    assert_eq!(code, 0, "{env}");
    // An unpatchable writer makes the ablation unsupported.
    let mut unsupported = third_party();
    unsupported["unprojected_writers"] = json!(["pos.weight (positions apprises)"]);
    std::fs::write(&spec, unsupported.to_string()).unwrap();
    let (code, env) = run_json(&[
        "apply",
        "--model",
        m,
        "--patch",
        patch.to_str().unwrap(),
        "--arch-spec",
        s,
        "--accept-approximate",
    ]);
    assert_eq!(
        (code, env["error"]["kind"].as_str()),
        (2, Some("unsupported_architecture")),
        "{env}"
    );
}

#[test]
fn invalid_third_party_specs_are_refused_with_a_kind() {
    let dir = tempfile::tempdir().unwrap();
    let model = dense_model(dir.path(), r#"{"model_type":"qwen2"}"#, false);
    let spec = dir.path().join("bad.json");
    let mut bad_placeholder = third_party();
    bad_placeholder["safetensors"]["attn_out"] = json!(["blocks.{layr}.out.weight"]);
    let mut unknown_field = third_party();
    unknown_field["extra"] = json!(1);
    let mut embed_with_layer = third_party();
    embed_with_layer["safetensors"]["embed"] = json!(["blocks.{layer}.tok.weight"]);
    for bad in [bad_placeholder, unknown_field, embed_with_layer] {
        std::fs::write(&spec, bad.to_string()).unwrap();
        let (code, env) = run_json(&[
            "inspect",
            "--model",
            model.to_str().unwrap(),
            "--arch-spec",
            spec.to_str().unwrap(),
        ]);
        assert_eq!(
            (code, env["error"]["kind"].as_str()),
            (2, Some("invalid_arch_spec")),
            "{env}"
        );
    }
}

// ------------------------------------------------------------------ CLI and MCP

#[test]
fn arch_list_detect_and_schema() {
    let (code, env) = run_json(&["arch", "list"]);
    assert_eq!(code, 0, "{env}");
    let list = env["data"]["architectures"].as_array().unwrap();
    assert_eq!(list.len(), arch::builtins().len());
    let gemma = list.iter().find(|a| a["id"] == "gemma2").unwrap();
    assert_eq!(gemma["exactness"]["kind"], "approximate");
    assert_eq!(gemma["verification"], "topology");
    let qwen2 = list.iter().find(|a| a["id"] == "qwen2").unwrap();
    assert_eq!(
        (qwen2["forward"].as_bool(), qwen2["verification"].as_str()),
        (Some(true), Some("forward"))
    );

    let dir = tempfile::tempdir().unwrap();
    let model = dense_model(
        dir.path(),
        r#"{"model_type":"qwen3","tie_word_embeddings":true}"#,
        false,
    );
    let (code, env) = run_json(&["arch", "detect", "--model", model.to_str().unwrap()]);
    assert_eq!(code, 0, "{env}");
    assert_eq!(env["data"]["id"], "qwen3");
    assert_eq!(env["data"]["tied_embeddings"], true);
    assert_eq!(env["data"]["exactness"]["kind"], "exact");

    let out = surgeon()
        .args(["schema", "--kind", "arch"])
        .output()
        .unwrap();
    let schema: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(schema["title"], "ArchSpec");
    assert!(jsonschema::validator_for(&schema)
        .unwrap()
        .is_valid(&third_party()));
}

fn mcp(dir: &Path, read_only: bool, calls: &[Value]) -> Vec<Value> {
    let mut cmd = surgeon();
    cmd.args(["mcp", "--allow-dir", dir.to_str().unwrap()]);
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
            "clientInfo": {"name": "p16", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    ];
    for (i, args) in calls.iter().enumerate() {
        lines.push(json!({"jsonrpc": "2.0", "id": i + 1, "method": "tools/call", "params": args}));
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
        .map(|id| {
            replies.iter().find(|r| r["id"] == id).unwrap()["result"]["structuredContent"].clone()
        })
        .collect()
}

#[test]
fn mcp_takes_the_arch_fields_inside_the_sandbox() {
    let dir = tempfile::tempdir().unwrap();
    let model = dense_model(dir.path(), r#"{"model_type":"gemma2"}"#, false);
    let original = std::fs::read(&model).unwrap();
    std::fs::write(dir.path().join("toy.json"), third_party().to_string()).unwrap();
    let patch: Value = {
        write_ablation(dir.path(), &unit(&mut Rng(71), H), 1, 1.0);
        serde_json::from_str(&std::fs::read_to_string(dir.path().join("ablate.json")).unwrap())
            .unwrap()
    };
    let apply = |extra: Value| {
        let mut args = json!({"model": "model.safetensors", "patch": patch, "dry_run": true});
        for (k, v) in extra.as_object().unwrap() {
            args[k] = v.clone();
        }
        json!({"name": "surgeon_apply", "arguments": args})
    };
    let replies = mcp(
        dir.path(),
        false,
        &[
            apply(json!({})),
            apply(json!({"accept_approximate": true})),
            json!({"name": "surgeon_inspect", "arguments": {"model": "model.safetensors", "arch_spec": "../toy.json"}}),
            json!({"name": "surgeon_inspect", "arguments": {"model": "model.safetensors"}}),
        ],
    );
    assert_eq!(
        replies[0]["error"]["kind"], "approximate_ablation",
        "{}",
        replies[0]
    );
    assert_eq!(replies[1]["ok"], true, "{}", replies[1]);
    assert_eq!(replies[1]["data"]["arch"]["accepted_approximate"], true);
    assert_eq!(
        replies[2]["error"]["kind"], "path_not_allowed",
        "{}",
        replies[2]
    );
    assert_eq!(replies[3]["data"]["arch"]["id"], "gemma2");
    let replies = mcp(
        dir.path(),
        true,
        &[apply(json!({"accept_approximate": true, "dry_run": false}))],
    );
    assert_eq!(replies[0]["error"]["kind"], "read_only");
    assert_eq!(std::fs::read(&model).unwrap(), original);
}
