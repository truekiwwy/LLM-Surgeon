//! SafeTensors -> GGUF for Qwen2 in pure Rust, so a GPU llama.cpp build (Vulkan, HIP...)
//! can run a model on a machine with no python / torch. Other architectures still go
//! through llama.cpp's `convert_hf_to_gguf.py` (see `export_gguf`).
//!
//! Same layout as that script for Qwen2: llama.cpp tensor names, GGUF dims (reversed, no
//! permutation for Qwen2), the byte-level BPE vocabulary of `tokenizer.json` padded to the
//! rows of the embedding, 1-D tensors (norms, biases) in f32.

use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use half::f16;
use serde_json::Value;

use super::error::{EvalError, Result};
use crate::engine::{DType, ReadOnlyModel};
use crate::kernel::simd::{bf16_load, f16_load};

const ALIGNMENT: u64 = 32;
const GGUF_VERSION: u32 = 3;
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
/// llama.cpp token types.
const TOKEN_NORMAL: i32 = 1;
const TOKEN_CONTROL: i32 = 3;
const TOKEN_USER_DEFINED: i32 = 4;
const TOKEN_UNUSED: i32 = 5;
/// GGUF metadata value tags.
const TAG_I32: u32 = 5;
const TAG_U32: u32 = 4;
const TAG_F32: u32 = 6;
const TAG_BOOL: u32 = 7;
const TAG_STR: u32 = 8;
const TAG_ARRAY: u32 = 9;
/// Elements converted per buffer, so the 150k x 896 embedding is never copied whole.
const CHUNK_ELEMENTS: usize = 1 << 16;

/// ggml type of the matrices (1-D tensors are always f32).
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
    clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum WeightType {
    #[default]
    F16,
    /// Lossless from bf16 / f16 sources, twice the size.
    F32,
}

impl WeightType {
    fn ggml(self) -> u32 {
        match self {
            Self::F32 => GGML_F32,
            Self::F16 => GGML_F16,
        }
    }
}

struct Planned<'a> {
    name: String,
    model: &'a ReadOnlyModel,
    source: &'a str,
    ne: Vec<u64>,
    ggml: u32,
    bytes: u64,
}

fn io(path: &Path, e: impl std::fmt::Display) -> EvalError {
    EvalError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    }
}

fn invalid(msg: impl Into<String>) -> EvalError {
    EvalError::InvalidConfig(msg.into())
}

fn read_json(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).map_err(|e| io(path, e))?;
    serde_json::from_str(&text).map_err(|e| invalid(format!("{} : {e}", path.display())))
}

/// llama.cpp name of a Qwen2 tensor; `None` for buffers that are not weights.
fn gguf_name(hf: &str) -> Result<Option<String>> {
    let unknown = || invalid(format!("tenseur Qwen2 inconnu : {hf}"));
    let fixed = match hf {
        "model.embed_tokens.weight" => "token_embd.weight",
        "model.norm.weight" => "output_norm.weight",
        "lm_head.weight" => "output.weight",
        _ if hf.ends_with("rotary_emb.inv_freq") => return Ok(None),
        _ => {
            let rest = hf.strip_prefix("model.layers.").ok_or_else(unknown)?;
            let (layer, rest) = rest.split_once('.').ok_or_else(unknown)?;
            let mapped = match rest {
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.q_proj.bias" => "attn_q.bias",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.k_proj.bias" => "attn_k.bias",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.v_proj.bias" => "attn_v.bias",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                _ => return Err(unknown()),
            };
            return Ok(Some(format!("blk.{layer}.{mapped}")));
        }
    };
    Ok(Some(fixed.into()))
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// The key/value section of the header, with its entry count.
#[derive(Default)]
struct Meta {
    buf: Vec<u8>,
    entries: u64,
}

impl Meta {
    fn put(&mut self, key: &str, tag: u32, value: &[u8]) {
        put_str(&mut self.buf, key);
        self.buf.extend_from_slice(&tag.to_le_bytes());
        self.buf.extend_from_slice(value);
        self.entries += 1;
    }

    fn u32(&mut self, key: &str, v: u64) {
        self.put(key, TAG_U32, &(v as u32).to_le_bytes());
    }

    fn f32(&mut self, key: &str, v: f32) {
        self.put(key, TAG_F32, &v.to_le_bytes());
    }

    fn text(&mut self, key: &str, v: &str) {
        let mut value = Vec::new();
        put_str(&mut value, v);
        self.put(key, TAG_STR, &value);
    }

    fn array(&mut self, key: &str, elem_tag: u32, count: usize, items: &[u8]) {
        let mut value = elem_tag.to_le_bytes().to_vec();
        value.extend_from_slice(&(count as u64).to_le_bytes());
        value.extend_from_slice(items);
        self.put(key, TAG_ARRAY, &value);
    }

    fn strings(&mut self, key: &str, items: &[String]) {
        let mut buf = Vec::new();
        items.iter().for_each(|s| put_str(&mut buf, s));
        self.array(key, TAG_STR, items.len(), &buf);
    }
}

fn config_u64(config: &Value, key: &str) -> Result<u64> {
    config[key]
        .as_u64()
        .ok_or_else(|| invalid(format!("config.json : « {key} » absent ou invalide")))
}

/// An id that is a number or a list of numbers (`eos_token_id` may be either).
fn first_id(v: &Value) -> Option<u64> {
    v.as_u64().or_else(|| v.as_array()?.first()?.as_u64())
}

/// Token texts and types by id, `n_vocab` of them (the rows of the embedding).
fn vocabulary(tokenizer: &Value, n_vocab: usize) -> Result<(Vec<String>, Vec<i32>)> {
    let mut tokens: Vec<String> = (0..n_vocab).map(|i| format!("[PAD{i}]")).collect();
    let mut types = vec![TOKEN_UNUSED; n_vocab];
    let mut place = |id: Option<u64>, text: &str, kind: i32| -> Result<()> {
        let id = id.map(|i| i as usize).filter(|&i| i < n_vocab);
        let id =
            id.ok_or_else(|| invalid(format!("token « {text} » hors de vocab_size ({n_vocab})")))?;
        tokens[id] = text.to_owned();
        types[id] = kind;
        Ok(())
    };
    let kind = tokenizer["model"]["type"].as_str().unwrap_or("?");
    let vocab = tokenizer["model"]["vocab"]
        .as_object()
        .filter(|_| kind == "BPE");
    let vocab = vocab.ok_or_else(|| {
        invalid(format!(
            "tokenizer.json : seul un tokenizer BPE (byte-level) se convertit, pas « {kind} »"
        ))
    })?;
    for (text, id) in vocab {
        place(id.as_u64(), text, TOKEN_NORMAL)?;
    }
    for added in tokenizer["added_tokens"].as_array().into_iter().flatten() {
        let kind = if added["special"].as_bool() == Some(true) {
            TOKEN_CONTROL
        } else {
            TOKEN_USER_DEFINED
        };
        let text = added["content"].as_str().unwrap_or_default();
        place(added["id"].as_u64(), text, kind)?;
    }
    Ok((tokens, types))
}

/// Merge rules as "a b", whether the tokenizer stores strings or pairs.
fn merges(tokenizer: &Value) -> Result<Vec<String>> {
    let list = tokenizer["model"]["merges"]
        .as_array()
        .ok_or_else(|| invalid("tokenizer.json : model.merges absent"))?;
    list.iter()
        .map(|m| match (m.as_str(), m.as_array()) {
            (Some(s), _) => Ok(s.to_owned()),
            (_, Some(pair)) => Ok(pair
                .iter()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>()
                .join(" ")),
            _ => Err(invalid("tokenizer.json : fusion BPE mal formée")),
        })
        .collect()
}

fn metadata(dir: &Path, config: &Value, weights: WeightType, n_vocab: usize) -> Result<Meta> {
    let tokenizer = read_json(&dir.join("tokenizer.json"))?;
    // Optional: only the pad token and the chat template come from it.
    let tok_config = read_json(&dir.join("tokenizer_config.json")).unwrap_or(Value::Null);
    let (tokens, types) = vocabulary(&tokenizer, n_vocab)?;
    let name = dir
        .file_name()
        .map_or_else(|| "model".into(), |n| n.to_string_lossy());

    let mut m = Meta::default();
    m.text("general.architecture", "qwen2");
    m.text("general.name", &name);
    m.u32("general.file_type", u64::from(weights == WeightType::F16));
    m.u32("general.quantization_version", 2);
    for (key, from) in [
        ("block_count", "num_hidden_layers"),
        ("context_length", "max_position_embeddings"),
        ("embedding_length", "hidden_size"),
        ("feed_forward_length", "intermediate_size"),
        ("attention.head_count", "num_attention_heads"),
        ("attention.head_count_kv", "num_key_value_heads"),
    ] {
        m.u32(&format!("qwen2.{key}"), config_u64(config, from)?);
    }
    m.f32(
        "qwen2.rope.freq_base",
        config["rope_theta"].as_f64().unwrap_or(10_000.0) as f32,
    );
    m.f32(
        "qwen2.attention.layer_norm_rms_epsilon",
        config["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
    );
    m.text("tokenizer.ggml.model", "gpt2");
    m.text("tokenizer.ggml.pre", "qwen2");
    m.strings("tokenizer.ggml.tokens", &tokens);
    let kinds: Vec<u8> = types.iter().flat_map(|t| t.to_le_bytes()).collect();
    m.array("tokenizer.ggml.token_type", TAG_I32, types.len(), &kinds);
    m.strings("tokenizer.ggml.merges", &merges(&tokenizer)?);
    let pad = tok_config["pad_token"]
        .as_str()
        .and_then(|p| tokens.iter().position(|t| t == p));
    for (key, id) in [
        ("bos", first_id(&config["bos_token_id"])),
        ("eos", first_id(&config["eos_token_id"])),
        ("padding", pad.map(|p| p as u64)),
    ] {
        if let Some(id) = id {
            m.u32(&format!("tokenizer.ggml.{key}_token_id"), id);
        }
    }
    m.put("tokenizer.ggml.add_bos_token", TAG_BOOL, &[0]);
    if let Some(template) = tok_config["chat_template"].as_str() {
        m.text("tokenizer.chat_template", template);
    }
    Ok(m)
}

/// Every `*.safetensors` of the directory, in name order (shards or a single file).
fn safetensors_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| io(dir, e))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    match files.is_empty() {
        true => Err(invalid(format!(
            "aucun .safetensors dans {}",
            dir.display()
        ))),
        false => Ok(files),
    }
}

fn plan<'a>(models: &'a [ReadOnlyModel], weights: WeightType) -> Result<Vec<Planned<'a>>> {
    let mut planned = Vec::new();
    for model in models {
        for info in &model.index().tensors {
            let Some(name) = gguf_name(&info.name)? else {
                continue;
            };
            if info.dtype.size_of().is_none() {
                return Err(invalid(format!(
                    "{} : dtype {} non géré",
                    info.name, info.dtype
                )));
            }
            let ggml = if info.shape.len() == 1 {
                GGML_F32
            } else {
                weights.ggml()
            };
            let width = if ggml == GGML_F16 { 2 } else { 4 };
            let elements: usize = info.shape.iter().product();
            planned.push(Planned {
                name,
                model,
                source: &info.name,
                // GGUF lists the fastest-varying dimension first.
                ne: info.shape.iter().rev().map(|&d| d as u64).collect(),
                ggml,
                bytes: (elements * width) as u64,
            });
        }
    }
    Ok(planned)
}

/// Converts one tensor to `ggml` in chunks. A finite value that overflows f16 is an
/// error: the file would silently carry an inf.
fn write_tensor(out: &mut impl Write, t: &Planned) -> Result<()> {
    let view = t
        .model
        .tensor(t.source)
        .map_err(|e| invalid(e.to_string()))?;
    let load: fn(&[u8]) -> f32 = match view.info.dtype {
        DType::BF16 => |b| bf16_load([b[0], b[1]]),
        DType::F16 => |b| f16_load([b[0], b[1]]),
        _ => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
    };
    let width = view.info.dtype.size_of().unwrap_or(4);
    let mut buf = Vec::with_capacity(CHUNK_ELEMENTS * 4);
    for chunk in view.bytes.chunks(CHUNK_ELEMENTS * width) {
        buf.clear();
        for cell in chunk.chunks_exact(width) {
            let x = load(cell);
            if t.ggml == GGML_F16 {
                let h = f16::from_f32(x);
                if h.is_infinite() && x.is_finite() {
                    return Err(invalid(format!("{} : {x} dépasse la plage f16", t.name)));
                }
                buf.extend_from_slice(&h.to_le_bytes());
            } else {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        out.write_all(&buf).map_err(|e| invalid(e.to_string()))?;
    }
    Ok(())
}

fn pad_to_alignment(out: &mut impl Write, written: u64) -> std::io::Result<()> {
    let pad = written.next_multiple_of(ALIGNMENT) - written;
    out.write_all(&vec![0; pad as usize])
}

/// Whether `dir` holds a Qwen2 `config.json` (the only architecture converted natively).
pub(crate) fn is_qwen2_dir(dir: &Path) -> bool {
    read_json(&dir.join("config.json")).is_ok_and(|c| c["model_type"].as_str() == Some("qwen2"))
}

/// Writes `output` from the Qwen2 SafeTensors directory `dir`. The model is only read.
pub fn convert_qwen2(dir: &Path, output: &Path, weights: WeightType) -> Result<()> {
    let config = read_json(&dir.join("config.json"))?;
    if config["model_type"].as_str() != Some("qwen2") {
        return Err(invalid(format!(
            "conversion native : seul qwen2 est géré (model_type {}) ; passez --llama-cpp-dir \
             vers un llama.cpp qui contient convert_hf_to_gguf.py",
            config["model_type"]
        )));
    }
    let models = safetensors_files(dir)?
        .iter()
        .map(|p| ReadOnlyModel::open(p).map_err(|e| invalid(format!("{} : {e}", p.display()))))
        .collect::<Result<Vec<_>>>()?;
    let mut planned = plan(&models, weights)?;
    let n_vocab = planned
        .iter()
        .find(|t| t.name == "token_embd.weight")
        .and_then(|t| t.ne.get(1))
        .ok_or_else(|| invalid("model.embed_tokens.weight absent"))?;
    let n_vocab = *n_vocab as usize;
    // Tied embeddings: llama.cpp reuses token_embd when there is no output.weight.
    if config["tie_word_embeddings"].as_bool() == Some(true) {
        planned.retain(|t| t.name != "output.weight");
    }
    let meta = metadata(dir, &config, weights, n_vocab)?;

    let mut head = b"GGUF".to_vec();
    head.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    head.extend_from_slice(&(planned.len() as u64).to_le_bytes());
    head.extend_from_slice(&meta.entries.to_le_bytes());
    head.extend_from_slice(&meta.buf);
    let mut offset = 0u64;
    for t in &planned {
        put_str(&mut head, &t.name);
        head.extend_from_slice(&(t.ne.len() as u32).to_le_bytes());
        t.ne.iter()
            .for_each(|d| head.extend_from_slice(&d.to_le_bytes()));
        head.extend_from_slice(&t.ggml.to_le_bytes());
        head.extend_from_slice(&offset.to_le_bytes());
        offset += t.bytes.next_multiple_of(ALIGNMENT);
    }

    let file = std::fs::File::create(output).map_err(|e| io(output, e))?;
    let mut out = BufWriter::new(file);
    let wrote = |e: std::io::Error| io(output, e);
    out.write_all(&head).map_err(wrote)?;
    pad_to_alignment(&mut out, head.len() as u64).map_err(wrote)?;
    for t in &planned {
        write_tensor(&mut out, t)?;
        pad_to_alignment(&mut out, t.bytes).map_err(wrote)?;
    }
    out.flush().map_err(wrote)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::gguf::{parse, parse::MetaValue};
    use half::bf16;
    use serde_json::json;

    /// Tensors back to back, all BF16.
    fn write_safetensors(path: &Path, tensors: &[(&str, Vec<usize>, Vec<u8>)]) {
        let (mut header, mut data) = (serde_json::Map::new(), Vec::new());
        for (name, shape, bytes) in tensors {
            let start = data.len();
            data.extend_from_slice(bytes);
            let offsets = [start, data.len()];
            header.insert(
                (*name).into(),
                json!({"dtype": "BF16", "shape": shape, "data_offsets": offsets}),
            );
        }
        let json = Value::Object(header).to_string();
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(json.as_bytes());
        file.extend_from_slice(&data);
        std::fs::write(path, file).unwrap();
    }

    /// 1 layer, hidden 4, 6 token rows: 4 in the vocabulary, one added, one padding row.
    fn tiny_model(dir: &Path) {
        let config = json!({
            "model_type": "qwen2", "num_hidden_layers": 1, "max_position_embeddings": 64,
            "hidden_size": 4, "intermediate_size": 8, "num_attention_heads": 2,
            "num_key_value_heads": 1, "rope_theta": 1e6, "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true, "bos_token_id": 4, "eos_token_id": 5,
        });
        let tokenizer = json!({
            "added_tokens": [{"id": 4, "content": "<|endoftext|>", "special": true}],
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "Ġ": 3}, "merges": ["a b"]},
        });
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        let l = "model.layers.0.";
        let shapes = [
            ("model.embed_tokens.weight".to_owned(), vec![6, 4]),
            ("model.norm.weight".to_owned(), vec![4]),
            (format!("{l}input_layernorm.weight"), vec![4]),
            (format!("{l}self_attn.q_proj.weight"), vec![4, 4]),
            (format!("{l}self_attn.q_proj.bias"), vec![4]),
            (format!("{l}mlp.down_proj.weight"), vec![4, 8]),
        ];
        // Element i of every tensor is i / 4.
        let tensors: Vec<_> = shapes
            .iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                let bytes = (0..n).flat_map(|i| bf16::from_f32(i as f32 * 0.25).to_le_bytes());
                (name.as_str(), shape.clone(), bytes.collect::<Vec<u8>>())
            })
            .collect();
        write_safetensors(&dir.join("model.safetensors"), &tensors);
    }

    #[test]
    fn converts_a_tiny_qwen2_to_a_readable_gguf() {
        let dir = tempfile::tempdir().unwrap();
        tiny_model(dir.path());
        let out = dir.path().join("m.gguf");
        convert_qwen2(dir.path(), &out, WeightType::F16).unwrap();

        let bytes = std::fs::read(&out).unwrap();
        let gguf = parse(&bytes).unwrap();
        assert_eq!(gguf.architecture(), Some("qwen2"));
        assert_eq!(gguf.tensors.len(), 6);
        let down = gguf.tensor("blk.0.ffn_down.weight").unwrap();
        // HF [out 4, in 8] -> ne [in 8, out 4], f16.
        assert_eq!((down.ne.clone(), down.ggml_type.0), (vec![8, 4], GGML_F16));
        assert_eq!(
            gguf.tensor("blk.0.attn_q.bias").unwrap().ggml_type.0,
            GGML_F32
        );
        assert_eq!(
            gguf.tensor("output_norm.weight").unwrap().ggml_type.0,
            GGML_F32
        );
        assert!(gguf.tensor("output.weight").is_none());
        // 6 token rows: 4 vocabulary entries, 1 added token, 1 padding row.
        let tokens = gguf.meta("tokenizer.ggml.tokens").unwrap();
        assert!(matches!(tokens, MetaValue::Array { len: 6, .. }));
        // Values survive: embedding row 1 is [1.0, 1.25, 1.5, 1.75] as f16.
        let emb = gguf.tensor("token_embd.weight").unwrap();
        let start = emb.start as usize + 4 * 2;
        let row: Vec<f32> = bytes[start..start + 8]
            .chunks(2)
            .map(|b| f16_load([b[0], b[1]]))
            .collect();
        assert_eq!(row, [1.0, 1.25, 1.5, 1.75]);
    }

    #[test]
    fn refuses_other_architectures() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.json"), r#"{"model_type": "llama"}"#).unwrap();
        let err = convert_qwen2(dir.path(), &dir.path().join("m.gguf"), WeightType::F16);
        assert!(matches!(err, Err(EvalError::InvalidConfig(m)) if m.contains("seul qwen2")));
    }

    #[test]
    fn a_word_level_tokenizer_is_refused_with_its_type() {
        let dir = tempfile::tempdir().unwrap();
        tiny_model(dir.path());
        let tokenizer = dir.path().join("tokenizer.json");
        let text = std::fs::read_to_string(&tokenizer).unwrap();
        std::fs::write(&tokenizer, text.replace("\"BPE\"", "\"WordLevel\"")).unwrap();
        let err = convert_qwen2(dir.path(), &dir.path().join("m.gguf"), WeightType::F16);
        assert!(matches!(err, Err(EvalError::InvalidConfig(m)) if m.contains("WordLevel")));
    }

    #[test]
    fn maps_hugging_face_names() {
        assert_eq!(
            gguf_name("model.layers.12.self_attn.o_proj.weight")
                .unwrap()
                .as_deref(),
            Some("blk.12.attn_output.weight")
        );
        assert_eq!(
            gguf_name("model.layers.0.rotary_emb.inv_freq").unwrap(),
            None
        );
        assert!(gguf_name("model.layers.0.mlp.experts.0.w1").is_err());
    }
}
