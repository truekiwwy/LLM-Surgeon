//! SafeTensors -> GGUF in pure Rust, so a GPU llama.cpp build (Vulkan, HIP...) can run a
//! model on a machine with no python / torch. Families: Qwen2, Llama (2, 3, 3.1, 3.2) and
//! Mistral, Gemma and Gemma 2, Phi-3. Other architectures still go through llama.cpp's
//! `convert_hf_to_gguf.py` (see `export_gguf`).
//!
//! Same layout as that script: llama.cpp tensor names, GGUF dims (reversed), the Q/K rows of
//! Llama regrouped per rotary pair, Gemma norms stored as `1 + w`, the rope factor tensors of
//! Llama 3.1 and Phi-3 LongRoPE, 1-D tensors in f32. The vocabulary comes from
//! `tokenizer.json`, padded to the rows of the embedding: a byte-level BPE (Qwen2, Llama 3)
//! as `gpt2`, a SentencePiece BPE with byte fallback (Llama 2, Mistral, Gemma, Phi-3) as
//! `llama`.

use std::collections::HashMap;
use std::f64::consts::PI;
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
const TOKEN_UNKNOWN: i32 = 2;
const TOKEN_CONTROL: i32 = 3;
const TOKEN_USER_DEFINED: i32 = 4;
const TOKEN_UNUSED: i32 = 5;
const TOKEN_BYTE: i32 = 6;
/// GGUF metadata value tags.
const TAG_I32: u32 = 5;
const TAG_U32: u32 = 4;
const TAG_F32: u32 = 6;
const TAG_BOOL: u32 = 7;
const TAG_STR: u32 = 8;
const TAG_ARRAY: u32 = 9;
/// Elements converted per buffer, so the 150k x 896 embedding is never copied whole.
const CHUNK_ELEMENTS: usize = 1 << 16;
/// The `model_type` values converted natively.
const SUPPORTED: &str = "qwen2, llama, mistral, gemma, gemma2, phi3";
/// The split regex of the Llama 3 tokenizer (llama.cpp pre-tokenizer `llama-bpe`).
const LLAMA3_SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

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

/// The llama.cpp architecture a `model_type` converts to (Mistral is a Llama for llama.cpp).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arch {
    Qwen2,
    Llama,
    Gemma,
    Gemma2,
    Phi3,
}

impl Arch {
    fn from_config(config: &Value) -> Result<Self> {
        match config["model_type"].as_str() {
            Some("qwen2") => Ok(Self::Qwen2),
            Some("llama" | "mistral") => Ok(Self::Llama),
            Some("gemma") => Ok(Self::Gemma),
            Some("gemma2") => Ok(Self::Gemma2),
            Some("phi3") => Ok(Self::Phi3),
            _ => Err(invalid(format!(
                "conversion native : model_type {} non géré ({SUPPORTED}) ; passez \
                 --llama-cpp-dir vers un llama.cpp qui contient convert_hf_to_gguf.py",
                config["model_type"]
            ))),
        }
    }

    /// `general.architecture`, also the prefix of the model keys.
    fn name(self) -> &'static str {
        match self {
            Self::Qwen2 => "qwen2",
            Self::Llama => "llama",
            Self::Gemma => "gemma",
            Self::Gemma2 => "gemma2",
            Self::Phi3 => "phi3",
        }
    }

    fn is_gemma(self) -> bool {
        matches!(self, Self::Gemma | Self::Gemma2)
    }
}

/// The attention shape of `config.json`.
struct Heads {
    heads: u64,
    kv: u64,
    dim: u64,
    /// `head_dim` is given (it may differ from hidden / heads).
    explicit_dim: bool,
}

impl Heads {
    fn read(config: &Value) -> Result<Self> {
        let heads = config_u64(config, "num_attention_heads")?;
        if heads == 0 {
            return Err(invalid("config.json : num_attention_heads vaut 0"));
        }
        let explicit = config["head_dim"].as_u64();
        Ok(Self {
            heads,
            kv: config["num_key_value_heads"].as_u64().unwrap_or(heads),
            dim: match explicit {
                Some(d) => d,
                None => config_u64(config, "hidden_size")? / heads,
            },
            explicit_dim: explicit.is_some(),
        })
    }
}

/// Where the bytes of a tensor come from.
enum Source<'a> {
    Tensor {
        model: &'a ReadOnlyModel,
        name: &'a str,
    },
    /// Computed by the converter (rope factors), f32.
    Values(Vec<f32>),
}

struct Planned<'a> {
    name: String,
    source: Source<'a>,
    ne: Vec<u64>,
    ggml: u32,
    bytes: u64,
    /// Llama Q/K: the number of heads whose rows are regrouped per rotary pair.
    permute_heads: Option<usize>,
    /// Added to every value (Gemma stores its norms as `w` for `1 + w`).
    offset: f32,
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

/// llama.cpp name of a tensor; `None` for buffers that are not weights and for the Gemma
/// head, always tied to the embedding.
fn gguf_name(arch: Arch, hf: &str) -> Result<Option<String>> {
    let unknown = || invalid(format!("tenseur {} inconnu : {hf}", arch.name()));
    match hf {
        "model.embed_tokens.weight" => return Ok(Some("token_embd.weight".into())),
        "model.norm.weight" => return Ok(Some("output_norm.weight".into())),
        "lm_head.weight" if arch.is_gemma() => return Ok(None),
        "lm_head.weight" => return Ok(Some("output.weight".into())),
        _ if hf.ends_with("rotary_emb.inv_freq") => return Ok(None),
        _ => {}
    }
    let rest = hf.strip_prefix("model.layers.").ok_or_else(unknown)?;
    let (layer, rest) = rest.split_once('.').ok_or_else(unknown)?;
    let (stem, kind) = rest
        .rsplit_once('.')
        .filter(|(_, kind)| matches!(*kind, "weight" | "bias"))
        .ok_or_else(unknown)?;
    let mapped = match (arch, stem) {
        (Arch::Gemma2, "post_attention_layernorm") => "post_attention_norm",
        (Arch::Gemma2, "pre_feedforward_layernorm") => "ffn_norm",
        (Arch::Gemma2, "post_feedforward_layernorm") => "post_ffw_norm",
        (Arch::Phi3, "self_attn.qkv_proj") => "attn_qkv",
        // Gate and up stacked: llama.cpp splits the 2 x n_ff rows itself.
        (Arch::Phi3, "mlp.gate_up_proj") => "ffn_up",
        (_, "input_layernorm") => "attn_norm",
        (_, "post_attention_layernorm") => "ffn_norm",
        (_, "self_attn.q_proj") => "attn_q",
        (_, "self_attn.k_proj") => "attn_k",
        (_, "self_attn.v_proj") => "attn_v",
        (_, "self_attn.o_proj") => "attn_output",
        (_, "mlp.gate_proj") => "ffn_gate",
        (_, "mlp.up_proj") => "ffn_up",
        (_, "mlp.down_proj") => "ffn_down",
        _ => return Err(unknown()),
    };
    Ok(Some(format!("blk.{layer}.{mapped}.{kind}")))
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

    fn bool(&mut self, key: &str, v: bool) {
        self.put(key, TAG_BOOL, &[u8::from(v)]);
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
            "tokenizer.json : seul un tokenizer BPE (byte-level ou SentencePiece) se convertit, \
             pas « {kind} »"
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

/// SentencePiece scores for llama.cpp, which merges the pair whose result scores highest:
/// minus the rank of the first merge producing the piece, so the BPE order is kept. A piece
/// no merge produces scores below every merge.
fn piece_scores(tokens: &[String], merges: &[String]) -> Vec<f32> {
    let mut rank: HashMap<String, usize> = HashMap::new();
    for (i, merge) in merges.iter().enumerate() {
        if let Some((a, b)) = merge.split_once(' ') {
            rank.entry(format!("{a}{b}")).or_insert(i);
        }
    }
    let lowest = -(merges.len() as f32) - 1.0;
    tokens
        .iter()
        .map(|t| rank.get(t).map_or(lowest, |&r| -(r as f32)))
        .collect()
}

/// `<0x0A>`: a byte fallback piece.
fn is_byte_piece(text: &str) -> bool {
    text.len() == 6
        && text.starts_with("<0x")
        && text.ends_with('>')
        && text[3..5].bytes().all(|b| b.is_ascii_hexdigit())
}

/// The llama.cpp pre-tokenizer of a byte-level BPE. llama.cpp names it, `tokenizer.json`
/// only holds its regex: an unknown one is refused rather than tokenized wrongly.
fn bpe_pre(arch: Arch, tokenizer: &Value) -> Result<&'static str> {
    if arch == Arch::Qwen2 {
        return Ok("qwen2");
    }
    let llama3 = tokenizer["pre_tokenizer"]["pretokenizers"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|p| p["pattern"]["Regex"].as_str() == Some(LLAMA3_SPLIT));
    match llama3 {
        true => Ok("llama-bpe"),
        false => Err(invalid(format!(
            "tokenizer.json : BPE byte-level dont le pré-découpage n'est pas celui de Llama 3 \
             ({}) ; passez --llama-cpp-dir vers un llama.cpp qui contient convert_hf_to_gguf.py",
            arch.name()
        ))),
    }
}

/// The `rope_scaling` type, `""` without one.
fn rope_kind(config: &Value) -> &str {
    let scaling = &config["rope_scaling"];
    scaling["rope_type"]
        .as_str()
        .or_else(|| scaling["type"].as_str())
        .unwrap_or("")
}

/// Phi-3 rotary dimensions: `partial_rotary_factor` of each head.
fn phi3_rope_dims(config: &Value, heads: &Heads) -> Result<u64> {
    let factor = config["partial_rotary_factor"].as_f64().unwrap_or(1.0);
    Ok((factor * config_u64(config, "hidden_size")? as f64) as u64 / heads.heads)
}

/// Tensors llama.cpp reads but `rope_scaling` describes: the Llama 3.1 frequency factors and
/// the Phi-3 LongRoPE factors. Another scaling of those families is refused (the model would
/// run with a wrong rotary embedding); Qwen2 and Gemma ignore it, as before.
fn rope_tensors(arch: Arch, config: &Value, heads: &Heads) -> Result<Vec<(String, Vec<f32>)>> {
    let scaling = &config["rope_scaling"];
    match (arch, rope_kind(config)) {
        (_, "") | (Arch::Qwen2 | Arch::Gemma | Arch::Gemma2, _) => Ok(Vec::new()),
        (Arch::Llama, "llama3") => {
            let get = |key: &str, default: f64| scaling[key].as_f64().unwrap_or(default);
            let base = config["rope_theta"].as_f64().unwrap_or(10_000.0);
            let factor = get("factor", 8.0);
            let (low, high) = (get("low_freq_factor", 1.0), get("high_freq_factor", 4.0));
            let old = get("original_max_position_embeddings", 8192.0);
            let dim = heads.dim as f64;
            let factors = (0..heads.dim)
                .step_by(2)
                .map(|i| {
                    let wavelen = 2.0 * PI * base.powf(i as f64 / dim);
                    let f = if wavelen < old / high {
                        1.0
                    } else if wavelen > old / low {
                        factor
                    } else {
                        let smooth = (old / wavelen - low) / (high - low);
                        1.0 / ((1.0 - smooth) / factor + smooth)
                    };
                    f as f32
                })
                .collect();
            Ok(vec![("rope_freqs.weight".into(), factors)])
        }
        (Arch::Phi3, "su" | "longrope") => {
            let half = phi3_rope_dims(config, heads)? as usize / 2;
            let factors = |key: &str| -> Result<Vec<f32>> {
                let list = scaling[key].as_array().filter(|l| l.len() == half);
                let list = list.ok_or_else(|| {
                    invalid(format!(
                        "config.json : rope_scaling.{key} absent ou de longueur autre que {half}"
                    ))
                })?;
                list.iter()
                    .map(|v| {
                        v.as_f64().map(|v| v as f32).ok_or_else(|| {
                            invalid(format!("config.json : rope_scaling.{key} non numérique"))
                        })
                    })
                    .collect()
            };
            Ok(vec![
                ("rope_factors_long.weight".into(), factors("long_factor")?),
                ("rope_factors_short.weight".into(), factors("short_factor")?),
            ])
        }
        (_, kind) => Err(invalid(format!(
            "conversion native : rope_scaling « {kind} » non géré pour {} ; passez \
             --llama-cpp-dir vers un llama.cpp qui contient convert_hf_to_gguf.py",
            arch.name()
        ))),
    }
}

/// The `<arch>.*` keys llama.cpp reads for the family.
fn model_metadata(m: &mut Meta, arch: Arch, config: &Value, heads: &Heads) -> Result<()> {
    let a = arch.name();
    for (key, from) in [
        ("block_count", "num_hidden_layers"),
        ("context_length", "max_position_embeddings"),
        ("embedding_length", "hidden_size"),
        ("feed_forward_length", "intermediate_size"),
        ("attention.head_count", "num_attention_heads"),
    ] {
        m.u32(&format!("{a}.{key}"), config_u64(config, from)?);
    }
    m.u32(&format!("{a}.attention.head_count_kv"), heads.kv);
    m.f32(
        &format!("{a}.rope.freq_base"),
        config["rope_theta"].as_f64().unwrap_or(10_000.0) as f32,
    );
    m.f32(
        &format!("{a}.attention.layer_norm_rms_epsilon"),
        config["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
    );
    if heads.explicit_dim {
        m.u32(&format!("{a}.attention.key_length"), heads.dim);
        m.u32(&format!("{a}.attention.value_length"), heads.dim);
    }
    match arch {
        Arch::Llama => m.u32(&format!("{a}.rope.dimension_count"), heads.dim),
        Arch::Gemma2 => {
            for key in ["attn_logit_softcapping", "final_logit_softcapping"] {
                if let Some(v) = config[key].as_f64() {
                    m.f32(&format!("{a}.{key}"), v as f32);
                }
            }
            if let Some(window) = config["sliding_window"].as_u64() {
                m.u32(&format!("{a}.attention.sliding_window"), window);
            }
        }
        Arch::Phi3 => {
            m.u32(
                &format!("{a}.rope.dimension_count"),
                phi3_rope_dims(config, heads)?,
            );
            let max = config_u64(config, "max_position_embeddings")?;
            let orig = config["original_max_position_embeddings"]
                .as_u64()
                .unwrap_or(max);
            m.u32(&format!("{a}.rope.scaling.original_context_length"), orig);
            if let Some(window) = config["sliding_window"].as_u64() {
                m.u32(&format!("{a}.attention.sliding_window"), window);
            }
            if matches!(rope_kind(config), "su" | "longrope") {
                let scale = max as f64 / orig.max(1) as f64;
                let factor = match scale > 1.0 {
                    true => (1.0 + scale.ln() / (orig as f64).ln()).sqrt(),
                    false => 1.0,
                };
                m.f32(&format!("{a}.rope.scaling.attn_factor"), factor as f32);
            }
        }
        Arch::Qwen2 | Arch::Gemma => {}
    }
    Ok(())
}

fn metadata(
    dir: &Path,
    config: &Value,
    arch: Arch,
    heads: &Heads,
    weights: WeightType,
    n_vocab: usize,
) -> Result<Meta> {
    let tokenizer = read_json(&dir.join("tokenizer.json"))?;
    // Optional: the pad token, add_bos_token and the chat template come from it.
    let tok_config = read_json(&dir.join("tokenizer_config.json")).unwrap_or(Value::Null);
    let (tokens, mut types) = vocabulary(&tokenizer, n_vocab)?;
    let sentencepiece = tokenizer["model"]["byte_fallback"].as_bool() == Some(true);
    let name = dir
        .file_name()
        .map_or_else(|| "model".into(), |n| n.to_string_lossy());

    let mut m = Meta::default();
    m.text("general.architecture", arch.name());
    m.text("general.name", &name);
    m.u32("general.file_type", u64::from(weights == WeightType::F16));
    m.u32("general.quantization_version", 2);
    model_metadata(&mut m, arch, config, heads)?;

    let unk = tokenizer["model"]["unk_token"]
        .as_str()
        .and_then(|u| tokens.iter().position(|t| t == u));
    let merges = merges(&tokenizer);
    if sentencepiece {
        m.text("tokenizer.ggml.model", "llama");
        for (kind, text) in types.iter_mut().zip(&tokens) {
            if is_byte_piece(text) {
                *kind = TOKEN_BYTE;
            }
        }
        if let Some(unk) = unk {
            types[unk] = TOKEN_UNKNOWN;
        }
    } else {
        m.text("tokenizer.ggml.model", "gpt2");
        m.text("tokenizer.ggml.pre", bpe_pre(arch, &tokenizer)?);
    }
    m.strings("tokenizer.ggml.tokens", &tokens);
    let kinds: Vec<u8> = types.iter().flat_map(|t| t.to_le_bytes()).collect();
    m.array("tokenizer.ggml.token_type", TAG_I32, types.len(), &kinds);
    if sentencepiece {
        // A SentencePiece tokenizer.json may come without merges: every score is then equal.
        let scores = piece_scores(&tokens, &merges.unwrap_or_default());
        let bytes: Vec<u8> = scores.iter().flat_map(|s| s.to_le_bytes()).collect();
        m.array("tokenizer.ggml.scores", TAG_F32, scores.len(), &bytes);
    } else {
        m.strings("tokenizer.ggml.merges", &merges?);
    }
    let pad = tok_config["pad_token"]
        .as_str()
        .and_then(|p| tokens.iter().position(|t| t == p));
    for (key, id) in [
        ("bos", first_id(&config["bos_token_id"])),
        ("eos", first_id(&config["eos_token_id"])),
        ("unknown", unk.filter(|_| sentencepiece).map(|u| u as u64)),
        ("padding", pad.map(|p| p as u64)),
    ] {
        if let Some(id) = id {
            m.u32(&format!("tokenizer.ggml.{key}_token_id"), id);
        }
    }
    // Qwen2 never adds one; the others follow tokenizer_config.json, BOS by default.
    let add_bos = arch != Arch::Qwen2 && tok_config["add_bos_token"].as_bool().unwrap_or(true);
    m.bool("tokenizer.ggml.add_bos_token", add_bos);
    if arch.is_gemma() {
        m.bool("tokenizer.ggml.add_space_prefix", false);
    }
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

fn plan<'a>(
    models: &'a [ReadOnlyModel],
    arch: Arch,
    heads: &Heads,
    weights: WeightType,
) -> Result<Vec<Planned<'a>>> {
    let mut planned = Vec::new();
    for model in models {
        for info in &model.index().tensors {
            let Some(name) = gguf_name(arch, &info.name)? else {
                continue;
            };
            if info.dtype.size_of().is_none() {
                return Err(invalid(format!(
                    "{} : dtype {} non géré",
                    info.name, info.dtype
                )));
            }
            let permute_heads = match (arch, name.split('.').nth(2)) {
                (Arch::Llama, Some("attn_q")) => Some(heads.heads as usize),
                (Arch::Llama, Some("attn_k")) => Some(heads.kv as usize),
                _ => None,
            };
            if let Some(n) = permute_heads {
                let rows = info.shape.first().copied().unwrap_or(0);
                if n == 0 || rows % n != 0 || (rows / n) % 2 != 0 {
                    return Err(invalid(format!(
                        "{} : {rows} lignes ne se répartissent pas en {n} têtes paires",
                        info.name
                    )));
                }
            }
            let ggml = if info.shape.len() == 1 {
                GGML_F32
            } else {
                weights.ggml()
            };
            let width = if ggml == GGML_F16 { 2 } else { 4 };
            let elements: usize = info.shape.iter().product();
            let offset = match arch.is_gemma() && name.ends_with("norm.weight") {
                true => 1.0,
                false => 0.0,
            };
            planned.push(Planned {
                name,
                source: Source::Tensor {
                    model,
                    name: &info.name,
                },
                // GGUF lists the fastest-varying dimension first.
                ne: info.shape.iter().rev().map(|&d| d as u64).collect(),
                ggml,
                bytes: (elements * width) as u64,
                permute_heads,
                offset,
            });
        }
    }
    Ok(planned)
}

/// Converts one tensor to `ggml` in chunks. A finite value that overflows f16 is an
/// error: the file would silently carry an inf.
fn write_tensor(out: &mut impl Write, t: &Planned) -> Result<()> {
    let wrote = |e: std::io::Error| invalid(e.to_string());
    let (model, source) = match &t.source {
        Source::Values(values) => {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            return out.write_all(&bytes).map_err(wrote);
        }
        Source::Tensor { model, name } => (model, name),
    };
    let view = model.tensor(source).map_err(|e| invalid(e.to_string()))?;
    let load: fn(&[u8]) -> f32 = match view.info.dtype {
        DType::BF16 => |b| bf16_load([b[0], b[1]]),
        DType::F16 => |b| f16_load([b[0], b[1]]),
        _ => |b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
    };
    let width = view.info.dtype.size_of().unwrap_or(4);
    // Llama Q/K (llama.cpp's `permute`): row `i` of a head takes source row `i / 2` of its
    // first half when `i` is even, of its second half when odd.
    let slices: Vec<&[u8]> = match t.permute_heads {
        None => vec![view.bytes],
        Some(heads) => {
            let rows = view.info.shape[0];
            let (row_bytes, dim) = (view.bytes.len() / rows, rows / heads);
            (0..rows)
                .map(|r| {
                    let i = r % dim;
                    let src = r - i + (i % 2) * (dim / 2) + i / 2;
                    &view.bytes[src * row_bytes..(src + 1) * row_bytes]
                })
                .collect()
        }
    };
    let mut buf = Vec::with_capacity(CHUNK_ELEMENTS * 4);
    for chunk in slices.iter().flat_map(|s| s.chunks(CHUNK_ELEMENTS * width)) {
        buf.clear();
        for cell in chunk.chunks_exact(width) {
            let x = load(cell) + t.offset;
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
        out.write_all(&buf).map_err(wrote)?;
    }
    Ok(())
}

fn pad_to_alignment(out: &mut impl Write, written: u64) -> std::io::Result<()> {
    let pad = written.next_multiple_of(ALIGNMENT) - written;
    out.write_all(&vec![0; pad as usize])
}

/// Whether `dir` holds a `config.json` of a family converted natively.
pub(crate) fn is_convertible_dir(dir: &Path) -> bool {
    read_json(&dir.join("config.json")).is_ok_and(|c| Arch::from_config(&c).is_ok())
}

/// Writes `output` from the SafeTensors directory `dir` (Qwen2, Llama, Mistral, Gemma,
/// Gemma 2 or Phi-3). The model is only read.
pub fn convert_safetensors(dir: &Path, output: &Path, weights: WeightType) -> Result<()> {
    let config = read_json(&dir.join("config.json"))?;
    let arch = Arch::from_config(&config)?;
    let heads = Heads::read(&config)?;
    let models = safetensors_files(dir)?
        .iter()
        .map(|p| ReadOnlyModel::open(p).map_err(|e| invalid(format!("{} : {e}", p.display()))))
        .collect::<Result<Vec<_>>>()?;
    let mut planned = plan(&models, arch, &heads, weights)?;
    for (name, values) in rope_tensors(arch, &config, &heads)? {
        planned.push(Planned {
            name,
            ne: vec![values.len() as u64],
            ggml: GGML_F32,
            bytes: 4 * values.len() as u64,
            source: Source::Values(values),
            permute_heads: None,
            offset: 0.0,
        });
    }
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
    let meta = metadata(dir, &config, arch, &heads, weights, n_vocab)?;

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
    use crate::patch::gguf::{parse, parse::GgufFile, parse::MetaValue};
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

    /// A model directory where element i of every tensor is i / 4.
    fn write_model(dir: &Path, config: &Value, tokenizer: &Value, shapes: &[(String, Vec<usize>)]) {
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
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

    fn layer(names: &[(&str, Vec<usize>)]) -> Vec<(String, Vec<usize>)> {
        names
            .iter()
            .map(|(n, s)| (format!("model.layers.0.{n}"), s.clone()))
            .collect()
    }

    /// 4 vocabulary entries and one added token: a byte-level BPE.
    fn byte_level(split: Option<&str>) -> Value {
        let mut tokenizer = json!({
            "added_tokens": [{"id": 4, "content": "<|endoftext|>", "special": true}],
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2, "Ġ": 3}, "merges": ["a b"]},
        });
        if let Some(regex) = split {
            tokenizer["pre_tokenizer"] = json!({"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated"},
                {"type": "ByteLevel", "add_prefix_space": false},
            ]});
        }
        tokenizer
    }

    /// A SentencePiece BPE with byte fallback (Llama 2, Mistral, Gemma, Phi-3).
    fn sentencepiece() -> Value {
        json!({
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true},
                {"id": 1, "content": "<s>", "special": true},
                {"id": 2, "content": "</s>", "special": true},
            ],
            "model": {
                "type": "BPE", "byte_fallback": true, "unk_token": "<unk>",
                "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2, "<0x0A>": 3, "▁": 4, "a": 5, "▁a": 6},
                "merges": ["▁ a"],
            },
        })
    }

    fn convert(dir: &Path) -> (Vec<u8>, GgufFile) {
        let out = dir.join("m.gguf");
        convert_safetensors(dir, &out, WeightType::F16).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        let gguf = parse(&bytes).unwrap();
        (bytes, gguf)
    }

    /// The values of a tensor, f16 or f32.
    fn values(bytes: &[u8], gguf: &GgufFile, name: &str) -> Vec<f32> {
        let t = gguf.tensor(name).unwrap_or_else(|| panic!("{name} absent"));
        let n: u64 = t.ne.iter().product();
        let start = t.start as usize;
        match t.ggml_type.0 {
            GGML_F16 => bytes[start..start + 2 * n as usize]
                .chunks(2)
                .map(|b| f16_load([b[0], b[1]]))
                .collect(),
            _ => bytes[start..start + 4 * n as usize]
                .chunks(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
        }
    }

    fn meta_u64(gguf: &GgufFile, key: &str) -> Option<u64> {
        gguf.meta(key).and_then(MetaValue::as_u64)
    }

    fn meta_f32(gguf: &GgufFile, key: &str) -> Option<f32> {
        match gguf.meta(key) {
            Some(MetaValue::F32(v)) => Some(*v),
            _ => None,
        }
    }

    fn array_head(gguf: &GgufFile, key: &str) -> Vec<MetaValue> {
        match gguf.meta(key) {
            Some(MetaValue::Array { head, .. }) => head.clone(),
            other => panic!("{key} : {other:?}"),
        }
    }

    /// 1 layer, hidden 4, 6 token rows: 4 in the vocabulary, one added, one padding row.
    fn tiny_qwen2(dir: &Path) {
        let config = json!({
            "model_type": "qwen2", "num_hidden_layers": 1, "max_position_embeddings": 64,
            "hidden_size": 4, "intermediate_size": 8, "num_attention_heads": 2,
            "num_key_value_heads": 1, "rope_theta": 1e6, "rms_norm_eps": 1e-6,
            "tie_word_embeddings": true, "bos_token_id": 4, "eos_token_id": 5,
        });
        let mut shapes = vec![
            ("model.embed_tokens.weight".to_owned(), vec![6, 4]),
            ("model.norm.weight".to_owned(), vec![4]),
        ];
        shapes.extend(layer(&[
            ("input_layernorm.weight", vec![4]),
            ("self_attn.q_proj.weight", vec![4, 4]),
            ("self_attn.q_proj.bias", vec![4]),
            ("mlp.down_proj.weight", vec![4, 8]),
        ]));
        write_model(dir, &config, &byte_level(None), &shapes);
    }

    #[test]
    fn converts_a_tiny_qwen2_to_a_readable_gguf() {
        let dir = tempfile::tempdir().unwrap();
        tiny_qwen2(dir.path());
        let (bytes, gguf) = convert(dir.path());
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
        assert_eq!(
            gguf.meta("tokenizer.ggml.pre").and_then(MetaValue::as_str),
            Some("qwen2")
        );
        assert_eq!(
            gguf.meta("tokenizer.ggml.add_bos_token"),
            Some(&MetaValue::Bool(false))
        );
        // Values survive: embedding row 1 is [1.0, 1.25, 1.5, 1.75] as f16.
        assert_eq!(
            values(&bytes, &gguf, "token_embd.weight")[4..8],
            [1.0, 1.25, 1.5, 1.75]
        );
        // No permutation for Qwen2: q rows stay in order.
        assert_eq!(values(&bytes, &gguf, "blk.0.attn_q.weight")[4], 1.0);
    }

    /// Llama 3.1: hidden 8, 2 heads of 4, 1 KV head, untied head, llama3 rope scaling.
    #[test]
    fn llama_regroups_q_and_k_per_rotary_pair_and_computes_the_rope_factors() {
        let dir = tempfile::tempdir().unwrap();
        let config = json!({
            "model_type": "llama", "num_hidden_layers": 1, "max_position_embeddings": 131072,
            "hidden_size": 8, "intermediate_size": 16, "num_attention_heads": 2,
            "num_key_value_heads": 1, "rope_theta": 500000.0, "rms_norm_eps": 1e-5,
            "tie_word_embeddings": false, "bos_token_id": 4, "eos_token_id": [4],
            "rope_scaling": {"rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
                             "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
        });
        let mut shapes = vec![
            ("model.embed_tokens.weight".to_owned(), vec![5, 8]),
            ("model.norm.weight".to_owned(), vec![8]),
            ("lm_head.weight".to_owned(), vec![5, 8]),
        ];
        shapes.extend(layer(&[
            ("input_layernorm.weight", vec![8]),
            ("post_attention_layernorm.weight", vec![8]),
            ("self_attn.q_proj.weight", vec![8, 8]),
            ("self_attn.k_proj.weight", vec![4, 8]),
            ("self_attn.v_proj.weight", vec![4, 8]),
            ("self_attn.o_proj.weight", vec![8, 8]),
            ("mlp.gate_proj.weight", vec![16, 8]),
            ("mlp.up_proj.weight", vec![16, 8]),
            ("mlp.down_proj.weight", vec![8, 16]),
        ]));
        write_model(
            dir.path(),
            &config,
            &byte_level(Some(LLAMA3_SPLIT)),
            &shapes,
        );
        let (bytes, gguf) = convert(dir.path());

        assert_eq!(gguf.architecture(), Some("llama"));
        assert_eq!(meta_u64(&gguf, "llama.rope.dimension_count"), Some(4));
        assert_eq!(meta_u64(&gguf, "llama.attention.head_count_kv"), Some(1));
        assert_eq!(meta_u64(&gguf, "llama.context_length"), Some(131072));
        assert_eq!(meta_f32(&gguf, "llama.rope.freq_base"), Some(500000.0));
        assert_eq!(
            gguf.meta("tokenizer.ggml.pre").and_then(MetaValue::as_str),
            Some("llama-bpe")
        );
        assert_eq!(
            gguf.meta("tokenizer.ggml.add_bos_token"),
            Some(&MetaValue::Bool(true))
        );
        assert!(gguf.tensor("output.weight").is_some());
        assert!(gguf.tensor("blk.0.ffn_norm.weight").is_some());
        // Row r of HF starts with 8r / 4 = 2r; each head of 4 rows becomes [0, 2, 1, 3].
        let first = |name: &str, cols: usize| -> Vec<f32> {
            values(&bytes, &gguf, name)
                .chunks(cols)
                .map(|r| r[0])
                .collect()
        };
        let q: Vec<f32> = [0, 2, 1, 3, 4, 6, 5, 7]
            .iter()
            .map(|r| 2.0 * *r as f32)
            .collect();
        assert_eq!(first("blk.0.attn_q.weight", 8), q);
        assert_eq!(first("blk.0.attn_k.weight", 8), [0.0, 4.0, 2.0, 6.0]);
        assert_eq!(first("blk.0.attn_v.weight", 8), [0.0, 2.0, 4.0, 6.0]);
        // One factor per rotary pair: the fast pair keeps 1, the slow one is smoothed.
        let freqs = values(&bytes, &gguf, "rope_freqs.weight");
        assert_eq!(freqs.len(), 2);
        assert_eq!(freqs[0], 1.0);
        assert!((freqs[1] - 2.6946).abs() < 1e-3, "{freqs:?}");
    }

    #[test]
    fn mistral_with_a_sentencepiece_vocabulary_becomes_a_llama_spm_vocabulary() {
        let dir = tempfile::tempdir().unwrap();
        let config = json!({
            "model_type": "mistral", "num_hidden_layers": 1, "max_position_embeddings": 32,
            "hidden_size": 4, "intermediate_size": 8, "num_attention_heads": 2,
            "num_key_value_heads": 2, "rope_theta": 1e6, "rms_norm_eps": 1e-5,
            "bos_token_id": 1, "eos_token_id": 2, "sliding_window": null,
        });
        let shapes = vec![
            ("model.embed_tokens.weight".to_owned(), vec![8, 4]),
            ("model.norm.weight".to_owned(), vec![4]),
        ];
        write_model(dir.path(), &config, &sentencepiece(), &shapes);
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            json!({"add_bos_token": true, "chat_template": "{{ x }}"}).to_string(),
        )
        .unwrap();
        let (_, gguf) = convert(dir.path());

        assert_eq!(gguf.architecture(), Some("llama"));
        assert_eq!(
            gguf.meta("tokenizer.ggml.model")
                .and_then(MetaValue::as_str),
            Some("llama")
        );
        assert!(gguf.meta("tokenizer.ggml.pre").is_none());
        assert!(gguf.meta("tokenizer.ggml.merges").is_none());
        let types: Vec<i32> = array_head(&gguf, "tokenizer.ggml.token_type")
            .iter()
            .map(|v| match v {
                MetaValue::I32(t) => *t,
                other => panic!("{other:?}"),
            })
            .collect();
        let expected = [
            TOKEN_UNKNOWN,
            TOKEN_CONTROL,
            TOKEN_CONTROL,
            TOKEN_BYTE,
            TOKEN_NORMAL,
            TOKEN_NORMAL,
            TOKEN_NORMAL,
            TOKEN_UNUSED,
        ];
        assert_eq!(types, expected);
        // "▁a" comes from merge 0; nothing else is a merge result.
        let scores = array_head(&gguf, "tokenizer.ggml.scores");
        assert_eq!(scores[6], MetaValue::F32(0.0));
        assert_eq!(scores[5], MetaValue::F32(-2.0));
        assert_eq!(meta_u64(&gguf, "tokenizer.ggml.unknown_token_id"), Some(0));
        assert_eq!(meta_u64(&gguf, "tokenizer.ggml.bos_token_id"), Some(1));
        assert_eq!(
            gguf.meta("tokenizer.chat_template")
                .and_then(MetaValue::as_str),
            Some("{{ x }}")
        );
    }

    #[test]
    fn gemma_norms_store_one_plus_w_and_the_head_stays_tied() {
        let dir = tempfile::tempdir().unwrap();
        let config = json!({
            "model_type": "gemma", "num_hidden_layers": 1, "max_position_embeddings": 32,
            "hidden_size": 4, "intermediate_size": 8, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": 8, "rope_theta": 10000.0,
            "rms_norm_eps": 1e-6, "bos_token_id": 1, "eos_token_id": 2,
        });
        let mut shapes = vec![
            ("model.embed_tokens.weight".to_owned(), vec![8, 4]),
            ("model.norm.weight".to_owned(), vec![4]),
            ("lm_head.weight".to_owned(), vec![8, 4]),
        ];
        shapes.extend(layer(&[
            ("input_layernorm.weight", vec![4]),
            ("self_attn.q_proj.weight", vec![8, 4]),
        ]));
        write_model(dir.path(), &config, &sentencepiece(), &shapes);
        let (bytes, gguf) = convert(dir.path());

        assert_eq!(gguf.architecture(), Some("gemma"));
        assert!(gguf.tensor("output.weight").is_none());
        assert_eq!(
            values(&bytes, &gguf, "blk.0.attn_norm.weight"),
            [1.0, 1.25, 1.5, 1.75]
        );
        assert_eq!(
            values(&bytes, &gguf, "output_norm.weight"),
            [1.0, 1.25, 1.5, 1.75]
        );
        // Matrices are not shifted, and Gemma's rotary embedding needs no permutation.
        assert_eq!(values(&bytes, &gguf, "blk.0.attn_q.weight")[4], 1.0);
        assert_eq!(meta_u64(&gguf, "gemma.attention.key_length"), Some(8));
        assert_eq!(meta_u64(&gguf, "gemma.attention.value_length"), Some(8));
        assert_eq!(
            gguf.meta("tokenizer.ggml.add_space_prefix"),
            Some(&MetaValue::Bool(false))
        );
    }

    #[test]
    fn gemma2_maps_its_four_norms_and_writes_the_softcaps() {
        let dir = tempfile::tempdir().unwrap();
        let config = json!({
            "model_type": "gemma2", "num_hidden_layers": 1, "max_position_embeddings": 32,
            "hidden_size": 4, "intermediate_size": 8, "num_attention_heads": 1,
            "num_key_value_heads": 1, "head_dim": 4, "rms_norm_eps": 1e-6,
            "attn_logit_softcapping": 50.0, "final_logit_softcapping": 30.0,
            "sliding_window": 4096,
        });
        let mut shapes = vec![("model.embed_tokens.weight".to_owned(), vec![8, 4])];
        shapes.extend(layer(&[
            ("input_layernorm.weight", vec![4]),
            ("post_attention_layernorm.weight", vec![4]),
            ("pre_feedforward_layernorm.weight", vec![4]),
            ("post_feedforward_layernorm.weight", vec![4]),
        ]));
        write_model(dir.path(), &config, &sentencepiece(), &shapes);
        let (bytes, gguf) = convert(dir.path());

        assert_eq!(gguf.architecture(), Some("gemma2"));
        for name in [
            "blk.0.attn_norm.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.ffn_norm.weight",
            "blk.0.post_ffw_norm.weight",
        ] {
            assert_eq!(values(&bytes, &gguf, name)[0], 1.0, "{name}");
        }
        assert_eq!(meta_f32(&gguf, "gemma2.attn_logit_softcapping"), Some(50.0));
        assert_eq!(
            meta_f32(&gguf, "gemma2.final_logit_softcapping"),
            Some(30.0)
        );
        assert_eq!(
            meta_u64(&gguf, "gemma2.attention.sliding_window"),
            Some(4096)
        );
    }

    #[test]
    fn phi3_keeps_its_fused_projections_and_longrope_factors() {
        let dir = tempfile::tempdir().unwrap();
        let config = json!({
            "model_type": "phi3", "num_hidden_layers": 1, "max_position_embeddings": 131072,
            "original_max_position_embeddings": 4096, "hidden_size": 8,
            "intermediate_size": 16, "num_attention_heads": 2, "num_key_value_heads": 2,
            "rope_theta": 10000.0, "rms_norm_eps": 1e-5, "sliding_window": 262144,
            "rope_scaling": {"type": "longrope", "long_factor": [1.5, 2.5],
                             "short_factor": [1.0, 1.25]},
        });
        let mut shapes = vec![
            ("model.embed_tokens.weight".to_owned(), vec![8, 8]),
            ("lm_head.weight".to_owned(), vec![8, 8]),
        ];
        shapes.extend(layer(&[
            ("self_attn.qkv_proj.weight", vec![24, 8]),
            ("mlp.gate_up_proj.weight", vec![32, 8]),
        ]));
        write_model(dir.path(), &config, &sentencepiece(), &shapes);
        let (bytes, gguf) = convert(dir.path());

        assert_eq!(gguf.architecture(), Some("phi3"));
        assert_eq!(gguf.tensor("blk.0.attn_qkv.weight").unwrap().ne, [8, 24]);
        assert_eq!(gguf.tensor("blk.0.ffn_up.weight").unwrap().ne, [8, 32]);
        assert!(gguf.tensor("output.weight").is_some());
        assert_eq!(
            values(&bytes, &gguf, "rope_factors_long.weight"),
            [1.5, 2.5]
        );
        assert_eq!(
            values(&bytes, &gguf, "rope_factors_short.weight"),
            [1.0, 1.25]
        );
        assert_eq!(meta_u64(&gguf, "phi3.rope.dimension_count"), Some(4));
        assert_eq!(
            meta_u64(&gguf, "phi3.rope.scaling.original_context_length"),
            Some(4096)
        );
        // sqrt(1 + ln(32) / ln(4096))
        let factor = meta_f32(&gguf, "phi3.rope.scaling.attn_factor").unwrap();
        assert!((factor - 1.1902).abs() < 1e-3, "{factor}");
    }

    #[test]
    fn unsupported_inputs_are_refused_with_their_reason() {
        let refused = |config: Value, tokenizer: Value| -> String {
            let dir = tempfile::tempdir().unwrap();
            let shapes = vec![("model.embed_tokens.weight".to_owned(), vec![5, 4])];
            write_model(dir.path(), &config, &tokenizer, &shapes);
            match convert_safetensors(dir.path(), &dir.path().join("m.gguf"), WeightType::F16) {
                Err(EvalError::InvalidConfig(m)) => m,
                other => panic!("{other:?}"),
            }
        };
        let base = json!({
            "model_type": "llama", "num_hidden_layers": 1, "max_position_embeddings": 32,
            "hidden_size": 4, "intermediate_size": 8, "num_attention_heads": 2,
        });
        let mut falcon = base.clone();
        falcon["model_type"] = json!("falcon");
        assert!(refused(falcon, byte_level(None)).contains("non géré"));
        // A byte-level BPE whose split is not Llama 3's (Mistral Nemo's Tekken, for one).
        assert!(refused(base.clone(), byte_level(Some("other"))).contains("Llama 3"));
        let mut yarn = base.clone();
        yarn["rope_scaling"] = json!({"type": "yarn", "factor": 4.0});
        assert!(refused(yarn, byte_level(Some(LLAMA3_SPLIT))).contains("yarn"));
        let mut words = byte_level(None);
        words["model"]["type"] = json!("WordLevel");
        let mut qwen2 = base;
        qwen2["model_type"] = json!("qwen2");
        assert!(refused(qwen2, words).contains("WordLevel"));
    }

    #[test]
    fn maps_hugging_face_names() {
        let name = |arch, hf| gguf_name(arch, hf).unwrap();
        assert_eq!(
            name(Arch::Qwen2, "model.layers.12.self_attn.o_proj.weight").as_deref(),
            Some("blk.12.attn_output.weight")
        );
        assert_eq!(
            name(Arch::Llama, "model.layers.0.rotary_emb.inv_freq"),
            None
        );
        assert_eq!(
            name(
                Arch::Llama,
                "model.layers.3.post_attention_layernorm.weight"
            )
            .as_deref(),
            Some("blk.3.ffn_norm.weight")
        );
        assert_eq!(
            name(
                Arch::Gemma2,
                "model.layers.3.post_attention_layernorm.weight"
            )
            .as_deref(),
            Some("blk.3.post_attention_norm.weight")
        );
        assert_eq!(name(Arch::Gemma, "lm_head.weight"), None);
        assert_eq!(
            name(Arch::Phi3, "model.layers.1.mlp.gate_up_proj.weight").as_deref(),
            Some("blk.1.ffn_up.weight")
        );
        assert!(gguf_name(Arch::Llama, "model.layers.0.mlp.gate_up_proj.weight").is_err());
        assert!(gguf_name(Arch::Qwen2, "model.layers.0.mlp.experts.0.w1").is_err());
        assert!(is_byte_piece("<0x0A>") && !is_byte_piece("<0xZZ>") && !is_byte_piece("<s>"));
    }
}
