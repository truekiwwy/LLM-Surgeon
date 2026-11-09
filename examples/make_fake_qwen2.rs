//! Writes the deterministic mini Qwen2 of the test suite (4 layers, hidden size 16,
//! vocabulary 64, F32, one file) and two prompt files, to try `surgeon extract` and the
//! extension API without a real model.
//!
//! ```text
//! cargo run --example make_fake_qwen2 -- <dir>
//! ```
//!
//! - `<dir>/qwen2-mini/`: `config.json`, `tokenizer.json` (word level), `model.safetensors`.
//! - `<dir>/pos.json`, `<dir>/neg.json`: six prompts each, made of vocabulary words.
//!
//! The bytes are those of `write_fake_qwen2_dir(dir, "F32", false)` and
//! `write_prompt_files` in `tests/integration.rs`. `tests/compat.rs`, `tests/plugin_api.rs`
//! and `examples/plugin_extraction_example.rs` include this file with `#[path]`.

use std::error::Error;
use std::path::{Path, PathBuf};

use serde_json::{json, Value};

pub type Res<T> = Result<T, Box<dyn Error + Send + Sync>>;

const POS_WORDS: [&str; 12] = [
    "joie", "soleil", "rire", "fete", "danse", "musique", "ami", "lumiere", "chaleur", "sourire",
    "jardin", "plage",
];
const NEG_WORDS: [&str; 12] = [
    "pluie", "nuit", "froid", "brume", "silence", "gris", "orage", "vide", "hiver", "ombre",
    "perte", "peur",
];
const FILLER: [&str; 10] = [
    "le", "la", "un", "une", "est", "et", "tres", "il", "elle", "nous",
];
pub const HIDDEN: usize = 16;
const INTER: usize = 32;
pub const LAYERS: usize = 4;
const VOCAB: usize = 64;

struct Rng(u64);

impl Rng {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    }
}

fn vocab() -> Vec<String> {
    ["[UNK]", "<|im_start|>", "<|im_end|>", "user", "assistant"]
        .iter()
        .chain(&POS_WORDS)
        .chain(&NEG_WORDS)
        .chain(&FILLER)
        .map(|w| w.to_string())
        .collect()
}

/// Every tensor of the 4-layer model, final norm and `lm_head` included.
fn qwen_tensors() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut rng = Rng(42);
    let head = HIDDEN / 2;
    let mut shapes = vec![("model.embed_tokens.weight".to_owned(), vec![VOCAB, HIDDEN])];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        for (suffix, shape) in [
            ("self_attn.q_proj.weight", vec![2 * head, HIDDEN]),
            ("self_attn.q_proj.bias", vec![2 * head]),
            ("self_attn.k_proj.weight", vec![2 * head, HIDDEN]),
            ("self_attn.k_proj.bias", vec![2 * head]),
            ("self_attn.v_proj.weight", vec![2 * head, HIDDEN]),
            ("self_attn.v_proj.bias", vec![2 * head]),
            ("self_attn.o_proj.weight", vec![HIDDEN, 2 * head]),
            ("mlp.gate_proj.weight", vec![INTER, HIDDEN]),
            ("mlp.up_proj.weight", vec![INTER, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, INTER]),
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
        ] {
            shapes.push((format!("{p}.{suffix}"), shape));
        }
    }
    shapes.push(("model.norm.weight".into(), vec![HIDDEN]));
    shapes.push(("lm_head.weight".into(), vec![VOCAB, HIDDEN]));
    shapes
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let values = (0..n)
                .map(|_| match name.contains("norm") {
                    true => 1.0 + rng.next(0.1),
                    false => rng.next(0.4),
                })
                .collect();
            (name, shape, values)
        })
        .collect()
}

/// F32 SafeTensors, header padded to 8 bytes like the test helper.
fn write_tensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) -> Res<()> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        data.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        header.insert(
            name.clone(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = serde_json::to_string(&header)?;
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    Ok(std::fs::write(path, file)?)
}

fn write_tokenizer(path: &Path) -> Res<()> {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::{AddedToken, Tokenizer};
    let words = vocab().into_iter().enumerate();
    let model = WordLevel::builder()
        .vocab(words.map(|(i, w)| (w, i as u32)).collect())
        .unk_token("[UNK]".into())
        .build()?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer.add_special_tokens([
        AddedToken::from("<|im_start|>", true),
        AddedToken::from("<|im_end|>", true),
    ])?;
    tokenizer.save(path, false)
}

fn qwen_config() -> Value {
    json!({"architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2", "vocab_size": VOCAB,
        "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": LAYERS,
        "num_attention_heads": 2, "num_key_value_heads": 2, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false,
        "sliding_window": 128, "hidden_act": "silu", "torch_dtype": "float32"})
}

/// `<dir>/qwen2-mini` with `config.json`, `tokenizer.json` and `model.safetensors`.
pub fn write_fake_qwen2(dir: &Path) -> Res<PathBuf> {
    let model = dir.join("qwen2-mini");
    std::fs::create_dir_all(&model)?;
    std::fs::write(model.join("config.json"), qwen_config().to_string())?;
    write_tokenizer(&model.join("tokenizer.json"))?;
    write_tensors(&model.join("model.safetensors"), &qwen_tensors())?;
    Ok(model)
}

/// `<dir>/pos.json` and `<dir>/neg.json`: six prompts each.
pub fn write_prompt_files(dir: &Path) -> Res<(PathBuf, PathBuf)> {
    let make = |words: &[&str]| -> Vec<String> {
        (0..6)
            .map(|i| format!("{} {} est {}", FILLER[i % 4], words[i], words[(i + 3) % 12]))
            .collect()
    };
    let (pos, neg) = (dir.join("pos.json"), dir.join("neg.json"));
    std::fs::write(&pos, serde_json::to_string(&make(&POS_WORDS))?)?;
    std::fs::write(&neg, serde_json::to_string(&make(&NEG_WORDS))?)?;
    Ok((pos, neg))
}

fn main() -> Res<()> {
    let dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: make_fake_qwen2 <dir>")?;
    std::fs::create_dir_all(&dir)?;
    let model = write_fake_qwen2(&dir)?;
    let (pos, neg) = write_prompt_files(&dir)?;
    println!("{}\n{}\n{}", model.display(), pos.display(), neg.display());
    Ok(())
}
