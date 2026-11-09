//! Writes a tiny Qwen2-like base/tuned checkpoint pair to try `surgeon diff` and
//! `surgeon apply --patch x.surgery` without a real model.
//!
//! ```text
//! cargo run --example make_fake_checkpoints -- --out <dir> [--kind rank|full] [--rank 4]
//!     [--dtype bf16|f16|f32] [--seed 1] [--config] [--lora]
//! ```
//!
//! - `<dir>/base/model.safetensors` and `<dir>/tuned/model.safetensors`: 2 layers, hidden 32,
//!   intermediate 64, vocab 48. Per layer: `q_proj`, `up_proj`, `down_proj` weights (2D) and the
//!   two layer norms (1D); plus `embed_tokens`, `lm_head` (2D, excluded by `diff` by default)
//!   and `model.norm.weight` (identical in both files, so `diff` skips it).
//! - `--kind rank` (default): `tuned = base + B*A` with `B` [m x r], `A` [r x n], so the 2D
//!   difference has EXACT rank `--rank` (default 4, at most 32). Values are multiples of small
//!   powers of two, so every sum is exact in the stored dtype (no rounding noise breaks the
//!   rank); the layer norms get a small shift.
//!   `--kind full`: `tuned = round(base + noise)`, full-rank differences.
//! - `--dtype`: storage dtype of the weights, `bf16` by default; `round` is round-to-nearest-even
//!   (only `--kind full` rounds: the exact grid of `--kind rank` fits every dtype).
//! - `--config`: also writes a `config.json` next to each checkpoint (metadata only: the pair has
//!   no k/v/o/gate projections, so it is not loadable for inference).
//! - `--lora` (needs `--kind rank`): `<dir>/lora/adapter_model.safetensors` (f32, PEFT keys
//!   `base_model.model.<path>.lora_A.weight` / `lora_B.weight`) and `adapter_config.json` with
//!   `lora_alpha = r`, for the `q_proj`, `up_proj`, `down_proj` weights: merging it into `base`
//!   reproduces `tuned` on those tensors (scale `alpha / r = 1`).
//!
//! Everything comes from a fixed LCG seeded by `--seed`, so a run is reproducible.

use std::collections::BTreeMap;
use std::error::Error;
use std::path::Path;

use half::{bf16, f16};
use serde_json::json;

const HIDDEN: usize = 32;
const INTER: usize = 64;
const VOCAB: usize = 48;
const LAYERS: usize = 2;

type Res<T> = Result<T, Box<dyn Error>>;

#[derive(Clone, Copy, PartialEq)]
enum Dtype {
    Bf16,
    F16,
    F32,
}

impl Dtype {
    fn parse(s: &str) -> Res<Self> {
        match s {
            "bf16" => Ok(Self::Bf16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            other => Err(format!("unknown --dtype {other} (bf16, f16 or f32)").into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F16 => "F16",
            Self::F32 => "F32",
        }
    }

    /// Round-to-nearest-even into the dtype, back to f64.
    fn round(self, x: f64) -> f64 {
        match self {
            Self::Bf16 => bf16::from_f64(x).to_f64(),
            Self::F16 => f16::from_f64(x).to_f64(),
            Self::F32 => f64::from(x as f32),
        }
    }

    fn encode(self, vals: &[f64]) -> Vec<u8> {
        vals.iter()
            .flat_map(|&x| match self {
                Self::Bf16 => bf16::from_f64(x).to_le_bytes().to_vec(),
                Self::F16 => f16::from_f64(x).to_le_bytes().to_vec(),
                Self::F32 => (x as f32).to_le_bytes().to_vec(),
            })
            .collect()
    }
}

/// Deterministic uniform values in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    fn vec(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.next() * scale).collect()
    }

    /// Multiples of `step` in `[-max_k * step, max_k * step]`: sums and products of a few of
    /// them stay exactly representable in bf16, f16 and f32.
    fn grid(&mut self, n: usize, step: f64, max_k: i64) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let k = ((self.next() + 1.0) / 2.0 * (2 * max_k + 1) as f64) as i64 - max_k;
                k.min(max_k) as f64 * step
            })
            .collect()
    }
}

struct Spec {
    name: String,
    shape: Vec<usize>,
    /// Receives a LoRA adapter with `--lora`.
    lora: bool,
}

fn specs() -> Vec<Spec> {
    let mut out = vec![Spec {
        name: "model.embed_tokens.weight".into(),
        shape: vec![VOCAB, HIDDEN],
        lora: false,
    }];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        for (suffix, shape, lora) in [
            ("self_attn.q_proj.weight", vec![HIDDEN, HIDDEN], true),
            ("mlp.up_proj.weight", vec![INTER, HIDDEN], true),
            ("mlp.down_proj.weight", vec![HIDDEN, INTER], true),
            ("input_layernorm.weight", vec![HIDDEN], false),
            ("post_attention_layernorm.weight", vec![HIDDEN], false),
        ] {
            out.push(Spec {
                name: format!("{p}.{suffix}"),
                shape,
                lora,
            });
        }
    }
    out.push(Spec {
        name: "model.norm.weight".into(),
        shape: vec![HIDDEN],
        lora: false,
    });
    out.push(Spec {
        name: "lm_head.weight".into(),
        shape: vec![VOCAB, HIDDEN],
        lora: false,
    });
    out
}

/// `B*A` with `B` [m x r], `A` [r x n], all row-major.
fn product(b: &[f64], a: &[f64], m: usize, r: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|idx| {
            (0..r)
                .map(|k| b[(idx / n) * r + k] * a[k * n + idx % n])
                .sum()
        })
        .collect()
}

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, &str, Vec<u8>)]) -> Res<()> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, dtype, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            name.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = serde_json::to_string(&header)?;
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(path, file)?;
    Ok(())
}

struct Options {
    out: String,
    full: bool,
    rank: usize,
    dtype: Dtype,
    seed: u64,
    config: bool,
    lora: bool,
}

fn parse_args() -> Res<Options> {
    let mut opts = Options {
        out: String::new(),
        full: false,
        rank: 4,
        dtype: Dtype::Bf16,
        seed: 1,
        config: false,
        lora: false,
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || args.next().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--out" => opts.out = value()?,
            "--kind" => {
                opts.full = match value()?.as_str() {
                    "rank" => false,
                    "full" => true,
                    other => return Err(format!("unknown --kind {other} (rank or full)").into()),
                }
            }
            "--rank" => opts.rank = value()?.parse()?,
            "--dtype" => opts.dtype = Dtype::parse(&value()?)?,
            "--seed" => opts.seed = value()?.parse()?,
            "--config" => opts.config = true,
            "--lora" => opts.lora = true,
            other => return Err(format!("unknown argument {other}").into()),
        }
    }
    if opts.out.is_empty() {
        return Err("--out <dir> is required".into());
    }
    if opts.rank == 0 || opts.rank > HIDDEN {
        return Err(format!("--rank must be in 1..={HIDDEN}").into());
    }
    if opts.lora && opts.full {
        return Err("--lora needs --kind rank".into());
    }
    Ok(opts)
}

fn main() -> Res<()> {
    let o = parse_args()?;
    let (dtype, r) = (o.dtype, o.rank);
    let mut rng = Lcg(o.seed);
    let root = Path::new(&o.out);
    for sub in ["base", "tuned"] {
        std::fs::create_dir_all(root.join(sub))?;
    }

    let mut base = Vec::new();
    let mut tuned = Vec::new();
    // LoRA key -> (shape, f32 values).
    let mut adapter: BTreeMap<String, (Vec<usize>, Vec<f64>)> = BTreeMap::new();
    for spec in specs() {
        let n: usize = spec.shape.iter().product();
        let is_norm = spec.shape.len() == 1;
        let w: Vec<f64> = if is_norm {
            rng.grid(n, 1.0 / 64.0, 6).iter().map(|x| 1.0 + x).collect()
        } else {
            // Odd multiples of 1/128: base + B*A is never exactly 0 (a tiny reconstruction error
            // could then show up as a different value).
            rng.grid(n, 1.0 / 64.0, 32)
                .iter()
                .map(|x| x + 1.0 / 128.0)
                .collect()
        };
        let new: Vec<f64> = if spec.name == "model.norm.weight" {
            w.clone()
        } else if o.full {
            let noise = rng.vec(n, 0.02);
            w.iter()
                .zip(&noise)
                .map(|(a, b)| dtype.round(a + b))
                .collect()
        } else if is_norm {
            let shift = rng.grid(n, 1.0 / 64.0, 1);
            w.iter().zip(&shift).map(|(a, d)| a + d).collect()
        } else {
            // Steps of 1/8 up to 2/8: B*A is a multiple of 1/64 below r/16, so base + B*A is
            // exact in every dtype for r <= 32 and the difference is exactly rank r.
            let (m, cols) = (spec.shape[0], spec.shape[1]);
            let b = rng.grid(m * r, 0.125, 2);
            let a = rng.grid(r * cols, 0.125, 2);
            let delta = product(&b, &a, m, r, cols);
            if o.lora && spec.lora {
                let key = spec.name.trim_end_matches(".weight");
                adapter.insert(
                    format!("base_model.model.{key}.lora_A.weight"),
                    (vec![r, cols], a),
                );
                adapter.insert(
                    format!("base_model.model.{key}.lora_B.weight"),
                    (vec![m, r], b),
                );
            }
            w.iter().zip(&delta).map(|(a, d)| a + d).collect()
        };
        base.push((
            spec.name.clone(),
            spec.shape.clone(),
            dtype.name(),
            dtype.encode(&w),
        ));
        tuned.push((spec.name, spec.shape, dtype.name(), dtype.encode(&new)));
    }
    write_safetensors(&root.join("base/model.safetensors"), &base)?;
    write_safetensors(&root.join("tuned/model.safetensors"), &tuned)?;

    if o.config {
        let config = json!({
            "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
            "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": LAYERS,
            "num_attention_heads": 4, "num_key_value_heads": 4, "vocab_size": VOCAB,
            "max_position_embeddings": 128, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "tie_word_embeddings": false, "torch_dtype": "bfloat16",
        });
        for sub in ["base", "tuned"] {
            std::fs::write(
                root.join(sub).join("config.json"),
                serde_json::to_vec_pretty(&config)?,
            )?;
        }
    }
    if o.lora {
        let dir = root.join("lora");
        std::fs::create_dir_all(&dir)?;
        let tensors: Vec<_> = adapter
            .into_iter()
            .map(|(k, (shape, vals))| (k, shape, "F32", Dtype::F32.encode(&vals)))
            .collect();
        write_safetensors(&dir.join("adapter_model.safetensors"), &tensors)?;
        let cfg = json!({
            "peft_type": "LORA", "r": r, "lora_alpha": r, "bias": "none",
            "task_type": "CAUSAL_LM", "use_rslora": false, "use_dora": false,
            "fan_in_fan_out": false, "target_modules": ["q_proj", "up_proj", "down_proj"],
        });
        std::fs::write(
            dir.join("adapter_config.json"),
            serde_json::to_vec_pretty(&cfg)?,
        )?;
    }
    println!(
        "wrote {} tensors per checkpoint under {}",
        base.len(),
        root.display()
    );
    Ok(())
}
