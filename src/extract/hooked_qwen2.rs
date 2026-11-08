//! Qwen2 forward pass that stops at layer L and exposes its activation.
//!
//! Adapted from candle-transformers 0.11.1, `src/models/qwen2.rs`
//! (Copyright (c) the candle authors; dual-licensed under MIT OR Apache-2.0,
//! https://github.com/huggingface/candle). Same math, same candle ops, in the same order:
//! RMSNorm (`candle_nn::RmsNorm`), rotate-half RoPE (`candle_nn::rotary_emb::rope`, sin/cos
//! computed in the model dtype), GQA through `repeat_kv` (concatenation), biases on q/k/v
//! only, causal mask with sliding window (built in f32, then cast), softmax on the last
//! dimension, SwiGLU (`act(gate) * up -> down`). Changes: no KV cache (one pass, batch of
//! one, offset 0), sin/cos computed for the prompt length instead of every position (each
//! entry is a single product, so the values are identical), and an early exit at layer L
//! with a choice of site.

use std::collections::HashMap;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Activation, Embedding, Linear, RmsNorm, VarBuilder};
use serde::{Deserialize, Serialize};

use super::config::Qwen2Config;
use super::error::{ExtractError, Result};
use super::weights::{load_tensors, required_tensors, TensorSource};

/// Where the activation of layer L is read.
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum Site {
    /// Residual stream at the output of layer L, before any later norm.
    #[default]
    #[value(name = "resid_post")]
    ResidPost,
    /// Output of the MLP of layer L (before the residual addition).
    #[value(name = "mlp_out")]
    MlpOut,
}

impl Site {
    pub fn as_str(self) -> &'static str {
        match self {
            Site::ResidPost => "resid_post",
            Site::MlpOut => "mlp_out",
        }
    }
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &Qwen2Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let (h, i) = (cfg.hidden_size, cfg.intermediate_size);
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?,
            up_proj: candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?,
            down_proj: candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
}

/// `candle_transformers::utils::repeat_kv`.
fn repeat_kv(xs: Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

impl Attention {
    fn new(cfg: &Qwen2Config, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let (num_heads, num_kv_heads, head_dim) = (
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim(),
        );
        Ok(Self {
            q_proj: candle_nn::linear(h, num_heads * head_dim, vb.pp("q_proj"))?,
            k_proj: candle_nn::linear(h, num_kv_heads * head_dim, vb.pp("k_proj"))?,
            v_proj: candle_nn::linear(h, num_kv_heads * head_dim, vb.pp("v_proj"))?,
            o_proj: candle_nn::linear_no_bias(num_heads * head_dim, h, vb.pp("o_proj"))?,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: h,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        rope: &(Tensor, Tensor),
        mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let q = self
            .q_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v_proj
            .forward(xs)?
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let (cos, sin) = rope;
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let weights = match mask {
            None => weights,
            Some(mask) => weights.broadcast_add(mask)?,
        };
        let weights = candle_nn::ops::softmax_last_dim(&weights)?;
        weights
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(cfg: &Qwen2Config, vb: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            self_attn: Attention::new(cfg, vb.pp("self_attn"))?,
            mlp: Mlp::new(cfg, vb.pp("mlp"))?,
            input_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: candle_nn::rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    /// (residual stream after the layer, MLP output of the layer).
    fn forward(
        &self,
        xs: &Tensor,
        rope: &(Tensor, Tensor),
        mask: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, rope, mask)?;
        let xs = (xs + residual)?;
        let mlp_out = xs.apply(&self.post_attention_layernorm)?.apply(&self.mlp)?;
        Ok(((&xs + &mlp_out)?, mlp_out))
    }
}

/// Final norm and output projection, only loaded by [`HookedQwen2::from_tensors_full`].
struct LmHead {
    norm: RmsNorm,
    lm_head: Linear,
}

pub struct HookedQwen2 {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    head: Option<LmHead>,
    head_dim: usize,
    rope_theta: f64,
    sliding_window: usize,
    dtype: DType,
    device: Device,
}

impl HookedQwen2 {
    /// Reads the embeddings and layers `0..=up_to_layer` only.
    pub fn load(
        src: &dyn TensorSource,
        cfg: &Qwen2Config,
        up_to_layer: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let required = required_tensors(cfg, up_to_layer);
        let tensors = load_tensors(src, &required, dtype, device, &mut |_, _| {
            std::ops::ControlFlow::Continue(())
        })?;
        Self::from_tensors(tensors, cfg, up_to_layer, dtype, device)
    }

    /// Builds the model from tensors named as in the checkpoint (`model.layers.0...`).
    pub fn from_tensors(
        tensors: HashMap<String, Tensor>,
        cfg: &Qwen2Config,
        up_to_layer: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::build(tensors, cfg, up_to_layer, dtype, device, false)
    }

    /// Every layer, the final norm and `lm_head` (the embeddings when
    /// `tie_word_embeddings`), for [`Self::forward_logits`]. Tensors as listed by
    /// `weights::required_tensors_full`.
    pub fn from_tensors_full(
        tensors: HashMap<String, Tensor>,
        cfg: &Qwen2Config,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        Self::build(tensors, cfg, cfg.num_hidden_layers - 1, dtype, device, true)
    }

    fn build(
        tensors: HashMap<String, Tensor>,
        cfg: &Qwen2Config,
        up_to_layer: usize,
        dtype: DType,
        device: &Device,
        with_head: bool,
    ) -> Result<Self> {
        if up_to_layer >= cfg.num_hidden_layers {
            return Err(ExtractError::InvalidArgument(format!(
                "couche {up_to_layer} hors bornes : num_hidden_layers = {}",
                cfg.num_hidden_layers
            )));
        }
        let root = VarBuilder::from_tensors(tensors, dtype, device);
        let vb = root.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?;
        let vb_l = vb.pp("layers");
        let layers = (0..=up_to_layer)
            .map(|i| DecoderLayer::new(cfg, vb_l.pp(i)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        let head = match with_head {
            false => None,
            true => Some(LmHead {
                norm: candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
                lm_head: match cfg.tie_word_embeddings {
                    true => Linear::new(embed_tokens.embeddings().clone(), None),
                    false => candle_nn::linear_no_bias(
                        cfg.hidden_size,
                        cfg.vocab_size,
                        root.pp("lm_head"),
                    )?,
                },
            }),
        };
        Ok(Self {
            embed_tokens,
            layers,
            head,
            head_dim: cfg.head_dim(),
            rope_theta: cfg.rope_theta,
            sliding_window: cfg.sliding_window(),
            dtype,
            device: device.clone(),
        })
    }

    /// `RotaryEmbedding::new` of the reference, for the first `seq_len` positions.
    fn rope(&self, seq_len: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let dim = self.head_dim;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / self.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq =
            Tensor::from_vec(inv_freq, (1, inv_freq_len), &self.device)?.to_dtype(self.dtype)?;
        let t = Tensor::arange(0u32, seq_len as u32, &self.device)?
            .to_dtype(self.dtype)?
            .reshape((seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok((freqs.cos()?, freqs.sin()?))
    }

    /// `prepare_causal_attention_mask` of the reference with offset 0 and batch 1.
    fn causal_mask(&self, tgt_len: usize) -> candle_core::Result<Tensor> {
        let window = self.sliding_window;
        let mask: Vec<f32> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?
            .expand((1, 1, tgt_len, tgt_len))?
            .to_dtype(self.dtype)
    }

    fn run(
        &self,
        input_ids: &Tensor,
        mut visit: impl FnMut(usize, &Tensor, &Tensor),
    ) -> Result<Tensor> {
        let (_b, seq_len) = input_ids.dims2()?;
        let mask = match seq_len {
            0 => return Err(ExtractError::InvalidArgument("séquence vide".into())),
            1 => None,
            n => Some(self.causal_mask(n)?),
        };
        let rope = self.rope(seq_len)?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let (resid, mlp_out) = layer.forward(&xs, &rope, mask.as_ref())?;
            visit(i, &resid, &mlp_out);
            xs = resid;
        }
        Ok(xs)
    }

    /// `[1, seq, hidden]` at layer `up_to_layer`; nothing after it is computed.
    pub fn forward_capture(&self, input_ids: &Tensor, site: Site) -> Result<Tensor> {
        let last = self.layers.len() - 1;
        let mut mlp_last = None;
        let resid = self.run(input_ids, |i, _, mlp| {
            if i == last {
                mlp_last = Some(mlp.clone());
            }
        })?;
        match site {
            Site::ResidPost => Ok(resid),
            Site::MlpOut => {
                mlp_last.ok_or_else(|| ExtractError::Compute("couche cible absente".into()))
            }
        }
    }

    /// `[seq, vocab]` logits in f32 for a batch of one (teacher forcing on the prompt):
    /// every layer, final norm, `lm_head`. Needs [`Self::from_tensors_full`].
    pub fn forward_logits(&self, input_ids: &Tensor) -> Result<Tensor> {
        let head = self.head.as_ref().ok_or_else(|| {
            ExtractError::Compute("modèle chargé sans norme finale ni lm_head".into())
        })?;
        let hidden = self.run(input_ids, |_, _, _| {})?;
        let logits = hidden.apply(&head.norm)?.apply(&head.lm_head)?;
        Ok(logits.squeeze(0)?.to_dtype(DType::F32)?)
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    #[cfg(test)]
    fn residual_per_layer(&self, input_ids: &Tensor) -> Result<Vec<Tensor>> {
        let mut out = Vec::new();
        self.run(input_ids, |_, resid, _| out.push(resid.clone()))?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::qwen2 as reference;

    /// Deterministic LCG in [-scale, scale].
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self, scale: f32) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        }
    }

    fn cfg(num_kv_heads: usize) -> Qwen2Config {
        super::super::config::parse_config(&format!(
            r#"{{"model_type":"qwen2","vocab_size":64,"hidden_size":16,"intermediate_size":32,
            "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":{num_kv_heads},
            "max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000.0,
            "tie_word_embeddings":false,"sliding_window":128,"hidden_act":"silu"}}"#
        ))
        .unwrap()
    }

    fn reference_cfg(c: &Qwen2Config) -> reference::Config {
        reference::Config {
            vocab_size: c.vocab_size,
            hidden_size: c.hidden_size,
            intermediate_size: c.intermediate_size,
            num_hidden_layers: c.num_hidden_layers,
            num_attention_heads: c.num_attention_heads,
            num_key_value_heads: c.num_key_value_heads,
            max_position_embeddings: c.max_position_embeddings,
            sliding_window: c.sliding_window(),
            max_window_layers: c.num_hidden_layers,
            tie_word_embeddings: false,
            rope_theta: c.rope_theta,
            rms_norm_eps: c.rms_norm_eps,
            use_sliding_window: false,
            hidden_act: c.hidden_act,
        }
    }

    /// Every tensor of the full model, final norm and lm_head included.
    fn weights(c: &Qwen2Config, seed: u64) -> HashMap<String, Tensor> {
        let mut rng = Lcg(seed);
        let mut all = required_tensors(c, c.num_hidden_layers - 1);
        all.push(("model.norm.weight".into(), vec![c.hidden_size]));
        all.push(("lm_head.weight".into(), vec![c.vocab_size, c.hidden_size]));
        all.into_iter()
            .map(|(name, shape)| {
                let n: usize = shape.iter().product();
                let values: Vec<f32> = (0..n)
                    .map(|_| {
                        if name.contains("norm") {
                            1.0 + rng.next(0.2)
                        } else {
                            rng.next(0.3)
                        }
                    })
                    .collect();
                (name, Tensor::from_vec(values, shape, &Device::Cpu).unwrap())
            })
            .collect()
    }

    fn ids(values: &[u32]) -> Tensor {
        Tensor::new(values, &Device::Cpu)
            .unwrap()
            .unsqueeze(0)
            .unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
    }

    #[test]
    fn full_forward_matches_candle_reference_logits() {
        for kv in [2, 1] {
            let c = cfg(kv);
            let w = weights(&c, 7 + kv as u64);
            let input = ids(&[3, 17, 42, 5, 9, 60, 1]);
            let vb = VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
            let mut model = reference::ModelForCausalLM::new(&reference_cfg(&c), vb).unwrap();
            let expected = model
                .forward(&input, 0)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap();

            let hooked = HookedQwen2::from_tensors(
                w.clone(),
                &c,
                c.num_hidden_layers - 1,
                DType::F32,
                &Device::Cpu,
            )
            .unwrap();
            let hidden = hooked.forward_capture(&input, Site::ResidPost).unwrap();
            let norm = candle_nn::RmsNorm::new(w["model.norm.weight"].clone(), c.rms_norm_eps);
            let last = norm
                .forward(&hidden)
                .unwrap()
                .narrow(1, 6, 1)
                .unwrap()
                .squeeze(0)
                .unwrap()
                .squeeze(0)
                .unwrap();
            let logits = w["lm_head.weight"]
                .matmul(&last.unsqueeze(1).unwrap())
                .unwrap()
                .squeeze(1)
                .unwrap();
            let diff = max_abs_diff(&logits, &expected);
            assert!(diff < 1e-4, "kv={kv}: max |Δlogit| = {diff}");
        }
    }

    #[test]
    fn early_exit_is_bit_identical_to_the_full_forward_layer() {
        let c = cfg(1);
        let w = weights(&c, 11);
        let input = ids(&[1, 2, 3, 4, 5]);
        let full = HookedQwen2::from_tensors(w.clone(), &c, 3, DType::F32, &Device::Cpu).unwrap();
        let per_layer = full.residual_per_layer(&input).unwrap();
        assert_eq!(per_layer.len(), c.num_hidden_layers);
        for (layer, recorded) in per_layer.iter().enumerate() {
            let early =
                HookedQwen2::from_tensors(w.clone(), &c, layer, DType::F32, &Device::Cpu).unwrap();
            let captured = early.forward_capture(&input, Site::ResidPost).unwrap();
            let a: Vec<u32> = captured
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            let b: Vec<u32> = recorded
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect();
            assert_eq!(a, b, "layer {layer}");
        }
    }

    #[test]
    fn forward_logits_matches_the_reference_at_every_position() {
        for (kv, tied) in [(2, false), (1, true)] {
            let mut c = cfg(kv);
            c.tie_word_embeddings = tied;
            let mut w = weights(&c, 21 + kv as u64);
            if tied {
                w.remove("lm_head.weight");
            }
            let input = ids(&[3, 17, 42, 5, 9, 60, 1]);
            let vb = VarBuilder::from_tensors(w.clone(), DType::F32, &Device::Cpu);
            let mut rc = reference_cfg(&c);
            rc.tie_word_embeddings = tied;
            let mut reference = reference::Model::new(&rc, vb).unwrap();
            let hidden = reference
                .forward(&input, 0, None)
                .unwrap()
                .squeeze(0)
                .unwrap();
            let head = match tied {
                true => &w["model.embed_tokens.weight"],
                false => &w["lm_head.weight"],
            };
            let expected = hidden.matmul(&head.t().unwrap()).unwrap();

            let hooked = HookedQwen2::from_tensors_full(w, &c, DType::F32, &Device::Cpu).unwrap();
            let logits = hooked.forward_logits(&input).unwrap();
            assert_eq!(logits.dims(), [7, 64]);
            let diff = max_abs_diff(&logits, &expected);
            assert!(diff < 1e-4, "kv={kv} tied={tied}: max |Δlogit| = {diff}");
        }
        let c = cfg(2);
        let partial = HookedQwen2::from_tensors(weights(&c, 3), &c, 1, DType::F32, &Device::Cpu);
        assert!(partial.unwrap().forward_logits(&ids(&[1, 2])).is_err());
    }

    #[test]
    fn mlp_out_site_and_bf16_run() {
        let c = cfg(2);
        let w = weights(&c, 5);
        let model = HookedQwen2::from_tensors(w.clone(), &c, 1, DType::F32, &Device::Cpu).unwrap();
        let input = ids(&[4, 8]);
        let resid = model.forward_capture(&input, Site::ResidPost).unwrap();
        let mlp = model.forward_capture(&input, Site::MlpOut).unwrap();
        assert_eq!(resid.dims(), [1, 2, 16]);
        assert_eq!(mlp.dims(), [1, 2, 16]);
        assert!(max_abs_diff(&resid, &mlp) > 0.0);

        let bf16 = HookedQwen2::from_tensors(w, &c, 1, DType::BF16, &Device::Cpu).unwrap();
        let out = bf16
            .forward_capture(&input, Site::ResidPost)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        assert_eq!(bf16.dtype(), DType::BF16);
        assert!(max_abs_diff(&out, &resid) < 0.25, "bf16 stays close to f32");
        assert!(
            HookedQwen2::from_tensors(weights(&c, 1), &c, 4, DType::F32, &Device::Cpu).is_err(),
            "layer out of range"
        );
    }
}
