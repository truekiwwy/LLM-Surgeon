//! What extraction, generation, steering, `bench dump` and `autotune` need from a forward
//! pass, behind one trait, and the factory that picks the implementation from the model
//! type.
//!
//! ```text
//! build(cfg, tensors, Depth) -> Box<dyn ResidualForward>
//!   qwen2, llama, mistral, qwen3 -> HookedDecoder (decoder.rs, Variant from the config)
//!   anything else                -> unsupported_architecture (parse_config refuses it first)
//! ```
//!
//! A new architecture (Gemma 2's scaled embeddings, `(1 + w)` norms, post-sublayer norms
//! and logit soft-capping, for instance) is a new implementation of the trait, checked
//! against its candle model before it is added here.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use super::config::{DecoderConfig, SUPPORTED_MODEL_TYPES};
use super::decoder::{HookedDecoder, KvCache, Site};
use super::error::{ExtractError, Result};

/// A forward pass that exposes the residual stream.
pub trait ResidualForward: Send {
    /// Layers loaded.
    fn num_layers(&self) -> usize;
    fn dtype(&self) -> DType;
    /// `[1, seq, hidden]` at the last loaded layer: the residual stream after it, or its
    /// MLP output. Nothing after it is computed.
    fn forward_capture(&self, input_ids: &Tensor, site: Site) -> Result<Tensor>;
    /// `[seq, vocab]` f32 logits (teacher forcing). Needs [`Depth::Full`].
    fn forward_logits(&self, input_ids: &Tensor) -> Result<Tensor>;
    /// Layers `0..n_layers` on new tokens, continuing `cache`; `after_layer(i, resid)` may
    /// replace the residual stream after layer `i` (activation injection, capture,
    /// cancellation).
    fn forward_cached(
        &self,
        input_ids: &Tensor,
        cache: &mut KvCache,
        n_layers: usize,
        after_layer: &mut dyn FnMut(usize, Tensor) -> Result<Tensor>,
    ) -> Result<Tensor>;
    /// f32 logits of the last position of `hidden`. Needs [`Depth::Full`].
    fn logits_last(&self, hidden: &Tensor) -> Result<Vec<f32>>;
}

impl ResidualForward for HookedDecoder {
    fn num_layers(&self) -> usize {
        HookedDecoder::num_layers(self)
    }

    fn dtype(&self) -> DType {
        HookedDecoder::dtype(self)
    }

    fn forward_capture(&self, input_ids: &Tensor, site: Site) -> Result<Tensor> {
        HookedDecoder::forward_capture(self, input_ids, site)
    }

    fn forward_logits(&self, input_ids: &Tensor) -> Result<Tensor> {
        HookedDecoder::forward_logits(self, input_ids)
    }

    fn forward_cached(
        &self,
        input_ids: &Tensor,
        cache: &mut KvCache,
        n_layers: usize,
        after_layer: &mut dyn FnMut(usize, Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        HookedDecoder::forward_cached(self, input_ids, cache, n_layers, after_layer)
    }

    fn logits_last(&self, hidden: &Tensor) -> Result<Vec<f32>> {
        HookedDecoder::logits_last(self, hidden)
    }
}

/// How much of the model to build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Depth {
    /// Embeddings and layers `0..=layer` (extraction, autotune).
    UpTo(usize),
    /// Every layer, the final norm and the output head (logits, generation).
    Full,
}

/// The forward of `cfg`'s architecture, from tensors named as in the checkpoint.
pub fn build(
    cfg: &DecoderConfig,
    tensors: HashMap<String, Tensor>,
    depth: Depth,
    dtype: DType,
    device: &Device,
) -> Result<Box<dyn ResidualForward>> {
    if !SUPPORTED_MODEL_TYPES.contains(&cfg.model_type.as_str()) {
        return Err(ExtractError::UnsupportedArchitecture {
            model_type: cfg.model_type.clone(),
        });
    }
    Ok(Box::new(match depth {
        Depth::UpTo(layer) => HookedDecoder::from_tensors(tensors, cfg, layer, dtype, device)?,
        Depth::Full => HookedDecoder::from_tensors_full(tensors, cfg, dtype, device)?,
    }))
}
