//! Minimal weight loading: the embeddings and layers `0..=L` only, read through the
//! engine's memory maps (one shard per file, name -> shard map from their indexes).

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::ControlFlow;
use std::path::Path;

use candle_core::{DType, Device, Tensor};

use super::config::Qwen2Config;
use super::error::{ExtractError, Result};
use crate::engine::{self, ReadOnlyModel};

/// A tensor as stored: dtype, shape and little-endian bytes (borrowed from a mmap).
pub struct RawTensor<'a> {
    pub dtype: engine::DType,
    pub shape: Vec<usize>,
    pub bytes: Cow<'a, [u8]>,
}

/// Where weights come from. `meta` only reads index metadata; `read` hands out the bytes.
/// Mockable, so a test can prove that no tensor past layer L is ever read.
pub trait TensorSource {
    fn meta(&self, name: &str) -> Option<(engine::DType, Vec<usize>)>;
    fn read(&self, name: &str) -> Result<RawTensor<'_>>;
}

/// Every `.safetensors` shard of a model directory (index or listing, as `hardware` sees
/// it), memory-mapped read-only by the engine: opening reads the headers only.
pub struct SafetensorsDir {
    shards: Vec<ReadOnlyModel>,
    owner: HashMap<String, usize>,
}

impl SafetensorsDir {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let files = crate::hardware::compute_model_size(model_dir)
            .map_err(|e| ExtractError::Weights(e.to_string()))?
            .shard_files;
        let mut shards = Vec::with_capacity(files.len());
        let mut owner = HashMap::new();
        for (i, file) in files.iter().enumerate() {
            let shard =
                ReadOnlyModel::open(file).map_err(|e| ExtractError::Weights(e.to_string()))?;
            for info in &shard.index().tensors {
                owner.entry(info.name.clone()).or_insert(i);
            }
            shards.push(shard);
        }
        Ok(Self { shards, owner })
    }
}

impl TensorSource for SafetensorsDir {
    fn meta(&self, name: &str) -> Option<(engine::DType, Vec<usize>)> {
        let info = self
            .shards
            .get(*self.owner.get(name)?)?
            .index()
            .get(name)
            .ok()?;
        Some((info.dtype.clone(), info.shape.clone()))
    }

    fn read(&self, name: &str) -> Result<RawTensor<'_>> {
        let missing = || ExtractError::MissingTensor(name.to_owned());
        let shard = self
            .shards
            .get(*self.owner.get(name).ok_or_else(missing)?)
            .ok_or_else(missing)?;
        let view = shard.tensor(name).map_err(|_| missing())?;
        Ok(RawTensor {
            dtype: view.info.dtype.clone(),
            shape: view.info.shape.clone(),
            bytes: Cow::Borrowed(view.bytes),
        })
    }
}

/// Names and expected shapes needed to run layers `0..=layer`: no later layer, no final
/// norm, no `lm_head`. Biases and q/k norms as the model type has them (`Variant`): q/k/v
/// biases and no o_proj bias in Qwen2, the order of the list unchanged for it.
pub fn required_tensors(cfg: &Qwen2Config, layer: usize) -> Vec<(String, Vec<usize>)> {
    let (h, i, d) = (cfg.hidden_size, cfg.intermediate_size, cfg.head_dim());
    let q = cfg.num_attention_heads * d;
    let kv = cfg.num_key_value_heads * d;
    let v = cfg.variant();
    let mut names = vec![(
        "model.embed_tokens.weight".to_owned(),
        vec![cfg.vocab_size, h],
    )];
    for l in 0..=layer {
        let p = format!("model.layers.{l}");
        for (proj, out) in [("q_proj", q), ("k_proj", kv), ("v_proj", kv)] {
            names.push((format!("{p}.self_attn.{proj}.weight"), vec![out, h]));
            if v.qkv_bias {
                names.push((format!("{p}.self_attn.{proj}.bias"), vec![out]));
            }
        }
        if v.qk_norm {
            names.push((format!("{p}.self_attn.q_norm.weight"), vec![d]));
            names.push((format!("{p}.self_attn.k_norm.weight"), vec![d]));
        }
        names.push((format!("{p}.self_attn.o_proj.weight"), vec![h, q]));
        if v.o_bias {
            names.push((format!("{p}.self_attn.o_proj.bias"), vec![h]));
        }
        for (proj, shape) in [
            ("gate_proj", [i, h]),
            ("up_proj", [i, h]),
            ("down_proj", [h, i]),
        ] {
            names.push((format!("{p}.mlp.{proj}.weight"), shape.to_vec()));
            if v.mlp_bias {
                names.push((format!("{p}.mlp.{proj}.bias"), vec![shape[0]]));
            }
        }
        names.extend([
            (format!("{p}.input_layernorm.weight"), vec![h]),
            (format!("{p}.post_attention_layernorm.weight"), vec![h]),
        ]);
    }
    names
}

/// Every tensor of a full forward pass: all layers, `model.norm` and `lm_head` (absent
/// when the embeddings are tied, as in Qwen2.5-0.5B).
pub fn required_tensors_full(cfg: &Qwen2Config) -> Vec<(String, Vec<usize>)> {
    let mut names = required_tensors(cfg, cfg.num_hidden_layers - 1);
    names.push(("model.norm.weight".into(), vec![cfg.hidden_size]));
    if !cfg.tie_word_embeddings {
        names.push((
            "lm_head.weight".into(),
            vec![cfg.vocab_size, cfg.hidden_size],
        ));
    }
    names
}

fn candle_dtype(name: &str, dtype: &engine::DType) -> Result<DType> {
    match dtype {
        engine::DType::BF16 => Ok(DType::BF16),
        engine::DType::F16 => Ok(DType::F16),
        engine::DType::F32 => Ok(DType::F32),
        engine::DType::Other(other) => Err(ExtractError::UnsupportedDtype {
            name: name.to_owned(),
            dtype: other.clone(),
        }),
    }
}

/// Checks presence, shape and dtype of every required tensor from the index alone.
/// Returns (total elements, elements and source bytes of the largest tensor).
pub fn check_tensors(
    src: &dyn TensorSource,
    required: &[(String, Vec<usize>)],
) -> Result<TensorTotals> {
    let mut totals = TensorTotals::default();
    for (name, expected) in required {
        let (dtype, shape) = src
            .meta(name)
            .ok_or_else(|| ExtractError::MissingTensor(name.clone()))?;
        candle_dtype(name, &dtype)?;
        if &shape != expected {
            return Err(ExtractError::TensorShape {
                name: name.clone(),
                expected: expected.clone(),
                found: shape,
            });
        }
        let elements: u64 = shape.iter().map(|&d| d as u64).product();
        let source_bytes = elements * dtype.size_of().unwrap_or(4) as u64;
        totals.elements += elements;
        totals.largest_source_bytes = totals.largest_source_bytes.max(source_bytes);
    }
    Ok(totals)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TensorTotals {
    pub elements: u64,
    pub largest_source_bytes: u64,
}

/// Reads and converts every required tensor to `dtype`. `on_tensor(done, total)` returning
/// `Break` cancels the load.
pub fn load_tensors(
    src: &dyn TensorSource,
    required: &[(String, Vec<usize>)],
    dtype: DType,
    device: &Device,
    on_tensor: &mut dyn FnMut(usize, usize) -> ControlFlow<()>,
) -> Result<HashMap<String, Tensor>> {
    let mut tensors = HashMap::with_capacity(required.len());
    for (done, (name, expected)) in required.iter().enumerate() {
        let raw = src.read(name)?;
        if &raw.shape != expected {
            return Err(ExtractError::TensorShape {
                name: name.clone(),
                expected: expected.clone(),
                found: raw.shape,
            });
        }
        // from_raw_buffer copies into candle's storage and handles unaligned mmap bytes.
        let tensor = Tensor::from_raw_buffer(
            &raw.bytes,
            candle_dtype(name, &raw.dtype)?,
            &raw.shape,
            device,
        )?
        .to_dtype(dtype)?;
        tensors.insert(name.clone(), tensor);
        if on_tensor(done + 1, required.len()).is_break() {
            return Err(ExtractError::Cancelled);
        }
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    struct Fake {
        tensors: HashMap<String, (engine::DType, Vec<usize>)>,
        reads: RefCell<Vec<String>>,
    }

    impl TensorSource for Fake {
        fn meta(&self, name: &str) -> Option<(engine::DType, Vec<usize>)> {
            self.tensors.get(name).cloned()
        }
        fn read(&self, name: &str) -> Result<RawTensor<'_>> {
            self.reads.borrow_mut().push(name.to_owned());
            let (dtype, shape) = self
                .meta(name)
                .ok_or_else(|| ExtractError::MissingTensor(name.into()))?;
            let n: usize = shape.iter().product();
            Ok(RawTensor {
                dtype,
                shape,
                bytes: Cow::Owned(vec![0u8; n * 4]),
            })
        }
    }

    fn cfg() -> Qwen2Config {
        super::super::config::parse_config(
            r#"{"model_type":"qwen2","vocab_size":64,"hidden_size":16,"intermediate_size":32,
            "num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,
            "max_position_embeddings":128,"rms_norm_eps":1e-6,"rope_theta":10000.0}"#,
        )
        .unwrap()
    }

    fn fake(cfg: &Qwen2Config) -> Fake {
        let tensors = required_tensors(cfg, cfg.num_hidden_layers - 1)
            .into_iter()
            .map(|(n, s)| (n, (engine::DType::F32, s)))
            .collect();
        Fake {
            tensors,
            reads: RefCell::new(Vec::new()),
        }
    }

    #[test]
    fn minimal_set_stops_at_layer_l() {
        let cfg = cfg();
        let req = required_tensors(&cfg, 1);
        assert_eq!(req.len(), 1 + 2 * 12);
        assert!(req
            .iter()
            .all(|(n, _)| !n.contains("layers.2") && !n.contains("layers.3")));
        assert!(req
            .iter()
            .all(|(n, _)| n != "lm_head.weight" && n != "model.norm.weight"));
        let kv = req
            .iter()
            .find(|(n, _)| n.ends_with("0.self_attn.k_proj.weight"))
            .unwrap();
        assert_eq!(kv.1, [8, 16], "GQA: one KV head of dim 8");
    }

    #[test]
    fn missing_shape_and_dtype_errors_name_the_tensor() {
        let cfg = cfg();
        let req = required_tensors(&cfg, 0);
        let mut src = fake(&cfg);
        assert_eq!(
            check_tensors(&src, &req).unwrap().elements,
            req.iter()
                .map(|(_, s)| s.iter().product::<usize>() as u64)
                .sum::<u64>()
        );
        src.tensors.remove("model.layers.0.mlp.up_proj.weight");
        let err = check_tensors(&src, &req).unwrap_err();
        assert!(
            matches!(&err, ExtractError::MissingTensor(n) if n == "model.layers.0.mlp.up_proj.weight")
        );

        let mut src = fake(&cfg);
        src.tensors.insert(
            "model.layers.0.mlp.down_proj.weight".into(),
            (engine::DType::F32, vec![32, 16]),
        );
        let err = check_tensors(&src, &req).unwrap_err();
        assert!(
            err.to_string().contains("down_proj") && err.to_string().contains("[32, 16]"),
            "{err}"
        );

        let mut src = fake(&cfg);
        src.tensors.insert(
            "model.embed_tokens.weight".into(),
            (engine::DType::Other("I8".into()), vec![64, 16]),
        );
        assert!(matches!(
            check_tensors(&src, &req),
            Err(ExtractError::UnsupportedDtype { .. })
        ));
    }

    #[test]
    fn loading_reads_only_required_names_and_can_be_cancelled() {
        let cfg = cfg();
        let req = required_tensors(&cfg, 1);
        let src = fake(&cfg);
        let loaded = load_tensors(&src, &req, DType::F32, &Device::Cpu, &mut |_, _| {
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(loaded.len(), req.len());
        assert!(src
            .reads
            .borrow()
            .iter()
            .all(|n| !n.contains("layers.2") && !n.contains("layers.3")));

        let src = fake(&cfg);
        let err = load_tensors(&src, &req, DType::F32, &Device::Cpu, &mut |done, _| {
            if done == 3 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap_err();
        assert!(matches!(err, ExtractError::Cancelled));
        assert_eq!(src.reads.borrow().len(), 3);
    }
}
