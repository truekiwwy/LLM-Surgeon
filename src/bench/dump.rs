//! `surgeon bench dump`: logits of a Qwen2 model on prompts, in pure Rust (candle, CPU),
//! for `bench logits` before/after a surgery. Same loading path as `extract`
//! (`SafetensorsDir` maps, `check_tensors`, `choose_dtype`), but every layer, the final
//! norm and `lm_head` (or the tied embeddings).
//!
//! Output, written to a sibling `.tmp` then renamed (never a partial file, nothing left
//! on error or cancellation): `logits` `[n_positions, vocab]` (f32 or f16), `prompt_offsets`
//! I64 `[n_prompts + 1]`, and `__metadata__`. The header is computed up front, so rows
//! stream to disk prompt after prompt: only one prompt's logits are ever in memory.
//!
//! RAM check: weights of the whole model in the compute dtype + activations, plus a logits
//! margin of `max_tokens × vocab × 4 × 2` (the f32 logits of the longest prompt and their
//! encoded copy). The spec's `positions × vocab × 4` over all prompts would refuse dumps
//! that fit, since rows never accumulate in memory.

use std::collections::BTreeMap;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use candle_core::{Device, Tensor};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::error::{BenchError, Result};
use super::logits_io::{LogitsDtype, LOGITS_TENSOR, OFFSETS_TENSOR};
use super::report::{BenchDumpReport, BenchRun, DumpTiming};
use super::sha256_file;
use crate::extract::{
    check_tensors, check_texts, choose_dtype, dtype_name, load_config, load_tensors,
    load_tokenizer, model_name, plan_memory, required_tensors_full, tokenize, DTypeChoice,
    HookedQwen2, MemoryPlan, SafetensorsDir, TensorSource,
};
use crate::hardware::HardwareProbe;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema, clap::ValueEnum,
)]
#[serde(rename_all = "lowercase")]
pub enum OutDtype {
    #[default]
    F32,
    F16,
}

impl OutDtype {
    fn logits(self) -> LogitsDtype {
        match self {
            Self::F32 => LogitsDtype::F32,
            Self::F16 => LogitsDtype::F16,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DumpRequest {
    pub model: PathBuf,
    pub prompts: Vec<String>,
    pub out: PathBuf,
    pub dtype: DTypeChoice,
    pub out_dtype: OutDtype,
    pub max_tokens: usize,
    pub max_total_positions: usize,
    pub overwrite: bool,
}

/// Removes the temporary file unless committed.
struct TempOut {
    path: PathBuf,
    committed: bool,
}

impl Drop for TempOut {
    fn drop(&mut self) {
        if !self.committed {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

fn io_error(path: &Path, e: impl std::fmt::Display) -> BenchError {
    BenchError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    }
}

fn check_output(out: &Path, overwrite: bool) -> Result<()> {
    if !out
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("safetensors"))
    {
        return Err(BenchError::input(
            out,
            "l'extension .safetensors est requise",
        ));
    }
    let parent = match out.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    if !parent.is_dir() {
        return Err(BenchError::input(out, "dossier de sortie inexistant"));
    }
    if out.exists() && !overwrite {
        return Err(BenchError::OutputExists(out.to_path_buf()));
    }
    Ok(())
}

/// Hash of the token sequences: same prompts and tokenizer give the same hash.
fn prompts_hash(ids: &[Vec<u32>]) -> String {
    let mut hasher = Sha256::new();
    for prompt in ids {
        hasher.update((prompt.len() as u64).to_le_bytes());
        for id in prompt {
            hasher.update(id.to_le_bytes());
        }
    }
    hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// SafeTensors header (sorted keys, padded to 8 bytes) for the streamed layout.
fn header(
    n_positions: usize,
    vocab: usize,
    out_dtype: OutDtype,
    n_prompts: usize,
    metadata: &BTreeMap<String, String>,
) -> Vec<u8> {
    let logits_bytes = n_positions * vocab * out_dtype.logits().size();
    let offsets_bytes = 8 * (n_prompts + 1);
    let dtype = match out_dtype {
        OutDtype::F32 => "F32",
        OutDtype::F16 => "F16",
    };
    let mut text = serde_json::json!({
        "__metadata__": metadata,
        LOGITS_TENSOR: { "dtype": dtype, "shape": [n_positions, vocab], "data_offsets": [0, logits_bytes] },
        OFFSETS_TENSOR: { "dtype": "I64", "shape": [n_prompts + 1], "data_offsets": [logits_bytes, logits_bytes + offsets_bytes] },
    })
    .to_string();
    while !(8 + text.len()).is_multiple_of(8) {
        text.push(' ');
    }
    let mut bytes = (text.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(text.as_bytes());
    bytes
}

/// Opens the model directory and runs [`run_with`].
pub fn run(
    req: &DumpRequest,
    probe: &dyn HardwareProbe,
    cancel: &AtomicBool,
    on_prompt: &mut dyn FnMut(usize, usize),
) -> Result<BenchRun<BenchDumpReport>> {
    check_output(&req.out, req.overwrite)?;
    let source = SafetensorsDir::open(&req.model)?;
    run_with(req, &source, probe, cancel, on_prompt)
}

/// validate -> config -> tensor index -> RAM (code 1 before any weight is read) ->
/// tokenize -> position cap -> weights -> forward per prompt, streamed -> rename.
pub fn run_with(
    req: &DumpRequest,
    source: &dyn TensorSource,
    probe: &dyn HardwareProbe,
    cancel: &AtomicBool,
    on_prompt: &mut dyn FnMut(usize, usize),
) -> Result<BenchRun<BenchDumpReport>> {
    let started = Instant::now();
    check_texts(req.prompts.clone()).map_err(BenchError::InvalidArgument)?;
    if req.max_tokens == 0 || req.max_total_positions == 0 {
        return Err(BenchError::InvalidArgument(
            "--max-tokens et --max-total-positions doivent être >= 1".into(),
        ));
    }
    check_output(&req.out, req.overwrite)?;
    let cfg = load_config(&req.model)?;
    let required = required_tensors_full(&cfg);
    let totals = check_tensors(source, &required)?;
    let mut warnings = Vec::new();
    let plan = plan_memory(&cfg, totals, req.max_tokens);
    let margin = (req.max_tokens * cfg.vocab_size * 4 * 2) as u64;
    let plan = MemoryPlan {
        f32_bytes: plan.f32_bytes + margin,
        bf16_bytes: plan.bf16_bytes + margin,
    };
    let dtype = choose_dtype(req.dtype, plan, probe, &mut warnings)?;

    let tokenizer = load_tokenizer(&req.model)?;
    let mut ids = Vec::with_capacity(req.prompts.len());
    let mut truncated = 0;
    for (i, prompt) in req.prompts.iter().enumerate() {
        let t = tokenize(&tokenizer, prompt, req.max_tokens)?;
        if t.ids.is_empty() {
            return Err(BenchError::InvalidArgument(format!(
                "prompt {i} : aucun token"
            )));
        }
        truncated += usize::from(t.truncated);
        ids.push(t.ids);
    }
    let n_positions: usize = ids.iter().map(Vec::len).sum();
    let size = req.out_dtype.logits().size();
    if n_positions > req.max_total_positions {
        return Err(BenchError::InvalidArgument(format!(
            "{n_positions} positions > --max-total-positions {} : le dump ferait environ {:.2} GiB \
             ({n_positions} × {} × {size} octets) ; réduire les prompts, --max-tokens, ou relever le plafond",
            req.max_total_positions,
            (n_positions * cfg.vocab_size * size) as f64 / (1u64 << 30) as f64,
            cfg.vocab_size
        )));
    }
    if truncated > 0 {
        warnings.push(format!(
            "{truncated} prompt(s) tronqué(s) à {} tokens",
            req.max_tokens
        ));
    }
    let tokenizer_sha256 = sha256_file(&req.model.join("tokenizer.json"))?;
    let prompts_sha256 = prompts_hash(&ids);
    let dtype_used = dtype_name(dtype).to_owned();
    let mut offsets = vec![0i64];
    for prompt in &ids {
        offsets.push(offsets.last().copied().unwrap_or(0) + prompt.len() as i64);
    }
    let metadata: BTreeMap<String, String> = [
        ("model", model_name(&req.model)),
        ("dtype_used", dtype_used.clone()),
        ("n_prompts", ids.len().to_string()),
        ("n_positions", n_positions.to_string()),
        ("vocab_size", cfg.vocab_size.to_string()),
        ("prompts_sha256", prompts_sha256.clone()),
        ("tokenizer_sha256", tokenizer_sha256.clone()),
        ("max_tokens", req.max_tokens.to_string()),
        ("surgeon_version", env!("CARGO_PKG_VERSION").into()),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_owned(), v))
    .collect();

    let device = Device::Cpu;
    let tensors = load_tensors(source, &required, dtype, &device, &mut |_, _| match cancel
        .load(Ordering::SeqCst)
    {
        true => std::ops::ControlFlow::Break(()),
        false => std::ops::ControlFlow::Continue(()),
    })?;
    let model = HookedQwen2::from_tensors_full(tensors, &cfg, dtype, &device)?;

    let name = req
        .out
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "logits.safetensors".into());
    let mut temp = TempOut {
        path: req
            .out
            .with_file_name(format!(".{name}.tmp-{}", std::process::id())),
        committed: false,
    };
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temp.path)
        .map_err(|e| io_error(&temp.path, e))?;
    let mut writer = BufWriter::new(file);
    let write = |w: &mut BufWriter<std::fs::File>, bytes: &[u8]| {
        w.write_all(bytes).map_err(|e| io_error(&temp.path, e))
    };
    write(
        &mut writer,
        &header(
            n_positions,
            cfg.vocab_size,
            req.out_dtype,
            ids.len(),
            &metadata,
        ),
    )?;
    for (i, prompt) in ids.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            return Err(BenchError::Cancelled);
        }
        let input = Tensor::new(prompt.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model
            .forward_logits(&input)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        if logits.iter().any(|v| v.is_nan() || *v == f32::INFINITY) {
            return Err(BenchError::Compute(format!(
                "logits NaN/+Inf pour le prompt {i} : modèle numériquement cassé"
            )));
        }
        let mut row = Vec::with_capacity(logits.len() * size);
        match req.out_dtype {
            OutDtype::F32 => logits.iter().for_each(|v| row.extend(v.to_le_bytes())),
            OutDtype::F16 => logits
                .iter()
                .for_each(|v| row.extend(half::f16::from_f32(*v).to_le_bytes())),
        }
        write(&mut writer, &row)?;
        on_prompt(i + 1, ids.len());
    }
    let offset_bytes: Vec<u8> = offsets.iter().flat_map(|o| o.to_le_bytes()).collect();
    write(&mut writer, &offset_bytes)?;
    let file = writer
        .into_inner()
        .map_err(|e| io_error(&temp.path, e.error()))?;
    file.sync_all().map_err(|e| io_error(&temp.path, e))?;
    drop(file);
    // Re-read by the engine: header, offsets and size are validated before the rename.
    crate::engine::ReadOnlyModel::open(&temp.path).map_err(|e| io_error(&temp.path, e))?;
    if cancel.load(Ordering::SeqCst) {
        return Err(BenchError::Cancelled);
    }
    if req.out.exists() && !req.overwrite {
        return Err(BenchError::OutputExists(req.out.clone()));
    }
    std::fs::rename(&temp.path, &req.out).map_err(|e| io_error(&req.out, e))?;
    temp.committed = true;
    let file_bytes = std::fs::metadata(&req.out).map(|m| m.len()).unwrap_or(0);

    Ok(BenchRun {
        report: BenchDumpReport {
            out: req.out.clone(),
            model: req.model.clone(),
            dtype_used,
            out_dtype: req.out_dtype.logits(),
            n_prompts: ids.len(),
            n_positions,
            vocab_size: cfg.vocab_size,
            max_tokens: req.max_tokens,
            truncated_prompts: truncated,
            prompts_sha256,
            tokenizer_sha256,
            file_bytes,
            timing_ms: DumpTiming {
                total: started.elapsed().as_millis() as u64,
            },
        },
        warnings,
        threshold_warnings: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_aligned_sorted_and_parseable() {
        let meta: BTreeMap<String, String> = [("b".to_owned(), "2".to_owned())].into();
        let h = header(3, 4, OutDtype::F16, 2, &meta);
        assert_eq!(h.len() % 8, 0);
        let mut file = h.clone();
        file.extend(vec![0u8; 3 * 4 * 2 + 8 * 3]);
        let parsed = safetensors::SafeTensors::deserialize(&file).unwrap();
        assert_eq!(parsed.tensor(LOGITS_TENSOR).unwrap().shape(), [3, 4]);
        assert_eq!(parsed.tensor(OFFSETS_TENSOR).unwrap().shape(), [3]);
        let text = String::from_utf8_lossy(&h[8..]).into_owned();
        assert!(
            text.find("__metadata__") < text.find("logits"),
            "sorted keys"
        );
        assert_eq!(prompts_hash(&[vec![1, 2]]), prompts_hash(&[vec![1, 2]]));
        assert_ne!(
            prompts_hash(&[vec![1, 2]]),
            prompts_hash(&[vec![1], vec![2]])
        );
    }
}
