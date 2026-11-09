//! `surgeon lora extract`: rank-r adapter from the difference of two checkpoints.
//!
//! For every selected 2D tensor, `rsvd` approximates `patched - base ~ U diag(S) V` and the
//! adapter stores `B = U sqrt(S)` [m x r] and `A = sqrt(S) V` [r x n], so that `B A` is the
//! rank-r approximation and `lora_alpha = r` gives a merge scale of exactly 1. The adapter
//! (PEFT keys) and its `adapter_config.json` are written next to each other, atomically
//! (temporary files, then rename). The result is a pure function of the inputs and `--seed`.
//!
//! The rank is lowered to the smallest `min(m, n)` of the selected tensors when needed, so
//! every pair has the same `r` (the `alpha / r` merge formula needs a single rank).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use half::{bf16, f16};
use regex::Regex;
use safetensors::tensor::TensorView as StView;
use safetensors::Dtype as StDtype;
use serde::Serialize;
use serde_json::json;

use super::cli_args::LoraExtractArgs;
use super::config::{self, CONFIG_FILE};
use super::decode_into;
use super::keys::{adapter_key, Side};
use crate::app::{AppDeps, CoreResult, Outcome, ProgressEvent, ProgressSink};
use crate::engine::{DType, ReadOnlyModel, TensorView};
use crate::surgery::error::{Result, SurgeryError};
use crate::surgery::events::{SinkObserver, SurgeryEvent, SurgeryObserver};
use crate::surgery::format::FactorDtype;
use crate::surgery::rsvd::{rsvd, RowSource, Rsvd, RsvdParams};
use crate::surgery::target::{ensure_not_input, resolve_shards};

/// Attention q/k/v/o projections and MLP gate/up/down weights of the decoder layers.
const DEFAULT_TARGETS: &str =
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$";

#[derive(Debug, Clone, Serialize)]
pub struct ExtractTensorReport {
    pub name: String,
    pub rel_error: f64,
    pub energy: f64,
}

/// Serialized under `data.lora` of the envelope.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ExtractReport {
    pub out: String,
    pub config: String,
    pub rank: usize,
    pub dtype: String,
    pub tensors: usize,
    pub mean_rel_error: f64,
    pub warnings: Vec<String>,
    /// Sorted by tensor name.
    pub per_tensor: Vec<ExtractTensorReport>,
}

/// A checkpoint (file or shard directory) mapped read-only.
struct Checkpoint {
    shards: Vec<ReadOnlyModel>,
}

impl Checkpoint {
    fn open(path: &Path) -> Result<Self> {
        let shards = resolve_shards(path)?
            .iter()
            .map(|p| Ok(ReadOnlyModel::open(p)?))
            .collect::<Result<_>>()?;
        Ok(Self { shards })
    }

    /// `name -> (dtype, shape)`, sorted.
    fn catalog(&self) -> BTreeMap<&str, (&DType, &[usize])> {
        self.shards
            .iter()
            .flat_map(|s| &s.index().tensors)
            .map(|t| (t.name.as_str(), (&t.dtype, t.shape.as_slice())))
            .collect()
    }

    fn view(&self, name: &str) -> Result<TensorView<'_>> {
        let shard = self.shards.iter().find(|s| s.index().get(name).is_ok());
        Ok(shard
            .ok_or_else(|| SurgeryError::Failed(format!("{name} : tenseur introuvable")))?
            .tensor(name)?)
    }
}

/// `patched - base`, decoded to f32 row by row.
struct DiffSource<'a> {
    base: &'a [u8],
    patched: &'a [u8],
    dtype: &'a DType,
    rows: usize,
    cols: usize,
}

impl RowSource for DiffSource<'_> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn read_rows(&self, start: usize, out: &mut [f32]) -> Result<usize> {
        let count = (out.len() / self.cols).min(self.rows.saturating_sub(start));
        let elem = self.dtype.size_of().unwrap_or(1);
        let range = start * self.cols * elem..(start + count) * self.cols * elem;
        let (Some(base), Some(patched)) = (self.base.get(range.clone()), self.patched.get(range))
        else {
            return Err(SurgeryError::Failed("lecture hors du tenseur".into()));
        };
        let out = &mut out[..count * self.cols];
        let mut scratch = vec![0.0; out.len()];
        decode_into(self.dtype, base, out)?;
        decode_into(self.dtype, patched, &mut scratch)?;
        for (o, p) in out.iter_mut().zip(&scratch) {
            *o = p - *o;
        }
        Ok(count)
    }
}

/// Names, shapes and dtypes must be identical; every difference is listed.
fn check_same_layout(base: &Checkpoint, patched: &Checkpoint) -> Result<()> {
    let (b, p) = (base.catalog(), patched.catalog());
    let mut diffs = Vec::new();
    for (name, (dtype, shape)) in &b {
        match p.get(name) {
            None => diffs.push(format!("{name} : absent de --patched")),
            Some((pd, ps)) if pd != dtype || ps != shape => diffs.push(format!(
                "{name} : {dtype} {shape:?} dans --base, {pd} {ps:?} dans --patched"
            )),
            Some(_) => {}
        }
    }
    diffs.extend(
        p.keys()
            .filter(|name| !b.contains_key(*name))
            .map(|name| format!("{name} : absent de --base")),
    );
    match diffs.is_empty() {
        true => Ok(()),
        false => Err(SurgeryError::Usage(format!(
            "--base et --patched diffèrent ({}) :\n  - {}",
            diffs.len(),
            diffs.join("\n  - ")
        ))),
    }
}

/// Little-endian bytes of `values` in `dtype`. A finite value that becomes infinite
/// (f16 range is +-65504) is an error with a bf16 hint instead of a silently broken adapter.
fn encode(values: &[f32], dtype: FactorDtype, tensor: &str) -> Result<Vec<u8>> {
    let overflow = || {
        SurgeryError::Failed(format!(
            "{tensor} : un facteur déborde de {}, relancez avec --dtype bf16 (ou f32)",
            dtype.as_str()
        ))
    };
    let mut bytes = Vec::with_capacity(values.len() * dtype.size_of());
    for &x in values {
        match dtype {
            FactorDtype::F32 => bytes.extend_from_slice(&x.to_le_bytes()),
            FactorDtype::F16 => {
                let h = f16::from_f32(x);
                if h.is_infinite() && x.is_finite() {
                    return Err(overflow());
                }
                bytes.extend_from_slice(&h.to_le_bytes());
            }
            FactorDtype::Bf16 => {
                let h = bf16::from_f32(x);
                if h.is_infinite() && x.is_finite() {
                    return Err(overflow());
                }
                bytes.extend_from_slice(&h.to_le_bytes());
            }
        }
    }
    Ok(bytes)
}

fn st_dtype(dtype: FactorDtype) -> StDtype {
    match dtype {
        FactorDtype::F32 => StDtype::F32,
        FactorDtype::F16 => StDtype::F16,
        FactorDtype::Bf16 => StDtype::BF16,
    }
}

/// `(B, A)` = `(U sqrt(S), sqrt(S) V)`, row-major f32.
fn split_factors(f: &Rsvd) -> (Vec<f32>, Vec<f32>) {
    let root: Vec<f32> = f.s.iter().map(|s| s.max(0.0).sqrt()).collect();
    let b =
        f.u.chunks_exact(f.r)
            .flat_map(|row| row.iter().zip(&root).map(|(u, s)| u * s))
            .collect();
    let a =
        f.v.chunks_exact(f.n)
            .zip(&root)
            .flat_map(|(row, s)| row.iter().map(move |v| v * s))
            .collect();
    (b, a)
}

/// Directory name of the base checkpoint, never an absolute path (the adapter is shared).
fn base_name(base: &Path) -> String {
    let abs = std::path::absolute(base).unwrap_or_else(|_| base.to_path_buf());
    let dir = match abs.is_dir() {
        true => Some(abs.as_path()),
        false => abs.parent(),
    };
    dir.and_then(Path::file_name)
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

struct Factor {
    key: String,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// Adapter and config go to temporary files first, so a failure leaves any existing output
/// untouched.
fn write_atomic(
    out: &Path,
    config_path: &Path,
    factors: &[Factor],
    dtype: FactorDtype,
    config_text: &str,
) -> Result<()> {
    let tmp = |p: &Path| {
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy())
            .unwrap_or_default();
        p.with_file_name(format!(".{name}.tmp{}", std::process::id()))
    };
    let (out_tmp, config_tmp) = (tmp(out), tmp(config_path));
    let views = factors
        .iter()
        .map(|f| {
            let view = StView::new(st_dtype(dtype), f.shape.clone(), &f.bytes)
                .map_err(|e| SurgeryError::Failed(format!("{} : {e}", f.key)))?;
            Ok((f.key.clone(), view))
        })
        .collect::<Result<Vec<_>>>()?;
    let metadata = std::collections::HashMap::from([("format".to_owned(), "pt".to_owned())]);
    let result = safetensors::serialize_to_file(views, Some(metadata), &out_tmp)
        .map_err(|e| SurgeryError::Failed(format!("{} : {e}", out.display())))
        .and_then(|()| {
            std::fs::write(&config_tmp, config_text).map_err(|e| SurgeryError::io(&config_tmp, &e))
        })
        .and_then(|()| std::fs::rename(&out_tmp, out).map_err(|e| SurgeryError::io(out, &e)))
        .and_then(|()| {
            std::fs::rename(&config_tmp, config_path).map_err(|e| SurgeryError::io(config_path, &e))
        });
    if result.is_err() {
        // Best effort: the temporaries are ours, the error is the one to report.
        let _ = std::fs::remove_file(&out_tmp);
        let _ = std::fs::remove_file(&config_tmp);
    }
    result
}

pub fn run_extract(
    args: &LoraExtractArgs,
    obs: &mut dyn SurgeryObserver,
    cancel: &AtomicBool,
) -> Result<ExtractReport> {
    if args.rank == 0 {
        return Err(SurgeryError::Usage("--rank doit valoir au moins 1".into()));
    }
    let out = &args.out;
    let config_path: PathBuf = out.with_file_name(CONFIG_FILE);
    if out.file_name().is_none() {
        return Err(SurgeryError::Usage(format!(
            "--out invalide : {}",
            out.display()
        )));
    }
    let parent = out.parent().filter(|p| !p.as_os_str().is_empty());
    if !parent.unwrap_or(Path::new(".")).is_dir() {
        return Err(SurgeryError::Usage(format!(
            "le dossier de --out n'existe pas : {}",
            out.display()
        )));
    }
    for target in [out, &config_path] {
        ensure_not_input(target, &[&args.base, &args.patched])?;
    }
    if !args.overwrite {
        if let Some(existing) = [out, &config_path].into_iter().find(|p| p.exists()) {
            return Err(SurgeryError::Usage(format!(
                "{} existe déjà, utilisez --overwrite",
                existing.display()
            )));
        }
    }
    let pattern = args.target_modules.as_deref().unwrap_or(DEFAULT_TARGETS);
    let pattern = Regex::new(pattern)
        .map_err(|e| SurgeryError::Usage(format!("--target-modules invalide : {e}")))?;

    let base = Checkpoint::open(&args.base)?;
    let patched = Checkpoint::open(&args.patched)?;
    check_same_layout(&base, &patched)?;

    let mut problems = Vec::new();
    let mut selected = Vec::new();
    for (name, (dtype, shape)) in base.catalog() {
        let &[m, n] = shape else { continue };
        if !pattern.is_match(name) {
            continue;
        }
        match (name.ends_with(".weight"), dtype.size_of()) {
            (false, _) => problems.push(format!(
                "{name} : nom sans suffixe .weight, clé PEFT impossible"
            )),
            (_, None) => problems.push(format!("{name} : dtype {dtype} non supporté")),
            _ => selected.push((name, dtype, m, n)),
        }
    }
    if !problems.is_empty() {
        return Err(SurgeryError::Usage(format!(
            "tenseurs sélectionnés inutilisables ({}) :\n  - {}",
            problems.len(),
            problems.join("\n  - ")
        )));
    }
    if selected.is_empty() {
        return Err(SurgeryError::Usage(
            "aucun tenseur 2D ne correspond à --target-modules".into(),
        ));
    }

    let mut warnings = Vec::new();
    let limit = selected
        .iter()
        .map(|&(_, _, m, n)| m.min(n))
        .min()
        .unwrap_or(1);
    let rank = args.rank.min(limit);
    if rank < args.rank {
        warnings.push(format!(
            "rang réduit de {} à {rank} (plus petite dimension des tenseurs sélectionnés)",
            args.rank
        ));
    }
    let params = RsvdParams {
        rank,
        seed: args.seed,
        ..Default::default()
    };

    let mut factors = Vec::with_capacity(selected.len() * 2);
    let mut per_tensor = Vec::with_capacity(selected.len());
    let total = selected.len();
    for (i, &(name, dtype, m, n)) in selected.iter().enumerate() {
        let (b, p) = (base.view(name)?, patched.view(name)?);
        let source = DiffSource {
            base: b.bytes,
            patched: p.bytes,
            dtype,
            rows: m,
            cols: n,
        };
        let svd = rsvd(&source, &params, cancel)?;
        let (b_vals, a_vals) = split_factors(&svd);
        for (side, shape, vals) in [
            (Side::B, vec![m, rank], b_vals),
            (Side::A, vec![rank, n], a_vals),
        ] {
            let key = adapter_key(name, side)
                .ok_or_else(|| SurgeryError::Failed(format!("{name} : clé PEFT impossible")))?;
            let bytes = encode(&vals, args.dtype, name)?;
            factors.push(Factor { key, shape, bytes });
        }
        per_tensor.push(ExtractTensorReport {
            name: name.to_owned(),
            rel_error: svd.rel_error,
            energy: svd.energy,
        });
        let event = SurgeryEvent::Tensor {
            done: i + 1,
            total,
            name: name.to_owned(),
        };
        if obs.event(event).is_break() {
            return Err(SurgeryError::Cancelled);
        }
    }

    let mut modules: Vec<String> = selected
        .iter()
        .filter_map(|(name, ..)| name.strip_suffix(".weight")?.rsplit('.').next())
        .map(str::to_owned)
        .collect();
    modules.sort();
    modules.dedup();
    let config_text = config::render(rank, &modules, &base_name(&args.base));
    write_atomic(out, &config_path, &factors, args.dtype, &config_text)?;

    let mean_rel_error = per_tensor.iter().map(|t| t.rel_error).sum::<f64>() / total as f64;
    Ok(ExtractReport {
        out: out.display().to_string(),
        config: config_path.display().to_string(),
        rank,
        dtype: args.dtype.as_str().to_owned(),
        tensors: total,
        mean_rel_error,
        warnings,
        per_tensor,
    })
}

pub fn core_extract(
    args: &LoraExtractArgs,
    deps: &AppDeps,
    sink: &mut dyn ProgressSink,
) -> CoreResult {
    let mut obs = SinkObserver {
        sink: &mut *sink,
        cancel: &deps.interrupted,
    };
    let report = run_extract(args, &mut obs, &deps.interrupted)?;
    for w in &report.warnings {
        sink.event(ProgressEvent::Warn(w.clone()));
    }
    let human = format!(
        "{} tenseur(s) extraits (rang {}, {}), erreur relative moyenne {:.4}\n  adapter : {}\n  config  : {}\n",
        report.tensors, report.rank, report.dtype, report.mean_rel_error, report.out, report.config
    );
    let mut outcome = Outcome::ok(
        json!({ "lora": report, "warnings": report.warnings }),
        human,
    );
    outcome.warnings = report.warnings.clone();
    Ok(outcome)
}
