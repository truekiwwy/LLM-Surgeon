//! Logits dumps: `[positions, vocab]` (or `[batch, seq, vocab]`, leading dimensions
//! flattened) in f32, f16 or bf16.
//!
//! - `.safetensors`: memory-mapped by `engine::ReadOnlyModel`, rows decoded on demand.
//!   Nothing of size `positions × vocab` is ever allocated.
//! - `.npy` and `.pt`/`.pth`: best effort through candle (`Tensor::read_npy`,
//!   `pickle::PthTensors`), loaded whole in f32 after a RAM check. candle's pickle reader
//!   interprets a safe subset of the format (no code execution); it supports uncompressed
//!   (ZIP stored) archives as written by `torch.save`, contiguous tensors only, and prints
//!   `skipping: ...` on stderr for non-tensor entries of the dictionary.

use std::collections::BTreeMap;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use candle_core::{DType, Tensor};
use schemars::JsonSchema;
use serde::Serialize;

use super::error::{BenchError, Result};
use crate::engine::{self, ReadOnlyModel};
use crate::extract::required_ram;
use crate::hardware::{BudgetPolicy, HardwareProbe};

/// Default tensor name, as written by `surgeon bench dump`.
pub const LOGITS_TENSOR: &str = "logits";
pub const OFFSETS_TENSOR: &str = "prompt_offsets";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum LogitsDtype {
    F32,
    F16,
    Bf16,
}

impl LogitsDtype {
    pub fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum DumpFormat {
    Safetensors,
    Npy,
    Pt,
}

enum Data {
    Mapped(ReadOnlyModel),
    Owned(Vec<f32>),
}

pub struct LogitsDump {
    pub path: PathBuf,
    pub format: DumpFormat,
    pub tensor: String,
    /// Stored dtype (`.npy` and `.pt` are converted to f32 in memory).
    pub dtype: LogitsDtype,
    pub shape: Vec<usize>,
    pub positions: usize,
    pub vocab: usize,
    /// `__metadata__` of a SafeTensors dump, empty otherwise.
    pub metadata: BTreeMap<String, String>,
    /// `n_prompts + 1` boundaries, validated against `positions`.
    pub prompt_offsets: Option<Vec<usize>>,
    /// Why a present `prompt_offsets` was ignored.
    pub offsets_warning: Option<String>,
    data: Data,
}

/// Borrowed rows of a dump, decoded to f64 one row at a time.
#[derive(Clone, Copy)]
pub(crate) enum Rows<'a> {
    Le { bytes: &'a [u8], dtype: LogitsDtype },
    F32(&'a [f32]),
}

impl Rows<'_> {
    /// Row `i` into `out` (`out.len()` is the vocabulary size).
    pub(crate) fn decode(&self, i: usize, out: &mut [f64]) {
        let vocab = out.len();
        match *self {
            Rows::F32(values) => {
                for (o, v) in out.iter_mut().zip(&values[i * vocab..(i + 1) * vocab]) {
                    *o = f64::from(*v);
                }
            }
            Rows::Le { bytes, dtype } => {
                let size = dtype.size();
                let row = &bytes[i * vocab * size..(i + 1) * vocab * size];
                match dtype {
                    LogitsDtype::F32 => {
                        for (o, c) in out.iter_mut().zip(row.chunks_exact(4)) {
                            *o = f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                        }
                    }
                    LogitsDtype::F16 => {
                        for (o, c) in out.iter_mut().zip(row.chunks_exact(2)) {
                            *o = half::f16::from_le_bytes([c[0], c[1]]).to_f64();
                        }
                    }
                    LogitsDtype::Bf16 => {
                        for (o, c) in out.iter_mut().zip(row.chunks_exact(2)) {
                            *o = half::bf16::from_le_bytes([c[0], c[1]]).to_f64();
                        }
                    }
                }
            }
        }
    }
}

impl LogitsDump {
    pub(crate) fn rows(&self) -> Result<Rows<'_>> {
        match &self.data {
            Data::Owned(values) => Ok(Rows::F32(values)),
            Data::Mapped(model) => {
                let view = model
                    .tensor(&self.tensor)
                    .map_err(|e| BenchError::input(&self.path, e))?;
                Ok(Rows::Le {
                    bytes: view.bytes,
                    dtype: self.dtype,
                })
            }
        }
    }

    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// Opens a dump by extension. `probe` checks the RAM before a `.npy`/`.pt` is loaded.
pub fn open_dump(
    path: &Path,
    tensor: Option<&str>,
    probe: &dyn HardwareProbe,
) -> Result<LogitsDump> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "safetensors" => open_safetensors(path, tensor),
        "npy" => open_npy(path, probe),
        "pt" | "pth" => open_pt(path, tensor, probe),
        _ => Err(BenchError::input(
            path,
            "format non reconnu (attendu .safetensors, .npy, .pt ou .pth)",
        )),
    }
}

/// `[.., vocab]` -> (positions, vocab); at least 2 dimensions.
fn flatten(path: &Path, shape: &[usize]) -> Result<(usize, usize)> {
    match shape {
        [] | [_] => Err(BenchError::input(
            path,
            format!("forme {shape:?} : au moins 2 dimensions attendues [positions, vocab]"),
        )),
        [lead @ .., vocab] => Ok((lead.iter().product(), *vocab)),
    }
}

fn open_safetensors(path: &Path, tensor: Option<&str>) -> Result<LogitsDump> {
    let model = ReadOnlyModel::open(path).map_err(|e| BenchError::input(path, e))?;
    let name = tensor.unwrap_or(LOGITS_TENSOR).to_owned();
    let info = model.index().get(&name).map_err(|_| {
        let names: Vec<&str> = model
            .index()
            .tensors
            .iter()
            .map(|t| t.name.as_str())
            .collect();
        BenchError::input(
            path,
            format!(
                "tenseur « {name} » absent (présents : {names:?}) ; --tensor pour en choisir un"
            ),
        )
    })?;
    let dtype = match &info.dtype {
        engine::DType::F32 => LogitsDtype::F32,
        engine::DType::F16 => LogitsDtype::F16,
        engine::DType::BF16 => LogitsDtype::Bf16,
        engine::DType::Other(other) => {
            return Err(BenchError::input(
                path,
                format!("tenseur « {name} » en {other} : f32, f16 ou bf16 attendu"),
            ))
        }
    };
    let shape = info.shape.clone();
    let (positions, vocab) = flatten(path, &shape)?;
    let (prompt_offsets, offsets_warning) = match read_offsets(&model, positions) {
        Ok(offsets) => (offsets, None),
        Err(reason) => (None, Some(reason)),
    };
    Ok(LogitsDump {
        path: path.to_path_buf(),
        format: DumpFormat::Safetensors,
        tensor: name,
        dtype,
        shape,
        positions,
        vocab,
        metadata: model.metadata(),
        prompt_offsets,
        offsets_warning,
        data: Data::Mapped(model),
    })
}

/// `Ok(None)` when absent; `Err(reason)` when present but unusable.
fn read_offsets(
    model: &ReadOnlyModel,
    positions: usize,
) -> std::result::Result<Option<Vec<usize>>, String> {
    let Ok(view) = model.tensor(OFFSETS_TENSOR) else {
        return Ok(None);
    };
    let bad = |why: &str| Err(format!("{OFFSETS_TENSOR} ignoré : {why}"));
    if view.info.dtype != engine::DType::Other("I64".into()) || view.info.shape.len() != 1 {
        return bad("tenseur I64 1D attendu");
    }
    let offsets: Vec<i64> = view
        .bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect();
    let valid = offsets.len() >= 2
        && offsets.first() == Some(&0)
        && offsets.last().copied() == i64::try_from(positions).ok()
        && offsets.windows(2).all(|w| w[0] <= w[1]);
    match valid {
        true => Ok(Some(offsets.iter().map(|&o| o as usize).collect())),
        false => bad("bornes non croissantes ou incohérentes avec le nombre de positions"),
    }
}

/// candle reads the whole file, converts to f32 then copies out: about 5x the file in the
/// worst case (f16 input), checked against the available RAM first.
fn check_ram(path: &Path, bytes: u64, probe: &dyn HardwareProbe) -> Result<()> {
    let need = required_ram(bytes, &BudgetPolicy::default());
    let available = probe
        .system_ram()
        .map_err(|e| BenchError::InsufficientResources(e.to_string()))?
        .available_bytes;
    match need <= available {
        true => Ok(()),
        false => Err(BenchError::InsufficientResources(format!(
            "{} : chargement en mémoire de {:.2} GiB (marge incluse), {:.2} GiB disponibles",
            path.display(),
            need as f64 / (1u64 << 30) as f64,
            available as f64 / (1u64 << 30) as f64
        ))),
    }
}

/// Runs a best-effort reader; a panic on a malformed file becomes an input error.
fn guarded<T>(path: &Path, read: impl FnOnce() -> candle_core::Result<T>) -> Result<T> {
    match catch_unwind(AssertUnwindSafe(read)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(e)) => Err(BenchError::input(path, format!("lecture impossible : {e}"))),
        Err(_) => Err(BenchError::input(
            path,
            "lecture impossible : fichier malformé",
        )),
    }
}

fn owned(
    path: &Path,
    format: DumpFormat,
    name: String,
    tensor: &Tensor,
    dtype: LogitsDtype,
) -> Result<LogitsDump> {
    let shape = tensor.dims().to_vec();
    let (positions, vocab) = flatten(path, &shape)?;
    let values = guarded(path, || {
        tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    })?;
    Ok(LogitsDump {
        path: path.to_path_buf(),
        format,
        tensor: name,
        dtype,
        shape,
        positions,
        vocab,
        metadata: BTreeMap::new(),
        prompt_offsets: None,
        offsets_warning: None,
        data: Data::Owned(values),
    })
}

fn stored_dtype(path: &Path, dtype: DType) -> Result<LogitsDtype> {
    match dtype {
        DType::F32 | DType::F64 => Ok(LogitsDtype::F32),
        DType::F16 => Ok(LogitsDtype::F16),
        DType::BF16 => Ok(LogitsDtype::Bf16),
        other => Err(BenchError::input(
            path,
            format!("dtype {other:?} : flottant attendu"),
        )),
    }
}

fn open_npy(path: &Path, probe: &dyn HardwareProbe) -> Result<LogitsDump> {
    let bytes = std::fs::metadata(path)
        .map_err(|e| BenchError::input(path, e))?
        .len();
    check_ram(path, bytes.saturating_mul(5), probe)?;
    let tensor = guarded(path, || Tensor::read_npy(path))?;
    let dtype = stored_dtype(path, tensor.dtype())?;
    owned(path, DumpFormat::Npy, "npy".into(), &tensor, dtype)
}

fn open_pt(path: &Path, tensor: Option<&str>, probe: &dyn HardwareProbe) -> Result<LogitsDump> {
    let pth = guarded(path, || candle_core::pickle::PthTensors::new(path, None))?;
    let mut names: Vec<&String> = pth.tensor_infos().keys().collect();
    names.sort();
    let name = match (tensor, names.as_slice()) {
        (Some(t), _) => t.to_owned(),
        (None, _) if pth.tensor_infos().contains_key(LOGITS_TENSOR) => LOGITS_TENSOR.into(),
        (None, [only]) => (*only).clone(),
        (None, _) => {
            return Err(BenchError::input(
                path,
                format!("plusieurs tenseurs sans « logits » ({names:?}) : --tensor pour choisir"),
            ))
        }
    };
    let info = pth.tensor_infos().get(&name).ok_or_else(|| {
        BenchError::input(
            path,
            format!("tenseur « {name} » absent (présents : {names:?})"),
        )
    })?;
    let elements = info.layout.shape().elem_count() as u64;
    let stored = elements * info.dtype.size_in_bytes() as u64;
    check_ram(path, stored + 8 * elements, probe)?;
    let dtype = stored_dtype(path, info.dtype)?;
    let tensor = guarded(path, || pth.get(&name))?
        .ok_or_else(|| BenchError::input(path, format!("tenseur « {name} » illisible")))?;
    owned(path, DumpFormat::Pt, name, &tensor, dtype)
}

/// Problems that make a comparison meaningless are errors; doubtful ones are warnings.
pub fn check_pair(before: &LogitsDump, after: &LogitsDump) -> Result<Vec<String>> {
    if before.vocab != after.vocab {
        return Err(BenchError::VocabMismatch {
            before: before.vocab,
            after: after.vocab,
        });
    }
    if before.positions != after.positions {
        return Err(BenchError::ShapeMismatch {
            before: before.shape.clone(),
            after: after.shape.clone(),
        });
    }
    if before.positions == 0 || before.vocab == 0 {
        return Err(BenchError::input(
            &before.path,
            "dump vide : au moins 1 position et 1 token attendus",
        ));
    }
    let mut warnings: Vec<String> = [before, after]
        .iter()
        .filter_map(|d| d.offsets_warning.as_ref())
        .cloned()
        .collect();
    for key in ["prompts_sha256", "tokenizer_sha256"] {
        if let (Some(a), Some(b)) = (before.meta(key), after.meta(key)) {
            if a != b {
                warnings.push(format!(
                    "{key} différent entre les deux dumps : comparaison potentiellement invalide \
                     (prompts ou tokenizer différents)"
                ));
            }
        }
    }
    if let (Some(a), Some(b)) = (&before.prompt_offsets, &after.prompt_offsets) {
        if a != b {
            warnings.push("prompt_offsets différents : rapport par prompt omis".into());
        }
    }
    Ok(warnings)
}

/// Per-prompt boundaries when both dumps agree on them.
pub fn shared_offsets<'a>(before: &'a LogitsDump, after: &LogitsDump) -> Option<&'a [usize]> {
    match (&before.prompt_offsets, &after.prompt_offsets) {
        (Some(a), Some(b)) if a == b => Some(a),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::logits_metrics::{run, LogitsRequest};
    use crate::hardware::{GpuInfo, MemoryInfo};
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;
    use std::collections::HashMap;
    use std::sync::atomic::AtomicBool;

    struct Ram(u64);

    impl HardwareProbe for Ram {
        fn system_ram(&self) -> crate::hardware::Result<MemoryInfo> {
            Ok(MemoryInfo {
                total_bytes: self.0,
                available_bytes: self.0,
            })
        }
        fn gpus(&self) -> crate::hardware::Result<Vec<GpuInfo>> {
            Ok(Vec::new())
        }
    }

    const PLENTY: Ram = Ram(64 << 30);

    fn encode(values: &[f32], dtype: Dtype) -> Vec<u8> {
        match dtype {
            Dtype::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Dtype::F16 => values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
                .collect(),
            _ => values
                .iter()
                .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
                .collect(),
        }
    }

    fn write(
        dir: &Path,
        name: &str,
        dtype: Dtype,
        shape: &[usize],
        values: &[f32],
        meta: &[(&str, &str)],
        offsets: Option<&[i64]>,
    ) -> PathBuf {
        let data = encode(values, dtype);
        let offset_bytes: Vec<u8> = offsets
            .unwrap_or(&[])
            .iter()
            .flat_map(|o| o.to_le_bytes())
            .collect();
        let mut tensors = vec![(
            LOGITS_TENSOR.to_owned(),
            TensorView::new(dtype, shape.to_vec(), &data).unwrap(),
        )];
        if let Some(o) = offsets {
            tensors.push((
                OFFSETS_TENSOR.to_owned(),
                TensorView::new(Dtype::I64, vec![o.len()], &offset_bytes).unwrap(),
            ));
        }
        let meta: HashMap<String, String> = meta
            .iter()
            .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
            .collect();
        let bytes = safetensors::serialize(tensors, (!meta.is_empty()).then_some(meta)).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// Multiples of 1/64 within ±4: exact in f32, f16 and bf16 alike.
    fn logits(positions: usize, vocab: usize, seed: f32) -> Vec<f32> {
        (0..positions * vocab)
            .map(|i| ((i as f32 * 0.37 + seed).sin() * 4.0 * 64.0).round() / 64.0)
            .collect()
    }

    fn request(before: &Path, after: &Path) -> LogitsRequest {
        LogitsRequest {
            before: before.into(),
            after: after.into(),
            tensor: None,
            top_k: 2,
            worst: 2,
            warn_kl: None,
            warn_top1_shift: None,
        }
    }

    fn never() -> AtomicBool {
        AtomicBool::new(false)
    }

    #[test]
    fn dtypes_mixed_and_three_dimensional_dumps() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let values = logits(6, 4, 0.0);
        let f32_3d = write(
            d,
            "a.safetensors",
            Dtype::F32,
            &[2, 3, 4],
            &values,
            &[],
            None,
        );
        let f16 = write(d, "b.safetensors", Dtype::F16, &[6, 4], &values, &[], None);
        let bf16 = write(d, "c.safetensors", Dtype::BF16, &[6, 4], &values, &[], None);
        let a = open_dump(&f32_3d, None, &PLENTY).unwrap();
        assert_eq!((a.positions, a.vocab, a.dtype), (6, 4, LogitsDtype::F32));
        for other in [&f16, &bf16] {
            let r = run(&request(&f32_3d, other), &PLENTY, &never(), &mut |_, _| {}).unwrap();
            assert_eq!(r.report.kl_mean, Some(0.0), "{other:?}");
            assert_eq!(r.report.top1_shift_rate, 0.0);
            assert_eq!(r.report.positions, 6);
        }
        let shifted: Vec<f32> = values.iter().map(|v| v * 1.5).collect();
        let s = write(d, "d.safetensors", Dtype::F16, &[6, 4], &shifted, &[], None);
        let r = run(&request(&f32_3d, &s), &PLENTY, &never(), &mut |_, _| {}).unwrap();
        assert!(r.report.kl_mean.unwrap() > 0.0);
        assert_eq!(r.report.verdict, crate::bench::Verdict::None);
        assert_eq!(r.report.prompts, None);
    }

    #[test]
    fn prompt_offsets_give_worst_prompts_and_hash_mismatch_warns() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let a = logits(5, 3, 0.0);
        let mut b = a.clone();
        b[4 * 3] += 3.0; // only the last position (prompt 1) moves
        let offsets = [0, 2, 5];
        let meta_a = [("prompts_sha256", "x"), ("tokenizer_sha256", "t")];
        let meta_b = [("prompts_sha256", "y"), ("tokenizer_sha256", "t")];
        let before = write(
            d,
            "a.safetensors",
            Dtype::F32,
            &[5, 3],
            &a,
            &meta_a,
            Some(&offsets),
        );
        let after = write(
            d,
            "b.safetensors",
            Dtype::F32,
            &[5, 3],
            &b,
            &meta_b,
            Some(&offsets),
        );
        let r = run(&request(&before, &after), &PLENTY, &never(), &mut |_, _| {}).unwrap();
        assert_eq!(r.report.prompts, Some(2));
        assert_eq!(r.report.worst_prompts[0].index, 1);
        assert_eq!(r.report.per_prompt[0].kl_mean, Some(0.0));
        assert_eq!(r.warnings.len(), 1, "{:?}", r.warnings);
        assert!(r.warnings[0].contains("prompts_sha256"));
        assert!(r.threshold_warnings.is_empty());

        let other = write(
            d,
            "c.safetensors",
            Dtype::F32,
            &[5, 3],
            &b,
            &[],
            Some(&[0, 1, 5]),
        );
        let r = run(&request(&before, &other), &PLENTY, &never(), &mut |_, _| {}).unwrap();
        assert_eq!(
            r.report.prompts, None,
            "different offsets: no per-prompt report"
        );
        assert!(r.warnings.iter().any(|w| w.contains("prompt_offsets")));
    }

    #[test]
    fn thresholds_are_strict() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let before = write(
            d,
            "a.safetensors",
            Dtype::F32,
            &[1, 2],
            &[0.0, 0.0],
            &[],
            None,
        );
        let after = write(
            d,
            "b.safetensors",
            Dtype::F32,
            &[1, 2],
            &[0.0, 3f32.ln()],
            &[],
            None,
        );
        let kl = run(&request(&before, &after), &PLENTY, &never(), &mut |_, _| {})
            .unwrap()
            .report
            .kl_mean
            .unwrap();
        let mut req = request(&before, &after);
        req.top_k = 1;
        req.warn_kl = Some(kl);
        req.warn_top1_shift = Some(1.0);
        let equal = run(&req, &PLENTY, &never(), &mut |_, _| {}).unwrap();
        assert!(equal.threshold_warnings.is_empty(), "equal is not above");
        assert_eq!(equal.report.verdict, crate::bench::Verdict::Ok);
        req.warn_kl = Some(kl - 1e-9);
        req.warn_top1_shift = Some(0.0);
        let above = run(&req, &PLENTY, &never(), &mut |_, _| {}).unwrap();
        assert_eq!(
            above.threshold_warnings.len(),
            2,
            "{:?}",
            above.threshold_warnings
        );
        assert_eq!(above.report.verdict, crate::bench::Verdict::Warn);
        req.warn_kl = Some(-1.0);
        assert!(run(&req, &PLENTY, &never(), &mut |_, _| {}).is_err());
    }

    #[test]
    fn incompatible_corrupted_and_odd_files() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let a = write(
            d,
            "a.safetensors",
            Dtype::F32,
            &[2, 3],
            &logits(2, 3, 0.0),
            &[],
            None,
        );
        let vocab = write(
            d,
            "v.safetensors",
            Dtype::F32,
            &[2, 4],
            &logits(2, 4, 0.0),
            &[],
            None,
        );
        let pos = write(
            d,
            "p.safetensors",
            Dtype::F32,
            &[3, 3],
            &logits(3, 3, 0.0),
            &[],
            None,
        );
        let pair = |x: &Path, y: &Path| {
            check_pair(
                &open_dump(x, None, &PLENTY).unwrap(),
                &open_dump(y, None, &PLENTY).unwrap(),
            )
        };
        assert!(matches!(
            pair(&a, &vocab),
            Err(BenchError::VocabMismatch {
                before: 3,
                after: 4
            })
        ));
        let err = pair(&a, &pos).unwrap_err();
        assert!(
            matches!(&err, BenchError::ShapeMismatch { before, after } if before == &[2, 3] && after == &[3, 3]),
            "{err:?}"
        );

        let garbage = d.join("g.safetensors");
        std::fs::write(&garbage, b"definitely not a safetensors file").unwrap();
        assert!(matches!(
            open_dump(&garbage, None, &PLENTY),
            Err(BenchError::Input { .. })
        ));
        let bytes = std::fs::read(&a).unwrap();
        let truncated = d.join("t.safetensors");
        std::fs::write(&truncated, &bytes[..bytes.len() - 5]).unwrap();
        assert!(matches!(
            open_dump(&truncated, None, &PLENTY),
            Err(BenchError::Input { .. })
        ));
        let err = open_dump(&a, Some("nope"), &PLENTY).err().unwrap();
        assert!(
            err.to_string().contains("logits"),
            "lists the present tensors: {err}"
        );
        let one_d = write(
            d,
            "o.safetensors",
            Dtype::F32,
            &[3],
            &[1.0, 2.0, 3.0],
            &[],
            None,
        );
        assert!(open_dump(&one_d, None, &PLENTY).is_err());
        assert!(open_dump(&d.join("x.bin"), None, &PLENTY).is_err());
        let empty = write(d, "e.safetensors", Dtype::F32, &[0, 3], &[], &[], None);
        let e = open_dump(&empty, None, &PLENTY).unwrap();
        assert!(check_pair(&e, &open_dump(&empty, None, &PLENTY).unwrap()).is_err());
    }

    #[test]
    fn npy_is_read_after_a_ram_check() {
        let dir = tempfile::tempdir().unwrap();
        let values = logits(3, 5, 1.0);
        let path = dir.path().join("a.npy");
        Tensor::from_vec(values.clone(), (3, 5), &candle_core::Device::Cpu)
            .unwrap()
            .write_npy(&path)
            .unwrap();
        let dump = open_dump(&path, None, &PLENTY).unwrap();
        assert_eq!(
            (dump.positions, dump.vocab, dump.format),
            (3, 5, DumpFormat::Npy)
        );
        let mut row = vec![0.0; 5];
        dump.rows().unwrap().decode(2, &mut row);
        assert_eq!(row[4], f64::from(values[14]));
        assert!(matches!(
            open_dump(&path, None, &Ram(0)),
            Err(BenchError::InsufficientResources(_))
        ));
        let bad = dir.path().join("b.npy");
        std::fs::write(&bad, b"\x93NUMPY garbage").unwrap();
        assert!(matches!(
            open_dump(&bad, None, &PLENTY),
            Err(BenchError::Input { .. })
        ));
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut crc = !0u32;
        for &b in data {
            crc ^= u32::from(b);
            for _ in 0..8 {
                crc = if crc & 1 == 1 {
                    (crc >> 1) ^ 0xEDB8_8320
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    /// Minimal ZIP archive with stored (uncompressed) entries, as `torch.save` writes.
    fn stored_zip(entries: &[(&str, Vec<u8>)]) -> Vec<u8> {
        let (mut out, mut central) = (Vec::new(), Vec::new());
        for (name, data) in entries {
            let offset = out.len() as u32;
            let crc = crc32(data);
            out.extend(0x0403_4b50u32.to_le_bytes());
            for v in [20u16, 0, 0, 0, 0x21] {
                out.extend(v.to_le_bytes());
            }
            out.extend(crc.to_le_bytes());
            out.extend((data.len() as u32).to_le_bytes());
            out.extend((data.len() as u32).to_le_bytes());
            out.extend((name.len() as u16).to_le_bytes());
            out.extend(0u16.to_le_bytes());
            out.extend(name.as_bytes());
            out.extend(data);

            central.extend(0x0201_4b50u32.to_le_bytes());
            for v in [20u16, 20, 0, 0, 0, 0x21] {
                central.extend(v.to_le_bytes());
            }
            central.extend(crc.to_le_bytes());
            central.extend((data.len() as u32).to_le_bytes());
            central.extend((data.len() as u32).to_le_bytes());
            central.extend((name.len() as u16).to_le_bytes());
            for v in [0u16, 0, 0, 0] {
                central.extend(v.to_le_bytes());
            }
            central.extend(0u32.to_le_bytes());
            central.extend(offset.to_le_bytes());
            central.extend(name.as_bytes());
        }
        let cd_offset = out.len() as u32;
        out.extend(&central);
        out.extend(0x0605_4b50u32.to_le_bytes());
        for v in [0u16, 0, entries.len() as u16, entries.len() as u16] {
            out.extend(v.to_le_bytes());
        }
        out.extend((central.len() as u32).to_le_bytes());
        out.extend(cd_offset.to_le_bytes());
        out.extend(0u16.to_le_bytes());
        out
    }

    /// `{"logits": _rebuild_tensor_v2(<FloatStorage "0">, 0, (rows, cols), (cols, 1),
    /// False, {})}` in pickle protocol 2.
    fn pickle_state_dict(rows: u8, cols: u8) -> Vec<u8> {
        let mut p = vec![0x80, 2, b'}', b'('];
        let unicode = |p: &mut Vec<u8>, s: &str| {
            p.push(b'X');
            p.extend((s.len() as u32).to_le_bytes());
            p.extend(s.as_bytes());
        };
        unicode(&mut p, "logits");
        p.extend(b"ctorch._utils\n_rebuild_tensor_v2\n(");
        p.push(b'(');
        unicode(&mut p, "storage");
        p.extend(b"ctorch\nFloatStorage\n");
        unicode(&mut p, "0");
        unicode(&mut p, "cpu");
        p.extend([b'K', rows * cols, b't', b'Q']);
        p.extend([b'K', 0]);
        p.extend([b'(', b'K', rows, b'K', cols, b't']);
        p.extend([b'(', b'K', cols, b'K', 1, b't']);
        p.extend([0x89, b'}', b't', b'R', b'u', b'.']);
        p
    }

    #[test]
    fn pytorch_state_dict_best_effort() {
        let dir = tempfile::tempdir().unwrap();
        let values = logits(2, 3, 2.0);
        let storage: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let zip = stored_zip(&[
            ("archive/data.pkl", pickle_state_dict(2, 3)),
            ("archive/data/0", storage),
        ]);
        let path = dir.path().join("a.pt");
        std::fs::write(&path, zip).unwrap();
        let dump = open_dump(&path, None, &PLENTY).unwrap();
        assert_eq!(
            (dump.positions, dump.vocab, dump.format),
            (2, 3, DumpFormat::Pt)
        );
        assert_eq!(dump.tensor, "logits");
        let mut row = vec![0.0; 3];
        dump.rows().unwrap().decode(1, &mut row);
        let expected: Vec<f64> = values[3..].iter().map(|v| f64::from(*v)).collect();
        assert_eq!(row, expected);
        assert!(matches!(
            open_dump(&path, None, &Ram(0)),
            Err(BenchError::InsufficientResources(_))
        ));
        assert!(open_dump(&path, Some("absent"), &PLENTY).is_err());

        let corrupted = dir.path().join("b.pt");
        std::fs::write(&corrupted, b"PK\x03\x04 broken").unwrap();
        assert!(matches!(
            open_dump(&corrupted, None, &PLENTY),
            Err(BenchError::Input { .. })
        ));
    }

    /// A dump far larger than the scratch buffers, made of zero pages (`set_len`): read
    /// through the memory map row by row, never materialized.
    #[test]
    fn large_sparse_dump_streams() {
        let dir = tempfile::tempdir().unwrap();
        let (positions, vocab) = (100_000usize, 64usize);
        let path = dir.path().join("big.safetensors");
        let data_bytes = positions * vocab * 2;
        let mut header = serde_json::json!({
            LOGITS_TENSOR: { "dtype": "F16", "shape": [positions, vocab], "data_offsets": [0, data_bytes] }
        })
        .to_string();
        while !(8 + header.len()).is_multiple_of(8) {
            header.push(' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len((bytes.len() + data_bytes) as u64).unwrap();
        drop(file);
        let dump = open_dump(&path, None, &PLENTY).unwrap();
        assert!(
            matches!(dump.rows().unwrap(), Rows::Le { .. }),
            "borrowed from the map"
        );
        let mut blocks = 0;
        let r = run(&request(&path, &path), &PLENTY, &never(), &mut |_, _| {
            blocks += 1
        })
        .unwrap();
        assert_eq!(r.report.kl_mean, Some(0.0));
        assert_eq!(
            blocks,
            positions.div_ceil(crate::bench::logits_metrics::BLOCK_POSITIONS)
        );
    }

    #[test]
    fn report_json_is_deterministic_without_timing() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        let offsets = [0, 10, 40];
        let a = write(
            d,
            "a.safetensors",
            Dtype::F32,
            &[40, 7],
            &logits(40, 7, 0.0),
            &[],
            Some(&offsets),
        );
        let b = write(
            d,
            "b.safetensors",
            Dtype::F16,
            &[40, 7],
            &logits(40, 7, 0.3),
            &[],
            Some(&offsets),
        );
        let json = || {
            let r = run(&request(&a, &b), &PLENTY, &never(), &mut |_, _| {}).unwrap();
            let mut v = serde_json::to_value(&r.report).unwrap();
            v.as_object_mut().unwrap().remove("timing_ms");
            serde_json::to_string(&v).unwrap()
        };
        assert_eq!(json(), json());
    }
}
