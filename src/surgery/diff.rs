//! `surgeon diff`: low-rank approximation of `tuned - base`, written as a `.surgery`.
//!
//! Pipeline of [`run_diff`]: open both checkpoints (files or sharded directories, memory
//! mapped) -> compare names, shapes and dtypes of ALL tensors -> filter (`--include`,
//! `--exclude`, embeddings) -> scan: tensors with identical bytes are skipped, the others get
//! the SHA-256 of their base bytes -> write: 2D tensors go through `rsvd` on a [`RowSource`]
//! that computes `tuned - base` row by row, everything else is stored as an exact f32 delta.
//! An f16 factor overflow restarts the whole write with bf16 (the header has a single
//! `factor_dtype`).
//!
//! Tensors are processed in sorted name order and the header carries no timestamp unless
//! asked, so the output is a pure function of the inputs and the seed.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use half::{bf16, f16};
use regex::Regex;
use serde::Serialize;
use serde_json::json;

use super::cli_args::{DiffArgs, OnHighError};
use super::error::{Result, SurgeryError};
use super::events::{SinkObserver, SurgeryEvent, SurgeryObserver};
use super::format::{
    sha256_hex, Compression, CompressionAlgo, FactorDtype, LowRankFactors, MAX_RANK,
};
use super::rsvd::{rsvd, RowSource, RsvdParams};
use super::target::resolve_shards;
use super::writer::{HeaderInit, SurgeryWriter, TensorMeta, WriteSummary};
use crate::app::{exit, AppDeps, CoreResult, Outcome, ProgressEvent, ProgressSink};
use crate::engine::{DType, ReadOnlyModel, TensorView};
use crate::hardware::{self, BudgetPolicy, Verdict};

/// Deflate level of the blobs: factors are floats, a higher level costs time for little gain.
const COMPRESSION: Compression = Compression {
    algo: CompressionAlgo::Deflate,
    level: 1,
};

/// Worst tensors listed in the human summary.
const HUMAN_TENSORS: usize = 20;

#[derive(Debug, Clone, Serialize)]
pub struct DiffTensorReport {
    pub name: String,
    pub kind: String,
    /// Effective rank (`min(--rank, m, n)`), 0 for `full_delta`.
    pub rank: usize,
    /// 0 for `full_delta` (the delta is stored exactly).
    pub rel_error: f64,
    /// Share of `||delta||_F^2` captured by the factors, 1 for `full_delta`.
    pub energy: f64,
}

/// Serialized under `data.surgery` of the envelope.
#[derive(Debug, Clone, Serialize, Default)]
pub struct DiffReport {
    pub out: String,
    pub bytes: u64,
    pub sha256: String,
    /// Requested rank (`--rank`); each tensor reports its effective rank.
    pub rank: usize,
    pub factor_dtype: String,
    pub tensors: Vec<DiffTensorReport>,
    pub skipped_identical: usize,
    pub skipped_filtered: usize,
    pub warnings: Vec<String>,
}

/// Element type of the tensors `diff` can read.
#[derive(Clone, Copy)]
enum Elem {
    Bf16,
    F16,
    F32,
}

impl Elem {
    fn of(dtype: &DType, tensor: &str) -> Result<Self> {
        match dtype {
            DType::BF16 => Ok(Self::Bf16),
            DType::F16 => Ok(Self::F16),
            DType::F32 => Ok(Self::F32),
            DType::Other(name) => Err(SurgeryError::Usage(format!(
                "{tensor} : dtype {name} non pris en charge (BF16, F16 ou F32 attendus)"
            ))),
        }
    }

    fn size(self) -> usize {
        match self {
            Self::Bf16 | Self::F16 => 2,
            Self::F32 => 4,
        }
    }

    /// Decodes `out.len()` elements of `bytes` starting at element `first`.
    fn load(self, bytes: &[u8], first: usize, out: &mut [f32]) {
        let src = &bytes[first * self.size()..(first + out.len()) * self.size()];
        match self {
            Self::Bf16 => out
                .iter_mut()
                .zip(src.chunks_exact(2))
                .for_each(|(o, c)| *o = bf16::from_le_bytes([c[0], c[1]]).to_f32()),
            Self::F16 => out
                .iter_mut()
                .zip(src.chunks_exact(2))
                .for_each(|(o, c)| *o = f16::from_le_bytes([c[0], c[1]]).to_f32()),
            Self::F32 => out
                .iter_mut()
                .zip(src.chunks_exact(4))
                .for_each(|(o, c)| *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        }
    }
}

/// `tuned - base` of one tensor seen as a `rows x cols` matrix (leading dimensions folded into
/// rows, a 1D tensor is one row), computed in f32 while streaming.
struct DeltaRows<'a> {
    name: &'a str,
    base: &'a [u8],
    tuned: &'a [u8],
    elem: Elem,
    rows: usize,
    cols: usize,
}

impl RowSource for DeltaRows<'_> {
    fn rows(&self) -> usize {
        self.rows
    }

    fn cols(&self) -> usize {
        self.cols
    }

    fn read_rows(&self, start: usize, out: &mut [f32]) -> Result<usize> {
        let rows = (out.len() / self.cols).min(self.rows.saturating_sub(start));
        let out = &mut out[..rows * self.cols];
        let first = start * self.cols;
        self.elem.load(self.tuned, first, out);
        let mut base = vec![0.0f32; out.len()];
        self.elem.load(self.base, first, &mut base);
        for (i, (d, b)) in out.iter_mut().zip(&base).enumerate() {
            *d -= b;
            // NaN/Inf in either checkpoint, or a delta beyond the f32 range.
            if !d.is_finite() {
                let at = first + i;
                return Err(SurgeryError::Usage(format!(
                    "{} : valeur non finie (ligne {}, colonne {}) dans un checkpoint, \
                     ou delta hors de la plage f32",
                    self.name,
                    at / self.cols,
                    at % self.cols
                )));
            }
        }
        Ok(rows)
    }
}

/// One or several `.safetensors` files, memory mapped.
struct Checkpoint {
    models: Vec<ReadOnlyModel>,
}

impl Checkpoint {
    fn open(path: &Path) -> Result<Self> {
        let models = resolve_shards(path)?
            .iter()
            .map(|shard| Ok(ReadOnlyModel::open(shard)?))
            .collect::<Result<_>>()?;
        Ok(Self { models })
    }

    /// Every tensor by name (the first shard wins on a duplicate).
    fn tensors(&self) -> Result<Tensors<'_>> {
        let mut map = BTreeMap::new();
        for model in &self.models {
            for info in &model.index().tensors {
                if !map.contains_key(info.name.as_str()) {
                    map.insert(info.name.as_str(), model.tensor(&info.name)?);
                }
            }
        }
        Ok(map)
    }
}

type Tensors<'a> = BTreeMap<&'a str, TensorView<'a>>;

/// An error listing every difference of names, shapes and dtypes.
fn compare(base: &Tensors, tuned: &Tensors) -> Result<()> {
    let mut problems = Vec::new();
    for (name, b) in base {
        match tuned.get(name) {
            None => problems.push(format!("{name} : absent du checkpoint tuned")),
            Some(t) => {
                if b.info.shape != t.info.shape {
                    problems.push(format!(
                        "{name} : forme {:?} (base) contre {:?} (tuned)",
                        b.info.shape, t.info.shape
                    ));
                }
                if b.info.dtype != t.info.dtype {
                    problems.push(format!(
                        "{name} : dtype {} (base) contre {} (tuned)",
                        b.info.dtype, t.info.dtype
                    ));
                }
            }
        }
    }
    problems.extend(
        tuned
            .keys()
            .filter(|name| !base.contains_key(*name))
            .map(|name| format!("{name} : absent du checkpoint base")),
    );
    match problems.is_empty() {
        true => Ok(()),
        false => Err(SurgeryError::Usage(format!(
            "base et tuned diffèrent ({} écart(s)) :\n  {}",
            problems.len(),
            problems.join("\n  ")
        ))),
    }
}

/// A tensor that differs between the checkpoints, with the hash of its base bytes.
struct Work<'a> {
    name: &'a str,
    base: &'a [u8],
    tuned: &'a [u8],
    dtype: &'a DType,
    elem: Elem,
    shape: &'a [usize],
    base_sha256: String,
}

impl<'a> Work<'a> {
    fn rows(&self) -> Result<DeltaRows<'a>> {
        let Some((&cols, lead)) = self.shape.split_last() else {
            return Err(SurgeryError::Usage(format!(
                "{} : tenseur scalaire non pris en charge",
                self.name
            )));
        };
        Ok(DeltaRows {
            name: self.name,
            base: self.base,
            tuned: self.tuned,
            elem: self.elem,
            rows: lead.iter().product(),
            cols,
        })
    }
}

fn notify(obs: &mut dyn SurgeryObserver, ev: SurgeryEvent) -> Result<()> {
    match obs.event(ev) {
        ControlFlow::Continue(()) => Ok(()),
        ControlFlow::Break(()) => Err(SurgeryError::Cancelled),
    }
}

fn compile(flag: &str, pattern: &Option<String>) -> Result<Option<Regex>> {
    pattern
        .as_deref()
        .map(|p| {
            Regex::new(p)
                .map_err(|e| SurgeryError::Usage(format!("{flag} : expression invalide : {e}")))
        })
        .transpose()
}

fn is_embedding(name: &str) -> bool {
    name.contains("embed_tokens") || name.contains("lm_head")
}

/// File or directory name only: no absolute path ends up in the patch.
fn display_name(path: &Path) -> String {
    let abs = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    abs.file_name().map_or_else(
        || "checkpoint".to_owned(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// `YYYY-MM-DDTHH:MM:SSZ` (civil-from-days of H. Hinnant, no date dependency).
fn rfc3339_utc(secs: u64) -> String {
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

fn now_rfc3339() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    rfc3339_utc(secs)
}

/// Engine: paths and options are already validated by the caller (CLI or MCP sandbox).
pub fn run_diff(
    args: &DiffArgs,
    obs: &mut dyn SurgeryObserver,
    cancel: &AtomicBool,
) -> Result<DiffReport> {
    if args.rank == 0 || args.rank > MAX_RANK {
        return Err(SurgeryError::Usage(format!(
            "--rank doit être dans 1..={MAX_RANK}, reçu {}",
            args.rank
        )));
    }
    if args
        .max_rel_error
        .is_some_and(|t| !(t.is_finite() && t >= 0.0))
    {
        return Err(SurgeryError::Usage(
            "--max-rel-error doit être un nombre positif".into(),
        ));
    }
    let include = compile("--include", &args.include)?;
    let exclude = compile("--exclude", &args.exclude)?;
    if !args.overwrite && args.out.exists() {
        return Err(SurgeryError::Usage(format!(
            "{} existe déjà (utiliser --overwrite pour l'écraser)",
            args.out.display()
        )));
    }

    let base = Checkpoint::open(&args.base)?;
    let tuned = Checkpoint::open(&args.tuned)?;
    let (base_tensors, tuned_tensors) = (base.tensors()?, tuned.tensors()?);
    compare(&base_tensors, &tuned_tensors)?;

    let selected: Vec<&str> = base_tensors
        .keys()
        .copied()
        .filter(|name| {
            (args.include_embeddings || !is_embedding(name))
                && include.as_ref().is_none_or(|re| re.is_match(name))
                && !exclude.as_ref().is_some_and(|re| re.is_match(name))
        })
        .collect();
    let skipped_filtered = base_tensors.len() - selected.len();

    let mut skipped_identical = 0;
    let mut work = Vec::new();
    for (i, name) in selected.iter().enumerate() {
        notify(
            obs,
            SurgeryEvent::Hashing {
                done: i + 1,
                total: selected.len(),
                name: (*name).to_owned(),
            },
        )?;
        let (b, t) = (base_tensors[name], tuned_tensors[name]);
        if b.bytes == t.bytes {
            skipped_identical += 1;
            continue;
        }
        work.push(Work {
            name,
            base: b.bytes,
            tuned: t.bytes,
            dtype: &b.info.dtype,
            elem: Elem::of(&b.info.dtype, name)?,
            shape: &b.info.shape,
            base_sha256: sha256_hex(b.bytes),
        });
    }

    let init = |factor_dtype| HeaderInit {
        created_with: format!("weight-surgeon {}", env!("CARGO_PKG_VERSION")),
        base_name: display_name(&args.base),
        base_tensor_count: base_tensors.len(),
        tuned_name: display_name(&args.tuned),
        rank: args.rank,
        factor_dtype,
        compression: COMPRESSION,
        timestamp: args.timestamp.then(now_rfc3339),
    };
    let mut factor_dtype = args.factor_dtype;
    let mut warnings = Vec::new();
    let written = loop {
        match write_patch(args, &work, init(factor_dtype), obs, cancel) {
            Err(SurgeryError::FactorOverflow { tensor, dtype })
                if factor_dtype == FactorDtype::F16 =>
            {
                warnings.push(format!(
                    "{tensor} : un facteur déborde de {dtype}, écriture refaite en bf16 \
                     (plage plus large, mantisse plus courte)"
                ));
                factor_dtype = FactorDtype::Bf16;
            }
            other => break other?,
        }
    };
    warnings.extend(written.warnings);
    Ok(DiffReport {
        out: args.out.display().to_string(),
        bytes: written.summary.bytes,
        sha256: written.summary.sha256,
        rank: args.rank,
        factor_dtype: factor_dtype.as_str().to_owned(),
        tensors: written.tensors,
        skipped_identical,
        skipped_filtered,
        warnings,
    })
}

struct Written {
    summary: WriteSummary,
    tensors: Vec<DiffTensorReport>,
    warnings: Vec<String>,
}

/// One complete write attempt. Dropping the writer on any error removes its scratch file and
/// leaves nothing at `--out`.
fn write_patch(
    args: &DiffArgs,
    work: &[Work],
    init: HeaderInit,
    obs: &mut dyn SurgeryObserver,
    cancel: &AtomicBool,
) -> Result<Written> {
    let mut writer = SurgeryWriter::create(&args.out, init, args.overwrite)?;
    let params = RsvdParams {
        rank: args.rank,
        oversample: args.oversample,
        power_iters: args.power_iters,
        seed: args.seed,
    };
    let (mut tensors, mut warnings) = (Vec::new(), Vec::new());
    for (i, item) in work.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            return Err(SurgeryError::Cancelled);
        }
        let src = item.rows()?;
        let meta = |rel_error, energy| TensorMeta {
            name: item.name.to_owned(),
            shape: item.shape.to_vec(),
            target_dtype: item.dtype.to_string(),
            base_sha256: item.base_sha256.clone(),
            rel_error,
            energy,
        };
        // `None`: stored whole (1D and N-D tensors, or `--on-high-error full`).
        let mut approx = (item.shape.len() == 2)
            .then(|| rsvd(&src, &params, cancel))
            .transpose()?;
        if let (Some(res), Some(limit)) = (&approx, args.max_rel_error) {
            if res.rel_error > limit {
                let outcome = match args.on_high_error {
                    OnHighError::Warn => "gardé en rang faible",
                    OnHighError::Skip => "absent du patch",
                    OnHighError::Full => "stocké en full_delta",
                };
                warnings.push(format!(
                    "{} : erreur relative {:.4} au-dessus de {limit}, {outcome}",
                    item.name, res.rel_error
                ));
                match args.on_high_error {
                    OnHighError::Warn => {}
                    OnHighError::Full => approx = None,
                    OnHighError::Skip => {
                        notify(obs, tensor_event(i, work.len(), item))?;
                        continue;
                    }
                }
            }
        }
        tensors.push(match approx {
            Some(res) => {
                let (rank, rel_error, energy) = (res.r, res.rel_error, res.energy);
                let factors = LowRankFactors {
                    u: res.u,
                    s: res.s,
                    v: res.v,
                };
                writer.add_lowrank(meta(rel_error, energy), &factors)?;
                report(item, "lowrank", rank, rel_error, energy)
            }
            None => {
                let mut delta = vec![0.0f32; src.rows * src.cols];
                src.read_rows(0, &mut delta)?;
                writer.add_full_delta(meta(0.0, 1.0), &delta)?;
                report(item, "full_delta", 0, 0.0, 1.0)
            }
        });
        notify(obs, tensor_event(i, work.len(), item))?;
    }
    let summary = writer.finish()?;
    Ok(Written {
        summary,
        tensors,
        warnings,
    })
}

fn tensor_event(i: usize, total: usize, item: &Work) -> SurgeryEvent {
    SurgeryEvent::Tensor {
        done: i + 1,
        total,
        name: item.name.to_owned(),
    }
}

fn report(item: &Work, kind: &str, rank: usize, rel_error: f64, energy: f64) -> DiffTensorReport {
    DiffTensorReport {
        name: item.name.to_owned(),
        kind: kind.to_owned(),
        rank,
        rel_error,
        energy,
    }
}

fn mib(bytes: u64) -> String {
    format!("{:.2} MiB", bytes as f64 / f64::from(1u32 << 20))
}

fn human(report: &DiffReport) -> String {
    let mut lines = vec![
        format!(
            "patch écrit : {} ({}, factor_dtype {}, rang demandé {})",
            report.out,
            mib(report.bytes),
            report.factor_dtype,
            report.rank
        ),
        format!(
            "{} tenseur(s) dans le patch, {} identique(s) ignoré(s), {} filtré(s)",
            report.tensors.len(),
            report.skipped_identical,
            report.skipped_filtered
        ),
    ];
    let mut worst: Vec<&DiffTensorReport> = report.tensors.iter().collect();
    worst.sort_by(|a, b| b.rel_error.total_cmp(&a.rel_error));
    lines.extend(worst.iter().take(HUMAN_TENSORS).map(|t| {
        format!(
            "  {} · {} · rang {} · erreur relative {:.4} · énergie {:.4}",
            t.name, t.kind, t.rank, t.rel_error, t.energy
        )
    }));
    if worst.len() > HUMAN_TENSORS {
        lines.push(format!(
            "  ... {} autres tenseurs (le détail est dans --json)",
            worst.len() - HUMAN_TENSORS
        ));
    }
    lines.push(
        "Note : le .surgery est une APPROXIMATION du fine-tune (erreur relative par tenseur), \
         exacte seulement si la différence est de rang <= r. Sa taille vaut environ \
         la somme de r(m+n) x octets par facteur : des centaines de Mo pour un 7B au rang 16."
            .to_owned(),
    );
    lines.join("\n") + "\n"
}

/// Shared by the CLI and the `surgeon_diff` MCP tool.
pub fn core_diff(args: &DiffArgs, deps: &AppDeps, sink: &mut dyn ProgressSink) -> CoreResult {
    if args.include_embeddings {
        let budget =
            hardware::evaluate_budget(&args.base, deps.probe.as_ref(), &BudgetPolicy::default())?;
        if let Verdict::Insufficient { reason } = budget.verdict {
            return Err(Outcome::fail(
                exit::GENERAL,
                format!("--include-embeddings : ressources insuffisantes : {reason}"),
            )
            .with_kind("insufficient_resources"));
        }
    }
    let report = {
        let mut obs = SinkObserver {
            sink: &mut *sink,
            cancel: &deps.interrupted,
        };
        run_diff(args, &mut obs, &deps.interrupted)?
    };
    for w in &report.warnings {
        sink.event(ProgressEvent::Warn(w.clone()));
    }
    let mut outcome = Outcome::ok(json!({ "surgery": report }), human(&report));
    outcome.warnings = report.warnings;
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_known_dates() {
        assert_eq!(rfc3339_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339_utc(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(rfc3339_utc(1_700_000_000), "2023-11-14T22:13:20Z");
    }
}
