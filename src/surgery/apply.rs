//! `surgeon apply --patch x.surgery`: checks a `.surgery` against the target model, then
//! applies it in place (`W <- W + scale * delta`) with all-or-nothing semantics.
//!
//! Pipeline, shared by [`run_apply`] (library) and [`try_apply`] (CLI):
//! `prepare` (open patch and shards, presence/shape/dtype checks, base hashes: nothing is
//! written) -> backups (CLI only) -> `execute` (write, flush, commit; any error, NaN/Inf or
//! cancellation rolls back).

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};

use super::cli_args::VerifyBase;
use super::error::{Result, SurgeryError};
use super::events::{SinkObserver, SurgeryEvent, SurgeryObserver};
use super::format::{FormatError, TensorEntry, TensorKind};
use super::lowrank::{apply_full_delta, apply_lowrank_rows};
use super::reader::{is_surgery_file, SurgeryReader};
use super::target::{resolve_shards, Target};
use crate::app::{exit, AppDeps, Outcome, ProgressEvent, ProgressSink, RollbackOutcome};
use crate::cli::ApplyArgs;
use crate::engine::DType;
use crate::patch::{BackupSet, RollbackHandle};

#[derive(Debug, Clone)]
pub struct SurgeryApplyRequest {
    /// Shards of the target model (one file for a single SafeTensors).
    pub model_shards: Vec<PathBuf>,
    pub patch: PathBuf,
    pub scale: f64,
    pub verify_base: VerifyBase,
    pub dry_run: bool,
    /// Raised by Ctrl-C or an MCP cancellation (`AppDeps::interrupted`).
    pub cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorApplyReport {
    pub name: String,
    pub kind: String,
    pub rank: usize,
    pub elements_modified: u64,
    pub max_abs_delta: f32,
    /// `None` with `--verify-base off`.
    pub base_hash_ok: Option<bool>,
}

/// Serialized under `data.surgery` of the envelope.
#[derive(Debug, Clone, Serialize, Default)]
pub struct SurgeryApplyReport {
    pub patch: String,
    pub scale: f64,
    pub dry_run: bool,
    pub tensors: Vec<TensorApplyReport>,
    pub warnings: Vec<String>,
}

/// One patch tensor, resolved against the target.
struct Item {
    entry: TensorEntry,
    rows: usize,
    cols: usize,
    dtype: DType,
    hash_ok: Option<bool>,
}

struct Prepared {
    reader: SurgeryReader,
    target: Target,
    items: Vec<Item>,
    warnings: Vec<String>,
}

fn kind_name(kind: TensorKind) -> &'static str {
    match kind {
        TensorKind::Lowrank => "lowrank",
        TensorKind::FullDelta => "full_delta",
    }
}

/// Verifies the patch (global hash first), checks every tensor (presence, shape, dtype,
/// base hash per `verify_base`), then writes. Any error, NaN/Inf or cancellation calls
/// `guard.rollback()` before returning; success calls `guard.commit()`. Errors raised by the
/// checks happen before the first write and leave `guard` untouched.
pub fn run_apply(
    req: &SurgeryApplyRequest,
    guard: &mut dyn RollbackHandle,
    obs: &mut dyn SurgeryObserver,
) -> Result<SurgeryApplyReport> {
    let prepared = prepare(req, obs)?;
    execute(req, prepared, guard, obs)
}

fn cancelled(req: &SurgeryApplyRequest) -> bool {
    req.cancel.load(Ordering::SeqCst)
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `(rows, cols)` of the row-major view the kernel works on. Low-rank tensors must be 2D;
/// a dense delta is any tensor (1D = one row, ND = leading dimensions folded into rows).
fn dims(entry: &TensorEntry) -> std::result::Result<(usize, usize), String> {
    let shape = &entry.shape;
    match (entry.kind, shape.as_slice()) {
        (TensorKind::Lowrank, &[m, n]) => Ok((m, n)),
        (TensorKind::Lowrank, _) => Err(format!("lowrank exige une forme 2D, reçu {shape:?}")),
        (TensorKind::FullDelta, &[]) => Err("forme vide".into()),
        (TensorKind::FullDelta, [rest @ .., last]) => Ok((rest.iter().product(), *last)),
    }
}

fn prepare(req: &SurgeryApplyRequest, obs: &mut dyn SurgeryObserver) -> Result<Prepared> {
    let scale = req.scale as f32;
    if !scale.is_finite() {
        return Err(SurgeryError::Usage(format!(
            "--scale doit être fini, reçu {}",
            req.scale
        )));
    }
    let reader = SurgeryReader::open(&req.patch)?;
    let mut target = Target::open(&req.model_shards)?;

    let mut seen = HashSet::new();
    let mut items = Vec::new();
    let mut problems = Vec::new();
    for entry in &reader.header().tensors {
        if !seen.insert(entry.name.as_str()) {
            return Err(FormatError::Invalid(format!("tenseur en double : {}", entry.name)).into());
        }
        let Some(info) = target.info(&entry.name) else {
            problems.push(format!("{} : absent du modèle", entry.name));
            continue;
        };
        if info.shape != entry.shape {
            problems.push(format!(
                "{} : forme {:?} dans le modèle, {:?} dans le patch",
                entry.name, info.shape, entry.shape
            ));
        } else if !info.dtype.to_string().eq_ignore_ascii_case(&entry.target_dtype) {
            problems.push(format!(
                "{} : dtype {} dans le modèle, {} dans le patch",
                entry.name, info.dtype, entry.target_dtype
            ));
        } else if info.dtype.size_of().is_none() {
            problems.push(format!("{} : dtype {} non supporté", entry.name, info.dtype));
        } else {
            match dims(entry) {
                Ok((rows, cols)) => items.push(Item {
                    entry: entry.clone(),
                    rows,
                    cols,
                    dtype: info.dtype.clone(),
                    hash_ok: None,
                }),
                Err(why) => problems.push(format!("{} : {why}", entry.name)),
            }
        }
    }
    if !problems.is_empty() {
        return Err(SurgeryError::Usage(format!(
            "patch incompatible avec le modèle ({} tenseur(s)) :\n  - {}",
            problems.len(),
            problems.join("\n  - ")
        )));
    }

    let mut warnings = Vec::new();
    if req.verify_base != VerifyBase::Off {
        let total = items.len();
        for (i, item) in items.iter_mut().enumerate() {
            let actual = sha256_hex(target.tensor_bytes_mut(&item.entry.name)?);
            let ok = actual.eq_ignore_ascii_case(&item.entry.base_sha256);
            item.hash_ok = Some(ok);
            if !ok {
                let msg = format!(
                    "{} : le tenseur actuel n'est pas la base du patch (sha256 {} au lieu de {})",
                    item.entry.name, actual, item.entry.base_sha256
                );
                match req.verify_base {
                    VerifyBase::Strict => return Err(SurgeryError::Usage(msg)),
                    _ => warnings.push(msg),
                }
            }
            let ev = SurgeryEvent::Hashing {
                done: i + 1,
                total,
                name: item.entry.name.clone(),
            };
            if obs.event(ev).is_break() || cancelled(req) {
                return Err(SurgeryError::Cancelled);
            }
        }
    }
    Ok(Prepared {
        reader,
        target,
        items,
        warnings,
    })
}

fn execute(
    req: &SurgeryApplyRequest,
    prepared: Prepared,
    guard: &mut dyn RollbackHandle,
    obs: &mut dyn SurgeryObserver,
) -> Result<SurgeryApplyReport> {
    let Prepared {
        mut reader,
        mut target,
        items,
        warnings,
    } = prepared;
    let mut report = SurgeryApplyReport {
        patch: req.patch.display().to_string(),
        scale: req.scale,
        dry_run: req.dry_run,
        tensors: Vec::with_capacity(items.len()),
        warnings,
    };
    if req.dry_run {
        for item in &items {
            let e = &item.entry;
            let ev = SurgeryEvent::Preview {
                name: e.name.clone(),
                kind: kind_name(e.kind).to_owned(),
                rank: e.rank,
                rel_error: e.rel_error,
            };
            if obs.event(ev).is_break() {
                return Err(SurgeryError::Cancelled);
            }
            report.tensors.push(tensor_report(item, Default::default()));
        }
        return Ok(report);
    }

    let written = write_all(
        req,
        &mut reader,
        &mut target,
        &items,
        obs,
        &mut report.tensors,
    )
    .and_then(|()| target.flush());
    // The mappings flush on drop: a rollback must restore after them, not before.
    drop(target);
    match written {
        Ok(()) => {
            guard.commit();
            Ok(report)
        }
        Err(e) => match guard.rollback() {
            Ok(()) => Err(e),
            Err(rb) => Err(SurgeryError::Failed(format!(
                "{e} ; rollback impossible : {rb}"
            ))),
        },
    }
}

fn tensor_report(item: &Item, stats: super::lowrank::LowRankStats) -> TensorApplyReport {
    TensorApplyReport {
        name: item.entry.name.clone(),
        kind: kind_name(item.entry.kind).to_owned(),
        rank: item.entry.rank,
        elements_modified: stats.elements_modified,
        max_abs_delta: stats.max_abs_delta,
        base_hash_ok: item.hash_ok,
    }
}

fn write_all(
    req: &SurgeryApplyRequest,
    reader: &mut SurgeryReader,
    target: &mut Target,
    items: &[Item],
    obs: &mut dyn SurgeryObserver,
    out: &mut Vec<TensorApplyReport>,
) -> Result<()> {
    let scale = req.scale as f32;
    for (i, item) in items.iter().enumerate() {
        if cancelled(req) {
            return Err(SurgeryError::Cancelled);
        }
        let e = &item.entry;
        let stats = match e.kind {
            TensorKind::Lowrank => {
                let f = reader.read_lowrank(e)?;
                let r = e.rank;
                if f.s.len() != r || f.u.len() != item.rows * r {
                    return Err(FormatError::Invalid(format!(
                        "{} : facteurs incohérents avec le rang {r}",
                        e.name
                    ))
                    .into());
                }
                // Scale and singular values folded into U, the kernel only sees U_scaled.
                let u_scaled: Vec<f32> = f
                    .u
                    .chunks_exact(r.max(1))
                    .flat_map(|row| row.iter().zip(&f.s).map(|(u, s)| u * (s * scale)))
                    .collect();
                let bytes = target.tensor_bytes_mut(&e.name)?;
                apply_lowrank_rows(
                    bytes,
                    item.dtype.clone(),
                    item.rows,
                    item.cols,
                    &u_scaled,
                    &f.v,
                    r,
                    &req.cancel,
                )?
            }
            TensorKind::FullDelta => {
                let delta = reader.read_full_delta(e)?;
                let bytes = target.tensor_bytes_mut(&e.name)?;
                apply_full_delta(
                    bytes,
                    item.dtype.clone(),
                    item.rows,
                    item.cols,
                    &delta,
                    scale,
                    &req.cancel,
                )?
            }
        };
        if stats.nan_or_inf_produced > 0 {
            return Err(SurgeryError::Failed(format!(
                "{} : {} poids deviennent NaN ou Inf (dépassement de {}), patch annulé",
                e.name, stats.nan_or_inf_produced, item.dtype
            )));
        }
        out.push(tensor_report(item, stats));
        let ev = SurgeryEvent::Tensor {
            done: i + 1,
            total: items.len(),
            name: e.name.clone(),
        };
        if obs.event(ev).is_break() {
            return Err(SurgeryError::Cancelled);
        }
    }
    Ok(())
}

/// What happened to the backups once `execute` returned.
enum GuardState {
    Untouched,
    Restored,
    Failed(String),
    /// An error left the model modified and no backup was taken.
    NoBackup,
}

/// Rollback handle of the CLI path: the backups when there are any, a recorder otherwise
/// (`--no-backup`), so `try_apply` can tell the user what became of the files.
struct ApplyGuard {
    backups: Option<BackupSet>,
    state: GuardState,
}

impl RollbackHandle for ApplyGuard {
    fn rollback(&mut self) -> crate::patch::Result<()> {
        let Some(set) = self.backups.as_mut() else {
            self.state = GuardState::NoBackup;
            return Ok(());
        };
        let result = set.rollback_all();
        self.state = match &result {
            Ok(()) => GuardState::Restored,
            Err(e) => GuardState::Failed(e.to_string()),
        };
        result
    }

    fn commit(&mut self) {
        if let Some(set) = self.backups.as_mut() {
            set.commit_all();
        }
    }
}

/// CLI entry called by `app::apply` before `load_patch`: `Ok(None)` when `--patch` is not
/// a `.surgery` file (detected by magic), so the `patch.json` path runs unchanged. Builds
/// the backups of the touched shards, runs the apply pipeline, and shapes the `Outcome`
/// (`data.surgery`, `data.backups`).
pub fn try_apply(
    args: &ApplyArgs,
    deps: &AppDeps,
    sink: &mut dyn ProgressSink,
) -> std::result::Result<Option<Outcome>, Outcome> {
    if !is_surgery_file(&args.patch) {
        return Ok(None);
    }
    cli_apply(args, deps, sink).map(Some)
}

fn cli_apply(
    args: &ApplyArgs,
    deps: &AppDeps,
    sink: &mut dyn ProgressSink,
) -> std::result::Result<Outcome, Outcome> {
    let req = SurgeryApplyRequest {
        model_shards: resolve_shards(&args.model)?,
        patch: args.patch.clone(),
        scale: args.surgery.scale,
        verify_base: args.surgery.verify_base,
        dry_run: args.dry_run,
        cancel: Arc::clone(&deps.interrupted),
    };
    let mut obs = SinkObserver {
        sink: &mut *sink,
        cancel: &deps.interrupted,
    };
    let prepared = prepare(&req, &mut obs)?;
    for w in &prepared.warnings {
        sink.event(ProgressEvent::Warn(w.clone()));
    }

    let mut guard = ApplyGuard {
        backups: None,
        state: GuardState::Untouched,
    };
    let mut backup_files = Vec::new();
    if !req.dry_run {
        if cancelled(&req) {
            return Err(SurgeryError::Cancelled.into());
        }
        if !args.no_backup {
            let touched = prepared
                .target
                .shards_holding(prepared.items.iter().map(|i| i.entry.name.as_str()));
            let set = BackupSet::create_many_in(&touched, args.backup_dir.as_deref())?;
            backup_files = set
                .guards()
                .iter()
                .map(|g| g.backup_path().to_path_buf())
                .collect();
            guard.backups = Some(set);
        }
        for path in &backup_files {
            sink.event(ProgressEvent::Backup { path: path.clone() });
        }
    }

    let started = Instant::now();
    let mut obs = SinkObserver {
        sink: &mut *sink,
        cancel: &deps.interrupted,
    };
    match execute(&req, prepared, &mut guard, &mut obs) {
        Ok(report) => {
            if !req.dry_run {
                sink.event(ProgressEvent::Commit {
                    seconds: started.elapsed().as_secs_f64(),
                });
            }
            Ok(outcome(&report, &backup_files))
        }
        Err(e) => match guard.state {
            GuardState::Failed(why) => {
                sink.event(ProgressEvent::Rollback(RollbackOutcome::Failed {
                    backups: backup_files.clone(),
                }));
                let files = backup_files
                    .iter()
                    .map(|p| p.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(Outcome::fail(
                    exit::CRITICAL,
                    format!("{e} ; restauration impossible ({why}), backups : {files}"),
                )
                .with_kind("critical"))
            }
            GuardState::Restored => {
                sink.event(ProgressEvent::Rollback(RollbackOutcome::Restored));
                Err(e.into())
            }
            GuardState::NoBackup => {
                sink.event(ProgressEvent::Rollback(RollbackOutcome::NoBackup));
                Err(e.into())
            }
            GuardState::Untouched => Err(e.into()),
        },
    }
}

/// Backups are kept after success: they are the manual undo.
fn outcome(report: &SurgeryApplyReport, backups: &[PathBuf]) -> Outcome {
    let modified: u64 = report.tensors.iter().map(|t| t.elements_modified).sum();
    let max_delta = report
        .tensors
        .iter()
        .map(|t| t.max_abs_delta)
        .fold(0.0, f32::max);
    let human = match report.dry_run {
        true => format!(
            "dry-run : {} tenseur(s) vérifiés, aucun octet écrit\n",
            report.tensors.len()
        ),
        false => format!(
            "{} tenseur(s) patchés, {modified} éléments modifiés, max |delta| {max_delta:.6}\n",
            report.tensors.len()
        ),
    };
    let mut outcome = Outcome::ok(
        json!({ "surgery": report, "backups": backups, "warnings": report.warnings }),
        human,
    );
    outcome.warnings = report.warnings.clone();
    outcome
}
