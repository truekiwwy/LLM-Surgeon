//! `surgeon lora merge`: `W <- W + scale * (alpha / r) * B A` in place, all-or-nothing.
//!
//! Pipeline, shared by [`run_merge`] (library) and [`core_merge`] (CLI): `prepare` (adapter,
//! config, key pairing, shape and dtype checks against the model: nothing is written) ->
//! backups (CLI only) -> `execute` (write, flush, commit; any error, NaN/Inf or cancellation
//! rolls back). The kernel is `surgery::lowrank::apply_lowrank_rows` with `U = eff * B`,
//! `V = A`. Dry-run validates everything (including that the adapter values are finite) but
//! does not compute the merge, so it cannot predict an overflow of the target dtype.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;
use serde_json::json;

use super::cli_args::LoraMergeArgs;
use super::config::{self, AdapterConfig};
use super::decode_vec;
use super::keys::{self, KeyPair};
use crate::app::{
    exit, AppDeps, CoreResult, Outcome, ProgressEvent, ProgressSink, RollbackOutcome,
};
use crate::engine::{DType, ReadOnlyModel};
use crate::patch::{BackupSet, RollbackHandle};
use crate::surgery::error::{Result, SurgeryError};
use crate::surgery::events::{SinkObserver, SurgeryEvent, SurgeryObserver};
use crate::surgery::lowrank::{apply_lowrank_rows, LowRankStats};
use crate::surgery::target::{resolve_shards, Target};

#[derive(Debug, Clone)]
pub struct MergeRequest {
    pub model_shards: Vec<PathBuf>,
    pub lora: PathBuf,
    pub alpha: Option<f64>,
    pub adapter_config: Option<PathBuf>,
    pub scale: f64,
    pub dry_run: bool,
    pub cancel: Arc<AtomicBool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TensorMergeReport {
    pub name: String,
    pub elements_modified: u64,
    pub max_abs_delta: f32,
}

/// Serialized under `data.lora` of the envelope.
#[derive(Debug, Clone, Serialize, Default)]
pub struct MergeReport {
    pub rank: usize,
    pub lora_alpha: f64,
    /// Effective multiplier: scale * lora_alpha / r (or / sqrt(r) with rsLoRA).
    pub effective_scale: f64,
    pub tensors: usize,
    pub max_abs_delta: f32,
    pub dry_run: bool,
    pub warnings: Vec<String>,
    /// Sorted by tensor name; the deltas are zero in a dry-run.
    pub per_tensor: Vec<TensorMergeReport>,
}

/// One target weight with its factors, validated against the model.
struct Item {
    pair: KeyPair,
    rows: usize,
    cols: usize,
    dtype: DType,
}

struct Prepared {
    adapter: ReadOnlyModel,
    target: Target,
    items: Vec<Item>,
    rank: usize,
    alpha: f64,
    effective_scale: f64,
    warnings: Vec<String>,
}

fn usage_list(title: &str, problems: &[String]) -> SurgeryError {
    SurgeryError::Usage(format!(
        "{title} ({}) :\n  - {}",
        problems.len(),
        problems.join("\n  - ")
    ))
}

fn prepare(req: &MergeRequest) -> Result<Prepared> {
    if !req.scale.is_finite() {
        return Err(SurgeryError::Usage(format!(
            "--scale doit être fini, reçu {}",
            req.scale
        )));
    }
    if req.alpha.is_some_and(|a| !a.is_finite()) {
        return Err(SurgeryError::Usage("--alpha doit être fini".into()));
    }
    let cfg = match config::locate(&req.lora, req.adapter_config.as_deref()) {
        Some(path) => Some(AdapterConfig::load(&path)?),
        None => None,
    };
    if let Some(cfg) = &cfg {
        cfg.check_supported()?;
    }
    let alpha = req
        .alpha
        .or(cfg.as_ref().and_then(|c| c.lora_alpha))
        .ok_or_else(|| {
            SurgeryError::Usage(
                "lora_alpha introuvable : passez --alpha <valeur>, ou placez adapter_config.json \
                 à côté de --lora (ou donnez --adapter-config)"
                    .into(),
            )
        })?;

    let adapter = ReadOnlyModel::open(&req.lora).map_err(|e| {
        SurgeryError::Usage(format!("{} : adapter illisible ({e})", req.lora.display()))
    })?;
    let names: Vec<&str> = adapter
        .index()
        .tensors
        .iter()
        .map(|t| t.name.as_str())
        .collect();
    let mut refused: Vec<String> = Vec::new();
    for name in &names {
        if keys::is_embedding_key(name) {
            refused.push(format!("{name} : LoRA d'embedding non supporté"));
        } else if keys::is_dora_key(name) {
            refused.push(format!("{name} : DoRA non supporté"));
        }
    }
    if !refused.is_empty() {
        return Err(usage_list("adapter non supporté", &refused));
    }
    let pairs = keys::pair_keys(names.iter().copied())
        .map_err(|orphans| usage_list("clés d'adapter sans correspondance", &orphans))?;
    if pairs.is_empty() {
        return Err(SurgeryError::Usage(
            "adapter vide : aucune paire lora_A/lora_B".into(),
        ));
    }

    let target = Target::open(&req.model_shards)?;
    let mut problems = Vec::new();
    let mut items = Vec::new();
    let mut rank = None;
    for pair in pairs {
        match check_pair(&pair, &adapter, &target, &mut rank) {
            Ok((rows, cols, dtype)) => items.push(Item {
                pair,
                rows,
                cols,
                dtype,
            }),
            Err(why) => problems.push(format!("{} : {why}", pair.target)),
        }
    }
    if !problems.is_empty() {
        return Err(usage_list("adapter incompatible avec le modèle", &problems));
    }
    let rank = rank.unwrap_or(0);

    let use_rslora = cfg.as_ref().is_some_and(|c| c.use_rslora);
    let divisor = match use_rslora {
        true => (rank as f64).sqrt(),
        false => rank as f64,
    };
    let effective_scale = req.scale * alpha / divisor;
    if !(effective_scale as f32).is_finite() {
        return Err(SurgeryError::Usage(format!(
            "échelle effective non représentable : {effective_scale}"
        )));
    }
    let mut warnings = Vec::new();
    if let Some(cfg) = &cfg {
        if cfg.has_modules_to_save() {
            warnings.push(
                "modules_to_save ignoré : seuls les facteurs lora_A/lora_B sont fusionnés".into(),
            );
        }
        if cfg.r.is_some_and(|r| r != rank) {
            warnings.push(format!(
                "adapter_config.json annonce r={:?}, les matrices A ont {rank} lignes (valeur utilisée)",
                cfg.r
            ));
        }
    }
    Ok(Prepared {
        adapter,
        target,
        items,
        rank,
        alpha,
        effective_scale,
        warnings,
    })
}

/// Shapes `A [r, n]`, `B [m, r]`, target `[m, n]`, one rank for the whole adapter, supported
/// dtypes, finite values. Returns `(m, n, target dtype)` or the reason.
fn check_pair(
    pair: &KeyPair,
    adapter: &ReadOnlyModel,
    target: &Target,
    rank: &mut Option<usize>,
) -> std::result::Result<(usize, usize, DType), String> {
    let (a, b) = match (adapter.tensor(&pair.a), adapter.tensor(&pair.b)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) | (_, Err(e)) => return Err(e.to_string()),
    };
    let info = target
        .info(&pair.target)
        .ok_or("absent du modèle (aucun tenseur de ce nom)")?;
    let (&[r, n], &[m, rb], &[tm, tn]) = (&a.info.shape[..], &b.info.shape[..], &info.shape[..])
    else {
        return Err(format!(
            "formes 2D attendues (A {:?}, B {:?}, modèle {:?})",
            a.info.shape, b.info.shape, info.shape
        ));
    };
    if r != rb || r == 0 {
        return Err(format!("rangs incohérents (A [{r}, {n}], B [{m}, {rb}])"));
    }
    let first = *rank.get_or_insert(r);
    if first != r {
        return Err(format!(
            "rang {r} différent de {first} : rangs par module non supportés"
        ));
    }
    if (tm, tn) != (m, n) {
        return Err(format!(
            "forme {tm}x{tn} dans le modèle, B A donne {m}x{n} (A [{r}, {n}], B [{m}, {r}])"
        ));
    }
    for (what, dtype) in [
        ("A", &a.info.dtype),
        ("B", &b.info.dtype),
        ("modèle", &info.dtype),
    ] {
        if dtype.size_of().is_none() {
            return Err(format!(
                "dtype {dtype} non supporté pour {what} (BF16, F16, F32)"
            ));
        }
    }
    for (what, view) in [("A", &a), ("B", &b)] {
        let values = decode_vec(&view.info.dtype, view.bytes).map_err(|e| e.to_string())?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err(format!("{what} contient une valeur non finie"));
        }
    }
    Ok((m, n, info.dtype.clone()))
}

fn cancelled(req: &MergeRequest) -> bool {
    req.cancel.load(Ordering::SeqCst)
}

/// Any error, NaN/Inf or cancellation calls `guard.rollback()`; success calls `commit()`.
/// Errors raised by the checks happen before the first write and leave `guard` untouched.
pub fn run_merge(
    req: &MergeRequest,
    guard: &mut dyn RollbackHandle,
    obs: &mut dyn SurgeryObserver,
) -> Result<MergeReport> {
    let prepared = prepare(req)?;
    execute(req, prepared, guard, obs)
}

fn execute(
    req: &MergeRequest,
    prepared: Prepared,
    guard: &mut dyn RollbackHandle,
    obs: &mut dyn SurgeryObserver,
) -> Result<MergeReport> {
    let Prepared {
        adapter,
        mut target,
        items,
        rank,
        alpha,
        effective_scale,
        warnings,
    } = prepared;
    let mut report = MergeReport {
        rank,
        lora_alpha: alpha,
        effective_scale,
        tensors: items.len(),
        dry_run: req.dry_run,
        warnings,
        ..Default::default()
    };
    if req.dry_run {
        for (i, item) in items.iter().enumerate() {
            report
                .per_tensor
                .push(tensor_report(item, LowRankStats::default()));
            if progress(req, obs, i, &items).is_err() {
                return Err(SurgeryError::Cancelled);
            }
        }
        return Ok(report);
    }

    let written = write_all(
        req,
        &adapter,
        &mut target,
        &items,
        effective_scale as f32,
        obs,
    )
    .and_then(|per_tensor| {
        target.flush()?;
        Ok(per_tensor)
    });
    // The mappings are not flushed by drop, but a restore must land after every write
    // handle is gone.
    drop(target);
    match written {
        Ok(per_tensor) => {
            guard.commit();
            report.max_abs_delta = per_tensor
                .iter()
                .map(|t| t.max_abs_delta)
                .fold(0.0, f32::max);
            report.per_tensor = per_tensor;
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

fn tensor_report(item: &Item, stats: LowRankStats) -> TensorMergeReport {
    TensorMergeReport {
        name: item.pair.target.clone(),
        elements_modified: stats.elements_modified,
        max_abs_delta: stats.max_abs_delta,
    }
}

/// Emits the per-tensor event; `Err` when the observer or the flag asks to stop.
fn progress(
    req: &MergeRequest,
    obs: &mut dyn SurgeryObserver,
    index: usize,
    items: &[Item],
) -> std::result::Result<(), ()> {
    let ev = SurgeryEvent::Tensor {
        done: index + 1,
        total: items.len(),
        name: items[index].pair.target.clone(),
    };
    match obs.event(ev).is_break() || cancelled(req) {
        true => Err(()),
        false => Ok(()),
    }
}

fn write_all(
    req: &MergeRequest,
    adapter: &ReadOnlyModel,
    target: &mut Target,
    items: &[Item],
    eff: f32,
    obs: &mut dyn SurgeryObserver,
) -> Result<Vec<TensorMergeReport>> {
    let mut out = Vec::with_capacity(items.len());
    for (i, item) in items.iter().enumerate() {
        if cancelled(req) {
            return Err(SurgeryError::Cancelled);
        }
        let a = adapter.tensor(&item.pair.a)?;
        let b = adapter.tensor(&item.pair.b)?;
        let a_f32 = decode_vec(&a.info.dtype, a.bytes)?;
        // The scale is folded into U so the kernel only sees `U_scaled V`.
        let u_scaled: Vec<f32> = decode_vec(&b.info.dtype, b.bytes)?
            .into_iter()
            .map(|x| x * eff)
            .collect();
        let rank = a.info.shape[0];
        let bytes = target.tensor_bytes_mut(&item.pair.target)?;
        let stats = apply_lowrank_rows(
            bytes,
            item.dtype.clone(),
            item.rows,
            item.cols,
            &u_scaled,
            &a_f32,
            rank,
            &req.cancel,
        )?;
        if stats.nan_or_inf_produced > 0 {
            return Err(SurgeryError::Failed(format!(
                "{} : {} poids deviennent NaN ou Inf (dépassement de {}), fusion annulée",
                item.pair.target, stats.nan_or_inf_produced, item.dtype
            )));
        }
        out.push(tensor_report(item, stats));
        progress(req, obs, i, items).map_err(|()| SurgeryError::Cancelled)?;
    }
    Ok(out)
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
/// (`--no-backup`), so `core_merge` can tell the user what became of the files.
struct MergeGuard {
    backups: Option<BackupSet>,
    state: GuardState,
}

impl RollbackHandle for MergeGuard {
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

/// CLI and MCP entry: backups of the touched shards (mandatory unless `--no-backup`), merge,
/// `data.lora` and `data.backups` in the envelope. Backups are kept after success.
pub fn core_merge(args: &LoraMergeArgs, deps: &AppDeps, sink: &mut dyn ProgressSink) -> CoreResult {
    let req = MergeRequest {
        model_shards: resolve_shards(&args.model)?,
        lora: args.lora.clone(),
        alpha: args.alpha,
        adapter_config: args.adapter_config.clone(),
        scale: args.scale,
        dry_run: args.dry_run,
        cancel: Arc::clone(&deps.interrupted),
    };
    let prepared = prepare(&req)?;
    for w in &prepared.warnings {
        sink.event(ProgressEvent::Warn(w.clone()));
    }

    let mut guard = MergeGuard {
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
                .shards_holding(prepared.items.iter().map(|i| i.pair.target.as_str()));
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

fn outcome(report: &MergeReport, backups: &[PathBuf]) -> Outcome {
    let human = match report.dry_run {
        true => format!(
            "dry-run : {} tenseur(s) vérifiés (rang {}, échelle effective {}), aucun octet écrit\n",
            report.tensors, report.rank, report.effective_scale
        ),
        false => format!(
            "{} tenseur(s) fusionnés (rang {}, échelle effective {}), max |delta| {:.6}\n",
            report.tensors, report.rank, report.effective_scale, report.max_abs_delta
        ),
    };
    let mut outcome = Outcome::ok(
        json!({ "lora": report, "backups": backups, "warnings": report.warnings }),
        human,
    );
    outcome.warnings = report.warnings.clone();
    outcome
}
