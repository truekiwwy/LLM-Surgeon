//! `PatchBackend` over a GGUF file, and the "no bankruptcy" orchestration of an in-place
//! patch of a file of tens of GB:
//!
//! 1. preflight, nothing written: split name, crash marker, parse, options, every target of
//!    every operation resolved, typed (`unsupported_quant_type`), whole blocks, vector and
//!    direction lengths, tied embeddings refused unless allowed; the work and the backup
//!    size are estimated ([`Preflight`]);
//! 2. free space, then a full verified `BackupGuard` copy (unless `--force-no-backup`);
//! 3. crash marker `<file>.surgeon-inprogress` (backup path, original SHA-256, time),
//!    created before the first write;
//! 4. `apply_patch`: per tensor, chunk by chunk (`ChunkedPatcher`), each chunk computed in
//!    a copy, checked (finite scales and values) and only then written;
//! 5. `flush`: synchronous flush, header region hash unchanged, then (`--verify`) a
//!    read-back of every patched tensor against the SHA-256 computed while writing;
//! 6. any error -> rollback through the backup; the marker is removed after the commit or
//!    the rollback, and kept when the file may be left modified.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use memmap2::{Mmap, MmapMut};
use rayon::prelude::*;
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::mapping::{self, Layout, MappingError, Orientation};
use super::parse::{self, GgufError, GgufFile};
use super::project::{self, Projector, INTERRUPTED};
use super::quant::Requant;
use super::verify::{self, Fidelity, RowError, TensorFidelity, Written};
use crate::engine::SubspaceProjection;
use crate::kernel::{self, ChunkFailure, ChunkedPatcher, IoConfig, IoCounters, OsHints, RowDeltas};
use crate::patch::{
    ApplyReport, Axis, BackupGuard, LayerSelection, OpStats, OperationKind, PatchBackend,
    PatchError, Projection, RollbackHandle, ValidatedPatch, VectorTensor,
};

/// Suffix of the crash marker next to the model.
pub const MARKER_SUFFIX: &str = ".surgeon-inprogress";
/// Kept free on the backup volume beyond the copy itself.
pub const SPACE_MARGIN: u64 = 256 << 20;

#[derive(Debug, Clone, PartialEq)]
pub struct GgufOptions {
    pub requant: Requant,
    pub verify: bool,
    pub force_no_backup: bool,
    pub io: IoConfig,
    pub backup_dir: Option<PathBuf>,
    /// Patch `token_embd` even when it is also the output head (no `output.weight`).
    pub allow_tied_embedding: bool,
    /// `project_out`: quantization passes measured, 1 to [`project::MAX_REFINE_PASSES`].
    pub refine_passes: usize,
    /// `project_out`: refinement stops at this `requant_residual_fraction`, in `(0, 1]`.
    pub residual_target: f64,
}

impl Default for GgufOptions {
    fn default() -> Self {
        Self {
            requant: Requant::KeepScales,
            verify: true,
            force_no_backup: false,
            io: IoConfig::default(),
            backup_dir: None,
            allow_tied_embedding: false,
            refine_passes: project::DEFAULT_REFINE_PASSES,
            residual_target: project::DEFAULT_RESIDUAL_TARGET,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GgufApplyError {
    #[error(transparent)]
    Patch(#[from] PatchError),
    #[error("{} : {source}", .path.display())]
    Format { path: PathBuf, source: GgufError },
    #[error(transparent)]
    Mapping(#[from] MappingError),
    #[error("{0}")]
    Usage(String),
    #[error("une application précédente a été interrompue : le marqueur {} existe. Le modèle peut \
             être partiellement modifié. Restaurez le backup indiqué dans ce fichier (copiez-le sur \
             le modèle et vérifiez son SHA-256), puis supprimez le marqueur.", .marker.display())]
    InProgress { marker: PathBuf },
    #[error("espace disque insuffisant pour le backup dans {} : {need} octets requis, {available} disponibles", .dir.display())]
    NoSpace {
        dir: PathBuf,
        need: u64,
        available: u64,
    },
    #[error("erreur d'E/S sur {} : {message}", .path.display())]
    Io { path: PathBuf, message: String },
}

impl GgufApplyError {
    /// `error.kind` of the JSON envelope (`None`: the patch error's own kind).
    pub fn kind(&self) -> Option<&'static str> {
        Some(match self {
            GgufApplyError::Patch(_) => return None,
            GgufApplyError::Format { .. } => "invalid_gguf",
            GgufApplyError::Mapping(MappingError::UnsupportedQuantType { .. }) => {
                "unsupported_quant_type"
            }
            GgufApplyError::Mapping(_) => "invalid_target",
            GgufApplyError::Usage(_) => "invalid_input",
            GgufApplyError::InProgress { .. } => "apply_in_progress",
            GgufApplyError::NoSpace { .. } => "insufficient_resources",
            GgufApplyError::Io { .. } => "model_error",
        })
    }

    fn io(path: &Path, e: &std::io::Error) -> Self {
        GgufApplyError::Io {
            path: path.to_path_buf(),
            message: e.to_string(),
        }
    }
}

pub fn marker_path(model: &Path) -> PathBuf {
    let mut name = model.as_os_str().to_owned();
    name.push(MARKER_SUFFIX);
    PathBuf::from(name)
}

/// Content of the crash marker.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct Marker {
    pub model: PathBuf,
    pub backup: Option<PathBuf>,
    pub original_sha256: Option<String>,
    pub started_unix: u64,
    pub pid: u32,
}

enum Mapping {
    Read(Mmap),
    Write(MmapMut),
}

impl Mapping {
    fn bytes(&self) -> &[u8] {
        match self {
            Mapping::Read(m) => m,
            Mapping::Write(m) => m,
        }
    }
}

/// `PatchBackend` over one GGUF file. Read mode computes every metric without writing
/// (`--dry-run`); write mode patches in place.
pub struct GgufBackend {
    path: PathBuf,
    file: GgufFile,
    map: Mapping,
    requant: Requant,
    io: IoConfig,
    verify: bool,
    refine_passes: usize,
    residual_target: f64,
    cancel: Arc<AtomicBool>,
    header_sha256: [u8; 32],
    written: Vec<Written>,
    tensors: Vec<TensorFidelity>,
    counters: IoCounters,
    verified: bool,
}

fn backend_err(e: impl std::fmt::Display) -> PatchError {
    PatchError::Backend(e.to_string())
}

pub const TIED_WARNING: &str = "embeddings liés (output.weight absent) : la tête de sortie \
    de llama.cpp lit token_embd.weight, elle est donc modifiée aussi (--allow-tied-embedding)";

impl GgufBackend {
    fn new(
        path: &Path,
        file: GgufFile,
        map: Mapping,
        opts: &GgufOptions,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        let data_start = file.data_start as usize;
        let header_sha256 = verify::sha256(&map.bytes()[..data_start.min(map.bytes().len())]);
        Self {
            path: path.to_path_buf(),
            file,
            map,
            requant: opts.requant,
            io: opts.io,
            verify: opts.verify,
            refine_passes: opts.refine_passes,
            residual_target: opts.residual_target,
            cancel,
            header_sha256,
            written: Vec::new(),
            tensors: Vec::new(),
            counters: IoCounters::default(),
            verified: false,
        }
    }

    pub fn file(&self) -> &GgufFile {
        &self.file
    }

    /// Fidelity of every tensor shifted so far, in order.
    pub fn tensors(&self) -> &[TensorFidelity] {
        &self.tensors
    }

    pub fn io_counters(&self) -> IoCounters {
        self.counters
    }

    fn target(&self, name: &str) -> Result<(std::ops::Range<usize>, Layout), PatchError> {
        let entry = self
            .file
            .tensor(name)
            .ok_or_else(|| backend_err(format!("tenseur introuvable : {name}")))?;
        let layout = mapping::layout(entry).map_err(backend_err)?;
        let start = usize::try_from(entry.start).map_err(backend_err)?;
        let len = layout.rows * layout.row_bytes();
        Ok((start..start + len, layout))
    }

    fn ggml_type(&self, name: &str) -> String {
        self.file
            .tensor(name)
            .map(|t| t.ggml_type.name())
            .unwrap_or_default()
    }

    fn cancelled(&self) -> Result<(), PatchError> {
        match self.cancel.load(Ordering::SeqCst) {
            true => Err(backend_err(INTERRUPTED)),
            false => Ok(()),
        }
    }

    /// Writes a tensor chunk by chunk (write mode only): `work(next, first_row)` turns a
    /// copy of the original rows into the new ones, checked before the chunk is written.
    fn write_tensor(
        &mut self,
        name: &str,
        range: std::ops::Range<usize>,
        layout: Layout,
        mut work: impl FnMut(&mut [u8], usize) -> Result<(), PatchError>,
    ) -> Result<(), PatchError> {
        let Mapping::Write(map) = &mut self.map else {
            return Ok(());
        };
        let cancel = Arc::clone(&self.cancel);
        let mut hasher = Sha256::new();
        let counters = ChunkedPatcher::new(self.io)
            .run(
                map,
                range.clone(),
                layout.row_bytes(),
                &mut OsHints,
                &|| cancel.load(Ordering::SeqCst),
                |chunk, first_row| {
                    // Computed and checked in a copy: a failing chunk is never written.
                    let mut next = chunk.to_vec();
                    work(&mut next, first_row)?;
                    chunk.copy_from_slice(&next);
                    hasher.update(&next);
                    Ok(())
                },
            )
            .map_err(|failure| match failure {
                ChunkFailure::Cancelled { .. } => backend_err(INTERRUPTED),
                ChunkFailure::InvalidRange => backend_err("plage hors du fichier"),
                ChunkFailure::Work(e) => e,
            })?;
        self.counters.merge(counters);
        self.written.push(Written {
            name: name.to_owned(),
            range,
            layout,
            sha256: hasher.finalize().into(),
        });
        Ok(())
    }

    /// Records a tensor's fidelity, with the tied-embeddings warning when it applies.
    fn record(&mut self, mut fidelity: TensorFidelity) {
        if fidelity.name == mapping::TOKEN_EMBD && mapping::tied_embeddings(&self.file) {
            fidelity.push_warning(TIED_WARNING.to_owned());
        }
        self.tensors.push(fidelity);
    }
}

/// Patches the rows of `chunk` (whole rows starting at `first_row`) in place. Rows run in
/// parallel; the fidelity is folded in row order.
fn patch_chunk(
    chunk: &mut [u8],
    first_row: usize,
    layout: &Layout,
    mode: Requant,
    deltas: &[f32],
    axis: Axis,
) -> Result<Fidelity, RowError> {
    let rows: Vec<Fidelity> = chunk
        .par_chunks_mut(layout.row_bytes())
        .enumerate()
        .map(|(k, row)| {
            let d = match axis {
                Axis::Rows => RowDeltas::Each(deltas),
                Axis::Cols => RowDeltas::Splat(deltas.get(first_row + k).copied().unwrap_or(0.0)),
            };
            verify::patch_row(layout, mode, row, d)
        })
        .collect::<Result<_, _>>()?;
    let mut total = Fidelity::default();
    for f in &rows {
        total.merge(f);
    }
    Ok(total)
}

/// Fidelity of a strict no-op (alpha 0): nothing is re-rounded.
fn unchanged(layout: &Layout) -> Fidelity {
    let elements = (layout.rows * layout.cols) as u64;
    Fidelity {
        elements,
        unchanged: elements,
        ..Fidelity::default()
    }
}

impl PatchBackend for GgufBackend {
    fn resolve_targets(
        &self,
        layers: &LayerSelection,
        projections: &[Projection],
    ) -> crate::patch::Result<Vec<String>> {
        Ok(mapping::targets(&self.file, layers, projections)
            .into_iter()
            .map(|(_, _, t)| t.name.clone())
            .collect())
    }

    fn load_vector(&self, file: &Path, tensor: Option<&str>) -> crate::patch::Result<Vec<f32>> {
        crate::engine::load_direction_vector(file, tensor).map_err(backend_err)
    }

    fn load_directions(
        &self,
        file: &Path,
        tensor: Option<&str>,
    ) -> crate::patch::Result<(Option<usize>, Vec<f32>)> {
        crate::engine::load_direction_matrix(file, tensor).map_err(backend_err)
    }

    fn shift_tensor(
        &mut self,
        tensor_name: &str,
        direction: &[f32],
        alpha: f32,
        op: OperationKind,
        axis: Axis,
    ) -> crate::patch::Result<OpStats> {
        self.cancelled()?;
        let sign = match op {
            OperationKind::Subtract => -1.0f32,
            OperationKind::Add => 1.0,
            OperationKind::ProjectOut => {
                return Err(backend_err("project_out passe par project_out_subspace"));
            }
        };
        let (range, layout) = self.target(tensor_name)?;
        let expected = match axis {
            Axis::Rows => layout.cols,
            Axis::Cols => layout.rows,
        };
        if direction.len() != expected {
            return Err(backend_err(format!(
                "vecteur de longueur {} pour {expected} attendus (axe {axis:?})",
                direction.len()
            )));
        }
        if !alpha.is_finite() || direction.iter().any(|v| !v.is_finite()) {
            return Err(backend_err("alpha et v doivent être finis"));
        }
        // Same precomputed delta as the SafeTensors engine: sign * (alpha * v).
        let deltas: Vec<f32> = direction.iter().map(|v| sign * (alpha * v)).collect();
        let row_bytes = layout.row_bytes();
        let mode = self.requant;
        let mut fidelity = Fidelity::default();

        if alpha == 0.0 {
            // Strict no-op, as in the engine: nothing is re-rounded.
            fidelity = unchanged(&layout);
        } else if let Mapping::Read(map) = &self.map {
            let bytes = map
                .get(range.clone())
                .ok_or_else(|| backend_err("plage hors du fichier"))?;
            let chunk_len = (self.io.chunk_bytes / row_bytes).max(1) * row_bytes;
            for (i, chunk) in bytes.chunks(chunk_len).enumerate() {
                self.cancelled()?;
                let mut next = chunk.to_vec();
                let f = patch_chunk(
                    &mut next,
                    i * chunk_len / row_bytes,
                    &layout,
                    mode,
                    &deltas,
                    axis,
                )
                .map_err(backend_err)?;
                fidelity.merge(&f);
            }
        } else {
            self.write_tensor(tensor_name, range, layout, |next, first_row| {
                let f = patch_chunk(next, first_row, &layout, mode, &deltas, axis)
                    .map_err(backend_err)?;
                if f.nan_or_inf > 0 {
                    return Err(backend_err(format!(
                        "{} valeurs NaN/Inf produites",
                        f.nan_or_inf
                    )));
                }
                fidelity.merge(&f);
                Ok(())
            })?;
        }
        let ggml_type = self.ggml_type(tensor_name);
        self.record(TensorFidelity::new(
            tensor_name,
            ggml_type,
            &layout,
            &fidelity,
        ));
        Ok(OpStats {
            max_abs_delta: fidelity.max_abs_delta,
            elements_modified: fidelity.modified(),
            nan_or_inf_produced: fidelity.nan_or_inf,
        })
    }

    fn project_out_tensor(
        &mut self,
        tensor_name: &str,
        a: &[f32],
        b: &[f32],
        alpha: f32,
    ) -> crate::patch::Result<OpStats> {
        self.project_out_subspace(tensor_name, a, b, 1, alpha)
    }

    /// `W ← W − α A (Bᵀ W)` (`token_embd`: each row `x ← x − α Σ (x·b_t) a_t`) on the
    /// dequantized weights, requantized; see [`super::project`].
    fn project_out_subspace(
        &mut self,
        tensor_name: &str,
        a: &[f32],
        b: &[f32],
        k: usize,
        alpha: f32,
    ) -> crate::patch::Result<OpStats> {
        self.cancelled()?;
        let (range, layout) = self.target(tensor_name)?;
        let local = match mapping::parse_name(tensor_name).map(|(_, p)| Orientation::of(p)) {
            Some(Orientation::Writer) => false,
            Some(Orientation::WriterLocal) => true,
            _ => {
                return Err(backend_err(format!(
                    "{tensor_name} : project_out ne vise que down_proj, o_proj et embed_tokens"
                )))
            }
        };
        let len = if local { layout.cols } else { layout.rows };
        if k == 0 || a.len() != k * len || b.len() != a.len() {
            return Err(backend_err(format!(
                "{tensor_name} : directions de {} et {} valeurs pour {k} × {len} attendues",
                a.len(),
                b.len()
            )));
        }
        if !(alpha.is_finite() && (0.0..=1.0).contains(&alpha))
            || a.iter().chain(b).any(|v| !v.is_finite())
        {
            return Err(backend_err("alpha dans [0, 1], a et b finis"));
        }
        let ggml_type = self.ggml_type(tensor_name);
        if alpha == 0.0 {
            let mut fidelity =
                TensorFidelity::new(tensor_name, ggml_type, &layout, &unchanged(&layout));
            fidelity.residual_fraction = Some(1.0);
            fidelity.requant_residual_fraction = Some(0.0);
            fidelity.refine_passes = Some(0);
            self.record(fidelity);
            return Ok(OpStats::default());
        }
        let projector = Projector {
            p: SubspaceProjection { a, b, alpha, k },
            local,
            layout,
            mode: self.requant,
        };
        let cancel = Arc::clone(&self.cancel);
        let bytes = self
            .map
            .bytes()
            .get(range.clone())
            .ok_or_else(|| backend_err("plage hors du fichier"))?;
        let plan = projector
            .plan(bytes, self.refine_passes, self.residual_target, &|| {
                cancel.load(Ordering::SeqCst)
            })
            .map_err(backend_err)?;
        if plan.fidelity.nan_or_inf > 0 {
            return Err(backend_err(format!(
                "{} valeurs NaN/Inf produites",
                plan.fidelity.nan_or_inf
            )));
        }
        self.write_tensor(tensor_name, range, layout, |next, first_row| {
            projector
                .write_rows(next, first_row, &plan)
                .map_err(backend_err)
        })?;

        let mut fidelity = TensorFidelity::new(tensor_name, ggml_type, &layout, &plan.fidelity);
        fidelity.residual_fraction = plan.residual_fraction;
        fidelity.requant_residual_fraction = plan.requant_residual_fraction;
        fidelity.refine_passes = Some(plan.passes());
        for (n, pair) in plan.history.windows(2).enumerate() {
            if let [Some(prev), Some(next)] = pair {
                if next >= prev {
                    fidelity.push_warning(format!(
                        "la passe de raffinement {} n'a pas diminué le résidu ({prev:.3e} -> \
                         {next:.3e}) ; la meilleure passe est celle écrite",
                        n + 2
                    ));
                }
            }
        }
        if let Some(r) = plan
            .requant_residual_fraction
            .filter(|r| *r > self.residual_target)
        {
            fidelity.push_warning(format!(
                "fraction résiduelle due à la re-quantification {r:.3} > cible {} après {} \
                 passe(s) : une partie de la direction effacée revient ; --refine-passes plus \
                 grand, ou patchez le SafeTensors puis re-quantifiez",
                self.residual_target,
                plan.passes()
            ));
        }
        self.record(fidelity);
        Ok(OpStats {
            max_abs_delta: plan.fidelity.max_abs_delta,
            elements_modified: plan.fidelity.modified(),
            nan_or_inf_produced: plan.fidelity.nan_or_inf,
        })
    }

    fn flush(&mut self) -> crate::patch::Result<()> {
        let Mapping::Write(map) = &self.map else {
            return Ok(());
        };
        map.flush()
            .map_err(|e| backend_err(format!("{} : {e}", self.path.display())))?;
        let data_start = self.file.data_start as usize;
        // The mapping itself first (cheap), then, with --verify, a second, independent
        // mapping of the file on disk.
        verify::verify_file(map, data_start, &self.header_sha256, &[]).map_err(backend_err)?;
        if self.verify {
            let file = File::open(&self.path)
                .map_err(|e| backend_err(format!("{} : {e}", self.path.display())))?;
            let reread = kernel::map_read(&file).map_err(backend_err)?;
            verify::verify_file(&reread, data_start, &self.header_sha256, &self.written)
                .map_err(|e| backend_err(format!("vérification après écriture : {e}")))?;
            self.verified = true;
        }
        Ok(())
    }
}

/// Runs the patch on the backend (`apply_patch`), with the rollback guard if any.
pub type RunPatch<'a> = dyn FnMut(&mut GgufBackend, Option<&mut dyn RollbackHandle>) -> crate::patch::Result<ApplyReport>
    + 'a;

/// Result of [`apply_gguf`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GgufRun {
    #[serde(skip)]
    pub report: ApplyReport,
    pub requant: Requant,
    pub version: u32,
    pub architecture: Option<String>,
    pub tensors: Vec<TensorFidelity>,
    /// SHA-256 of the header region `[0, data_start)`, identical before and after.
    pub header_sha256: String,
    pub data_start: u64,
    /// `true` once the read-back check passed (always `false` in dry run).
    pub verified: bool,
    pub backup: Option<PathBuf>,
    #[serde(skip)]
    pub io: IoCounters,
    pub preflight: Preflight,
}

/// Runs a validated patch on a GGUF file (see the module doc). `run` executes the patch on
/// the backend (`apply_patch`, possibly wrapped for progress events); it receives the
/// rollback guard (none in dry run or with `force_no_backup`).
pub fn apply_gguf(
    model: &Path,
    patch: &ValidatedPatch,
    opts: &GgufOptions,
    dry_run: bool,
    cancel: Arc<AtomicBool>,
    run: &mut RunPatch<'_>,
) -> Result<GgufRun, GgufApplyError> {
    let name = model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if parse::is_split_name(&name) {
        return Err(GgufApplyError::Format {
            path: model.to_path_buf(),
            source: GgufError::Split(name),
        });
    }
    let marker = marker_path(model);
    if marker.exists() {
        return Err(GgufApplyError::InProgress { marker });
    }
    let file = File::open(model).map_err(|e| GgufApplyError::io(model, &e))?;
    let read = kernel::map_read(&file).map_err(|e| GgufApplyError::io(model, &e))?;
    let gguf = parse::parse(&read).map_err(|source| GgufApplyError::Format {
        path: model.to_path_buf(),
        source,
    })?;
    let preflight = preflight(&gguf, &read, patch, opts)?;

    let architecture = gguf.architecture().map(str::to_owned);
    let (version, data_start) = (gguf.version, gguf.data_start);
    let finish = |backend: &GgufBackend, report: ApplyReport, backup: Option<PathBuf>| GgufRun {
        report,
        requant: opts.requant,
        version,
        architecture: architecture.clone(),
        tensors: backend.tensors.clone(),
        header_sha256: verify::hex(&backend.header_sha256),
        data_start,
        verified: backend.verified,
        backup,
        io: backend.counters,
        preflight: preflight.clone(),
    };

    if dry_run {
        let mut backend = GgufBackend::new(model, gguf, Mapping::Read(read), opts, cancel);
        let mut report = run(&mut backend, None).map_err(|e| match e {
            // Nothing was written, whatever apply_patch assumes without a guard.
            PatchError::ExecutionFailed { op_id, cause, .. } => PatchError::ExecutionFailed {
                op_id,
                cause: format!("{cause} (dry-run : rien n'a été écrit)"),
                rolled_back: false,
                left_modified: false,
            },
            other => other,
        })?;
        report.dry_run = true;
        return Ok(finish(&backend, report, None));
    }
    drop(read);
    drop(file);

    let mut guard = match opts.force_no_backup {
        true => None,
        false => {
            check_space(model, opts.backup_dir.as_deref(), gguf.file_len)?;
            Some(BackupGuard::create(model, opts.backup_dir.as_deref())?)
        }
    };
    let backup = guard.as_ref().map(|g| g.backup_path().to_path_buf());
    write_marker(
        &marker,
        &Marker {
            model: model.to_path_buf(),
            backup: backup.clone(),
            original_sha256: guard.as_ref().map(|g| g.original_sha256().to_owned()),
            started_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            pid: std::process::id(),
        },
    )?;

    let opened = OpenOptions::new()
        .read(true)
        .write(true)
        .open(model)
        .and_then(|f| kernel::map_write(&f));
    let map = match opened {
        Ok(map) => map,
        Err(e) => {
            // Nothing written yet: the backup is not needed.
            let _ = fs::remove_file(&marker);
            return Err(GgufApplyError::io(model, &e));
        }
    };
    let mut backend = GgufBackend::new(model, gguf, Mapping::Write(map), opts, cancel);
    let result = run(
        &mut backend,
        guard.as_mut().map(|g| g as &mut dyn RollbackHandle),
    );
    let run = result.map(|report| finish(&backend, report, backup));
    drop(backend);
    let keep_marker = matches!(
        run,
        Err(PatchError::RollbackFailed { .. })
            | Err(PatchError::ExecutionFailed {
                left_modified: true,
                ..
            })
    );
    if !keep_marker {
        let _ = fs::remove_file(&marker);
    }
    Ok(run?)
}

/// `data.gguf.preflight`: what the run will do, measured before any write.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Preflight {
    /// Tensor patches planned (a tensor targeted by two operations counts twice).
    pub tensors: usize,
    pub target_bytes: u64,
    /// Bytes dequantized and requantized, every `project_out` pass counted (upper bound).
    pub work_bytes: u64,
    /// `work_bytes` over the throughput of a few rows of the first target, on every
    /// thread: an order of magnitude, not a promise. `None` when it could not be measured.
    pub estimated_seconds: Option<f64>,
    /// Size of the backup copy (the whole file); `None` with `--force-no-backup`.
    pub backup_bytes: Option<u64>,
}

/// Every check that does not need to write, over every operation before any vector is
/// read: options, targets typed and laid out (`unsupported_quant_type`), tied embeddings;
/// then vector and direction lengths. Also estimates the work.
fn preflight(
    gguf: &GgufFile,
    bytes: &[u8],
    patch: &ValidatedPatch,
    opts: &GgufOptions,
) -> Result<Preflight, GgufApplyError> {
    if !(1..=project::MAX_REFINE_PASSES).contains(&opts.refine_passes) {
        return Err(GgufApplyError::Usage(format!(
            "--refine-passes {} hors de [1, {}]",
            opts.refine_passes,
            project::MAX_REFINE_PASSES
        )));
    }
    if !(opts.residual_target > 0.0 && opts.residual_target <= 1.0) {
        return Err(GgufApplyError::Usage(format!(
            "--residual-target {} hors de ]0, 1]",
            opts.residual_target
        )));
    }
    let mut planned = Vec::new();
    for op in &patch.operations {
        for (_, projection, t) in mapping::targets(gguf, &op.layers, &op.projections) {
            planned.push((op, Orientation::of(projection), t, mapping::layout(t)?));
        }
    }
    let tied = mapping::tied_embeddings(gguf);
    if let Some((op, ..)) = planned
        .iter()
        .find(|(_, _, t, _)| t.name == mapping::TOKEN_EMBD && tied && !opts.allow_tied_embedding)
    {
        return Err(GgufApplyError::Usage(format!(
            "opération {} : token_embd.weight est aussi la tête de sortie de ce GGUF \
             (output.weight absent, embeddings liés comme dans Qwen2.5-0.5B) : le modifier \
             change aussi les logits de sortie. Refusé par défaut ; --allow-tied-embedding \
             pour l'accepter (avertissement dans le rapport), ou patchez le SafeTensors, où \
             surgeon détache d'abord lm_head",
            op.id
        )));
    }

    let mut work_bytes = 0u64;
    let mut vectors: Vec<(&str, usize)> = Vec::new();
    for (op, orientation, t, layout) in &planned {
        let project = op.kind == OperationKind::ProjectOut;
        // One direction length per operation, loaded once.
        let len = match vectors.iter().find(|(id, _)| *id == op.id) {
            Some((_, len)) => *len,
            None => {
                let tensor = op.vector_tensor.as_ref().map(VectorTensor::direction);
                let len = match project {
                    true => crate::engine::load_direction_matrix(&op.vector_file, tensor)
                        .map(|(k, v)| v.len() / k.unwrap_or(1).max(1)),
                    false => crate::engine::load_direction_vector(&op.vector_file, tensor)
                        .map(|v| v.len()),
                }
                .map_err(|e| GgufApplyError::Patch(backend_err(e)))?;
                vectors.push((&op.id, len));
                len
            }
        };
        let (expected, what) = match (project, orientation, op.axis) {
            (true, Orientation::WriterLocal, _) | (false, _, Axis::Rows) => {
                (layout.cols, "colonnes")
            }
            _ => (layout.rows, "lignes"),
        };
        if len != expected {
            return Err(GgufApplyError::Usage(format!(
                "opération {} : {} attend des directions de {expected} ({what}), {len} reçues",
                op.id, t.name
            )));
        }
        let size = (layout.rows * layout.row_bytes()) as u64;
        let passes = match (project, orientation) {
            (false, _) => 1,
            // R(W), the measuring passes, the write.
            (true, Orientation::Writer) => 2 + opts.refine_passes,
            (true, _) => 1 + opts.refine_passes,
        };
        work_bytes += size * passes as u64;
    }
    let target_bytes = planned
        .iter()
        .map(|(.., l)| (l.rows * l.row_bytes()) as u64)
        .sum();
    let estimated_seconds = planned
        .first()
        .and_then(|(_, _, t, layout)| throughput(bytes, t.start, layout))
        .map(|per_second| work_bytes as f64 / per_second);
    Ok(Preflight {
        tensors: planned.len(),
        target_bytes,
        work_bytes,
        estimated_seconds,
        backup_bytes: (!opts.force_no_backup).then_some(gguf.file_len),
    })
}

/// Bytes per second of patching up to 64 rows of a tensor in a copy, scaled to every
/// rayon thread.
fn throughput(bytes: &[u8], start: u64, layout: &Layout) -> Option<f64> {
    let start = usize::try_from(start).ok()?;
    let len = layout.rows.min(64) * layout.row_bytes();
    let mut sample = bytes.get(start..start.checked_add(len)?)?.to_vec();
    let clock = std::time::Instant::now();
    for row in sample.chunks_exact_mut(layout.row_bytes().max(1)) {
        verify::patch_row(layout, Requant::Recompute, row, RowDeltas::Splat(1e-3)).ok()?;
    }
    let seconds = clock.elapsed().as_secs_f64();
    (seconds > 0.0 && len > 0).then(|| len as f64 / seconds * rayon::current_num_threads() as f64)
}

fn check_space(
    model: &Path,
    backup_dir: Option<&Path>,
    file_len: u64,
) -> Result<(), GgufApplyError> {
    let dir = match backup_dir {
        Some(dir) => dir.to_path_buf(),
        None => model
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from(".")),
    };
    let need = file_len.saturating_add(SPACE_MARGIN);
    // A platform without the query falls back on the copy's own failure.
    match kernel::available_space(&dir) {
        Ok(available) if available < need => Err(GgufApplyError::NoSpace {
            dir,
            need,
            available,
        }),
        _ => Ok(()),
    }
}

/// Created with `create_new`: a concurrent apply cannot overwrite it.
fn write_marker(path: &Path, marker: &Marker) -> Result<(), GgufApplyError> {
    let json = serde_json::to_vec_pretty(marker).map_err(|e| GgufApplyError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    })?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::AlreadyExists => GgufApplyError::InProgress {
                marker: path.to_path_buf(),
            },
            _ => GgufApplyError::io(path, &e),
        })?;
    file.write_all(&json)
        .and_then(|()| file.sync_all())
        .map_err(|e| GgufApplyError::io(path, &e))
}
