//! `PatchBackend` over a GGUF file, and the "no bankruptcy" orchestration of an in-place
//! patch of a file of tens of GB:
//!
//! 1. preflight, nothing written: split name, crash marker, parse, every target resolved,
//!    typed (`unsupported_quant_type`), whole blocks, vector lengths, no `project_out`;
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

use super::mapping::{self, Layout, MappingError};
use super::parse::{self, GgufError, GgufFile};
use super::quant::Requant;
use super::verify::{self, Fidelity, RowError, TensorFidelity, Written};
use crate::kernel::{self, ChunkFailure, ChunkedPatcher, IoConfig, IoCounters, OsHints, RowDeltas};
use crate::patch::{
    ApplyReport, Axis, BackupGuard, LayerSelection, OpStats, OperationKind, PatchBackend,
    PatchError, Projection, RollbackHandle, ValidatedPatch, VectorTensor,
};

/// Suffix of the crash marker next to the model.
pub const MARKER_SUFFIX: &str = ".surgeon-inprogress";
/// Kept free on the backup volume beyond the copy itself.
pub const SPACE_MARGIN: u64 = 256 << 20;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufOptions {
    pub requant: Requant,
    pub verify: bool,
    pub force_no_backup: bool,
    pub io: IoConfig,
    pub backup_dir: Option<PathBuf>,
}

impl Default for GgufOptions {
    fn default() -> Self {
        Self {
            requant: Requant::KeepScales,
            verify: true,
            force_no_backup: false,
            io: IoConfig::default(),
            backup_dir: None,
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
    cancel: Arc<AtomicBool>,
    header_sha256: [u8; 32],
    written: Vec<Written>,
    tensors: Vec<TensorFidelity>,
    counters: IoCounters,
    verified: bool,
}

const INTERRUPTED: &str = "interrompu (Ctrl-C ou annulation)";

fn backend_err(e: impl std::fmt::Display) -> PatchError {
    PatchError::Backend(e.to_string())
}

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

impl PatchBackend for GgufBackend {
    fn resolve_targets(
        &self,
        layers: &LayerSelection,
        projections: &[Projection],
    ) -> crate::patch::Result<Vec<String>> {
        Ok(mapping::mlp_tensors(&self.file, layers, projections)
            .into_iter()
            .map(|(_, _, t)| t.name.clone())
            .collect())
    }

    fn load_vector(&self, file: &Path, tensor: Option<&str>) -> crate::patch::Result<Vec<f32>> {
        crate::engine::load_direction_vector(file, tensor).map_err(backend_err)
    }

    fn shift_tensor(
        &mut self,
        tensor_name: &str,
        direction: &[f32],
        alpha: f32,
        op: OperationKind,
        axis: Axis,
    ) -> crate::patch::Result<OpStats> {
        if self.cancel.load(Ordering::SeqCst) {
            return Err(backend_err(INTERRUPTED));
        }
        let sign = match op {
            OperationKind::Subtract => -1.0f32,
            OperationKind::Add => 1.0,
            OperationKind::ProjectOut => {
                return Err(backend_err("project_out non supporté sur un GGUF"));
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
        let ggml_type = self
            .file
            .tensor(tensor_name)
            .map(|t| t.ggml_type.name())
            .unwrap_or_default();
        let row_bytes = layout.row_bytes();
        let mode = self.requant;
        let mut fidelity = Fidelity::default();
        let mut hasher = Sha256::new();

        if alpha == 0.0 {
            // Strict no-op, as in the engine: nothing is re-rounded.
            fidelity.elements = (layout.rows * layout.cols) as u64;
            fidelity.unchanged = fidelity.elements;
        } else {
            match &mut self.map {
                Mapping::Write(map) => {
                    let cancel = Arc::clone(&self.cancel);
                    let counters = ChunkedPatcher::new(self.io)
                        .run(
                            map,
                            range.clone(),
                            row_bytes,
                            &mut OsHints,
                            &|| cancel.load(Ordering::SeqCst),
                            |chunk, first_row| {
                                // Computed and checked in a copy: a failing chunk is never
                                // written.
                                let mut next = chunk.to_vec();
                                let f =
                                    patch_chunk(&mut next, first_row, &layout, mode, &deltas, axis)
                                        .map_err(backend_err)?;
                                if f.nan_or_inf > 0 {
                                    return Err(backend_err(format!(
                                        "{} valeurs NaN/Inf produites",
                                        f.nan_or_inf
                                    )));
                                }
                                chunk.copy_from_slice(&next);
                                hasher.update(&next);
                                fidelity.merge(&f);
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
                        name: tensor_name.to_owned(),
                        range: range.clone(),
                        layout,
                        sha256: hasher.finalize().into(),
                    });
                }
                Mapping::Read(map) => {
                    let bytes = map
                        .get(range.clone())
                        .ok_or_else(|| backend_err("plage hors du fichier"))?;
                    let chunk_len = (self.io.chunk_bytes / row_bytes).max(1) * row_bytes;
                    for (i, chunk) in bytes.chunks(chunk_len).enumerate() {
                        if self.cancel.load(Ordering::SeqCst) {
                            return Err(backend_err(INTERRUPTED));
                        }
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
                }
            }
        }
        self.tensors.push(TensorFidelity::new(
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

    // project_out: the trait's default refusal (apply_gguf rejects it before any backup).

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
    if let Some(op) = patch
        .operations
        .iter()
        .find(|op| op.kind == OperationKind::ProjectOut)
    {
        return Err(GgufApplyError::Usage(format!(
            "opération {} : project_out n'est pas supporté sur un GGUF (patchez le SafeTensors)",
            op.id
        )));
    }

    let file = File::open(model).map_err(|e| GgufApplyError::io(model, &e))?;
    let read = kernel::map_read(&file).map_err(|e| GgufApplyError::io(model, &e))?;
    let gguf = parse::parse(&read).map_err(|source| GgufApplyError::Format {
        path: model.to_path_buf(),
        source,
    })?;
    preflight(&gguf, patch)?;

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

/// Every check that does not need to write: targets typed and laid out, vector lengths.
fn preflight(gguf: &GgufFile, patch: &ValidatedPatch) -> Result<(), GgufApplyError> {
    for op in &patch.operations {
        let targets = mapping::mlp_tensors(gguf, &op.layers, &op.projections);
        let vector = crate::engine::load_direction_vector(
            &op.vector_file,
            op.vector_tensor.as_ref().map(VectorTensor::direction),
        )
        .map_err(|e| GgufApplyError::Patch(backend_err(e)))?;
        for (_, _, t) in targets {
            let layout = mapping::layout(t)?;
            let expected = match op.axis {
                Axis::Rows => layout.cols,
                Axis::Cols => layout.rows,
            };
            if vector.len() != expected {
                return Err(GgufApplyError::Usage(format!(
                    "opération {} : {} attend un vecteur de {expected} (axe {:?}), {} reçu",
                    op.id,
                    t.name,
                    op.axis,
                    vector.len()
                )));
            }
        }
    }
    Ok(())
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
