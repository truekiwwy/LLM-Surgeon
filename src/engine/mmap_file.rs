use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use memmap2::{Mmap, MmapMut, MmapOptions};

use super::error::{EngineError, Result};
use super::index::{parse_index, ModelIndex, TensorInfo};
use super::project::{project_out_embedding, project_out_tensor};
use super::shift::{decode_f32, BroadcastAxis, ShiftOp, ShiftPlan, ShiftStats};
use crate::kernel::{ChunkFailure, ChunkedPatcher, IoConfig, IoCounters, KernelConfig, OsHints};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Clone, Copy)]
pub struct TensorView<'a> {
    pub info: &'a TensorInfo,
    pub bytes: &'a [u8],
}

/// Read-only mapping. Not `Clone`: duplicating it would mean remapping the file.
#[derive(Debug)]
pub struct ReadOnlyModel {
    mmap: Mmap,
    index: ModelIndex,
}

/// Header and weights parsed straight from the mapping, no copy of the weights.
pub fn read_index(path: &Path) -> Result<ModelIndex> {
    Ok(ReadOnlyModel::open(path)?.index)
}

fn open_checked(path: &Path, write: bool) -> Result<File> {
    let file = OpenOptions::new()
        .read(true)
        .write(write)
        .open(path)
        .map_err(|e| EngineError::io(path, &e))?;
    // Checked before mapping: mapping an empty file fails on some platforms.
    let len = file
        .metadata()
        .map_err(|e| EngineError::io(path, &e))?
        .len();
    if len < 8 {
        return Err(EngineError::FileTooSmall {
            path: path.to_path_buf(),
            len,
        });
    }
    Ok(file)
}

impl ReadOnlyModel {
    pub fn open(path: &Path) -> Result<Self> {
        let file = open_checked(path, false)?;
        // SAFETY: the mapping is only read. Like every mmap user, we assume no other process
        // truncates or rewrites the file while it is open; that is the CLI's usage contract.
        let mmap =
            unsafe { MmapOptions::new().map(&file) }.map_err(|e| EngineError::io(path, &e))?;
        let index = parse_index(path, &mmap)?;
        Ok(Self { mmap, index })
    }

    pub fn index(&self) -> &ModelIndex {
        &self.index
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let info = self.index.get(name)?;
        // In bounds: parse_index validated every range against this same mapping.
        let bytes = &self.mmap[info.byte_start..info.byte_end];
        Ok(TensorView { info, bytes })
    }

    /// The header's `__metadata__`, sorted; empty when absent. Re-parsed from the mapping
    /// (header only) so `ModelIndex` keeps its shape.
    pub fn metadata(&self) -> BTreeMap<String, String> {
        safetensors::SafeTensors::read_metadata(&self.mmap)
            .ok()
            .and_then(|(_, m)| m.metadata().clone())
            .map(|m| m.into_iter().collect())
            .unwrap_or_default()
    }
}

/// Read-write mapping for in-place edits. Only tensor ranges are ever handed out, so the
/// header and the bytes of untouched tensors are never written.
#[derive(Debug)]
pub struct WritableModel {
    path: PathBuf,
    mmap: MmapMut,
    index: ModelIndex,
    io: IoConfig,
    kernel: KernelConfig,
    counters: IoCounters,
    cancel: Option<Arc<AtomicBool>>,
}

impl WritableModel {
    pub fn open(path: &Path) -> Result<Self> {
        let file = open_checked(path, true)?;
        // SAFETY: same contract as ReadOnlyModel::open; in addition this process is the only
        // writer and only writes inside validated tensor ranges.
        let mmap =
            unsafe { MmapOptions::new().map_mut(&file) }.map_err(|e| EngineError::io(path, &e))?;
        let index = parse_index(path, &mmap)?;
        Ok(Self {
            path: path.to_path_buf(),
            mmap,
            index,
            io: IoConfig::default(),
            kernel: KernelConfig::from_env(),
            counters: IoCounters::default(),
            cancel: None,
        })
    }

    pub fn index(&self) -> &ModelIndex {
        &self.index
    }

    /// Chunking / prefetch of [`Self::shift_tensor`], and a flag checked between chunks
    /// (raised: the shift stops with an `EngineError::Io` of kind `Interrupted`, the tensor
    /// possibly half-shifted, so the caller must hold a backup).
    pub fn set_io(&mut self, io: IoConfig, cancel: Option<Arc<AtomicBool>>) {
        self.io = io;
        self.cancel = cancel;
    }

    /// Kernel level used by [`Self::shift_tensor`] (default: `SURGEON_KERNEL` or best).
    pub fn set_kernel(&mut self, kernel: KernelConfig) {
        self.kernel = kernel;
    }

    /// Chunks, bytes and hints of every shift since `open`.
    pub fn io_counters(&self) -> IoCounters {
        self.counters
    }

    pub fn tensor_bytes_mut(&mut self, name: &str) -> Result<&mut [u8]> {
        let info = self.index.get(name)?;
        Ok(&mut self.mmap[info.byte_start..info.byte_end])
    }

    pub fn shift_tensor(
        &mut self,
        name: &str,
        direction: &[f32],
        alpha: f32,
        op: ShiftOp,
        axis: BroadcastAxis,
    ) -> Result<ShiftStats> {
        let info = self.index.get(name)?;
        let range = info.byte_start..info.byte_end;
        let Some(plan) = ShiftPlan::new(
            range.len(),
            info.dtype.clone(),
            &info.shape,
            direction,
            alpha,
            op,
            axis,
        )?
        else {
            return Ok(ShiftStats::default());
        };
        let cancel = self.cancel.clone();
        let cancelled = || cancel.as_ref().is_some_and(|c| c.load(Ordering::SeqCst));
        let kernel = self.kernel;
        let mut stats = ShiftStats::default();
        let counters = ChunkedPatcher::new(self.io)
            .run(
                &mut self.mmap,
                range,
                plan.row_bytes(),
                &mut OsHints,
                &cancelled,
                |chunk, first_row| {
                    stats = stats.merge(plan.run(chunk, first_row, &kernel)?);
                    Ok(())
                },
            )
            .map_err(|failure| match failure {
                ChunkFailure::Cancelled { .. } => EngineError::io(
                    &self.path,
                    &std::io::Error::new(
                        std::io::ErrorKind::Interrupted,
                        "interrompu entre deux blocs (tenseur possiblement partiel)",
                    ),
                ),
                ChunkFailure::Work(e) => e,
                ChunkFailure::InvalidRange => EngineError::DataSizeMismatch {
                    path: self.path.clone(),
                },
            })?;
        self.counters.merge(counters);
        Ok(stats)
    }

    /// `W ← W − α a (bᵀ W)`, see [`project_out_tensor`].
    pub fn project_out_tensor(
        &mut self,
        name: &str,
        a: &[f32],
        b: &[f32],
        alpha: f32,
    ) -> Result<ShiftStats> {
        let info = self.index.get(name)?;
        let bytes = &mut self.mmap[info.byte_start..info.byte_end];
        project_out_tensor(bytes, info.dtype.clone(), &info.shape, a, b, alpha)
    }

    /// `W ← W − α (W b) aᵀ`, see [`project_out_embedding`].
    pub fn project_out_embedding(
        &mut self,
        name: &str,
        a: &[f32],
        b: &[f32],
        alpha: f32,
    ) -> Result<ShiftStats> {
        let info = self.index.get(name)?;
        let bytes = &mut self.mmap[info.byte_start..info.byte_end];
        project_out_embedding(bytes, info.dtype.clone(), &info.shape, a, b, alpha)
    }

    pub fn flush(&self) -> Result<()> {
        self.mmap
            .flush()
            .map_err(|e| EngineError::io(&self.path, &e))
    }
}

/// Loads a 1D direction from a `.safetensors` file (bf16/f16/f32, `tensor_name` required
/// when the file holds several tensors) or from a `.json` array of numbers.
pub fn load_direction_vector(path: &Path, tensor_name: Option<&str>) -> Result<Vec<f32>> {
    let invalid = |message: String| EngineError::InvalidDirectionSource {
        path: path.to_path_buf(),
        message,
    };
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("json") => {
            let raw = std::fs::read_to_string(path).map_err(|e| EngineError::io(path, &e))?;
            serde_json::from_str(&raw).map_err(|e| invalid(e.to_string()))
        }
        Some("safetensors") => {
            let model = ReadOnlyModel::open(path)?;
            let view = match (tensor_name, model.index.tensors.as_slice()) {
                (Some(name), _) => model.tensor(name)?,
                (None, [only]) => model.tensor(&only.name)?,
                (None, all) => {
                    return Err(invalid(format!(
                        "{} tenseurs dans le fichier, précisez lequel",
                        all.len()
                    )))
                }
            };
            if view.info.shape.len() != 1 {
                return Err(EngineError::InvalidShape {
                    shape: view.info.shape.clone(),
                    reason: "vecteur 1D attendu".to_owned(),
                });
            }
            decode_f32(view.bytes, &view.info.dtype)
        }
        _ => Err(invalid(
            "extension attendue : .safetensors ou .json".to_owned(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::index::DType;
    use crate::engine::test_support::{encode, raw_file, reference_shift, write_fake_safetensors};
    use std::fs;

    const GATE: &str = "model.layers.0.mlp.gate_proj.weight";
    const UP: &str = "model.layers.0.mlp.up_proj.weight";
    const DOWN: &str = "model.layers.0.mlp.down_proj.weight";
    const NORM: &str = "model.norm.weight";

    /// 4x8 bf16 weights covering zero, signed zero, negatives, subnormals, large values.
    fn bf16_weights() -> Vec<u8> {
        let mut bits: Vec<u16> = vec![
            0x0000, 0x8000, 0x0001, 0x8003, 0x007F, 0x0080, 0x7F7F,
            0xFF7F, // 0, -0, subnormals, min normal, +-max
        ];
        let values = [
            1.0f32, -1.0, 0.5, -0.25, 3.0e38, -3.0e38, 1e-38, 123.456, -9.75, 2.0, 1e4, -1e4,
            0.333, -0.333, 7.0, 64.5, 0.001, -0.001, 1.5, -1.5, 100.0, -100.0, 1e20, -1e20,
        ];
        bits.extend(values.iter().map(|&x| half::bf16::from_f32(x).to_bits()));
        bits.iter().flat_map(|b| b.to_le_bytes()).collect()
    }

    fn four_tensor_file(dir: &Path) -> PathBuf {
        let path = dir.join("model.safetensors");
        write_fake_safetensors(
            &path,
            &[
                (GATE, "BF16", vec![4, 8], bf16_weights()),
                (UP, "BF16", vec![4, 8], bf16_weights()),
                (DOWN, "BF16", vec![8, 4], bf16_weights()),
                (NORM, "F32", vec![8], encode(&DType::F32, &[1.0; 8])),
            ],
        );
        path
    }

    #[test]
    fn read_index_finds_exact_names_dtypes_shapes_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let path = four_tensor_file(dir.path());
        let header_len = u64::from_le_bytes(fs::read(&path).unwrap()[..8].try_into().unwrap());
        let index = read_index(&path).unwrap();
        assert_eq!(index.header_len as u64, header_len);
        let base = 8 + index.header_len;
        let got: Vec<_> = index
            .tensors
            .iter()
            .map(|t| {
                (
                    t.name.as_str(),
                    t.dtype.clone(),
                    t.shape.clone(),
                    t.byte_start - base,
                    t.byte_end - base,
                )
            })
            .collect();
        assert_eq!(
            got,
            vec![
                (GATE, DType::BF16, vec![4, 8], 0, 64),
                (UP, DType::BF16, vec![4, 8], 64, 128),
                (DOWN, DType::BF16, vec![8, 4], 128, 192),
                (NORM, DType::F32, vec![8], 192, 224),
            ]
        );
        let model = ReadOnlyModel::open(&path).unwrap();
        assert_eq!(model.tensor(UP).unwrap().bytes, bf16_weights().as_slice());
        assert!(matches!(
            model.tensor("nope"),
            Err(EngineError::TensorNotFound(_))
        ));
    }

    #[test]
    fn corrupt_files_give_distinct_errors() {
        let dir = tempfile::tempdir().unwrap();
        let check = |name: &str, bytes: Vec<u8>| {
            let p = dir.path().join(name);
            fs::write(&p, bytes).unwrap();
            read_index(&p).unwrap_err()
        };
        let ok_header = r#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let good = raw_file(ok_header, &[0; 8]);

        assert!(matches!(
            check("short", good[..5].to_vec()),
            EngineError::FileTooSmall { len: 5, .. }
        ));
        // Truncated inside the header: header_len points past the end of the file.
        assert!(matches!(
            check("cut", good[..20].to_vec()),
            EngineError::InvalidHeaderLength { .. }
        ));
        assert!(matches!(
            check("json", raw_file("{not json", &[])),
            EngineError::InvalidHeaderJson { .. }
        ));
        // Last offset beyond the data actually present.
        let oob = r#"{"t":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        assert!(matches!(
            check("oob", raw_file(oob, &[0; 8])),
            EngineError::DataSizeMismatch { .. }
        ));
        let overlap = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"b":{"dtype":"F32","shape":[2],"data_offsets":[4,12]}}"#;
        assert!(matches!(
            check("overlap", raw_file(overlap, &[0; 12])),
            EngineError::InvalidOffsets { .. }
        ));
        let bad_size = r#"{"t":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#;
        assert!(matches!(
            check("size", raw_file(bad_size, &[0; 8])),
            EngineError::TensorSizeMismatch { .. }
        ));
        assert!(matches!(
            read_index(&dir.path().join("missing")),
            Err(EngineError::Io { .. })
        ));
    }

    #[test]
    fn bit_exact_bf16_shift_through_file_leaves_everything_else_intact() {
        let dir = tempfile::tempdir().unwrap();
        let path = four_tensor_file(dir.path());
        let before = fs::read(&path).unwrap();
        let v = [1.0, -2.0, 0.0, 1e-39, 3.0, -0.5, 1e30, 0.125];

        let mut model = WritableModel::open(&path).unwrap();
        let info = model.index().get(UP).unwrap().clone();
        let stats = model
            .shift_tensor(UP, &v, 0.5, ShiftOp::Subtract, BroadcastAxis::Rows)
            .unwrap();
        model.flush().unwrap();
        drop(model);
        assert!(stats.elements_modified > 0);

        let after = fs::read(&path).unwrap();
        let expected = reference_shift(
            &DType::BF16,
            [4, 8],
            &before[info.byte_start..info.byte_end],
            &v,
            0.5,
            ShiftOp::Subtract,
            BroadcastAxis::Rows,
        );
        assert_eq!(&after[info.byte_start..info.byte_end], expected.as_slice());
        assert_eq!(
            after[..info.byte_start],
            before[..info.byte_start],
            "header or earlier tensor changed"
        );
        assert_eq!(
            after[info.byte_end..],
            before[info.byte_end..],
            "later tensor changed"
        );
        // Reopening parses the same index.
        assert_eq!(
            read_index(&path).unwrap().tensors,
            read_index(&path).unwrap().tensors
        );
    }

    #[test]
    fn rejected_shifts_leave_the_file_byte_identical() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.safetensors");
        write_fake_safetensors(
            &path,
            &[
                (
                    "w",
                    "BF16",
                    vec![2, 2],
                    encode(&DType::BF16, &[1.0, 2.0, 3.0, 4.0]),
                ),
                ("ints", "I32", vec![2, 2], vec![0; 16]),
            ],
        );
        let before = fs::read(&path).unwrap();
        let mut model = WritableModel::open(&path).unwrap();
        let (sub, rows) = (ShiftOp::Subtract, BroadcastAxis::Rows);
        assert!(matches!(
            model.shift_tensor("ints", &[1.0; 2], 1.0, sub, rows),
            Err(EngineError::UnsupportedDtype(_))
        ));
        assert!(matches!(
            model.shift_tensor("w", &[1.0; 3], 1.0, sub, rows),
            Err(EngineError::DirectionLengthMismatch { .. })
        ));
        assert!(matches!(
            model.shift_tensor("w", &[1.0; 2], f32::NAN, sub, rows),
            Err(EngineError::InvalidAlpha(_))
        ));
        assert!(matches!(
            model.shift_tensor("w", &[1.0, f32::INFINITY], 1.0, sub, rows),
            Err(EngineError::NonFiniteDirection { index: 1 })
        ));
        assert!(matches!(
            model.shift_tensor("absent", &[1.0; 2], 1.0, sub, rows),
            Err(EngineError::TensorNotFound(_))
        ));
        // alpha = 0 succeeds but must not touch a byte either.
        model.shift_tensor("w", &[1.0; 2], 0.0, sub, rows).unwrap();
        model.flush().unwrap();
        drop(model);
        assert_eq!(fs::read(&path).unwrap(), before);
    }

    #[test]
    fn direction_vector_from_safetensors_and_json() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        let single = p.join("v.safetensors");
        write_fake_safetensors(
            &single,
            &[(
                "dir",
                "BF16",
                vec![3],
                encode(&DType::BF16, &[1.0, -0.5, 2.0]),
            )],
        );
        assert_eq!(
            load_direction_vector(&single, None).unwrap(),
            vec![1.0, -0.5, 2.0]
        );

        let multi = p.join("multi.safetensors");
        write_fake_safetensors(
            &multi,
            &[
                ("a", "F16", vec![2], encode(&DType::F16, &[0.25, 4.0])),
                ("m", "F32", vec![1, 2], encode(&DType::F32, &[1.0, 2.0])),
            ],
        );
        assert_eq!(
            load_direction_vector(&multi, Some("a")).unwrap(),
            vec![0.25, 4.0]
        );
        assert!(matches!(
            load_direction_vector(&multi, None),
            Err(EngineError::InvalidDirectionSource { .. })
        ));
        assert!(matches!(
            load_direction_vector(&multi, Some("m")),
            Err(EngineError::InvalidShape { .. })
        ));

        let json = p.join("v.json");
        fs::write(&json, "[1, -2.5, 0.125]").unwrap();
        assert_eq!(
            load_direction_vector(&json, None).unwrap(),
            vec![1.0, -2.5, 0.125]
        );
        fs::write(&json, r#"{"not": "an array"}"#).unwrap();
        assert!(matches!(
            load_direction_vector(&json, None),
            Err(EngineError::InvalidDirectionSource { .. })
        ));

        let txt = p.join("v.txt");
        fs::write(&txt, "1 2 3").unwrap();
        assert!(matches!(
            load_direction_vector(&txt, None),
            Err(EngineError::InvalidDirectionSource { .. })
        ));
    }
}
