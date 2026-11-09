//! Deterministic `.surgery` writer.
//!
//! The header carries absolute blob offsets and precedes the blobs, so blobs are spooled to a
//! temporary file next to the output (a 7B model at rank 16 is hundreds of MB, too big for
//! memory) and the final file is assembled in `finish`, then renamed into place atomically.

use std::borrow::Cow;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use serde::Serialize;
use sha2::{Digest, Sha256};

use super::error::{Result, SurgeryError};
use super::format::{
    hex, sha256_hex, BaseInfo, BlobRef, Blobs, Compression, CompressionAlgo, FactorDtype,
    LowRankFactors, Provenance, SurgeryHeader, TensorEntry, TensorKind, TunedInfo, FOOTER_LEN,
    FORMAT_VERSION, MAGIC, MAX_HEADER_LEN, PREAMBLE_LEN,
};

/// Header fields known before the first tensor.
#[derive(Debug, Clone)]
pub struct HeaderInit {
    pub created_with: String,
    pub base_name: String,
    pub base_tensor_count: usize,
    pub tuned_name: String,
    pub rank: usize,
    pub factor_dtype: FactorDtype,
    pub compression: Compression,
    /// RFC 3339 string; `None` keeps the file reproducible.
    pub timestamp: Option<String>,
}

/// Per-tensor metadata supplied by the caller.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub target_dtype: String,
    pub base_sha256: String,
    pub rel_error: f64,
    pub energy: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WriteSummary {
    pub bytes: u64,
    pub sha256: String,
    pub tensors: usize,
}

pub struct SurgeryWriter {
    path: PathBuf,
    overwrite: bool,
    init: HeaderInit,
    spool_path: PathBuf,
    spool: Option<BufWriter<File>>,
    spool_len: u64,
    /// Blob offsets are relative to the blob section until `finish` knows `header_len`.
    tensors: Vec<TensorEntry>,
    names: HashSet<String>,
}

impl SurgeryWriter {
    /// Fails when `path` exists and `overwrite` is false. Nothing is visible at `path`
    /// until `finish` succeeds (temporary file next to it, then rename).
    pub fn create(path: &Path, init: HeaderInit, overwrite: bool) -> Result<Self> {
        if init.rank == 0 {
            return Err(SurgeryError::Usage("rang du .surgery nul".into()));
        }
        if !overwrite && path.exists() {
            return Err(exists_error(path));
        }
        let spool_path = sibling(path, "spool")?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&spool_path)
            .map_err(|e| SurgeryError::io(&spool_path, &e))?;
        Ok(Self {
            path: path.to_path_buf(),
            overwrite,
            init,
            spool_path,
            spool: Some(BufWriter::new(file)),
            spool_len: 0,
            tensors: Vec::new(),
            names: HashSet::new(),
        })
    }

    /// `rank` is the stored rank of `factors` (`s.len()`). Returns
    /// `SurgeryError::FactorOverflow` when a factor does not fit `factor_dtype`.
    pub fn add_lowrank(&mut self, meta: TensorMeta, factors: &LowRankFactors) -> Result<()> {
        let rank = factors.s.len();
        let mut entry = self.entry(meta, TensorKind::Lowrank, rank)?;
        let [m, n] = entry.shape[..] else {
            return Err(SurgeryError::Usage(format!(
                "{} : lowrank exige une forme 2D",
                entry.name
            )));
        };
        if m.checked_mul(rank) != Some(factors.u.len())
            || rank.checked_mul(n) != Some(factors.v.len())
        {
            return Err(SurgeryError::Usage(format!(
                "{} : tailles de U/S/V incohérentes avec la forme {m}x{n} et le rang {rank}",
                entry.name
            )));
        }
        let dtype = self.init.factor_dtype;
        // Encode everything first: an overflow must not leave a half-written tensor in the spool.
        let u = encode(&factors.u, dtype, &entry.name)?;
        let s = encode(&factors.s, dtype, &entry.name)?;
        let v = encode(&factors.v, dtype, &entry.name)?;
        entry.blobs = Blobs {
            u: Some(self.put_blob(&u)?),
            s: Some(self.put_blob(&s)?),
            v: Some(self.put_blob(&v)?),
            delta: None,
        };
        self.push(entry);
        Ok(())
    }

    /// Exact f32 delta of a 1D tensor (or of a tensor kept whole by `--on-high-error full`).
    pub fn add_full_delta(&mut self, meta: TensorMeta, delta: &[f32]) -> Result<()> {
        let mut entry = self.entry(meta, TensorKind::FullDelta, 0)?;
        if entry.numel() != u64::try_from(delta.len()).ok() {
            return Err(SurgeryError::Usage(format!(
                "{} : {} valeurs pour la forme {:?}",
                entry.name,
                delta.len(),
                entry.shape
            )));
        }
        let raw = encode(delta, FactorDtype::F32, &entry.name)?;
        entry.blobs.delta = Some(self.put_blob(&raw)?);
        self.push(entry);
        Ok(())
    }

    pub fn finish(mut self) -> Result<WriteSummary> {
        let mut spool = self
            .spool
            .take()
            .ok_or_else(|| SurgeryError::Failed("writer déjà terminé".into()))?;
        spool
            .flush()
            .map_err(|e| SurgeryError::io(&self.spool_path, &e))?;
        drop(spool);

        let tmp = sibling(&self.path, "tmp")?;
        let result = self.assemble(&tmp);
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result
    }

    fn assemble(&self, tmp: &Path) -> Result<WriteSummary> {
        let header_bytes = self.header_bytes()?;
        let header_len = header_bytes.len() as u64;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(tmp)
            .map_err(|e| SurgeryError::io(tmp, &e))?;
        let mut out = HashingWriter::new(BufWriter::new(file));
        let io = |e: std::io::Error| SurgeryError::io(tmp, &e);

        out.write_all(&MAGIC).map_err(io)?;
        out.write_all(&FORMAT_VERSION.to_le_bytes()).map_err(io)?;
        out.write_all(&[0u8; 6]).map_err(io)?; // flags u16 + reserved u32
        out.write_all(&header_len.to_le_bytes()).map_err(io)?;
        out.write_all(&header_bytes).map_err(io)?;

        let mut spool =
            File::open(&self.spool_path).map_err(|e| SurgeryError::io(&self.spool_path, &e))?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = spool
                .read(&mut buf)
                .map_err(|e| SurgeryError::io(&self.spool_path, &e))?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n]).map_err(io)?;
        }

        let (mut inner, digest, bytes) = out.finish();
        inner.write_all(&digest).map_err(io)?;
        inner.write_all(&MAGIC).map_err(io)?;
        let file = inner
            .into_inner()
            .map_err(|e| SurgeryError::io(tmp, e.error()))?;
        file.sync_all().map_err(io)?;
        drop(file);

        // Re-check at the last moment: another process may have created the target meanwhile.
        if !self.overwrite && self.path.exists() {
            return Err(exists_error(&self.path));
        }
        fs::rename(tmp, &self.path).map_err(|e| SurgeryError::io(&self.path, &e))?;
        Ok(WriteSummary {
            bytes: bytes + FOOTER_LEN,
            sha256: hex(&digest),
            tensors: self.tensors.len(),
        })
    }

    /// Canonical header bytes. Offsets depend on `header_len`, which depends on the digits of the
    /// offsets: iterate to the fixed point (lengths only grow, so it terminates in a few steps).
    fn header_bytes(&self) -> Result<Vec<u8>> {
        let mut header_len = 0u64;
        for _ in 0..16 {
            let header = self.header(PREAMBLE_LEN + header_len);
            header.validate()?;
            let bytes = header.to_canonical_json()?;
            if bytes.len() as u64 == header_len {
                if header_len > MAX_HEADER_LEN {
                    return Err(SurgeryError::Failed("en-tête trop volumineux".into()));
                }
                return Ok(bytes);
            }
            header_len = bytes.len() as u64;
        }
        Err(SurgeryError::Failed(
            "la longueur de l'en-tête ne converge pas".into(),
        ))
    }

    fn header(&self, blobs_start: u64) -> SurgeryHeader {
        let shift = |b: &Option<BlobRef>| {
            b.as_ref().map(|b| BlobRef {
                offset: b.offset + blobs_start,
                ..b.clone()
            })
        };
        let tensors = self
            .tensors
            .iter()
            .map(|t| TensorEntry {
                blobs: Blobs {
                    u: shift(&t.blobs.u),
                    s: shift(&t.blobs.s),
                    v: shift(&t.blobs.v),
                    delta: shift(&t.blobs.delta),
                },
                ..t.clone()
            })
            .collect();
        SurgeryHeader {
            format_version: FORMAT_VERSION,
            created_with: self.init.created_with.clone(),
            base: BaseInfo {
                name: self.init.base_name.clone(),
                tensor_count: self.init.base_tensor_count,
            },
            tuned: TunedInfo {
                name: self.init.tuned_name.clone(),
            },
            rank: self.init.rank,
            factor_dtype: self.init.factor_dtype,
            compression: self.init.compression,
            provenance: self
                .init
                .timestamp
                .clone()
                .map(|timestamp| Provenance { timestamp }),
            tensors,
        }
    }

    /// Entry without blobs; checks everything that does not need the data.
    fn entry(&self, meta: TensorMeta, kind: TensorKind, rank: usize) -> Result<TensorEntry> {
        let entry = TensorEntry {
            name: meta.name,
            shape: meta.shape,
            kind,
            rank,
            target_dtype: meta.target_dtype,
            base_sha256: meta.base_sha256,
            rel_error: meta.rel_error,
            energy: meta.energy,
            blobs: Blobs::default(),
        };
        entry.validate_fields(self.init.rank)?;
        if self.names.contains(&entry.name) {
            return Err(SurgeryError::Usage(format!(
                "tenseur déjà ajouté : {}",
                entry.name
            )));
        }
        Ok(entry)
    }

    fn push(&mut self, entry: TensorEntry) {
        self.names.insert(entry.name.clone());
        self.tensors.push(entry);
    }

    /// Hash the raw bytes, compress, append to the spool.
    fn put_blob(&mut self, raw: &[u8]) -> Result<BlobRef> {
        let Compression { algo, level } = self.init.compression;
        let stored: Cow<[u8]> = match algo {
            CompressionAlgo::None => Cow::Borrowed(raw),
            CompressionAlgo::Deflate => {
                Cow::Owned(miniz_oxide::deflate::compress_to_vec(raw, level))
            }
        };
        let spool = self
            .spool
            .as_mut()
            .ok_or_else(|| SurgeryError::Failed("writer déjà terminé".into()))?;
        spool
            .write_all(&stored)
            .map_err(|e| SurgeryError::io(&self.spool_path, &e))?;
        let blob = BlobRef {
            offset: self.spool_len,
            comp_len: stored.len() as u64,
            raw_len: raw.len() as u64,
            sha256: sha256_hex(raw),
        };
        self.spool_len += blob.comp_len;
        Ok(blob)
    }
}

impl Drop for SurgeryWriter {
    fn drop(&mut self) {
        // Also reached after a successful `finish`: the spool is then just leftover scratch.
        self.spool = None;
        let _ = fs::remove_file(&self.spool_path);
    }
}

/// Little-endian bytes of `values` in `dtype`, round-to-nearest-even. A finite value that does
/// not fit (f16 above 65504, bf16 rounding past f32::MAX) is an error, never a silent infinity.
fn encode(values: &[f32], dtype: FactorDtype, tensor: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(values.len() * dtype.size_of());
    let overflow = || SurgeryError::FactorOverflow {
        tensor: tensor.to_string(),
        dtype: dtype.as_str(),
    };
    for &x in values {
        if !x.is_finite() {
            return Err(SurgeryError::Usage(format!(
                "{tensor} : valeur non finie dans les données"
            )));
        }
        match dtype {
            FactorDtype::F32 => out.extend_from_slice(&x.to_le_bytes()),
            FactorDtype::F16 => {
                let h = f16::from_f32(x);
                if !h.is_finite() {
                    return Err(overflow());
                }
                out.extend_from_slice(&h.to_bits().to_le_bytes());
            }
            FactorDtype::Bf16 => {
                let b = bf16::from_f32(x);
                if !b.is_finite() {
                    return Err(overflow());
                }
                out.extend_from_slice(&b.to_bits().to_le_bytes());
            }
        }
    }
    Ok(out)
}

fn exists_error(path: &Path) -> SurgeryError {
    SurgeryError::Usage(format!(
        "{} existe déjà (utiliser --overwrite pour l'écraser)",
        path.display()
    ))
}

/// Hidden scratch file next to `path`, so rename stays on one filesystem.
fn sibling(path: &Path, suffix: &str) -> Result<PathBuf> {
    let name = path
        .file_name()
        .ok_or_else(|| SurgeryError::Usage(format!("chemin invalide : {}", path.display())))?
        .to_string_lossy();
    let file = format!(".{name}.{}.{suffix}", std::process::id());
    Ok(path.with_file_name(file))
}

/// Hashes exactly the bytes written, so the footer digest covers the whole file before it.
struct HashingWriter<W: Write> {
    inner: W,
    hasher: Sha256,
    written: u64,
}

impl<W: Write> HashingWriter<W> {
    fn new(inner: W) -> Self {
        Self {
            inner,
            hasher: Sha256::new(),
            written: 0,
        }
    }

    fn finish(self) -> (W, [u8; 32], u64) {
        (self.inner, self.hasher.finalize().into(), self.written)
    }
}

impl<W: Write> Write for HashingWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.written += n as u64;
        Ok(n)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}
