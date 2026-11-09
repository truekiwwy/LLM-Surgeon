//! `.surgery` reader. The file is untrusted input: every length is checked against the real file
//! size with checked arithmetic, and nothing is allocated from a value that was not validated.

use std::collections::HashMap;
use std::fs::File;
use std::io::{ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use half::{bf16, f16};
use miniz_oxide::inflate::{decompress_to_vec_with_limit, TINFLStatus};
use sha2::{Digest, Sha256};

use super::error::{Result, SurgeryError};
use super::format::{
    sha256_hex, BlobRole, CompressionAlgo, FactorDtype, FormatError, LowRankFactors, SurgeryHeader,
    TensorEntry, TensorKind, FOOTER_LEN, FORMAT_VERSION, MAGIC, MAX_HEADER_LEN, PREAMBLE_LEN,
};

/// True when the file starts with the `SURG` magic (cheap: reads 4 bytes, never fails).
pub fn is_surgery_file(path: &Path) -> bool {
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok()
        && magic == MAGIC
}

pub struct SurgeryReader {
    file: File,
    path: PathBuf,
    header: SurgeryHeader,
    index: HashMap<String, usize>,
}

impl SurgeryReader {
    /// Validates magic, version, `header_len` against the real size, blob bounds and
    /// overlaps, declared lengths, and the global SHA-256 (streamed) before returning:
    /// a reader that exists is a reader on a file whose integrity was checked.
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| SurgeryError::io(path, &e))?;
        let len = file
            .metadata()
            .map_err(|e| SurgeryError::io(path, &e))?
            .len();
        let io = |e: std::io::Error| map_io(path, e);

        let mut pre = [0u8; PREAMBLE_LEN as usize];
        let got = read_up_to(&mut file, &mut pre).map_err(io)?;
        if got < 4 {
            return Err(if MAGIC.starts_with(&pre[..got]) {
                truncated(len, "moins de 4 octets")
            } else {
                FormatError::BadMagic.into()
            });
        }
        if pre[..4] != MAGIC {
            return Err(FormatError::BadMagic.into());
        }
        if len < PREAMBLE_LEN + FOOTER_LEN {
            return Err(truncated(len, "plus court que le préambule et le pied"));
        }
        let version = u16::from_le_bytes([pre[4], pre[5]]);
        if version != FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion {
                found: version,
                supported: FORMAT_VERSION,
            }
            .into());
        }
        let flags = u16::from_le_bytes([pre[6], pre[7]]);
        let reserved = u32::from_le_bytes([pre[8], pre[9], pre[10], pre[11]]);
        if flags != 0 || reserved != 0 {
            return Err(FormatError::UnsupportedFlags { flags, reserved }.into());
        }

        let header_len = le_u64(&pre[12..20]);
        let available = len - PREAMBLE_LEN - FOOTER_LEN;
        if header_len > available {
            return Err(FormatError::HeaderLenMismatch {
                declared: header_len,
                available,
            }
            .into());
        }
        if header_len > MAX_HEADER_LEN {
            return Err(FormatError::InvalidHeader(format!(
                "en-tête de {header_len} octets, limite {MAX_HEADER_LEN}"
            ))
            .into());
        }
        // header_len <= available <= len, so both fit in the file and in usize on 64-bit hosts.
        let blobs_start = PREAMBLE_LEN + header_len;
        let footer_start = len - FOOTER_LEN;

        let mut json = vec![0u8; header_len as usize];
        file.read_exact(&mut json).map_err(io)?;
        let header: SurgeryHeader =
            serde_json::from_slice(&json).map_err(|e| FormatError::InvalidHeader(e.to_string()))?;
        header.validate()?;
        check_layout(&header, blobs_start, footer_start)?;

        let mut footer = [0u8; FOOTER_LEN as usize];
        file.seek(SeekFrom::Start(footer_start)).map_err(io)?;
        file.read_exact(&mut footer).map_err(io)?;
        if footer[32..] != MAGIC {
            return Err(FormatError::BadMagic.into());
        }
        file.seek(SeekFrom::Start(0)).map_err(io)?;
        if hash_prefix(&mut file, footer_start).map_err(io)?[..] != footer[..32] {
            return Err(FormatError::GlobalHashMismatch.into());
        }

        let index = header
            .tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        Ok(Self {
            file,
            path: path.to_path_buf(),
            header,
            index,
        })
    }

    pub fn header(&self) -> &SurgeryHeader {
        &self.header
    }

    /// Inflates (never beyond the declared `raw_len`), verifies the blob hash, converts to f32.
    pub fn read_lowrank(&mut self, entry: &TensorEntry) -> Result<LowRankFactors> {
        let entry = self.own_entry(entry, TensorKind::Lowrank)?;
        let dtype = self.header.factor_dtype;
        let factors = LowRankFactors {
            u: decode(&self.read_blob(&entry, BlobRole::U)?, dtype),
            s: decode(&self.read_blob(&entry, BlobRole::S)?, dtype),
            v: decode(&self.read_blob(&entry, BlobRole::V)?, dtype),
        };
        ensure_finite(&entry, &factors)?;
        Ok(factors)
    }

    pub fn read_full_delta(&mut self, entry: &TensorEntry) -> Result<Vec<f32>> {
        let entry = self.own_entry(entry, TensorKind::FullDelta)?;
        let delta = decode(&self.read_blob(&entry, BlobRole::Delta)?, FactorDtype::F32);
        match delta.iter().all(|x| x.is_finite()) {
            true => Ok(delta),
            false => Err(FormatError::NonFiniteValue {
                tensor: entry.name,
                blob: BlobRole::Delta.as_str(),
            }
            .into()),
        }
    }

    /// The caller's entry is never trusted: use the validated copy held by the header.
    fn own_entry(&self, entry: &TensorEntry, kind: TensorKind) -> Result<TensorEntry> {
        let own = self
            .index
            .get(&entry.name)
            .and_then(|&i| self.header.tensors.get(i))
            .filter(|own| *own == entry)
            .ok_or_else(|| {
                SurgeryError::Usage(format!(
                    "{} : entrée absente de {}",
                    entry.name,
                    self.path.display()
                ))
            })?;
        if own.kind != kind {
            return Err(SurgeryError::Usage(format!(
                "{} : type de tenseur {:?} inattendu ici",
                own.name, own.kind
            )));
        }
        Ok(own.clone())
    }

    /// Raw (decompressed, hash-checked) bytes of one blob; length is exactly `raw_len`.
    fn read_blob(&mut self, entry: &TensorEntry, role: BlobRole) -> Result<Vec<u8>> {
        let (tensor, blob_name) = (entry.name.as_str(), role.as_str());
        let blob = entry.blob(role).ok_or_else(|| {
            FormatError::InvalidHeader(format!("{tensor} : blob {blob_name} absent"))
        })?;
        // comp_len was bounded by the real file size in `open`.
        let mut stored = vec![0u8; blob.comp_len as usize];
        self.file
            .seek(SeekFrom::Start(blob.offset))
            .and_then(|_| self.file.read_exact(&mut stored))
            .map_err(|e| map_io(&self.path, e))?;

        let raw = match self.header.compression.algo {
            CompressionAlgo::None => stored,
            CompressionAlgo::Deflate => {
                let limit = blob.raw_len as usize;
                decompress_to_vec_with_limit(&stored, limit).map_err(|e| match e.status {
                    TINFLStatus::HasMoreOutput => FormatError::DecompressionBomb {
                        tensor: tensor.to_string(),
                        blob: blob_name,
                        limit: blob.raw_len,
                    },
                    _ => FormatError::DecompressionFailed {
                        tensor: tensor.to_string(),
                        blob: blob_name,
                        reason: e.to_string(),
                    },
                })?
            }
        };
        if raw.len() as u64 != blob.raw_len {
            return Err(FormatError::RawLenMismatch {
                tensor: tensor.to_string(),
                blob: blob_name,
                declared: blob.raw_len,
                expected: raw.len() as u64,
            }
            .into());
        }
        if sha256_hex(&raw) != blob.sha256 {
            return Err(FormatError::BlobHashMismatch {
                tensor: tensor.to_string(),
                blob: blob_name,
            }
            .into());
        }
        Ok(raw)
    }
}

fn truncated(len: u64, what: &str) -> SurgeryError {
    FormatError::Truncated(format!("{len} octets, {what}")).into()
}

fn map_io(path: &Path, e: std::io::Error) -> SurgeryError {
    if e.kind() == ErrorKind::UnexpectedEof {
        FormatError::Truncated("fin de fichier inattendue".into()).into()
    } else {
        SurgeryError::io(path, &e)
    }
}

fn le_u64(bytes: &[u8]) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&bytes[..8]);
    u64::from_le_bytes(b)
}

/// Fills `buf` as far as the file goes; returns the number of bytes read.
fn read_up_to(file: &mut File, buf: &mut [u8]) -> std::io::Result<usize> {
    let mut got = 0;
    while got < buf.len() {
        match file.read(&mut buf[got..])? {
            0 => break,
            n => got += n,
        }
    }
    Ok(got)
}

/// SHA-256 of the first `len` bytes, streamed in 1 MiB chunks.
fn hash_prefix(file: &mut File, len: u64) -> std::io::Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut left = len;
    while left > 0 {
        let want = buf.len().min(usize::try_from(left).unwrap_or(usize::MAX));
        let n = file.read(&mut buf[..want])?;
        if n == 0 {
            return Err(ErrorKind::UnexpectedEof.into());
        }
        hasher.update(&buf[..n]);
        left -= n as u64;
    }
    Ok(hasher.finalize().into())
}

/// Blobs must lie inside `[blobs_start, footer_start)`, must not overlap, and tile it exactly.
fn check_layout(
    header: &SurgeryHeader,
    blobs_start: u64,
    footer_start: u64,
) -> std::result::Result<(), FormatError> {
    // (offset, end, label)
    let mut spans = Vec::new();
    for t in &header.tensors {
        for (role, b) in t.blob_list() {
            let out_of_bounds = |detail: String| FormatError::OffsetOutOfBounds {
                tensor: t.name.clone(),
                blob: role.as_str(),
                detail,
            };
            let end = b
                .offset
                .checked_add(b.comp_len)
                .ok_or_else(|| out_of_bounds("offset + comp_len déborde".into()))?;
            if b.offset < blobs_start || end > footer_start {
                return Err(out_of_bounds(format!(
                    "[{}, {end}) hors de [{blobs_start}, {footer_start})",
                    b.offset
                )));
            }
            spans.push((b.offset, end, format!("{}/{}", t.name, role.as_str())));
        }
    }
    spans.sort();
    let mut cursor = blobs_start;
    for (offset, end, label) in &spans {
        if *offset < cursor {
            let first = spans
                .iter()
                .find(|s| s.1 == cursor)
                .map_or_else(String::new, |s| s.2.clone());
            return Err(FormatError::BlobOverlap {
                first,
                second: label.clone(),
            });
        }
        if *offset > cursor {
            return Err(FormatError::InvalidHeader(format!(
                "trou de {} octets avant {label}",
                offset - cursor
            )));
        }
        cursor = *end;
    }
    if cursor != footer_start {
        return Err(FormatError::InvalidHeader(format!(
            "{} octets non référencés avant le pied",
            footer_start - cursor
        )));
    }
    Ok(())
}

/// Little-endian bytes to f32. Lengths were validated against the header (multiples of 2 or 4).
fn decode(raw: &[u8], dtype: FactorDtype) -> Vec<f32> {
    match dtype {
        FactorDtype::F32 => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        FactorDtype::F16 => raw
            .chunks_exact(2)
            .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        FactorDtype::Bf16 => raw
            .chunks_exact(2)
            .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
    }
}

/// A hostile file can store NaN/Inf bit patterns that pass every hash; refuse them here.
fn ensure_finite(entry: &TensorEntry, f: &LowRankFactors) -> Result<()> {
    for (role, data) in [
        (BlobRole::U, &f.u),
        (BlobRole::S, &f.s),
        (BlobRole::V, &f.v),
    ] {
        if !data.iter().all(|x| x.is_finite()) {
            return Err(FormatError::NonFiniteValue {
                tensor: entry.name.clone(),
                blob: role.as_str(),
            }
            .into());
        }
    }
    Ok(())
}
