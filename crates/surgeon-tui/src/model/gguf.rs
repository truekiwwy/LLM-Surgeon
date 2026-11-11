//! Minimal GGUF reader (versions 2 and 3): mmap-backed, parses the header,
//! skips the key-value metadata (keeping `general.architecture` and
//! `general.alignment`) and the tensor infos, and resolves each tensor's
//! absolute byte range against the aligned data section.
//!
//! ```text
//! magic "GGUF" | version u32 | tensor_count u64 | kv_count u64
//! kv:     key string | type u32 | value            (string: len u64 + bytes)
//! tensor: name string | n_dims u32 | dims u64[n] | type i32 | offset u64
//! data_start = align_up(after header, alignment = 32 default)
//! ```
//!
//! GGUF dims list the fastest dimension first; entries are stored reversed so
//! `shape.last()` matches the SafeTensors row-major convention.

use std::path::{Path, PathBuf};

use crate::model::dtype::DType;
use crate::model::ModelError;

/// A raw entry before role classification.
pub type RawEntry = (String, Vec<u64>, DType, u64, PathBuf, u64);

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], ModelError> {
        let end = self.pos.checked_add(n).ok_or_else(truncated)?;
        if end > self.data.len() {
            return Err(truncated());
        }
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u32(&mut self) -> Result<u32, ModelError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, ModelError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn string(&mut self) -> Result<String, ModelError> {
        let len = self.u64()? as usize;
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| ModelError::Gguf("clé ou valeur non UTF-8".into()))
    }

    /// Skips one metadata value of the given type, walking nested arrays.
    fn skip_value(&mut self, value_type: u32) -> Result<(), ModelError> {
        match value_type {
            0 | 1 | 7 => self.take(1).map(|_| ()),
            2 | 3 => self.take(2).map(|_| ()),
            4..=6 => self.take(4).map(|_| ()),
            10..=12 => self.take(8).map(|_| ()),
            8 => self.string().map(|_| ()),
            9 => {
                let elem_type = self.u32()?;
                let count = self.u64()?;
                for _ in 0..count {
                    self.skip_value(elem_type)?;
                }
                Ok(())
            }
            other => Err(ModelError::Gguf(format!(
                "type de métadonnée inconnu : {other}"
            ))),
        }
    }
}

fn truncated() -> ModelError {
    ModelError::Gguf("en-tête tronqué".into())
}

/// Opens one `.gguf` file and returns its tensors plus the detected family.
pub fn open_file(path: &Path) -> Result<(Vec<RawEntry>, Option<String>), ModelError> {
    use crate::model::map_file;

    let map = map_file(path)?;
    let mut r = Reader {
        data: &map[..],
        pos: 0,
    };

    if r.take(4)? != b"GGUF" {
        return Err(ModelError::Gguf("magic GGUF absent".into()));
    }
    let version = r.u32()?;
    if version < 2 {
        return Err(ModelError::Gguf(format!(
            "version GGUF non gérée : {version}"
        )));
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;

    let mut family = None;
    let mut alignment = 32u64;
    for _ in 0..kv_count {
        let key = r.string()?;
        let value_type = r.u32()?;
        match (key.as_str(), value_type) {
            ("general.architecture", 8) => family = Some(r.string()?),
            ("general.alignment", 4 | 5) => alignment = u64::from(r.u32()?),
            ("general.alignment", 10 | 11) => alignment = r.u64()?,
            _ => r.skip_value(value_type)?,
        }
    }
    if alignment == 0 {
        alignment = 32;
    }

    let mut infos = Vec::new();
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        let mut dims = Vec::with_capacity(n_dims.min(8));
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let ggml_type = r.u32()? as i32;
        let offset = r.u64()?;
        infos.push((name, dims, ggml_type, offset));
    }

    let data_start = align_up(r.pos as u64, alignment);
    let file_len = map.len() as u64;

    let mut entries = Vec::with_capacity(infos.len());
    for (name, dims, ggml_type, offset) in infos {
        if dims.is_empty() {
            return Err(ModelError::Gguf(format!("{name} : tenseur sans dimension")));
        }
        let dtype = DType::from_ggml(ggml_type as u32);
        let numel: u64 = dims.iter().product();
        let byte_len = byte_len(dtype.block(), numel);
        let absolute = data_start
            .checked_add(offset)
            .ok_or_else(|| ModelError::Gguf(format!("{name} : déplacement invalide")))?;
        if absolute + byte_len > file_len {
            return Err(ModelError::Gguf(format!(
                "{name} : données au-delà de la fin du fichier"
            )));
        }
        let mut shape = dims;
        shape.reverse();
        entries.push((name, shape, dtype, byte_len, path.to_path_buf(), absolute));
    }
    Ok((entries, family))
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

/// Bytes on disk for `numel` elements of a block dtype.
fn byte_len(block: Option<(u64, u64)>, numel: u64) -> u64 {
    match block {
        None => 0,
        Some((elems, bytes)) => numel.div_ceil(elems) * bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built GGUF: one f32 1D tensor and one Q8_0 tensor, 32-byte aligned,
    /// with an array-valued KV that must be skipped without desynchronizing.
    fn build_gguf() -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes()); // tensor count
        out.extend_from_slice(&2u64.to_le_bytes()); // kv count

        let mut kv1 = Vec::new();
        kv1.extend_from_slice(&20u64.to_le_bytes());
        kv1.extend_from_slice(b"general.architecture");
        kv1.extend_from_slice(&8u32.to_le_bytes());
        kv1.extend_from_slice(&5u64.to_le_bytes());
        kv1.extend_from_slice(b"llama");

        let mut kv2 = Vec::new();
        kv2.extend_from_slice(&18u64.to_le_bytes());
        kv2.extend_from_slice(b"general.some_array");
        kv2.extend_from_slice(&9u32.to_le_bytes());
        kv2.extend_from_slice(&8u32.to_le_bytes());
        kv2.extend_from_slice(&2u64.to_le_bytes());
        for text in ["alpha", "beta"] {
            kv2.extend_from_slice(&(text.len() as u64).to_le_bytes());
            kv2.extend_from_slice(text.as_bytes());
        }
        out.extend_from_slice(&kv1);
        out.extend_from_slice(&kv2);

        // tensor 1: token_embd.weight, dims [4], F32, offset 0
        out.extend_from_slice(&17u64.to_le_bytes());
        out.extend_from_slice(b"token_embd.weight");
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&4u64.to_le_bytes());
        out.extend_from_slice(&0i32.to_le_bytes());
        out.extend_from_slice(&0u64.to_le_bytes());
        // tensor 2: blk.0.attn_output.weight, dims [32, 2] fastest-first, Q8_0, offset 16
        out.extend_from_slice(&24u64.to_le_bytes());
        out.extend_from_slice(b"blk.0.attn_output.weight");
        out.extend_from_slice(&2u32.to_le_bytes());
        out.extend_from_slice(&32u64.to_le_bytes());
        out.extend_from_slice(&2u64.to_le_bytes());
        out.extend_from_slice(&8i32.to_le_bytes());
        out.extend_from_slice(&16u64.to_le_bytes());

        while out.len() % 32 != 0 {
            out.push(0);
        }
        // data
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            out.extend_from_slice(&v.to_le_bytes());
        }
        let mut block = Vec::new();
        block.extend_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        for i in 0..32u8 {
            block.push(i + 1);
        }
        out.extend_from_slice(&block);
        out.extend_from_slice(&block);
        out
    }

    #[test]
    fn hand_built_gguf_parses_with_family_offsets_and_dtypes() {
        let bytes = build_gguf();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("m.gguf");
        std::fs::write(&path, &bytes).expect("write");

        let (entries, family) = open_file(&path).expect("parse");
        assert_eq!(family.as_deref(), Some("llama"));
        assert_eq!(entries.len(), 2);

        let (name, shape, dtype, byte_len, file, offset) = &entries[0];
        assert_eq!(name, "token_embd.weight");
        assert_eq!(shape, &[4]);
        assert_eq!(*dtype, DType::F32);
        assert_eq!(*byte_len, 16);
        assert_eq!(file, &path);
        // data_start = align_up(header_len, 32)
        assert_eq!(*offset % 32, 0);

        let (name, shape, dtype, byte_len, _, offset) = &entries[1];
        assert_eq!(name, "blk.0.attn_output.weight");
        // dims written fastest-first are reversed for the row-major convention.
        assert_eq!(shape, &[2, 32]);
        assert_eq!(*dtype, DType::Q8_0);
        assert_eq!(*byte_len, 2 * 34);
        assert_eq!(*offset, entries[0].5 + 16);
    }

    #[test]
    fn corrupted_magic_and_truncation_are_typed_errors() {
        let mut bytes = build_gguf();
        bytes[0] = b'X';
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("bad.gguf");
        std::fs::write(&path, &bytes).expect("write");
        assert!(matches!(
            open_file(&path),
            Err(ModelError::Gguf(message)) if message.contains("magic")
        ));

        let bytes = build_gguf();
        let path = dir.path().join("cut.gguf");
        std::fs::write(&path, &bytes[..40]).expect("write");
        assert!(matches!(
            open_file(&path),
            Err(ModelError::Gguf(message)) if message.contains("tronqué")
        ));
    }

    #[test]
    fn block_byte_lengths_cover_the_common_types() {
        assert_eq!(byte_len(Some((1, 4)), 7), 28);
        assert_eq!(byte_len(Some((32, 34)), 64), 68);
        assert_eq!(byte_len(Some((256, 144)), 512), 288);
        assert_eq!(byte_len(None, 100), 0);
    }
}
