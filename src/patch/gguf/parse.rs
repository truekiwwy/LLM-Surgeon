//! GGUF v2/v3 reader (little-endian only), over a byte slice (a mapping or a test buffer).
//!
//! Every read goes through [`Cursor::take`], which checks the bounds first: a truncated or
//! corrupted file is a typed error, never a panic nor a read past the end. No allocation is
//! sized by an untrusted count: vectors are reserved at most `remaining bytes / minimal
//! record size`, and arrays keep only their first [`ARRAY_HEAD`] items.
//!
//! ```text
//! "GGUF" | version u32 | tensor_count u64 | kv_count u64
//! kv_count x  { key: string, type u32, value }        string = len u64 + UTF-8 bytes
//! tensor_count x { name: string, n_dims u32, ne[n_dims] u64, type u32, offset u64 }
//! padding to general.alignment (default 32)
//! data: tensor i at data_start + offset_i (offset_i multiple of the alignment)
//! ```

use serde::Serialize;
use thiserror::Error;

pub const MAGIC: [u8; 4] = *b"GGUF";
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// GGML_MAX_DIMS.
pub const MAX_DIMS: u32 = 4;
/// Longest string accepted (a chat template is a few KiB, a token a few bytes).
pub const MAX_STRING_LEN: u64 = 16 << 20;
/// Items kept per metadata array (the rest is validated and skipped).
pub const ARRAY_HEAD: usize = 16;
/// Nesting depth of arrays of arrays.
const MAX_ARRAY_DEPTH: u32 = 4;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum GgufError {
    #[error("pas un fichier GGUF (magie {0:02x?}, attendu « GGUF »)")]
    NotGguf([u8; 4]),
    #[error("GGUF big-endian non supporté (seul le little-endian est lu)")]
    BigEndian,
    #[error("version GGUF {0} non supportée (2 ou 3)")]
    UnsupportedVersion(u32),
    #[error(
        "fichier tronqué : {what} ({need} octets à l'offset {offset}, fichier de {len} octets)"
    )]
    Truncated {
        what: String,
        offset: u64,
        need: u64,
        len: u64,
    },
    #[error("{what} aberrant : {value} (plafond {limit})")]
    Limit {
        what: String,
        value: u64,
        limit: u64,
    },
    #[error("type de valeur de métadonnée inconnu {0}")]
    ValueType(u32),
    #[error("chaîne non UTF-8 : {0}")]
    Utf8(String),
    #[error("clé de métadonnée dupliquée : {0}")]
    DuplicateKey(String),
    #[error("tenseur dupliqué : {0}")]
    DuplicateTensor(String),
    #[error("tenseur {name} : {reason}")]
    Tensor { name: String, reason: String },
    #[error("general.alignment invalide : {0} (puissance de 2 attendue)")]
    Alignment(u64),
    #[error(
        "GGUF multi-parties non supporté ({0}) : fusionnez d'abord avec llama-gguf-split --merge"
    )]
    Split(String),
}

type Result<T> = std::result::Result<T, GgufError>;

/// A `ggml_type` id. Only [`GgmlType::is_patchable`] types can be patched; any other type
/// is still parsed (its tensors are never touched).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub struct GgmlType(pub u32);

impl GgmlType {
    pub const F32: GgmlType = GgmlType(0);
    pub const F16: GgmlType = GgmlType(1);
    pub const Q8_0: GgmlType = GgmlType(8);
    pub const Q4_K: GgmlType = GgmlType(12);
    pub const Q6_K: GgmlType = GgmlType(14);
    pub const BF16: GgmlType = GgmlType(30);

    pub fn name(self) -> String {
        const NAMES: [&str; 31] = [
            "F32", "F16", "Q4_0", "Q4_1", "type4", "type5", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K",
            "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S",
            "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M",
            "BF16",
        ];
        match NAMES.get(self.0 as usize) {
            Some(name) => (*name).to_owned(),
            None => format!("type{}", self.0),
        }
    }

    /// `(elements per block, bytes per block)` of the patchable types.
    pub fn block(self) -> Option<(u64, u64)> {
        match self {
            GgmlType::F32 => Some((1, 4)),
            GgmlType::F16 | GgmlType::BF16 => Some((1, 2)),
            GgmlType::Q8_0 => Some((32, 34)),
            GgmlType::Q4_K => Some((256, 144)),
            GgmlType::Q6_K => Some((256, 210)),
            _ => None,
        }
    }

    pub fn is_patchable(self) -> bool {
        self.block().is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    /// `head` holds the first [`ARRAY_HEAD`] items.
    Array {
        elem_type: u32,
        len: u64,
        head: Vec<MetaValue>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            MetaValue::U8(v) => Some(u64::from(v)),
            MetaValue::U16(v) => Some(u64::from(v)),
            MetaValue::U32(v) => Some(u64::from(v)),
            MetaValue::U64(v) => Some(v),
            MetaValue::I8(v) => u64::try_from(v).ok(),
            MetaValue::I16(v) => u64::try_from(v).ok(),
            MetaValue::I32(v) => u64::try_from(v).ok(),
            MetaValue::I64(v) => u64::try_from(v).ok(),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::String(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TensorEntry {
    pub name: String,
    /// `ne[0]` is the contiguous dimension (columns, `in_features`), `ne[1]` the rows.
    pub ne: Vec<u64>,
    pub ggml_type: GgmlType,
    /// Relative to the start of the data section.
    pub offset: u64,
    /// Absolute position in the file.
    pub start: u64,
    /// Byte size, known for patchable types only.
    pub size: Option<u64>,
}

impl TensorEntry {
    pub fn n_elements(&self) -> u64 {
        self.ne.iter().product()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: Vec<(String, MetaValue)>,
    pub tensors: Vec<TensorEntry>,
    pub alignment: u64,
    /// End of the header region `[0, data_start)`: magic, metadata, tensor table, padding.
    pub data_start: u64,
    pub file_len: u64,
}

impl GgufFile {
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|t| t.name == name)
    }

    pub fn meta(&self, key: &str) -> Option<&MetaValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.meta("general.architecture")
            .and_then(MetaValue::as_str)
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn remaining(&self) -> u64 {
        (self.bytes.len() - self.pos) as u64
    }

    fn take(&mut self, n: u64, what: &str) -> Result<&'a [u8]> {
        let truncated = || GgufError::Truncated {
            what: what.to_owned(),
            offset: self.pos as u64,
            need: n,
            len: self.bytes.len() as u64,
        };
        let n = usize::try_from(n).map_err(|_| truncated())?;
        let end = self.pos.checked_add(n).ok_or_else(truncated)?;
        let slice = self.bytes.get(self.pos..end).ok_or_else(truncated)?;
        self.pos = end;
        Ok(slice)
    }

    fn array<const N: usize>(&mut self, what: &str) -> Result<[u8; N]> {
        let mut out = [0u8; N];
        out.copy_from_slice(self.take(N as u64, what)?);
        Ok(out)
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        self.array(what).map(u32::from_le_bytes)
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        self.array(what).map(u64::from_le_bytes)
    }

    fn string(&mut self, what: &str) -> Result<String> {
        let len = self.u64(what)?;
        if len > MAX_STRING_LEN {
            return Err(GgufError::Limit {
                what: format!("longueur de chaîne ({what})"),
                value: len,
                limit: MAX_STRING_LEN,
            });
        }
        let bytes = self.take(len, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| GgufError::Utf8(what.to_owned()))
    }

    /// A count of records of at least `min_size` bytes each: larger than what the rest of
    /// the file can hold means corruption.
    fn count(&mut self, what: &str, min_size: u64) -> Result<u64> {
        let n = self.u64(what)?;
        let limit = self.remaining() / min_size;
        if n > limit {
            return Err(GgufError::Limit {
                what: what.to_owned(),
                value: n,
                limit,
            });
        }
        Ok(n)
    }

    fn value(&mut self, ty: u32, what: &str, depth: u32) -> Result<MetaValue> {
        Ok(match ty {
            0 => MetaValue::U8(u8::from_le_bytes(self.array(what)?)),
            1 => MetaValue::I8(i8::from_le_bytes(self.array(what)?)),
            2 => MetaValue::U16(u16::from_le_bytes(self.array(what)?)),
            3 => MetaValue::I16(i16::from_le_bytes(self.array(what)?)),
            4 => MetaValue::U32(self.u32(what)?),
            5 => MetaValue::I32(i32::from_le_bytes(self.array(what)?)),
            6 => MetaValue::F32(f32::from_le_bytes(self.array(what)?)),
            7 => MetaValue::Bool(self.array::<1>(what)?[0] != 0),
            8 => MetaValue::String(self.string(what)?),
            9 => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(GgufError::Limit {
                        what: format!("imbrication de tableaux ({what})"),
                        value: u64::from(depth) + 1,
                        limit: u64::from(MAX_ARRAY_DEPTH),
                    });
                }
                let elem_type = self.u32(what)?;
                let min_size = match elem_type {
                    0 | 1 | 7 => 1,
                    2 | 3 => 2,
                    4..=6 => 4,
                    8 | 10..=12 => 8,
                    9 => 12,
                    other => return Err(GgufError::ValueType(other)),
                };
                let len = self.count(&format!("longueur de tableau ({what})"), min_size)?;
                let mut head = Vec::with_capacity(len.min(ARRAY_HEAD as u64) as usize);
                for i in 0..len {
                    let item = self.value(elem_type, what, depth + 1)?;
                    if i < ARRAY_HEAD as u64 {
                        head.push(item);
                    }
                }
                MetaValue::Array {
                    elem_type,
                    len,
                    head,
                }
            }
            10 => MetaValue::U64(self.u64(what)?),
            11 => MetaValue::I64(i64::from_le_bytes(self.array(what)?)),
            12 => MetaValue::F64(f64::from_le_bytes(self.array(what)?)),
            other => return Err(GgufError::ValueType(other)),
        })
    }
}

/// Parses the header region and checks every tensor against the file bounds.
pub fn parse(bytes: &[u8]) -> Result<GgufFile> {
    let mut c = Cursor { bytes, pos: 0 };
    let magic: [u8; 4] = c.array("magie")?;
    if magic != MAGIC {
        return Err(GgufError::NotGguf(magic));
    }
    let version = c.u32("version")?;
    match version {
        2 | 3 => {}
        v if matches!(v.swap_bytes(), 2 | 3) => return Err(GgufError::BigEndian),
        v => return Err(GgufError::UnsupportedVersion(v)),
    }
    // Minimal records: tensor info = name len 8 + n_dims 4 + type 4 + offset 8;
    // key/value = key len 8 + type 4 + one byte.
    let tensor_count = c.count("nombre de tenseurs", 24)?;
    let kv_count = c.count("nombre de métadonnées", 13)?;

    let mut metadata: Vec<(String, MetaValue)> = Vec::with_capacity(kv_count as usize);
    for i in 0..kv_count {
        let key = c.string(&format!("clé de métadonnée #{i}"))?;
        let ty = c.u32(&format!("type de {key}"))?;
        let value = c.value(ty, &key, 0)?;
        if metadata.iter().any(|(k, _)| *k == key) {
            return Err(GgufError::DuplicateKey(key));
        }
        metadata.push((key, value));
    }
    let meta = |key: &str| metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    let alignment = match meta("general.alignment") {
        None => DEFAULT_ALIGNMENT,
        Some(v) => match v.as_u64() {
            Some(a) if a.is_power_of_two() => a,
            Some(a) => return Err(GgufError::Alignment(a)),
            None => return Err(GgufError::Alignment(0)),
        },
    };
    if let Some(n) = meta("split.count").and_then(MetaValue::as_u64) {
        if n > 1 {
            return Err(GgufError::Split(format!("split.count = {n}")));
        }
    }

    let mut tensors: Vec<TensorEntry> = Vec::with_capacity(tensor_count as usize);
    for i in 0..tensor_count {
        let name = c.string(&format!("nom du tenseur #{i}"))?;
        let bad = |reason: String| GgufError::Tensor {
            name: name.clone(),
            reason,
        };
        let n_dims = c.u32(&format!("n_dims de {name}"))?;
        if n_dims > MAX_DIMS {
            return Err(bad(format!("{n_dims} dimensions (maximum {MAX_DIMS})")));
        }
        let mut ne = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = c.u64(&format!("dimensions de {name}"))?;
            // Stored as int64 by ggml: the top bit set is a negative dimension.
            if dim > i64::MAX as u64 {
                return Err(bad(format!("dimension négative ({})", dim as i64)));
            }
            ne.push(dim);
        }
        let n_elements = ne
            .iter()
            .try_fold(1u64, |acc, &d| acc.checked_mul(d))
            .ok_or_else(|| bad("nombre d'éléments hors limites".into()))?;
        let ggml_type = GgmlType(c.u32(&format!("type de {name}"))?);
        let offset = c.u64(&format!("offset de {name}"))?;
        if !offset.is_multiple_of(alignment) {
            return Err(bad(format!(
                "offset {offset} non multiple de l'alignement {alignment}"
            )));
        }
        let size = match ggml_type.block() {
            Some((elems, block_bytes)) if n_elements.is_multiple_of(elems) => Some(
                (n_elements / elems)
                    .checked_mul(block_bytes)
                    .ok_or_else(|| bad("taille en octets hors limites".into()))?,
            ),
            Some((elems, _)) => {
                return Err(bad(format!(
                    "{n_elements} éléments, pas un multiple du bloc {} de {elems}",
                    ggml_type.name()
                )))
            }
            None => None,
        };
        if tensors.iter().any(|t| t.name == name) {
            return Err(GgufError::DuplicateTensor(name));
        }
        tensors.push(TensorEntry {
            name,
            ne,
            ggml_type,
            offset,
            start: 0,
            size,
        });
    }

    let file_len = bytes.len() as u64;
    let data_start = (c.pos as u64)
        .checked_next_multiple_of(alignment)
        .ok_or(GgufError::Alignment(alignment))?;
    for t in &mut tensors {
        t.start = data_start
            .checked_add(t.offset)
            .ok_or_else(|| GgufError::Tensor {
                name: t.name.clone(),
                reason: "offset hors limites".into(),
            })?;
        let end = t.start.checked_add(t.size.unwrap_or(0));
        if end.is_none_or(|end| end > file_len) {
            return Err(GgufError::Truncated {
                what: format!("données du tenseur {}", t.name),
                offset: t.start,
                need: t.size.unwrap_or(0),
                len: file_len,
            });
        }
    }
    Ok(GgufFile {
        version,
        metadata,
        tensors,
        alignment,
        data_start,
        file_len,
    })
}

/// `model-00001-of-00003.gguf`: one part of a split model.
pub fn is_split_name(file_name: &str) -> bool {
    let stem = file_name.strip_suffix(".gguf").unwrap_or(file_name);
    let mut parts = stem.rsplit('-');
    let total = parts.next().unwrap_or_default();
    let of = parts.next().unwrap_or_default();
    let index = parts.next().unwrap_or_default();
    of == "of"
        && [index, total]
            .iter()
            .all(|p| p.len() == 5 && p.bytes().all(|b| b.is_ascii_digit()))
}
