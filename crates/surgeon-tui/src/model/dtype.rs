//! Weight dtypes and their decoders.
//!
//! Floating dtypes decode element-wise; the llama.cpp K-quants decode per block
//! with the exact layouts used by LLM-Surgeon's patcher (`patch::gguf::quant`):
//!
//! ```text
//! Q8_0  32 elems /  34 B : d f16 | qs i8[32]                     y = qs * d
//! Q4_K 256 elems / 144 B : d f16 | dmin f16 | scales u8[12] | nibbles u8[128]
//! Q5_K 256 elems / 176 B : d f16 | dmin f16 | scales u8[12] | qh u8[32] | nibbles u8[128]
//! Q6_K 256 elems / 210 B : nibbles u8[128] | qh u8[64] | scales i8[16] | d f16
//!   y = d * scales[e / 16] * (q - 32)          (q holds 6 bits: 4 low + 2 high)
//!   Q4_K / Q5_K: y = d * sc[j] * q - dmin * m[j]   (8 sub-blocks of 32, 6-bit scales)
//! ```

use std::fmt;

/// The dtype of a tensor, as stored on disk.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE f32, little endian.
    F32,
    /// IEEE f16.
    F16,
    /// Brain float 16.
    Bf16,
    /// llama.cpp Q8_0.
    Q8_0,
    /// llama.cpp Q4_K.
    Q4K,
    /// llama.cpp Q5_K.
    Q5K,
    /// llama.cpp Q6_K.
    Q6K,
    /// Anything else (IQ quants, integer dtypes, ...): listed but not decoded.
    Other(String),
}

impl DType {
    /// Maps a GGML type id (GGUF tensor type) to a [`DType`].
    pub fn from_ggml(id: u32) -> DType {
        match id {
            0 => DType::F32,
            1 => DType::F16,
            8 => DType::Q8_0,
            12 => DType::Q4K,
            13 => DType::Q5K,
            14 => DType::Q6K,
            30 => DType::Bf16,
            other => DType::Other(ggml_name(other)),
        }
    }

    /// Short display name, as written on disk.
    pub fn name(&self) -> &str {
        match self {
            DType::F32 => "F32",
            DType::F16 => "F16",
            DType::Bf16 => "BF16",
            DType::Q8_0 => "Q8_0",
            DType::Q4K => "Q4_K",
            DType::Q5K => "Q5_K",
            DType::Q6K => "Q6_K",
            DType::Other(name) => name,
        }
    }

    /// `(elements per block, bytes per block)`; `None` for dtypes that are not decoded.
    pub fn block(&self) -> Option<(u64, u64)> {
        match self {
            DType::F32 => Some((1, 4)),
            DType::F16 | DType::Bf16 => Some((1, 2)),
            DType::Q8_0 => Some((32, 34)),
            DType::Q4K => Some((256, 144)),
            DType::Q5K => Some((256, 176)),
            DType::Q6K => Some((256, 210)),
            DType::Other(_) => None,
        }
    }

    /// Whether the metrics can decode values at all.
    pub fn decodable(&self) -> bool {
        self.block().is_some()
    }

    /// Bits per weight, for the UI.
    pub fn bits_per_weight(&self) -> Option<f64> {
        self.block()
            .map(|(elems, bytes)| 8.0 * f64::from(bytes as u32) / f64::from(elems as u32))
    }

    /// Decodes whole blocks into `out`: `out.len()` must be a multiple of the
    /// block element count and the byte slice must cover the matching blocks.
    /// Missing trailing bytes decode as zero instead of panicking.
    pub fn decode_blocks(&self, bytes: &[u8], out: &mut [f32]) {
        let Some((elems, block_bytes)) = self.block() else {
            out.fill(0.0);
            return;
        };
        let (elems, block_bytes) = (elems as usize, block_bytes as usize);
        let mut block = vec![0.0f32; elems];
        for (b, y) in bytes
            .chunks(block_bytes)
            .zip(out.chunks_mut(elems))
        {
            if b.len() == block_bytes {
                decode_block(self, b, y);
            } else {
                // Truncated tail: decode what is there into the scratch block,
                // padded with zeros, then copy.
                let mut padded = b.to_vec();
                padded.resize(block_bytes, 0);
                decode_block(self, &padded, &mut block);
                y.copy_from_slice(&block[..y.len()]);
            }
        }
    }

    /// Decodes the element range `[start, start + count)` of the flattened
    /// tensor into `out` (`out.len() == count as usize`).
    pub fn decode_range(&self, bytes: &[u8], start: u64, count: u64, out: &mut Vec<f32>) {
        out.clear();
        if !self.decodable() {
            out.resize(count as usize, 0.0);
            return;
        }
        let (elems, _) = self.block().expect("decodable");
        if count == 0 {
            return;
        }
        let last = start + count - 1;
        if self == &DType::F32 {
            for i in start..=last {
                out.push(read_f32(bytes, i * 4));
            }
            return;
        }
        if matches!(self, DType::F16 | DType::Bf16) {
            for i in start..=last {
                out.push(read_half(self, bytes, i * 2));
            }
            return;
        }
        let first_block = start / elems;
        let last_block = last / elems;
        let mut scratch = vec![0.0f32; elems as usize];
        for b in first_block..=last_block {
            decode_one_block(self, bytes, b, &mut scratch);
            for i in 0..elems {
                let e = b * elems + i;
                if e >= start && e <= last {
                    out.push(scratch[i as usize]);
                }
            }
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

fn decode_one_block(dtype: &DType, bytes: &[u8], block: u64, out: &mut [f32]) {
    let (_, block_bytes) = dtype.block().expect("decodable");
    let from = (block * block_bytes) as usize;
    let to = ((block + 1) * block_bytes) as usize;
    if from >= bytes.len() {
        out.fill(0.0);
        return;
    }
    let end = to.min(bytes.len());
    let mut padded = bytes[from..end].to_vec();
    padded.resize(block_bytes as usize, 0);
    decode_block(dtype, &padded, out);
}

/// GGML type names for the ids the TUI lists but does not decode.
fn ggml_name(id: u32) -> String {
    const NAMES: [&str; 31] = [
        "F32", "F16", "Q4_0", "Q4_1", "type4", "type5", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "Q2_K",
        "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL",
        "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M", "BF16",
    ];
    match NAMES.get(id as usize) {
        Some(name) => (*name).to_owned(),
        None => format!("type{id}"),
    }
}

fn read_f32(bytes: &[u8], at: u64) -> f32 {
    let at = at as usize;
    if at + 4 <= bytes.len() {
        f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    } else {
        0.0
    }
}

fn read_half(dtype: &DType, bytes: &[u8], at: u64) -> f32 {
    let at = at as usize;
    if at + 2 > bytes.len() {
        return 0.0;
    }
    let bits = u16::from_le_bytes([bytes[at], bytes[at + 1]]);
    match dtype {
        DType::F16 => half::f16::from_bits(bits).to_f32(),
        _ => half::bf16::from_bits(bits).to_f32(),
    }
}

fn f16_at(b: &[u8], at: usize) -> f32 {
    if at + 2 <= b.len() {
        half::f16::from_le_bytes([b[at], b[at + 1]]).to_f32()
    } else {
        0.0
    }
}

/// The 6-bit `(scale, min)` pair `j` of a Q4_K / Q5_K block, packed in 12 bytes.
fn scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j - 4] >> 6) << 4),
        )
    }
}

fn decode_block(dtype: &DType, b: &[u8], out: &mut [f32]) {
    match dtype {
        DType::F32 => out[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        DType::F16 => out[0] = half::f16::from_le_bytes([b[0], b[1]]).to_f32(),
        DType::Bf16 => out[0] = half::bf16::from_le_bytes([b[0], b[1]]).to_f32(),
        DType::Q8_0 => {
            let d = f16_at(b, 0);
            for (y, &q) in out.iter_mut().zip(&b[2..34]) {
                *y = f32::from(q as i8) * d;
            }
        }
        DType::Q4K | DType::Q5K => {
            let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
            let (qh, qs) = match dtype {
                DType::Q4K => (None, &b[16..144]),
                _ => (Some(&b[16..48]), &b[48..176]),
            };
            let mut q = [0u8; 256];
            let high = |l: usize, bit: usize| qh.map_or(0, |qh| ((qh[l] >> bit) & 1) << 4);
            for c in 0..4 {
                for i in 0..32 {
                    q[64 * c + i] = (qs[32 * c + i] & 0xF) | high(i, 2 * c);
                    q[64 * c + 32 + i] = (qs[32 * c + i] >> 4) | high(i, 2 * c + 1);
                }
            }
            for j in 0..8 {
                let (sc, m) = scale_min_k4(j, &b[4..16]);
                let (d1, m1) = (d * f32::from(sc), dmin * f32::from(m));
                for i in 32 * j..32 * j + 32 {
                    out[i] = d1 * f32::from(q[i]) - m1;
                }
            }
        }
        DType::Q6K => {
            let d = f16_at(b, 208);
            let (ql, qh) = (&b[..128], &b[128..192]);
            let mut q = [0u8; 256];
            for n in 0..2 {
                for l in 0..32 {
                    let (lo, hi) = (&ql[64 * n..], qh[32 * n + l]);
                    q[128 * n + l] = (lo[l] & 0xF) | ((hi & 3) << 4);
                    q[128 * n + l + 32] = (lo[l + 32] & 0xF) | (((hi >> 2) & 3) << 4);
                    q[128 * n + l + 64] = (lo[l] >> 4) | (((hi >> 4) & 3) << 4);
                    q[128 * n + l + 96] = (lo[l + 32] >> 4) | (((hi >> 6) & 3) << 4);
                }
            }
            for (e, y) in out.iter_mut().enumerate().take(256) {
                let sc = f32::from(b[192 + e / 16] as i8);
                *y = d * sc * f32::from(i16::from(q[e]) - 32);
            }
        }
        DType::Other(_) => out.fill(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_and_halves_decode_element_wise() {
        let bytes: Vec<u8> = [1.5f32, -2.25]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut out = Vec::new();
        DType::F32.decode_range(&bytes, 0, 2, &mut out);
        assert_eq!(out, vec![1.5, -2.25]);

        let f16 = half::f16::from_f32(0.75);
        let bytes = f16.to_le_bytes().to_vec();
        let mut out = Vec::new();
        DType::F16.decode_range(&bytes, 0, 1, &mut out);
        assert_eq!(out, vec![0.75]);

        let bf16 = half::bf16::from_f32(-3.5);
        let bytes = bf16.to_le_bytes().to_vec();
        let mut out = Vec::new();
        DType::Bf16.decode_range(&bytes, 0, 1, &mut out);
        assert_eq!(out, vec![-3.5]);
    }

    #[test]
    fn q8_0_block_is_scale_times_integers() {
        let mut b = vec![0u8; 34];
        b[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        for (i, q) in b[2..34].iter_mut().enumerate() {
            *q = (i as i8 + 1) as u8;
        }
        let mut out = [0.0f32; 32];
        DType::Q8_0.decode_blocks(&b, &mut out);
        for (i, y) in out.iter().enumerate() {
            assert_eq!(*y, 0.5 * (i + 1) as f32);
        }
    }

    /// Independent Q4_K reference: unpacks the nibbles, then applies
    /// `y[i] = d * sc[j] * q[i] - dmin * m[j]` with the 6-bit scales.
    fn reference_q4k(b: &[u8]) -> [f32; 256] {
        let d = half::f16::from_le_bytes([b[0], b[1]]).to_f32();
        let dmin = half::f16::from_le_bytes([b[2], b[3]]).to_f32();
        let mut out = [0.0f32; 256];
        for j in 0..8 {
            let s = &b[4..16];
            let (sc, m) = if j < 4 {
                (s[j] & 63, s[j + 4] & 63)
            } else {
                (
                    (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
                    (s[j + 4] >> 4) | ((s[j - 4] >> 6) << 4),
                )
            };
            for i in 32 * j..32 * j + 32 {
                let byte = b[16 + 32 * (i / 64) + (i % 32)];
                let q = if (i / 32) % 2 == 0 {
                    byte & 0xF
                } else {
                    byte >> 4
                };
                out[i] = d * f32::from(sc) * f32::from(q) - dmin * f32::from(m);
            }
        }
        out
    }

    #[test]
    fn q4_k_matches_an_independent_reference() {
        let mut b = vec![0u8; 144];
        b[..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        b[2..4].copy_from_slice(&half::f16::from_f32(0.25).to_le_bytes());
        for (i, v) in b[4..16].iter_mut().enumerate() {
            *v = (i * 7 + 3) as u8;
        }
        for (i, v) in b[16..144].iter_mut().enumerate() {
            *v = (i * 13 + 5) as u8;
        }
        let mut out = [0.0f32; 256];
        DType::Q4K.decode_blocks(&b, &mut out);
        let reference = reference_q4k(&b);
        assert_eq!(out, reference);
    }

    #[test]
    fn q6_k_matches_an_independent_reference() {
        let mut b = vec![0u8; 210];
        b[208..210].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        for (i, v) in b[..128].iter_mut().enumerate() {
            *v = (i * 3 + 1) as u8;
        }
        for (i, v) in b[128..192].iter_mut().enumerate() {
            *v = (i * 5 + 2) as u8;
        }
        for (i, v) in b[192..208].iter_mut().enumerate() {
            *v = (i + 1) as u8;
        }
        let mut out = [0.0f32; 256];
        DType::Q6K.decode_blocks(&b, &mut out);
        // Reference: unpack the 6-bit integers (4 low bits from `ql`, 2 high
        // bits from `qh`), then `y[e] = d * scales[e / 16] * (q - 32)`.
        let mut reference = [0.0f32; 256];
        for n in 0..2 {
            for l in 0..32 {
                let (lo, hi) = (&b[64 * n..], b[128 + 32 * n + l]);
                let parts = [
                    (lo[l] & 0xF, (hi & 3) << 4, 0),
                    (lo[l + 32] & 0xF, ((hi >> 2) & 3) << 4, 32),
                    (lo[l] >> 4, ((hi >> 4) & 3) << 4, 64),
                    (lo[l + 32] >> 4, ((hi >> 6) & 3) << 4, 96),
                ];
                for (low, high, offset) in parts {
                    let e = 128 * n + l + offset;
                    let q = i16::from(low | high) - 32;
                    reference[e] = 2.0 * f32::from(b[192 + e / 16] as i8) * f32::from(q);
                }
            }
        }
        assert_eq!(out, reference);
    }

    #[test]
    fn block_layouts_follow_the_spec() {
        assert_eq!(DType::Q8_0.block(), Some((32, 34)));
        assert_eq!(DType::Q4K.block(), Some((256, 144)));
        assert_eq!(DType::Q5K.block(), Some((256, 176)));
        assert_eq!(DType::Q6K.block(), Some((256, 210)));
        assert_eq!(DType::F32.bits_per_weight(), Some(32.0));
        assert_eq!(DType::Q4K.bits_per_weight(), Some(4.5));
        assert!(!DType::Other("IQ2_XXS".into()).decodable());
        assert_eq!(DType::from_ggml(30), DType::Bf16);
        assert_eq!(DType::from_ggml(12), DType::Q4K);
        assert_eq!(DType::from_ggml(99), DType::Other("type99".into()));
    }
}
