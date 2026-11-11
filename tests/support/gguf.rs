//! Shared by the GGUF test suites (p12, p15): a minimal GGUF v3 writer, and a second
//! decoder / keep-scales requantizer written independently of `patch::gguf::quant` (index
//! formulas instead of unpack loops). Each suite uses a subset.
#![allow(dead_code)]

use half::{bf16, f16};
use sha2::{Digest, Sha256};

use weight_surgeon::patch::gguf::{quant, Codec, GgmlType};

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    pub fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.uniform().max(1e-12), self.uniform());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

pub const F32: u32 = 0;
pub const F16: u32 = 1;
pub const Q3_K: u32 = 11;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q6_K: u32 = 14;
pub const BF16: u32 = 30;
pub const Q4_0: u32 = 2;
pub const Q4_1: u32 = 3;
pub const Q5_0: u32 = 6;
pub const Q5_1: u32 = 7;
pub const Q5_K: u32 = 13;

pub fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn f16v(b: &[u8], at: usize) -> f32 {
    f16::from_bits(u16::from_le_bytes([b[at], b[at + 1]])).to_f32()
}

pub fn codec(ty: u32) -> Codec {
    Codec::from_type(GgmlType(ty)).unwrap()
}

// ------------------------------------------------------------------ GGUF v3 writer

pub enum Kv {
    U32(u32),
    Str(&'static str),
    Strs(Vec<String>),
    I32s(Vec<i32>),
    F32(f32),
    Bool(bool),
}

pub struct T {
    pub name: String,
    pub ne: Vec<u64>,
    pub ty: u32,
    pub data: Vec<u8>,
}

pub fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

/// GGUF v3, alignment 32, tensors laid out in order, each at a multiple of 32.
pub fn gguf_bytes(kv: &[(&str, Kv)], tensors: &[T]) -> Vec<u8> {
    const ALIGN: usize = 32;
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend((kv.len() as u64).to_le_bytes());
    for (key, value) in kv {
        put_str(&mut b, key);
        match value {
            Kv::U32(x) => {
                b.extend(4u32.to_le_bytes());
                b.extend(x.to_le_bytes());
            }
            Kv::Str(s) => {
                b.extend(8u32.to_le_bytes());
                put_str(&mut b, s);
            }
            Kv::Strs(list) => {
                b.extend(9u32.to_le_bytes());
                b.extend(8u32.to_le_bytes());
                b.extend((list.len() as u64).to_le_bytes());
                list.iter().for_each(|s| put_str(&mut b, s));
            }
            Kv::I32s(list) => {
                b.extend(9u32.to_le_bytes());
                b.extend(5u32.to_le_bytes());
                b.extend((list.len() as u64).to_le_bytes());
                list.iter().for_each(|x| b.extend(x.to_le_bytes()));
            }
            Kv::F32(x) => {
                b.extend(6u32.to_le_bytes());
                b.extend(x.to_le_bytes());
            }
            Kv::Bool(x) => {
                b.extend(7u32.to_le_bytes());
                b.push(u8::from(*x));
            }
        }
    }
    let mut offsets = Vec::new();
    let mut offset = 0usize;
    for t in tensors {
        offsets.push(offset);
        offset = (offset + t.data.len()).next_multiple_of(ALIGN);
    }
    for (t, off) in tensors.iter().zip(&offsets) {
        put_str(&mut b, &t.name);
        b.extend((t.ne.len() as u32).to_le_bytes());
        t.ne.iter().for_each(|d| b.extend(d.to_le_bytes()));
        b.extend(t.ty.to_le_bytes());
        b.extend((*off as u64).to_le_bytes());
    }
    b.resize(b.len().next_multiple_of(ALIGN), 0);
    let data_start = b.len();
    for (t, off) in tensors.iter().zip(&offsets) {
        b.resize(data_start + off, 0);
        b.extend(&t.data);
    }
    b
}

pub fn metadata() -> Vec<(&'static str, Kv)> {
    vec![
        ("general.architecture", Kv::Str("qwen2")),
        ("general.name", Kv::Str("p12-fixture")),
        ("general.alignment", Kv::U32(32)),
        ("qwen2.block_count", Kv::U32(2)),
        ("qwen2.rope.freq_base", Kv::F32(1e6)),
        ("tokenizer.ggml.add_bos_token", Kv::Bool(false)),
        (
            "tokenizer.ggml.tokens",
            Kv::Strs((0..40).map(|i| format!("tok{i}")).collect()),
        ),
        ("tokenizer.ggml.token_type", Kv::I32s((0..40).collect())),
    ]
}

/// `rows x cols` weights ~ N(0, sigma²), encoded with the library's fresh quantizer.
pub fn tensor_data(ty: u32, rows: usize, cols: usize, sigma: f32, rng: &mut Rng) -> Vec<u8> {
    let c = codec(ty);
    let w: Vec<f32> = (0..rows * cols).map(|_| sigma * rng.normal()).collect();
    let mut data = vec![0u8; c.row_bytes(cols) * rows];
    for (x, b) in w
        .chunks(c.block_elems())
        .zip(data.chunks_mut(c.block_bytes()))
    {
        quant::quantize_block(c, x, b).unwrap();
    }
    data
}

// ------------------------------------------------------------------ independent decoder

/// Q4_K 6-bit (sc, m) of sub-block j, read as the spec describes the 12 bytes.
pub fn q4k_scales(s: &[u8], j: usize) -> (f32, f32) {
    let (sc, m) = match j {
        0..=3 => (s[j] & 63, s[j + 4] & 63),
        _ => (
            (s[j + 4] & 15) + 16 * (s[j - 4] >> 6),
            (s[j + 4] >> 4) + 16 * (s[j] >> 6),
        ),
    };
    (f32::from(sc), f32::from(m))
}

/// Byte index and shift of element `e`'s low 4 bits (Q4_K nibble, Q6_K low part).
pub fn q4k_nibble(e: usize) -> (usize, u32) {
    (16 + 32 * (e / 64) + e % 32, if e % 64 < 32 { 0 } else { 4 })
}

pub fn q6k_low(e: usize) -> (usize, u32) {
    let (n, r) = (e / 128, e % 128);
    let (g, l) = (r / 32, r % 32);
    (64 * n + l + 32 * (g & 1), 4 * (g as u32 >> 1))
}

pub fn q6k_high(e: usize) -> (usize, u32) {
    let (n, r) = (e / 128, e % 128);
    (128 + 32 * n + r % 32, 2 * (r as u32 / 32))
}

/// One block, element by element from index formulas.
pub fn decode_block(ty: u32, b: &[u8]) -> Vec<f32> {
    match ty {
        F32 => vec![f32::from_le_bytes([b[0], b[1], b[2], b[3]])],
        F16 => vec![f16v(b, 0)],
        BF16 => vec![bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()],
        Q8_0 => (0..32)
            .map(|j| f32::from(b[2 + j] as i8) * f16v(b, 0))
            .collect(),
        Q4_K => (0..256)
            .map(|e| {
                let (sc, m) = q4k_scales(&b[4..16], e / 32);
                let (at, shift) = q4k_nibble(e);
                let q = f32::from((b[at] >> shift) & 15);
                f16v(b, 0) * sc * q - f16v(b, 2) * m
            })
            .collect(),
        Q6_K => (0..256)
            .map(|e| {
                let ((lo_at, lo_shift), (hi_at, hi_shift)) = (q6k_low(e), q6k_high(e));
                let q = ((b[lo_at] >> lo_shift) & 15) | (((b[hi_at] >> hi_shift) & 3) << 4);
                let sc = f32::from(b[192 + e / 16] as i8);
                f16v(b, 208) * sc * (i32::from(q) - 32) as f32
            })
            .collect(),
        Q4_0 | Q4_1 | Q5_0 | Q5_1 => (0..32)
            .map(|l| {
                let q = f32::from(legacy_q(ty, b, l));
                match ty {
                    Q4_0 => (q - 8.0) * f16v(b, 0),
                    Q5_0 => (q - 16.0) * f16v(b, 0),
                    _ => q * f16v(b, 0) + f16v(b, 2),
                }
            })
            .collect(),
        Q5_K => (0..256)
            .map(|e| {
                let (sc, m) = q4k_scales(&b[4..16], e / 32);
                f16v(b, 0) * sc * f32::from(q5k_q(b, e)) - f16v(b, 2) * m
            })
            .collect(),
        other => panic!("type {other}"),
    }
}

/// Legacy 32-element blocks: offsets of `qs` and, for the 5-bit types, `qh`.
fn legacy_offsets(ty: u32) -> (usize, Option<usize>) {
    match ty {
        Q4_0 => (2, None),
        Q4_1 => (4, None),
        Q5_0 => (6, Some(2)),
        _ => (8, Some(4)),
    }
}

/// Element `l`: nibble `l / 16` of qs[l % 16]; the 5-bit types add bit `l` of the
/// little-endian u32 `qh` as bit 4.
pub fn legacy_q(ty: u32, b: &[u8], l: usize) -> u8 {
    let (qs, qh) = legacy_offsets(ty);
    let low = (b[qs + l % 16] >> (4 * (l / 16))) & 15;
    let high = qh.map_or(0, |at| (b[at + l / 8] >> (l % 8)) & 1);
    low | (high << 4)
}

pub fn set_legacy_q(ty: u32, b: &mut [u8], l: usize, q: u8) {
    let (qs, qh) = legacy_offsets(ty);
    let shift = 4 * (l / 16);
    b[qs + l % 16] = (b[qs + l % 16] & !(15 << shift)) | ((q & 15) << shift);
    if let Some(at) = qh {
        let bit = l % 8;
        b[at + l / 8] = (b[at + l / 8] & !(1 << bit)) | (((q >> 4) & 1) << bit);
    }
}

/// Q5_K element `e`: chunk `c = e / 64`, half `h`, lane `i`: nibble `h` of qs[32c + i]
/// (qs at byte 48), bit 4 is bit `2c + h` of qh[i] (qh at byte 16).
pub fn q5k_q(b: &[u8], e: usize) -> u8 {
    let (c, h, i) = (e / 64, (e % 64) / 32, e % 32);
    let low = (b[48 + 32 * c + i] >> (4 * h)) & 15;
    let high = (b[16 + i] >> (2 * c + h)) & 1;
    low | (high << 4)
}

pub fn set_q5k_q(b: &mut [u8], e: usize, q: u8) {
    let (c, h, i) = (e / 64, (e % 64) / 32, e % 32);
    let at = 48 + 32 * c + i;
    b[at] = (b[at] & !(15 << (4 * h))) | ((q & 15) << (4 * h));
    let bit = 2 * c + h;
    b[16 + i] = (b[16 + i] & !(1 << bit)) | (((q >> 4) & 1) << bit);
}

pub fn decode_tensor(ty: u32, data: &[u8]) -> Vec<f32> {
    data.chunks(codec(ty).block_bytes())
        .flat_map(|b| decode_block(ty, b))
        .collect()
}

/// Independent keep-scales requantization of one block: re-rounds the integer of every
/// changed element on the stored grid. Returns the saturated count.
pub fn requant_keep_scales(ty: u32, b: &mut [u8], before: &[f32], exact: &[f32]) -> u64 {
    let mut saturated = 0;
    for e in 0..before.len() {
        if before[e].to_bits() == exact[e].to_bits() {
            continue;
        }
        let x = exact[e];
        match ty {
            F32 => b.copy_from_slice(&x.to_le_bytes()),
            F16 => b.copy_from_slice(&f16::from_f32(x).to_le_bytes()),
            BF16 => b.copy_from_slice(&bf16::from_f32(x).to_le_bytes()),
            Q8_0 => {
                let d = f16v(b, 0);
                if d == 0.0 {
                    continue;
                }
                let r = (x / d).round();
                saturated += u64::from(r.abs() > 127.0);
                b[2 + e] = r.clamp(-127.0, 127.0) as i8 as u8;
            }
            Q4_K => {
                let (sc, m) = q4k_scales(&b[4..16], e / 32);
                let (d1, m1) = (f16v(b, 0) * sc, f16v(b, 2) * m);
                if d1 == 0.0 {
                    continue;
                }
                let r = ((x + m1) / d1).round_ties_even();
                saturated += u64::from(!(0.0..=15.0).contains(&r));
                let q = r.clamp(0.0, 15.0) as u8;
                let (at, shift) = q4k_nibble(e);
                b[at] = (b[at] & !(15 << shift)) | (q << shift);
            }
            Q6_K => {
                let d1 = f16v(b, 208) * f32::from(b[192 + e / 16] as i8);
                if d1 == 0.0 {
                    continue;
                }
                let r = (x / d1).round_ties_even();
                saturated += u64::from(!(-32.0..=31.0).contains(&r));
                let u = (r.clamp(-32.0, 31.0) as i32 + 32) as u8;
                let ((lo_at, lo_shift), (hi_at, hi_shift)) = (q6k_low(e), q6k_high(e));
                b[lo_at] = (b[lo_at] & !(15 << lo_shift)) | ((u & 15) << lo_shift);
                b[hi_at] = (b[hi_at] & !(3 << hi_shift)) | ((u >> 4) << hi_shift);
            }
            Q4_0 | Q4_1 | Q5_0 | Q5_1 => {
                let d = f16v(b, 0);
                if d == 0.0 {
                    continue;
                }
                // (int)(v + 0.5), the rounding of the legacy _ref quantizers.
                let (r, top) = match ty {
                    Q4_0 => ((x / d + 8.5).floor(), 15.0),
                    Q5_0 => ((x / d + 16.5).floor(), 31.0),
                    Q4_1 => (((x - f16v(b, 2)) / d + 0.5).floor(), 15.0),
                    _ => (((x - f16v(b, 2)) / d + 0.5).floor(), 31.0),
                };
                saturated += u64::from(!(0.0..=top).contains(&r));
                set_legacy_q(ty, b, e, r.clamp(0.0, top) as u8);
            }
            Q5_K => {
                let (sc, m) = q4k_scales(&b[4..16], e / 32);
                let (d1, m1) = (f16v(b, 0) * sc, f16v(b, 2) * m);
                if d1 == 0.0 {
                    continue;
                }
                let r = ((x + m1) / d1).round_ties_even();
                saturated += u64::from(!(0.0..=31.0).contains(&r));
                set_q5k_q(b, e, r.clamp(0.0, 31.0) as u8);
            }
            other => panic!("type {other}"),
        }
    }
    saturated
}
