//! Codecs of the patchable GGUF types and the in-place requantization of a patched row.
//!
//! Block layouts, from `ggml-common.h` (`QK_K = 256`, `K_SCALE_SIZE = 12`); `q` is the
//! unsigned integer of an element, `l` its index in the block:
//!
//! ```text
//! Q4_0  32 elems / 18 B  : d f16 | qs u8[16]                    y = (q - 8) * d
//! Q4_1  32 elems / 20 B  : d f16 | m f16 | qs u8[16]            y = q * d + m
//! Q5_0  32 elems / 22 B  : d f16 | qh u32 | qs u8[16]           y = (q - 16) * d
//! Q5_1  32 elems / 24 B  : d f16 | m f16 | qh u32 | qs u8[16]   y = q * d + m
//!       l < 16: low nibble of qs[l], l >= 16: high nibble of qs[l - 16];
//!       Q5: bit 4 of q is bit l of qh (little-endian u32)
//! Q8_0  32 elems / 34 B  : d f16 | qs i8[32]                    y = qs[l] * d
//! Q4_K 256 elems / 144 B : d f16 | dmin f16 | scales u8[12] | qs u8[128]
//! Q5_K 256 elems / 176 B : d f16 | dmin f16 | scales u8[12] | qh u8[32] | qs u8[128]
//!       8 sub-blocks of 32, 6-bit (sc, m) packed by get_scale_min_k4:
//!         j < 4 : sc = s[j] & 63,                        m = s[j+4] & 63
//!         j >= 4: sc = (s[j+4] & 0xF) | (s[j-4] >> 6) << 4, m = (s[j+4] >> 4) | (s[j] >> 6) << 4
//!       64-element chunk c, lane i: elems 64c + i = low nibble of qs[32c + i],
//!       64c + 32 + i = high nibble; Q5_K: bit 4 is bit 2c (resp. 2c + 1) of qh[i]
//!       y = (d * sc) * q - (dmin * m)
//! Q6_K 256 elems / 210 B : ql u8[128] | qh u8[64] | scales i8[16] | d f16
//!       128-element half n, group g (0..4) of 32, lane l: 4 low bits in ql[64n + l + 32(g&1)]
//!       (low nibble for g < 2, high for g >= 2), 2 high bits in qh[32n + l] >> 2g,
//!       q = (low | high << 4) - 32, scale scales[e / 16];   y = (d * sc) * q
//! ```
//!
//! The float evaluation order of the dequantizers is the one of `dequantize_row_*` in
//! `ggml-quants.c`, so the values are those llama.cpp computes (`q * d + m` is two IEEE
//! operations: a C build that contracts it into an FMA rounds once, see the limits in the
//! module doc). Rounding: `roundf` (ties away from zero) for Q8_0 as in
//! `quantize_row_q8_0_ref`, `(int)(x + 0.5)` style truncation for Q4_0 / Q4_1 / Q5_0 / Q5_1
//! as in their `_ref` quantizers, `nearest_int` (ties to even) for the K types. Written
//! from the reference algorithms, cross-checked against candle's Rust port of
//! `ggml-quants.c` (`candle-core` `k_quants.rs`, read offline: the llama.cpp sources were
//! not available), by hand-built blocks and by a second, independent decoder in the tests.
//!
//! Requantization modes ([`Requant`]):
//! - `keep-scales` (default): `d`, `m` / `dmin` and the sub-block scales are kept byte for
//!   byte; only the integer of each element whose patched value differs is re-rounded on
//!   the existing grid, saturating (and counting) at the format bounds.
//! - `recompute`: every block holding a changed element gets new scales from its patched
//!   values. Q8_0, Q4_0, Q4_1, Q5_0 and Q5_1 are their `quantize_row_*_ref` exactly (except
//!   that Q5_1 clamps at 31 where the reference would wrap a value rounded past it). The
//!   K types use a min/max quantizer whose 6-bit (Q4_K, Q5_K) / 8-bit (Q6_K) scales and
//!   f16 super-scales are rounded UP, so every value stays inside the representable range
//!   (error <= half a step, no clipping). It is not the iterative search of
//!   `llama-quantize` (`make_qkx2_quants` / `make_qx_quants`) and uses no importance matrix.

use half::f16;
use serde::Serialize;

use super::parse::GgmlType;

/// Block-quantization patch mode.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
    clap::ValueEnum,
)]
#[serde(rename_all = "kebab-case")]
pub enum Requant {
    #[default]
    KeepScales,
    Recompute,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Codec {
    F32,
    F16,
    Bf16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q5K,
    Q6K,
}

impl Codec {
    pub fn from_type(t: GgmlType) -> Option<Codec> {
        Some(match t {
            GgmlType::F32 => Codec::F32,
            GgmlType::F16 => Codec::F16,
            GgmlType::BF16 => Codec::Bf16,
            GgmlType::Q4_0 => Codec::Q4_0,
            GgmlType::Q4_1 => Codec::Q4_1,
            GgmlType::Q5_0 => Codec::Q5_0,
            GgmlType::Q5_1 => Codec::Q5_1,
            GgmlType::Q8_0 => Codec::Q8_0,
            GgmlType::Q4_K => Codec::Q4K,
            GgmlType::Q5_K => Codec::Q5K,
            GgmlType::Q6_K => Codec::Q6K,
            _ => return None,
        })
    }

    pub fn block_elems(self) -> usize {
        match self {
            Codec::F32 | Codec::F16 | Codec::Bf16 => 1,
            Codec::Q4_0 | Codec::Q4_1 | Codec::Q5_0 | Codec::Q5_1 | Codec::Q8_0 => 32,
            Codec::Q4K | Codec::Q5K | Codec::Q6K => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Codec::F32 => 4,
            Codec::F16 | Codec::Bf16 => 2,
            Codec::Q4_0 => 18,
            Codec::Q4_1 => 20,
            Codec::Q5_0 => 22,
            Codec::Q5_1 => 24,
            Codec::Q8_0 => 34,
            Codec::Q4K => 144,
            Codec::Q5K => 176,
            Codec::Q6K => 210,
        }
    }

    /// Bytes of a row of `cols` elements (`cols` multiple of the block).
    pub fn row_bytes(self, cols: usize) -> usize {
        cols / self.block_elems() * self.block_bytes()
    }

    /// Offsets of `qh` (5-bit types) and `qs` in a 32-element block.
    fn legacy_layout(self) -> (Option<usize>, usize) {
        match self {
            Codec::Q4_0 => (None, 2),
            Codec::Q4_1 => (None, 4),
            Codec::Q5_0 => (Some(2), 6),
            _ => (Some(4), 8),
        }
    }
}

fn f16_at(b: &[u8], at: usize) -> f32 {
    f16::from_le_bytes([b[at], b[at + 1]]).to_f32()
}

/// Smallest f16 >= `x` (x >= 0), so a scale never shrinks the range it must cover.
/// `None` when it is not finite.
fn f16_ceil(x: f32) -> Option<f16> {
    let h = f16::from_f32(x);
    let h = match h.to_f32() < x {
        true => f16::from_bits(h.to_bits() + 1),
        false => h,
    };
    h.is_finite().then_some(h)
}

/// Round-to-nearest f16, `None` when not finite.
fn f16_finite(x: f32) -> Option<f16> {
    Some(f16::from_f32(x)).filter(|h| h.is_finite())
}

/// ggml's `nearest_int` (round half to even), clamped to `[lo, hi]`; `true` when clamped.
fn nearest_clamped(x: f32, lo: i32, hi: i32) -> (i32, bool) {
    clamped(x.round_ties_even(), lo, hi)
}

/// `(int)(x)` of an already offset `x + 0.5` (the legacy `_ref` quantizers), clamped.
fn floor_clamped(x: f32, lo: i32, hi: i32) -> (i32, bool) {
    clamped(x.floor(), lo, hi)
}

fn clamped(r: f32, lo: i32, hi: i32) -> (i32, bool) {
    if r < lo as f32 {
        (lo, true)
    } else if r > hi as f32 {
        (hi, true)
    } else {
        (r as i32, false)
    }
}

pub fn scale_min_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

fn pack_scale_min_k4(sc: &[u8; 8], m: &[u8; 8], s: &mut [u8]) {
    s[..12].fill(0);
    for j in 0..8 {
        if j < 4 {
            s[j] = sc[j];
            s[j + 4] = m[j];
        } else {
            s[j + 4] = (sc[j] & 0xF) | ((m[j] & 0xF) << 4);
            s[j - 4] |= (sc[j] >> 4) << 6;
            s[j] |= (m[j] >> 4) << 6;
        }
    }
}

// ------------------------------------------------------------------ unpack / pack

/// The stored integers of a block-quantized block, element order (the first
/// `block_elems` entries): unsigned for every type but Q8_0, whose bytes are `i8`.
fn unpack(codec: Codec, b: &[u8]) -> [u8; 256] {
    let mut q = [0u8; 256];
    match codec {
        Codec::Q4_0 | Codec::Q4_1 | Codec::Q5_0 | Codec::Q5_1 => {
            let (qh_at, qs_at) = codec.legacy_layout();
            let qh = qh_at.map_or(0, |at| {
                u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
            });
            let high = |l: usize| (((qh >> l) & 1) as u8) << 4;
            for l in 0..16 {
                q[l] = (b[qs_at + l] & 0xF) | high(l);
                q[l + 16] = (b[qs_at + l] >> 4) | high(l + 16);
            }
        }
        Codec::Q8_0 => q[..32].copy_from_slice(&b[2..34]),
        Codec::Q4K | Codec::Q5K => {
            let (qh, qs) = match codec {
                Codec::Q4K => (None, &b[16..144]),
                _ => (Some(&b[16..48]), &b[48..176]),
            };
            let high = |l: usize, bit: usize| qh.map_or(0, |qh| ((qh[l] >> bit) & 1) << 4);
            for c in 0..4 {
                for i in 0..32 {
                    q[64 * c + i] = (qs[32 * c + i] & 0xF) | high(i, 2 * c);
                    q[64 * c + 32 + i] = (qs[32 * c + i] >> 4) | high(i, 2 * c + 1);
                }
            }
        }
        Codec::Q6K => {
            let (ql, qh) = (&b[..128], &b[128..192]);
            for n in 0..2 {
                for l in 0..32 {
                    let (lo, hi) = (&ql[64 * n..], qh[32 * n + l]);
                    q[128 * n + l] = (lo[l] & 0xF) | ((hi & 3) << 4);
                    q[128 * n + l + 32] = (lo[l + 32] & 0xF) | (((hi >> 2) & 3) << 4);
                    q[128 * n + l + 64] = (lo[l] >> 4) | (((hi >> 4) & 3) << 4);
                    q[128 * n + l + 96] = (lo[l + 32] >> 4) | (((hi >> 6) & 3) << 4);
                }
            }
        }
        // Floats hold no packed integers: never called for them.
        Codec::F32 | Codec::F16 | Codec::Bf16 => {}
    }
    q
}

/// Inverse of [`unpack`]: writes the integers, leaves the scale fields untouched.
fn pack(codec: Codec, q: &[u8; 256], b: &mut [u8]) {
    match codec {
        Codec::Q4_0 | Codec::Q4_1 | Codec::Q5_0 | Codec::Q5_1 => {
            let (qh_at, qs_at) = codec.legacy_layout();
            let mut qh = 0u32;
            for l in 0..16 {
                b[qs_at + l] = (q[l] & 0xF) | ((q[l + 16] & 0xF) << 4);
                qh |= u32::from(q[l] >> 4) << l | u32::from(q[l + 16] >> 4) << (l + 16);
            }
            if let Some(at) = qh_at {
                b[at..at + 4].copy_from_slice(&qh.to_le_bytes());
            }
        }
        Codec::Q8_0 => b[2..34].copy_from_slice(&q[..32]),
        Codec::Q4K | Codec::Q5K => {
            let qs_at = if codec == Codec::Q4K { 16 } else { 48 };
            for c in 0..4 {
                for i in 0..32 {
                    b[qs_at + 32 * c + i] = (q[64 * c + i] & 0xF) | (q[64 * c + 32 + i] << 4);
                }
            }
            if codec == Codec::Q5K {
                for i in 0..32 {
                    b[16 + i] = (0..4).fold(0u8, |h, c| {
                        h | ((q[64 * c + i] >> 4) << (2 * c))
                            | ((q[64 * c + 32 + i] >> 4) << (2 * c + 1))
                    });
                }
            }
        }
        Codec::Q6K => {
            for n in 0..2 {
                for l in 0..32 {
                    let e = |k: usize| q[128 * n + l + 32 * k];
                    b[64 * n + l] = (e(0) & 0xF) | ((e(2) & 0xF) << 4);
                    b[64 * n + l + 32] = (e(1) & 0xF) | ((e(3) & 0xF) << 4);
                    b[128 + 32 * n + l] =
                        (e(0) >> 4) | ((e(1) >> 4) << 2) | ((e(2) >> 4) << 4) | ((e(3) >> 4) << 6);
                }
            }
        }
        Codec::F32 | Codec::F16 | Codec::Bf16 => {}
    }
}

// ------------------------------------------------------------------ dequantize

/// Dequantizes one block (`out.len() == codec.block_elems()`).
pub fn dequantize_block(codec: Codec, b: &[u8], out: &mut [f32]) {
    match codec {
        Codec::F32 => out[0] = f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        Codec::F16 => out[0] = crate::kernel::simd::f16_load([b[0], b[1]]),
        Codec::Bf16 => out[0] = crate::kernel::simd::bf16_load([b[0], b[1]]),
        Codec::Q8_0 => {
            let d = f16_at(b, 0);
            for (y, &q) in out.iter_mut().zip(&b[2..34]) {
                *y = f32::from(q as i8) * d;
            }
        }
        Codec::Q4_0 | Codec::Q5_0 => {
            let d = f16_at(b, 0);
            let zero = if codec == Codec::Q4_0 { 8 } else { 16 };
            let q = unpack(codec, b);
            for (y, &q) in out.iter_mut().zip(&q[..32]) {
                *y = f32::from(q as i8 - zero) * d;
            }
        }
        Codec::Q4_1 | Codec::Q5_1 => {
            let (d, m) = (f16_at(b, 0), f16_at(b, 2));
            let q = unpack(codec, b);
            for (y, &q) in out.iter_mut().zip(&q[..32]) {
                *y = f32::from(q) * d + m;
            }
        }
        Codec::Q4K | Codec::Q5K => {
            let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
            let l = unpack(codec, b);
            for j in 0..8 {
                let (sc, m) = scale_min_k4(j, &b[4..16]);
                let (d1, m1) = (d * f32::from(sc), dmin * f32::from(m));
                for i in 32 * j..32 * j + 32 {
                    out[i] = d1 * f32::from(l[i]) - m1;
                }
            }
        }
        Codec::Q6K => {
            let d = f16_at(b, 208);
            let q = unpack(codec, b);
            for (e, y) in out.iter_mut().enumerate().take(256) {
                let sc = f32::from(b[192 + e / 16] as i8);
                *y = d * sc * f32::from(i16::from(q[e]) - 32);
            }
        }
    }
}

/// Dequantizes a row of whole blocks.
pub fn dequantize_row(codec: Codec, bytes: &[u8], out: &mut [f32]) {
    for (b, y) in bytes
        .chunks_exact(codec.block_bytes())
        .zip(out.chunks_exact_mut(codec.block_elems()))
    {
        dequantize_block(codec, b, y);
    }
}

// ------------------------------------------------------------------ quantize (recompute)

/// A recomputed scale is not finite in f16 (the block cannot be written).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NonFiniteScale;

/// Q4_K / Q5_K min/max quantizer (`nmax` 15 or 31): writes `d`, `dmin` and the scales,
/// returns the integers.
fn quantize_k_min(x: &[f32], nmax: f32, b: &mut [u8]) -> Result<[u8; 256], NonFiniteScale> {
    let mut lo = [0.0f32; 8];
    let mut hi = [0.0f32; 8];
    for j in 0..8 {
        let sub = &x[32 * j..32 * j + 32];
        lo[j] = sub.iter().fold(0.0f32, |m, &v| m.min(v));
        hi[j] = sub.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
    }
    let max_min = lo.iter().fold(0.0f32, |m, &v| m.max(-v));
    let dmin = f16_ceil(max_min / 63.0).ok_or(NonFiniteScale)?;
    let dmin_f = dmin.to_f32();
    let mut m = [0u8; 8];
    let mut need = [0.0f32; 8];
    for j in 0..8 {
        if dmin_f > 0.0 {
            m[j] = (-lo[j] / dmin_f).ceil().min(63.0) as u8;
        }
        need[j] = ((hi[j] + dmin_f * f32::from(m[j])) / nmax).max(0.0);
    }
    let max_need = need.iter().fold(0.0f32, |a, &v| a.max(v));
    let d = f16_ceil(max_need / 63.0).ok_or(NonFiniteScale)?;
    let d_f = d.to_f32();
    let mut sc = [0u8; 8];
    if d_f > 0.0 {
        for j in 0..8 {
            sc[j] = (need[j] / d_f).ceil().min(63.0) as u8;
        }
    }
    b[..2].copy_from_slice(&d.to_le_bytes());
    b[2..4].copy_from_slice(&dmin.to_le_bytes());
    pack_scale_min_k4(&sc, &m, &mut b[4..16]);
    let mut l = [0u8; 256];
    for j in 0..8 {
        let (sc, m) = scale_min_k4(j, &b[4..16]);
        let (d1, m1) = (d_f * f32::from(sc), dmin_f * f32::from(m));
        if d1 == 0.0 {
            continue;
        }
        for i in 32 * j..32 * j + 32 {
            l[i] = nearest_clamped((x[i] + m1) / d1, 0, nmax as i32).0 as u8;
        }
    }
    Ok(l)
}

/// Fresh quantization of one block: the `recompute` mode, and the reference "fresh
/// quantization" of the fidelity checks.
pub fn quantize_block(codec: Codec, x: &[f32], b: &mut [u8]) -> Result<(), NonFiniteScale> {
    match codec {
        Codec::F32 => b[..4].copy_from_slice(&x[0].to_le_bytes()),
        Codec::F16 => b[..2].copy_from_slice(&crate::kernel::simd::f16_store(x[0])),
        Codec::Bf16 => b[..2].copy_from_slice(&crate::kernel::simd::bf16_store(x[0])),
        Codec::Q8_0 => {
            // quantize_row_q8_0_ref.
            let amax = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let dh = f16_finite(d).ok_or(NonFiniteScale)?;
            b[..2].copy_from_slice(&dh.to_le_bytes());
            for (q, &v) in b[2..34].iter_mut().zip(x) {
                *q = ((v * id).round() as i8) as u8;
            }
        }
        Codec::Q4_0 | Codec::Q5_0 => {
            // quantize_row_q{4,5}_0_ref: the signed extreme maps to -half.
            let half: f32 = if codec == Codec::Q4_0 { 8.0 } else { 16.0 };
            let (mut amax, mut max) = (0.0f32, 0.0f32);
            for &v in x {
                if amax < v.abs() {
                    amax = v.abs();
                    max = v;
                }
            }
            let d = max / -half;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            b[..2].copy_from_slice(&f16_finite(d).ok_or(NonFiniteScale)?.to_le_bytes());
            let mut q = [0u8; 256];
            for (q, &v) in q.iter_mut().zip(x) {
                *q = ((v * id + (half + 0.5)) as i8).min(2 * half as i8 - 1) as u8;
            }
            pack(codec, &q, b);
        }
        Codec::Q4_1 | Codec::Q5_1 => {
            // quantize_row_q{4,5}_1_ref.
            let top: f32 = if codec == Codec::Q4_1 { 15.0 } else { 31.0 };
            let min = x.iter().fold(f32::INFINITY, |m, &v| m.min(v));
            let max = x.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let d = (max - min) / top;
            let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            b[..2].copy_from_slice(&f16_finite(d).ok_or(NonFiniteScale)?.to_le_bytes());
            b[2..4].copy_from_slice(&f16_finite(min).ok_or(NonFiniteScale)?.to_le_bytes());
            let mut q = [0u8; 256];
            for (q, &v) in q.iter_mut().zip(x) {
                *q = (((v - min) * id + 0.5) as u8).min(top as u8);
            }
            pack(codec, &q, b);
        }
        Codec::Q4K | Codec::Q5K => {
            let nmax = if codec == Codec::Q4K { 15.0 } else { 31.0 };
            let l = quantize_k_min(x, nmax, b)?;
            pack(codec, &l, b);
        }
        Codec::Q6K => {
            let mut s = [0.0f32; 16];
            for (j, sj) in s.iter_mut().enumerate() {
                *sj = x[16 * j..16 * j + 16]
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()))
                    / 31.0;
            }
            let max_s = s.iter().fold(0.0f32, |a, &v| a.max(v));
            let d = f16_ceil(max_s / 127.0).ok_or(NonFiniteScale)?;
            let d_f = d.to_f32();
            let mut q = [32u8; 256];
            for (j, &sj) in s.iter().enumerate() {
                let sc = match d_f > 0.0 {
                    true => (sj / d_f).ceil().min(127.0) as i8,
                    false => 0,
                };
                b[192 + j] = sc as u8;
                let d1 = d_f * f32::from(sc);
                if d1 == 0.0 {
                    continue;
                }
                for i in 16 * j..16 * j + 16 {
                    q[i] = (nearest_clamped(x[i] / d1, -32, 31).0 + 32) as u8;
                }
            }
            pack(codec, &q, b);
            b[208..210].copy_from_slice(&d.to_le_bytes());
        }
    }
    Ok(())
}

// ------------------------------------------------------------------ requantize a patched row

/// Per-row outcome of [`requantize_row`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RowRequant {
    /// Elements whose integer was clamped at the format bounds.
    pub saturated: u64,
    /// A recomputed scale overflowed f16 (the row must not be written).
    pub nonfinite_scale: bool,
}

/// The stored integer of element `i` of block `b` for the value `x` on the block's own
/// grid, and whether it was clamped. `None` when the element's scale is zero (kept).
fn keep_scales_int(codec: Codec, b: &[u8], i: usize, x: f32) -> Option<(u8, bool)> {
    let nonzero = |d: f32| (d != 0.0).then_some(d);
    let (q, clamped) = match codec {
        Codec::Q8_0 => {
            // roundf, as quantize_row_q8_0_ref; +-127: the range the reference produces
            // (-128 would break the x86 dot product's sign trick).
            let d = nonzero(f16_at(b, 0))?;
            let (q, c) = clamped((x / d).round(), -127, 127);
            ((q as i8) as u8, c)
        }
        Codec::Q4_0 => floor_u8(floor_clamped(x / nonzero(f16_at(b, 0))? + 8.5, 0, 15)),
        Codec::Q5_0 => floor_u8(floor_clamped(x / nonzero(f16_at(b, 0))? + 16.5, 0, 31)),
        Codec::Q4_1 | Codec::Q5_1 => {
            let (d, m) = (nonzero(f16_at(b, 0))?, f16_at(b, 2));
            let top = if codec == Codec::Q4_1 { 15 } else { 31 };
            floor_u8(floor_clamped((x - m) / d + 0.5, 0, top))
        }
        Codec::Q4K | Codec::Q5K => {
            let (sc, m) = scale_min_k4(i / 32, &b[4..16]);
            let d1 = nonzero(f16_at(b, 0) * f32::from(sc))?;
            let m1 = f16_at(b, 2) * f32::from(m);
            let top = if codec == Codec::Q4K { 15 } else { 31 };
            floor_u8(nearest_clamped((x + m1) / d1, 0, top))
        }
        Codec::Q6K => {
            let d1 = nonzero(f16_at(b, 208) * f32::from(b[192 + i / 16] as i8))?;
            let (v, c) = nearest_clamped(x / d1, -32, 31);
            ((v + 32) as u8, c)
        }
        // Stored directly by `quantize_block`: never called for them.
        Codec::F32 | Codec::F16 | Codec::Bf16 => return None,
    };
    Some((q, clamped))
}

fn floor_u8((q, clamped): (i32, bool)) -> (u8, bool) {
    (q as u8, clamped)
}

/// Writes into `out` (a copy of the original row) the requantized patched values `exact`.
/// `before` are the dequantized original values. Elements with `exact == before` (same
/// bits) keep their stored bytes; blocks without any such change are left untouched.
pub fn requantize_row(
    codec: Codec,
    mode: Requant,
    before: &[f32],
    exact: &[f32],
    out: &mut [u8],
) -> RowRequant {
    let mut result = RowRequant::default();
    let (be, bb) = (codec.block_elems(), codec.block_bytes());
    for ((b, x0), x1) in out
        .chunks_exact_mut(bb)
        .zip(before.chunks_exact(be))
        .zip(exact.chunks_exact(be))
    {
        let changed = |i: usize| x0[i].to_bits() != x1[i].to_bits();
        if !(0..be).any(changed) {
            continue;
        }
        match (codec, mode) {
            (Codec::F32 | Codec::F16 | Codec::Bf16, _) | (_, Requant::Recompute) => {
                if quantize_block(codec, x1, b).is_err() {
                    result.nonfinite_scale = true;
                }
            }
            (_, Requant::KeepScales) => {
                let mut q = unpack(codec, b);
                for (i, &x) in x1.iter().enumerate() {
                    if let Some((v, clamped)) = changed(i)
                        .then(|| keep_scales_int(codec, b, i, x))
                        .flatten()
                    {
                        result.saturated += u64::from(clamped);
                        q[i] = v;
                    }
                }
                pack(codec, &q, b);
            }
        }
    }
    result
}

/// Every scale field of a row is finite (`d`, `m` / `dmin`; floats: the values themselves
/// are checked by the caller).
pub fn scales_finite(codec: Codec, row: &[u8]) -> bool {
    row.chunks_exact(codec.block_bytes()).all(|b| match codec {
        Codec::Q4_0 | Codec::Q5_0 | Codec::Q8_0 => f16_at(b, 0).is_finite(),
        Codec::Q4_1 | Codec::Q5_1 | Codec::Q4K | Codec::Q5K => {
            f16_at(b, 0).is_finite() && f16_at(b, 2).is_finite()
        }
        Codec::Q6K => f16_at(b, 208).is_finite(),
        Codec::F32 | Codec::F16 | Codec::Bf16 => true,
    })
}
