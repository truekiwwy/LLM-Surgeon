//! Codecs of the patchable GGUF types and the in-place requantization of a patched row.
//!
//! Block layouts, from `ggml-common.h` (`QK_K = 256`, `K_SCALE_SIZE = 12`):
//!
//! ```text
//! Q8_0  32 elems / 34 B  : d f16 | qs i8[32]                 y = qs[j] * d
//! Q4_K 256 elems / 144 B : d f16 | dmin f16 | scales u8[12] | qs u8[128]
//!       8 sub-blocks of 32, 6-bit (sc, m) packed by get_scale_min_k4:
//!         j < 4 : sc = s[j] & 63,                        m = s[j+4] & 63
//!         j >= 4: sc = (s[j+4] & 0xF) | (s[j-4] >> 6) << 4, m = (s[j+4] >> 4) | (s[j] >> 6) << 4
//!       64-element chunk c: elems 0..32 = low nibbles of qs[32c..32c+32], 32..64 = high
//!       y = (d * sc) * q - (dmin * m)
//! Q6_K 256 elems / 210 B : ql u8[128] | qh u8[64] | scales i8[16] | d f16
//!       128-element half n, group g (0..4) of 32, lane l: 4 low bits in ql[64n + l + 32(g&1)]
//!       (low nibble for g < 2, high for g >= 2), 2 high bits in qh[32n + l] >> 2g,
//!       q = (low | high << 4) - 32, scale scales[e / 16];   y = (d * sc) * q
//! ```
//!
//! The float evaluation order of the dequantizers is the one of `dequantize_row_*` in
//! `ggml-quants.c`, so the values are those llama.cpp computes. Rounding: `roundf` (ties
//! away from zero) for Q8_0 as in `quantize_row_q8_0_ref`, `nearest_int` (ties to even) for
//! Q4_K / Q6_K. Written from the reference algorithms as known (llama.cpp sources were not
//! available offline): the layouts are cross-checked by hand-built blocks and a second,
//! independent decoder in the tests, not against the C code itself.
//!
//! Requantization modes ([`Requant`]):
//! - `keep-scales` (default): `d`, `dmin` and the sub-block scales are kept byte for byte;
//!   only the integer of each element whose patched value differs is re-rounded on the
//!   existing grid, saturating (and counting) at the format bounds.
//! - `recompute`: every block holding a changed element gets new scales from its patched
//!   values. Q8_0 is `quantize_row_q8_0_ref` exactly. Q4_K / Q6_K use a min/max quantizer
//!   whose 6-bit (Q4_K) / 8-bit (Q6_K) scales and f16 super-scales are rounded UP, so every
//!   value stays inside the representable range (error <= half a step, no clipping). It is
//!   not the iterative search of `llama-quantize` (`make_qkx2_quants` / `make_qx_quants`)
//!   and uses no importance matrix.

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
    Q8_0,
    Q4K,
    Q6K,
}

impl Codec {
    pub fn from_type(t: GgmlType) -> Option<Codec> {
        Some(match t {
            GgmlType::F32 => Codec::F32,
            GgmlType::F16 => Codec::F16,
            GgmlType::BF16 => Codec::Bf16,
            GgmlType::Q8_0 => Codec::Q8_0,
            GgmlType::Q4_K => Codec::Q4K,
            GgmlType::Q6_K => Codec::Q6K,
            _ => return None,
        })
    }

    pub fn block_elems(self) -> usize {
        match self {
            Codec::F32 | Codec::F16 | Codec::Bf16 => 1,
            Codec::Q8_0 => 32,
            Codec::Q4K | Codec::Q6K => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Codec::F32 => 4,
            Codec::F16 | Codec::Bf16 => 2,
            Codec::Q8_0 => 34,
            Codec::Q4K => 144,
            Codec::Q6K => 210,
        }
    }

    /// Bytes of a row of `cols` elements (`cols` multiple of the block).
    pub fn row_bytes(self, cols: usize) -> usize {
        cols / self.block_elems() * self.block_bytes()
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

/// ggml's `nearest_int` (round half to even), clamped to `[lo, hi]`; `true` when clamped.
fn nearest_clamped(x: f32, lo: i32, hi: i32) -> (i32, bool) {
    let r = x.round_ties_even();
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

/// The integers of a Q4_K block, element order (0..=15).
fn q4k_unpack(b: &[u8]) -> [u8; 256] {
    let qs = &b[16..144];
    let mut l = [0u8; 256];
    for c in 0..4 {
        for i in 0..32 {
            l[64 * c + i] = qs[32 * c + i] & 0xF;
            l[64 * c + 32 + i] = qs[32 * c + i] >> 4;
        }
    }
    l
}

fn q4k_pack(l: &[u8; 256], b: &mut [u8]) {
    for c in 0..4 {
        for i in 0..32 {
            b[16 + 32 * c + i] = l[64 * c + i] | (l[64 * c + 32 + i] << 4);
        }
    }
}

/// The integers of a Q6_K block, element order, unsigned (0..=63, value = q - 32).
fn q6k_unpack(b: &[u8]) -> [u8; 256] {
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
    q
}

fn q6k_pack(q: &[u8; 256], b: &mut [u8]) {
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
        Codec::Q4K => {
            let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
            let l = q4k_unpack(b);
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
            let q = q6k_unpack(b);
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
            let dh = f16::from_f32(d);
            if !dh.is_finite() {
                return Err(NonFiniteScale);
            }
            b[..2].copy_from_slice(&dh.to_le_bytes());
            for (q, &v) in b[2..34].iter_mut().zip(x) {
                *q = ((v * id).round() as i8) as u8;
            }
        }
        Codec::Q4K => {
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
                need[j] = ((hi[j] + dmin_f * f32::from(m[j])) / 15.0).max(0.0);
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
                    l[i] = nearest_clamped((x[i] + m1) / d1, 0, 15).0 as u8;
                }
            }
            q4k_pack(&l, b);
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
            q6k_pack(&q, b);
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
            (Codec::Q8_0, Requant::KeepScales) => {
                let d = f16_at(b, 0);
                if d == 0.0 {
                    continue;
                }
                for (i, &x) in x1.iter().enumerate() {
                    if changed(i) {
                        // roundf, as quantize_row_q8_0_ref; +-127: the range the reference
                        // produces (-128 would break the x86 dot product's sign trick).
                        let r = (x / d).round();
                        result.saturated += u64::from(!(-127.0..=127.0).contains(&r));
                        b[2 + i] = (r.clamp(-127.0, 127.0) as i8) as u8;
                    }
                }
            }
            (Codec::Q4K, Requant::KeepScales) => {
                let (d, dmin) = (f16_at(b, 0), f16_at(b, 2));
                let mut l = q4k_unpack(b);
                for (i, &x) in x1.iter().enumerate() {
                    let (sc, m) = scale_min_k4(i / 32, &b[4..16]);
                    let (d1, m1) = (d * f32::from(sc), dmin * f32::from(m));
                    if changed(i) && d1 != 0.0 {
                        let (q, clamped) = nearest_clamped((x + m1) / d1, 0, 15);
                        result.saturated += u64::from(clamped);
                        l[i] = q as u8;
                    }
                }
                q4k_pack(&l, b);
            }
            (Codec::Q6K, Requant::KeepScales) => {
                let d = f16_at(b, 208);
                let mut q = q6k_unpack(b);
                for (i, &x) in x1.iter().enumerate() {
                    let d1 = d * f32::from(b[192 + i / 16] as i8);
                    if changed(i) && d1 != 0.0 {
                        let (v, clamped) = nearest_clamped(x / d1, -32, 31);
                        result.saturated += u64::from(clamped);
                        q[i] = (v + 32) as u8;
                    }
                }
                q6k_pack(&q, b);
            }
        }
    }
    result
}

/// Every scale field of a row is finite (`d`, `dmin`; floats: the values themselves are
/// checked by the caller).
pub fn scales_finite(codec: Codec, row: &[u8]) -> bool {
    row.chunks_exact(codec.block_bytes()).all(|b| match codec {
        Codec::Q8_0 => f16_at(b, 0).is_finite(),
        Codec::Q4K => f16_at(b, 0).is_finite() && f16_at(b, 2).is_finite(),
        Codec::Q6K => f16_at(b, 208).is_finite(),
        Codec::F32 | Codec::F16 | Codec::Bf16 => true,
    })
}
