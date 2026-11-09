//! Row kernels for `W[r][c] += delta`, where `delta = sign * (alpha * v[.])` is precomputed
//! once per vector element by the engine.
//!
//! Bit-exactness contract: every level writes exactly the bytes of [`shift_row_scalar`] and
//! returns the same [`ShiftStats`]. This holds because each element costs a single IEEE
//! addition `old + delta`, done in f32 by every level, followed by the same rounding:
//! - bf16: round to nearest even with `half`'s NaN rule (`(x >> 16) | 0x40`), rebuilt from
//!   integer operations;
//! - f16: the F16C / AVX-512 conversions with `_MM_FROUND_TO_NEAREST_INT`, the instruction
//!   `half` itself uses on x86_64; NEON keeps the scalar `half` conversion for f16, because the
//!   NEON f16 intrinsics are not stable in Rust.
//!
//! No FMA anywhere: the multiplication `alpha * v` happens once per vector element (scalar,
//! in the engine), the kernel only adds. A fused multiply-add would round once instead of
//! twice and drift from the scalar reference; the operation is memory bound anyway.
//!
//! Alignment: tensor data in a mapped file is only 8-byte aligned (SafeTensors header) or
//! 32-byte aligned (GGUF), so every vector access is an unaligned load/store
//! (`loadu`/`storeu`, `vld1q`), with no alignment assumption.

use half::{bf16, f16};
use serde::Serialize;

/// Element type of a tensor the kernels can shift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Elem {
    F32,
    F16,
    Bf16,
}

impl Elem {
    pub fn size(self) -> usize {
        match self {
            Elem::F32 => 4,
            Elem::F16 | Elem::Bf16 => 2,
        }
    }
}

/// Deltas of a row-major `[rows, cols]` slice, already `sign * (alpha * v[i])`.
#[derive(Debug, Clone, Copy)]
pub enum Deltas<'a> {
    /// `len == cols`: element `c` of every row receives `deltas[c]` (engine axis `Rows`).
    PerCol(&'a [f32]),
    /// `len == rows`: every element of row `r` receives `deltas[r]` (engine axis `Cols`).
    PerRow(&'a [f32]),
}

/// Deltas of one row.
#[derive(Debug, Clone, Copy)]
pub enum RowDeltas<'a> {
    Splat(f32),
    Each(&'a [f32]),
}

impl<'a> RowDeltas<'a> {
    /// The deltas of elements `from..`.
    fn skip(self, from: usize) -> RowDeltas<'a> {
        match self {
            RowDeltas::Splat(d) => RowDeltas::Splat(d),
            RowDeltas::Each(ds) => RowDeltas::Each(ds.get(from..).unwrap_or(&[])),
        }
    }

    /// Number of elements a row of `n` cells can actually shift (zip semantics).
    fn usable(self, n: usize) -> usize {
        match self {
            RowDeltas::Splat(_) => n,
            RowDeltas::Each(ds) => n.min(ds.len()),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct ShiftStats {
    /// Elements whose stored bits actually changed (a delta lost to rounding does not count).
    pub elements_modified: u64,
    /// Largest |new - old| over finite results, measured after rounding to the dtype.
    pub max_abs_delta: f32,
    /// Finite weights that became NaN or Inf (e.g. f16 overflow).
    pub nan_or_inf_produced: u64,
}

impl ShiftStats {
    /// Exact and order-independent (sums of integers, max of non-NaN floats).
    pub fn merge(self, other: Self) -> Self {
        Self {
            elements_modified: self.elements_modified + other.elements_modified,
            max_abs_delta: self.max_abs_delta.max(other.max_abs_delta),
            nan_or_inf_produced: self.nan_or_inf_produced + other.nan_or_inf_produced,
        }
    }
}

// Little-endian byte arrays, never pointer casts: tensor data has no alignment guarantee.
pub fn bf16_load(b: [u8; 2]) -> f32 {
    bf16::from_le_bytes(b).to_f32()
}
pub fn bf16_store(x: f32) -> [u8; 2] {
    bf16::from_f32(x).to_le_bytes() // round to nearest, ties to even
}
pub fn f16_load(b: [u8; 2]) -> f32 {
    f16::from_le_bytes(b).to_f32()
}
pub fn f16_store(x: f32) -> [u8; 2] {
    f16::from_f32(x).to_le_bytes()
}

/// The scalar reference: one element at a time, `store(load(cell) + delta)`.
#[inline]
pub fn shift_cells<const N: usize>(
    cells: &mut [[u8; N]],
    deltas: impl Iterator<Item = f32>,
    load: &impl Fn([u8; N]) -> f32,
    store: &impl Fn(f32) -> [u8; N],
) -> ShiftStats {
    let mut stats = ShiftStats::default();
    for (cell, delta) in cells.iter_mut().zip(deltas) {
        let old = load(*cell);
        let new_bits = store(old + delta);
        if new_bits != *cell {
            let new = load(new_bits);
            stats.elements_modified += 1;
            if new.is_finite() {
                stats.max_abs_delta = stats.max_abs_delta.max((new - old).abs());
            } else if old.is_finite() {
                stats.nan_or_inf_produced += 1;
            }
            *cell = new_bits;
        }
    }
    stats
}

/// Scalar row kernel, also the tail of every SIMD kernel. A trailing partial cell (row
/// length not a multiple of the element size) is left untouched.
pub fn shift_row_scalar(row: &mut [u8], elem: Elem, d: RowDeltas<'_>) -> ShiftStats {
    fn cells<const N: usize>(
        row: &mut [u8],
        d: RowDeltas<'_>,
        load: &impl Fn([u8; N]) -> f32,
        store: &impl Fn(f32) -> [u8; N],
    ) -> ShiftStats {
        let (cells, _) = row.as_chunks_mut::<N>();
        match d {
            RowDeltas::Splat(x) => shift_cells(cells, std::iter::repeat(x), load, store),
            RowDeltas::Each(ds) => shift_cells(cells, ds.iter().copied(), load, store),
        }
    }
    match elem {
        Elem::Bf16 => cells(row, d, &bf16_load, &bf16_store),
        Elem::F16 => cells(row, d, &f16_load, &f16_store),
        Elem::F32 => cells(row, d, &f32::from_le_bytes, &f32::to_le_bytes),
    }
}

/// Splits a row into the SIMD body (`lanes`-wide groups) and the scalar tail, so that the
/// vector loop never reads or writes past `n` cells, nor past the deltas.
fn body_len(row: &[u8], elem: Elem, d: RowDeltas<'_>, lanes: usize) -> (usize, usize) {
    let n = d.usable(row.len() / elem.size());
    (n - n % lanes, n)
}

/// Shifts the tail `[from, n)` with the scalar kernel.
fn tail(row: &mut [u8], elem: Elem, d: RowDeltas<'_>, from: usize, n: usize) -> ShiftStats {
    let size = elem.size();
    match row.get_mut(from * size..n * size) {
        Some(rest) => shift_row_scalar(rest, elem, d.skip(from)),
        None => ShiftStats::default(),
    }
}

#[cfg(target_arch = "x86_64")]
pub(super) mod x86 {
    use std::arch::x86_64::*;

    use super::{body_len, tail, Elem, RowDeltas, ShiftStats};

    const EXP_MASK: i32 = 0x7F80_0000;

    /// Running statistics of a vector loop, folded into `ShiftStats` at the end.
    struct Acc {
        modified: u64,
        nan_or_inf: u64,
        max: [f32; 16],
    }

    impl Acc {
        fn finish(self, tail: ShiftStats) -> ShiftStats {
            let max = self.max.iter().fold(0.0f32, |m, &x| m.max(x));
            ShiftStats {
                elements_modified: self.modified,
                max_abs_delta: max,
                nan_or_inf_produced: self.nan_or_inf,
            }
            .merge(tail)
        }
    }

    // ------------------------------------------------------------------ AVX2 + F16C

    /// f32 bits of 8 lanes -> bf16 bits in the low 16 bits of each u32 lane, exactly as
    /// `half::bf16::from_f32`: NaN keeps its high half with the quiet bit set, everything
    /// else rounds to nearest even (`x + 0x7FFF + lsb`, then `>> 16`).
    #[target_feature(enable = "avx2")]
    fn bf16_round_avx2(x: __m256) -> __m256i {
        let bits = _mm256_castps_si256(x);
        let high = _mm256_srli_epi32::<16>(bits);
        let lsb = _mm256_and_si256(high, _mm256_set1_epi32(1));
        let bias = _mm256_add_epi32(_mm256_set1_epi32(0x7FFF), lsb);
        let rounded = _mm256_srli_epi32::<16>(_mm256_add_epi32(bits, bias));
        let quiet = _mm256_or_si256(high, _mm256_set1_epi32(0x40));
        let abs = _mm256_and_si256(bits, _mm256_set1_epi32(0x7FFF_FFFF));
        let is_nan = _mm256_cmpgt_epi32(abs, _mm256_set1_epi32(EXP_MASK));
        _mm256_blendv_epi8(rounded, quiet, is_nan)
    }

    /// Stats of 8 lanes: `changed` is all-ones where the stored bits changed.
    #[target_feature(enable = "avx2")]
    fn stats_avx2(acc: &mut Acc, max: &mut __m256, old: __m256, new: __m256, changed: __m256i) {
        let exp = _mm256_set1_epi32(EXP_MASK);
        let old_bad = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(old), exp), exp);
        let new_bad = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_castps_si256(new), exp), exp);
        let produced = _mm256_andnot_si256(old_bad, _mm256_and_si256(changed, new_bad));
        acc.modified += u64::from(_mm256_movemask_ps(_mm256_castsi256_ps(changed)).count_ones());
        acc.nan_or_inf += u64::from(_mm256_movemask_ps(_mm256_castsi256_ps(produced)).count_ones());
        let sign = _mm256_set1_ps(-0.0);
        let diff = _mm256_andnot_ps(sign, _mm256_sub_ps(new, old));
        // Non-finite results do not count; unchanged cells have diff == 0.
        let diff = _mm256_andnot_ps(_mm256_castsi256_ps(new_bad), diff);
        *max = _mm256_max_ps(*max, diff);
    }

    #[target_feature(enable = "avx2")]
    fn delta_avx2(d: RowDeltas<'_>, i: usize) -> __m256 {
        match d {
            RowDeltas::Splat(x) => _mm256_set1_ps(x),
            RowDeltas::Each(ds) => match ds.get(i..i + 8) {
                // SAFETY: `get` proved that the 8 floats [i, i+8) are inside `ds`;
                // `loadu` has no alignment requirement.
                Some(s) => unsafe { _mm256_loadu_ps(s.as_ptr()) },
                None => _mm256_setzero_ps(),
            },
        }
    }

    /// AVX2 (+F16C for f16) row kernel, 8 elements per step.
    #[target_feature(enable = "avx2,f16c")]
    pub(in crate::kernel) fn row_avx2(row: &mut [u8], elem: Elem, d: RowDeltas<'_>) -> ShiftStats {
        let (body, n) = body_len(row, elem, d, 8);
        let mut acc = Acc {
            modified: 0,
            nan_or_inf: 0,
            max: [0.0; 16],
        };
        let mut max = _mm256_setzero_ps();
        let size = elem.size();
        let mut i = 0;
        while i < body {
            let Some(cell) = row.get_mut(i * size..(i + 8) * size) else {
                break;
            };
            let p = cell.as_mut_ptr();
            let delta = delta_avx2(d, i);
            match elem {
                Elem::F32 => {
                    // SAFETY: `cell` is exactly 32 bytes (8 f32) of `row`, unaligned load.
                    let old = unsafe { _mm256_loadu_ps(p.cast::<f32>()) };
                    let new = _mm256_add_ps(old, delta);
                    let same =
                        _mm256_cmpeq_epi32(_mm256_castps_si256(old), _mm256_castps_si256(new));
                    let changed = _mm256_xor_si256(same, _mm256_set1_epi32(-1));
                    stats_avx2(&mut acc, &mut max, old, new, changed);
                    // SAFETY: same 32 bytes, exclusively borrowed through `cell`.
                    unsafe { _mm256_storeu_ps(p.cast::<f32>(), new) };
                }
                Elem::Bf16 => {
                    // SAFETY: `cell` is exactly 16 bytes (8 bf16) of `row`, unaligned load.
                    let raw = unsafe { _mm_loadu_si128(p.cast::<__m128i>()) };
                    let old_bits = _mm256_cvtepu16_epi32(raw);
                    let old = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(old_bits));
                    let new_bits = bf16_round_avx2(_mm256_add_ps(old, delta));
                    let new = _mm256_castsi256_ps(_mm256_slli_epi32::<16>(new_bits));
                    let same = _mm256_cmpeq_epi32(old_bits, new_bits);
                    let changed = _mm256_xor_si256(same, _mm256_set1_epi32(-1));
                    stats_avx2(&mut acc, &mut max, old, new, changed);
                    // u32 lanes hold values <= 0xFFFF: packus never saturates; the permute
                    // gathers the two 64-bit halves holding lanes 0..4 and 4..8.
                    let packed = _mm256_packus_epi32(new_bits, new_bits);
                    let packed = _mm256_permute4x64_epi64::<0b00_00_10_00>(packed);
                    // SAFETY: same 16 bytes, exclusively borrowed through `cell`.
                    unsafe {
                        _mm_storeu_si128(p.cast::<__m128i>(), _mm256_castsi256_si128(packed))
                    };
                }
                Elem::F16 => {
                    // SAFETY: `cell` is exactly 16 bytes (8 f16) of `row`, unaligned load.
                    let raw = unsafe { _mm_loadu_si128(p.cast::<__m128i>()) };
                    let old = _mm256_cvtph_ps(raw);
                    let new_raw =
                        _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm256_add_ps(old, delta));
                    let new = _mm256_cvtph_ps(new_raw);
                    let same = _mm256_cvtepi16_epi32(_mm_cmpeq_epi16(raw, new_raw));
                    let changed = _mm256_xor_si256(same, _mm256_set1_epi32(-1));
                    stats_avx2(&mut acc, &mut max, old, new, changed);
                    // SAFETY: same 16 bytes, exclusively borrowed through `cell`.
                    unsafe { _mm_storeu_si128(p.cast::<__m128i>(), new_raw) };
                }
            }
            i += 8;
        }
        let mut lanes = [0.0f32; 8];
        // SAFETY: `lanes` is a local array of exactly 8 f32.
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), max) };
        acc.max[..8].copy_from_slice(&lanes);
        acc.finish(tail(row, elem, d, i, n))
    }

    /// Test hook: the AVX2 / F16C conversions of 8 f32 to (bf16 bits, f16 bits).
    #[cfg(test)]
    #[target_feature(enable = "avx2,f16c")]
    pub(in crate::kernel) fn convert_avx2(xs: &[f32; 8]) -> ([u16; 8], [u16; 8]) {
        // SAFETY: `xs` is exactly 8 f32.
        let x = unsafe { _mm256_loadu_ps(xs.as_ptr()) };
        let mut bf = [0u32; 8];
        // SAFETY: `bf` is exactly 8 u32 (32 bytes).
        unsafe { _mm256_storeu_si256(bf.as_mut_ptr().cast::<__m256i>(), bf16_round_avx2(x)) };
        let mut f = [0u16; 8];
        // SAFETY: `f` is exactly 8 u16 (16 bytes).
        unsafe {
            _mm_storeu_si128(
                f.as_mut_ptr().cast::<__m128i>(),
                _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x),
            )
        };
        (bf.map(|b| b as u16), f)
    }

    /// Test hook: the AVX-512 conversions of 16 f32 to (bf16 bits, f16 bits).
    #[cfg(test)]
    #[target_feature(enable = "avx512f")]
    pub(in crate::kernel) fn convert_avx512(xs: &[f32; 16]) -> ([u16; 16], [u16; 16]) {
        // SAFETY: `xs` is exactly 16 f32.
        let x = unsafe { _mm512_loadu_ps(xs.as_ptr()) };
        let mut bf = [0u16; 16];
        // SAFETY: `bf` is exactly 16 u16 (32 bytes).
        unsafe {
            _mm256_storeu_si256(
                bf.as_mut_ptr().cast::<__m256i>(),
                _mm512_cvtepi32_epi16(bf16_round_avx512(x)),
            )
        };
        let mut f = [0u16; 16];
        // SAFETY: `f` is exactly 16 u16 (32 bytes).
        unsafe {
            _mm256_storeu_si256(
                f.as_mut_ptr().cast::<__m256i>(),
                _mm512_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(x),
            )
        };
        (bf, f)
    }

    // ------------------------------------------------------------------ AVX-512F

    #[target_feature(enable = "avx512f")]
    fn bf16_round_avx512(x: __m512) -> __m512i {
        let bits = _mm512_castps_si512(x);
        let high = _mm512_srli_epi32::<16>(bits);
        let lsb = _mm512_and_si512(high, _mm512_set1_epi32(1));
        let bias = _mm512_add_epi32(_mm512_set1_epi32(0x7FFF), lsb);
        let rounded = _mm512_srli_epi32::<16>(_mm512_add_epi32(bits, bias));
        let quiet = _mm512_or_si512(high, _mm512_set1_epi32(0x40));
        let abs = _mm512_and_si512(bits, _mm512_set1_epi32(0x7FFF_FFFF));
        let is_nan = _mm512_cmpgt_epi32_mask(abs, _mm512_set1_epi32(EXP_MASK));
        _mm512_mask_blend_epi32(is_nan, rounded, quiet)
    }

    #[target_feature(enable = "avx512f")]
    fn stats_avx512(acc: &mut Acc, max: &mut __m512, old: __m512, new: __m512, changed: u16) {
        let exp = _mm512_set1_epi32(EXP_MASK);
        let old_bad = _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(old), exp), exp);
        let new_bad = _mm512_cmpeq_epi32_mask(_mm512_and_si512(_mm512_castps_si512(new), exp), exp);
        acc.modified += u64::from(changed.count_ones());
        acc.nan_or_inf += u64::from((changed & new_bad & !old_bad).count_ones());
        let diff = _mm512_abs_ps(_mm512_sub_ps(new, old));
        *max = _mm512_mask_max_ps(*max, !new_bad, *max, diff);
    }

    #[target_feature(enable = "avx512f")]
    fn delta_avx512(d: RowDeltas<'_>, i: usize) -> __m512 {
        match d {
            RowDeltas::Splat(x) => _mm512_set1_ps(x),
            RowDeltas::Each(ds) => match ds.get(i..i + 16) {
                // SAFETY: `get` proved that the 16 floats [i, i+16) are inside `ds`.
                Some(s) => unsafe { _mm512_loadu_ps(s.as_ptr()) },
                None => _mm512_setzero_ps(),
            },
        }
    }

    /// AVX-512F row kernel, 16 elements per step (no AVX-512BW/VL needed: every lane
    /// operation is on 32 bits, 16-bit values are widened/narrowed with `vpmovzxwd`/`vpmovdw`).
    #[target_feature(enable = "avx512f")]
    pub(in crate::kernel) fn row_avx512(
        row: &mut [u8],
        elem: Elem,
        d: RowDeltas<'_>,
    ) -> ShiftStats {
        let (body, n) = body_len(row, elem, d, 16);
        let mut acc = Acc {
            modified: 0,
            nan_or_inf: 0,
            max: [0.0; 16],
        };
        let mut max = _mm512_setzero_ps();
        let size = elem.size();
        let mut i = 0;
        while i < body {
            let Some(cell) = row.get_mut(i * size..(i + 16) * size) else {
                break;
            };
            let p = cell.as_mut_ptr();
            let delta = delta_avx512(d, i);
            match elem {
                Elem::F32 => {
                    // SAFETY: `cell` is exactly 64 bytes (16 f32) of `row`, unaligned load.
                    let old = unsafe { _mm512_loadu_ps(p.cast::<f32>()) };
                    let new = _mm512_add_ps(old, delta);
                    let changed = _mm512_cmpneq_epi32_mask(
                        _mm512_castps_si512(old),
                        _mm512_castps_si512(new),
                    );
                    stats_avx512(&mut acc, &mut max, old, new, changed);
                    // SAFETY: same 64 bytes, exclusively borrowed through `cell`.
                    unsafe { _mm512_storeu_ps(p.cast::<f32>(), new) };
                }
                Elem::Bf16 => {
                    // SAFETY: `cell` is exactly 32 bytes (16 bf16) of `row`, unaligned load.
                    let raw = unsafe { _mm256_loadu_si256(p.cast::<__m256i>()) };
                    let old_bits = _mm512_cvtepu16_epi32(raw);
                    let old = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(old_bits));
                    let new_bits = bf16_round_avx512(_mm512_add_ps(old, delta));
                    let new = _mm512_castsi512_ps(_mm512_slli_epi32::<16>(new_bits));
                    let changed = _mm512_cmpneq_epi32_mask(old_bits, new_bits);
                    stats_avx512(&mut acc, &mut max, old, new, changed);
                    // SAFETY: same 32 bytes; lanes hold values <= 0xFFFF, so the truncating
                    // narrow `vpmovdw` is exact.
                    unsafe {
                        _mm256_storeu_si256(p.cast::<__m256i>(), _mm512_cvtepi32_epi16(new_bits))
                    };
                }
                Elem::F16 => {
                    // SAFETY: `cell` is exactly 32 bytes (16 f16) of `row`, unaligned load.
                    let raw = unsafe { _mm256_loadu_si256(p.cast::<__m256i>()) };
                    let old = _mm512_cvtph_ps(raw);
                    let new_raw =
                        _mm512_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm512_add_ps(old, delta));
                    let new = _mm512_cvtph_ps(new_raw);
                    let changed = _mm512_cmpneq_epi32_mask(
                        _mm512_cvtepu16_epi32(raw),
                        _mm512_cvtepu16_epi32(new_raw),
                    );
                    stats_avx512(&mut acc, &mut max, old, new, changed);
                    // SAFETY: same 32 bytes, exclusively borrowed through `cell`.
                    unsafe { _mm256_storeu_si256(p.cast::<__m256i>(), new_raw) };
                }
            }
            i += 16;
        }
        // SAFETY: `acc.max` is a local array of exactly 16 f32.
        unsafe { _mm512_storeu_ps(acc.max.as_mut_ptr(), max) };
        acc.finish(tail(row, elem, d, i, n))
    }
}

#[cfg(target_arch = "aarch64")]
pub(super) mod neon {
    use std::arch::aarch64::*;

    use super::{body_len, shift_row_scalar, tail, Elem, RowDeltas, ShiftStats};

    const EXP_MASK: u32 = 0x7F80_0000;

    #[target_feature(enable = "neon")]
    fn delta_neon(d: RowDeltas<'_>, i: usize) -> float32x4_t {
        match d {
            RowDeltas::Splat(x) => vdupq_n_f32(x),
            RowDeltas::Each(ds) => match ds.get(i..i + 4) {
                // SAFETY: `get` proved that the 4 floats [i, i+4) are inside `ds`.
                Some(s) => unsafe { vld1q_f32(s.as_ptr()) },
                None => vdupq_n_f32(0.0),
            },
        }
    }

    /// Counts the all-ones lanes of a mask.
    #[target_feature(enable = "neon")]
    fn count(mask: uint32x4_t) -> u64 {
        u64::from(vaddvq_u32(vshrq_n_u32::<31>(mask)))
    }

    /// f32 bits -> bf16 bits in each u32 lane, same rule as the x86 kernels.
    #[target_feature(enable = "neon")]
    fn bf16_round_neon(bits: uint32x4_t) -> uint32x4_t {
        let high = vshrq_n_u32::<16>(bits);
        let lsb = vandq_u32(high, vdupq_n_u32(1));
        let rounded = vshrq_n_u32::<16>(vaddq_u32(bits, vaddq_u32(vdupq_n_u32(0x7FFF), lsb)));
        let quiet = vorrq_u32(high, vdupq_n_u32(0x40));
        let is_nan = vcgtq_u32(
            vandq_u32(bits, vdupq_n_u32(0x7FFF_FFFF)),
            vdupq_n_u32(EXP_MASK),
        );
        vbslq_u32(is_nan, quiet, rounded)
    }

    /// Test hook: the NEON bf16 rounding of 4 f32.
    #[cfg(test)]
    #[target_feature(enable = "neon")]
    pub(in crate::kernel) fn bf16_bits_neon(xs: &[f32; 4]) -> [u16; 4] {
        // SAFETY: `xs` is exactly 4 f32.
        let x = unsafe { vld1q_f32(xs.as_ptr()) };
        let bits = vmovn_u32(bf16_round_neon(vreinterpretq_u32_f32(x)));
        let mut out = [0u16; 4];
        // SAFETY: `out` is exactly 4 u16.
        unsafe { vst1_u16(out.as_mut_ptr(), bits) };
        out
    }

    /// NEON row kernel, 4 elements per step for f32 and bf16. f16 uses the scalar `half`
    /// conversion: the NEON f16 conversion intrinsics are not stable in Rust.
    #[target_feature(enable = "neon")]
    pub(in crate::kernel) fn row_neon(row: &mut [u8], elem: Elem, d: RowDeltas<'_>) -> ShiftStats {
        if elem == Elem::F16 {
            return shift_row_scalar(row, elem, d);
        }
        let (body, n) = body_len(row, elem, d, 4);
        let exp = vdupq_n_u32(EXP_MASK);
        let mut modified = 0;
        let mut nan_or_inf = 0;
        let mut max = vdupq_n_f32(0.0);
        let size = elem.size();
        let mut i = 0;
        while i < body {
            let Some(cell) = row.get_mut(i * size..(i + 4) * size) else {
                break;
            };
            let p = cell.as_mut_ptr();
            let delta = delta_neon(d, i);
            let (old_bits, new_bits, old, new) = match elem {
                Elem::Bf16 => {
                    // SAFETY: `cell` is exactly 8 bytes (4 bf16) of `row`; vld1 is unaligned.
                    let raw = unsafe { vld1_u16(p.cast::<u16>()) };
                    let old_bits = vmovl_u16(raw);
                    let old = vreinterpretq_f32_u32(vshlq_n_u32::<16>(old_bits));
                    let new_bits = bf16_round_neon(vreinterpretq_u32_f32(vaddq_f32(old, delta)));
                    let new = vreinterpretq_f32_u32(vshlq_n_u32::<16>(new_bits));
                    // SAFETY: same 8 bytes, exclusively borrowed through `cell`.
                    unsafe { vst1_u16(p.cast::<u16>(), vmovn_u32(new_bits)) };
                    (old_bits, new_bits, old, new)
                }
                _ => {
                    // SAFETY: `cell` is exactly 16 bytes (4 f32) of `row`; vld1q is unaligned.
                    let old = unsafe { vld1q_f32(p.cast::<f32>()) };
                    let new = vaddq_f32(old, delta);
                    // SAFETY: same 16 bytes, exclusively borrowed through `cell`.
                    unsafe { vst1q_f32(p.cast::<f32>(), new) };
                    (
                        vreinterpretq_u32_f32(old),
                        vreinterpretq_u32_f32(new),
                        old,
                        new,
                    )
                }
            };
            let changed = vmvnq_u32(vceqq_u32(old_bits, new_bits));
            let old_bad = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(old), exp), exp);
            let new_bad = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(new), exp), exp);
            modified += count(changed);
            nan_or_inf += count(vbicq_u32(vandq_u32(changed, new_bad), old_bad));
            let diff = vabsq_f32(vsubq_f32(new, old));
            max = vmaxq_f32(max, vbslq_f32(new_bad, vdupq_n_f32(0.0), diff));
            i += 4;
        }
        ShiftStats {
            elements_modified: modified,
            max_abs_delta: vmaxvq_f32(max),
            nan_or_inf_produced: nan_or_inf,
        }
        .merge(tail(row, elem, d, i, n))
    }
}
