//! Kernel tests: every available SIMD level against the scalar reference, bit for bit.
//! Levels this CPU lacks are skipped (and named by `levels_under_test`).

use half::{bf16, f16};
use memmap2::MmapMut;

use super::dispatch::{choose, shift_rows, shift_scalar, KernelConfig, Level};
use super::mmap_io::{ChunkFailure, ChunkedPatcher, IoConfig, OsHints, PageHints};
use super::simd::{Deltas, Elem, ShiftStats};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 16) as u32
    }
    fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next() as f32 / u32::MAX as f32)
    }
}

/// Levels other than scalar that this CPU can run.
fn levels_under_test() -> Vec<Level> {
    [Level::Avx2, Level::Avx512, Level::Neon]
        .into_iter()
        .filter(|l| l.is_available())
        .collect()
}

const COLS: [usize; 10] = [1, 7, 8, 15, 16, 17, 255, 256, 257, 4096];
const ROWS: [usize; 6] = [1, 2, 3, 5, 17, 64];
const ELEMS: [Elem; 3] = [Elem::Bf16, Elem::F16, Elem::F32];

fn encode(elem: Elem, x: f32) -> Vec<u8> {
    match elem {
        Elem::Bf16 => bf16::from_f32(x).to_le_bytes().to_vec(),
        Elem::F16 => f16::from_f32(x).to_le_bytes().to_vec(),
        Elem::F32 => x.to_le_bytes().to_vec(),
    }
}

fn decode(elem: Elem, b: &[u8]) -> f32 {
    match elem {
        Elem::Bf16 => bf16::from_le_bytes([b[0], b[1]]).to_f32(),
        Elem::F16 => f16::from_le_bytes([b[0], b[1]]).to_f32(),
        Elem::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
    }
}

/// Every NaN replaced by one canonical pattern: payloads may legitimately differ.
fn canonical(elem: Elem, bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks(elem.size())
        .flat_map(|c| match decode(elem, c).is_nan() {
            true => vec![0xAA; elem.size()],
            false => c.to_vec(),
        })
        .collect()
}

/// Random weights with every edge class mixed in: ±0, subnormals (of the dtype and of
/// f32), ±Inf, NaN, the largest finite values, and RNE ties built by hand (`w` exact in
/// the dtype, `w + delta` exactly halfway between two neighbours).
fn tensor(elem: Elem, rows: usize, cols: usize, rng: &mut Rng) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let half_ulp_at_one = match elem {
        Elem::Bf16 => 2f32.powi(-8),
        Elem::F16 => 2f32.powi(-11),
        Elem::F32 => 2f32.powi(-24),
    };
    let specials = [
        0.0,
        -0.0,
        1e-41,
        -1e-40,
        6.0e-8,
        -5.9e-8,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        65504.0,
        -3.0e38,
        1.0,
        1.0 + 2.0 * half_ulp_at_one,
    ];
    let mut bytes = Vec::with_capacity(rows * cols * elem.size());
    for i in 0..rows * cols {
        let x = match rng.next() % 5 {
            0 => specials[i % specials.len()],
            _ => rng.f32_in(-4.0, 4.0),
        };
        bytes.extend(encode(elem, x));
    }
    let col_deltas: Vec<f32> = (0..cols)
        .map(|c| match c % 4 {
            // Ties at 1.0 (even, stays) and at 1 + ulp (odd, rounds up).
            0 => half_ulp_at_one,
            1 => -half_ulp_at_one,
            2 => rng.f32_in(-1e-3, 1e-3),
            _ => rng.f32_in(-0.5, 0.5),
        })
        .collect();
    let row_deltas: Vec<f32> = (0..rows)
        .map(|r| match r % 3 {
            0 => half_ulp_at_one,
            1 => 70000.0, // overflows f16
            _ => rng.f32_in(-0.25, 0.25),
        })
        .collect();
    (bytes, col_deltas, row_deltas)
}

fn assert_same(elem: Elem, label: &str, got: (&[u8], ShiftStats), want: (&[u8], ShiftStats)) {
    assert_eq!(
        canonical(elem, got.0),
        canonical(elem, want.0),
        "{label}: bytes"
    );
    assert_eq!(got.1, want.1, "{label}: stats");
}

#[test]
fn every_level_matches_the_scalar_reference_bit_for_bit() {
    let levels = levels_under_test();
    let mut rng = Rng(0x5EED);
    for elem in ELEMS {
        for rows in ROWS {
            for cols in COLS {
                let (original, per_col, per_row) = tensor(elem, rows, cols, &mut rng);
                for deltas in [Deltas::PerCol(&per_col), Deltas::PerRow(&per_row)] {
                    let mut want = original.clone();
                    let want_stats = shift_scalar(&mut want, elem, cols, deltas).unwrap();
                    for &level in &levels {
                        let mut got = original.clone();
                        let stats =
                            shift_rows(&mut got, elem, cols, deltas, &KernelConfig::new(level))
                                .unwrap();
                        let label = format!("{level} {elem:?} {rows}x{cols} {deltas:?}");
                        assert_same(elem, &label, (&got, stats), (&want, want_stats));
                    }
                }
            }
        }
    }
}

/// The scalar reference itself agrees with the engine's original formula, element by
/// element: `store(load(w) + delta)`.
#[test]
fn scalar_reference_is_store_of_load_plus_delta() {
    let mut rng = Rng(9);
    for elem in ELEMS {
        let (original, per_col, _) = tensor(elem, 3, 17, &mut rng);
        let mut got = original.clone();
        shift_scalar(&mut got, elem, 17, Deltas::PerCol(&per_col)).unwrap();
        for (i, (cell, before)) in got
            .chunks(elem.size())
            .zip(original.chunks(elem.size()))
            .enumerate()
        {
            let want = encode(elem, decode(elem, before) + per_col[i % 17]);
            assert_eq!(
                canonical(elem, cell),
                canonical(elem, &want),
                "{elem:?} #{i}"
            );
        }
    }
}

#[test]
fn unaligned_slices_and_sentinels_are_respected() {
    let mut rng = Rng(77);
    for level in levels_under_test() {
        for elem in ELEMS {
            for offset in [1usize, 3, 5] {
                let (rows, cols) = (5, 257);
                let (original, per_col, _) = tensor(elem, rows, cols, &mut rng);
                let mut want = original.clone();
                shift_scalar(&mut want, elem, cols, Deltas::PerCol(&per_col)).unwrap();
                // Sentinels on both sides, and a start offset that breaks any alignment.
                let mut buffer = vec![0xC3u8; offset + original.len() + 64];
                let body = offset..offset + original.len();
                buffer[body.clone()].copy_from_slice(&original);
                let cfg = KernelConfig::new(level);
                shift_rows(
                    &mut buffer[body.clone()],
                    elem,
                    cols,
                    Deltas::PerCol(&per_col),
                    &cfg,
                )
                .unwrap();
                assert!(
                    buffer[..offset].iter().all(|&b| b == 0xC3),
                    "{level} before"
                );
                assert!(
                    buffer[body.end..].iter().all(|&b| b == 0xC3),
                    "{level} after"
                );
                assert_eq!(
                    canonical(elem, &buffer[body]),
                    canonical(elem, &want),
                    "{level} {elem:?} +{offset}"
                );
            }
        }
    }
}

#[test]
fn result_does_not_depend_on_threads_or_block_size() {
    let mut rng = Rng(3);
    let (rows, cols) = (64, 300);
    let (original, _per_col, per_row) = tensor(Elem::Bf16, rows, cols, &mut rng);
    let mut want = original.clone();
    let want_stats = shift_scalar(&mut want, Elem::Bf16, cols, Deltas::PerRow(&per_row)).unwrap();
    let mut levels = levels_under_test();
    levels.push(Level::Scalar);
    for level in levels {
        for threads in [1, 2, 8] {
            for block_bytes in [1, 600, 7000, 1 << 20] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let mut got = original.clone();
                let cfg = KernelConfig { level, block_bytes };
                let stats = pool.install(|| {
                    shift_rows(&mut got, Elem::Bf16, cols, Deltas::PerRow(&per_row), &cfg).unwrap()
                });
                let label = format!("{level} {threads} threads, blocks of {block_bytes}");
                assert_same(Elem::Bf16, &label, (&got, stats), (&want, want_stats));
            }
        }
    }
}

#[test]
fn stats_count_changes_overflows_and_the_largest_delta() {
    // f16: 65504 + 70000 overflows, 1.0 + 2^-12 is lost to rounding, 2 + 0.5 is exact.
    let mut bytes: Vec<u8> = [65504.0f32, 1.0, 2.0]
        .iter()
        .flat_map(|&x| encode(Elem::F16, x))
        .collect();
    let deltas = [70000.0, 2f32.powi(-12), 0.5];
    for level in levels_under_test().into_iter().chain([Level::Scalar]) {
        let mut b = bytes.clone();
        let stats = shift_rows(
            &mut b,
            Elem::F16,
            3,
            Deltas::PerCol(&deltas),
            &KernelConfig::new(level),
        )
        .unwrap();
        assert_eq!(stats.elements_modified, 2, "{level}");
        assert_eq!(stats.nan_or_inf_produced, 1, "{level}");
        assert_eq!(stats.max_abs_delta, 0.5, "{level}");
    }
    let stats = shift_scalar(&mut bytes, Elem::F16, 3, Deltas::PerCol(&deltas)).unwrap();
    assert!(decode(Elem::F16, &bytes[..2]).is_infinite());
    assert_eq!(stats.elements_modified, 2);
}

#[test]
fn wrong_shapes_are_errors_not_panics() {
    let mut bytes = vec![0u8; 12];
    let cfg = KernelConfig::new(Level::best());
    assert!(shift_rows(&mut bytes, Elem::F32, 2, Deltas::PerCol(&[1.0]), &cfg).is_err());
    assert!(shift_rows(&mut bytes, Elem::F32, 2, Deltas::PerRow(&[1.0]), &cfg).is_err());
    assert!(shift_rows(&mut bytes, Elem::F32, 5, Deltas::PerCol(&[1.0; 5]), &cfg).is_err());
    assert!(shift_rows(&mut bytes, Elem::F32, 0, Deltas::PerCol(&[]), &cfg).is_err());
    assert_eq!(
        shift_rows(&mut [], Elem::F32, 3, Deltas::PerCol(&[1.0; 3]), &cfg),
        Ok(ShiftStats::default())
    );
}

#[test]
fn level_choice_honours_the_request_and_falls_back_with_a_warning() {
    assert_eq!(choose(Some("scalar")).level, Level::Scalar);
    assert_eq!(choose(Some(" SCALAR ")).warning, None);
    assert_eq!(choose(None).level, Level::best());
    assert!(choose(None).warning.is_none());
    let unknown = choose(Some("sse9"));
    assert_eq!(unknown.level, Level::best());
    assert!(unknown.warning.unwrap().contains("inconnu"));
    for level in [Level::Avx2, Level::Avx512, Level::Neon] {
        let choice = choose(Some(level.name()));
        match level.is_available() {
            true => assert_eq!((choice.level, choice.warning), (level, None)),
            false => {
                assert_eq!(choice.level, Level::best());
                assert!(choice.warning.unwrap().contains("indisponible"));
            }
        }
    }
}

/// A forced level the CPU lacks never reaches its instructions: the scalar kernel runs.
#[test]
fn unavailable_level_runs_the_scalar_kernel() {
    let missing: Vec<Level> = Level::ALL
        .into_iter()
        .filter(|l| !l.is_available())
        .collect();
    assert!(!missing.is_empty(), "no CPU has both x86 and NEON levels");
    let mut rng = Rng(5);
    let (original, per_col, _) = tensor(Elem::F16, 4, 33, &mut rng);
    let mut want = original.clone();
    let want_stats = shift_scalar(&mut want, Elem::F16, 33, Deltas::PerCol(&per_col)).unwrap();
    for level in missing {
        let mut got = original.clone();
        let stats = shift_rows(
            &mut got,
            Elem::F16,
            33,
            Deltas::PerCol(&per_col),
            &KernelConfig::new(level),
        )
        .unwrap();
        assert_same(
            Elem::F16,
            &level.to_string(),
            (&got, stats),
            (&want, want_stats),
        );
    }
}

// ------------------------------------------------------------------ conversions

/// The f32 patterns of the sampled conversion test: a fixed stride over all 2^32 patterns
/// plus every edge class (zeros, subnormals, Inf, NaN payloads, overflow thresholds and
/// RNE ties of both bf16 and f16).
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn sampled_patterns() -> Vec<u32> {
    let mut bits: Vec<u32> = (0..=u32::MAX).step_by(65_521).collect();
    for sign in [0u32, 0x8000_0000] {
        for b in [
            0x0000_0000,
            0x0000_0001,
            0x007F_FFFF,
            0x0080_0000,
            0x7F7F_FFFF,
            0x7F80_0000,
            0x7F80_0001,
            0x7FBF_FFFF,
            0x7FC0_0000,
            0x7FFF_FFFF,
            0x477F_E000, // 65504, f16 max
            0x477F_EFFF, // just below the f16 overflow threshold
            0x477F_F000, // 65520: f16 tie between max and Inf -> Inf
            0x3380_0000, // 2^-24, smallest f16 subnormal
            0x3300_0000, // 2^-25: tie between 0 and the smallest subnormal -> 0
            0x3300_0001,
        ] {
            bits.push(sign | b);
        }
        // bf16 ties: low half exactly 0x8000, with an even and an odd kept bit.
        for high in [0x3F80u32, 0x3F81, 0x7F7F, 0x0001] {
            bits.push(sign | (high << 16) | 0x8000);
            bits.push(sign | (high << 16) | 0x7FFF);
            bits.push(sign | (high << 16) | 0x8001);
        }
        // f16 ties: 13 dropped bits exactly 0x1000 above a value of each parity.
        for kept in [0x3F80_0000u32, 0x3F80_2000, 0x3880_0000, 0x3880_2000] {
            bits.push(sign | kept | 0x1000);
            bits.push(sign | kept | 0x0FFF);
            bits.push(sign | kept | 0x1001);
        }
    }
    bits
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
fn same_half(a: u16, b: u16, is_nan: impl Fn(u16) -> bool) -> bool {
    a == b || (is_nan(a) && is_nan(b))
}

/// Checks the SIMD conversions of `xs` against `half`; returns the number of mismatches.
#[cfg(target_arch = "x86_64")]
fn conversion_mismatches(xs: &[f32]) -> usize {
    use super::simd::x86;
    let bf_nan = |b: u16| bf16::from_bits(b).is_nan();
    let f_nan = |b: u16| f16::from_bits(b).is_nan();
    let mut bad = 0;
    if Level::Avx2.is_available() {
        for group in xs.chunks_exact(8) {
            let group: &[f32; 8] = group.try_into().unwrap();
            // SAFETY: avx2 and f16c detected just above.
            let (bf, f) = unsafe { x86::convert_avx2(group) };
            for (i, &x) in group.iter().enumerate() {
                bad += usize::from(!same_half(bf[i], bf16::from_f32(x).to_bits(), bf_nan));
                bad += usize::from(!same_half(f[i], f16::from_f32(x).to_bits(), f_nan));
            }
        }
    }
    if Level::Avx512.is_available() {
        for group in xs.chunks_exact(16) {
            let group: &[f32; 16] = group.try_into().unwrap();
            // SAFETY: avx512f detected just above.
            let (bf, f) = unsafe { x86::convert_avx512(group) };
            for (i, &x) in group.iter().enumerate() {
                bad += usize::from(!same_half(bf[i], bf16::from_f32(x).to_bits(), bf_nan));
                bad += usize::from(!same_half(f[i], f16::from_f32(x).to_bits(), f_nan));
            }
        }
    }
    bad
}

/// NEON: bf16 only (f16 uses the scalar `half` conversion on aarch64).
#[cfg(target_arch = "aarch64")]
fn conversion_mismatches(xs: &[f32]) -> usize {
    use super::simd::neon;
    let bf_nan = |b: u16| bf16::from_bits(b).is_nan();
    let mut bad = 0;
    if Level::Neon.is_available() {
        for group in xs.chunks_exact(4) {
            let group: &[f32; 4] = group.try_into().unwrap();
            // SAFETY: neon detected just above.
            let bf = unsafe { neon::bf16_bits_neon(group) };
            for (i, &x) in group.iter().enumerate() {
                bad += usize::from(!same_half(bf[i], bf16::from_f32(x).to_bits(), bf_nan));
            }
        }
    }
    bad
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
fn simd_conversions_match_half_on_sampled_patterns() {
    let mut xs: Vec<f32> = sampled_patterns().into_iter().map(f32::from_bits).collect();
    xs.resize(xs.len().div_ceil(16) * 16, 0.0);
    assert_eq!(conversion_mismatches(&xs), 0);
}

/// All 2^32 f32 patterns, every available SIMD level. Run once in release:
/// `cargo test --release --lib -- --ignored exhaustive`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[test]
#[ignore]
fn exhaustive_f32_to_bf16_and_f16_conversions() {
    use rayon::prelude::*;
    let bad: usize = (0u32..=u16::MAX as u32)
        .into_par_iter()
        .map(|high| {
            let xs: Vec<f32> = (0u32..=u16::MAX as u32)
                .map(|low| f32::from_bits(high << 16 | low))
                .collect();
            conversion_mismatches(&xs)
        })
        .sum();
    assert_eq!(bad, 0);
}

// ------------------------------------------------------------------ chunked patcher

/// Hints that always fail: the patch must still complete, failures only counted.
struct FailingHints;

impl PageHints for FailingHints {
    fn sequential(&mut self, _: &MmapMut, _: std::ops::Range<usize>) -> std::io::Result<()> {
        Err(std::io::Error::other("injected"))
    }
    fn will_need(&mut self, _: &MmapMut, _: std::ops::Range<usize>) -> std::io::Result<()> {
        Err(std::io::Error::other("injected"))
    }
    fn flush_async(&mut self, _: &MmapMut, _: std::ops::Range<usize>) -> std::io::Result<()> {
        Err(std::io::Error::other("injected"))
    }
}

/// Runs a per-row shift chunk by chunk over a mapping, returns the bytes and counters.
fn chunked(
    original: &[u8],
    cols: usize,
    per_row: &[f32],
    io: IoConfig,
    hints: &mut dyn PageHints,
) -> (Vec<u8>, super::IoCounters) {
    let head = 24; // the tensor starts mid-page, after an unrelated header
    let mut map = MmapMut::map_anon(head + original.len() + 16).unwrap();
    map[head..head + original.len()].copy_from_slice(original);
    let row_bytes = cols * 2;
    let counters = ChunkedPatcher::new(io)
        .run(
            &mut map,
            head..head + original.len(),
            row_bytes,
            hints,
            &|| false,
            |chunk, first_row| {
                let rows = chunk.len() / row_bytes;
                let d = Deltas::PerRow(&per_row[first_row..first_row + rows]);
                shift_rows(
                    chunk,
                    Elem::Bf16,
                    cols,
                    d,
                    &KernelConfig::new(Level::best()),
                )
                .map(|_| ())
            },
        )
        .unwrap();
    assert!(map[..head].iter().all(|&b| b == 0));
    assert!(map[head + original.len()..].iter().all(|&b| b == 0));
    (map[head..head + original.len()].to_vec(), counters)
}

#[test]
fn chunked_patch_equals_one_pass_for_any_chunk_size() {
    let mut rng = Rng(11);
    let (rows, cols) = (37, 129);
    let (original, _, per_row) = tensor(Elem::Bf16, rows, cols, &mut rng);
    let mut want = original.clone();
    shift_scalar(&mut want, Elem::Bf16, cols, Deltas::PerRow(&per_row)).unwrap();
    let row_bytes = cols * 2;
    for chunk_bytes in [
        1,
        3,
        row_bytes - 1,
        row_bytes,
        row_bytes + 1,
        5 * row_bytes + 7,
        1 << 30,
    ] {
        for prefetch in [true, false] {
            let io = IoConfig {
                chunk_bytes,
                prefetch,
            };
            let (got, counters) = chunked(&original, cols, &per_row, io, &mut OsHints);
            assert_eq!(
                canonical(Elem::Bf16, &got),
                canonical(Elem::Bf16, &want),
                "{io:?}"
            );
            let rows_per_chunk = (chunk_bytes / row_bytes).max(1);
            assert_eq!(counters.chunks, rows.div_ceil(rows_per_chunk) as u64);
            assert_eq!(counters.bytes, original.len() as u64);
            if !prefetch {
                assert_eq!(counters.prefetch_requests + counters.flush_requests, 0);
            }
        }
    }
}

#[test]
fn injected_hint_failures_are_counted_never_fatal() {
    let mut rng = Rng(12);
    let (original, _, per_row) = tensor(Elem::Bf16, 20, 64, &mut rng);
    let mut want = original.clone();
    shift_scalar(&mut want, Elem::Bf16, 64, Deltas::PerRow(&per_row)).unwrap();
    let io = IoConfig {
        chunk_bytes: 3 * 128,
        prefetch: true,
    };
    let (got, counters) = chunked(&original, 64, &per_row, io, &mut FailingHints);
    assert_eq!(canonical(Elem::Bf16, &got), canonical(Elem::Bf16, &want));
    assert!(counters.prefetch_requests > 0);
    assert_eq!(counters.prefetch_failures, counters.prefetch_requests);
    assert_eq!(counters.flush_failures, counters.flush_requests);
    assert_eq!(counters.flush_requests, counters.chunks - 1);
}

#[test]
fn cancellation_stops_between_chunks() {
    let mut map = MmapMut::map_anon(1000).unwrap();
    let calls = std::cell::Cell::new(0);
    let result = ChunkedPatcher::new(IoConfig {
        chunk_bytes: 100,
        prefetch: false,
    })
    .run(
        &mut map,
        0..1000,
        10,
        &mut OsHints,
        &|| calls.get() == 3,
        |chunk, _| {
            calls.set(calls.get() + 1);
            chunk.fill(1);
            Ok::<(), ()>(())
        },
    );
    assert!(matches!(
        result,
        Err(ChunkFailure::Cancelled { chunks_done: 3 })
    ));
    assert_eq!(map.iter().filter(|&&b| b == 1).count(), 300);
}

#[test]
fn invalid_ranges_are_refused_before_any_work() {
    let mut map = MmapMut::map_anon(100).unwrap();
    let patcher = ChunkedPatcher::new(IoConfig::default());
    for (range, row) in [(0..101, 1), (0..99, 2), (0..10, 0)] {
        let result = patcher.run(&mut map, range, row, &mut OsHints, &|| false, |_, _| {
            Err::<(), &str>("must not run")
        });
        assert!(matches!(result, Err(ChunkFailure::InvalidRange)));
    }
}

#[cfg(windows)]
#[test]
fn windows_prefetch_on_a_file_mapping_succeeds_or_is_counted() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.bin");
    std::fs::write(&path, vec![0u8; 3 << 20]).unwrap();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let mut map = super::map_write(&file).unwrap();
    let mut hints = OsHints;
    assert!(hints.will_need(&map, 4096..(2 << 20)).is_ok());
    assert!(hints.sequential(&map, 0..(3 << 20)).is_ok());
    assert!(
        hints.will_need(&map, 0..(4 << 20)).is_err(),
        "out of the view"
    );
    map[0] = 1;
    assert!(hints.flush_async(&map, 0..4096).is_ok());
}

#[cfg(unix)]
#[test]
fn unix_madvise_on_a_file_mapping_succeeds() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.bin");
    std::fs::write(&path, vec![0u8; 3 << 20]).unwrap();
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .unwrap();
    let mut map = super::map_write(&file).unwrap();
    let mut hints = OsHints;
    assert!(hints.will_need(&map, 4097..(2 << 20)).is_ok());
    assert!(hints.sequential(&map, 0..(3 << 20)).is_ok());
    map[0] = 1;
    assert!(hints.flush_async(&map, 0..4096).is_ok());
}

#[test]
fn page_faults_and_free_space_are_reported() {
    let before = super::page_faults().expect("page fault counters on this OS");
    // A fresh anonymous mapping faults on the first write to each page. A `Vec` does not
    // always: once another test has freed a large buffer, glibc's dynamic mmap threshold
    // can serve it from heap pages that are already resident, and nothing faults.
    let mut fresh = MmapMut::map_anon(8 << 20).unwrap();
    for page in fresh.chunks_mut(4096) {
        page[0] = 1;
    }
    let touched: u64 = fresh.iter().map(|&b| u64::from(b)).sum();
    assert_eq!(touched, 2048);
    let after = super::page_faults().unwrap();
    assert!(after.since(before).total > 0);
    let dir = tempfile::tempdir().unwrap();
    assert!(super::available_space(dir.path()).unwrap() > 0);
}
