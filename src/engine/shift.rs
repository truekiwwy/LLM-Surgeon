use half::{bf16, f16};
use rayon::prelude::*;
use serde::Serialize;

use super::error::{EngineError, Result};
use super::index::DType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ShiftOp {
    Subtract,
    Add,
}

/// How the 1D direction is broadcast over a 2D tensor `[rows, cols]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum BroadcastAxis {
    /// `direction.len() == cols`: every row receives v, `W[r][c] -= alpha * v[c]`.
    Rows,
    /// `direction.len() == rows`: `W[r][c] -= alpha * v[r]`.
    Cols,
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
    fn merge(self, other: Self) -> Self {
        Self {
            elements_modified: self.elements_modified + other.elements_modified,
            max_abs_delta: self.max_abs_delta.max(other.max_abs_delta),
            nan_or_inf_produced: self.nan_or_inf_produced + other.nan_or_inf_produced,
        }
    }
}

// Little-endian byte arrays, never pointer casts: tensor data has no alignment guarantee.
pub(super) fn bf16_load(b: [u8; 2]) -> f32 {
    bf16::from_le_bytes(b).to_f32()
}
pub(super) fn bf16_store(x: f32) -> [u8; 2] {
    bf16::from_f32(x).to_le_bytes() // round to nearest, ties to even
}
pub(super) fn f16_load(b: [u8; 2]) -> f32 {
    f16::from_le_bytes(b).to_f32()
}
pub(super) fn f16_store(x: f32) -> [u8; 2] {
    f16::from_f32(x).to_le_bytes()
}

/// `W ∓= alpha * v` in place on a row-major `[rows, cols]` tensor. Every check runs before
/// the first write, so on `Err` not a single byte has changed.
pub fn apply_vector_shift(
    tensor_bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    direction: &[f32],
    alpha: f32,
    op: ShiftOp,
    axis: BroadcastAxis,
) -> Result<ShiftStats> {
    let elem = dtype
        .size_of()
        .ok_or_else(|| EngineError::UnsupportedDtype(dtype.to_string()))?;
    let invalid_shape = |reason: &str| EngineError::InvalidShape {
        shape: shape.to_vec(),
        reason: reason.to_owned(),
    };
    let &[rows, cols] = shape else {
        return Err(invalid_shape("tenseur 2D attendu"));
    };
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(elem))
        .ok_or_else(|| invalid_shape("dépassement de capacité"))?;
    if tensor_bytes.len() != expected {
        return Err(EngineError::ByteLengthMismatch {
            expected,
            got: tensor_bytes.len(),
        });
    }
    let expected_dir = match axis {
        BroadcastAxis::Rows => cols,
        BroadcastAxis::Cols => rows,
    };
    if direction.len() != expected_dir {
        return Err(EngineError::DirectionLengthMismatch {
            expected: expected_dir,
            got: direction.len(),
        });
    }
    if !alpha.is_finite() {
        return Err(EngineError::InvalidAlpha(alpha));
    }
    if let Some(index) = direction.iter().position(|v| !v.is_finite()) {
        return Err(EngineError::NonFiniteDirection { index });
    }
    // alpha == 0 is a strict no-op: computing it would still turn -0.0 into +0.0
    // whenever v has the matching sign.
    if alpha == 0.0 || tensor_bytes.is_empty() {
        return Ok(ShiftStats::default());
    }

    // a - b == a + (-b) exactly in IEEE 754 (signed zeros included), so folding the sign
    // into one precomputed delta per v[i] stays bit-identical to `w - alpha * v[i]`.
    let sign = match op {
        ShiftOp::Subtract => -1.0,
        ShiftOp::Add => 1.0,
    };
    let deltas: Vec<f32> = direction.iter().map(|v| sign * (alpha * v)).collect();
    let deltas = match axis {
        BroadcastAxis::Rows => Deltas::PerCol(&deltas),
        BroadcastAxis::Cols => Deltas::PerRow(&deltas),
    };
    Ok(match dtype {
        DType::BF16 => shift_rows(tensor_bytes, cols, deltas, bf16_load, bf16_store),
        DType::F16 => shift_rows(tensor_bytes, cols, deltas, f16_load, f16_store),
        DType::F32 => shift_rows(
            tensor_bytes,
            cols,
            deltas,
            f32::from_le_bytes,
            f32::to_le_bytes,
        ),
        DType::Other(name) => return Err(EngineError::UnsupportedDtype(name)),
    })
}

#[derive(Clone, Copy)]
enum Deltas<'a> {
    PerCol(&'a [f32]),
    PerRow(&'a [f32]),
}

/// One rayon task per group of rows. Rows are disjoint slices, each element depends only
/// on itself, and the stats reduction (sum, max) is exact, so the result is identical to a
/// sequential run.
fn shift_rows<const N: usize, L, S>(
    bytes: &mut [u8],
    cols: usize,
    deltas: Deltas<'_>,
    load: L,
    store: S,
) -> ShiftStats
where
    L: Fn([u8; N]) -> f32 + Sync,
    S: Fn(f32) -> [u8; N] + Sync,
{
    let row_bytes = cols * N;
    bytes
        .par_chunks_mut(row_bytes)
        // >= 64 KiB per task: amortises scheduling and keeps threads on separate cache lines.
        .with_min_len((64 * 1024 / row_bytes).max(1))
        .enumerate()
        .map(|(r, row)| {
            let (cells, _) = row.as_chunks_mut::<N>();
            match deltas {
                Deltas::PerCol(d) => shift_cells(cells, d.iter().copied(), &load, &store),
                // r < rows == d.len(), checked by apply_vector_shift.
                Deltas::PerRow(d) => shift_cells(cells, std::iter::repeat(d[r]), &load, &store),
            }
        })
        .reduce(ShiftStats::default, ShiftStats::merge)
}

#[inline]
pub(super) fn shift_cells<const N: usize>(
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

/// Decodes a tensor's bytes to f32 (used to load a direction vector).
pub(crate) fn decode_f32(bytes: &[u8], dtype: &DType) -> Result<Vec<f32>> {
    fn collect<const N: usize>(bytes: &[u8], load: impl Fn([u8; N]) -> f32) -> Vec<f32> {
        bytes.as_chunks::<N>().0.iter().map(|c| load(*c)).collect()
    }
    Ok(match dtype {
        DType::BF16 => collect(bytes, bf16_load),
        DType::F16 => collect(bytes, f16_load),
        DType::F32 => collect(bytes, f32::from_le_bytes),
        DType::Other(name) => return Err(EngineError::UnsupportedDtype(name.clone())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::test_support::{encode, reference_shift, Lcg};

    const ALL: [(ShiftOp, BroadcastAxis); 4] = [
        (ShiftOp::Subtract, BroadcastAxis::Rows),
        (ShiftOp::Subtract, BroadcastAxis::Cols),
        (ShiftOp::Add, BroadcastAxis::Rows),
        (ShiftOp::Add, BroadcastAxis::Cols),
    ];

    fn dir_len(shape: [usize; 2], axis: BroadcastAxis) -> usize {
        match axis {
            BroadcastAxis::Rows => shape[1],
            BroadcastAxis::Cols => shape[0],
        }
    }

    #[test]
    fn matches_reference_for_every_dtype_op_and_axis() {
        let values: Vec<f32> = vec![
            0.0, -0.0, 1.0, -1.5, 3.25, -7.0, 1e-3, 1e-41, -1e-40, 65000.0, -2.5e38, 1e30,
        ];
        let shape = [3, 4];
        let mut rng = Lcg(7);
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            for (op, axis) in ALL {
                let v: Vec<f32> = (0..dir_len(shape, axis))
                    .map(|_| rng.f32_in(-4.0, 4.0))
                    .collect();
                let original = encode(&dtype, &values);
                let expected = reference_shift(&dtype, shape, &original, &v, 0.5, op, axis);
                let mut bytes = original.clone();
                apply_vector_shift(&mut bytes, dtype.clone(), &shape, &v, 0.5, op, axis).unwrap();
                assert_eq!(bytes, expected, "{dtype:?} {op:?} {axis:?}");
            }
        }
    }

    #[test]
    fn parallel_512x512_equals_sequential_reference() {
        let shape = [512, 512];
        let mut rng = Lcg(42);
        let values: Vec<f32> = (0..512 * 512).map(|_| rng.f32_in(-2.0, 2.0)).collect();
        let original = encode(&DType::BF16, &values);
        for (op, axis) in ALL {
            let v: Vec<f32> = (0..512).map(|_| rng.f32_in(-0.1, 0.1)).collect();
            let expected = reference_shift(&DType::BF16, shape, &original, &v, 0.75, op, axis);
            let mut bytes = original.clone();
            let stats =
                apply_vector_shift(&mut bytes, DType::BF16, &shape, &v, 0.75, op, axis).unwrap();
            assert_eq!(bytes, expected, "{op:?} {axis:?}");
            let changed = bytes
                .chunks(2)
                .zip(original.chunks(2))
                .filter(|(a, b)| a != b)
                .count() as u64;
            assert_eq!(stats.elements_modified, changed);
        }
    }

    #[test]
    fn alpha_zero_is_a_strict_noop() {
        let original = encode(&DType::BF16, &[-0.0, 0.0, 1.0, -2.0]);
        let mut bytes = original.clone();
        let stats = apply_vector_shift(
            &mut bytes,
            DType::BF16,
            &[2, 2],
            &[-1.0, 1.0],
            0.0,
            ShiftOp::Subtract,
            BroadcastAxis::Rows,
        )
        .unwrap();
        assert_eq!(bytes, original);
        assert_eq!(stats, ShiftStats::default());
    }

    /// Round trip error is at most 1 bf16 ULP as long as |alpha * v| stays small next to
    /// |w| (here |delta| <= 0.005, |w| >= 0.5), so `w + d` stays within one binade of w.
    /// With |delta| >> |w| the information is lost by design (bf16 keeps 8 bits).
    #[test]
    fn add_then_subtract_stays_within_one_bf16_ulp() {
        // Order-preserving key so the ULP distance is a plain subtraction.
        fn key(b: &[u8]) -> i32 {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            let mag = i32::from(bits & 0x7fff);
            if bits & 0x8000 != 0 {
                -mag
            } else {
                mag
            }
        }
        let mut rng = Lcg(3);
        let shape = [64, 64];
        let values: Vec<f32> = (0..64 * 64)
            .map(|_| {
                let m = rng.f32_in(0.5, 4.0);
                if rng.next_u32().is_multiple_of(2) {
                    m
                } else {
                    -m
                }
            })
            .collect();
        let v: Vec<f32> = (0..64).map(|_| rng.f32_in(-0.01, 0.01)).collect();
        let original = encode(&DType::BF16, &values);
        let mut bytes = original.clone();
        for op in [ShiftOp::Add, ShiftOp::Subtract] {
            apply_vector_shift(
                &mut bytes,
                DType::BF16,
                &shape,
                &v,
                0.5,
                op,
                BroadcastAxis::Rows,
            )
            .unwrap();
        }
        let worst = bytes
            .chunks(2)
            .zip(original.chunks(2))
            .map(|(a, b)| (key(a) - key(b)).abs())
            .max()
            .unwrap();
        assert!(worst <= 1, "worst ULP error {worst}");
    }

    #[test]
    fn f16_overflow_is_reported() {
        let mut bytes = encode(&DType::F16, &[65504.0, 1.0]);
        let stats = apply_vector_shift(
            &mut bytes,
            DType::F16,
            &[1, 2],
            &[-1000.0, 0.0],
            1.0,
            ShiftOp::Subtract,
            BroadcastAxis::Rows,
        )
        .unwrap();
        assert_eq!(stats.nan_or_inf_produced, 1);
        assert_eq!(stats.elements_modified, 1);
        assert_eq!(stats.max_abs_delta, 0.0); // the only change is non-finite
        assert!(f16::from_le_bytes([bytes[0], bytes[1]]).is_infinite());
    }

    #[test]
    fn degenerate_shapes() {
        let sub = ShiftOp::Subtract;
        // 0 x N and N x 0: valid, nothing to do.
        let empty = apply_vector_shift(
            &mut [],
            DType::BF16,
            &[0, 5],
            &[1.0; 5],
            1.0,
            sub,
            BroadcastAxis::Rows,
        );
        assert_eq!(empty.unwrap(), ShiftStats::default());
        let empty = apply_vector_shift(
            &mut [],
            DType::F32,
            &[5, 0],
            &[1.0; 5],
            1.0,
            sub,
            BroadcastAxis::Cols,
        );
        assert_eq!(empty.unwrap(), ShiftStats::default());
        let empty = apply_vector_shift(
            &mut [],
            DType::F32,
            &[5, 0],
            &[],
            1.0,
            sub,
            BroadcastAxis::Rows,
        );
        assert_eq!(empty.unwrap(), ShiftStats::default());
        // 1 x 1.
        let mut one = encode(&DType::F32, &[2.0]);
        let stats = apply_vector_shift(
            &mut one,
            DType::F32,
            &[1, 1],
            &[1.0],
            0.5,
            sub,
            BroadcastAxis::Cols,
        )
        .unwrap();
        assert_eq!(one, encode(&DType::F32, &[1.5]));
        assert_eq!(stats.max_abs_delta, 0.5);
    }

    #[test]
    fn validation_errors_leave_bytes_untouched() {
        let original = encode(&DType::BF16, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let shape = [2, 3];
        let rows = BroadcastAxis::Rows;
        let sub = ShiftOp::Subtract;
        let cases: Vec<(EngineError, Result<ShiftStats>)> = {
            let run = |dtype: DType, shape: &[usize], v: &[f32], alpha: f32| {
                let mut bytes = original.clone();
                let r = apply_vector_shift(&mut bytes, dtype, shape, v, alpha, sub, rows);
                assert_eq!(bytes, original);
                r
            };
            vec![
                (
                    EngineError::UnsupportedDtype("I32".into()),
                    run(DType::Other("I32".into()), &shape, &[1.0; 3], 1.0),
                ),
                (
                    EngineError::DirectionLengthMismatch {
                        expected: 3,
                        got: 2,
                    },
                    run(DType::BF16, &shape, &[1.0; 2], 1.0),
                ),
                (
                    EngineError::InvalidAlpha(f32::NAN),
                    run(DType::BF16, &shape, &[1.0; 3], f32::NAN),
                ),
                (
                    EngineError::NonFiniteDirection { index: 1 },
                    run(DType::BF16, &shape, &[1.0, f32::INFINITY, 1.0], 1.0),
                ),
                (
                    EngineError::ByteLengthMismatch {
                        expected: 24,
                        got: 12,
                    },
                    run(DType::F32, &shape, &[1.0; 3], 1.0),
                ),
                (
                    EngineError::InvalidShape {
                        shape: vec![6],
                        reason: String::new(),
                    },
                    run(DType::BF16, &[6], &[1.0; 6], 1.0),
                ),
            ]
        };
        for (want, got) in cases {
            let got = got.unwrap_err();
            assert_eq!(
                std::mem::discriminant(&want),
                std::mem::discriminant(&got),
                "{want:?} vs {got:?}"
            );
        }
    }
}
