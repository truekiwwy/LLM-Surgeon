use serde::Serialize;

use super::error::{EngineError, Result};
use super::index::DType;
use crate::kernel::{self, Deltas, Elem, KernelConfig};

// Scalar conversions and the reference cell loop live in `kernel::simd`; re-exported here
// for `project` and `decode_f32`.
pub(super) use crate::kernel::simd::{bf16_load, bf16_store, f16_load, f16_store, shift_cells};
pub use crate::kernel::ShiftStats;
// The tests below reach `f16` through `use super::*`.
#[cfg(test)]
use half::f16;

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

/// `W ∓= alpha * v` in place on a row-major `[rows, cols]` tensor. Every check runs before
/// the first write, so on `Err` not a single byte has changed. The kernel level comes from
/// `SURGEON_KERNEL` (best available by default); every level writes the same bytes.
pub fn apply_vector_shift(
    tensor_bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    direction: &[f32],
    alpha: f32,
    op: ShiftOp,
    axis: BroadcastAxis,
) -> Result<ShiftStats> {
    apply_vector_shift_with(
        tensor_bytes,
        dtype,
        shape,
        direction,
        alpha,
        op,
        axis,
        &KernelConfig::from_env(),
    )
}

/// [`apply_vector_shift`] with an explicit kernel level and block size.
#[allow(clippy::too_many_arguments)]
pub fn apply_vector_shift_with(
    tensor_bytes: &mut [u8],
    dtype: DType,
    shape: &[usize],
    direction: &[f32],
    alpha: f32,
    op: ShiftOp,
    axis: BroadcastAxis,
    kernel: &KernelConfig,
) -> Result<ShiftStats> {
    match ShiftPlan::new(tensor_bytes.len(), dtype, shape, direction, alpha, op, axis)? {
        Some(plan) => plan.run(tensor_bytes, 0, kernel),
        None => Ok(ShiftStats::default()),
    }
}

/// A validated shift: element type, row length and the precomputed deltas. Lets the
/// chunked writer shift a tensor block by block with the arithmetic of one pass.
#[derive(Debug, Clone)]
pub(crate) struct ShiftPlan {
    elem: Elem,
    cols: usize,
    deltas: Vec<f32>,
    axis: BroadcastAxis,
}

impl ShiftPlan {
    /// Every check of [`apply_vector_shift`]. `None` is a strict no-op (alpha 0 or empty
    /// tensor).
    pub(crate) fn new(
        byte_len: usize,
        dtype: DType,
        shape: &[usize],
        direction: &[f32],
        alpha: f32,
        op: ShiftOp,
        axis: BroadcastAxis,
    ) -> Result<Option<Self>> {
        let elem = match dtype {
            DType::BF16 => Elem::Bf16,
            DType::F16 => Elem::F16,
            DType::F32 => Elem::F32,
            DType::Other(name) => return Err(EngineError::UnsupportedDtype(name)),
        };
        let invalid_shape = |reason: &str| EngineError::InvalidShape {
            shape: shape.to_vec(),
            reason: reason.to_owned(),
        };
        let &[rows, cols] = shape else {
            return Err(invalid_shape("tenseur 2D attendu"));
        };
        let expected = rows
            .checked_mul(cols)
            .and_then(|n| n.checked_mul(elem.size()))
            .ok_or_else(|| invalid_shape("dépassement de capacité"))?;
        if byte_len != expected {
            return Err(EngineError::ByteLengthMismatch {
                expected,
                got: byte_len,
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
        if alpha == 0.0 || byte_len == 0 {
            return Ok(None);
        }
        // a - b == a + (-b) exactly in IEEE 754 (signed zeros included), so folding the sign
        // into one precomputed delta per v[i] stays bit-identical to `w - alpha * v[i]`.
        let sign = match op {
            ShiftOp::Subtract => -1.0,
            ShiftOp::Add => 1.0,
        };
        Ok(Some(Self {
            elem,
            cols,
            deltas: direction.iter().map(|v| sign * (alpha * v)).collect(),
            axis,
        }))
    }

    pub(crate) fn row_bytes(&self) -> usize {
        self.cols * self.elem.size()
    }

    /// Shifts `bytes`, whole rows starting at row `first_row` of the tensor.
    pub(crate) fn run(
        &self,
        bytes: &mut [u8],
        first_row: usize,
        kernel: &KernelConfig,
    ) -> Result<ShiftStats> {
        let rows = bytes.len() / self.row_bytes().max(1);
        let deltas = match self.axis {
            BroadcastAxis::Rows => Deltas::PerCol(&self.deltas),
            BroadcastAxis::Cols => {
                Deltas::PerRow(self.deltas.get(first_row..first_row + rows).ok_or(
                    EngineError::ByteLengthMismatch {
                        expected: self.deltas.len() * self.row_bytes(),
                        got: (first_row + rows) * self.row_bytes(),
                    },
                )?)
            }
        };
        kernel::shift_rows(bytes, self.elem, self.cols, deltas, kernel).map_err(|e| {
            EngineError::InvalidShape {
                shape: vec![rows, self.cols],
                reason: e.to_string(),
            }
        })
    }
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
