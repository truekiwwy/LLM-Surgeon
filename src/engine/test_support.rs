//! Test-only helpers: fake safetensors files, an independent naive reference for the
//! shift, and a deterministic RNG.

use std::path::Path;

use half::{bf16, f16};

use super::index::DType;
use super::shift::{BroadcastAxis, ShiftOp};

/// Raw safetensors bytes: u64 LE header length + JSON header + data.
pub fn raw_file(header_json: &str, data: &[u8]) -> Vec<u8> {
    let mut out = (header_json.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header_json.as_bytes());
    out.extend_from_slice(data);
    out
}

/// Writes tensors back to back, in the given order, so offsets are predictable.
pub fn write_fake_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        header.insert(
            (*name).to_owned(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let json = serde_json::Value::Object(header).to_string();
    std::fs::write(path, raw_file(&json, &data)).unwrap();
}

pub fn encode(dtype: &DType, values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&x| match dtype {
            DType::BF16 => bf16::from_f32(x).to_bits().to_le_bytes().to_vec(),
            DType::F16 => f16::from_f32(x).to_bits().to_le_bytes().to_vec(),
            DType::F32 => x.to_le_bytes().to_vec(),
            DType::Other(_) => unreachable!(),
        })
        .collect()
}

/// Naive sequential reference written straight from the spec, sharing no code with
/// `apply_vector_shift`: read bits, f32 math `w - alpha * v`, round, write bits.
pub fn reference_shift(
    dtype: &DType,
    [rows, cols]: [usize; 2],
    data: &[u8],
    v: &[f32],
    alpha: f32,
    op: ShiftOp,
    axis: BroadcastAxis,
) -> Vec<u8> {
    let mut out = data.to_vec();
    for r in 0..rows {
        for c in 0..cols {
            let vi = match axis {
                BroadcastAxis::Rows => v[c],
                BroadcastAxis::Cols => v[r],
            };
            let delta = alpha * vi;
            let apply = |w: f32| match op {
                ShiftOp::Subtract => w - delta,
                ShiftOp::Add => w + delta,
            };
            let i = r * cols + c;
            match dtype {
                DType::BF16 => {
                    let w = bf16::from_bits(u16::from_le_bytes([data[2 * i], data[2 * i + 1]]));
                    let new = bf16::from_f32(apply(w.to_f32())).to_bits().to_le_bytes();
                    out[2 * i..2 * i + 2].copy_from_slice(&new);
                }
                DType::F16 => {
                    let w = f16::from_bits(u16::from_le_bytes([data[2 * i], data[2 * i + 1]]));
                    let new = f16::from_f32(apply(w.to_f32())).to_bits().to_le_bytes();
                    out[2 * i..2 * i + 2].copy_from_slice(&new);
                }
                DType::F32 => {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(&data[4 * i..4 * i + 4]);
                    let new = apply(f32::from_le_bytes(b)).to_le_bytes();
                    out[4 * i..4 * i + 4].copy_from_slice(&new);
                }
                DType::Other(_) => unreachable!(),
            }
        }
    }
    out
}

/// Deterministic LCG (Knuth MMIX constants), good enough for test data.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }

    pub fn f32_in(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * (self.next_u32() as f32 / (1u64 << 31) as f32)
    }
}
