//! Writes a tiny GGUF v3 file (2 layers, hidden size 32, intermediate size 64, Q8_0 MLP
//! tensors) to try `surgeon inspect` and `surgeon apply --gguf` without a real model.
//!
//! ```text
//! cargo run --example make_fake_gguf -- <file.gguf>
//! ```
//!
//! Tensor names follow llama.cpp (`blk.{i}.ffn_{gate,up,down}.weight`); the weights are
//! N(0, 0.02²) from a fixed LCG, so a run is reproducible. The file is not loadable by
//! llama.cpp (no real vocabulary or attention tensors): it only feeds the patcher.

use std::error::Error;

use weight_surgeon::patch::gguf::{quant, Codec, GgmlType};

const Q8_0: u32 = 8;
const HIDDEN: usize = 32;
const INTER: usize = 64;
const LAYERS: usize = 2;
const ALIGN: usize = 32;

struct Lcg(u64);

impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller.
    fn normal(&mut self) -> f32 {
        let (u1, u2) = (self.uniform().max(1e-12), self.uniform());
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    }
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    b.extend((s.len() as u64).to_le_bytes());
    b.extend(s.as_bytes());
}

fn main() -> Result<(), Box<dyn Error>> {
    let out = std::env::args()
        .nth(1)
        .ok_or("usage: make_fake_gguf <file.gguf>")?;
    let codec = Codec::from_type(GgmlType(Q8_0)).ok_or("Q8_0 codec")?;
    let mut rng = Lcg(7);

    // (name, ne = [columns, rows], data)
    let mut tensors: Vec<(String, [u64; 2], Vec<u8>)> = Vec::new();
    for layer in 0..LAYERS {
        for proj in ["gate", "up", "down"] {
            let (cols, rows) = if proj == "down" {
                (INTER, HIDDEN)
            } else {
                (HIDDEN, INTER)
            };
            let weights: Vec<f32> = (0..rows * cols).map(|_| 0.02 * rng.normal()).collect();
            let mut data = vec![0u8; codec.row_bytes(cols) * rows];
            for (x, b) in weights
                .chunks(codec.block_elems())
                .zip(data.chunks_mut(codec.block_bytes()))
            {
                quant::quantize_block(codec, x, b).map_err(|_| "non-finite Q8_0 scale")?;
            }
            tensors.push((
                format!("blk.{layer}.ffn_{proj}.weight"),
                [cols as u64, rows as u64],
                data,
            ));
        }
    }

    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend(2u64.to_le_bytes()); // two metadata keys
    put_str(&mut b, "general.architecture");
    b.extend(8u32.to_le_bytes());
    put_str(&mut b, "qwen2");
    put_str(&mut b, "general.alignment");
    b.extend(4u32.to_le_bytes());
    b.extend((ALIGN as u32).to_le_bytes());

    let mut offset = 0usize;
    for (name, ne, data) in &tensors {
        put_str(&mut b, name);
        b.extend(2u32.to_le_bytes());
        ne.iter().for_each(|d| b.extend(d.to_le_bytes()));
        b.extend(Q8_0.to_le_bytes());
        b.extend((offset as u64).to_le_bytes());
        offset = (offset + data.len()).next_multiple_of(ALIGN);
    }
    b.resize(b.len().next_multiple_of(ALIGN), 0);
    let data_start = b.len();
    let mut offset = 0usize;
    for (_, _, data) in &tensors {
        b.resize(data_start + offset, 0);
        b.extend(data);
        offset = (offset + data.len()).next_multiple_of(ALIGN);
    }
    std::fs::write(&out, b)?;
    println!("{out}");
    Ok(())
}
