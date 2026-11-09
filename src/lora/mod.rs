//! Native LoRA merge and extract on SafeTensors (PEFT key layout).
//!
//! ```text
//! cli_args -> surgery::format (FactorDtype)
//! merge    -> keys, config, surgery::{lowrank, target, events, error}
//! extract  -> keys, config, surgery::{rsvd, events, error}
//! ```
//!
//! `merge` computes `W <- W + scale * (alpha / r) B A` (rsLoRA: `alpha / sqrt(r)`) with the
//! `.surgery` kernel. `extract` approximates `patched - base` per 2D tensor with a rank-r
//! randomized SVD and writes `B = U sqrt(S)`, `A = sqrt(S) V` as an adapter.
//!
//! PEFT compatibility: key names and `adapter_config.json` follow the PEFT layout, but the
//! files written by `extract` are NOT tested against HuggingFace PEFT in CI (no Python
//! dependency); only the round trip `extract -> merge` is.

pub mod cli_args;
pub mod config;
pub mod extract;
pub mod keys;
pub mod merge;

use cli_args::{LoraArgs, LoraCommand};
use half::{bf16, f16};

use crate::app::{AppDeps, CoreResult, ProgressSink};
use crate::engine::DType;
use crate::surgery::error::{Result, SurgeryError};

/// Shared by the CLI and the `surgeon_lora` MCP tool.
pub fn core_lora(args: &LoraArgs, deps: &AppDeps, sink: &mut dyn ProgressSink) -> CoreResult {
    match &args.command {
        LoraCommand::Merge(a) => merge::core_merge(a, deps, sink),
        LoraCommand::Extract(a) => extract::core_extract(a, deps, sink),
    }
}

/// Decodes little-endian `bytes` of `dtype` (BF16, F16, F32) into `out`.
pub(crate) fn decode_into(dtype: &DType, bytes: &[u8], out: &mut [f32]) -> Result<()> {
    let elem = dtype
        .size_of()
        .ok_or_else(|| SurgeryError::Usage(format!("dtype {dtype} non supporté")))?;
    if bytes.len() != out.len() * elem {
        return Err(SurgeryError::Usage(format!(
            "{} octets pour {} éléments {dtype}",
            bytes.len(),
            out.len()
        )));
    }
    match dtype {
        DType::BF16 => decode_with(bytes, out, |b: [u8; 2]| bf16::from_le_bytes(b).to_f32()),
        DType::F16 => decode_with(bytes, out, |b: [u8; 2]| f16::from_le_bytes(b).to_f32()),
        _ => decode_with(bytes, out, f32::from_le_bytes),
    }
    Ok(())
}

fn decode_with<const N: usize>(bytes: &[u8], out: &mut [f32], load: impl Fn([u8; N]) -> f32) {
    for (o, chunk) in out.iter_mut().zip(bytes.as_chunks::<N>().0) {
        *o = load(*chunk);
    }
}

/// `decode_into` into a fresh vector.
pub(crate) fn decode_vec(dtype: &DType, bytes: &[u8]) -> Result<Vec<f32>> {
    let elem = dtype.size_of().unwrap_or(1);
    let mut out = vec![0.0; bytes.len() / elem];
    decode_into(dtype, bytes, &mut out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_each_dtype_and_rejects_others() {
        let v = [1.5f32, -0.25];
        let bf: Vec<u8> = v
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect();
        let h: Vec<u8> = v
            .iter()
            .flat_map(|x| f16::from_f32(*x).to_le_bytes())
            .collect();
        let f: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        for (dtype, bytes) in [(DType::BF16, bf), (DType::F16, h), (DType::F32, f)] {
            assert_eq!(decode_vec(&dtype, &bytes).unwrap(), v);
        }
        assert!(decode_vec(&DType::Other("I8".into()), &[0]).is_err());
        assert!(decode_into(&DType::F32, &[0; 3], &mut [0.0]).is_err());
    }
}
