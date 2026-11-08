//! Pooling of a captured activation into one vector per prompt.

use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use super::error::{ExtractError, Result};

#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    Default,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
    clap::ValueEnum,
)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    /// Activation of the last token (the one that would predict the answer).
    #[default]
    Last,
    /// Mean over every token of the prompt.
    Mean,
}

impl Pooling {
    pub fn as_str(self) -> &'static str {
        match self {
            Pooling::Last => "last",
            Pooling::Mean => "mean",
        }
    }
}

/// `[1, seq, hidden]` -> `hidden` values in f64. Any NaN/Inf in the activation is an error
/// naming the prompt: it would poison the mean silently.
pub fn pool(activation: &Tensor, pooling: Pooling, prompt: &str) -> Result<Vec<f64>> {
    let (_, seq, hidden) = activation.dims3()?;
    if seq == 0 {
        return Err(ExtractError::Compute(format!(
            "activation vide pour {prompt}"
        )));
    }
    let values: Vec<f32> = activation.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err(ExtractError::NonFiniteActivation {
            prompt: prompt.to_owned(),
        });
    }
    Ok(match pooling {
        Pooling::Last => values[(seq - 1) * hidden..]
            .iter()
            .map(|&v| f64::from(v))
            .collect(),
        Pooling::Mean => {
            let mut sums = vec![0f64; hidden];
            for token in values.chunks_exact(hidden) {
                for (sum, &v) in sums.iter_mut().zip(token) {
                    *sum += f64::from(v);
                }
            }
            sums.iter().map(|s| s / seq as f64).collect()
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn activation(values: &[f32], seq: usize, hidden: usize) -> Tensor {
        Tensor::from_slice(values, (1, seq, hidden), &Device::Cpu).unwrap()
    }

    #[test]
    fn last_and_mean_pooling_exact_values() {
        let a = activation(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2);
        assert_eq!(pool(&a, Pooling::Last, "p").unwrap(), [5.0, 6.0]);
        assert_eq!(pool(&a, Pooling::Mean, "p").unwrap(), [3.0, 4.0]);
        let one = activation(&[0.5, -0.25], 1, 2);
        assert_eq!(pool(&one, Pooling::Mean, "p").unwrap(), [0.5, -0.25]);
        let bf16 = a.to_dtype(DType::BF16).unwrap();
        assert_eq!(pool(&bf16, Pooling::Last, "p").unwrap(), [5.0, 6.0]);
    }

    #[test]
    fn non_finite_activation_names_the_prompt() {
        let a = activation(&[1.0, f32::NAN, 3.0, 4.0], 2, 2);
        let err = pool(&a, Pooling::Last, "pos[3]").unwrap_err();
        assert!(matches!(&err, ExtractError::NonFiniteActivation { prompt } if prompt == "pos[3]"));
        let inf = activation(&[f32::INFINITY, 0.0], 1, 2);
        assert!(pool(&inf, Pooling::Mean, "neg[0]").is_err());
    }
}
