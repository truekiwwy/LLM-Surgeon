//! Error type shared by `surgery` and `lora`, and its mapping to the CLI/MCP envelope.

use thiserror::Error;

use super::format::FormatError;
use crate::app::{exit, Outcome};
use crate::engine::EngineError;

pub type Result<T> = std::result::Result<T, SurgeryError>;

#[derive(Debug, Error)]
pub enum SurgeryError {
    /// Phase 0 stub: the command is wired but its engine is not written yet.
    #[error("{0} : pas encore implémenté")]
    NotImplemented(&'static str),
    /// Bad arguments or inconsistent inputs, nothing touched (exit 2).
    #[error("arguments invalides : {0}")]
    Usage(String),
    /// Malformed or corrupt `.surgery` file (exit 2).
    #[error("fichier .surgery invalide : {0}")]
    Format(#[from] FormatError),
    #[error("{path} : {message}")]
    Io { path: String, message: String },
    #[error(transparent)]
    Engine(#[from] EngineError),
    #[error("opération annulée")]
    Cancelled,
    /// A factor does not fit the chosen `factor_dtype` (f16 overflow): the caller may retry
    /// with bf16.
    #[error("{tensor} : un facteur déborde de {dtype}")]
    FactorOverflow { tensor: String, dtype: &'static str },
    /// Execution failed after validation (NaN/Inf produced, numeric failure...), exit 3.
    #[error("{0}")]
    Failed(String),
}

impl SurgeryError {
    pub fn io(path: &std::path::Path, e: &std::io::Error) -> Self {
        Self::Io {
            path: path.display().to_string(),
            message: e.to_string(),
        }
    }

    /// Stable machine-readable category (`error.kind` of the envelope).
    pub fn kind(&self) -> &'static str {
        match self {
            Self::NotImplemented(_) => "not_implemented",
            Self::Usage(_) => "invalid_arguments",
            Self::Format(_) => "invalid_surgery",
            Self::Io { .. } | Self::Engine(_) => "model_error",
            Self::Cancelled => "cancelled",
            Self::FactorOverflow { .. } => "factor_overflow",
            Self::Failed(_) => "execution_failed",
        }
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            Self::NotImplemented(_) => exit::INTERNAL,
            Self::Usage(_) | Self::Format(_) => exit::INVALID_INPUT,
            Self::Io { .. } | Self::Engine(_) | Self::Cancelled => exit::GENERAL,
            Self::FactorOverflow { .. } | Self::Failed(_) => exit::EXECUTION_FAILED,
        }
    }
}

impl From<SurgeryError> for Outcome {
    fn from(e: SurgeryError) -> Self {
        Outcome::fail(e.exit_code(), &e).with_kind(e.kind())
    }
}
