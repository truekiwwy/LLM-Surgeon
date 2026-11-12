//! Errors of `surgeon graph` and their mapping to the CLI/MCP envelope.

use thiserror::Error;

use crate::app::{exit, Outcome};
use crate::surgery::SurgeryError;

pub type Result<T> = std::result::Result<T, GraphError>;

#[derive(Debug, Error)]
pub enum GraphError {
    /// Bad arguments or an inconsistent request, nothing written (exit 2).
    #[error("arguments invalides : {0}")]
    Usage(String),
    /// A model, cache or output path that cannot be read or written (exit 2).
    #[error("{path} : {message}")]
    Io { path: String, message: String },
    /// The model is structurally usable but not by this command (exit 2).
    #[error("modèle invalide : {0}")]
    Model(String),
    /// Numerical failure or an impossible computation (exit 7).
    #[error("calcul impossible : {0}")]
    Compute(String),
    /// Ctrl-C, the MCP cancellation flag or a `max_seconds` deadline (exit 1).
    #[error("opération annulée")]
    Cancelled,
}

impl GraphError {
    pub fn io(path: &std::path::Path, error: &std::io::Error) -> Self {
        Self::Io {
            path: path.display().to_string(),
            message: error.to_string(),
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Usage(_) => "invalid_input",
            Self::Io { .. } => "invalid_input",
            Self::Model(_) => "model_error",
            Self::Compute(_) => "compute_failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub fn exit_code(&self) -> i32 {
        match self {
            Self::Usage(_) | Self::Io { .. } | Self::Model(_) => exit::INVALID_INPUT,
            Self::Compute(_) => exit::EXTRACT_FAILED,
            Self::Cancelled => exit::GENERAL,
        }
    }
}

impl From<GraphError> for Outcome {
    fn from(error: GraphError) -> Self {
        Outcome::fail(error.exit_code(), &error).with_kind(error.kind())
    }
}

impl From<crate::engine::EngineError> for GraphError {
    fn from(error: crate::engine::EngineError) -> Self {
        Self::Model(error.to_string())
    }
}

impl From<crate::arch::ArchError> for GraphError {
    fn from(error: crate::arch::ArchError) -> Self {
        Self::Model(error.to_string())
    }
}

impl From<crate::patch::gguf::GgufError> for GraphError {
    fn from(error: crate::patch::gguf::GgufError) -> Self {
        Self::Model(error.to_string())
    }
}

impl From<crate::linalg::LinalgError> for GraphError {
    fn from(error: crate::linalg::LinalgError) -> Self {
        Self::Compute(error.to_string())
    }
}

/// `rsvd` shares the `.surgery` error type; its semantics map cleanly.
pub fn surgery(error: SurgeryError) -> GraphError {
    match error {
        SurgeryError::Usage(message) => GraphError::Usage(message),
        SurgeryError::Cancelled => GraphError::Cancelled,
        other => GraphError::Compute(other.to_string()),
    }
}

/// Shorthand used by the pipeline.
pub fn cancelled() -> GraphError {
    GraphError::Cancelled
}
