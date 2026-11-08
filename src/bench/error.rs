use std::path::PathBuf;

use crate::eval::EvalError;
use crate::extract::ExtractError;

/// Every failure of `surgeon bench`. The exit code is chosen by `app` (0/1/2/8/9/70), the
/// stable `error.kind` here.
#[derive(Debug, Clone, thiserror::Error)]
pub enum BenchError {
    #[error("argument invalide : {0}")]
    InvalidArgument(String),

    #[error("{} : {message}", .path.display())]
    Input { path: PathBuf, message: String },

    #[error("{} existe déjà (--overwrite pour le remplacer)", .0.display())]
    OutputExists(PathBuf),

    #[error("ressources insuffisantes : {0}")]
    InsufficientResources(String),

    #[error("{0}")]
    BinaryNotFound(String),

    #[error("vocabulaires différents : avant {before}, après {after}")]
    VocabMismatch { before: usize, after: usize },

    #[error("formes incompatibles : avant {before:?}, après {after:?}")]
    ShapeMismatch {
        before: Vec<usize>,
        after: Vec<usize>,
    },

    #[error(
        "logits non finis (NaN, +Inf ou ligne entièrement à -Inf) dans le dump {side}, positions {positions:?}"
    )]
    NonFiniteLogits {
        side: &'static str,
        positions: Vec<usize>,
    },

    #[error("{step} a échoué (code {code}) :\n{stderr_tail}")]
    ProcessFailed {
        step: String,
        code: i32,
        stderr_tail: String,
    },

    #[error("perplexité illisible dans la sortie de llama-perplexity ({reason}) :\n{stderr_tail}")]
    PplParse { reason: String, stderr_tail: String },

    #[error("délai dépassé après {after_ms} ms, processus arrêté")]
    Timeout { after_ms: u128 },

    #[error("conversion GGUF : {0}")]
    Conversion(String),

    #[error("génération impossible : {0}")]
    Generation(String),

    #[error("modèle : {0}")]
    Model(String),

    #[error("calcul : {0}")]
    Compute(String),

    #[error("écriture de {} : {message}", .path.display())]
    Io { path: PathBuf, message: String },

    #[error("bench annulé, aucun fichier écrit")]
    Cancelled,
}

impl BenchError {
    /// Stable `error.kind` of the JSON envelope.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::InvalidArgument(_) | Self::Input { .. } => "invalid_input",
            Self::OutputExists(_) => "output_exists",
            Self::InsufficientResources(_) => "insufficient_resources",
            Self::BinaryNotFound(_) => "binary_not_found",
            Self::VocabMismatch { .. } => "vocab_mismatch",
            Self::ShapeMismatch { .. } => "shape_mismatch",
            Self::NonFiniteLogits { .. } => "non_finite_logits",
            Self::ProcessFailed { .. } => "process_failed",
            Self::PplParse { .. } => "ppl_parse_failed",
            Self::Timeout { .. } => "timeout",
            Self::Conversion(_) => "conversion_failed",
            Self::Generation(_) => "generation_failed",
            Self::Model(_) => "model_error",
            Self::Compute(_) | Self::Io { .. } => "bench_failed",
            Self::Cancelled => "cancelled",
        }
    }

    pub(crate) fn input(path: &std::path::Path, message: impl std::fmt::Display) -> Self {
        Self::Input {
            path: path.to_path_buf(),
            message: message.to_string(),
        }
    }
}

impl From<EvalError> for BenchError {
    fn from(e: EvalError) -> Self {
        match e {
            EvalError::BinaryNotFound { .. } => Self::BinaryNotFound(e.to_string()),
            EvalError::Timeout { after_ms } => Self::Timeout { after_ms },
            EvalError::CommandFailed {
                step,
                code,
                stderr_tail,
            } => Self::ProcessFailed {
                step,
                code,
                stderr_tail,
            },
            other => Self::Generation(other.to_string()),
        }
    }
}

impl From<ExtractError> for BenchError {
    fn from(e: ExtractError) -> Self {
        match e {
            ExtractError::InvalidArgument(m) => Self::InvalidArgument(m),
            ExtractError::Prompts { message, .. } => Self::InvalidArgument(message),
            ExtractError::OutputExists(p) => Self::OutputExists(p),
            ExtractError::InsufficientResources(m) => Self::InsufficientResources(m),
            ExtractError::Cancelled => Self::Cancelled,
            ExtractError::Io { path, message } => Self::Io { path, message },
            ExtractError::Compute(m) | ExtractError::NonFiniteActivation { prompt: m } => {
                Self::Compute(m)
            }
            other => Self::Model(other.to_string()),
        }
    }
}

impl From<candle_core::Error> for BenchError {
    fn from(e: candle_core::Error) -> Self {
        Self::Compute(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, BenchError>;
