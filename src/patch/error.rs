use std::path::PathBuf;

use serde::Serialize;

/// One semantic problem in a patch, located by a JSON path such as `operations[1].alpha`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ValidationIssue {
    pub path: String,
    pub message: String,
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} : {}", self.path, self.message)
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum PatchError {
    #[error("patch illisible {} : {message}", .path.display())]
    PatchUnreadable { path: PathBuf, message: String },

    /// Malformed JSON or schema violation (unknown field, wrong type...).
    #[error("patch JSON invalide (ligne {line}, colonne {column}) : {message}")]
    InvalidJson {
        message: String,
        line: usize,
        column: usize,
    },

    #[error("patch invalide :\n{}", .0.iter().map(|i| format!("  - {i}")).collect::<Vec<_>>().join("\n"))]
    Invalid(Vec<ValidationIssue>),

    #[error("erreur d'E/S sur {} : {message}", .path.display())]
    Io { path: PathBuf, message: String },

    /// Backup creation or verification failed. Nothing was modified.
    #[error("backup : {0}")]
    Backup(String),

    /// Error raised by an external backend (engine, hardware...), converted by the integration.
    #[error("backend : {0}")]
    Backend(String),

    /// `rolled_back`: the original was restored. `left_modified`: no backup was available
    /// and the model may be partially patched.
    #[error("échec de l'opération {op_id} : {cause} (rollback : {rolled_back}, fichier laissé modifié : {left_modified})")]
    ExecutionFailed {
        op_id: String,
        cause: String,
        rolled_back: bool,
        left_modified: bool,
    },

    #[error("CRITIQUE : rollback impossible après l'échec de {op_id} : {cause}")]
    RollbackFailed { op_id: String, cause: String },
}

pub type Result<T> = std::result::Result<T, PatchError>;

impl PatchError {
    pub(crate) fn io(path: impl Into<PathBuf>, err: &std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            message: err.to_string(),
        }
    }
}

/// Process exit codes, see `crate::cli`.
pub mod exit_code {
    pub const OK: i32 = 0;
    /// Patch unreadable, malformed or semantically invalid. Nothing was touched.
    pub const INVALID_PATCH: i32 = 2;
    /// Execution error; the original is intact (never touched, or restored from backup).
    pub const EXECUTION_FAILED: i32 = 3;
    /// Rollback failed, or no backup and the model was left partially patched.
    pub const CRITICAL: i32 = 4;
}

impl From<&PatchError> for i32 {
    fn from(err: &PatchError) -> i32 {
        match err {
            PatchError::PatchUnreadable { .. }
            | PatchError::InvalidJson { .. }
            | PatchError::Invalid(_) => exit_code::INVALID_PATCH,
            PatchError::RollbackFailed { .. }
            | PatchError::ExecutionFailed {
                left_modified: true,
                ..
            } => exit_code::CRITICAL,
            PatchError::Io { .. }
            | PatchError::Backup(_)
            | PatchError::Backend(_)
            | PatchError::ExecutionFailed { .. } => exit_code::EXECUTION_FAILED,
        }
    }
}
