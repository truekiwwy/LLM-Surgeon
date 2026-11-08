use std::path::PathBuf;

#[derive(Debug, Clone, thiserror::Error)]
pub enum EvalError {
    #[error("binaire « {name} » introuvable : {hint}")]
    BinaryNotFound { name: String, hint: String },

    #[error("impossible de lancer {} : {message}", .program.display())]
    Spawn { program: PathBuf, message: String },

    #[error("erreur d'E/S sur {} : {message}", .path.display())]
    Io { path: PathBuf, message: String },

    /// The child was killed and reaped before this error was returned.
    #[error("délai dépassé après {after_ms} ms, processus arrêté")]
    Timeout { after_ms: u128 },

    #[error("l'étape « {step} » a échoué (code {code}) :\n{stderr_tail}")]
    CommandFailed {
        step: String,
        code: i32,
        stderr_tail: String,
    },

    #[error("quantification « {0} » refusée (attendu : {list})", list = super::gguf::ALLOWED_QUANTS.join(", "))]
    InvalidQuant(String),

    #[error("configuration invalide : {0}")]
    InvalidConfig(String),

    #[error("GGUF final invalide {} : {reason}", .path.display())]
    InvalidGguf { path: PathBuf, reason: String },

    #[error("fichier de prompts {} : {message}", .path.display())]
    Prompts { path: PathBuf, message: String },

    /// Generic failure reported by a `TextGenerator` implementation.
    #[error("génération : {0}")]
    Generator(String),
}

pub type Result<T> = std::result::Result<T, EvalError>;
