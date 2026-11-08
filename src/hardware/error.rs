use std::path::PathBuf;

/// Errors of the hardware module.
///
/// Sources are kept as strings so the type stays `Clone` (contract: every public
/// type is `Debug + Clone`); `io::Error` and `serde_json::Error` are not.
#[derive(Debug, Clone, thiserror::Error)]
pub enum HardwareError {
    #[error("chemin introuvable : {}", .0.display())]
    PathNotFound(PathBuf),

    #[error("aucun fichier .safetensors dans {}", .0.display())]
    NoSafetensors(PathBuf),

    #[error("extension invalide (attendu .safetensors) : {}", .0.display())]
    InvalidExtension(PathBuf),

    #[error("index corrompu {} : {message}", .path.display())]
    CorruptIndex { path: PathBuf, message: String },

    #[error("shard {} référencé par l'index mais introuvable", .shard.display())]
    MissingShard { shard: PathBuf },

    #[error("erreur d'E/S sur {} : {message}", .path.display())]
    Io { path: PathBuf, message: String },

    #[error("lecture de la RAM système impossible : {0}")]
    RamUnavailable(String),

    #[error("ressources insuffisantes : {reason}")]
    InsufficientResources { reason: String },
}

pub type Result<T> = std::result::Result<T, HardwareError>;

impl HardwareError {
    pub(crate) fn io(path: impl Into<PathBuf>, err: &std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            message: err.to_string(),
        }
    }
}
