use std::path::PathBuf;

/// Errors of the engine module. Sources are kept as strings so the type stays `Clone`.
#[derive(Debug, Clone, thiserror::Error)]
pub enum EngineError {
    #[error("erreur d'E/S sur {} : {message}", .path.display())]
    Io { path: PathBuf, message: String },

    #[error("fichier trop petit ({len} octets, minimum 8) : {}", .path.display())]
    FileTooSmall { path: PathBuf, len: u64 },

    #[error("longueur de header aberrante (dépasse le fichier ou 100 Mo) : {}", .path.display())]
    InvalidHeaderLength { path: PathBuf },

    #[error("header JSON invalide dans {} : {message}", .path.display())]
    InvalidHeaderJson { path: PathBuf, message: String },

    #[error("offsets invalides (chevauchement, trou ou inversion) pour {tensor} dans {}", .path.display())]
    InvalidOffsets { path: PathBuf, tensor: String },

    #[error("taille d'un tenseur différente de produit(shape) x taille du dtype dans {}", .path.display())]
    TensorSizeMismatch { path: PathBuf },

    #[error("zone de données incohérente avec la taille du fichier (tronqué ou offsets hors bornes) : {}", .path.display())]
    DataSizeMismatch { path: PathBuf },

    #[error("tenseur introuvable : {0}")]
    TensorNotFound(String),

    #[error("dtype non supporté : {0} (attendu BF16, F16 ou F32)")]
    UnsupportedDtype(String),

    #[error("shape invalide {shape:?} : {reason}")]
    InvalidShape { shape: Vec<usize>, reason: String },

    #[error("taille des octets incohérente : attendu {expected}, reçu {got}")]
    ByteLengthMismatch { expected: usize, got: usize },

    #[error("longueur du vecteur direction incohérente : attendu {expected}, reçu {got}")]
    DirectionLengthMismatch { expected: usize, got: usize },

    #[error("alpha doit être fini, reçu {0}")]
    InvalidAlpha(f32),

    #[error("le vecteur direction contient une valeur non finie à l'index {index}")]
    NonFiniteDirection { index: usize },

    #[error("vecteur direction illisible depuis {} : {message}", .path.display())]
    InvalidDirectionSource { path: PathBuf, message: String },
}

pub type Result<T> = std::result::Result<T, EngineError>;

impl EngineError {
    pub(crate) fn io(path: impl Into<PathBuf>, err: &std::io::Error) -> Self {
        Self::Io {
            path: path.into(),
            message: err.to_string(),
        }
    }
}
