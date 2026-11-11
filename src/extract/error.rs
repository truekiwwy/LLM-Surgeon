use std::path::PathBuf;

#[derive(Debug, Clone, thiserror::Error)]
pub enum ExtractError {
    #[error("argument invalide : {0}")]
    InvalidArgument(String),

    #[error("prompts {set} : {message}")]
    Prompts { set: String, message: String },

    #[error("{} existe déjà (--overwrite pour le remplacer)", .0.display())]
    OutputExists(PathBuf),

    #[error("ressources insuffisantes : {0}")]
    InsufficientResources(String),

    #[error("config.json invalide : {0}")]
    Config(String),

    #[error(
        "architecture « {model_type} » non supportée par le forward natif (extraction, \
         génération, bench dump) : Qwen2/Qwen2.5, Llama 2/3, Mistral et Qwen3 seulement ; ses \
         poids restent patchables (surgeon arch list)"
    )]
    UnsupportedArchitecture { model_type: String },

    #[error("tokenizer : {0}")]
    Tokenizer(String),

    #[error("tenseur manquant : {0}")]
    MissingTensor(String),

    #[error("tenseur {name} : forme {found:?}, attendu {expected:?}")]
    TensorShape {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    #[error("tenseur {name} : dtype {dtype} non supporté (bf16, f16 ou f32)")]
    UnsupportedDtype { name: String, dtype: String },

    #[error("poids : {0}")]
    Weights(String),

    #[error("calcul : {0}")]
    Compute(String),

    #[error("activation non finie (NaN/Inf) pour le prompt {prompt}")]
    NonFiniteActivation { prompt: String },

    #[error("direction dégénérée : ‖Δ‖ = {norm:e} (prompts positifs et négatifs indiscernables)")]
    DegenerateDirection { norm: f64 },

    #[error("écriture de {} : {message}", .path.display())]
    Io { path: PathBuf, message: String },

    #[error("extraction annulée, aucun fichier écrit")]
    Cancelled,

    #[error(
        "méthode svd : prompts non appariés ({pos} positifs, {neg} négatifs) ; il en faut autant, appariés par index"
    )]
    UnpairedPrompts { pos: usize, neg: usize },
}

impl ExtractError {
    /// Stable `error.kind` of the JSON envelope.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::InvalidArgument(_) | Self::Prompts { .. } => "invalid_input",
            Self::UnpairedPrompts { .. } => "unpaired_prompts",
            Self::OutputExists(_) => "output_exists",
            Self::InsufficientResources(_) => "insufficient_resources",
            Self::UnsupportedArchitecture { .. } => "unsupported_architecture",
            Self::Config(_)
            | Self::Tokenizer(_)
            | Self::MissingTensor(_)
            | Self::TensorShape { .. }
            | Self::UnsupportedDtype { .. }
            | Self::Weights(_) => "model_error",
            Self::NonFiniteActivation { .. } => "non_finite_activation",
            Self::DegenerateDirection { .. } => "degenerate_direction",
            Self::Compute(_) | Self::Io { .. } => "extract_failed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl From<candle_core::Error> for ExtractError {
    fn from(e: candle_core::Error) -> Self {
        Self::Compute(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ExtractError>;
