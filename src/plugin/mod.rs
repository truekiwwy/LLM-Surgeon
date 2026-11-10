//! Extension API `v0` of the Surgery Engine: the direction methods of `surgeon extract`
//! as a trait, a registry, and the three built-in methods implemented through it.
//!
//! `v0` is unstable: any release may change it. A third-party method is compiled into a
//! program that depends on this library (static linking, no dynamic loading, no code run
//! from a file), registered in a [`Registry`], then run with
//! [`crate::extract::run_extraction_plugin`]. `docs/PLUGINS.md` walks through
//! `examples/plugin_extraction_example.rs`.
//!
//! Who does what during an extraction:
//!
//! ```text
//! engine (extract)                          method (ExtractionMethod)
//! validate request, options ------------->  validate(params)          before any forward
//! requires_paired, declared sidecars        (describe)
//! forward pass, capture, pooling
//! ActivationSet { pos, neg } ------------>  extract(ctx, acts, cancel) -> ExtractionOutput
//! check direction (size, finite, unit)
//! statistics along the direction (d', threshold, projections)
//! header = engine keys + method metadata + `method` (absent for mean_diff)
//! <out>, <out>.stats.json, <out>.<suffix>.safetensors   written together, atomically
//! ```
//!
//! The `surgeon` binary only knows [`Registry::with_builtin`]: its CLI (`--method`) and its
//! MCP server (`method`) accept exactly `mean_diff`, `svd` and `leace`.

#![warn(missing_docs)]

mod builtin;
mod registry;

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

use serde::{Serialize, Serializer};
use serde_json::{Map, Value};

use crate::extract::{ExtractError, MethodDiagnostics, Pooling, Site};

pub(crate) use builtin::Builtin;
pub use registry::{Origin, Registry};

/// Version of this API, reported by `surgeon plugins list`.
pub const API_VERSION: &str = "v0";

/// Identifier of the extraction-method extension point.
pub const EXTRACTION_METHOD_POINT: &str = "extraction_method";

/// Method used when none is given (CLI default, MCP default).
pub const DEFAULT_METHOD: &str = "mean_diff";

/// A way to turn the pooled activations of two prompt sets into one unit direction.
///
/// Implementations must be deterministic for a given input: the engine promises
/// reproducible files.
pub trait ExtractionMethod: Send + Sync {
    /// Stable identifier, `[a-z][a-z0-9_]{1,31}` (checked by [`Registry::register`]).
    fn id(&self) -> &'static str;

    /// Title, summary, parameters and requirements, read before any computation.
    fn describe(&self) -> MethodInfo;

    /// Checks `params` before the forward pass, so a bad value costs nothing.
    /// The default accepts everything.
    fn validate(&self, params: &Value) -> Result<(), PluginError> {
        let _ = params;
        Ok(())
    }

    /// The direction. A long computation should return [`PluginError::Cancelled`] soon
    /// after `cancel` becomes true; nothing is written then.
    fn extract(
        &self,
        ctx: &ExtractContext,
        acts: &ActivationSet,
        cancel: &AtomicBool,
    ) -> Result<ExtractionOutput, PluginError>;
}

/// What a method declares about itself.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MethodInfo {
    /// Short name for humans.
    pub title: &'static str,
    /// One sentence: what the direction is.
    pub summary: &'static str,
    /// Parameters read from [`ExtractContext::params`].
    pub params: Vec<ParamInfo>,
    /// Needs as many positive as negative prompts, matched by index.
    pub requires_paired: bool,
    /// Suffixes of the extra files the method may write, `<out>.<suffix>.safetensors`.
    /// Same rule as identifiers. The engine refuses an existing one without `overwrite`
    /// before any computation, and refuses an undeclared one after.
    pub sidecars: Vec<&'static str>,
}

/// One parameter of a method.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ParamInfo {
    /// Key in the `params` object.
    pub name: &'static str,
    /// JSON type of the value.
    pub kind: ParamKind,
    /// Value used when the key is absent.
    pub default: Value,
    /// What it changes.
    pub summary: &'static str,
}

/// JSON type of a parameter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamKind {
    /// `true` or `false`.
    Boolean,
    /// Any finite number.
    Number,
    /// A whole number.
    Integer,
    /// A JSON array.
    Array,
    /// A string.
    String,
}

/// Where the activations come from, and the method's parameters.
#[derive(Debug, Clone, Copy)]
pub struct ExtractContext<'a> {
    /// Size of each activation row, and of the direction to return.
    pub hidden_size: usize,
    /// Layer the activations were captured from (0-based).
    pub layer: usize,
    /// Capture site inside the layer.
    pub site: Site,
    /// How the token activations of a prompt were pooled into one row.
    pub pooling: Pooling,
    /// The method's parameters as given by the caller (`null` when none), already
    /// accepted by [`ExtractionMethod::validate`].
    pub params: &'a Value,
}

/// Pooled activations, one row of `hidden_size` values per prompt.
#[derive(Debug, Clone, Copy)]
pub struct ActivationSet<'a> {
    /// Positive prompts (label `true`), in file order.
    pub pos: &'a [Vec<f64>],
    /// Negative prompts (label `false`), in file order.
    pub neg: &'a [Vec<f64>],
}

impl<'a> ActivationSet<'a> {
    /// Every row with its label, positives first.
    pub fn labeled(&self) -> impl Iterator<Item = (&'a [f64], bool)> {
        let pos = self.pos.iter().map(|r| (r.as_slice(), true));
        pos.chain(self.neg.iter().map(|r| (r.as_slice(), false)))
    }
}

/// What a method returns.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ExtractionOutput {
    /// Unit direction (L2 norm 1 within 1e-6, `hidden_size` finite values). Kept in f64
    /// so the separation statistics are computed on it exactly; the file stores it as f32.
    pub direction: Vec<f64>,
    /// Written as `method_diagnostics` in the report and in `<out>.stats.json`.
    pub diagnostics: Option<Diagnostics>,
    /// Header entries of `<out>` and of every sidecar. A key the engine writes itself
    /// (`model`, `layer`, `n_pos`, `method`...) is refused.
    pub metadata: BTreeMap<String, String>,
    /// Extra files, each with a suffix declared in [`MethodInfo::sidecars`].
    pub sidecars: Vec<Sidecar>,
    /// 2D sidecar tensors, by `(suffix, tensor name)`: a listed tensor holds that many
    /// directions of `hidden_size` values and is written with the shape
    /// `[rows, hidden_size]`, one direction per row. The others stay 1D.
    pub sidecar_rows: BTreeMap<(String, String), usize>,
    /// Shown after the engine's own warnings.
    pub warnings: Vec<String>,
}

impl ExtractionOutput {
    /// A direction and nothing else.
    pub fn new(direction: Vec<f64>) -> Self {
        Self {
            direction,
            ..Self::default()
        }
    }
}

/// Diagnostics of a method.
#[derive(Debug, Clone, PartialEq)]
pub enum Diagnostics {
    /// Built-in methods: typed, so their files keep the field order they always had.
    Builtin(MethodDiagnostics),
    /// Any JSON object. The engine sets its `method` key to the method id.
    Json(Map<String, Value>),
}

impl Serialize for Diagnostics {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Builtin(d) => d.serialize(serializer),
            Self::Json(map) => map.serialize(serializer),
        }
    }
}

/// An extra `.safetensors` file next to the direction.
#[derive(Debug, Clone, PartialEq)]
pub struct Sidecar {
    /// Declared suffix: the file is `<out>.<suffix>.safetensors`.
    pub suffix: &'static str,
    /// 1D F32 tensors, in file order, unique non-empty names.
    pub tensors: Vec<(String, Vec<f32>)>,
    /// Entries of this file's header only. The engine adds the method metadata and the
    /// provenance (`model`, `layer`, `site`, `pooling`, `n_pos`, `n_neg`,
    /// `surgeon_version`); a key already present is refused.
    pub metadata: BTreeMap<String, String>,
}

/// Errors of the extension API. Converted to [`ExtractError`] by the engine, so they get
/// the exit codes of `surgeon extract`.
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    /// Bad `params` (exit code 2).
    #[error("paramètres invalides : {0}")]
    InvalidParams(String),
    /// The computation failed (exit code 7).
    #[error("{0}")]
    Failed(String),
    /// Stopped on request; nothing is written (exit code 1).
    #[error("extraction annulée")]
    Cancelled,
    /// The method returned something the engine cannot write (exit code 7).
    #[error("sortie de la méthode invalide : {0}")]
    InvalidOutput(String),
    /// [`Registry::register`]: the identifier breaks `[a-z][a-z0-9_]{1,31}`.
    #[error("identifiant invalide {0:?} : attendu [a-z][a-z0-9_]{{1,31}}")]
    InvalidId(String),
    /// [`Registry::register`]: the identifier belongs to a built-in method.
    #[error("identifiant réservé à une méthode intégrée : {0}")]
    ReservedId(String),
    /// [`Registry::register`]: the identifier is already registered.
    #[error("identifiant déjà enregistré : {0}")]
    DuplicateId(String),
    /// An engine error, kept as is (the built-in methods use it).
    #[error(transparent)]
    Extract(#[from] ExtractError),
}

impl From<PluginError> for ExtractError {
    fn from(e: PluginError) -> Self {
        match e {
            PluginError::Extract(e) => e,
            PluginError::Cancelled => ExtractError::Cancelled,
            PluginError::InvalidParams(m) => ExtractError::InvalidArgument(m),
            PluginError::Failed(m) => ExtractError::Compute(m),
            e @ PluginError::InvalidOutput(_) => ExtractError::Compute(e.to_string()),
            e => ExtractError::InvalidArgument(e.to_string()),
        }
    }
}

/// `true` for `[a-z][a-z0-9_]{1,31}`: method identifiers and sidecar suffixes.
pub fn is_valid_id(id: &str) -> bool {
    let bytes = id.as_bytes();
    (2..=32).contains(&bytes.len())
        && bytes[0].is_ascii_lowercase()
        && bytes[1..]
            .iter()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || *b == b'_')
}

/// The message serde gives for an unknown enum variant, so a front-end that validates
/// against a registry answers exactly as the former `Method` enum did.
pub fn unknown_method_message(id: &str, known: &[&str]) -> String {
    let quoted: Vec<String> = known.iter().map(|k| format!("`{k}`")).collect();
    let expected = match quoted.as_slice() {
        [] => return format!("unknown variant `{id}`, there are no variants"),
        [one] => one.clone(),
        [a, b] => format!("{a} or {b}"),
        many => format!("one of {}", many.join(", ")),
    };
    format!("unknown variant `{id}`, expected {expected}")
}

/// `surgeon plugins list`: the extension points and what implements them.
#[derive(Debug, Clone, Serialize)]
pub struct Catalog {
    /// [`API_VERSION`].
    pub api_version: &'static str,
    /// Implemented extension points only; the roadmap is in `docs/ARCHITECTURE.md`.
    pub extension_points: Vec<ExtensionPoint>,
}

/// One extension point.
#[derive(Debug, Clone, Serialize)]
pub struct ExtensionPoint {
    /// [`EXTRACTION_METHOD_POINT`].
    pub id: &'static str,
    /// Path of the trait to implement.
    pub rust_trait: &'static str,
    /// Registered implementations, built-in first, then by identifier.
    pub implementations: Vec<Implementation>,
}

/// One registered implementation.
#[derive(Debug, Clone, Serialize)]
pub struct Implementation {
    /// [`ExtractionMethod::id`].
    pub id: &'static str,
    /// Built into the engine, or registered by the program.
    pub origin: Origin,
    /// [`ExtractionMethod::describe`].
    #[serde(flatten)]
    pub info: MethodInfo,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identifiers_follow_the_rule() {
        for ok in ["ab", "mean_diff", "m2", "a_", &"x".repeat(32)] {
            assert!(is_valid_id(ok), "{ok}");
        }
        for bad in [
            "",
            "a",
            "1ab",
            "_ab",
            "Ab",
            "a-b",
            "a b",
            "é_a",
            &"x".repeat(33),
        ] {
            assert!(!is_valid_id(bad), "{bad}");
        }
    }

    #[test]
    fn unknown_method_message_is_the_serde_one() {
        use serde::de::Error as _;
        let serde = |known: &'static [&'static str]| {
            serde::de::value::Error::unknown_variant("pca", known).to_string()
        };
        for known in [
            &[][..],
            &["svd"],
            &["svd", "leace"],
            &["mean_diff", "svd", "leace"],
        ] {
            assert_eq!(unknown_method_message("pca", known), serde(known));
        }
    }

    #[test]
    fn labeled_rows_put_positives_first() {
        let (pos, neg) = (vec![vec![1.0]], vec![vec![2.0], vec![3.0]]);
        let acts = ActivationSet {
            pos: &pos,
            neg: &neg,
        };
        let labels: Vec<(f64, bool)> = acts.labeled().map(|(r, l)| (r[0], l)).collect();
        assert_eq!(labels, [(1.0, true), (2.0, false), (3.0, false)]);
    }
}
