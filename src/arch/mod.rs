//! Data-driven description of model architectures: which tensors write into the residual
//! stream, where (SafeTensors and GGUF names) and with which orientation, plus the topology
//! facts that decide whether a weight-level `project_out` is exact.
//!
//! No dependency on `engine`, `patch` or `extract`: they read an [`ArchSpec`] to select
//! tensors (`engine::selector`, `patch::gguf::mapping`), to orient `project_out`, and to
//! gate an approximate ablation (`apply --accept-approximate`).
//!
//! ```text
//! detect (config.json model_type / architectures, GGUF general.architecture, --arch-spec)
//!   -> Resolution { spec, source, detected }
//! ArchSpec::role_of(name, format) -> TensorRole { role, layer, expert, layout }
//! ArchSpec::exactness() -> Exactness  (topology) ; with_biases(..) (the model's own biases)
//! ```
//!
//! Pattern syntax: dot-separated segments matched against the END of a tensor name (any
//! prefix: `model.`, `model.language_model.`...), `{layer}` and `{expert}` match a decimal
//! segment. `layers.{layer}.mlp.down_proj.weight` matches
//! `model.layers.3.mlp.down_proj.weight` with layer 3.
//!
//! Exactness of an ablation by weights (`project_out` on every writer): the residual stream
//! is the sum of the embeddings and of every sublayer output when each norm only feeds a
//! sublayer (pre-norm, parallel or sequential, RMSNorm or LayerNorm): erasing the direction
//! from every writer erases it from the stream, exactly. A learned norm applied to a
//! sublayer OUTPUT before the addition (Gemma 2/3 post-norms) rescales each coordinate and
//! brings part of the direction back: approximate. A writer bias, or a writer the spec
//! cannot patch, also leaves the direction in the stream. `tests/arch.rs` measures it.

mod builtin;

use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub use builtin::{builtin, builtins, legacy, LEGACY_ID};

/// What a tensor is to the residual stream.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// Input embeddings, one residual vector per row (`[vocab, hidden]`).
    Embed,
    /// Attention output projection, a writer (`[hidden, *]`).
    AttnOut,
    /// MLP gate projection, a reader.
    MlpGate,
    /// MLP up projection, a reader.
    MlpUp,
    /// MLP output projection: dense, per expert, fused experts or shared experts.
    MlpOut,
    /// MoE router: listed for audit, never modified.
    Router,
}

impl Role {
    pub fn is_writer(self) -> bool {
        matches!(self, Role::Embed | Role::AttnOut | Role::MlpOut)
    }
}

/// How a tensor is laid out with respect to the residual stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Layout {
    /// 2D `[hidden, in]` (`[out, in]`): the rows index the residual stream.
    OutIn,
    /// 2D `[n, hidden]`: each row is a residual vector (embeddings).
    Rows,
    /// 3D `[experts, hidden, in]`: fused experts, `[out, in]` per expert.
    ExpertsOutIn,
}

/// A name pattern, or a pattern with an explicit layout.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Pattern {
    /// The layout of the role: `rows` for `embed`, `out_in` otherwise.
    Name(String),
    Full {
        name: String,
        layout: Layout,
    },
}

impl Pattern {
    pub fn name(&self) -> &str {
        match self {
            Pattern::Name(name) | Pattern::Full { name, .. } => name,
        }
    }

    fn layout(&self, role: Role) -> Layout {
        match self {
            Pattern::Full { layout, .. } => *layout,
            Pattern::Name(_) if role == Role::Embed => Layout::Rows,
            Pattern::Name(_) => Layout::OutIn,
        }
    }
}

/// Patterns per role for one file format.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RoleNames {
    #[serde(default)]
    pub embed: Vec<Pattern>,
    #[serde(default)]
    pub attn_out: Vec<Pattern>,
    #[serde(default)]
    pub mlp_gate: Vec<Pattern>,
    #[serde(default)]
    pub mlp_up: Vec<Pattern>,
    #[serde(default)]
    pub mlp_out: Vec<Pattern>,
    #[serde(default)]
    pub router: Vec<Pattern>,
}

impl RoleNames {
    /// Every pattern with its role, in role order.
    pub fn iter(&self) -> impl Iterator<Item = (Role, &Pattern)> {
        [
            (Role::Embed, &self.embed),
            (Role::AttnOut, &self.attn_out),
            (Role::MlpGate, &self.mlp_gate),
            (Role::MlpUp, &self.mlp_up),
            (Role::MlpOut, &self.mlp_out),
            (Role::Router, &self.router),
        ]
        .into_iter()
        .flat_map(|(role, patterns)| patterns.iter().map(move |p| (role, p)))
    }

    pub fn of(&self, role: Role) -> &[Pattern] {
        match role {
            Role::Embed => &self.embed,
            Role::AttnOut => &self.attn_out,
            Role::MlpGate => &self.mlp_gate,
            Role::MlpUp => &self.mlp_up,
            Role::MlpOut => &self.mlp_out,
            Role::Router => &self.router,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NormKind {
    RmsNorm,
    LayerNorm,
}

/// Whether the output head reuses the input embeddings.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TiedRule {
    /// Key of `config.json` (`tie_word_embeddings`); its value decides.
    pub config_key: String,
    /// When the key is missing from `config.json`.
    pub default: bool,
}

/// One architecture. Built-ins come from [`builtins`]; `--arch-spec` reads one from JSON
/// (`surgeon schema --kind arch`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArchSpec {
    pub id: String,
    /// `model_type` values of `config.json`.
    #[serde(default)]
    pub hf_model_types: Vec<String>,
    /// `architectures` entries of `config.json`, tried when `model_type` is unknown.
    #[serde(default)]
    pub hf_architectures: Vec<String>,
    /// `general.architecture` values of a GGUF file.
    #[serde(default)]
    pub gguf_arch: Vec<String>,
    pub safetensors: RoleNames,
    #[serde(default)]
    pub gguf: RoleNames,
    pub norm_kind: NormKind,
    /// A learned norm on each sublayer output before it is added to the stream.
    pub post_sublayer_norm: bool,
    /// Attention and MLP read the same normalized input (Falcon, Command-R).
    #[serde(default)]
    pub parallel_block: bool,
    pub tied_embeddings: TiedRule,
    /// Writers of the residual stream the spec cannot patch (learned position embeddings,
    /// a fused writer...): an ablation by weights then leaves the direction in the stream.
    #[serde(default)]
    pub unprojected_writers: Vec<String>,
    /// Definitions the spec was checked against.
    #[serde(default)]
    pub sources: Vec<String>,
}

/// Exactness of `project_out` on every writer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Exactness {
    /// The direction leaves the residual stream entirely (up to float rounding).
    Exact,
    /// Part of the direction comes back; `apply` needs `--accept-approximate`.
    Approximate { reason: String },
    /// Refused by `apply`.
    Unsupported { reason: String },
}

impl Exactness {
    pub fn label(&self) -> &'static str {
        match self {
            Exactness::Exact => "exact",
            Exactness::Approximate { .. } => "approximate",
            Exactness::Unsupported { .. } => "unsupported",
        }
    }

    pub fn reason(&self) -> Option<&str> {
        match self {
            Exactness::Exact => None,
            Exactness::Approximate { reason } | Exactness::Unsupported { reason } => Some(reason),
        }
    }

    /// The worse of two classifications, reasons joined.
    fn and(self, other: Exactness) -> Exactness {
        let rank = |e: &Exactness| match e {
            Exactness::Exact => 0,
            Exactness::Approximate { .. } => 1,
            Exactness::Unsupported { .. } => 2,
        };
        match (rank(&self), rank(&other)) {
            (a, b) if a > b => self,
            (a, b) if a < b => other,
            _ => match (self, other) {
                (Exactness::Approximate { reason: a }, Exactness::Approximate { reason: b }) => {
                    Exactness::Approximate {
                        reason: format!("{a} ; {b}"),
                    }
                }
                (Exactness::Unsupported { reason: a }, Exactness::Unsupported { reason: b }) => {
                    Exactness::Unsupported {
                        reason: format!("{a} ; {b}"),
                    }
                }
                (same, _) => same,
            },
        }
    }
}

/// File format of the tensor names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    SafeTensors,
    Gguf,
}

/// What a name is to an architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRole {
    pub role: Role,
    pub layer: Option<usize>,
    pub expert: Option<usize>,
    pub layout: Layout,
}

/// Matches `pattern` against the end of `name` (the whole name when `anchored`):
/// `(layer, expert)` captured.
fn match_pattern(
    pattern: &str,
    name: &str,
    anchored: bool,
) -> Option<(Option<usize>, Option<usize>)> {
    let pattern: Vec<&str> = pattern.split('.').collect();
    let name: Vec<&str> = name.split('.').collect();
    if anchored && name.len() != pattern.len() {
        return None;
    }
    let tail = name.get(name.len().checked_sub(pattern.len())?..)?;
    let (mut layer, mut expert) = (None, None);
    for (p, n) in pattern.iter().zip(tail) {
        match *p {
            "{layer}" => layer = Some(n.parse().ok()?),
            "{expert}" => expert = Some(n.parse().ok()?),
            literal if literal == *n => {}
            _ => return None,
        }
    }
    Some((layer, expert))
}

impl ArchSpec {
    pub fn names(&self, format: Format) -> &RoleNames {
        match format {
            Format::SafeTensors => &self.safetensors,
            Format::Gguf => &self.gguf,
        }
    }

    /// The role of a tensor name, the first matching pattern in role order. GGUF names
    /// have no prefix: they must match the whole pattern.
    pub fn role_of(&self, name: &str, format: Format) -> Option<TensorRole> {
        let anchored = format == Format::Gguf;
        self.names(format).iter().find_map(|(role, pattern)| {
            let (layer, expert) = match_pattern(pattern.name(), name, anchored)?;
            Some(TensorRole {
                role,
                layer,
                expert,
                layout: pattern.layout(role),
            })
        })
    }

    /// The classification from the topology alone (no model at hand).
    pub fn exactness(&self) -> Exactness {
        let mut out = Exactness::Exact;
        if !self.unprojected_writers.is_empty() {
            out = out.and(Exactness::Unsupported {
                reason: format!(
                    "écrivains du flux résiduel non projetables : {}",
                    self.unprojected_writers.join(", ")
                ),
            });
        }
        let missing: Vec<&str> = [
            ("embed", &self.safetensors.embed),
            ("attn_out", &self.safetensors.attn_out),
            ("mlp_out", &self.safetensors.mlp_out),
        ]
        .iter()
        .filter(|(_, patterns)| patterns.is_empty())
        .map(|(role, _)| *role)
        .collect();
        if !missing.is_empty() {
            out = out.and(Exactness::Unsupported {
                reason: format!(
                    "rôle(s) écrivain(s) sans motif : {} (la direction n'est pas retirée de \
                     tous les écrivains)",
                    missing.join(", ")
                ),
            });
        }
        if self.post_sublayer_norm {
            out = out.and(Exactness::Approximate {
                reason: "une norme apprise est appliquée à la sortie de chaque sous-couche \
                         avant l'ajout au flux : son échelle élément par élément ne commute pas \
                         avec la projection et réintroduit une partie de la direction"
                    .into(),
            });
        }
        out
    }

    /// [`Self::exactness`] for a given model, whose writer biases (`.bias` tensors next to
    /// a writer) are not projected.
    pub fn exactness_with_biases(&self, writer_biases: &[String]) -> Exactness {
        match writer_biases {
            [] => self.exactness(),
            biases => self.exactness().and(Exactness::Approximate {
                reason: format!(
                    "biais d'écrivains non projetés ({} tenseur(s), par exemple {})",
                    biases.len(),
                    biases[0]
                ),
            }),
        }
    }

    /// Checks what the JSON schema cannot: non-empty id and patterns, placeholders,
    /// layouts that fit their role.
    pub fn validate(&self) -> Result<(), ArchError> {
        let mut issues = Vec::new();
        if self.id.trim().is_empty() {
            issues.push("id vide".to_owned());
        }
        for (format, names) in [("safetensors", &self.safetensors), ("gguf", &self.gguf)] {
            for (role, pattern) in names.iter() {
                let at = format!("{format}.{}", json_role(role));
                let name = pattern.name();
                let segments: Vec<&str> = name.split('.').collect();
                if name.is_empty() || segments.iter().any(|s| s.is_empty()) {
                    issues.push(format!(
                        "{at} : motif « {name} » vide ou avec un segment vide"
                    ));
                    continue;
                }
                for s in &segments {
                    if s.contains(['{', '}']) && !matches!(*s, "{layer}" | "{expert}") {
                        issues.push(format!(
                            "{at} : « {s} » dans « {name} » (seuls {{layer}} et {{expert}})"
                        ));
                    }
                }
                let has = |p: &str| segments.contains(&p);
                let layout = pattern.layout(role);
                let fits = match (role, layout) {
                    (Role::Embed, Layout::Rows) => !has("{layer}") && !has("{expert}"),
                    (Role::Embed, _) => false,
                    (Role::MlpOut, Layout::ExpertsOutIn) => has("{layer}") && !has("{expert}"),
                    (_, Layout::ExpertsOutIn | Layout::Rows) => false,
                    (_, Layout::OutIn) => has("{layer}"),
                };
                if !fits {
                    issues.push(format!(
                        "{at} : « {name} » ({layout:?}) : embed est `rows` sans {{layer}}, les \
                         autres rôles ont {{layer}} ; `experts_out_in` seulement pour mlp_out, \
                         sans {{expert}}"
                    ));
                }
            }
        }
        match issues.is_empty() {
            true => Ok(()),
            false => Err(ArchError::InvalidSpec(issues)),
        }
    }

    /// Whether `config.json` ties the output head to the embeddings.
    pub fn tied_in_config(&self, config: &serde_json::Value) -> bool {
        config
            .get(&self.tied_embeddings.config_key)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(self.tied_embeddings.default)
    }
}

fn json_role(role: Role) -> String {
    serde_json::to_value(role)
        .ok()
        .and_then(|v| v.as_str().map(str::to_owned))
        .unwrap_or_default()
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArchError {
    #[error("spécification d'architecture invalide : {}", .0.join(" ; "))]
    InvalidSpec(Vec<String>),
    #[error("{path} : {message}")]
    Read { path: String, message: String },
}

impl ArchError {
    pub fn kind(&self) -> &'static str {
        match self {
            ArchError::InvalidSpec(_) => "invalid_arch_spec",
            ArchError::Read { .. } => "invalid_input",
        }
    }
}

/// Where the architecture of a model came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    ConfigModelType,
    ConfigArchitectures,
    Gguf,
    ArchSpecFile,
    /// Nothing recognized: the historical Llama/Qwen naming, as before this module.
    Default,
}

/// The architecture used for a model.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Resolution {
    pub spec: ArchSpec,
    pub source: Source,
    /// What the file said (`model_type`, `general.architecture`), recognized or not.
    pub declared: Option<String>,
}

impl Resolution {
    pub fn detected(&self) -> bool {
        self.source != Source::Default
    }

    fn default(declared: Option<String>) -> Self {
        Self {
            spec: legacy().clone(),
            source: Source::Default,
            declared,
        }
    }
}

/// From a parsed `config.json`: `model_type`, then `architectures`, else the default.
pub fn detect_config(config: &serde_json::Value) -> Resolution {
    let model_type = config.get("model_type").and_then(|v| v.as_str());
    if let Some(spec) = model_type.and_then(|t| {
        builtins()
            .iter()
            .find(|s| s.hf_model_types.iter().any(|m| m == t))
    }) {
        return Resolution {
            spec: spec.clone(),
            source: Source::ConfigModelType,
            declared: model_type.map(str::to_owned),
        };
    }
    let architectures: Vec<&str> = config
        .get("architectures")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    if let Some(spec) = builtins().iter().find(|s| {
        s.hf_architectures
            .iter()
            .any(|a| architectures.contains(&a.as_str()))
    }) {
        return Resolution {
            spec: spec.clone(),
            source: Source::ConfigArchitectures,
            declared: model_type.map(str::to_owned),
        };
    }
    Resolution::default(model_type.map(str::to_owned))
}

/// From `general.architecture` of a GGUF file.
pub fn detect_gguf(architecture: Option<&str>) -> Resolution {
    match architecture.and_then(|a| {
        builtins()
            .iter()
            .find(|s| s.gguf_arch.iter().any(|g| g == a))
    }) {
        Some(spec) => Resolution {
            spec: spec.clone(),
            source: Source::Gguf,
            declared: architecture.map(str::to_owned),
        },
        None => Resolution::default(architecture.map(str::to_owned)),
    }
}

/// The `config.json` of a SafeTensors model (folder, file or index), parsed. `None` when
/// there is none or it is not JSON (the default naming then applies, as before).
pub fn read_config(model: &Path) -> Option<serde_json::Value> {
    let dir = match model.is_dir() {
        true => model,
        false => model.parent()?,
    };
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    serde_json::from_str(&text).ok()
}

/// A third-party spec from a JSON file, schema-checked (unknown fields refused) and
/// validated.
pub fn load_spec(path: &Path) -> Result<ArchSpec, ArchError> {
    let read = |message: String| ArchError::Read {
        path: path.display().to_string(),
        message,
    };
    let text = std::fs::read_to_string(path).map_err(|e| read(e.to_string()))?;
    let spec: ArchSpec =
        serde_json::from_str(&text).map_err(|e| ArchError::InvalidSpec(vec![e.to_string()]))?;
    spec.validate()?;
    Ok(spec)
}

/// The architecture of a model: `--arch-spec` when given, else detection (`gguf_arch` for
/// a GGUF file, `config.json` otherwise).
pub fn resolve(
    model: &Path,
    gguf_architecture: Option<Option<&str>>,
    arch_spec: Option<&Path>,
) -> Result<Resolution, ArchError> {
    if let Some(path) = arch_spec {
        return Ok(Resolution {
            spec: load_spec(path)?,
            source: Source::ArchSpecFile,
            declared: None,
        });
    }
    Ok(match gguf_architecture {
        Some(architecture) => detect_gguf(architecture),
        None => match read_config(model) {
            Some(config) => detect_config(&config),
            None => Resolution::default(None),
        },
    })
}

/// Ids with a native forward (extraction, generation, `bench dump`): `extract::forward`
/// accepts exactly these, the others are `unsupported_architecture` there.
pub const FORWARD_IDS: [&str; 3] = ["llama", "qwen2", "qwen3"];

/// How the classification of an architecture was checked numerically (`tests/arch.rs`):
/// on its real forward, or on a residual stack with its topology only.
pub fn verification(id: &str) -> &'static str {
    match FORWARD_IDS.contains(&id) {
        true => "forward",
        false => "topology",
    }
}

/// What `inspect`, `apply` and `arch detect` say about a model's architecture.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Report {
    pub id: String,
    pub source: Source,
    pub detected: bool,
    /// What the file declared, recognized or not.
    pub declared: Option<String>,
    /// Tensors of the model per role.
    pub roles: std::collections::BTreeMap<Role, usize>,
    pub writer_biases: Vec<String>,
    /// `None` when unknown (no `config.json`).
    pub tied_embeddings: Option<bool>,
    pub exactness: Exactness,
    pub forward: bool,
    pub verification: &'static str,
}

impl Report {
    /// `names`: every tensor of the model; `tied`: whether the head reuses the embeddings.
    pub fn new<'a>(
        resolution: &Resolution,
        names: impl IntoIterator<Item = &'a str>,
        format: Format,
        tied: Option<bool>,
    ) -> Self {
        let names: Vec<&str> = names.into_iter().collect();
        let spec = &resolution.spec;
        let mut roles = std::collections::BTreeMap::new();
        let mut writer_biases = Vec::new();
        for name in &names {
            let Some(role) = spec.role_of(name, format) else {
                continue;
            };
            *roles.entry(role.role).or_insert(0) += 1;
            let bias = name.strip_suffix(".weight").map(|s| format!("{s}.bias"));
            if let Some(bias) =
                bias.filter(|b| role.role.is_writer() && names.contains(&b.as_str()))
            {
                writer_biases.push(bias);
            }
        }
        writer_biases.sort();
        Self {
            id: spec.id.clone(),
            source: resolution.source,
            detected: resolution.detected(),
            declared: resolution.declared.clone(),
            roles,
            exactness: spec.exactness_with_biases(&writer_biases),
            writer_biases,
            tied_embeddings: tied,
            forward: FORWARD_IDS.contains(&spec.id.as_str()),
            verification: verification(&spec.id),
        }
    }

    /// One line for the human outputs.
    pub fn summary(&self) -> String {
        let source = match self.source {
            Source::ConfigModelType => "config.json model_type",
            Source::ConfigArchitectures => "config.json architectures",
            Source::Gguf => "general.architecture",
            Source::ArchSpecFile => "--arch-spec",
            Source::Default => "non détectée : nommage Llama/Qwen supposé",
        };
        let roles: Vec<String> = self
            .roles
            .iter()
            .map(|(role, n)| format!("{} {n}", json_role(*role)))
            .collect();
        let mut line = format!(
            "{} ({source}) · project_out {} · {}",
            self.id,
            self.exactness.label(),
            match roles.is_empty() {
                true => "aucun rôle résolu".to_owned(),
                false => roles.join(", "),
            }
        );
        if let Some(reason) = self.exactness.reason() {
            line.push_str(&format!(" · {reason}"));
        }
        line
    }
}

/// Why a weight ablation is refused: `error.kind` and message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    pub kind: &'static str,
    pub message: String,
}

/// The gate of a `project_out` (`project_out` false: nothing to decide). `exact` passes,
/// `approximate` needs `accept` and then returns its warning, `unsupported` is refused.
pub fn gate(report: &Report, project_out: bool, accept: bool) -> Result<Option<String>, Refusal> {
    if !project_out {
        return Ok(None);
    }
    match &report.exactness {
        Exactness::Exact => Ok(None),
        Exactness::Unsupported { reason } => Err(Refusal {
            kind: "unsupported_architecture",
            message: format!(
                "project_out refusé sur l'architecture {} : {reason}",
                report.id
            ),
        }),
        Exactness::Approximate { reason } if !accept => Err(Refusal {
            kind: "approximate_ablation",
            message: format!(
                "project_out sur l'architecture {} n'est qu'approximatif : {reason}. Relancez \
                 avec --accept-approximate (accept_approximate en MCP) pour l'accepter, puis \
                 mesurez le résultat",
                report.id
            ),
        }),
        Exactness::Approximate { reason } => Ok(Some(format!(
            "ablation approximative acceptée (--accept-approximate) sur {} : {reason}",
            report.id
        ))),
    }
}

/// JSON Schema (draft-07) of an architecture spec (`surgeon schema --kind arch`).
pub fn schema() -> serde_json::Value {
    serde_json::to_value(schemars::schema_for!(ArchSpec)).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patterns_match_the_end_of_a_name_with_any_prefix() {
        let p = "layers.{layer}.mlp.experts.{expert}.down_proj.weight";
        assert_eq!(
            match_pattern(p, "model.layers.3.mlp.experts.17.down_proj.weight", false),
            Some((Some(3), Some(17)))
        );
        assert_eq!(
            match_pattern(p, "model.layers.3.mlp.down_proj.weight", false),
            None
        );
        assert_eq!(
            match_pattern(p, "layers.x.mlp.experts.1.down_proj.weight", false),
            None
        );
        assert_eq!(
            match_pattern("embed_tokens.weight", "embed_tokens.weight", false),
            Some((None, None))
        );
        assert_eq!(
            match_pattern("embed_tokens.weight", "xembed_tokens.weight", false),
            None
        );
        assert_eq!(match_pattern("a.b.c", "b.c", false), None);
    }

    #[test]
    fn exactness_follows_the_topology_and_the_biases() {
        let mut spec = builtin("qwen2").unwrap().clone();
        assert_eq!(spec.exactness(), Exactness::Exact);
        let biased = spec.exactness_with_biases(&["model.layers.0.self_attn.o_proj.bias".into()]);
        assert_eq!(biased.label(), "approximate");
        spec.post_sublayer_norm = true;
        assert!(spec.exactness().reason().unwrap().contains("norme apprise"));
        spec.unprojected_writers = vec!["wpe".into()];
        assert_eq!(spec.exactness().label(), "unsupported");
    }
}
