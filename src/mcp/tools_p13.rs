//! MCP tools `surgeon_diff` and `surgeon_lora`: argument types (their schemas are the
//! `inputSchema`s), definitions, and execution through `core_diff` / `core_lora`, the same
//! cores as the CLI. The tools.rs helpers are private, so the few lines needed are repeated.

use std::path::{Path, PathBuf};

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};
use crate::lora::cli_args::{LoraArgs, LoraCommand, LoraExtractArgs, LoraMergeArgs};
use crate::lora::core_lora;
use crate::patch::ValidationIssue;
use crate::surgery::cli_args::{DiffArgs, OnHighError};
use crate::surgery::core_diff;
use crate::surgery::format::FactorDtype;

pub const TOOL_DIFF: &str = "surgeon_diff";
pub const TOOL_LORA: &str = "surgeon_lora";
pub(super) const NAMES: [&str; 2] = [TOOL_DIFF, TOOL_LORA];

const DEFAULT_RANK: u32 = 16;

/// Mirror of `OnHighError` (a clap enum) with serde and schemars.
#[derive(Debug, Clone, Copy, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum OnHighErrorArg {
    Warn,
    Skip,
    Full,
}

impl From<OnHighErrorArg> for OnHighError {
    fn from(value: OnHighErrorArg) -> Self {
        match value {
            OnHighErrorArg::Warn => Self::Warn,
            OnHighErrorArg::Skip => Self::Skip,
            OnHighErrorArg::Full => Self::Full,
        }
    }
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_diff : approximation de rang r de la différence entre deux checkpoints, écrite dans un fichier .surgery."
)]
pub struct DiffToolArgs {
    #[schemars(
        description = "Checkpoint de base : .safetensors ou dossier shardé, dans un dossier autorisé."
    )]
    pub base: PathBuf,
    #[schemars(
        description = "Checkpoint affiné : mêmes noms, formes et dtypes de tenseurs que base."
    )]
    pub tuned: PathBuf,
    #[schemars(description = "Fichier .surgery produit, dans un dossier autorisé.")]
    pub out: PathBuf,
    #[serde(default)]
    #[schemars(range(min = 1), description = "Rang de l'approximation (défaut 16).")]
    pub rank: Option<u32>,
    #[serde(default)]
    #[schemars(description = "Dtype de stockage des facteurs : f16 (défaut), bf16 ou f32.")]
    pub factor_dtype: Option<FactorDtype>,
    #[serde(default)]
    #[schemars(description = "Regex : seuls les tenseurs dont le nom correspond sont traités.")]
    pub include: Option<String>,
    #[serde(default)]
    #[schemars(description = "Regex : les tenseurs dont le nom correspond sont ignorés.")]
    pub exclude: Option<String>,
    #[serde(default)]
    #[schemars(
        description = "Inclut aussi embed_tokens et lm_head (exclus par défaut : énormes)."
    )]
    pub include_embeddings: bool,
    #[serde(default)]
    #[schemars(
        description = "Erreur relative de reconstruction au-delà de laquelle on_high_error s'applique."
    )]
    pub max_rel_error: Option<f64>,
    #[serde(default)]
    #[schemars(
        description = "warn (défaut) : avertit ; skip : omet le tenseur ; full : stocke le delta exact."
    )]
    pub on_high_error: Option<OnHighErrorArg>,
    #[serde(default)]
    #[schemars(description = "Graine du SVD randomisé (défaut 0) ; même graine, mêmes octets.")]
    pub seed: Option<u64>,
    #[serde(default)]
    #[schemars(description = "Remplace out s'il existe (sinon erreur).")]
    pub overwrite: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum LoraMode {
    Merge,
    Extract,
}

impl LoraMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Merge => "merge",
            Self::Extract => "extract",
        }
    }
}

/// Flat on purpose (no `oneOf` at the top): which fields a mode takes is checked at
/// runtime, with one issue per offending field.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_lora : mode, puis les champs de ce mode (merge : model, lora, alpha, adapter_config, scale, dry_run, backup_dir ; extract : base, patched, out, rank, dtype, target_modules, seed, overwrite)."
)]
pub struct LoraToolArgs {
    #[schemars(
        description = "merge : fusionne un adaptateur PEFT dans les poids, en place ; extract : extrait un adaptateur de la différence de deux checkpoints."
    )]
    pub mode: LoraMode,
    #[serde(default)]
    #[schemars(description = "merge : modèle modifié en place (.safetensors ou dossier shardé).")]
    pub model: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "merge : adapter_model.safetensors (clés PEFT).")]
    pub lora: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "merge : lora_alpha ; remplace adapter_config.json. L'un des deux est requis."
    )]
    pub alpha: Option<f64>,
    #[serde(default)]
    #[schemars(description = "merge : adapter_config.json (défaut : celui à côté de lora).")]
    pub adapter_config: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "merge : multiplicateur supplémentaire du delta (défaut 1).")]
    pub scale: Option<f64>,
    #[serde(default)]
    #[schemars(description = "merge : valide sans écrire (défaut false).")]
    pub dry_run: Option<bool>,
    #[serde(default)]
    #[schemars(description = "merge : dossier des sauvegardes ; défaut : à côté du modèle.")]
    pub backup_dir: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "extract : checkpoint de base.")]
    pub base: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "extract : checkpoint modifié, mêmes noms, formes et dtypes.")]
    pub patched: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "extract : adapter_model.safetensors écrit (adapter_config.json à côté)."
    )]
    pub out: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "extract : rang de l'adaptateur (défaut 16)."
    )]
    pub rank: Option<u32>,
    #[serde(default)]
    #[schemars(description = "extract : dtype des matrices A et B : f32 (défaut), f16 ou bf16.")]
    pub dtype: Option<FactorDtype>,
    #[serde(default)]
    #[schemars(
        description = "extract : regex des tenseurs 2D retenus (défaut : projections d'attention et MLP)."
    )]
    pub target_modules: Option<String>,
    #[serde(default)]
    #[schemars(description = "extract : graine du SVD randomisé (défaut 0).")]
    pub seed: Option<u64>,
    #[serde(default)]
    #[schemars(description = "extract : remplace out s'il existe.")]
    pub overwrite: Option<bool>,
}

/// Draft-07 with every subschema inlined, like `schema_value` in tools.rs. `tool_definitions`
/// sorts the keys of the whole array afterwards.
fn schema_value<T: JsonSchema>() -> Value {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    let schema = settings.into_generator().into_root_schema_for::<T>();
    serde_json::to_value(schema).unwrap_or(Value::Null)
}

fn tool(name: &str, title: &str, description: &str, input: Value, annotations: Value) -> Value {
    let mut annotations = annotations;
    annotations["title"] = json!(title);
    annotations["openWorldHint"] = json!(false);
    json!({ "name": name, "description": description, "inputSchema": input, "annotations": annotations })
}

pub(super) fn definitions() -> Vec<Value> {
    vec![
        tool(
            TOOL_DIFF,
            "Différence de checkpoints",
            "Calcule la différence entre un checkpoint de base et sa version affinée, approximée par tenseur à rang r (SVD randomisé), et l'écrit dans un fichier .surgery neuf (jamais en place). C'est une APPROXIMATION du fine-tuning complet, erreur relative reportée par tenseur ; taille environ somme r(m+n) fois les octets du dtype (des centaines de Mo pour un 7B à rang 16). Un .surgery ne s'applique pas via surgeon_apply : utiliser la CLI.",
            schema_value::<DiffToolArgs>(),
            json!({ "readOnlyHint": false, "destructiveHint": false, "idempotentHint": true }),
        ),
        tool(
            TOOL_LORA,
            "Fusion et extraction LoRA",
            "mode merge : fusionne un adaptateur LoRA PEFT dans les poids EN PLACE (W += scale·alpha/r·B·A), sauvegarde vérifiée imposée avant écriture, rollback automatique en cas d'erreur ou d'annulation ; faire d'abord dry_run: true. mode extract : écrit un adaptateur LoRA de rang r (format PEFT, adapter_config.json à côté) depuis la différence de deux checkpoints, sans toucher aux modèles. Format PEFT non testé contre HuggingFace en CI ; fusion dans un GGUF non supportée.",
            schema_value::<LoraToolArgs>(),
            json!({ "readOnlyHint": false, "destructiveHint": true, "idempotentHint": false }),
        ),
    ]
}

fn invalid(kind: &str, message: impl std::fmt::Display) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, message).with_kind(kind)
}

fn parse<T: serde::de::DeserializeOwned>(arguments: Value) -> Result<T, Outcome> {
    serde_json::from_value(arguments)
        .map_err(|e| invalid("invalid_arguments", format!("arguments invalides : {e}")))
}

fn sandboxed(ctx: &ToolContext, argument: &str, path: &Path) -> Result<PathBuf, Outcome> {
    ctx.cfg
        .sandbox
        .resolve(argument, path, ctx.cfg.sandbox.base())
        .map_err(|e| invalid("path_not_allowed", e))
}

fn required_path(
    ctx: &ToolContext,
    argument: &str,
    path: &Option<PathBuf>,
) -> Result<PathBuf, Outcome> {
    match path {
        Some(path) => sandboxed(ctx, argument, path),
        None => Err(invalid("invalid_arguments", format!("{argument} requis"))),
    }
}

fn optional_path(
    ctx: &ToolContext,
    argument: &str,
    path: &Option<PathBuf>,
) -> Result<Option<PathBuf>, Outcome> {
    path.as_ref()
        .map(|p| sandboxed(ctx, argument, p))
        .transpose()
}

/// Envelope `command` of the call, known before parsing so that argument errors carry it.
fn command_name(name: &str, arguments: &Value) -> &'static str {
    match (name, arguments.get("mode").and_then(Value::as_str)) {
        (TOOL_DIFF, _) => "diff",
        (_, Some("merge")) => "lora merge",
        (_, Some("extract")) => "lora extract",
        _ => "lora",
    }
}

pub(super) fn call(
    name: &str,
    arguments: Value,
    ctx: &mut ToolContext,
) -> (&'static str, CoreResult) {
    let command = command_name(name, &arguments);
    let result = match name {
        TOOL_DIFF => diff(arguments, ctx),
        _ => lora(arguments, ctx),
    };
    (command, result)
}

fn refuse_read_only(ctx: &ToolContext, tool: &str) -> Result<(), Outcome> {
    match ctx.cfg.read_only {
        true => Err(invalid(
            "read_only",
            format!("serveur lancé avec --read-only : {tool} écrit des fichiers, il est désactivé"),
        )),
        false => Ok(()),
    }
}

fn diff(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    refuse_read_only(ctx, TOOL_DIFF)?;
    let args: DiffToolArgs = parse(arguments)?;
    let core_args = DiffArgs {
        base: sandboxed(ctx, "base", &args.base)?,
        tuned: sandboxed(ctx, "tuned", &args.tuned)?,
        out: sandboxed(ctx, "out", &args.out)?,
        rank: args.rank.unwrap_or(DEFAULT_RANK) as usize,
        factor_dtype: args.factor_dtype.unwrap_or(FactorDtype::F16),
        include: args.include,
        exclude: args.exclude,
        include_embeddings: args.include_embeddings,
        max_rel_error: args.max_rel_error,
        on_high_error: args.on_high_error.map_or(OnHighError::Warn, Into::into),
        oversample: 8,
        power_iters: 3,
        seed: args.seed.unwrap_or(0),
        // The header stays reproducible: no wall-clock time over MCP.
        timestamp: false,
        overwrite: args.overwrite,
    };
    core_diff(&core_args, ctx.deps, ctx.sink)
}

/// One issue per missing required field and per field the mode does not take.
fn lora_issues(a: &LoraToolArgs) -> Vec<ValidationIssue> {
    // (field, present, required in merge, required in extract); None: the mode ignores it.
    let fields: [(&str, bool, Option<bool>, Option<bool>); 15] = [
        ("model", a.model.is_some(), Some(true), None),
        ("lora", a.lora.is_some(), Some(true), None),
        ("alpha", a.alpha.is_some(), Some(false), None),
        (
            "adapter_config",
            a.adapter_config.is_some(),
            Some(false),
            None,
        ),
        ("scale", a.scale.is_some(), Some(false), None),
        ("dry_run", a.dry_run.is_some(), Some(false), None),
        ("backup_dir", a.backup_dir.is_some(), Some(false), None),
        ("base", a.base.is_some(), None, Some(true)),
        ("patched", a.patched.is_some(), None, Some(true)),
        ("out", a.out.is_some(), None, Some(true)),
        ("rank", a.rank.is_some(), None, Some(false)),
        ("dtype", a.dtype.is_some(), None, Some(false)),
        (
            "target_modules",
            a.target_modules.is_some(),
            None,
            Some(false),
        ),
        ("seed", a.seed.is_some(), None, Some(false)),
        ("overwrite", a.overwrite.is_some(), None, Some(false)),
    ];
    let mode = a.mode.as_str();
    fields
        .iter()
        .filter_map(|&(path, present, merge, extract)| {
            let role = match a.mode {
                LoraMode::Merge => merge,
                LoraMode::Extract => extract,
            };
            let message = match (role, present) {
                (Some(true), false) => format!("requis en mode {mode}"),
                (None, true) => format!("non utilisé en mode {mode}"),
                _ => return None,
            };
            Some(ValidationIssue {
                path: path.to_owned(),
                message,
            })
        })
        .collect()
}

fn lora(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    refuse_read_only(ctx, TOOL_LORA)?;
    let args: LoraToolArgs = parse(arguments)?;
    let issues = lora_issues(&args);
    if !issues.is_empty() {
        let mut failure = invalid(
            "invalid_arguments",
            format!("champs incohérents avec le mode {}", args.mode.as_str()),
        );
        if let Some(body) = failure.error.as_mut() {
            body.issues = issues;
        }
        return Err(failure);
    }
    let command = match args.mode {
        LoraMode::Merge => LoraCommand::Merge(LoraMergeArgs {
            model: required_path(ctx, "model", &args.model)?,
            lora: required_path(ctx, "lora", &args.lora)?,
            alpha: args.alpha,
            adapter_config: optional_path(ctx, "adapter_config", &args.adapter_config)?,
            scale: args.scale.unwrap_or(1.0),
            dry_run: args.dry_run.unwrap_or(false),
            // An agent never decides to skip the backup: there is no argument for it.
            no_backup: false,
            backup_dir: optional_path(ctx, "backup_dir", &args.backup_dir)?,
        }),
        LoraMode::Extract => LoraCommand::Extract(LoraExtractArgs {
            base: required_path(ctx, "base", &args.base)?,
            patched: required_path(ctx, "patched", &args.patched)?,
            out: required_path(ctx, "out", &args.out)?,
            rank: args.rank.unwrap_or(DEFAULT_RANK) as usize,
            dtype: args.dtype.unwrap_or(FactorDtype::F32),
            target_modules: args.target_modules,
            seed: args.seed.unwrap_or(0),
            overwrite: args.overwrite.unwrap_or(false),
        }),
    };
    core_lora(&LoraArgs { command }, ctx.deps, ctx.sink)
}
