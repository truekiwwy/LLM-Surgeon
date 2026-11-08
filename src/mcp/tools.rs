//! The four MCP tools: argument types (their schemas are the `inputSchema`s), definitions,
//! and execution through the same command cores as the CLI.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use super::server::McpConfig;
use crate::app::{
    core_apply, core_check, core_eval, core_inspect, envelope_of, exit, AppDeps, ApplyRequest,
    CoreResult, Outcome, ProgressSink,
};
use crate::cli::{CheckArgs, EvalArgs, InspectArgs, SchemaKind};
use crate::eval::{load_prompts, EvalPrompt};
use crate::patch::{LayerSelection, LayersField, PatchFile, Projection};
use crate::ui::Ui;

pub const TOOL_CHECK: &str = "surgeon_check";
pub const TOOL_INSPECT: &str = "surgeon_inspect";
pub const TOOL_APPLY: &str = "surgeon_apply";
pub const TOOL_EVAL: &str = "surgeon_eval";
const TOOLS: [&str; 4] = [TOOL_CHECK, TOOL_INSPECT, TOOL_APPLY, TOOL_EVAL];

const DEFAULT_MAX_TENSORS: u32 = 500;
const DEFAULT_MAX_NEW_TOKENS: u32 = 256;
const DEFAULT_QUANT: &str = "Q4_K_M";

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(description = "Arguments de surgeon_check.")]
pub struct CheckToolArgs {
    #[schemars(
        description = "Modèle : fichier .safetensors, dossier shardé ou model.safetensors.index.json, dans un dossier autorisé."
    )]
    pub model: PathBuf,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(description = "Arguments de surgeon_inspect.")]
pub struct InspectToolArgs {
    #[schemars(
        description = "Modèle : fichier .safetensors, dossier shardé ou index, dans un dossier autorisé."
    )]
    pub model: PathBuf,
    /// Same format as a patch operation's `layers`, not the internal `LayerSelection`.
    #[serde(default)]
    pub layers: Option<LayersField>,
    #[serde(default)]
    #[schemars(description = "Projections à lister ; absent ou vide : les trois.")]
    pub projections: Option<Vec<Projection>>,
    #[serde(default)]
    #[schemars(
        description = "Calcule min, max, moyenne et norme L2 de chaque matrice (lecture complète, plus lent)."
    )]
    pub show_stats: bool,
    #[serde(default = "default_max_tensors")]
    #[schemars(
        range(min = 1),
        description = "Nombre maximal de tenseurs renvoyés (défaut 500) ; au-delà, data.truncated = true et data.total = N."
    )]
    pub max_tensors: u32,
}

fn default_max_tensors() -> u32 {
    DEFAULT_MAX_TENSORS
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(description = "Arguments de surgeon_apply.")]
pub struct ApplyToolArgs {
    #[schemars(description = "Modèle à modifier en place, dans un dossier autorisé.")]
    pub model: PathBuf,
    pub patch: PatchFile,
    #[serde(default)]
    #[schemars(
        description = "Valide et résout les cibles sans écrire ni sauvegarder (défaut false)."
    )]
    pub dry_run: bool,
    #[serde(default)]
    #[schemars(description = "Dossier des sauvegardes ; défaut : à côté du modèle.")]
    pub backup_dir: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "Dossier de résolution des vector_file relatifs ; défaut : le premier dossier autorisé."
    )]
    pub base_dir: Option<PathBuf>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_eval : fournir exactement un de prompts_file ou prompts."
)]
pub struct EvalToolArgs {
    #[schemars(
        description = "Modèle évalué : .gguf directement, ou SafeTensors converti en GGUF temporaire."
    )]
    pub model: PathBuf,
    #[serde(default)]
    #[schemars(
        description = "Fichier de prompts : tableau JSON, JSONL ou texte (un prompt par ligne)."
    )]
    pub prompts_file: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "Prompts fournis directement (au lieu de prompts_file).")]
    pub prompts: Option<Vec<EvalPrompt>>,
    #[serde(default = "default_max_new_tokens")]
    #[schemars(
        range(min = 1),
        description = "Tokens générés au plus par prompt (défaut 256)."
    )]
    pub max_new_tokens: u32,
    #[serde(default)]
    #[schemars(description = "Chemin du GGUF quantifié à produire si le modèle est sain.")]
    pub export_gguf: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "Quantification de l'export (Q4_K_M par défaut, Q8_0, F16...).")]
    pub quant: Option<String>,
    #[serde(default)]
    #[schemars(
        description = "Racine de llama.cpp (convertisseur, llama-quantize) ; défaut : $LLAMA_CPP_DIR du serveur."
    )]
    pub llama_cpp_dir: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "Exporte aussi un modèle dégradé (jamais un modèle cassé).")]
    pub allow_degraded: bool,
}

fn default_max_new_tokens() -> u32 {
    DEFAULT_MAX_NEW_TOKENS
}

/// Draft-07 with every subschema inlined: many MCP clients mishandle `$ref`.
fn schema_value<T: JsonSchema>() -> Value {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    let schema = settings.into_generator().into_root_schema_for::<T>();
    sorted(serde_json::to_value(schema).unwrap_or(Value::Null))
}

/// Keys sorted recursively, whatever serde_json's map implementation: the output is
/// deterministic byte for byte.
fn sorted(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut entries: Vec<(String, Value)> = map.into_iter().collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(
                entries
                    .into_iter()
                    .map(|(k, v)| (k, sorted(v)))
                    .collect::<Map<_, _>>(),
            )
        }
        Value::Array(items) => Value::Array(items.into_iter().map(sorted).collect()),
        other => other,
    }
}

pub fn patch_schema() -> Value {
    schema_value::<PatchFile>()
}

fn tool(name: &str, title: &str, description: &str, input: Value, annotations: Value) -> Value {
    let mut annotations = annotations;
    annotations["title"] = json!(title);
    annotations["openWorldHint"] = json!(false);
    json!({ "name": name, "description": description, "inputSchema": input, "annotations": annotations })
}

/// Exactly the `tools` array of `tools/list`, and the output of `surgeon schema --kind tools`.
pub fn tool_definitions() -> Value {
    sorted(json!([
        tool(
            TOOL_CHECK,
            "Budget matériel",
            "Vérifie si la machine peut charger le modèle : taille, RAM et VRAM nécessaires et disponibles, verdict fits_in_vram / fits_in_ram_only / insufficient. Lecture seule.",
            schema_value::<CheckToolArgs>(),
            json!({ "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true }),
        ),
        tool(
            TOOL_INSPECT,
            "Topologie MLP",
            "Liste les matrices MLP (gate_proj, up_proj, down_proj) par couche : dtype, forme, taille, et en option statistiques min/max/moyenne/L2. Sert à choisir les couches et l'axe d'un patch. Lecture seule.",
            schema_value::<InspectToolArgs>(),
            json!({ "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true }),
        ),
        tool(
            TOOL_APPLY,
            "Chirurgie de poids",
            "Applique un patch (W ∓= alpha·v) EN PLACE sur les poids du modèle. Sauvegarde vérifiée imposée avant écriture, rollback automatique en cas d'erreur ou d'annulation. Faire d'abord un dry_run: true pour vérifier les cibles.",
            schema_value::<ApplyToolArgs>(),
            json!({ "readOnlyHint": false, "destructiveHint": true, "idempotentHint": false }),
        ),
        tool(
            TOOL_EVAL,
            "Évaluation de non-régression",
            "Fait générer le modèle (llama.cpp) sur des prompts, détecte les boucles dégénérées et les attentes non tenues, verdict healthy / degraded / broken ; exporte un GGUF quantifié si demandé et si le modèle est sain.",
            schema_value::<EvalToolArgs>(),
            json!({ "readOnlyHint": false, "destructiveHint": false, "idempotentHint": false }),
        ),
    ]))
}

/// Pretty JSON, sorted keys, exactly one trailing newline.
pub fn render_schema(kind: SchemaKind) -> String {
    let value = match kind {
        SchemaKind::Patch => patch_schema(),
        SchemaKind::Tools => tool_definitions(),
    };
    let mut text = serde_json::to_string_pretty(&value).unwrap_or_default();
    text.push('\n');
    text
}

pub(crate) fn is_known(name: &str) -> bool {
    TOOLS.contains(&name)
}

pub(crate) struct ToolContext<'a> {
    pub deps: &'a AppDeps,
    pub cfg: &'a McpConfig,
    pub sink: &'a mut dyn ProgressSink,
}

/// `tools/call` result for a known tool. Every failure here is a tool execution error
/// (`isError: true`), never a JSON-RPC error: the agent reads it and can correct itself.
pub(crate) fn call(name: &str, arguments: Value, ctx: &mut ToolContext) -> Value {
    let (command, result) = match name {
        TOOL_CHECK => ("check", check(arguments, ctx)),
        TOOL_INSPECT => ("inspect", inspect(arguments, ctx)),
        TOOL_APPLY => ("apply", apply(arguments, ctx)),
        _ => ("eval", eval(arguments, ctx)),
    };
    let mut outcome = result.unwrap_or_else(|failure| failure);
    if !outcome.is_ok() && ctx.deps.interrupted.load(Ordering::SeqCst) {
        outcome = outcome.with_kind("cancelled");
    }
    let envelope = envelope_of(Some(command), &outcome);
    json!({
        "content": [{ "type": "text", "text": serde_json::to_string(&envelope).unwrap_or_default() }],
        "structuredContent": envelope,
        "isError": !outcome.is_ok(),
    })
}

fn invalid(kind: &str, message: impl std::fmt::Display) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, message).with_kind(kind)
}

fn parse<T: DeserializeOwned>(arguments: Value) -> Result<T, Outcome> {
    serde_json::from_value(arguments)
        .map_err(|e| invalid("invalid_arguments", format!("arguments invalides : {e}")))
}

fn sandboxed(
    ctx: &ToolContext,
    argument: &str,
    path: &Path,
    base: &Path,
) -> Result<PathBuf, Outcome> {
    ctx.cfg
        .sandbox
        .resolve(argument, path, base)
        .map_err(|e| invalid("path_not_allowed", e))
}

fn check(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: CheckToolArgs = parse(arguments)?;
    let model = sandboxed(ctx, "model", &args.model, ctx.cfg.sandbox.base())?;
    let args = CheckArgs {
        model,
        policy_ram_margin_gib: None,
    };
    core_check(&args, ctx.deps, &Ui::off())
}

fn inspect(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: InspectToolArgs = parse(arguments)?;
    let model = sandboxed(ctx, "model", &args.model, ctx.cfg.sandbox.base())?;
    let layers = match &args.layers {
        Some(layers) => layers
            .to_selection()
            .map_err(|e| invalid("invalid_arguments", format!("layers : {e}")))?,
        None => LayerSelection::All,
    };
    let core_args = InspectArgs {
        model,
        layers,
        projections: args.projections.unwrap_or_default(),
        show_stats: args.show_stats,
    };
    let mut outcome = core_inspect(&core_args, ctx.deps, &Ui::off())?;
    let max = args.max_tensors as usize;
    if let Some(tensors) = outcome
        .data
        .get_mut("tensors")
        .and_then(Value::as_array_mut)
    {
        let total = tensors.len();
        if total > max {
            tensors.truncate(max);
            outcome.data["truncated"] = json!(true);
            outcome.data["total"] = json!(total);
        }
    }
    Ok(outcome)
}

fn apply(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    if ctx.cfg.read_only {
        return Err(invalid(
            "read_only",
            "serveur lancé avec --read-only : surgeon_apply est désactivé",
        ));
    }
    let args: ApplyToolArgs = parse(arguments)?;
    let root = ctx.cfg.sandbox.base().to_path_buf();
    let model = sandboxed(ctx, "model", &args.model, &root)?;
    let base_dir = match &args.base_dir {
        Some(dir) => sandboxed(ctx, "base_dir", dir, &root)?,
        None => root.clone(),
    };
    let backup_dir = match &args.backup_dir {
        Some(dir) => Some(sandboxed(ctx, "backup_dir", dir, &root)?),
        None => None,
    };
    let mut patch = args.patch;
    for (i, op) in patch.operations.iter_mut().enumerate() {
        let argument = format!("patch.operations[{i}].vector_file");
        op.vector_file = sandboxed(ctx, &argument, &op.vector_file, &base_dir)?;
    }
    // An agent never decides to skip the backup: only the operator can, at launch.
    patch.backup |= !ctx.cfg.allow_no_backup;
    let request = ApplyRequest {
        model,
        patch,
        patch_label: PathBuf::from("patch"),
        base_dir,
        dry_run: args.dry_run,
        no_backup: false,
        backup_dir,
    };
    core_apply(&request, ctx.deps, ctx.sink)
}

fn eval(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: EvalToolArgs = parse(arguments)?;
    if ctx.cfg.read_only && args.export_gguf.is_some() {
        return Err(invalid(
            "read_only",
            "serveur lancé avec --read-only : export_gguf est désactivé",
        ));
    }
    let root = ctx.cfg.sandbox.base().to_path_buf();
    let model = sandboxed(ctx, "model", &args.model, &root)?;
    let prompts = match (&args.prompts_file, args.prompts) {
        (Some(file), None) => {
            let file = sandboxed(ctx, "prompts_file", file, &root)?;
            load_prompts(&file).map_err(|e| invalid("invalid_input", e))?
        }
        (None, Some(prompts)) => normalize_prompts(prompts)?,
        _ => {
            return Err(invalid(
                "invalid_arguments",
                "fournir exactement un de prompts_file ou prompts",
            ))
        }
    };
    let export_gguf = match &args.export_gguf {
        Some(path) => Some(sandboxed(ctx, "export_gguf", path, &root)?),
        None => None,
    };
    // A tool-provided llama.cpp directory is sandboxed (it holds scripts that get run);
    // $LLAMA_CPP_DIR comes from the operator who launched the server and is trusted.
    let llama_cpp_dir = match &args.llama_cpp_dir {
        Some(dir) => Some(sandboxed(ctx, "llama_cpp_dir", dir, &root)?),
        None => std::env::var_os("LLAMA_CPP_DIR").map(PathBuf::from),
    };
    let core_args = EvalArgs {
        model,
        prompts: PathBuf::new(),
        max_new_tokens: args.max_new_tokens,
        export_gguf,
        quant: args.quant.unwrap_or_else(|| DEFAULT_QUANT.into()),
        llama_cpp_dir,
        allow_degraded: args.allow_degraded,
    };
    core_eval(&core_args, prompts, ctx.deps, &Ui::off(), ctx.sink)
}

/// Same rules as `load_prompts` for inline prompts: at least one, missing ids become
/// `prompt-<n>`, non-empty texts, unique ids.
fn normalize_prompts(mut prompts: Vec<EvalPrompt>) -> Result<Vec<EvalPrompt>, Outcome> {
    if prompts.is_empty() {
        return Err(invalid("invalid_arguments", "prompts : au moins un prompt"));
    }
    let mut seen = HashSet::new();
    for (i, p) in prompts.iter_mut().enumerate() {
        if p.id.trim().is_empty() {
            p.id = format!("prompt-{}", i + 1);
        }
        if p.text.trim().is_empty() {
            return Err(invalid(
                "invalid_arguments",
                format!("prompt « {} » : texte vide", p.id),
            ));
        }
        if !seen.insert(p.id.clone()) {
            return Err(invalid(
                "invalid_arguments",
                format!("prompt « {} » : id dupliqué", p.id),
            ));
        }
    }
    Ok(prompts)
}
