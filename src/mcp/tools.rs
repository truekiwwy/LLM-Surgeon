//! The four MCP tools: argument types (their schemas are the `inputSchema`s), definitions,
//! and execution through the same command cores as the CLI.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use schemars::gen::{SchemaGenerator, SchemaSettings};
use schemars::schema::{InstanceType, Metadata, Schema, SchemaObject, SubschemaValidation};
use schemars::JsonSchema;
use serde::de::{self, DeserializeOwned, DeserializeSeed, EnumAccess, VariantAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{json, Map, Value};

use super::server::McpConfig;
use crate::app::{
    core_apply_routed, core_autotune, core_bench_dump, core_bench_eval, core_bench_logits,
    core_bench_ppl, core_check, core_eval, core_extract_method, core_inspect, envelope_of, exit,
    AppDeps, ApplyRequest, CoreResult, Outcome, ProgressSink,
};
use crate::bench;
use crate::bench::dump::DumpRequest;
use crate::bench::logits_metrics::LogitsRequest;
use crate::bench::ppl_runner::{PplBaseline, PplRequest};
use crate::bench::prompt::Template;
use crate::bench::score::EvalRequest;
use crate::cli::{CheckArgs, EvalArgs, InspectArgs, SchemaKind};
use crate::eval::{load_prompts, EvalPrompt};
use crate::extract::{
    self, ChatTemplate, DTypeChoice, ExtractRequest, MethodOptions, Pooling, Site,
    DEFAULT_LEACE_RCOND, DEFAULT_LEACE_SHRINKAGE,
};
use crate::patch::autotune::{parse_layer_range, AutotuneRequest, L2Reg};
use crate::patch::{LayerSelection, LayersField, PatchFile, Projection, ValidationIssue};
use crate::plugin::{unknown_method_message, Registry, DEFAULT_METHOD};
use crate::ui::Ui;

pub const TOOL_CHECK: &str = "surgeon_check";
pub const TOOL_INSPECT: &str = "surgeon_inspect";
pub const TOOL_APPLY: &str = "surgeon_apply";
pub const TOOL_EVAL: &str = "surgeon_eval";
pub const TOOL_EXTRACT: &str = "surgeon_extract";
pub const TOOL_BENCH: &str = "surgeon_bench";
pub const TOOL_AUTOTUNE: &str = "surgeon_autotune";
const TOOLS: [&str; 7] = [
    TOOL_CHECK,
    TOOL_INSPECT,
    TOOL_APPLY,
    TOOL_EVAL,
    TOOL_EXTRACT,
    TOOL_BENCH,
    TOOL_AUTOTUNE,
];

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
    #[serde(default)]
    #[schemars(
        description = "Modèle .gguf patché en place (re-quantification avec perte, backup toujours créé). Requis pour un .gguf."
    )]
    pub gguf: bool,
    #[serde(default)]
    #[schemars(description = "GGUF : keep-scales (défaut) ou recompute.")]
    pub requant: Option<crate::patch::gguf::Requant>,
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

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_extract : pour chaque ensemble, fournir exactement un de *_prompts ou *_prompts_file."
)]
pub struct ExtractToolArgs {
    #[schemars(
        description = "Dossier du modèle Qwen2/Qwen2.5 (config.json, tokenizer.json, *.safetensors), dans un dossier autorisé."
    )]
    pub model: PathBuf,
    #[serde(default)]
    #[schemars(description = "Prompts positifs : ceux qui présentent le comportement à isoler.")]
    pub pos_prompts: Option<Vec<String>>,
    #[serde(default)]
    #[schemars(description = "Fichier de prompts positifs (tableau JSON, JSONL ou texte).")]
    pub pos_prompts_file: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "Prompts négatifs ou neutres.")]
    pub neg_prompts: Option<Vec<String>>,
    #[serde(default)]
    #[schemars(description = "Fichier de prompts négatifs (tableau JSON, JSONL ou texte).")]
    pub neg_prompts_file: Option<PathBuf>,
    #[serde(default = "default_extract_layer")]
    #[schemars(
        description = "Couche (base 0), défaut 20 ; doit être < num_hidden_layers du modèle."
    )]
    pub layer: u32,
    #[schemars(
        description = "Fichier .safetensors produit (un tenseur 1D F32 `direction`), dans un dossier autorisé."
    )]
    pub out: PathBuf,
    #[serde(default)]
    pub pooling: Pooling,
    #[serde(default)]
    pub site: Site,
    #[serde(default)]
    pub chat_template: ChatTemplate,
    #[serde(default = "default_extract_max_tokens")]
    #[schemars(
        range(min = 1),
        description = "Tokens gardés par prompt (troncature à droite), défaut 512."
    )]
    pub max_tokens: u32,
    #[serde(default)]
    pub dtype: DTypeChoice,
    #[serde(default)]
    #[schemars(description = "Remplace un fichier out existant (sinon erreur).")]
    pub overwrite: bool,
    #[serde(default)]
    #[schemars(
        description = "Méthode d'extraction, identifiant du registre du moteur : mean_diff (défaut), svd (prompts appariés par index), leace (écrit aussi <out>.leace.safetensors pour project_out)."
    )]
    pub method: Option<MethodId>,
    #[serde(default)]
    #[schemars(
        description = "svd : centre les différences appariées (paires d'ordre arbitraire)."
    )]
    pub svd_center: bool,
    #[serde(default)]
    #[schemars(
        range(min = 0.0, max = 1.0),
        description = "leace : rétrécissement λ de Σ vers tr(Σ)/d·I, défaut 0.1 (0 : pseudo-inverse)."
    )]
    pub leace_shrinkage: Option<f64>,
    #[serde(default)]
    #[schemars(
        description = "leace avec λ = 0 : seuil relatif des valeurs propres conservées (défaut 1e-10)."
    )]
    pub leace_rcond: Option<f64>,
    #[serde(default)]
    #[schemars(
        range(min = 0.0, max = 1.0),
        description = "Fraction hors-échantillon (par index) pour les diagnostics ; 10 prompts par classe au moins."
    )]
    pub holdout_frac: Option<f64>,
}

/// `method` of `surgeon_extract`: an identifier of the built-in registry. Its schema lists
/// the registry (a `oneOf` of string constants) and its deserialization follows the former
/// `Method` enum exactly (string or single-key map, same error messages). `Serialize` lets
/// schemars write `"default": null`, as for the enum.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(transparent)]
pub struct MethodId(pub String);

impl JsonSchema for MethodId {
    fn schema_name() -> String {
        "Method".into()
    }

    fn json_schema(_: &mut SchemaGenerator) -> Schema {
        let variants = Registry::with_builtin()
            .list()
            .map(|(m, _)| {
                Schema::Object(SchemaObject {
                    metadata: Some(Box::new(Metadata {
                        description: Some(m.describe().summary.into()),
                        ..Metadata::default()
                    })),
                    instance_type: Some(InstanceType::String.into()),
                    enum_values: Some(vec![m.id().into()]),
                    ..SchemaObject::default()
                })
            })
            .collect();
        Schema::Object(SchemaObject {
            subschemas: Some(Box::new(SubschemaValidation {
                one_of: Some(variants),
                ..SubschemaValidation::default()
            })),
            ..SchemaObject::default()
        })
    }
}

impl<'de> Deserialize<'de> for MethodId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Id;
        impl Visitor<'_> for Id {
            type Value = String;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("variant identifier")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<String, E> {
                let ids = Registry::with_builtin().ids();
                match ids.contains(&v) {
                    true => Ok(v.to_owned()),
                    false => Err(E::custom(unknown_method_message(v, &ids))),
                }
            }
        }
        impl<'de> DeserializeSeed<'de> for Id {
            type Value = String;
            fn deserialize<D: Deserializer<'de>>(self, d: D) -> Result<String, D::Error> {
                d.deserialize_identifier(self)
            }
        }
        struct Enum;
        impl<'de> Visitor<'de> for Enum {
            type Value = MethodId;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("enum Method")
            }
            fn visit_enum<A: EnumAccess<'de>>(self, data: A) -> Result<MethodId, A::Error> {
                let (id, variant) = data.variant_seed(Id)?;
                variant.unit_variant()?;
                Ok(MethodId(id))
            }
        }
        // The variant list is not used by serde_json: `Id` checks the registry.
        deserializer.deserialize_enum("Method", &[], Enum)
    }
}

fn default_extract_layer() -> u32 {
    20
}

fn default_extract_max_tokens() -> u32 {
    512
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
    let mut tools = base_tool_definitions();
    if let Some(list) = tools.as_array_mut() {
        list.extend(super::tools_p13::definitions());
    }
    sorted(tools)
}

fn base_tool_definitions() -> Value {
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
        tool(
            TOOL_EXTRACT,
            "Extraction de direction",
            "Calcule un vecteur de direction (steering) à partir des activations d'une couche d'un modèle Qwen2, en Rust pur sur CPU, avec une méthode du registre du moteur (method : mean_diff par défaut, v = normalize(mean(pos) - mean(neg)) ; svd ; leace), et l'écrit dans un .safetensors (tenseur `direction`) à utiliser comme vector_file d'un patch surgeon_apply. Renvoie des diagnostics de séparation (d′, accuracy).",
            schema_value::<ExtractToolArgs>(),
            json!({ "readOnlyHint": false, "destructiveHint": false, "idempotentHint": true }),
        ),
        tool(
            TOOL_BENCH,
            "Mesure d'une chirurgie",
            "Mesure l'effet d'une chirurgie, selon mode : dump (logits d'un modèle Qwen2 sur des prompts, en Rust pur, écrits dans un .safetensors), logits (KL(before‖after), cosinus, changements top-1 / top-k entre deux dumps), ppl (perplexité via llama-perplexity et écart à une référence arrondi à 4 décimales), eval (Pass exact match glouton sur un dataset GSM8K / MATH, intervalle de Wilson, comparaison appariée). Recette : dump avant et après un apply sur une copie, logits, ppl, eval, puis décision.",
            schema_value::<BenchToolArgs>(),
            json!({ "readOnlyHint": false, "destructiveHint": false, "idempotentHint": true }),
        ),
        tool(
            TOOL_AUTOTUNE,
            "Réglage des coefficients par couche",
            "Mesure en mémoire la réponse de la projection d'une couche de lecture sur le vecteur quand chaque couche candidate est décalée (différences finies, arrondi au dtype comme sur disque), ajuste les α par régression ridge (λ fixe ou LOOCV), valide en appliquant tous les α ensemble, et produit un patch.json compatible surgeon_apply (écrit seulement si out est fourni). Ne mesure pas les effets secondaires : utiliser surgeon_bench ensuite.",
            schema_value::<AutotuneToolArgs>(),
            json!({ "readOnlyHint": false, "destructiveHint": false, "idempotentHint": true }),
        ),
    ]))
}

/// Pretty JSON, sorted keys, exactly one trailing newline.
pub fn render_schema(kind: SchemaKind) -> String {
    let value = match kind {
        SchemaKind::Patch => patch_schema(),
        SchemaKind::Tools => tool_definitions(),
        SchemaKind::Surgery => crate::surgery::schema::header_schema(),
    };
    let mut text = serde_json::to_string_pretty(&value).unwrap_or_default();
    text.push('\n');
    text
}

pub(crate) fn is_known(name: &str) -> bool {
    TOOLS.contains(&name) || super::tools_p13::NAMES.contains(&name)
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
        TOOL_EVAL => ("eval", eval(arguments, ctx)),
        TOOL_BENCH => (bench_command(&arguments), bench(arguments, ctx)),
        TOOL_AUTOTUNE => ("autotune", autotune(arguments, ctx)),
        super::tools_p13::TOOL_DIFF | super::tools_p13::TOOL_LORA => {
            super::tools_p13::call(name, arguments, ctx)
        }
        _ => ("extract", extract_direction(arguments, ctx)),
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
    // GGUF: same guards, and never --force-no-backup from an agent.
    let gguf = crate::patch::gguf::GgufApplyArgs {
        gguf: args.gguf,
        requant: args.requant.unwrap_or_default(),
        ..Default::default()
    };
    core_apply_routed(&request, &gguf, ctx.deps, &Ui::off(), ctx.sink)
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

fn extract_direction(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    if ctx.cfg.read_only {
        return Err(invalid(
            "read_only",
            "serveur lancé avec --read-only : surgeon_extract écrit un fichier, il est désactivé",
        ));
    }
    let args: ExtractToolArgs = parse(arguments)?;
    let root = ctx.cfg.sandbox.base().to_path_buf();
    let model_dir = sandboxed(ctx, "model", &args.model, &root)?;
    let pos_prompts = prompt_set(
        ctx,
        "pos",
        args.pos_prompts,
        args.pos_prompts_file.as_deref(),
        &root,
    )?;
    let neg_prompts = prompt_set(
        ctx,
        "neg",
        args.neg_prompts,
        args.neg_prompts_file.as_deref(),
        &root,
    )?;
    let out = sandboxed(ctx, "out", &args.out, &root)?;
    let request = ExtractRequest {
        model_dir,
        pos_prompts,
        neg_prompts,
        layer: args.layer as usize,
        out,
        pooling: args.pooling,
        site: args.site,
        chat_template: args.chat_template,
        max_tokens: args.max_tokens as usize,
        dtype: args.dtype,
        overwrite: args.overwrite,
    };
    let params = MethodOptions {
        svd_center: args.svd_center,
        leace_shrinkage: args.leace_shrinkage.unwrap_or(DEFAULT_LEACE_SHRINKAGE),
        leace_rcond: args.leace_rcond.unwrap_or(DEFAULT_LEACE_RCOND),
        holdout_frac: args.holdout_frac.unwrap_or(0.0),
        ..MethodOptions::default()
    }
    .params();
    let method = args.method.map_or(DEFAULT_METHOD.to_owned(), |m| m.0);
    core_extract_method(&request, &method, &params, ctx.deps, &Ui::off(), ctx.sink)
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_autotune : coefficients α par couche ajustés par ridge sur des réponses mesurées en mémoire."
)]
pub struct AutotuneToolArgs {
    #[schemars(description = "Dossier du modèle Qwen2, dans un dossier autorisé.")]
    pub model: PathBuf,
    #[schemars(
        description = "Fichier direction de surgeon_extract (tenseur direction et métadonnées)."
    )]
    pub vector: PathBuf,
    #[schemars(description = "Tableau JSON (ou JSONL) de chaînes ou d'objets {text, target?}.")]
    pub eval_dataset: PathBuf,
    #[serde(default)]
    #[schemars(
        description = "Couches candidates \"A-B\" ; défaut : 0 jusqu'à la couche de lecture."
    )]
    pub target_layer_range: Option<String>,
    #[serde(default)]
    #[schemars(description = "Couche lue ; défaut : la couche du vecteur.")]
    pub readout_layer: Option<u32>,
    #[serde(default)]
    #[schemars(description = "Projection décalée ; défaut down_proj.")]
    pub proj: Option<Projection>,
    #[serde(default)]
    pub objective: Option<crate::patch::autotune::Objective>,
    #[serde(default)]
    #[schemars(description = "Projection cible (obligatoire avec objective target).")]
    pub target_projection: Option<f64>,
    #[serde(default)]
    #[schemars(description = "λ > 0, ou \"auto\" (LOOCV sur 1e-6..1e2) ; défaut 1e-3.")]
    pub l2_reg: Option<String>,
    #[serde(default)]
    #[schemars(description = "Pas δ des différences finies, défaut 0.05.")]
    pub probe_alpha: Option<f64>,
    #[serde(default)]
    #[schemars(description = "Écrêtage |α| ≤ alpha_max, défaut 1.")]
    pub alpha_max: Option<f64>,
    #[serde(default)]
    #[schemars(description = "Couches à |α| < min_alpha retirées, défaut 1e-4.")]
    pub min_alpha: Option<f64>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "Plafond de forwards vérifié avant calcul, défaut 2000."
    )]
    pub max_forwards: Option<u32>,
    #[serde(default)]
    #[schemars(range(min = 1), description = "Troncature par prompt, défaut 256.")]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub dtype: DTypeChoice,
    #[serde(default)]
    pub site: Option<Site>,
    #[serde(default)]
    pub pooling: Option<Pooling>,
    #[serde(default)]
    pub chat_template: Option<ChatTemplate>,
    #[serde(default)]
    #[schemars(
        description = "patch.json à écrire (refusé en --read-only) ; sans out, le patch est seulement dans data.patch."
    )]
    pub out: Option<PathBuf>,
    #[serde(default)]
    pub overwrite: bool,
}

fn autotune(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: AutotuneToolArgs = parse(arguments)?;
    if ctx.cfg.read_only && args.out.is_some() {
        return Err(invalid(
            "read_only",
            "serveur lancé avec --read-only : surgeon_autotune peut calculer mais pas écrire out",
        ));
    }
    let root = ctx.cfg.sandbox.base().to_path_buf();
    let bad = |m: String| invalid("invalid_arguments", m);
    let request = AutotuneRequest {
        model: sandboxed(ctx, "model", &args.model, &root)?,
        vector: sandboxed(ctx, "vector", &args.vector, &root)?,
        eval_dataset: sandboxed(ctx, "eval_dataset", &args.eval_dataset, &root)?,
        layer_range: args
            .target_layer_range
            .as_deref()
            .map(parse_layer_range)
            .transpose()
            .map_err(&bad)?,
        readout_layer: args.readout_layer.map(|r| r as usize),
        proj: args.proj.unwrap_or(Projection::Down),
        objective: args.objective.unwrap_or_default(),
        target_projection: args.target_projection,
        l2_reg: args
            .l2_reg
            .as_deref()
            .unwrap_or("1e-3")
            .parse::<L2Reg>()
            .map_err(&bad)?,
        probe_alpha: args.probe_alpha.unwrap_or(0.05),
        alpha_max: args.alpha_max.unwrap_or(1.0),
        min_alpha: args.min_alpha.unwrap_or(1e-4),
        max_forwards: args.max_forwards.map_or(2000, |m| m as usize),
        max_tokens: args.max_tokens.map_or(256, |m| m as usize),
        dtype: args.dtype,
        site: args.site,
        pooling: args.pooling,
        chat_template: args.chat_template,
        out: args
            .out
            .as_ref()
            .map(|o| sandboxed(ctx, "out", o, &root))
            .transpose()?,
        overwrite: args.overwrite,
    };
    core_autotune(&request, ctx.deps, &Ui::off(), ctx.sink)
}

/// Exactly one of the inline list or the (sandboxed) file.
fn prompt_set(
    ctx: &ToolContext,
    set: &str,
    inline: Option<Vec<String>>,
    file: Option<&Path>,
    root: &Path,
) -> Result<Vec<String>, Outcome> {
    let texts = match (inline, file) {
        (Some(list), None) => extract::check_texts(list),
        (None, Some(path)) => {
            let path = sandboxed(ctx, &format!("{set}_prompts_file"), path, root)?;
            extract::load_prompt_texts(&path)
        }
        _ => {
            return Err(invalid(
                "invalid_arguments",
                format!("fournir exactement un de {set}_prompts ou {set}_prompts_file"),
            ))
        }
    };
    texts.map_err(|m| invalid("invalid_input", format!("prompts {set} : {m}")))
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum BenchMode {
    /// Compare deux dumps de logits (before, after).
    Logits,
    /// Perplexité via llama-perplexity (model, text).
    Ppl,
    /// Pass@1 exact match glouton sur un dataset (model, dataset).
    Eval,
    /// Écrit les logits d'un modèle Qwen2 sur des prompts (model, prompts, out).
    Dump,
}

/// Flat on purpose (no `oneOf` at the top): which fields a mode takes is checked at
/// runtime, with one issue per offending field.
#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_bench : mode, puis les champs de ce mode (logits : before, after ; ppl : model, text ; eval : model, dataset ; dump : model, prompts, out)."
)]
pub struct BenchToolArgs {
    pub mode: BenchMode,
    #[serde(default)]
    #[schemars(description = "logits : dump de référence (.safetensors, .npy, .pt/.pth).")]
    pub before: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "logits : dump comparé, même forme.")]
    pub after: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "logits : tenseur lu (défaut logits).")]
    pub tensor: Option<String>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "logits : taille des ensembles top-k (défaut 5)."
    )]
    pub top_k: Option<u32>,
    #[serde(default)]
    #[schemars(description = "logits : prompts listés dans worst_prompts (défaut 5).")]
    pub worst: Option<u32>,
    #[serde(default)]
    #[schemars(description = "logits : avertit si la KL moyenne dépasse strictement ce seuil.")]
    pub warn_kl: Option<f64>,
    #[serde(default)]
    #[schemars(
        description = "logits : avertit si la part de top-1 changés (0..1) dépasse strictement ce seuil."
    )]
    pub warn_top1_shift: Option<f64>,
    #[serde(default)]
    #[schemars(description = "logits, ppl, eval : seuil dépassé => exit_code 9 (isError).")]
    pub fail_on_warn: Option<bool>,
    #[serde(default)]
    #[schemars(
        description = "ppl, eval : .gguf ou dossier SafeTensors (converti en GGUF temporaire) ; dump : dossier Qwen2."
    )]
    pub model: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "ppl : fichier texte évalué.")]
    pub text: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "ppl, eval : modèle de référence, mesuré en premier.")]
    pub baseline_model: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "ppl : perplexité de référence connue (exclusif avec baseline_model)."
    )]
    pub baseline_ppl: Option<f64>,
    #[serde(default)]
    #[schemars(description = "ppl : incertitude ± de baseline_ppl.")]
    pub baseline_ppl_err: Option<f64>,
    #[serde(default)]
    #[schemars(range(min = 1), description = "ppl : contexte -c (défaut 512).")]
    pub ctx: Option<u32>,
    #[serde(default)]
    #[schemars(range(min = 1), description = "ppl : --chunks.")]
    pub chunks: Option<u32>,
    #[serde(default)]
    #[schemars(range(min = 1), description = "ppl : -t.")]
    pub threads: Option<u32>,
    #[serde(default)]
    #[schemars(description = "ppl : couches sur GPU (défaut 0, CPU reproductible).")]
    pub n_gpu_layers: Option<u32>,
    #[serde(default)]
    #[schemars(
        description = "ppl, eval : racine de llama.cpp ; défaut : $LLAMA_CPP_DIR du serveur."
    )]
    pub llama_cpp_dir: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "ppl : avertit si le delta arrondi dépasse strictement ce seuil (défaut 0.2)."
    )]
    pub warn_delta: Option<f64>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "ppl : délai par run ; eval : délai par question (secondes)."
    )]
    pub timeout_secs: Option<u64>,
    #[serde(default)]
    #[schemars(
        description = "eval : tableau JSON (ou JSONL) de {question|problem, answer|solution, id?}."
    )]
    pub dataset: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "eval : N premières entrées dans l'ordre du fichier (défaut 50)."
    )]
    pub limit: Option<u32>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "eval : tokens générés au plus par question (défaut 512)."
    )]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    #[schemars(description = "eval : gabarit contenant {question} (texte, jamais un chemin).")]
    pub prompt_template: Option<String>,
    #[serde(default)]
    pub format: Option<crate::bench::dataset::DatasetFormat>,
    #[serde(default)]
    #[schemars(description = "eval : avertit si l'exactitude (0..1) est strictement inférieure.")]
    pub min_accuracy: Option<f64>,
    #[serde(default)]
    #[schemars(
        description = "dump : fichier de prompts (tableau JSON de chaînes ou de {text}, JSONL ou texte)."
    )]
    pub prompts: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "dump : .safetensors écrit (logits, prompt_offsets).")]
    pub out: Option<PathBuf>,
    #[serde(default)]
    pub dtype: Option<DTypeChoice>,
    #[serde(default)]
    pub out_dtype: Option<crate::bench::dump::OutDtype>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "dump : troncature par prompt (défaut 256)."
    )]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "dump : plafond de positions, vérifié avant calcul (défaut 4096)."
    )]
    pub max_total_positions: Option<u32>,
    #[serde(default)]
    #[schemars(description = "dump : remplace out s'il existe.")]
    pub overwrite: Option<bool>,
}

/// (mode, required fields, optional fields).
const BENCH_FIELDS: [(&str, &[&str], &[&str]); 4] = [
    (
        "logits",
        &["before", "after"],
        &[
            "tensor",
            "top_k",
            "worst",
            "warn_kl",
            "warn_top1_shift",
            "fail_on_warn",
        ],
    ),
    (
        "ppl",
        &["model", "text"],
        &[
            "baseline_model",
            "baseline_ppl",
            "baseline_ppl_err",
            "ctx",
            "chunks",
            "threads",
            "n_gpu_layers",
            "llama_cpp_dir",
            "warn_delta",
            "timeout_secs",
            "fail_on_warn",
        ],
    ),
    (
        "eval",
        &["model", "dataset"],
        &[
            "baseline_model",
            "limit",
            "max_new_tokens",
            "prompt_template",
            "format",
            "llama_cpp_dir",
            "timeout_secs",
            "min_accuracy",
            "fail_on_warn",
        ],
    ),
    (
        "dump",
        &["model", "prompts", "out"],
        &[
            "dtype",
            "out_dtype",
            "max_tokens",
            "max_total_positions",
            "overwrite",
        ],
    ),
];

fn bench_command(arguments: &Value) -> &'static str {
    match arguments.get("mode").and_then(Value::as_str) {
        Some("logits") => "bench logits",
        Some("ppl") => "bench ppl",
        Some("eval") => "bench eval",
        Some("dump") => "bench dump",
        _ => "bench",
    }
}

/// `-32602` for surgeon_bench: a property its schema does not declare. (The five older
/// tools report it as an `invalid_arguments` tool error, kept for compatibility.)
pub(crate) fn unknown_property(name: &str, arguments: &Value) -> Option<String> {
    if name != TOOL_BENCH {
        return None;
    }
    let schema = schema_value::<BenchToolArgs>();
    let known = schema.get("properties")?.as_object()?;
    let unknown: Vec<&String> = arguments
        .as_object()?
        .keys()
        .filter(|k| !known.contains_key(*k))
        .collect();
    (!unknown.is_empty()).then(|| format!("{TOOL_BENCH} : propriété(s) inconnue(s) {unknown:?}"))
}

/// One issue per missing field and per field the mode does not take.
fn bench_issues(mode: &str, arguments: &Map<String, Value>) -> Vec<ValidationIssue> {
    let Some((_, required, optional)) = BENCH_FIELDS.iter().find(|(m, _, _)| *m == mode) else {
        return Vec::new();
    };
    let present = |k: &str| arguments.get(k).is_some_and(|v| !v.is_null());
    let issue = |path: &str, message: String| ValidationIssue {
        path: path.to_owned(),
        message,
    };
    let mut issues: Vec<ValidationIssue> = required
        .iter()
        .filter(|k| !present(k))
        .map(|k| issue(k, format!("requis en mode {mode}")))
        .collect();
    issues.extend(
        arguments
            .keys()
            .filter(|k| *k != "mode" && present(k))
            .filter(|k| !required.contains(&k.as_str()) && !optional.contains(&k.as_str()))
            .map(|k| issue(k, format!("non utilisé en mode {mode}"))),
    );
    if present("baseline_model") && present("baseline_ppl") {
        issues.push(issue("baseline_ppl", "exclusif avec baseline_model".into()));
    }
    if present("baseline_ppl_err") && !present("baseline_ppl") {
        issues.push(issue("baseline_ppl_err", "requiert baseline_ppl".into()));
    }
    issues
}

fn bench(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: BenchToolArgs = parse(arguments.clone())?;
    if ctx.cfg.read_only && args.mode == BenchMode::Dump {
        return Err(invalid(
            "read_only",
            "serveur lancé avec --read-only : surgeon_bench mode dump écrit un fichier, il est désactivé",
        ));
    }
    let mode = bench_command(&arguments).trim_start_matches("bench ");
    let issues = arguments
        .as_object()
        .map(|o| bench_issues(mode, o))
        .unwrap_or_default();
    if !issues.is_empty() {
        let mut failure = invalid(
            "invalid_arguments",
            format!("champs incohérents avec le mode {mode}"),
        );
        if let Some(body) = failure.error.as_mut() {
            body.issues = issues;
        }
        return Err(failure);
    }
    let root = ctx.cfg.sandbox.base().to_path_buf();
    let path = |ctx: &ToolContext, name: &str, value: &Option<PathBuf>| {
        let value = value
            .as_ref()
            .ok_or_else(|| invalid("invalid_arguments", format!("{name} requis")))?;
        sandboxed(ctx, name, value, &root)
    };
    let optional_path = |ctx: &ToolContext, name: &str, value: &Option<PathBuf>| {
        value
            .as_ref()
            .map(|v| sandboxed(ctx, name, v, &root))
            .transpose()
    };
    // Same trust rule as surgeon_eval: a tool-provided directory is sandboxed, the
    // operator's $LLAMA_CPP_DIR is not.
    let llama_cpp_dir = match optional_path(ctx, "llama_cpp_dir", &args.llama_cpp_dir)? {
        Some(dir) => Some(dir),
        None => std::env::var_os("LLAMA_CPP_DIR").map(PathBuf::from),
    };
    let timeout = args.timeout_secs.map(std::time::Duration::from_secs);
    let fail_on_warn = args.fail_on_warn.unwrap_or(false);
    let ui = Ui::off();
    match args.mode {
        BenchMode::Logits => {
            let req = LogitsRequest {
                before: path(ctx, "before", &args.before)?,
                after: path(ctx, "after", &args.after)?,
                tensor: args.tensor,
                top_k: args.top_k.map_or(bench::DEFAULT_TOP_K, |k| k as usize),
                worst: args.worst.map_or(bench::DEFAULT_WORST, |w| w as usize),
                warn_kl: args.warn_kl,
                warn_top1_shift: args.warn_top1_shift,
            };
            core_bench_logits(&req, fail_on_warn, ctx.deps, &ui, ctx.sink)
        }
        BenchMode::Ppl => {
            let baseline_model = optional_path(ctx, "baseline_model", &args.baseline_model)?;
            let baseline = match (baseline_model, args.baseline_ppl) {
                (Some(model), _) => Some(PplBaseline::Model(model)),
                (None, Some(ppl)) => Some(PplBaseline::Value {
                    ppl,
                    err: args.baseline_ppl_err,
                }),
                (None, None) => None,
            };
            let req = PplRequest {
                model: path(ctx, "model", &args.model)?,
                text: path(ctx, "text", &args.text)?,
                baseline,
                ctx: args.ctx.unwrap_or(bench::DEFAULT_CTX),
                chunks: args.chunks,
                threads: args.threads,
                n_gpu_layers: args.n_gpu_layers.unwrap_or(0),
                llama_cpp_dir,
                warn_delta: args.warn_delta.unwrap_or(bench::DEFAULT_WARN_DELTA),
                timeout,
                // Never from an agent: no free-form argument reaches llama.cpp over MCP.
                llama_args: Vec::new(),
            };
            core_bench_ppl(&req, fail_on_warn, ctx.deps, &ui, ctx.sink)
        }
        BenchMode::Eval => {
            let template = match &args.prompt_template {
                Some(text) => Template::new(text),
                None => Ok(Template::default()),
            }
            .map_err(|e| invalid("invalid_arguments", e))?;
            let req = EvalRequest {
                model: path(ctx, "model", &args.model)?,
                dataset: path(ctx, "dataset", &args.dataset)?,
                baseline_model: optional_path(ctx, "baseline_model", &args.baseline_model)?,
                limit: args.limit.map_or(bench::DEFAULT_LIMIT, |l| l as usize),
                max_new_tokens: args.max_new_tokens.unwrap_or(bench::DEFAULT_MAX_NEW_TOKENS),
                template,
                format: args.format.unwrap_or_default(),
                llama_cpp_dir,
                timeout,
                min_accuracy: args.min_accuracy,
            };
            core_bench_eval(&req, fail_on_warn, ctx.deps, &ui, ctx.sink)
        }
        BenchMode::Dump => {
            let prompts_file = path(ctx, "prompts", &args.prompts)?;
            let prompts = extract::load_prompt_texts(&prompts_file)
                .map_err(|m| invalid("invalid_input", format!("prompts : {m}")))?;
            let req = DumpRequest {
                model: path(ctx, "model", &args.model)?,
                prompts,
                out: path(ctx, "out", &args.out)?,
                dtype: args.dtype.unwrap_or_default(),
                out_dtype: args.out_dtype.unwrap_or_default(),
                max_tokens: args
                    .max_tokens
                    .map_or(bench::DEFAULT_DUMP_MAX_TOKENS, |m| m as usize),
                max_total_positions: args
                    .max_total_positions
                    .map_or(bench::DEFAULT_MAX_TOTAL_POSITIONS, |m| m as usize),
                overwrite: args.overwrite.unwrap_or(false),
            };
            core_bench_dump(&req, ctx.deps, &ui, ctx.sink)
        }
    }
}
