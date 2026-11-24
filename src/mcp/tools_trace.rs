use std::path::PathBuf;

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::{trace_prompts, ToolContext};
use crate::app::{core_causal_trace, exit, CoreResult, Outcome};
use crate::extract::{self, ChatTemplate, Component, DTypeChoice, TraceRequest};
use crate::patch::{LayerSelection, LayersField};
use crate::ui::Ui;

pub const TOOL_CAUSAL_TRACE: &str = "surgeon_causal_trace";
pub(super) const NAMES: [&str; 1] = [TOOL_CAUSAL_TRACE];

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_causal_trace : des paires (prompt propre, prompt corrompu) appariées par index, pour chaque ensemble exactement un de *_prompts ou *_prompts_file."
)]
pub struct CausalTraceToolArgs {
    #[schemars(
        description = "Dossier du modèle Qwen2/Qwen2.5, Llama/Mistral ou Qwen3 (config.json, tokenizer.json, *.safetensors), dans un dossier autorisé (lecture seule possible)."
    )]
    pub model: PathBuf,
    #[serde(default)]
    #[schemars(
        description = "Prompts propres : ceux qui produisent le comportement à localiser (une requête refusée, le style visé)."
    )]
    pub clean_prompts: Option<Vec<String>>,
    #[serde(default)]
    #[schemars(
        description = "Fichier de prompts propres (tableau JSON, JSONL ou texte), ou un jeu intégré builtin:<set>."
    )]
    pub clean_prompts_file: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "Prompts corrompus ou neutres, autant que de propres, appariés par index."
    )]
    pub corrupted_prompts: Option<Vec<String>>,
    #[serde(default)]
    #[schemars(
        description = "Fichier de prompts corrompus (tableau JSON, JSONL ou texte), ou un jeu intégré builtin:<set>."
    )]
    pub corrupted_prompts_file: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "Composant restauré au dernier token : resid (flux résiduel après la couche, défaut), attn (sortie de l'attention, après o_proj) ou mlp (sortie du MLP, après down_proj)."
    )]
    pub component: Option<Component>,
    #[serde(default)]
    #[schemars(
        description = "Couches tracées : \"all\" (défaut), \"A-B\", \"3,5,7\", un tableau ou {start, end}."
    )]
    pub layers: Option<LayersField>,
    #[serde(default)]
    #[schemars(
        description = "Cible mesurée : texte dont le premier token (encodé sans token spécial, une espace initiale compte) est suivi en log-probabilité. Absent : le token le plus probable du prompt propre."
    )]
    pub target: Option<String>,
    #[serde(default)]
    #[schemars(
        range(min = 0.0, max = 1.0),
        description = "Fenêtre causale : couches contiguës autour du pic dont le score reste au moins cette part du pic, dans ]0, 1] (défaut 0.5)."
    )]
    pub window_threshold: Option<f64>,
    #[serde(default)]
    #[schemars(
        description = "Habillage des prompts : auto (défaut, le format déclaré par le modèle), none, chatml, llama3, inst ou gemma."
    )]
    pub chat_template: ChatTemplate,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "Tokens gardés par prompt (troncature à droite), défaut 512."
    )]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub dtype: DTypeChoice,
}

pub(super) fn definitions() -> Vec<Value> {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    let schema = settings
        .into_generator()
        .into_root_schema_for::<CausalTraceToolArgs>();
    vec![json!({
        "name": TOOL_CAUSAL_TRACE,
        "description": "Localise les couches responsables d'un comportement par restauration d'activation (causal tracing), en Rust pur sur CPU. Pour chaque paire (prompt propre, prompt corrompu), la sortie propre du composant choisi (resid, attn ou mlp) au dernier token est réinjectée couche par couche dans la passe corrompue ; score de la couche = (log p_restauré − log p_corrompu) / (log p_propre − log p_corrompu) sur le token cible, moyenné sur les paires (1 : la couche suffit à rétablir la sortie propre). Renvoie le profil par couche (data.layers), le classement (data.ranking) et la fenêtre causale contiguë autour du pic (data.window.layers, au format layers de surgeon_compose et de surgeon_graph). Coût : 2 + nombre de couches passes avant par paire. N'écrit aucun fichier.",
        "inputSchema": serde_json::to_value(schema).unwrap_or(Value::Null),
        "annotations": {
            "title": "Traçage causal par couche",
            "readOnlyHint": true,
            "destructiveHint": false,
            "idempotentHint": true,
            "openWorldHint": false,
        },
    })]
}

fn invalid(kind: &str, message: impl std::fmt::Display) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, message).with_kind(kind)
}

pub(super) fn call(arguments: Value, ctx: &mut ToolContext) -> (&'static str, CoreResult) {
    ("causal_trace", causal_trace(arguments, ctx))
}

fn causal_trace(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: CausalTraceToolArgs = serde_json::from_value(arguments)
        .map_err(|e| invalid("invalid_arguments", format!("arguments invalides : {e}")))?;
    let sandbox = &ctx.cfg.sandbox;
    let model = sandbox
        .resolve_read("model", &args.model, sandbox.base())
        .map_err(|e| invalid("path_not_allowed", e))?;
    let clean_prompts = trace_prompts(
        ctx,
        "clean",
        args.clean_prompts,
        args.clean_prompts_file.as_deref(),
    )?;
    let corrupted_prompts = trace_prompts(
        ctx,
        "corrupted",
        args.corrupted_prompts,
        args.corrupted_prompts_file.as_deref(),
    )?;
    let layers = match &args.layers {
        Some(layers) => layers
            .to_selection()
            .map_err(|e| invalid("invalid_arguments", format!("layers : {e}")))?,
        None => LayerSelection::All,
    };
    let request = TraceRequest {
        model_dir: extract::model_folder(&model).to_path_buf(),
        clean_prompts,
        corrupted_prompts,
        component: args.component.unwrap_or_default(),
        layers,
        target: args.target,
        window_threshold: args
            .window_threshold
            .unwrap_or(extract::DEFAULT_WINDOW_THRESHOLD),
        chat_template: args.chat_template,
        max_tokens: args
            .max_tokens
            .map_or(extract::DEFAULT_MAX_TOKENS, |m| m as usize),
        dtype: args.dtype,
    };
    core_causal_trace(&request, ctx.deps, &Ui::off(), ctx.sink)
}
