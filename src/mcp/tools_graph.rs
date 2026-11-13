//! MCP tool `surgeon_graph`: argument type (its schema is the `inputSchema`), definition,
//! and execution through `core_graph_with`, the same core as the CLI.
//!
//! Without `out_dir` the call is read-only and returns the bounded summary of
//! [`crate::graph::output::summary`]; with `out_dir` it writes the selected files. A server
//! started with `--read-only` therefore accepts the former and refuses the latter (and a
//! cache directory, which also writes).

use std::path::{Path, PathBuf};
use std::str::FromStr;

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};
use crate::graph::{
    ColorScaleArg, EdgeMode, EdgeWeight, GraphArgs, MatrixArg, OutputFormat, PaletteArg,
};

pub const TOOL_GRAPH: &str = "surgeon_graph";
pub(super) const NAMES: [&str; 1] = [TOOL_GRAPH];

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_graph : graphe orienté des écritures/lectures du flux résiduel, avec bornes spectrales certifiées. Sans out_dir, renvoie une synthèse bornée et n'écrit rien."
)]
pub struct GraphToolArgs {
    #[schemars(
        description = "Modèle : .safetensors, dossier shardé ou .gguf, dans un dossier autorisé."
    )]
    pub model: PathBuf,
    #[serde(default)]
    #[schemars(
        description = "Dossier des sorties (graph.json, graph.dot, graph.mmd, matrices.csv, edges.csv, graph.ndjson). Absent : synthèse bornée seulement, aucun fichier écrit."
    )]
    pub out_dir: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "Formats écrits avec out_dir : json, dot, mmd, csv, ndjson (graph.ndjson : instantané GraphPatch, ouvert et suivi par surgeon-gui)."
    )]
    pub format: Option<Vec<OutputFormat>>,
    #[serde(default)]
    #[schemars(description = "Couches retenues : 0, 1-3, 0,2,4 ou all.")]
    pub layers: Option<String>,
    #[serde(default)]
    #[schemars(description = "Matrices retenues : q,k,v,o,gate,up,down,embed,lm_head.")]
    pub matrices: Option<Vec<MatrixArg>>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "Rang cible du SVD randomisé (défaut 32)."
    )]
    pub rank: Option<u32>,
    #[serde(default)]
    #[schemars(
        range(min = 0),
        description = "Colonnes supplémentaires du sous-espace aléatoire (défaut 10)."
    )]
    pub oversample: Option<u32>,
    #[serde(default)]
    #[schemars(
        range(min = 0, max = 64),
        description = "Itérations de puissance du SVD (défaut 3)."
    )]
    pub power_iters: Option<u32>,
    #[serde(default)]
    #[schemars(description = "Graine du SVD ; même graine, mêmes nombres (défaut 0).")]
    pub seed: Option<u64>,
    #[serde(default)]
    #[schemars(
        range(min = 0),
        description = "Dimension maximale pour le spectre complet exact (défaut 2048 ; 0 désactive)."
    )]
    pub full_spectrum_max_dim: Option<usize>,
    #[serde(default)]
    #[schemars(description = "Arêtes : adjacent (défaut), all, ou topk:N.")]
    pub edges: Option<String>,
    #[serde(default)]
    #[schemars(description = "Alignement brut minimal (0 garde tout).")]
    pub min_alignment: Option<f64>,
    #[serde(default)]
    #[schemars(description = "Poids d'arête : alignment (défaut) ou coupling.")]
    pub edge_weight: Option<EdgeWeight>,
    #[serde(default)]
    #[schemars(description = "Vecteur résiduel JSON ([f32]) ou .bin f32 pour E_write/E_read.")]
    pub vector: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "Kurtosis par canal au-delà duquel un canal est outlier (défaut 10)."
    )]
    pub kurtosis_threshold: Option<f64>,
    #[serde(default)]
    #[schemars(description = "Échelle de couleur des arêtes : quantile, log, linear.")]
    pub color_scale: Option<ColorScaleArg>,
    #[serde(default)]
    #[schemars(description = "Palette : diverging (défaut) ou cb-safe.")]
    pub palette: Option<PaletteArg>,
    #[serde(default)]
    #[schemars(
        range(min = 1),
        description = "Nombre maximal de matrices traitées (les plus grosses d'abord)."
    )]
    pub max_nodes: Option<usize>,
    #[serde(default)]
    #[schemars(description = "Dossier du cache JSONL (aucun cache si absent).")]
    pub cache_dir: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "N'écrit ni ne lit le cache.")]
    pub no_cache: bool,
    #[serde(default)]
    #[schemars(description = "Réutilise un cache partiel (exige cache_dir).")]
    pub resume: bool,
    #[serde(default)]
    #[schemars(description = "Estime les passes, octets et la durée sans exécuter les SVD.")]
    pub estimate: bool,
    #[serde(default)]
    #[schemars(
        range(min = 0, max = 1024),
        description = "Fils rayon (0 : défaut du système)."
    )]
    pub jobs: Option<u32>,
    #[serde(default)]
    #[schemars(description = "Remplace les sorties existantes.")]
    pub overwrite: bool,
    #[serde(default)]
    #[schemars(
        description = "Plafond wall-clock en secondes ; la tâche est annulée (kind cancelled) au-delà."
    )]
    pub max_seconds: Option<f64>,
    #[serde(default)]
    #[schemars(
        description = "Référence comparée au modèle, dans un dossier autorisé : modèle (analysé avec les mêmes paramètres) ou graph.json déjà exporté. Ajoute baseline_diff (deltas de rang effectif, de norme et d'alignement)."
    )]
    pub baseline: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "Empreinte rapide (taille, date, 3 fenêtres de 1 Mio par fichier) au lieu du SHA-256 complet ; model.hash_mode vaut alors fast."
    )]
    pub fast_hash: bool,
}

pub(super) fn definitions() -> Vec<Value> {
    vec![tool(
        TOOL_GRAPH,
        "Graphe de poids",
        "Transforme les matrices du flux résiduel en graphe orienté : écritures (embed, o_proj, down_proj) vers lectures postérieures (q/k/v, gate/up, lm_head), avec alignement de sous-espaces, couplage énergétique, angle principal, erank énergétique borné, rang stable certifié, kurtosis et canaux outliers. Sans out_dir : synthèse bornée seulement, aucun fichier. Avec out_dir : graph.json (schéma surgeon schema --kind graph), graph.dot, graph.mmd, matrices.csv, edges.csv, graph.ndjson (flux GraphPatch pour surgeon-gui). baseline compare à une référence (modèle ou graph.json). Le graphe complet n'est jamais renvoyé en ligne.",
        schema_value::<GraphToolArgs>(),
        json!({ "readOnlyHint": false, "destructiveHint": false, "idempotentHint": true }),
    )]
}

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

pub(super) fn call(arguments: Value, ctx: &mut ToolContext) -> (&'static str, CoreResult) {
    ("graph", graph(arguments, ctx))
}

fn graph(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: GraphToolArgs = parse(arguments)?;
    if ctx.cfg.read_only && (args.out_dir.is_some() || args.cache_dir.is_some()) {
        return Err(invalid(
            "read_only",
            "serveur lancé avec --read-only : surgeon_graph n'écrit ni out_dir ni cache_dir",
        ));
    }
    let edges = match &args.edges {
        Some(text) => EdgeMode::from_str(text).map_err(|e| invalid("invalid_arguments", e))?,
        None => EdgeMode::Adjacent,
    };
    let format = args.format.unwrap_or_else(|| vec![OutputFormat::Json]);
    let core = GraphArgs {
        model: sandboxed(ctx, "model", &args.model)?,
        out_dir: optional_path(ctx, "out_dir", &args.out_dir)?,
        format,
        layers: args.layers,
        matrices: args.matrices.unwrap_or_default(),
        rank: args.rank.unwrap_or(32) as usize,
        oversample: args.oversample.unwrap_or(10) as usize,
        power_iters: args.power_iters.unwrap_or(3) as usize,
        seed: args.seed.unwrap_or(0),
        full_spectrum_max_dim: args.full_spectrum_max_dim.unwrap_or(2048),
        edges,
        min_alignment: args.min_alignment.unwrap_or(0.0),
        edge_weight: args.edge_weight.unwrap_or(EdgeWeight::Alignment),
        vector: optional_path(ctx, "vector", &args.vector)?,
        kurtosis_threshold: args.kurtosis_threshold.unwrap_or(10.0),
        color_scale: args.color_scale.unwrap_or(ColorScaleArg::Quantile),
        palette: args.palette.unwrap_or(PaletteArg::Diverging),
        max_nodes: args.max_nodes,
        cache_dir: optional_path(ctx, "cache_dir", &args.cache_dir)?,
        no_cache: args.no_cache,
        resume: args.resume,
        estimate: args.estimate,
        jobs: args.jobs.unwrap_or(0) as usize,
        overwrite: args.overwrite,
        baseline: optional_path(ctx, "baseline", &args.baseline)?,
        fast_hash: args.fast_hash,
    };
    crate::graph::core_graph_with(&core, ctx.deps, ctx.sink, args.max_seconds)
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

fn optional_path(
    ctx: &ToolContext,
    argument: &str,
    path: &Option<PathBuf>,
) -> Result<Option<PathBuf>, Outcome> {
    path.as_ref()
        .map(|p| sandboxed(ctx, argument, p))
        .transpose()
}

fn invalid(kind: &str, message: impl std::fmt::Display) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, message).with_kind(kind)
}
