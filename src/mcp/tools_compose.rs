//! MCP tool `surgeon_compose`: `surgeon compose` through `patch::compose::core_compose`, the
//! same core as the CLI. The vectors are only read (an `--allow-read` root is fine); the
//! composed direction and the patch are written in an `--allow-dir` folder.

use std::path::{Path, PathBuf};

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};
use crate::cli::ComposeArgs;
use crate::patch::compose::ComposeMode;

pub const TOOL_COMPOSE: &str = "surgeon_compose";
pub(super) const NAMES: [&str; 1] = [TOOL_COMPOSE];

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_compose : les vecteurs à retirer (ablate), les directions à préserver (keep), le fichier de sortie (out), et facultativement le patch qui retire la direction composée (patch_out)."
)]
pub struct ComposeToolArgs {
    #[schemars(
        length(min = 1),
        description = "Vecteurs à retirer : fichiers de surgeon_extract (ou tableaux JSON), chacun « chemin » ou « chemin:alpha » (poids dans la somme, défaut 1)."
    )]
    pub ablate: Vec<String>,
    #[serde(default)]
    #[schemars(
        description = "Directions à préserver (vecteurs de capacité) : la direction composée leur est rendue orthogonale en mode orthogonal_preserve."
    )]
    pub keep: Option<Vec<PathBuf>>,
    #[serde(default)]
    #[schemars(
        description = "orthogonal_preserve (défaut) : vecteurs retirés projetés hors du sous-espace des directions préservées (Gram-Schmidt modifié, f64), refus explicite d'un vecteur quasi colinéaire ; standard : somme pondérée directe."
    )]
    pub mode: Option<ComposeMode>,
    #[schemars(
        description = "Direction composée (.safetensors) : tenseurs direction, a et b (= direction unitaire), dans un dossier autorisé en écriture."
    )]
    pub out: PathBuf,
    #[serde(default)]
    #[schemars(
        description = "patch.json à écrire : project_out de la direction composée, avec la recette (sources, sha256, alpha, cosinus avant/après, ratio de norme) dans composition."
    )]
    pub patch_out: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "Couches du patch : all (défaut), A-B ou liste.")]
    pub layers: Option<String>,
    #[serde(default)]
    #[schemars(description = "Matrices du patch ; défaut o_proj et down_proj.")]
    pub projections: Option<Vec<String>>,
    #[serde(default)]
    #[schemars(
        range(min = 0.0, max = 1.0),
        description = "Force du project_out, dans [0, 1] (défaut 1 : effacement complet)."
    )]
    pub alpha: Option<f32>,
    #[serde(default)]
    #[schemars(description = "Remplace out et patch_out s'ils existent.")]
    pub overwrite: Option<bool>,
}

pub(super) fn definitions() -> Vec<Value> {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    let schema = settings
        .into_generator()
        .into_root_schema_for::<ComposeToolArgs>();
    vec![json!({
        "name": TOOL_COMPOSE,
        "description": "Compose une direction à partir de plusieurs vecteurs extraits : retirer le refus en préservant des directions de capacité (maths, logique). En mode orthogonal_preserve, les directions préservées sont orthonormées (Gram-Schmidt modifié en f64) et chaque vecteur retiré est projeté hors de leur sous-espace ; un vecteur nul, non fini, de dimension différente ou quasi colinéaire est refusé avec son nom. Écrit la direction composée et, avec patch_out, un patch project_out qui porte la recette. L'orthogonalité est linéaire : mesurer ensuite avec surgeon_bench (refusal et eval). Ne modifie aucun modèle.",
        "inputSchema": serde_json::to_value(schema).unwrap_or(Value::Null),
        "annotations": {
            "title": "Composition de directions",
            "readOnlyHint": false,
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
    ("compose", compose(arguments, ctx))
}

fn compose(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    if ctx.cfg.read_only {
        return Err(invalid(
            "read_only",
            "serveur lancé avec --read-only : surgeon_compose écrit des fichiers, il est désactivé",
        ));
    }
    let args: ComposeToolArgs = serde_json::from_value(arguments)
        .map_err(|e| invalid("invalid_arguments", format!("arguments invalides : {e}")))?;
    let sandbox = &ctx.cfg.sandbox;
    let base = sandbox.base();
    let read = |argument: &str, path: &Path| {
        sandbox
            .resolve_read(argument, path, base)
            .map_err(|e| invalid("path_not_allowed", e))
    };
    let write = |argument: &str, path: &Path| {
        let resolved = sandbox
            .resolve(argument, path, base)
            .map_err(|e| invalid("path_not_allowed", e))?;
        match sandbox.is_read_only(&resolved) {
            true => Err(invalid(
                "path_not_allowed",
                format!("{argument} : {} est en lecture seule", resolved.display()),
            )),
            false => Ok(resolved),
        }
    };
    let ablate = args
        .ablate
        .iter()
        .enumerate()
        .map(|(i, spec)| {
            let (path, alpha) = crate::patch::compose::parse_weighted(spec)
                .map_err(|e| invalid("invalid_arguments", format!("ablate[{i}] : {e}")))?;
            let path = read(&format!("ablate[{i}]"), &path)?;
            Ok(format!("{}:{alpha}", path.display()))
        })
        .collect::<Result<Vec<_>, Outcome>>()?;
    let keep = args
        .keep
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(i, path)| read(&format!("keep[{i}]"), path))
        .collect::<Result<Vec<_>, _>>()?;
    let defaults = ComposeArgs {
        ablate,
        keep,
        mode: args.mode.unwrap_or_default(),
        out: write("out", &args.out)?,
        patch_out: args.patch_out.map(|p| write("patch_out", &p)).transpose()?,
        layers: args.layers.unwrap_or_else(|| "all".into()),
        projections: args
            .projections
            .unwrap_or_else(|| vec!["o_proj".into(), "down_proj".into()]),
        alpha: args.alpha.unwrap_or(1.0),
        overwrite: args.overwrite.unwrap_or(false),
    };
    crate::patch::compose::core_compose(&defaults)
}
