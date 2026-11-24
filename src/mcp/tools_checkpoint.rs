use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::{readable, readable_or_builtin, ToolContext};
use crate::app::{exit, CoreResult, Outcome};
use crate::checkpoint::sweep::{self, Drift, EngineLab, Target};
use crate::checkpoint::{CheckpointError, Store};
use crate::patch::{OperationKind, Projection};
use crate::pipeline::config::PillarKind;

pub const TOOL_CHECKPOINT: &str = "surgeon_checkpoint";
pub const TOOL_SWEEP: &str = "surgeon_sweep";
pub(super) const NAMES: [&str; 2] = [TOOL_CHECKPOINT, TOOL_SWEEP];
const MAX_POINTS: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CheckpointAction {
    #[schemars(
        description = "Instantané de la copie de travail model sous work/checkpoints/<id>/model/ : liens physiques (copie si autre volume), manifeste SHA-256, étiquette, métriques, parent."
    )]
    Create,
    #[schemars(
        description = "Arbre des checkpoints (parent, enfants, métriques) et empreinte disque réelle : un fichier partagé par liens physiques n'est compté qu'une fois."
    )]
    List,
    #[schemars(
        description = "Remet la copie de travail model dans l'état du checkpoint id : SHA-256 du checkpoint vérifié, puis échange atomique du dossier. Les sauvegardes .bak-* et sous-dossiers de la copie sont gardés."
    )]
    Restore,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_checkpoint : action, la copie de travail model (create, restore), id (restore), label et metrics (create)."
)]
pub struct CheckpointToolArgs {
    pub action: CheckpointAction,
    #[serde(default)]
    #[schemars(
        description = "Copie de travail (dossier ou fichier de poids) dans un dossier autorisé en écriture (--allow-dir), jamais le modèle d'origine. Obligatoire pour create et restore."
    )]
    pub model: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "restore : identifiant du checkpoint (data.id de create, ou de list)."
    )]
    pub id: Option<String>,
    #[serde(default)]
    #[schemars(description = "create : étiquette libre (128 caractères au plus).")]
    pub label: Option<String>,
    #[serde(default)]
    #[schemars(
        description = "create : métriques mesurées sur cet état, par exemple {\"gain\": 0.3, \"drift\": 0.02}."
    )]
    pub metrics: Option<BTreeMap<String, f64>>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SweepTarget {
    #[schemars(
        description = "Métrique cible, taux de succès comme un pilier du pipeline : unblocking (réponses sans refus), safety (refus) ou gsm8k (exactitude)."
    )]
    pub kind: PillarKind,
    #[schemars(
        description = "Prompts (unblocking, safety) ou dataset GSM8K, fichier ou builtin:refusals, builtin:math..."
    )]
    pub dataset: PathBuf,
    #[serde(default)]
    #[schemars(range(min = 1), description = "Tokens générés au plus par réponse.")]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    #[schemars(range(min = 1), description = "gsm8k : premières questions retenues.")]
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SweepDrift {
    #[serde(default)]
    #[schemars(
        description = "kl (défaut) : KL(base‖candidat) moyenne par prompt ; logits : part des positions dont le token le plus probable change."
    )]
    pub kind: Option<PillarKind>,
    #[schemars(description = "Prompts neutres de la dérive, fichier ou builtin:general...")]
    pub dataset: PathBuf,
    #[serde(default)]
    #[schemars(range(min = 1), description = "Tokens lus au plus par prompt du dump.")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_sweep : la copie de travail, les vecteurs, l'opération, la grille alphas × fenêtres layers, le budget max_drift, la métrique cible et la dérive."
)]
pub struct SweepToolArgs {
    #[schemars(
        description = "Copie de travail réécrite à chaque point puis remise en état ; dans un dossier autorisé en écriture (--allow-dir)."
    )]
    pub model: PathBuf,
    #[schemars(
        length(min = 1),
        description = "Vecteurs de direction (surgeon_extract, surgeon_compose : tenseur direction ; ou tableaux JSON)."
    )]
    pub vectors: Vec<PathBuf>,
    #[serde(default)]
    #[schemars(description = "Tenseur à lire dans chaque .safetensors (défaut direction).")]
    pub vector_tensor: Option<String>,
    pub operation: OperationKind,
    #[serde(default)]
    #[schemars(description = "Matrices visées ; défaut o_proj et down_proj.")]
    pub projections: Option<Vec<Projection>>,
    #[schemars(
        length(min = 1),
        description = "Grille des coefficients, par exemple [0.5, 0.8, 1.0, 1.2] (project_out : dans [0, 1] ; add_direction : amplification ≥ 0)."
    )]
    pub alphas: Vec<f32>,
    #[schemars(
        length(min = 1),
        description = "Fenêtres de couches candidates, chacune au format layers : \"all\", \"10-20\" ou \"3,5,7\" (la fenêtre causale de surgeon_causal_trace)."
    )]
    pub layers: Vec<String>,
    #[schemars(
        range(min = 0.0),
        description = "Dérive maximale admise : un point au-delà est annulé sans checkpoint."
    )]
    pub max_drift: f64,
    pub target: SweepTarget,
    pub drift: SweepDrift,
}

fn schema<T: JsonSchema>() -> Value {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    serde_json::to_value(settings.into_generator().into_root_schema_for::<T>())
        .unwrap_or(Value::Null)
}

pub(super) fn definitions() -> Vec<Value> {
    vec![
        json!({
            "name": TOOL_CHECKPOINT,
            "description": "Checkpoints instantanés d'une copie de travail, sous work/checkpoints/ du dossier de travail. create : liens physiques des fichiers du modèle (repli sur copie si autre volume) et manifeste SHA-256, avec étiquette, métriques et parent (le dernier checkpoint créé ou restauré de cette copie) ; list : l'arbre et l'empreinte disque réelle (disk_bytes, exclusive_bytes) ; restore : vérifie le SHA-256 du checkpoint puis remet la copie dans cet état par échange atomique de dossier, sans jamais toucher au modèle d'origine. Un fichier partagé avec un checkpoint est recopié avant toute écriture en place (surgeon_apply, surgeon_lora merge) : le checkpoint ne change jamais. Les outils ne peuvent pas écrire sous work/checkpoints/.",
            "inputSchema": schema::<CheckpointToolArgs>(),
            "annotations": {
                "title": "Checkpoints d'une copie de travail",
                "readOnlyHint": false,
                "destructiveHint": true,
                "idempotentHint": false,
                "openWorldHint": false,
            },
        }),
        json!({
            "name": TOOL_SWEEP,
            "description": "Balayage Pareto automatique sur une copie de travail : pour chaque point de la grille vecteurs × fenêtres layers × alphas (64 au plus, tous validés par dry-run avant la première écriture), applique le patch, mesure la métrique cible (target, génération gloutonne native) et la dérive (drift, logits natifs contre l'état de départ). Un point qui dépasse max_drift ou que le front domine est annulé (rollback vers le checkpoint de base) ; un point qui améliore le front de Pareto (gain plus haut ou dérive plus basse) devient un checkpoint, puis la copie revient à la base. À la fin : data.front (indices de data.points non dominés, par dérive croissante), data.best (le gain le plus haut sous max_drift, à égalité la dérive la plus basse) et la copie restaurée sur ce meilleur compromis (data.restored ; la base si aucun point n'a de gain). Erreur ou annulation : la copie revient à la base.",
            "inputSchema": schema::<SweepToolArgs>(),
            "annotations": {
                "title": "Balayage Pareto",
                "readOnlyHint": false,
                "destructiveHint": true,
                "idempotentHint": false,
                "openWorldHint": false,
            },
        }),
    ]
}

fn invalid(kind: &str, message: impl std::fmt::Display) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, message).with_kind(kind)
}

fn failure(e: CheckpointError) -> Outcome {
    let code = match e {
        CheckpointError::Invalid(_) => exit::INVALID_INPUT,
        CheckpointError::Failed(_) => exit::EXECUTION_FAILED,
        _ => exit::GENERAL,
    };
    let kind = e.kind();
    Outcome::fail(code, e).with_kind(kind)
}

pub(super) fn call(
    name: &str,
    arguments: Value,
    ctx: &mut ToolContext,
) -> (&'static str, CoreResult) {
    match name {
        TOOL_SWEEP => ("sweep", sweep_tool(arguments, ctx)),
        _ => ("checkpoint", checkpoint(arguments, ctx)),
    }
}

fn parse<T: serde::de::DeserializeOwned>(arguments: Value) -> Result<T, Outcome> {
    serde_json::from_value(arguments)
        .map_err(|e| invalid("invalid_arguments", format!("arguments invalides : {e}")))
}

fn working_copy(ctx: &ToolContext, model: Option<&Path>) -> Result<PathBuf, Outcome> {
    let model = model.ok_or_else(|| invalid("invalid_arguments", "model requis"))?;
    let sandbox = &ctx.cfg.sandbox;
    let path = sandbox
        .resolve("model", model, sandbox.base())
        .map_err(|e| invalid("path_not_allowed", e))?;
    if sandbox.is_read_only(&path) {
        return Err(invalid(
            "path_not_allowed",
            format!(
                "model : {} est dans un dossier en lecture seule",
                path.display()
            ),
        ));
    }
    if !path.exists() {
        return Err(invalid(
            "invalid_input",
            format!("{} introuvable", path.display()),
        ));
    }
    Ok(path)
}

fn refuse_read_only(ctx: &ToolContext, tool: &str) -> Result<(), Outcome> {
    match ctx.cfg.read_only {
        true => Err(invalid(
            "read_only",
            format!("serveur lancé avec --read-only : {tool} écrit la copie de travail"),
        )),
        false => Ok(()),
    }
}

fn checkpoint(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: CheckpointToolArgs = parse(arguments)?;
    let store = Store::in_work(&ctx.cfg.sandbox.work_dir());
    match args.action {
        CheckpointAction::List => {
            let listing = store.list().map_err(failure)?;
            let human = listing
                .checkpoints
                .iter()
                .map(|c| {
                    let m = &c.manifest;
                    format!(
                        "{} {:<24} parent {:<4} {} Mio exclusifs\n",
                        m.id,
                        m.label,
                        m.parent.as_deref().unwrap_or("-"),
                        c.exclusive_bytes >> 20
                    )
                })
                .chain([format!(
                    "{} checkpoint(s), {} Mio sur disque pour {} Mio logiques\n",
                    listing.checkpoints.len(),
                    listing.disk_bytes >> 20,
                    listing.logical_bytes >> 20
                )])
                .collect();
            Ok(Outcome::ok(json!(listing), human))
        }
        CheckpointAction::Create => {
            refuse_read_only(ctx, TOOL_CHECKPOINT)?;
            let model = working_copy(ctx, args.model.as_deref())?;
            let label = args.label.unwrap_or_default();
            if label.chars().count() > 128 {
                return Err(invalid(
                    "invalid_arguments",
                    "label : 128 caractères au plus",
                ));
            }
            let manifest = store
                .create(&model, &label, args.metrics.unwrap_or_default())
                .map_err(failure)?;
            let human = format!(
                "checkpoint {} de {} ({} fichier(s), sha256 {})\n",
                manifest.id,
                model.display(),
                manifest.files.len(),
                manifest.sha256
            );
            Ok(Outcome::ok(json!(manifest), human))
        }
        CheckpointAction::Restore => {
            refuse_read_only(ctx, TOOL_CHECKPOINT)?;
            let model = working_copy(ctx, args.model.as_deref())?;
            let id = args
                .id
                .ok_or_else(|| invalid("invalid_arguments", "id requis pour restore"))?;
            let restored = store.restore(&id, &model).map_err(failure)?;
            let human = format!(
                "{} remis dans l'état du checkpoint {} (sha256 {})\n",
                model.display(),
                restored.id,
                restored.sha256
            );
            Ok(Outcome::ok(json!(restored), human))
        }
    }
}

fn sweep_tool(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    refuse_read_only(ctx, TOOL_SWEEP)?;
    let args: SweepToolArgs = parse(arguments)?;
    let model = working_copy(ctx, Some(&args.model))?;
    let base = ctx.cfg.sandbox.base().to_path_buf();
    let vectors = args
        .vectors
        .iter()
        .enumerate()
        .map(|(i, v)| readable(ctx, &format!("vectors[{i}]"), v, &base))
        .collect::<Result<Vec<_>, _>>()?;
    let grid = sweep::grid(&vectors, &args.layers, &args.alphas);
    if grid.is_empty() || grid.len() > MAX_POINTS {
        return Err(invalid(
            "invalid_arguments",
            format!(
                "grille de {} point(s) : entre 1 et {MAX_POINTS} (vectors × layers × alphas)",
                grid.len()
            ),
        ));
    }
    let target = Target {
        kind: args.target.kind,
        dataset: readable_or_builtin(ctx, "target.dataset", &args.target.dataset, &base)?,
        max_new_tokens: args.target.max_new_tokens,
        limit: args.target.limit,
    };
    let drift = Drift {
        kind: args.drift.kind.unwrap_or(PillarKind::Kl),
        dataset: readable_or_builtin(ctx, "drift.dataset", &args.drift.dataset, &base)?,
        max_tokens: args.drift.max_tokens,
    };
    let work = ctx.cfg.sandbox.work_dir();
    let scratch = work.join(format!(".sweep-{}", std::process::id()));
    std::fs::create_dir_all(&scratch)
        .map_err(|e| invalid("invalid_input", format!("{} : {e}", scratch.display())))?;
    let store = Store::in_work(&work);
    let deps = ctx.deps;
    let mut lab = EngineLab {
        deps,
        sink: &mut *ctx.sink,
        operation: args.operation,
        projections: args
            .projections
            .unwrap_or_else(|| vec![Projection::AttnOut, Projection::Down]),
        vector_tensor: args.vector_tensor,
        target,
        drift,
        scratch: scratch.clone(),
    };
    let report = lab.validate().and_then(|()| {
        sweep::run(
            &store,
            &model,
            &grid,
            args.max_drift,
            &mut lab,
            &deps.interrupted,
        )
    });
    let _ = std::fs::remove_dir_all(&scratch);
    let report = report.map_err(failure)?;
    let human = match report.best {
        Some(i) => {
            let best = &report.points[i];
            format!(
                "{} point(s), front de {} ; meilleur compromis {} : gain {:+.4}, dérive {:.4}, copie sur le checkpoint {}\n",
                report.points.len(),
                report.front.len(),
                best.candidate.label(),
                best.gain,
                best.drift,
                report.restored
            )
        }
        None => format!(
            "{} point(s), aucun gain sous max_drift {} : copie remise sur la base (checkpoint {})\n",
            report.points.len(),
            report.max_drift,
            report.restored
        ),
    };
    Ok(Outcome::ok(json!(report), human))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schemas_ask_for_what_each_action_needs() {
        let defs = definitions();
        let names: Vec<_> = defs.iter().map(|d| d["name"].as_str().unwrap()).collect();
        assert_eq!(names, NAMES);
        let sweep = &defs[1]["inputSchema"];
        let required: Vec<_> = sweep["required"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        for key in [
            "model",
            "vectors",
            "operation",
            "alphas",
            "layers",
            "max_drift",
            "target",
            "drift",
        ] {
            assert!(required.contains(&key), "{key}");
        }
        let actions = &defs[0]["inputSchema"]["properties"]["action"];
        assert_eq!(
            actions.to_string().matches("description").count(),
            3,
            "{actions}"
        );
    }
}
