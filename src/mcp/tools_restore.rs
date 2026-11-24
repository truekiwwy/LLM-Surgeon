//! MCP tool `surgeon_restore`: the rollback of `surgeon_apply`, through `core_restore`, the
//! same core as `surgeon restore`. The restored model is a write path (`--allow-dir`, never
//! under an `--allow-read` root); the backups and the origin model are only read.

use std::path::{Path, PathBuf};

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};
use crate::cli::RestoreArgs;

pub const TOOL_RESTORE: &str = "surgeon_restore";
pub(super) const NAMES: [&str; 1] = [TOOL_RESTORE];

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_restore : model, puis au plus un de backup, backup_dir ou from_origin (aucun : l'état d'avant le dernier surgeon_apply, depuis les sauvegardes à côté du modèle)."
)]
pub struct RestoreToolArgs {
    #[schemars(
        description = "Fichier ou dossier modèle à remettre en état, dans un dossier autorisé en écriture (--allow-dir)."
    )]
    pub model: PathBuf,
    #[serde(default)]
    #[schemars(
        description = "Sauvegardes écrites par surgeon_apply (data.backups, <nom>.bak-<horodatage>), une par fichier à restaurer."
    )]
    pub backup: Option<Vec<PathBuf>>,
    #[serde(default)]
    #[schemars(
        description = "Dossier où chercher les sauvegardes (le backup_dir de surgeon_apply) ; défaut : à côté du modèle."
    )]
    pub backup_dir: Option<PathBuf>,
    #[serde(default)]
    #[schemars(
        description = "Modèle d'origine (dossier, ou fichier de poids recopié sous son nom dans model, lisible en --allow-read) : chacun de ses fichiers est recopié sur model."
    )]
    pub from_origin: Option<PathBuf>,
}

pub(super) fn definitions() -> Vec<Value> {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    let schema = settings
        .into_generator()
        .into_root_schema_for::<RestoreToolArgs>();
    vec![json!({
        "name": TOOL_RESTORE,
        "description": "Remet un modèle octet pour octet dans un état connu : depuis les sauvegardes de surgeon_apply (backup, ou sans argument l'état d'avant le dernier surgeon_apply, data.checkpoint = \"last_apply\" ; répéter l'appel ne remonte pas plus loin) ou depuis le modèle d'origine (from_origin). SHA-256 de chaque fichier avant et après dans data.files ; échec si l'empreinte finale diffère de la source. À appeler quand surgeon_bench mesure une régression après un surgeon_apply. Les sauvegardes et l'origine ne sont jamais modifiées.",
        "inputSchema": serde_json::to_value(schema).unwrap_or(Value::Null),
        "annotations": {
            "title": "Restauration d'un modèle",
            "readOnlyHint": false,
            "destructiveHint": true,
            "idempotentHint": true,
            "openWorldHint": false,
        },
    })]
}

fn invalid(kind: &str, message: impl std::fmt::Display) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, message).with_kind(kind)
}

pub(super) fn call(arguments: Value, ctx: &mut ToolContext) -> (&'static str, CoreResult) {
    ("restore", restore(arguments, ctx))
}

fn restore(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    if ctx.cfg.read_only {
        return Err(invalid(
            "read_only",
            "serveur lancé avec --read-only : surgeon_restore réécrit le modèle, il est désactivé",
        ));
    }
    let args: RestoreToolArgs = serde_json::from_value(arguments)
        .map_err(|e| invalid("invalid_arguments", format!("arguments invalides : {e}")))?;
    let sandbox = &ctx.cfg.sandbox;
    let base = sandbox.base();
    let model = sandbox
        .resolve("model", &args.model, base)
        .map_err(|e| invalid("path_not_allowed", e))?;
    if sandbox.is_read_only(&model) {
        return Err(invalid(
            "path_not_allowed",
            format!(
                "model : {} est dans un dossier en lecture seule",
                model.display()
            ),
        ));
    }
    let read = |argument: &str, path: &Path| {
        sandbox
            .resolve_read(argument, path, base)
            .map_err(|e| invalid("path_not_allowed", e))
    };
    let backup = args
        .backup
        .unwrap_or_default()
        .iter()
        .enumerate()
        .map(|(i, path)| read(&format!("backup[{i}]"), path))
        .collect::<Result<_, _>>()?;
    let request = RestoreArgs {
        model,
        backup,
        backup_dir: args
            .backup_dir
            .map(|dir| read("backup_dir", &dir))
            .transpose()?,
        from_origin: args
            .from_origin
            .map(|origin| read("from_origin", &origin))
            .transpose()?,
    };
    crate::patch::core_restore(&request, ctx.deps)
}
