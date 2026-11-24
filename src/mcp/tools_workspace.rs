//! MCP tool `surgeon_workspace`: makes the server's folder (its first allowed directory) a
//! surgeon-gui project, for an agent that works outside the studio (Claude Code, Codex... with
//! the `weight-surgeon` skill). `init` writes `project.surgeon.json` with the origin model
//! referenced exactly as the studio records it (canonical path, SHA-256 of the file or of the
//! folder's top-level files, their size), so the studio reopens the project with its origin
//! known and locked. The origin is only read; an existing manifest is replaced only with
//! `force`.

use std::path::{Path, PathBuf};

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};
use crate::pipeline::job;

pub const TOOL_WORKSPACE: &str = "surgeon_workspace";
pub(super) const NAMES: [&str; 1] = [TOOL_WORKSPACE];
/// `MANIFEST_VERSION` of surgeon-gui (`studio::project`): its `external_project` test reads
/// what this writes.
const MANIFEST_VERSION: u32 = 1;
/// The folders `Project::create` of the studio makes.
const LAYOUT: [&str; 4] = ["runs", "artifacts", "copies", "chat"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum WorkspaceAction {
    /// Fait du dossier de travail un projet surgeon-gui : project.surgeon.json avec le modèle d'origine (chemin, SHA-256, taille).
    Init,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_workspace : action init, le modèle d'origine, un nom facultatif, force pour remplacer un manifeste existant."
)]
pub struct WorkspaceToolArgs {
    pub action: WorkspaceAction,
    #[schemars(
        description = "Modèle d'origine, lu seulement : dossier SafeTensors (config.json, *.safetensors) ou fichier .safetensors / .gguf, dans un dossier autorisé (--allow-read)."
    )]
    pub model: PathBuf,
    #[serde(default)]
    #[schemars(
        description = "Nom du projet affiché par le studio (64 caractères au plus) ; défaut : le nom du dossier de travail."
    )]
    pub name: Option<String>,
    #[serde(default)]
    #[schemars(description = "Remplace un project.surgeon.json existant (sinon erreur).")]
    pub force: bool,
}

pub(super) fn definitions() -> Vec<Value> {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    let schema = settings
        .into_generator()
        .into_root_schema_for::<WorkspaceToolArgs>();
    vec![json!({
        "name": TOOL_WORKSPACE,
        "description": "Projet surgeon-gui pour un agent hors du studio. action init : écrit project.surgeon.json dans le dossier de travail (le premier dossier autorisé) avec le modèle d'origine comme le studio l'enregistre (chemin canonique, SHA-256, taille), crée runs/, artifacts/, copies/ et chat/. Ensuite chaque appel d'outil est journalisé dans chat/mcp_runs.jsonl avec son run_id, record_finding remplit le journal des preuves, et surgeon-gui rouvre le projet avec son origine reconnue et verrouillée. À appeler une fois, au début. Le modèle n'est que lu ; un manifeste existant n'est remplacé qu'avec force.",
        "inputSchema": serde_json::to_value(schema).unwrap_or(Value::Null),
        "annotations": {
            "title": "Projet du studio",
            "readOnlyHint": false,
            "destructiveHint": false,
            "idempotentHint": false,
            "openWorldHint": false,
        },
    })]
}

fn invalid(kind: &str, message: impl std::fmt::Display) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, message).with_kind(kind)
}

pub(super) fn call(arguments: Value, ctx: &mut ToolContext) -> (&'static str, CoreResult) {
    ("workspace", workspace(arguments, ctx))
}

fn workspace(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    if ctx.cfg.read_only {
        return Err(invalid(
            "read_only",
            "serveur lancé avec --read-only : surgeon_workspace écrit le manifeste du projet",
        ));
    }
    let args: WorkspaceToolArgs = serde_json::from_value(arguments)
        .map_err(|e| invalid("invalid_arguments", format!("arguments invalides : {e}")))?;
    let sandbox = &ctx.cfg.sandbox;
    let model = sandbox
        .resolve_read("model", &args.model, sandbox.base())
        .map_err(|e| invalid("path_not_allowed", e))?;
    match args.action {
        WorkspaceAction::Init => init(sandbox.base(), &model, args.name, args.force),
    }
}

/// The origin as the studio's `project::source_model` records it.
fn source(model: &Path) -> Result<Value, Outcome> {
    let failed = |e: crate::pipeline::PipelineError| invalid("invalid_input", e);
    let path = std::fs::canonicalize(model)
        .map_err(|e| invalid("invalid_input", format!("{} : {e}", model.display())))?;
    let weights = |p: &Path| {
        p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e.to_ascii_lowercase().as_str(), "safetensors" | "gguf"))
    };
    let (sha256, bytes) = match path.is_dir() {
        true => {
            let files = job::files(&path).map_err(failed)?;
            if !files.iter().any(|(_, f)| weights(f)) {
                return Err(invalid(
                    "invalid_input",
                    format!(
                        "{} : aucun fichier de poids (.safetensors, .gguf)",
                        path.display()
                    ),
                ));
            }
            (
                job::sha256_dir(&path).map_err(failed)?,
                job::dir_size(&path).map_err(failed)?,
            )
        }
        false if weights(&path) => (
            job::sha256_file(&path).map_err(failed)?,
            std::fs::metadata(&path)
                .map_err(|e| invalid("invalid_input", format!("{} : {e}", path.display())))?
                .len(),
        ),
        false => {
            return Err(invalid(
                "invalid_input",
                format!(
                    "{} : ni dossier de modèle, ni fichier .safetensors ou .gguf",
                    path.display()
                ),
            ))
        }
    };
    Ok(json!({ "path": path, "sha256": sha256, "bytes": bytes }))
}

fn init(base: &Path, model: &Path, name: Option<String>, force: bool) -> CoreResult {
    let manifest_path = base.join(super::tools_session::PROJECT_FILE);
    let replaced = manifest_path.exists();
    if replaced && !force {
        return Err(invalid(
            "output_exists",
            format!(
                "{} existe déjà : ce dossier est un projet (force pour le remplacer)",
                manifest_path.display()
            ),
        ));
    }
    let name = match name {
        Some(n) => n.trim().to_owned(),
        None => base
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
    };
    if name.is_empty() || name.chars().count() > 64 {
        return Err(invalid(
            "invalid_arguments",
            format!("name « {name} » : 1 à 64 caractères"),
        ));
    }
    let model = std::fs::canonicalize(model).unwrap_or_else(|_| model.to_path_buf());
    let base_dir = std::fs::canonicalize(base).unwrap_or_else(|_| base.to_path_buf());
    if model == base_dir || model.starts_with(base_dir.join("copies")) {
        return Err(invalid(
            "invalid_arguments",
            "model : le modèle d'origine, pas le projet ni une copie de travail",
        ));
    }
    let source = source(&model)?;
    let manifest = json!({
        "schema_version": MANIFEST_VERSION,
        "name": name,
        "created_unix": super::tools_session::now_unix(),
        "source": source,
        "state": "ready",
    });
    for sub in LAYOUT {
        let dir = base.join(sub);
        std::fs::create_dir_all(&dir)
            .map_err(|e| invalid("invalid_input", format!("{} : {e}", dir.display())))?;
    }
    let bytes = serde_json::to_vec_pretty(&manifest).unwrap_or_default();
    job::write_atomic(&manifest_path, &bytes).map_err(|e| invalid("invalid_input", e))?;
    let human = format!(
        "{} {} : origine {} (sha256 {})\n",
        manifest_path.display(),
        if replaced { "remplacé" } else { "écrit" },
        source["path"].as_str().unwrap_or_default(),
        source["sha256"].as_str().unwrap_or_default(),
    );
    Ok(Outcome::ok(
        json!({
            "manifest": manifest_path,
            "name": manifest["name"],
            "source": source,
            "replaced": replaced,
        }),
        human,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(dir: &Path) -> PathBuf {
        let model = dir.join("origin");
        std::fs::create_dir_all(&model).unwrap();
        std::fs::write(model.join("config.json"), b"{}").unwrap();
        std::fs::write(model.join("model.safetensors"), b"weights").unwrap();
        model
    }

    fn kind(o: &Outcome) -> Option<&str> {
        o.error.as_ref().and_then(|e| e.kind.as_deref())
    }

    fn data(result: CoreResult) -> Value {
        result.unwrap_or_else(|o| panic!("{:?}", o.error)).data
    }

    #[test]
    fn init_records_the_origin_once_and_replaces_it_only_with_force() {
        let root = tempfile::tempdir().unwrap();
        let origin = model(root.path());
        let project = root.path().join("my-project");
        std::fs::create_dir_all(&project).unwrap();

        let first = data(init(&project, &origin, None, false));
        assert_eq!(first["name"], "my-project", "the folder's name");
        assert_eq!(first["replaced"], false);
        let source = &first["source"];
        let canonical = std::fs::canonicalize(&origin).unwrap();
        assert_eq!(source["path"], json!(canonical));
        assert_eq!(source["sha256"], job::sha256_dir(&canonical).unwrap());
        assert_eq!(source["bytes"], 2 + 7);
        for sub in LAYOUT {
            assert!(project.join(sub).is_dir(), "{sub}");
        }
        let written: Value =
            serde_json::from_slice(&std::fs::read(project.join("project.surgeon.json")).unwrap())
                .unwrap();
        assert_eq!(written["schema_version"], MANIFEST_VERSION);
        assert_eq!(written["state"], "ready");
        assert_eq!(&written["source"], source);

        // Never over a project by accident; on purpose, with force.
        let again = init(&project, &origin, Some("autre".into()), false).unwrap_err();
        assert_eq!(kind(&again), Some("output_exists"));
        let forced = data(init(&project, &origin, Some(" autre ".into()), true));
        assert_eq!(
            (&forced["name"], &forced["replaced"]),
            (&json!("autre"), &json!(true))
        );
    }

    #[test]
    fn init_refuses_what_is_not_an_origin_model() {
        let root = tempfile::tempdir().unwrap();
        let project = root.path().join("p");
        std::fs::create_dir_all(project.join("copies").join("c1")).unwrap();
        std::fs::write(project.join("copies/c1/model.safetensors"), b"w").unwrap();
        let empty = root.path().join("empty");
        std::fs::create_dir_all(&empty).unwrap();
        let notes = root.path().join("notes.txt");
        std::fs::write(&notes, b"x").unwrap();
        let origin = model(root.path());
        for (model, expected) in [
            (project.join("copies/c1"), "invalid_arguments"),
            (project.clone(), "invalid_arguments"),
            (empty, "invalid_input"),
            (notes, "invalid_input"),
        ] {
            let failure = init(&project, &model, None, false).unwrap_err();
            assert_eq!(kind(&failure), Some(expected), "{}", model.display());
        }
        assert!(
            !project.join("project.surgeon.json").exists(),
            "nothing written"
        );
        let long = "x".repeat(65);
        assert!(init(&project, &origin, Some(long), false).is_err());
        assert!(init(&project, &origin, Some("  ".into()), false).is_err());
        // A lone weights file is an origin too.
        data(init(
            &project,
            &origin.join("model.safetensors"),
            None,
            false,
        ));
    }
}
