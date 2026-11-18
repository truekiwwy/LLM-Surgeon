//! MCP tool `surgeon_files`: the working files of an agent that has no file tool of its own
//! (Claude Code and Codex as the studio runs them, API sessions). It lists and reads text in
//! the allowed folders, read-only roots included (the configured models folder), and writes
//! only under `<first allowed folder>/work/`: neither weights, nor a working copy, nor the
//! studio's own records (runs, ledger, approvals) can be reached through it.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};
use crate::pipeline::job::{write_atomic, PRIVATE_MARKER};

pub const TOOL_FILES: &str = "surgeon_files";
pub(super) const NAMES: [&str; 1] = [TOOL_FILES];
/// What a working file may be: text an agent prepares, never something the engine executes
/// or loads as weights.
const WRITABLE: [&str; 15] = [
    "jsonl", "json", "txt", "md", "csv", "tsv", "py", "ps1", "sh", "js", "toml", "yaml", "yml",
    "tmpl", "jinja",
];
const MAX_READ_BYTES: u64 = 1 << 20;
const MAX_ENTRIES: usize = 2000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FileAction {
    /// Contenu d'un dossier autorisé ; sans path, les dossiers autorisés et le premier d'entre eux.
    List,
    /// Fichier texte d'un dossier autorisé (1 Mio au plus, data.truncated au-delà).
    Read,
    /// Fichier de travail sous work/ (créé avec ses dossiers).
    Write,
}

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_files : action, puis path (requis pour read et write) et content (requis pour write)."
)]
pub struct FilesToolArgs {
    pub action: FileAction,
    #[serde(default)]
    #[schemars(
        description = "list, read : chemin absolu, ou relatif au premier dossier autorisé. write : relatif à work/ (prompts_pos.jsonl), ou absolu sous work/."
    )]
    pub path: Option<PathBuf>,
    #[serde(default)]
    #[schemars(description = "write : le texte du fichier.")]
    pub content: Option<String>,
    #[serde(default)]
    #[schemars(description = "write : ajoute à la fin du fichier au lieu de le créer.")]
    pub append: bool,
    #[serde(default)]
    #[schemars(description = "write : remplace un fichier existant (sinon erreur).")]
    pub overwrite: bool,
}

pub(super) fn definitions() -> Vec<Value> {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    let schema = settings
        .into_generator()
        .into_root_schema_for::<FilesToolArgs>();
    vec![json!({
        "name": TOOL_FILES,
        "description": "Fichiers de travail de l'agent. list et read parcourent les dossiers autorisés, dossiers de modèles en lecture seule compris (data.entries[].model signale un dossier de modèle). write crée des fichiers texte (.jsonl, .json, .txt, .md, .csv, .py...) uniquement sous work/ du dossier de travail et renvoie data.path, le chemin absolu à passer ensuite à surgeon_extract (pos_prompts_file, neg_prompts_file), surgeon_autotune (eval_dataset) ou surgeon_bench (dataset, prompts). Jamais de poids, de copie de travail ni de journal du projet.",
        "inputSchema": serde_json::to_value(schema).unwrap_or(Value::Null),
        "annotations": {
            "title": "Fichiers de travail",
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

fn io_failure(path: &Path, e: std::io::Error) -> Outcome {
    invalid("invalid_input", format!("{} : {e}", path.display()))
}

pub(super) fn call(arguments: Value, ctx: &mut ToolContext) -> (&'static str, CoreResult) {
    ("files", files(arguments, ctx))
}

fn files(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: FilesToolArgs = serde_json::from_value(arguments)
        .map_err(|e| invalid("invalid_arguments", format!("arguments invalides : {e}")))?;
    match args.action {
        FileAction::List => list(&args, ctx),
        FileAction::Read => read(&args, ctx),
        FileAction::Write => write(&args, ctx),
    }
}

fn required_path(args: &FilesToolArgs) -> Result<&Path, Outcome> {
    args.path
        .as_deref()
        .ok_or_else(|| invalid("invalid_arguments", "path requis"))
}

fn readable(ctx: &ToolContext, path: &Path) -> Result<PathBuf, Outcome> {
    let sandbox = &ctx.cfg.sandbox;
    sandbox
        .resolve_read("path", path, sandbox.base())
        .map_err(|e| invalid("path_not_allowed", e))
}

fn list(args: &FilesToolArgs, ctx: &ToolContext) -> CoreResult {
    let sandbox = &ctx.cfg.sandbox;
    let dir = readable(ctx, args.path.as_deref().unwrap_or(Path::new(".")))?;
    let mut entries: Vec<Value> = std::fs::read_dir(&dir)
        .map_err(|e| io_failure(&dir, e))?
        .flatten()
        .filter(|e| !e.path().join(PRIVATE_MARKER).is_file())
        .map(|e| {
            let path = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            match path.is_dir() {
                true => json!({
                    "name": name,
                    "kind": "dir",
                    "model": path.join("config.json").is_file(),
                }),
                false => json!({
                    "name": name,
                    "kind": "file",
                    "bytes": e.metadata().map_or(0, |m| m.len()),
                }),
            }
        })
        .collect();
    entries.sort_by_key(|e| e["name"].as_str().unwrap_or_default().to_owned());
    let total = entries.len();
    entries.truncate(MAX_ENTRIES);
    Ok(Outcome::ok(
        json!({
            "path": dir,
            "entries": entries,
            "total": total,
            "truncated": total > MAX_ENTRIES,
            "work_dir": sandbox.work_dir(),
            "writable_roots": sandbox.roots(),
            "read_only_roots": sandbox.read_roots(),
        }),
        format!("{} : {total} entrée(s)\n", dir.display()),
    ))
}

fn read(args: &FilesToolArgs, ctx: &ToolContext) -> CoreResult {
    let path = readable(ctx, required_path(args)?)?;
    let file = std::fs::File::open(&path).map_err(|e| io_failure(&path, e))?;
    let total = file.metadata().map_err(|e| io_failure(&path, e))?.len();
    let mut bytes = Vec::new();
    file.take(MAX_READ_BYTES)
        .read_to_end(&mut bytes)
        .map_err(|e| io_failure(&path, e))?;
    let truncated = total > MAX_READ_BYTES;
    // The cut may fall inside a character; anything else invalid is a binary file.
    let valid = match std::str::from_utf8(&bytes) {
        Ok(_) => bytes.len(),
        Err(e) if truncated && bytes.len() - e.valid_up_to() < 4 => e.valid_up_to(),
        Err(_) => {
            return Err(invalid(
                "invalid_input",
                format!("{} : pas un fichier texte UTF-8", path.display()),
            ))
        }
    };
    bytes.truncate(valid);
    let content = String::from_utf8(bytes).unwrap_or_default();
    Ok(Outcome::ok(
        json!({ "path": path, "content": content, "bytes": total, "truncated": truncated }),
        String::new(),
    ))
}

fn write(args: &FilesToolArgs, ctx: &ToolContext) -> CoreResult {
    if ctx.cfg.read_only {
        return Err(invalid(
            "read_only",
            "serveur lancé avec --read-only : surgeon_files ne peut pas écrire",
        ));
    }
    let content = args
        .content
        .as_deref()
        .ok_or_else(|| invalid("invalid_arguments", "content requis"))?;
    let requested = required_path(args)?;
    let sandbox = &ctx.cfg.sandbox;
    let path = sandbox.resolve_work("path", requested).map_err(|_| {
        invalid(
            "path_not_allowed",
            format!(
                "path : l'écriture est limitée à {}",
                sandbox.work_dir().display()
            ),
        )
    })?;
    let allowed = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| WRITABLE.contains(&e.to_ascii_lowercase().as_str()));
    if !allowed {
        return Err(invalid(
            "invalid_arguments",
            format!(
                "path : extension non permise, au choix {}",
                WRITABLE.join(", ")
            ),
        ));
    }
    if path.exists() && !args.overwrite && !args.append {
        return Err(invalid(
            "output_exists",
            format!("{} existe déjà (overwrite ou append)", path.display()),
        ));
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| io_failure(dir, e))?;
    }
    match args.append {
        true => std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .and_then(|mut f| f.write_all(content.as_bytes()))
            .map_err(|e| io_failure(&path, e))?,
        false => {
            write_atomic(&path, content.as_bytes()).map_err(|e| invalid("invalid_input", e))?
        }
    }
    let bytes = std::fs::metadata(&path).map_or(0, |m| m.len());
    Ok(Outcome::ok(
        json!({ "path": path, "bytes": bytes }),
        format!("{} écrit ({bytes} octets)\n", path.display()),
    ))
}
