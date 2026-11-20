//! The tools that report to the studio (surgeon-gui) rather than touch a model, for the
//! agents that reach the engine through MCP (Claude Code, Codex, OpenCode, Antigravity):
//! `record_finding` (the evidence log), `show_in_3d` (the node selected in the 3D view) and
//! `update_plan` (the plan shown to the user). They write under `chat/` of the studio project
//! the server is serving (its first allowed directory), which the studio reads while the
//! agent runs; outside a project they fail.
//!
//! The stdio server also journals every other tool call of a project in
//! `chat/mcp_runs.jsonl` and returns its `run_id`, so that a finding can cite the run that
//! measured its figures. The studio's own in-process calls (`call_tool`) are not journaled
//! here: the studio keeps its ledger with its own run ids.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};

pub const TOOL_FINDING: &str = "record_finding";
pub const TOOL_SHOW: &str = "show_in_3d";
pub const TOOL_PLAN: &str = "update_plan";
pub(super) const NAMES: [&str; 3] = [TOOL_FINDING, TOOL_SHOW, TOOL_PLAN];

/// The manifest that marks a studio project folder.
const PROJECT_FILE: &str = "project.surgeon.json";
pub const FINDINGS: &str = "findings.jsonl";
pub const FOCUS: &str = "focus.json";
pub const PLAN: &str = "plan.json";
pub const RUNS: &str = "mcp_runs.jsonl";
/// A larger result is journaled without its data (a graph report): its values cannot back
/// a finding.
const MAX_JOURNALED: usize = 256 * 1024;

pub(super) fn definitions() -> Vec<Value> {
    let annotations = json!({
        "readOnlyHint": false,
        "destructiveHint": false,
        "idempotentHint": false,
        "openWorldHint": false,
    });
    let def = |name: &str, title: &str, description: &str, schema: Value| {
        let mut annotations = annotations.clone();
        annotations["title"] = json!(title);
        json!({"name": name, "description": description, "inputSchema": schema, "annotations": annotations})
    };
    vec![
        def(
            TOOL_FINDING,
            "Consigner un constat",
            "Consigne un constat dans le journal des preuves du projet (onglet Preuves de surgeon-gui). Tout chiffre doit citer le run qui l'a mesuré : evidence[].run_id (le run_id que renvoie chaque appel d'outil surgeon_*) et la valeur exacte du résultat dans value ; sans run, donner le raisonnement (reasoning) : le constat sera « Déduit », sinon « Inconnu ».",
            json!({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "properties": {
                "text": {"type": "string", "description": "Le constat, une phrase."},
                "evidence": {"type": "array", "items": {"type": "object", "properties": {
                    "run_id": {"type": "string"},
                    "metric": {"type": "string"},
                    "value": {"type": "number"}},
                    "required": ["run_id"], "additionalProperties": false}},
                "reasoning": {"type": "string"}},
                "required": ["text"], "additionalProperties": false}),
        ),
        def(
            TOOL_SHOW,
            "Montrer dans la vue 3D",
            "Sélectionne un nœud de la vue 3D de surgeon-gui : nom de tenseur ou identifiant de nœud du graphe (surgeon_graph).",
            json!({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "properties": {"target": {"type": "string"}},
                "required": ["target"], "additionalProperties": false}),
        ),
        def(
            TOOL_PLAN,
            "Mettre à jour le plan",
            "Remplace le plan affiché à l'utilisateur dans surgeon-gui (étapes courtes, dans l'ordre).",
            json!({"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "properties": {"steps": {"type": "array",
                "items": {"type": "string"}}}, "required": ["steps"],
                "additionalProperties": false}),
        ),
    ]
}

/// The `chat/` folder of the studio project being served, when it is one.
pub(super) fn chat_dir(ctx: &ToolContext) -> Option<PathBuf> {
    let base = ctx.cfg.sandbox.base();
    base.join(PROJECT_FILE).is_file().then(|| base.join("chat"))
}

pub(super) fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

fn failure(kind: &str, message: impl std::fmt::Display) -> Outcome {
    Outcome::fail(exit::INVALID_INPUT, message).with_kind(kind)
}

fn write(path: &Path, bytes: &[u8], append: bool) -> Result<(), Outcome> {
    use std::io::Write;
    let parent = path.parent().unwrap_or(path);
    std::fs::create_dir_all(parent)
        .and_then(|()| {
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .append(append)
                .truncate(!append)
                .open(path)
        })
        .and_then(|mut f| f.write_all(bytes))
        .map_err(|e| failure("io", format!("{} : {e}", path.display())))
}

fn line(value: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).unwrap_or_default();
    bytes.push(b'\n');
    bytes
}

pub(super) fn call(
    name: &str,
    arguments: Value,
    ctx: &mut ToolContext,
) -> (&'static str, CoreResult) {
    let result = (|| {
        let chat = chat_dir(ctx).ok_or_else(|| {
            failure(
                "no_project",
                "aucun projet surgeon-gui servi (project.surgeon.json absent du premier dossier autorisé)",
            )
        })?;
        let text = |key: &str| {
            arguments[key]
                .as_str()
                .map(str::trim)
                .filter(|t| !t.is_empty())
                .map(str::to_owned)
        };
        match name {
            TOOL_FINDING => {
                let finding =
                    text("text").ok_or_else(|| failure("invalid_arguments", "text manquant"))?;
                let evidence = match &arguments["evidence"] {
                    Value::Null => json!([]),
                    Value::Array(items) if items.iter().all(|e| e["run_id"].as_str().is_some()) => {
                        Value::Array(items.clone())
                    }
                    _ => {
                        return Err(failure(
                            "invalid_arguments",
                            "evidence : tableau de {run_id, metric?, value?}",
                        ))
                    }
                };
                let mut entry = json!({
                    "text": finding,
                    "evidence": evidence,
                    "conversation": "mcp",
                    "created_unix": now_unix(),
                });
                if let Some(reasoning) = text("reasoning") {
                    entry["reasoning"] = json!(reasoning);
                }
                write(&chat.join(FINDINGS), &line(&entry), true)?;
                Ok(Outcome::ok(
                    json!({"recorded": true}),
                    "constat consigné dans l'onglet Preuves".into(),
                ))
            }
            TOOL_SHOW => {
                let target = text("target")
                    .ok_or_else(|| failure("invalid_arguments", "target manquant"))?;
                let focus = json!({"target": target, "created_unix": now_unix()});
                write(&chat.join(FOCUS), &line(&focus), false)?;
                Ok(Outcome::ok(json!({"selected": target}), String::new()))
            }
            _ => {
                let steps: Vec<String> = arguments["steps"]
                    .as_array()
                    .ok_or_else(|| failure("invalid_arguments", "steps : tableau de chaînes"))?
                    .iter()
                    .filter_map(|s| s.as_str().map(str::to_owned))
                    .collect();
                write(&chat.join(PLAN), &line(&json!({"steps": steps})), false)?;
                Ok(Outcome::ok(json!({"steps": steps.len()}), String::new()))
            }
        }
    })();
    let command = match name {
        TOOL_FINDING => "record_finding",
        TOOL_SHOW => "show_in_3d",
        _ => "update_plan",
    };
    (command, result)
}

/// Journals a finished engine tool call of a project and adds its `run_id` to the result
/// (envelope and text); nothing outside a project, for the studio tools, or when the journal
/// cannot be written (the call itself already succeeded).
pub(super) fn stamp(
    ctx: &ToolContext,
    tool: &str,
    arguments: &Value,
    started_unix: u64,
    result: &mut Value,
) {
    if NAMES.contains(&tool) || tool == super::tools_files::TOOL_FILES {
        return;
    }
    let Some(run_id) = journal(
        ctx,
        tool,
        arguments,
        started_unix,
        &result["structuredContent"],
    ) else {
        return;
    };
    result["structuredContent"]["run_id"] = json!(run_id);
    let text = serde_json::to_string(&result["structuredContent"]).unwrap_or_default();
    result["content"] = json!([{ "type": "text", "text": text }]);
}

fn journal(
    ctx: &ToolContext,
    tool: &str,
    arguments: &Value,
    started_unix: u64,
    envelope: &Value,
) -> Option<String> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let chat = chat_dir(ctx)?;
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let run_id = format!("mcp-{millis}-{}", COUNTER.fetch_add(1, Ordering::Relaxed));
    let data = &envelope["data"];
    let large = serde_json::to_vec(data).map_or(0, |d| d.len()) > MAX_JOURNALED;
    let entry = json!({
        "run_id": run_id,
        "tool": tool,
        // What the studio's summary of the experiments reads (layers, alpha, mode).
        "arguments": arguments,
        "ok": envelope["ok"],
        "exit_code": envelope["exit_code"],
        "data": if large { Value::Null } else { data.clone() },
        "truncated": large,
        "started_unix": started_unix,
        "finished_unix": now_unix(),
    });
    write(&chat.join(RUNS), &line(&entry), true).ok()?;
    Some(run_id)
}
