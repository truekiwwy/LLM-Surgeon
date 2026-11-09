//! MCP tools `surgeon_diff` and `surgeon_lora` (frozen names; the real argument types,
//! schemas and execution are written by the `mcpdocs` agent, wave 2).

use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};

pub const TOOL_DIFF: &str = "surgeon_diff";
pub const TOOL_LORA: &str = "surgeon_lora";
pub(super) const NAMES: [&str; 2] = [TOOL_DIFF, TOOL_LORA];

pub(super) fn definitions() -> Vec<Value> {
    NAMES
        .iter()
        .map(|name| {
            json!({
                "name": name,
                "description": "Outil en cours d'implémentation : répond not_implemented tant que le moteur n'est pas livré.",
                "inputSchema": { "type": "object" },
                "annotations": { "title": name, "openWorldHint": false, "readOnlyHint": false, "destructiveHint": false },
            })
        })
        .collect()
}

pub(super) fn call(
    name: &str,
    _arguments: Value,
    _ctx: &mut ToolContext,
) -> (&'static str, CoreResult) {
    let command = if name == TOOL_DIFF { "diff" } else { "lora" };
    (
        command,
        Err(Outcome::fail(exit::INTERNAL, "pas encore implémenté").with_kind("not_implemented")),
    )
}
