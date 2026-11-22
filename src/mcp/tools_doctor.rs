//! MCP tool `surgeon_doctor`: `surgeon doctor` through `doctor::core_doctor`, read only. The
//! model is a read path of the sandbox; the cache is never cleaned from here (`--clean-cache`
//! is a CLI decision), and hipfire is the one of the configuration.

use std::path::PathBuf;

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};
use crate::doctor::{core_doctor, DoctorRequest};
use crate::ui::Ui;

pub const TOOL_DOCTOR: &str = "surgeon_doctor";
pub(super) const NAMES: [&str; 1] = [TOOL_DOCTOR];

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(description = "Arguments de surgeon_doctor : un modèle facultatif.")]
pub struct DoctorToolArgs {
    #[serde(default)]
    #[schemars(
        description = "Dossier de modèle : son besoin exact en RAM, le dtype qu'il obtient (f32, ou repli bf16 lent) et le lot de génération native."
    )]
    pub model: Option<PathBuf>,
}

pub(super) fn definitions() -> Vec<Value> {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    let schema = settings
        .into_generator()
        .into_root_schema_for::<DoctorToolArgs>();
    vec![json!({
        "name": TOOL_DOCTOR,
        "description": "Vérifie en quelques secondes que la machine est prête pour une campagne de mesures, sans charger de modèle : RAM libre et taille de modèle exécutable en f32 ou en bf16 sans repli lent (avec model : son besoin exact et son lot natif), cœurs physiques et threads de génération native, cache des références ref_bench (dossier, entrées, taille, entrées illisibles ou obsolètes), et le démon hipfire quand il est configuré ou répond (URL, /health, /metrics, mémoire GPU lue dans son log : 0 Mo au total = démon figé à relancer avec start-hipfire.ps1). data.ready dit si aucun contrôle n'échoue. Lecture seule.",
        "inputSchema": serde_json::to_value(schema).unwrap_or(Value::Null),
        "annotations": {
            "title": "Diagnostic de la machine",
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
    ("doctor", doctor(arguments, ctx))
}

fn doctor(arguments: Value, ctx: &mut ToolContext) -> CoreResult {
    let args: DoctorToolArgs = serde_json::from_value(arguments)
        .map_err(|e| invalid("invalid_arguments", format!("arguments invalides : {e}")))?;
    let sandbox = &ctx.cfg.sandbox;
    let model = args
        .model
        .map(|m| {
            sandbox
                .resolve_read("model", &m, sandbox.base())
                .map_err(|e| invalid("path_not_allowed", e))
        })
        .transpose()?;
    let req = DoctorRequest {
        model,
        ..DoctorRequest::default()
    };
    core_doctor(&req, ctx.deps.probe.as_ref(), &Ui::off())
}
