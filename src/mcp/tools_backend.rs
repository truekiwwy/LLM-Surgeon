//! MCP tool `surgeon_backends`: the GPUs and the Rust inference engines of the machine the
//! server runs on, the user configuration and the recommendation (the first-start probe of
//! `surgeon backend detect`, read only). It runs nothing the operator did not configure.

use schemars::gen::SchemaSettings;
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use super::tools::ToolContext;
use crate::app::{exit, CoreResult, Outcome};
use crate::backend::{describe_config, detect_hardware_and_backends, load_user_config};

pub const TOOL_BACKENDS: &str = "surgeon_backends";
pub(super) const NAMES: [&str; 1] = [TOOL_BACKENDS];

#[derive(Debug, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
#[schemars(
    description = "Arguments de surgeon_backends : aucun. Renvoie les GPU, les moteurs d'inférence Rust (hipfire, mistralrs, native), la configuration utilisateur et le backend recommandé."
)]
pub struct BackendsToolArgs {}

pub(super) fn definitions() -> Vec<Value> {
    let mut settings = SchemaSettings::draft07();
    settings.inline_subschemas = true;
    let schema = settings
        .into_generator()
        .into_root_schema_for::<BackendsToolArgs>();
    vec![json!({
        "name": TOOL_BACKENDS,
        "description": "Liste les GPU de la machine et ses moteurs d'inférence Rust : hipfire (démon des GPU AMD), mistralrs (mistral.rs, intégré à surgeon quand il est compilé avec cuda ou metal, sinon son binaire : GPU NVIDIA ou Apple Silicon, dossier SafeTensors tel quel) et native (CPU, seulement sur une machine sans GPU), la configuration utilisateur persistante et le backend recommandé (vendor, target, recommended, reason). Sur une machine avec GPU, target et recommended sont toujours son moteur GPU, même pas encore prêt : reason dit ce qui manque (voir surgeon_doctor) et le backend auto de surgeon_bench le prend. Lecture seule ; à appeler avant surgeon_bench ou surgeon_pipeline pour savoir quel moteur tourne.",
        "inputSchema": serde_json::to_value(schema).unwrap_or(Value::Null),
        "annotations": {
            "title": "Moteurs d'inférence",
            "readOnlyHint": true,
            "destructiveHint": false,
            "idempotentHint": true,
            "openWorldHint": false,
        },
    })]
}

pub(super) fn call(arguments: Value, _ctx: &mut ToolContext) -> (&'static str, CoreResult) {
    ("backends", backends(arguments))
}

fn backends(arguments: Value) -> CoreResult {
    serde_json::from_value::<BackendsToolArgs>(arguments).map_err(|e| {
        Outcome::fail(exit::INVALID_INPUT, format!("arguments invalides : {e}"))
            .with_kind("invalid_arguments")
    })?;
    let user = load_user_config().map_err(|e| {
        Outcome::fail(
            exit::INVALID_INPUT,
            format!("configuration utilisateur : {e}"),
        )
        .with_kind("invalid_input")
    })?;
    let report = detect_hardware_and_backends(&user);
    let effective = describe_config(&user);
    Ok(Outcome::ok(
        json!({"report": report, "user_config": user, "effective": effective}),
        String::new(),
    ))
}
