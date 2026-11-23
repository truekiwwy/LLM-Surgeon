//! The real executor: an engine tool goes through `weight_surgeon::mcp::call_tool`, the
//! code path of `surgeon mcp` (same argument schema, sandbox, cores, backups and rollback),
//! with the project folder as allowed directory and the source model's folder as read-only
//! root. The working copy is the only job the studio runs itself.

use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use serde_json::{json, Value};
use weight_surgeon::eval::{quantize_safetensors, GgufQuant};
use weight_surgeon::events::SurgeonEvent;
use weight_surgeon::mcp::{call_tool, McpConfig, PipelineJobs};

use super::bus::Recorder;
use super::copy;
use super::jobs::{Executor, JobSpec, Outcome, RunContext, COPY_TOOL, EXPORT_TOOL};

#[derive(Default)]
pub struct EngineExecutor {
    /// Keeps a detached `surgeon_pipeline` worker alive while the studio runs.
    pipeline: PipelineJobs,
}

impl Executor for EngineExecutor {
    fn run(&self, spec: &JobSpec, args: &Value, ctx: &RunContext, rec: &mut Recorder) -> Outcome {
        if spec.tool == COPY_TOOL {
            return working_copy(args, ctx, rec);
        }
        if spec.tool == EXPORT_TOOL {
            return gguf_export(args, ctx, rec);
        }
        // The source model and the models folder are read only, as for the CLI runtimes
        // (`--allow-read`): the engine itself refuses a write there, not only the policy.
        let read: Vec<_> = ctx
            .source
            .iter()
            .map(|s| super::copy::model_root(&s.path).to_path_buf())
            .chain(weight_surgeon::backend::models_dir())
            .collect();
        // allow_no_backup false: the apply backups are imposed, as over MCP.
        let cfg = McpConfig::new(
            std::slice::from_ref(&ctx.project_dir),
            spec.read_only,
            false,
        )
        .and_then(|cfg| cfg.with_read_roots(&read));
        let cfg = match cfg {
            Ok(cfg) => cfg,
            Err(e) => return Outcome::failed(format!("dossiers autorisés : {e}")),
        };
        let deps = crate::runner::deps(Arc::clone(&ctx.cancel));
        match call_tool(&spec.tool, args.clone(), &deps, &cfg, rec, &self.pipeline) {
            Ok(result) => {
                let envelope = result["structuredContent"].clone();
                Outcome {
                    exit_code: envelope["exit_code"].as_i64().map_or(1, |c| c as i32),
                    cancelled: envelope["error"]["kind"] == "cancelled",
                    envelope,
                }
            }
            Err(e) => Outcome::failed(e),
        }
    }
}

fn gguf_export(args: &Value, ctx: &RunContext, rec: &mut Recorder) -> Outcome {
    let quant = match serde_json::from_value::<GgufQuant>(args["format"].clone()) {
        Ok(quant) => quant,
        Err(e) => return Outcome::failed(format!("format : {e}")),
    };
    let (Some(model), Some(out)) = (args["model"].as_str(), args["out"].as_str()) else {
        return Outcome::failed(format!("{EXPORT_TOOL} : model et out requis"));
    };
    let made = quantize_safetensors(
        Path::new(model),
        Path::new(out),
        quant,
        &ctx.cancel,
        &mut |done, total| {
            rec.emit(SurgeonEvent::Progress {
                done: done as u64,
                total: Some(total as u64),
                unit: "tensor".into(),
            })
        },
    );
    match made {
        Ok(()) => {
            rec.emit(SurgeonEvent::Artifact {
                kind: "gguf".into(),
                path: out.into(),
                sha256: None,
            });
            Outcome {
                exit_code: 0,
                envelope: json!({ "ok": true, "command": EXPORT_TOOL, "exit_code": 0,
                    "data": { "gguf": out, "format": quant } }),
                cancelled: false,
            }
        }
        Err(e) => Outcome {
            cancelled: ctx.cancel.load(Ordering::SeqCst),
            ..Outcome::failed(e.to_string())
        },
    }
}

/// `studio_copy {"id"?: "..."}`: a working copy of the source model in `copies/<id>/`.
fn working_copy(args: &Value, ctx: &RunContext, rec: &mut Recorder) -> Outcome {
    let Some(source) = &ctx.source else {
        return Outcome::failed("le projet n'a pas de modèle d'origine");
    };
    let id = match args["id"].as_str() {
        Some(id) => id.to_owned(),
        None => format!("copy-{}", super::project::now_unix()),
    };
    if let Err(e) = super::project::validate_name(&id) {
        return Outcome::failed(e);
    }
    let Some(free) = copy::free_space(&ctx.copies_dir) else {
        return Outcome::failed(format!(
            "espace libre inconnu pour {}",
            ctx.copies_dir.display()
        ));
    };
    let made = copy::make(
        &source.path,
        &ctx.copies_dir,
        &id,
        free,
        &ctx.cancel,
        |done, total| {
            rec.emit(SurgeonEvent::Progress {
                done: done as u64,
                total: Some(total as u64),
                unit: "file".into(),
            })
        },
    );
    match made {
        Ok(path) => {
            rec.emit(SurgeonEvent::Artifact {
                kind: "working_copy".into(),
                path: path.clone(),
                sha256: None,
            });
            Outcome {
                exit_code: 0,
                envelope: json!({ "ok": true, "command": COPY_TOOL, "exit_code": 0,
                    "data": { "copy": path, "source_sha256": source.sha256 } }),
                cancelled: false,
            }
        }
        Err(e) if ctx.cancel.load(std::sync::atomic::Ordering::SeqCst) => Outcome {
            cancelled: true,
            ..Outcome::failed(e)
        },
        Err(e) => Outcome::failed(e),
    }
}
