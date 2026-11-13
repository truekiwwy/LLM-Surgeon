//! `surgeon-gui --self-test`: the studio foundation end to end, without a window, on a
//! throwaway workspace in the temporary folder: a project over a tiny SafeTensors model, a
//! working copy, a real engine job, the refusal of an in-place write on the source, the
//! replay of every run, the ledger and the generated forms. Prints one line per check.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use weight_surgeon::pipeline::job::sha256_dir;

use super::bus::{RunView, EVENTS_FILE};
use super::exec::EngineExecutor;
use super::jobs::{Budgets, JobSpec, Manager, COPY_TOOL};
use super::ledger::{self, Verdict};
use super::project::Project;

pub fn run(out: &mut dyn Write) -> Result<(), String> {
    let root = std::env::temp_dir().join(format!("surgeon-gui-self-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let result = checks(&root, out);
    let _ = std::fs::remove_dir_all(&root);
    result
}

fn ok(out: &mut dyn Write, what: &str) {
    let _ = writeln!(out, "ok   {what}");
}

fn spec(tool: &str, args: serde_json::Value) -> JobSpec {
    JobSpec {
        tool: tool.into(),
        label: format!("self-test {tool}"),
        args,
        priority: 0,
        budgets: Budgets::default(),
        resumed_from: None,
        read_only: false,
    }
}

/// One F32 tensor `[2, 2]` named like an MLP writer, in a model folder.
pub(crate) fn write_model(dir: &Path) -> Result<(), String> {
    let mut header =
        br#"{"model.layers.0.mlp.down_proj.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#
            .to_vec();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend([0u8; 16]);
    std::fs::create_dir_all(dir)
        .and_then(|()| std::fs::write(dir.join("model.safetensors"), bytes))
        .map_err(|e| format!("{} : {e}", dir.display()))
}

fn checks(root: &Path, out: &mut dyn Write) -> Result<(), String> {
    let model = root.join("model");
    write_model(&model)?;
    let before = sha256_dir(&model).map_err(|e| e.to_string())?;
    let project = Project::create(root, "self-test", Some(&model))?;
    ok(out, "projet créé, modèle d'origine haché");

    let mut m = Manager::open(
        &project.dir,
        Arc::new(EngineExecutor::default()),
        Arc::new(|| {}),
    )?;
    // The check reads the copy: it is submitted once the copy has ended, not alongside.
    m.submit(spec(COPY_TOOL, json!({"id": "c1"})))?;
    let mut ended = m.wait(Duration::from_secs(120))?;
    m.submit(spec("surgeon_check", json!({"model": "copies/c1"})))?;
    ended.extend(m.wait(Duration::from_secs(120))?);
    for e in &ended {
        if e.verdict != Verdict::Ok {
            return Err(format!("{} : {:?} {}", e.spec.tool, e.verdict, e.envelope));
        }
    }
    ok(out, "copie de travail puis surgeon_check par le moteur");

    let refused = m.submit(spec(
        "surgeon_apply",
        json!({"model": model, "patch": "patch.json"}),
    ));
    if refused.is_ok() {
        return Err("un apply sur le modèle d'origine a été accepté".into());
    }
    if sha256_dir(&model).map_err(|e| e.to_string())? != before {
        return Err("le modèle d'origine a changé".into());
    }
    ok(out, "écriture sur l'origine refusée, empreinte inchangée");

    for e in &ended {
        let file = m.project.runs_dir().join(&e.id).join(EVENTS_FILE);
        if RunView::replay(&file)? != e.view || e.view.end.is_none() {
            return Err(format!("{} : rejeu différent", e.id));
        }
    }
    ok(out, "rejeu des events.ndjson identique");

    let entries = ledger::read(&m.project.ledger_path())?;
    if entries.len() != ended.len() || entries.iter().any(|e| e.hashes.binary.is_none()) {
        return Err(format!("journal : {} entrée(s)", entries.len()));
    }
    ok(out, "journal d'expériences complet");

    let tools = weight_surgeon::mcp::tool_definitions();
    let forms = super::forms::forms();
    if Some(forms.len()) != tools.as_array().map(Vec::len) {
        return Err("formulaires et outils du moteur divergent".into());
    }
    ok(
        out,
        &format!(
            "{} formulaires générés depuis les schémas du moteur",
            forms.len()
        ),
    );
    Ok(())
}
