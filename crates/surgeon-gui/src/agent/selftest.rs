//! `--self-test`, agent part: a scripted dialogue (no network) against the real engine on a
//! throwaway project. The agent inspects a working copy in "Ask" mode, is refused an apply,
//! records a finding linked to its run, and the source model is untouched.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use weight_surgeon::pipeline::job::sha256_dir;

use super::evidence::{self, Status};
use super::mock::{self, Mock, Step};
use super::policy::AgentMode;
use super::provider::{Block, Request, StopReason, Turn, Usage};
use super::session::{Conversation, Dirs, Session, Settings};
use crate::studio::exec::EngineExecutor;
use crate::studio::jobs::{Budgets, JobSpec, Manager, COPY_TOOL};
use crate::studio::ledger;
use crate::studio::project::Project;

pub fn run() -> Result<String, String> {
    let root = std::env::temp_dir().join(format!("surgeon-gui-agent-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    let result = checks(&root);
    let _ = std::fs::remove_dir_all(&root);
    result
}

fn checks(root: &Path) -> Result<String, String> {
    let model = root.join("model");
    crate::studio::selftest::write_model(&model)?;
    let before = sha256_dir(&model).map_err(|e| e.to_string())?;
    let project = Project::create(root, "agent", Some(&model))?;
    let mut m = Manager::open(
        &project.dir,
        Arc::new(EngineExecutor::default()),
        Arc::new(|| {}),
    )?;
    m.submit(JobSpec {
        tool: COPY_TOOL.into(),
        label: "copie".into(),
        args: json!({"id": "c1"}),
        priority: 0,
        budgets: Budgets::default(),
        resumed_from: None,
        read_only: false,
    })?;
    m.wait(Duration::from_secs(120))?;

    let record = Step::Reply(Box::new(|req: &Request| {
        let run = req
            .messages
            .last()
            .into_iter()
            .flat_map(|m| &m.blocks)
            .find_map(|b| match b {
                Block::ToolResult { content, .. } => content.split("run=\"").nth(1),
                _ => None,
            })
            .and_then(|r| r.split('"').next())
            .unwrap_or_default()
            .to_owned();
        Ok(Turn {
            blocks: vec![Block::ToolUse {
                id: "f".into(),
                name: "record_finding".into(),
                input: json!({"text": "la copie a 1 couche MLP", "evidence": [{"run_id": run, "value": 1}]}),
            }],
            usage: Usage::default(),
            stop: StopReason::ToolUse,
        })
    }));
    let mock = Arc::new(Mock::new(vec![
        mock::call(vec![
            ("a", "surgeon_inspect", json!({"model": "copies/c1"})),
            (
                "b",
                "surgeon_apply",
                json!({"model": "copies/c1", "patch": "patch.json"}),
            ),
        ]),
        record,
        mock::say("fin"),
    ]));
    let dirs = Dirs::of(&m.project.dir);
    let conv = Conversation {
        id: Session::new_id(&dirs),
        model: "mock-1".into(),
        connection: "self-test".into(),
        ..Conversation::default()
    };
    let settings = Settings {
        mode: AgentMode::Ask,
        ..Settings::default()
    };
    let mut s = Session::new(
        conv,
        mock.clone(),
        settings,
        Vec::new(),
        dirs.clone(),
        Arc::new(|| {}),
    );
    s.consented = true;
    s.send("inspecte la copie")?;
    let deadline = Instant::now() + Duration::from_secs(120);
    while s.busy() || m.busy() {
        s.tick(&mut m);
        for e in m.tick(Instant::now()) {
            s.on_ended(&e);
        }
        if Instant::now() > deadline {
            return Err("l'agent n'a pas terminé".into());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    if let Some(e) = &s.error {
        return Err(format!("agent : {e}"));
    }
    let runs: Vec<String> = m.history().into_iter().map(|r| r.spec.tool).collect();
    if runs != [COPY_TOOL, "surgeon_inspect"] {
        return Err(format!("runs inattendus : {runs:?}"));
    }
    let refused = mock.requests()[1].messages.last().is_some_and(|msg| {
        msg.blocks.iter().any(|b| {
            matches!(b, Block::ToolResult { content, is_error: true, .. } if content.contains("interdit"))
        })
    });
    if !refused {
        return Err("l'apply du mode Demander n'a pas été refusé".into());
    }
    let ledger = ledger::read(&m.project.ledger_path())?;
    let findings = evidence::read(&dirs.findings());
    let status = findings.first().map(|f| f.status(&ledger));
    if !matches!(status, Some(Status::Measured(_))) {
        return Err(format!(
            "constat non mesuré : {status:?} ({})",
            inspect_result(&ledger)
        ));
    }
    if sha256_dir(&model).map_err(|e| e.to_string())? != before {
        return Err("le modèle d'origine a changé".into());
    }
    Ok("dialogue scripté : inspect lu, apply refusé, constat mesuré, origine intacte".into())
}

fn inspect_result(ledger: &[ledger::Entry]) -> Value {
    ledger
        .iter()
        .find(|e| e.tool == "surgeon_inspect")
        .map(|e| e.result.clone())
        .unwrap_or_default()
}
