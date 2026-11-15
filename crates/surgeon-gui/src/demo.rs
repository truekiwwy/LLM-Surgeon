//! `--demo-data`: a throwaway workspace that fills every view with real state, for the
//! captures, the rendering tests and a first look. Nothing is invented on screen: a tiny
//! model is written, copied and analysed by the real engine (ledger, events, activity), the
//! graph is the test fixture (a qwen2-mini with its baseline diff), and the agent is a
//! scripted provider that inspects the copy, records a finding tied to its run, then asks to
//! apply a patch, which waits on an approval card. No network, no key.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use eframe::egui;
use serde_json::json;
use weight_surgeon::graph::stream::GraphPatch;

use crate::agent::config::{Connection, Consent, ProviderKind};
use crate::agent::mock::{Mock, Step};
use crate::agent::policy::AgentMode;
use crate::agent::provider::{Block, Request, StopReason, Turn, Usage};
use crate::agent::session::{Conversation, Dirs, Session, Settings};
use crate::agent::vault::SessionStore;
use crate::agent::view::Hub;
use crate::app::SurgeonApp;
use crate::source::Input;
use crate::studio::exec::EngineExecutor;
use crate::studio::forms;
use crate::studio::jobs::{Budgets, JobSpec, Manager, COPY_TOOL};
use crate::studio::project::{now_unix, Project};
use crate::studio::view::{default_values, Studio};

const FIXTURE: &str = include_str!("../tests/fixtures/qwen2-mini.ndjson");
const WAIT: Duration = Duration::from_secs(120);

/// A fresh demo folder: one per call, so that tests in one process do not share it.
fn root() -> PathBuf {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("surgeon-gui-demo-{}-{n}", std::process::id()))
}

/// Builds a demo workspace in the temporary folder and puts it in `app`.
pub fn install(ctx: &egui::Context, app: &mut SurgeonApp) -> Result<(), String> {
    let root = root();
    let _ = std::fs::remove_dir_all(&root);
    let workspace = root.join("projects");
    let project = build_project(&root, &workspace)?;
    graph(ctx, app)?;

    app.studio = Studio::new(workspace);
    app.studio.open_project(ctx, &project);
    let manager = app
        .studio
        .manager
        .as_mut()
        .ok_or("projet de démo non ouvert")?;
    let inspect = manager
        .history()
        .into_iter()
        .rev()
        .find(|r| r.spec.tool == "surgeon_inspect")
        .map(|r| r.id);
    app.studio.followed = inspect;
    // The first-run backend dialog would cover the demo.
    app.dialog.open = false;

    app.agents = agents(&root, &project)?;
    let session = session(app, &project)?;
    app.agents.session = Some(session);
    Ok(())
}

/// A tiny model, its project, a working copy, a check and an inspection of the copy.
fn build_project(root: &Path, workspace: &Path) -> Result<PathBuf, String> {
    let model = root.join("model");
    crate::studio::selftest::write_model(&model)?;
    std::fs::create_dir_all(workspace).map_err(|e| e.to_string())?;
    let project = Project::create(workspace, "demo-qwen2", Some(&model))?;
    let mut m = Manager::open(
        &project.dir,
        Arc::new(EngineExecutor::default()),
        Arc::new(|| {}),
    )?;
    let spec = |tool: &str, label: &str, args| JobSpec {
        tool: tool.into(),
        label: label.into(),
        args,
        priority: 0,
        budgets: Budgets::default(),
        resumed_from: None,
        read_only: false,
    };
    m.submit(spec(COPY_TOOL, "copie de travail", json!({"id": "c1"})))?;
    m.wait(WAIT)?;
    let all = forms::forms();
    for (tool, label) in [
        ("surgeon_check", "contrôle c1"),
        ("surgeon_inspect", "inspection c1"),
    ] {
        let form = all.iter().find(|f| f.tool == tool).ok_or(tool)?;
        let args = form.args(&default_values(form, &m.project))?;
        m.submit(spec(tool, label, args))?;
        m.wait(WAIT)?;
    }
    Ok(project.dir)
}

/// The fixture graph, its changed tensor selected.
fn graph(ctx: &egui::Context, app: &mut SurgeonApp) -> Result<(), String> {
    let GraphPatch::Snapshot { report } =
        serde_json::from_str(FIXTURE.trim()).map_err(|e| e.to_string())?
    else {
        return Err("fixture : instantané attendu".into());
    };
    app.set_input(ctx, Input::Report(report));
    let step = app.timeline.current().ok_or("graphe vide")?;
    let changed = step.report.nodes.iter().find(|n| n.id == "L1.attn_out");
    if let Some(node) = changed {
        app.open_layers.insert(node.layer);
        app.selected = Some(node.id.clone());
    }
    app.diff = true;
    Ok(())
}

/// Two connections (a local endpoint, consented; a cloud one without key), the local one
/// used by the project in `Modifier` mode.
fn agents(root: &Path, project: &Path) -> Result<Hub, String> {
    let mut hub = Hub::new(
        root.join("agents.json"),
        Box::new(SessionStore::default()),
        Arc::new(|| {}),
    );
    hub.cfg.connections = vec![
        Connection {
            id: "ollama".into(),
            kind: ProviderKind::Compatible,
            label: "Ollama (local)".into(),
            base_url: "http://127.0.0.1:11434/v1".into(),
            models: vec!["qwen2.5:7b".into(), "llama3.1:8b".into()],
            default_model: None,
            verified: false,
            prices: Default::default(),
        },
        Connection {
            id: "anthropic".into(),
            kind: ProviderKind::Anthropic,
            label: "Anthropic".into(),
            base_url: String::new(),
            models: Vec::new(),
            default_model: None,
            verified: false,
            prices: Default::default(),
        },
    ];
    hub.cfg.consents.push(Consent {
        connection: "ollama".into(),
        accepted_unix: now_unix(),
        redact_paths: true,
    });
    hub.save_cfg();
    hub.selected = Some("ollama".into());
    hub.project = Some(project.to_path_buf());
    hub.agent.connection = Some("ollama".into());
    hub.agent.model = Some("qwen2.5:7b".into());
    hub.agent.mode = AgentMode::Modify;
    hub.save_agent();
    Ok(hub)
}

fn turn(blocks: Vec<Block>) -> Result<Turn, crate::agent::provider::ProviderError> {
    Ok(Turn {
        blocks,
        usage: Usage {
            input_tokens: 1200,
            output_tokens: 180,
        },
        stop: StopReason::ToolUse,
    })
}

/// The scripted session, run until its apply waits on the approval card.
fn session(app: &mut SurgeonApp, project: &Path) -> Result<Session, String> {
    let inspect = Step::Reply(Box::new(|_: &Request| {
        turn(vec![
            Block::Text {
                text: "Je lis d'abord la copie c1 (lecture seule).".into(),
            },
            Block::ToolUse {
                id: "a".into(),
                name: "surgeon_inspect".into(),
                input: json!({"model": "copies/c1"}),
            },
        ])
    }));
    let finding = Step::Reply(Box::new(|req: &Request| {
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
        turn(vec![Block::ToolUse {
            id: "f".into(),
            name: "record_finding".into(),
            input: json!({"text": "la copie c1 a une couche MLP", "evidence": [{"run_id": run, "value": 1}]}),
        }])
    }));
    let apply = Step::Reply(Box::new(|_: &Request| {
        turn(vec![
            Block::Text {
                text: "Je propose d'appliquer patch.json sur la copie c1. Rien ne part sans votre accord.".into(),
            },
            Block::ToolUse {
                id: "b".into(),
                name: "surgeon_apply".into(),
                input: json!({"model": "copies/c1", "patch": "patch.json"}),
            },
        ])
    }));
    let mock = Arc::new(Mock::new(vec![inspect, finding, apply]));
    let dirs = Dirs::of(project);
    let conv = Conversation {
        id: Session::new_id(&dirs),
        created_unix: now_unix(),
        connection: "ollama".into(),
        model: "qwen2.5:7b".into(),
        ..Conversation::default()
    };
    let settings = Settings {
        mode: AgentMode::Modify,
        ..Settings::default()
    };
    let mut s = Session::new(conv, mock, settings, Vec::new(), dirs, Arc::new(|| {}));
    s.consented = true;
    s.send("Audite la copie c1 et propose une correction si besoin.")?;
    let m = app
        .studio
        .manager
        .as_mut()
        .ok_or("projet de démo non ouvert")?;
    let deadline = Instant::now() + WAIT;
    while s.cards.is_empty() {
        s.tick(m);
        for e in m.tick(Instant::now()) {
            s.on_ended(&e);
        }
        if let Some(e) = &s.error {
            return Err(format!("agent de démo : {e}"));
        }
        if Instant::now() > deadline {
            return Err("agent de démo : pas de carte d'approbation".into());
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    Ok(s)
}
