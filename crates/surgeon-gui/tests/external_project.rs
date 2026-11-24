//! A project folder made only by an external agent (Claude Code, Codex... with the
//! `weight-surgeon` skill) through `surgeon mcp`: the manifest `surgeon_workspace` writes with
//! the origin model, a working copy under `copies/`, working files and a patch under `work/`,
//! the runs the server journals in `chat/mcp_runs.jsonl` and the findings it records. The
//! studio opens it as one of its own: listed, opened with its origin recognised and locked,
//! its runs in the results, the experiments summary, and the evidence log with its
//! "Heuristique" and "Mesuré" badges on screen.

#[path = "../../../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use eframe::egui;
use serde_json::{json, Value};
use surgeon_gui::actions::Env;
use surgeon_gui::agent::evidence::{self, Status};
use surgeon_gui::agent::paths::ProjectPaths;
use surgeon_gui::agent::session::Dirs;
use surgeon_gui::agent::{experiments, vault::SessionStore, view::Hub};
use surgeon_gui::app::{SurgeonApp, Tab};
use surgeon_gui::desktop::Desktop;
use surgeon_gui::settings::{Lang, Settings};
use surgeon_gui::studio::project::{self, Project};
use surgeon_gui::studio::{exec::EngineExecutor, jobs::Manager, results};
use surgeon_gui::{raster, studio::ledger::Verdict};
use weight_surgeon::app::{self, AppDeps};
use weight_surgeon::mcp::McpConfig;

fn deps() -> AppDeps {
    AppDeps {
        probe: Box::new(weight_surgeon::hardware::SystemProbe::detect()),
        runner: Box::new(weight_surgeon::eval::SystemCommandRunner),
        generator_factory: Box::new(app::real_generator),
        interrupted: Arc::new(AtomicBool::new(false)),
        ui: weight_surgeon::ui::UiConfig {
            out: weight_surgeon::ui::ColorMode::Off,
            err: weight_surgeon::ui::ColorMode::Off,
            err_is_tty: false,
        },
    }
}

/// One client session of the engine's MCP server on `cfg`, as a harness drives it: the
/// handshake, then `calls` in order; the `structuredContent` of each answer, each one a success.
fn session(cfg: &McpConfig, calls: &[(&str, Value)]) -> Vec<Value> {
    let contents = answers(cfg, calls);
    for (content, (tool, _)) in contents.iter().zip(calls) {
        assert_eq!(content["ok"], true, "{tool}: {content}");
    }
    contents
}

/// [`session`], failures included.
fn answers(cfg: &McpConfig, calls: &[(&str, Value)]) -> Vec<Value> {
    let mut lines = vec![
        json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "external-harness", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
    ];
    for (i, (tool, arguments)) in calls.iter().enumerate() {
        lines.push(
            json!({"jsonrpc": "2.0", "id": i + 1, "method": "tools/call",
            "params": {"name": tool, "arguments": arguments}}),
        );
    }
    let input: String = lines.iter().map(|l| format!("{l}\n")).collect();
    let mut out = Vec::new();
    let code = weight_surgeon::mcp::serve(
        Cursor::new(input.into_bytes()),
        &mut out,
        &mut std::io::sink(),
        &deps(),
        cfg,
    );
    assert_eq!(code, 0);
    let replies: Vec<Value> = String::from_utf8(out)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    (1..=calls.len())
        .map(|id| {
            let reply = replies.iter().find(|r| r["id"] == id).unwrap();
            reply["result"]["structuredContent"].clone()
        })
        .collect()
}

fn copy_dir(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, to.join(path.file_name().unwrap())).unwrap();
    }
}

/// What the external agent leaves: `(workspace, project, origin)`.
fn external_project(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let origin = fake_qwen2::write_fake_qwen2(&root.join("models")).unwrap();
    let workspace = root.join("projects");
    let project = workspace.join("qwen-refus");
    std::fs::create_dir_all(&project).unwrap();
    let (prompts, _) = fake_qwen2::write_prompt_files(root).unwrap();
    let prompts = std::fs::read_to_string(prompts).unwrap();

    // `surgeon mcp` with no --allow-dir serves the harness's folder; the origin is read only.
    let cfg = McpConfig::new(std::slice::from_ref(&project), false, false)
        .unwrap()
        .with_read_roots(std::slice::from_ref(&origin))
        .unwrap();
    // The skill's first call: the folder becomes a project with its origin recorded.
    let init = json!({"action": "init", "model": origin, "name": "qwen-refus"});
    let made = session(&cfg, &[("surgeon_workspace", init.clone())]);
    assert_eq!(made[0]["data"]["replaced"], false);
    // Its own shell tool: the working copy.
    copy_dir(&origin, &project.join("copies").join("copy-1"));
    let patch = json!({"schema_version": 1, "operations": [{"id": "po",
        "operation": "project_out", "projections": ["o_proj", "down_proj"], "layers": "all",
        "vector_file": "work/dir_L2.safetensors", "alpha": 0.6}]});
    let first = session(
        &cfg,
        &[
            ("surgeon_doctor", json!({"model": "copies/copy-1"})),
            (
                "surgeon_files",
                json!({"action": "write", "path": "prompts.json", "content": prompts}),
            ),
            (
                "surgeon_bench",
                json!({"mode": "refusal_probe", "model": "copies/copy-1",
                       "prompts": "work/prompts.json", "probe_method": "first-token-prob"}),
            ),
            (
                "surgeon_bench",
                json!({"mode": "refusal", "model": "copies/copy-1",
                       "prompts": "work/prompts.json", "baseline_model": origin,
                       "max_new_tokens": 6}),
            ),
            (
                "surgeon_files",
                json!({"action": "write", "path": "patch_po.json",
                       "content": patch.to_string()}),
            ),
        ],
    );
    let (probe, refusal) = (&first[2], &first[3]);
    assert_eq!(probe["data"]["is_heuristic"], true);
    let probed = probe["data"]["probe_refusal_rate"].as_f64().unwrap();
    let measured = refusal["data"]["refusal_rate"].as_f64().unwrap();
    // A later turn of the agent: the findings cite the run ids the server returned.
    session(
        &cfg,
        &[
            (
                "record_finding",
                json!({"text": format!("refus sondé {probed}"), "evidence": [{
                    "run_id": probe["run_id"], "metric": "probe_refusal_rate", "value": probed}]}),
            ),
            (
                "record_finding",
                json!({"text": format!("refus {measured} en génération"), "evidence": [{
                    "run_id": refusal["run_id"], "metric": "refusal_rate", "value": measured}]}),
            ),
            (
                "update_plan",
                json!({"steps": ["Diagnostic", "Crible", "Certification"]}),
            ),
        ],
    );
    // A second init never replaces the project by accident.
    let again = answers(&cfg, &[("surgeon_workspace", init)]);
    assert_eq!(again[0]["error"]["kind"], "output_exists", "{}", again[0]);
    (workspace, project, origin)
}

#[test]
fn a_project_made_only_through_mcp_opens_in_the_studio() {
    // The test never reads the user's configuration nor writes the user's reference cache.
    std::env::set_var(weight_surgeon::backend::config::NO_USER_CONFIG_ENV, "1");
    let root = tempfile::tempdir().unwrap();
    let (workspace, project, origin) = external_project(root.path());

    // Listed and opened like a project of the studio, its origin the one the studio records.
    let listed = Project::list(&workspace);
    assert_eq!(listed.len(), 1);
    let manifest = listed[0].1.as_ref().unwrap();
    assert_eq!(manifest.name, "qwen-refus");
    let source = manifest.source.clone().expect("the origin is recorded");
    assert_eq!(
        source,
        project::source_model(&origin).unwrap(),
        "path, SHA-256, size"
    );
    let manager = Manager::open(
        &project,
        Arc::new(EngineExecutor::default()),
        Arc::new(|| {}),
    )
    .unwrap();
    assert!(
        manager.history().is_empty(),
        "no studio job, nothing missing"
    );
    drop(manager);
    let paths = ProjectPaths::of(&project);
    assert_eq!(paths.copy, Some(project.join("copies").join("copy-1")));
    assert_eq!(paths.origin, Some(source.path.clone()));
    assert!(project.join("work").join("patch_po.json").is_file());

    // The server's journal: every engine call, the working files aside, the refused init too.
    let dirs = Dirs::of(&project);
    let runs = evidence::mcp_runs(&dirs.mcp_runs());
    let tools: Vec<&str> = runs.iter().map(|r| r.tool.as_str()).collect();
    assert_eq!(
        tools,
        [
            "surgeon_workspace",
            "surgeon_doctor",
            "surgeon_bench",
            "surgeon_bench",
            "surgeon_workspace"
        ]
    );
    let verdicts: Vec<Verdict> = runs.iter().map(|r| r.verdict).collect();
    assert_eq!(&verdicts[..4], [Verdict::Ok; 4]);
    assert_eq!(verdicts[4], Verdict::Failed);
    let listed = results::runs(&runs);
    assert_eq!(listed.len(), 2, "both measures in the results");
    assert!(
        listed.iter().any(|r| r.bench.is_some()),
        "the refusal report"
    );

    // The evidence: the probe is a heuristic, the full generation a measure.
    let findings = evidence::read(&dirs.findings());
    let statuses: Vec<&str> = findings.iter().map(|f| f.status(&runs).label()).collect();
    assert_eq!(statuses, ["Heuristique", "Mesuré"]);
    assert!(matches!(findings[0].status(&runs), Status::Heuristic(_)));

    // The experiments summary the studio's own agent reads when the project goes on there.
    let summary = experiments::project_summary(&project);
    for needle in [
        "sonde refusal_probe sur copy-1",
        "(heuristique)",
        "bench refusal sur copy-1 : refus",
    ] {
        assert!(summary.contains(needle), "{needle} in {summary}");
    }

    // On screen: the project open in the window, its findings with their badges.
    let ctx = egui::Context::default();
    let env = Env {
        components: root.path().join("components"),
        config: None,
        settings: root.path().join("settings.json"),
    };
    let settings = Settings {
        language: Some(Lang::Fr),
        wizard_done: true,
        ..Default::default()
    };
    let desktop = Desktop::detached(None, settings, env);
    let mut app = SurgeonApp::with_shell(&ctx, None, desktop, workspace);
    app.dialog.read_only();
    app.agents = Hub::new(
        root.path().join("agents.json"),
        Box::new(SessionStore::default()),
        Arc::new(|| {}),
    );
    app.studio.open_project(&ctx, &project);
    app.tab = Tab::Agent;
    let mut texts = Vec::new();
    for _ in 0..3 {
        texts = raster::render(&ctx, [1440, 900], 1.0, |ui| app.draw(ui)).texts;
    }
    assert_eq!(app.agents.project.as_deref(), Some(project.as_path()));
    for badge in ["Heuristique (sonde, à confirmer par génération)", "Mesuré"] {
        assert!(texts.iter().any(|t| t == badge), "{badge}: {texts:?}");
    }
    assert_eq!(app.results().len(), 2);
}
