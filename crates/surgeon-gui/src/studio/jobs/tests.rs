//! The manager with a scripted executor (no GPU, no model), then once with the engine.

#[path = "../../../../../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

use std::sync::atomic::AtomicUsize;
use std::sync::Mutex;

use serde_json::json;

use super::*;
use crate::studio::exec::EngineExecutor;

/// `args.behave`: `ok`, `fail`, `block` (until cancelled or released), `write` (writes
/// `args.out`, reports it, succeeds).
#[derive(Default)]
struct Script {
    active: AtomicUsize,
    max_heavy: AtomicUsize,
    heavy: AtomicUsize,
    started: Mutex<Vec<String>>,
    release: AtomicBool,
}

impl Executor for Script {
    fn run(&self, spec: &JobSpec, args: &Value, ctx: &RunContext, rec: &mut Recorder) -> Outcome {
        self.started.lock().unwrap().push(spec.label.clone());
        self.active.fetch_add(1, Ordering::SeqCst);
        let heavy = is_heavy(spec);
        if heavy {
            let now = self.heavy.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_heavy.fetch_max(now, Ordering::SeqCst);
        }
        let ok = Outcome {
            exit_code: 0,
            envelope: json!({"ok": true, "exit_code": 0, "data": {"out": args["out"]}}),
            cancelled: false,
        };
        let outcome = match args["behave"].as_str().unwrap_or("ok") {
            "fail" => Outcome::failed("scripted failure"),
            "block" => {
                let mut i = 0;
                while !ctx.cancel.load(Ordering::SeqCst) && !self.release.load(Ordering::SeqCst) {
                    i += 1;
                    rec.emit(SurgeonEvent::Progress {
                        done: i,
                        total: None,
                        unit: "tick".into(),
                    });
                    std::thread::sleep(Duration::from_millis(2));
                }
                match ctx.cancel.load(Ordering::SeqCst) {
                    true => Outcome {
                        cancelled: true,
                        ..Outcome::failed("cancelled")
                    },
                    false => ok,
                }
            }
            "write" => {
                let out = PathBuf::from(args["out"].as_str().unwrap());
                std::fs::write(&out, b"result").unwrap();
                ok
            }
            _ => {
                std::thread::sleep(Duration::from_millis(5));
                ok
            }
        };
        if heavy {
            self.heavy.fetch_sub(1, Ordering::SeqCst);
        }
        self.active.fetch_sub(1, Ordering::SeqCst);
        outcome
    }
}

fn project() -> (tempfile::TempDir, PathBuf) {
    let ws = tempfile::tempdir().unwrap();
    let model = ws.path().join("source");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("model.safetensors"), b"weights").unwrap();
    let p = Project::create(ws.path(), "p", Some(&model)).unwrap();
    (ws, p.dir)
}

fn open(dir: &Path, exec: Arc<dyn Executor>) -> Manager {
    Manager::open(dir, exec, Arc::new(|| {})).unwrap()
}

fn spec(tool: &str, label: &str, args: Value) -> JobSpec {
    JobSpec {
        tool: tool.into(),
        label: label.into(),
        args,
        priority: 0,
        budgets: Budgets::default(),
        resumed_from: None,
        read_only: false,
    }
}

fn settle(m: &mut Manager) -> Vec<Ended> {
    m.wait(Duration::from_secs(30)).unwrap()
}

#[test]
fn heavy_jobs_run_one_at_a_time_by_priority_and_light_ones_alongside() {
    let (_ws, dir) = project();
    let script = Arc::new(Script::default());
    let mut m = open(&dir, script.clone());
    for (label, priority) in [("h-low", 0), ("h-high", 5), ("h-mid", 2)] {
        let mut s = spec("surgeon_graph", label, json!({}));
        s.priority = priority;
        m.submit(s).unwrap();
    }
    m.submit(spec("surgeon_check", "light", json!({}))).unwrap();
    let ended = settle(&mut m);
    assert_eq!(ended.len(), 4);
    assert!(ended.iter().all(|e| e.verdict == Verdict::Ok));
    assert_eq!(script.max_heavy.load(Ordering::SeqCst), 1);
    let order = script.started.lock().unwrap().clone();
    let heavy: Vec<&str> = order
        .iter()
        .map(String::as_str)
        .filter(|l| l.starts_with('h'))
        .collect();
    assert_eq!(heavy, ["h-high", "h-mid", "h-low"]);
    assert_eq!(m.project.manifest.state, ProjectState::Ready);
    let ledger = ledger::read(&m.project.ledger_path()).unwrap();
    assert_eq!(ledger.len(), 4);
    for e in &ledger {
        let view =
            RunView::replay(&m.project.runs_dir().join(&e.run_id).join(EVENTS_FILE)).unwrap();
        assert_eq!(view.end, Some(SurgeonEvent::Finished { exit_code: 0 }));
    }
}

#[test]
fn a_cancelled_run_leaves_no_output_and_the_project_ready() {
    let (_ws, dir) = project();
    let mut m = open(&dir, Arc::new(Script::default()));
    let id = m
        .submit(spec(
            "surgeon_graph",
            "b",
            json!({"behave": "block", "out": "{out}/x"}),
        ))
        .unwrap();
    while m.running.first().is_none_or(|r| r.view.progress.is_empty()) {
        m.tick(Instant::now());
        std::thread::sleep(bus::FRAME);
    }
    assert_eq!(m.project.manifest.state, ProjectState::Running);
    let run_dir = m.project.runs_dir().join(&id);
    assert!(run_dir.join("out.partial").is_dir());
    m.cancel(&id).unwrap();
    let ended = settle(&mut m);
    assert_eq!(ended[0].verdict, Verdict::Cancelled);
    assert_eq!(ended[0].view.end, Some(SurgeonEvent::Cancelled));
    assert!(ended[0].view.progress.contains_key("tick"));
    assert!(!run_dir.join("out.partial").exists() && !run_dir.join("out").exists());
    assert_eq!(read_status(&run_dir).unwrap().status, RunStatus::Cancelled);
    assert_eq!(m.project.manifest.state, ProjectState::Ready);
}

#[test]
fn an_estimate_runs_beside_a_heavy_run_but_a_pipeline_plan_waits() {
    let (_ws, dir) = project();
    let script = Arc::new(Script::default());
    let mut m = open(&dir, script.clone());
    let heavy = m
        .submit(spec("surgeon_graph", "h", json!({"behave": "block"})))
        .unwrap();
    let estimate = json!({"model": "copies/c", "estimate": true});
    m.submit(spec("surgeon_graph", "estimate", estimate))
        .unwrap();
    m.submit(spec("surgeon_pipeline", "plan", json!({"action": "plan"})))
        .unwrap();
    let mut done = Vec::new();
    while done.is_empty() {
        done.extend(m.tick(Instant::now()));
        std::thread::sleep(bus::FRAME);
    }
    assert_eq!(done[0].spec.label, "estimate", "the preview did not wait");
    assert_eq!(m.queue.len(), 1, "the plan waits for the heavy slot");
    script.release.store(true, Ordering::SeqCst);
    settle(&mut m);
    assert_eq!(script.max_heavy.load(Ordering::SeqCst), 1);
    // "h" and the estimate start in the same tick, on two threads: only the plan is ordered.
    let order = script.started.lock().unwrap().clone();
    assert_eq!(order.len(), 3);
    assert_eq!(order[2], "plan");
    let _ = heavy;
}

#[test]
fn a_queued_run_is_cancelled_without_starting() {
    let (_ws, dir) = project();
    let script = Arc::new(Script::default());
    let mut m = open(&dir, script.clone());
    let id = m.submit(spec("surgeon_graph", "never", json!({}))).unwrap();
    m.cancel(&id).unwrap();
    assert!(settle(&mut m).is_empty());
    assert!(script.started.lock().unwrap().is_empty());
    let ledger = ledger::read(&m.project.ledger_path()).unwrap();
    assert_eq!(ledger[0].verdict, Verdict::Cancelled);
}

#[test]
fn a_success_publishes_its_output_and_a_failure_needs_attention() {
    let (_ws, dir) = project();
    let mut m = open(&dir, Arc::new(Script::default()));
    let id = m
        .submit(spec(
            "surgeon_extract",
            "w",
            json!({"behave": "write", "out": "{out}/v.safetensors"}),
        ))
        .unwrap();
    let ended = settle(&mut m);
    let out = m
        .project
        .runs_dir()
        .join(&id)
        .join("out")
        .join("v.safetensors");
    assert_eq!(std::fs::read(&out).unwrap(), b"result");
    let reported = ended[0].envelope["data"]["out"].as_str().unwrap();
    assert_eq!(Path::new(reported), out, "repointed to out/");
    let params = read_params(&m.project.runs_dir().join(&id)).unwrap();
    assert_eq!(
        params.args["out"], "{out}/v.safetensors",
        "params keep the placeholder"
    );

    m.submit(spec(
        "surgeon_graph",
        "f",
        json!({"behave": "fail", "out": "{out}/x"}),
    ))
    .unwrap();
    let ended = settle(&mut m);
    assert_eq!(ended[0].verdict, Verdict::Failed);
    assert_eq!(m.project.manifest.state, ProjectState::NeedsAttention);
    m.acknowledge().unwrap();
    assert_eq!(m.project.manifest.state, ProjectState::Ready);
}

#[test]
fn the_wall_time_budget_stops_a_run_and_flags_the_project() {
    let (_ws, dir) = project();
    let mut m = open(&dir, Arc::new(Script::default()));
    let mut s = spec("surgeon_graph", "slow", json!({"behave": "block"}));
    s.budgets.max_wall_time_secs = Some(0);
    let id = m.submit(s).unwrap();
    m.tick(Instant::now());
    std::thread::sleep(Duration::from_millis(20));
    let ended = settle(&mut m);
    assert_eq!(ended[0].verdict, Verdict::Cancelled);
    let status = read_status(&m.project.runs_dir().join(&id)).unwrap();
    assert!(status.note.unwrap().contains("max_wall_time"));
    assert_eq!(m.project.manifest.state, ProjectState::NeedsAttention);
}

#[test]
fn a_crash_leaves_an_interrupted_project_that_resumes() {
    let (_ws, dir) = project();
    let script = Arc::new(Script::default());
    let mut m = open(&dir, script.clone());
    let id = m
        .submit(spec(
            "surgeon_graph",
            "long",
            json!({"behave": "block", "out": "{out}/x"}),
        ))
        .unwrap();
    m.submit(spec("surgeon_graph", "waiting", json!({})))
        .unwrap();
    m.tick(Instant::now());
    // The studio dies: nothing is closed, the lock (this pid) stays.
    std::mem::forget(m);

    let mut m = open(&dir, script.clone());
    assert_eq!(m.project.manifest.state, ProjectState::Interrupted);
    let resumable: Vec<String> = m.resumable().into_iter().map(|r| r.id).collect();
    assert_eq!(resumable.len(), 2, "the running and the queued run");
    assert!(!m.project.runs_dir().join(&id).join("out.partial").exists());
    let ledger = ledger::read(&m.project.ledger_path()).unwrap();
    assert!(ledger.iter().all(|e| e.verdict == Verdict::Interrupted));
    script.release.store(true, Ordering::SeqCst);

    let new = m.resume(&id).unwrap();
    assert!(m.resume(&id).is_err(), "resumed once");
    let ended = settle(&mut m);
    assert_eq!(ended[0].id, new);
    assert_eq!(ended[0].spec.resumed_from.as_deref(), Some(id.as_str()));
    assert_eq!(m.project.manifest.state, ProjectState::Ready);
    assert_eq!(m.resumable().len(), 1);
}

#[test]
fn the_project_lock_is_held_while_a_job_thread_still_runs() {
    let (_ws, dir) = project();
    let script = Arc::new(Script::default());
    let mut m = open(&dir, script.clone());
    m.submit(spec("surgeon_graph", "long", json!({"behave": "block"})))
        .unwrap();
    m.tick(Instant::now());
    let deadline = Instant::now() + Duration::from_secs(10);
    while script.active.load(Ordering::SeqCst) == 0 {
        assert!(Instant::now() < deadline, "the job never started");
        std::thread::sleep(Duration::from_millis(2));
    }
    // The manager goes (the window closes): its detached thread still writes in the project.
    drop(m);
    assert!(dir.join(LOCK).exists(), "held by the running job");
    script.release.store(true, Ordering::SeqCst);
    let deadline = Instant::now() + Duration::from_secs(10);
    while dir.join(LOCK).exists() {
        assert!(Instant::now() < deadline, "released once the job ends");
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn a_project_open_in_a_live_process_is_refused_and_a_dead_lock_taken_over() {
    let (_ws, dir) = project();
    let mut system = sysinfo::System::new().unwrap();
    system.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
    let other = system
        .processes()
        .keys()
        .map(|p| p.as_u32())
        .find(|p| *p != std::process::id() && *p != 0)
        .unwrap();
    std::fs::write(dir.join(LOCK), other.to_string()).unwrap();
    let e = Manager::open(&dir, Arc::new(Script::default()), Arc::new(|| {}))
        .err()
        .unwrap();
    assert!(e.contains("déjà ouvert"), "{e}");
    std::fs::write(dir.join(LOCK), (u32::MAX - 7).to_string()).unwrap();
    let m = open(&dir, Arc::new(Script::default()));
    drop(m);
    assert!(!dir.join(LOCK).exists(), "released on close");
}

#[test]
fn writes_to_the_source_model_are_refused_before_queuing() {
    let (_ws, dir) = project();
    let mut m = open(&dir, Arc::new(Script::default()));
    let source = m.project.manifest.source.clone().unwrap().path;
    let copies = m.project.copies_dir();
    std::fs::create_dir_all(copies.join("c1")).unwrap();
    let refused = [
        spec(
            "surgeon_apply",
            "src",
            json!({"model": source, "patch": "p.json"}),
        ),
        spec(
            "surgeon_apply",
            "rel",
            json!({"model": "copies", "patch": "p.json"}),
        ),
        spec(
            "surgeon_lora",
            "merge",
            json!({"mode": "merge", "model": source}),
        ),
        spec(
            "surgeon_extract",
            "out",
            json!({"model": "copies/c1", "out": source.join("v.safetensors")}),
        ),
        spec("frobnicate", "x", json!({})),
    ];
    for s in refused {
        assert!(m.submit(s.clone()).is_err(), "{}", s.label);
    }
    let allowed = [
        spec(
            "surgeon_apply",
            "copy",
            json!({"model": "copies/c1", "patch": "p.json"}),
        ),
        spec(
            "surgeon_apply",
            "dry",
            json!({"model": source, "patch": "p.json", "dry_run": true}),
        ),
        spec(
            "surgeon_lora",
            "extract",
            json!({"mode": "extract", "base": source}),
        ),
    ];
    for s in allowed {
        m.submit(s.clone())
            .unwrap_or_else(|e| panic!("{}: {e}", s.label));
    }
    assert!(
        m.history().iter().all(|r| r.spec.label != "src"),
        "no run dir for a refusal"
    );
}

#[test]
fn a_pipeline_over_the_candidate_budget_is_refused() {
    let (_ws, dir) = project();
    let mut m = open(&dir, Arc::new(Script::default()));
    std::fs::write(
        dir.join("pipeline.json"),
        r#"{"budgets":{"max_candidates":12}}"#,
    )
    .unwrap();
    let mut s = spec(
        "surgeon_pipeline",
        "p",
        json!({"action": "plan", "job_dir": "{out}/job", "config": "pipeline.json"}),
    );
    s.budgets.max_candidates = Some(8);
    assert!(m.submit(s.clone()).unwrap_err().contains("max_candidates"));
    s.budgets.max_candidates = Some(12);
    m.submit(s).unwrap();
}

#[test]
fn the_engine_runs_a_working_copy_and_an_inspection_without_touching_the_source() {
    let ws = tempfile::tempdir().unwrap();
    let model = fake_qwen2::write_fake_qwen2(ws.path()).unwrap();
    let before = weight_surgeon::pipeline::job::sha256_dir(&model).unwrap();
    let p = Project::create(ws.path(), "real", Some(&model)).unwrap();
    let mut m = open(&p.dir, Arc::new(EngineExecutor::default()));
    m.submit(spec(COPY_TOOL, "copie", json!({"id": "c1"})))
        .unwrap();
    let ended = settle(&mut m);
    assert_eq!(ended[0].verdict, Verdict::Ok, "{}", ended[0].envelope);
    m.submit(spec(
        "surgeon_inspect",
        "inspect",
        json!({"model": "copies/c1"}),
    ))
    .unwrap();
    let ended = settle(&mut m);
    assert_eq!(ended[0].verdict, Verdict::Ok, "{}", ended[0].envelope);
    assert_eq!(ended[0].envelope["command"], "inspect");
    assert_eq!(
        ended[0].view.command.as_deref(),
        Some("surgeon_inspect"),
        "first line"
    );
    let copied = weight_surgeon::pipeline::job::sha256_dir(&p.copies_dir().join("c1")).unwrap();
    assert_eq!(copied, before);
    assert_eq!(
        weight_surgeon::pipeline::job::sha256_dir(&model).unwrap(),
        before
    );
    let ledger = ledger::read(&m.project.ledger_path()).unwrap();
    assert_eq!(ledger.len(), 2);
    assert!(ledger[1]
        .hashes
        .model
        .as_deref()
        .unwrap()
        .starts_with("fast-"));
    assert!(ledger[1].hashes.binary.is_some());
}
