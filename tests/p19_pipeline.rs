//! P19: `surgeon pipeline` and the MCP tool `surgeon_pipeline`, end to end with the real
//! binary on the mini Qwen2 of `examples/make_fake_qwen2.rs`: a full decision (Dev, then one
//! Test consultation), early stopping, power, the Test Set registry, resume after a kill,
//! cancellation, the safety gate, and the asynchronous MCP flow.
//!
//! The mini model has random weights and never refuses: its refusal and GSM8K outcomes are
//! all equal before and after the patch, so the gates' verdicts depend on the sample sizes
//! and margins chosen here (known truth), the KL drift on the patch strength.

#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

#[path = "support/fake_server.rs"]
mod support;

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

const WORDS: [&str; 18] = [
    "joie", "fete", "soleil", "danse", "rire", "musique", "ami", "lumiere", "chaleur", "pluie",
    "brume", "nuit", "silence", "froid", "gris", "orage", "vide", "hiver",
];

fn prompts(n: usize, offset: usize) -> Value {
    let det = ["le", "la", "un", "une"];
    json!((0..n)
        .map(|i| format!(
            "{} {} est {} et {}",
            det[(i + offset) % 4],
            WORDS[(i * 5 + offset) % 18],
            WORDS[(i * 7 + offset + 3) % 18],
            WORDS[(i * 11 + offset) % 18]
        ))
        .collect::<Vec<_>>())
}

fn gsm(offset: usize) -> Value {
    json!((0..8)
        .map(|i| json!({
            "id": format!("q{i}"),
            "question": format!("{} et {}", WORDS[(i + offset) % 18], WORDS[(i * 3 + offset) % 18]),
            "answer": format!("#### {}", i + offset),
        }))
        .collect::<Vec<_>>())
}

struct Ws {
    dir: tempfile::TempDir,
}

impl Ws {
    /// The mini model, a direction, Dev and Test sets of every kind, `pipeline.json`.
    /// `unblock_dev` prompts in the Dev set of the first generation pillar (a long one makes
    /// the run last long enough to interrupt).
    fn new(unblock_dev: usize) -> Self {
        let dir = tempfile::tempdir().unwrap();
        let ws = Self { dir };
        fake_qwen2::write_fake_qwen2(ws.dir.path()).unwrap();
        fake_qwen2::write_prompt_files(ws.dir.path()).unwrap();
        let (code, out) = ws.surgeon(&[
            "extract",
            "--model",
            "qwen2-mini",
            "--pos-prompts",
            "pos.json",
            "--neg-prompts",
            "neg.json",
            "--layer",
            "1",
            "--out",
            "v.safetensors",
        ]);
        assert_eq!(code, 0, "{out}");
        for (name, value) in [
            ("neutral_dev.json", prompts(10, 4)),
            ("neutral_test.json", prompts(10, 5)),
            ("gsm_dev.json", gsm(0)),
            ("gsm_test.json", gsm(9)),
            ("unblock_dev.json", prompts(unblock_dev, 2)),
            ("unblock_test.json", prompts(12, 3)),
            ("safety_dev.json", prompts(12, 0)),
            ("safety_test.json", prompts(12, 1)),
        ] {
            ws.write(name, &value);
        }
        ws.write("pipeline.json", &Self::config());
        ws
    }

    fn config() -> Value {
        json!({
            "schema_version": 1,
            "model": "qwen2-mini",
            "patch": {"schema_version": 1, "description": "ablation", "backup": false, "operations": [{
                "id": "po", "operation": "project_out", "alpha": 1.0, "layers": "all",
                "projections": ["down_proj"], "vector_file": "v.safetensors",
                "vector_tensor": "direction", "vector_tensor_b": "direction"}]},
            "candidates": {"alphas": [0.5, 1.0]},
            "pillars": [
                {"id": "kl", "kind": "kl", "dataset_dev": "neutral_dev.json",
                 "dataset_test": "neutral_test.json", "max_tokens": 16,
                 "gate": {"max_drift": {"value": 5.0, "confidence": 0.9}}},
                {"id": "gsm", "kind": "gsm8k", "dataset_dev": "gsm_dev.json",
                 "dataset_test": "gsm_test.json", "max_new_tokens": 6,
                 "gate": {"non_inferiority": {"margin": 0.3, "confidence": 0.9}}},
                {"id": "unblock", "kind": "unblocking", "dataset_dev": "unblock_dev.json",
                 "dataset_test": "unblock_test.json", "max_new_tokens": 6,
                 "gate": {"non_inferiority": {"margin": 0.3, "confidence": 0.9}}},
                {"id": "safety", "kind": "safety", "dataset_dev": "safety_dev.json",
                 "dataset_test": "safety_test.json", "max_new_tokens": 6,
                 "gate": {"non_inferiority": {"margin": 0.3, "confidence": 0.9}}}
            ],
            "selection": {"pillar": "unblock"},
            "budgets": {"max_wall_time_secs": 3600, "max_candidates": 4},
            "seed": 7,
            "working_copy": "copy"
        })
    }

    fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    fn write(&self, name: &str, value: &Value) {
        std::fs::write(self.path(name), value.to_string()).unwrap();
    }

    fn read(&self, name: &str) -> Value {
        serde_json::from_str(&std::fs::read_to_string(self.path(name)).unwrap()).unwrap()
    }

    fn command(&self) -> Command {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_surgeon"));
        cmd.current_dir(self.dir.path()).env("NO_COLOR", "1");
        cmd
    }

    /// Exit code and envelope of `surgeon --json <args>` (strict JSON on stdout).
    fn surgeon(&self, args: &[&str]) -> (i32, Value) {
        let out = self.command().arg("--json").args(args).output().unwrap();
        let stdout = String::from_utf8(out.stdout).unwrap();
        let envelope = serde_json::from_str(&stdout)
            .unwrap_or_else(|e| panic!("{args:?}: not strict JSON ({e}): {stdout}"));
        (out.status.code().unwrap(), envelope)
    }

    fn run(&self, job: &str, extra: &[&str]) -> (i32, Value) {
        let mut args = vec![
            "pipeline",
            "run",
            "--config",
            "pipeline.json",
            "--job-dir",
            job,
        ];
        args.extend(extra);
        self.surgeon(&args)
    }

    fn model_bytes(&self) -> Vec<(String, Vec<u8>)> {
        let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(self.path("qwen2-mini"))
            .unwrap()
            .map(|e| e.unwrap())
            .map(|e| {
                (
                    e.file_name().to_string_lossy().into_owned(),
                    std::fs::read(e.path()).unwrap(),
                )
            })
            .collect();
        files.sort();
        files
    }
}

/// Every file under `dir`, recursively.
fn tree(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        match path.is_dir() {
            true => out.extend(tree(&path)),
            false => out.push(path),
        }
    }
    out
}

fn wait_for(what: &str, limit: Duration, mut done: impl FnMut() -> bool) {
    let started = Instant::now();
    while !done() {
        assert!(started.elapsed() < limit, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn a_full_decision_selects_on_dev_confirms_on_test_and_never_touches_the_model() {
    let ws = Ws::new(12);
    let before = ws.model_bytes();
    let (code, plan) = ws.surgeon(&[
        "pipeline",
        "plan",
        "--config",
        "pipeline.json",
        "--job-dir",
        "job",
    ]);
    assert_eq!(code, 0, "{plan}");
    let plan = &plan["data"];
    assert_eq!(plan["candidates"], 2);
    assert_eq!(plan["pillars"][2]["n_dev"], 12);
    assert!(plan["disk"]["total_bytes"].as_u64().unwrap() > 0);
    assert!(plan["calibration"]["tokens_per_second"].as_f64().unwrap() > 0.0);
    assert!(ws.path("job/plan.json").is_file());

    let (code, run) = ws.run("job", &[]);
    assert_eq!(code, 0, "{run}");
    let results = &run["data"];
    assert_eq!(results["verdict"], "GO", "{results}");
    // Both candidates pass the Dev; the tie on the selection pillar goes to the smaller alpha.
    assert_eq!(results["selected"], 0);
    assert_eq!(results["candidates"][1]["passed_dev"], true);
    assert_eq!(results["test"]["candidate"], 0);
    assert_eq!(results["model"]["unchanged"], true);
    assert_eq!(results["test_contaminated"], false);
    assert_eq!(
        ws.model_bytes(),
        before,
        "original model untouched, byte for byte"
    );
    assert_eq!(results, &ws.read("job/results.json"));

    // A GSM8K gate says how many generations the token budget cut, before and after.
    let dev = &results["candidates"][0]["dev"];
    let cut = |who: &str| {
        ws.read(&format!("job/checkpoints/{who}_dev_gsm.json"))["items"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|i| i["truncated"] == true)
            .count()
    };
    assert_eq!(dev[1]["pillar"], "gsm");
    assert_eq!(dev[1]["truncated_before"], cut("base"));
    assert_eq!(dev[1]["truncated_after"], cut("cand0"));
    assert_eq!(dev[1]["truncated_before"], 8, "6 tokens end no answer here");
    let reasons = |who: &str, reason: &str| {
        ws.read(&format!("job/checkpoints/{who}_dev_gsm.json"))["items"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|i| i["reason"] == reason)
            .count()
    };
    assert_eq!(dev[1]["loops_after"], reasons("cand0", "loop"));
    assert_eq!(dev[1]["no_answer_before"], reasons("base", "no_answer"));
    assert_eq!(
        dev[1]["loops_before"].as_u64().unwrap() + dev[1]["no_answer_before"].as_u64().unwrap(),
        8,
        "random tokens: a loop or no answer, never an extracted one"
    );
    assert!(dev[0]["truncated_before"].is_null(), "not a generation");

    // The Test part of the report holds aggregates only; the items are in private/.
    let report = ws.read("job/report.json");
    for pillar in report["results"]["test"]["pillars"].as_array().unwrap() {
        assert!(pillar.get("items").is_none());
    }
    let report_text = report.to_string();
    for prompt in ws.read("safety_test.json").as_array().unwrap() {
        assert!(!report_text.contains(prompt.as_str().unwrap()));
    }
    assert!(ws.path("job/private/cand0_test_safety.json").is_file());
    assert!(!ws.path("job/work").exists(), "working copy removed");
    assert_eq!(
        report["manifest"]["llama_cpp"],
        "non utilisé (backend natif, décodage glouton)"
    );
    assert_eq!(ws.read("test_access.json").as_array().unwrap().len(), 1);

    let (code, promoted) = ws.surgeon(&["pipeline", "promote", "--job-dir", "job"]);
    assert_eq!(code, 0, "{promoted}");
    assert_eq!(promoted["data"]["applied"], false);
    let patch = ws.read("job/promoted/patch.json");
    assert_eq!(patch["operations"][0]["alpha"], 0.5);
    assert_eq!(patch["backup"], true);
    assert_eq!(ws.model_bytes(), before, "promote never applies");

    // The schema of pipeline.json accepts the configuration.
    let out = ws
        .command()
        .args(["schema", "--kind", "pipeline"])
        .output()
        .unwrap();
    let schema: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert!(jsonschema::draft7::meta::is_valid(&schema));
    assert!(jsonschema::draft7::is_valid(&schema, &Ws::config()));
}

#[test]
fn plan_counts_the_dumps_that_coexist_at_their_real_size() {
    // The base dumps of the Dev and Test sets stay until the end of the job, one candidate
    // dump exists at a time; a dump holds one row per real token, not `max_tokens` rows.
    let ws = Ws::new(12);
    let positions = |file: &str| {
        let out = format!("{file}.safetensors");
        let (code, dump) = ws.surgeon(&[
            "bench",
            "dump",
            "--model",
            "qwen2-mini",
            "--prompts",
            file,
            "--out",
            &out,
            "--max-tokens",
            "16",
        ]);
        assert_eq!(code, 0, "{dump}");
        let data = &dump["data"];
        (
            data["n_positions"].as_u64().unwrap(),
            data["vocab_size"].as_u64().unwrap(),
        )
    };
    let (dev, vocab) = positions("neutral_dev.json");
    let (test, _) = positions("neutral_test.json");
    let (code, plan) = ws.surgeon(&[
        "pipeline",
        "plan",
        "--config",
        "pipeline.json",
        "--job-dir",
        "job",
    ]);
    assert_eq!(code, 0, "{plan}");
    assert_eq!(
        plan["data"]["disk"]["dumps_bytes"],
        (dev + test + dev.max(test)) * vocab * 4,
        "{plan}"
    );
}

#[test]
fn a_failed_cheap_gate_stops_before_the_costly_pillars() {
    let ws = Ws::new(12);
    let mut config = Ws::config();
    // Any drift fails: the KL gate is NO_GO for every candidate.
    config["pillars"][0]["gate"] = json!({"max_drift": {"value": 0.0, "confidence": 0.9}});
    ws.write("pipeline.json", &config);
    let (code, run) = ws.run("job", &[]);
    assert_eq!(code, 0, "{run}");
    let results = &run["data"];
    assert_eq!(results["verdict"], "NO_GO");
    assert!(results["test"].is_null() && results["selected"].is_null());
    for c in results["candidates"].as_array().unwrap() {
        assert_eq!(c["stopped_at"], "kl");
        assert_eq!(c["dev"].as_array().unwrap().len(), 1);
    }
    // The call counter: one measurement per candidate, the generation pillars never ran
    // (not even on the base model), and the Test Set was never consulted.
    let measured: Vec<String> = tree(&ws.path("job/checkpoints"))
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect();
    assert_eq!(measured.len(), 2, "{measured:?}");
    assert!(measured.iter().all(|m| m.ends_with("_dev_kl.json")));
    assert!(!ws.path("test_access.json").exists());
}

#[test]
fn too_small_a_sample_is_inconclusive_with_the_n_needed() {
    let ws = Ws::new(12);
    let mut config = Ws::config();
    config["pillars"][3]["gate"] = json!({"non_inferiority": {"margin": 0.02, "confidence": 0.95}});
    ws.write("pipeline.json", &config);
    let (_, plan) = ws.surgeon(&[
        "pipeline",
        "plan",
        "--config",
        "pipeline.json",
        "--job-dir",
        "job",
        "--no-calibration",
    ]);
    let safety = &plan["data"]["pillars"][3];
    assert_eq!(safety["dev_underpowered"], true, "{plan}");
    let needed = safety["min_n_for_go"].as_u64().unwrap();
    assert!(needed > 12);
    assert!(plan["data"]["estimated_secs"].is_null());
    let (code, run) = ws.run("job", &[]);
    assert_eq!(code, 0, "{run}");
    let results = &run["data"];
    assert_eq!(results["verdict"], "INCONCLUSIVE");
    let gate = &results["candidates"][0]["dev"][3]["gate"];
    assert_eq!(gate["verdict"], "INCONCLUSIVE");
    assert_eq!(gate["n_required"].as_u64(), Some(needed));
    assert!(results["test"].is_null(), "never GO, never tested");
}

#[test]
fn the_test_set_is_consulted_once() {
    let ws = Ws::new(12);
    assert_eq!(ws.run("job-a", &[]).1["data"]["verdict"], "GO");
    // Another job on the same Test sets: refused, nothing measured on them.
    let (code, second) = ws.run("job-b", &[]);
    assert_eq!(code, 1, "{second}");
    assert_eq!(second["error"]["kind"], "test_set_already_used");
    assert!(!ws.path("job-b/private/cand0_test_kl.json").exists());
    assert!(!ws.path("job-b/results.json").exists());
    // Explicit reuse (CLI only): measured, marked as contaminated, never a GO.
    let (code, reused) = ws.run("job-c", &["--allow-test-reuse"]);
    assert_eq!(code, 0, "{reused}");
    assert_eq!(reused["data"]["test_contaminated"], true);
    assert_eq!(reused["data"]["test"]["verdict"], "GO");
    assert_eq!(reused["data"]["verdict"], "INCONCLUSIVE");
    let registry = ws.read("test_access.json");
    let entries = registry.as_array().unwrap();
    assert_eq!(entries.len(), 2);
    assert_eq!(
        (entries[0]["reuse"].clone(), entries[1]["reuse"].clone()),
        (json!(false), json!(true))
    );
}

#[test]
fn a_resumed_job_writes_the_same_results_byte_for_byte() {
    let ws = Ws::new(200);
    let (code, _) = ws.run("reference", &[]);
    assert_eq!(code, 0);
    let reference = std::fs::read(ws.path("reference/results.json")).unwrap();
    // A fresh registry: the second job is the same decision, not a second consultation.
    std::fs::remove_file(ws.path("test_access.json")).unwrap();

    let mut worker = ws
        .command()
        .args([
            "pipeline",
            "run",
            "--config",
            "pipeline.json",
            "--job-dir",
            "crashed",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    // Killed once the base model has been measured on the long set, mid-candidate.
    wait_for(
        "a first generation checkpoint",
        Duration::from_secs(300),
        || {
            ws.path("crashed/checkpoints/base_dev_unblock.json")
                .exists()
        },
    );
    worker.kill().unwrap();
    worker.wait().unwrap();
    assert!(!ws.path("crashed/results.json").exists());
    for file in tree(&ws.path("crashed/checkpoints")) {
        serde_json::from_slice::<Value>(&std::fs::read(&file).unwrap())
            .unwrap_or_else(|e| panic!("{}: partial checkpoint ({e})", file.display()));
    }
    // Resumed without --config: the job is bound to its configuration.
    let (code, resumed) = ws.surgeon(&["pipeline", "run", "--job-dir", "crashed"]);
    assert_eq!(code, 0, "{resumed}");
    assert_eq!(
        std::fs::read(ws.path("crashed/results.json")).unwrap(),
        reference
    );
    // Bound: another configuration in the same job directory is refused.
    let mut other = Ws::config();
    other["seed"] = json!(8);
    ws.write("pipeline.json", &other);
    let (code, refused) = ws.run("crashed", &[]);
    assert_eq!(code, 2, "{refused}");
}

#[test]
fn cancel_stops_the_worker_without_residue() {
    let ws = Ws::new(200);
    let mut worker = ws
        .command()
        .args([
            "pipeline",
            "run",
            "--config",
            "pipeline.json",
            "--job-dir",
            "job",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_for("the worker", Duration::from_secs(120), || {
        std::fs::read_to_string(ws.path("job/state.json"))
            .is_ok_and(|s| s.contains("\"stage\": \"candidate_dev\""))
    });
    let (code, status) = ws.surgeon(&["pipeline", "status", "--job-dir", "job"]);
    assert_eq!(code, 0);
    assert_eq!(status["data"]["worker_running"], true, "{status}");
    assert!(status["data"]["state"]["units_total"].as_u64().unwrap() > 0);
    let (code, _) = ws.surgeon(&["pipeline", "cancel", "--job-dir", "job"]);
    assert_eq!(code, 0);
    let exit = worker.wait().unwrap();
    assert_eq!(exit.code(), Some(1), "cancelled");
    let (_, status) = ws.surgeon(&["pipeline", "status", "--job-dir", "job"]);
    let data = &status["data"];
    assert_eq!(data["state"]["status"], "cancelled", "{data}");
    assert_eq!(data["worker_running"], false);
    assert_eq!(data["resumable"], true);
    assert!(!ws.path("job/work").exists(), "working copy removed");
    assert!(!ws.path("job/lock").exists() && !ws.path("job/cancel").exists());
    for file in tree(&ws.path("job")) {
        assert!(
            !file.to_string_lossy().ends_with(".tmp"),
            "{}",
            file.display()
        );
    }
}

#[test]
fn the_safety_gate_is_mandatory_and_its_cli_override_is_logged() {
    let ws = Ws::new(12);
    let mut config = Ws::config();
    config["pillars"].as_array_mut().unwrap().pop();
    ws.write("pipeline.json", &config);
    let (code, refused) = ws.run("job", &[]);
    assert_eq!(code, 2, "{refused}");
    assert!(refused["error"]["message"]
        .as_str()
        .unwrap()
        .contains("aucun pilier safety"));
    let (code, run) = ws.run("job-2", &["--allow-disable-safety"]);
    assert_eq!(code, 0, "{run}");
    assert_eq!(run["data"]["safety_disabled"], true);
    assert_eq!(
        ws.read("job-2/report.json")["manifest"]["options"]["allow_disable_safety"],
        true
    );
}

// ------------------------------------------------------------------ MCP

/// An open `surgeon mcp` session: requests one at a time, replies by id.
struct Mcp {
    child: Child,
    stdin: Option<ChildStdin>,
    replies: Receiver<Value>,
    next: u64,
}

impl Mcp {
    fn start(ws: &Ws, read_only: bool) -> Self {
        let mut cmd = ws.command();
        cmd.args(["mcp", "--allow-dir", ws.dir.path().to_str().unwrap()]);
        if read_only {
            cmd.arg("--read-only");
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, replies) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                let value: Value = serde_json::from_str(&line).expect("strict JSON-RPC");
                if tx.send(value).is_err() {
                    break;
                }
            }
        });
        let mut mcp = Self {
            stdin: child.stdin.take(),
            child,
            replies,
            next: 1,
        };
        mcp.send(
            json!({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "p19", "version": "0"}}}),
        );
        mcp.wait(0);
        mcp.send(json!({"jsonrpc": "2.0", "method": "notifications/initialized"}));
        mcp
    }

    fn send(&mut self, message: Value) {
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(stdin, "{message}").unwrap();
        stdin.flush().unwrap();
    }

    fn wait(&self, id: u64) -> Value {
        loop {
            let reply = self.replies.recv_timeout(Duration::from_secs(300)).unwrap();
            if reply["id"] == id {
                return reply;
            }
        }
    }

    /// `tools/call` of `name`: the raw reply.
    fn call(&mut self, name: &str, arguments: Value) -> Value {
        let id = self.next;
        self.next += 1;
        self.send(json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
            "params": {"name": name, "arguments": arguments}}));
        self.wait(id)
    }

    fn pipeline(&mut self, arguments: Value) -> Value {
        self.call("surgeon_pipeline", arguments)["result"]["structuredContent"].clone()
    }

    /// Closes stdin: the server ends.
    fn close(mut self) -> std::process::ExitStatus {
        drop(self.stdin.take());
        self.child.wait().unwrap()
    }
}

#[test]
fn mcp_start_status_report_promote_without_ever_showing_a_test_item() {
    let ws = Ws::new(12);
    let mut mcp = Mcp::start(&ws, false);
    let start =
        mcp.pipeline(json!({"action": "start", "config": "pipeline.json", "job_dir": "job"}));
    assert_eq!(start["ok"], true, "{start}");
    assert_eq!(start["data"]["job_id"].as_str().unwrap().len(), 16);
    // One job at a time.
    let again =
        mcp.pipeline(json!({"action": "start", "config": "pipeline.json", "job_dir": "job2"}));
    assert_eq!(again["error"]["kind"], "busy", "{again}");
    let mut transcript = Vec::new();
    wait_for("the report", Duration::from_secs(300), || {
        let status = mcp.pipeline(json!({"action": "status", "job_dir": "job"}));
        transcript.push(status.to_string());
        status["data"]["report_available"] == true && status["data"]["worker_running"] == false
    });
    let report = mcp.pipeline(json!({"action": "report", "job_dir": "job"}));
    transcript.push(report.to_string());
    assert_eq!(report["data"]["results"]["verdict"], "GO", "{report}");
    let promoted = mcp.pipeline(json!({"action": "promote_patch", "job_dir": "job"}));
    assert_eq!(promoted["ok"], true, "{promoted}");
    assert!(ws.path("job/promoted/patch.json").is_file());
    // Test items are behind the sandbox, even for the other tools.
    let private = mcp.call(
        "surgeon_bench",
        json!({"mode": "refusal", "model": "qwen2-mini", "prompts": "job/private/cand0_test_safety.json"}),
    );
    assert_eq!(
        private["result"]["structuredContent"]["error"]["kind"], "path_not_allowed",
        "{private}"
    );
    // No Test prompt ever crossed the protocol.
    let transcript = transcript.join("\n");
    for name in ["safety_test.json", "unblock_test.json", "neutral_test.json"] {
        for prompt in ws.read(name).as_array().unwrap() {
            assert!(!transcript.contains(prompt.as_str().unwrap()), "{prompt}");
        }
    }
    assert_eq!(mcp.close().code(), Some(0));
}

#[test]
fn mcp_has_no_way_around_the_safety_gate_the_sandbox_or_read_only() {
    let ws = Ws::new(12);
    let mut config = Ws::config();
    config["pillars"].as_array_mut().unwrap().pop();
    ws.write("unsafe.json", &config);
    let outside = tempfile::tempdir().unwrap();
    let mut mcp = Mcp::start(&ws, false);
    let no_safety =
        mcp.pipeline(json!({"action": "start", "config": "unsafe.json", "job_dir": "job"}));
    assert_eq!(no_safety["error"]["kind"], "invalid_input", "{no_safety}");
    for extra in [
        json!({"allow_disable_safety": true}),
        json!({"allow_test_reuse": true}),
        json!({"args": ["--allow-disable-safety"]}),
    ] {
        let mut args = json!({"action": "start", "config": "unsafe.json", "job_dir": "job"});
        for (k, v) in extra.as_object().unwrap() {
            args[k] = v.clone();
        }
        let refused = mcp.pipeline(args);
        assert_eq!(refused["error"]["kind"], "invalid_arguments", "{refused}");
    }
    let escape = mcp.pipeline(json!({"action": "status",
        "job_dir": outside.path().to_str().unwrap()}));
    assert_eq!(escape["error"]["kind"], "path_not_allowed");
    // A configuration naming a file outside the sandbox.
    let mut config = Ws::config();
    config["pillars"][0]["dataset_test"] = json!(outside.path().join("t.json"));
    ws.write("leaky.json", &config);
    let leaky = mcp.pipeline(json!({"action": "plan", "config": "leaky.json", "job_dir": "job"}));
    assert_eq!(leaky["error"]["kind"], "path_not_allowed", "{leaky}");
    mcp.close();

    let mut read_only = Mcp::start(&ws, true);
    let start =
        read_only.pipeline(json!({"action": "start", "config": "pipeline.json", "job_dir": "job"}));
    assert_eq!(start["error"]["kind"], "read_only");
    let plan =
        read_only.pipeline(json!({"action": "plan", "config": "pipeline.json", "job_dir": "job"}));
    assert_eq!(plan["ok"], true, "{plan}");
    assert!(
        !ws.path("job/plan.json").exists(),
        "read-only: nothing written"
    );
    read_only.close();
}

#[test]
fn mcp_status_and_cancel_see_the_worker_as_soon_as_start_returns() {
    // `start` returns before the worker has taken the job lock. In that window the server
    // must still say that its worker runs, and a cancel must still stop it.
    let ws = Ws::new(200);
    let mut mcp = Mcp::start(&ws, false);
    let start =
        mcp.pipeline(json!({"action": "start", "config": "pipeline.json", "job_dir": "job"}));
    assert_eq!(start["ok"], true, "{start}");
    let status = mcp.pipeline(json!({"action": "status", "job_dir": "job"}));
    assert_eq!(status["data"]["worker_running"], true, "{status}");
    assert_eq!(status["data"]["resumable"], false, "{status}");
    let cancel = mcp.pipeline(json!({"action": "cancel", "job_dir": "job"}));
    assert_eq!(cancel["ok"], true, "{cancel}");
    wait_for("the cancellation", Duration::from_secs(60), || {
        mcp.pipeline(json!({"action": "status", "job_dir": "job"}))["data"]["worker_running"]
            == false
    });
    let end = mcp.pipeline(json!({"action": "status", "job_dir": "job"}));
    assert_eq!(end["data"]["state"]["status"], "cancelled", "{end}");
    assert!(!ws.path("job/results.json").exists());
    assert!(!ws.path("job/work").exists());
    mcp.close();
}

#[test]
fn the_worker_stops_with_the_mcp_server() {
    let ws = Ws::new(200);
    let mut mcp = Mcp::start(&ws, false);
    let start =
        mcp.pipeline(json!({"action": "start", "config": "pipeline.json", "job_dir": "job"}));
    assert_eq!(start["ok"], true, "{start}");
    wait_for("the worker", Duration::from_secs(120), || {
        std::fs::read_to_string(ws.path("job/state.json")).is_ok_and(|s| s.contains("running"))
    });
    assert_eq!(mcp.close().code(), Some(0));
    let (_, status) = ws.surgeon(&["pipeline", "status", "--job-dir", "job"]);
    let data = &status["data"];
    assert_eq!(data["worker_running"], false, "{data}");
    assert_eq!(data["state"]["status"], "cancelled", "{data}");
    assert!(!ws.path("job/work").exists());
}

#[test]
fn a_patch_template_with_the_default_backup_passes_the_static_checks() {
    // `backup` defaults to true in a patch file. The pipeline patches a disposable working
    // copy, so the flag of the template must not fail the dry run of every candidate.
    let ws = Ws::new(12);
    let mut config = Ws::config();
    config["patch"].as_object_mut().unwrap().remove("backup");
    ws.write("pipeline.json", &config);
    let (code, run) = ws.run("job", &[]);
    assert_eq!(code, 0, "{run}");
    let results = &run["data"];
    for c in results["candidates"].as_array().unwrap() {
        assert!(c["static_error"].is_null(), "{c}");
    }
    assert_eq!(results["verdict"], "GO", "{results}");
}

#[test]
fn candidates_refused_by_the_static_checks_are_not_reported_as_a_dev_failure() {
    let ws = Ws::new(12);
    let mut config = Ws::config();
    // The mini model has two layers: no candidate can be applied.
    config["patch"]["operations"][0]["layers"] = json!("7-8");
    ws.write("pipeline.json", &config);
    let (code, run) = ws.run("job", &[]);
    assert_eq!(code, 0, "{run}");
    let results = &run["data"];
    assert_eq!(results["verdict"], "NO_GO");
    for c in results["candidates"].as_array().unwrap() {
        assert_eq!(c["stopped_at"], "static_checks");
        assert!(c["dev"].as_array().unwrap().is_empty());
    }
    let reason = results["verdict_reason"].as_str().unwrap();
    assert!(reason.contains("contrôles statiques"), "{reason}");
    assert!(
        tree(&ws.path("job/checkpoints")).is_empty(),
        "nothing measured"
    );
}

#[test]
fn plan_does_not_announce_a_guaranteed_inconclusive_that_a_table_can_refute() {
    // GSM8K Dev has 8 questions. At a margin of 0.2 and 90 %, 4 right on both sides and 4
    // wrong on both sides is GO (lower bound -0.178), although 8 right on both sides is not
    // (-0.253): 8 questions are not "INCONCLUSIVE whatever the model does".
    let ws = Ws::new(12);
    let mut config = Ws::config();
    config["pillars"][1]["gate"] = json!({"non_inferiority": {"margin": 0.2, "confidence": 0.9}});
    ws.write("pipeline.json", &config);
    let (code, plan) = ws.surgeon(&[
        "pipeline",
        "plan",
        "--config",
        "pipeline.json",
        "--job-dir",
        "job",
        "--no-calibration",
    ]);
    assert_eq!(code, 0, "{plan}");
    let gsm = &plan["data"]["pillars"][1];
    assert_eq!(gsm["n_dev"], 8);
    assert_eq!(gsm["dev_underpowered"], false, "{gsm}");
    assert_eq!(gsm["min_n_possible"], 8, "{gsm}");
    assert_eq!(gsm["min_n_for_go"], 11, "{gsm}");
    let warnings = plan["data"]["warnings"].to_string();
    assert!(!warnings.contains("garanti"), "{warnings}");
    // Between the two sizes the plan still says what the outcome depends on.
    assert!(warnings.contains("taux de base"), "{warnings}");
}

/// Raises the interruption flag the first time the RAM is read while the working copy
/// exists: that is inside `apply` on the copy, after the worker's last cancellation check.
struct CancelDuringApply {
    inner: weight_surgeon::hardware::SystemProbe,
    work_model: PathBuf,
    flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl weight_surgeon::hardware::HardwareProbe for CancelDuringApply {
    fn system_ram(&self) -> weight_surgeon::hardware::Result<weight_surgeon::hardware::MemoryInfo> {
        if self.work_model.exists() {
            self.flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        self.inner.system_ram()
    }

    fn gpus(&self) -> weight_surgeon::hardware::Result<Vec<weight_surgeon::hardware::GpuInfo>> {
        self.inner.gpus()
    }
}

#[test]
fn a_cancellation_that_lands_during_apply_is_a_cancellation_not_a_failure() {
    use weight_surgeon::pipeline::job::Job;
    use weight_surgeon::pipeline::run::{self, RunOptions};
    use weight_surgeon::pipeline::PipelineError;

    let ws = Ws::new(12);
    let mut deps = weight_surgeon::app::real_deps();
    deps.probe = Box::new(CancelDuringApply {
        inner: weight_surgeon::hardware::SystemProbe::detect(),
        work_model: ws.path("job/work/model"),
        flag: std::sync::Arc::clone(&deps.interrupted),
    });
    let options = RunOptions::default();
    let config = run::load_config(&ws.path("pipeline.json"), &options).unwrap();
    let job = Job::new(&ws.path("job"));
    let file = run::bind(&job, Some(config), options).unwrap();
    match run::run(&job, file, &deps) {
        Err(PipelineError::Cancelled) => {}
        Err(other) => panic!("reported as a failure: {other}"),
        Ok(_) => panic!("the run was not interrupted"),
    }
    let state = ws.read("job/state.json");
    assert_eq!(state["status"], "cancelled", "{state}");
    assert!(state["error"].is_null(), "{state}");
    assert!(!ws.path("job/work").exists(), "working copy removed");
}

/// The binary pillars generate through the configured backend (here a fake llama-server
/// standing for a GPU llama.cpp), the drift pillars stay native, and the job records which.
#[test]
fn the_binary_pillars_run_on_the_configured_backend_and_the_job_is_bound_to_it() {
    let ws = Ws::new(12);
    std::fs::create_dir(ws.path("llama")).unwrap();
    std::fs::copy(
        support::fake_server(),
        ws.path("llama").join(support::server_name()),
    )
    .unwrap();
    let flags = [
        "--backend",
        "llama-server",
        "--llama-cpp-dir",
        "llama",
        "--gpu-layers",
        "99",
    ];
    let label = "llama-server (Vulkan, ngl=99, batch=1)";
    // The mini model's word-level tokenizer cannot be a GGUF vocabulary: give it empty merges
    // as a BPE (the fake server never reads the file).
    let tokenizer = ws.path("qwen2-mini/tokenizer.json");
    let mut json: Value =
        serde_json::from_str(&std::fs::read_to_string(&tokenizer).unwrap()).unwrap();
    json["model"]["type"] = json!("BPE");
    json["model"]["merges"] = json!([]);
    std::fs::write(&tokenizer, json.to_string()).unwrap();

    // The calibration measures the native CPU engine: no estimate for another backend.
    let mut plan_args = vec![
        "pipeline",
        "plan",
        "--config",
        "pipeline.json",
        "--job-dir",
        "job",
    ];
    plan_args.extend(flags);
    let (code, plan) = ws.surgeon(&plan_args);
    assert_eq!(code, 0, "{plan}");
    assert_eq!(plan["data"]["backend"], label);
    assert_eq!(plan["data"]["estimated_secs"], Value::Null);
    assert!(plan["warnings"][0]
        .as_str()
        .unwrap()
        .contains("non estimée"));

    let (code, run) = ws.run("job", &flags);
    assert_eq!(code, 0, "{run}");
    assert_eq!(run["data"]["backend"], label);
    assert_eq!(run["data"]["model"]["unchanged"], true);
    assert_eq!(
        ws.read("job/job.json")["config"]["backend"]["backend"],
        "llama-server"
    );
    let report = ws.read("job/report.json");
    assert_eq!(report["manifest"]["backend"], label);
    assert_eq!(report["manifest"]["llama_cpp"], "llama");
    // The fake server answered the generation pillars: no temporary GGUF is left in the job.
    assert!(tree(&ws.path("job"))
        .iter()
        .all(|p| p.extension().is_none_or(|e| e != "gguf")));

    // Bound to its backend: the same file without the flags is another configuration, and a
    // resume without --config cannot change it.
    let (code, v) = ws.run("job", &[]);
    assert_eq!(code, 2, "{v}");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("autre configuration"));
    let (code, v) = ws.surgeon(&["pipeline", "run", "--job-dir", "job", "--backend", "native"]);
    assert_eq!(code, 2, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("reprise"));
    let (code, v) = ws.run("job-hip", &["--backend", "hipfire"]);
    assert_eq!(code, 2, "{v}");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("--backend llama-server"));
}
