//! The `--events` NDJSON protocol: golden lines of every event (the v1 wire format), and
//! the option on a real command, whose envelope stays what it is without it.

#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use regex::Regex;
use serde_json::Value;
use weight_surgeon::events::{
    parse_line, EventLine, LogLevel, NodeStatus, SurgeonEvent, EVENTS_VERSION,
};

const GOLDEN: &str = include_str!("fixtures/events/v1.ndjson");

fn golden_events() -> Vec<SurgeonEvent> {
    use SurgeonEvent as S;
    vec![
        S::JobStarted {
            command: "apply".into(),
        },
        S::Progress {
            done: 3,
            total: Some(8),
            unit: "tensor".into(),
        },
        S::Progress {
            done: 2,
            total: None,
            unit: "chunk".into(),
        },
        S::NodeState {
            node_id: "model.layers.1.mlp.down_proj.weight".into(),
            state: NodeStatus::Running,
        },
        S::Metric {
            node_id: Some("ref".into()),
            name: "max_delta".into(),
            value: 0.125,
        },
        S::Metric {
            node_id: None,
            name: "edges".into(),
            value: 42.0,
        },
        S::Log {
            level: LogLevel::Info,
            message: "écriture validée en 0.2 s".into(),
        },
        S::Artifact {
            kind: "backup".into(),
            path: PathBuf::from("m/model.safetensors.bak-1700000000"),
            sha256: None,
        },
        S::Artifact {
            kind: "vector".into(),
            path: PathBuf::from("artifacts/v.safetensors"),
            sha256: Some("ab".repeat(32)),
        },
        S::Warning {
            message: "rollback : modèle restauré".into(),
        },
        S::Finished { exit_code: 0 },
        S::Failed {
            kind: "invalid_patch".into(),
            message: "operations[0] : alpha manquant".into(),
            exit_code: 2,
        },
        S::Cancelled,
    ]
}

#[test]
fn every_event_matches_its_golden_line_both_ways() {
    let lines: Vec<&str> = GOLDEN.lines().collect();
    let events = golden_events();
    assert_eq!(lines.len(), events.len());
    for (i, (line, event)) in lines.iter().zip(events).enumerate() {
        let expected = EventLine {
            v: EVENTS_VERSION,
            seq: i as u64,
            t_ms: 10 * i as u64,
            event,
        };
        assert_eq!(serde_json::to_string(&expected).unwrap(), *line, "line {i}");
        assert_eq!(parse_line(line).unwrap(), expected, "line {i}");
    }
}

#[test]
fn the_events_schema_is_printed_by_the_schema_command() {
    let out = surgeon(Path::new("."), &["schema", "--kind", "events"]);
    assert_eq!(out.status.code(), Some(0));
    let schema: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(schema, weight_surgeon::events::schema());
    assert_eq!(schema["title"], "EventLine");
}

fn surgeon(dir: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_surgeon"))
        .args(args)
        .current_dir(dir)
        .env("NO_COLOR", "1")
        .env_remove("CLICOLOR_FORCE")
        .output()
        .unwrap()
}

const VECTOR: &str =
    "[0.25,-0.25,0.5,-0.5,0.125,-0.125,0.0,1.0,0.25,-0.25,0.5,-0.5,0.125,-0.125,0.0,1.0]";
const PATCH: &str = r#"{"schema_version":1,"operations":[{"id":"ref","layers":"1-2","projections":["down_proj"],"operation":"subtract","vector_file":"vec.json","alpha":0.5,"axis":"cols"}]}"#;

fn fixture_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    fake_qwen2::write_fake_qwen2(dir.path()).unwrap();
    std::fs::write(dir.path().join("vec.json"), VECTOR).unwrap();
    std::fs::write(dir.path().join("patch.json"), PATCH).unwrap();
    dir
}

/// The envelope without what differs between two runs: the temporary directory, backup
/// stamps and the I/O measures (`data.io`: timings, page faults).
fn normalized(stdout: &[u8], dir: &Path) -> Value {
    let mut envelope: Value = serde_json::from_slice(stdout).unwrap();
    envelope["data"]
        .as_object_mut()
        .unwrap()
        .remove("io")
        .expect("apply reports its I/O");
    // The directory as it appears inside a JSON string (escaped backslashes on Windows).
    let quoted = Value::from(dir.display().to_string()).to_string();
    let text = envelope
        .to_string()
        .replace(&quoted[1..quoted.len() - 1], "<dir>");
    let stamps = Regex::new(r"\.bak-\d+(-\d+)?").unwrap();
    serde_json::from_str(&stamps.replace_all(&text, ".bak-<stamp>")).unwrap()
}

fn read_events(path: &Path) -> Vec<EventLine> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|l| parse_line(l).unwrap())
        .collect()
}

#[test]
fn apply_with_events_writes_the_stream_and_keeps_its_envelope() {
    let args = [
        "--json",
        "apply",
        "--model",
        "qwen2-mini",
        "--patch",
        "patch.json",
    ];
    let plain_dir = fixture_dir();
    let plain = surgeon(plain_dir.path(), &args);
    assert_eq!(plain.status.code(), Some(0), "{plain:?}");

    let dir = fixture_dir();
    let mut with_events = vec!["--events", "events.ndjson"];
    with_events.extend(args);
    let observed = surgeon(dir.path(), &with_events);
    assert_eq!(observed.status.code(), Some(0), "{observed:?}");
    assert_eq!(
        normalized(&observed.stdout, dir.path()),
        normalized(&plain.stdout, plain_dir.path())
    );

    let events = read_events(&dir.path().join("events.ndjson"));
    for (i, line) in events.iter().enumerate() {
        assert_eq!(line.seq, i as u64, "no gap");
    }
    assert_eq!(
        events.first().unwrap().event,
        SurgeonEvent::JobStarted {
            command: "apply".into()
        }
    );
    assert_eq!(
        events.last().unwrap().event,
        SurgeonEvent::Finished { exit_code: 0 }
    );
    let has = |f: &dyn Fn(&SurgeonEvent) -> bool| events.iter().any(|l| f(&l.event));
    assert!(has(
        &|e| matches!(e, SurgeonEvent::Artifact { kind, .. } if kind == "backup")
    ));
    // Layers 1-2, down_proj: two tensors shifted, each reported done.
    for layer in [1, 2] {
        let id = format!("model.layers.{layer}.mlp.down_proj.weight");
        assert!(has(
            &|e| matches!(e, SurgeonEvent::NodeState { node_id, state: NodeStatus::Done } if *node_id == id)
        ));
    }
    assert!(has(&|e| matches!(e,
        SurgeonEvent::Progress { done: 2, total: Some(2), unit } if unit == "tensor")));
}

#[test]
fn a_failure_closes_the_stream_with_its_kind() {
    let dir = fixture_dir();
    std::fs::write(dir.path().join("patch.json"), "{").unwrap();
    let out = surgeon(
        dir.path(),
        &[
            "--json",
            "--events",
            "ev.ndjson",
            "apply",
            "--model",
            "qwen2-mini",
            "--patch",
            "patch.json",
        ],
    );
    assert_ne!(out.status.code(), Some(0));
    let envelope: Value = serde_json::from_slice(&out.stdout).unwrap();
    let events = read_events(&dir.path().join("ev.ndjson"));
    match &events.last().unwrap().event {
        SurgeonEvent::Failed {
            kind, exit_code, ..
        } => {
            assert_eq!(Some(kind.as_str()), envelope["error"]["kind"].as_str());
            assert_eq!(Some(i64::from(*exit_code)), envelope["exit_code"].as_i64());
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn dash_writes_the_events_to_stderr_and_a_bad_target_stops_before_any_work() {
    let dir = fixture_dir();
    let out = surgeon(
        dir.path(),
        &[
            "--json",
            "--events",
            "-",
            "inspect",
            "--model",
            "qwen2-mini",
        ],
    );
    assert_eq!(out.status.code(), Some(0));
    let stderr = String::from_utf8(out.stderr).unwrap();
    let events: Vec<EventLine> = stderr.lines().map(|l| parse_line(l).unwrap()).collect();
    assert_eq!(events.len(), 2, "{stderr}");
    assert!(events[1].event.is_terminal());
    serde_json::from_slice::<Value>(&out.stdout).expect("stdout is the envelope alone");

    let bad = surgeon(
        dir.path(),
        &[
            "--json",
            "--events",
            "missing/dir/ev.ndjson",
            "inspect",
            "--model",
            "qwen2-mini",
        ],
    );
    assert_eq!(bad.status.code(), Some(2));
    let envelope: Value = serde_json::from_slice(&bad.stdout).unwrap();
    assert_eq!(envelope["error"]["kind"], "invalid_input");
}
