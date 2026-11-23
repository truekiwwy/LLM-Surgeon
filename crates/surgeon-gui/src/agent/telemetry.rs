//! The Live telemetry of a session: from the agent's engine calls, the stage of the protocol
//! it is at (health, probe, compose, measure), the method of its newest call and what it
//! targets; from the project's runs, where the measured metric stands against the origin.

use serde_json::Value;

use super::activity;
use super::provider::Block;
use super::session::Conversation;
use crate::studio::ledger::{Entry, Verdict};

/// The four stages of the protocol, in order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Stage {
    Health,
    Probe,
    Compose,
    Measure,
}

impl Stage {
    pub const ALL: [Stage; 4] = [Stage::Health, Stage::Probe, Stage::Compose, Stage::Measure];
}

/// What a figure rests on: a screening without generation, or a real measure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Evidence {
    Heuristic,
    Measured,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Method {
    /// `orthogonal-preserve · f64`, `refusal_probe · first-token-prob`, `mean_diff · resid_post`.
    pub text: String,
    pub evidence: Option<Evidence>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Telemetry {
    pub stage: Option<Stage>,
    pub method: Option<Method>,
    /// The layers and matrices of the newest call that names some.
    pub target: Option<String>,
}

fn text(v: &Value, default: &str) -> String {
    v.as_str().unwrap_or(default).to_owned()
}

/// The stage and the method of a call to `name`; `None` outside the four stages (the plan,
/// the findings, the files).
pub fn classify(name: &str, input: &Value) -> Option<(Stage, Method)> {
    let tool = name.rsplit("__").next().unwrap_or(name);
    let method = |text: String, evidence| Method { text, evidence };
    Some(match tool {
        "surgeon_doctor" | "surgeon_check" | "surgeon_workspace" | "surgeon_inspect"
        | "surgeon_graph" => (
            Stage::Health,
            method(tool.trim_start_matches("surgeon_").into(), None),
        ),
        "surgeon_extract" => {
            let how = format!(
                "{} · {}",
                text(&input["method"], "mean_diff"),
                text(&input["site"], "resid_post")
            );
            (Stage::Probe, method(how, None))
        }
        "surgeon_autotune" => (Stage::Probe, method("autotune · ridge".into(), None)),
        "surgeon_causal_trace" => {
            let how = format!("causal · {}", text(&input["component"], "resid"));
            (Stage::Probe, method(how, None))
        }
        "surgeon_bench" if input["mode"] == "refusal_probe" => {
            let probe = text(&input["probe_method"], "first-token-prob");
            let how = format!("refusal_probe · {probe}");
            (Stage::Probe, method(how, Some(Evidence::Heuristic)))
        }
        // Compose works in f64 and writes f32 (patch::compose).
        "surgeon_compose" => {
            let how = format!("{} · f64", text(&input["mode"], "orthogonal-preserve"));
            (Stage::Compose, method(how, None))
        }
        "surgeon_apply" => {
            let op = &input["patch"]["operations"][0];
            let how = match (op["operation"].as_str(), op["alpha"].as_f64()) {
                (Some(operation), Some(alpha)) => format!("{operation} · α {alpha}"),
                (Some(operation), None) => operation.to_owned(),
                // A patch given as a file: its name.
                _ => text(&input["patch"], "apply"),
            };
            (Stage::Compose, method(how, None))
        }
        "surgeon_restore" | "surgeon_lora" | "surgeon_diff" => (
            Stage::Compose,
            method(tool.trim_start_matches("surgeon_").into(), None),
        ),
        "surgeon_bench" | "surgeon_eval" => {
            let mode = text(&input["mode"], "eval");
            let how = match input["dataset"].as_str() {
                Some(dataset) => format!("{mode} · {dataset}"),
                None => mode,
            };
            (Stage::Measure, method(how, Some(Evidence::Measured)))
        }
        _ => return None,
    })
}

/// The newest staged call of `conv` sets the stage and the method; the newest call that names
/// layers or matrices sets the target.
pub fn of(conv: &Conversation) -> Telemetry {
    let mut out = Telemetry::default();
    let calls = conv.messages.iter().flat_map(|m| &m.blocks);
    for (name, input) in calls.filter_map(|b| match b {
        Block::ToolUse { name, input, .. } => Some((name, input)),
        _ => None,
    }) {
        if let Some((stage, method)) = classify(name, input) {
            out.stage = Some(stage);
            out.method = Some(method);
        }
        if let Some(a) = activity::of(name, input).filter(|a| !a.target.is_empty()) {
            out.target = Some(a.target);
        }
    }
    out
}

/// Where the measured metric stands: the copy, and the origin when the run compared both.
#[derive(Debug, Clone, PartialEq)]
pub struct Progress {
    pub metric: &'static str,
    pub before: Option<f64>,
    pub after: f64,
    pub heuristic: bool,
}

/// The newest successful `surgeon_bench` run of `entries` with a rate it can read.
pub fn progress(entries: &[Entry]) -> Option<Progress> {
    entries
        .iter()
        .filter(|e| e.tool == "surgeon_bench" && e.verdict == Verdict::Ok)
        .filter_map(|e| {
            let r = &e.result;
            let metric = ["probe_refusal_rate", "refusal_rate", "accuracy"]
                .into_iter()
                .find(|m| r[*m].is_number())?;
            let progress = Progress {
                metric,
                before: r["baseline"][metric].as_f64(),
                after: r[metric].as_f64()?,
                heuristic: metric == "probe_refusal_rate" || r["is_heuristic"] == true,
            };
            Some((e.finished_unix, progress))
        })
        .max_by_key(|(finished, _)| *finished)
        .map(|(_, progress)| progress)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::agent::provider::{Message, Role};
    use crate::studio::ledger::{Hashes, Versions};

    #[test]
    fn each_engine_tool_falls_in_its_stage_with_its_method() {
        let (stage, m) = classify("surgeon_doctor", &json!({})).unwrap();
        assert_eq!((stage, m.text.as_str()), (Stage::Health, "doctor"));
        let (stage, m) = classify("mcp__surgeon__surgeon_extract", &json!({"layer": 14})).unwrap();
        assert_eq!(
            (stage, m.text.as_str()),
            (Stage::Probe, "mean_diff · resid_post")
        );
        let probe = json!({"mode": "refusal_probe"});
        let (stage, m) = classify("surgeon_bench", &probe).unwrap();
        assert_eq!(stage, Stage::Probe);
        assert_eq!(m.text, "refusal_probe · first-token-prob");
        assert_eq!(m.evidence, Some(Evidence::Heuristic));
        let compose = json!({"mode": "orthogonal-preserve"});
        let (stage, m) = classify("surgeon_compose", &compose).unwrap();
        assert_eq!(
            (stage, m.text.as_str()),
            (Stage::Compose, "orthogonal-preserve · f64")
        );
        let apply = json!({"patch": {"operations": [{"operation": "project_out", "alpha": 0.6}]}});
        assert_eq!(
            classify("surgeon_apply", &apply).unwrap().1.text,
            "project_out · α 0.6"
        );
        let file = json!({"patch": "work/patch_compose.json"});
        assert_eq!(
            classify("surgeon_apply", &file).unwrap().1.text,
            "work/patch_compose.json"
        );
        let eval = json!({"mode": "eval", "dataset": "builtin:math"});
        let (stage, m) = classify("surgeon_bench", &eval).unwrap();
        assert_eq!(
            (stage, m.text.as_str()),
            (Stage::Measure, "eval · builtin:math")
        );
        assert_eq!(m.evidence, Some(Evidence::Measured));
        assert!(classify("record_finding", &json!({})).is_none());
    }

    #[test]
    fn the_newest_staged_call_sets_the_stage_and_the_target_sticks() {
        let call = |name: &str, input: Value| Block::ToolUse {
            id: name.into(),
            name: name.into(),
            input,
        };
        let mut conv = Conversation::default();
        assert_eq!(of(&conv), Telemetry::default());
        conv.messages.push(Message {
            role: Role::Assistant,
            blocks: vec![
                call("surgeon_extract", json!({"layer": 14})),
                call("surgeon_compose", json!({"mode": "orthogonal-preserve"})),
                call("update_plan", json!({})),
            ],
        });
        let t = of(&conv);
        assert_eq!(t.stage, Some(Stage::Compose));
        assert_eq!(t.method.unwrap().text, "orthogonal-preserve · f64");
        assert_eq!(t.target.as_deref(), Some("L14"), "compose names no layer");
    }

    fn bench(id: &str, finished: u64, result: Value) -> Entry {
        Entry {
            run_id: id.into(),
            tool: "surgeon_bench".into(),
            label: String::new(),
            params: Value::Null,
            resumed_from: None,
            started_unix: 0,
            finished_unix: finished,
            hashes: Hashes::default(),
            versions: Versions::default(),
            verdict: Verdict::Ok,
            exit_code: 0,
            result,
            error: None,
        }
    }

    #[test]
    fn progress_is_the_newest_readable_bench_against_its_baseline() {
        assert_eq!(progress(&[]), None);
        let refusal = json!({"refusal_rate": 0.08, "baseline": {"refusal_rate": 0.62}});
        let probe = json!({"probe_refusal_rate": 0.3, "is_heuristic": true});
        let entries = [
            bench("a", 10, refusal),
            bench("b", 5, probe),
            bench("c", 20, json!({"mode": "ppl"})),
        ];
        let p = progress(&entries).unwrap();
        assert_eq!(
            p,
            Progress {
                metric: "refusal_rate",
                before: Some(0.62),
                after: 0.08,
                heuristic: false
            }
        );
        let p = progress(&entries[1..]).unwrap();
        assert_eq!(
            (p.metric, p.before, p.heuristic),
            ("probe_refusal_rate", None, true)
        );
    }
}
