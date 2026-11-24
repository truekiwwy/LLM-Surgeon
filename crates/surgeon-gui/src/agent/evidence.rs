//! The evidence log: every finding the agent records, and its status computed from the
//! project's ledger each time it is shown, never taken from the agent.
//!
//! - Measured: it cites runs that exist and succeeded, and each cited value appears in that
//!   run's result. A finding with figures must link at least one value.
//! - Heuristic: the same, but one of its runs is an exploration heuristic (its result says
//!   `is_heuristic: true`: `surgeon_bench` mode `refusal_probe`). It points somewhere; a
//!   full generation (mode `refusal` or `eval`) has yet to prove it.
//! - Inferred: no run backs it, but the agent gave its reasoning (shown).
//! - Unknown: anything else, with the reason.

use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::studio::ledger::{Entry, Hashes, Verdict, Versions};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Evidence {
    pub run_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    pub text: String,
    #[serde(default)]
    pub evidence: Vec<Evidence>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    pub conversation: String,
    pub created_unix: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Status {
    /// `(run id, model hash)` of every cited run.
    Measured(Vec<(String, Option<String>)>),
    /// Backed like `Measured`, by at least one heuristic run: never a final proof.
    Heuristic(Vec<(String, Option<String>)>),
    Inferred(String),
    Unknown(String),
}

impl Status {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Measured(_) => "Mesuré",
            Self::Heuristic(_) => "Heuristique",
            Self::Inferred(_) => "Déduit",
            Self::Unknown(_) => "Inconnu",
        }
    }
}

impl Finding {
    /// Reads `record_finding` arguments.
    pub fn from_args(args: &Value, conversation: &str) -> Result<Self, String> {
        let text = args["text"]
            .as_str()
            .filter(|t| !t.trim().is_empty())
            .ok_or("record_finding : text manquant")?;
        let evidence = match &args["evidence"] {
            Value::Null => Vec::new(),
            v => serde_json::from_value(v.clone())
                .map_err(|e| format!("record_finding : evidence illisible : {e}"))?,
        };
        Ok(Self {
            text: text.into(),
            evidence,
            reasoning: args["reasoning"]
                .as_str()
                .filter(|r| !r.trim().is_empty())
                .map(String::from),
            conversation: conversation.into(),
            created_unix: crate::studio::project::now_unix(),
        })
    }

    pub fn status(&self, ledger: &[Entry]) -> Status {
        let figures = self.text.chars().any(|c| c.is_ascii_digit());
        if self.evidence.is_empty() {
            return match (&self.reasoning, figures) {
                (Some(r), _) => Status::Inferred(r.clone()),
                (None, true) => Status::Unknown("chiffre sans run associé".into()),
                (None, false) => Status::Unknown("ni run ni raisonnement".into()),
            };
        }
        let mut runs = Vec::new();
        for ev in &self.evidence {
            let Some(entry) = ledger.iter().find(|e| e.run_id == ev.run_id) else {
                return Status::Unknown(format!("run {} absent du journal", ev.run_id));
            };
            if entry.verdict != Verdict::Ok {
                return Status::Unknown(format!("run {} : {:?}", ev.run_id, entry.verdict));
            }
            if let Some(v) = ev.value {
                if !contains_number(&entry.result, v) {
                    return Status::Unknown(format!(
                        "valeur {v} absente du résultat de {}",
                        ev.run_id
                    ));
                }
            }
            runs.push((entry.run_id.clone(), entry.hashes.model.clone()));
        }
        if figures && self.evidence.iter().all(|e| e.value.is_none()) {
            return Status::Unknown("chiffre sans valeur liée à un run".into());
        }
        let heuristic = self.evidence.iter().any(|ev| {
            ledger
                .iter()
                .find(|e| e.run_id == ev.run_id)
                .is_some_and(|e| is_heuristic(&e.result))
        });
        match heuristic {
            true => Status::Heuristic(runs),
            false => Status::Measured(runs),
        }
    }
}

/// What the engine says of a result that is a screening, not a proof.
fn is_heuristic(result: &Value) -> bool {
    result["is_heuristic"] == true || result["evidence_level"] == "heuristic_screening"
}

fn contains_number(v: &Value, x: f64) -> bool {
    match v {
        Value::Number(n) => n
            .as_f64()
            .is_some_and(|n| (n - x).abs() <= 1e-9 + 1e-6 * x.abs()),
        Value::Array(a) => a.iter().any(|v| contains_number(v, x)),
        Value::Object(o) => o.values().any(|v| contains_number(v, x)),
        _ => false,
    }
}

pub fn append(path: &Path, finding: &Finding) -> Result<(), String> {
    let mut line = serde_json::to_vec(finding).map_err(|e| e.to_string())?;
    line.push(b'\n');
    use std::io::Write;
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(&line))
        .map_err(|e| format!("{} : {e}", path.display()))
}

/// The runs a CLI agent made through the engine's MCP server (`chat/mcp_runs.jsonl`), as
/// ledger entries a finding can cite: the run id the server returned, its verdict and data.
pub fn mcp_runs(path: &Path) -> Vec<Entry> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .filter_map(|run| {
            Some(Entry {
                run_id: run["run_id"].as_str()?.to_owned(),
                tool: run["tool"].as_str().unwrap_or_default().to_owned(),
                label: "MCP".into(),
                params: run["arguments"].clone(),
                resumed_from: None,
                started_unix: run["started_unix"].as_u64().unwrap_or_default(),
                finished_unix: run["finished_unix"].as_u64().unwrap_or_default(),
                hashes: Hashes::default(),
                versions: Versions::default(),
                verdict: match run["ok"].as_bool() {
                    Some(true) => Verdict::Ok,
                    _ => Verdict::Failed,
                },
                exit_code: run["exit_code"].as_i64().unwrap_or(1) as i32,
                result: run["data"].clone(),
                error: None,
                elapsed_ms: None,
            })
        })
        .collect()
}

pub fn read(path: &Path) -> Vec<Finding> {
    std::fs::read_to_string(path)
        .unwrap_or_default()
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::studio::ledger::{Hashes, Versions};

    fn entry(id: &str, verdict: Verdict, result: Value) -> Entry {
        Entry {
            run_id: id.into(),
            tool: "surgeon_bench".into(),
            label: "".into(),
            params: json!({}),
            resumed_from: None,
            started_unix: 0,
            finished_unix: 0,
            hashes: Hashes {
                model: Some("fast-abc".into()),
                ..Hashes::default()
            },
            versions: Versions::default(),
            verdict,
            exit_code: 0,
            result,
            error: None,
            elapsed_ms: None,
        }
    }

    fn finding(args: Value) -> Finding {
        Finding::from_args(&args, "c1").unwrap()
    }

    #[test]
    fn a_figure_without_a_run_is_never_measured() {
        let ledger = vec![
            entry("r1", Verdict::Ok, json!({"refusal_rate": 0.31})),
            entry("r2", Verdict::Failed, json!({})),
        ];
        let f = finding(json!({"text": "le refus tombe à 0.31"}));
        assert!(matches!(f.status(&ledger), Status::Unknown(_)));
        let f =
            finding(json!({"text": "le refus tombe à 0.31", "reasoning": "vu au run précédent"}));
        assert!(matches!(f.status(&ledger), Status::Inferred(_)));
        let f =
            finding(json!({"text": "refus 0.31", "evidence": [{"run_id": "r1", "value": 0.31}]}));
        assert_eq!(
            f.status(&ledger),
            Status::Measured(vec![("r1".into(), Some("fast-abc".into()))])
        );
        // A value the run does not hold, an unknown run, a failed run, a figure with no value.
        for ev in [
            json!([{"run_id": "r1", "value": 0.5}]),
            json!([{"run_id": "r9", "value": 0.31}]),
            json!([{"run_id": "r2", "value": 0.31}]),
            json!([{"run_id": "r1"}]),
        ] {
            let f = finding(json!({"text": "refus 0.31", "evidence": ev}));
            assert!(matches!(f.status(&ledger), Status::Unknown(_)), "{ev}");
        }
        // The status follows the ledger: the same finding without its run is not measured.
        let f =
            finding(json!({"text": "refus 0.31", "evidence": [{"run_id": "r1", "value": 0.31}]}));
        assert!(matches!(f.status(&[]), Status::Unknown(_)));
    }

    #[test]
    fn a_cli_finding_is_measured_by_the_runs_the_mcp_server_journaled() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp_runs.jsonl");
        // The lines the engine's MCP server writes (weight_surgeon src/mcp/tools_session.rs).
        let runs = [
            json!({"run_id": "mcp-1-0", "tool": "surgeon_bench", "ok": true, "exit_code": 0,
                   "data": {"refusal_rate": 0.4}, "truncated": false,
                   "started_unix": 1, "finished_unix": 2}),
            json!({"run_id": "mcp-1-1", "tool": "surgeon_bench", "ok": false, "exit_code": 2,
                   "data": null, "truncated": false, "started_unix": 3, "finished_unix": 4}),
        ];
        let text: String = runs.iter().map(|r| format!("{r}\n")).collect();
        std::fs::write(&path, text + "not json\n").unwrap();
        let ledger = mcp_runs(&path);
        assert_eq!(ledger.len(), 2);
        assert_eq!(ledger[1].verdict, Verdict::Failed);
        let f = finding(
            json!({"text": "refus à 0.4", "evidence": [{"run_id": "mcp-1-0", "value": 0.4}]}),
        );
        assert!(matches!(f.status(&ledger), Status::Measured(_)));
        let f = finding(
            json!({"text": "refus à 0.4", "evidence": [{"run_id": "mcp-1-1", "value": 0.4}]}),
        );
        assert!(matches!(f.status(&ledger), Status::Unknown(_)));
        assert!(mcp_runs(&dir.path().join("absent.jsonl")).is_empty());
    }

    #[test]
    fn a_probe_run_backs_a_heuristic_never_a_measure() {
        let probe = json!({"is_heuristic": true, "evidence_level": "heuristic_screening",
                           "probe_refusal_rate": 0.12, "mean_refusal_prob": 0.2});
        let ledger = vec![
            entry("probe", Verdict::Ok, probe),
            entry("gen", Verdict::Ok, json!({"refusal_rate": 0.1})),
        ];
        let f = finding(
            json!({"text": "refus sondé 0.12", "evidence": [{"run_id": "probe", "value": 0.12}]}),
        );
        let status = f.status(&ledger);
        assert!(matches!(status, Status::Heuristic(_)), "{status:?}");
        assert_eq!(status.label(), "Heuristique");
        // One heuristic run among the cited ones is enough to keep it a heuristic.
        let f = finding(json!({"text": "refus 0.1 puis 0.12", "evidence": [
            {"run_id": "gen", "value": 0.1}, {"run_id": "probe", "value": 0.12}]}));
        assert!(matches!(f.status(&ledger), Status::Heuristic(_)));
        // The full generation alone is the measure.
        let f =
            finding(json!({"text": "refus 0.1", "evidence": [{"run_id": "gen", "value": 0.1}]}));
        assert!(matches!(f.status(&ledger), Status::Measured(_)));
        // Through the MCP journal too.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("mcp_runs.jsonl");
        let run = json!({"run_id": "mcp-2-0", "tool": "surgeon_bench", "ok": true, "exit_code": 0,
                         "data": {"is_heuristic": true, "probe_refusal_rate": 0.3}});
        std::fs::write(&path, format!("{run}\n")).unwrap();
        let f = finding(
            json!({"text": "refus 0.3", "evidence": [{"run_id": "mcp-2-0", "value": 0.3}]}),
        );
        assert!(matches!(f.status(&mcp_runs(&path)), Status::Heuristic(_)));
    }
}
