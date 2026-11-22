//! What the agent already tried in the project, read from the ledger and from the MCP journal
//! of a CLI agent: a compact summary for its prompt, so that it neither runs a configuration
//! twice nor raises alpha after a loss of retention; the detection of an exact repeat, answered
//! with the earlier run instead of a new one; and the rule that a heuristic probe never closes
//! a task without a full generation on the final model.

use std::collections::HashSet;
use std::path::Path;

use serde::Serialize;
use serde_json::Value;

use super::evidence;
use super::provider::{Block, Message};
use super::session::Dirs;
use crate::studio::ledger::{self, Entry, Hashes, Verdict};

/// Tools whose run is set by its arguments, the model state and the input files (greedy
/// decoding, fixed seeds): running one again with all three unchanged teaches nothing.
pub const REPEATABLE: [&str; 5] = [
    "surgeon_apply",
    "surgeon_bench",
    "surgeon_autotune",
    "surgeon_compose",
    "surgeon_extract",
];

/// Runs listed in the prompt: the newest ones.
const SUMMARY_RUNS: usize = 30;

/// What the studio tells an agent that closes on a heuristic probe alone.
pub const CERTIFY_REMINDER: &str = "[Rappel du studio] Tu t'apprêtes à conclure alors que la \
     seule mesure du comportement sur le modèle dans son état final est une sonde \
     (surgeon_bench mode refusal_probe, is_heuristic: true) : c'est un criblage, pas une preuve. \
     Avant de déclarer un candidat gagnant ou de clore l'objectif, certifie working_copy_path \
     par génération complète : surgeon_bench mode refusal (builtin:refusals) et mode eval \
     (builtin:math, builtin:logic, builtin:general), chacun avec baseline_model = origin_path. \
     Si tu ne peux pas, dis explicitement que le résultat n'est pas certifié.";

fn file_name(v: &Value) -> String {
    v.as_str()
        .map(|s| {
            Path::new(s)
                .file_name()
                .map_or(s.to_owned(), |n| n.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "?".into())
}

fn rate(v: &Value) -> Option<String> {
    v.as_f64().map(|x| format!("{x:.3}"))
}

/// One line for an experiment run (patch, composition, autotune, measure); `None` otherwise.
pub fn describe(e: &Entry) -> Option<String> {
    let (p, r) = (&e.params, &e.result);
    let model = file_name(&p["model"]);
    let text = match e.tool.as_str() {
        "surgeon_apply" if p["dry_run"] != true => {
            let ops: Vec<String> = p["patch"]["operations"]
                .as_array()
                .into_iter()
                .flatten()
                .map(|op| {
                    format!(
                        "{} couches {} {} alpha {} ({})",
                        op["operation"].as_str().unwrap_or("?"),
                        op["layers"],
                        op["projections"],
                        op["alpha"],
                        file_name(&op["vector_file"])
                    )
                })
                .collect();
            let what = match ops.is_empty() {
                true => format!("patch {}", file_name(&p["patch"])),
                false => ops.join(" ; "),
            };
            format!("patch sur {model} : {what}")
        }
        "surgeon_restore" => format!("restauration de {model}"),
        "surgeon_compose" => {
            let names = |v: &Value| {
                v.as_array()
                    .into_iter()
                    .flatten()
                    .map(file_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            format!(
                "composition {} : retire {} ; garde {}",
                p["mode"].as_str().unwrap_or("orthogonal_preserve"),
                names(&p["ablate"]),
                names(&p["keep"])
            )
        }
        "surgeon_extract" => format!(
            "extraction couche {} ({}) -> {}",
            p["layer"],
            p["method"].as_str().unwrap_or("mean_diff"),
            file_name(&p["out"])
        ),
        "surgeon_autotune" => match r["selection"]["selected_scale"].as_f64() {
            Some(c) => format!(
                "autotune {} : échelle {c} retenue par génération",
                file_name(&p["vector"])
            ),
            None => format!(
                "autotune {} : alpha du ridge, sans échelle validée en génération",
                file_name(&p["vector"])
            ),
        },
        "surgeon_bench" => match p["mode"].as_str()? {
            "refusal_probe" => format!(
                "sonde refusal_probe sur {model} : refus sondé {} (heuristique)",
                rate(&r["probe_refusal_rate"])?
            ),
            // The reports flatten their score: `refusal_rate`, `baseline.refusal_rate`.
            "refusal" => format!(
                "bench refusal sur {model} : refus {}{}",
                rate(&r["refusal_rate"])?,
                rate(&r["baseline"]["refusal_rate"])
                    .map_or(String::new(), |b| format!(" (base {b})"))
            ),
            "eval" => {
                let after = r["accuracy"].as_f64()?;
                let before = r["baseline"]["accuracy"].as_f64();
                format!(
                    "bench eval {} sur {model} : exactitude {after:.3}{}",
                    file_name(&p["dataset"]),
                    match before {
                        Some(b) if after < b => format!(" (base {b:.3}, ↓ rétention)"),
                        Some(b) => format!(" (base {b:.3})"),
                        None => String::new(),
                    }
                )
            }
            _ => return None,
        },
        _ => return None,
    };
    Some(format!("- {} : {text}", e.run_id))
}

/// The successful experiment runs of `entries`, the newest [`SUMMARY_RUNS`] of them, with
/// the two rules they serve; empty when there are none.
pub fn summary(entries: &[Entry]) -> String {
    let lines: Vec<String> = entries
        .iter()
        .filter(|e| e.verdict == Verdict::Ok)
        .filter_map(describe)
        .collect();
    if lines.is_empty() {
        return String::new();
    }
    let skipped = lines.len().saturating_sub(SUMMARY_RUNS);
    format!(
        "Essais déjà faits dans ce projet (du plus ancien au plus récent{}) :\n{}\n\
         Ne relance pas une configuration identique (mêmes arguments sur le même état du \
         modèle) : le studio répond alors avec le run existant. Après une baisse de \
         rétention (↓), n'augmente pas alpha sur les mêmes couches : réduis-le, change de \
         couches ou compose la direction avec surgeon_compose.",
        match skipped {
            0 => String::new(),
            n => format!(", {n} plus anciens omis"),
        },
        lines[skipped..].join("\n")
    )
}

/// [`summary`] of the project: its ledger and the MCP journal of its CLI agents, in time order.
pub fn project_summary(project: &Path) -> String {
    let mut entries = ledger::read(&project.join("ledger.jsonl")).unwrap_or_default();
    entries.extend(evidence::mcp_runs(&Dirs::of(project).mcp_runs()));
    entries.sort_by_key(|e| e.started_unix);
    summary(&entries)
}

/// One bench run as the exported report lists it: its scores and the strength of its evidence.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ExperimentRun {
    /// `run_id` of the ledger or of the MCP journal.
    pub run: String,
    pub tool: String,
    /// Refusal rate: measured by generation, or screened by the probe.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal_rate: Option<f64>,
    /// Accuracy of the `eval` mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_accuracy: Option<f64>,
    /// `heuristic` for a probe, `measure` for a full generation.
    pub status: &'static str,
}

/// Whether the run is an exploration probe rather than a full generation.
fn is_heuristic(e: &Entry) -> bool {
    e.result["is_heuristic"] == true
        || e.result["evidence_level"] == "heuristic_screening"
        || e.params["mode"] == "refusal_probe"
}

/// The `surgeon_bench` runs of `entries` that carry a score, oldest first.
pub fn experiment_runs(entries: &[Entry]) -> Vec<ExperimentRun> {
    entries
        .iter()
        .filter(|e| e.tool == "surgeon_bench")
        .filter_map(|e| {
            let refusal_rate = e.result["refusal_rate"]
                .as_f64()
                .or_else(|| e.result["probe_refusal_rate"].as_f64());
            let eval_accuracy = e.result["accuracy"].as_f64();
            (refusal_rate.is_some() || eval_accuracy.is_some()).then(|| ExperimentRun {
                run: e.run_id.clone(),
                tool: e.tool.clone(),
                refusal_rate,
                eval_accuracy,
                status: if is_heuristic(e) {
                    "heuristic"
                } else {
                    "measure"
                },
            })
        })
        .collect()
}

fn score(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_owned(), |x| format!("{x:.3}"))
}

/// A Markdown report of the project's bench runs, for a human: the legend of the two kinds of
/// evidence, then one row per run. Pure: nothing is written.
pub fn export_experiments_markdown(title: &str, entries: &[Entry]) -> String {
    let title = match title.trim() {
        "" => "projet",
        title => title,
    };
    let runs = experiment_runs(entries);
    let mut out = format!(
        "# Bilan des essais : {title}\n\n\
         ## Comment lire ce tableau\n\n\
         - **Heuristique** : une sonde rapide sur les activations, comme prendre la température \
         en deux secondes. Elle oriente, elle ne prouve rien.\n\
         - **Mesure** : une génération complète de texte, le vrai test sur route.\n\n"
    );
    if runs.is_empty() {
        out.push_str(
            "Aucun essai mesuré dans ce projet : lancez `surgeon_bench` (modes `refusal`, `eval` \
             ou `refusal_probe`), puis exportez à nouveau.\n",
        );
        return out;
    }
    out.push_str(
        "| Run | Outil | Score refus | Score eval | Statut |\n\
         | --- | --- | --- | --- | --- |\n",
    );
    for run in &runs {
        let status = match run.status {
            "heuristic" => "Heuristique",
            _ => "Mesure",
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {status} |\n",
            run.run,
            run.tool,
            score(run.refusal_rate),
            score(run.eval_accuracy),
        ));
    }
    out
}

/// The same runs as indented JSON, for a script.
pub fn export_experiments_json(entries: &[Entry]) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&experiment_runs(entries))
}

/// The finished run that `tool` with `args` would repeat: same arguments, same model state,
/// and every input file it read unchanged (a file it wrote itself, absent then, is ignored).
pub fn already_tried<'a>(
    tool: &str,
    args: &Value,
    now: &Hashes,
    entries: &'a [Entry],
) -> Option<&'a Entry> {
    if !REPEATABLE.contains(&tool) {
        return None;
    }
    entries.iter().rev().find(|e| {
        e.verdict == Verdict::Ok
            && e.tool == tool
            && &e.params == args
            && e.hashes.model == now.model
            && e.hashes
                .inputs
                .iter()
                .all(|(k, v)| now.inputs.get(k) == Some(v))
    })
}

/// Whether `messages` (one execution) used a heuristic probe without a full generation
/// (`surgeon_bench` mode refusal or eval) on the model in its final state: closing then would
/// rest on a screening.
pub fn uncertified(messages: &[Message]) -> bool {
    let blocks = || messages.iter().flat_map(|m| &m.blocks);
    let succeeded: HashSet<&str> = blocks()
        .filter_map(|b| match b {
            Block::ToolResult {
                id,
                is_error: false,
                ..
            } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    let (mut probed, mut certified) = (false, false);
    for b in blocks() {
        let Block::ToolUse { id, name, input } = b else {
            continue;
        };
        if !succeeded.contains(id.as_str()) {
            continue;
        }
        let mode = input["mode"].as_str();
        match name.as_str() {
            "surgeon_bench" if mode == Some("refusal_probe") => probed = true,
            "surgeon_bench" if matches!(mode, Some("refusal" | "eval")) => certified = true,
            // A write makes the earlier measures those of another model.
            "surgeon_apply" if input["dry_run"] != true => certified = false,
            "surgeon_lora" if mode == Some("merge") && input["dry_run"] != true => {
                certified = false
            }
            "surgeon_restore" => certified = false,
            _ => {}
        }
    }
    probed && !certified
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::studio::ledger::Versions;

    fn entry(id: &str, tool: &str, params: Value, result: Value, model: &str) -> Entry {
        Entry {
            run_id: id.into(),
            tool: tool.into(),
            label: String::new(),
            params,
            resumed_from: None,
            started_unix: 0,
            finished_unix: 0,
            hashes: Hashes {
                model: Some(model.into()),
                ..Hashes::default()
            },
            versions: Versions::default(),
            verdict: Verdict::Ok,
            exit_code: 0,
            result,
            error: None,
        }
    }

    fn apply(alpha: f64) -> Value {
        json!({"model": "copies/c1", "patch": {"schema_version": 1, "operations": [{
            "id": "po", "operation": "project_out", "layers": "12-16", "alpha": alpha,
            "projections": ["o_proj", "down_proj"], "vector_file": "work/dir_L14.safetensors"}]}})
    }

    #[test]
    fn the_summary_lists_layers_alpha_mode_and_probe_against_measure() {
        let entries = [
            entry("r1", "surgeon_apply", apply(0.8), json!({}), "fast-a"),
            entry(
                "r2",
                "surgeon_bench",
                json!({"mode": "refusal_probe", "model": "copies/c1"}),
                json!({"is_heuristic": true, "probe_refusal_rate": 0.05}),
                "fast-b",
            ),
            entry(
                "r3",
                "surgeon_bench",
                json!({"mode": "refusal", "model": "copies/c1"}),
                json!({"refusal_rate": 0.11, "baseline": {"refusal_rate": 0.4}}),
                "fast-b",
            ),
            entry(
                "r4",
                "surgeon_bench",
                json!({"mode": "eval", "model": "copies/c1", "dataset": "builtin:math"}),
                json!({"accuracy": 0.35, "baseline": {"accuracy": 0.5}}),
                "fast-b",
            ),
            entry(
                "r5",
                "surgeon_compose",
                json!({"ablate": ["work/dir_L14.safetensors"], "keep": ["work/math_L14.safetensors"],
                       "mode": "orthogonal_preserve"}),
                json!({}),
                "",
            ),
            entry(
                "r6",
                "surgeon_check",
                json!({"model": "copies/c1"}),
                json!({}),
                "fast-b",
            ),
        ];
        let text = summary(&entries);
        for needle in [
            "r1 : patch sur c1 : project_out couches \"12-16\" [\"o_proj\",\"down_proj\"] alpha 0.8 (dir_L14.safetensors)",
            "r2 : sonde refusal_probe sur c1 : refus sondé 0.050 (heuristique)",
            "r3 : bench refusal sur c1 : refus 0.110 (base 0.400)",
            "r4 : bench eval builtin:math sur c1 : exactitude 0.350 (base 0.500, ↓ rétention)",
            "r5 : composition orthogonal_preserve : retire dir_L14.safetensors ; garde math_L14.safetensors",
            "n'augmente pas alpha",
        ] {
            assert!(text.contains(needle), "{needle} in {text}");
        }
        assert!(!text.contains("r6"), "not an experiment");
        assert_eq!(summary(&entries[5..]), "");
    }

    #[test]
    fn a_repeat_needs_the_same_arguments_model_state_and_inputs() {
        let mut first = entry("r1", "surgeon_apply", apply(0.8), json!({}), "fast-a");
        first.hashes.inputs.insert("vector".into(), "sha-v".into());
        let entries = [first];
        let now = |model: &str, inputs: &[(&str, &str)]| Hashes {
            model: Some(model.into()),
            inputs: inputs
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            ..Hashes::default()
        };
        let same = now("fast-a", &[("vector", "sha-v"), ("out", "written-by-r1")]);
        assert_eq!(
            already_tried("surgeon_apply", &apply(0.8), &same, &entries).map(|e| &e.run_id[..]),
            Some("r1")
        );
        // Another alpha, another model state (after an apply), an input that changed.
        assert!(already_tried("surgeon_apply", &apply(0.6), &same, &entries).is_none());
        assert!(already_tried(
            "surgeon_apply",
            &apply(0.8),
            &now("fast-b", &[("vector", "sha-v")]),
            &entries
        )
        .is_none());
        assert!(already_tried(
            "surgeon_apply",
            &apply(0.8),
            &now("fast-a", &[("vector", "sha-w")]),
            &entries
        )
        .is_none());
        // Never for a tool whose result is not set by its arguments.
        assert!(already_tried("surgeon_files", &apply(0.8), &same, &entries).is_none());
    }

    #[test]
    fn the_project_summary_reads_the_ledger_and_the_mcp_journal() {
        let dir = tempfile::tempdir().unwrap();
        let applied = entry("r1", "surgeon_apply", apply(0.6), json!({}), "fast-a");
        ledger::append(&dir.path().join("ledger.jsonl"), &applied).unwrap();
        let chat = dir.path().join("chat");
        std::fs::create_dir_all(&chat).unwrap();
        // A CLI agent's run, journaled by the engine's MCP server with its arguments.
        let line = json!({"run_id": "mcp-1-0", "tool": "surgeon_bench", "ok": true,
            "exit_code": 0, "arguments": {"mode": "refusal_probe", "model": "copies/c1"},
            "data": {"probe_refusal_rate": 0.07}, "started_unix": 5, "finished_unix": 6});
        std::fs::write(chat.join("mcp_runs.jsonl"), format!("{line}\n")).unwrap();
        let text = project_summary(dir.path());
        let (at_apply, at_probe) = (
            text.find("r1 : patch sur c1").unwrap(),
            text.find("mcp-1-0 : sonde refusal_probe sur c1 : refus sondé 0.070")
                .unwrap(),
        );
        assert!(at_apply < at_probe, "time order: {text}");
        assert_eq!(project_summary(&dir.path().join("absent")), "");
    }

    fn call(id: &str, name: &str, input: Value, ok: bool) -> [Message; 2] {
        [
            Message {
                role: super::super::provider::Role::Assistant,
                blocks: vec![Block::ToolUse {
                    id: id.into(),
                    name: name.into(),
                    input,
                }],
            },
            Message {
                role: super::super::provider::Role::User,
                blocks: vec![Block::ToolResult {
                    id: id.into(),
                    content: "…".into(),
                    is_error: !ok,
                }],
            },
        ]
    }

    #[test]
    fn a_probe_needs_a_full_generation_after_the_last_write() {
        let probe = call(
            "t1",
            "surgeon_bench",
            json!({"mode": "refusal_probe"}),
            true,
        );
        let refusal = call("t2", "surgeon_bench", json!({"mode": "refusal"}), true);
        let apply = call("t3", "surgeon_apply", json!({"dry_run": false}), true);
        let dry = call("t4", "surgeon_apply", json!({"dry_run": true}), true);
        let failed = call("t5", "surgeon_bench", json!({"mode": "eval"}), false);
        let run = |parts: &[&[Message; 2]]| -> bool {
            uncertified(
                &parts
                    .iter()
                    .flat_map(|p| p.iter().cloned())
                    .collect::<Vec<_>>(),
            )
        };
        assert!(!run(&[]), "nothing probed");
        assert!(run(&[&probe]));
        assert!(!run(&[&probe, &refusal]));
        assert!(!run(&[&refusal, &probe]), "same model state");
        assert!(
            run(&[&probe, &refusal, &apply]),
            "measured before the last write"
        );
        assert!(
            !run(&[&apply, &probe, &dry, &refusal]),
            "a dry run writes nothing"
        );
        assert!(run(&[&probe, &failed]), "a failed bench certifies nothing");
    }

    #[test]
    fn the_export_says_when_there_is_nothing_to_report() {
        let markdown = export_experiments_markdown("qwen-0.5b-refus", &[]);
        assert!(markdown.contains("# Bilan des essais : qwen-0.5b-refus"));
        assert!(markdown.contains("Aucun essai mesuré"), "{markdown}");
        assert!(!markdown.contains("| Run |"), "{markdown}");
        let json: Value = serde_json::from_str(&export_experiments_json(&[]).unwrap()).unwrap();
        assert_eq!(json, json!([]));
    }

    #[test]
    fn the_export_keeps_probes_and_measures_apart() {
        let entries = [
            entry(
                "r1",
                "surgeon_bench",
                json!({"mode": "refusal_probe", "model": "copies/c1"}),
                json!({"is_heuristic": true, "probe_refusal_rate": 0.85}),
                "fast-a",
            ),
            entry(
                "r2",
                "surgeon_bench",
                json!({"mode": "refusal", "model": "copies/c1"}),
                json!({"refusal_rate": 0.12, "baseline": {"refusal_rate": 0.85}}),
                "fast-b",
            ),
            entry(
                "r3",
                "surgeon_bench",
                json!({"mode": "eval", "model": "copies/c1", "dataset": "builtin:math"}),
                json!({"accuracy": 0.64, "baseline": {"accuracy": 0.85}}),
                "fast-b",
            ),
            entry("r4", "surgeon_apply", json!({}), json!({}), "fast-b"),
        ];
        let markdown = export_experiments_markdown("Qwen2.5-0.5B-Instruct", &entries);
        for row in [
            "| r1 | surgeon_bench | 0.850 | - | Heuristique |",
            "| r2 | surgeon_bench | 0.120 | - | Mesure |",
            "| r3 | surgeon_bench | - | 0.640 | Mesure |",
        ] {
            assert!(markdown.contains(row), "{row} in {markdown}");
        }
        assert!(!markdown.contains("r4"), "not a bench run: {markdown}");
        let json: Value =
            serde_json::from_str(&export_experiments_json(&entries).unwrap()).unwrap();
        let rows = json.as_array().unwrap();
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0]["status"], "heuristic");
        assert_eq!(rows[0]["refusal_rate"], 0.85);
        assert!(rows[0].get("eval_accuracy").is_none(), "{json}");
        assert_eq!(rows[2]["eval_accuracy"], 0.64);
        assert_eq!(rows[2]["status"], "measure");
    }
}
