//! The results of the project's measuring runs, as the Results panel shows them: every
//! `surgeon_bench`, `surgeon_graph`, `surgeon_extract`, `surgeon_autotune` and
//! `surgeon_lora` run of the ledger (studio jobs) and of `chat/mcp_runs.jsonl` (CLI agents),
//! newest first. A bench run is read from its report: dataset, score of the copy against the
//! origin with their 95 % intervals, gained and lost items, and each item with its question
//! when the set is known. A LoRA extraction gives its rank and the fidelity of its SVD.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::Value;
use sha2::{Digest, Sha256};
use weight_surgeon::bench::{builtin, dataset};

use super::jobs::{Budgets, JobSpec};
use super::ledger::{Entry, Verdict};

/// The tools whose runs the panel lists.
pub const TOOLS: [&str; 5] = [
    "surgeon_bench",
    "surgeon_graph",
    "surgeon_extract",
    "surgeon_autotune",
    "surgeon_lora",
];

/// The ranks the Results panel offers for a LoRA extraction.
pub const LORA_RANKS: [u32; 4] = [8, 16, 32, 64];

/// The job that extracts the working copy `copy` against the origin `origin` as a rank-`rank`
/// PEFT adapter, written in `out/lora-r<rank>/` of the project (created here) and replaced
/// by the next extraction of that rank. The origin is only read.
pub fn lora_job(project: &Path, origin: &Path, copy: &Path, rank: u32) -> Result<JobSpec, String> {
    let out = project.join("out").join(format!("lora-r{rank}"));
    std::fs::create_dir_all(&out).map_err(|e| format!("{} : {e}", out.display()))?;
    Ok(JobSpec {
        tool: "surgeon_lora".into(),
        label: format!("LoRA r{rank}"),
        args: serde_json::json!({"mode": "extract", "base": origin, "patched": copy,
                                 "out": out, "rank": rank, "overwrite": true}),
        priority: 0,
        budgets: Budgets::default(),
        resumed_from: None,
        read_only: false,
    })
}

/// What a LoRA extraction wrote.
#[derive(Debug, Clone, PartialEq)]
pub struct Lora {
    pub rank: u64,
    pub tensors: u64,
    /// `1 − mean ‖ΔW − BA‖F / ‖ΔW‖F` over the tensors: how much of the difference the
    /// adapter rebuilds.
    pub fidelity: f64,
    /// The tensor the rank fits worst, and its relative error.
    pub worst: Option<(String, f64)>,
    pub out: PathBuf,
}

/// `data.lora` of an extraction (a merge reports no error: `None`).
fn lora(data: &Value) -> Option<Lora> {
    let l = &data["lora"];
    let worst = l["per_tensor"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| Some((t["name"].as_str()?.to_owned(), t["rel_error"].as_f64()?)))
        .max_by(|a, b| a.1.total_cmp(&b.1));
    Some(Lora {
        rank: l["rank"].as_u64()?,
        tensors: l["tensors"].as_u64()?,
        fidelity: 1.0 - l["mean_rel_error"].as_f64()?,
        worst,
        out: PathBuf::from(l["out"].as_str()?),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub id: String,
    pub tool: String,
    pub verdict: Verdict,
    pub finished_unix: u64,
    pub millis: Option<u64>,
    pub error: Option<String>,
    /// The graph file a `surgeon_graph` run wrote, still on disk.
    pub graph: Option<PathBuf>,
    pub bench: Option<Bench>,
    pub lora: Option<Lora>,
    pub ppl: Option<Ppl>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ppl {
    pub before: Option<f64>,
    pub after: f64,
    pub delta_pct: Option<f64>,
}

fn ppl(data: &Value) -> Option<Ppl> {
    Some(Ppl {
        before: data["baseline"]["ppl"].as_f64(),
        after: data["ppl"].as_f64()?,
        delta_pct: data["delta"]["delta_relative_pct"].as_f64(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Accuracy on questions with a gold answer.
    Eval,
    /// Refusal rate on prompts.
    Refusal,
}

/// A rate and its Wilson 95 % interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub n: u64,
    pub rate: f64,
    pub low: f64,
    pub high: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Bench {
    pub mode: Mode,
    /// `builtin:math`, a dataset path, or `None` when the report cannot tell.
    pub dataset: Option<String>,
    /// The model measured (the working copy), then the baseline (the origin).
    pub after: Score,
    pub before: Option<Score>,
    /// Items the surgery fixed (eval: now right; refusal: no longer refused) and broke.
    pub gained: usize,
    pub lost: usize,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub id: String,
    pub question: Option<String>,
    /// The reference answer (eval only).
    pub gold: Option<String>,
    pub after: Answer,
    pub before: Option<Answer>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// The extracted answer (eval) or the reply (refusal).
    pub text: String,
    /// Right (eval) or not refused (refusal).
    pub good: bool,
}

/// The listed runs of `entries`, newest first.
pub fn runs<'a>(entries: impl IntoIterator<Item = &'a Entry>) -> Vec<Run> {
    let mut runs: Vec<Run> = entries
        .into_iter()
        .filter(|e| TOOLS.contains(&e.tool.as_str()))
        .map(|e| Run {
            id: e.run_id.clone(),
            tool: e.tool.clone(),
            verdict: e.verdict,
            finished_unix: e.finished_unix,
            millis: e.elapsed_ms.or_else(|| {
                (e.started_unix > 0 && e.finished_unix > e.started_unix)
                    .then(|| (e.finished_unix - e.started_unix) * 1000)
            }),
            error: e.error.clone(),
            graph: graph_output(e),
            bench: (e.tool == "surgeon_bench")
                .then(|| bench(&e.result, &e.params))
                .flatten(),
            lora: (e.tool == "surgeon_lora")
                .then(|| lora(&e.result))
                .flatten(),
            ppl: (e.tool == "surgeon_bench")
                .then(|| ppl(&e.result))
                .flatten(),
        })
        .collect();
    runs.sort_by(|a, b| b.finished_unix.cmp(&a.finished_unix).then(b.id.cmp(&a.id)));
    runs
}

/// The newest successful `surgeon_graph` run that left a graph on disk: `(run id, file)`.
pub fn newest_graph<'a>(entries: impl IntoIterator<Item = &'a Entry>) -> Option<(String, PathBuf)> {
    entries
        .into_iter()
        .filter(|e| e.tool == "surgeon_graph" && e.verdict == Verdict::Ok)
        .filter_map(|e| Some((e.finished_unix, e.run_id.clone(), graph_output(e)?)))
        .max_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)))
        .map(|(_, id, path)| (id, path))
}

/// `graph.json` among the run's outputs, else `graph.ndjson`, if the file is still there.
fn graph_output(e: &Entry) -> Option<PathBuf> {
    let outputs: Vec<PathBuf> = e.result["outputs"]
        .as_array()?
        .iter()
        .filter_map(|p| p.as_str().map(PathBuf::from))
        .collect();
    ["graph.json", "graph.ndjson"].into_iter().find_map(|name| {
        outputs
            .iter()
            .find(|p| p.file_name().is_some_and(|f| f == name) && p.is_file())
            .cloned()
    })
}

fn score(v: &Value, rate: &str) -> Option<Score> {
    Some(Score {
        n: v["n"].as_u64()?,
        rate: v[rate].as_f64()?,
        low: v["ci95_low"].as_f64()?,
        high: v["ci95_high"].as_f64()?,
    })
}

fn count(v: &Value) -> usize {
    v.as_array().map_or(0, Vec::len)
}

/// An eval or refusal report; other modes (logits, ppl, dump...) are listed without detail.
fn bench(data: &Value, params: &Value) -> Option<Bench> {
    if data.get("accuracy").is_some() {
        return eval(data);
    }
    data.get("refusal_rate").and_then(|_| refusal(data, params))
}

fn eval(data: &Value) -> Option<Bench> {
    let dataset = data["dataset"].as_str().map(str::to_owned);
    let questions: HashMap<String, String> = dataset
        .as_deref()
        .and_then(|d| dataset::load(Path::new(d), dataset::DatasetFormat::Auto, usize::MAX).ok())
        .into_iter()
        .flatten()
        .map(|i| (i.id, i.question))
        .collect();
    let answer = |item: &Value| Answer {
        text: item["extracted"].as_str().unwrap_or("-").to_owned(),
        good: item["correct"].as_bool().unwrap_or(false),
    };
    let baseline = &data["baseline"];
    let before: HashMap<&str, Answer> = baseline["items"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|i| Some((i["id"].as_str()?, answer(i))))
        .collect();
    let items = data["items"]
        .as_array()?
        .iter()
        .filter_map(|i| {
            let id = i["id"].as_str()?;
            Some(Item {
                id: id.to_owned(),
                question: questions.get(id).cloned(),
                gold: i["gold"].as_str().map(str::to_owned),
                after: answer(i),
                before: before.get(id).cloned(),
            })
        })
        .collect();
    Some(Bench {
        mode: Mode::Eval,
        dataset,
        after: score(data, "accuracy")?,
        before: score(baseline, "accuracy"),
        gained: count(&data["paired"]["gained"]),
        lost: count(&data["paired"]["lost"]),
        items,
    })
}

type PromptSet = (String, String, Vec<String>);

fn builtin_prompt_sets() -> &'static [PromptSet] {
    static SETS: OnceLock<Vec<PromptSet>> = OnceLock::new();
    SETS.get_or_init(|| {
        builtin::names()
            .split(", ")
            .filter_map(|name| {
                let prompts = builtin::prompts(Path::new(name))?.ok()?;
                let json = serde_json::to_string(&prompts).ok()?;
                let sha: String = Sha256::digest(json.as_bytes())
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                Some((name.to_owned(), sha, prompts))
            })
            .collect()
    })
}

// The report keeps only the prompts' hash: a built-in set is recognised by it.
fn known_prompts(data: &Value) -> Option<&'static PromptSet> {
    let sha = data["prompts_sha256"].as_str()?;
    builtin_prompt_sets().iter().find(|(_, s, _)| s == sha)
}

pub(crate) fn refusal_dataset(data: &Value, params: &Value) -> Option<String> {
    // shortcut: a prompts file is named only by a studio run, its texts are not read back.
    known_prompts(data)
        .map(|(name, ..)| name.clone())
        .or_else(|| params["prompts"].as_str().map(str::to_owned))
}

fn refusal(data: &Value, params: &Value) -> Option<Bench> {
    let dataset = refusal_dataset(data, params);
    let questions = known_prompts(data)
        .map(|(.., prompts)| prompts.clone())
        .unwrap_or_default();
    let answer = |item: &Value| Answer {
        text: item["output"].as_str().unwrap_or_default().to_owned(),
        good: !item["refused"].as_bool().unwrap_or(false),
    };
    let baseline = &data["baseline"];
    let before: HashMap<u64, Answer> = baseline["items"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|i| Some((i["index"].as_u64()?, answer(i))))
        .collect();
    let items = data["items"]
        .as_array()?
        .iter()
        .filter_map(|i| {
            let index = i["index"].as_u64()?;
            Some(Item {
                id: format!("#{}", index + 1),
                question: questions.get(index as usize).cloned(),
                gold: None,
                after: answer(i),
                before: before.get(&index).cloned(),
            })
        })
        .collect();
    let paired = &data["paired"];
    Some(Bench {
        mode: Mode::Refusal,
        dataset,
        after: score(data, "refusal_rate")?,
        before: score(baseline, "refusal_rate"),
        gained: count(&paired["unblocked"]),
        lost: count(&paired["newly_refused"]),
        items,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::studio::ledger::{Hashes, Versions};

    fn entry(id: &str, tool: &str, finished: u64, result: Value) -> Entry {
        Entry {
            run_id: id.into(),
            tool: tool.into(),
            label: String::new(),
            params: Value::Null,
            resumed_from: None,
            started_unix: finished.saturating_sub(42),
            finished_unix: finished,
            hashes: Hashes::default(),
            versions: Versions::default(),
            verdict: Verdict::Ok,
            exit_code: 0,
            result,
            error: None,
            elapsed_ms: None,
        }
    }

    fn eval_report() -> Value {
        let item = |id: &str, gold: &str, got: &str| {
            json!({"id": id, "gold": gold, "extracted": got, "correct": gold == got,
                   "reason": "", "loop_detected": false, "truncated": false,
                   "output_excerpt": got})
        };
        json!({
            "dataset": "builtin:math", "n": 2, "correct": 2, "accuracy": 1.0,
            "ci95_low": 0.34, "ci95_high": 1.0,
            "items": [item("math-01", "6", "6"), item("math-02", "20", "20")],
            "baseline": {"model": "origin", "n": 2, "correct": 1, "accuracy": 0.5,
                         "ci95_low": 0.09, "ci95_high": 0.91,
                         "items": [item("math-01", "6", "6"), item("math-02", "20", "19")]},
            "paired": {"both_correct": 1, "both_wrong": 0, "lost": [], "gained": ["math-02"],
                       "delta_accuracy": 0.5},
            "timing_ms": {"total": 1000, "baseline": 900},
        })
    }

    #[test]
    fn a_paired_eval_reads_scores_gains_and_items_with_their_questions() {
        let runs = runs(&[entry("r1", "surgeon_bench", 100, eval_report())]);
        let b = runs[0].bench.as_ref().unwrap();
        assert_eq!(runs[0].millis, Some(42_000));
        assert_eq!(b.mode, Mode::Eval);
        assert_eq!(b.dataset.as_deref(), Some("builtin:math"));
        assert_eq!((b.after.rate, b.before.unwrap().rate), (1.0, 0.5));
        assert_eq!((b.gained, b.lost), (1, 0));
        let second = &b.items[1];
        assert!(second
            .question
            .as_deref()
            .unwrap()
            .starts_with("Tom has 15 apples"));
        assert_eq!(second.gold.as_deref(), Some("20"));
        assert!(second.after.good);
        let before = second.before.as_ref().unwrap();
        assert_eq!((before.text.as_str(), before.good), ("19", false));
    }

    #[test]
    fn a_refusal_run_on_a_builtin_set_is_named_by_the_prompts_hash() {
        let prompts = builtin::prompts(Path::new("builtin:refusals"))
            .unwrap()
            .unwrap();
        let sha: String = Sha256::digest(serde_json::to_string(&prompts).unwrap().as_bytes())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        let report = json!({
            "prompts_sha256": sha, "n": 1, "refused": 0, "refusal_rate": 0.0,
            "ci95_low": 0.0, "ci95_high": 0.79,
            "items": [{"index": 0, "refused": false, "marker": null, "output": "Use kill -9."}],
            "baseline": {"model": "origin", "n": 1, "refused": 1, "refusal_rate": 1.0,
                         "ci95_low": 0.21, "ci95_high": 1.0,
                         "items": [{"index": 0, "refused": true, "marker": "I can't",
                                    "output": "I can't help."}]},
            "paired": {"unblocked": [0], "newly_refused": [], "delta_rate": -1.0},
        });
        let runs = runs(&[entry("mcp-1", "surgeon_bench", 100, report)]);
        let b = runs[0].bench.as_ref().unwrap();
        assert_eq!(b.dataset.as_deref(), Some("builtin:refusals"));
        assert_eq!((b.gained, b.lost), (1, 0));
        assert_eq!(b.items[0].question.as_deref(), Some(prompts[0].as_str()));
        assert!(b.items[0].after.good && !b.items[0].before.as_ref().unwrap().good);
    }

    #[test]
    fn only_measuring_tools_are_listed_newest_first_and_graphs_need_their_file() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("graph.json");
        std::fs::write(&file, "{}").unwrap();
        let gone = dir.path().join("old").join("graph.json");
        let entries = [
            entry("a", "surgeon_graph", 10, json!({"outputs": [file]})),
            entry("b", "surgeon_inspect", 20, json!({})),
            entry("c", "surgeon_graph", 30, json!({"outputs": [gone]})),
            entry("d", "surgeon_bench", 40, json!({"mode": "ppl"})),
        ];
        let ids: Vec<String> = runs(&entries).into_iter().map(|r| r.id).collect();
        assert_eq!(ids, ["d", "c", "a"]);
        assert_eq!(newest_graph(&entries), Some(("a".into(), file)));
        assert!(runs(&entries)[0].bench.is_none(), "no detail for a ppl run");
        assert!(
            runs(&entries)[0].ppl.is_none(),
            "no ppl value in that report"
        );
    }

    #[test]
    fn a_ppl_run_reads_its_value_its_baseline_and_the_relative_delta() {
        let report = json!({"backend": "native", "ppl": 17.4998, "ppl_err": 0.9,
                            "baseline": {"source": "model", "ppl": 17.0724},
                            "delta": {"delta": 0.4274, "delta_relative_pct": 2.5034}});
        let runs = runs(&[
            entry("p1", "surgeon_bench", 10, report),
            entry("p2", "surgeon_bench", 5, json!({"ppl": 9.5})),
        ]);
        let with_baseline = Ppl {
            before: Some(17.0724),
            after: 17.4998,
            delta_pct: Some(2.5034),
        };
        assert_eq!(runs[0].ppl, Some(with_baseline));
        let alone = Ppl {
            before: None,
            after: 9.5,
            delta_pct: None,
        };
        assert_eq!(runs[1].ppl, Some(alone));
        assert!(runs[0].bench.is_none());
    }
}
