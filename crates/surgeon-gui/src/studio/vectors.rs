use std::collections::BTreeMap;
use std::path::Path;

use serde_json::json;
use weight_surgeon::patch::LayerSelection;

use super::jobs::{Budgets, JobSpec};
use super::ledger::{Entry, Verdict};
use crate::ui::i18n::t;

pub const SWEEP_TOOL: &str = "surgeon_extract";
pub const TRACE_TOOL: &str = "surgeon_causal_trace";
pub const METHODS: [&str; 2] = ["mean_diff", "svd"];
pub const COMPONENTS: [&str; 3] = ["resid", "attn", "mlp"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorMode {
    #[default]
    Sweep,
    Trace,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorForm {
    pub open: bool,
    pub mode: VectorMode,
    pub method: &'static str,
    pub component: &'static str,
    pub positive: String,
    pub negative: String,
    pub layers: String,
    pub name: String,
}

impl Default for VectorForm {
    fn default() -> Self {
        Self {
            open: false,
            mode: VectorMode::Sweep,
            method: METHODS[0],
            component: COMPONENTS[0],
            positive: String::new(),
            negative: String::new(),
            layers: "all".into(),
            name: "direction".into(),
        }
    }
}

fn lines(text: &str) -> Vec<String> {
    text.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect()
}

impl VectorForm {
    pub fn out(&self) -> String {
        format!("work/{}.safetensors", self.name.trim())
    }

    pub fn job(&self, model: &Path) -> Result<JobSpec, String> {
        let (positive, negative) = (lines(&self.positive), lines(&self.negative));
        if positive.is_empty() || negative.is_empty() {
            return Err(t("vector.error.prompts").into());
        }
        let paired = self.mode == VectorMode::Trace || self.method == "svd";
        if paired && positive.len() != negative.len() {
            return Err(t("vector.error.pairs").into());
        }
        let layers = match self.layers.trim() {
            "" => "all",
            layers => layers,
        };
        layers
            .parse::<LayerSelection>()
            .map_err(|e| format!("{} : {e}", t("vector.layers")))?;
        let (tool, args) = match self.mode {
            VectorMode::Sweep => {
                super::project::validate_name(self.name.trim())
                    .map_err(|_| t("vector.error.name").to_owned())?;
                let args = json!({"model": model, "pos_prompts": positive,
                    "neg_prompts": negative, "layers": layers, "method": self.method,
                    "out": self.out()});
                (SWEEP_TOOL, args)
            }
            VectorMode::Trace => {
                let args = json!({"model": model, "clean_prompts": positive,
                    "corrupted_prompts": negative, "layers": layers,
                    "component": self.component});
                (TRACE_TOOL, args)
            }
        };
        Ok(JobSpec {
            tool: tool.into(),
            label: t("vector.title").into(),
            args,
            priority: 0,
            budgets: Budgets::default(),
            resumed_from: None,
            read_only: false,
        })
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerScores {
    pub causal: BTreeMap<usize, f64>,
    pub component: Option<String>,
    pub window: Option<(usize, usize)>,
    pub auc: BTreeMap<usize, f64>,
    pub margin: BTreeMap<usize, f64>,
}

fn tool_of(entry: &Entry) -> &str {
    entry.tool.rsplit("__").next().unwrap_or(&entry.tool)
}

fn per_layer(rows: &serde_json::Value, key: &str) -> BTreeMap<usize, f64> {
    rows.as_array()
        .into_iter()
        .flatten()
        .filter_map(|row| Some((row["layer"].as_u64()? as usize, row[key].as_f64()?)))
        .collect()
}

impl LayerScores {
    pub fn of<'a>(runs: impl IntoIterator<Item = &'a Entry>) -> Self {
        let runs: Vec<&Entry> = runs
            .into_iter()
            .filter(|e| e.verdict == Verdict::Ok)
            .collect();
        let newest = |accept: &dyn Fn(&Entry) -> bool| {
            runs.iter()
                .copied()
                .filter(|e| accept(e))
                .max_by_key(|e| e.finished_unix)
        };
        let mut scores = Self::default();
        if let Some(trace) = newest(&|e| tool_of(e) == TRACE_TOOL) {
            let data = &trace.result;
            scores.causal = per_layer(&data["layers"], "score");
            scores.component = data["component"].as_str().map(String::from);
            let window = &data["window"];
            scores.window = window["start"]
                .as_u64()
                .zip(window["end"].as_u64())
                .map(|(start, end)| (start as usize, end as usize));
        }
        let sweep = |e: &Entry| tool_of(e) == SWEEP_TOOL && e.result["layers"].is_array();
        if let Some(sweep) = newest(&sweep) {
            scores.auc = per_layer(&sweep.result["layers"], "auc");
            scores.margin = per_layer(&sweep.result["layers"], "margin");
        }
        scores
    }

    pub fn lit(&self, layer: Option<usize>) -> bool {
        matches!((self.window, layer), (Some((start, end)), Some(l)) if (start..=end).contains(&l))
    }

    pub fn is_empty(&self) -> bool {
        self.causal.is_empty() && self.auc.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Value};

    use super::*;
    use crate::settings::Lang;
    use crate::studio::ledger::{Hashes, Versions};
    use crate::ui::i18n;

    fn entry(tool: &str, finished: u64, verdict: Verdict, result: Value) -> Entry {
        Entry {
            run_id: format!("r{finished}"),
            tool: tool.into(),
            label: String::new(),
            params: Value::Null,
            resumed_from: None,
            started_unix: finished,
            finished_unix: finished,
            hashes: Hashes::default(),
            versions: Versions::default(),
            verdict,
            exit_code: 0,
            result,
            error: None,
        }
    }

    fn form(mode: VectorMode) -> VectorForm {
        VectorForm {
            mode,
            positive: "refuse a\n\n  refuse b \n".into(),
            negative: "accept a\naccept b".into(),
            layers: "2-5".into(),
            name: "refusal_sweep".into(),
            ..VectorForm::default()
        }
    }

    #[test]
    fn the_form_becomes_a_sweep_or_a_trace_job_writing_under_work() {
        i18n::set_lang(Lang::Fr);
        let model = Path::new("C:/models/qwen");
        let sweep = form(VectorMode::Sweep).job(model).unwrap();
        assert_eq!(sweep.tool, SWEEP_TOOL);
        assert_eq!(
            sweep.args,
            json!({"model": "C:/models/qwen", "pos_prompts": ["refuse a", "refuse b"],
                "neg_prompts": ["accept a", "accept b"], "layers": "2-5",
                "method": "mean_diff", "out": "work/refusal_sweep.safetensors"})
        );
        let trace = VectorForm {
            component: "mlp",
            ..form(VectorMode::Trace)
        }
        .job(model)
        .unwrap();
        assert_eq!(trace.tool, TRACE_TOOL);
        assert_eq!(trace.args["component"], "mlp");
        assert_eq!(trace.args["clean_prompts"][1], "refuse b");
        assert!(trace.args.get("out").is_none(), "a trace writes nothing");
    }

    #[test]
    fn bad_prompts_names_and_layers_are_refused_before_any_job() {
        i18n::set_lang(Lang::Fr);
        let model = Path::new("m");
        let unpaired = VectorForm {
            negative: "only one".into(),
            ..form(VectorMode::Trace)
        };
        assert_eq!(unpaired.job(model).unwrap_err(), t("vector.error.pairs"));
        let svd = VectorForm {
            method: "svd",
            ..unpaired.clone()
        };
        assert!(VectorForm {
            mode: VectorMode::Sweep,
            ..svd
        }
        .job(model)
        .is_err());
        let mean_diff = VectorForm {
            mode: VectorMode::Sweep,
            ..unpaired
        };
        assert!(mean_diff.job(model).is_ok(), "mean_diff needs no pairs");
        let empty = VectorForm {
            positive: " \n".into(),
            ..form(VectorMode::Sweep)
        };
        assert_eq!(empty.job(model).unwrap_err(), t("vector.error.prompts"));
        let escape = VectorForm {
            name: "../model".into(),
            ..form(VectorMode::Sweep)
        };
        assert_eq!(escape.job(model).unwrap_err(), t("vector.error.name"));
        let layers = VectorForm {
            layers: "5-2".into(),
            ..form(VectorMode::Sweep)
        };
        assert!(layers.job(model).is_err());
    }

    #[test]
    fn the_newest_successful_trace_and_sweep_give_the_layer_scores() {
        let trace = |finished, start: u64, end: u64| {
            let result = json!({"component": "mlp",
                "window": {"start": start, "end": end, "layers": ""},
                "layers": [{"layer": 0, "score": 0.1}, {"layer": 1, "score": 0.8},
                           {"layer": 2, "score": 0.6}]});
            entry(
                "mcp__surgeon__surgeon_causal_trace",
                finished,
                Verdict::Ok,
                result,
            )
        };
        let runs = [
            trace(10, 0, 0),
            trace(30, 1, 2),
            entry(
                TRACE_TOOL,
                40,
                Verdict::Failed,
                json!({"window": {"start": 0, "end": 0}}),
            ),
            entry(
                SWEEP_TOOL,
                20,
                Verdict::Ok,
                json!({"layers": [{"layer": 1, "auc": 0.9, "margin": -0.5},
                                  {"layer": 2, "auc": 1.0, "margin": 0.25}]}),
            ),
            entry(
                SWEEP_TOOL,
                50,
                Verdict::Ok,
                json!({"layer": 4, "stats": {}}),
            ),
        ];
        let scores = LayerScores::of(&runs);
        assert_eq!(scores.window, Some((1, 2)));
        assert_eq!(scores.component.as_deref(), Some("mlp"));
        assert_eq!(scores.causal[&1], 0.8);
        assert_eq!(scores.auc[&2], 1.0);
        assert_eq!(scores.margin[&1], -0.5);
        assert!(scores.lit(Some(2)) && !scores.lit(Some(0)) && !scores.lit(None));
        assert!(LayerScores::of(&[]).is_empty());
    }
}
