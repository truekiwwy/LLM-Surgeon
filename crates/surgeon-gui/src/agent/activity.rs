//! What the agent's running tool works on, for the 3D view: the last tool call of the
//! conversation still without its result (an API session or a CLI run alike), read into a
//! short label (`Extraction L14…`, `Patch down_proj L11–L14…`) and the layers it targets.

use std::collections::{BTreeSet, HashSet};

use serde_json::Value;
use weight_surgeon::patch::{LayerSelection, LayersField};

use super::provider::Block;
use super::session::Conversation;
use crate::ui::i18n::{t, tf};

#[derive(Debug, Clone, PartialEq)]
pub struct Activity {
    pub label: String,
    /// What the call works on, without its verb: `down_proj L11–L14`, `builtin:math`.
    pub target: String,
    /// `None`: the whole model (a benchmark), no layer stands out.
    pub layers: Option<LayerSelection>,
}

impl Activity {
    pub fn covers(&self, layer: Option<usize>) -> bool {
        match (&self.layers, layer) {
            (Some(LayerSelection::All), Some(_)) => true,
            (Some(LayerSelection::List(list)), Some(l)) => list.contains(&l),
            (
                Some(LayerSelection::Range {
                    start,
                    end_inclusive,
                }),
                Some(l),
            ) => (*start..=*end_inclusive).contains(&l),
            _ => false,
        }
    }
}

/// The running call of `conv`, if it is one the 3D view can show.
pub fn current(conv: &Conversation) -> Option<Activity> {
    let blocks = || conv.messages.iter().flat_map(|m| &m.blocks);
    let answered: HashSet<&str> = blocks()
        .filter_map(|b| match b {
            Block::ToolResult { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    blocks()
        .filter_map(|b| match b {
            Block::ToolUse { id, name, input } if !answered.contains(id.as_str()) => {
                Some((name, input))
            }
            _ => None,
        })
        .next_back()
        .and_then(|(name, input)| of(name, input))
}

/// The newest call of `conv` the 3D view can show, running or answered: what Live mode keeps
/// in focus between two calls.
pub fn last(conv: &Conversation) -> Option<Activity> {
    conv.messages
        .iter()
        .flat_map(|m| &m.blocks)
        .rev()
        .find_map(|b| match b {
            Block::ToolUse { name, input, .. } => of(name, input),
            _ => None,
        })
}

/// `14`, `"11-14"`, `"3,5"`, `"all"`, `[3, 5]` or `{"start", "end"}`.
fn layers(v: &Value) -> Option<LayerSelection> {
    match v {
        Value::Number(n) => Some(LayerSelection::List(vec![n.as_u64()? as usize])),
        Value::Null => None,
        v => serde_json::from_value::<LayersField>(v.clone())
            .ok()?
            .to_selection()
            .ok(),
    }
}

fn names(v: &Value) -> Vec<String> {
    match v {
        Value::String(s) => vec![s.clone()],
        Value::Array(items) => items
            .iter()
            .filter_map(|s| s.as_str().map(str::to_owned))
            .collect(),
        _ => Vec::new(),
    }
}

/// `L14`, `L11–L14`, `L3, L5`, or every layer.
fn layers_text(sel: &LayerSelection) -> String {
    match sel {
        LayerSelection::All => t("activity.all").to_owned(),
        LayerSelection::Range {
            start,
            end_inclusive,
        } if start == end_inclusive => format!("L{start}"),
        LayerSelection::Range {
            start,
            end_inclusive,
        } => format!("L{start}–L{end_inclusive}"),
        LayerSelection::List(list) => {
            let (lo, hi) = (list.iter().min(), list.iter().max());
            match (lo, hi) {
                (Some(lo), Some(hi)) if hi - lo + 1 == list.len() && lo != hi => {
                    format!("L{lo}–L{hi}")
                }
                _ => list
                    .iter()
                    .map(|l| format!("L{l}"))
                    .collect::<Vec<_>>()
                    .join(", "),
            }
        }
    }
}

/// The layers and projections of every operation of an inline patch.
fn patch(v: &Value) -> (Option<LayerSelection>, Vec<String>) {
    // shortcut: a patch given as a file path is not read here (one read per frame), its
    // layers do not pulse; read it once per call if agents start passing paths.
    let ops = v["operations"].as_array().map(Vec::as_slice).unwrap_or(&[]);
    let projections: BTreeSet<String> = ops
        .iter()
        .flat_map(|op| names(&op["projections"]))
        .collect();
    (
        union(ops.iter().map(|op| layers(&op["layers"]))),
        projections.into_iter().collect(),
    )
}

fn union(selections: impl IntoIterator<Item = Option<LayerSelection>>) -> Option<LayerSelection> {
    let mut all = false;
    let mut list = BTreeSet::new();
    for sel in selections {
        match sel {
            Some(LayerSelection::All) => all = true,
            Some(LayerSelection::List(l)) => list.extend(l),
            Some(LayerSelection::Range {
                start,
                end_inclusive,
            }) => list.extend(start..=end_inclusive),
            None => {}
        }
    }
    match (all, list.is_empty()) {
        (true, _) => Some(LayerSelection::All),
        (false, false) => Some(LayerSelection::List(list.into_iter().collect())),
        (false, true) => None,
    }
}

/// The activity of a call to `name` (an engine tool, `mcp__<server>__` prefix or not).
pub fn of(name: &str, input: &Value) -> Option<Activity> {
    type Label = fn(&str) -> String;
    let tool = name.rsplit("__").next().unwrap_or(name);
    let (label, sel, what): (Label, _, _) = match tool {
        "surgeon_inspect" => (
            |t| tf("activity.inspect", &[("target", &t)]),
            layers(&input["layers"]),
            names(&input["projections"]),
        ),
        "surgeon_extract" => (
            |t| tf("activity.extract", &[("target", &t)]),
            layers(&input["layers"]).or_else(|| layers(&input["layer"])),
            Vec::new(),
        ),
        "surgeon_causal_trace" => (
            |t| tf("activity.trace", &[("target", &t)]),
            layers(&input["layers"]).or(Some(LayerSelection::All)),
            names(&input["component"]),
        ),
        "surgeon_autotune" => (
            |t| tf("activity.autotune", &[("target", &t)]),
            layers(&input["target_layer_range"]),
            names(&input["proj"]),
        ),
        "surgeon_apply" => {
            let (sel, what) = patch(&input["patch"]);
            (|t| tf("activity.apply", &[("target", &t)]), sel, what)
        }
        "surgeon_sweep" => {
            let windows = input["layers"].as_array().map(Vec::as_slice).unwrap_or(&[]);
            (
                |t| tf("activity.sweep", &[("target", &t)]),
                union(windows.iter().map(layers)),
                names(&input["operation"]),
            )
        }
        "surgeon_bench" => {
            let what = names(&input["dataset"])
                .into_iter()
                .chain(names(&input["mode"]))
                .take(1)
                .collect();
            (|t| tf("activity.bench", &[("target", &t)]), None, what)
        }
        _ => return None,
    };
    let mut target = what;
    target.extend(sel.as_ref().map(layers_text));
    let target = target.join(" ");
    Some(Activity {
        label: label(&target).trim().replace(" …", "…"),
        target,
        layers: sel,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::agent::provider::{Message, Role};
    use crate::settings::Lang;
    use crate::ui::i18n;

    #[test]
    fn each_tool_reads_its_own_layer_arguments() {
        i18n::set_lang(Lang::Fr);
        let a = of("surgeon_extract", &json!({"layer": 14})).unwrap();
        assert_eq!(a.label, "Extraction L14…");
        assert!(a.covers(Some(14)) && !a.covers(Some(13)) && !a.covers(None));

        let input = json!({"target_layer_range": "11-14", "proj": "down_proj"});
        let a = of("mcp__surgeon__surgeon_autotune", &input).unwrap();
        assert_eq!(a.label, "Autotune down_proj L11–L14…");
        assert!(a.covers(Some(11)) && a.covers(Some(14)) && !a.covers(Some(15)));

        let patch = json!({"patch": {"operations": [
            {"layers": "11-12", "projections": ["down_proj"]},
            {"layers": [13, 14], "projections": ["down_proj"]},
        ]}});
        let a = of("surgeon_apply", &patch).unwrap();
        assert_eq!(a.label, "Patch down_proj L11–L14…");

        let a = of("surgeon_inspect", &json!({"layers": "all"})).unwrap();
        assert!(a.covers(Some(0)) && a.covers(Some(40)));

        let a = of("surgeon_extract", &json!({"layer": 20, "layers": "4-6"})).unwrap();
        assert_eq!(a.label, "Extraction L4–L6…");
        let a = of(
            "surgeon_sweep",
            &json!({"layers": ["4-6", "10"], "operation": "project_out"}),
        )
        .unwrap();
        assert_eq!(a.label, "Balayage Pareto project_out L4, L5, L6, L10…");
        assert!(a.covers(Some(10)) && !a.covers(Some(7)));
        let a = of("surgeon_causal_trace", &json!({"component": "mlp"})).unwrap();
        assert_eq!(a.label, "Traçage causal mlp toutes les couches…");
        assert!(a.covers(Some(0)) && a.covers(Some(27)));

        let a = of(
            "surgeon_bench",
            &json!({"mode": "eval", "dataset": "builtin:math"}),
        )
        .unwrap();
        assert_eq!(a.label, "Benchmark builtin:math…");
        assert!(!a.covers(Some(0)));

        assert!(of("surgeon_check", &json!({})).is_none());
    }

    #[test]
    fn only_a_call_still_waiting_for_its_result_is_running() {
        i18n::set_lang(Lang::Fr);
        let call = |id: &str, layer: u64| Block::ToolUse {
            id: id.into(),
            name: "surgeon_extract".into(),
            input: json!({"layer": layer}),
        };
        let result = |id: &str| Block::ToolResult {
            id: id.into(),
            content: String::new(),
            is_error: false,
        };
        let mut conv = Conversation::default();
        conv.messages.push(Message {
            role: Role::Assistant,
            blocks: vec![call("a", 3)],
        });
        assert_eq!(current(&conv).unwrap().label, "Extraction L3…");
        conv.messages.push(Message {
            role: Role::User,
            blocks: vec![result("a")],
        });
        assert!(current(&conv).is_none());
        // Answered, the call stays the last one Live mode keeps in focus.
        assert_eq!(last(&conv).unwrap().target, "L3");
        conv.messages.push(Message {
            role: Role::Assistant,
            blocks: vec![call("b", 7)],
        });
        assert_eq!(current(&conv).unwrap().label, "Extraction L7…");
        assert_eq!(last(&conv).unwrap().target, "L7");
    }
}
