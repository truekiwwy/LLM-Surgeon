//! Cost and time before launch. A tool with a preview mode of its own (`surgeon_graph`
//! `estimate`, `surgeon_apply` and `surgeon_lora` `dry_run`, `surgeon_pipeline` `plan`) is
//! first run in that mode as a normal job; its envelope gives the estimate, and a run whose
//! estimate exceeds its time budget is refused before it is queued.

use serde_json::Value;

/// The arguments of the preview of `tool` called with `args`, or `None` when the engine has
/// no preview for it.
pub fn preview_args(tool: &str, args: &Value) -> Option<Value> {
    let (key, value) = match tool {
        "surgeon_graph" => ("estimate", Value::Bool(true)),
        "surgeon_apply" | "surgeon_lora" => ("dry_run", Value::Bool(true)),
        "surgeon_pipeline" => ("action", Value::from("plan")),
        _ => return None,
    };
    let mut preview = args.clone();
    preview.as_object_mut()?.insert(key.into(), value);
    Some(preview)
}

/// Whether `args` already are a preview: such a run writes no model.
pub fn is_preview(tool: &str, args: &Value) -> bool {
    preview_args(tool, args).is_some_and(|p| &p == args)
}

/// What a preview said about the real run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Estimate {
    pub seconds: Option<f64>,
    /// Bytes read or written, as the tool counts them.
    pub bytes: Option<u64>,
    /// The engine's own warnings (budget, power, disk).
    pub warnings: Vec<String>,
}

/// Where the tools put their figures: graph `estimated_seconds` / `bytes_read`, GGUF apply
/// `preflight.estimated_seconds` / `work_bytes`, pipeline `estimated_secs` / `disk.total_bytes`.
const SECONDS: [&str; 2] = ["estimated_seconds", "estimated_secs"];
const BYTES: [&str; 3] = ["bytes_read", "work_bytes", "total_bytes"];

pub fn read(envelope: &Value) -> Estimate {
    let data = &envelope["data"];
    Estimate {
        seconds: find(data, &SECONDS).and_then(Value::as_f64),
        bytes: find(data, &BYTES).and_then(Value::as_u64),
        warnings: envelope["warnings"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(Into::into)
            .collect(),
    }
}

/// The first non-null value under one of `keys`, breadth first.
fn find<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    let mut level = vec![value];
    while !level.is_empty() {
        for v in &level {
            if let Some(found) = keys.iter().find_map(|k| v.get(k).filter(|x| !x.is_null())) {
                return Some(found);
            }
        }
        level = level
            .iter()
            .flat_map(|v| match v {
                Value::Object(map) => map.values().collect(),
                Value::Array(items) => items.iter().collect(),
                _ => Vec::new(),
            })
            .collect();
    }
    None
}

impl Estimate {
    pub fn summary(&self) -> String {
        let time = self
            .seconds
            .map_or("durée non estimée".into(), |s| format!("~{s:.1} s"));
        match self.bytes {
            Some(b) => format!("{time} · {:.2} Gio", b as f64 / f64::from(1u32 << 30)),
            None => time,
        }
    }
}

/// The check before queueing. A tool with a preview and a time budget needs the estimate of
/// these exact arguments, and that estimate must fit the budget. Without a budget, or for a
/// tool the engine cannot preview, nothing is required.
pub fn gate(
    has_preview: bool,
    max_wall_time_secs: Option<u64>,
    estimate: Option<&Estimate>,
) -> Result<(), String> {
    let Some(max) = max_wall_time_secs.filter(|_| has_preview) else {
        return Ok(());
    };
    match estimate.map(|e| e.seconds) {
        None => Err(format!(
            "budget de {max} s : estimez d'abord ces paramètres"
        )),
        Some(Some(s)) if s > max as f64 => Err(format!(
            "estimation ~{s:.0} s au-delà du budget de {max} s : relevez le budget ou réduisez le travail"
        )),
        // An estimate without a duration (uncalibrated) does not block: the budget still
        // stops the run when it is reached.
        Some(_) => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn previews_use_each_tool_mode_and_leave_other_arguments_alone() {
        let args = json!({"model": "copies/c1", "out_dir": "{out}"});
        let p = preview_args("surgeon_graph", &args).unwrap();
        assert_eq!(
            p,
            json!({"model": "copies/c1", "out_dir": "{out}", "estimate": true})
        );
        assert!(is_preview("surgeon_graph", &p));
        assert!(!is_preview("surgeon_graph", &args));
        let apply = json!({"model": "copies/c1", "patch": "p.json", "dry_run": false});
        assert_eq!(
            preview_args("surgeon_apply", &apply).unwrap()["dry_run"],
            true
        );
        let pipe = json!({"action": "start", "job_dir": "{out}/job"});
        assert_eq!(
            preview_args("surgeon_pipeline", &pipe).unwrap()["action"],
            "plan"
        );
        assert_eq!(preview_args("surgeon_check", &args), None);
        assert!(!is_preview("surgeon_check", &args));
        // Every previewed tool exists in the engine schemas, with that argument.
        let tools = weight_surgeon::mcp::tool_definitions();
        for (tool, key) in [
            ("surgeon_graph", "estimate"),
            ("surgeon_apply", "dry_run"),
            ("surgeon_lora", "dry_run"),
            ("surgeon_pipeline", "action"),
        ] {
            let def = tools
                .as_array()
                .unwrap()
                .iter()
                .find(|t| t["name"] == tool)
                .unwrap_or_else(|| panic!("{tool}"));
            assert!(
                def["inputSchema"]["properties"][key].is_object(),
                "{tool}.{key}"
            );
        }
    }

    #[test]
    fn estimates_are_read_from_every_tool_layout() {
        let graph = json!({"data": {"estimate": true, "estimated_seconds": 12.5, "bytes_read": 1073741824u64}});
        let e = read(&graph);
        assert_eq!((e.seconds, e.bytes), (Some(12.5), Some(1 << 30)));
        assert_eq!(e.summary(), "~12.5 s · 1.00 Gio");
        let gguf = json!({"data": {"runs": [{"preflight": {"estimated_seconds": null, "work_bytes": 10}}]}});
        assert_eq!(
            read(&gguf),
            Estimate {
                seconds: None,
                bytes: Some(10),
                warnings: vec![]
            }
        );
        let plan =
            json!({"data": {"estimated_secs": 600, "disk": {"total_bytes": 5}}, "warnings": ["w"]});
        let e = read(&plan);
        assert_eq!(
            (e.seconds, e.bytes, e.warnings.len()),
            (Some(600.0), Some(5), 1)
        );
        assert_eq!(read(&json!({"data": {}})).summary(), "durée non estimée");
    }

    #[test]
    fn the_budget_gate_needs_a_fitting_estimate_only_when_it_can_have_one() {
        let fits = Estimate {
            seconds: Some(30.0),
            ..Default::default()
        };
        let over = Estimate {
            seconds: Some(90.0),
            ..Default::default()
        };
        let unknown = Estimate::default();
        assert!(gate(true, None, None).is_ok(), "no budget");
        assert!(
            gate(false, Some(60), None).is_ok(),
            "no preview for this tool"
        );
        assert!(gate(true, Some(60), None).unwrap_err().contains("estimez"));
        assert!(gate(true, Some(60), Some(&fits)).is_ok());
        assert!(gate(true, Some(60), Some(&over))
            .unwrap_err()
            .contains("90"));
        assert!(gate(true, Some(60), Some(&unknown)).is_ok());
    }
}
