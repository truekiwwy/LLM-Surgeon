//! A third-party extraction method, `median_diff`: `normalize(median(pos) - median(neg))`,
//! coordinate by coordinate, less sensitive than `mean_diff` to a few outlier prompts.
//! It is registered next to the built-in methods and run end to end, through the library,
//! on the mini Qwen2 of `make_fake_qwen2.rs`. Walkthrough: `docs/PLUGINS.md`.
//!
//! ```text
//! cargo run --example plugin_extraction_example [-- <dir>]
//! ```
//!
//! Writes `<dir>/median.safetensors` (+ `.stats.json`) and prints the report as JSON. Without
//! `<dir>`, works in a temporary directory removed at the end.

#[path = "make_fake_qwen2.rs"]
#[allow(dead_code)]
pub mod fake_qwen2;

use std::error::Error;
use std::ops::ControlFlow;
use std::path::Path;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};
use weight_surgeon::extract::{
    load_prompt_texts, run_extraction_plugin, ChatTemplate, DTypeChoice, ExtractEvent,
    ExtractObserver, ExtractReport, ExtractRequest, Pooling, Site,
};
use weight_surgeon::hardware::SystemProbe;
use weight_surgeon::plugin::{
    ActivationSet, Diagnostics, ExtractContext, ExtractionMethod, ExtractionOutput, MethodInfo,
    PluginError, Registry,
};

// ------------------------------------------------------------------ the plugin

pub struct MedianDiff;

fn median(rows: &[Vec<f64>], j: usize) -> f64 {
    let mut column: Vec<f64> = rows.iter().map(|r| r[j]).collect();
    column.sort_by(f64::total_cmp);
    let n = column.len();
    match n % 2 {
        1 => column[n / 2],
        _ => (column[n / 2 - 1] + column[n / 2]) / 2.0,
    }
}

impl ExtractionMethod for MedianDiff {
    fn id(&self) -> &'static str {
        "median_diff"
    }

    fn describe(&self) -> MethodInfo {
        MethodInfo {
            title: "Différence des médianes",
            summary: "normalize(médiane(pos) − médiane(neg)), coordonnée par coordonnée.",
            params: vec![],
            requires_paired: false,
            sidecars: vec![],
        }
    }

    fn extract(
        &self,
        ctx: &ExtractContext,
        acts: &ActivationSet,
        _cancel: &AtomicBool,
    ) -> Result<ExtractionOutput, PluginError> {
        let diff: Vec<f64> = (0..ctx.hidden_size)
            .map(|j| median(acts.pos, j) - median(acts.neg, j))
            .collect();
        let norm = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm.is_finite() && norm > 1e-12) {
            return Err(PluginError::Failed(format!(
                "médianes confondues (‖Δ‖ = {norm})"
            )));
        }
        let mut out = ExtractionOutput::new(diff.iter().map(|x| x / norm).collect());
        let diagnostics = json!({ "raw_median_diff_norm": norm });
        out.diagnostics = diagnostics.as_object().cloned().map(Diagnostics::Json);
        Ok(out)
    }
}

// ------------------------------------------------------------------ using it

/// No progress display, never cancels.
struct Quiet;

impl ExtractObserver for Quiet {
    fn on_event(&mut self, _: ExtractEvent) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

pub type Res<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// Mini model and prompts in `dir`, then a `median_diff` extraction at layer 2.
pub fn run(dir: &Path) -> Res<ExtractReport> {
    let model = fake_qwen2::write_fake_qwen2(dir)?;
    let (pos, neg) = fake_qwen2::write_prompt_files(dir)?;

    let mut registry = Registry::with_builtin();
    registry.register(Box::new(MedianDiff))?;
    let method = registry
        .get("median_diff")
        .ok_or("median_diff not registered")?;

    let request = ExtractRequest {
        model_dir: model,
        pos_prompts: load_prompt_texts(&pos)?,
        neg_prompts: load_prompt_texts(&neg)?,
        layer: 2,
        out: dir.join("median.safetensors"),
        pooling: Pooling::Last,
        site: Site::ResidPost,
        chat_template: ChatTemplate::NoTemplate,
        max_tokens: 512,
        dtype: DTypeChoice::Auto,
        overwrite: true,
    };
    let never = AtomicBool::new(false);
    let probe = SystemProbe::detect();
    let report = run_extraction_plugin(&request, method, &Value::Null, &probe, &mut Quiet, &never)?;
    Ok(report)
}

fn main() -> Res<()> {
    let temp = tempfile::tempdir()?;
    let dir = std::env::args_os()
        .nth(1)
        .map_or(temp.path().to_path_buf(), Into::into);
    std::fs::create_dir_all(&dir)?;
    let report = run(&dir)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
