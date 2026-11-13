//! `graph.json`, DOT, Mermaid and CSV for `surgeon graph`, plus safe
//! output writes.
//!
//! The report is the schema source (`surgeon schema --kind graph`). It contains no
//! wall-clock timestamp: a run is reproducible from the model digest, the parameters and
//! the code version. Files are written with exclusive creation, symlink refusal and a
//! temporary file renamed into place; `--overwrite` is required to replace an existing
//! target.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::diff::{BaselineDiff, EdgeDiff, NodeDiff};
use super::error::{GraphError, Result};
use super::layout::Position;
use super::metrics::NodeMetrics;
use super::model::SkippedTensor;
use super::GraphRole;

pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ToolInfo {
    pub name: String,
    pub version: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ModelReport {
    pub path: String,
    /// Full SHA-256, or `fast-...` with `--fast-hash` (see `hash_mode`).
    pub sha256: String,
    /// `sha256` (complete digest of every file) or `fast` (`--fast-hash`: sizes, dates and
    /// three sampled windows per file). Absent from reports older than this field: `sha256`.
    #[serde(default = "full_hash_mode")]
    pub hash_mode: String,
    pub format: String,
    pub architecture: String,
    pub arch_source: String,
    pub hidden: usize,
    pub layers: usize,
    pub tied_lm_head: bool,
    pub tensor_count: usize,
    pub total_weight_bytes: u64,
    pub skipped: Vec<SkippedTensor>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ParametersReport {
    pub rank: usize,
    pub oversample: usize,
    pub power_iters: usize,
    pub seed: u64,
    pub full_spectrum_max_dim: usize,
    pub edge_mode: String,
    pub min_alignment: f64,
    pub edge_weight: String,
    pub vector: Option<String>,
    pub kurtosis_threshold: f64,
    pub color_scale: String,
    pub palette: String,
    pub max_nodes: Option<usize>,
    pub formats: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GlobalReport {
    pub nodes: usize,
    pub edges: usize,
    pub structural_edges: usize,
    pub layers: usize,
    pub hidden: usize,
    pub outlier_channels: Vec<usize>,
    pub kurtosis_median: f64,
    pub kurtosis_max: f64,
    pub mean_erank_energy_lower: f64,
    pub mean_erank_energy_upper: f64,
    pub full_spectrum_nodes: usize,
    pub non_finite_values: u64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NodeKind {
    Writer,
    Reader,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NodeReport {
    pub id: String,
    pub role: GraphRole,
    pub kind: NodeKind,
    pub layer: Option<usize>,
    pub expert: Option<usize>,
    pub tensor: String,
    pub dtype: String,
    pub weight_bytes: u64,
    pub rows: usize,
    pub cols: usize,
    pub seq: usize,
    pub metrics: NodeMetrics,
    /// `Colonne par couches` position in the unit ball (`layout::column`). Zero in reports
    /// older than this field.
    #[serde(default)]
    pub pos_column: Position,
    /// `Constellation 3D` position in the unit ball (`layout::constellation`).
    #[serde(default)]
    pub pos_constellation: Position,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EdgeReport {
    pub source: String,
    pub target: String,
    pub writer_role: GraphRole,
    pub reader_role: GraphRole,
    pub source_seq: usize,
    pub target_seq: usize,
    pub layer_gap: i64,
    pub rank_used: usize,
    pub alignment: f64,
    pub chance: f64,
    pub alignment_norm: f64,
    pub coupling: f64,
    pub cos_theta: f64,
    pub weight: f64,
    pub structural: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct GraphReport {
    pub schema_version: u32,
    pub tool: ToolInfo,
    pub model: ModelReport,
    pub parameters: ParametersReport,
    pub global: GlobalReport,
    pub nodes: Vec<NodeReport>,
    pub edges: Vec<EdgeReport>,
    /// With `--baseline` only: deltas of the model against the reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub baseline_diff: Option<BaselineDiff>,
}

fn full_hash_mode() -> String {
    "sha256".into()
}

/// JSON Schema (draft-07) of `graph.json` (`surgeon schema --kind graph`).
pub fn schema() -> Value {
    serde_json::to_value(schemars::schema_for!(GraphReport)).unwrap_or_default()
}

pub fn render_json(report: &GraphReport) -> Result<String> {
    let mut text = serde_json::to_string_pretty(report)
        .map_err(|e| GraphError::Compute(format!("graph.json : {e}")))?;
    text.push('\n');
    Ok(text)
}

// ------------------------------------------------------------------ colours

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorScale {
    Quantile,
    Log,
    Linear,
}

impl ColorScale {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Quantile => "quantile",
            Self::Log => "log",
            Self::Linear => "linear",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Palette {
    Diverging,
    CbSafe,
}

impl Palette {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Diverging => "diverging",
            Self::CbSafe => "cb-safe",
        }
    }

    fn stops(self) -> &'static [[u8; 3]] {
        const DIVERGING: [[u8; 3]; 9] = [
            [33, 102, 172],
            [67, 147, 195],
            [146, 197, 222],
            [209, 229, 240],
            [247, 247, 247],
            [253, 219, 199],
            [244, 165, 130],
            [214, 96, 77],
            [178, 24, 43],
        ];
        const CB_SAFE: [[u8; 3]; 10] = [
            [68, 1, 84],
            [72, 40, 120],
            [62, 73, 137],
            [49, 104, 142],
            [38, 130, 142],
            [31, 158, 137],
            [53, 183, 121],
            [109, 205, 89],
            [180, 222, 44],
            [253, 231, 37],
        ];
        match self {
            Self::Diverging => &DIVERGING,
            Self::CbSafe => &CB_SAFE,
        }
    }
}

fn interpolate(palette: Palette, t: f64) -> [u8; 3] {
    let stops = palette.stops();
    let t = t.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
    let low = t.floor() as usize;
    let high = (low + 1).min(stops.len() - 1);
    let fraction = t - low as f64;
    [
        (f64::from(stops[low][0])
            + (f64::from(stops[high][0]) - f64::from(stops[low][0])) * fraction)
            .round() as u8,
        (f64::from(stops[low][1])
            + (f64::from(stops[high][1]) - f64::from(stops[low][1])) * fraction)
            .round() as u8,
        (f64::from(stops[low][2])
            + (f64::from(stops[high][2]) - f64::from(stops[low][2])) * fraction)
            .round() as u8,
    ]
}

fn edge_position(report: &GraphReport, weight: f64, sorted: &[f64]) -> f64 {
    match match report.parameters.color_scale.as_str() {
        "log" => ColorScale::Log,
        "linear" => ColorScale::Linear,
        _ => ColorScale::Quantile,
    } {
        ColorScale::Linear => {
            let min = sorted.first().copied().unwrap_or(0.0);
            let max = sorted.last().copied().unwrap_or(0.0);
            match max > min {
                true => (weight - min) / (max - min),
                false => 0.0,
            }
        }
        ColorScale::Log => {
            let max = sorted.last().copied().unwrap_or(0.0);
            match max > 0.0 {
                true => (1.0 + weight.max(0.0)).ln() / (1.0 + max).ln(),
                false => 0.0,
            }
        }
        ColorScale::Quantile => {
            if sorted.is_empty() {
                0.0
            } else {
                let at = sorted.partition_point(|v| *v <= weight);
                at as f64 / sorted.len() as f64
            }
        }
    }
    .clamp(0.0, 1.0)
}

fn edge_color(report: &GraphReport, weight: f64, sorted: &[f64]) -> [u8; 3] {
    let palette = match report.parameters.palette.as_str() {
        "cb-safe" => Palette::CbSafe,
        _ => Palette::Diverging,
    };
    interpolate(palette, edge_position(report, weight, sorted))
}

fn hex(rgb: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])
}

fn rgba(rgb: [u8; 3], alpha: u8) -> String {
    format!("#{:02x}{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2], alpha)
}

fn role_color(role: GraphRole) -> [u8; 3] {
    match role {
        GraphRole::Embed => [76, 120, 168],
        GraphRole::AttnOut => [245, 133, 24],
        GraphRole::Q | GraphRole::K | GraphRole::V => [178, 121, 162],
        GraphRole::Gate | GraphRole::Up => [228, 87, 86],
        GraphRole::MlpOut => [84, 162, 75],
        GraphRole::LmHead => [114, 183, 178],
    }
}

fn node_size(metrics: &NodeMetrics) -> f64 {
    let log = (1.0 + metrics.frobenius_norm.max(0.0)).log10();
    (0.45 + 0.35 * log).clamp(0.45, 2.4)
}

fn weight_strength(report: &GraphReport, weight: f64, sorted: &[f64]) -> f64 {
    edge_position(report, weight, sorted)
}

// ------------------------------------------------------------------ DOT

fn dot_escape(text: &str) -> String {
    text.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "")
}

/// Graphviz DOT with one cluster per layer, a legend, log node sizes, octagon outlier
/// nodes and RGBA edges coloured by the selected scale and palette.
pub fn render_dot(report: &GraphReport) -> String {
    let (changed_nodes, changed_edges) = changed(report.baseline_diff.as_ref());
    let dot_node = |report: &GraphReport, node: &NodeReport| {
        dot_node(report, node, changed_nodes.get(node.id.as_str()).copied())
    };
    let mut weights: Vec<f64> = report.edges.iter().map(|e| e.weight).collect();
    weights.sort_by(f64::total_cmp);
    let mut out = String::new();
    out.push_str(
        "// surgeon graph - deterministic projection of the residual-stream write/read alignment\n",
    );
    out.push_str("digraph surgeon_graph {\n");
    out.push_str("  graph [rankdir=LR, bgcolor=\"#ffffff\", fontname=\"Helvetica\", labelloc=t, label=\"surgeon graph\"];
");
    out.push_str(
        "  node [fontname=\"Helvetica\", fontsize=10, style=\"filled\", penwidth=1.2];
",
    );
    out.push_str(
        "  edge [fontname=\"Helvetica\", fontsize=8];
",
    );
    out.push_str("  subgraph cluster_embed {\n    label=\"embedding\";\n    color=\"#dddddd\";\n");
    for node in report.nodes.iter().filter(|n| n.role == GraphRole::Embed) {
        out.push_str(&dot_node(report, node));
    }
    out.push_str("  }\n");
    for layer in 0..report.global.layers {
        out.push_str(&format!(
            "  subgraph cluster_L{layer} {{\n    label=\"layer {layer}\";\n    color=\"#dddddd\";\n"
        ));
        for node in report.nodes.iter().filter(|n| n.layer == Some(layer)) {
            out.push_str(&dot_node(report, node));
        }
        out.push_str("  }\n");
    }
    let heads: Vec<&NodeReport> = report
        .nodes
        .iter()
        .filter(|n| n.role == GraphRole::LmHead)
        .collect();
    if !heads.is_empty() {
        out.push_str(
            "  subgraph cluster_head {\n    label=\"output head\";\n    color=\"#dddddd\";\n",
        );
        for node in heads {
            out.push_str(&dot_node(report, node));
        }
        out.push_str("  }\n");
    }
    for edge in &report.edges {
        let source = dot_id(&edge.source);
        let target = dot_id(&edge.target);
        if edge.structural {
            out.push_str(&format!(
                "  {source} -> {target} [style=dashed, color=\"#88888880\", penwidth=1.0, label=\"structural\"];
"
            ));
            continue;
        }
        if let Some(diff) = changed_edges.get(&(edge.source.as_str(), edge.target.as_str())) {
            out.push_str(&format!(
                "  {source} -> {target} [color=\"{CHANGED}\", penwidth=3.40, style=bold, label=\"{:.3} (Δ {:+.3})\"];\n",
                edge.weight, diff.alignment_delta
            ));
            continue;
        }
        let color = edge_color(report, edge.weight, &weights);
        let width = 0.6 + 2.8 * weight_strength(report, edge.weight, &weights);
        out.push_str(&format!(
            "  {source} -> {target} [color=\"{}\", penwidth={width:.2}, label=\"{:.3}\"];
",
            rgba(color, 215),
            edge.weight
        ));
    }
    out.push_str("  subgraph cluster_legend {\n    label=\"legend\";\n    color=\"#dddddd\";\n");
    out.push_str("    legend_writer [label=\"writer (W·basis)\", shape=box, fillcolor=\"#f5f5f5\", color=\"#666666\"];\n");
    out.push_str("    legend_reader [label=\"reader (basis)\", shape=box, style=\"filled,dashed\", fillcolor=\"#f5f5f5\", color=\"#666666\"];\n");
    out.push_str("    legend_octagon [label=\"outlier / kurtosis\", shape=octagon, fillcolor=\"#f5f5f5\", color=\"#d62728\"];\n");
    out.push_str("    legend_writer -> legend_reader [label=\"alignment\", color=\"#999999\"];\n");
    if let Some(diff) = &report.baseline_diff {
        out.push_str(&format!(
            "    legend_changed [label=\"changed vs baseline (>= {:.1}%)\", shape=box, penwidth=3.4, fillcolor=\"#f5f5f5\", color=\"{CHANGED}\"];\n",
            diff.threshold * 100.0
        ));
    }
    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

fn dot_id(id: &str) -> String {
    let clean: String = id
        .chars()
        .map(|c| match c.is_ascii_alphanumeric() || c == '_' {
            true => c,
            false => '_',
        })
        .collect();
    format!("\"{clean}\"")
}

/// Colour of what changed against `--baseline`, in every renderer.
const CHANGED: &str = "#e6550d";

type ChangedEdges<'a> = BTreeMap<(&'a str, &'a str), &'a EdgeDiff>;

/// Changed nodes and edges of the baseline diff, by id (empty without `--baseline`).
fn changed(diff: Option<&BaselineDiff>) -> (BTreeMap<&str, &NodeDiff>, ChangedEdges<'_>) {
    let Some(diff) = diff else {
        return Default::default();
    };
    (
        diff.nodes
            .iter()
            .filter(|n| n.changed)
            .map(|n| (n.id.as_str(), n))
            .collect(),
        diff.edges
            .iter()
            .filter(|e| e.changed)
            .map(|e| ((e.source.as_str(), e.target.as_str()), e))
            .collect(),
    )
}

fn dot_node(report: &GraphReport, node: &NodeReport, diff: Option<&NodeDiff>) -> String {
    let size = node_size(&node.metrics);
    let shape = match node
        .metrics
        .is_outlier(report.parameters.kurtosis_threshold)
    {
        true => "octagon",
        false => match node.kind {
            NodeKind::Writer => "box",
            NodeKind::Reader => "box",
        },
    };
    let style = match node.kind {
        NodeKind::Writer => "filled",
        NodeKind::Reader => "filled,dashed",
    };
    let mut label = format!(
        "{}\n{}\nerank {:.1}-{:.1}\nkurt {:.1}",
        node.id,
        node.tensor.replace('"', "'"),
        node.metrics.erank_energy_lower,
        node.metrics.erank_energy_upper,
        node.metrics.kurtosis
    );
    if let Some(diff) = diff {
        label.push_str(&format!(
            "\nΔ norm {:+.2}% · Δ erank {:+.2}",
            diff.relative_change * 100.0,
            diff.effective_rank_delta
        ));
    }
    let outlier = node
        .metrics
        .is_outlier(report.parameters.kurtosis_threshold);
    let (border, pen) = match (diff, outlier) {
        (Some(_), _) => (CHANGED, 3.4),
        (None, true) => ("#d62728", 1.2),
        (None, false) => ("#555555", 1.2),
    };
    format!(
        "    {} [label=\"{}\", shape={shape}, style=\"{style}\", fillcolor=\"{}\", color=\"{border}\", penwidth={pen:.1}, width={size:.2}, height={size:.2}];\n",
        dot_id(&node.id),
        dot_escape(&label),
        rgba(role_color(node.role), 70),
    )
}

// ------------------------------------------------------------------ Mermaid

fn mermaid_escape(text: &str) -> String {
    // Labels are built here (`<br/>` is intentional); only ampersands and quotes could
    // break the Mermaid parser.
    text.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('\n', " ")
}

fn mermaid_id(id: &str) -> String {
    let mut clean: String = id
        .chars()
        .map(|c| match c.is_ascii_alphanumeric() || c == '_' {
            true => c,
            false => '_',
        })
        .collect();
    if clean.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        clean.insert(0, 'n');
    }
    clean
}

/// Mermaid flowchart with hexagon nodes, class definitions and one `linkStyle` per edge.
pub fn render_mermaid(report: &GraphReport) -> String {
    let mut weights: Vec<f64> = report.edges.iter().map(|e| e.weight).collect();
    weights.sort_by(f64::total_cmp);
    let mut out =
        String::from("%% surgeon graph - deterministic write/read alignment\nflowchart LR\n");
    for node in &report.nodes {
        if node.role == GraphRole::Embed {
            out.push_str(&mermaid_node(report, node));
        }
    }
    for layer in 0..report.global.layers {
        out.push_str(&format!("  subgraph layer_{layer}[\"layer {layer}\"]\n"));
        for node in report.nodes.iter().filter(|n| n.layer == Some(layer)) {
            out.push_str(&mermaid_node(report, node));
        }
        out.push_str("  end\n");
    }
    for node in report.nodes.iter().filter(|n| n.role == GraphRole::LmHead) {
        out.push_str(&mermaid_node(report, node));
    }
    for edge in &report.edges {
        let label = match edge.structural {
            true => "structural".to_owned(),
            false => format!("{:.3}", edge.weight),
        };
        let arrow = match edge.structural {
            true => "-.->",
            false => "-->",
        };
        out.push_str(&format!(
            "  {} {arrow}|\"{}\"| {}\n",
            mermaid_id(&edge.source),
            mermaid_escape(&label),
            mermaid_id(&edge.target)
        ));
    }
    out.push_str("  classDef writer fill:#4c78a8,stroke:#26445f,color:#ffffff\n");
    out.push_str(
        "  classDef reader fill:#f2f2f2,stroke:#666666,color:#111111,stroke-dasharray:4 3\n",
    );
    out.push_str("  classDef outlier fill:#fde0dd,stroke:#d62728,stroke-width:3px,color:#111111\n");
    let (changed_nodes, changed_edges) = changed(report.baseline_diff.as_ref());
    if !changed_nodes.is_empty() {
        out.push_str(&format!(
            "  classDef changed stroke:{CHANGED},stroke-width:4px\n"
        ));
    }
    let mut classes: Vec<String> = Vec::new();
    for node in &report.nodes {
        let class = match node
            .metrics
            .is_outlier(report.parameters.kurtosis_threshold)
        {
            true => "outlier",
            false => match node.kind {
                NodeKind::Writer => "writer",
                NodeKind::Reader => "reader",
            },
        };
        classes.push(format!("  class {} {class}\n", mermaid_id(&node.id)));
    }
    for line in classes {
        out.push_str(&line);
    }
    for id in changed_nodes.keys() {
        out.push_str(&format!("  class {} changed\n", mermaid_id(id)));
    }
    for (index, edge) in report.edges.iter().enumerate() {
        if changed_edges.contains_key(&(edge.source.as_str(), edge.target.as_str())) {
            out.push_str(&format!(
                "  linkStyle {index} stroke:{CHANGED},stroke-width:4px\n"
            ));
        } else if edge.structural {
            out.push_str(&format!(
                "  linkStyle {index} stroke:#888888,stroke-width:1px,stroke-dasharray:4 3\n"
            ));
        } else {
            let color = edge_color(report, edge.weight, &weights);
            let width = 0.8 + 3.0 * weight_strength(report, edge.weight, &weights);
            out.push_str(&format!(
                "  linkStyle {index} stroke:{},stroke-width:{width:.2}px\n",
                hex(color)
            ));
        }
    }
    out
}

fn mermaid_node(report: &GraphReport, node: &NodeReport) -> String {
    let label = format!(
        "{}<br/>{}<br/>erank {:.1}-{:.1}",
        node.id,
        match node.kind {
            NodeKind::Writer => "writer",
            NodeKind::Reader => "reader",
        },
        node.metrics.erank_energy_lower,
        node.metrics.erank_energy_upper
    );
    let _ = report;
    format!(
        "    {}{{{{\"{}\"}}}}\n",
        mermaid_id(&node.id),
        mermaid_escape(&label)
    )
}

// ------------------------------------------------------------------ CSV

pub fn render_matrices_csv(report: &GraphReport) -> String {
    let mut out = String::from(
        "id,role,kind,layer,expert,tensor,dtype,rows,cols,rank,frobenius_norm,energy,residual_energy,range_energy,rel_error,sigma1,sigma1_lower,sigma1_upper,d_spec,d_spec_lower,d_spec_upper,erank_energy_lower,erank_energy_upper,erank_l1,full_spectrum,kurtosis,kurtosis_excess,channel_outliers,non_finite,weight_bytes\n",
    );
    for node in &report.nodes {
        let m = &node.metrics;
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            csv(&node.id),
            node.role.as_str(),
            match node.kind {
                NodeKind::Writer => "writer",
                NodeKind::Reader => "reader",
            },
            optional(node.layer),
            optional(node.expert),
            csv(&node.tensor),
            csv(&node.dtype),
            m.rows,
            m.cols,
            m.rank,
            num(m.frobenius_norm),
            num(m.energy),
            num(m.residual_energy),
            num(m.range_energy),
            num(m.rel_error),
            optional_num(m.sigma1),
            optional_num(m.sigma1_lower),
            optional_num(m.sigma1_upper),
            optional_num(m.d_spec),
            optional_num(m.d_spec_lower),
            optional_num(m.d_spec_upper),
            num(m.erank_energy_lower),
            num(m.erank_energy_upper),
            optional_num(m.erank_l1),
            m.full_spectrum,
            num(m.kurtosis),
            num(m.kurtosis_excess),
            csv(&m.channels.outliers.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(";")),
            m.non_finite,
            node.weight_bytes
        ));
    }
    out
}

pub fn render_edges_csv(report: &GraphReport) -> String {
    let mut out = String::from(
        "source,target,writer_role,reader_role,source_seq,target_seq,layer_gap,rank_used,alignment,chance,alignment_norm,coupling,cos_theta,weight,structural\n",
    );
    for edge in &report.edges {
        out.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            csv(&edge.source),
            csv(&edge.target),
            edge.writer_role.as_str(),
            edge.reader_role.as_str(),
            edge.source_seq,
            edge.target_seq,
            edge.layer_gap,
            edge.rank_used,
            num(edge.alignment),
            num(edge.chance),
            num(edge.alignment_norm),
            num(edge.coupling),
            num(edge.cos_theta),
            num(edge.weight),
            edge.structural
        ));
    }
    out
}

fn csv(value: &str) -> String {
    match value.contains([',', '"', '\n', '\r']) {
        true => format!("\"{}\"", value.replace('"', "\"\"")),
        false => value.to_owned(),
    }
}

fn num(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.12}")
    } else {
        "null".to_owned()
    }
}

fn optional<T: std::fmt::Display>(value: Option<T>) -> String {
    value.map_or_else(String::new, |v| v.to_string())
}

fn optional_num(value: Option<f64>) -> String {
    value.map_or_else(String::new, num)
}

// ------------------------------------------------------------------ summary

/// Bounded inline view: never the full graph, even without `--out-dir`.
pub fn summary(
    report: &GraphReport,
    outputs: &[PathBuf],
    max_nodes: usize,
    max_edges: usize,
) -> Value {
    let node_rows: Vec<Value> = report
        .nodes
        .iter()
        .take(max_nodes)
        .map(|node| {
            json!({
                "id": node.id,
                "role": node.role.as_str(),
                "kind": node.kind,
                "layer": node.layer,
                "expert": node.expert,
                "rank": node.metrics.rank,
                "erank_energy_lower": node.metrics.erank_energy_lower,
                "erank_energy_upper": node.metrics.erank_energy_upper,
                "kurtosis": node.metrics.kurtosis,
                "outlier": node.metrics.is_outlier(report.parameters.kurtosis_threshold),
            })
        })
        .collect();
    let mut top: Vec<&EdgeReport> = report
        .edges
        .iter()
        .filter(|edge| !edge.structural)
        .collect();
    top.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.source.cmp(&b.source))
            .then_with(|| a.target.cmp(&b.target))
    });
    let edge_rows: Vec<Value> = top
        .iter()
        .take(max_edges)
        .map(|edge| {
            json!({
                "source": edge.source,
                "target": edge.target,
                "alignment": edge.alignment,
                "alignment_norm": edge.alignment_norm,
                "coupling": edge.coupling,
                "cos_theta": edge.cos_theta,
                "weight": edge.weight,
            })
        })
        .collect();
    json!({
        "schema_version": report.schema_version,
        "model": {
            "path": &report.model.path,
            "sha256": &report.model.sha256,
            "hash_mode": &report.model.hash_mode,
            "format": &report.model.format,
            "architecture": &report.model.architecture,
            "hidden": report.model.hidden,
            "layers": report.model.layers,
            "tied_lm_head": report.model.tied_lm_head,
        },
        "parameters": &report.parameters,
        "global": &report.global,
        "outputs": outputs.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(),
        "nodes": node_rows,
        "top_edges": edge_rows,
        "truncated": {
            "nodes": report.nodes.len() > node_rows.len(),
            "edges": report.edges.len() > edge_rows.len(),
        },
        "baseline_diff": report.baseline_diff.as_ref().map(|diff| json!({
            "baseline": diff.baseline,
            "source": diff.source,
            "baseline_sha256": diff.baseline_sha256,
            "baseline_hash_mode": diff.baseline_hash_mode,
            "threshold": diff.threshold,
            "nodes_compared": diff.nodes_compared,
            "nodes_changed": diff.nodes_changed,
            "edges_compared": diff.edges_compared,
            "edges_changed": diff.edges_changed,
            "only_in_model": diff.only_in_model.len(),
            "only_in_baseline": diff.only_in_baseline.len(),
            "warnings": diff.warnings,
            "top_changed_layers": diff.top_changed_layers,
        })),
        "note": "graph.json is the complete report; this envelope is a bounded summary",
    })
}

pub fn human_summary(report: &GraphReport, outputs: &[PathBuf]) -> String {
    let mut text = format!(
        "surgeon graph · {} matrices · {} edges\nmodel: {} (sha256 {})\n",
        report.global.nodes,
        report.global.edges,
        report.model.path,
        &report.model.sha256[..report.model.sha256.len().min(12)]
    );
    if report.global.outlier_channels.is_empty() {
        text.push_str("outlier channels: none\n");
    } else {
        text.push_str(&format!(
            "outlier channels: {} ({})\n",
            report.global.outlier_channels.len(),
            report
                .global
                .outlier_channels
                .iter()
                .take(12)
                .map(|c| c.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    if let Some(diff) = &report.baseline_diff {
        text.push_str(&format!(
            "baseline: {} · {}/{} matrices and {}/{} edges changed\n",
            diff.baseline,
            diff.nodes_changed,
            diff.nodes_compared,
            diff.edges_changed,
            diff.edges_compared
        ));
        for layer in diff.top_changed_layers.iter().take(3) {
            text.push_str(&format!(
                "  {}: {:.3}% ({})\n",
                layer
                    .layer
                    .map_or_else(|| "embed/head".to_owned(), |l| format!("layer {l}")),
                layer.max_relative_change * 100.0,
                layer.top_node
            ));
        }
    }
    if outputs.is_empty() {
        text.push_str("outputs: none (no --out-dir)\n");
    } else {
        text.push_str("outputs:\n");
        for path in outputs {
            text.push_str(&format!("  {}\n", path.display()));
        }
    }
    text
}

// ------------------------------------------------------------------ safe writes

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct OutputFile {
    pub name: &'static str,
    pub contents: String,
}

/// Writes `files` under `out_dir` with exclusive creation and symlink refusal.
pub fn write_outputs(
    out_dir: &Path,
    files: &[OutputFile],
    overwrite: bool,
) -> Result<Vec<PathBuf>> {
    prepare_dir(out_dir)?;
    let mut written = Vec::with_capacity(files.len());
    for file in files {
        let target = out_dir.join(file.name);
        refuse_symlink(&target)?;
        if target.exists() {
            if !overwrite {
                return Err(GraphError::Io {
                    path: target.display().to_string(),
                    message: "existe déjà (utiliser --overwrite)".into(),
                });
            }
            if target.is_dir() {
                return Err(GraphError::Io {
                    path: target.display().to_string(),
                    message: "est un dossier".into(),
                });
            }
        }
        let temp = out_dir.join(format!(
            ".{}.tmp-{}-{}",
            file.name,
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        refuse_symlink(&temp)?;
        let result = write_one(&temp, &target, file.contents.as_bytes(), overwrite);
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        result?;
        written.push(target);
    }
    Ok(written)
}

fn prepare_dir(dir: &Path) -> Result<()> {
    match dir.symlink_metadata() {
        Ok(meta) if meta.file_type().is_symlink() => Err(GraphError::Io {
            path: dir.display().to_string(),
            message: "dossier symbolique refusé".into(),
        }),
        Ok(meta) if !meta.is_dir() => Err(GraphError::Io {
            path: dir.display().to_string(),
            message: "n'est pas un dossier".into(),
        }),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            std::fs::create_dir_all(dir).map_err(|e| GraphError::io(dir, &e))
        }
        Err(error) => Err(GraphError::io(dir, &error)),
    }
}

fn refuse_symlink(path: &Path) -> Result<()> {
    match path.symlink_metadata() {
        Ok(meta) if meta.file_type().is_symlink() => Err(GraphError::Io {
            path: path.display().to_string(),
            message: "lien symbolique refusé".into(),
        }),
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(GraphError::io(path, &error)),
    }
}

fn write_one(temp: &Path, target: &Path, bytes: &[u8], overwrite: bool) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(temp)
        .map_err(|e| GraphError::io(temp, &e))?;
    file.write_all(bytes)
        .map_err(|e| GraphError::io(temp, &e))?;
    file.sync_all().map_err(|e| GraphError::io(temp, &e))?;
    drop(file);
    if overwrite {
        match std::fs::remove_file(target) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(GraphError::io(target, &error)),
        }
    }
    std::fs::rename(temp, target).map_err(|e| GraphError::io(target, &e))
}

/// Removes an old cache file, refusing a symlink.
pub fn remove_cache_file(path: &Path) -> Result<()> {
    refuse_symlink(path)?;
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(GraphError::io(path, &error)),
    }
}

pub fn open_cache_append(path: &Path) -> Result<File> {
    refuse_symlink(path)?;
    if let Some(parent) = path.parent() {
        prepare_dir(parent)?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| GraphError::io(path, &e))
}

/// Cache directories follow the same symlink refusal as output directories.
pub fn prepare_cache_dir(dir: &Path) -> Result<()> {
    prepare_dir(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::metrics::ChannelStats;

    fn metrics() -> NodeMetrics {
        NodeMetrics {
            rows: 4,
            cols: 4,
            rank: 2,
            frobenius_norm: 3.0,
            energy: 0.9,
            residual_energy: 0.1,
            range_energy: 0.95,
            rel_error: 0.3,
            sigma1: Some(2.0),
            sigma1_lower: Some(2.0),
            sigma1_upper: Some(2.5),
            d_spec: Some(2.25),
            d_spec_lower: Some(1.4),
            d_spec_upper: Some(2.25),
            erank_energy_lower: 1.7,
            erank_energy_upper: 2.2,
            erank_l1: None,
            full_spectrum: false,
            kurtosis: 4.0,
            kurtosis_excess: 1.0,
            channels: ChannelStats {
                count: 4,
                norm_median: 1.0,
                norm_max: 2.0,
                norm_argmax: 1,
                kurtosis_max: 4.0,
                outliers: vec![],
            },
            e_write: None,
            e_read: None,
            non_finite: 0,
            singular_values: vec![2.0, 1.0],
        }
    }

    fn report() -> GraphReport {
        GraphReport {
            schema_version: SCHEMA_VERSION,
            tool: ToolInfo {
                name: "surgeon".into(),
                version: "0.0.0".into(),
            },
            model: ModelReport {
                path: "m".into(),
                sha256: "a".repeat(64),
                hash_mode: "sha256".into(),
                format: "safetensors".into(),
                architecture: "qwen2".into(),
                arch_source: "config_model_type".into(),
                hidden: 4,
                layers: 1,
                tied_lm_head: false,
                tensor_count: 2,
                total_weight_bytes: 64,
                skipped: vec![],
                warnings: vec![],
            },
            parameters: ParametersReport {
                rank: 2,
                oversample: 2,
                power_iters: 1,
                seed: 0,
                full_spectrum_max_dim: 2048,
                edge_mode: "adjacent".into(),
                min_alignment: 0.0,
                edge_weight: "alignment".into(),
                vector: None,
                kurtosis_threshold: 10.0,
                color_scale: "quantile".into(),
                palette: "diverging".into(),
                max_nodes: None,
                formats: vec!["json".into()],
            },
            global: GlobalReport {
                nodes: 2,
                edges: 1,
                structural_edges: 0,
                layers: 1,
                hidden: 4,
                outlier_channels: vec![],
                kurtosis_median: 4.0,
                kurtosis_max: 4.0,
                mean_erank_energy_lower: 1.7,
                mean_erank_energy_upper: 2.2,
                full_spectrum_nodes: 0,
                non_finite_values: 0,
                warnings: vec![],
            },
            nodes: vec![
                NodeReport {
                    id: "embed".into(),
                    role: GraphRole::Embed,
                    kind: NodeKind::Writer,
                    layer: None,
                    expert: None,
                    tensor: "embed".into(),
                    dtype: "F32".into(),
                    weight_bytes: 32,
                    rows: 4,
                    cols: 4,
                    seq: 0,
                    metrics: metrics(),
                    pos_column: [0.0; 3],
                    pos_constellation: [0.0; 3],
                },
                NodeReport {
                    id: "L0.q".into(),
                    role: GraphRole::Q,
                    kind: NodeKind::Reader,
                    layer: Some(0),
                    expert: None,
                    tensor: "q".into(),
                    dtype: "F32".into(),
                    weight_bytes: 32,
                    rows: 4,
                    cols: 4,
                    seq: 1,
                    metrics: metrics(),
                    pos_column: [0.0; 3],
                    pos_constellation: [0.0; 3],
                },
            ],
            edges: vec![EdgeReport {
                source: "embed".into(),
                target: "L0.q".into(),
                writer_role: GraphRole::Embed,
                reader_role: GraphRole::Q,
                source_seq: 0,
                target_seq: 1,
                layer_gap: 0,
                rank_used: 2,
                alignment: 0.8,
                chance: 0.5,
                alignment_norm: 0.6,
                coupling: 1.2,
                cos_theta: 0.9,
                weight: 0.8,
                structural: false,
            }],
            baseline_diff: None,
        }
    }

    #[test]
    fn renderers_include_every_node_and_edge() {
        let report = report();
        let dot = render_dot(&report);
        assert!(dot.contains("cluster_L0"));
        assert!(dot.contains("\"L0_q\""));
        assert!(dot.contains("\"embed\" -> \"L0_q\""));
        assert!(dot.contains("octagon"));
        let mmd = render_mermaid(&report);
        assert!(mmd.contains("{{"));
        assert!(mmd.contains("linkStyle 0"));
        let csv = render_matrices_csv(&report);
        assert_eq!(csv.lines().count(), 3);
        let edges = render_edges_csv(&report);
        assert_eq!(edges.lines().count(), 2);
    }

    #[test]
    fn summary_is_bounded_and_never_contains_the_full_graph() {
        let report = report();
        let summary = summary(&report, &[], 1, 1);
        assert_eq!(summary["nodes"].as_array().unwrap().len(), 1);
        assert_eq!(summary["top_edges"].as_array().unwrap().len(), 1);
        assert_eq!(summary["truncated"]["nodes"], true);
        assert_eq!(summary["truncated"]["edges"], false);
    }

    #[test]
    fn writes_refuse_overwrite_and_symlinks() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let files = [OutputFile {
            name: "graph.json",
            contents: "{}".into(),
        }];
        let written = write_outputs(&out, &files, false).unwrap();
        assert_eq!(written.len(), 1);
        assert!(write_outputs(&out, &files, false).is_err());
        assert!(write_outputs(&out, &files, true).is_ok());
    }
}
