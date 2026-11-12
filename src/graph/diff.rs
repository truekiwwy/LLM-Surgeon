//! `--baseline`: spectral and structural difference between the model and a reference.
//!
//! The reference is either a model, analysed with the same parameters, or a `graph.json`
//! already written by `surgeon graph` (no SVD to redo). Nodes are paired by id (`L3.q`,
//! `embed`, ...), edges by `(source, target)`. Every delta is `model - baseline`, so two runs on
//! the same weights with the same parameters give exact zeros. The deltas come from the
//! reports, not from the weights: a change that keeps the Frobenius norm and the spectrum (a
//! rotation) is not seen.

use std::collections::BTreeMap;
use std::path::Path;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::error::{GraphError, Result};
use super::metrics::NodeMetrics;
use super::output::{GraphReport, NodeReport, SCHEMA_VERSION};
use super::GraphRole;

/// A node or an edge is `changed` when one of its relative deltas reaches this value (0.1 %).
pub const CHANGE_THRESHOLD: f64 = 1e-3;
/// Layers listed in `top_changed_layers`.
const TOP_LAYERS: usize = 10;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BaselineDiff {
    /// The `--baseline` path as given.
    pub baseline: String,
    /// `model` (analysed in this run) or `graph_json` (read from a previous report).
    pub source: String,
    pub baseline_sha256: String,
    pub baseline_hash_mode: String,
    /// Relative delta from which a node or an edge counts as changed.
    pub threshold: f64,
    pub nodes_compared: usize,
    pub nodes_changed: usize,
    pub edges_compared: usize,
    pub edges_changed: usize,
    /// Node ids present on one side only (`--layers`, `--matrices` or architecture differ).
    pub only_in_model: Vec<String>,
    pub only_in_baseline: Vec<String>,
    /// Parameters that differ between the two reports: their deltas mix both effects.
    pub warnings: Vec<String>,
    /// Layers with a changed node, most changed first (at most 10). `layer` is null for the
    /// embedding and the output head.
    pub top_changed_layers: Vec<LayerChange>,
    pub nodes: Vec<NodeDiff>,
    pub edges: Vec<EdgeDiff>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct NodeDiff {
    pub id: String,
    pub role: GraphRole,
    pub layer: Option<usize>,
    /// Delta of the middle of the certified energy effective-rank interval.
    pub effective_rank_delta: f64,
    /// Delta of the Frobenius norm.
    pub norm_delta: f64,
    /// `norm_delta / baseline Frobenius norm`.
    pub relative_change: f64,
    /// Delta of `sigma1` (spectral norm), when both reports have it.
    pub sigma1_delta: Option<f64>,
    pub changed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EdgeDiff {
    pub source: String,
    pub target: String,
    pub alignment_delta: f64,
    pub coupling_delta: f64,
    /// `alignment_delta / baseline alignment`.
    pub relative_change: f64,
    pub changed: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LayerChange {
    pub layer: Option<usize>,
    pub nodes_changed: usize,
    /// Largest `|relative_change|` of the layer's nodes, and the node that has it.
    pub max_relative_change: f64,
    pub top_node: String,
    pub max_abs_effective_rank_delta: f64,
}

/// A `.json` file that is not a SafeTensors index is read as a `graph.json`.
pub fn is_report_file(path: &Path) -> bool {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    path.is_file() && name.ends_with(".json") && !name.ends_with(".index.json")
}

pub fn read_report(path: &Path) -> Result<GraphReport> {
    let bytes = std::fs::read(path).map_err(|e| GraphError::io(path, &e))?;
    let report: GraphReport = serde_json::from_slice(&bytes).map_err(|e| {
        GraphError::Usage(format!(
            "--baseline {} : graph.json illisible ({e})",
            path.display()
        ))
    })?;
    if report.schema_version != SCHEMA_VERSION {
        return Err(GraphError::Usage(format!(
            "--baseline {} : schema_version {} (attendu {SCHEMA_VERSION})",
            path.display(),
            report.schema_version
        )));
    }
    Ok(report)
}

/// `delta / |base|`; from a zero baseline, ±1 (a value that appears), never infinite, so the
/// report stays valid JSON.
fn relative(delta: f64, base: f64) -> f64 {
    match base.abs() > f64::MIN_POSITIVE {
        true => delta / base.abs(),
        false if delta == 0.0 => 0.0,
        false => delta.signum(),
    }
}

fn erank_mid(metrics: &NodeMetrics) -> f64 {
    (metrics.erank_energy_lower + metrics.erank_energy_upper) / 2.0
}

fn node_diff(base: &NodeReport, node: &NodeReport) -> NodeDiff {
    let (b, m) = (&base.metrics, &node.metrics);
    let effective_rank_delta = erank_mid(m) - erank_mid(b);
    let norm_delta = m.frobenius_norm - b.frobenius_norm;
    let relative_change = relative(norm_delta, b.frobenius_norm);
    let rank_change = relative(effective_rank_delta, erank_mid(b).max(1.0));
    NodeDiff {
        id: node.id.clone(),
        role: node.role,
        layer: node.layer,
        effective_rank_delta,
        norm_delta,
        relative_change,
        sigma1_delta: m.sigma1.zip(b.sigma1).map(|(m, b)| m - b),
        changed: relative_change.abs() >= CHANGE_THRESHOLD || rank_change.abs() >= CHANGE_THRESHOLD,
    }
}

/// Compares `model` with `baseline`; `path` is the `--baseline` argument.
pub fn compare(baseline: &GraphReport, model: &GraphReport, path: &Path) -> BaselineDiff {
    let base_nodes: BTreeMap<&str, &NodeReport> =
        baseline.nodes.iter().map(|n| (n.id.as_str(), n)).collect();
    let model_ids: BTreeMap<&str, ()> = model.nodes.iter().map(|n| (n.id.as_str(), ())).collect();
    let nodes: Vec<NodeDiff> = model
        .nodes
        .iter()
        .filter_map(|node| Some(node_diff(base_nodes.get(node.id.as_str())?, node)))
        .collect();

    let base_edges: BTreeMap<(&str, &str), _> = baseline
        .edges
        .iter()
        .filter(|e| !e.structural)
        .map(|e| ((e.source.as_str(), e.target.as_str()), e))
        .collect();
    let edges: Vec<EdgeDiff> = model
        .edges
        .iter()
        .filter(|e| !e.structural)
        .filter_map(|edge| {
            let base = base_edges.get(&(edge.source.as_str(), edge.target.as_str()))?;
            let alignment_delta = edge.alignment - base.alignment;
            let relative_change = relative(alignment_delta, base.alignment);
            Some(EdgeDiff {
                source: edge.source.clone(),
                target: edge.target.clone(),
                alignment_delta,
                coupling_delta: edge.coupling - base.coupling,
                relative_change,
                changed: relative_change.abs() >= CHANGE_THRESHOLD,
            })
        })
        .collect();

    let mut layers: BTreeMap<Option<usize>, LayerChange> = BTreeMap::new();
    for node in nodes.iter().filter(|n| n.changed) {
        let entry = layers.entry(node.layer).or_insert_with(|| LayerChange {
            layer: node.layer,
            nodes_changed: 0,
            max_relative_change: 0.0,
            top_node: node.id.clone(),
            max_abs_effective_rank_delta: 0.0,
        });
        entry.nodes_changed += 1;
        if node.relative_change.abs() > entry.max_relative_change {
            entry.max_relative_change = node.relative_change.abs();
            entry.top_node = node.id.clone();
        }
        entry.max_abs_effective_rank_delta = entry
            .max_abs_effective_rank_delta
            .max(node.effective_rank_delta.abs());
    }
    let mut top_changed_layers: Vec<LayerChange> = layers.into_values().collect();
    top_changed_layers.sort_by(|a, b| {
        b.max_relative_change
            .total_cmp(&a.max_relative_change)
            .then_with(|| b.nodes_changed.cmp(&a.nodes_changed))
            .then_with(|| a.layer.cmp(&b.layer))
    });
    top_changed_layers.truncate(TOP_LAYERS);

    BaselineDiff {
        baseline: path.display().to_string(),
        source: match is_report_file(path) {
            true => "graph_json",
            false => "model",
        }
        .into(),
        baseline_sha256: baseline.model.sha256.clone(),
        baseline_hash_mode: baseline.model.hash_mode.clone(),
        threshold: CHANGE_THRESHOLD,
        nodes_compared: nodes.len(),
        nodes_changed: nodes.iter().filter(|n| n.changed).count(),
        edges_compared: edges.len(),
        edges_changed: edges.iter().filter(|e| e.changed).count(),
        only_in_model: model
            .nodes
            .iter()
            .filter(|n| !base_nodes.contains_key(n.id.as_str()))
            .map(|n| n.id.clone())
            .collect(),
        only_in_baseline: baseline
            .nodes
            .iter()
            .filter(|n| !model_ids.contains_key(n.id.as_str()))
            .map(|n| n.id.clone())
            .collect(),
        warnings: parameter_warnings(baseline, model),
        top_changed_layers,
        nodes,
        edges,
    }
}

fn parameter_warnings(baseline: &GraphReport, model: &GraphReport) -> Vec<String> {
    let (b, m) = (&baseline.parameters, &model.parameters);
    let pairs = [
        ("rank", b.rank.to_string(), m.rank.to_string()),
        (
            "oversample",
            b.oversample.to_string(),
            m.oversample.to_string(),
        ),
        (
            "power_iters",
            b.power_iters.to_string(),
            m.power_iters.to_string(),
        ),
        ("seed", b.seed.to_string(), m.seed.to_string()),
        (
            "full_spectrum_max_dim",
            b.full_spectrum_max_dim.to_string(),
            m.full_spectrum_max_dim.to_string(),
        ),
        ("edge_mode", b.edge_mode.clone(), m.edge_mode.clone()),
        (
            "min_alignment",
            b.min_alignment.to_string(),
            m.min_alignment.to_string(),
        ),
        ("edge_weight", b.edge_weight.clone(), m.edge_weight.clone()),
        (
            "kurtosis_threshold",
            b.kurtosis_threshold.to_string(),
            m.kurtosis_threshold.to_string(),
        ),
        (
            "vector",
            format!("{:?}", b.vector),
            format!("{:?}", m.vector),
        ),
    ];
    pairs
        .into_iter()
        .filter(|(_, b, m)| b != m)
        .map(|(name, b, m)| format!("paramètre {name} différent : référence {b}, modèle {m}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_change_stays_finite() {
        assert_eq!(relative(1.0, 4.0), 0.25);
        assert_eq!(relative(-1.0, -4.0), -0.25);
        assert_eq!(relative(0.0, 0.0), 0.0);
        assert_eq!(relative(3.0, 0.0), 1.0);
        assert_eq!(relative(-3.0, 0.0), -1.0);
    }

    #[test]
    fn only_a_plain_json_file_is_a_report() {
        let dir = tempfile::tempdir().unwrap();
        for name in [
            "graph.json",
            "model.safetensors.index.json",
            "model.safetensors",
        ] {
            std::fs::write(dir.path().join(name), "{}").unwrap();
        }
        assert!(is_report_file(&dir.path().join("graph.json")));
        assert!(!is_report_file(
            &dir.path().join("model.safetensors.index.json")
        ));
        assert!(!is_report_file(&dir.path().join("model.safetensors")));
        assert!(!is_report_file(dir.path()));
    }
}
