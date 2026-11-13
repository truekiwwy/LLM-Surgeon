//! Incremental graph states for a live view (an orchestrator, a GUI) or a replayed history,
//! one JSON object per NDJSON line.
//!
//! A stream starts with a `snapshot` (the full report). Each later state is an `update` that
//! carries only what changed: the nodes whose analysis changed, with their delta against the
//! previous state, the nodes that only moved, the changed edges and the new `model` and
//! `global` blocks (warnings included) when they differ. A different node or edge set, other
//! parameters or another `baseline_diff` make a new snapshot. Applying the patches in order
//! rebuilds every state exactly: `prev.apply(between(prev, next)) == next`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::diff::{node_diff, NodeDiff};
use super::error::{GraphError, Result};
use super::layout::Position;
use super::output::{EdgeReport, GlobalReport, GraphReport, ModelReport, NodeReport};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GraphPatch {
    Snapshot {
        report: Box<GraphReport>,
    },
    Update {
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        nodes: Vec<NodeUpdate>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        moves: Vec<NodeMove>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        edges: Vec<EdgeReport>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        model: Option<Box<ModelReport>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        global: Option<Box<GlobalReport>>,
    },
}

/// A node whose analysis changed, with its delta against the previous state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeUpdate {
    pub node: NodeReport,
    pub diff: NodeDiff,
}

/// A node whose analysis is unchanged but whose layout positions moved.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeMove {
    pub id: String,
    pub pos_column: Position,
    pub pos_constellation: Position,
}

impl GraphPatch {
    pub fn snapshot(report: &GraphReport) -> Self {
        Self::Snapshot {
            report: Box::new(report.clone()),
        }
    }

    /// The patch that turns `prev` into `next`.
    pub fn between(prev: &GraphReport, next: &GraphReport) -> Self {
        let same_shape = prev.schema_version == next.schema_version
            && prev.tool == next.tool
            && prev.parameters == next.parameters
            && prev.baseline_diff == next.baseline_diff
            && prev.nodes.len() == next.nodes.len()
            && prev.edges.len() == next.edges.len()
            && prev
                .nodes
                .iter()
                .zip(&next.nodes)
                .all(|(a, b)| a.id == b.id)
            && prev
                .edges
                .iter()
                .zip(&next.edges)
                .all(|(a, b)| a.source == b.source && a.target == b.target);
        if !same_shape {
            return Self::snapshot(next);
        }
        let mut nodes = Vec::new();
        let mut moves = Vec::new();
        for (old, new) in prev.nodes.iter().zip(&next.nodes) {
            if old == new {
                continue;
            }
            let mut moved = old.clone();
            moved.pos_column = new.pos_column;
            moved.pos_constellation = new.pos_constellation;
            match moved == *new {
                true => moves.push(NodeMove {
                    id: new.id.clone(),
                    pos_column: new.pos_column,
                    pos_constellation: new.pos_constellation,
                }),
                false => nodes.push(NodeUpdate {
                    node: new.clone(),
                    diff: node_diff(old, new),
                }),
            }
        }
        Self::Update {
            nodes,
            moves,
            edges: prev
                .edges
                .iter()
                .zip(&next.edges)
                .filter(|(old, new)| old != new)
                .map(|(_, new)| new.clone())
                .collect(),
            model: (prev.model != next.model).then(|| Box::new(next.model.clone())),
            global: (prev.global != next.global).then(|| Box::new(next.global.clone())),
        }
    }

    /// Applies the patch to `report`. An update naming an unknown node or edge is refused
    /// before anything changes: the receiver should ask for a new snapshot.
    pub fn apply(&self, report: &mut GraphReport) -> Result<()> {
        let (nodes, moves, edges, model, global) = match self {
            Self::Snapshot { report: next } => {
                *report = (**next).clone();
                return Ok(());
            }
            Self::Update {
                nodes,
                moves,
                edges,
                model,
                global,
            } => (nodes, moves, edges, model, global),
        };
        let node_index: HashMap<&str, usize> = report
            .nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.id.as_str(), i))
            .collect();
        let edge_index: HashMap<(&str, &str), usize> = report
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| ((e.source.as_str(), e.target.as_str()), i))
            .collect();
        let find_node = |id: &str| {
            node_index
                .get(id)
                .copied()
                .ok_or_else(|| GraphError::Usage(format!("patch : nœud inconnu {id}")))
        };
        let node_slots = nodes
            .iter()
            .map(|u| find_node(&u.node.id))
            .collect::<Result<Vec<_>>>()?;
        let move_slots = moves
            .iter()
            .map(|m| find_node(&m.id))
            .collect::<Result<Vec<_>>>()?;
        let edge_slots = edges
            .iter()
            .map(|e| {
                edge_index
                    .get(&(e.source.as_str(), e.target.as_str()))
                    .copied()
                    .ok_or_else(|| {
                        GraphError::Usage(format!(
                            "patch : arête inconnue {} -> {}",
                            e.source, e.target
                        ))
                    })
            })
            .collect::<Result<Vec<_>>>()?;

        for (slot, update) in node_slots.into_iter().zip(nodes) {
            report.nodes[slot] = update.node.clone();
        }
        for (slot, step) in move_slots.into_iter().zip(moves) {
            report.nodes[slot].pos_column = step.pos_column;
            report.nodes[slot].pos_constellation = step.pos_constellation;
        }
        for (slot, edge) in edge_slots.into_iter().zip(edges) {
            report.edges[slot] = edge.clone();
        }
        if let Some(model) = model {
            report.model = (**model).clone();
        }
        if let Some(global) = global {
            report.global = (**global).clone();
        }
        Ok(())
    }

    /// One NDJSON line (compact JSON and `\n`).
    pub fn to_ndjson(&self) -> Result<String> {
        let mut line = serde_json::to_string(self)
            .map_err(|e| GraphError::Compute(format!("patch NDJSON : {e}")))?;
        line.push('\n');
        Ok(line)
    }
}

/// `graph.ndjson` of a `graph` run: the snapshot of the model alone, or with `--baseline` the
/// snapshot of the reference then the update to the model. Neither state carries
/// `baseline_diff` (the per-node deltas of the update are that comparison), and the reference
/// takes the run's `formats`, which describe the export rather than the analysis: a
/// `graph.json` exported with other formats still gives an update, not a second snapshot.
pub fn history(report: &GraphReport, baseline: Option<&GraphReport>) -> Result<String> {
    let mut after = report.clone();
    after.baseline_diff = None;
    let Some(baseline) = baseline else {
        return GraphPatch::snapshot(&after).to_ndjson();
    };
    let mut before = baseline.clone();
    before.baseline_diff = None;
    before.parameters.formats = after.parameters.formats.clone();
    Ok(GraphPatch::snapshot(&before).to_ndjson()?
        + &GraphPatch::between(&before, &after).to_ndjson()?)
}
