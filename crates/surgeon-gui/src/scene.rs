//! Where the 3D view puts each node, from its role, layer and expert alone (the report's
//! spectral layouts are not used: a graph reads by its architecture first).
//!
//! - Column: one row per layer, bottom to top; attention on the left, MLP on the right (a
//!   third of a row higher, it runs after), the writers next to the residual stream in the middle,
//!   embeddings and head on the stream itself. Experts step back in depth.
//! - Helix: the same heights, each layer a ring around the stream turned a little from the
//!   one below, so that depth carries the relief.

use std::f32::consts::TAU;

use weight_surgeon::graph::layout::Position;
use weight_surgeon::graph::output::NodeReport;
use weight_surgeon::graph::GraphRole;

/// World radius of a node (the viewport scales it with depth).
pub const NODE_RADIUS: f32 = 0.02;
/// Radius of a layer's ring in the helix.
pub const RING: f32 = 0.6;
/// Turn of each ring over the one below.
const TWIST: f32 = 0.45;

/// The two layouts of a graph and the anchors of its layer marks.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Scene {
    pub column: Vec<Position>,
    pub helix: Vec<Position>,
    /// `(layer, height)` of each row.
    pub layers: Vec<(usize, f32)>,
    /// The residual stream, bottom and top.
    pub spine: [Position; 2],
}

/// Rows of a model of `layers` layers: two node diameters apart at least, and the whole
/// model within the default view when it fits.
fn spacing(layers: usize) -> f32 {
    (2.0 / (layers as f32 + 2.0)).clamp(2.5 * NODE_RADIUS, 0.12)
}

/// Horizontal place of a role in its row; MLP rows sit a third of a row up.
fn place(role: GraphRole) -> (f32, f32) {
    match role {
        GraphRole::Q => (-0.78, 0.0),
        GraphRole::K => (-0.62, 0.0),
        GraphRole::V => (-0.46, 0.0),
        GraphRole::AttnOut => (-0.2, 0.0),
        GraphRole::MlpOut => (0.2, 0.3),
        GraphRole::Gate => (0.46, 0.3),
        GraphRole::Up => (0.62, 0.3),
        GraphRole::Embed | GraphRole::LmHead => (0.0, 0.0),
    }
}

/// Angle of a role on its ring, before the layer's twist.
fn angle(role: GraphRole) -> f32 {
    let order = [
        GraphRole::Q,
        GraphRole::K,
        GraphRole::V,
        GraphRole::AttnOut,
        GraphRole::MlpOut,
        GraphRole::Gate,
        GraphRole::Up,
    ];
    let k = order.iter().position(|r| *r == role).unwrap_or(0);
    k as f32 * TAU / order.len() as f32
}

impl Scene {
    pub fn new(nodes: &[NodeReport], layers: usize) -> Self {
        let layers = nodes
            .iter()
            .filter_map(|n| n.layer.map(|l| l + 1))
            .max()
            .unwrap_or(0)
            .max(layers);
        let s = spacing(layers);
        // Row 0 is the embedding, rows 1..=layers the layers, the head on top.
        let bottom = -s * (layers as f32 + 1.0) / 2.0;
        let row = |r: f32| bottom + s * r;
        let experts = nodes
            .iter()
            .filter_map(|n| n.expert)
            .max()
            .map_or(1, |e| e + 1);
        let depth = |e: Option<usize>| match e {
            Some(e) => (e as f32 - (experts as f32 - 1.0) / 2.0) * 2.5 * NODE_RADIUS,
            None => 0.0,
        };
        let (mut column, mut helix) = (Vec::new(), Vec::new());
        for n in nodes {
            let y = match (n.role, n.layer) {
                (GraphRole::Embed, _) => row(0.0),
                (GraphRole::LmHead, _) => row(layers as f32 + 1.0),
                (role, Some(l)) => row(l as f32 + 1.0 + place(role).1),
                (_, None) => row(0.0),
            };
            let (x, _) = place(n.role);
            let z = depth(n.expert);
            column.push([x, y, z]);
            match n.role {
                GraphRole::Embed | GraphRole::LmHead => helix.push([0.0, y, 0.0]),
                role => {
                    let a = angle(role) + TWIST * n.layer.unwrap_or(0) as f32;
                    let r = RING + z.abs();
                    helix.push([r * a.cos(), y, r * a.sin()]);
                }
            }
        }
        let marks = (0..layers).map(|l| (l, row(l as f32 + 1.0))).collect();
        Self {
            column,
            helix,
            layers: marks,
            spine: [[0.0, row(0.0), 0.0], [0.0, row(layers as f32 + 1.0), 0.0]],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit;

    fn report(layers: usize) -> Vec<NodeReport> {
        let template = testkit::graph(1).nodes[0].clone();
        let roles = [
            GraphRole::Q,
            GraphRole::K,
            GraphRole::V,
            GraphRole::AttnOut,
            GraphRole::Gate,
            GraphRole::Up,
            GraphRole::MlpOut,
        ];
        let mut nodes = Vec::new();
        for l in 0..layers {
            for role in roles {
                let mut n = template.clone();
                (n.role, n.layer, n.expert) = (role, Some(l), None);
                nodes.push(n);
            }
        }
        for role in [GraphRole::Embed, GraphRole::LmHead] {
            let mut n = template.clone();
            (n.role, n.layer) = (role, None);
            nodes.push(n);
        }
        nodes
    }

    fn min_distance(points: &[Position]) -> f32 {
        let mut min = f32::MAX;
        for (i, a) in points.iter().enumerate() {
            for b in &points[i + 1..] {
                let d = (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f32>().sqrt();
                min = min.min(d);
            }
        }
        min
    }

    #[test]
    fn no_two_nodes_overlap_in_either_layout_even_on_deep_models() {
        for layers in [2, 24, 80] {
            let nodes = report(layers);
            let scene = Scene::new(&nodes, layers);
            assert!(min_distance(&scene.column) > 2.0 * NODE_RADIUS, "{layers}");
            assert!(min_distance(&scene.helix) > 2.0 * NODE_RADIUS, "{layers}");
            assert_eq!(scene.layers.len(), layers);
        }
    }

    #[test]
    fn the_column_puts_attention_left_mlp_right_and_the_stream_in_the_middle() {
        let nodes = report(3);
        let scene = Scene::new(&nodes, 3);
        for (n, p) in nodes.iter().zip(&scene.column) {
            match n.role {
                GraphRole::Q | GraphRole::K | GraphRole::V | GraphRole::AttnOut => {
                    assert!(p[0] < 0.0)
                }
                GraphRole::Gate | GraphRole::Up | GraphRole::MlpOut => assert!(p[0] > 0.0),
                GraphRole::Embed | GraphRole::LmHead => assert_eq!(p[0], 0.0),
            }
        }
        let y = |role, layer| {
            let i = nodes
                .iter()
                .position(|n| n.role == role && n.layer == layer)
                .unwrap();
            scene.column[i][1]
        };
        assert!(y(GraphRole::Embed, None) < y(GraphRole::Q, Some(0)));
        assert!(y(GraphRole::Q, Some(0)) < y(GraphRole::MlpOut, Some(0)));
        assert!(y(GraphRole::MlpOut, Some(0)) < y(GraphRole::Q, Some(1)));
        assert!(y(GraphRole::Q, Some(2)) < y(GraphRole::LmHead, None));
    }

    #[test]
    fn the_helix_turns_from_layer_to_layer_and_has_depth() {
        let nodes = report(4);
        let scene = Scene::new(&nodes, 4);
        let q: Vec<Position> = nodes
            .iter()
            .zip(&scene.helix)
            .filter(|(n, _)| n.role == GraphRole::Q)
            .map(|(_, p)| *p)
            .collect();
        assert!(q
            .windows(2)
            .all(|w| w[0][0] != w[1][0] || w[0][2] != w[1][2]));
        let depths: Vec<f32> = scene.helix.iter().map(|p| p[2]).collect();
        let spread = depths.iter().cloned().fold(f32::MIN, f32::max)
            - depths.iter().cloned().fold(f32::MAX, f32::min);
        assert!(spread > RING, "{spread}");
    }
}
