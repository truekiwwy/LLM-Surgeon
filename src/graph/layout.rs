//! Deterministic 3D layouts of the graph, computed once in Rust and stored in every node
//! (`pos_column`, `pos_constellation`), so the HTML view and any future GUI only draw.
//!
//! Both layouts are centred on the origin and scaled to the unit ball. They use the basic
//! IEEE-754 operations and `sqrt` only, which are correctly rounded everywhere, never the
//! platform libm (`sin`, `cos`, `cbrt` differ in the last bit between Windows, Linux and
//! macOS): the same report gives bit-identical coordinates on every platform. The repulsion
//! is split by node across threads, each node summing in a fixed order, so the thread count
//! changes nothing either.

use std::collections::HashMap;

use rayon::prelude::*;

use super::output::{EdgeReport, NodeReport};
use super::GraphRole;

/// A normalised 3D position (unit ball).
pub type Position = [f32; 3];

/// Ideal edge length of the force model, in simulation units.
const K: f64 = 1.8;
/// Pull of every node toward the origin, which keeps disconnected parts in view.
const GRAVITY: f64 = 0.05;
/// Adaptive step (Hu, 2005): shrunk by this factor whenever the force energy rises (an
/// oscillation), grown back after `PROGRESS` decreasing iterations in a row.
const DAMPING: f64 = 0.9;
const PROGRESS: usize = 5;
/// Converged once the mean squared step per node falls below `(TOLERANCE * K)^2`.
const TOLERANCE: f64 = 1e-3;
/// Iteration budget: `WORK / N^2` (pair interactions), within these bounds.
const WORK: f64 = 2e9;
const MIN_ITERATIONS: usize = 50;
const MAX_ITERATIONS: usize = 1000;

/// Result of the force simulation.
#[derive(Debug, Clone, PartialEq)]
pub struct Constellation {
    pub positions: Vec<Position>,
    pub iterations: usize,
    /// Mean squared step per node of the last iteration (kinetic energy of the system).
    pub energy: f64,
    /// `energy` reached the tolerance before the iteration budget ran out.
    pub converged: bool,
}

/// Fills `pos_column` and `pos_constellation` of every node.
pub fn apply(nodes: &mut [NodeReport], edges: &[EdgeReport]) {
    let column = column(nodes);
    let forces = constellation(nodes.len(), &links(nodes, edges)).positions;
    for ((node, column), forces) in nodes.iter_mut().zip(column).zip(forces) {
        node.pos_column = column;
        node.pos_constellation = forces;
    }
}

/// `Colonne par couches`: causal order (`seq`) on the vertical axis, one ring angle per
/// role, experts spread on wider rings.
pub fn column(nodes: &[NodeReport]) -> Vec<Position> {
    let max_seq = nodes.iter().map(|n| n.seq).max().unwrap_or(0).max(1) as f64;
    let points = nodes.iter().map(|node| {
        let expert = node.expert.unwrap_or(0);
        let ring = match node.role {
            GraphRole::Embed | GraphRole::LmHead => 0.0,
            _ => 3.2 + 0.45 * (expert % 8) as f64,
        };
        let angle =
            role_angle(node.role) + 0.39 * expert as f64 + 0.18 * node.layer.unwrap_or(0) as f64;
        let (sin, cos) = sin_cos(angle);
        [
            ring * cos,
            (node.seq as f64 - max_seq / 2.0) * 0.9,
            ring * sin,
        ]
    });
    normalize(points.collect())
}

fn role_angle(role: GraphRole) -> f64 {
    match role {
        GraphRole::Q | GraphRole::Embed | GraphRole::LmHead => 0.0,
        GraphRole::Gate => 0.52,
        GraphRole::AttnOut => 1.05,
        GraphRole::K => 2.09,
        GraphRole::Up => 3.67,
        GraphRole::V => 4.19,
        GraphRole::MlpOut => 5.24,
    }
}

/// Edges as `(source, target, strength in [0, 1])`: structural edges pull at 0.5, spectral
/// ones from 0.2 to 1 with their coupling. Edges to unknown nodes are ignored.
fn links(nodes: &[NodeReport], edges: &[EdgeReport]) -> Vec<(usize, usize, f64)> {
    let index: HashMap<&str, usize> = nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id.as_str(), i))
        .collect();
    let coupling = |e: &EdgeReport| match e.coupling.is_finite() {
        true => e.coupling.max(0.0),
        false => 0.0,
    };
    let max_coupling = edges.iter().map(coupling).fold(0.0, f64::max);
    edges
        .iter()
        .filter_map(|edge| {
            let source = *index.get(edge.source.as_str())?;
            let target = *index.get(edge.target.as_str())?;
            let strength = match (edge.structural, max_coupling > 0.0) {
                (true, _) => 0.5,
                (false, true) => 0.2 + 0.8 * coupling(edge) / max_coupling,
                (false, false) => 0.2,
            };
            Some((source, target, strength))
        })
        .collect()
}

/// `Constellation 3D`: Fruchterman-Reingold in 3D on `count` nodes. Every pair repels,
/// every link `(a, b, strength)` pulls with its strength, the start is a Fibonacci sphere
/// (distinct points, no random source) and an adaptive step, damped whenever the force
/// energy rises, caps each move. Stops when the kinetic energy (mean squared move) falls
/// below the tolerance or the iteration budget runs out.
pub fn constellation(count: usize, links: &[(usize, usize, f64)]) -> Constellation {
    let radius = K * cbrt(count as f64);
    let mut pos: Vec<[f64; 3]> = (0..count).map(|i| fibonacci(i, count, radius)).collect();
    // shortcut: exact O(N^2) repulsion with a shrinking iteration budget, switch to
    // Barnes-Hut if reports with several thousand matrices become common.
    let budget = match count {
        0 => 0,
        n => ((WORK / (n as f64 * n as f64)) as usize).clamp(MIN_ITERATIONS, MAX_ITERATIONS),
    };
    let threshold = (TOLERANCE * K) * (TOLERANCE * K);
    let mut disp = vec![[0.0f64; 3]; count];
    let (mut iterations, mut energy) = (0, 0.0);
    let (mut step, mut progress, mut previous_force_energy) = (0.1 * radius, 0, f64::INFINITY);
    while iterations < budget {
        disp.par_iter_mut().enumerate().for_each(|(i, d)| {
            *d = [0.0; 3];
            for (j, other) in pos.iter().enumerate() {
                if i != j {
                    let delta = sub(pos[i], *other);
                    let force = K * K / (dot(delta, delta) + 1e-6);
                    add_scaled(d, delta, force);
                }
            }
        });
        for &(a, b, strength) in links {
            if a < count && b < count && a != b {
                let delta = sub(pos[a], pos[b]);
                let force = strength * dot(delta, delta).sqrt() / K;
                add_scaled(&mut disp[a], delta, -force);
                add_scaled(&mut disp[b], delta, force);
            }
        }
        let mut force_energy = 0.0;
        energy = 0.0;
        for (p, d) in pos.iter_mut().zip(&disp) {
            let mut net = *d;
            add_scaled(&mut net, *p, -GRAVITY);
            let length = dot(net, net).sqrt();
            if length > 0.0 {
                let moved = length.min(step);
                add_scaled(p, net, moved / length);
                energy += moved * moved;
                force_energy += length * length;
            }
        }
        energy /= count as f64;
        iterations += 1;
        if force_energy < previous_force_energy {
            progress += 1;
            if progress >= PROGRESS {
                progress = 0;
                step /= DAMPING;
            }
        } else {
            progress = 0;
            step *= DAMPING;
        }
        previous_force_energy = force_energy;
        if energy < threshold {
            break;
        }
    }
    Constellation {
        positions: normalize(pos),
        iterations,
        energy,
        converged: count < 2 || energy < threshold,
    }
}

fn fibonacci(i: usize, count: usize, radius: f64) -> [f64; 3] {
    let y = match count > 1 {
        true => 1.0 - 2.0 * i as f64 / (count - 1) as f64,
        false => 0.0,
    };
    let ring = (1.0 - y * y).max(0.0).sqrt();
    // The golden angle.
    let (sin, cos) = sin_cos(i as f64 * 2.399963229728653);
    [radius * ring * cos, radius * y, radius * ring * sin]
}

/// Centres on the mean and scales the farthest point to distance 1.
fn normalize(points: Vec<[f64; 3]>) -> Vec<Position> {
    let count = points.len().max(1) as f64;
    let mut mean = [0.0; 3];
    for p in &points {
        add_scaled(&mut mean, *p, 1.0 / count);
    }
    let radius = points
        .iter()
        .map(|p| dot(sub(*p, mean), sub(*p, mean)).sqrt())
        .fold(0.0, f64::max);
    let scale = match radius > 0.0 {
        true => 1.0 / radius,
        false => 0.0,
    };
    points
        .iter()
        .map(|p| {
            let c = sub(*p, mean);
            [
                (c[0] * scale) as f32,
                (c[1] * scale) as f32,
                (c[2] * scale) as f32,
            ]
        })
        .collect()
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add_scaled(target: &mut [f64; 3], v: [f64; 3], k: f64) {
    target[0] += v[0] * k;
    target[1] += v[1] * k;
    target[2] += v[2] * k;
}

/// `sin` and `cos` from a Taylor series (error below 1e-15 after reduction to [-pi, pi]).
fn sin_cos(x: f64) -> (f64, f64) {
    use std::f64::consts::TAU;
    let x = x - (x / TAU).round() * TAU;
    let x2 = x * x;
    let (mut sin, mut cos) = (0.0, 0.0);
    let (mut term_sin, mut term_cos) = (x, 1.0);
    for k in 1..=15 {
        sin += term_sin;
        cos += term_cos;
        let k = k as f64;
        term_sin *= -x2 / ((2.0 * k) * (2.0 * k + 1.0));
        term_cos *= -x2 / ((2.0 * k - 1.0) * (2.0 * k));
    }
    (sin, cos)
}

/// Cube root by Newton's method from `max(x, 1)`, enough steps for any node count.
fn cbrt(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let mut y = x.max(1.0);
    for _ in 0..100 {
        y = (2.0 * y + x / (y * y)) / 3.0;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portable_trigonometry_matches_the_platform_one() {
        for i in -400..=400 {
            let x = i as f64 * 0.137;
            let (sin, cos) = sin_cos(x);
            assert!((sin - x.sin()).abs() < 1e-12, "sin {x}");
            assert!((cos - x.cos()).abs() < 1e-12, "cos {x}");
        }
        for x in [1.0, 2.0, 27.0, 30.0, 9000.0, 1e6] {
            assert!((cbrt(x) - x.cbrt()).abs() < 1e-9 * x.cbrt(), "cbrt {x}");
        }
    }

    #[test]
    fn degenerate_graphs_stay_finite() {
        assert!(constellation(0, &[]).positions.is_empty());
        assert_eq!(constellation(1, &[]).positions, vec![[0.0; 3]]);
        // Out of range and self links are ignored.
        let two = constellation(2, &[(0, 5, 1.0), (1, 1, 1.0), (0, 1, 1.0)]);
        assert!(two.positions.iter().flatten().all(|v| v.is_finite()));
        assert!(normalize(vec![[2.0; 3], [2.0; 3]]) == vec![[0.0; 3]; 2]);
    }
}
