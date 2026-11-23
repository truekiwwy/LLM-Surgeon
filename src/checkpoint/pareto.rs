use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub gain: f64,
    pub drift: f64,
}

impl Point {
    pub fn is_finite(self) -> bool {
        self.gain.is_finite() && self.drift.is_finite()
    }

    pub fn dominates(self, other: Point) -> bool {
        self.gain >= other.gain
            && self.drift <= other.drift
            && (self.gain > other.gain || self.drift < other.drift)
    }
}

pub fn improves(front: &[Point], candidate: Point) -> bool {
    candidate.is_finite()
        && !front
            .iter()
            .any(|p| *p == candidate || p.dominates(candidate))
}

pub fn front(points: &[Point]) -> Vec<usize> {
    let mut kept: Vec<usize> = Vec::new();
    for (i, p) in points.iter().enumerate() {
        let shadowed = !p.is_finite()
            || points
                .iter()
                .enumerate()
                .any(|(j, q)| q.is_finite() && (q.dominates(*p) || (*q == *p && j < i)));
        if !shadowed {
            kept.push(i);
        }
    }
    kept.sort_by(|&a, &b| {
        points[a]
            .drift
            .total_cmp(&points[b].drift)
            .then(points[b].gain.total_cmp(&points[a].gain))
    });
    kept
}

pub fn best(points: &[Point], front: &[usize]) -> Option<usize> {
    front.iter().copied().max_by(|&a, &b| {
        points[a]
            .gain
            .total_cmp(&points[b].gain)
            .then(points[b].drift.total_cmp(&points[a].drift))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(gain: f64, drift: f64) -> Point {
        Point { gain, drift }
    }

    #[test]
    fn the_front_keeps_exactly_the_non_dominated_points_sorted_by_drift() {
        let points = [
            p(0.30, 0.20),
            p(0.10, 0.05),
            p(0.25, 0.30),
            p(0.30, 0.10),
            p(0.05, 0.05),
            p(0.40, 0.50),
            p(0.30, 0.10),
            p(f64::NAN, 0.0),
            p(0.90, f64::INFINITY),
        ];
        assert_eq!(front(&points), vec![1, 3, 5]);
        assert_eq!(best(&points, &front(&points)), Some(5));
    }

    #[test]
    fn a_tie_on_gain_goes_to_the_lower_drift() {
        let points = [p(0.2, 0.3), p(0.2, 0.1), p(0.1, 0.0)];
        let f = front(&points);
        assert_eq!(f, vec![2, 1]);
        assert_eq!(best(&points, &f), Some(1));
        assert_eq!(best(&points, &[]), None);
    }

    #[test]
    fn only_a_point_outside_the_shadow_of_the_front_improves_it() {
        let front = [p(0.0, 0.0), p(0.3, 0.2)];
        assert!(improves(&front, p(0.1, 0.1)));
        assert!(improves(&front, p(0.4, 0.5)));
        assert!(!improves(&front, p(0.3, 0.2)));
        assert!(!improves(&front, p(0.2, 0.3)));
        assert!(!improves(&front, p(0.0, 0.1)));
        assert!(!improves(&front, p(-0.1, 0.0)));
        assert!(!improves(&front, p(f64::NAN, 0.0)));
    }
}
