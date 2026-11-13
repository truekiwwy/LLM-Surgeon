//! Orbit camera over the unit ball the layouts live in: perspective projection to screen
//! pixels and nearest-node picking. Pure math, no GPU: the viewport paints 2D shapes.

use eframe::egui::{Pos2, Rect};

type Vec3 = [f32; 3];

/// Vertical field of view.
const FOV_Y: f32 = 0.85;
/// Points closer than this to the eye are not drawn.
const NEAR: f32 = 0.05;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Orbit {
    /// Rotation around the vertical axis, radians.
    pub yaw: f32,
    /// Elevation, radians, kept inside (-pi/2, pi/2).
    pub pitch: f32,
    pub distance: f32,
    pub target: Vec3,
}

impl Default for Orbit {
    fn default() -> Self {
        Self {
            yaw: 0.6,
            pitch: 0.3,
            distance: 2.8,
            target: [0.0; 3],
        }
    }
}

/// A point on screen.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projected {
    pub pos: Pos2,
    /// Distance along the view axis: larger is farther.
    pub depth: f32,
    /// Pixels per world unit at that depth (sizes shrink with distance).
    pub scale: f32,
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(a: Vec3) -> Vec3 {
    let n = dot(a, a).sqrt().max(f32::MIN_POSITIVE);
    [a[0] / n, a[1] / n, a[2] / n]
}

impl Orbit {
    /// `(eye, right, up, forward)`.
    fn frame(&self) -> (Vec3, Vec3, Vec3, Vec3) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        let back = [cp * sy, sp, cp * cy];
        let eye = [
            self.target[0] + self.distance * back[0],
            self.target[1] + self.distance * back[1],
            self.target[2] + self.distance * back[2],
        ];
        let forward = [-back[0], -back[1], -back[2]];
        let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
        let up = cross(right, forward);
        (eye, right, up, forward)
    }

    fn focal(rect: Rect) -> f32 {
        0.5 * rect.height() / (0.5 * FOV_Y).tan()
    }

    /// Left drag: orbit around the target.
    pub fn rotate(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.5, 1.5);
    }

    /// Right drag: move the target in the view plane, the scene following the pointer.
    pub fn pan(&mut self, dx: f32, dy: f32, rect: Rect) {
        let (_, right, up, _) = self.frame();
        let per_pixel = self.distance / Self::focal(rect);
        for i in 0..3 {
            self.target[i] += (-right[i] * dx + up[i] * dy) * per_pixel;
        }
    }

    /// Wheel: exponential zoom, so every notch scales the same.
    pub fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (-scroll * 0.002).exp()).clamp(0.25, 25.0);
    }

    /// `None` behind the eye.
    pub fn project(&self, p: Vec3, rect: Rect) -> Option<Projected> {
        let (eye, right, up, forward) = self.frame();
        let d = sub(p, eye);
        let depth = dot(d, forward);
        if depth < NEAR {
            return None;
        }
        let scale = Self::focal(rect) / depth;
        let c = rect.center();
        Some(Projected {
            pos: Pos2::new(c.x + dot(d, right) * scale, c.y - dot(d, up) * scale),
            depth,
            scale,
        })
    }
}

/// The index of the point under `pointer`: among `(position, radius)` the pointer is inside
/// of (plus `slack` pixels), the closest to it.
pub fn pick(points: &[(usize, Pos2, f32)], pointer: Pos2, slack: f32) -> Option<usize> {
    points
        .iter()
        .map(|&(i, pos, radius)| (i, pos.distance(pointer), radius + slack))
        .filter(|&(_, d, reach)| d <= reach)
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect() -> Rect {
        Rect::from_min_max(Pos2::ZERO, Pos2::new(800.0, 600.0))
    }

    #[test]
    fn the_target_is_at_the_center_and_depth_grows_away_from_the_eye() {
        let camera = Orbit::default();
        let center = camera.project(camera.target, rect()).unwrap();
        assert!(center.pos.distance(rect().center()) < 1e-3);
        assert!((center.depth - camera.distance).abs() < 1e-4);
        // A point behind the target (seen from the eye) is farther and drawn smaller.
        let (eye, ..) = camera.frame();
        let behind = [-eye[0], -eye[1], -eye[2]];
        let far = camera.project(behind, rect()).unwrap();
        assert!(far.depth > center.depth && far.scale < center.scale);
        // The eye itself is not drawable.
        assert!(camera.project(eye, rect()).is_none());
    }

    #[test]
    fn panning_moves_the_scene_with_the_pointer() {
        let mut camera = Orbit::default();
        let origin = camera.project([0.0; 3], rect()).unwrap().pos;
        camera.pan(40.0, -25.0, rect());
        let moved = camera.project([0.0; 3], rect()).unwrap().pos;
        assert!((moved.x - origin.x - 40.0).abs() < 1e-2, "{moved:?}");
        assert!((moved.y - origin.y + 25.0).abs() < 1e-2, "{moved:?}");
    }

    #[test]
    fn zoom_and_pitch_stay_bounded() {
        let mut camera = Orbit::default();
        camera.zoom(1e6);
        assert_eq!(camera.distance, 0.25);
        camera.zoom(-1e6);
        assert_eq!(camera.distance, 25.0);
        camera.rotate(0.0, 1e6);
        assert_eq!(camera.pitch, 1.5);
    }

    #[test]
    fn picking_takes_the_closest_point_within_reach() {
        let points = [
            (0, Pos2::new(10.0, 10.0), 4.0),
            (1, Pos2::new(14.0, 10.0), 4.0),
            (2, Pos2::new(100.0, 100.0), 4.0),
        ];
        assert_eq!(pick(&points, Pos2::new(13.0, 10.0), 2.0), Some(1));
        assert_eq!(pick(&points, Pos2::new(50.0, 50.0), 2.0), None);
        assert_eq!(pick(&points, Pos2::new(105.0, 100.0), 2.0), Some(2));
        assert_eq!(pick(&[], Pos2::ZERO, 2.0), None);
    }
}
