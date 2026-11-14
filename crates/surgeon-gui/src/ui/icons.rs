//! Icons drawn with the `Painter`: one set, a 16-point box, one stroke width, segments
//! snapped to pixel centers so that horizontal and vertical strokes stay crisp. No glyph of
//! a font is ever used as an icon, and a status dot is a real circle.

use eframe::egui::emath::GuiRounding as _;
use eframe::egui::{Color32, Painter, Pos2, Rect, Shape, Stroke, Vec2};

use super::theme::{RoleShape, TOKENS};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Icon {
    Minimize,
    Maximize,
    Restore,
    Close,
    ChevronRight,
    ChevronLeft,
    ChevronDown,
    Search,
    Play,
    Pause,
    SkipStart,
    SkipEnd,
    StepBack,
    StepForward,
    /// A crosshair: put the camera back.
    Recenter,
}

/// Points of the icon in a unit box, as polylines (`true`: closed).
fn paths(icon: Icon) -> Vec<(Vec<[f32; 2]>, bool)> {
    let open = |p: &[[f32; 2]]| (p.to_vec(), false);
    let closed = |p: &[[f32; 2]]| (p.to_vec(), true);
    match icon {
        Icon::Minimize => vec![open(&[[0.2, 0.5], [0.8, 0.5]])],
        Icon::Maximize => vec![closed(&[[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])],
        Icon::Restore => vec![
            closed(&[[0.2, 0.35], [0.65, 0.35], [0.65, 0.8], [0.2, 0.8]]),
            open(&[
                [0.35, 0.35],
                [0.35, 0.2],
                [0.8, 0.2],
                [0.8, 0.65],
                [0.65, 0.65],
            ]),
        ],
        Icon::Close => vec![
            open(&[[0.2, 0.2], [0.8, 0.8]]),
            open(&[[0.8, 0.2], [0.2, 0.8]]),
        ],
        Icon::ChevronRight => vec![open(&[[0.38, 0.25], [0.65, 0.5], [0.38, 0.75]])],
        Icon::ChevronLeft => vec![open(&[[0.62, 0.25], [0.35, 0.5], [0.62, 0.75]])],
        Icon::ChevronDown => vec![open(&[[0.25, 0.38], [0.5, 0.65], [0.75, 0.38]])],
        Icon::Search => vec![
            closed(&[
                [0.45, 0.15],
                [0.66, 0.24],
                [0.75, 0.45],
                [0.66, 0.66],
                [0.45, 0.75],
                [0.24, 0.66],
                [0.15, 0.45],
                [0.24, 0.24],
            ]),
            open(&[[0.67, 0.67], [0.87, 0.87]]),
        ],
        Icon::Play => vec![closed(&[[0.3, 0.2], [0.8, 0.5], [0.3, 0.8]])],
        Icon::Pause => vec![
            open(&[[0.35, 0.2], [0.35, 0.8]]),
            open(&[[0.65, 0.2], [0.65, 0.8]]),
        ],
        Icon::SkipStart => vec![
            open(&[[0.25, 0.2], [0.25, 0.8]]),
            closed(&[[0.8, 0.2], [0.4, 0.5], [0.8, 0.8]]),
        ],
        Icon::SkipEnd => vec![
            open(&[[0.75, 0.2], [0.75, 0.8]]),
            closed(&[[0.2, 0.2], [0.6, 0.5], [0.2, 0.8]]),
        ],
        Icon::StepBack => vec![open(&[[0.62, 0.25], [0.35, 0.5], [0.62, 0.75]])],
        Icon::StepForward => vec![open(&[[0.38, 0.25], [0.65, 0.5], [0.38, 0.75]])],
        Icon::Recenter => vec![
            open(&[[0.5, 0.1], [0.5, 0.35]]),
            open(&[[0.5, 0.65], [0.5, 0.9]]),
            open(&[[0.1, 0.5], [0.35, 0.5]]),
            open(&[[0.65, 0.5], [0.9, 0.5]]),
            closed(&[[0.42, 0.42], [0.58, 0.42], [0.58, 0.58], [0.42, 0.58]]),
        ],
    }
}

/// Paints `icon` centered in `rect`, in a box of the icon size.
pub fn paint(painter: &Painter, rect: Rect, icon: Icon, color: Color32) {
    let ppp = painter.pixels_per_point();
    let side = TOKENS.size.icon;
    let origin = (rect.center() - Vec2::splat(side / 2.0)).round_to_pixels(ppp);
    let stroke = Stroke::new(TOKENS.stroke, color);
    for (points, closed) in paths(icon) {
        let points: Vec<Pos2> = points
            .iter()
            .map(|p| (origin + Vec2::new(p[0], p[1]) * side).round_to_pixel_center(ppp))
            .collect();
        painter.add(match closed {
            true => Shape::closed_line(points, stroke),
            false => Shape::line(points, stroke),
        });
    }
}

/// The states a status dot can show. Always next to a word, never alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Ok,
    Warn,
    Error,
    /// Something is running.
    Active,
    /// Nothing to report: waiting, off, unknown.
    Idle,
}

pub fn state_color(state: State) -> Color32 {
    let c = &TOKENS.color;
    match state {
        State::Ok => c.ok,
        State::Warn => c.warn,
        State::Error => c.error,
        State::Active => c.text,
        State::Idle => c.text_disabled,
    }
}

/// A crisp circle: filled for a settled state, a ring for idle.
pub fn dot(painter: &Painter, center: Pos2, state: State) {
    let ppp = painter.pixels_per_point();
    let center = center.round_to_pixel_center(ppp);
    let radius = TOKENS.size.dot / 2.0;
    let color = state_color(state);
    match state {
        State::Idle => {
            painter.circle_stroke(center, radius - 0.75, Stroke::new(TOKENS.stroke, color));
        }
        _ => {
            painter.circle_filled(center, radius, color);
        }
    }
}

/// The outline of a node of `shape`, centered on `c` with half-size `r`.
pub fn role_outline(shape: RoleShape, c: Pos2, r: f32, stroke: Stroke) -> Shape {
    match shape {
        RoleShape::Circle => Shape::circle_stroke(c, r, stroke),
        RoleShape::Square => Shape::closed_line(
            vec![
                c + Vec2::new(-r, -r),
                c + Vec2::new(r, -r),
                c + Vec2::new(r, r),
                c + Vec2::new(-r, r),
            ],
            stroke,
        ),
        RoleShape::Diamond => Shape::closed_line(
            vec![
                c + Vec2::new(0.0, -1.3 * r),
                c + Vec2::new(1.3 * r, 0.0),
                c + Vec2::new(0.0, 1.3 * r),
                c + Vec2::new(-1.3 * r, 0.0),
            ],
            stroke,
        ),
    }
}

/// The filled body of a node of `shape`.
pub fn role_fill(shape: RoleShape, c: Pos2, r: f32, fill: Color32) -> Shape {
    match shape {
        RoleShape::Circle => Shape::circle_filled(c, r, fill),
        RoleShape::Square => {
            Shape::rect_filled(Rect::from_center_size(c, Vec2::splat(2.0 * r)), 0.0, fill)
        }
        RoleShape::Diamond => Shape::convex_polygon(
            vec![
                c + Vec2::new(0.0, -1.3 * r),
                c + Vec2::new(1.3 * r, 0.0),
                c + Vec2::new(0.0, 1.3 * r),
                c + Vec2::new(-1.3 * r, 0.0),
            ],
            fill,
            Stroke::NONE,
        ),
    }
}
