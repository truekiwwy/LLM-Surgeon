//! The shell: a title bar, then the activity bar and three zones side by side, separated by a
//! one-pixel line. Column widths are multiples of 8, dragged by the lines between the zones,
//! held between their stops and persisted in the settings.

use eframe::egui::{CursorIcon, Pos2, Rect, Sense, Ui, Vec2};

use super::components::{self, vline};
use super::theme::TOKENS as T;

/// The grid every layout dimension sits on.
pub const GRID: f32 = 8.0;

pub fn snap8(v: f32) -> f32 {
    (v / GRID).round() * GRID
}

/// Widths of the two side columns, in points; `collapsed` folds the left one away.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Columns {
    pub left: f32,
    pub right: f32,
    pub collapsed: bool,
}

impl Default for Columns {
    fn default() -> Self {
        Self {
            left: T.size.left,
            right: T.size.right,
            collapsed: false,
        }
    }
}

impl Columns {
    /// Stored widths, put back on the grid and inside their stops.
    pub fn from_saved(left: Option<f32>, right: Option<f32>) -> Self {
        let hold = |v: Option<f32>, default: f32, [min, max]: [f32; 2]| {
            snap8(v.unwrap_or(default)).clamp(min, max)
        };
        Self {
            left: hold(left, T.size.left, T.size.left_range),
            right: hold(right, T.size.right, T.size.right_range),
            collapsed: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Regions {
    pub titlebar: Rect,
    pub activity: Rect,
    pub left: Rect,
    pub center: Rect,
    pub right: Rect,
}

/// The zones of `screen`. A window too narrow for the stored columns takes from the right
/// one, then from the left one, down to their smallest widths: the center keeps its floor.
/// A folded left column has no width; the center takes it.
pub fn regions(screen: Rect, columns: Columns) -> Regions {
    let titlebar = Rect::from_min_size(screen.min, Vec2::new(screen.width(), T.size.titlebar));
    let body = Rect::from_min_max(Pos2::new(screen.left(), titlebar.bottom()), screen.max);
    let activity = Rect::from_min_size(body.min, Vec2::new(T.size.activity, body.height()));
    let room = body.width() - activity.width() - T.size.center_min;
    let mut right = columns.right;
    let mut left = match columns.collapsed {
        true => 0.0,
        false => columns.left,
    };
    if left + right > room {
        right = snap8((room - left).max(T.size.right_range[0]));
    }
    if left + right > room && !columns.collapsed {
        left = snap8((room - right).max(T.size.left_range[0]));
    }
    let left_rect = Rect::from_min_size(
        Pos2::new(activity.right(), body.top()),
        Vec2::new(left, body.height()),
    );
    let right_rect = Rect::from_min_max(Pos2::new(body.right() - right, body.top()), body.max);
    let center = Rect::from_min_max(
        Pos2::new(left_rect.right(), body.top()),
        Pos2::new(right_rect.left(), body.bottom()),
    );
    Regions {
        titlebar,
        activity,
        left: left_rect,
        center,
        right: right_rect,
    }
}

/// The lines between the zones of the body.
pub fn separators(ui: &Ui, regions: &Regions) {
    let painter = ui.painter();
    let y = regions.activity.y_range();
    let mut xs = vec![regions.activity.right(), regions.right.left()];
    if regions.left.width() > 0.0 {
        xs.push(regions.left.right());
    }
    for x in xs {
        vline(painter, x, y, T.color.separator);
    }
}

/// The lines on both sides of the center resize the columns, in steps of 8. `true` when a
/// width changed (the caller persists it).
pub fn splitters(ui: &mut Ui, regions: &Regions, columns: &mut Columns) -> bool {
    let before = *columns;
    let lines = [
        ("split-left", regions.left.right(), true),
        ("split-right", regions.right.left(), false),
    ];
    // A folded column has no width to drag.
    for (name, x, is_left) in lines
        .into_iter()
        .filter(|(_, _, left)| !(*left && columns.collapsed))
    {
        let rect = Rect::from_center_size(
            Pos2::new(x, regions.center.center().y),
            Vec2::new(T.space.s, regions.center.height()),
        );
        let response = ui
            .interact(rect, ui.id().with(name), Sense::drag())
            .on_hover_cursor(CursorIcon::ResizeHorizontal);
        if response.hovered() || response.dragged() {
            vline(ui.painter(), x, rect.y_range(), T.color.border);
        }
        if let Some(pointer) = response
            .interact_pointer_pos()
            .filter(|_| response.dragged())
        {
            ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
            if is_left {
                let width = snap8(pointer.x - regions.left.left());
                columns.left = width.clamp(T.size.left_range[0], T.size.left_range[1]);
            } else {
                let width = snap8(regions.right.right() - pointer.x);
                columns.right = width.clamp(T.size.right_range[0], T.size.right_range[1]);
            }
        }
    }
    *columns != before
}

/// One part of a column stacked top to bottom.
#[derive(Debug, Clone, Copy)]
pub enum Part {
    /// A height in points (a multiple of 8).
    Fixed(f32),
    /// A share of what the fixed parts leave.
    Flex(f32),
}

/// Splits `rect` top to bottom into `parts`, edge to edge (each panel draws the line on its
/// top).
pub fn stack(rect: Rect, parts: &[Part]) -> Vec<Rect> {
    let fixed: f32 = parts
        .iter()
        .map(|p| match p {
            Part::Fixed(h) => *h,
            Part::Flex(_) => 0.0,
        })
        .sum();
    let shares: f32 = parts
        .iter()
        .map(|p| match p {
            Part::Flex(s) => *s,
            Part::Fixed(_) => 0.0,
        })
        .sum();
    let free = (rect.height() - fixed).max(0.0);
    let mut heights: Vec<f32> = parts
        .iter()
        .map(|p| match p {
            Part::Fixed(h) => *h,
            Part::Flex(s) => (free * s / shares.max(1.0) / GRID).floor() * GRID,
        })
        .collect();
    // The last flexible part takes the remainder: the stack ends on the column's bottom edge.
    if let Some(i) = parts.iter().rposition(|p| matches!(p, Part::Flex(_))) {
        let others: f32 = heights.iter().sum::<f32>() - heights[i];
        heights[i] = (rect.height() - others).max(0.0);
    }
    let mut y = rect.top();
    heights
        .into_iter()
        .map(|h| {
            let r = Rect::from_min_size(Pos2::new(rect.left(), y), Vec2::new(rect.width(), h));
            y += h;
            r
        })
        .collect()
}

/// Records the zones in the layout dump.
pub fn record(ui: &Ui, regions: &Regions) {
    for (name, rect) in [
        ("titlebar", regions.titlebar),
        ("activity", regions.activity),
        ("left", regions.left),
        ("center", regions.center),
        ("right", regions.right),
    ] {
        components::record(ui.ctx(), name, rect);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen(w: f32, h: f32) -> Rect {
        Rect::from_min_size(Pos2::ZERO, Vec2::new(w, h))
    }

    #[test]
    fn zones_sit_on_the_grid_at_both_capture_sizes() {
        for (w, h) in [(1440.0, 900.0), (1920.0, 1080.0)] {
            let r = regions(screen(w, h), Columns::default());
            assert_eq!(r.titlebar.height(), T.size.titlebar);
            assert_eq!(r.left.top(), r.titlebar.bottom());
            assert_eq!(r.center.bottom(), h);
            for v in [
                r.activity.width(),
                r.left.left(),
                r.left.width(),
                r.center.left(),
                r.center.width(),
                r.right.width(),
                w - r.right.right(),
            ] {
                assert_eq!(v % GRID, 0.0, "{v} at {w}x{h}");
            }
            assert_eq!(r.center.left(), r.left.right(), "edge to edge");
            assert_eq!(r.right.left(), r.center.right(), "edge to edge");
            assert!(r.center.width() >= T.size.center_min);
        }
    }

    #[test]
    fn a_folded_left_column_gives_its_width_to_the_center() {
        let open = regions(screen(1440.0, 900.0), Columns::default());
        let folded = Columns {
            collapsed: true,
            ..Columns::default()
        };
        let r = regions(screen(1440.0, 900.0), folded);
        assert_eq!(r.left.width(), 0.0);
        assert_eq!(r.center.left(), r.activity.right());
        assert_eq!(r.right.width(), open.right.width());
        assert_eq!(r.center.width() - open.center.width(), open.left.width());
    }

    #[test]
    fn a_narrow_window_takes_from_the_columns_before_the_center() {
        let wide = Columns {
            left: 400.0,
            right: 480.0,
            collapsed: false,
        };
        let r = regions(screen(960.0, 600.0), wide);
        assert_eq!(r.right.width(), T.size.right_range[0]);
        assert_eq!(r.left.width(), T.size.left_range[0]);
        assert!(r.center.width() > 0.0);
    }

    #[test]
    fn saved_widths_come_back_on_the_grid_and_inside_their_stops() {
        let c = Columns::from_saved(Some(283.0), Some(9000.0));
        assert_eq!(c.left, 280.0);
        assert_eq!(c.right, T.size.right_range[1]);
        assert_eq!(Columns::from_saved(None, None), Columns::default());
    }

    #[test]
    fn a_stack_fills_its_column_edge_to_edge() {
        let rect = Rect::from_min_size(Pos2::new(48.0, 48.0), Vec2::new(320.0, 820.0));
        let parts = stack(
            rect,
            &[Part::Fixed(240.0), Part::Flex(1.0), Part::Fixed(160.0)],
        );
        assert_eq!(parts[0].height(), 240.0);
        assert_eq!(parts[1].top(), parts[0].bottom());
        assert_eq!(parts[2].top(), parts[1].bottom());
        assert_eq!(parts[2].bottom(), rect.bottom());
    }
}
