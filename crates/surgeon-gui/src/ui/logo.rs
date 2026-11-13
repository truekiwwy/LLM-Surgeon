//! The logomark: the square 5×5 block grid of the validated Atelier mockup
//! (`docs/design/mockups`), drawn as vector cells in the accent color at three strengths.

use eframe::egui::emath::GuiRounding as _;
use eframe::egui::{Painter, Rect, Vec2};

use super::theme::{self, TOKENS};

/// Cell strengths, row by row: 3 full, 2 medium, 1 dim, 0 empty.
const CELLS: [[u8; 5]; 5] = [
    [3, 3, 0, 3, 3],
    [3, 2, 3, 0, 1],
    [0, 1, 3, 1, 2],
    [3, 2, 0, 3, 3],
    [3, 3, 0, 3, 1],
];

/// Paints the mark in `rect` (a square); cells are snapped to whole pixels.
pub fn paint(painter: &Painter, rect: Rect) {
    let ppp = painter.pixels_per_point();
    let cell = rect.width() / 5.0;
    for (row, cells) in CELLS.iter().enumerate() {
        for (col, &strength) in cells.iter().enumerate() {
            let t = match strength {
                0 => continue,
                1 => 0.3,
                2 => 0.6,
                _ => 1.0,
            };
            let min = rect.min + Vec2::new(col as f32 * cell, row as f32 * cell);
            let r = Rect::from_min_size(min, Vec2::splat(cell)).round_to_pixels(ppp);
            let color = theme::mix(TOKENS.color.accent, TOKENS.color.window, t);
            painter.rect_filled(r, 0.0, color);
        }
    }
}
