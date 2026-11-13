//! Theme: the colors of `surgeon-tui` (role accents from its ANSI-256 palette, cyan focus,
//! magenta agent highlight) on a flat, matte dark base. Semantic colors are reserved: cyan
//! for the active selection, emerald for an improvement, amber for a warning, carmine for a
//! regression. No gradient, no shadow, tight corners.

use eframe::egui::{
    self, Color32, CornerRadius, FontFamily, FontId, Shadow, Stroke, TextStyle, Theme,
};
use weight_surgeon::graph::GraphRole;

/// Viewport background.
pub const BG: Color32 = Color32::from_rgb(0x0D, 0x0F, 0x12);
/// Docked panels.
pub const PANEL: Color32 = Color32::from_rgb(0x13, 0x16, 0x1B);
/// Panel borders and separators.
pub const BORDER: Color32 = Color32::from_rgb(0x23, 0x28, 0x30);
/// Widgets at rest.
const WIDGET: Color32 = Color32::from_rgb(0x1A, 0x1E, 0x24);
const WIDGET_HOVER: Color32 = Color32::from_rgb(0x21, 0x26, 0x2E);
/// Base text (TUI `Gray`).
pub const TEXT: Color32 = Color32::from_rgb(0xC5, 0xC9, 0xD0);
/// Hints, filtered-out items (TUI `DarkGray`).
pub const DIM: Color32 = Color32::from_rgb(0x6A, 0x71, 0x7C);
/// Active selection only (TUI `BORDER_FOCUS`).
pub const SELECTION: Color32 = Color32::from_rgb(0x3F, 0xB8, 0xD4);
const SELECTION_BG: Color32 = Color32::from_rgb(0x17, 0x2A, 0x33);
/// Nodes touched by the last stream patch (TUI `HIGHLIGHT`: a layer flagged by an agent).
pub const AGENT: Color32 = Color32::from_rgb(0xD7, 0x5F, 0xD7);
/// GO, spectral improvement.
pub const GOOD: Color32 = Color32::from_rgb(0x2B, 0xB6, 0x73);
/// Warnings.
pub const WARN: Color32 = Color32::from_rgb(0xE0, 0xA0, 0x30);
/// NO-GO, regression, errors.
pub const BAD: Color32 = Color32::from_rgb(0xD2, 0x3C, 0x4B);

/// The role accents of `surgeon-tui` (`ui/theme.rs`, xterm-256 indices in comments). The
/// TUI has no q/k/v roles: attention readers take the attention color.
pub const fn role_color(role: GraphRole) -> Color32 {
    match role {
        GraphRole::Embed | GraphRole::LmHead => Color32::from_rgb(95, 175, 255), // 75
        GraphRole::AttnOut | GraphRole::Q | GraphRole::K | GraphRole::V => {
            Color32::from_rgb(135, 175, 255) // 111
        }
        GraphRole::Gate => Color32::from_rgb(255, 175, 0), // 214
        GraphRole::Up => Color32::from_rgb(215, 175, 95),  // 179
        GraphRole::MlpOut => Color32::from_rgb(255, 95, 95), // 203
    }
}

/// Monospace text: every number, index, shape, hash and delta is tabular.
pub fn mono(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text).monospace()
}

/// Installs the theme on the dark style (the only one the app uses).
pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(Theme::Dark);
    ctx.style_mut_of(Theme::Dark, |style| {
        style.text_styles = [
            (
                TextStyle::Small,
                FontId::new(10.5, FontFamily::Proportional),
            ),
            (TextStyle::Body, FontId::new(12.5, FontFamily::Proportional)),
            (
                TextStyle::Button,
                FontId::new(12.5, FontFamily::Proportional),
            ),
            (
                TextStyle::Heading,
                FontId::new(14.0, FontFamily::Proportional),
            ),
            (
                TextStyle::Monospace,
                FontId::new(12.0, FontFamily::Monospace),
            ),
        ]
        .into();
        style.spacing.item_spacing = egui::vec2(6.0, 4.0);
        style.spacing.button_padding = egui::vec2(6.0, 2.0);
        style.spacing.interact_size.y = 20.0;

        let v = &mut style.visuals;
        v.dark_mode = true;
        v.panel_fill = PANEL;
        v.window_fill = PANEL;
        v.extreme_bg_color = BG;
        v.faint_bg_color = Color32::from_rgb(0x16, 0x1A, 0x20);
        v.code_bg_color = BG;
        v.window_stroke = Stroke::new(1.0, BORDER);
        v.window_corner_radius = CornerRadius::same(3);
        v.menu_corner_radius = CornerRadius::same(2);
        v.window_shadow = Shadow::NONE;
        v.popup_shadow = Shadow::NONE;
        v.selection.bg_fill = SELECTION_BG;
        v.selection.stroke = Stroke::new(1.0, SELECTION);
        v.hyperlink_color = SELECTION;
        v.warn_fg_color = WARN;
        v.error_fg_color = BAD;

        let w = &mut v.widgets;
        for state in [
            &mut w.noninteractive,
            &mut w.inactive,
            &mut w.hovered,
            &mut w.active,
            &mut w.open,
        ] {
            state.corner_radius = CornerRadius::same(2);
            state.expansion = 0.0;
            state.bg_stroke = Stroke::new(1.0, BORDER);
            state.fg_stroke = Stroke::new(1.0, TEXT);
        }
        w.noninteractive.bg_fill = PANEL;
        w.noninteractive.weak_bg_fill = PANEL;
        w.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        w.inactive.bg_fill = WIDGET;
        w.inactive.weak_bg_fill = WIDGET;
        w.hovered.bg_fill = WIDGET_HOVER;
        w.hovered.weak_bg_fill = WIDGET_HOVER;
        w.hovered.bg_stroke = Stroke::new(1.0, Color32::from_rgb(0x3A, 0x41, 0x4C));
        w.active.bg_fill = SELECTION_BG;
        w.active.weak_bg_fill = SELECTION_BG;
        w.active.bg_stroke = Stroke::new(1.0, SELECTION);
        w.open.bg_fill = WIDGET_HOVER;
        w.open.weak_bg_fill = WIDGET_HOVER;
    });
}
