//! Design tokens (`docs/design/DIRECTION.md`): the only place where a color, a spacing step,
//! a size or a radius is written down. Views go through [`TOKENS`] and `ui::components`;
//! `tests/ui_lint.rs` fails on a `Color32::` anywhere else.
//!
//! Two blues, two jobs: `accent` marks the one focus of attention of a region (active tab,
//! selection, active mode, caret, focus ring), `action` fills the one primary action of a
//! screen. Neither is ever small running text (4.0:1 and 2.3:1 on a panel).

use eframe::egui::{
    self, Color32, CornerRadius, FontFamily, FontId, Margin, Shadow, Stroke, TextStyle, Theme,
};
use weight_surgeon::graph::GraphRole;

pub struct Colors {
    /// Window background, visible between the panels.
    pub window: Color32,
    /// 3D canvas, histogram, text fields.
    pub canvas: Color32,
    /// Panels.
    pub surface: Color32,
    /// A hovered row or secondary button.
    pub surface_hover: Color32,
    /// The selected row.
    pub selection: Color32,
    /// Structural borders, one physical pixel.
    pub border: Color32,
    /// Separators inside a panel.
    pub separator: Color32,
    /// The focus of attention of a region.
    pub accent: Color32,
    /// The primary action of a screen; its text is `on_action`.
    pub action: Color32,
    pub action_hover: Color32,
    pub on_action: Color32,
    pub text: Color32,
    pub text_secondary: Color32,
    pub text_disabled: Color32,
    pub ok: Color32,
    pub warn: Color32,
    pub error: Color32,
    /// Data scales: seven discrete steps of one blue, picked by quantile.
    pub data: [Color32; 7],
    /// The scrim behind a modal and the drop hint.
    pub scrim: Color32,
}

/// Inner paddings. Layout dimensions are multiples of 8 and live in [`Sizes`].
pub struct Space {
    pub xs: f32,
    pub s: f32,
    pub m: f32,
    pub l: f32,
    pub xl: f32,
}

/// Layout dimensions, all multiples of 8.
pub struct Sizes {
    pub titlebar: f32,
    pub statusbar: f32,
    /// Gap between zones and around the body.
    pub gap: f32,
    /// A list or table row.
    pub row: f32,
    /// A panel header.
    pub header: f32,
    /// Buttons, fields, segments.
    pub control: f32,
    /// The primary action.
    pub cta: f32,
    pub window_button: [f32; 2],
    pub left: f32,
    pub left_range: [f32; 2],
    pub right: f32,
    pub right_range: [f32; 2],
    /// Smallest width the center keeps when a column is dragged.
    pub center_min: f32,
    pub icon: f32,
    pub dot: f32,
}

/// The type scale: five sizes.
pub struct TypeScale {
    pub meta: f32,
    pub mono: f32,
    pub body: f32,
    pub title: f32,
    pub brand: f32,
}

pub struct ThemeTokens {
    pub color: Colors,
    pub space: Space,
    pub size: Sizes,
    pub text: TypeScale,
    pub radius: u8,
    /// Longest feedback transition, seconds.
    pub motion: f32,
    /// Icon and glyph stroke, points.
    pub stroke: f32,
}

pub const TOKENS: ThemeTokens = ThemeTokens {
    color: Colors {
        window: Color32::from_rgb(0x0E, 0x0F, 0x14),
        canvas: Color32::from_rgb(0x0D, 0x0E, 0x12),
        surface: Color32::from_rgb(0x16, 0x1B, 0x22),
        surface_hover: Color32::from_rgb(0x1C, 0x22, 0x2B),
        selection: Color32::from_rgb(0x1B, 0x29, 0x42),
        border: Color32::from_rgb(0x2A, 0x34, 0x44),
        separator: Color32::from_rgb(0x1B, 0x20, 0x28),
        accent: Color32::from_rgb(0x27, 0x73, 0xF5),
        action: Color32::from_rgb(0x09, 0x46, 0xC8),
        action_hover: Color32::from_rgb(0x0B, 0x52, 0xE6),
        on_action: Color32::from_rgb(0xFF, 0xFF, 0xFF),
        text: Color32::from_rgb(0xE6, 0xED, 0xF3),
        text_secondary: Color32::from_rgb(0x9A, 0xA7, 0xB8),
        text_disabled: Color32::from_rgb(0x6B, 0x77, 0x88),
        ok: Color32::from_rgb(0x3F, 0xB9, 0x50),
        warn: Color32::from_rgb(0xD2, 0x99, 0x22),
        error: Color32::from_rgb(0xFA, 0x5C, 0x54),
        data: [
            Color32::from_rgb(0x2B, 0x36, 0x48),
            Color32::from_rgb(0x2D, 0x42, 0x66),
            Color32::from_rgb(0x2F, 0x4F, 0x86),
            Color32::from_rgb(0x31, 0x5D, 0xA8),
            Color32::from_rgb(0x3D, 0x74, 0xD6),
            Color32::from_rgb(0x7F, 0xA6, 0xEE),
            Color32::from_rgb(0xC2, 0xD6, 0xF8),
        ],
        scrim: Color32::from_rgba_premultiplied(0x07, 0x08, 0x0A, 0xB8),
    },
    space: Space {
        xs: 4.0,
        s: 8.0,
        m: 12.0,
        l: 16.0,
        xl: 24.0,
    },
    size: Sizes {
        titlebar: 40.0,
        statusbar: 24.0,
        gap: 8.0,
        row: 24.0,
        header: 32.0,
        control: 24.0,
        cta: 32.0,
        window_button: [48.0, 40.0],
        left: 280.0,
        left_range: [240.0, 400.0],
        right: 320.0,
        right_range: [280.0, 480.0],
        center_min: 480.0,
        icon: 16.0,
        dot: 8.0,
    },
    text: TypeScale {
        meta: 11.0,
        mono: 12.0,
        body: 13.0,
        title: 14.0,
        brand: 16.0,
    },
    radius: 2,
    motion: 0.12,
    stroke: 1.5,
};

pub const TRANSPARENT: Color32 = Color32::TRANSPARENT;

// Names the older modules use: the same tokens.
pub const BG: Color32 = TOKENS.color.canvas;
pub const PANEL: Color32 = TOKENS.color.surface;
pub const BORDER: Color32 = TOKENS.color.border;
pub const TEXT: Color32 = TOKENS.color.text;
pub const DIM: Color32 = TOKENS.color.text_secondary;
pub const SELECTION: Color32 = TOKENS.color.accent;
/// Nodes touched by the last stream patch.
pub const AGENT: Color32 = TOKENS.color.data[6];
pub const GOOD: Color32 = TOKENS.color.ok;
pub const WARN: Color32 = TOKENS.color.warn;
pub const BAD: Color32 = TOKENS.color.error;

/// The medium weight of the sans-serif: panel titles and the brand.
pub fn medium() -> FontFamily {
    FontFamily::Name("medium".into())
}

/// The three role groups of a graph node. A role reads by the shape of its node, color is
/// kept for data and selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoleShape {
    Circle,
    Square,
    Diamond,
}

pub const fn role_shape(role: GraphRole) -> RoleShape {
    match role {
        GraphRole::AttnOut | GraphRole::Q | GraphRole::K | GraphRole::V => RoleShape::Circle,
        GraphRole::Gate | GraphRole::Up | GraphRole::MlpOut => RoleShape::Square,
        GraphRole::Embed | GraphRole::LmHead => RoleShape::Diamond,
    }
}

/// `a` over `b` at `t` in 0..=1, opaque: the discrete strengths of a state color.
pub fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let ch = |x: u8, y: u8| (f32::from(x) * t + f32::from(y) * (1.0 - t)).round() as u8;
    Color32::from_rgb(ch(a.r(), b.r()), ch(a.g(), b.g()), ch(a.b(), b.b()))
}

/// The step of the data scale for `position` in 0..=1 (a quantile).
pub fn data_step(position: f64) -> Color32 {
    let steps = TOKENS.color.data.len();
    let i = (position.clamp(0.0, 1.0) * steps as f64) as usize;
    TOKENS.color.data[i.min(steps - 1)]
}

/// The rank of `value` among `sorted` values as a position in 0..=1.
pub fn quantile(value: f64, sorted: &[f64]) -> f64 {
    match sorted.len() {
        0 | 1 => 1.0,
        n => sorted.partition_point(|v| *v < value) as f64 / (n - 1) as f64,
    }
}

/// Monospace text: every number, index, shape, hash and delta is tabular.
pub fn mono(text: impl Into<String>) -> egui::RichText {
    egui::RichText::new(text).monospace()
}

/// Relative luminance (WCAG 2.x).
pub fn luminance(c: Color32) -> f64 {
    let lin = |v: u8| {
        let s = f64::from(v) / 255.0;
        match s <= 0.04045 {
            true => s / 12.92,
            false => ((s + 0.055) / 1.055).powf(2.4),
        }
    };
    0.2126 * lin(c.r()) + 0.7152 * lin(c.g()) + 0.0722 * lin(c.b())
}

/// Contrast ratio between two opaque colors (WCAG 2.x), 1..=21.
pub fn contrast(a: Color32, b: Color32) -> f64 {
    let (la, lb) = (luminance(a), luminance(b));
    (la.max(lb) + 0.05) / (la.min(lb) + 0.05)
}

/// A color pair the interface uses, with the ratio it must reach.
pub struct Pair {
    pub name: String,
    pub fg: Color32,
    pub bg: Color32,
    pub min: f64,
}

/// Every foreground / background pair in use. `theme_contrast` and `--self-test` check them.
pub fn pairs() -> Vec<Pair> {
    let c = &TOKENS.color;
    let mut out = Vec::new();
    for (bg_name, bg) in [
        ("surface", c.surface),
        ("window", c.window),
        ("canvas", c.canvas),
        ("surface_hover", c.surface_hover),
        ("selection", c.selection),
    ] {
        out.push((format!("text on {bg_name}"), c.text, bg, 7.0));
        out.push((
            format!("text_secondary on {bg_name}"),
            c.text_secondary,
            bg,
            4.5,
        ));
        out.push((
            format!("text_disabled on {bg_name}"),
            c.text_disabled,
            bg,
            3.0,
        ));
        out.push((format!("ok on {bg_name}"), c.ok, bg, 4.5));
        out.push((format!("warn on {bg_name}"), c.warn, bg, 4.5));
        out.push((format!("error on {bg_name}"), c.error, bg, 4.5));
        // Focus ring, underline, selection marker: an indicator, not text.
        out.push((format!("accent on {bg_name}"), c.accent, bg, 3.0));
    }
    out.push(("on_action on action".into(), c.on_action, c.action, 7.0));
    out.push((
        "on_action on action_hover".into(),
        c.on_action,
        c.action_hover,
        4.5,
    ));
    // The top of the data scale carries labels on the canvas.
    out.push(("data[6] on canvas".into(), c.data[6], c.canvas, 7.0));
    out.push(("data[5] on canvas".into(), c.data[5], c.canvas, 4.5));
    out.into_iter()
        .map(|(name, fg, bg, min)| Pair { name, fg, bg, min })
        .collect()
}

/// The pairs under their threshold, as `name: ratio < min`.
pub fn contrast_failures() -> Vec<String> {
    pairs()
        .iter()
        .filter_map(|p| {
            let ratio = contrast(p.fg, p.bg);
            (ratio < p.min).then(|| format!("{}: {ratio:.2} < {}", p.name, p.min))
        })
        .collect()
}

/// The style every frame runs on, built from the tokens alone.
pub fn style() -> egui::Style {
    let t = &TOKENS;
    let c = &t.color;
    let mut style = egui::Style {
        text_styles: [
            (
                TextStyle::Small,
                FontId::new(t.text.meta, FontFamily::Proportional),
            ),
            (
                TextStyle::Body,
                FontId::new(t.text.body, FontFamily::Proportional),
            ),
            (
                TextStyle::Button,
                FontId::new(t.text.body, FontFamily::Proportional),
            ),
            (TextStyle::Heading, FontId::new(t.text.brand, medium())),
            (
                TextStyle::Monospace,
                FontId::new(t.text.mono, FontFamily::Monospace),
            ),
        ]
        .into(),
        ..Default::default()
    };
    style.animation_time = t.motion;

    let s = &mut style.spacing;
    s.item_spacing = egui::vec2(t.space.s, t.space.xs);
    s.button_padding = egui::vec2(t.space.s, t.space.xs);
    s.window_margin = Margin::same(t.space.l as i8);
    s.menu_margin = Margin::same(t.space.xs as i8);
    s.indent = t.space.l;
    s.interact_size = egui::vec2(t.size.control, t.size.control);
    s.icon_width = t.size.icon;
    s.icon_width_inner = t.size.dot;
    s.icon_spacing = t.space.s;
    s.combo_height = t.size.row * 10.0;
    s.scroll = egui::style::ScrollStyle::solid();
    s.scroll.bar_width = t.space.s;
    s.scroll.handle_min_length = t.size.row;
    s.scroll.bar_inner_margin = t.space.xs;
    s.scroll.bar_outer_margin = 0.0;
    s.scroll.foreground_color = false;

    let v = &mut style.visuals;
    v.dark_mode = true;
    v.override_text_color = None;
    v.weak_text_color = Some(c.text_secondary);
    v.hyperlink_color = c.text;
    v.faint_bg_color = c.surface_hover;
    v.extreme_bg_color = c.canvas;
    v.text_edit_bg_color = Some(c.canvas);
    v.code_bg_color = c.canvas;
    v.warn_fg_color = c.warn;
    v.error_fg_color = c.error;
    v.window_corner_radius = CornerRadius::same(t.radius);
    v.menu_corner_radius = CornerRadius::same(t.radius);
    v.window_shadow = Shadow::NONE;
    v.popup_shadow = Shadow::NONE;
    v.window_fill = c.surface;
    v.window_stroke = Stroke::new(1.0, c.border);
    v.window_highlight_topmost = false;
    v.panel_fill = c.window;
    v.selection.bg_fill = c.selection;
    // egui paints the text of a selected widget with this stroke: text, not the accent.
    v.selection.stroke = Stroke::new(1.0, c.text);
    v.text_cursor.stroke = Stroke::new(t.stroke, c.accent);
    v.text_cursor.blink = false;
    v.striped = false;
    v.slider_trailing_fill = true;
    v.collapsing_header_frame = false;
    v.indent_has_left_vline = false;
    v.image_loading_spinners = false;
    v.disabled_alpha = 1.0;

    let w = &mut v.widgets;
    for state in [
        &mut w.noninteractive,
        &mut w.inactive,
        &mut w.hovered,
        &mut w.active,
        &mut w.open,
    ] {
        state.corner_radius = CornerRadius::same(t.radius);
        state.expansion = 0.0;
        state.bg_stroke = Stroke::new(1.0, c.border);
        state.fg_stroke = Stroke::new(1.0, c.text);
        state.bg_fill = c.surface;
        state.weak_bg_fill = c.surface;
    }
    w.noninteractive.bg_stroke = Stroke::new(1.0, c.separator);
    w.noninteractive.fg_stroke = Stroke::new(1.0, c.text);
    w.inactive.fg_stroke = Stroke::new(1.0, c.text);
    w.hovered.bg_fill = c.surface_hover;
    w.hovered.weak_bg_fill = c.surface_hover;
    w.active.bg_fill = c.selection;
    w.active.weak_bg_fill = c.selection;
    w.active.bg_stroke = Stroke::new(1.0, c.accent);
    w.open.bg_fill = c.surface_hover;
    w.open.weak_bg_fill = c.surface_hover;
    style
}

/// Installs the theme on the dark style (the only one the app uses).
pub fn apply(ctx: &egui::Context) {
    ctx.set_theme(Theme::Dark);
    ctx.set_style_of(Theme::Dark, style());
}

/// Every color a [`egui::Visuals`] built by [`style`] may hold: the tokens and nothing else.
pub fn token_colors() -> Vec<Color32> {
    let c = &TOKENS.color;
    let mut all = vec![
        c.window,
        c.canvas,
        c.surface,
        c.surface_hover,
        c.selection,
        c.border,
        c.separator,
        c.accent,
        c.action,
        c.action_hover,
        c.on_action,
        c.text,
        c.text_secondary,
        c.text_disabled,
        c.ok,
        c.warn,
        c.error,
        c.scrim,
    ];
    all.extend(c.data);
    all
}
