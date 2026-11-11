//! Theme: a modern dark palette on ANSI-256 colors (safe on every terminal,
//! vivid under TrueColor). One accent per role, one verdict color per badge.

use ratatui::style::{Color, Style};

/// Base text.
pub const TEXT: Color = Color::Gray;
/// Dimmed text (layers outside the filter, hints).
pub const DIM: Color = Color::DarkGray;
/// Panel borders.
pub const BORDER: Color = Color::Indexed(238);
/// Border of the focused panel.
pub const BORDER_FOCUS: Color = Color::Cyan;
/// Selection background.
pub const SELECTED_BG: Color = Color::Indexed(236);
/// Agent highlight (a layer flagged over IPC).
pub const HIGHLIGHT: Color = Color::Magenta;
/// Title / branding.
pub const ACCENT: Color = Color::Indexed(213);

pub const fn role_color(role: crate::model::roles::Role) -> Color {
    use crate::model::roles::Role;
    match role {
        Role::Embed => Color::Indexed(75),
        Role::AttnOut => Color::Indexed(111),
        Role::MlpGate => Color::Indexed(214),
        Role::MlpUp => Color::Indexed(179),
        Role::MlpOut => Color::Indexed(203),
        Role::Router => Color::Indexed(176),
        Role::Other => Color::DarkGray,
    }
}

/// Green for exact, amber for approximate, red for unsupported.
pub const fn exactness_color(exactness: crate::model::roles::Exactness) -> Color {
    use crate::model::roles::Exactness;
    match exactness {
        Exactness::Exact => Color::Green,
        Exactness::Approximate => Color::Yellow,
        Exactness::Unsupported => Color::Red,
    }
}

/// Severity colors for the event log.
pub fn log_color(line: &str) -> Color {
    if line.contains("échec") || line.contains("erreur") || line.contains("impossible") {
        Color::Red
    } else if line.contains("agent") {
        Color::Indexed(75)
    } else {
        TEXT
    }
}

pub fn base() -> Style {
    Style::new().fg(TEXT)
}

pub fn dim() -> Style {
    Style::new().fg(DIM)
}

pub fn focused_border() -> Style {
    Style::new().fg(BORDER_FOCUS)
}

pub fn border() -> Style {
    Style::new().fg(BORDER)
}
