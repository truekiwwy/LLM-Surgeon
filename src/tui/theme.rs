//! TUI styles from the palette of `ui.rs` (royal blue, bold white, gray). In `Plain` /
//! `NO_COLOR` mode no color is emitted at all, only modifiers. One explicit exception to
//! the palette: the NaN/Inf alert red `#FF5555`.

use ratatui::style::{Color, Modifier, Style};

use crate::ui::{rgb_to_ansi256, ColorMode, BLUE, GRAY, WHITE};

/// The only color outside the palette, reserved to the NaN/Inf alert.
pub const ALERT_RED: (u8, u8, u8) = (255, 85, 85);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    mode: ColorMode,
}

impl Theme {
    pub fn new(mode: ColorMode) -> Self {
        Self { mode }
    }

    pub fn mode(&self) -> ColorMode {
        self.mode
    }

    fn color(&self, rgb: (u8, u8, u8)) -> Option<Color> {
        match self.mode {
            ColorMode::TrueColor => Some(Color::Rgb(rgb.0, rgb.1, rgb.2)),
            ColorMode::Ansi256 => Some(Color::Indexed(rgb_to_ansi256(rgb))),
            ColorMode::Plain | ColorMode::Off => None,
        }
    }

    fn fg(&self, rgb: (u8, u8, u8)) -> Style {
        self.color(rgb)
            .map_or(Style::default(), |c| Style::default().fg(c))
    }

    /// Badges, borders, bars.
    pub fn accent(&self) -> Style {
        self.fg(BLUE)
    }

    /// Values and metrics.
    pub fn strong(&self) -> Style {
        self.fg(WHITE).add_modifier(Modifier::BOLD)
    }

    /// Labels, paths, hints.
    pub fn dim(&self) -> Style {
        self.fg(GRAY)
    }

    /// Selected item: reversed in every mode, so it stays visible without color.
    pub fn selected(&self) -> Style {
        self.accent()
            .add_modifier(Modifier::REVERSED | Modifier::BOLD)
    }

    /// Blinking alert: only the style alternates, the text is always drawn.
    pub fn alert(&self, on: bool) -> Style {
        match (self.color(ALERT_RED), on) {
            (Some(red), true) => Style::default().fg(red).add_modifier(Modifier::BOLD),
            (Some(_), false) => self.strong(),
            (None, true) => Style::default().add_modifier(Modifier::REVERSED | Modifier::BOLD),
            (None, false) => Style::default().add_modifier(Modifier::BOLD),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn palette_per_mode() {
        let true_color = Theme::new(ColorMode::TrueColor);
        assert_eq!(true_color.accent().fg, Some(Color::Rgb(65, 105, 225)));
        assert_eq!(true_color.alert(true).fg, Some(Color::Rgb(255, 85, 85)));
        assert_eq!(true_color.alert(false).fg, Some(Color::Rgb(255, 255, 255)));
        assert_eq!(
            Theme::new(ColorMode::Ansi256).accent().fg,
            Some(Color::Indexed(62))
        );
        assert_eq!(
            Theme::new(ColorMode::Ansi256).alert(true).fg,
            Some(Color::Indexed(203))
        );
        let plain = Theme::new(ColorMode::Plain);
        for style in [
            plain.accent(),
            plain.strong(),
            plain.dim(),
            plain.alert(true),
            plain.selected(),
        ] {
            assert_eq!(
                (style.fg, style.bg),
                (None, None),
                "no color without color support"
            );
        }
        assert!(plain.alert(true).add_modifier.contains(Modifier::REVERSED));
    }
}
