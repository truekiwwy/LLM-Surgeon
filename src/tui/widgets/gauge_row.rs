//! Labelled gauge: the bar is clamped, the exact value is always printed.

use ratatui::layout::Rect;
use ratatui::text::Span;
use ratatui::widgets::{Block, Gauge};
use ratatui::Frame;

use crate::tui::theme::Theme;

pub fn render_gauge(
    frame: &mut Frame,
    area: Rect,
    title: &str,
    ratio: f64,
    value: String,
    theme: &Theme,
) {
    let ratio = if ratio.is_finite() {
        ratio.clamp(0.0, 1.0)
    } else {
        1.0
    };
    let gauge = Gauge::default()
        .block(
            Block::bordered()
                .border_style(theme.accent())
                .title(Span::styled(format!(" {title} "), theme.dim())),
        )
        .gauge_style(theme.accent())
        .ratio(ratio)
        .label(Span::styled(value, theme.strong()));
    frame.render_widget(gauge, area);
}
