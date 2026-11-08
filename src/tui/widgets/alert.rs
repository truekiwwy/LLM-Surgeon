//! NaN/Inf banner: the text is always drawn, only its style blinks.

use ratatui::layout::Rect;
use ratatui::text::Line;
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::engine::stats::TensorStats;
use crate::tui::theme::Theme;

pub fn alert_text(stats: &TensorStats) -> String {
    format!(
        "⚠ NaN/Inf DÉTECTÉ : {} NaN · {} +Inf · {} -Inf",
        stats.nan, stats.pos_inf, stats.neg_inf
    )
}

pub fn render_alert(
    frame: &mut Frame,
    area: Rect,
    stats: &TensorStats,
    theme: &Theme,
    blink_on: bool,
) {
    let line = Line::styled(alert_text(stats), theme.alert(blink_on));
    frame.render_widget(Paragraph::new(line), area);
}
