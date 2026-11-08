//! Distribution of the finite values in eighth blocks, linear or logarithmic.

use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use super::{bar_glyph, eighths, resample};
use crate::engine::stats::Histogram;
use crate::tui::theme::Theme;

/// Rows of glyphs (top first) for `counts` resampled to `width` x `height`.
pub fn histogram_rows(counts: &[u64], width: usize, height: usize, log: bool) -> Vec<String> {
    let values: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
    let columns = resample(&values, width);
    let max = columns.iter().copied().fold(0.0, f64::max);
    let heights: Vec<usize> = columns
        .iter()
        .map(|&v| eighths(v, max, height, log))
        .collect();
    (0..height)
        .rev()
        .map(|row| heights.iter().map(|&h| bar_glyph(h, row)).collect())
        .collect()
}

pub fn render_histogram(
    frame: &mut Frame,
    area: Rect,
    hist: Option<&Histogram>,
    log: bool,
    theme: &Theme,
    title: &str,
) {
    let scale = if log { "log" } else { "linéaire" };
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(format!(" {title} ({scale}) "), theme.strong()));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let Some(hist) = hist else {
        let message = Line::styled("aucune valeur finie : histogramme vide", theme.dim());
        frame.render_widget(Paragraph::new(message), inner);
        return;
    };
    let (width, height) = (
        inner.width as usize,
        inner.height.saturating_sub(1) as usize,
    );
    if width == 0 || height == 0 {
        return;
    }
    let mut lines: Vec<Line> = histogram_rows(&hist.counts, width, height, log)
        .into_iter()
        .map(|row| Line::styled(row, theme.accent()))
        .collect();
    let (min, max) = (format!("{:.4e}", hist.min), format!("{:.4e}", hist.max));
    let gap = width.saturating_sub(min.chars().count() + max.chars().count());
    lines.push(Line::from(vec![
        Span::styled(min, theme.dim()),
        Span::raw(" ".repeat(gap)),
        Span::styled(max, theme.dim()),
    ]));
    frame.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_are_bottom_anchored_and_log_lifts_small_bins() {
        let rows = histogram_rows(&[8, 0, 4], 3, 1, false);
        assert_eq!(rows, ["█ ▄"]);
        let linear = histogram_rows(&[1000, 1], 2, 4, false);
        let log = histogram_rows(&[1000, 1], 2, 4, true);
        assert_eq!(linear[3], "█▁");
        assert_eq!(log[3], "█▃", "log scale lifts the small bin");
        assert_eq!(histogram_rows(&[5], 4, 2, false), ["████", "████"]);
    }
}
