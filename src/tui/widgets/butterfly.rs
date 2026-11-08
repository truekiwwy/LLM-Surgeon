//! Overlap of the positive and negative projections: positives grow up from a common
//! axis (blue), negatives grow down (white), and the optimal threshold is marked.

use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::tui::theme::Theme;

/// Per-column counts of `values` over `[lo, hi]` in `columns` bins.
pub fn bin_counts(values: &[f64], lo: f64, hi: f64, columns: usize) -> Vec<usize> {
    let mut counts = vec![0usize; columns];
    if columns == 0 {
        return counts;
    }
    let width = hi - lo;
    for &v in values.iter().filter(|v| v.is_finite()) {
        let column = match width > 0.0 {
            true => ((v - lo) / width * columns as f64) as usize,
            false => columns / 2,
        };
        counts[column.min(columns - 1)] += 1;
    }
    counts
}

/// Column of `threshold` on the same axis.
pub fn threshold_column(threshold: f64, lo: f64, hi: f64, columns: usize) -> usize {
    match (hi - lo > 0.0, columns) {
        (_, 0) => 0,
        (true, n) => {
            (((threshold - lo) / (hi - lo)) * n as f64).clamp(0.0, (n - 1) as f64) as usize
        }
        (false, n) => n / 2,
    }
}

/// (rows above the axis, axis line, rows below), plain glyphs.
pub fn butterfly_rows(
    pos: &[f64],
    neg: &[f64],
    threshold: f64,
    columns: usize,
    half: usize,
) -> (Vec<String>, String, Vec<String>) {
    let finite = pos.iter().chain(neg).copied().filter(|v| v.is_finite());
    let (lo, hi) = finite.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| {
        (a.min(v), b.max(v))
    });
    let (lo, hi) = if lo.is_finite() { (lo, hi) } else { (0.0, 0.0) };
    let (up, down) = (
        bin_counts(pos, lo, hi, columns),
        bin_counts(neg, lo, hi, columns),
    );
    let max = up.iter().chain(&down).copied().max().unwrap_or(0).max(1);
    let height = |count: usize| match count {
        0 => 0,
        c => (c * half).div_ceil(max).max(1),
    };
    let above = (0..half)
        .rev()
        .map(|row| {
            up.iter()
                .map(|&c| if height(c) > row { "█" } else { " " })
                .collect()
        })
        .collect();
    let marker = threshold_column(threshold, lo, hi, columns);
    let axis = (0..columns)
        .map(|c| if c == marker { "┼" } else { "─" })
        .collect();
    let below = (0..half)
        .map(|row| {
            down.iter()
                .map(|&c| if height(c) > row { "█" } else { " " })
                .collect()
        })
        .collect();
    (above, axis, below)
}

pub fn render_butterfly(
    frame: &mut Frame,
    area: Rect,
    pos: &[f64],
    neg: &[f64],
    threshold: f64,
    theme: &Theme,
) {
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(
            " Projections : pos ▲ / neg ▼ (┼ seuil) ",
            theme.strong(),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let columns = inner.width as usize;
    let half = inner.height.saturating_sub(2) as usize / 2;
    if columns == 0 || half == 0 {
        return;
    }
    let (above, axis, below) = butterfly_rows(pos, neg, threshold, columns, half);
    let mut lines: Vec<Line> = above
        .into_iter()
        .map(|r| Line::styled(r, theme.accent()))
        .collect();
    lines.push(Line::styled(axis, theme.dim()));
    lines.extend(below.into_iter().map(|r| Line::styled(r, theme.strong())));
    let label = format!("seuil {threshold:.4}");
    lines.push(Line::from(Span::styled(label, theme.dim())));
    frame.render_widget(Paragraph::new(lines), inner);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binning_axis_and_marker() {
        assert_eq!(bin_counts(&[0.0, 1.0, 2.0, f64::NAN], 0.0, 2.0, 2), [1, 2]);
        assert_eq!(
            bin_counts(&[3.0, 3.0], 3.0, 3.0, 4),
            [0, 0, 2, 0],
            "constant: middle column"
        );
        assert_eq!(threshold_column(1.0, 0.0, 2.0, 4), 2);
        let (above, axis, below) = butterfly_rows(&[2.0, 2.0], &[0.0], 1.0, 4, 2);
        assert_eq!(above, ["   █", "   █"]);
        assert_eq!(axis, "──┼─");
        assert_eq!(below, ["█   ", "    "]);
    }
}
