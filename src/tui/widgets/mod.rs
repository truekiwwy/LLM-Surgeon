//! Widgets drawn by `ui::view`. They only read state: no computation is ever triggered.
//!
//! ```text
//! view -> matrix_panel (layers, projections, metrics) -> alert, histogram
//!      -> steering_panel (vector, top-10, components)  -> alert, histogram, gauge_row, butterfly
//! ```

pub mod alert;
pub mod butterfly;
pub mod gauge_row;
pub mod histogram;
pub mod matrix_panel;
pub mod steering_panel;

/// Eighth blocks, from 1/8 to 8/8.
pub const BLOCKS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];

/// Resamples `values` to `columns` (sums when shrinking, repeats when stretching).
pub fn resample(values: &[f64], columns: usize) -> Vec<f64> {
    if values.is_empty() || columns == 0 {
        return vec![0.0; columns];
    }
    (0..columns)
        .map(|c| {
            let start = c * values.len() / columns;
            let end = ((c + 1) * values.len() / columns).max(start + 1);
            values[start..end.min(values.len())].iter().sum()
        })
        .collect()
}

/// Glyph of row `row` (0 = bottom) for a bar of `eighths` eighths.
pub fn bar_glyph(eighths: usize, row: usize) -> &'static str {
    match eighths.saturating_sub(row * 8) {
        0 => " ",
        n => BLOCKS[n.min(8) - 1],
    }
}

/// Bar height in eighths for `value` out of `max`, over `rows` rows. A non-zero value
/// always shows at least one eighth.
pub fn eighths(value: f64, max: f64, rows: usize, log: bool) -> usize {
    if value <= 0.0 || max <= 0.0 || !value.is_finite() {
        return 0;
    }
    let ratio = match log {
        true => (1.0 + value).ln() / (1.0 + max).ln(),
        false => value / max,
    };
    ((ratio * (rows * 8) as f64).round() as usize).clamp(1, rows * 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampling_and_bars() {
        assert_eq!(resample(&[1.0, 2.0, 3.0, 4.0], 2), [3.0, 7.0]);
        assert_eq!(resample(&[1.0, 2.0], 4), [1.0, 1.0, 2.0, 2.0]);
        assert_eq!(resample(&[], 3), [0.0; 3]);
        assert_eq!(eighths(0.0, 10.0, 2, false), 0);
        assert_eq!(eighths(0.01, 10.0, 2, false), 1, "visible even when tiny");
        assert_eq!(eighths(10.0, 10.0, 2, true), 16);
        assert_eq!(bar_glyph(12, 0), "█");
        assert_eq!(bar_glyph(12, 1), "▄");
        assert_eq!(bar_glyph(12, 2), " ");
    }
}
