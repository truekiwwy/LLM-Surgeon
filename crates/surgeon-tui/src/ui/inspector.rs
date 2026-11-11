//! Center panel: the full identity card of the selected tensor — topology
//! role, layout, quantization, static metrics with their sampled badge, the
//! ablation exactness verdict and the row-magnitude profile.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::app::{App, Focus};
use crate::model::metrics::Stats;
use crate::model::MoeKind;
use crate::ui::theme;

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == Focus::Inspector;
    let title = app
        .selected()
        .map(|e| format!(" {} ", e.name))
        .unwrap_or_else(|| " Inspecteur ".to_owned());
    let block = Block::bordered()
        .title(Span::styled(
            title,
            Style::new().fg(if focused { theme::BORDER_FOCUS } else { theme::ACCENT }),
        ))
        .border_style(if focused { theme::focused_border() } else { theme::border() });
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let Some(entry) = app.selected() else {
        return;
    };
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(6), Constraint::Length(1), Constraint::Min(3)])
        .split(inner);
    render_card(frame, rows[0], app, entry);
    if rows[1].height >= 1 {
        frame.render_widget(
            Paragraph::new(Span::styled(
                " profil de magnitude des lignes (échantillon) :",
                theme::dim(),
            )),
            rows[1],
        );
    }
    render_profile(frame, rows[2], app);
}

fn render_card(frame: &mut Frame, area: Rect, app: &App, entry: &crate::model::TensorEntry) {
    let mut lines: Vec<Line<'static>> = Vec::new();
    let push = |lines: &mut Vec<Line<'static>>, key: &str, value: Span<'static>| {
        lines.push(Line::from(vec![
            Span::styled(format!("{key:<24}"), theme::dim()),
            value,
        ]));
    };
    push(
        &mut lines,
        "rôle",
        Span::styled(
            format!("{} {}", entry.role, moe_label(entry.moe)),
            Style::new().fg(theme::role_color(entry.role)),
        ),
    );
    if let Some(layer) = entry.layer {
        push(
            &mut lines,
            "couche",
            Span::styled(format!("L{layer}"), theme::base()),
        );
    }
    if let Some(expert) = entry.expert {
        push(
            &mut lines,
            "expert",
            Span::styled(format!("#{expert}"), theme::base()),
        );
    }
    push(
        &mut lines,
        "forme",
        Span::styled(
            format!("{} ({} éléments)", entry.shape_text(), entry.numel()),
            theme::base(),
        ),
    );
    let dtype_line = entry
        .dtype
        .bits_per_weight()
        .map_or_else(
            || format!("{} (non décodé)", entry.dtype.name()),
            |bpw| format!("{} ({bpw:.2} bits/poids)", entry.dtype.name()),
        );
    push(&mut lines, "dtype", Span::styled(dtype_line, theme::base()));
    push(
        &mut lines,
        "octets / déplacement",
        Span::styled(
            format!("{} o @ {}", entry.byte_len, entry.file.display()),
            theme::base(),
        ),
    );
    match app.selected_exactness() {
        Some(exactness) => push(
            &mut lines,
            "ablation (project_out)",
            Span::styled(
                format!("{} — effacement du résidu", exactness.label()),
                Style::new().fg(theme::exactness_color(exactness)),
            ),
        ),
        None => push(
            &mut lines,
            "ablation (project_out)",
            Span::styled("— (rôle non écrivain)", theme::dim()),
        ),
    }
    lines.push(Line::from(Span::styled("", theme::dim())));

    match app.selected_index().and_then(|i| app.stats.get(&i)) {
        Some(stats) if stats.available => {
            metrics_lines(&mut lines, stats);
        }
        Some(_) => lines.push(Line::from(Span::styled(
            format!(
                "dtype {} non décodé — métriques indisponibles",
                entry.dtype.name()
            ),
            Style::new().fg(theme::DIM),
        ))),
        None => lines.push(Line::from(Span::styled(
            "métriques en cours de calcul…",
            theme::dim(),
        ))),
    }
    frame.render_widget(Paragraph::new(lines), area);
}

fn metrics_lines(lines: &mut Vec<Line<'static>>, stats: &Stats) {
    let sample = if stats.sampled < stats.numel {
        format!("échantillon {}/{}", stats.sampled, stats.numel)
    } else {
        "exhaustif".to_owned()
    };
    let mut row = |key: &str, value: String| {
        lines.push(Line::from(vec![
            Span::styled(format!("{key:<24}"), theme::dim()),
            Span::styled(value, theme::base()),
        ]));
    };
    row("norme L2", format!("{:.4} ({sample})", stats.l2));
    row("min / max", format!("{:.6} / {:.6}", stats.min, stats.max));
    row("moyenne / |moyenne|", format!("{:.6} / {:.6}", stats.mean, stats.abs_mean));
    row(
        "kurtosis (excès)",
        format!("{:.4} — outliers", stats.kurtosis),
    );
    row(
        "rang effectif approx.",
        format!("{:.2} / {} (projection aléatoire)", stats.r_eff, crate::model::metrics::BASIS),
    );
}

fn render_profile(frame: &mut Frame, area: Rect, app: &App) {
    let Some(stats) = app.selected_index().and_then(|i| app.stats.get(&i)) else {
        return;
    };
    if !stats.available || stats.row_profile.is_empty() {
        return;
    }
    let bars = block_bars(&stats.row_profile, area.width as usize);
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            bars,
            Style::new().fg(theme::ACCENT),
        ))),
        area,
    );
}

/// Renders a sparkline with the Unicode block ramp, right-aligned to `width`.
pub fn block_bars(values: &[f64], width: usize) -> String {
    const RAMP: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if width == 0 || values.is_empty() {
        return String::new();
    }
    let max = values.iter().copied().fold(0.0f64, f64::max).max(1e-12);
    let mut out = String::with_capacity(width * 3);
    for i in 0..width {
        let value = if values.len() <= width {
            values.get(i).copied().unwrap_or(0.0)
        } else {
            let stride = values.len() / width;
            values[i * stride]
        };
        let level = ((value / max) * 7.0).round().clamp(0.0, 7.0) as usize;
        out.push(RAMP[level]);
    }
    out
}

fn moe_label(kind: MoeKind) -> &'static str {
    match kind {
        MoeKind::NotMoe => "",
        MoeKind::Dense => "(dense)",
        MoeKind::Fused => "(experts fusionnés)",
        MoeKind::Shared => "(expert partagé)",
        MoeKind::Router => "(routeur)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkline_covers_the_full_width() {
        let bars = block_bars(&[0.0, 0.25, 0.5, 0.75, 1.0], 10);
        assert_eq!(bars.chars().count(), 10);
        assert!(bars.contains('█'));
        assert!(bars.starts_with('▁'));
        assert_eq!(block_bars(&[], 5), "");
        assert_eq!(block_bars(&[1.0], 0), "");
    }
}
