//! Tab 2: the steering vector (norm, top-10, components) and its separation quality.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;

use super::alert::render_alert;
use super::butterfly::render_butterfly;
use super::gauge_row::render_gauge;
use super::histogram::render_histogram;
use crate::tui::app::{App, CompanionState, VectorView};
use crate::tui::theme::Theme;

/// d′ above this fills the gauge (the exact value is still printed).
pub const D_PRIME_SCALE: f64 = 4.0;

pub fn render_steering(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let Some(vector) = &app.vector else {
        let message = Line::styled(
            "Aucun vecteur : relancer avec --vector direction.safetensors",
            theme.dim(),
        );
        frame.render_widget(
            Paragraph::new(message).block(Block::bordered().border_style(theme.accent())),
            area,
        );
        return;
    };
    let [left, right] =
        Layout::horizontal([Constraint::Percentage(50), Constraint::Percentage(50)]).areas(area);
    let alert_rows = u16::from(vector.stats.non_finite > 0);
    let [alert, summary, middle, components] = Layout::vertical([
        Constraint::Length(alert_rows),
        Constraint::Length(7),
        Constraint::Length(12),
        Constraint::Min(0),
    ])
    .areas(left);
    if vector.stats.non_finite > 0 {
        render_alert(frame, alert, &vector.stats, theme, app.blink_on());
    }
    render_vector(frame, summary, app, vector);
    let [top, histogram] =
        Layout::horizontal([Constraint::Length(26), Constraint::Min(0)]).areas(middle);
    render_top(frame, top, app, vector);
    render_histogram(
        frame,
        histogram,
        vector.stats.histogram.as_ref(),
        false,
        theme,
        "Valeurs",
    );
    render_components(frame, components, app, vector);
    render_separation(frame, right, app, vector);
}

fn kv<'a>(theme: &Theme, key: &'a str, value: Span<'a>) -> Line<'a> {
    Line::from(vec![Span::styled(format!("{key:<16}"), theme.dim()), value])
}

fn render_vector(frame: &mut Frame, area: Rect, app: &App, v: &VectorView) {
    let theme = &app.theme;
    let s = &v.stats;
    let norm_flag = match v.is_normalized() {
        true => "normalisé ✓",
        false => "non normalisé ⚠",
    };
    let mut lines = vec![
        kv(
            theme,
            "taille",
            Span::styled(v.values.len().to_string(), theme.strong()),
        ),
        kv(
            theme,
            "norme L2",
            Span::styled(format!("{:.3}  {norm_flag}", v.norm()), theme.strong()),
        ),
        kv(
            theme,
            "min · max",
            Span::styled(format!("{:.4} · {:.4}", s.min, s.max), theme.strong()),
        ),
        kv(
            theme,
            "moyenne",
            Span::styled(format!("{:.4e}", s.mean), theme.strong()),
        ),
    ];
    if let Some(hidden) = app.model.as_ref().and_then(|m| m.hidden_size) {
        let compat = match hidden == v.values.len() {
            true => Span::styled("compatible ✓".to_owned(), theme.strong()),
            false => Span::styled(
                format!("incompatible ✗ ({} ≠ {hidden})", v.values.len()),
                theme.alert(true),
            ),
        };
        lines.push(kv(theme, "modèle", compat));
    }
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(" Vecteur ", theme.strong()));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_top(frame: &mut Frame, area: Rect, app: &App, v: &VectorView) {
    let theme = &app.theme;
    let lines: Vec<Line> = v
        .top
        .iter()
        .enumerate()
        .skip(app.top_scroll)
        .map(|(rank, (i, value))| {
            Line::from(vec![
                Span::styled(format!("{:>2}. dim {i:<5}", rank + 1), theme.dim()),
                Span::styled(format!("{value:+.4}"), theme.strong()),
            ])
        })
        .collect();
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(" TOP-10 |v_i| ", theme.strong()));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

/// Components `pan..` sampled every `zoom`, as bars around a zero axis.
pub fn component_rows(
    values: &[f32],
    pan: usize,
    zoom: usize,
    columns: usize,
    half: usize,
) -> (Vec<String>, Vec<String>) {
    let picked: Vec<f32> = (0..columns)
        .filter_map(|c| values.get(pan + c * zoom.max(1)).copied())
        .collect();
    let max = picked
        .iter()
        .filter(|v| v.is_finite())
        .map(|v| v.abs())
        .fold(0.0f32, f32::max);
    let height = |v: f32| match v.is_finite() && max > 0.0 {
        true => ((v.abs() / max) * half as f32).ceil() as usize,
        false => 0,
    };
    let above = (0..half)
        .rev()
        .map(|row| {
            picked
                .iter()
                .map(|&v| {
                    if v > 0.0 && height(v) > row {
                        "█"
                    } else {
                        " "
                    }
                })
                .collect()
        })
        .collect();
    let below = (0..half)
        .map(|row| {
            picked
                .iter()
                .map(|&v| {
                    if v < 0.0 && height(v) > row {
                        "█"
                    } else {
                        " "
                    }
                })
                .collect()
        })
        .collect();
    (above, below)
}

fn render_components(frame: &mut Frame, area: Rect, app: &App, v: &VectorView) {
    let theme = &app.theme;
    let end = (app.pan + app.zoom * 80).min(v.values.len());
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(
            format!(" Composantes {}..{} (zoom ×{}) ", app.pan, end, app.zoom),
            theme.strong(),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let half = inner.height.saturating_sub(1) as usize / 2;
    if half == 0 {
        return;
    }
    let (above, below) = component_rows(&v.values, app.pan, app.zoom, inner.width as usize, half);
    let mut lines: Vec<Line> = above
        .into_iter()
        .map(|r| Line::styled(r, theme.accent()))
        .collect();
    lines.push(Line::styled("─".repeat(inner.width as usize), theme.dim()));
    lines.extend(below.into_iter().map(|r| Line::styled(r, theme.strong())));
    frame.render_widget(Paragraph::new(lines), inner);
}

fn render_separation(frame: &mut Frame, area: Rect, app: &App, v: &VectorView) {
    let theme = &app.theme;
    let meta = |key: &str| v.metadata.get(key).cloned().unwrap_or_else(|| "?".into());
    let companion = match &v.companion {
        CompanionState::Loaded(c) => Some(c),
        _ => None,
    };
    let d_prime = companion
        .map(|c| c.d_prime)
        .unwrap_or_else(|| meta("d_prime").parse().ok().filter(|d: &f64| d.is_finite()));
    let accuracy = companion
        .map(|c| c.threshold_accuracy)
        .or_else(|| meta("threshold_accuracy").parse().ok());
    let [gauges, details, butterfly] = Layout::vertical([
        Constraint::Length(6),
        Constraint::Length(9),
        Constraint::Min(0),
    ])
    .areas(area);
    let [g1, g2] = Layout::vertical([Constraint::Length(3), Constraint::Length(3)]).areas(gauges);
    match d_prime {
        Some(d) => render_gauge(
            frame,
            g1,
            "d′ (séparation)",
            d / D_PRIME_SCALE,
            format!("{d:.2}"),
            theme,
        ),
        None => render_gauge(
            frame,
            g1,
            "d′ (séparation)",
            1.0,
            "∞ ou inconnu".into(),
            theme,
        ),
    }
    match accuracy {
        Some(a) => render_gauge(
            frame,
            g2,
            "accuracy au seuil",
            a,
            format!("{:.0} %", a * 100.0),
            theme,
        ),
        None => render_gauge(
            frame,
            g2,
            "accuracy au seuil",
            0.0,
            "inconnue".into(),
            theme,
        ),
    }

    let value = |text: String| Span::styled(text, theme.strong());
    let number = |key: &str| {
        meta(key)
            .parse::<f64>()
            .map_or_else(|_| meta(key), |x| format!("{x:.4}"))
    };
    let lines = vec![
        kv(
            theme,
            "proj. μ₊ · μ₋",
            value(format!(
                "{} · {}",
                number("pos_mean_proj"),
                number("neg_mean_proj")
            )),
        ),
        kv(
            theme,
            "cos(μ₊, μ₋)",
            value(companion.map_or_else(
                || number("cos_mean_pos_neg"),
                |c| format!("{:.4}", c.cos_mean_pos_neg),
            )),
        ),
        kv(
            theme,
            "couche · site",
            value(format!("{} · {}", meta("layer"), meta("site"))),
        ),
        kv(theme, "pooling", value(meta("pooling"))),
        kv(
            theme,
            "n_pos · n_neg",
            value(format!("{} · {}", meta("n_pos"), meta("n_neg"))),
        ),
        kv(theme, "modèle source", value(meta("model"))),
    ];
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(" Séparation ", theme.strong()));
    frame.render_widget(Paragraph::new(lines).block(block), details);

    match &v.companion {
        CompanionState::Loaded(c) => render_butterfly(
            frame,
            butterfly,
            &c.pos_projections,
            &c.neg_projections,
            c.threshold,
            theme,
        ),
        CompanionState::Missing => {
            let message = Line::styled(
                "projections indisponibles : relancer `surgeon extract` (v4.0+)",
                theme.dim(),
            );
            let paragraph = Paragraph::new(message).wrap(Wrap { trim: true });
            frame.render_widget(
                paragraph.block(Block::bordered().border_style(theme.accent())),
                butterfly,
            );
        }
        CompanionState::Invalid(why) => {
            let message = Line::styled(
                format!("⚠ fichier compagnon ignoré : {why}"),
                theme.strong(),
            );
            let paragraph = Paragraph::new(message).wrap(Wrap { trim: true });
            frame.render_widget(
                paragraph.block(Block::bordered().border_style(theme.accent())),
                butterfly,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn components_around_zero() {
        let (above, below) = component_rows(&[1.0, -0.5, 0.0, f32::NAN], 0, 1, 4, 2);
        assert_eq!(above, ["█   ", "█   "]);
        assert_eq!(below, [" █  ", "    "]);
        let (above, _) = component_rows(&[1.0, 2.0, 3.0, 4.0], 1, 2, 4, 1);
        assert_eq!(above, ["██"], "pan 1, zoom 2: components 1 and 3");
    }
}
