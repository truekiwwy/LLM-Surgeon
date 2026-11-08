//! Tab 1: layer list with projection selector, live metrics, alert and histogram.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use super::alert::render_alert;
use super::histogram::render_histogram;
use crate::tui::app::{App, Computation, PROJECTIONS};
use crate::tui::theme::Theme;

const LIST_WIDTH: u16 = 26;

pub fn render_inspector(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let Some(model) = &app.model else {
        let message = Line::styled(
            "Aucun modèle : relancer avec --model <MODELS_DIR>/mon-modele",
            theme.dim(),
        );
        frame.render_widget(
            Paragraph::new(message).block(Block::bordered().border_style(theme.accent())),
            area,
        );
        return;
    };
    let [list, right] =
        Layout::horizontal([Constraint::Length(LIST_WIDTH), Constraint::Min(0)]).areas(area);
    render_layers(frame, list, app);

    let alert_rows = u16::from(app.current_stats().is_some_and(|s| s.non_finite > 0));
    let [alert, metrics, histogram] = Layout::vertical([
        Constraint::Length(alert_rows),
        Constraint::Length(11),
        Constraint::Min(0),
    ])
    .areas(right);
    if let Some(stats) = app.current_stats().filter(|s| s.non_finite > 0) {
        render_alert(frame, alert, stats, theme, app.blink_on());
    }
    if model.layers.is_empty() {
        let message = Line::styled(
            "aucune couche MLP (model.layers.N.mlp.*_proj) dans ce fichier",
            theme.dim(),
        );
        frame.render_widget(
            Paragraph::new(message).block(Block::bordered().border_style(theme.accent())),
            metrics,
        );
        return;
    }
    render_metrics(frame, metrics, app);
    let hist = app.current_stats().and_then(|s| s.histogram.as_ref());
    match &app.computation {
        Computation::Done(Ok(_)) => render_histogram(
            frame,
            histogram,
            hist,
            app.log_scale,
            theme,
            "Distribution des valeurs finies",
        ),
        _ => frame.render_widget(Block::bordered().border_style(theme.accent()), histogram),
    }
}

fn render_layers(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let Some(model) = &app.model else { return };
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(
            format!(" Couches ({}) ", model.layers.len()),
            theme.strong(),
        ));
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let layer = app.selection.layer;
    let header: Vec<Span> = PROJECTIONS
        .iter()
        .enumerate()
        .flat_map(|(p, name)| {
            let label = name.trim_end_matches("_proj");
            let available = model
                .layers
                .get(layer)
                .and_then(|l| l[p].as_ref())
                .is_some();
            // Brackets as well as reverse video: the selection stays readable without color.
            let (text, style) = match (p == app.selection.proj, available) {
                (true, _) => (format!("[{label}]"), theme.selected()),
                (false, true) => (format!(" {label} "), theme.strong()),
                (false, false) => (format!(" {label} "), theme.dim()),
            };
            [Span::styled(text, style), Span::styled("│", theme.dim())]
        })
        .collect();
    let mut lines = vec![Line::from(header)];

    let rows = (inner.height as usize).saturating_sub(1);
    let start = layer
        .saturating_sub(rows / 2)
        .min(model.layers.len().saturating_sub(rows));
    for (i, tensors) in model.layers.iter().enumerate().skip(start).take(rows) {
        let warn = PROJECTIONS
            .iter()
            .enumerate()
            .any(|(p, _)| app.flagged.get(&(i, p)).copied().unwrap_or(false));
        let marker = if warn { " ⚠" } else { "" };
        let missing = tensors.iter().filter(|t| t.is_none()).count();
        let note = if missing > 0 {
            format!(" ({missing} absent)")
        } else {
            String::new()
        };
        let style = if i == layer {
            theme.selected()
        } else if missing == 3 {
            theme.dim()
        } else {
            theme.strong()
        };
        let cursor = if i == layer { "▶" } else { " " };
        lines.push(Line::from(vec![
            Span::styled(format!("{cursor}L{i:<4}{marker}"), style),
            Span::styled(note, theme.dim()),
        ]));
    }
    frame.render_widget(Paragraph::new(lines), inner);
}

fn kv<'a>(theme: &Theme, key: &'a str, value: String) -> Line<'a> {
    Line::from(vec![
        Span::styled(format!("{key:<16}"), theme.dim()),
        Span::styled(value, theme.strong()),
    ])
}

fn render_metrics(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(" Métriques ", theme.strong()));
    let Some(entry) = app.current_entry() else {
        let message = Line::styled("projection absente pour cette couche", theme.dim());
        frame.render_widget(Paragraph::new(message).block(block), area);
        return;
    };
    let mut lines = vec![
        kv(theme, "tenseur", entry.name.clone()),
        kv(
            theme,
            "dtype · shape",
            format!(
                "{} · {:?} · {} octets",
                entry.dtype, entry.shape, entry.bytes
            ),
        ),
    ];
    match &app.computation {
        Computation::Running { .. } | Computation::Idle => {
            lines.push(Line::styled(
                format!("{} calcul…", app.spinner()),
                theme.strong(),
            ));
        }
        Computation::Done(Err(message)) => lines.push(Line::styled(
            format!("erreur : {message}"),
            theme.alert(true),
        )),
        Computation::Done(Ok(s)) if s.count == 0 => {
            lines.push(kv(theme, "éléments finis", format!("0 / {}", s.elements)));
            lines.push(Line::styled(
                "aucune valeur finie (tenseur vide ou entièrement NaN/Inf)",
                theme.dim(),
            ));
        }
        Computation::Done(Ok(s)) => {
            lines.push(kv(
                theme,
                "min · max",
                format!("{:.6} · {:.6}", s.min, s.max),
            ));
            lines.push(kv(theme, "moyenne", format!("{:.6e}", s.mean)));
            lines.push(kv(
                theme,
                "écart-type",
                format!("{:.6e} (population)", s.std),
            ));
            lines.push(kv(
                theme,
                "sparsité",
                format!("{:.3} % de zéros exacts", s.sparsity() * 100.0),
            ));
            lines.push(kv(
                theme,
                "NaN · ±Inf",
                format!("{} NaN · {} +Inf · {} -Inf", s.nan, s.pos_inf, s.neg_inf),
            ));
            lines.push(kv(
                theme,
                "éléments finis",
                format!("{} / {}", s.count, s.elements),
            ));
            lines.push(kv(theme, "norme L2", format!("{:.6}", s.l2)));
        }
    }
    frame.render_widget(Paragraph::new(lines).block(block), area);
}
