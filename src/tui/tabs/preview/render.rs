//! `view` of the tab: reads [`PreviewState`] only.

use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::Frame;

use super::{ModelStatus, PaneState, PaneStatus, PreviewState, NOTE};
use crate::tui::Theme;

const SPINNER: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
/// Loop ticks per spinner frame.
const SPINNER_TICKS: u64 = 6;

pub fn view(frame: &mut Frame, area: Rect, state: &PreviewState, theme: &Theme) {
    let [input, controls, panes, footer] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(6),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(area);
    render_input(frame, input, state, theme);
    render_controls(frame, controls, state, theme);
    match &state.model {
        ModelStatus::Unavailable(message) => render_unavailable(frame, panes, message, theme),
        _ => render_panes(frame, panes, state, theme),
    }
    render_footer(frame, footer, state, theme);
}

fn render_input(frame: &mut Frame, area: Rect, state: &PreviewState, theme: &Theme) {
    let title = match state.input_focus {
        true => " Prompt · Entrée générer · Esc quitter la saisie ",
        false => " Prompt · i ou Entrée pour éditer ",
    };
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(title, theme.strong()));
    let inner = block.inner(area);
    let (shown, col) = state.editor.visible(usize::from(inner.width));
    let style = match state.input_focus {
        true => theme.strong(),
        false => theme.dim(),
    };
    frame.render_widget(
        Paragraph::new(Span::styled(shown, style)).block(block),
        area,
    );
    if state.input_focus {
        let x = inner.x
            + u16::try_from(col)
                .unwrap_or(0)
                .min(inner.width.saturating_sub(1));
        frame.set_cursor_position((x, inner.y));
    }
}

/// `─────●────`: the slider position inside the range.
fn slider_bar(alpha: f64, (lo, hi): (f64, f64), width: usize) -> String {
    let last = width.saturating_sub(1);
    let at = (((alpha - lo) / (hi - lo)).clamp(0.0, 1.0) * last as f64).round() as usize;
    (0..width)
        .map(|i| match i == at {
            true => '●',
            false => '─',
        })
        .collect()
}

fn render_controls(frame: &mut Frame, area: Rect, state: &PreviewState, theme: &Theme) {
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(" Steering ", theme.strong()));
    let inner = block.inner(area);
    let (lo, hi) = state.alpha_range;
    let (lo_text, hi_text) = (format!("{lo:+.2}"), format!("{hi:+.2}"));
    let bar_width = usize::from(inner.width)
        .saturating_sub("alpha ".len() + lo_text.len() + hi_text.len() + 4)
        .max(3);
    let slider = Line::from(vec![
        Span::styled("alpha ", theme.dim()),
        Span::styled(lo_text, theme.dim()),
        Span::raw(" "),
        Span::styled(
            slider_bar(state.alpha, state.alpha_range, bar_width),
            theme.accent(),
        ),
        Span::raw(" "),
        Span::styled(hi_text, theme.dim()),
    ]);
    let values = match (state.alpha_abs(), state.scale) {
        (Some(abs), Some(scale)) => format!(
            "alpha = {:+.2}   alpha_abs = {abs:+.3}   (s = {scale:.3})",
            state.alpha
        ),
        _ => format!(
            "alpha = {:+.2}   alpha_abs = mesure de l'échelle s…",
            state.alpha
        ),
    };
    let lines = vec![
        slider,
        Line::styled(values, theme.strong()),
        Line::styled(
            format!(
                "couche {} · gabarit {} · {} tokens max",
                state.layer,
                state.template.as_str(),
                state.tokens
            ),
            theme.dim(),
        ),
        Line::styled(NOTE, theme.dim()),
    ];
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn render_panes(frame: &mut Frame, area: Rect, state: &PreviewState, theme: &Theme) {
    let [left, right] = Layout::horizontal([Constraint::Percentage(50); 2]).areas(area);
    let steered_title = format!("Steered (alpha={:+.2})", state.shown_alpha);
    render_pane(frame, left, "Baseline (alpha=0)", &state.baseline, theme);
    render_pane(frame, right, &steered_title, &state.steered, theme);
}

fn render_pane(frame: &mut Frame, area: Rect, title: &str, pane: &PaneState, theme: &Theme) {
    let speed = match pane.tok_per_s > 0.0 {
        true => format!(" · {:.1} tok/s", pane.tok_per_s),
        false => String::new(),
    };
    let status = match &pane.status {
        PaneStatus::Running => " · en cours",
        _ => "",
    };
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(
            format!(" {title}{status}{speed} "),
            theme.strong(),
        ));
    let body = match &pane.status {
        PaneStatus::Failed(message) => Span::styled(format!("erreur : {message}"), theme.strong()),
        _ => Span::styled(pane.text.clone(), theme.strong()),
    };
    frame.render_widget(
        Paragraph::new(body).wrap(Wrap { trim: false }).block(block),
        area,
    );
}

fn render_unavailable(frame: &mut Frame, area: Rect, message: &str, theme: &Theme) {
    let lines = vec![
        Line::styled("Prévisualisation indisponible", theme.strong()),
        Line::raw(""),
        Line::styled(message.to_owned(), theme.strong()),
        Line::raw(""),
        Line::styled(
            "Les onglets 1 et 2 restent utilisables (Tab pour changer d'onglet).",
            theme.dim(),
        ),
    ];
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(" Modèle ", theme.strong()));
    frame.render_widget(
        Paragraph::new(lines).wrap(Wrap { trim: true }).block(block),
        area,
    );
}

fn render_footer(frame: &mut Frame, area: Rect, state: &PreviewState, theme: &Theme) {
    let mut parts: Vec<String> = Vec::new();
    if state.model == ModelStatus::Loading {
        let frame_index = (state.ticks / SPINNER_TICKS) as usize % SPINNER.len();
        parts.push(format!("{} chargement du modèle…", SPINNER[frame_index]));
    }
    if let Some(speed) = state.slow() {
        parts.push(format!(
            "avertissement : génération lente ({speed:.1} tok/s, moins de 1)"
        ));
    }
    parts.extend(state.warnings.iter().cloned());
    if state.pending() {
        parts.push("alpha modifié, génération dans un instant…".into());
    }
    parts.extend(state.notice.iter().cloned());
    frame.render_widget(
        Paragraph::new(Line::styled(parts.join(" · "), theme.dim())),
        area,
    );
}
