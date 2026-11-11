//! The interface: three responsive panels — the layer tree, the tensor
//! inspector with its magnitude profile, and the agent bridge with the event
//! log and the patch preview — over a one-line status footer.
//!
//! Layout adapts: below 100 columns everything stacks vertically, below 44
//! columns the inspector collapses to the essentials. Rendering is pure:
//! every panel reads [`App`] and draws, the app state never changes here.

mod bridge;
mod inspector;
mod layers;
pub mod theme;

pub use layers::filter_label;

use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::Terminal;
use std::io::Stdout;

use crate::app::App;

/// Renders one frame (called at 60 Hz).
pub fn draw(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &App,
) -> std::io::Result<()> {
    terminal
        .draw(|frame| {
        let area = frame.area();
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(1)])
            .split(area);
        let panels = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(0)])
            .split(rows[0]);
        draw_filter_bar(frame, panels[0], app);
        if area.width >= 100 {
            let columns = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([
                    Constraint::Percentage(28),
                    Constraint::Percentage(46),
                    Constraint::Percentage(26),
                ])
                .split(panels[1]);
            layers::render(frame, columns[0], app);
            inspector::render(frame, columns[1], app);
            bridge::render(frame, columns[2], app);
        } else {
            let stacked = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Percentage(34),
                    Constraint::Percentage(38),
                    Constraint::Percentage(28),
                ])
                .split(panels[1]);
            layers::render(frame, stacked[0], app);
            inspector::render(frame, stacked[1], app);
            bridge::render(frame, stacked[2], app);
        }
        draw_status(frame, rows[1], app);
        })
        .map(|_| ())
}

fn draw_filter_bar(frame: &mut ratatui::Frame, area: ratatui::prelude::Rect, app: &App) {
    use ratatui::text::{Line, Span};
    use ratatui::widgets::Paragraph;
    let spans = if app.searching {
        vec![
            Span::styled("/ ", theme::dim()),
            Span::styled(
                app.text_filter.clone(),
                ratatui::style::Style::new().fg(theme::ACCENT),
            ),
            Span::styled("▌", theme::dim()),
        ]
    } else {
        vec![Span::styled(
            format!(
                "  / recherche   r rôle : {}   filtre texte : {}",
                filter_label(app),
                if app.text_filter.is_empty() {
                    "—".to_owned()
                } else {
                    app.text_filter.clone()
                }
            ),
            theme::dim(),
        )]
    };
    frame.render_widget(
        Paragraph::new(Line::from(spans)).style(theme::base()),
        area,
    );
}

fn draw_status(frame: &mut ratatui::Frame, area: ratatui::prelude::Rect, app: &App) {
    use ratatui::text::{Line, Span};
    use ratatui::style::Style;
    use ratatui::widgets::Paragraph;
    let Some(entry) = app.selected() else {
        frame.render_widget(
            Paragraph::new(Line::from(Span::styled(
                " aucun tenseur — vérifiez le chemin du modèle",
                Style::new().fg(theme::DIM),
            ))),
            area,
        );
        return;
    };
    let metrics_state = match app.selected_index().and_then(|i| app.stats.get(&i)) {
        Some(stats) if stats.available => format!(
            "L2 {:.3} · kurt {:.2} · r_eff {:.2}/{}",
            stats.l2,
            stats.kurtosis,
            stats.r_eff,
            crate::model::metrics::BASIS
        ),
        Some(_) => "métriques indisponibles".to_owned(),
        None => "métriques en cours…".to_owned(),
    };
    let exactness = app
        .selected_exactness()
        .map(|e| e.label())
        .unwrap_or("—");
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" ● ", Style::new().fg(theme::role_color(entry.role))),
            Span::styled(&entry.name, Style::new().fg(theme::TEXT)),
            Span::styled(format!("  {}  {exactness}  ", entry.role.label()), {
                let color = app
                    .selected_exactness()
                    .map(theme::exactness_color)
                    .unwrap_or(theme::DIM);
                Style::new().fg(color)
            }),
            Span::styled(metrics_state, Style::new().fg(theme::DIM)),
        ])),
        area,
    );
}
