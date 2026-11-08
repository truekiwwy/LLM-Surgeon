//! `view`: pure rendering of the state. It reads `App` and never computes statistics.

use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Clear, Paragraph, Tabs, Wrap};
use ratatui::Frame;

use super::app::{App, Computation, Tab, MIN_HEIGHT, MIN_WIDTH};
use super::widgets::{matrix_panel, steering_panel};

pub fn view(frame: &mut Frame, app: &App) {
    let area = frame.area();
    let theme = &app.theme;
    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        let message = Paragraph::new(Line::styled(
            format!("Terminal trop petit ({MIN_WIDTH}×{MIN_HEIGHT} minimum)"),
            theme.strong(),
        ))
        .alignment(Alignment::Center)
        .wrap(Wrap { trim: true });
        frame.render_widget(message, area);
        return;
    }
    let [tabs_area, body, status] = Layout::vertical([
        Constraint::Length(3),
        Constraint::Min(0),
        Constraint::Length(1),
    ])
    .areas(area);

    let selected = match app.tab {
        Tab::Inspector => 0,
        Tab::Steering => 1,
    };
    let tabs = Tabs::new(["1 Inspecteur de poids", "2 Steering vector"])
        .select(selected)
        .style(theme.dim())
        .highlight_style(theme.selected())
        .block(
            Block::bordered()
                .border_style(theme.accent())
                .title(Span::styled(" surgeon tui ", theme.strong())),
        );
    frame.render_widget(tabs, tabs_area);

    match (app.model.is_none() && app.vector.is_none(), app.tab) {
        (true, _) => welcome(frame, body, app),
        (false, Tab::Inspector) => matrix_panel::render_inspector(frame, body, app),
        (false, Tab::Steering) => steering_panel::render_steering(frame, body, app),
    }
    status_bar(frame, status, app);
    if app.help {
        help_overlay(frame, area, app);
    }
}

fn welcome(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let lines = vec![
        Line::styled("Aucun modèle ni vecteur chargé.", theme.strong()),
        Line::raw(""),
        Line::styled(
            "Inspecter les poids MLP d'un modèle SafeTensors :",
            theme.dim(),
        ),
        Line::styled(
            "  surgeon tui --model <MODELS_DIR>/mon-modele",
            theme.strong(),
        ),
        Line::raw(""),
        Line::styled(
            "Visualiser un vecteur produit par surgeon extract :",
            theme.dim(),
        ),
        Line::styled(
            "  surgeon tui --vector direction.safetensors",
            theme.strong(),
        ),
        Line::raw(""),
        Line::styled(
            "Les deux ensemble vérifient la compatibilité des tailles.",
            theme.dim(),
        ),
        Line::styled("? aide · q quitter", theme.dim()),
    ];
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(" Bienvenue ", theme.strong()));
    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn status_bar(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let keys = match app.tab {
        Tab::Inspector => "Tab onglet · ↑↓ couche · ←→ projection · l log · ? aide · q quitter",
        Tab::Steering => "Tab onglet · ←→ défiler · +/- zoom · ↑↓ top-10 · ? aide · q quitter",
    };
    let state = match &app.computation {
        Computation::Running { .. } => format!("{} calcul…", app.spinner()),
        Computation::Done(Err(_)) => "erreur".to_owned(),
        Computation::Done(Ok(_)) => "prêt".to_owned(),
        Computation::Idle => String::new(),
    };
    let path = app
        .model
        .as_ref()
        .map(|m| m.path.display().to_string())
        .or_else(|| app.vector.as_ref().map(|v| v.path.display().to_string()))
        .unwrap_or_default();
    let room =
        (area.width as usize).saturating_sub(keys.chars().count() + state.chars().count() + 6);
    let line = Line::from(vec![
        Span::styled(keys, theme.dim()),
        Span::raw("  "),
        Span::styled(crate::ui::truncate_middle(&path, room), theme.dim()),
        Span::raw("  "),
        Span::styled(state, theme.strong()),
    ]);
    frame.render_widget(Paragraph::new(line), area);
}

fn help_overlay(frame: &mut Frame, area: Rect, app: &App) {
    let theme = &app.theme;
    let rows = [
        ("Tab / Shift+Tab", "changer d'onglet"),
        ("↑ / ↓", "couche (onglet 1) · top-10 (onglet 2)"),
        (
            "← / →",
            "projection (onglet 1) · défiler les composantes (onglet 2)",
        ),
        ("PgUp / PgDn", "±10 couches"),
        ("Home / End", "première / dernière couche"),
        ("l", "échelle linéaire / logarithmique"),
        ("+ / -", "zoom des composantes"),
        ("?", "afficher / masquer cette aide"),
        ("q · Esc · Ctrl-C", "quitter (Esc ferme d'abord l'aide)"),
    ];
    let lines: Vec<Line> = rows
        .iter()
        .map(|(k, v)| {
            Line::from(vec![
                Span::styled(format!("{k:<18}"), theme.strong()),
                Span::styled(*v, theme.dim()),
            ])
        })
        .collect();
    let width = 72.min(area.width);
    let height = (rows.len() as u16 + 2).min(area.height);
    let popup = Rect::new(
        area.x + (area.width - width) / 2,
        area.y + (area.height - height) / 2,
        width,
        height,
    );
    frame.render_widget(Clear, popup);
    let block = Block::bordered()
        .border_style(theme.accent())
        .title(Span::styled(" Aide ", theme.strong()));
    frame.render_widget(Paragraph::new(lines).block(block), popup);
}
