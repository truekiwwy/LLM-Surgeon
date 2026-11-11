//! Right panel: the agent bridge — the IPC endpoint banner, the live event
//! log of everything the bridge received, the patch preview proposed by an
//! agent, and the key hints for exporting context / launching the agent.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;

use crate::app::{App, Focus};
use crate::ui::theme;

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == Focus::Bridge;
    let block = Block::bordered()
        .title(Span::styled(
            " Bridge agents ",
            Style::new().fg(if focused { theme::BORDER_FOCUS } else { theme::ACCENT }),
        ))
        .border_style(if focused { theme::focused_border() } else { theme::border() });
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(4), Constraint::Length(3), Constraint::Length(2)])
        .split(inner);
    render_log(frame, rows[0], app);
    render_preview(frame, rows[1], app);
    render_hints(frame, rows[2], app);
}

fn render_log(frame: &mut Frame, area: Rect, app: &App) {
    if area.height < 1 {
        return;
    }
    let visible = area.height as usize;
    let skip = app.log.len().saturating_sub(visible);
    let mut lines: Vec<Line> = app
        .log
        .iter()
        .skip(skip)
        .map(|line| {
            Line::from(Span::styled(
                line.clone(),
                Style::new().fg(theme::log_color(line)),
            ))
        })
        .collect();
    lines.push(Line::from(Span::styled(
        format!(" ipc : {}", app.ipc_display),
        theme::dim(),
    )));
    frame.render_widget(Paragraph::new(lines), area);
}

fn render_preview(frame: &mut Frame, area: Rect, app: &App) {
    let text = match &app.preview {
        Some(preview) => {
            let age = preview.at.elapsed().as_secs();
            Line::from(vec![
                Span::styled(" diff ", Style::new().fg(theme::HIGHLIGHT)),
                Span::styled(
                    format!(
                        "{} · α={:.4} · rang 1 · il y a {age} s",
                        preview.tensor,
                        preview.alpha
                    ),
                    theme::base(),
                ),
            ])
        }
        None => Line::from(Span::styled(
            " aucun patch en attente — preview_patch via IPC",
            theme::dim(),
        )),
    };
    let block = Block::bordered().title(Span::styled(" Aperçu patch ", theme::dim()));
    let inner = block.inner(area);
    frame.render_widget(block, area);
    frame.render_widget(Paragraph::new(text), inner);
}

fn render_hints(frame: &mut Frame, area: Rect, app: &App) {
    frame.render_widget(
        Paragraph::new(Line::from(Span::styled(
            format!(
                " Ctrl+E exporte · Ctrl+P lance « {} » · Tab panneau · q quitter",
                app.agent_command
            ),
            theme::dim(),
        ))),
        area,
    );
}

#[cfg(test)]
mod tests {
    use crate::ui::inspector;

    #[test]
    fn sparkline_reuse_keeps_its_contract() {
        assert_eq!(inspector::block_bars(&[1.0, 2.0], 2), "▅█");
    }
}
