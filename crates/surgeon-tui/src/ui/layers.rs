//! Left panel: the layer tree of the filtered tensors, with the `/` search
//! and the `r` role cycle feeding it.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Paragraph};
use ratatui::Frame;
use std::time::Duration;

use crate::app::{App, Focus};
use crate::model::roles::Role;
use crate::ui::theme;

pub fn render(frame: &mut Frame, area: Rect, app: &App) {
    let focused = app.focus == Focus::Layers;
    let block = Block::bordered()
        .title(Span::styled(
            format!(" Couches ({}) ", app.filtered.len()),
            Style::new().fg(if focused {
                theme::BORDER_FOCUS
            } else {
                theme::ACCENT
            }),
        ))
        .border_style(if focused {
            theme::focused_border()
        } else {
            theme::border()
        });
    let inner = block.inner(area);
    frame.render_widget(block, area);
    if inner.height < 1 || app.filtered.is_empty() {
        frame.render_widget(Paragraph::new(" aucun tenseur").style(theme::dim()), inner);
        return;
    }
    let visible = inner.height as usize;
    let start = app.sel.saturating_sub(visible / 2);
    let start = start.min(
        app.filtered
            .len()
            .saturating_sub(visible.min(app.filtered.len())),
    );
    let now = std::time::Instant::now();
    for (row, position) in (start..app.filtered.len()).take(visible).enumerate() {
        let index = app.filtered[position];
        let entry = &app.index.tensors[index];
        let selected = position == app.sel;
        let highlighted = app.highlight.is_some_and(|(layer, at)| {
            entry.layer == Some(layer) && now.duration_since(at) < Duration::from_secs(5)
        });
        let mut spans = Vec::new();
        spans.push(Span::styled(
            if highlighted {
                " ◉ "
            } else if selected {
                " ▸ "
            } else {
                "   "
            },
            Style::new().fg(if highlighted {
                theme::HIGHLIGHT
            } else {
                theme::DIM
            }),
        ));
        spans.push(Span::styled(
            format!("{:>4} ", entry.layer_text()),
            Style::new().fg(if entry.layer.is_some() {
                theme::DIM
            } else {
                theme::BORDER
            }),
        ));
        let role_tag = match entry.moe {
            crate::model::MoeKind::Fused => "f",
            crate::model::MoeKind::Shared => "s",
            _ if entry.expert.is_some() => "e",
            _ => "·",
        };
        spans.push(Span::styled(
            format!("{}{role_tag} ", short_role(entry.role)),
            Style::new().fg(theme::role_color(entry.role)),
        ));
        spans.push(Span::styled(
            truncate_tail(&entry.name, inner.width.saturating_sub(14) as usize),
            if selected {
                Style::new().fg(theme::TEXT).bg(theme::SELECTED_BG)
            } else {
                theme::base()
            },
        ));
        frame.render_widget(
            Paragraph::new(Line::from(spans)),
            Rect {
                y: inner.y + row as u16,
                ..inner
            },
        );
    }
}

fn short_role(role: Role) -> &'static str {
    match role {
        Role::Embed => "emb",
        Role::AttnOut => "att",
        Role::MlpGate => "gat",
        Role::MlpUp => "up ",
        Role::MlpOut => "dwn",
        Role::Router => "rtr",
        Role::Other => "oth",
    }
}

/// Cuts in the middle to keep the tail (the discriminating part) visible.
fn truncate_tail(name: &str, width: usize) -> String {
    use unicode_width::UnicodeWidthStr;
    if name.width() <= width {
        return name.to_owned();
    }
    let keep = width.saturating_sub(1);
    let head = keep / 2;
    let tail = keep - head;
    let mut out = String::new();
    let mut used = 0usize;
    let mut chars = name.chars();
    for ch in chars.by_ref() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if used + w > head {
            break;
        }
        out.push(ch);
        used += w;
    }
    out.push('…');
    let tail_text: String = name
        .chars()
        .rev()
        .take(tail * 2)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let mut tail_used = 0usize;
    let mut tail_out = String::new();
    for ch in tail_text.chars().rev() {
        let w = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
        if tail_used + w > tail {
            break;
        }
        tail_out.insert(0, ch);
        tail_used += w;
    }
    out.push_str(&tail_out);
    out
}

/// The current role filter, for the filter bar.
pub fn filter_label(app: &App) -> String {
    app.role_filter
        .map_or_else(|| "tous".to_owned(), |r| r.label().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_keeps_the_tail() {
        let cut = truncate_tail("model.layers.14.mlp.down_proj.weight", 20);
        assert!(cut.contains('…'));
        assert!(cut.ends_with("weight") || cut.ends_with("…"));
        assert!(truncate_tail("short", 20) == "short");
    }
}
