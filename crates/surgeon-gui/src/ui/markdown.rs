use eframe::egui::text::{LayoutJob, TextFormat};
use eframe::egui::{self, FontId, Label, ScrollArea, Stroke, Ui};

use super::components as c;
use super::theme::TOKENS as T;

#[derive(Debug, Clone, PartialEq)]
pub enum Block {
    Heading(u8, String),
    Para(String),
    Item {
        depth: usize,
        marker: String,
        text: String,
    },
    Quote(String),
    Code(String),
    Rule,
    Table(Vec<Vec<String>>),
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Style {
    pub bold: bool,
    pub italic: bool,
    pub code: bool,
    pub strike: bool,
}

fn is_rule(line: &str) -> bool {
    let marks: String = line.chars().filter(|c| !c.is_whitespace()).collect();
    marks.len() >= 3
        && ['-', '*', '_']
            .iter()
            .any(|m| marks.chars().all(|c| c == *m))
}

fn heading(line: &str) -> Option<(u8, &str)> {
    let level = line.chars().take_while(|c| *c == '#').count();
    let rest = &line[level..];
    ((1..=6).contains(&level) && rest.starts_with(' ')).then(|| (level as u8, rest.trim()))
}

fn list_item(line: &str) -> Option<(String, &str)> {
    for bullet in ["- ", "* ", "+ "] {
        if let Some(rest) = line.strip_prefix(bullet) {
            return Some(("\u{2022}".into(), rest.trim()));
        }
    }
    let digits = line.chars().take_while(char::is_ascii_digit).count();
    let rest = &line[digits..];
    match (digits, rest.strip_prefix(". ").or(rest.strip_prefix(") "))) {
        (1..=3, Some(text)) => Some((format!("{}.", &line[..digits]), text.trim())),
        _ => None,
    }
}

fn cells(line: &str) -> Vec<String> {
    let line = line.trim();
    let line = line.strip_prefix('|').unwrap_or(line);
    let line = line.strip_suffix('|').unwrap_or(line);
    line.split('|').map(|c| c.trim().to_owned()).collect()
}

fn is_table_rule(line: &str) -> bool {
    let row = cells(line);
    line.contains('-')
        && row.iter().all(|c| {
            let c = c.trim_matches(':');
            !c.is_empty() && c.chars().all(|ch| ch == '-')
        })
}

pub fn parse(text: &str) -> Vec<Block> {
    let lines: Vec<&str> = text.lines().collect();
    let mut out = Vec::new();
    let mut para: Vec<&str> = Vec::new();
    let flush = |para: &mut Vec<&str>, out: &mut Vec<Block>| {
        if !para.is_empty() {
            out.push(Block::Para(para.join("\n")));
            para.clear();
        }
    };
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        let indent = line.len() - line.trim_start().len();
        i += 1;
        if let Some(fence) = ["```", "~~~"].into_iter().find(|f| trimmed.starts_with(f)) {
            flush(&mut para, &mut out);
            let mut code = Vec::new();
            while i < lines.len() && !lines[i].trim_start().starts_with(fence) {
                code.push(lines[i]);
                i += 1;
            }
            i += 1;
            out.push(Block::Code(code.join("\n")));
            continue;
        }
        if trimmed.is_empty() {
            flush(&mut para, &mut out);
            continue;
        }
        if trimmed.starts_with('|') && lines.get(i).is_some_and(|l| is_table_rule(l)) {
            flush(&mut para, &mut out);
            let mut rows = vec![cells(trimmed)];
            i += 1;
            while i < lines.len() && lines[i].trim().starts_with('|') {
                rows.push(cells(lines[i]));
                i += 1;
            }
            out.push(Block::Table(rows));
            continue;
        }
        let block = if is_rule(trimmed) {
            Block::Rule
        } else if let Some((level, title)) = heading(trimmed) {
            Block::Heading(level, title.to_owned())
        } else if let Some((marker, text)) = list_item(trimmed) {
            Block::Item {
                depth: indent / 2,
                marker,
                text: text.to_owned(),
            }
        } else if let Some(quote) = trimmed.strip_prefix('>') {
            Block::Quote(quote.trim().to_owned())
        } else {
            para.push(trimmed);
            continue;
        };
        flush(&mut para, &mut out);
        out.push(block);
    }
    flush(&mut para, &mut out);
    out
}

pub fn spans(text: &str) -> Vec<(String, Style)> {
    let mut out: Vec<(String, Style)> = Vec::new();
    let mut style = Style::default();
    let mut buf = String::new();
    let push = |buf: &mut String, style: Style, out: &mut Vec<(String, Style)>| {
        if !buf.is_empty() {
            out.push((std::mem::take(buf), style));
        }
    };
    let mut rest = text;
    while let Some(ch) = rest.chars().next() {
        let after = &rest[ch.len_utf8()..];
        let word_before = buf.chars().last().is_some_and(char::is_alphanumeric);
        if ch == '\\' {
            if let Some(next) = after.chars().next().filter(|c| c.is_ascii_punctuation()) {
                buf.push(next);
                rest = &after[next.len_utf8()..];
                continue;
            }
        }
        if ch == '`' {
            if let Some(end) = after.find('`') {
                push(&mut buf, style, &mut out);
                let code = Style {
                    code: true,
                    ..style
                };
                out.push((after[..end].to_owned(), code));
                rest = &after[end + 1..];
                continue;
            }
        }
        if ch == '[' {
            if let Some((label, tail)) = after.split_once("](") {
                if let Some(close) = tail.find(')').filter(|_| !label.contains(']')) {
                    buf.push_str(label);
                    rest = &tail[close + 1..];
                    continue;
                }
            }
        }
        let toggles = [("**", 0), ("__", 0), ("~~", 1), ("*", 2)];
        if let Some((mark, kind)) = toggles.iter().find(|(m, _)| rest.starts_with(m)) {
            let on = match kind {
                0 => style.bold,
                1 => style.strike,
                _ => style.italic,
            };
            let tail = &rest[mark.len()..];
            let opens = !on
                && tail.chars().next().is_some_and(|c| !c.is_whitespace())
                && tail.contains(mark)
                && !(*mark != "**" && *mark != "~~" && word_before);
            if on || opens {
                push(&mut buf, style, &mut out);
                match kind {
                    0 => style.bold = !on,
                    1 => style.strike = !on,
                    _ => style.italic = !on,
                }
                rest = tail;
                continue;
            }
        }
        buf.push(ch);
        rest = after;
    }
    push(&mut buf, style, &mut out);
    out
}

fn job(text: &str, font: FontId, color: egui::Color32) -> LayoutJob {
    let mut job = LayoutJob::default();
    for (part, style) in spans(text) {
        let font = match (style.code, style.bold) {
            (true, _) => c::font_mono(),
            (false, true) => FontId::new(font.size, super::theme::medium()),
            (false, false) => font.clone(),
        };
        let mut format = TextFormat {
            font_id: font,
            color,
            italics: style.italic,
            ..Default::default()
        };
        if style.code {
            format.background = T.color.surface_hover;
        }
        if style.strike {
            format.strikethrough = Stroke::new(1.0, color);
        }
        job.append(&part, 0.0, format);
    }
    job
}

fn paragraph(ui: &mut Ui, text: &str, font: FontId, color: egui::Color32) {
    ui.add(Label::new(job(text, font, color)).wrap());
}

pub fn show(ui: &mut Ui, text: &str) {
    let blocks = parse(text);
    for (k, block) in blocks.iter().enumerate() {
        if k > 0 {
            ui.add_space(T.space.xs);
        }
        match block {
            Block::Heading(level, title) => {
                let font = match level {
                    1 | 2 => c::font_title(),
                    _ => c::font_strong(),
                };
                ui.add_space(T.space.xs);
                paragraph(ui, title, font, T.color.text);
            }
            Block::Para(text) => paragraph(ui, text, c::font_body(), T.color.text),
            Block::Quote(text) => {
                ui.horizontal_top(|ui| {
                    ui.add_space(T.space.m);
                    paragraph(ui, text, c::font_body(), T.color.text_secondary);
                });
            }
            Block::Item {
                depth,
                marker,
                text,
            } => {
                ui.horizontal_top(|ui| {
                    ui.spacing_mut().item_spacing.x = T.space.xs;
                    ui.add_space(*depth as f32 * T.space.l);
                    ui.add_sized(
                        [T.space.l + T.space.xs, T.size.row * 0.75],
                        Label::new(
                            egui::RichText::new(marker)
                                .font(c::font_body())
                                .color(T.color.text_secondary),
                        ),
                    );
                    paragraph(ui, text, c::font_body(), T.color.text);
                });
            }
            Block::Code(code) => {
                c::inset_frame().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    c::mono_wrapped(ui, code.as_str(), T.color.text);
                });
            }
            Block::Rule => c::separator(ui),
            Block::Table(rows) => table(ui, rows, k),
        }
    }
}

fn table(ui: &mut Ui, rows: &[Vec<String>], salt: usize) {
    let columns = rows.iter().map(Vec::len).max().unwrap_or(0);
    ScrollArea::horizontal()
        .id_salt(("md-table", salt))
        .auto_shrink([false, true])
        .show(ui, |ui| {
            egui::Grid::new(("md-grid", salt))
                .striped(true)
                .spacing([T.space.l, T.space.xs])
                .show(ui, |ui| {
                    for (r, row) in rows.iter().enumerate() {
                        for k in 0..columns {
                            let cell = row.get(k).map_or("", String::as_str);
                            let text = match r {
                                0 => format!("**{cell}**"),
                                _ => cell.to_owned(),
                            };
                            ui.add(Label::new(job(&text, c::font_body(), T.color.text)));
                        }
                        ui.end_row();
                    }
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_are_read_line_by_line() {
        let text = "# Titre\nUne ligne\nla suite\n\n- un\n  - deux\n1. trois\n---\n> cite\n```\nlet x = 1;\n```\n| a | b |\n|---|:-:|\n| 1 | 2 |";
        assert_eq!(
            parse(text),
            vec![
                Block::Heading(1, "Titre".into()),
                Block::Para("Une ligne\nla suite".into()),
                Block::Item {
                    depth: 0,
                    marker: "\u{2022}".into(),
                    text: "un".into()
                },
                Block::Item {
                    depth: 1,
                    marker: "\u{2022}".into(),
                    text: "deux".into()
                },
                Block::Item {
                    depth: 0,
                    marker: "1.".into(),
                    text: "trois".into()
                },
                Block::Rule,
                Block::Quote("cite".into()),
                Block::Code("let x = 1;".into()),
                Block::Table(vec![
                    vec!["a".into(), "b".into()],
                    vec!["1".into(), "2".into()]
                ]),
            ]
        );
    }

    #[test]
    fn an_unclosed_fence_keeps_the_rest_as_code() {
        assert_eq!(parse("```\na\nb"), vec![Block::Code("a\nb".into())]);
    }

    #[test]
    fn inline_marks_become_styles_and_identifiers_stay_intact() {
        let bold = Style {
            bold: true,
            ..Style::default()
        };
        let code = Style {
            code: true,
            ..Style::default()
        };
        assert_eq!(
            spans(
                "Le **score** de `surgeon_check` sur [la doc](https://x.y) a 2*3 snake_case_name"
            ),
            vec![
                ("Le ".into(), Style::default()),
                ("score".into(), bold),
                (" de ".into(), Style::default()),
                ("surgeon_check".into(), code),
                (" sur la doc a 2*3 snake_case_name".into(), Style::default()),
            ]
        );
        let italic = spans("*penché* et ~~barré~~");
        assert!(italic[0].1.italic && italic[2].1.strike);
        assert_eq!(spans("\\*pas\\* italique")[0].0, "*pas* italique");
    }
}
