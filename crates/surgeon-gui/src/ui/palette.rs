//! The command palette (`Ctrl+K`): every action of the window by name, filtered as you type,
//! walked with the arrows, run with Enter. A query that looks like a path offers to open it.

use eframe::egui::{self, Id, Key, Order, Pos2, Rect, Vec2};

use super::components as c;
use super::i18n::{t, tf};
use super::shell::tab_label;
use super::theme::TOKENS as T;
use crate::app::{Mode, Section, SurgeonApp, Tab};
use crate::settings::Lang;

/// On the 8-point grid, like every layout dimension.
const WIDTH: f32 = 560.0;
/// Commands shown at once; typing narrows the rest.
const ROWS: usize = 10;

#[derive(Default)]
pub struct Palette {
    pub open: bool,
    pub query: String,
    pub cursor: usize,
}

impl Palette {
    pub fn toggle(&mut self) {
        self.open = !self.open;
        self.query.clear();
        self.cursor = 0;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    Tab(Tab),
    /// A new session of this kind.
    Session(Mode),
    Open(String),
    Settings,
    Backend,
    Wizard,
    Language(Lang),
    ResetCamera,
    NewCopy,
    ExportLedger,
    CheckUpdate,
    /// The interference radar of the open project's direction files.
    Radar,
}

fn label(command: &Command) -> String {
    match command {
        Command::Tab(tab) => tf("cmd.tab", &[("tab", &tab_label(*tab))]),
        Command::Session(Mode::Advance) => t("cmd.session.advance").into(),
        Command::Session(Mode::Scalping) => t("cmd.session.scalping").into(),
        Command::Open(path) => tf("cmd.open", &[("path", path)]),
        Command::Settings => t("cmd.settings").into(),
        Command::Backend => t("cmd.backend").into(),
        Command::Wizard => t("cmd.wizard").into(),
        Command::Language(Lang::Fr) => t("cmd.lang.fr").into(),
        Command::Language(Lang::En) => t("cmd.lang.en").into(),
        Command::ResetCamera => t("cmd.camera").into(),
        Command::NewCopy => t("cmd.copy").into(),
        Command::ExportLedger => t("cmd.ledger").into(),
        Command::CheckUpdate => t("cmd.update").into(),
        Command::Radar => t("cmd.radar").into(),
    }
}

/// The commands offered for `query`, in a stable order.
pub fn commands(query: &str) -> Vec<(Command, String)> {
    let mut all: Vec<Command> = Tab::ALL.into_iter().map(Command::Tab).collect();
    all.extend([
        Command::Session(Mode::Advance),
        Command::Session(Mode::Scalping),
        Command::NewCopy,
        Command::ExportLedger,
        Command::Radar,
        Command::ResetCamera,
        Command::Settings,
        Command::Backend,
        Command::Wizard,
        Command::CheckUpdate,
        Command::Language(Lang::Fr),
        Command::Language(Lang::En),
    ]);
    let q = query.trim();
    let needle = q.to_lowercase();
    let mut out: Vec<(Command, String)> = all
        .into_iter()
        .map(|cmd| {
            let text = label(&cmd);
            (cmd, text)
        })
        .filter(|(_, text)| needle.is_empty() || text.to_lowercase().contains(&needle))
        .collect();
    let looks_like_path = q.contains(['/', '\\']) || q.ends_with(".json") || q.ends_with(".ndjson");
    if looks_like_path {
        let cmd = Command::Open(q.to_owned());
        let text = label(&cmd);
        out.insert(0, (cmd, text));
    }
    out
}

fn run(ctx: &egui::Context, app: &mut SurgeonApp, command: Command) {
    match command {
        Command::Tab(tab) => {
            app.tab = tab;
            app.live = false;
        }
        Command::Session(mode) => {
            app.set_mode(mode);
            app.agents.new_conversation();
            app.tab = Tab::Agent;
        }
        Command::Settings => app.settings = Some(Section::General),
        Command::Open(path) => app
            .desktop
            .pending
            .push(crate::cli::classify(std::path::Path::new(&path))),
        Command::Backend => app.dialog.open = true,
        Command::Wizard => app.desktop.open_wizard(),
        Command::Language(lang) => app.desktop.set_lang(lang),
        Command::ResetCamera => app.camera = crate::camera::Orbit::default(),
        Command::NewCopy => drop(app.studio.submit_copy()),
        Command::ExportLedger => app.studio.export_ledger(true),
        Command::CheckUpdate => {
            app.settings = Some(Section::Updates);
            app.desktop.check_update(ctx);
        }
        Command::Radar => match &app.studio.manager {
            Some(manager) => app.radar.open_on(&manager.project.dir.join("work")),
            None => app.studio.message = Some((t("radar.no_project").into(), true)),
        },
    }
}

pub fn show(ctx: &egui::Context, app: &mut SurgeonApp) {
    if !app.palette.open {
        return;
    }
    let list = commands(&app.palette.query);
    let (up, down, enter, escape) = ctx.input_mut(|i| {
        (
            i.consume_key(egui::Modifiers::NONE, Key::ArrowUp),
            i.consume_key(egui::Modifiers::NONE, Key::ArrowDown),
            i.consume_key(egui::Modifiers::NONE, Key::Enter),
            i.consume_key(egui::Modifiers::NONE, Key::Escape),
        )
    });
    let p = &mut app.palette;
    if down {
        p.cursor = (p.cursor + 1).min(list.len().min(ROWS).saturating_sub(1));
    }
    if up {
        p.cursor = p.cursor.saturating_sub(1);
    }
    p.cursor = p.cursor.min(list.len().min(ROWS).saturating_sub(1));
    if escape {
        p.toggle();
        return;
    }

    let screen = ctx.content_rect();
    let scrim = egui::LayerId::new(Order::Foreground, Id::new("palette-scrim"));
    ctx.layer_painter(scrim)
        .rect_filled(screen, 0.0, T.color.scrim);
    let width = WIDTH;
    let rows = list.len().min(ROWS) as f32;
    let height = T.size.control + T.space.s + rows * T.size.row + 2.0 * T.space.s;
    let rect = Rect::from_min_size(
        Pos2::new(
            screen.center().x - width / 2.0,
            screen.top() + 2.0 * T.size.titlebar,
        ),
        Vec2::new(width, height),
    );
    let mut picked = enter
        .then(|| list.get(app.palette.cursor).map(|(c, _)| c.clone()))
        .flatten();
    let mut close = false;
    egui::Area::new(Id::new("palette"))
        .order(Order::Tooltip)
        .fixed_pos(rect.min)
        .show(ctx, |ui| {
            c::popup_frame().show(ui, |ui| {
                ui.set_width(width - 2.0 * T.space.s);
                let field = c::search(
                    ui,
                    Id::new("palette-query"),
                    &mut app.palette.query,
                    t("palette.hint"),
                );
                field.request_focus();
                if field.changed() {
                    app.palette.cursor = 0;
                }
                ui.add_space(T.space.s);
                if list.is_empty() {
                    c::secondary(ui, t("palette.empty"));
                }
                for (k, (command, text)) in list.iter().enumerate().take(ROWS) {
                    let response = c::row(ui, k == app.palette.cursor, text, |ui| {
                        c::label(ui, text.as_str());
                    });
                    if response.clicked() {
                        picked = Some(command.clone());
                    }
                }
            });
        });
    // A click outside the palette closes it.
    if ctx.input(|i| i.pointer.any_click())
        && ctx
            .pointer_interact_pos()
            .is_some_and(|p| !rect.contains(p))
        && picked.is_none()
    {
        close = true;
    }
    if let Some(command) = picked {
        app.palette.toggle();
        run(ctx, app, command);
    } else if close {
        app.palette.toggle();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::i18n;

    #[test]
    fn the_query_filters_and_a_path_is_offered_first() {
        i18n::set_lang(Lang::En);
        let all = commands("");
        assert!(all.len() >= 12, "{}", all.len());
        let graph = commands("graph");
        assert!(graph.iter().any(|(c, _)| *c == Command::Tab(Tab::Graph)));
        assert!(graph.len() < all.len());
        let path = commands("runs/r1/out/graph.json");
        assert_eq!(path[0].0, Command::Open("runs/r1/out/graph.json".into()));
    }
}
