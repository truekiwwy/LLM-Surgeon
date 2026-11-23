//! The shell around the views, laid out like an IDE without its editor: a title bar without
//! system frame (logomark and name; the running work, local only, the agent's connection,
//! the settings, the window buttons; dragged, double-clicked, resized by its four edges and
//! four corners), the activity bar (the agent, the 3D graph, the commands), the views, and
//! the keyboard: `Ctrl+K` the palette, `Ctrl+1` / `Ctrl+2` the views, `Ctrl+B` folds the
//! left column, `Ctrl+,` the settings.

use eframe::egui::{
    self, Align, Align2, CursorIcon, Id, Key, Layout, Modifiers, Pos2, Rect, ResizeDirection,
    Sense, Ui, Vec2, ViewportCommand,
};

use super::components as c;
use super::i18n::{t, tf};
use super::icons::{Icon, State};
use super::layout::{self, Regions};
use super::theme::TOKENS as T;
use super::{logo, palette, views};
use crate::app::{Section, SurgeonApp, Tab};

/// Below this window width the title bar drops the name.
const COMPACT_NAME: f32 = 1104.0;

pub fn tab_label(tab: Tab) -> &'static str {
    match tab {
        Tab::Agent => t("tab.agent"),
        Tab::Graph => t("tab.graph"),
    }
}

fn tab_icon(tab: Tab) -> Icon {
    match tab {
        Tab::Agent => Icon::Chat,
        Tab::Graph => Icon::Cube,
    }
}

/// One frame of the window.
pub fn show(ui: &mut Ui, app: &mut SurgeonApp) {
    let ctx = ui.ctx().clone();
    c::begin_dump(&ctx, true);
    keys(&ctx, app);
    let screen = ui.max_rect();
    ui.painter().rect_filled(screen, 0.0, T.color.window);
    let regions = layout::regions(screen, app.columns);
    layout::record(ui, &regions);

    // The first-run wizard covers the body; the title bar then shows only the brand.
    let wizard = app.desktop.wizard.is_some();
    if wizard {
        let body = Rect::from_min_max(regions.activity.min, regions.right.max);
        super::wizard::show(ui, app, body);
    } else {
        match (app.live, app.tab) {
            // Live mode splits the body itself: no columns to separate or drag.
            (true, _) => views::live::show(ui, app, &regions),
            (false, Tab::Agent) => views::agents::show(ui, app, &regions),
            (false, Tab::Graph) => views::graph::show(ui, app, &regions),
        }
        if !app.live {
            layout::separators(ui, &regions);
            splitters(ui, app, &regions);
        }
        activity_bar(ui, app, regions.activity);
    }
    titlebar(ui, app, regions.titlebar, !wizard);
    resize_edges(ui, screen);
    views::system::dialog(&ctx, app);
    views::radar::dialog(&ctx, app);
    views::vector::dialog(&ctx, app);
    palette::show(&ctx, app);
}

fn keys(ctx: &egui::Context, app: &mut SurgeonApp) {
    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::K)) {
        app.palette.toggle();
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::B)) {
        app.toggle_left();
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::Comma)) {
        app.settings = Some(app.settings.unwrap_or(Section::General));
    }
    for (tab, key) in Tab::ALL.into_iter().zip([Key::Num1, Key::Num2]) {
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, key)) {
            app.tab = tab;
            app.live = false;
        }
    }
}

/// Column widths follow the lines; they are saved once the pointer is released.
fn splitters(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    let dirty = Id::new("columns-dirty");
    if layout::splitters(ui, regions, &mut app.columns) {
        ui.ctx().data_mut(|d| d.insert_temp(dirty, true));
    }
    let pending = ui
        .ctx()
        .data(|d| d.get_temp::<bool>(dirty))
        .unwrap_or(false);
    if pending && !ui.input(|i| i.pointer.any_down()) {
        ui.ctx().data_mut(|d| d.remove::<bool>(dirty));
        app.columns_changed();
    }
}

// ------------------------------------------------------------------ activity bar

/// The views on top (the active one again folds the left column), the commands at the
/// bottom. The agent carries the count of what waits for the user.
fn activity_bar(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect) {
    let side = T.size.activity;
    let cell = |k: f32| Rect::from_min_size(rect.min + Vec2::new(0.0, k * side), Vec2::splat(side));
    let pending = app.agents.pending();
    for (k, tab) in Tab::ALL.into_iter().enumerate() {
        let badge = if tab == Tab::Agent { pending } else { 0 };
        let tip = tab_label(tab);
        let active = app.tab == tab && !app.live;
        if c::activity(ui, cell(k as f32), tab_icon(tab), active, tip, badge).clicked() {
            match active {
                true => app.toggle_left(),
                false => {
                    app.tab = tab;
                    app.live = false;
                }
            }
        }
    }
    let bottom = Rect::from_min_size(
        Pos2::new(rect.left(), rect.bottom() - side),
        Vec2::splat(side),
    );
    let tip = tf("activity.commands", &[("key", &t("status.palette.key"))]);
    if c::activity(ui, bottom, Icon::Search, app.palette.open, &tip, 0).clicked() {
        app.palette.toggle();
    }
}

// ------------------------------------------------------------------ title bar

fn titlebar(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect, controls: bool) {
    let ctx = ui.ctx().clone();
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, T.color.window);
    c::hline(
        &painter,
        rect.x_range(),
        rect.bottom() - c::px(&painter),
        T.color.separator,
    );

    // The empty bar moves the window; registered first, so every control sits above it.
    let drag = ui.interact(rect, Id::new("titlebar-drag"), Sense::click_and_drag());
    let maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
    if drag.double_clicked() {
        ctx.send_viewport_cmd(ViewportCommand::Maximized(!maximized));
    } else if drag.drag_started_by(egui::PointerButton::Primary) {
        ctx.send_viewport_cmd(ViewportCommand::StartDrag);
    }

    // Brand: the mark over the activity bar, then the name.
    let mark = Rect::from_center_size(
        Pos2::new(rect.left() + T.size.activity / 2.0, rect.center().y),
        Vec2::splat(T.size.icon + T.space.s),
    );
    logo::paint(&painter, mark);
    let mut left = mark.right() + T.space.m;
    if rect.width() >= COMPACT_NAME {
        left = painter
            .text(
                Pos2::new(left, rect.center().y),
                Align2::LEFT_CENTER,
                t("app.name"),
                c::font_brand(),
                T.color.text,
            )
            .right()
            + T.space.l;
    }

    window_buttons(ui, rect, maximized);
    if !controls {
        return;
    }
    let [w, _] = T.size.window_button;
    let area = Rect::from_min_max(
        Pos2::new(left, rect.top()),
        Pos2::new(rect.right() - 3.0 * w - T.space.s, rect.bottom()),
    );
    c::region(ui, area, "titlebar-controls", |ui| {
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = T.space.s;
            if c::icon_button(ui, Icon::Gear, t("settings.open")).clicked() {
                app.settings = Some(app.settings.unwrap_or(Section::General));
            }
            c::switch(
                ui,
                Icon::Play,
                t("live.toggle"),
                &mut app.live,
                t("live.toggle.tip"),
            );
            if app.agents.project.is_some() {
                views::agents::picker(ui, app, T.size.left);
            }
            let local = app.desktop.settings.local_only;
            let local_seg = c::Seg {
                text: t("status.local"),
                state: if local { State::Ok } else { State::Idle },
                tip: t("policy.local_only.tip"),
                max: T.size.left,
                on: Some(local),
            };
            let flipped = match work_status(app) {
                Some((state, text, tip)) => {
                    let work = c::Seg {
                        text: &text,
                        state,
                        tip: &tip,
                        max: T.size.left,
                        on: None,
                    };
                    c::capsule(ui, &[work, local_seg])[1].clicked()
                }
                None => c::capsule(ui, &[local_seg])[0].clicked(),
            };
            if flipped {
                app.set_local_only(!local);
            }
        });
    });
}

fn window_buttons(ui: &mut Ui, bar: Rect, maximized: bool) {
    let [w, h] = T.size.window_button;
    let items = [
        (
            Icon::Minimize,
            t("window.minimize"),
            ViewportCommand::Minimized(true),
        ),
        (
            if maximized {
                Icon::Restore
            } else {
                Icon::Maximize
            },
            if maximized {
                t("window.restore")
            } else {
                t("window.maximize")
            },
            ViewportCommand::Maximized(!maximized),
        ),
        (Icon::Close, t("window.close"), ViewportCommand::Close),
    ];
    let mut x = bar.right() - 3.0 * w;
    for (k, (icon, tooltip, command)) in items.into_iter().enumerate() {
        let rect = Rect::from_min_size(Pos2::new(x, bar.top()), Vec2::new(w, h));
        x += w;
        if c::window_button(ui, rect, icon, tooltip, k == 2).clicked() {
            ui.ctx().send_viewport_cmd(command);
        }
    }
}

/// Four edges and four corners resize the window (none while maximized).
fn resize_edges(ui: &mut Ui, screen: Rect) {
    let ctx = ui.ctx().clone();
    if ctx.input(|i| i.viewport().maximized.unwrap_or(false)) {
        return;
    }
    let e = T.space.xs;
    let k = T.space.s;
    let (l, r, t_, b) = (screen.left(), screen.right(), screen.top(), screen.bottom());
    use ResizeDirection as D;
    let zones = [
        (
            Rect::from_min_max(Pos2::new(l + k, t_), Pos2::new(r - k, t_ + e)),
            D::North,
            CursorIcon::ResizeNorth,
        ),
        (
            Rect::from_min_max(Pos2::new(l + k, b - e), Pos2::new(r - k, b)),
            D::South,
            CursorIcon::ResizeSouth,
        ),
        (
            Rect::from_min_max(Pos2::new(l, t_ + k), Pos2::new(l + e, b - k)),
            D::West,
            CursorIcon::ResizeWest,
        ),
        (
            Rect::from_min_max(Pos2::new(r - e, t_ + k), Pos2::new(r, b - k)),
            D::East,
            CursorIcon::ResizeEast,
        ),
        (
            Rect::from_min_max(Pos2::new(l, t_), Pos2::new(l + k, t_ + k)),
            D::NorthWest,
            CursorIcon::ResizeNorthWest,
        ),
        (
            Rect::from_min_max(Pos2::new(r - k, t_), Pos2::new(r, t_ + k)),
            D::NorthEast,
            CursorIcon::ResizeNorthEast,
        ),
        (
            Rect::from_min_max(Pos2::new(l, b - k), Pos2::new(l + k, b)),
            D::SouthWest,
            CursorIcon::ResizeSouthWest,
        ),
        (
            Rect::from_min_max(Pos2::new(r - k, b - k), Pos2::new(r, b)),
            D::SouthEast,
            CursorIcon::ResizeSouthEast,
        ),
    ];
    for (n, (rect, direction, cursor)) in zones.into_iter().enumerate() {
        let response = ui
            .interact(rect, Id::new(("resize-edge", n)), Sense::drag())
            .on_hover_cursor(cursor);
        if response.drag_started_by(egui::PointerButton::Primary) {
            ctx.send_viewport_cmd(ViewportCommand::BeginResize(direction));
        }
    }
}

/// The first segment of the title bar's capsule, with its tooltip: an install, the
/// project's jobs, an error, else the open project and the state of its working copy.
/// `None` when no project is open and all is quiet.
fn work_status(app: &mut SurgeonApp) -> Option<(State, String, String)> {
    if let Some(p) = app.desktop.setup.progress().first() {
        let pct = p
            .fraction()
            .map_or(String::new(), |f| format!(" {:.0} %", f * 100.0));
        let text = tf("status.install", &[("label", &p.label), ("pct", &pct)]);
        return Some((State::Active, text.clone(), text));
    }
    let running = app.studio.manager.as_ref().map_or(0, |m| m.running.len());
    let queued = app.studio.manager.as_ref().map_or(0, |m| m.queue.len());
    if running + queued > 0 {
        let text = tf("status.jobs", &[("running", &running), ("queued", &queued)]);
        return Some((State::Active, text.clone(), text));
    }
    let error = match app.studio.message.clone() {
        Some((text, true)) => Some(text),
        _ => app.desktop.message.clone(),
    };
    if let Some(text) = error {
        return Some((State::Error, text.clone(), text));
    }
    let name = app.studio.manager.as_ref()?.project.manifest.name.clone();
    let entries = app.ledger().to_vec();
    Some(match app.studio.active_copy(&entries) {
        Some(copy) => {
            let (state, word) = match copy.writes {
                0 => (State::Ok, t("copy.intact")),
                _ => (State::Warn, t("copy.modified")),
            };
            (state, format!("{name} · {word}"), copy.path)
        }
        None => (
            State::Idle,
            format!("{name} · {}", t("copy.none")),
            t("copy.new.tip").to_owned(),
        ),
    })
}
