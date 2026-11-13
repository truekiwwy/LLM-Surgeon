//! The shell around the views: a title bar without system frame (logomark, name, five tabs,
//! the `[SCALPING]` / `[ADVANCE]` switch, window buttons; dragged, double-clicked, resized by
//! its four edges and four corners), the three zones of the body, a status bar and the
//! keyboard: `Ctrl+K` the palette, `Ctrl+1` to `Ctrl+5` the tabs.

use eframe::egui::{
    self, Align2, CursorIcon, Id, Key, Modifiers, Pos2, Rect, ResizeDirection, Sense, Ui, Vec2,
    ViewportCommand,
};

use super::components::{self as c, ButtonKind};
use super::i18n::{t, tf};
use super::icons::{Icon, State};
use super::layout::{self, Regions};
use super::theme::TOKENS as T;
use super::{logo, palette, views};
use crate::app::{Mode, SurgeonApp, Tab};
use crate::settings::Lang;

/// Below these window widths the title bar drops the name, then the edition.
const COMPACT_NAME: f32 = 1104.0;
const COMPACT_EDITION: f32 = 1280.0;

pub fn tab_label(tab: Tab) -> &'static str {
    match tab {
        Tab::Atelier => t("tab.atelier"),
        Tab::Graph => t("tab.graph"),
        Tab::Analyses => t("tab.analyses"),
        Tab::Agents => t("tab.agents"),
        Tab::Diagnostics => t("tab.diagnostics"),
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

    match app.tab {
        Tab::Atelier => views::atelier::show(ui, app, &regions),
        Tab::Graph => views::graph::show(ui, app, &regions),
        Tab::Analyses => views::analyses::show(ui, app, &regions),
        Tab::Agents => views::agents::show(ui, app, &regions),
        Tab::Diagnostics => views::diagnostics::show(ui, app, &regions),
    }
    splitters(ui, app, &regions);
    titlebar(ui, app, regions.titlebar);
    statusbar(ui, app, regions.statusbar);
    resize_edges(ui, screen);
    palette::show(&ctx, app);
}

fn keys(ctx: &egui::Context, app: &mut SurgeonApp) {
    let palette = ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, Key::K));
    if palette {
        app.palette.toggle();
    }
    let digits = [Key::Num1, Key::Num2, Key::Num3, Key::Num4, Key::Num5];
    for (tab, key) in Tab::ALL.into_iter().zip(digits) {
        if ctx.input_mut(|i| i.consume_key(Modifiers::COMMAND, key)) {
            app.tab = tab;
        }
    }
}

/// Column widths follow the gaps; they are saved once the pointer is released.
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

// ------------------------------------------------------------------ title bar

fn titlebar(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect) {
    let ctx = ui.ctx().clone();
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, T.color.window);
    c::hline(
        &painter,
        rect.x_range(),
        rect.bottom() - c::px(&painter),
        T.color.border,
    );

    // The empty bar moves the window; registered first, so every control sits above it.
    let drag = ui.interact(rect, Id::new("titlebar-drag"), Sense::click_and_drag());
    let maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
    if drag.double_clicked() {
        ctx.send_viewport_cmd(ViewportCommand::Maximized(!maximized));
    } else if drag.drag_started_by(egui::PointerButton::Primary) {
        ctx.send_viewport_cmd(ViewportCommand::StartDrag);
    }

    // Brand: the mark, the name, the edition.
    let mark = Rect::from_min_size(
        Pos2::new(
            rect.left() + T.space.l,
            rect.center().y - T.space.l / 2.0 - T.space.xs,
        ),
        Vec2::splat(T.size.icon + T.space.s),
    );
    logo::paint(&painter, mark);
    // A narrow window drops the edition, then the name: the mark, the tabs and the mode
    // switch always fit (960 is the smallest window).
    let mut x = mark.right();
    if rect.width() >= COMPACT_NAME {
        x = painter
            .text(
                Pos2::new(x + T.space.m, rect.center().y),
                Align2::LEFT_CENTER,
                t("app.name"),
                c::font_brand(),
                T.color.text,
            )
            .right();
    }
    if rect.width() >= COMPACT_EDITION {
        x = painter
            .text(
                Pos2::new(x + T.space.s, rect.center().y + 1.0),
                Align2::LEFT_CENTER,
                tf("app.edition", &[("version", &env!("CARGO_PKG_VERSION"))]),
                c::font_meta(),
                T.color.text_secondary,
            )
            .right();
    }

    // Tabs.
    x += T.space.xl;
    for tab in Tab::ALL {
        let label = tab_label(tab);
        let width = painter
            .layout_no_wrap(label.to_owned(), c::font_tab(), T.color.text)
            .size()
            .x
            + 2.0 * T.space.l;
        let r = Rect::from_min_size(Pos2::new(x, rect.top()), Vec2::new(width, rect.height()));
        if c::tab(ui, r, label, app.tab == tab).clicked() {
            app.tab = tab;
        }
        x += width;
    }

    // Right: the mode switch, then the window buttons.
    let buttons = 3.0 * T.size.window_button[0];
    let controls = Rect::from_min_max(
        Pos2::new(x + T.space.l, rect.top()),
        Pos2::new(rect.right() - buttons - T.space.l, rect.bottom()),
    );
    let mut mode = app.mode;
    c::region(ui, controls, "titlebar-mode", |ui| {
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.set_height(controls.height());
            c::segmented(
                ui,
                &mut mode,
                &[
                    c::Segment {
                        value: Mode::Scalping,
                        label: t("mode.scalping"),
                        tooltip: t("mode.scalping.tip"),
                    },
                    c::Segment {
                        value: Mode::Advance,
                        label: t("mode.advance"),
                        tooltip: t("mode.advance.tip"),
                    },
                ],
                c::font_mono(),
            );
        });
    });
    if mode != app.mode {
        app.set_mode(mode);
    }
    window_buttons(ui, rect, maximized);
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

// ------------------------------------------------------------------ status bar

fn statusbar(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect) {
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 0.0, T.color.window);
    c::hline(&painter, rect.x_range(), rect.top(), T.color.border);
    let inner = rect.shrink2(Vec2::new(T.space.l, 0.0));
    c::region(ui, inner, "statusbar", |ui| {
        ui.horizontal_centered(|ui| {
            ui.spacing_mut().item_spacing.x = T.space.m;
            context_status(ui, app);
            job_status(ui, app);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.spacing_mut().item_spacing.x = T.space.m;
                c::meta(ui, format!("v{}", env!("CARGO_PKG_VERSION")));
                let mut lang = app.desktop.lang();
                let picked = c::segmented_small(
                    ui,
                    &mut lang,
                    &[(Lang::En, "EN"), (Lang::Fr, "FR")],
                    t("status.language"),
                );
                if picked {
                    app.desktop.set_lang(lang);
                }
                if c::link(ui, &app.backend_label(), t("status.backend.tip")).clicked() {
                    app.dialog.open = true;
                }
                match app.hardware_label() {
                    Some(hw) => c::meta(ui, hw),
                    None => c::meta(ui, t("status.probing")),
                };
                c::keycap(ui, t("status.palette.key"));
                c::meta(ui, t("status.palette"));
            });
        });
    });
}

/// The project and its working copy, or what to do without one.
fn context_status(ui: &mut Ui, app: &mut SurgeonApp) {
    let project = app
        .studio
        .manager
        .as_ref()
        .map(|m| m.project.manifest.name.clone());
    let Some(project) = project else {
        c::status(ui, State::Idle, t("status.no_project"));
        return;
    };
    let entries = app.ledger().to_vec();
    match app.studio.active_copy(&entries) {
        Some(copy) => {
            let (state, word) = match copy.writes {
                0 => (State::Ok, t("copy.intact")),
                _ => (State::Warn, t("copy.modified")),
            };
            c::status(ui, state, word);
            c::mono_secondary(ui, format!("{project}:{}", copy.path));
        }
        None => {
            c::status(ui, State::Idle, t("copy.none"));
            c::mono_secondary(ui, project);
        }
    }
}

/// The running work: card or editor step, project jobs, the last message.
fn job_status(ui: &mut Ui, app: &mut SurgeonApp) {
    if let Some(job) = app.job.as_ref().filter(|j| !j.finished) {
        let label = job.labels.get(job.current).map_or("", String::as_str);
        c::status(
            ui,
            State::Active,
            tf(
                "status.step",
                &[
                    ("i", &(job.current + 1)),
                    ("n", &job.labels.len()),
                    ("label", &label),
                    ("s", &format!("{:.1}", job.started.elapsed().as_secs_f64())),
                ],
            ),
        );
        if c::button_if(
            ui,
            !job.cancelled(),
            t("action.cancel"),
            ButtonKind::Ghost,
            t("status.cancelling"),
        )
        .clicked()
        {
            job.cancel();
        }
        return;
    }
    let running = app.studio.manager.as_ref().map_or(0, |m| m.running.len());
    let queued = app.studio.manager.as_ref().map_or(0, |m| m.queue.len());
    if running + queued > 0 {
        c::status(
            ui,
            State::Active,
            tf("status.jobs", &[("running", &running), ("queued", &queued)]),
        );
        return;
    }
    let message = app
        .studio
        .message
        .clone()
        .or_else(|| app.desktop.message.clone().map(|m| (m, true)));
    if let Some((text, error)) = message {
        let state = if error { State::Error } else { State::Ok };
        c::status(ui, state, text);
    }
}
