use eframe::egui::{self, Id, ScrollArea, Sense, Ui, Vec2};

use super::super::components::{self as c, ButtonKind};
use super::super::i18n::{t, tf};
use super::super::icons::Icon;
use super::super::theme::TOKENS as T;
use crate::app::SurgeonApp;
use crate::studio::vectors::{VectorMode, COMPONENTS, METHODS};

const SIZE: [f32; 2] = [720.0, 704.0];

pub fn open(app: &mut SurgeonApp) {
    match app.studio.manager.is_some() {
        true => app.vector.open = true,
        false => app.studio.message = Some((t("vector.none").into(), true)),
    }
}

pub fn dialog(ctx: &egui::Context, app: &mut SurgeonApp) {
    if !app.vector.open {
        return;
    }
    if app.studio.manager.is_none() {
        app.vector.open = false;
        return;
    }
    let screen = ctx.content_rect();
    let size = Vec2::new(
        SIZE[0].min(screen.width() - 2.0 * T.size.activity),
        SIZE[1].min(screen.height() - 2.0 * T.size.titlebar),
    );
    let mut close = false;
    egui::Modal::new(Id::new("vector"))
        .backdrop_color(T.color.scrim)
        .frame(c::popup_frame().inner_margin(0.0))
        .show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
            c::Panel::new("vector")
                .bare()
                .title(t("vector.title"))
                .show_with(
                    ui,
                    rect,
                    &mut close,
                    |ui, close| {
                        if c::icon_button(ui, Icon::Close, t("vector.close")).clicked() {
                            *close = true;
                        }
                    },
                    |ui, close| body(ui, app, close),
                );
        });
    if close {
        app.vector.open = false;
    }
}

fn body(ui: &mut Ui, app: &mut SurgeonApp, close: &mut bool) {
    ScrollArea::vertical()
        .id_salt("vector-form")
        .auto_shrink(false)
        .show(ui, |ui| {
            ui.add_space(T.space.s);
            let form = &mut app.vector;
            let modes = [
                (VectorMode::Sweep, t("vector.mode.sweep")),
                (VectorMode::Trace, t("vector.mode.trace")),
            ];
            c::segmented_small(ui, &mut form.mode, &modes, t("vector.mode.tip"));
            ui.add_space(T.space.xs);
            let (hint, positive, negative) = match form.mode {
                VectorMode::Sweep => (
                    t("vector.sweep.hint"),
                    t("vector.positive"),
                    t("vector.negative"),
                ),
                VectorMode::Trace => (
                    t("vector.trace.hint"),
                    t("vector.clean"),
                    t("vector.corrupted"),
                ),
            };
            c::secondary(ui, hint);
            ui.add_space(T.space.s);
            match form.mode {
                VectorMode::Sweep => {
                    c::meta(ui, t("vector.method"));
                    let methods = [(METHODS[0], "mean_diff"), (METHODS[1], "pca (svd)")];
                    c::segmented_small(ui, &mut form.method, &methods, t("vector.method.tip"));
                }
                VectorMode::Trace => {
                    c::meta(ui, t("vector.component"));
                    let components = COMPONENTS.map(|name| (name, name));
                    c::segmented_small(
                        ui,
                        &mut form.component,
                        &components,
                        t("vector.component.tip"),
                    );
                }
            }
            ui.add_space(T.space.s);
            c::meta(ui, positive);
            c::field_multiline(ui, &mut form.positive, t("vector.positive.hint"), 4);
            ui.add_space(T.space.xs);
            c::meta(ui, negative);
            c::field_multiline(ui, &mut form.negative, t("vector.negative.hint"), 4);
            ui.add_space(T.space.s);
            let width = ui.available_width() / 2.0;
            ui.horizontal(|ui| {
                c::meta(ui, t("vector.layers"));
                c::field(ui, &mut form.layers, "all, 8-20, 3,5,7", width);
            });
            if form.mode == VectorMode::Sweep {
                ui.horizontal(|ui| {
                    c::meta(ui, t("vector.name"));
                    c::field(ui, &mut form.name, "refusal_sweep", width);
                });
                c::mono_secondary(ui, form.out().as_str());
            }
            ui.add_space(T.space.m);
            if c::button(ui, t("vector.submit"), ButtonKind::Primary).clicked() {
                *close = submit(app);
            }
        });
}

fn submit(app: &mut SurgeonApp) -> bool {
    let Some(manager) = app.studio.manager.as_mut() else {
        return false;
    };
    let Some(model) = manager
        .project
        .manifest
        .source
        .as_ref()
        .map(|s| s.path.clone())
    else {
        app.studio.message = Some((t("vector.none").into(), true));
        return false;
    };
    let submitted = app.vector.job(&model).and_then(|job| manager.submit(job));
    match submitted {
        Ok(id) => {
            app.studio.message = Some((tf("vector.submitted", &[("id", &id)]), false));
            true
        }
        Err(e) => {
            app.studio.message = Some((e, true));
            false
        }
    }
}
