//! The two views and the sections of the settings dialog, and what they share: the left
//! column (the project, its working copy, the tensor tree) and number formats.

pub mod agents;
pub mod graph;
pub mod live;
pub mod radar;
pub mod results;
pub mod system;
pub mod vector;

use std::collections::BTreeMap;

use eframe::egui::{self, Rect, ScrollArea, Sense, Ui, Vec2};

use super::components::{self as c, ButtonKind};
use super::i18n::{t, tf};
use super::icons::{self, Icon};
use super::theme::{self, TOKENS as T};
use crate::app::SurgeonApp;
use crate::viewport;

// ------------------------------------------------------------------ numbers

/// Fixed precision in a readable range, scientific outside.
pub fn num(v: f64) -> String {
    match v.abs() {
        a if a != 0.0 && !(1e-3..1e5).contains(&a) => format!("{v:.3e}"),
        _ => format!("{v:.4}"),
    }
}

pub fn opt(v: Option<f64>) -> String {
    v.map_or("-".into(), num)
}

/// Seconds as `42 s`, `1 min 05 s`.
pub fn duration(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s} s"),
        s => format!("{} min {:02} s", s / 60, s % 60),
    }
}

/// How long ago, in the largest whole unit: `now`, `5 min`, `3 h`, `2 d`.
pub fn age(secs: u64) -> String {
    match secs {
        s if s < 60 => t("age.now").into(),
        s if s < 3600 => tf("age.minutes", &[("n", &(s / 60))]),
        s if s < 86_400 => tf("age.hours", &[("n", &(s / 3600))]),
        s => tf("age.days", &[("n", &(s / 86_400))]),
    }
}

// ------------------------------------------------------------------ explorer

/// The left column of the 3D view: the project and its copy, then the tensor tree.
pub fn explorer(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect) {
    left_column(ui, app, rect, "explorer", t("panel.explorer"), |ui, app| {
        context(ui, app);
        c::separator(ui);
        tree(ui, app);
    });
}

/// The left column of a view: a panel whose header folds it (`Ctrl+B` too; the activity
/// bar unfolds it). Folded, it has no width and draws nothing.
pub fn left_column(
    ui: &mut Ui,
    app: &mut SurgeonApp,
    rect: Rect,
    name: &'static str,
    title: &str,
    body: impl FnOnce(&mut Ui, &mut SurgeonApp),
) {
    if app.columns.collapsed {
        return;
    }
    c::Panel::new(name).title(title).show_with(
        ui,
        rect,
        app,
        |ui, app| {
            if c::icon_button(ui, Icon::ChevronLeft, t("left.collapse")).clicked() {
                app.toggle_left();
            }
        },
        body,
    );
}

/// The project picker of `width`, when there is a project to pick; `true` when a project is
/// open.
pub fn project_select(ui: &mut Ui, app: &mut SurgeonApp, width: f32) -> bool {
    let ctx = ui.ctx().clone();
    let current = app
        .studio
        .manager
        .as_ref()
        .map(|m| m.project.manifest.name.clone());
    let projects: Vec<(std::path::PathBuf, String)> = app
        .studio
        .projects
        .iter()
        .filter_map(|(dir, m)| m.as_ref().ok().map(|m| (dir.clone(), m.name.clone())))
        .collect();
    let shown = current.clone().unwrap_or_else(|| t("project.none").into());
    let mut open = None;
    if !projects.is_empty() {
        c::select(ui, "project-select", &shown, width, |ui| {
            for (dir, name) in &projects {
                let mut picked = current.clone();
                if c::option(ui, &mut picked, Some(name.clone()), name) {
                    open = Some(dir.clone());
                }
            }
        });
    }
    if let Some(dir) = open {
        app.studio.open_project(&ctx, &dir);
    }
    current.is_some()
}

/// The source model of a new project: the system picker (a weights file or a model folder)
/// and the path it gave. Outside Windows, the path typed by hand.
fn source_picker(ui: &mut Ui, app: &mut SurgeonApp, width: f32) {
    if !cfg!(windows) {
        c::field_mono(
            ui,
            &mut app.studio.new_source,
            t("project.source.hint"),
            width,
        );
        return;
    }
    let ctx = ui.ctx().clone();
    let free = !app.studio.picking();
    ui.horizontal(|ui| {
        let choices = [
            (false, t("project.source.file")),
            (true, t("project.source.folder")),
        ];
        for (folder, text) in choices {
            let why = t("project.source.picking");
            if c::button_if(ui, free, text, ButtonKind::Secondary, why).clicked() {
                app.studio
                    .pick_source(&ctx, folder, t("project.source.title"));
            }
        }
    });
    ui.add_space(T.space.xs);
    let path = app.studio.new_source.trim().to_owned();
    match path.is_empty() {
        true => c::meta(ui, t("project.source.none")),
        false => c::mono_secondary(ui, path.as_str()).on_hover_text(path.as_str()),
    };
}

/// The project picker, or the creation form, and the working copy.
pub fn context(ui: &mut Ui, app: &mut SurgeonApp) {
    let ctx = ui.ctx().clone();
    let width = ui.available_width();
    if !project_select(ui, app, width) {
        ui.add_space(T.space.s);
        source_picker(ui, app, width);
        ui.add_space(T.space.m);
        c::field(ui, &mut app.studio.new_name, t("project.name.hint"), width);
        ui.add_space(T.space.m);
        let ready = !app.studio.new_name.trim().is_empty() && !app.studio.creating();
        let text = match app.studio.creating() {
            true => t("project.creating"),
            false => t("project.create"),
        };
        if c::button_if(
            ui,
            ready,
            text,
            ButtonKind::Primary,
            t("project.create.why"),
        )
        .clicked()
        {
            app.studio.create_project(&ctx);
        }
        return;
    }
    ui.add_space(T.space.s);
    let entries = app.ledger().to_vec();
    let copy = app.studio.active_copy(&entries);
    ui.horizontal(|ui| match &copy {
        Some(copy) => {
            let (state, word) = match copy.writes {
                0 => (icons::State::Ok, t("copy.intact")),
                _ => (icons::State::Warn, t("copy.modified")),
            };
            c::status(ui, state, word);
            c::mono_secondary(ui, copy.path.as_str());
        }
        None => {
            c::status(ui, icons::State::Idle, t("copy.none"));
        }
    });
    match &app.studio.source_lock {
        Some(Ok(held)) => {
            c::status(
                ui,
                icons::State::Ok,
                tf("source.locked", &[("n", &held.files)]),
            )
            .on_hover_text(t("source.locked.tip"));
        }
        Some(Err(e)) => {
            c::status(ui, icons::State::Warn, t("source.unlocked")).on_hover_text(e.as_str());
        }
        None => {}
    }
    ui.add_space(T.space.s);
    ui.horizontal_wrapped(|ui| {
        if c::button(ui, t("copy.new"), ButtonKind::Secondary)
            .on_hover_text(t("copy.new.tip"))
            .clicked()
        {
            app.studio.submit_copy();
        }
        if c::button(ui, t("vector.open"), ButtonKind::Secondary)
            .on_hover_text(t("vector.open.tip"))
            .clicked()
        {
            vector::open(app);
        }
    });
}

/// The tensors of the graph on screen, by layer; Up / Down walk it, Escape clears.
pub fn tree(ui: &mut Ui, app: &mut SurgeonApp) {
    c::search(
        ui,
        egui::Id::new("tree-search"),
        &mut app.filter,
        t("tree.search"),
    );
    ui.add_space(T.space.xs);
    ui.horizontal(|ui| {
        let names = [t("group.attn"), t("group.mlp"), t("group.embed")];
        for (g, name) in names.into_iter().enumerate() {
            c::checkbox(ui, &mut app.groups[g], name);
        }
    });
    ui.add_space(T.space.xs);
    let Some(step) = app.timeline.current() else {
        c::empty(ui, Icon::Cube, t("tree.empty"), t("tree.empty.hint"));
        return;
    };
    let visible = app.visible(&step.report);
    let mut layers: BTreeMap<Option<usize>, Vec<usize>> = BTreeMap::new();
    for (i, node) in step.report.nodes.iter().enumerate() {
        if visible[i] {
            layers.entry(node.layer).or_default().push(i);
        }
    }
    let shown = visible.iter().filter(|v| **v).count();
    c::meta(
        ui,
        tf(
            "tree.count",
            &[("shown", &shown), ("total", &visible.len())],
        ),
    );
    let order: Vec<usize> = layers.values().flatten().copied().collect();
    let before = app.selected_index();
    let ctx = ui.ctx().clone();
    app.arrow_selection(&ctx, &order);
    let Some(step) = app.timeline.current() else {
        return;
    };
    let selected = app.selected_index();
    let moved = selected != before;
    let filtering = !app.filter.trim().is_empty();
    let run_states = app.run_view().map(|_| app.layer.states.clone());
    let mut picked: Option<Option<String>> = None;
    let mut toggled = None;
    ScrollArea::vertical()
        .id_salt("tree")
        .auto_shrink(false)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = 0.0;
            for (layer, nodes) in &layers {
                let open = filtering || app.open_layers.contains(layer);
                let title = match layer {
                    Some(l) => format!("layers.{l}"),
                    None => t("tree.outside").to_owned(),
                };
                let header = c::row(ui, false, &title, |ui| {
                    let icon = if open {
                        Icon::ChevronDown
                    } else {
                        Icon::ChevronRight
                    };
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(T.size.icon), Sense::hover());
                    icons::paint(ui.painter(), r, icon, T.color.text_secondary);
                    c::mono(ui, title.as_str());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        c::meta(ui, nodes.len().to_string());
                    });
                });
                if header.clicked() {
                    toggled = Some(*layer);
                }
                if !open {
                    continue;
                }
                for &i in nodes {
                    let node = &step.report.nodes[i];
                    let is_selected = Some(i) == selected;
                    let response = c::row(ui, is_selected, &node.tensor, |ui| {
                        ui.add_space(T.space.l);
                        let (r, _) =
                            ui.allocate_exact_size(Vec2::splat(T.size.icon), Sense::hover());
                        let shape = theme::role_shape(node.role);
                        let ink = if is_selected {
                            T.color.accent
                        } else {
                            T.color.text_secondary
                        };
                        ui.painter().add(icons::role_outline(
                            shape,
                            r.center(),
                            T.size.dot / 2.0 - 1.0,
                            egui::Stroke::new(T.stroke, ink),
                        ));
                        let mut text = viewport::role_label(node.role).to_owned();
                        if let Some(e) = node.expert {
                            text.push_str(&format!(" · e{e}"));
                        }
                        c::mono(ui, text);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if let Some(state) = run_states
                                .as_ref()
                                .and_then(|s| s.get(i).copied().flatten())
                            {
                                c::meta(ui, viewport::state_style(state).2);
                            } else if step.touched.contains_key(&node.id) {
                                c::meta(ui, t("tree.patched"));
                            }
                        });
                    });
                    let response = response.on_hover_text(&node.tensor);
                    if is_selected && moved {
                        response.scroll_to_me(None);
                    }
                    if response.clicked() {
                        picked = Some(Some(node.id.clone()));
                    }
                }
            }
        });
    if let Some(layer) = toggled {
        if !app.open_layers.remove(&layer) {
            app.open_layers.insert(layer);
        }
    }
    if let Some(picked) = picked {
        app.selected = picked;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use weight_surgeon::graph::output::GraphReport;
    use weight_surgeon::graph::stream::GraphPatch;

    pub fn report() -> GraphReport {
        let line = include_str!("../../../tests/fixtures/qwen2-mini.ndjson");
        match serde_json::from_str(line.trim()).unwrap() {
            GraphPatch::Snapshot { report } => *report,
            GraphPatch::Update { .. } => unreachable!(),
        }
    }

    #[test]
    fn numbers_switch_to_scientific_outside_the_readable_range() {
        assert_eq!(num(0.0), "0.0000");
        assert_eq!(num(12.5), "12.5000");
        assert_eq!(num(1.5e-5), "1.500e-5");
        assert_eq!(num(-2.5e6), "-2.500e6");
        assert_eq!(duration(42), "42 s");
        assert_eq!(duration(65), "1 min 05 s");
        crate::ui::i18n::set_lang(crate::settings::Lang::En);
        assert_eq!(age(59), "now");
        assert_eq!(age(3599), "59 min");
        assert_eq!(age(7200), "2 h");
        assert_eq!(age(3 * 86_400 + 5), "3 d");
    }
}
