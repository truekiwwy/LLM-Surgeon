//! The vector interference radar: the two direction files of the project's `work/`,
//! compared layer by layer before composing them. Read-only: it opens the vectors, writes
//! nothing, and never touches a model.

use std::path::{Path, PathBuf};

use eframe::egui::{self, Id, ScrollArea, Sense, Ui, Vec2};

use super::super::components::{self as c, ButtonKind};
use super::super::i18n::{t, tf};
use super::super::icons::{Icon, State};
use super::super::theme::TOKENS as T;
use crate::app::SurgeonApp;
use crate::studio::vector_radar::{self, LayerRow, RadarReport, Risk};

/// The largest the panel gets.
const SIZE: [f32; 2] = [820.0, 640.0];

/// The panel state: the files found in `work/`, the two selections and their comparison.
#[derive(Debug, Default)]
pub struct Radar {
    pub open: bool,
    pub files: Vec<PathBuf>,
    pub a: Option<PathBuf>,
    pub b: Option<PathBuf>,
    pub report: Option<Result<RadarReport, String>>,
    computed_for: Option<(PathBuf, PathBuf)>,
}

impl Radar {
    /// Opens the panel and scans the `work/` folder of the project.
    pub fn open_on(&mut self, work: &Path) {
        self.open = true;
        self.refresh(work);
    }

    /// Rescans `work/`, keeps the selections that still exist, then compares.
    pub fn refresh(&mut self, work: &Path) {
        self.files = vector_radar::work_files(work);
        if self
            .a
            .as_ref()
            .is_none_or(|path| !self.files.contains(path))
        {
            self.a = self.files.first().cloned();
        }
        if self
            .b
            .as_ref()
            .is_none_or(|path| !self.files.contains(path))
        {
            self.b = self.files.get(1).or_else(|| self.files.first()).cloned();
        }
        self.computed_for = None;
        self.compute();
    }

    /// Compares the two selected files, once per selection.
    fn compute(&mut self) {
        let (Some(a), Some(b)) = (self.a.clone(), self.b.clone()) else {
            self.report = None;
            self.computed_for = None;
            return;
        };
        if self.computed_for.as_ref() == Some(&(a.clone(), b.clone())) {
            return;
        }
        self.report = Some(vector_radar::compare_direction_files(&a, &b));
        self.computed_for = Some((a, b));
    }
}

/// The radar over the window, open on `app.radar`.
pub fn dialog(ctx: &egui::Context, app: &mut SurgeonApp) {
    if !app.radar.open {
        return;
    }
    let Some(work) = app
        .studio
        .manager
        .as_ref()
        .map(|manager| manager.project.dir.join("work"))
    else {
        app.radar.open = false;
        return;
    };
    app.radar.compute();
    let screen = ctx.content_rect();
    let size = Vec2::new(
        SIZE[0].min(screen.width() - 2.0 * T.size.activity),
        SIZE[1].min(screen.height() - 2.0 * T.size.titlebar),
    );
    let mut close = false;
    egui::Modal::new(Id::new("radar"))
        .backdrop_color(T.color.scrim)
        .frame(c::popup_frame().inner_margin(0.0))
        .show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
            c::Panel::new("radar")
                .bare()
                .title(t("radar.title"))
                .show_with(
                    ui,
                    rect,
                    &mut close,
                    |ui, close| {
                        if c::icon_button(ui, Icon::Close, t("radar.close")).clicked() {
                            *close = true;
                        }
                    },
                    |ui, _| body(ui, app, &work),
                );
        });
    if close {
        app.radar.open = false;
    }
}

fn body(ui: &mut Ui, app: &mut SurgeonApp, work: &Path) {
    ui.add_space(T.space.s);
    pickers(ui, app, work);
    ui.add_space(T.space.s);
    c::separator(ui);
    match &app.radar.report {
        None => c::empty(ui, Icon::List, t("radar.none"), t("radar.none.hint")),
        Some(Err(error)) => {
            c::status(ui, State::Error, error.as_str());
        }
        Some(Ok(report)) if report.rows.is_empty() => {
            c::empty(ui, Icon::List, t("radar.empty"), t("radar.empty.hint"))
        }
        Some(Ok(report)) => report_body(ui, report),
    }
}

/// One line per vector: a row wider than the panel would widen it past its clip.
fn pickers(ui: &mut Ui, app: &mut SurgeonApp, work: &Path) {
    let width = ui.available_width() / 2.0;
    ui.horizontal(|ui| {
        c::meta(ui, t("radar.vector.a"));
        select_file(ui, "radar-a", &mut app.radar.a, &app.radar.files, width);
    });
    ui.horizontal(|ui| {
        c::meta(ui, t("radar.vector.b"));
        select_file(ui, "radar-b", &mut app.radar.b, &app.radar.files, width);
        if c::button(ui, t("radar.refresh"), ButtonKind::Secondary).clicked() {
            app.radar.refresh(work);
        }
    });
}

/// `L14 · direction`, or the tensor alone when no layer is known.
fn row_name(row: &LayerRow) -> String {
    match row.layer {
        Some(layer) => format!("L{layer} · {}", row.tensor),
        None => row.tensor.clone(),
    }
}

fn select_file(
    ui: &mut Ui,
    id: &str,
    current: &mut Option<PathBuf>,
    files: &[PathBuf],
    width: f32,
) {
    let shown = current
        .as_ref()
        .and_then(|path| path.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("-");
    let mut picked = current.clone();
    c::select(ui, id, shown, width, |ui| {
        for file in files {
            let name = file
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            c::option(ui, &mut picked, Some(file.clone()), name);
        }
    });
    *current = picked;
}

fn report_body(ui: &mut Ui, report: &RadarReport) {
    summary(ui, report);
    ui.add_space(T.space.s);
    ScrollArea::vertical()
        .id_salt("radar-rows")
        .auto_shrink(false)
        .show(ui, |ui| {
            for row in &report.rows {
                layer_row(ui, row);
            }
        });
}

fn summary(ui: &mut Ui, report: &RadarReport) {
    let (state, word) = match report.collision {
        true => (State::Warn, t("radar.summary.collision")),
        false => (State::Ok, t("radar.summary.clear")),
    };
    c::status(ui, state, word);
    ui.add_space(T.space.xs);
    let mean = format!("{:.4}", report.mean_abs);
    c::meta(ui, tf("radar.summary.mean", &[("value", &mean)]));
    if let Some(row) = &report.strongest {
        let cosine = format!("{:+.4}", row.cosine);
        c::mono(
            ui,
            tf(
                "radar.summary.strongest",
                &[("tensor", &row_name(row)), ("value", &cosine)],
            ),
        );
    }
    c::secondary(
        ui,
        match report.collision {
            true => t("radar.advice.preserve"),
            false => t("radar.advice.free"),
        },
    );
}

fn layer_row(ui: &mut Ui, row: &LayerRow) {
    let (label, color) = match row.risk() {
        Risk::Orthogonal => (t("radar.risk.orthogonal"), T.color.ok),
        Risk::Moderate => (t("radar.risk.moderate"), T.color.warn),
        Risk::Strong => (t("radar.risk.strong"), T.color.error),
    };
    let name = row_name(row);
    c::row(ui, false, &name, |ui| {
        c::mono(ui, name.as_str());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let cosine = format!("{:+.4}", row.cosine);
            ui.add(egui::Label::new(
                egui::RichText::new(cosine)
                    .font(c::font_mono())
                    .color(color),
            ));
            c::meta(ui, label);
        });
    });
    let (rect, _) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), T.space.s), Sense::hover());
    let painter = ui.painter();
    let track = egui::Rect::from_center_size(rect.center(), Vec2::new(rect.width(), T.space.xs));
    painter.rect_filled(track, T.radius, T.color.surface_hover);
    let mut fill = track;
    fill.set_width(track.width() * row.cosine.abs() as f32);
    painter.rect_filled(fill, T.radius, color);
    ui.add_space(T.space.xs);
}
