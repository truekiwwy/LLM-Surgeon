use eframe::egui::{self, Align2, Pos2, ScrollArea, Sense, Stroke, Ui, Vec2};
use weight_surgeon::checkpoint::pareto::Point;
use weight_surgeon::checkpoint::{Listing, Manifest};

use super::super::components::{self as c, ButtonKind};
use super::super::i18n::{t, tf};
use super::super::icons::Icon;
use super::super::theme::TOKENS as T;
use super::age;
use crate::agent::paths::ProjectPaths;
use crate::app::SurgeonApp;
use crate::studio::checkpoints;
use crate::studio::project::now_unix;

const PLOT_HEIGHT: f32 = 168.0;
const DOT: f32 = 4.0;

pub fn show(ui: &mut Ui, app: &mut SurgeonApp) {
    let listing = match app.checkpoints() {
        Some(Ok(listing)) if !listing.checkpoints.is_empty() => listing.clone(),
        Some(Err(e)) => {
            let e = e.clone();
            c::message(ui, &e, true);
            return;
        }
        _ => {
            c::empty(
                ui,
                Icon::List,
                t("checkpoints.none"),
                t("checkpoints.none.hint"),
            );
            return;
        }
    };
    let head = app
        .agents
        .project
        .as_deref()
        .and_then(|p| ProjectPaths::of(p).copy)
        .and_then(|copy| checkpoints::head(&listing, &copy).map(str::to_owned));
    let front = checkpoints::front(&listing);
    let depths = checkpoints::depths(&listing);
    if app
        .checkpoint_selected
        .as_ref()
        .is_none_or(|id| !listing.checkpoints.iter().any(|c| c.manifest.id == *id))
    {
        app.checkpoint_selected = head.clone();
    }
    c::section(ui, t("checkpoints.title"));
    c::kv(
        ui,
        t("checkpoints.disk"),
        &tf(
            "checkpoints.bytes",
            &[
                ("disk", &mib(listing.disk_bytes)),
                ("logical", &mib(listing.logical_bytes)),
            ],
        ),
    );
    plot(ui, app, &listing, &front, head.as_deref());
    ui.add_space(T.space.s);
    ScrollArea::vertical()
        .id_salt("checkpoints")
        .auto_shrink(false)
        .show(ui, |ui| {
            for (listed, depth) in listing.checkpoints.iter().zip(&depths) {
                let m = &listed.manifest;
                let selected = app.checkpoint_selected.as_deref() == Some(m.id.as_str());
                let row = c::row(ui, selected, &m.label, |ui| {
                    ui.add_space(*depth as f32 * T.space.m);
                    c::mono(ui, m.id.as_str());
                    c::line(ui, label(m));
                    if head.as_deref() == Some(m.id.as_str()) {
                        c::meta(ui, t("checkpoints.current"));
                    }
                    if front.contains(&m.id) {
                        c::meta(ui, t("checkpoints.front"));
                    }
                });
                if row.clicked() {
                    app.checkpoint_selected = Some(m.id.clone());
                }
            }
            let selected = listing
                .checkpoints
                .iter()
                .find(|c| app.checkpoint_selected.as_deref() == Some(c.manifest.id.as_str()));
            if let Some(listed) = selected {
                detail(
                    ui,
                    app,
                    &listed.manifest,
                    listed.exclusive_bytes,
                    head.as_deref(),
                );
            }
        });
}

fn mib(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / f64::from(1 << 20))
}

fn label(m: &Manifest) -> String {
    match checkpoints::point(m) {
        Some(p) => format!("{} · {:+.3} / {:.3}", m.label, p.gain, p.drift),
        None => m.label.clone(),
    }
}

fn plot(
    ui: &mut Ui,
    app: &mut SurgeonApp,
    listing: &Listing,
    front: &[String],
    head: Option<&str>,
) {
    let points: Vec<(&str, Point)> = listing
        .checkpoints
        .iter()
        .filter_map(|c| Some((c.manifest.id.as_str(), checkpoints::point(&c.manifest)?)))
        .collect();
    if points.is_empty() {
        c::secondary(ui, t("checkpoints.plot.none"));
        return;
    }
    let (rect, response) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), PLOT_HEIGHT), Sense::click());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, T.radius, T.color.surface);
    let area = rect.shrink(T.space.m);
    let max_drift = points
        .iter()
        .map(|(_, p)| p.drift)
        .fold(0.0, f64::max)
        .max(1e-9);
    let (low, high) = points.iter().fold((0.0f64, 0.0f64), |(lo, hi), (_, p)| {
        (lo.min(p.gain), hi.max(p.gain))
    });
    let span = (high - low).max(1e-9);
    let at = |p: Point| {
        Pos2::new(
            area.left() + (p.drift / max_drift) as f32 * area.width(),
            area.bottom() - ((p.gain - low) / span) as f32 * area.height(),
        )
    };
    let axis = Stroke::new(1.0, T.color.border);
    painter.line_segment([area.left_bottom(), area.right_bottom()], axis);
    painter.line_segment([area.left_bottom(), area.left_top()], axis);
    painter.text(
        area.right_bottom(),
        Align2::RIGHT_BOTTOM,
        t("checkpoints.plot.drift"),
        c::font_meta(),
        T.color.text_secondary,
    );
    painter.text(
        area.left_top(),
        Align2::LEFT_TOP,
        t("checkpoints.plot.gain"),
        c::font_meta(),
        T.color.text_secondary,
    );
    let steps: Vec<Pos2> = front
        .iter()
        .filter_map(|id| points.iter().find(|(p, _)| p == id).map(|(_, p)| at(*p)))
        .collect();
    painter.add(egui::Shape::line(steps, Stroke::new(1.5, T.color.accent)));
    let hover = response.hover_pos();
    let mut nearest: Option<(&str, Pos2, f32)> = None;
    for (id, p) in &points {
        let pos = at(*p);
        let on_front = front.iter().any(|f| f == id);
        let color = match on_front {
            true => T.color.accent,
            false => T.color.text_secondary,
        };
        painter.circle_filled(pos, DOT, color);
        if Some(*id) == head {
            painter.circle_stroke(pos, DOT + 3.0, Stroke::new(1.5, T.color.ok));
        }
        if app.checkpoint_selected.as_deref() == Some(*id) {
            painter.circle_stroke(pos, DOT + 6.0, Stroke::new(1.0, T.color.text));
        }
        if let Some(d) = hover.map(|h| h.distance(pos)).filter(|d| *d < 4.0 * DOT) {
            if nearest.is_none_or(|(_, _, best)| d < best) {
                nearest = Some((id, pos, d));
            }
        }
    }
    if let Some((id, pos, _)) = nearest {
        let text = listing
            .checkpoints
            .iter()
            .find(|c| c.manifest.id == id)
            .map(|c| format!("{} {}", c.manifest.id, label(&c.manifest)))
            .unwrap_or_default();
        let anchor = match pos.x > rect.center().x {
            true => Align2::RIGHT_BOTTOM,
            false => Align2::LEFT_BOTTOM,
        };
        painter.text(
            pos - Vec2::new(0.0, DOT + 2.0),
            anchor,
            text,
            c::font_meta(),
            T.color.text,
        );
        if response.clicked() {
            app.checkpoint_selected = Some(id.to_owned());
        }
    }
}

fn detail(ui: &mut Ui, app: &mut SurgeonApp, m: &Manifest, exclusive: u64, head: Option<&str>) {
    ui.add_space(T.space.s);
    c::section(ui, &m.label);
    c::kv(ui, "id", &m.id);
    c::kv(
        ui,
        t("checkpoints.parent"),
        m.parent.as_deref().unwrap_or("-"),
    );
    c::kv(
        ui,
        t("checkpoints.created"),
        &age(now_unix().saturating_sub(m.created_unix)),
    );
    c::kv(ui, "sha256", m.sha256.get(..16).unwrap_or(&m.sha256));
    c::kv(ui, t("checkpoints.files"), &m.files.len().to_string());
    c::kv(
        ui,
        t("checkpoints.exclusive"),
        &tf("checkpoints.mib", &[("n", &mib(exclusive))]),
    );
    if let (Some(target), Some(p)) = (m.metrics.get("target"), checkpoints::point(m)) {
        c::kv(
            ui,
            t("checkpoints.before"),
            &format!("{:.1} %", (target - p.gain) * 100.0),
        );
        c::kv(
            ui,
            t("checkpoints.after"),
            &format!("{:.1} %", target * 100.0),
        );
        c::kv(ui, t("checkpoints.gain"), &format!("{:+.4}", p.gain));
        c::kv(ui, t("checkpoints.drift"), &format!("{:.4}", p.drift));
    }
    ui.add_space(T.space.s);
    let current = head == Some(m.id.as_str());
    let (enabled, why) = match (app.studio.manager.is_some(), current) {
        (false, _) => (false, t("checkpoints.restore.none")),
        (true, true) => (false, t("checkpoints.restore.current")),
        (true, false) => (true, t("checkpoints.restore.tip")),
    };
    if c::button_if(
        ui,
        enabled,
        t("checkpoints.restore"),
        ButtonKind::Primary,
        why,
    )
    .clicked()
    {
        app.restore_checkpoint(m);
    }
}
