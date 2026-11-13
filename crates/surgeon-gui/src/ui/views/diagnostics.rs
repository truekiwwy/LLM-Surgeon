//! Diagnostics: sections and the project on the left; the machine and the dependencies (with
//! their repair), the updates or the setup wizard in the center; the technical log and the
//! diagnostics report (no personal path, no secret) on the right.

use eframe::egui::{self, ScrollArea, Ui};

use super::super::components::{self as c, ButtonKind};
use super::super::i18n::{t, tf};
use super::super::icons::State;
use super::super::layout::{self, Part, Regions};
use super::super::theme::TOKENS as T;
use super::context;
use crate::app::SurgeonApp;
use crate::desktop::UpdateState;
use crate::diag::{Fix, Status};

fn section_label(k: usize) -> &'static str {
    match k {
        0 => t("diag.section.system"),
        1 => t("diag.section.updates"),
        _ => t("diag.section.wizard"),
    }
}

pub fn show(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    c::Panel::new("sections")
        .title(t("panel.sections"))
        .show(ui, regions.left, |ui| {
            for k in 0..3 {
                if c::row(ui, app.diag_section == k, section_label(k), |ui| {
                    c::label(ui, section_label(k));
                })
                .clicked()
                {
                    app.diag_section = k;
                }
            }
            c::separator(ui);
            context(ui, app);
        });
    match app.diag_section {
        0 => system(ui, app, regions),
        1 => c::Panel::new("updates")
            .title(t("panel.updates"))
            .show(ui, regions.center, |ui| updates(ui, app)),
        _ => c::Panel::new("wizard")
            .title(t("panel.wizard"))
            .show(ui, regions.center, |ui| {
                c::label(ui, t("diag.wizard.explain"));
                ui.add_space(T.space.m);
                if c::button(ui, t("diag.wizard.open"), ButtonKind::Secondary).clicked() {
                    app.desktop.open_wizard();
                }
            }),
    }
    journal(ui, app, regions.right);
}

fn system(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    let rows = app.desktop.system.as_ref().map_or(2, |s| {
        4 + s.gpus.len().max(1) + s.notes.len() - usize::from(s.disk_free_mib.is_none())
    });
    // A kv row is one row plus the item gap.
    let height = T.size.header + 2.0 * T.space.m + rows as f32 * (T.size.row + T.space.xs);
    let parts = layout::stack(
        regions.center,
        &[Part::Fixed(layout::snap8(height)), Part::Flex(1.0)],
    );
    let ctx = ui.ctx().clone();
    c::Panel::new("hardware")
        .title(t("panel.hardware"))
        .show_with(
            ui,
            parts[0],
            app,
            |ui, app| {
                let probing = app.desktop.probing();
                let label = if probing {
                    t("diag.probing")
                } else {
                    t("diag.refresh")
                };
                if c::button_if(ui, !probing, label, ButtonKind::Ghost, t("diag.probing")).clicked()
                {
                    app.desktop.refresh_system(&ctx);
                }
            },
            |ui, app| {
                // Keys and values stay readable together on a wide window.
                ui.set_max_width(T.size.center_min);
                let Some(s) = &app.desktop.system else {
                    c::empty(ui, t("diag.no_probe"), t("diag.no_probe.hint"));
                    return;
                };
                c::kv(ui, t("diag.os"), &format!("{} {}", s.os, s.arch));
                match s.gpus.is_empty() {
                    true => c::kv(ui, t("diag.gpu"), t("diag.gpu.none")),
                    false => {
                        for gpu in &s.gpus {
                            let vram = gpu
                                .vram_mib
                                .map_or(String::new(), |v| format!(" · {v} MiB"));
                            c::kv(ui, t("diag.gpu"), &format!("{}{vram}", gpu.name));
                        }
                    }
                }
                let vulkan = match s.vulkan {
                    Some(true) => t("diag.yes"),
                    Some(false) => t("diag.no"),
                    None => t("diag.unknown"),
                };
                c::kv(ui, "Vulkan", vulkan);
                c::kv(
                    ui,
                    t("diag.ram"),
                    &tf(
                        "diag.mib_free",
                        &[("free", &s.ram_free_mib), ("total", &s.ram_total_mib)],
                    ),
                );
                if let Some(d) = s.disk_free_mib {
                    c::kv(ui, t("diag.disk"), &tf("diag.mib", &[("n", &d)]));
                }
                for note in &s.notes {
                    c::status(ui, State::Warn, note.as_str());
                }
            },
        );
    let rows = app.desktop.rows(app.dialog.saved());
    let mut fix = None;
    c::Panel::new("dependencies")
        .title(t("panel.dependencies"))
        .show(ui, parts[1], |ui| {
            ScrollArea::vertical()
                .id_salt("deps")
                .auto_shrink(false)
                .show(ui, |ui| {
                    for row in &rows {
                        let state = match row.status {
                            Status::Ok => State::Ok,
                            Status::Missing => State::Error,
                            Status::Outdated => State::Warn,
                        };
                        ui.horizontal(|ui| {
                            c::status(ui, state, row.name);
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if row.fix != Fix::None {
                                        if c::button(ui, t("diag.reinstall"), ButtonKind::Ghost)
                                            .clicked()
                                        {
                                            fix = Some((row.fix, true));
                                        }
                                        if c::button(ui, t("diag.repair"), ButtonKind::Secondary)
                                            .clicked()
                                        {
                                            fix = Some((row.fix, false));
                                        }
                                    }
                                    c::meta(
                                        ui,
                                        crate::diag::status_word(row.status, app.desktop.lang()),
                                    );
                                },
                            );
                        });
                        c::secondary(ui, row.detail.as_str());
                        ui.add_space(T.space.s);
                    }
                });
        });
    if let Some((what, force)) = fix {
        app.desktop.open_fix(what, force);
    }
}

fn updates(ui: &mut Ui, app: &mut SurgeonApp) {
    let ctx = ui.ctx().clone();
    c::secondary(ui, t("update.explain"));
    ui.add_space(T.space.m);
    match &app.desktop.update {
        UpdateState::Idle | UpdateState::UpToDate | UpdateState::Failed(_) => {
            match &app.desktop.update {
                UpdateState::UpToDate => {
                    c::status(ui, State::Ok, t("update.latest"));
                }
                UpdateState::Failed(e) => c::message(ui, e, true),
                _ => {}
            }
            ui.add_space(T.space.s);
            if c::button(ui, t("update.check"), ButtonKind::Secondary).clicked() {
                app.desktop.check_update(&ctx);
            }
        }
        UpdateState::Checking(_) => {
            c::status(ui, State::Active, t("update.checking"));
        }
        UpdateState::Available(u) => {
            c::status(
                ui,
                State::Ok,
                tf("update.available", &[("version", &u.version)]),
            );
            c::kv(ui, t("update.file"), &u.name);
            c::kv(
                ui,
                t("update.size"),
                &format!("{:.1} MiB", u.size as f64 / (1 << 20) as f64),
            );
            c::mono_wrapped(ui, u.url.as_str(), T.color.text_secondary);
            let hash = match &u.sha256 {
                Some(h) => format!("SHA-256 {h}"),
                None => t("update.no_hash").into(),
            };
            c::mono_wrapped(ui, hash, T.color.text_secondary);
            ui.add_space(T.space.s);
            if c::button(ui, t("update.download"), ButtonKind::Secondary).clicked() {
                app.desktop.download_update(&ctx);
            }
        }
        UpdateState::Downloading(u, task) => {
            c::mono(ui, u.name.as_str());
            let p = task.progress();
            c::progress(ui, p.fraction().unwrap_or(0.0), false);
            if c::button(ui, t("action.cancel"), ButtonKind::Ghost).clicked() {
                task.cancel();
            }
        }
        UpdateState::Downloaded(path) => {
            c::status(ui, State::Ok, t("update.downloaded"));
            c::mono_wrapped(ui, path.display().to_string(), T.color.text_secondary);
        }
    }
}

/// `1794605966 INFO …` as `14:59:26 INFO …` (UTC): the log stamps lines in Unix seconds.
fn clock(line: &str) -> String {
    match line.split_once(' ') {
        Some((stamp, rest)) => match stamp.parse::<u64>() {
            Ok(t) => {
                let s = t % 86_400;
                format!("{:02}:{:02}:{:02} {rest}", s / 3600, s % 3600 / 60, s % 60)
            }
            Err(_) => line.to_owned(),
        },
        None => line.to_owned(),
    }
}

fn journal(ui: &mut Ui, app: &mut SurgeonApp, rect: egui::Rect) {
    let parts = layout::stack(
        rect,
        &[
            Part::Flex(1.0),
            Part::Fixed(T.size.header + 3.0 * T.size.row),
        ],
    );
    c::Panel::new("journal")
        .title(t("panel.journal"))
        .show(ui, parts[0], |ui| {
            // shortcut: the log tail is read every frame; cache it if logs grow past a few MiB.
            let lines = crate::logging::tail(&crate::paths::logs_dir(), 200);
            if lines.is_empty() {
                c::empty(ui, t("journal.empty"), t("journal.empty.hint"));
                return;
            }
            ScrollArea::vertical()
                .id_salt("journal")
                .auto_shrink(false)
                .stick_to_bottom(true)
                .show(ui, |ui| {
                    for line in lines.iter().map(|l| clock(l)) {
                        let state = match () {
                            _ if line.contains("WARN") => Some(State::Warn),
                            _ if line.contains("ERROR") => Some(State::Error),
                            _ => None,
                        };
                        match state {
                            Some(state) => c::status(ui, state, line),
                            None => c::mono_secondary(ui, line),
                        };
                    }
                });
        });
    let rows = app.desktop.rows(app.dialog.saved());
    c::Panel::new("report")
        .title(t("panel.report"))
        .show(ui, parts[1], |ui| {
            c::meta(ui, t("report.explain"));
            ui.add_space(T.space.s);
            if c::button_wide(ui, t("report.generate"), ButtonKind::Primary).clicked() {
                let ctx = ui.ctx().clone();
                app.desktop.export_diagnostics(&ctx, &rows);
            }
            if let Some(text) = &app.desktop.export {
                c::meta(ui, tf("report.copied", &[("n", &text.lines().count())]));
            }
        });
}

#[cfg(test)]
mod tests {
    #[test]
    fn log_stamps_read_as_a_utc_clock() {
        assert_eq!(super::clock("86399 INFO x"), "23:59:59 INFO x");
        assert_eq!(super::clock("3661 WARN y z"), "01:01:01 WARN y z");
        assert_eq!(super::clock("not a stamp"), "not a stamp");
    }
}
