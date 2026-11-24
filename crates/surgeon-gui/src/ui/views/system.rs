//! The settings dialog, the one place where the environment is configured, sections on the
//! left: general (language, local only, projects folder, the wizard to replay), the AI
//! providers (keys in the vault, models, prices, CLI runtimes), the external harnesses (the
//! `weight-surgeon` skill and the MCP configuration of `surgeon mcp`), the engine (machine,
//! dependencies with their repair, inference backend), the updates, and the technical log
//! with the diagnostics report (no personal path, no secret).

use eframe::egui::{self, Id, Pos2, Rect, ScrollArea, Sense, Ui, Vec2};

use super::super::components::{self as c, ButtonKind, Segment};
use super::super::i18n::{t, tf};
use super::super::icons::{self, Icon, State};
use super::super::layout::Regions;
use super::super::theme::TOKENS as T;
use crate::agent::config::Connection;
use crate::agent::detect::{Preset, Reason};
use crate::app::{Section, SurgeonApp};
use crate::desktop::UpdateState;
use crate::diag::{actions_needed, Fix, Group, Row, Severity, Status, Uses};
use crate::integration::{self, Harness, SkillState};
use crate::settings::Lang;
use weight_surgeon::doctor::Level;

/// The column of the sections.
const NAV: f32 = 232.0;
/// The largest the dialog gets.
const SIZE: [f32; 2] = [1120.0, 760.0];

fn section_label(section: Section) -> &'static str {
    match section {
        Section::General => t("settings.section.general"),
        Section::Providers => t("settings.section.providers"),
        Section::Harnesses => t("settings.section.harnesses"),
        Section::Engine => t("settings.section.engine"),
        Section::Updates => t("settings.section.updates"),
        Section::Journal => t("settings.section.journal"),
    }
}

fn section_icon(section: Section) -> Icon {
    match section {
        Section::General => Icon::Sliders,
        Section::Providers => Icon::Sparkle,
        Section::Harnesses => Icon::Cube,
        Section::Engine => Icon::Chip,
        Section::Updates => Icon::Download,
        Section::Journal => Icon::List,
    }
}

/// The settings dialog over the window, open on `app.settings`. Escape, a click outside or
/// the close button closes it.
pub fn dialog(ctx: &egui::Context, app: &mut SurgeonApp) {
    let Some(section) = app.settings else {
        return;
    };
    let screen = ctx.content_rect();
    let size = Vec2::new(
        SIZE[0].min(screen.width() - 2.0 * T.size.activity),
        SIZE[1].min(screen.height() - 2.0 * T.size.titlebar),
    );
    let mut close = false;
    let modal = egui::Modal::new(Id::new("settings"))
        .backdrop_color(T.color.scrim)
        .frame(c::popup_frame().inner_margin(0.0))
        .show(ctx, |ui| {
            let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
            c::record(ui.ctx(), "settings", rect);
            let nav = Rect::from_min_size(rect.min, Vec2::new(NAV, rect.height()));
            let body = Rect::from_min_max(Pos2::new(nav.right(), rect.top()), rect.max);
            c::vline(ui.painter(), nav.right(), rect.y_range(), T.color.separator);
            c::region(ui, nav.shrink(T.space.m), "settings-nav", |ui| {
                ui.add_space(T.space.xs);
                c::strong(ui, t("settings.title"));
                ui.add_space(T.space.m);
                ui.spacing_mut().item_spacing.y = T.space.xs / 2.0;
                for s in Section::ALL {
                    let label = section_label(s);
                    let picked = s == section;
                    let row = c::row(ui, picked, label, |ui| {
                        let (glyph, _) =
                            ui.allocate_exact_size(Vec2::splat(T.size.icon), Sense::hover());
                        let ink = if picked {
                            T.color.accent
                        } else {
                            T.color.text_secondary
                        };
                        icons::paint(ui.painter(), glyph, section_icon(s), ink);
                        c::line(ui, label);
                    });
                    if row.clicked() {
                        app.settings = Some(s);
                    }
                }
            });
            let shut = Rect::from_min_size(
                Pos2::new(
                    body.right() - T.space.s - T.size.control,
                    body.top() + (T.size.header - T.size.control) / 2.0,
                ),
                Vec2::splat(T.size.control),
            );
            close = c::region(ui, shut, "settings-close", |ui| {
                c::icon_button(ui, Icon::Close, t("settings.close")).clicked()
            });
            let right =
                Rect::from_min_max(Pos2::new(body.right() - T.size.right, body.top()), body.max);
            let center = Rect::from_min_max(body.min, Pos2::new(right.left(), body.bottom()));
            let regions = Regions {
                titlebar: Rect::NOTHING,
                activity: Rect::NOTHING,
                left: nav,
                center,
                right,
            };
            if matches!(
                section,
                Section::Providers | Section::Harnesses | Section::Engine | Section::Journal
            ) {
                c::vline(
                    ui.painter(),
                    right.left(),
                    right.y_range(),
                    T.color.separator,
                );
            }
            match section {
                Section::General => c::Panel::new("general")
                    .bare()
                    .title(t("settings.section.general"))
                    .show(ui, body, |ui| general(ui, app)),
                Section::Providers => providers(ui, app, &regions),
                Section::Harnesses => harnesses(ui, app, &regions),
                Section::Engine => machine(ui, app, &regions),
                Section::Updates => c::Panel::new("updates")
                    .bare()
                    .title(t("panel.updates"))
                    .show(ui, body, |ui| updates(ui, app)),
                Section::Journal => journal(ui, app, &regions),
            }
        });
    if close || modal.should_close() {
        app.settings = None;
    }
}

/// One setting: its name and what it does on the left, its control on the right.
fn setting(ui: &mut Ui, title: &str, hint: &str, control: impl FnOnce(&mut Ui)) {
    ui.horizontal_top(|ui| {
        let width = (ui.available_width() * 0.55).floor();
        ui.allocate_ui_with_layout(
            Vec2::new(width, T.size.control),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                ui.set_width(width);
                c::strong(ui, title);
                c::secondary(ui, hint);
            },
        );
        ui.add_space(T.space.l);
        control(ui);
    });
    ui.add_space(T.space.l);
}

/// Language, local only, the projects folder, the wizard to replay, the version.
fn general(ui: &mut Ui, app: &mut SurgeonApp) {
    ScrollArea::vertical()
        .id_salt("general")
        .auto_shrink(false)
        .show(ui, |ui| {
            setting(ui, t("settings.lang"), t("settings.lang.hint"), |ui| {
                let mut lang = app.desktop.lang();
                let options = [
                    Segment {
                        value: Lang::Fr,
                        label: "Français",
                        tooltip: t("status.language"),
                    },
                    Segment {
                        value: Lang::En,
                        label: "English",
                        tooltip: t("status.language"),
                    },
                ];
                if c::segmented(ui, &mut lang, &options, c::font_body()) {
                    app.desktop.set_lang(lang);
                }
            });
            setting(ui, t("settings.local"), t("policy.local_only.tip"), |ui| {
                let mut local = app.desktop.settings.local_only;
                if c::switch(
                    ui,
                    Icon::Shield,
                    t("status.local"),
                    &mut local,
                    t("policy.local_only.tip"),
                ) {
                    app.set_local_only(local);
                }
            });
            let folder = app.studio.workspace.display().to_string();
            let ctx = ui.ctx().clone();
            setting(ui, t("settings.workspace"), &folder, |ui| {
                if app.workspace_picker.busy() {
                    c::busy(ui, t("project.source.picking"));
                } else if cfg!(windows)
                    && c::button(ui, t("settings.workspace.change"), ButtonKind::Secondary)
                        .clicked()
                {
                    app.pick_workspace(&ctx, t("wiz.workspace.hint"));
                }
            });
            setting(ui, t("system.wizard"), t("system.wizard.tip"), |ui| {
                if c::button(ui, t("settings.wizard.run"), ButtonKind::Secondary).clicked() {
                    app.settings = None;
                    app.desktop.open_wizard();
                }
            });
            let version = format!("v{}", env!("CARGO_PKG_VERSION"));
            setting(ui, t("settings.version"), &version, |_| {});
        });
}

/// The dependencies in the center, the machine and the inference backend on the right.
fn machine(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    let ctx = ui.ctx().clone();
    c::Panel::new("hardware")
        .bare()
        .title(t("panel.hardware"))
        .show_with(
            ui,
            regions.right,
            app,
            |ui, app| {
                ui.add_space(T.size.control + T.space.s);
                let probing = app.desktop.probing();
                if probing {
                    c::spinner(ui);
                }
                let label = if probing {
                    t("diag.probing")
                } else {
                    t("diag.refresh")
                };
                if c::button_if(ui, !probing, label, ButtonKind::Ghost, t("diag.probing")).clicked()
                {
                    app.desktop.refresh_system(&ctx);
                    app.refresh_doctor(&ctx, false);
                }
            },
            |ui, app| {
                machine_facts(ui, app);
                c::separator(ui);
                // Its label is long (backend, API, layers, batch): it takes a line of its own.
                c::secondary(ui, t("settings.backend"));
                c::mono_wrapped(ui, app.backend_label(), T.color.text);
                ui.add_space(T.space.s);
                if c::button(ui, t("settings.backend.configure"), ButtonKind::Secondary).clicked() {
                    app.dialog.open = true;
                }
            },
        );
    let rows = app.desktop.rows(app.dialog.saved(), uses(app));
    let mut fix = None;
    c::Panel::new("dependencies")
        .bare()
        .title(t("panel.dependencies"))
        .show(ui, regions.center, |ui| {
            readiness(ui, &rows);
            ScrollArea::vertical()
                .id_salt("deps")
                .auto_shrink(false)
                .show(ui, |ui| {
                    health(ui, app);
                    for group in [Group::Essential, Group::Acceleration, Group::Samples] {
                        c::section(ui, group_label(group));
                        for row in rows.iter().filter(|r| r.group == group) {
                            if let Some(f) = dependency(ui, row) {
                                fix = Some(f);
                            }
                        }
                    }
                });
        });
    if let Some((what, force)) = fix {
        app.desktop.open_fix(what, force);
    }
}

/// The engine's machine check (`surgeon doctor`, run the first time the section shows): one
/// line per check with the engine's own words, then what can be done from here: purge the
/// stale reference cache, start or restart a hipfire daemon that is down or lost its GPU.
fn health(ui: &mut Ui, app: &mut SurgeonApp) {
    let ctx = ui.ctx().clone();
    c::section(ui, t("doctor.title"));
    let checking = app.desktop.doctoring();
    if app.desktop.doctor.is_none() && !checking {
        app.refresh_doctor(&ctx, false);
    }
    let Some(report) = app.desktop.doctor.clone() else {
        c::status(ui, State::Active, t("doctor.running"));
        ui.add_space(T.space.m);
        return;
    };
    match report.ready {
        true => c::status(ui, State::Ok, t("doctor.ready")),
        false => c::status(ui, State::Error, t("doctor.not_ready")),
    };
    ui.add_space(T.space.xs);
    for check in &report.checks {
        let state = match check.level {
            Level::Ok => State::Ok,
            Level::Warn => State::Warn,
            Level::Fail => State::Error,
        };
        let name = match check.name {
            "memory" => t("doctor.memory"),
            "cpu" => t("doctor.cpu"),
            "cache" => t("doctor.cache"),
            _ => t("doctor.hipfire"),
        };
        c::status(ui, state, name);
        c::secondary(ui, check.message.as_str());
        ui.add_space(T.space.xs);
    }
    let stale = report.cache.corrupt + report.cache.obsolete;
    let hipfire = report.hipfire.as_ref();
    let down = hipfire.is_some_and(|h| !h.up);
    let frozen = hipfire
        .and_then(|h| h.gpu_memory.as_ref())
        .is_some_and(|g| g.total_mb == 0);
    let busy = app.desktop.setup.hipfire_busy();
    ui.add_space(T.space.s);
    ui.horizontal_wrapped(|ui| {
        let why = t("doctor.running");
        if stale > 0
            && c::button_if(
                ui,
                !checking,
                &tf("doctor.purge", &[("n", &stale)]),
                ButtonKind::Secondary,
                why,
            )
            .clicked()
        {
            app.refresh_doctor(&ctx, true);
        }
        if down || frozen {
            let label = match down {
                true => t("doctor.hipfire.start"),
                false => t("doctor.hipfire.restart"),
            };
            if c::button_if(
                ui,
                !busy,
                label,
                ButtonKind::Secondary,
                t("doctor.hipfire.busy"),
            )
            .clicked()
            {
                app.restart_hipfire(&ctx);
            }
        }
    });
    if busy {
        c::status(ui, State::Active, t("doctor.hipfire.busy"));
    }
    if let Some(Err(e)) = &app.desktop.setup.hipfire {
        c::message(ui, e, true);
    }
    if report.cache.removed > 0 {
        c::meta(ui, tf("doctor.purged", &[("n", &report.cache.removed)]));
    }
    c::separator(ui);
}

/// The machine in key / value rows: OS, GPUs, Vulkan, memory, disk, notes.
pub(crate) fn machine_facts(ui: &mut Ui, app: &SurgeonApp) {
    let Some(s) = &app.desktop.system else {
        c::empty(ui, Icon::Chip, t("diag.no_probe"), t("diag.no_probe.hint"));
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
}

/// What the machine calls for, the severities follow it: its GPU, the hipfire daemon.
fn uses(app: &SurgeonApp) -> Uses {
    let gpus = app.desktop.system.iter().flat_map(|s| s.gpus.iter());
    Uses {
        vendor: crate::backend_choice::main_vendor(gpus.map(|g| g.name.as_str())),
        hipfire_up: app.hipfire_up,
    }
}

/// One line over the list: ready, or how many rows need the user.
fn readiness(ui: &mut Ui, rows: &[Row]) {
    let blocking = rows.iter().any(|r| r.severity == Severity::Blocking);
    match (actions_needed(rows), blocking) {
        (0, _) => c::status(ui, State::Ok, t("deps.ready")),
        (n, true) => c::status(ui, State::Error, tf("deps.actions", &[("n", &n)])),
        (n, false) => c::status(ui, State::Warn, tf("deps.actions", &[("n", &n)])),
    };
    c::separator(ui);
}

fn group_label(group: Group) -> &'static str {
    match group {
        Group::Essential => t("deps.group.essential"),
        Group::Acceleration => t("deps.group.acceleration"),
        Group::Samples => t("deps.group.samples"),
    }
}

/// One dependency: its state in words, what to do about it, its detail. The repair asked for.
fn dependency(ui: &mut Ui, row: &Row) -> Option<(Fix, bool)> {
    let (state, word) = match (row.severity, row.status) {
        (Severity::Ok, _) => (State::Ok, t("deps.ok").to_owned()),
        (Severity::Optional, Status::Outdated) => (State::Idle, t("deps.outdated").to_owned()),
        (Severity::Optional, _) => (State::Idle, t("deps.optional").to_owned()),
        (Severity::NeededBy(who), _) => (State::Warn, tf("deps.needed_by", &[("who", &who)])),
        (Severity::Blocking, _) => (State::Error, t("deps.required").to_owned()),
    };
    let mut fix = None;
    ui.horizontal(|ui| {
        c::status(ui, state, row.name);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if row.fix != Fix::None {
                let (text, kind, force) = match row.status {
                    // hipfire runs outside the app: its button opens the guide.
                    _ if row.fix == Fix::Hipfire => (t("diag.guide"), ButtonKind::Secondary, false),
                    Status::Missing => (t("diag.install"), ButtonKind::Secondary, false),
                    Status::Outdated => (t("diag.repair"), ButtonKind::Secondary, false),
                    Status::Ok => (t("diag.reinstall"), ButtonKind::Ghost, true),
                };
                if c::button(ui, text, kind).clicked() {
                    fix = Some((row.fix, force));
                }
            }
            c::meta(ui, word);
        });
    });
    c::secondary(ui, row.detail.as_str());
    ui.add_space(T.space.s);
    fix
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
            let local = app.desktop.settings.local_only;
            if c::button_if(
                ui,
                !local,
                t("update.check"),
                ButtonKind::Secondary,
                t("update.local_only"),
            )
            .clicked()
            {
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

/// The technical log in the center, the diagnostics report on the right.
fn journal(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    c::Panel::new("journal")
        .bare()
        .title(t("panel.journal"))
        .show(ui, regions.center, |ui| {
            // shortcut: the log tail is read every frame; cache it if logs grow past a few MiB.
            let lines = crate::logging::tail(&crate::paths::logs_dir(), 200);
            if lines.is_empty() {
                c::empty(ui, Icon::List, t("journal.empty"), t("journal.empty.hint"));
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
    let rows = app.desktop.rows(app.dialog.saved(), uses(app));
    c::Panel::new("report")
        .bare()
        .title(t("panel.report"))
        .show(ui, regions.right, |ui| {
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

// ------------------------------------------------------------------ external harnesses

/// The program the MCP configuration starts, looked for once per session.
fn surgeon_command() -> &'static std::path::Path {
    static COMMAND: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
    COMMAND.get_or_init(integration::surgeon_command)
}

/// The skill of each harness in the center, the MCP configuration on the right. A click
/// writes; the states are read from disk on every frame.
fn harnesses(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    let ctx = ui.ctx().clone();
    let home = crate::paths::home_dir();
    let mut note = None;
    c::Panel::new("skill")
        .bare()
        .title(t("panel.skill"))
        .show(ui, regions.center, |ui| {
            c::secondary(ui, t("skill.explain"));
            ui.add_space(T.space.m);
            let Some(home) = &home else {
                c::message(ui, t("skill.no_home"), true);
                return;
            };
            for harness in Harness::ALL {
                let dir = harness.skill_dir(home);
                let (state, word, action) = match integration::skill_state(&dir) {
                    SkillState::UpToDate => (State::Ok, t("skill.installed"), None),
                    SkillState::Outdated => {
                        (State::Warn, t("skill.outdated"), Some(t("skill.update")))
                    }
                    SkillState::Missing => {
                        (State::Idle, t("skill.missing"), Some(t("skill.install")))
                    }
                };
                ui.horizontal(|ui| {
                    c::status(ui, state, harness.name());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let clicked = action.is_some_and(|text| {
                            c::button(ui, text, ButtonKind::Secondary).clicked()
                        });
                        if clicked {
                            note = Some(match integration::install_skill(&dir) {
                                Ok(()) => (tf("skill.done", &[("path", &dir.display())]), false),
                                Err(e) => (e, true),
                            });
                        }
                        c::meta(ui, word);
                    });
                });
                c::mono_wrapped(ui, dir.display().to_string(), T.color.text_secondary);
                ui.add_space(T.space.s);
            }
        });
    c::Panel::new("mcp")
        .bare()
        .title(t("panel.mcp"))
        .show(ui, regions.right, |ui| {
            c::secondary(ui, t("mcp.explain"));
            ui.add_space(T.space.s);
            c::meta(ui, t("mcp.command"));
            c::mono_wrapped(ui, surgeon_command().display().to_string(), T.color.text);
            ui.add_space(T.space.s);
            let snippet = integration::mcp_snippet(surgeon_command());
            let text = serde_json::to_string_pretty(&snippet).unwrap_or_default();
            c::mono_wrapped(ui, text.as_str(), T.color.text_secondary);
            ui.add_space(T.space.m);
            if c::button_wide(ui, t("mcp.copy"), ButtonKind::Primary).clicked() {
                ctx.copy_text(text.clone());
                note = Some((t("mcp.copied").into(), false));
            }
            if let Some(home) = &home {
                let file = integration::claude_config(home);
                if c::button_wide(ui, t("mcp.add_claude"), ButtonKind::Secondary).clicked() {
                    note = Some(
                        match integration::merge_mcp_config(&file, surgeon_command()) {
                            Ok(Some(bak)) => (
                                tf(
                                    "mcp.added",
                                    &[("path", &file.display()), ("backup", &bak.display())],
                                ),
                                false,
                            ),
                            Ok(None) => (tf("mcp.created", &[("path", &file.display())]), false),
                            Err(e) => (e, true),
                        },
                    );
                }
            }
            if let Some((text, error)) = &app.harness_note {
                ui.add_space(T.space.s);
                c::message(ui, text, *error);
            }
        });
    if note.is_some() {
        app.harness_note = note;
    }
}

// ------------------------------------------------------------------ AI providers

fn reason_label(reason: Reason) -> &'static str {
    match reason {
        Reason::NotSupported => t("scan.reason.unsupported"),
    }
}

/// `2.1.3 (Claude Code)`, `codex-cli 0.166.0`, `opencode v2.2.7`: the number alone.
fn short_version(version: &str) -> &str {
    version
        .split_whitespace()
        .find(|w| {
            w.trim_start_matches('v')
                .starts_with(|c: char| c.is_ascii_digit())
        })
        .unwrap_or(version)
}

fn preset_label(preset: Option<Preset>) -> &'static str {
    match preset {
        Some(Preset::Anthropic) => "Anthropic",
        Some(Preset::OpenAi) => "OpenAI",
        Some(Preset::OpenRouter) => "OpenRouter",
        None => t("prov.custom"),
    }
}

/// What the machine offers on the left, the connection picked on the right. The first visit
/// scans (CLIs, loopback servers, environment keys: nothing is spent).
fn providers(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    c::Panel::new("providers")
        .bare()
        .title(t("panel.providers"))
        .show_with(
            ui,
            regions.center,
            app,
            |ui, app| {
                if app.agents.scanning() {
                    c::busy(ui, t("scan.running"));
                } else if c::button(ui, t("scan.again"), ButtonKind::Ghost).clicked() {
                    app.agents.rescan();
                }
            },
            |ui, app| {
                ScrollArea::vertical()
                    .id_salt("providers")
                    .auto_shrink(false)
                    .show(ui, |ui| provider_list(ui, app, Show::All));
            },
        );
    c::Panel::new("provider")
        .bare()
        .title(t("panel.provider"))
        .show(ui, regions.right, |ui| {
            ScrollArea::vertical()
                .id_salt("provider")
                .auto_shrink(false)
                .show(ui, |ui| provider_detail(ui, app));
        });
}

/// One line of the list: state and name, then what it is, a star for the default, an
/// action. `(row clicked, action clicked)`.
fn entry(
    ui: &mut Ui,
    selected: bool,
    state: State,
    name: &str,
    word: &str,
    star: bool,
    action: Option<&str>,
) -> (bool, bool) {
    let mut acted = false;
    let row = c::row(ui, selected, name, |ui| {
        c::status(ui, state, name);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if let Some(text) = action {
                acted = c::button(ui, text, ButtonKind::Secondary).clicked();
            }
            c::meta(ui, word);
            if star {
                c::meta(ui, t("prov.star"));
            }
        });
    });
    (row.clicked() && !acted, acted)
}

/// The state of a connection in words.
fn conn_state(app: &SurgeonApp, conn: &Connection) -> (State, String) {
    let hub = &app.agents;
    if hub.cfg.local_only && !conn.is_local() {
        return (State::Idle, t("prov.blocked").into());
    }
    match (conn.ready(), conn.kind.is_cli()) {
        (true, true) => (State::Ok, t("prov.ready").into()),
        (true, false) => (State::Ok, tf("prov.models", &[("n", &conn.models.len())])),
        (false, true) => (State::Idle, t("prov.to_test").into()),
        (false, false) if conn.kind.needs_key() && hub.key_tail(&conn.id).is_none() => {
            (State::Warn, t("prov.key.missing").into())
        }
        (false, false) => (State::Idle, t("prov.to_test").into()),
    }
}

/// Which part of the provider list to show: all of it (System), or one tab of the wizard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Show {
    All,
    /// The CLI runtimes.
    Agents,
    /// Local servers and cloud providers, with the add form.
    Providers,
}

pub(crate) fn provider_list(ui: &mut Ui, app: &mut SurgeonApp, show: Show) {
    if show == Show::All {
        let mut local = app.desktop.settings.local_only;
        if c::checkbox(ui, &mut local, t("policy.local_only"))
            .on_hover_text(t("policy.local_only.tip"))
            .changed()
        {
            app.set_local_only(local);
        }
    }
    if app.agents.scan.is_none() && !app.agents.scanning() {
        app.agents.rescan();
    }
    let Some(scan) = app.agents.scan.clone() else {
        ui.add_space(T.space.s);
        c::secondary(ui, t("scan.running"));
        return;
    };
    let default = app.agents.cfg.default_connection.clone();
    let selected = app.agents.selected.clone();
    let conns = app.agents.cfg.connections.clone();
    let is_selected = |c: &Connection| selected.as_deref() == Some(c.id.as_str());
    let is_default = |c: &Connection| default.as_deref() == Some(c.id.as_str());
    let mut pick = None;

    // Agents: the CLI runtimes.
    let agents = show != Show::Providers;
    if agents {
        c::section(ui, t("prov.group.cli"));
    }
    for d in scan.runtimes.iter().filter(|_| agents) {
        let conn = conns.iter().find(|c| c.kind == d.kind);
        let name = format!("{} {}", d.kind.name(), short_version(&d.version));
        let (state, word, action) = match (conn, d.usable()) {
            (_, false) => (
                State::Warn,
                tf(
                    "prov.missing_options",
                    &[("options", &d.missing.join(", "))],
                ),
                None,
            ),
            (Some(c), true) => {
                let (state, word) = conn_state(app, c);
                let test = (!c.verified && !app.agents.cfg.local_only).then(|| t("prov.test"));
                (state, word, test)
            }
            (None, true) => (State::Idle, t("prov.found").into(), Some(t("prov.connect"))),
        };
        let star = conn.is_some_and(is_default);
        let sel = conn.is_some_and(is_selected);
        let (row, acted) = entry(ui, sel, state, &name, &word, star, action);
        match (conn, acted) {
            (Some(c), true) => app.agents.check_cli(&c.id),
            (None, true) => app.agents.connect_cli(d.kind),
            _ => {}
        }
        if row {
            pick = conn.map(|c| c.id.clone());
        }
    }
    for c in conns
        .iter()
        .filter(|c| agents && c.kind.is_cli() && !scan.runtimes.iter().any(|d| d.kind == c.kind))
    {
        let (row, _) = entry(
            ui,
            is_selected(c),
            State::Warn,
            &c.label,
            t("prov.cli_gone"),
            is_default(c),
            None,
        );
        if row {
            pick = Some(c.id.clone());
        }
    }
    for o in scan.others.iter().filter(|_| agents) {
        let name = format!("{} {}", o.name, short_version(&o.version));
        c::row(ui, false, &name, |ui| {
            c::status(ui, State::Idle, &name);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                c::meta(ui, t("prov.unsupported"));
            });
        })
        .on_hover_text(reason_label(o.reason));
    }
    if agents && scan.runtimes.is_empty() && scan.others.is_empty() {
        c::secondary(ui, t("prov.no_cli"));
    }
    if show == Show::Agents {
        if let Some(id) = pick {
            app.agents.selected = Some(id);
        }
        return;
    }

    // Local servers.
    c::section(ui, t("prov.group.local"));
    for s in &scan.servers {
        let conn = conns.iter().find(|c| c.base() == s.base);
        let word = tf("prov.models", &[("n", &s.models.len())]);
        let action = conn.is_none().then(|| t("prov.connect"));
        let (row, acted) = entry(
            ui,
            conn.is_some_and(is_selected),
            State::Ok,
            s.name,
            &word,
            conn.is_some_and(is_default),
            action,
        );
        if acted {
            app.agents.connect_server(s);
        }
        if row {
            pick = conn.map(|c| c.id.clone());
        }
    }
    for c in conns.iter().filter(|c| {
        !c.kind.is_cli() && c.is_local() && !scan.servers.iter().any(|s| s.base == c.base())
    }) {
        let (row, _) = entry(
            ui,
            is_selected(c),
            State::Idle,
            &c.label,
            t("prov.stopped"),
            is_default(c),
            None,
        );
        if row {
            pick = Some(c.id.clone());
        }
    }
    if scan.servers.is_empty() {
        c::secondary(ui, t("prov.no_server"));
    }

    // Cloud: the API connections, then the environment keys not used yet.
    c::section(ui, t("prov.group.cloud"));
    for c in conns.iter().filter(|c| !c.kind.is_cli() && !c.is_local()) {
        let (state, word) = conn_state(app, c);
        let (row, _) = entry(
            ui,
            is_selected(c),
            state,
            &c.label,
            &word,
            is_default(c),
            None,
        );
        if row {
            pick = Some(c.id.clone());
        }
    }
    for key in &scan.keys {
        let used = key.preset.is_some_and(|p| {
            conns
                .iter()
                .any(|c| c.kind == p.kind() && c.base_url == p.base())
        });
        if used {
            continue;
        }
        let name = preset_label(key.preset);
        let found = tf("scan.key", &[("var", &key.var)]);
        match key.preset {
            Some(_) => {
                let blocked = app.agents.cfg.local_only;
                let action = (!blocked).then(|| t("prov.use"));
                let (_, acted) = entry(ui, false, State::Idle, name, &found, false, action);
                if acted {
                    app.agents.use_env_key(key);
                }
            }
            None => {
                c::row(ui, false, key.var, |ui| {
                    c::status(ui, State::Idle, key.var);
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        c::meta(ui, t("prov.custom_only"));
                    });
                });
            }
        }
    }
    if let Some(id) = pick {
        app.agents.selected = Some(id);
        app.agents.key_input.clear();
    }

    c::section(ui, t("prov.add"));
    let width = ui.available_width();
    let hub = &mut app.agents;
    let mut preset = hub.new_preset;
    c::select(ui, "new-preset", preset_label(preset), width, |ui| {
        for p in [
            Some(Preset::Anthropic),
            Some(Preset::OpenAi),
            Some(Preset::OpenRouter),
            None,
        ] {
            c::option(ui, &mut preset, p, preset_label(p));
        }
    });
    hub.new_preset = preset;
    ui.add_space(T.space.xs);
    c::field(ui, &mut hub.new_label, t("conn.label.hint"), width);
    if preset.is_none() {
        ui.add_space(T.space.xs);
        c::field_mono(ui, &mut hub.new_url, t("conn.url.hint"), width);
    }
    ui.add_space(T.space.xs);
    if c::button(ui, t("conn.add.action"), ButtonKind::Secondary).clicked() {
        hub.add_connection();
    }
}

/// The connection picked: default, sign-in or key, models, prices, consent, removal.
pub(crate) fn provider_detail(ui: &mut Ui, app: &mut SurgeonApp) {
    let ctx = ui.ctx().clone();
    let width = ui.available_width();
    let local_only = app.agents.cfg.local_only;
    let hub = &mut app.agents;
    let Some(conn) = hub
        .selected
        .clone()
        .and_then(|id| hub.cfg.connection(&id).cloned())
    else {
        c::empty(ui, Icon::Chat, t("prov.pick"), t("prov.pick.hint"));
        message(ui, hub);
        return;
    };
    let id = conn.id.clone();
    c::strong(ui, conn.label.as_str());
    c::meta(ui, conn.kind.name());
    if !conn.kind.is_cli() {
        c::mono_secondary(ui, conn.base());
    }
    ui.add_space(T.space.s);
    if hub.cfg.default_connection.as_deref() == Some(id.as_str()) {
        c::status(ui, State::Ok, t("prov.is_default"));
    } else if c::button_if(
        ui,
        conn.ready(),
        t("prov.make_default"),
        ButtonKind::Secondary,
        t("prov.make_default.why"),
    )
    .clicked()
    {
        hub.set_default(&id);
    }

    if conn.kind.is_cli() {
        c::section(ui, t("prov.signin"));
        let (state, word) = match conn.verified {
            true => (State::Ok, t("prov.ready")),
            false => (State::Idle, t("prov.to_test")),
        };
        c::status(ui, state, word);
        if hub.checking_cli(&id) {
            c::busy(ui, t("conn.testing"));
        } else if c::button_if(
            ui,
            !local_only,
            t("prov.test"),
            ButtonKind::Secondary,
            t("prov.test.why"),
        )
        .on_hover_text(t("prov.test.cli_tip"))
        .clicked()
        {
            hub.check_cli(&id);
        }
        if let Some((_, command)) = hub.login.clone().filter(|(k, _)| *k == id) {
            ui.add_space(T.space.s);
            c::secondary(ui, t("prov.login"));
            c::mono_wrapped(ui, command, T.color.text);
            if c::button(ui, t("prov.copy"), ButtonKind::Ghost).clicked() {
                ctx.copy_text(command.to_owned());
            }
        }
    } else {
        if !conn.is_local() {
            c::section(ui, t("prov.key"));
            match hub.key_tail(&id) {
                Some(tail) => {
                    c::status(ui, State::Ok, tf("prov.key.stored", &[("tail", &tail)]));
                    if c::button(ui, t("conn.key.revoke"), ButtonKind::Ghost).clicked() {
                        hub.revoke_key(&id);
                    }
                }
                None => {
                    c::status(ui, State::Warn, t("prov.key.missing"));
                }
            }
            c::field_secret(ui, &mut hub.key_input, t("conn.key.hint"), width);
            ui.add_space(T.space.xs);
            if hub.trying_key(&id) {
                c::busy(ui, t("conn.testing"));
            } else if c::button_if(
                ui,
                !hub.key_input.trim().is_empty(),
                t("prov.key.try"),
                ButtonKind::Secondary,
                t("prov.key.try.why"),
            )
            .clicked()
            {
                hub.try_typed_key(&id);
            }
        }
        c::section(ui, t("prov.models.title"));
        if hub.testing(&id) {
            c::busy(ui, t("conn.testing"));
        } else if c::button(ui, t("prov.test"), ButtonKind::Secondary).clicked() {
            hub.test_models(&conn);
        }
        if !conn.models.is_empty() {
            ui.add_space(T.space.xs);
            let current = conn.inherited_model().unwrap_or_default().to_owned();
            let mut picked = current.clone();
            c::select(ui, "default-model", &current, width, |ui| {
                for m in &conn.models {
                    c::option(ui, &mut picked, m.clone(), m);
                }
            });
            if picked != current {
                hub.set_default_model(&id, &picked);
            }
        }
        c::section(ui, t("conn.prices"));
        hub.price_model = conn.inherited_model().unwrap_or_default().to_owned();
        ui.horizontal(|ui| {
            let w = (ui.available_width() - 2.0 * T.space.s) / 3.0;
            c::field_mono(ui, &mut hub.price_in, t("conn.price.in"), w);
            c::field_mono(ui, &mut hub.price_out, t("conn.price.out"), w);
            if c::button(ui, t("conn.price.save"), ButtonKind::Ghost).clicked() {
                hub.set_price(&id);
            }
        });
    }
    ui.add_space(T.space.m);
    if hub.cfg.consent(&id).is_some() {
        c::status(ui, State::Ok, t("conn.consented"));
        if c::button(ui, t("conn.consent.withdraw"), ButtonKind::Ghost).clicked() {
            hub.withdraw_consent(&id);
        }
    }
    if c::button(ui, t("conn.delete"), ButtonKind::Ghost).clicked() {
        hub.delete_connection(&id);
    }
    message(ui, hub);
}

fn message(ui: &mut Ui, hub: &crate::agent::view::Hub) {
    if let Some((text, error)) = &hub.message {
        ui.add_space(T.space.s);
        c::message(ui, text, *error);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_cli_version_keeps_its_number() {
        assert_eq!(super::short_version("2.1.3 (Claude Code)"), "2.1.3");
        assert_eq!(super::short_version("codex-cli 0.166.0"), "0.166.0");
        assert_eq!(super::short_version("opencode v2.2.7"), "v2.2.7");
        assert_eq!(super::short_version("weird"), "weird");
    }

    #[test]
    fn log_stamps_read_as_a_utc_clock() {
        assert_eq!(super::clock("86399 INFO x"), "23:59:59 INFO x");
        assert_eq!(super::clock("3661 WARN y z"), "01:01:01 WARN y z");
        assert_eq!(super::clock("not a stamp"), "not a stamp");
    }
}
