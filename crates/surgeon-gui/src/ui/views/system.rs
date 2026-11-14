//! System: the one place where the environment is configured. Four sections on the left
//! (with the wizard to replay); machine and dependencies (with their repair), the AI
//! providers (keys in the vault, models, prices, CLI runtimes), the updates, and the
//! technical log with the diagnostics report (no personal path, no secret).

use eframe::egui::{self, Rect, ScrollArea, Ui};

use super::super::components::{self as c, ButtonKind};
use super::super::i18n::{t, tf};
use super::super::icons::State;
use super::super::layout::Regions;
use super::super::theme::TOKENS as T;
use super::left_column;
use crate::agent::config::ProviderKind;
use crate::app::{Section, SurgeonApp};
use crate::desktop::UpdateState;
use crate::diag::{Fix, Status};

fn section_label(section: Section) -> &'static str {
    match section {
        Section::Machine => t("system.section.machine"),
        Section::Providers => t("system.section.providers"),
        Section::Updates => t("system.section.updates"),
        Section::Journal => t("system.section.journal"),
    }
}

pub fn show(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    left_column(
        ui,
        app,
        regions.left,
        "sections",
        t("panel.sections"),
        |ui, app| {
            for section in Section::ALL {
                let label = section_label(section);
                if c::row(ui, app.system_section == section, label, |ui| {
                    c::label(ui, label);
                })
                .clicked()
                {
                    app.system_section = section;
                }
            }
            c::separator(ui);
            if c::button(ui, t("system.wizard"), ButtonKind::Secondary)
                .on_hover_text(t("system.wizard.tip"))
                .clicked()
            {
                app.desktop.open_wizard();
            }
        },
    );
    // A section without a side panel takes the center and the right column.
    let wide = Rect::from_min_max(regions.center.min, regions.right.max);
    match app.system_section {
        Section::Machine => machine(ui, app, regions),
        Section::Providers => c::Panel::new("connections")
            .title(t("panel.connections"))
            .show(ui, wide, |ui| connections(ui, app)),
        Section::Updates => {
            c::Panel::new("updates")
                .title(t("panel.updates"))
                .show(ui, wide, |ui| updates(ui, app))
        }
        Section::Journal => journal(ui, app, regions),
    }
}

/// The dependencies in the center, the machine on the right.
fn machine(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    let ctx = ui.ctx().clone();
    c::Panel::new("hardware")
        .title(t("panel.hardware"))
        .show_with(
            ui,
            regions.right,
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
        .show(ui, regions.center, |ui| {
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

/// The technical log in the center, the diagnostics report on the right.
fn journal(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    c::Panel::new("journal")
        .title(t("panel.journal"))
        .show(ui, regions.center, |ui| {
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

// ------------------------------------------------------------------ connections

fn kind_label(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Anthropic => t("provider.anthropic"),
        ProviderKind::OpenAi => t("provider.openai"),
        ProviderKind::Compatible => t("provider.compatible"),
        ProviderKind::ClaudeCode => t("provider.claude_code"),
        ProviderKind::Codex => t("provider.codex"),
    }
}

fn connections(ui: &mut Ui, app: &mut SurgeonApp) {
    let width = ui.available_width();
    ScrollArea::vertical()
        .id_salt("connections")
        .auto_shrink(false)
        .show(ui, |ui| {
            let hub = &mut app.agents;
            if hub.cfg.connections.is_empty() {
                c::empty(ui, t("conn.empty"), t("conn.empty.hint"));
            }
            let mut pick = None;
            for conn in &hub.cfg.connections {
                let consented = hub.cfg.consent(&conn.id).is_some();
                let used = hub.agent.connection.as_deref() == Some(conn.id.as_str());
                let response = c::row(
                    ui,
                    hub.selected.as_deref() == Some(conn.id.as_str()),
                    &conn.label,
                    |ui| {
                        let (state, word) = match (consented, conn.is_local()) {
                            (_, true) => (State::Ok, t("conn.local")),
                            (true, false) => (State::Ok, t("conn.consented")),
                            (false, false) => (State::Idle, t("conn.no_consent")),
                        };
                        c::status(ui, state, conn.label.as_str()).on_hover_text(word);
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if used {
                                c::meta(ui, t("conn.used"));
                            }
                        });
                    },
                );
                if response.clicked() {
                    pick = Some(conn.id.clone());
                }
            }
            if let Some(id) = pick {
                hub.selected = Some(id);
            }
            if let Some(id) = hub.selected.clone() {
                if let Some(conn) = hub.cfg.connection(&id).cloned() {
                    c::separator(ui);
                    c::meta(ui, kind_label(conn.kind));
                    c::mono_secondary(ui, conn.base());
                    if conn.kind.needs_key() {
                        let stored = hub.store.get(&id).ok().flatten().is_some();
                        let (state, word) = match stored {
                            true => (State::Ok, t("conn.key.stored")),
                            false => (State::Warn, t("conn.key.missing")),
                        };
                        c::status(ui, state, word);
                        ui.horizontal(|ui| {
                            let w = ui.available_width() - 2.0 * T.size.row * 3.0;
                            c::field_secret(
                                ui,
                                &mut hub.key_input,
                                t("conn.key.hint"),
                                w.max(T.size.row * 4.0),
                            );
                            if c::button(ui, t("conn.key.save"), ButtonKind::Secondary).clicked() {
                                hub.save_key(&id);
                            }
                        });
                        if stored
                            && c::button(ui, t("conn.key.revoke"), ButtonKind::Secondary).clicked()
                        {
                            hub.revoke_key(&id);
                        }
                    }
                    ui.add_space(T.space.s);
                    let testing = hub.testing(&id);
                    let label = if testing {
                        t("conn.testing")
                    } else {
                        t("conn.test")
                    };
                    if c::button_if(
                        ui,
                        !testing,
                        label,
                        ButtonKind::Secondary,
                        t("conn.testing"),
                    )
                    .clicked()
                    {
                        hub.test_models(&conn);
                    }
                    if !conn.models.is_empty() {
                        let current = hub.agent.model.clone().unwrap_or_default();
                        let mut picked = current.clone();
                        let shown = if current.is_empty() {
                            t("conn.model.pick")
                        } else {
                            current.as_str()
                        };
                        c::select(ui, "model-select", shown, width, |ui| {
                            for m in &conn.models {
                                c::option(ui, &mut picked, m.clone(), m);
                            }
                        });
                        if picked != current {
                            hub.agent.connection = Some(id.clone());
                            hub.agent.model = Some(picked.clone());
                            hub.price_model = picked;
                            hub.agent_changed();
                        }
                    }
                    if !conn.kind.is_cli() {
                        c::section(ui, t("conn.prices"));
                        ui.horizontal(|ui| {
                            let w = (ui.available_width() - 2.0 * T.space.s) / 3.0;
                            c::field_mono(ui, &mut hub.price_in, t("conn.price.in"), w);
                            c::field_mono(ui, &mut hub.price_out, t("conn.price.out"), w);
                            if c::button(ui, t("conn.price.save"), ButtonKind::Ghost).clicked() {
                                hub.set_price(&id);
                            }
                        });
                    }
                    ui.horizontal(|ui| {
                        if hub.cfg.consent(&id).is_some()
                            && c::button(ui, t("conn.consent.withdraw"), ButtonKind::Secondary)
                                .clicked()
                        {
                            hub.withdraw_consent(&id);
                        }
                        if c::button(ui, t("conn.delete"), ButtonKind::Ghost).clicked() {
                            hub.delete_connection(&id);
                        }
                    });
                }
            }
            c::separator(ui);
            c::section(ui, t("conn.add"));
            let mut kind = hub.new_kind;
            c::select(ui, "new-kind", kind_label(kind), width, |ui| {
                for k in [
                    ProviderKind::Anthropic,
                    ProviderKind::OpenAi,
                    ProviderKind::Compatible,
                    ProviderKind::ClaudeCode,
                    ProviderKind::Codex,
                ] {
                    c::option(ui, &mut kind, k, kind_label(k));
                }
            });
            hub.new_kind = kind;
            ui.add_space(T.space.xs);
            c::field(ui, &mut hub.new_label, t("conn.label.hint"), width);
            if kind == ProviderKind::Compatible {
                ui.add_space(T.space.xs);
                c::field_mono(ui, &mut hub.new_url, t("conn.url.hint"), width);
            }
            ui.add_space(T.space.xs);
            ui.horizontal(|ui| {
                if c::button(ui, t("conn.add.action"), ButtonKind::Secondary).clicked() {
                    hub.add_connection();
                }
                let detecting = hub.detecting();
                if c::button_if(
                    ui,
                    !detecting,
                    t("conn.detect"),
                    ButtonKind::Ghost,
                    t("conn.detecting"),
                )
                .clicked()
                {
                    hub.detect_runtimes();
                }
            });
            for d in &hub.detected {
                let state = if d.usable() { State::Ok } else { State::Warn };
                c::status(ui, state, format!("{} {}", kind_label(d.kind), d.version));
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
