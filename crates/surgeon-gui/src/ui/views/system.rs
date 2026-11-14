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
use crate::agent::config::Connection;
use crate::agent::detect::{Preset, Reason};
use crate::app::{Section, SurgeonApp};
use crate::desktop::UpdateState;
use crate::diag::{actions_needed, Fix, Group, Row, Severity, Status, Uses};
use weight_surgeon::bench::score::Backend;

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
        Section::Providers => providers(ui, app, regions),
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
    let rows = app.desktop.rows(app.dialog.saved(), uses(app));
    let mut fix = None;
    c::Panel::new("dependencies")
        .title(t("panel.dependencies"))
        .show(ui, regions.center, |ui| {
            readiness(ui, &rows);
            ScrollArea::vertical()
                .id_salt("deps")
                .auto_shrink(false)
                .show(ui, |ui| {
                    for group in [
                        Group::Essential,
                        Group::Acceleration,
                        Group::Agents,
                        Group::Samples,
                    ] {
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

/// The active choices the severities follow: the backend picked, a CLI runtime configured.
fn uses(app: &SurgeonApp) -> Uses {
    Uses {
        llama_backend: app.dialog.saved().backend == Some(Backend::LlamaServer),
        cli: app
            .agents
            .cfg
            .connections
            .iter()
            .find(|c| c.kind.is_cli())
            .map(|c| c.kind.name()),
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
        Group::Agents => t("deps.group.agents"),
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
    let rows = app.desktop.rows(app.dialog.saved(), uses(app));
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

// ------------------------------------------------------------------ AI providers

fn reason_label(reason: Reason) -> &'static str {
    match reason {
        Reason::NoToolBarrier => t("scan.reason.barrier"),
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
    if app.agents.scan.is_none() && !app.agents.scanning() {
        app.agents.rescan();
    }
    c::Panel::new("providers")
        .title(t("panel.providers"))
        .show_with(
            ui,
            regions.center,
            app,
            |ui, app| {
                let scanning = app.agents.scanning();
                let label = if scanning {
                    t("scan.running")
                } else {
                    t("scan.again")
                };
                if c::button_if(ui, !scanning, label, ButtonKind::Ghost, t("scan.running"))
                    .clicked()
                {
                    app.agents.rescan();
                }
            },
            |ui, app| {
                ScrollArea::vertical()
                    .id_salt("providers")
                    .auto_shrink(false)
                    .show(ui, |ui| provider_list(ui, app));
            },
        );
    c::Panel::new("provider")
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

fn provider_list(ui: &mut Ui, app: &mut SurgeonApp) {
    let mut local = app.desktop.settings.local_only;
    if c::checkbox(ui, &mut local, t("policy.local_only"))
        .on_hover_text(t("policy.local_only.tip"))
        .changed()
    {
        app.set_local_only(local);
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
    c::section(ui, t("prov.group.cli"));
    for d in &scan.runtimes {
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
        .filter(|c| c.kind.is_cli() && !scan.runtimes.iter().any(|d| d.kind == c.kind))
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
    for o in &scan.others {
        let name = format!("{} {}", o.name, short_version(&o.version));
        c::row(ui, false, &name, |ui| {
            c::status(ui, State::Idle, &name);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                c::meta(ui, t("prov.unsupported"));
            });
        })
        .on_hover_text(reason_label(o.reason));
    }
    if scan.runtimes.is_empty() && scan.others.is_empty() {
        c::secondary(ui, t("prov.no_cli"));
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
fn provider_detail(ui: &mut Ui, app: &mut SurgeonApp) {
    let ctx = ui.ctx().clone();
    let width = ui.available_width();
    let local_only = app.agents.cfg.local_only;
    let hub = &mut app.agents;
    let Some(conn) = hub
        .selected
        .clone()
        .and_then(|id| hub.cfg.connection(&id).cloned())
    else {
        c::empty(ui, t("prov.pick"), t("prov.pick.hint"));
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
        let checking = hub.checking_cli(&id);
        if c::button_if(
            ui,
            !checking && !local_only,
            if checking {
                t("conn.testing")
            } else {
                t("prov.test")
            },
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
            ui.horizontal(|ui| {
                c::mono(ui, command);
                if c::button(ui, t("prov.copy"), ButtonKind::Ghost).clicked() {
                    ctx.copy_text(command.to_owned());
                }
            });
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
            let trying = hub.trying_key(&id);
            if c::button_if(
                ui,
                !trying && !hub.key_input.trim().is_empty(),
                if trying {
                    t("conn.testing")
                } else {
                    t("prov.key.try")
                },
                ButtonKind::Secondary,
                t("prov.key.try.why"),
            )
            .clicked()
            {
                hub.try_typed_key(&id);
            }
        }
        c::section(ui, t("prov.models.title"));
        let testing = hub.testing(&id);
        if c::button_if(
            ui,
            !testing,
            if testing {
                t("conn.testing")
            } else {
                t("prov.test")
            },
            ButtonKind::Secondary,
            t("conn.testing"),
        )
        .clicked()
        {
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
