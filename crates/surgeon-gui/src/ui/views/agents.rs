//! The agent, the main view: the project and its sessions on the left; in the center either
//! the new-session form (`Advance`, a full scan where the agent finds the problems itself, or
//! `Scalping`, a precise symptom the user describes) or the session itself (the thread with
//! its approval cards and question, the message field); on the right the evidence (what the
//! agent established). What the agent may do is chosen in the form; connections are
//! configured in the settings. The consent dialog opens before the first request to a remote connection.

use std::collections::{HashMap, HashSet};

use eframe::egui::{self, Id, Order, Pos2, Rect, ScrollArea, Ui, Vec2};

use super::super::components::{self as c, ButtonKind, Segment};
use super::super::i18n::{t, tf};
use super::super::icons::{Icon, State};
use super::super::layout::Regions;
use super::super::theme::TOKENS as T;
use super::{age, context, left_column, project_select};
use crate::agent::evidence::{self, Status};
use crate::agent::mission::{Mission, Operation, Symptom, Tensors};
use crate::agent::policy::AgentMode;
use crate::agent::provider::{Block, Role};
use crate::agent::session::{Choice, Dirs};
use crate::app::{Mode, RightPane, Section, SurgeonApp, Tab};
use crate::studio::project::now_unix;

pub fn show(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    left_column(
        ui,
        app,
        regions.left,
        "sessions",
        t("panel.sessions"),
        sessions,
    );
    c::Panel::new("session").center().flush().show_with(
        ui,
        regions.center,
        app,
        |_, _| {},
        |ui, app| {
            let rect = ui.max_rect();
            match (
                app.agents.project.is_some(),
                app.agents.conversation().is_some(),
            ) {
                (false, _) => no_project(ui, app, rect),
                (true, false) => form(ui, app, rect),
                (true, true) => thread(ui, app, rect),
            }
        },
    );
    let title = match app.right {
        RightPane::Evidence => t("panel.evidence"),
        RightPane::Results => t("panel.results"),
    };
    c::Panel::new("evidence").title(title).show_with(
        ui,
        regions.right,
        app,
        |ui, app| {
            let panes = [
                (RightPane::Evidence, t("panel.evidence")),
                (RightPane::Results, t("panel.results")),
            ];
            c::segmented_small(ui, &mut app.right, &panes, t("pane.tip"));
        },
        |ui, app| match app.right {
            RightPane::Evidence => evidence_log(ui, app),
            RightPane::Results => super::results::show(ui, app),
        },
    );
    consent(ui.ctx(), app);
}

/// Opens the settings on the AI providers.
fn configure(app: &mut SurgeonApp) {
    app.settings = Some(Section::Providers);
}

/// The centered reading column of the center zone.
fn column(rect: Rect) -> Rect {
    let width = (rect.width() - 2.0 * T.space.xl).min(T.size.reading);
    Rect::from_center_size(rect.center(), Vec2::new(width, rect.height()))
}

fn symptom_label(symptom: Symptom) -> &'static str {
    match symptom {
        Symptom::Refusals => t("symptom.refusals"),
        Symptom::Repetition => t("symptom.repetition"),
        Symptom::Hallucination => t("symptom.hallucination"),
        Symptom::Language => t("symptom.language"),
        Symptom::Reasoning => t("symptom.reasoning"),
        Symptom::Other => t("symptom.other"),
    }
}

fn symptom_icon(symptom: Symptom) -> Icon {
    match symptom {
        Symptom::Refusals => Icon::Shield,
        Symptom::Repetition => Icon::List,
        Symptom::Hallucination => Icon::Sparkle,
        Symptom::Language => Icon::Chat,
        Symptom::Reasoning => Icon::Chip,
        Symptom::Other => Icon::Plus,
    }
}

/// An example of what the user may write for `symptom`.
fn symptom_idea(symptom: Symptom) -> &'static str {
    match symptom {
        Symptom::Refusals => t("idea.refusals"),
        Symptom::Repetition => t("idea.repetition"),
        Symptom::Hallucination => t("idea.hallucination"),
        Symptom::Language => t("idea.language"),
        Symptom::Reasoning => t("idea.reasoning"),
        Symptom::Other => t("idea.other"),
    }
}

fn mission_label(mission: Option<Mission>) -> String {
    match mission {
        Some(Mission::Advance) => t("mode.advance").into(),
        Some(Mission::Scalping(s)) => format!("{} · {}", t("mode.scalping"), symptom_label(s)),
        None => t("panel.session").into(),
    }
}

// ------------------------------------------------------------------ left column

/// New session, the project and its copy, then the project's sessions.
fn sessions(ui: &mut Ui, app: &mut SurgeonApp) {
    // Without a project the center holds the form that creates one.
    if app.agents.project.is_none() {
        c::empty(ui, Icon::Plus, t("project.empty"), t("project.empty.hint"));
        return;
    }
    let busy = app.agents.busy();
    let new = ui.add_enabled_ui(!busy, |ui| {
        c::button_wide(ui, t("chat.new"), ButtonKind::Secondary)
    });
    if new.inner.on_disabled_hover_text(t("chat.busy")).clicked() {
        app.agents.new_conversation();
    }
    c::section(ui, t("panel.project"));
    context(ui, app);
    c::section(ui, t("panel.sessions"));
    let current = app.agents.conversation().map(|c| c.id.clone());
    let now = now_unix();
    let lines = app.agents.conversations().to_vec();
    if lines.is_empty() {
        c::empty(
            ui,
            Icon::Chat,
            t("chat.list.empty"),
            t("chat.list.empty.hint"),
        );
        return;
    }
    let mut open = None;
    ScrollArea::vertical()
        .id_salt("sessions")
        .auto_shrink(false)
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing.y = T.space.xs / 2.0;
            for (id, title, created) in &lines {
                let selected = current.as_deref() == Some(id.as_str());
                // Right to left: the age keeps its room, the title takes the rest.
                let row = c::row(ui, selected, title, |ui| {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        c::meta(ui, age(now.saturating_sub(*created)));
                        c::line(ui, title.as_str());
                    });
                });
                if row.on_hover_text(id).clicked() && !selected {
                    open = Some(id.clone());
                }
            }
        });
    if let Some(id) = open {
        app.agents.open_conversation(&id);
    }
}

// ------------------------------------------------------------------ center

/// Without a project: what to do first, and the form that creates one.
fn no_project(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect) {
    let col = column(rect).shrink2(Vec2::new(0.0, T.space.xl * 2.0));
    c::region(ui, col, "no-project", |ui| {
        c::heading(ui, t("start.title"), t("start.lead"));
        ui.add_space(T.space.xl);
        c::card_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            context(ui, app);
        });
    });
}

/// The form of a new session: the kind, the model, the symptom (`Scalping`), the words of
/// the user, the connection and what the agent may do, then Start.
fn form(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect) {
    let col = column(rect).shrink2(Vec2::new(0.0, T.space.xl));
    c::region(ui, col, "session-form", |ui| {
        ScrollArea::vertical()
            .id_salt("session-form")
            .auto_shrink(false)
            .show(ui, |ui| {
                let mut mode = app.mode;
                let kinds = [
                    Segment {
                        value: Mode::Advance,
                        label: t("mode.advance"),
                        tooltip: t("mode.advance.tip"),
                    },
                    Segment {
                        value: Mode::Scalping,
                        label: t("mode.scalping"),
                        tooltip: t("mode.scalping.tip"),
                    },
                ];
                if c::segmented(ui, &mut mode, &kinds, c::font_body()) {
                    app.set_mode(mode);
                }
                ui.add_space(T.space.xl);
                match mode {
                    Mode::Advance => {
                        c::heading(ui, t("form.advance.title"), t("form.advance.lead"))
                    }
                    Mode::Scalping => {
                        c::heading(ui, t("form.scalping.title"), t("form.scalping.lead"))
                    }
                }
                ui.add_space(T.space.xl);
                c::card_frame().show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    form_steps(ui, app, mode);
                });
            });
    });
}

fn form_steps(ui: &mut Ui, app: &mut SurgeonApp, mode: Mode) {
    c::step(ui, 1, t("form.model"));
    ui.add_space(T.space.m);
    project_select(ui, app, ui.available_width());
    ui.add_space(T.space.s);
    copy_line(ui, app);
    ui.add_space(T.space.xl);

    let mut n = 2;
    if mode == Mode::Scalping {
        c::step(ui, n, t("form.symptom"));
        ui.add_space(T.space.m);
        let width = ((ui.available_width() - 2.0 * T.space.s) / 3.0).floor();
        for row in Symptom::ALL.chunks(3) {
            ui.horizontal(|ui| {
                for &symptom in row {
                    let picked = app.agents.symptom == symptom;
                    let label = symptom_label(symptom);
                    if c::tile(ui, symptom_icon(symptom), label, picked, width).clicked() {
                        app.agents.symptom = symptom;
                    }
                }
            });
            ui.add_space(T.space.s);
        }
        ui.add_space(T.space.l);
        n += 1;
    }

    let (title, hint) = match mode {
        Mode::Advance => (t("form.notes"), t("form.notes.hint")),
        Mode::Scalping => (t("form.describe"), t("form.describe.hint")),
    };
    c::step(ui, n, title);
    ui.add_space(T.space.m);
    c::field_multiline(ui, &mut app.agents.input, hint, 4);
    ui.add_space(T.space.s);
    let ideas: Vec<&str> = match mode {
        Mode::Advance => vec![t("idea.advance.layers"), t("idea.advance.origin")],
        Mode::Scalping => vec![symptom_idea(app.agents.symptom)],
    };
    ui.horizontal_wrapped(|ui| {
        c::secondary(ui, t("form.ideas"));
        for idea in ideas {
            if c::chip(ui, idea).clicked() {
                app.agents.input = idea.to_owned();
            }
        }
    });
    ui.add_space(T.space.xl);
    advanced(ui, app);
    ui.add_space(T.space.xl);

    ui.horizontal(|ui| {
        c::secondary(ui, t("form.allow"));
        let mut mode = app.agents.agent.mode;
        let options = [
            Segment {
                value: AgentMode::Ask,
                label: t("policy.ask"),
                tooltip: t("policy.ask.tip"),
            },
            Segment {
                value: AgentMode::Modify,
                label: t("policy.modify"),
                tooltip: t("policy.modify.tip"),
            },
        ];
        if c::segmented(ui, &mut mode, &options, c::font_body()) {
            app.agents.agent.mode = mode;
            app.agents.agent_changed();
        }
    });
    ui.add_space(T.space.m);
    ui.horizontal(|ui| {
        c::secondary(ui, t("form.by"));
        if app.agents.connection().is_some() {
            picker(ui, app, T.size.left);
        } else if c::button(ui, t("picker.configure"), ButtonKind::Secondary).clicked() {
            configure(app);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            start(ui, app, mode);
        });
    });
    if let Some((text, error)) = app.agents.message.clone() {
        ui.add_space(T.space.s);
        c::message(ui, &text, error);
    }
}

/// The advanced (CLI) panel: what the agent must write in its patch instead of choosing it,
/// as in a `patch.json` of the CLI. Closed, the agent follows the protocol alone.
fn advanced(ui: &mut Ui, app: &mut SurgeonApp) {
    let hub = &mut app.agents;
    c::switch(
        ui,
        Icon::Sliders,
        t("form.advanced"),
        &mut hub.forced_on,
        t("form.advanced.tip"),
    );
    if !hub.forced_on {
        return;
    }
    ui.add_space(T.space.s);
    c::inset_frame().show(ui, |ui| {
        ui.set_width(ui.available_width());
        let forced = &mut hub.forced;
        let operations = [
            Segment {
                value: Operation::Auto,
                label: "auto",
                tooltip: t("forced.auto.tip"),
            },
            Segment {
                value: Operation::ProjectOut,
                label: "project_out",
                tooltip: t("forced.project_out.tip"),
            },
            Segment {
                value: Operation::Add,
                label: "add",
                tooltip: t("forced.add.tip"),
            },
        ];
        let tensors = [
            Segment {
                value: Tensors::Both,
                label: t("forced.both"),
                tooltip: t("forced.tensors.tip"),
            },
            Segment {
                value: Tensors::OProj,
                label: "o_proj",
                tooltip: t("forced.tensors.tip"),
            },
            Segment {
                value: Tensors::DownProj,
                label: "down_proj",
                tooltip: t("forced.tensors.tip"),
            },
        ];
        ui.horizontal(|ui| {
            c::secondary(ui, t("form.advanced.operation"));
            c::segmented(ui, &mut forced.operation, &operations, c::font_body());
        });
        ui.add_space(T.space.s);
        ui.horizontal(|ui| {
            c::secondary(ui, t("form.advanced.tensors"));
            c::segmented(ui, &mut forced.tensors, &tensors, c::font_body());
        });
        ui.add_space(T.space.s);
        let width = ((ui.available_width() - T.space.l) / 2.0).floor();
        ui.horizontal(|ui| {
            c::secondary(ui, t("form.advanced.layers"));
            c::field_mono(
                ui,
                &mut forced.layers,
                t("form.advanced.layers.hint"),
                width * 0.6,
            );
            c::secondary(ui, t("form.advanced.alpha"));
            c::field_mono(
                ui,
                &mut forced.alpha,
                t("form.advanced.alpha.hint"),
                width * 0.6,
            );
        });
        if let Err(e) = forced.check() {
            ui.add_space(T.space.s);
            c::message(ui, &e, true);
        }
    });
}

/// The working copy the agent works on, or the button that makes one.
fn copy_line(ui: &mut Ui, app: &mut SurgeonApp) {
    let entries = app.ledger().to_vec();
    let copy = app.studio.active_copy(&entries);
    ui.horizontal(|ui| match &copy {
        Some(copy) => {
            let (state, word) = match copy.writes {
                0 => (State::Ok, t("copy.intact")),
                _ => (State::Warn, t("copy.modified")),
            };
            c::status(ui, state, word);
            c::mono_secondary(ui, copy.path.as_str());
        }
        None => {
            c::status(ui, State::Idle, t("copy.none"));
            if c::button(ui, t("copy.new"), ButtonKind::Secondary)
                .on_hover_text(t("copy.new.tip"))
                .clicked()
            {
                let _ = app.studio.submit_copy();
            }
        }
    });
}

/// Start: the one primary action of the form.
fn start(ui: &mut Ui, app: &mut SurgeonApp, mode: Mode) {
    let hub = &app.agents;
    let described = mode == Mode::Advance || !hub.input.trim().is_empty();
    let copying = app.pending_start.is_some();
    let forced_ok = !hub.forced_on || hub.forced.check().is_ok();
    let why = match (hub.connection().is_some(), hub.busy() || copying, described) {
        (false, _, _) => Some(t("agents.none")),
        (_, true, _) if copying => Some(t("form.copying")),
        (_, true, _) => Some(t("chat.busy")),
        (_, _, false) => Some(t("form.describe.why")),
        _ if !forced_ok => Some(t("form.advanced.why")),
        _ => None,
    };
    if copying {
        c::status(ui, State::Active, t("form.copying"));
    }
    let clicked = c::button_if(
        ui,
        why.is_none(),
        t("form.start"),
        ButtonKind::Primary,
        why.unwrap_or_default(),
    )
    .clicked();
    if !clicked {
        return;
    }
    let mission = match mode {
        Mode::Advance => Mission::Advance,
        Mode::Scalping => Mission::Scalping(app.agents.symptom),
    };
    if app.agents.input.trim().is_empty() {
        app.agents.input = t("form.advance.default").to_owned();
    }
    app.start_session(mission);
    ui.ctx().request_repaint();
}

/// The provider, then its model: the provider's choice resets the model, so the model list
/// only ever holds what the active provider takes.
pub fn picker(ui: &mut Ui, app: &mut SurgeonApp, width: f32) {
    let (conn_w, model_w) = (width * 0.4, width * 0.6);
    // The title bar lays out right to left: the model goes first to land after its provider.
    match ui.layout().prefer_right_to_left() {
        true => {
            model_picker(ui, app, model_w);
            connection_picker(ui, app, conn_w);
        }
        false => {
            connection_picker(ui, app, conn_w);
            model_picker(ui, app, model_w);
        }
    }
}

/// The ready connections, the global default, and a way to the settings.
fn connection_picker(ui: &mut Ui, app: &mut SurgeonApp, width: f32) {
    #[derive(Clone, PartialEq)]
    enum Pick {
        /// The global default (no project choice).
        Default,
        Use(String),
        Configure,
    }
    let hub = &app.agents;
    let shown = hub
        .connection()
        .map_or_else(|| t("picker.none").to_owned(), |c| c.label.clone());
    let current = hub
        .agent
        .connection
        .clone()
        .map_or(Pick::Default, Pick::Use);
    let default = hub
        .cfg
        .default_connection
        .as_deref()
        .and_then(|id| hub.cfg.connection(id))
        .map(|c| c.label.clone());
    let ready: Vec<_> = hub
        .cfg
        .connections
        .iter()
        .filter(|c| c.ready())
        .map(|c| (c.id.clone(), c.label.clone()))
        .collect();
    let mut picked = current.clone();
    c::select(ui, "session-picker", &shown, width, |ui| {
        if let Some(label) = &default {
            let text = tf("picker.default", &[("connection", label)]);
            c::option(ui, &mut picked, Pick::Default, &text);
        }
        for (id, label) in ready {
            c::option(ui, &mut picked, Pick::Use(id), &label);
        }
        c::option(ui, &mut picked, Pick::Configure, t("picker.configure"));
    });
    if picked != current {
        match picked {
            Pick::Default => app.agents.pick_connection(None),
            Pick::Use(id) => app.agents.pick_connection(Some(id)),
            Pick::Configure => configure(app),
        }
    }
}

/// The models of the active connection, and a free field for a name it does not list.
fn model_picker(ui: &mut Ui, app: &mut SurgeonApp, width: f32) {
    let hub = &mut app.agents;
    let Some(choices) = hub.connection().map(|c| c.model_choices()) else {
        return;
    };
    // The free field lives in this place's memory: the title bar and the form each have one.
    let id = ui.id().with("custom-model");
    if let Some(mut typed) = ui.data(|d| d.get_temp::<String>(id)) {
        let field = c::field_mono(ui, &mut typed, t("picker.model.custom.hint"), width);
        if field.lost_focus() {
            ui.data_mut(|d| d.remove::<String>(id));
            if ui.input(|i| i.key_pressed(egui::Key::Enter)) && !typed.trim().is_empty() {
                hub.pick_model(typed.trim().to_owned());
            }
            return;
        }
        if !field.has_focus() {
            field.request_focus();
        }
        ui.data_mut(|d| d.insert_temp(id, typed));
        return;
    }
    let current = hub.model();
    let shown = current
        .clone()
        .unwrap_or_else(|| t("picker.model.default").to_owned());
    let mut picked = current.clone();
    let mut custom = false;
    c::select(ui, "model-picker", &shown, width, |ui| {
        for m in &choices {
            c::option(ui, &mut picked, Some(m.clone()), m);
        }
        c::option(ui, &mut custom, true, t("picker.model.custom"));
    });
    if custom {
        ui.data_mut(|d| d.insert_temp(id, current.unwrap_or_default()));
    } else if picked != current {
        if let Some(model) = picked {
            hub.pick_model(model);
        }
    }
}

/// The session: its header, the thread, the message field.
fn thread(ui: &mut Ui, app: &mut SurgeonApp, rect: Rect) {
    let col = column(rect);
    let header = Rect::from_min_size(col.min, Vec2::new(col.width(), T.size.header + T.space.s));
    let field = Rect::from_min_max(
        Pos2::new(col.left(), col.bottom() - T.space.l - c::composer_height()),
        Pos2::new(col.right(), col.bottom() - T.space.l),
    );
    let body = Rect::from_min_max(
        Pos2::new(col.left(), header.bottom()),
        Pos2::new(col.right(), field.top() - T.space.s),
    );
    c::region(ui, header, "session-header", |ui| session_header(ui, app));
    c::region(ui, body, "session-thread", |ui| {
        ScrollArea::vertical()
            .id_salt("session")
            .auto_shrink(false)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                transcript_body(ui, app);
                approval(ui, app);
            });
    });
    c::region(ui, field, "session-field", |ui| {
        let hub = &app.agents;
        let ready = !hub.busy() && !hub.input.trim().is_empty() && hub.connected();
        let note = hub.message.clone();
        if c::composer(
            ui,
            &mut app.agents.input,
            t("chat.hint"),
            t("chat.keys"),
            note.as_ref().map(|(text, error)| (text.as_str(), *error)),
            ready,
            t("chat.send"),
        ) {
            app.agents.send_input();
            ui.ctx().request_repaint();
        }
    });
}

/// The kind of session and its connection on the left; export, pause and stop on the right.
fn session_header(ui: &mut Ui, app: &mut SurgeonApp) {
    let mission = app.agents.conversation().and_then(|c| c.mission);
    let model = app
        .agents
        .conversation()
        .map(|c| match c.model.as_str() {
            "" => format!("{} · {}", c.connection, t("picker.model.default")),
            m => format!("{} · {m}", c.connection),
        })
        .unwrap_or_default();
    ui.horizontal_centered(|ui| {
        let icon = match mission {
            Some(Mission::Scalping(_)) => Icon::Target,
            _ => Icon::Sparkle,
        };
        let (glyph, _) = ui.allocate_exact_size(Vec2::splat(T.size.icon), egui::Sense::hover());
        super::super::icons::paint(ui.painter(), glyph, icon, T.color.text_secondary);
        c::strong(ui, mission_label(mission));
        c::meta(ui, model);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let busy = app.agents.busy();
            let paused = app.agents.session.as_ref().is_some_and(|s| s.paused);
            if busy || paused {
                if c::icon_button(ui, Icon::Stop, t("chat.stop")).clicked() {
                    app.agents.stop(&mut app.studio);
                }
                match paused {
                    true => {
                        if c::icon_button(ui, Icon::Play, t("chat.resume")).clicked() {
                            app.agents.resume_run(&mut app.studio);
                        }
                    }
                    false => {
                        if c::icon_button(ui, Icon::Pause, t("chat.pause")).clicked() {
                            app.agents.pause();
                        }
                    }
                }
            }
            if let Some(project) = app.agents.project.clone() {
                if c::button(ui, t("chat.export"), ButtonKind::Ghost).clicked() {
                    app.agents.export(&project);
                }
            }
            if let Some(target) = app.focus.clone() {
                let text = tf("focus.show", &[("target", &target)]);
                if c::button(ui, &text, ButtonKind::Ghost)
                    .on_hover_text(t("focus.show.tip"))
                    .clicked()
                {
                    app.tab = Tab::Graph;
                }
            }
        });
    });
}

fn transcript_body(ui: &mut Ui, app: &SurgeonApp) {
    let Some(conv) = app.agents.conversation() else {
        return;
    };
    let configured = app.dialog.saved().backend;
    let blocks = || conv.messages.iter().flat_map(|m| &m.blocks);
    // A result is shown inside its call's row, not as a row of its own.
    let results: HashMap<&str, (&str, bool)> = blocks()
        .filter_map(|b| match b {
            Block::ToolResult {
                id,
                content,
                is_error,
            } => Some((id.as_str(), (content.as_str(), *is_error))),
            _ => None,
        })
        .collect();
    let calls: HashSet<&str> = blocks()
        .filter_map(|b| match b {
            Block::ToolUse { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect();
    for (n, message) in conv.messages.iter().enumerate() {
        let mut named = false;
        for (k, block) in message.blocks.iter().enumerate() {
            let row_id = Id::new(("tool", &conv.id, n, k));
            match (block, message.role) {
                (Block::Text { text }, Role::User) => {
                    c::bubble(ui, text);
                    ui.add_space(T.space.m);
                }
                (Block::Text { text }, Role::Assistant) => {
                    if !named {
                        c::meta(ui, t("chat.agent"));
                        named = true;
                    }
                    c::label(ui, text.as_str());
                    ui.add_space(T.space.m);
                }
                (Block::ToolUse { id, name, input }, _) => {
                    let result = results.get(id.as_str());
                    // shortcut: read with today's backend configuration, not the one of the
                    // call; record it with the call if the configuration changes mid-project.
                    let project = app.agents.project.as_deref();
                    let cli = weight_surgeon::mcp::cli_line(name, input, configured, project);
                    let pretty = serde_json::to_string_pretty(input).unwrap_or_default();
                    let mut details: Vec<&str> = cli.iter().map(|l| l.command.as_str()).collect();
                    details.push(pretty.as_str());
                    details.extend(result.map(|(content, _)| *content));
                    let failed = result.is_some_and(|(_, error)| *error);
                    let summary = cli
                        .as_ref()
                        .map_or_else(|| tool_summary(input), |l| l.command.clone());
                    c::tool_row(ui, row_id, name, &summary, failed, &details);
                    if cli.as_ref().is_some_and(|l| l.hipfire_set_aside) {
                        c::status(ui, State::Warn, t("bench.native"))
                            .on_hover_text(weight_surgeon::mcp::HIPFIRE_SET_ASIDE);
                    }
                }
                (
                    Block::ToolResult {
                        id,
                        content,
                        is_error,
                    },
                    _,
                ) if !calls.contains(id.as_str()) => {
                    let summary = result_line(content);
                    c::tool_row(
                        ui,
                        row_id,
                        t("chat.result"),
                        &summary,
                        *is_error,
                        &[content],
                    );
                }
                _ => {}
            }
        }
    }
    if !conv.plan.is_empty() {
        ui.add_space(T.space.s);
        c::inset_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            c::meta(ui, t("chat.plan"));
            for (k, step) in conv.plan.iter().enumerate() {
                c::label(ui, format!("{}. {}", k + 1, plan_step(step)));
            }
        });
        ui.add_space(T.space.m);
    }
    if let Some(s) = &app.agents.session {
        if !s.streaming.is_empty() {
            c::meta(ui, t("chat.agent"));
            c::label(ui, s.streaming.as_str());
        }
        if let Some(e) = &s.error {
            c::message(ui, e, true);
        }
        c::meta(
            ui,
            tf(
                "chat.usage",
                &[
                    ("in", &s.conv.usage.input_tokens),
                    ("out", &s.conv.usage.output_tokens),
                    ("cost", &format!("{:.4}", s.conv.cost_usd)),
                ],
            ),
        );
    }
}

/// The step without the number or bullet the agent may have written: the card numbers it.
fn plan_step(step: &str) -> &str {
    let step = step.trim_start();
    let rest = step.trim_start_matches(|c: char| c.is_ascii_digit());
    let marked = match rest.len() < step.len() {
        true => rest.strip_prefix(['.', ')']),
        false => rest.strip_prefix(['-', '*', '\u{2022}']),
    };
    marked.map_or(step, str::trim_start)
}

/// What a folded tool row says of its input: the command, prompt or path it names, else
/// its compact JSON, on one line.
fn tool_summary(input: &serde_json::Value) -> String {
    const KEYS: [&str; 10] = [
        "command",
        "cmd",
        "description",
        "prompt",
        "file_path",
        "path",
        "pattern",
        "text",
        "target",
        "model",
    ];
    let text = KEYS
        .iter()
        .find_map(|k| input[k].as_str())
        .map_or_else(|| input.to_string(), str::to_owned);
    let line = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    line.trim().chars().take(120).collect()
}

/// One line for a tool result: `← surgeon_inspect · run r…` for an engine result (its
/// untrusted-data frame), else its first line.
fn result_line(content: &str) -> String {
    let attr = |name: &str| {
        content
            .split(&format!("{name}=\""))
            .nth(1)
            .and_then(|r| r.split('"').next())
    };
    match (
        content.starts_with("<donnees_non_fiables"),
        attr("outil"),
        attr("run"),
    ) {
        (true, Some(tool), Some(run)) => tf("chat.tool_result", &[("tool", &tool), ("run", &run)]),
        _ => content
            .lines()
            .next()
            .unwrap_or("")
            .chars()
            .take(120)
            .collect(),
    }
}

/// The approval cards and the agent's question, where the agent asks for them.
fn approval(ui: &mut Ui, app: &mut SurgeonApp) {
    let Some(session) = app.agents.session.as_ref() else {
        return;
    };
    let cards = session.cards.clone();
    let question = session.question.clone();
    let mut decision = None;
    for card in &cards {
        ui.add_space(T.space.s);
        c::inset_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            c::status(ui, State::Warn, t("approval.waiting"));
            c::strong(ui, card.title.as_str());
            c::mono_wrapped(ui, card.params.as_str(), T.color.text_secondary);
            for line in &card.consequences {
                c::secondary(ui, format!("· {line}"));
            }
            c::meta(ui, card.cost.as_str());
            ui.add_space(T.space.s);
            ui.horizontal(|ui| {
                if c::button(ui, t("approval.once"), ButtonKind::Primary).clicked() {
                    decision = Some((card.id, Choice::Once));
                }
                if c::button(ui, t("approval.deny"), ButtonKind::Ghost).clicked() {
                    decision = Some((card.id, Choice::Deny));
                }
            });
        });
    }
    if let Some((card, choice)) = decision {
        app.agents.decide(&mut app.studio, card, choice);
    }
    if let Some((_, q)) = question {
        ui.add_space(T.space.s);
        c::inset_frame().show(ui, |ui| {
            ui.set_width(ui.available_width());
            c::status(ui, State::Active, t("approval.question"));
            c::label(ui, q);
            ui.add_space(T.space.s);
            c::field(
                ui,
                &mut app.agents.answer,
                t("approval.answer.hint"),
                ui.available_width(),
            );
            ui.add_space(T.space.s);
            if c::button(ui, t("approval.answer"), ButtonKind::Secondary).clicked() {
                app.agents.answer_question(&mut app.studio);
            }
        });
    }
}

// ------------------------------------------------------------------ right column

/// What the agent established: each finding with the measure it rests on, newest first.
fn evidence_log(ui: &mut Ui, app: &mut SurgeonApp) {
    let entries = app.runs();
    let findings = app
        .agents
        .project
        .as_ref()
        .map(|p| evidence::read(&Dirs::of(p).findings()))
        .unwrap_or_default();
    if findings.is_empty() {
        c::empty(ui, Icon::Check, t("finding.none"), t("finding.none.hint"));
        return;
    }
    ScrollArea::vertical()
        .id_salt("evidence")
        .auto_shrink(false)
        .show(ui, |ui| {
            for f in findings.iter().rev() {
                let (state, word) = match f.status(&entries) {
                    Status::Measured(_) => (State::Ok, t("finding.measured")),
                    Status::Inferred(_) => (State::Idle, t("finding.inferred")),
                    Status::Unknown(_) => (State::Warn, t("finding.unknown")),
                };
                c::status(ui, state, word);
                c::label(ui, f.text.as_str());
                ui.add_space(T.space.s);
            }
        });
}

// ------------------------------------------------------------------ consent

/// Before the first request to a connection: what leaves the machine, and to whom.
fn consent(ctx: &egui::Context, app: &mut SurgeonApp) {
    let Some(id) = app.agents.consent_for.clone() else {
        return;
    };
    let label = app
        .agents
        .cfg
        .connection(&id)
        .map_or(id.clone(), |c| c.label.clone());
    let screen = ctx.content_rect();
    let scrim = egui::LayerId::new(Order::Foreground, Id::new("consent-scrim"));
    ctx.layer_painter(scrim)
        .rect_filled(screen, 0.0, T.color.scrim);
    let width = T.size.reading * 0.6;
    let pos = Pos2::new(
        screen.center().x - width / 2.0,
        screen.top() + 12.0 * T.space.s,
    );
    let mut answer = None;
    egui::Area::new(Id::new("consent"))
        .order(Order::Tooltip)
        .fixed_pos(pos)
        .show(ctx, |ui| {
            c::popup_frame().inner_margin(T.space.xl).show(ui, |ui| {
                ui.set_width(width - 2.0 * T.space.xl);
                c::strong(ui, tf("consent.title", &[("connection", &label)]));
                ui.add_space(T.space.s);
                let lines = [
                    t("consent.messages"),
                    t("consent.tools"),
                    t("consent.paths"),
                    t("consent.never"),
                ];
                for line in lines {
                    c::label(ui, format!("· {line}"));
                }
                ui.add_space(T.space.s);
                c::checkbox(ui, &mut app.agents.consent_redact, t("consent.redact"));
                ui.add_space(T.space.l);
                ui.horizontal(|ui| {
                    if c::button(ui, t("consent.accept"), ButtonKind::Primary).clicked() {
                        answer = Some(true);
                    }
                    if c::button(ui, t("action.cancel"), ButtonKind::Ghost).clicked() {
                        answer = Some(false);
                    }
                });
            });
        });
    match answer {
        Some(true) => app.agents.accept_consent(),
        Some(false) => app.agents.consent_for = None,
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plan_step_loses_the_number_the_agent_wrote() {
        assert_eq!(plan_step("1. Inspecter la copie"), "Inspecter la copie");
        assert_eq!(plan_step(" 12) Mesurer"), "Mesurer");
        assert_eq!(plan_step("- Corriger"), "Corriger");
        assert_eq!(plan_step("3 couches à revoir"), "3 couches à revoir");
        assert_eq!(plan_step("Sans numéro"), "Sans numéro");
    }

    #[test]
    fn a_tool_summary_names_the_command_else_the_compact_input() {
        let shell = serde_json::json!({"command": "Get-ChildItem\n-Recurse", "timeout": 5});
        assert_eq!(tool_summary(&shell), "Get-ChildItem");
        let engine = serde_json::json!({"model": "copies/c1"});
        assert_eq!(tool_summary(&engine), "copies/c1");
        let other = serde_json::json!({"n": 1});
        assert_eq!(tool_summary(&other), r#"{"n":1}"#);
    }
}
