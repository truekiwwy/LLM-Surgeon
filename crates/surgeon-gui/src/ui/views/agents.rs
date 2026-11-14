//! Agents: the explorer; in the center the session of the open project (conversation,
//! approval cards, question, plan, the message field); on the right the policy (mode,
//! budgets, local only, the "always" rules, the evidence log). The connections live in
//! System. The consent dialog opens before the first request.

use eframe::egui::{self, Id, Order, Pos2, ScrollArea, Ui};

use super::super::components::{self as c, ButtonKind, Segment};
use super::super::i18n::{t, tf};
use super::super::icons::{Icon, State};
use super::super::layout::Regions;
use super::super::theme::TOKENS as T;
use super::explorer;
use crate::agent::config::AskMode;
use crate::agent::evidence::{self, Status};
use crate::agent::policy::AgentMode;
use crate::agent::provider::{Block, Role};
use crate::agent::session::{Choice, Dirs};
use crate::app::SurgeonApp;

pub fn show(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    explorer(ui, app, regions.left);
    c::Panel::new("session")
        .title(t("panel.session"))
        .show_with(ui, regions.center, app, session_actions, session);
    c::Panel::new("policy")
        .title(t("panel.policy"))
        .show(ui, regions.right, |ui| policy(ui, app));
    consent(ui.ctx(), app);
}

fn session_actions(ui: &mut Ui, app: &mut SurgeonApp) {
    let project = app.agents.project.clone();
    let has_conv = app.agents.conversation().is_some();
    if c::button_if(
        ui,
        has_conv && project.is_some(),
        t("chat.export"),
        ButtonKind::Ghost,
        t("chat.export.why"),
    )
    .clicked()
    {
        if let Some(p) = project {
            app.agents.export(&p);
        }
    }
    if c::button_if(
        ui,
        !app.agents.busy(),
        t("chat.new"),
        ButtonKind::Ghost,
        t("chat.busy"),
    )
    .clicked()
    {
        app.agents.new_conversation();
    }
}

fn session(ui: &mut Ui, app: &mut SurgeonApp) {
    if app.agents.project.is_none() {
        c::empty(ui, t("project.missing"), t("agents.project.hint"));
        return;
    }
    let line = match app.agents.connection() {
        Some(conn) => tf(
            "chat.connection",
            &[
                ("connection", &conn.label),
                (
                    "model",
                    &app.agents.agent.model.clone().unwrap_or_else(|| "-".into()),
                ),
            ],
        ),
        None => t("chat.no_connection").into(),
    };
    c::meta(ui, line);
    ui.add_space(T.space.s);
    let footer = 4.0 * T.size.row + T.size.cta + T.space.l;
    let height = (ui.available_height() - footer).max(T.size.row * 3.0);
    ui.allocate_ui(egui::vec2(ui.available_width(), height), |ui| {
        ScrollArea::vertical()
            .id_salt("session")
            .auto_shrink(false)
            .stick_to_bottom(true)
            .show(ui, |ui| {
                transcript_body(ui, app);
                approval(ui, app, false);
            });
    });
    c::separator(ui);
    c::field_multiline(ui, &mut app.agents.input, t("chat.hint"), 3);
    ui.add_space(T.space.s);
    let ctx = ui.ctx().clone();
    ui.horizontal(|ui| {
        let busy = app.agents.busy();
        let paused = app.agents.session.as_ref().is_some_and(|s| s.paused);
        if paused {
            if c::button(ui, t("chat.resume"), ButtonKind::Secondary).clicked() {
                app.agents.resume_run(&mut app.studio);
            }
        } else if c::button_if(
            ui,
            busy,
            t("chat.pause"),
            ButtonKind::Secondary,
            t("chat.idle"),
        )
        .clicked()
        {
            app.agents.pause();
        }
        if c::button_if(ui, busy, t("chat.stop"), ButtonKind::Ghost, t("chat.idle")).clicked() {
            app.agents.stop(&mut app.studio);
        }
        let ready = !busy && !app.agents.input.trim().is_empty() && app.agents.connected();
        let send = ui.add_enabled_ui(ready, |ui| {
            ui.set_width(ui.available_width());
            c::button_wide(ui, t("chat.send"), ButtonKind::Primary)
        });
        let enter = ui.input(|i| i.key_pressed(egui::Key::Enter) && i.modifiers.command);
        if ready && (send.inner.clicked() || enter) {
            app.agents.send_input();
            ctx.request_repaint();
        }
    });
    if let Some((text, error)) = app.agents.message.clone() {
        c::message(ui, &text, error);
    }
}

fn transcript_body(ui: &mut Ui, app: &SurgeonApp) {
    let Some(conv) = app.agents.conversation() else {
        c::empty(ui, t("chat.empty"), t("chat.empty.hint"));
        return;
    };
    for message in &conv.messages {
        let who = match message.role {
            Role::User => t("chat.you"),
            Role::Assistant => t("chat.agent"),
        };
        for block in &message.blocks {
            match block {
                Block::Text { text } => {
                    c::meta(ui, who);
                    c::label(ui, text.as_str());
                    ui.add_space(T.space.s);
                }
                Block::ToolUse { name, input, .. } => {
                    let line = tf("chat.tool_call", &[("name", name), ("input", &input)]);
                    c::icon_line(ui, Icon::StepForward, line).on_hover_text(input.to_string());
                }
                Block::ToolResult {
                    content, is_error, ..
                } => {
                    let first = result_line(content);
                    match is_error {
                        true => c::status(ui, State::Error, first),
                        false => c::icon_line(ui, Icon::StepBack, first),
                    };
                }
                Block::Opaque { .. } => {}
            }
        }
    }
    if let Some(s) = &app.agents.session {
        if !s.streaming.is_empty() {
            c::meta(ui, t("chat.agent"));
            c::label(ui, s.streaming.as_str());
        }
        if !s.conv.plan.is_empty() {
            c::section(ui, t("chat.plan"));
            for (k, step) in s.conv.plan.iter().enumerate() {
                c::label(ui, format!("{}. {step}", k + 1));
            }
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

/// The approval cards and the agent's question; `compact` (the Atelier) shows the first
/// card with its parameters cut to a few lines.
pub fn approval(ui: &mut Ui, app: &mut SurgeonApp, compact: bool) {
    let Some(session) = app.agents.session.as_ref() else {
        if compact {
            c::empty(ui, t("approval.none"), t("approval.none.hint"));
        }
        return;
    };
    let cards = session.cards.clone();
    let question = session.question.clone();
    if cards.is_empty() && question.is_none() {
        if compact {
            c::empty(ui, t("approval.none"), t("approval.none.hint"));
        }
        return;
    }
    let mut decision = None;
    for card in cards.iter().take(if compact { 1 } else { usize::MAX }) {
        c::status(ui, State::Warn, t("approval.waiting"));
        c::strong(ui, card.title.as_str());
        let params = match compact {
            true => card.params.lines().take(3).collect::<Vec<_>>().join("\n"),
            false => card.params.clone(),
        };
        c::mono_wrapped(ui, params, T.color.text_secondary).on_hover_text(&card.params);
        if !compact {
            for line in &card.consequences {
                c::secondary(ui, format!("· {line}"));
            }
            c::meta(ui, card.cost.as_str());
        }
        ui.add_space(T.space.s);
        ui.horizontal(|ui| {
            if c::button(ui, t("approval.once"), ButtonKind::Secondary).clicked() {
                decision = Some((card.id, Choice::Once));
            }
            if card.allow_always
                && c::button(ui, t("approval.always"), ButtonKind::Ghost)
                    .on_hover_text(t("approval.always.tip"))
                    .clicked()
            {
                decision = Some((card.id, Choice::Always));
            }
            if c::button(ui, t("approval.deny"), ButtonKind::Ghost).clicked() {
                decision = Some((card.id, Choice::Deny));
            }
        });
        ui.add_space(T.space.m);
    }
    if compact && cards.len() > 1 {
        c::meta(ui, tf("approval.more", &[("n", &(cards.len() - 1))]));
    }
    if let Some((card, choice)) = decision {
        app.agents.decide(&mut app.studio, card, choice);
    }
    if let Some((_, q)) = question {
        c::status(ui, State::Active, t("approval.question"));
        c::label(ui, q);
        c::field(
            ui,
            &mut app.agents.answer,
            t("approval.answer.hint"),
            ui.available_width(),
        );
        ui.add_space(T.space.xs);
        if c::button(ui, t("approval.answer"), ButtonKind::Secondary).clicked() {
            app.agents.answer_question(&mut app.studio);
        }
    }
}

// ------------------------------------------------------------------ policy

fn policy(ui: &mut Ui, app: &mut SurgeonApp) {
    let entries = app.ledger().to_vec();
    let project = app.agents.project.clone();
    let hub = &mut app.agents;
    ScrollArea::vertical()
        .id_salt("policy")
        .auto_shrink(false)
        .show(ui, |ui| {
            let mut mode = hub.agent.mode;
            c::segmented(
                ui,
                &mut mode,
                &[
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
                ],
                c::font_body(),
            );
            ui.add_space(T.space.xs);
            let mut ask = hub.agent.ask;
            c::segmented(
                ui,
                &mut ask,
                &[
                    Segment {
                        value: AskMode::AskMe,
                        label: t("policy.ask_me"),
                        tooltip: t("policy.ask_me.tip"),
                    },
                    Segment {
                        value: AskMode::DecideAlone,
                        label: t("policy.decide"),
                        tooltip: t("policy.decide.tip"),
                    },
                ],
                c::font_body(),
            );
            if mode != hub.agent.mode || ask != hub.agent.ask {
                hub.agent.mode = mode;
                hub.agent.ask = ask;
                hub.agent_changed();
            }
            let mut local = hub.cfg.local_only;
            if c::checkbox(ui, &mut local, t("policy.local_only"))
                .on_hover_text(t("policy.local_only.tip"))
                .changed()
            {
                hub.cfg.local_only = local;
                hub.save_cfg();
            }
            c::section(ui, t("policy.budgets"));
            let names = [
                t("budget.tokens"),
                t("budget.cost"),
                t("budget.secs"),
                t("budget.calls"),
            ];
            for (k, name) in names.into_iter().enumerate() {
                ui.horizontal(|ui| {
                    let label_width = (ui.available_width() * 0.5).floor();
                    ui.allocate_ui(egui::vec2(label_width, T.size.control), |ui| {
                        ui.set_width(label_width);
                        c::secondary(ui, name);
                    });
                    c::field_mono(
                        ui,
                        &mut hub.budget_text[k],
                        t("budget.none"),
                        ui.available_width(),
                    );
                });
                ui.add_space(T.space.xs);
            }
            if c::button(ui, t("policy.apply"), ButtonKind::Secondary).clicked() {
                hub.apply_budgets();
            }
            c::section(ui, t("policy.rules"));
            if hub.agent.rules.is_empty() {
                c::secondary(ui, t("policy.rules.none"));
            }
            let mut revoke = None;
            for (k, rule) in hub.agent.rules.iter().enumerate() {
                ui.horizontal(|ui| {
                    c::mono(ui, rule.tool.as_str());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if c::button(ui, t("policy.revoke"), ButtonKind::Ghost).clicked() {
                            revoke = Some(k);
                        }
                    });
                });
            }
            if let Some(k) = revoke {
                hub.revoke_rule(k);
            }
            c::section(ui, t("policy.findings"));
            let findings = project
                .as_ref()
                .map(|p| evidence::read(&Dirs::of(p).findings()))
                .unwrap_or_default();
            if findings.is_empty() {
                c::secondary(ui, t("policy.findings.none"));
            }
            for f in findings.iter().rev() {
                let (state, word) = match f.status(&entries) {
                    Status::Measured(_) => (State::Ok, t("finding.measured")),
                    Status::Inferred(_) => (State::Idle, t("finding.inferred")),
                    Status::Unknown(_) => (State::Warn, t("finding.unknown")),
                };
                c::status(ui, state, word);
                c::label(ui, f.text.as_str());
                ui.add_space(T.space.xs);
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
    let width = 60.0 * T.size.gap;
    let pos = Pos2::new(
        screen.center().x - width / 2.0,
        screen.top() + 12.0 * T.size.gap,
    );
    let mut answer = None;
    egui::Area::new(Id::new("consent"))
        .order(Order::Tooltip)
        .fixed_pos(pos)
        .show(ctx, |ui| {
            c::popup_frame().inner_margin(T.space.l).show(ui, |ui| {
                ui.set_width(width - 2.0 * T.space.l);
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
                ui.add_space(T.space.m);
                ui.horizontal(|ui| {
                    if c::button(ui, t("action.cancel"), ButtonKind::Ghost).clicked() {
                        answer = Some(false);
                    }
                    if c::button(ui, t("consent.accept"), ButtonKind::Primary).clicked() {
                        answer = Some(true);
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
