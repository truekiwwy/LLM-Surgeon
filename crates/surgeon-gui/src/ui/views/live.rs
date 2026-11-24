//! Live mode: the agent's session on the left, the 3D graph on the right, the graph following
//! the agent's tool calls (the running one ripples, the layers of the newest stay lit and the
//! rest steps back). Over the graph, the Live telemetry: where the agent works, when in the
//! protocol (health, probe, compose, measure), how, and where the measured metric stands.

use std::collections::HashSet;

use eframe::egui::{self, Label, Pos2, Rect, RichText, Sense, Ui, Vec2};

use super::super::components as c;
use super::super::i18n::t;
use super::super::icons::{self, Icon, State};
use super::super::layout::{snap8, Regions};
use super::super::theme::TOKENS as T;
use super::{agents, graph};
use crate::agent::activity::Activity;
use crate::agent::telemetry::{self, Evidence, Progress, Stage};
use crate::app::SurgeonApp;
use crate::source::Step;
use crate::studio::live::Layer;
use crate::studio::vectors::LayerScores;

/// Share of the body the agent takes; the graph gets the rest.
const SPLIT: f32 = 0.42;
/// The telemetry strip: four cards of a title row and two lines, on the 8-point grid.
const HUD: f32 = 88.0;

pub fn show(ui: &mut Ui, app: &mut SurgeonApp, regions: &Regions) {
    let body = Rect::from_min_max(regions.left.min, regions.right.max);
    let x = snap8(body.left() + body.width() * SPLIT);
    let left = Rect::from_min_max(body.min, Pos2::new(x, body.bottom()));
    let right = Rect::from_min_max(Pos2::new(x, body.top()), body.max);
    agents::session(ui, app, left);
    c::Panel::new("live-graph")
        .title(t("live.telemetry"))
        .canvas()
        .show_with(
            ui,
            right,
            app,
            |ui, app| {
                c::switch(
                    ui,
                    Icon::Cube,
                    t("live.ghost"),
                    &mut app.ghost,
                    t("live.ghost.tip"),
                );
            },
            |ui, app| {
                let rect = ui.max_rect();
                let hud = Rect::from_min_size(rect.min, Vec2::new(rect.width(), HUD));
                let view = Rect::from_min_max(Pos2::new(rect.left(), hud.bottom()), rect.max);
                c::region(ui, hud, "live-hud", |ui| strip(ui, app));
                c::region(ui, view, "live-view", |ui| graph::viewport(ui, app, true));
            },
        );
    let painter = ui.painter();
    for line in [regions.activity.right(), x] {
        c::vline(painter, line, body.y_range(), T.color.separator);
    }
    agents::consent(ui.ctx(), app);
}

/// The nodes Live mode keeps lit: the layers of the agent's newest call, the nodes its run
/// reached, those the last patch touched and those that differ from the baseline. `None`
/// when nothing stands out yet: no node steps back then.
pub fn focus(
    step: &Step,
    layer: Option<&Layer>,
    activity: Option<&Activity>,
    scores: &LayerScores,
) -> Option<Vec<bool>> {
    let changed: HashSet<&str> = step
        .report
        .baseline_diff
        .iter()
        .flat_map(|d| &d.nodes)
        .filter(|d| d.changed)
        .map(|d| d.id.as_str())
        .collect();
    let mask: Vec<bool> = step
        .report
        .nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            activity.is_some_and(|a| a.covers(node.layer))
                || scores.lit(node.layer)
                || layer.is_some_and(|l| l.states.get(i).copied().flatten().is_some())
                || step.touched.contains_key(&node.id)
                || changed.contains(node.id.as_str())
        })
        .collect();
    mask.contains(&true).then_some(mask)
}

/// Where, when, how, toward what: four cards, the protocol's frieze the widest.
fn strip(ui: &mut Ui, app: &mut SurgeonApp) {
    let now = app
        .agents
        .conversation()
        .map(telemetry::of)
        .unwrap_or_default();
    let progress = telemetry::progress(&app.runs());
    let target = now.target.clone().or_else(|| app.focus.clone());
    let rect = ui.max_rect().shrink(T.space.s);
    let shares = [0.9, 1.8, 1.1, 1.1];
    let unit = (rect.width() - 3.0 * T.space.s) / shares.iter().sum::<f32>();
    let mut x = rect.left();
    let [place, when, how, whither] = shares.map(|share| {
        let size = Vec2::new(share * unit, rect.height());
        let cell = Rect::from_min_size(Pos2::new(x, rect.top()), size);
        x = cell.right() + T.space.s;
        cell
    });
    card(ui, place, "live-where", t("live.where"), None, |ui| {
        let target = target.as_deref().unwrap_or("-");
        c::mono(ui, target).on_hover_text(target);
    });
    card(ui, when, "live-when", t("live.when"), None, |ui| {
        frieze(ui, now.stage);
    });
    let evidence = now.method.as_ref().and_then(|m| m.evidence);
    card(ui, how, "live-how", t("live.how"), evidence, |ui| {
        let method = now.method.as_ref().map_or("-", |m| m.text.as_str());
        c::mono(ui, method).on_hover_text(method);
    });
    let evidence = progress.as_ref().map(|p| match p.heuristic {
        true => Evidence::Heuristic,
        false => Evidence::Measured,
    });
    card(
        ui,
        whither,
        "live-whither",
        t("live.whither"),
        evidence,
        |ui| match &progress {
            Some(p) => {
                let tip = match p.metric {
                    "false_refusal_rate" => t("live.false_refusal.tip"),
                    "safety_refusal_rate" => t("live.safety_refusal.tip"),
                    _ => p.metric,
                };
                let tip = match &p.dataset {
                    Some(dataset) => format!("{tip}\n{dataset}"),
                    None => tip.to_owned(),
                };
                c::mono_secondary(ui, p.metric).on_hover_text(&tip);
                c::mono(ui, rates(p)).on_hover_text(&tip);
            }
            None => {
                c::mono(ui, "-");
            }
        },
    );
}

/// A card of the strip: surface, one-pixel border, its title and badge, then `body`.
fn card(
    ui: &mut Ui,
    rect: Rect,
    salt: &str,
    title: &str,
    evidence: Option<Evidence>,
    body: impl FnOnce(&mut Ui),
) {
    let painter = ui.painter();
    let rect = c::snap(painter, rect);
    painter.rect_filled(rect, T.radius, T.color.surface);
    c::outline(painter, rect, T.radius, T.color.border);
    let inner = rect.shrink2(Vec2::new(T.space.m, T.space.s));
    // The title is one text line, not a control row: the body keeps two lines below it.
    c::region(ui, inner, salt, |ui| {
        ui.spacing_mut().item_spacing.y = T.space.xs / 2.0;
        c::meta(ui, title);
        body(ui);
    });
    if let Some(evidence) = evidence {
        let top = Rect::from_min_size(inner.min, Vec2::new(inner.width(), T.space.l));
        c::region(ui, top, &format!("{salt}-badge"), |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                badge(ui, evidence);
            });
        });
    }
}

/// `Heuristique` (a screening) or `Mesuré` (a real measure), behind its state dot.
fn badge(ui: &mut Ui, evidence: Evidence) {
    let (state, word) = match evidence {
        Evidence::Heuristic => (State::Warn, t("live.heuristic")),
        Evidence::Measured => (State::Ok, t("finding.measured")),
    };
    // Right to left: the word, then its dot on its left.
    c::meta(ui, word);
    let (rect, _) = ui.allocate_exact_size(Vec2::splat(T.size.dot), Sense::hover());
    icons::dot(ui.painter(), rect.center(), state);
}

/// `1. Santé → 2. Sonde → 3. Compose → 4. Mesure`: the stage reached in full text, the ones
/// before it secondary, the ones after it disabled.
fn stage_name(s: Stage) -> &'static str {
    match s {
        Stage::Health => t("live.stage.health"),
        Stage::Probe => t("live.stage.probe"),
        Stage::Compose => t("live.stage.compose"),
        Stage::Measure => t("live.stage.measure"),
    }
}

fn rows(widths: &[f32], room: f32, gap: f32) -> usize {
    let mut rows = 1;
    let mut x = 0.0;
    for w in widths {
        if x > 0.0 && x + w > room {
            rows += 1;
            x = 0.0;
        }
        x += w + gap;
    }
    rows
}

fn frieze(ui: &mut Ui, stage: Option<Stage>) {
    let labels: Vec<String> = Stage::ALL
        .into_iter()
        .enumerate()
        .map(|(k, s)| format!("{}. {}", k + 1, stage_name(s)))
        .collect();
    let measure = |text: &str| {
        ui.painter()
            .layout_no_wrap(text.to_owned(), c::font_meta(), T.color.text)
            .size()
            .x
    };
    let arrow = measure("→");
    let widths: Vec<f32> = labels
        .iter()
        .enumerate()
        .map(|(k, l)| measure(l) + if k > 0 { arrow + T.space.xs } else { 0.0 })
        .collect();
    if rows(&widths, ui.available_width(), T.space.xs) > 2 {
        let full = labels.join(" → ");
        let now = stage.map_or("-".to_owned(), |s| {
            let k = Stage::ALL.iter().position(|x| *x == s).unwrap_or(0);
            format!("{}/{} {}", k + 1, Stage::ALL.len(), stage_name(s))
        });
        ui.add(Label::new(rich(&now, T.color.text)).truncate())
            .on_hover_text(full);
        return;
    }
    // Rows as tall as the text, not as a control: two of them fit the card when it wraps.
    ui.spacing_mut().interact_size.y = T.space.l;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing.x = T.space.xs;
        for (k, s) in Stage::ALL.into_iter().enumerate() {
            if k > 0 {
                ui.add(Label::new(rich("→", T.color.text_disabled)));
            }
            let name = stage_name(s);
            let color = match stage.map(|now| s.cmp(&now)) {
                Some(std::cmp::Ordering::Equal) => T.color.text,
                Some(std::cmp::Ordering::Less) => T.color.text_secondary,
                _ => T.color.text_disabled,
            };
            ui.add(Label::new(rich(&format!("{}. {name}", k + 1), color)));
        }
    });
}

fn rich(text: &str, color: egui::Color32) -> RichText {
    RichText::new(text).font(c::font_meta()).color(color)
}

/// `62.0 % → 8.0 %`, or the copy's rate alone when the run had no baseline.
fn rates(p: &Progress) -> String {
    let pct = |v: f64| format!("{:.1} %", v * 100.0);
    match p.before {
        Some(before) => format!("{} → {}", pct(before), pct(p.after)),
        None => pct(p.after),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::Timeline;
    use crate::testkit;

    #[test]
    fn the_newest_call_s_layers_stay_lit_and_nothing_dims_before_it() {
        let timeline = Timeline::from_report(testkit::graph(28));
        let step = timeline.current().unwrap();
        let none = LayerScores::default();
        assert_eq!(
            focus(step, None, None, &none),
            None,
            "nothing stands out yet"
        );
        let extract =
            crate::agent::activity::of("surgeon_extract", &serde_json::json!({"layer": 2}));
        let mask = focus(step, None, extract.as_ref(), &none).unwrap();
        for (lit, node) in mask.iter().zip(&step.report.nodes) {
            assert_eq!(*lit, node.layer == Some(2), "{}", node.id);
        }
    }

    #[test]
    fn the_last_causal_window_stays_lit_between_calls() {
        let timeline = Timeline::from_report(testkit::graph(28));
        let step = timeline.current().unwrap();
        let scores = LayerScores {
            window: Some((1, 2)),
            ..LayerScores::default()
        };
        let mask = focus(step, None, None, &scores).unwrap();
        for (lit, node) in mask.iter().zip(&step.report.nodes) {
            let causal = matches!(node.layer, Some(1 | 2));
            assert_eq!(*lit, causal, "{}", node.id);
        }
    }

    #[test]
    fn rates_read_from_the_origin_to_the_copy() {
        let p = Progress {
            metric: "refusal_rate",
            dataset: None,
            before: Some(0.62),
            after: 0.08,
            heuristic: false,
        };
        assert_eq!(rates(&p), "62.0 % → 8.0 %");
        assert_eq!(rates(&Progress { before: None, ..p }), "8.0 %");
    }

    #[test]
    fn the_frieze_counts_the_rows_it_would_wrap_into() {
        assert_eq!(rows(&[40.0, 40.0, 40.0, 40.0], 400.0, 4.0), 1);
        assert_eq!(rows(&[40.0, 40.0, 40.0, 40.0], 90.0, 4.0), 2);
        assert_eq!(rows(&[40.0, 40.0, 40.0, 40.0], 50.0, 4.0), 4);
    }
}
