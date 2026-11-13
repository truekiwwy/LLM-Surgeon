//! State and layout of the window: top bar, layer list (left), 3D viewport (center),
//! inspector (right) and, for a stream, the patch timeline (bottom).

use std::collections::BTreeMap;
use std::sync::mpsc::Receiver;

use eframe::egui::{
    self, Align, CentralPanel, CollapsingHeader, Grid, Key, Layout, Panel, RichText, ScrollArea,
    Sense, Slider, TextEdit,
};
use weight_surgeon::backend::{self, config::BackendConfig, detect::BackendReport};
use weight_surgeon::bench::score::Backend;
use weight_surgeon::graph::diff::NodeDiff;
use weight_surgeon::graph::metrics::NodeMetrics;
use weight_surgeon::graph::output::{GraphReport, NodeReport};
use weight_surgeon::graph::stream::GraphPatch;
use weight_surgeon::graph::GraphRole;

use crate::backend_dialog::BackendDialog;
use crate::camera::Orbit;
use crate::source::{self, Input, Step, Timeline};
use crate::theme::{self, mono};
use crate::viewport::{self, node_title, View};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LayoutMode {
    Column,
    Constellation,
}

/// The role filters of the layer list.
const GROUPS: [(&str, &[GraphRole]); 3] = [
    (
        "attn",
        &[GraphRole::Q, GraphRole::K, GraphRole::V, GraphRole::AttnOut],
    ),
    ("mlp", &[GraphRole::Gate, GraphRole::Up, GraphRole::MlpOut]),
    ("embed", &[GraphRole::Embed, GraphRole::LmHead]),
];

pub struct SurgeonApp {
    timeline: Timeline,
    /// Patches of an NDJSON stream; `None` for a single `graph.json` (no timeline bar).
    feed: Option<Receiver<Result<GraphPatch, String>>>,
    stream_error: Option<String>,
    layout: LayoutMode,
    diff: bool,
    camera: Orbit,
    selected: Option<String>,
    filter: String,
    groups: [bool; 3],
    show_layers: bool,
    show_inspector: bool,
    dialog: BackendDialog,
    probe: Option<Receiver<BackendReport>>,
    hardware: Option<BackendReport>,
}

impl SurgeonApp {
    pub fn new(ctx: &egui::Context, input: Option<Input>) -> Self {
        theme::apply(ctx);
        let (timeline, feed) = match input {
            Some(Input::Report(report)) => (Timeline::from_report(*report), None),
            Some(Input::Stream { reader, follow }) => {
                let wake = ctx.clone();
                let rx = source::spawn_reader(reader, follow, move || wake.request_repaint());
                (Timeline::default(), Some(rx))
            }
            None => (Timeline::default(), None),
        };
        let dialog = BackendDialog::load();
        // `llama-server --list-devices` can take seconds: probe off the UI thread.
        let (tx, probe) = std::sync::mpsc::channel();
        let user = dialog.saved().clone();
        let wake = ctx.clone();
        std::thread::spawn(move || {
            let _ = tx.send(backend::detect::detect_hardware_and_backends(&user));
            wake.request_repaint();
        });
        Self {
            timeline,
            feed,
            stream_error: None,
            layout: LayoutMode::Column,
            diff: false,
            camera: Orbit::default(),
            selected: None,
            filter: String::new(),
            groups: [true; 3],
            show_layers: true,
            show_inspector: true,
            dialog,
            probe: Some(probe),
            hardware: None,
        }
    }

    /// Drains the background channels.
    fn poll(&mut self) {
        if let Some(feed) = &self.feed {
            for patch in feed.try_iter() {
                if let Err(e) = patch.and_then(|p| self.timeline.push(&p)) {
                    self.stream_error = Some(e);
                }
            }
        }
        if let Some(report) = self.probe.as_ref().and_then(|rx| rx.try_recv().ok()) {
            self.dialog.prefill(&report);
            self.hardware = Some(report);
            self.probe = None;
        }
    }

    fn visible(&self, report: &GraphReport) -> Vec<bool> {
        let needle = self.filter.trim().to_lowercase();
        report
            .nodes
            .iter()
            .map(|n| {
                let group = GROUPS
                    .iter()
                    .position(|(_, roles)| roles.contains(&n.role))
                    .is_none_or(|g| self.groups[g]);
                group
                    && (needle.is_empty()
                        || n.id.to_lowercase().contains(&needle)
                        || n.tensor.to_lowercase().contains(&needle))
            })
            .collect()
    }

    fn hardware_label(&self) -> String {
        match &self.hardware {
            None => "détection…".into(),
            Some(r) => match r.llama.as_ref().and_then(|l| l.devices.first()) {
                Some(gpu) => format!("{} · {}", gpu.name, gpu.api()),
                None => "CPU".into(),
            },
        }
    }

    fn backend_label(&self) -> String {
        let saved: &BackendConfig = self.dialog.saved();
        let kind = saved.backend.unwrap_or(Backend::Native);
        let api = match kind {
            Backend::LlamaServer => self
                .hardware
                .as_ref()
                .and_then(|r| r.llama.as_ref())
                .and_then(|l| l.devices.first())
                .map(|d| d.api()),
            _ => None,
        };
        let text = backend::describe(
            kind.label(),
            api,
            saved.gpu_layers,
            saved.batch_size.unwrap_or(1),
        );
        format!("Backend: {text}")
    }

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_centered(|ui| {
            match self.timeline.current() {
                Some(step) => {
                    let m = &step.report.model;
                    ui.label(RichText::new(&m.architecture).strong());
                    ui.label(mono(format!(
                        "L{} · d{} · {} nœuds · {} arêtes",
                        m.layers,
                        m.hidden,
                        step.report.nodes.len(),
                        step.report.edges.len()
                    )))
                    .on_hover_text(&m.path);
                }
                None => {
                    ui.label(RichText::new("aucun graphe").color(theme::DIM));
                }
            }
            ui.separator();
            ui.selectable_value(&mut self.layout, LayoutMode::Column, "Colonne");
            ui.selectable_value(&mut self.layout, LayoutMode::Constellation, "Constellation");
            ui.separator();
            let has_diff = self.timeline.current().is_some_and(|s| {
                s.report.baseline_diff.is_some() || s.touched.values().any(Option::is_some)
            });
            let toggle = ui
                .add_enabled(has_diff, egui::Button::selectable(self.diff, "Diff (Δ)"))
                .on_disabled_hover_text("ni --baseline dans le rapport, ni patch d'analyse");
            if toggle.clicked() {
                self.diff = !self.diff;
            }
            ui.separator();
            ui.toggle_value(&mut self.show_layers, "Couches");
            ui.toggle_value(&mut self.show_inspector, "Inspecteur");

            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                if ui.button("Backend…").clicked() {
                    self.dialog.open = true;
                }
                ui.label(mono(self.backend_label()).color(theme::DIM));
                ui.separator();
                ui.label(mono(self.hardware_label()));
            });
        });
    }

    fn layer_panel(&mut self, ui: &mut egui::Ui) {
        ui.add(
            TextEdit::singleline(&mut self.filter)
                .hint_text("filtre : id ou tenseur")
                .desired_width(f32::INFINITY),
        );
        ui.horizontal(|ui| {
            for (g, (name, _)) in GROUPS.iter().enumerate() {
                ui.toggle_value(&mut self.groups[g], mono(*name));
            }
        });
        ui.separator();
        let Some(step) = self.timeline.current() else {
            return;
        };
        let visible = self.visible(&step.report);
        let mut layers: BTreeMap<Option<usize>, Vec<usize>> = BTreeMap::new();
        for (i, node) in step.report.nodes.iter().enumerate() {
            if visible[i] {
                layers.entry(node.layer).or_default().push(i);
            }
        }
        let filtering = !self.filter.trim().is_empty();
        let mut picked = None;
        ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            for (layer, nodes) in &layers {
                let title = match layer {
                    Some(l) => format!("Layer {l:>3}"),
                    None => "embed / lm_head".into(),
                };
                let holds_selection = nodes
                    .iter()
                    .any(|&i| self.selected.as_deref() == Some(&step.report.nodes[i].id));
                CollapsingHeader::new(mono(format!("{title}  ({})", nodes.len())))
                    .id_salt(("layer", *layer))
                    .open(filtering.then_some(true))
                    .default_open(holds_selection)
                    .show(ui, |ui| {
                        for &i in nodes {
                            let node = &step.report.nodes[i];
                            let selected = self.selected.as_deref() == Some(&node.id);
                            ui.horizontal(|ui| {
                                let (swatch, _) =
                                    ui.allocate_exact_size(egui::vec2(8.0, 8.0), Sense::hover());
                                ui.painter().rect_filled(
                                    swatch,
                                    egui::CornerRadius::same(1),
                                    theme::role_color(node.role),
                                );
                                let mut text = viewport::role_label(node.role).to_owned();
                                if let Some(e) = node.expert {
                                    text.push_str(&format!(" · e{e}"));
                                }
                                if ui.selectable_label(selected, mono(text)).clicked() {
                                    picked = Some(node.id.clone());
                                }
                                if step.touched.contains_key(&node.id) {
                                    ui.label(mono("patch").small().color(theme::AGENT));
                                }
                            });
                        }
                    });
            }
        });
        if picked.is_some() {
            self.selected = picked;
        }
    }

    fn inspector(&self, ui: &mut egui::Ui) {
        let Some(step) = self.timeline.current() else {
            ui.label(RichText::new("aucun graphe chargé").color(theme::DIM));
            return;
        };
        ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
            let node = self
                .selected
                .as_deref()
                .and_then(|id| step.report.nodes.iter().find(|n| n.id == id));
            match node {
                Some(node) => node_inspector(ui, step, node),
                None => model_summary(ui, &step.report),
            }
        });
    }

    fn timeline_bar(&mut self, ui: &mut egui::Ui) {
        ui.horizontal_centered(|ui| {
            let len = self.timeline.len();
            let mut cursor = self.timeline.cursor();
            let last = len.saturating_sub(1);
            let enabled = len > 1;
            if ui
                .add_enabled(enabled, egui::Button::new(mono("|<")))
                .clicked()
            {
                cursor = 0;
            }
            if ui
                .add_enabled(enabled, egui::Button::new(mono("<")))
                .clicked()
            {
                cursor = cursor.saturating_sub(1);
            }
            if ui
                .add_enabled(enabled, egui::Button::new(mono(">")))
                .clicked()
            {
                cursor = (cursor + 1).min(last);
            }
            if ui
                .add_enabled(enabled, egui::Button::new(mono(">|")))
                .clicked()
            {
                cursor = last;
            }
            ui.add_enabled(
                enabled,
                Slider::new(&mut cursor, 0..=last).show_value(false),
            );
            self.timeline.seek(cursor);
            let touched = self.timeline.current().map_or(0, |s| s.touched.len());
            ui.label(mono(format!(
                "état {:>3}/{:<3} · {touched} nœud(s) touché(s)",
                cursor + usize::from(len > 0),
                len
            )));
            ui.label(match self.timeline.is_live() {
                true => mono("direct"),
                false => mono("rejeu").color(theme::DIM),
            });
            if let Some(e) = &self.stream_error {
                ui.label(RichText::new(e).small().color(theme::BAD));
            }
        });
    }

    fn viewport(&mut self, ui: &mut egui::Ui) {
        let target = match self.layout {
            LayoutMode::Column => 0.0,
            LayoutMode::Constellation => 1.0,
        };
        let t = ui
            .ctx()
            .animate_value_with_time(egui::Id::new("layout-blend"), target, 0.45);
        let blend = t * t * (3.0 - 2.0 * t);
        let Some(step) = self.timeline.current() else {
            let hint = match self.feed {
                Some(_) => "en attente du premier snapshot NDJSON…",
                None => "surgeon-gui <graph.json | graph.ndjson | ->",
            };
            ui.centered_and_justified(|ui| ui.label(mono(hint).color(theme::DIM)));
            return;
        };
        let visible = self.visible(&step.report);
        let selected = self
            .selected
            .as_deref()
            .and_then(|id| step.report.nodes.iter().position(|n| n.id == id));
        let view = View {
            step,
            blend,
            diff: self.diff,
            visible: &visible,
            selected,
        };
        if let Some(click) = viewport::show(ui, &view, &mut self.camera) {
            self.selected = click.map(|i| step.report.nodes[i].id.clone());
        }
    }
}

impl eframe::App for SurgeonApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.poll();
        let ctx = ui.ctx().clone();
        if !ctx.egui_wants_keyboard_input() && ctx.input(|i| i.key_pressed(Key::R)) {
            self.camera = Orbit::default();
        }

        Panel::top("top-bar")
            .exact_size(30.0)
            .show(ui, |ui| self.top_bar(ui));
        if self.feed.is_some() {
            Panel::bottom("timeline")
                .exact_size(32.0)
                .show(ui, |ui| self.timeline_bar(ui));
        }
        let mut open = self.show_layers;
        Panel::left("layers")
            .default_size(255.0)
            .size_range(240.0..=270.0)
            .show_collapsible(ui, &mut open, |ui| self.layer_panel(ui));
        self.show_layers = open;
        let mut open = self.show_inspector;
        Panel::right("inspector")
            .default_size(295.0)
            .size_range(280.0..=310.0)
            .show_collapsible(ui, &mut open, |ui| self.inspector(ui));
        self.show_inspector = open;
        CentralPanel::no_frame().show(ui, |ui| self.viewport(ui));

        self.dialog.show(&ctx, self.hardware.as_ref());
    }
}

// ------------------------------------------------------------------ inspector

/// Fixed precision in a readable range, scientific outside.
fn num(v: f64) -> String {
    match v.abs() {
        a if a != 0.0 && !(1e-3..1e5).contains(&a) => format!("{v:.3e}"),
        _ => format!("{v:.4}"),
    }
}

fn opt(v: Option<f64>) -> String {
    v.map_or("-".into(), num)
}

/// `σ1 / σ_last` over the singular values the report keeps: the condition number when the
/// spectrum is complete, the spread of the top-r spectrum otherwise (`None` if singular).
fn spread(m: &NodeMetrics) -> Option<(String, f64)> {
    let first = f64::from(*m.singular_values.first()?);
    let last = f64::from(*m.singular_values.last()?);
    let label = match m.full_spectrum {
        true => "κ (σ1/σmin)".into(),
        false => format!("σ1/σ{}", m.singular_values.len()),
    };
    (last > 0.0).then(|| (label, first / last))
}

/// What deserves attention on a node, from its metrics and the report warnings.
fn node_warnings(report: &GraphReport, node: &NodeReport) -> Vec<String> {
    let m = &node.metrics;
    let mut out = Vec::new();
    if m.non_finite > 0 {
        out.push(format!("{} valeurs non finies", m.non_finite));
    }
    let threshold = report.parameters.kurtosis_threshold;
    if m.kurtosis > threshold {
        out.push(format!(
            "kurtosis {} > seuil {}",
            num(m.kurtosis),
            num(threshold)
        ));
    }
    if !m.channels.outliers.is_empty() {
        out.push(format!("{} canaux outliers", m.channels.outliers.len()));
    }
    out.extend(
        report
            .model
            .warnings
            .iter()
            .chain(&report.global.warnings)
            .filter(|w| w.contains(&node.id) || w.contains(&node.tensor))
            .cloned(),
    );
    out
}

fn section(ui: &mut egui::Ui, title: &str) {
    ui.add_space(6.0);
    ui.label(RichText::new(title).small().color(theme::DIM));
}

fn rows(ui: &mut egui::Ui, id: &str, rows: &[(&str, String)]) {
    Grid::new(id)
        .num_columns(2)
        .striped(true)
        .spacing([10.0, 3.0])
        .show(ui, |ui| {
            for (key, value) in rows {
                ui.label(RichText::new(*key).color(theme::DIM));
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.label(mono(value.as_str()));
                });
                ui.end_row();
            }
        });
}

fn node_inspector(ui: &mut egui::Ui, step: &Step, node: &NodeReport) {
    let m = &node.metrics;
    ui.label(
        RichText::new(node_title(node))
            .strong()
            .color(theme::SELECTION),
    );
    ui.label(mono(&node.tensor).small().color(theme::DIM));

    section(ui, "TENSEUR");
    rows(
        ui,
        "tensor",
        &[
            ("forme", format!("{} × {}", node.rows, node.cols)),
            ("dtype", node.dtype.clone()),
            ("octets", node.weight_bytes.to_string()),
            ("id", node.id.clone()),
        ],
    );

    section(ui, "SPECTRE");
    let interval = |lo: Option<f64>, hi: Option<f64>| match (lo, hi) {
        (Some(lo), Some(hi)) => format!("[{}, {}]", num(lo), num(hi)),
        _ => "-".into(),
    };
    let mut spectral = vec![
        ("σ1 (norme spectrale)", opt(m.sigma1)),
        ("σ1 certifié", interval(m.sigma1_lower, m.sigma1_upper)),
        ("rang stable", opt(m.d_spec)),
        (
            "rang stable certifié",
            interval(m.d_spec_lower, m.d_spec_upper),
        ),
        (
            "erank énergie",
            format!(
                "[{}, {}]",
                num(m.erank_energy_lower),
                num(m.erank_energy_upper)
            ),
        ),
        ("‖A‖F", num(m.frobenius_norm)),
        ("rang r", m.rank.to_string()),
        ("énergie captée", num(m.energy)),
        ("kurtosis", num(m.kurtosis)),
    ];
    let spread = spread(m);
    if let Some((label, value)) = &spread {
        spectral.push((label.as_str(), num(*value)));
    }
    rows(ui, "spectral", &spectral);

    let warnings = node_warnings(&step.report, node);
    if !warnings.is_empty() {
        section(ui, "AVERTISSEMENTS");
        for w in &warnings {
            ui.label(RichText::new(w).small().color(theme::WARN));
        }
    }

    let baseline = step
        .report
        .baseline_diff
        .as_ref()
        .and_then(|d| d.nodes.iter().find(|n| n.id == node.id));
    if let Some(diff) = baseline {
        section(ui, "BASELINE VS ACTUEL (Δ)");
        delta_table(ui, "baseline", m, diff);
    }
    if let Some(Some(diff)) = step.touched.get(&node.id) {
        section(ui, "ÉTAT PRÉCÉDENT VS ACTUEL (Δ PATCH)");
        delta_table(ui, "patch", m, diff);
    }
}

/// Rows `metric | before | now | Δ`; before is now minus the delta.
fn delta_table(ui: &mut egui::Ui, id: &str, m: &NodeMetrics, diff: &NodeDiff) {
    let erank = (m.erank_energy_lower + m.erank_energy_upper) / 2.0;
    let mut lines = vec![
        ("‖A‖F", m.frobenius_norm, diff.norm_delta, theme::TEXT),
        (
            "erank",
            erank,
            diff.effective_rank_delta,
            match diff.effective_rank_delta {
                d if d > 0.0 => theme::GOOD,
                d if d < 0.0 => theme::BAD,
                _ => theme::TEXT,
            },
        ),
    ];
    if let (Some(s), Some(d)) = (m.sigma1, diff.sigma1_delta) {
        lines.push(("σ1", s, d, theme::TEXT));
    }
    Grid::new(id)
        .num_columns(4)
        .striped(true)
        .spacing([10.0, 3.0])
        .show(ui, |ui| {
            for head in ["", "base", "actuel", "Δ"] {
                ui.label(RichText::new(head).small().color(theme::DIM));
            }
            ui.end_row();
            for (name, now, delta, color) in lines {
                ui.label(RichText::new(name).color(theme::DIM));
                ui.label(mono(num(now - delta)));
                ui.label(mono(num(now)));
                ui.label(mono(format!("{delta:+.4}")).color(color));
                ui.end_row();
            }
        });
    let rel = format!("{:+.2} %", diff.relative_change * 100.0);
    ui.label(match diff.changed {
        true => mono(format!("Δ‖A‖F relatif {rel} · modifié")).small(),
        false => mono(format!("Δ‖A‖F relatif {rel} · inchangé"))
            .small()
            .color(theme::DIM),
    });
}

fn model_summary(ui: &mut egui::Ui, report: &GraphReport) {
    let (m, g) = (&report.model, &report.global);
    ui.label(RichText::new(&m.architecture).strong());
    ui.label(mono(&m.path).small().color(theme::DIM));
    section(ui, "MODÈLE");
    rows(
        ui,
        "model",
        &[
            ("format", m.format.clone()),
            ("couches", m.layers.to_string()),
            ("hidden", m.hidden.to_string()),
            ("tenseurs", m.tensor_count.to_string()),
            ("sha256", m.sha256.chars().take(12).collect()),
        ],
    );
    section(ui, "GRAPHE");
    rows(
        ui,
        "global",
        &[
            ("nœuds", g.nodes.to_string()),
            ("arêtes", g.edges.to_string()),
            ("kurtosis médiane", num(g.kurtosis_median)),
            ("kurtosis max", num(g.kurtosis_max)),
            ("canaux outliers", g.outlier_channels.len().to_string()),
            ("valeurs non finies", g.non_finite_values.to_string()),
        ],
    );
    if let Some(diff) = &report.baseline_diff {
        section(ui, "BASELINE");
        rows(
            ui,
            "baseline",
            &[
                (
                    "nœuds modifiés",
                    format!("{}/{}", diff.nodes_changed, diff.nodes_compared),
                ),
                (
                    "arêtes modifiées",
                    format!("{}/{}", diff.edges_changed, diff.edges_compared),
                ),
            ],
        );
        for layer in &diff.top_changed_layers {
            let name = layer
                .layer
                .map_or("hors couches".into(), |l| format!("Layer {l}"));
            // max_relative_change is an absolute value: no sign to show.
            ui.label(mono(format!(
                "{name:<12} |Δ| {:.2} %  {}",
                layer.max_relative_change * 100.0,
                layer.top_node
            )));
        }
    }
    let warnings: Vec<&String> = m.warnings.iter().chain(&g.warnings).collect();
    if !warnings.is_empty() {
        section(ui, "AVERTISSEMENTS");
        for w in warnings {
            ui.label(RichText::new(w).small().color(theme::WARN));
        }
    }
    ui.add_space(8.0);
    ui.label(
        RichText::new("Sélectionnez un nœud pour l'inspecter.")
            .small()
            .color(theme::DIM),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report() -> GraphReport {
        let line = include_str!("../tests/fixtures/qwen2-mini.ndjson");
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
    }

    #[test]
    fn spread_is_a_condition_number_only_with_the_full_spectrum() {
        let mut m = report().nodes[0].metrics.clone();
        m.singular_values = vec![4.0, 2.0, 0.5];
        m.full_spectrum = true;
        assert_eq!(spread(&m), Some(("κ (σ1/σmin)".into(), 8.0)));
        m.full_spectrum = false;
        assert_eq!(spread(&m).unwrap().0, "σ1/σ3");
        m.singular_values = vec![1.0, 0.0];
        assert_eq!(spread(&m), None);
    }

    #[test]
    fn warnings_come_from_the_metrics_and_the_report() {
        let mut report = report();
        let node = report.nodes[2].clone();
        assert!(node_warnings(&report, &node).is_empty());
        report
            .global
            .warnings
            .push(format!("{} : spectre tronqué", node.id));
        let mut noisy = node;
        noisy.metrics.non_finite = 3;
        noisy.metrics.kurtosis = report.parameters.kurtosis_threshold + 1.0;
        let found = node_warnings(&report, &noisy);
        assert_eq!(found.len(), 3, "{found:?}");
        assert!(found[0].starts_with("3 valeurs"));
        assert!(found[1].starts_with("kurtosis"));
    }

    #[test]
    fn the_fixture_carries_the_baseline_the_diff_view_needs() {
        let report = report();
        let diff = report.baseline_diff.as_ref().unwrap();
        assert_eq!(diff.nodes_changed, 1);
        let changed = diff.nodes.iter().find(|n| n.changed).unwrap();
        assert_eq!(changed.id, "L1.attn_out");
        let node = report.nodes.iter().find(|n| n.id == changed.id).unwrap();
        assert_eq!(node_title(node), "Layer 1 · o_proj");
    }
}
