//! The editor drawer: file and model fields, the run buttons, the text with its gutter, and
//! each diagnostic painted on its own line.

use eframe::egui::{
    self, Align2, FontId, Key, KeyboardShortcut, Margin, Modifiers, RichText, ScrollArea, Stroke,
    TextEdit,
};

use super::{highlight, Editor, Run};
use crate::theme::{self, mono};

/// Width of the line-number gutter, inside the text margin.
const GUTTER: f32 = 34.0;

/// What the user asked for this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Request {
    Run(Run),
    Rollback,
}

/// `busy`: a job runs, nothing else can start. `can_rollback`: the last apply left backups.
pub fn show(
    ui: &mut egui::Ui,
    editor: &mut Editor,
    busy: bool,
    can_rollback: bool,
) -> Option<Request> {
    let mut request = None;
    let save = KeyboardShortcut::new(Modifiers::COMMAND, Key::S);
    let dry = KeyboardShortcut::new(Modifiers::COMMAND, Key::Enter);
    // Consumed before the text edit, which would otherwise take Enter as a newline.
    if ui.input_mut(|i| i.consume_shortcut(&save)) {
        editor.message = Some(match editor.export() {
            Ok(()) => (format!("enregistré : {}", editor.file.trim()), false),
            Err(e) => (e, true),
        });
    }
    if ui.input_mut(|i| i.consume_shortcut(&dry)) && !busy {
        request = Some(Request::Run(Run::DryRun));
    }

    ui.label(RichText::new("PATCH.JSON").small().color(theme::DIM));
    egui::Grid::new("editor-fields")
        .num_columns(2)
        .spacing([6.0, 4.0])
        .show(ui, |ui| {
            ui.label(RichText::new("fichier").color(theme::DIM));
            ui.horizontal(|ui| {
                ui.add(
                    TextEdit::singleline(&mut editor.file)
                        .desired_width(ui.available_width() - 120.0),
                );
                if ui.button("Importer").clicked() {
                    editor.message = editor.import().err().map(|e| (e, true));
                }
                if ui.button("Exporter").on_hover_text("Ctrl+S").clicked() {
                    editor.message = Some(match editor.export() {
                        Ok(()) => ("exporté".into(), false),
                        Err(e) => (e, true),
                    });
                }
            });
            ui.end_row();
            ui.label(RichText::new("modèle").color(theme::DIM));
            ui.add(
                TextEdit::singleline(&mut editor.model)
                    .hint_text("dossier SafeTensors ou fichier")
                    .desired_width(f32::INFINITY),
            );
            ui.end_row();
        });

    ui.horizontal(|ui| {
        let ready = !busy && editor.checked.valid;
        if ui
            .add_enabled(ready, egui::Button::new("Dry-Run"))
            .on_hover_text("simulation sans écriture (Ctrl+Entrée)")
            .clicked()
        {
            request = Some(Request::Run(Run::DryRun));
        }
        if ui
            .add_enabled(ready, egui::Button::new("Apply (avec backup)"))
            .on_hover_text("écrit le modèle après une copie vérifiée (SHA-256)")
            .clicked()
        {
            request = Some(Request::Run(Run::Apply));
        }
        if ui
            .add_enabled(!busy && can_rollback, egui::Button::new("Rollback"))
            .on_hover_text("restaure les fichiers du dernier Apply depuis leurs backups")
            .clicked()
        {
            request = Some(Request::Rollback);
        }
    });

    let errors = editor
        .checked
        .diagnostics
        .iter()
        .filter(|d| d.error)
        .count();
    let warnings = editor.checked.diagnostics.len() - errors;
    let status = match (errors, warnings) {
        (0, 0) => mono("valide").color(theme::GOOD),
        (0, w) => mono(format!("valide · {w} avertissement(s)")).color(theme::WARN),
        (e, w) => mono(format!("{e} erreur(s) · {w} avertissement(s)")).color(theme::BAD),
    };
    ui.horizontal(|ui| {
        ui.label(status);
        ui.label(
            mono(format!(
                "{} sélection(s) en 3D",
                editor.checked.selections.len()
            ))
            .color(theme::DIM),
        );
    });
    if let Some((text, error)) = &editor.message {
        let color = if *error { theme::BAD } else { theme::DIM };
        ui.label(RichText::new(text).small().color(color));
    }
    ui.separator();

    let font = FontId::monospace(12.0);
    let mut layouter = |ui: &egui::Ui, buf: &dyn egui::TextBuffer, _wrap: f32| {
        ui.painter()
            .layout_job(highlight(buf.as_str(), font.clone()))
    };
    ScrollArea::both().auto_shrink(false).show(ui, |ui| {
        let output = TextEdit::multiline(&mut editor.text)
            .code_editor()
            .lock_focus(true)
            .desired_width(f32::INFINITY)
            .desired_rows(24)
            .margin(Margin {
                left: GUTTER as i8 + 6,
                right: 6,
                top: 4,
                bottom: 4,
            })
            .layouter(&mut layouter)
            .show(ui);
        if output.response.changed() {
            editor.touch();
        }
        let painter = ui.painter_at(output.response.rect);
        let origin = output.galley_pos;
        let left = output.response.rect.left();
        painter.line_segment(
            [
                egui::pos2(left + GUTTER, output.response.rect.top()),
                egui::pos2(left + GUTTER, output.response.rect.bottom()),
            ],
            Stroke::new(1.0, theme::BORDER),
        );
        let marks: std::collections::HashMap<usize, bool> = editor
            .checked
            .diagnostics
            .iter()
            .map(|d| (d.line, d.error))
            .collect();
        for (line, row) in output.galley.rows.iter().enumerate() {
            let rect = row.rect().translate(origin.to_vec2());
            let color = match marks.get(&line) {
                Some(true) => theme::BAD,
                Some(false) => theme::WARN,
                None => theme::DIM,
            };
            painter.text(
                egui::pos2(left + GUTTER - 6.0, rect.center().y),
                Align2::RIGHT_CENTER,
                (line + 1).to_string(),
                FontId::monospace(10.5),
                color,
            );
        }
        // Each diagnostic on its line: a faint band and the message after the text.
        for d in &editor.checked.diagnostics {
            let Some(row) = output.galley.rows.get(d.line) else {
                continue;
            };
            let rect = row.rect().translate(origin.to_vec2());
            let color = if d.error { theme::BAD } else { theme::WARN };
            let band = egui::Rect::from_x_y_ranges(
                left + GUTTER..=output.response.rect.right(),
                rect.y_range(),
            );
            painter.rect_filled(band, 0.0, color.gamma_multiply(0.10));
            painter.text(
                egui::pos2(rect.right() + 18.0, rect.center().y),
                Align2::LEFT_CENTER,
                diagnostic_text(d),
                FontId::monospace(11.0),
                color,
            );
        }
    });
    request
}

fn diagnostic_text(d: &super::Diagnostic) -> String {
    match d.path.is_empty() {
        true => format!("<- {}", d.message),
        false => format!("<- {} : {}", d.path, d.message),
    }
}
