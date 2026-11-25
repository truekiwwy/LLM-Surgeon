//! The backend setup: one modal over the user `config.toml` that the CLI reads too
//! (`weight_surgeon::backend::config`), Rust engines only: mistral.rs, hipfire, native. The
//! first launch writes the file itself (`backend_choice`, from the hardware); the modal opens
//! from `Configurer…`, or when the file cannot be read. A save by hand ends the automatic
//! choice.

use std::path::PathBuf;

use eframe::egui::{self, Align, Grid, Layout, RichText, TextEdit};
use weight_surgeon::backend::config::{self, BackendConfig, Requantize};
use weight_surgeon::backend::detect::BackendReport;
use weight_surgeon::bench::score::Backend;

use crate::theme;
use crate::ui::components::{self as c, ButtonKind};
use crate::ui::i18n::{t, tf};
use crate::ui::icons::Icon;

/// Default of the form when the user file has no value.
const BATCH_SIZE: usize = 1;

pub struct BackendDialog {
    pub open: bool,
    /// The user file (`None`: disabled by the environment, nothing can be saved).
    path: Option<PathBuf>,
    /// What the user file holds; its other fields (`hipfire_bin`, `gguf_dtype`...) are kept.
    saved: BackendConfig,
    backend: Backend,
    mistralrs_bin: String,
    hipfire_url: String,
    batch_size: String,
    /// `hipfire_requantize` `auto` (the default): a SafeTensors folder is requantized under
    /// the alias of its weights before its measure.
    requantize: bool,
    /// Saved by hand since the app last asked ([`Self::take_saved`]).
    saved_by_hand: bool,
    /// Why the last load or save failed.
    error: Option<String>,
}

impl BackendDialog {
    /// Loads the user file; opens at once only when it cannot be read.
    pub fn load() -> Self {
        let path = config::user_config_path();
        let (saved, error) = match path.as_deref().map(config::load_from) {
            Some(Err(e)) => (BackendConfig::default(), Some(e)),
            Some(Ok(saved)) => (saved, None),
            None => (BackendConfig::default(), None),
        };
        let open = error.is_some();
        Self::from_config(path, saved, open, error)
    }

    fn from_config(
        path: Option<PathBuf>,
        saved: BackendConfig,
        open: bool,
        error: Option<String>,
    ) -> Self {
        let mut dialog = Self {
            open,
            path,
            backend: Backend::Native,
            mistralrs_bin: String::new(),
            hipfire_url: String::new(),
            batch_size: String::new(),
            requantize: true,
            saved_by_hand: false,
            saved: BackendConfig::default(),
            error,
        };
        dialog.show_config(saved);
        dialog
    }

    /// The form shows `saved`, which becomes what the file holds.
    fn show_config(&mut self, saved: BackendConfig) {
        // llama.cpp is not offered: a file that still names it shows native until saved.
        self.backend = match saved.backend {
            Some(b @ (Backend::MistralRs | Backend::Hipfire)) => b,
            _ => Backend::Native,
        };
        self.mistralrs_bin = shown(&saved.mistralrs_bin);
        self.hipfire_url = saved.hipfire_url.clone().unwrap_or_default();
        self.batch_size = saved.batch_size.unwrap_or(BATCH_SIZE).to_string();
        self.requantize = saved.hipfire_requantize != Some(Requantize::Off);
        self.saved = saved;
    }

    /// The saved configuration, as the CLI resolves it.
    pub fn saved(&self) -> &BackendConfig {
        &self.saved
    }

    /// Nothing will be written (a capture): the file is still read, never saved.
    pub fn read_only(&mut self) {
        self.path = None;
    }

    /// The user file can be written (not disabled by the environment).
    pub fn writable(&self) -> bool {
        self.path.is_some()
    }

    /// `true` once after a save by hand.
    pub fn take_saved(&mut self) -> bool {
        std::mem::take(&mut self.saved_by_hand)
    }

    /// Fills an empty `mistralrs_bin` from the probe (never overrides a typed value).
    pub fn prefill(&mut self, report: &BackendReport) {
        if self.mistralrs_bin.is_empty() {
            self.mistralrs_bin = shown(&report.mistralrs.as_ref().map(|m| m.binary.clone()));
        }
    }

    /// Writes `cfg` (the automatic choice) and shows it.
    pub fn save_config(&mut self, cfg: BackendConfig) -> Result<(), String> {
        let path = self.path.clone().ok_or(t("backend.disabled"))?;
        config::save_to(&path, &cfg)?;
        self.show_config(cfg);
        Ok(())
    }

    /// The form as a configuration; numbers are checked here, at the trust boundary.
    fn to_config(&self) -> Result<BackendConfig, String> {
        let batch_size = self
            .batch_size
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&b| b >= 1)
            .ok_or_else(|| tf("backend.batch_size.invalid", &[("value", &self.batch_size)]))?;
        let text = |text: &str| Some(text.trim().to_owned()).filter(|t| !t.is_empty());
        // Ticked over an absent key keeps it absent (absent is `auto`); unticked is `off`.
        let requantize = match (self.requantize, self.saved.hipfire_requantize) {
            (true, None) => None,
            (true, Some(_)) => Some(Requantize::Auto),
            (false, _) => Some(Requantize::Off),
        };
        Ok(BackendConfig {
            backend: Some(self.backend),
            batch_size: Some(batch_size),
            mistralrs_bin: text(&self.mistralrs_bin).map(PathBuf::from),
            hipfire_url: text(&self.hipfire_url),
            hipfire_requantize: requantize,
            ..self.saved.clone()
        })
    }

    fn save(&mut self) {
        let Some(path) = self.path.clone() else {
            return;
        };
        match self
            .to_config()
            .and_then(|cfg| config::save_to(&path, &cfg).map(|()| cfg))
        {
            Ok(cfg) => {
                self.saved = cfg;
                self.saved_by_hand = true;
                self.error = None;
                self.open = false;
            }
            Err(e) => self.error = Some(e),
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, report: Option<&BackendReport>) {
        if !self.open {
            return;
        }
        let mut close = false;
        let modal = egui::Modal::new(egui::Id::new("backend-setup")).show(ctx, |ui| {
            ui.set_width(460.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(t("backend.title")).strong());
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    close = c::icon_button(ui, Icon::Close, t("settings.close")).clicked();
                });
            });
            ui.label(RichText::new(t("backend.lead")).small().color(theme::DIM));
            ui.separator();
            self.choices(ui, report);
            ui.separator();
            self.fields(ui);
            ui.separator();
            self.buttons(ui);
        });
        if close || modal.should_close() {
            self.discard();
        }
    }

    fn discard(&mut self) {
        self.show_config(self.saved.clone());
        self.error = None;
        self.open = false;
    }

    fn choices(&mut self, ui: &mut egui::Ui, report: Option<&BackendReport>) {
        let mistralrs_note = match report.and_then(|r| r.mistralrs.as_ref()) {
            Some(m) => tf("backend.mistralrs.found", &[("path", &m.binary.display())]),
            None => t("backend.mistralrs.missing").into(),
        };
        let hipfire_note = report.map_or(t("backend.detecting").into(), |r| r.hipfire.note.clone());
        for (backend, name, note) in [
            (Backend::MistralRs, "mistralrs", mistralrs_note),
            (Backend::Hipfire, "hipfire", hipfire_note),
            (Backend::Native, "native", t("backend.native.note").into()),
        ] {
            ui.radio_value(&mut self.backend, backend, name);
            ui.label(RichText::new(note).small().color(theme::DIM));
        }
    }

    fn fields(&mut self, ui: &mut egui::Ui) {
        let server = matches!(self.backend, Backend::Hipfire | Backend::MistralRs);
        Grid::new("backend-fields")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                for (name, value) in [
                    ("mistralrs_bin", &mut self.mistralrs_bin),
                    ("hipfire_url", &mut self.hipfire_url),
                ] {
                    ui.label(theme::mono(name));
                    ui.add(
                        TextEdit::singleline(value)
                            .font(egui::TextStyle::Monospace)
                            .desired_width(300.0),
                    );
                    ui.end_row();
                }
                ui.label(theme::mono("batch_size"));
                ui.add_enabled(
                    server,
                    TextEdit::singleline(&mut self.batch_size)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(60.0),
                );
                ui.end_row();
                ui.label(theme::mono("hipfire_requantize"));
                ui.checkbox(&mut self.requantize, "auto")
                    .on_hover_text(t("backend.requantize.tip"));
                ui.end_row();
            });
    }

    fn buttons(&mut self, ui: &mut egui::Ui) {
        let target = match &self.path {
            Some(path) => theme::mono(path.display().to_string()).color(theme::DIM),
            None => RichText::new(t("backend.disabled")).color(theme::WARN),
        };
        let shown = ui.add(egui::Label::new(target.small()).truncate());
        if let Some(path) = &self.path {
            shown.on_hover_text(path.display().to_string());
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            let why = t("backend.disabled");
            if c::button_if(
                ui,
                self.path.is_some(),
                t("backend.save"),
                ButtonKind::Primary,
                why,
            )
            .clicked()
            {
                self.save();
            }
            if c::button(ui, t("action.cancel"), ButtonKind::Ghost).clicked() {
                self.discard();
            }
        });
        if let Some(e) = &self.error {
            ui.label(RichText::new(e).small().color(theme::BAD));
        }
    }
}

/// A path as the form shows it, empty when unset.
fn shown(path: &Option<PathBuf>) -> String {
    path.as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dialog(saved: BackendConfig) -> BackendDialog {
        BackendDialog::from_config(None, saved, false, None)
    }

    #[test]
    fn the_form_offers_rust_engines_only() {
        let empty = dialog(BackendConfig::default());
        assert_eq!(
            (empty.backend, empty.batch_size.as_str()),
            (Backend::Native, "1")
        );
        let llama = dialog(BackendConfig {
            backend: Some(Backend::LlamaServer),
            ..Default::default()
        });
        assert_eq!(llama.backend, Backend::Native, "llama.cpp is not offered");
        let amd = dialog(BackendConfig {
            backend: Some(Backend::Hipfire),
            hipfire_url: Some("http://127.0.0.1:11435".into()),
            ..Default::default()
        });
        assert_eq!(amd.backend, Backend::Hipfire);
        assert_eq!(amd.hipfire_url, "http://127.0.0.1:11435");
    }

    #[test]
    fn the_form_is_validated_and_keeps_the_other_fields() {
        let mut form = dialog(BackendConfig {
            hipfire_bin: Some(PathBuf::from("/opt/hipfire")),
            gpu_layers: Some(20),
            ..Default::default()
        });
        form.backend = Backend::MistralRs;
        form.mistralrs_bin = "  C:/m/mistralrs.exe ".into();
        let cfg = form.to_config().unwrap();
        assert_eq!(cfg.backend, Some(Backend::MistralRs));
        assert_eq!(cfg.mistralrs_bin, Some(PathBuf::from("C:/m/mistralrs.exe")));
        assert_eq!(
            (cfg.hipfire_bin.clone(), cfg.gpu_layers),
            (Some(PathBuf::from("/opt/hipfire")), Some(20))
        );
        assert_eq!(config::parse_toml(&config::to_toml(&cfg)).unwrap(), cfg);
        form.hipfire_url.clear();
        assert_eq!(form.to_config().unwrap().hipfire_url, None);
        // hipfire_requantize: absent is auto (ticked) and stays absent, unticked is off,
        // ticked again after it is auto.
        assert!(form.requantize, "auto by default");
        assert_eq!(form.to_config().unwrap().hipfire_requantize, None);
        form.requantize = false;
        let off = form.to_config().unwrap();
        assert_eq!(off.hipfire_requantize, Some(Requantize::Off));
        assert_eq!(config::parse_toml(&config::to_toml(&off)).unwrap(), off);
        let mut form = dialog(off);
        assert!(!form.requantize, "shown from the file");
        form.requantize = true;
        assert_eq!(
            form.to_config().unwrap().hipfire_requantize,
            Some(Requantize::Auto)
        );
        form.batch_size = "0".into();
        assert!(form.to_config().unwrap_err().contains("batch_size"));
    }

    #[test]
    fn closing_without_saving_drops_the_edits() {
        let mut form = dialog(BackendConfig {
            backend: Some(Backend::Hipfire),
            ..Default::default()
        });
        form.open = true;
        form.backend = Backend::Native;
        form.batch_size = "0".into();
        form.error = Some("batch_size".into());
        form.discard();
        assert_eq!(
            (
                form.backend,
                form.batch_size.as_str(),
                form.open,
                form.error
            ),
            (Backend::Hipfire, "1", false, None)
        );
    }

    #[test]
    fn an_automatic_save_is_not_a_choice_by_hand() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weight-surgeon").join("config.toml");
        let mut form =
            BackendDialog::from_config(Some(path.clone()), BackendConfig::default(), true, None);
        let auto = BackendConfig {
            backend: Some(Backend::MistralRs),
            ..Default::default()
        };
        form.save_config(auto.clone()).unwrap();
        assert_eq!((form.saved(), form.backend), (&auto, Backend::MistralRs));
        assert!(!form.take_saved(), "written by the app");
        form.backend = Backend::Native;
        form.save();
        assert!(!form.open && form.take_saved(), "saved by hand");
        assert!(!form.take_saved(), "once");
        assert_eq!(
            config::load_from(&path).unwrap().backend,
            Some(Backend::Native)
        );
    }
}
