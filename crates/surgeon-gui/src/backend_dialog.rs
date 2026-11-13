//! The backend setup: one modal over the user `config.toml` that the CLI reads too
//! (`weight_surgeon::backend::config`), prefilled by the hardware probe. Opened at first
//! start (no user file yet) and from the `Backend…` button.

use std::path::PathBuf;

use eframe::egui::{self, Align, Grid, Layout, RichText, TextEdit};
use weight_surgeon::backend::config::{self, BackendConfig};
use weight_surgeon::backend::detect::BackendReport;
use weight_surgeon::bench::score::Backend;

use crate::theme;

/// Defaults of the form when the user file has no value.
const GPU_LAYERS: u32 = 99;
const BATCH_SIZE: usize = 6;

pub struct BackendDialog {
    pub open: bool,
    /// The user file (`None`: disabled by the environment, nothing can be saved).
    path: Option<PathBuf>,
    /// What the user file holds; its other fields (`hipfire_bin`, `gguf_dtype`...) are kept.
    saved: BackendConfig,
    backend: Backend,
    /// The user picked a backend: the probe, answering later, no longer changes it.
    touched: bool,
    llama_cpp_dir: String,
    mistralrs_bin: String,
    gpu_layers: String,
    batch_size: String,
    /// Why the last load or save failed.
    error: Option<String>,
}

impl BackendDialog {
    /// Loads the user file; opens at once when there is none yet or it cannot be read.
    pub fn load() -> Self {
        let path = config::user_config_path();
        let (saved, error) = match path.as_deref().map(config::load_from) {
            Some(Err(e)) => (BackendConfig::default(), Some(e)),
            Some(Ok(saved)) => (saved, None),
            None => (BackendConfig::default(), None),
        };
        let open = error.is_some() || path.as_ref().is_some_and(|p| !p.exists());
        Self::from_config(path, saved, open, error)
    }

    fn from_config(
        path: Option<PathBuf>,
        saved: BackendConfig,
        open: bool,
        error: Option<String>,
    ) -> Self {
        Self {
            open,
            path,
            backend: saved.backend.unwrap_or(Backend::Native),
            touched: false,
            llama_cpp_dir: shown(&saved.llama_cpp_dir),
            mistralrs_bin: shown(&saved.mistralrs_bin),
            gpu_layers: saved.gpu_layers.unwrap_or(GPU_LAYERS).to_string(),
            batch_size: saved.batch_size.unwrap_or(BATCH_SIZE).to_string(),
            saved,
            error,
        }
    }

    /// The saved configuration, as the CLI resolves it.
    pub fn saved(&self) -> &BackendConfig {
        &self.saved
    }

    /// Fills what the user file and the user left empty from the probe (never overrides a
    /// saved value, nor a backend already clicked).
    pub fn prefill(&mut self, report: &BackendReport) {
        if self.saved.backend.is_none() && !self.touched {
            self.backend = report.recommended.backend.unwrap_or(Backend::Native);
        }
        if self.llama_cpp_dir.is_empty() {
            self.llama_cpp_dir = shown(&report.llama.as_ref().map(|l| l.dir.clone()));
        }
        if self.mistralrs_bin.is_empty() {
            self.mistralrs_bin = shown(&report.mistralrs.as_ref().map(|m| m.binary.clone()));
        }
    }

    /// The form as a configuration; numbers are checked here, at the trust boundary.
    fn to_config(&self) -> Result<BackendConfig, String> {
        let gpu_layers = self
            .gpu_layers
            .trim()
            .parse::<u32>()
            .map_err(|_| format!("gpu_layers « {} » : entier attendu", self.gpu_layers))?;
        let batch_size = self
            .batch_size
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&b| b >= 1)
            .ok_or_else(|| format!("batch_size « {} » : entier >= 1 attendu", self.batch_size))?;
        let path = |text: &str| Some(text.trim()).filter(|t| !t.is_empty()).map(PathBuf::from);
        Ok(BackendConfig {
            backend: Some(self.backend),
            gpu_layers: Some(gpu_layers),
            batch_size: Some(batch_size),
            llama_cpp_dir: path(&self.llama_cpp_dir),
            mistralrs_bin: path(&self.mistralrs_bin),
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
        let modal = egui::Modal::new(egui::Id::new("backend-setup")).show(ctx, |ui| {
            ui.set_width(460.0);
            ui.label(RichText::new("Backend d'inférence").strong());
            ui.label(
                RichText::new(match report {
                    Some(r) => r.reason.clone(),
                    None => "détection du matériel…".into(),
                })
                .small()
                .color(theme::DIM),
            );
            ui.separator();
            self.choices(ui, report);
            ui.separator();
            self.fields(ui);
            ui.separator();
            self.buttons(ui);
        });
        if modal.should_close() {
            self.open = false;
        }
    }

    fn choices(&mut self, ui: &mut egui::Ui, report: Option<&BackendReport>) {
        let gpu = report
            .and_then(|r| r.llama.as_ref())
            .and_then(|l| l.devices.first());
        let mistralrs_note = match report.and_then(|r| r.mistralrs.as_ref()) {
            Some(_) => "recommandé · mistral.rs, dossier SafeTensors tel quel, GPU selon le build",
            None => "mistral.rs (mistralrs-server), dossier SafeTensors tel quel, GPU selon le build",
        };
        let llama_note = match gpu {
            Some(gpu) => format!("{} ({}) via llama.cpp", gpu.name, gpu.api()),
            None => "llama.cpp, GPU selon le build (Vulkan, CUDA, Metal)".into(),
        };
        // The probe may still be running: no choice waits for it, and a click is final.
        let hipfire_note = report.map_or("détection en cours".into(), |r| r.hipfire.note.clone());
        for (backend, name, note) in [
            (Backend::MistralRs, "mistralrs", mistralrs_note.to_string()),
            (Backend::LlamaServer, "llama-server", llama_note),
            (Backend::Hipfire, "hipfire", hipfire_note),
            (Backend::Native, "native", "CPU, Rust pur, dossier SafeTensors".into()),
        ] {
            if ui.radio_value(&mut self.backend, backend, name).changed() {
                self.touched = true;
            }
            ui.label(RichText::new(note).small().color(theme::DIM));
        }
    }

    fn fields(&mut self, ui: &mut egui::Ui) {
        let llama = self.backend == Backend::LlamaServer;
        let server = matches!(
            self.backend,
            Backend::LlamaServer | Backend::Hipfire | Backend::MistralRs
        );
        Grid::new("backend-fields")
            .num_columns(2)
            .spacing([12.0, 6.0])
            .show(ui, |ui| {
                ui.label(theme::mono("llama_cpp_dir"));
                ui.add(
                    TextEdit::singleline(&mut self.llama_cpp_dir)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(300.0),
                );
                ui.end_row();
                ui.label(theme::mono("mistralrs_bin"));
                ui.add(
                    TextEdit::singleline(&mut self.mistralrs_bin)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(300.0),
                );
                ui.end_row();
                ui.label(theme::mono("gpu_layers"));
                ui.add_enabled(
                    llama,
                    TextEdit::singleline(&mut self.gpu_layers)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(60.0),
                );
                ui.end_row();
                ui.label(theme::mono("batch_size"));
                ui.add_enabled(
                    server,
                    TextEdit::singleline(&mut self.batch_size)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(60.0),
                );
                ui.end_row();
            });
    }

    fn buttons(&mut self, ui: &mut egui::Ui) {
        let target = match &self.path {
            Some(path) => theme::mono(path.display().to_string()).color(theme::DIM),
            None => RichText::new(
                "configuration utilisateur désactivée (WEIGHT_SURGEON_NO_USER_CONFIG)",
            )
            .color(theme::WARN),
        };
        let shown = ui.add(egui::Label::new(target.small()).truncate());
        if let Some(path) = &self.path {
            shown.on_hover_text(path.display().to_string());
        }
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            if ui
                .add_enabled(self.path.is_some(), egui::Button::new("Enregistrer"))
                .clicked()
            {
                self.save();
            }
            if ui.button("Annuler").clicked() {
                self.open = false;
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
    fn the_form_starts_from_the_file_then_from_the_defaults() {
        let empty = dialog(BackendConfig::default());
        assert_eq!(empty.backend, Backend::Native);
        assert_eq!(
            (empty.gpu_layers.as_str(), empty.batch_size.as_str()),
            ("99", "6")
        );
        let saved = dialog(BackendConfig {
            backend: Some(Backend::LlamaServer),
            gpu_layers: Some(20),
            ..Default::default()
        });
        assert_eq!(saved.backend, Backend::LlamaServer);
        assert_eq!(saved.gpu_layers, "20");
    }

    #[test]
    fn the_form_is_validated_and_keeps_the_other_fields() {
        let mut form = dialog(BackendConfig {
            hipfire_bin: Some(PathBuf::from("/opt/hipfire")),
            ..Default::default()
        });
        form.backend = Backend::LlamaServer;
        form.llama_cpp_dir = "  C:/tools/llama-vulkan ".into();
        let cfg = form.to_config().unwrap();
        assert_eq!(cfg.backend, Some(Backend::LlamaServer));
        assert_eq!((cfg.gpu_layers, cfg.batch_size), (Some(99), Some(6)));
        assert_eq!(
            cfg.llama_cpp_dir,
            Some(PathBuf::from("C:/tools/llama-vulkan"))
        );
        assert_eq!(cfg.hipfire_bin, Some(PathBuf::from("/opt/hipfire")));
        // The written file reads back to the same configuration.
        assert_eq!(config::parse_toml(&config::to_toml(&cfg)).unwrap(), cfg);

        form.llama_cpp_dir.clear();
        assert_eq!(form.to_config().unwrap().llama_cpp_dir, None);
        form.batch_size = "0".into();
        assert!(form.to_config().unwrap_err().contains("batch_size"));
        form.batch_size = "4".into();
        form.gpu_layers = "-1".into();
        assert!(form.to_config().unwrap_err().contains("gpu_layers"));
    }

    fn report() -> BackendReport {
        use weight_surgeon::backend::detect::{HipfireDetection, MistralRsDetection};
        BackendReport {
            os: "windows",
            mistralrs: Some(MistralRsDetection {
                binary: PathBuf::from("C:/tools/mistralrs-server.exe"),
            }),
            llama: None,
            hipfire: HipfireDetection {
                binary: None,
                note: "hipfire introuvable".into(),
            },
            recommended: BackendConfig {
                backend: Some(Backend::MistralRs),
                ..Default::default()
            },
            reason: "mistral.rs".into(),
        }
    }

    #[test]
    fn the_probe_fills_the_form_until_the_user_picks() {
        let mut form = dialog(BackendConfig::default());
        form.prefill(&report());
        assert_eq!(form.backend, Backend::MistralRs);
        assert_eq!(form.mistralrs_bin, "C:/tools/mistralrs-server.exe");
        assert_eq!(
            form.to_config().unwrap().mistralrs_bin,
            Some(PathBuf::from("C:/tools/mistralrs-server.exe"))
        );

        // hipfire clicked before the probe answered: the answer does not take it back.
        let mut form = dialog(BackendConfig::default());
        form.backend = Backend::Hipfire;
        form.touched = true;
        form.prefill(&report());
        assert_eq!(form.backend, Backend::Hipfire);
    }

    #[test]
    fn saving_writes_the_user_file_and_closes() {
        let dir = std::env::temp_dir().join(format!("surgeon-gui-cfg-{}", std::process::id()));
        let path = dir.join("weight-surgeon").join("config.toml");
        let mut form =
            BackendDialog::from_config(Some(path.clone()), BackendConfig::default(), true, None);
        form.save();
        assert!(!form.open);
        assert_eq!(form.error, None);
        let back = config::load_from(&path).unwrap();
        assert_eq!(&back, form.saved());
        assert_eq!(back.backend, Some(Backend::Native));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
