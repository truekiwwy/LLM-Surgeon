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
    /// What the user file holds; its other fields (`hipfire_bin`, `gguf_dtype`) are kept.
    saved: BackendConfig,
    backend: Backend,
    llama_cpp_dir: String,
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
            llama_cpp_dir: saved
                .llama_cpp_dir
                .as_ref()
                .map(|d| d.display().to_string())
                .unwrap_or_default(),
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

    /// Fills what the user file leaves empty from the probe (never overrides a saved value).
    pub fn prefill(&mut self, report: &BackendReport) {
        if self.saved.backend.is_none() {
            self.backend = report.recommended.backend.unwrap_or(Backend::Native);
        }
        if self.llama_cpp_dir.is_empty() {
            if let Some(llama) = &report.llama {
                self.llama_cpp_dir = llama.dir.display().to_string();
            }
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
        let dir = self.llama_cpp_dir.trim();
        Ok(BackendConfig {
            backend: Some(self.backend),
            gpu_layers: Some(gpu_layers),
            batch_size: Some(batch_size),
            llama_cpp_dir: (!dir.is_empty()).then(|| PathBuf::from(dir)),
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
        let llama_note = match gpu {
            Some(gpu) => format!("recommandé · {} ({})", gpu.name, gpu.api()),
            None => "llama.cpp, GPU selon le build (Vulkan, CUDA, Metal)".into(),
        };
        ui.radio_value(&mut self.backend, Backend::LlamaServer, "llama-server");
        ui.label(RichText::new(llama_note).small().color(theme::DIM));

        // Only the probe knows the OS verdict; until it answers, hipfire stays disabled.
        let hipfire_ok = report.is_some_and(|r| r.hipfire.supported_os);
        ui.add_enabled_ui(hipfire_ok, |ui| {
            ui.radio_value(&mut self.backend, Backend::Hipfire, "hipfire");
        });
        let hipfire_note = report.map_or("détection en cours".into(), |r| r.hipfire.note.clone());
        ui.label(RichText::new(hipfire_note).small().color(match hipfire_ok {
            true => theme::DIM,
            false => theme::WARN,
        }));

        ui.radio_value(&mut self.backend, Backend::Native, "native");
        ui.label(
            RichText::new("CPU, Rust pur, dossier SafeTensors")
                .small()
                .color(theme::DIM),
        );
    }

    fn fields(&mut self, ui: &mut egui::Ui) {
        let gpu = matches!(self.backend, Backend::LlamaServer | Backend::Hipfire);
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
                ui.label(theme::mono("gpu_layers"));
                ui.add_enabled(
                    gpu,
                    TextEdit::singleline(&mut self.gpu_layers)
                        .font(egui::TextStyle::Monospace)
                        .desired_width(60.0),
                );
                ui.end_row();
                ui.label(theme::mono("batch_size"));
                ui.add_enabled(
                    gpu,
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
