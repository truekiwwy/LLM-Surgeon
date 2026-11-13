//! What this machine can run, and the backend to recommend: the first-start probe of the CLI
//! (`surgeon backend detect`) and of a future GUI.

use std::path::{Path, PathBuf};

use serde::Serialize;

use super::config::BackendConfig;
use super::hipfire;
use crate::bench::score::Backend;
use crate::eval::{locate_binary, WeightType};

/// A device listed by `llama-server --list-devices`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GpuDevice {
    /// `Vulkan0`, `CUDA0`, `MTL0`...
    pub id: String,
    pub name: String,
    pub vram_mib: Option<u64>,
}

impl GpuDevice {
    /// `Vulkan`, `CUDA`...: the id without its index.
    pub fn api(&self) -> &str {
        self.id.trim_end_matches(|c: char| c.is_ascii_digit())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct LlamaDetection {
    pub server: PathBuf,
    /// Folder to record as `llama_cpp_dir`.
    pub dir: PathBuf,
    pub devices: Vec<GpuDevice>,
    /// `--list-devices` failed or timed out: the build may still work on the CPU.
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HipfireDetection {
    pub binary: Option<PathBuf>,
    /// The daemon's endpoint, or why it cannot be used.
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct MistralRsDetection {
    /// `mistralrs-server` or `mistralrs`.
    pub binary: PathBuf,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendReport {
    pub os: &'static str,
    pub mistralrs: Option<MistralRsDetection>,
    pub llama: Option<LlamaDetection>,
    pub hipfire: HipfireDetection,
    pub recommended: BackendConfig,
    pub reason: String,
}

/// One `llama-server --list-devices` line, `Vulkan0: AMD Radeon RX 7600 XT (16368 MiB, ...)`.
fn parse_device(line: &str) -> Option<GpuDevice> {
    let (id, rest) = line.trim().split_once(": ")?;
    let indexed = id.ends_with(|c: char| c.is_ascii_digit());
    let word = id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if !indexed || !word || id.to_ascii_lowercase().starts_with("cpu") {
        return None;
    }
    let (name, tail) = rest.rsplit_once(" (").unwrap_or((rest, ""));
    Some(GpuDevice {
        id: id.into(),
        name: name.trim().into(),
        vram_mib: tail.split_whitespace().next().and_then(|v| v.parse().ok()),
    })
}

pub fn parse_devices(output: &str) -> Vec<GpuDevice> {
    output.lines().filter_map(parse_device).collect()
}

/// The GPUs a `llama-server` build sees; the error when it cannot be asked.
pub fn list_devices(server: &Path) -> Result<Vec<GpuDevice>, String> {
    crate::eval::server::capture(server, &["--list-devices"])
        .map(|out| parse_devices(&out))
        .map_err(|e| e.to_string())
}

/// `~/tools/llama-vulkan`, where the docs unpack the prebuilt Windows GPU build. Only the
/// probe looks there: the commands themselves never guess a folder.
fn conventional_dir() -> Option<PathBuf> {
    let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"))?;
    Some(PathBuf::from(home).join("tools").join("llama-vulkan"))
}

fn detect_llama(user: &BackendConfig) -> Option<LlamaDetection> {
    // The configured folder, $LLAMA_CPP_DIR and the PATH first; the conventional folder last.
    let server = locate_binary("llama-server", user.llama_cpp_dir.as_deref())
        .ok()
        .or_else(|| {
            conventional_dir().and_then(|d| locate_binary("llama-server", Some(&d)).ok())
        })?;
    let (devices, error) = match list_devices(&server) {
        Ok(devices) => (devices, None),
        Err(e) => (Vec::new(), Some(e)),
    };
    Some(LlamaDetection {
        dir: server.parent().map(Path::to_path_buf).unwrap_or_default(),
        server,
        devices,
        error,
    })
}

/// Probes the machine (mistral.rs, llama.cpp and its devices, hipfire) and recommends a
/// backend: mistral.rs when installed (it reads the SafeTensors folder as is), else
/// `llama-server` on the GPU when one is usable, else the native CPU engine. hipfire is never
/// recommended: it is an opt-in AMD daemon. The engines found are recorded either way, so
/// that an explicit `--backend` finds them.
pub fn detect_hardware_and_backends(user: &BackendConfig) -> BackendReport {
    let mistralrs = super::mistralrs::locate(user).map(|binary| MistralRsDetection { binary });
    let llama = detect_llama(user);
    let hipfire = hipfire::detect(user);
    let gpu = llama.as_ref().and_then(|l| l.devices.first());
    let found = BackendConfig {
        llama_cpp_dir: llama.as_ref().map(|l| l.dir.clone()),
        mistralrs_bin: mistralrs.as_ref().map(|m| m.binary.clone()),
        hipfire_bin: hipfire.binary.clone(),
        hipfire_url: user.hipfire_url.clone(),
        ..Default::default()
    };
    let (recommended, reason) = match (&mistralrs, &llama, gpu) {
        (Some(m), _, _) => (
            BackendConfig {
                backend: Some(Backend::MistralRs),
                batch_size: Some(1),
                ..found
            },
            format!(
                "mistral.rs ({}) : dossier SafeTensors chargé tel quel, GPU selon son build",
                m.binary.display()
            ),
        ),
        (None, Some(_), Some(gpu)) => (
            BackendConfig {
                backend: Some(Backend::LlamaServer),
                gpu_layers: Some(99),
                batch_size: Some(1),
                gguf_dtype: Some(WeightType::F16),
                ..found
            },
            format!(
                "llama-server voit {} ({}) : tous les calques sur le GPU",
                gpu.name,
                gpu.api()
            ),
        ),
        (None, Some(l), None) => (
            BackendConfig {
                backend: Some(Backend::Native),
                ..found
            },
            format!(
                "llama-server ({}) ne voit aucun GPU : moteur natif (CPU)",
                l.server.display()
            ),
        ),
        (None, None, _) => (
            BackendConfig {
                backend: Some(Backend::Native),
                ..found
            },
            "ni mistral.rs ni llama-server : moteur natif (CPU)".into(),
        ),
    };
    BackendReport {
        os: std::env::consts::OS,
        mistralrs,
        llama,
        hipfire,
        recommended,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn devices_are_read_from_the_list_devices_output() {
        let out = "ggml_vulkan: Found 1 Vulkan devices:\nAvailable devices:\n  \
                   Vulkan0: AMD Radeon RX 7600 XT (16368 MiB, 15557 MiB free)\n  CPU: Intel (12 cores)\n";
        let devices = parse_devices(out);
        assert_eq!(
            devices,
            [GpuDevice {
                id: "Vulkan0".into(),
                name: "AMD Radeon RX 7600 XT".into(),
                vram_mib: Some(16368)
            }]
        );
        assert_eq!(devices[0].api(), "Vulkan");
        assert!(parse_devices("Available devices:\n").is_empty());
    }

    #[test]
    fn the_recommendation_follows_what_the_machine_has() {
        // No hint, and the PATH is not controllable here: the report must at least be
        // coherent whatever the machine has.
        let report = detect_hardware_and_backends(&BackendConfig::default());
        let backend = report.recommended.backend.unwrap();
        let gpu = report.llama.as_ref().and_then(|l| l.devices.first());
        match (&report.mistralrs, gpu) {
            (Some(m), _) => {
                assert_eq!(backend, Backend::MistralRs);
                assert_eq!(report.recommended.mistralrs_bin.as_ref(), Some(&m.binary));
            }
            (None, Some(_)) => assert_eq!(backend, Backend::LlamaServer),
            (None, None) => assert_eq!(backend, Backend::Native),
        }
        assert_ne!(backend, Backend::Hipfire, "hipfire is opt-in");
    }

    #[test]
    fn mistralrs_comes_first() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("mistralrs-server");
        std::fs::write(&bin, b"").unwrap();
        let user = BackendConfig {
            mistralrs_bin: Some(bin.clone()),
            ..Default::default()
        };
        let report = detect_hardware_and_backends(&user);
        assert_eq!(report.recommended.backend, Some(Backend::MistralRs));
        assert_eq!(report.recommended.mistralrs_bin, Some(bin));
        assert_eq!(report.recommended.batch_size, Some(1));
        assert!(report.reason.contains("mistral.rs"), "{}", report.reason);
    }
}
