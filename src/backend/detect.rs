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
    /// hipfire targets Linux with ROCm.
    pub supported_os: bool,
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendReport {
    pub os: &'static str,
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
    crate::eval::server::capture(server, "--list-devices")
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

/// Probes the machine (llama.cpp and its devices, hipfire) and recommends a backend:
/// `llama-server` on the GPU when one is usable, the native CPU engine otherwise. hipfire is
/// never recommended: it is an opt-in AMD engine.
pub fn detect_hardware_and_backends(user: &BackendConfig) -> BackendReport {
    let llama = detect_llama(user);
    let hipfire = hipfire::detect(user);
    let gpu = llama.as_ref().and_then(|l| l.devices.first());
    let (recommended, reason) = match (&llama, gpu) {
        (Some(l), Some(gpu)) => (
            BackendConfig {
                backend: Some(Backend::LlamaServer),
                gpu_layers: Some(99),
                batch_size: Some(1),
                llama_cpp_dir: Some(l.dir.clone()),
                hipfire_bin: hipfire.binary.clone(),
                gguf_dtype: Some(WeightType::F16),
            },
            format!(
                "llama-server voit {} ({}) : tous les calques sur le GPU",
                gpu.name,
                gpu.api()
            ),
        ),
        (Some(l), None) => (
            BackendConfig::default(),
            format!(
                "llama-server ({}) ne voit aucun GPU : moteur natif (CPU)",
                l.server.display()
            ),
        ),
        _ => (
            BackendConfig::default(),
            "llama-server introuvable : moteur natif (CPU)".into(),
        ),
    };
    let recommended = BackendConfig {
        backend: Some(recommended.backend.unwrap_or(Backend::Native)),
        ..recommended
    };
    BackendReport {
        os: std::env::consts::OS,
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
    fn a_missing_llama_means_native() {
        // No hint, and an empty PATH / LLAMA_CPP_DIR is not controllable here: the report
        // must at least be coherent whatever the machine has.
        let report = detect_hardware_and_backends(&BackendConfig::default());
        let backend = report.recommended.backend.unwrap();
        match report.llama.as_ref().and_then(|l| l.devices.first()) {
            Some(_) => assert_eq!(backend, Backend::LlamaServer),
            None => assert_eq!(backend, Backend::Native),
        }
        assert_ne!(backend, Backend::Hipfire, "hipfire is opt-in");
    }
}
