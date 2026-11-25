//! What this machine can run, and the backend to recommend: the first-start probe of the CLI
//! (`surgeon backend detect`), of the MCP tool `surgeon_backends` and of the GUI.
//!
//! Rust engines only: hipfire for an AMD GPU, mistral.rs for NVIDIA, Apple Silicon and plain
//! CPUs, and the built-in native engine until the target is ready. llama.cpp is still a
//! backend one can name by hand, never one the probe looks for or recommends.

use std::path::{Path, PathBuf};

use serde::Serialize;

use super::config::BackendConfig;
use super::hipfire;
use crate::bench::score::Backend;
use crate::hardware::{HardwareProbe, SystemProbe};

/// A device listed by `llama-server --list-devices` (report labels of a llama.cpp run).
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

/// Who made the GPU that decides the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Vendor {
    Nvidia,
    Amd,
    Apple,
    /// No GPU an engine uses (Intel, none): the CPU.
    Cpu,
}

#[derive(Debug, Clone, Serialize)]
pub struct HipfireDetection {
    pub binary: Option<PathBuf>,
    /// The daemon answers.
    pub up: bool,
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
    /// The GPUs seen, by name.
    pub gpus: Vec<String>,
    pub vendor: Vendor,
    /// The engine the hardware calls for ([`gpu_engine`]).
    pub target: Backend,
    pub mistralrs: Option<MistralRsDetection>,
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

/// The vendor that decides: an NVIDIA GPU first, then an AMD one, then Apple Silicon (an
/// integrated Intel GPU next to them does not count).
pub fn vendor<'a>(gpus: impl IntoIterator<Item = &'a str>) -> Vendor {
    let found: Vec<Vendor> = gpus
        .into_iter()
        .map(|gpu| {
            let name = gpu.to_ascii_lowercase();
            let has = |words: &[&str]| words.iter().any(|w| name.contains(w));
            match () {
                _ if has(&["nvidia", "geforce", "quadro", "tesla", "rtx "]) => Vendor::Nvidia,
                _ if has(&["amd", "radeon", "instinct"]) => Vendor::Amd,
                _ if has(&["apple"]) => Vendor::Apple,
                _ => Vendor::Cpu,
            }
        })
        .collect();
    [Vendor::Nvidia, Vendor::Amd, Vendor::Apple]
        .into_iter()
        .find(|v| found.contains(v))
        .unwrap_or(Vendor::Cpu)
}

/// The GPU names: NVML, sysfs and Apple Silicon through the hardware probe; on Windows the
/// display adapters, since the probe sees no AMD GPU there.
fn gpu_names() -> Vec<String> {
    let mut names: Vec<String> = SystemProbe::detect()
        .gpus()
        .unwrap_or_default()
        .into_iter()
        .map(|g| g.name)
        .collect();
    #[cfg(windows)]
    names.extend(windows_adapters());
    names.dedup();
    names
}

#[cfg(windows)]
fn windows_adapters() -> Vec<String> {
    let output = crate::process::command("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance Win32_VideoController).Name",
        ])
        .stdin(std::process::Stdio::null())
        .output();
    output
        .map(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// The engine a GPU calls for: hipfire on AMD, mistral.rs on NVIDIA and Apple Silicon, the
/// native CPU engine only when there is no GPU. What `--backend auto` (the default) runs.
pub fn gpu_engine(vendor: Vendor) -> Backend {
    match vendor {
        Vendor::Amd => Backend::Hipfire,
        Vendor::Nvidia | Vendor::Apple => Backend::MistralRs,
        Vendor::Cpu => Backend::Native,
    }
}

/// Sets the GPU [`machine_vendor`] reports: `none`, `amd`, `nvidia` or `apple`. For a machine
/// whose GPU is not detected, and for the tests (`.cargo/config.toml`).
pub const GPU_ENV: &str = "WEIGHT_SURGEON_GPU";

fn forced_vendor() -> Option<Vendor> {
    let forced = std::env::var(GPU_ENV).unwrap_or_default();
    match forced.trim().to_ascii_lowercase().as_str() {
        "none" | "cpu" => Some(Vendor::Cpu),
        "amd" => Some(Vendor::Amd),
        "nvidia" => Some(Vendor::Nvidia),
        "apple" => Some(Vendor::Apple),
        _ => None,
    }
}

/// The vendor of this machine's GPU (see [`vendor`]), or the one [`GPU_ENV`] names.
pub fn machine_vendor() -> Vendor {
    forced_vendor().unwrap_or_else(|| vendor(gpu_names().iter().map(String::as_str)))
}

/// The configuration to run with, and why: the GPU's engine whether or not it is ready (a
/// GPU machine never falls back to the CPU: the bench fails and says what is missing), and
/// on a machine without a GPU mistral.rs when available, else the native engine.
pub fn recommend(
    vendor: Vendor,
    mistralrs: Option<&Path>,
    hipfire_up: bool,
    found: BackendConfig,
) -> (BackendConfig, String) {
    let missing = "pas encore prêt : la mesure échouera avec son diagnostic (surgeon doctor) \
                   plutôt que de passer sur le CPU";
    use super::mistralrs::{EMBEDDED, EMBEDDED_GPU};
    let mistral = match mistralrs {
        _ if EMBEDDED_GPU => "mistral.rs intégré à surgeon (GPU)".to_owned(),
        Some(bin) => format!("mistral.rs ({})", bin.display()),
        None if EMBEDDED => "mistral.rs intégré à surgeon (CPU)".to_owned(),
        None => format!("mistral.rs introuvable, {missing}"),
    };
    let (backend, reason) = match (gpu_engine(vendor), hipfire_up) {
        (Backend::Hipfire, true) => (Backend::Hipfire, "GPU AMD : démon hipfire en écoute".into()),
        (Backend::Hipfire, false) => (
            Backend::Hipfire,
            format!("GPU AMD : démon hipfire {missing}"),
        ),
        (Backend::MistralRs, _) => (Backend::MistralRs, format!("GPU {vendor:?} : {mistral}")),
        _ if mistralrs.is_some() || EMBEDDED => {
            (Backend::MistralRs, format!("aucun GPU : {mistral}"))
        }
        _ => (
            Backend::Native,
            "aucun GPU : moteur natif (CPU, Rust)".into(),
        ),
    };
    let config = BackendConfig {
        backend: Some(backend),
        batch_size: Some(1),
        ..found
    };
    (config, reason)
}

/// Probes the machine (GPUs, mistral.rs, the hipfire daemon) and recommends a Rust engine.
/// The engines found are recorded either way, so that an explicit `--backend` finds them.
pub fn detect_hardware_and_backends(user: &BackendConfig) -> BackendReport {
    let gpus = gpu_names();
    let vendor = forced_vendor().unwrap_or_else(|| vendor(gpus.iter().map(String::as_str)));
    // An AMD machine reaches its daemon at hipfire's default when no address is set.
    let mut probed = user.clone();
    if vendor == Vendor::Amd && probed.hipfire_url.is_none() {
        probed.hipfire_url = Some(format!("http://127.0.0.1:{}", hipfire::DEFAULT_PORT));
    }
    let mistralrs = super::mistralrs::locate(user).map(|binary| MistralRsDetection { binary });
    let hipfire = hipfire::detect(&probed);
    let found = BackendConfig {
        mistralrs_bin: mistralrs.as_ref().map(|m| m.binary.clone()),
        hipfire_bin: hipfire.binary.clone(),
        hipfire_url: probed.hipfire_url.clone(),
        ..Default::default()
    };
    let (recommended, reason) = recommend(
        vendor,
        mistralrs.as_ref().map(|m| m.binary.as_path()),
        hipfire.up,
        found,
    );
    BackendReport {
        os: std::env::consts::OS,
        gpus,
        vendor,
        target: gpu_engine(vendor),
        mistralrs,
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
    fn the_vendor_comes_from_the_gpu_names() {
        assert_eq!(vendor(["AMD Radeon RX 7600 XT"]), Vendor::Amd);
        assert_eq!(
            vendor(["Intel(R) UHD Graphics", "NVIDIA GeForce RTX 4070"]),
            Vendor::Nvidia
        );
        assert_eq!(vendor(["Apple Silicon (mémoire unifiée)"]), Vendor::Apple);
        assert_eq!(vendor(["Intel(R) UHD Graphics"]), Vendor::Cpu);
        assert_eq!(vendor([]), Vendor::Cpu);
    }

    #[test]
    fn auto_is_the_gpu_engine_and_the_cpu_only_without_a_gpu() {
        assert_eq!(gpu_engine(Vendor::Amd), Backend::Hipfire);
        assert_eq!(gpu_engine(Vendor::Nvidia), Backend::MistralRs);
        assert_eq!(gpu_engine(Vendor::Apple), Backend::MistralRs);
        assert_eq!(gpu_engine(Vendor::Cpu), Backend::Native);
        // The tests run under WEIGHT_SURGEON_GPU=none (.cargo/config.toml).
        assert_eq!(machine_vendor(), Vendor::Cpu);
        use super::super::resolve_auto;
        assert_eq!(resolve_auto(Backend::Auto, false), Backend::Native);
        assert_eq!(
            resolve_auto(Backend::Auto, true),
            Backend::Llama,
            "a GGUF on the CPU"
        );
        assert_eq!(resolve_auto(Backend::Hipfire, true), Backend::Hipfire);
    }

    #[test]
    fn a_gpu_is_never_recommended_the_cpu() {
        let bin = Path::new("mistralrs.exe");
        let pick = |vendor, mistral, up| recommend(vendor, mistral, up, Default::default());
        assert_eq!(
            pick(Vendor::Amd, None, true).0.backend,
            Some(Backend::Hipfire)
        );
        // A daemon that is not up yet: still hipfire, the reason says what is missing.
        let (waiting, why) = pick(Vendor::Amd, Some(bin), false);
        assert_eq!(waiting.backend, Some(Backend::Hipfire));
        assert!(why.contains("surgeon doctor"), "{why}");
        assert_eq!(
            pick(Vendor::Nvidia, Some(bin), true).0.backend,
            Some(Backend::MistralRs),
            "hipfire is for AMD GPUs only"
        );
        for vendor in [Vendor::Nvidia, Vendor::Apple, Vendor::Amd] {
            let (config, why) = pick(vendor, None, false);
            assert_ne!(config.backend, Some(Backend::Native), "{vendor:?}");
            assert_eq!(config.batch_size, Some(1));
            assert!(!why.contains("llama"), "{why}");
        }
        let cpu = |mistral| pick(Vendor::Cpu, mistral, false).0.backend;
        assert_eq!(cpu(Some(bin)), Some(Backend::MistralRs));
        if !crate::backend::mistralrs::EMBEDDED {
            assert_eq!(cpu(None), Some(Backend::Native), "no GPU: the CPU");
        }
    }

    #[test]
    fn mistralrs_found_is_recorded_and_recommended_off_amd() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("mistralrs-server");
        std::fs::write(&bin, b"").unwrap();
        let user = BackendConfig {
            mistralrs_bin: Some(bin.clone()),
            ..Default::default()
        };
        let report = detect_hardware_and_backends(&user);
        assert_eq!(report.recommended.mistralrs_bin, Some(bin));
        let backend = report.recommended.backend.unwrap();
        match report.hipfire.up && report.vendor == Vendor::Amd {
            true => assert_eq!(backend, Backend::Hipfire),
            false => assert_eq!(backend, Backend::MistralRs),
        }
        assert!(report.recommended.llama_cpp_dir.is_none());
    }
}
