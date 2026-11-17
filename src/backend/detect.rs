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
    /// The engine the hardware calls for: `hipfire` (AMD) or `mistralrs`.
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
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    let output = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "(Get-CimInstance Win32_VideoController).Name",
        ])
        .stdin(std::process::Stdio::null())
        .creation_flags(CREATE_NO_WINDOW)
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

/// The configuration to run with, and why: the target when it is ready, else mistral.rs when
/// installed (an AMD machine whose daemon is not up yet), else the native engine.
pub fn recommend(
    vendor: Vendor,
    mistralrs: Option<&Path>,
    hipfire_up: bool,
    found: BackendConfig,
) -> (BackendConfig, String) {
    let amd = vendor == Vendor::Amd;
    let waiting = if amd {
        " ; hipfire, le moteur des GPU AMD, prendra le relais dès que son démon répondra"
    } else {
        ""
    };
    let (backend, reason) = match (amd && hipfire_up, mistralrs) {
        (true, _) => (
            Backend::Hipfire,
            "GPU AMD : démon hipfire en écoute".to_owned(),
        ),
        (false, Some(bin)) => (
            Backend::MistralRs,
            format!(
                "mistral.rs ({}) : dossier SafeTensors chargé tel quel, GPU selon son build{waiting}",
                bin.display()
            ),
        ),
        (false, None) => (
            Backend::Native,
            format!("mistral.rs introuvable : moteur natif (CPU, Rust){waiting}"),
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
    let vendor = vendor(gpus.iter().map(String::as_str));
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
        target: match vendor {
            Vendor::Amd => Backend::Hipfire,
            _ => Backend::MistralRs,
        },
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
    fn the_recommendation_is_a_rust_engine() {
        let bin = Path::new("mistralrs.exe");
        let pick = |vendor, mistral, up| recommend(vendor, mistral, up, Default::default());
        assert_eq!(
            pick(Vendor::Amd, None, true).0.backend,
            Some(Backend::Hipfire)
        );
        let (waiting, why) = pick(Vendor::Amd, Some(bin), false);
        assert_eq!(waiting.backend, Some(Backend::MistralRs));
        assert!(why.contains("hipfire"), "{why}");
        assert_eq!(
            pick(Vendor::Nvidia, Some(bin), true).0.backend,
            Some(Backend::MistralRs),
            "hipfire is for AMD GPUs only"
        );
        for vendor in [Vendor::Nvidia, Vendor::Apple, Vendor::Cpu, Vendor::Amd] {
            let (config, why) = pick(vendor, None, false);
            assert_eq!(config.backend, Some(Backend::Native));
            assert_eq!(config.batch_size, Some(1));
            assert!(!why.contains("llama"), "{why}");
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
