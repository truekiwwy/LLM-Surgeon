use serde::Serialize;

use super::error::{HardwareError, Result};

#[derive(Debug, Clone, Serialize)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    AppleSilicon,
}

#[derive(Debug, Clone, Serialize)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    pub total_vram_bytes: u64,
    pub free_vram_bytes: u64,
    pub unified_memory: bool,
}

pub trait HardwareProbe {
    fn system_ram(&self) -> Result<MemoryInfo>;
    /// Empty list when no GPU is found: that is not an error.
    fn gpus(&self) -> Result<Vec<GpuInfo>>;
    /// Non-fatal detection notes (e.g. NVML missing), copied into the budget report.
    fn warnings(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Real probe. Discrete GPUs are detected once at construction (NVML init is not
/// free); RAM, and the Apple unified pool that mirrors it, are read live.
#[derive(Debug, Clone, Default)]
pub struct SystemProbe {
    discrete_gpus: Vec<GpuInfo>,
    warnings: Vec<String>,
}

impl SystemProbe {
    pub fn detect() -> Self {
        #[cfg_attr(target_os = "macos", allow(unused_mut))] // no discrete GPU probing on macOS
        let mut probe = Self::default();
        #[cfg(not(target_os = "macos"))]
        probe.detect_nvidia();
        #[cfg(target_os = "linux")]
        probe
            .discrete_gpus
            .extend(amd_gpus_in(std::path::Path::new("/sys/class/drm")));
        probe
    }

    #[cfg(not(target_os = "macos"))]
    fn detect_nvidia(&mut self) {
        let nvml = match nvml_wrapper::Nvml::init() {
            Ok(nvml) => nvml,
            Err(e) => {
                self.warnings
                    .push(format!("NVML indisponible, GPU NVIDIA ignorés : {e}"));
                return;
            }
        };
        let count = match nvml.device_count() {
            Ok(n) => n,
            Err(e) => {
                self.warnings
                    .push(format!("NVML : impossible de compter les GPU : {e}"));
                return;
            }
        };
        for index in 0..count {
            let read = nvml
                .device_by_index(index)
                .and_then(|d| Ok((d.name()?, d.memory_info()?)));
            match read {
                Ok((name, mem)) => self.discrete_gpus.push(GpuInfo {
                    vendor: GpuVendor::Nvidia,
                    name,
                    total_vram_bytes: mem.total,
                    free_vram_bytes: mem.free,
                    unified_memory: false,
                }),
                Err(e) => self
                    .warnings
                    .push(format!("NVML : GPU #{index} ignoré : {e}")),
            }
        }
    }
}

impl HardwareProbe for SystemProbe {
    fn system_ram(&self) -> Result<MemoryInfo> {
        let mut sys =
            sysinfo::System::new().map_err(|e| HardwareError::RamUnavailable(e.to_string()))?;
        sys.refresh_memory();
        // sysinfo >= 0.30 reports bytes.
        Ok(MemoryInfo {
            total_bytes: sys.total_memory(),
            available_bytes: sys.available_memory(),
        })
    }

    fn gpus(&self) -> Result<Vec<GpuInfo>> {
        #[cfg_attr(
            not(all(target_os = "macos", target_arch = "aarch64")),
            allow(unused_mut)
        )] // only mutated on Apple Silicon
        let mut gpus = self.discrete_gpus.clone();
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            let ram = self.system_ram()?;
            gpus.push(GpuInfo {
                vendor: GpuVendor::AppleSilicon,
                name: "Apple Silicon (mémoire unifiée)".to_owned(),
                total_vram_bytes: ram.total_bytes,
                free_vram_bytes: ram.available_bytes,
                unified_memory: true,
            });
        }
        Ok(gpus)
    }

    fn warnings(&self) -> Vec<String> {
        self.warnings.clone()
    }
}

/// AMD GPUs from a sysfs DRM root (`/sys/class/drm`). Any unreadable card is skipped.
/// Compiled for tests on every OS so the parsing is checked off Linux too.
#[cfg(any(target_os = "linux", test))]
fn amd_gpus_in(drm_root: &std::path::Path) -> Vec<GpuInfo> {
    let Ok(entries) = std::fs::read_dir(drm_root) else {
        return Vec::new();
    };
    let mut gpus: Vec<(String, GpuInfo)> = entries
        .flatten()
        .filter_map(|entry| {
            let card = entry.file_name().into_string().ok()?;
            // `card0` only; `card0-DP-1` style entries are connectors.
            card.strip_prefix("card")?.parse::<u32>().ok()?;
            let device = entry.path().join("device");
            let read = |file: &str| std::fs::read_to_string(device.join(file)).ok();
            if read("vendor")?.trim() != "0x1002" {
                return None;
            }
            let total: u64 = read("mem_info_vram_total")?.trim().parse().ok()?;
            let used: u64 = read("mem_info_vram_used")?.trim().parse().ok()?;
            let name = read("product_name")
                .map(|n| n.trim().to_owned())
                .filter(|n| !n.is_empty())
                .unwrap_or_else(|| format!("AMD GPU ({card})"));
            Some((
                card,
                GpuInfo {
                    vendor: GpuVendor::Amd,
                    name,
                    total_vram_bytes: total,
                    free_vram_bytes: total.saturating_sub(used),
                    unified_memory: false,
                },
            ))
        })
        .collect();
    gpus.sort_by(|a, b| a.0.cmp(&b.0));
    gpus.into_iter().map(|(_, gpu)| gpu).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

    fn card(root: &Path, name: &str, files: &[(&str, &str)]) {
        let dev = root.join(name).join("device");
        fs::create_dir_all(&dev).unwrap();
        for (f, v) in files {
            fs::write(dev.join(f), v).unwrap();
        }
    }

    #[test]
    fn amd_sysfs_parsing_keeps_only_readable_amd_cards() {
        let root = tempfile::tempdir().unwrap();
        let r = root.path();
        card(
            r,
            "card0",
            &[
                ("vendor", "0x1002\n"),
                ("mem_info_vram_total", "17179869184\n"),
                ("mem_info_vram_used", "1073741824\n"),
            ],
        );
        card(r, "card1", &[("vendor", "0x10de\n")]); // NVIDIA: ignored
        card(r, "card2", &[("vendor", "0x1002\n")]); // unreadable VRAM: ignored
        card(r, "card0-DP-1", &[("vendor", "0x1002\n")]); // connector: ignored

        let gpus = amd_gpus_in(r);
        assert_eq!(gpus.len(), 1);
        assert_eq!(gpus[0].vendor, GpuVendor::Amd);
        assert_eq!(gpus[0].total_vram_bytes, 16 << 30);
        assert_eq!(gpus[0].free_vram_bytes, 15 << 30);
        assert!(amd_gpus_in(&r.join("missing")).is_empty());
    }

    #[test]
    fn system_probe_never_fails_without_gpu() {
        let probe = SystemProbe::detect();
        assert!(probe.system_ram().unwrap().total_bytes > 0);
        assert!(probe.gpus().is_ok());
    }
}
