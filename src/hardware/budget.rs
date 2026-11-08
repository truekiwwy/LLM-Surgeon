use std::path::Path;

use serde::Serialize;

use super::error::{HardwareError, Result};
use super::model_size::{compute_model_size, ModelSize};
use super::probe::{GpuInfo, HardwareProbe, SystemProbe};

const GIB: f64 = (1u64 << 30) as f64;

#[derive(Debug, Clone, Serialize)]
pub struct HardwareBudget {
    pub model_bytes: u64,
    pub required_ram_bytes: u64,
    pub available_ram_bytes: u64,
    pub required_vram_bytes: Option<u64>,
    pub best_gpu: Option<GpuInfo>,
    pub verdict: Verdict,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Verdict {
    FitsInVram,
    FitsInRamOnly,
    Insufficient { reason: String },
}

#[derive(Debug, Clone)]
pub struct BudgetPolicy {
    pub ram_overhead_factor: f64,
    pub vram_overhead_factor: f64,
    pub ram_safety_margin_bytes: u64,
    pub allow_cpu_fallback: bool,
}

impl Default for BudgetPolicy {
    fn default() -> Self {
        Self {
            ram_overhead_factor: 1.15,
            vram_overhead_factor: 1.20,
            ram_safety_margin_bytes: 2 << 30,
            allow_cpu_fallback: true,
        }
    }
}

/// Real hardware, default policy. `Err(InsufficientResources)` when the model cannot load.
pub fn can_load_model(model_path: &Path) -> Result<HardwareBudget> {
    can_load_model_with(model_path, &SystemProbe::detect(), &BudgetPolicy::default())
}

pub fn can_load_model_with(
    model_path: &Path,
    probe: &dyn HardwareProbe,
    policy: &BudgetPolicy,
) -> Result<HardwareBudget> {
    let budget = evaluate_budget(model_path, probe, policy)?;
    match &budget.verdict {
        Verdict::Insufficient { reason } => Err(HardwareError::InsufficientResources {
            reason: reason.clone(),
        }),
        _ => Ok(budget),
    }
}

/// Report only: an `Insufficient` verdict is `Ok`. `Err` means the model or the
/// hardware could not be read at all.
pub fn evaluate_budget(
    model_path: &Path,
    probe: &dyn HardwareProbe,
    policy: &BudgetPolicy,
) -> Result<HardwareBudget> {
    evaluate_with_size(&compute_model_size(model_path)?, probe, policy)
}

pub(crate) fn evaluate_with_size(
    size: &ModelSize,
    probe: &dyn HardwareProbe,
    policy: &BudgetPolicy,
) -> Result<HardwareBudget> {
    let ram = probe.system_ram()?;
    let mut warnings = probe.warnings();
    let model = size.total_bytes;
    let required_ram =
        scale(model, policy.ram_overhead_factor).saturating_add(policy.ram_safety_margin_bytes);
    let required_vram = scale(model, policy.vram_overhead_factor);
    let best_gpu = probe.gpus()?.into_iter().max_by_key(|g| g.free_vram_bytes);

    let verdict = match &best_gpu {
        // One shared pool: check it once against the stricter requirement, never
        // add RAM and VRAM together, and no CPU fallback since it is the same memory.
        Some(gpu) if gpu.unified_memory => {
            let need = required_ram.max(required_vram);
            let free = gpu.free_vram_bytes.min(ram.available_bytes);
            if free >= need {
                Verdict::FitsInVram
            } else {
                insufficient(format!(
                    "mémoire unifiée : besoin de {}, disponible {}",
                    gib(need),
                    gib(free)
                ))
            }
        }
        Some(gpu) if gpu.free_vram_bytes >= required_vram => Verdict::FitsInVram,
        _ => {
            let gpu_part = match &best_gpu {
                Some(gpu) => format!(
                    "VRAM : besoin de {}, {} libres sur {}",
                    gib(required_vram),
                    gib(gpu.free_vram_bytes),
                    gpu.name
                ),
                None => "aucun GPU détecté".to_owned(),
            };
            if !policy.allow_cpu_fallback {
                insufficient(format!("{gpu_part} ; repli CPU désactivé"))
            } else if ram.available_bytes >= required_ram {
                warnings.push(format!("{gpu_part} ; chargement en RAM (CPU), plus lent"));
                Verdict::FitsInRamOnly
            } else {
                insufficient(format!(
                    "{gpu_part} ; RAM : besoin de {} (marge incluse), disponible {}",
                    gib(required_ram),
                    gib(ram.available_bytes)
                ))
            }
        }
    };

    Ok(HardwareBudget {
        model_bytes: model,
        required_ram_bytes: required_ram,
        available_ram_bytes: ram.available_bytes,
        required_vram_bytes: best_gpu.as_ref().map(|_| required_vram),
        best_gpu,
        verdict,
        warnings,
    })
}

fn insufficient(reason: String) -> Verdict {
    Verdict::Insufficient { reason }
}

/// `bytes * factor`, rounded up. A NaN, infinite or < 1.0 factor counts as 1.0 so a
/// bad policy can never shrink the requirement below the model size. `as` from f64
/// saturates, so `u64::MAX * 1.2` stays `u64::MAX`.
fn scale(bytes: u64, factor: f64) -> u64 {
    let factor = if factor.is_finite() {
        factor.max(1.0)
    } else {
        1.0
    };
    (bytes as f64 * factor).ceil() as u64
}

fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / GIB)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::probe::{GpuVendor, MemoryInfo};

    const G: u64 = 1 << 30;

    struct MockProbe {
        ram_available: u64,
        gpus: Vec<GpuInfo>,
    }

    impl HardwareProbe for MockProbe {
        fn system_ram(&self) -> Result<MemoryInfo> {
            Ok(MemoryInfo {
                total_bytes: self.ram_available,
                available_bytes: self.ram_available,
            })
        }
        fn gpus(&self) -> Result<Vec<GpuInfo>> {
            Ok(self.gpus.clone())
        }
    }

    fn gpu(free: u64) -> GpuInfo {
        GpuInfo {
            vendor: GpuVendor::Nvidia,
            name: "mock".to_owned(),
            total_vram_bytes: free,
            free_vram_bytes: free,
            unified_memory: false,
        }
    }

    fn model(bytes: u64) -> ModelSize {
        ModelSize {
            total_bytes: bytes,
            shard_files: Vec::new(),
        }
    }

    fn eval(model_bytes: u64, probe: &MockProbe, policy: &BudgetPolicy) -> HardwareBudget {
        evaluate_with_size(&model(model_bytes), probe, policy).unwrap()
    }

    #[test]
    fn fits_in_vram_picks_gpu_with_most_free_memory() {
        let probe = MockProbe {
            ram_available: 16 * G,
            gpus: vec![gpu(8 * G), gpu(24 * G)],
        };
        let b = eval(10 * G, &probe, &BudgetPolicy::default());
        assert_eq!(b.verdict, Verdict::FitsInVram);
        assert_eq!(b.best_gpu.unwrap().free_vram_bytes, 24 * G);
        assert_eq!(b.required_vram_bytes, Some(12 * G));
    }

    #[test]
    fn small_gpu_big_ram_falls_back_with_warning() {
        let probe = MockProbe {
            ram_available: 64 * G,
            gpus: vec![gpu(8 * G)],
        };
        let b = eval(10 * G, &probe, &BudgetPolicy::default());
        assert_eq!(b.verdict, Verdict::FitsInRamOnly);
        assert_eq!(b.warnings.len(), 1);
    }

    #[test]
    fn insufficient_is_ok_in_report_and_err_in_guard() {
        let probe = MockProbe {
            ram_available: 8 * G,
            gpus: vec![gpu(8 * G)],
        };
        let b = eval(10 * G, &probe, &BudgetPolicy::default());
        assert!(matches!(b.verdict, Verdict::Insufficient { .. }));

        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("m.safetensors");
        // Sparse file: 10 GiB logical size without writing it.
        std::fs::File::create(&file)
            .unwrap()
            .set_len(10 * G)
            .unwrap();
        let policy = BudgetPolicy::default();
        assert!(evaluate_budget(&file, &probe, &policy).is_ok());
        let err = can_load_model_with(&file, &probe, &policy).unwrap_err();
        let HardwareError::InsufficientResources { reason } = err else {
            panic!("expected InsufficientResources, got {err:?}");
        };
        assert!(reason.contains("12.00 GiB"), "{reason}");
        assert!(reason.contains("8.00 GiB"), "{reason}");
    }

    #[test]
    fn unified_memory_is_counted_once() {
        let unified = GpuInfo {
            vendor: GpuVendor::AppleSilicon,
            name: "Apple".to_owned(),
            total_vram_bytes: 16 * G,
            free_vram_bytes: 16 * G,
            unified_memory: true,
        };
        let probe = MockProbe {
            ram_available: 16 * G,
            gpus: vec![unified],
        };
        let policy = BudgetPolicy::default();
        // 10 GiB: 13.5 GiB needed <= 16 free.
        assert_eq!(eval(10 * G, &probe, &policy).verdict, Verdict::FitsInVram);
        // 14 GiB: needs ~18.1 GiB; summing RAM + VRAM (32) would wrongly pass,
        // and the same pool must not be re-offered as a CPU fallback.
        assert!(matches!(
            eval(14 * G, &probe, &policy).verdict,
            Verdict::Insufficient { .. }
        ));
    }

    #[test]
    fn no_gpu_without_fallback_is_insufficient() {
        let probe = MockProbe {
            ram_available: 512 * G,
            gpus: Vec::new(),
        };
        let policy = BudgetPolicy {
            allow_cpu_fallback: false,
            ..BudgetPolicy::default()
        };
        let b = eval(G, &probe, &policy);
        assert!(matches!(b.verdict, Verdict::Insufficient { .. }));
        assert_eq!(b.required_vram_bytes, None);
    }

    #[test]
    fn edge_cases_zero_and_u64_max_do_not_overflow() {
        let roomy = MockProbe {
            ram_available: 4 * G,
            gpus: vec![gpu(G)],
        };
        let b = eval(0, &roomy, &BudgetPolicy::default());
        assert_eq!(b.verdict, Verdict::FitsInVram);
        assert_eq!(b.required_ram_bytes, 2 * G);

        let maxed = MockProbe {
            ram_available: u64::MAX,
            gpus: vec![gpu(u64::MAX)],
        };
        let policy = BudgetPolicy {
            ram_safety_margin_bytes: u64::MAX,
            ..BudgetPolicy::default()
        };
        let b = eval(u64::MAX, &maxed, &policy);
        assert_eq!(b.required_ram_bytes, u64::MAX);
        assert_eq!(b.required_vram_bytes, Some(u64::MAX));
        assert_eq!(b.verdict, Verdict::FitsInVram);
    }

    #[test]
    fn bad_overhead_factor_never_shrinks_requirement() {
        assert_eq!(scale(100, f64::NAN), 100);
        assert_eq!(scale(100, 0.5), 100);
        assert_eq!(scale(100, f64::INFINITY), 100);
        assert_eq!(scale(100, 1.2), 120);
    }
}
