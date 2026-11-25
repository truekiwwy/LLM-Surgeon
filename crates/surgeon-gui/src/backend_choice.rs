//! The app's side of the automatic backend choice: when the app may pick, the hipfire daemon
//! it keeps running, the configuration it writes and how mistral.rs gets installed. The
//! engine itself is the engine's recommendation (`backend::detect::recommend`): hipfire on an
//! AMD GPU, mistral.rs on NVIDIA and Apple Silicon even before they are ready, never the CPU
//! while a GPU is there. Pure: the GPUs and what answers come in, the choice comes out.

use std::path::PathBuf;

use weight_surgeon::backend::config::BackendConfig;
use weight_surgeon::backend::hipfire::DEFAULT_PORT;
use weight_surgeon::bench::score::Backend;

/// Who made a GPU, read from its name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Vendor {
    Nvidia,
    Amd,
    /// Apple Silicon, Intel, or nothing known: mistral.rs either way.
    Other,
}

pub fn vendor(gpu: &str) -> Vendor {
    let name = gpu.to_ascii_lowercase();
    let has = |words: &[&str]| words.iter().any(|w| name.contains(w));
    match () {
        _ if has(&["nvidia", "geforce", "quadro", "tesla", "rtx "]) => Vendor::Nvidia,
        _ if has(&["amd", "radeon", "instinct"]) => Vendor::Amd,
        _ => Vendor::Other,
    }
}

/// The GPU that decides: an NVIDIA one first, then an AMD one (an integrated Intel GPU next
/// to them does not count).
pub fn main_vendor<'a>(gpus: impl IntoIterator<Item = &'a str>) -> Vendor {
    let vendors: Vec<Vendor> = gpus.into_iter().map(vendor).collect();
    [Vendor::Nvidia, Vendor::Amd]
        .into_iter()
        .find(|v| vendors.contains(v))
        .unwrap_or(Vendor::Other)
}

/// Whether the app picks the backend: always while the choice is automatic, and over a file
/// that names none or a backend the app no longer offers (llama.cpp). A Rust engine saved by
/// hand stays.
pub fn automatic(saved: Option<Backend>, auto: bool) -> bool {
    auto || !matches!(
        saved,
        Some(Backend::MistralRs | Backend::Hipfire | Backend::Native)
    )
}

/// Whether the app should have a hipfire daemon running: the hardware or the saved backend
/// calls for it, and the native engine was not saved by hand (`auto`: the saved backend is
/// only the app's own latest choice).
pub fn wants_hipfire(target: Backend, saved: Option<Backend>, auto: bool) -> bool {
    let pinned_native = saved == Some(Backend::Native) && !auto;
    !pinned_native && (target == Backend::Hipfire || saved == Some(Backend::Hipfire))
}

/// The daemon address an AMD machine is given when none is set (hipfire's own default, also
/// reached through WSL2 or Docker).
pub fn hipfire_url() -> String {
    format!("http://127.0.0.1:{DEFAULT_PORT}")
}

/// The configuration of an automatic choice: `backend`, the hipfire daemon an AMD machine
/// waits for, the mistral.rs binary found. The other keys of `saved` are kept.
pub fn config(
    saved: &BackendConfig,
    backend: Backend,
    vendor: Vendor,
    mistral: Option<PathBuf>,
) -> BackendConfig {
    let hipfire_url = match (vendor, &saved.hipfire_url) {
        (_, Some(url)) => Some(url.clone()),
        (Vendor::Amd, None) => Some(hipfire_url()),
        _ => None,
    };
    BackendConfig {
        backend: Some(backend),
        mistralrs_bin: mistral.or_else(|| saved.mistralrs_bin.clone()),
        hipfire_url,
        batch_size: saved.batch_size.or(Some(1)),
        ..saved.clone()
    }
}

/// hipfire's install guide (Linux with ROCm, Windows notes).
pub const HIPFIRE_DOCS: &str =
    "https://github.com/warpfront/hipfire/blob/main/docs/GETTING_STARTED.md";

/// How mistral.rs gets on this machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MistralInstall {
    /// Windows: the official CPU build, downloaded and verified.
    Download,
    /// Windows with an NVIDIA GPU: built with CUDA (no official Windows CUDA build), the CPU
    /// build meanwhile.
    CudaBuild,
    /// Elsewhere: the official archive for this system, installed by hand.
    Manual(&'static str),
}

pub fn mistral_install(os: &str, arch: &str, vendor: Vendor) -> MistralInstall {
    match (os, arch, vendor) {
        ("windows", _, Vendor::Nvidia) => MistralInstall::CudaBuild,
        ("windows", _, _) => MistralInstall::Download,
        ("macos", "aarch64", _) => {
            MistralInstall::Manual("mistralrs-metal-aarch64-apple-darwin.tar.gz")
        }
        ("linux", "aarch64", _) => {
            MistralInstall::Manual("mistralrs-cpu-aarch64-unknown-linux-gnu.tar.gz")
        }
        _ => MistralInstall::Manual("mistralrs-cpu-x86_64-unknown-linux-gnu.tar.gz"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_gpu_vendor_comes_from_the_names() {
        assert_eq!(vendor("AMD Radeon RX 7600 XT"), Vendor::Amd);
        assert_eq!(vendor("NVIDIA GeForce RTX 4090"), Vendor::Nvidia);
        assert_eq!(vendor("Apple M2 Pro"), Vendor::Other);
        let laptop = ["Intel(R) UHD Graphics", "AMD Radeon RX 6600M"];
        assert_eq!(main_vendor(laptop), Vendor::Amd, "the iGPU does not decide");
        assert_eq!(
            main_vendor(["AMD Radeon", "NVIDIA RTX A2000"]),
            Vendor::Nvidia
        );
        assert_eq!(main_vendor([]), Vendor::Other, "CPU only");
    }

    #[test]
    fn llama_cpp_and_an_empty_file_are_taken_over_a_rust_choice_by_hand_is_kept() {
        assert!(automatic(None, false), "first launch");
        assert!(
            automatic(Some(Backend::LlamaServer), false),
            "llama.cpp is not offered"
        );
        assert!(!automatic(Some(Backend::Native), false), "saved by hand");
        assert!(automatic(Some(Backend::Native), true), "still automatic");
    }

    #[test]
    fn the_daemon_is_wanted_on_amd_or_by_choice_unless_native_is_pinned() {
        let (amd, other) = (Backend::Hipfire, Backend::MistralRs);
        assert!(wants_hipfire(amd, None, true), "first launch on AMD");
        assert!(
            wants_hipfire(amd, Some(Backend::Native), true),
            "native only as the automatic fallback"
        );
        assert!(wants_hipfire(amd, Some(Backend::MistralRs), false));
        assert!(wants_hipfire(other, Some(Backend::Hipfire), false));
        assert!(!wants_hipfire(amd, Some(Backend::Native), false), "pinned");
        assert!(!wants_hipfire(other, Some(Backend::MistralRs), true));
    }

    #[test]
    fn an_automatic_configuration_keeps_the_user_keys_and_points_amd_at_its_daemon() {
        let saved = BackendConfig {
            gpu_layers: Some(12),
            ..Default::default()
        };
        let bin = PathBuf::from("C:/m/mistralrs.exe");
        let amd = config(&saved, Backend::MistralRs, Vendor::Amd, Some(bin.clone()));
        assert_eq!(amd.backend, Some(Backend::MistralRs));
        assert_eq!(amd.hipfire_url.as_deref(), Some("http://127.0.0.1:11435"));
        assert_eq!((amd.mistralrs_bin, amd.gpu_layers), (Some(bin), Some(12)));
        let cpu = config(&saved, Backend::Native, Vendor::Other, None);
        assert_eq!(cpu.hipfire_url, None);
        assert_eq!(
            mistral_install("windows", "x86_64", Vendor::Nvidia),
            MistralInstall::CudaBuild
        );
        assert_eq!(
            mistral_install("windows", "x86_64", Vendor::Amd),
            MistralInstall::Download
        );
        assert!(
            matches!(mistral_install("macos", "aarch64", Vendor::Other), MistralInstall::Manual(a) if a.contains("metal"))
        );
    }
}
