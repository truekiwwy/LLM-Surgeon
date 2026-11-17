//! The inference backend a machine gets by default, Rust engines only (no llama.cpp):
//! mistral.rs for NVIDIA GPUs, Apple Silicon and plain CPUs (macOS, Windows, Linux), hipfire
//! for AMD GPUs. Pure: the GPUs and what answers come in, the choice comes out.
//!
//! What runs is the target once it is ready; until then mistral.rs when it is installed (an
//! AMD machine whose hipfire daemon is not up yet), else the built-in native engine (CPU,
//! Rust), so that the first launch always ends on a configuration that works.

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

/// The backend the hardware calls for.
pub fn target(vendor: Vendor) -> Backend {
    match vendor {
        Vendor::Amd => Backend::Hipfire,
        Vendor::Nvidia | Vendor::Other => Backend::MistralRs,
    }
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

/// What runs now: the target when it is ready, else mistral.rs when installed, else native.
pub fn effective(target: Backend, hipfire_up: bool, mistral_found: bool) -> Backend {
    match (target, hipfire_up, mistral_found) {
        (Backend::Hipfire, true, _) => Backend::Hipfire,
        (_, _, true) => Backend::MistralRs,
        _ => Backend::Native,
    }
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
    fn amd_goes_to_hipfire_and_everything_else_to_mistral_rs() {
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
        assert_eq!(target(Vendor::Amd), Backend::Hipfire);
        for v in [Vendor::Nvidia, Vendor::Other] {
            assert_eq!(target(v), Backend::MistralRs);
        }
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
    fn what_runs_falls_back_to_rust_engines_that_are_there() {
        assert_eq!(effective(Backend::Hipfire, true, false), Backend::Hipfire);
        assert_eq!(
            effective(Backend::Hipfire, false, true),
            Backend::MistralRs,
            "waiting for hipfire"
        );
        assert_eq!(
            effective(Backend::MistralRs, false, true),
            Backend::MistralRs
        );
        assert_eq!(
            effective(Backend::MistralRs, false, false),
            Backend::Native,
            "nothing installed yet"
        );
        assert_eq!(
            effective(Backend::MistralRs, true, false),
            Backend::Native,
            "hipfire is AMD's"
        );
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
