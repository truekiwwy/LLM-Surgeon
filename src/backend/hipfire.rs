//! hipfire (<https://github.com/warpfront/hipfire>): a Rust inference engine for AMD GPUs on
//! HIP / ROCm. Modelled as an optional external backend: located and checked here, and never
//! run when it cannot give the bench what it needs.
//!
//! What its README documents (read 2026-11-13): binary `hipfire`, subcommands `pull`, `run`,
//! `serve`, `bench`, `quantize`; a daemon on `127.0.0.1:11435` with the OpenAI-style
//! `/v1/chat/completions` (`model`, `messages`, `max_tokens`); Linux with ROCm 6, Windows and
//! WSL2 "untested"; its own quantized formats (`hipfire quantize`); Qwen2.5 not listed.
//! The bench sends prompts already rendered by its own template, with a seed and greedy
//! decoding. A chat endpoint would apply a second template and documents neither a seed nor
//! greedy decoding, so the numbers would not be comparable: no driver is wired until a
//! raw-completion endpoint exists to validate it on real hardware.

use std::path::PathBuf;

use super::config::BackendConfig;
use super::detect::HipfireDetection;
use crate::eval::locate_binary;

const ALTERNATIVE: &str = "--backend llama-server (Vulkan, llama.cpp) ou native";

fn binary(config: &BackendConfig) -> Option<PathBuf> {
    let given = config.hipfire_bin.clone();
    given
        .clone()
        .filter(|p| p.is_file())
        .or_else(|| locate_binary("hipfire", given.as_deref()).ok())
}

pub fn detect(config: &BackendConfig) -> HipfireDetection {
    let supported_os = cfg!(target_os = "linux");
    let binary = binary(config);
    let note = match (supported_os, &binary) {
        (false, _) => "hipfire cible Linux avec ROCm (Windows et WSL2 non testés par ses auteurs)",
        (true, None) => "hipfire absent du PATH et de hipfire_bin",
        (true, Some(_)) => "hipfire trouvé ; pilote du bench non branché (voir docs/BACKENDS.md)",
    };
    HipfireDetection {
        binary,
        supported_os,
        note: note.into(),
    }
}

/// Why `--backend hipfire` cannot run here, in the order a user can act on it: the OS, the
/// binary, then the driver. Always an error today; the message names the alternative.
pub fn unavailable(config: &BackendConfig) -> String {
    let found = detect(config);
    match (found.supported_os, found.binary) {
        (false, _) => format!(
            "hipfire : moteur HIP/ROCm pour Linux, non supporté sur {} (Windows et WSL2 non \
             testés par ses auteurs). Utilisez {ALTERNATIVE}.",
            std::env::consts::OS
        ),
        (true, None) => format!(
            "hipfire introuvable : donnez --hipfire-bin (ou hipfire_bin dans la configuration) \
             ou mettez-le dans le PATH. Sans ROCm, utilisez {ALTERNATIVE}."
        ),
        (true, Some(bin)) => format!(
            "hipfire ({}) est installé mais son pilote n'est pas branché : il n'expose que \
             /v1/chat/completions (son propre gabarit de chat fausserait les prompts déjà \
             rendus par le bench) et ne documente ni graine ni décodage glouton. Utilisez \
             {ALTERNATIVE}.",
            bin.display()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_error_names_the_cause_and_the_alternative() {
        let msg = unavailable(&BackendConfig::default());
        assert!(msg.contains("--backend llama-server"), "{msg}");
        if cfg!(target_os = "linux") {
            assert!(msg.contains("introuvable"), "{msg}");
        } else {
            assert!(msg.contains("non supporté"), "{msg}");
        }
    }

    #[test]
    fn an_installed_binary_is_found_through_hipfire_bin() {
        let dir = tempfile::tempdir().unwrap();
        let bin = dir.path().join("hipfire");
        std::fs::write(&bin, b"").unwrap();
        let config = BackendConfig {
            hipfire_bin: Some(bin.clone()),
            ..Default::default()
        };
        assert_eq!(detect(&config).binary, Some(bin));
        if cfg!(target_os = "linux") {
            assert!(unavailable(&config).contains("pilote n'est pas branché"));
        }
    }
}
