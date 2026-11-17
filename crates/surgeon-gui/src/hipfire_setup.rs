//! hipfire for an AMD GPU on Windows, installed and started in WSL2 by the app: Ubuntu 24.04
//! when absent, then (as root, in the distro) the ROCm 7.2.1 userspace, the librocdxg bridge
//! to the Windows driver, Rust and the official hipfire installer, then its daemon in the
//! background; done when `127.0.0.1:11435` answers.
//!
//! The method is hipfire's own "Windows - WSL2" guide (docs/GETTING_STARTED.md, v0.4.6) and
//! AMD's ROCm 7.2.1 quick start for Ubuntu 24.04. The Windows side (Adrenalin 26.2.2 or newer
//! for WSL2) is the user's to install. ROCm under WSL supports a short list of GPUs
//! ([`wsl_supported`]); hipfire itself is untested on Windows GPUs.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use weight_surgeon::backend::config::BackendConfig;

use crate::platform;
use crate::task::{self, Handle};
use crate::ui::i18n::t;

pub const DISTRO: &str = "Ubuntu-24.04";
pub const HIPFIRE_TAG: &str = "v0.4.6";
const ROCM_DEB: &str =
    "https://repo.radeon.com/amdgpu-install/7.2.1/ubuntu/noble/amdgpu-install_7.2.1.70201-1_all.deb";
const ROCDXG_DEB: &str =
    "https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb";
/// The digest the librocdxg release publishes for that package.
const ROCDXG_SHA256: &str = "28ded1254811192ebace1f76c0227580184af7b27ab2475fb9728295a702d541";
/// Where the official installer leaves the binary (`$HIPFIRE_HOME/src/target/release`).
const HIPFIRE_BIN: &str = "/root/.hipfire/src/target/release/hipfire";
/// How long the daemon may take to answer once started.
const SERVE_WAIT: Duration = Duration::from_secs(180);

/// The AMD GPUs ROCm 7.2.x runs under WSL (librocdxg 1.2 matrix, as hipfire's guide quotes
/// it), by the names Windows gives them.
const WSL_GPUS: [&str; 13] = [
    "RX 9070",
    "R9700",
    "RX 9060",
    "RX 7900",
    "W7900",
    "W7800",
    "RX 7800 XT",
    "W7700",
    "8060S",
    "8050S",
    "8040S",
    "890M",
    "880M",
];

pub fn wsl_supported(gpu: &str) -> bool {
    let name = gpu.to_ascii_uppercase();
    WSL_GPUS
        .iter()
        .any(|g| name.contains(&g.to_ascii_uppercase()))
}

/// The distributions `wsl -l -q` lists: UTF-16 read as UTF-8, so NUL bytes and a mangled
/// byte-order mark are dropped (the names are ASCII).
pub fn parse_distros(text: &str) -> Vec<String> {
    text.replace('\0', "")
        .lines()
        .map(|l| l.trim_matches(|c: char| c.is_whitespace() || c == '\u{feff}' || c == '\u{fffd}'))
        .filter(|l| !l.is_empty())
        .map(str::to_owned)
        .collect()
}

/// What runs as root in the distro: each step skipped when already done, so a rerun after a
/// failure resumes.
pub fn install_script() -> String {
    format!(
        r#"set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
echo "[1/5] paquets de base"
apt-get update -q
apt-get install -y -q curl ca-certificates git build-essential python3-setuptools python3-wheel
echo "[2/5] ROCm 7.2.1 (plusieurs Go)"
if [ ! -x /opt/rocm/bin/hipcc ]; then
  curl -fsSL -o /tmp/amdgpu-install.deb {ROCM_DEB}
  apt-get install -y -q /tmp/amdgpu-install.deb
  apt-get update -q
  apt-get install -y -q rocm
fi
echo "[3/5] librocdxg 1.2.2"
curl -fsSL -o /tmp/rocdxg.deb {ROCDXG_DEB}
echo "{ROCDXG_SHA256}  /tmp/rocdxg.deb" | sha256sum -c -
dpkg -i /tmp/rocdxg.deb
grep -q HSA_ENABLE_DXG_DETECTION /etc/environment || echo HSA_ENABLE_DXG_DETECTION=1 >> /etc/environment
echo "[4/5] Rust"
[ -x "$HOME/.cargo/bin/cargo" ] || curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
echo "[5/5] hipfire {HIPFIRE_TAG}"
export PATH="/opt/rocm/bin:$PATH" HSA_ENABLE_DXG_DETECTION=1
[ -x {HIPFIRE_BIN} ] || curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/{HIPFIRE_TAG}/scripts/install.sh | bash -s -- --yes --tag {HIPFIRE_TAG}
"#
    )
}

/// The daemon, in the foreground of its `wsl.exe` (which keeps the distro up while it runs).
fn serve_script() -> String {
    format!(
        r#"export PATH="/opt/rocm/bin:$PATH" HSA_ENABLE_DXG_DETECTION=1; exec {HIPFIRE_BIN} serve"#
    )
}

fn wsl() -> std::process::Command {
    platform::command(PathBuf::from("wsl.exe"))
}

fn as_root(script: &str) -> std::process::Command {
    let mut cmd = wsl();
    cmd.args(["-d", DISTRO, "-u", "root", "--", "bash", "-lc", script]);
    cmd
}

/// Installs what is missing, starts the daemon and waits for it to answer.
pub fn run(handle: &Handle) -> Result<(), String> {
    handle.report(t("hipfire.step.distro"), 0, None);
    let mut list = wsl();
    list.args(["-l", "-q"]);
    let (_, listed) =
        platform::run_capture(list, Duration::from_secs(30)).map_err(|e| e.to_string())?;
    if !parse_distros(&listed).iter().any(|d| d == DISTRO) {
        let mut install = wsl();
        install.args(["--install", "-d", DISTRO, "--no-launch"]);
        task::run_streamed(install, handle)?;
    }
    // A fresh distro may still want its first interactive start (or Windows a reboot).
    let (code, _) = platform::run_capture(as_root("true"), Duration::from_secs(120))
        .map_err(|e| e.to_string())?;
    if code != Some(0) {
        return Err(t("hipfire.distro.not_ready").into());
    }
    task::run_streamed(as_root(&install_script()), handle)?;

    handle.report(t("hipfire.step.serve"), 0, None);
    as_root(&serve_script())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| e.to_string())?;
    let config = BackendConfig {
        hipfire_url: Some(crate::backend_choice::hipfire_url()),
        ..Default::default()
    };
    let deadline = Instant::now() + SERVE_WAIT;
    while Instant::now() < deadline {
        if handle.cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return Err(t("setup.cancelled").into());
        }
        if weight_surgeon::backend::hipfire::probe(&config).is_ok() {
            return Ok(());
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    Err(t("hipfire.serve.silent").into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_gpus_rocm_runs_under_wsl_are_supported() {
        assert!(wsl_supported("AMD Radeon RX 7900 XTX"));
        assert!(wsl_supported("AMD Radeon RX 9070 XT"));
        assert!(wsl_supported("AMD Radeon(TM) 8060S Graphics"));
        assert!(
            !wsl_supported("AMD Radeon RX 7600 XT"),
            "gfx1102 is not in the matrix"
        );
        assert!(!wsl_supported("AMD Radeon RX 6700 XT"));
    }

    #[test]
    fn distributions_are_read_from_utf16() {
        let raw: Vec<u8> = "\u{feff}docker-desktop\r\nUbuntu-24.04\r\n"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        let read = String::from_utf8_lossy(&raw);
        assert_eq!(parse_distros(&read), ["docker-desktop", "Ubuntu-24.04"]);
    }

    #[test]
    fn the_script_verifies_librocdxg_and_pins_hipfire() {
        let script = install_script();
        assert!(script.contains(&format!(
            "{ROCDXG_SHA256}  /tmp/rocdxg.deb\" | sha256sum -c -"
        )));
        assert!(script.contains("--yes --tag v0.4.6"));
        assert!(
            script.starts_with("set -euo pipefail"),
            "the first failure stops it"
        );
        assert!(serve_script().ends_with("hipfire serve"));
    }
}
