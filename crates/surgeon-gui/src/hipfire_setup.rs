//! hipfire for an AMD GPU on Windows, installed and started in WSL2 by the app: Ubuntu 24.04
//! when absent, then (as root, in the distro) the ROCm 7.2.1 userspace, the librocdxg bridge
//! to the Windows driver, Rust and the official hipfire installer, then its daemon in the
//! background; done when `127.0.0.1:11435` answers.
//!
//! The method is hipfire's own "Windows - WSL2" guide (docs/GETTING_STARTED.md, v0.4.6) and
//! AMD's ROCm 7.2.1 quick start for Ubuntu 24.04. The Windows side (Adrenalin 26.2.2 or newer
//! for WSL2) is the user's to install. ROCm under WSL supports a short list of GPUs
//! ([`wsl_supported`]); hipfire itself is untested on Windows GPUs.
//!
//! The RX 7600 XT (gfx1102) is outside that list but runs with three [`Workarounds`], found on
//! that card (hipfire 0.4.6, ROCm 7.2.1, librocdxg 1.2.2): a line in librocdxg's `dids.conf`,
//! the HIP runtime kept loaded (`LD_PRELOAD`) and `hipfire serve --spec ngram`. Only the first
//! one writes a file: a block between `# >>> weight-surgeon` markers, `dids.conf.bak` taken
//! before, skipped when present, removed by [`cleanup_script`] (`surgeon-gui
//! --hipfire-cleanup`). The other two are in the commands this module runs, nowhere else.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use weight_surgeon::backend::config::BackendConfig;

use crate::platform;
use crate::task::{self, Handle};
use crate::ui::i18n::t;

pub const DISTRO: &str = weight_surgeon::backend::hipfire::WSL_DISTRO;
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

/// What a GPU outside librocdxg's matrix needs to run hipfire under WSL.
#[derive(Debug, PartialEq, Eq)]
pub struct Workarounds {
    /// Its `device_id,major,minor,stepping` line in [`DIDS_CONF`]: without it librocdxg finds
    /// no GPU agent.
    pub dids: &'static str,
    /// The HIP runtime the daemon keeps loaded (`LD_PRELOAD`): hipfire unloads it on exit and
    /// then crashes under WSL (SIGSEGV), which also fails `hipfire setup`. The path is that of
    /// the ROCm release [`install_script`] installs.
    pub preload: &'static str,
    /// `hipfire serve --spec`: the plain Qwen2 route of hipfire 0.4.6 emits no `gen_start`,
    /// so its HTTP requests answer 500. To drop once hipfire fixes it.
    pub spec: &'static str,
}

/// The RX 7600 XT: Navi 33, gfx1102, PCI `DEV_7480`.
const GFX1102: Workarounds = Workarounds {
    dids: "0x7480,11,0,2",
    preload: "/opt/rocm-7.2.1/lib/libamdhip64.so",
    spec: "ngram",
};

/// librocdxg's supplemental PCI ids, merged with its built-in table.
const DIDS_CONF: &str = "/opt/rocm/share/rocdxg/dids.conf";
const BLOCK_START: &str = "# >>> weight-surgeon (gfx1102) >>>";
const BLOCK_END: &str = "# <<< weight-surgeon <<<";

/// The AMD GPUs ROCm 7.2.x runs under WSL (librocdxg 1.2 matrix, as hipfire's guide quotes
/// it), by the names Windows gives them, then those that run with [`Workarounds`].
const WSL_GPUS: [(&str, Option<&Workarounds>); 14] = [
    ("RX 9070", None),
    ("R9700", None),
    ("RX 9060", None),
    ("RX 7900", None),
    ("W7900", None),
    ("W7800", None),
    ("RX 7800 XT", None),
    ("W7700", None),
    ("8060S", None),
    ("8050S", None),
    ("8040S", None),
    ("890M", None),
    ("880M", None),
    ("RX 7600 XT", Some(&GFX1102)),
];

fn wsl_gpu(gpu: &str) -> Option<Option<&'static Workarounds>> {
    let name = gpu.to_ascii_uppercase();
    WSL_GPUS
        .iter()
        .find(|(g, _)| name.contains(&g.to_ascii_uppercase()))
        .map(|(_, w)| *w)
}

pub fn wsl_supported(gpu: &str) -> bool {
    wsl_gpu(gpu).is_some()
}

/// What `gpu` needs beyond the stock install, if anything.
pub fn workarounds(gpu: &str) -> Option<&'static Workarounds> {
    wsl_gpu(gpu).flatten()
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
/// failure resumes. With `workarounds`, their `dids.conf` line after librocdxg and their
/// `LD_PRELOAD` for hipfire's installer (it runs `hipfire setup`).
pub fn install_script(workarounds: Option<&Workarounds>) -> String {
    let (dids, preload) = match workarounds {
        Some(w) => (dids_step(w, DIDS_CONF), preload_step(w)),
        None => Default::default(),
    };
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
{dids}echo "[4/5] Rust"
[ -x "$HOME/.cargo/bin/cargo" ] || curl -fsSL https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
echo "[5/5] hipfire {HIPFIRE_TAG}"
export PATH="/opt/rocm/bin:$PATH" HSA_ENABLE_DXG_DETECTION=1
{preload}[ -x {HIPFIRE_BIN} ] || curl -fsSL https://raw.githubusercontent.com/warpfront/hipfire/{HIPFIRE_TAG}/scripts/install.sh | bash -s -- --yes --tag {HIPFIRE_TAG}
"#
    )
}

/// The `dids.conf` line of `w` in `conf`, once: a line already there (in the weight-surgeon
/// block, or written by hand) is left alone; else `conf.bak` is taken if there is none yet,
/// and the line goes in between the markers [`cleanup_script`] looks for.
fn dids_step(w: &Workarounds, conf: &str) -> String {
    let dids = w.dids;
    format!(
        r#"echo "      {dids} dans {conf} (GPU hors de la matrice de librocdxg)"
if grep -qxF '{BLOCK_START}' {conf} 2>/dev/null; then
  echo "      déjà présente (bloc weight-surgeon)"
elif grep -qxF '{dids}' {conf} 2>/dev/null; then
  echo "      déjà présente, ajoutée hors weight-surgeon : laissée telle quelle"
else
  if [ -e {conf} ] && [ ! -e {conf}.bak ]; then
    cp {conf} {conf}.bak
    echo "      sauvegarde : {conf}.bak"
  fi
  printf '%s\n' '{BLOCK_START}' '{dids}' '{BLOCK_END}' >> {conf}
  echo "      ajoutée (retrait : surgeon-gui --hipfire-cleanup)"
fi
"#
    )
}

/// `LD_PRELOAD` of `w` when its library is there (another ROCm release puts it elsewhere).
fn preload_step(w: &Workarounds) -> String {
    let (lib, spec) = (w.preload, w.spec);
    format!(
        r#"if [ -f {lib} ]; then
  export LD_PRELOAD={lib}
  echo "      LD_PRELOAD={lib} ; le démon tournera avec et avec --spec {spec}"
else
  echo "      attention : {lib} absent, hipfire risque de planter à sa sortie (SIGSEGV)"
fi
"#
    )
}

/// Takes back what [`install_script`] wrote for a GPU's [`Workarounds`]: the weight-surgeon
/// block of `conf` (the `.bak` of the original stays). Nothing else was written: the
/// `LD_PRELOAD` and `--spec` only live in the commands this module runs.
pub fn cleanup_script(conf: &str) -> String {
    format!(
        r#"set -euo pipefail
if grep -qxF '{BLOCK_START}' {conf} 2>/dev/null; then
  sed -i '/^# >>> weight-surgeon/,/^# <<< weight-surgeon/d' {conf}
  echo "bloc weight-surgeon retiré de {conf} (original gardé : {conf}.bak)"
else
  echo "rien à retirer : aucun bloc weight-surgeon dans {conf}"
fi
"#
    )
}

/// The daemon, in the foreground of its `wsl.exe` (which keeps the distro up while it runs).
fn serve_script(workarounds: Option<&Workarounds>) -> String {
    let (preload, spec) = match workarounds {
        Some(w) => (
            format!(" LD_PRELOAD={}", w.preload),
            format!(" --spec {}", w.spec),
        ),
        None => Default::default(),
    };
    format!(
        r#"export PATH="/opt/rocm/bin:$PATH" HSA_ENABLE_DXG_DETECTION=1{preload}; exec {HIPFIRE_BIN} serve{spec}"#
    )
}

/// Stops every hipfire process of the distro: the HTTP front and the GPU process it owns. A
/// daemon that lost its GPU still answers `/health`, so a restart starts here. `[h]` keeps the
/// pattern from matching the shell that runs this script.
const STOP_SCRIPT: &str = r#"pkill -f '[h]ipfire' || true
for _ in $(seq 1 50); do
  pgrep -f '[h]ipfire' >/dev/null || exit 0
  sleep 0.2
done
pkill -9 -f '[h]ipfire' || true
"#;

/// What `start-hipfire.ps1` runs as the user: `~/start-hipfire.sh`, its output in the log
/// `surgeon doctor` reads.
const USER_SERVE: &str = r#"exec "$HOME/start-hipfire.sh" > "$HOME/hipfire-serve.log" 2>&1"#;

fn wsl() -> std::process::Command {
    platform::command(PathBuf::from("wsl.exe"))
}

/// `-e` hands the script to bash as is: `--` would run it through another shell first,
/// which strips its quotes.
fn as_user(user: &str, script: &str) -> std::process::Command {
    let mut cmd = wsl();
    cmd.args(["-d", DISTRO, "-u", user, "-e", "bash", "-lc", script]);
    cmd
}

fn as_root(script: &str) -> std::process::Command {
    as_user("root", script)
}

/// `surgeon-gui --hipfire-cleanup`: [`cleanup_script`] in the distribution; what it printed.
pub fn cleanup() -> Result<String, String> {
    let (code, out) =
        platform::run_capture(as_root(&cleanup_script(DIDS_CONF)), Duration::from_secs(60))
            .map_err(|e| e.to_string())?;
    match code {
        Some(0) => Ok(out),
        _ => Err(format!("{DISTRO} : nettoyage en échec ({code:?}) : {out}")),
    }
}

/// Installs what is missing (with what `gpu` needs beyond it), starts the daemon and waits
/// for it to answer.
pub fn run(handle: &Handle, gpu: &str) -> Result<(), String> {
    let workarounds = workarounds(gpu);
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
    task::run_streamed(as_root(&install_script(workarounds)), handle)?;
    serve(handle, as_root(&serve_script(workarounds)))
}

/// Stops the daemon of the distro (frozen at 0 MB of GPU, or stopped) and starts it again:
/// with the user's `~/start-hipfire.sh` when it is there, as `start-hipfire.ps1` does (that
/// user's models), else with what [`run`] installed as root. Nothing is installed here.
pub fn restart(handle: &Handle, gpu: &str) -> Result<(), String> {
    handle.report(t("hipfire.step.stop"), 0, None);
    let (code, out) = platform::run_capture(as_root(STOP_SCRIPT), Duration::from_secs(60))
        .map_err(|e| e.to_string())?;
    if code != Some(0) {
        return Err(format!(
            "{DISTRO} : arrêt de hipfire en échec ({code:?}) : {out}"
        ));
    }
    let user = std::env::var("USERNAME").ok().filter(|u| {
        let test = as_user(u, r#"test -x "$HOME/start-hipfire.sh""#);
        platform::run_capture(test, Duration::from_secs(30)).is_ok_and(|(code, _)| code == Some(0))
    });
    let command = match user {
        Some(u) => as_user(&u, USER_SERVE),
        None => as_root(&serve_script(workarounds(gpu))),
    };
    serve(handle, command)
}

/// Starts the daemon in the background of its `wsl.exe` (which keeps the distro up while it
/// runs) and waits for it to answer.
fn serve(handle: &Handle, mut command: std::process::Command) -> Result<(), String> {
    handle.report(t("hipfire.step.serve"), 0, None);
    command
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
        assert!(!wsl_supported("AMD Radeon RX 6700 XT"));
        assert_eq!(workarounds("AMD Radeon RX 7900 XTX"), None, "in the matrix");
        // gfx1102 is outside the matrix: supported through its workarounds.
        assert!(wsl_supported("AMD Radeon RX 7600 XT"));
        assert_eq!(workarounds("AMD Radeon RX 7600 XT"), Some(&GFX1102));
        assert_eq!(GFX1102.dids, "0x7480,11,0,2");
        assert_eq!(
            workarounds("AMD Radeon RX 7600"),
            None,
            "the XT only was tried"
        );
    }

    #[test]
    fn the_gfx1102_workarounds_reach_the_install_and_the_daemon() {
        let stock = install_script(None);
        assert!(!stock.contains("dids.conf") && !stock.contains("LD_PRELOAD"));
        assert!(serve_script(None).ends_with("hipfire serve"));

        let script = install_script(Some(&GFX1102));
        let dids = script.find(DIDS_CONF).unwrap();
        assert!(
            dids > script.find("dpkg -i /tmp/rocdxg.deb").unwrap(),
            "after librocdxg"
        );
        let preload = script.find("export LD_PRELOAD=").unwrap();
        assert!(
            preload < script.find("install.sh").unwrap(),
            "hipfire setup runs with it"
        );
        assert!(script.contains("weight-surgeon (gfx1102)") && script.contains(".bak"));
        assert_eq!(
            serve_script(Some(&GFX1102)),
            "export PATH=\"/opt/rocm/bin:$PATH\" HSA_ENABLE_DXG_DETECTION=1 \
             LD_PRELOAD=/opt/rocm-7.2.1/lib/libamdhip64.so; \
             exec /root/.hipfire/src/target/release/hipfire serve --spec ngram"
        );
    }

    /// The `dids.conf` step run twice, then the cleanup, on a scratch copy of the file.
    #[cfg(unix)]
    #[test]
    fn the_dids_line_goes_in_once_and_comes_out_again() {
        let dir = tempfile::tempdir().unwrap();
        let conf = dir.path().join("dids.conf");
        let original = "# Supplemental PCI device IDs\n0x1234,11,0,0\n";
        std::fs::write(&conf, original).unwrap();
        let path = conf.to_str().unwrap();
        let bash = |script: String| {
            let out = std::process::Command::new("bash")
                .args(["-c", &format!("set -euo pipefail\n{script}")])
                .output()
                .unwrap();
            assert!(out.status.success(), "{out:?}");
            String::from_utf8(out.stdout).unwrap()
        };
        let read = || std::fs::read_to_string(&conf).unwrap();

        assert!(bash(dids_step(&GFX1102, path)).contains("ajoutée"));
        let added = read();
        assert_eq!(
            added,
            format!("{original}{BLOCK_START}\n0x7480,11,0,2\n{BLOCK_END}\n")
        );
        assert_eq!(
            std::fs::read_to_string(dir.path().join("dids.conf.bak")).unwrap(),
            original
        );
        // Idempotent and detectable.
        assert!(bash(dids_step(&GFX1102, path)).contains("bloc weight-surgeon"));
        assert_eq!(read(), added);
        // Reversible: the block goes, the backup stays.
        assert!(bash(cleanup_script(path)).contains("retiré"));
        assert_eq!(read(), original);
        assert!(bash(cleanup_script(path)).contains("rien à retirer"));
        // A line written by hand is never touched, nor wrapped.
        std::fs::write(&conf, "0x7480,11,0,2\n").unwrap();
        assert!(bash(dids_step(&GFX1102, path)).contains("hors weight-surgeon"));
        assert_eq!(read(), "0x7480,11,0,2\n");
    }

    #[test]
    fn a_restart_stops_hipfire_but_never_its_own_shell() {
        // pkill -f reads command lines: the script's own text must not hold the plain word.
        assert!(!STOP_SCRIPT.contains("hipfire"), "{STOP_SCRIPT}");
        assert!(STOP_SCRIPT.contains("pkill -f '[h]ipfire'"));
        assert!(
            STOP_SCRIPT.contains("pkill -9"),
            "a daemon that ignores TERM"
        );
        // The user's script, as start-hipfire.ps1 runs it, logging where surgeon doctor reads.
        let args: Vec<String> = as_user("alice", USER_SERVE)
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            ["-d", DISTRO, "-u", "alice", "-e", "bash", "-lc", USER_SERVE]
        );
        assert!(
            USER_SERVE.contains("$HOME/start-hipfire.sh")
                && USER_SERVE.contains("hipfire-serve.log")
        );
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
        let script = install_script(None);
        assert!(script.contains(&format!(
            "{ROCDXG_SHA256}  /tmp/rocdxg.deb\" | sha256sum -c -"
        )));
        assert!(script.contains("--yes --tag v0.4.6"));
        assert!(
            script.starts_with("set -euo pipefail"),
            "the first failure stops it"
        );
    }

    #[test]
    fn scripts_reach_bash_unparsed() {
        let cmd = as_root("echo 'a b'");
        let args: Vec<_> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            [
                "-d",
                DISTRO,
                "-u",
                "root",
                "-e",
                "bash",
                "-lc",
                "echo 'a b'"
            ]
        );
    }
}
