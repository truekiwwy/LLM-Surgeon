//! What the first-run wizard and the Diagnostics page tell the user: the machine (GPU and
//! graphics API, memory, disk) and the state of every dependency, with an export that holds
//! no personal path and no secret.
//!
//! The OS is asked through its own tools (PowerShell/CIM on Windows) so that nothing here
//! needs `unsafe`; parsers are pure and tested. A failed query is a note in the report,
//! never a crash: the app stays usable on a machine it cannot describe.

use std::path::{Path, PathBuf};

use serde::Deserialize;
use weight_surgeon::backend::config::BackendConfig;
use weight_surgeon::hardware::{HardwareProbe, SystemProbe};

use crate::install::{self, Installed};
use crate::logging;
use crate::platform;
use crate::sample_model::QWEN;
use crate::settings::Lang;

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Gpu {
    pub name: String,
    pub vram_mib: Option<u64>,
    pub driver: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct SystemReport {
    pub os: String,
    pub arch: &'static str,
    pub gpus: Vec<Gpu>,
    /// `Some(false)`: no Vulkan loader, llama.cpp's GPU build cannot run and the app falls
    /// back to software rendering and the CPU engines.
    pub vulkan: Option<bool>,
    pub ram_total_mib: u64,
    pub ram_free_mib: u64,
    pub disk_free_mib: Option<u64>,
    pub notes: Vec<String>,
}

#[derive(Deserialize)]
struct RawGpu {
    #[serde(default)]
    name: String,
    #[serde(default)]
    adapter_ram: Option<u64>,
    #[serde(default)]
    qw: Option<u64>,
    #[serde(default)]
    driver: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum OneOrMany {
    Many(Vec<RawGpu>),
    One(RawGpu),
}

#[derive(Deserialize)]
struct RawProbe {
    #[serde(default)]
    gpus: Option<OneOrMany>,
    #[serde(default)]
    disk_free: Option<u64>,
}

/// The PowerShell answer: `{"gpus":[{name,adapter_ram,qw,driver}],"disk_free":N}`. `qw` (the
/// 64-bit VRAM size from the display driver's registry key) wins over the 32-bit
/// `AdapterRAM`, which tops out at 4 GiB.
pub fn parse_probe(json: &str) -> Result<(Vec<Gpu>, Option<u64>), String> {
    let raw: RawProbe = serde_json::from_str(json.trim()).map_err(|e| e.to_string())?;
    let gpus = match raw.gpus {
        None => Vec::new(),
        Some(OneOrMany::One(g)) => vec![g],
        Some(OneOrMany::Many(g)) => g,
    };
    let gpus = gpus
        .into_iter()
        .filter(|g| !g.name.is_empty())
        .map(|g| Gpu {
            vram_mib: g.qw.or(g.adapter_ram).filter(|b| *b > 0).map(|b| b >> 20),
            name: g.name,
            driver: g.driver,
        })
        .collect();
    Ok((gpus, raw.disk_free.map(|b| b >> 20)))
}

const PROBE_SCRIPT: &str = r"$ErrorActionPreference='SilentlyContinue';
$mem=@{};
Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0*' | ForEach-Object { if ($_.'HardwareInformation.qwMemorySize') { $mem[[string]$_.DriverDesc]=[uint64]$_.'HardwareInformation.qwMemorySize' } };
$g=@(Get-CimInstance Win32_VideoController | ForEach-Object { @{name=$_.Name;adapter_ram=[uint64]$_.AdapterRAM;qw=$mem[[string]$_.Name];driver=$_.DriverVersion} });
$d=$null; try { $d=[System.IO.DriveInfo]::new([System.IO.Path]::GetPathRoot($env:SURGEON_PROBE_DIR)).AvailableFreeSpace } catch {};
@{gpus=$g;disk_free=$d} | ConvertTo-Json -Compress -Depth 4";

/// Whether the Vulkan loader is installed: the file the drivers register.
pub fn vulkan_present() -> Option<bool> {
    if cfg!(windows) {
        let root = std::env::var_os("SystemRoot").map(PathBuf::from)?;
        return Some(root.join("System32").join("vulkan-1.dll").is_file());
    }
    let found = [
        "/usr/lib",
        "/usr/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
    ]
    .iter()
    .any(|d| Path::new(d).join("libvulkan.so.1").is_file());
    Some(found)
}

/// Asks the machine. `dir` is the folder whose drive's free space matters (the data folder).
pub fn probe(dir: &Path) -> SystemReport {
    let mut report = SystemReport {
        os: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH,
        vulkan: vulkan_present(),
        ..Default::default()
    };
    match SystemProbe::detect().system_ram() {
        Ok(ram) => {
            report.ram_total_mib = ram.total_bytes >> 20;
            report.ram_free_mib = ram.available_bytes >> 20;
        }
        Err(e) => report.notes.push(format!("RAM : {e}")),
    }
    if cfg!(windows) {
        let mut cmd = platform::command("powershell");
        cmd.args(["-NoProfile", "-NonInteractive", "-Command", PROBE_SCRIPT])
            .env("SURGEON_PROBE_DIR", dir);
        match platform::run_capture(cmd, std::time::Duration::from_secs(20)) {
            Ok((Some(0), out)) => match parse_probe(&out) {
                Ok((gpus, disk)) => {
                    report.gpus = gpus;
                    report.disk_free_mib = disk;
                }
                Err(e) => report.notes.push(format!("lecture des GPU : {e}")),
            },
            Ok((code, out)) => report.notes.push(format!(
                "PowerShell {code:?} : {}",
                out.lines().next().unwrap_or("")
            )),
            Err(e) => report.notes.push(format!("PowerShell : {e}")),
        }
    }
    report
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Ok,
    Missing,
    Outdated,
}

/// What a row's Repair / Reinstall buttons act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fix {
    LlamaCpp,
    SampleModel,
    Workspace,
    /// Nothing the app can do (a GPU driver, the CLI next to the exe): the detail says what.
    None,
}

/// What a dependency is for: the group it is listed under, in this order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Group {
    Essential,
    Acceleration,
    Agents,
    Samples,
}

/// How much a row's state matters, from what the user actually uses: only `Blocking` stops
/// the work, `NeededBy` names the active choice that needs an optional piece.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Ok,
    Optional,
    NeededBy(&'static str),
    Blocking,
}

/// The active choices that make an optional dependency needed.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Uses {
    /// The backend picked is `llama-server`.
    pub llama_backend: bool,
    /// A CLI runtime is configured (it starts `surgeon mcp`): its name.
    pub cli: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub struct Row {
    pub name: &'static str,
    pub status: Status,
    pub detail: String,
    pub fix: Fix,
    pub group: Group,
    pub severity: Severity,
}

/// `Blocking` and `NeededBy` rows: what the user has to act on.
pub fn actions_needed(rows: &[Row]) -> usize {
    rows.iter()
        .filter(|r| matches!(r.severity, Severity::Blocking | Severity::NeededBy(_)))
        .count()
}

/// A missing piece is `Blocking` when the work needs it, `NeededBy(who)` when an active
/// choice does, else `Optional`.
fn severity(status: Status, required: bool, needed_by: Option<&'static str>) -> Severity {
    match (status, required, needed_by) {
        (Status::Ok, _, _) => Severity::Ok,
        (_, true, _) => Severity::Blocking,
        (Status::Missing, false, Some(who)) => Severity::NeededBy(who),
        _ => Severity::Optional,
    }
}

/// `b12360` -> 12360.
pub fn build_number(tag: &str) -> Option<u64> {
    tag.trim_start_matches('b').parse().ok()
}

/// Whether `latest` (a release tag) is newer than `installed`.
pub fn is_outdated(installed: &str, latest: &str) -> bool {
    matches!((build_number(installed), build_number(latest)), (Some(i), Some(l)) if l > i)
}

/// The state of each dependency, its group and its severity under `uses`. `latest_tag` is
/// the newest llama.cpp release when the releases API was reachable (no network call is
/// made here).
pub fn dependencies(
    components: &Path,
    config: &BackendConfig,
    workspace: Option<&Path>,
    system: Option<&SystemReport>,
    latest_tag: Option<&str>,
    uses: Uses,
    lang: Lang,
) -> Vec<Row> {
    let tr = |fr: &str, en: &str| match lang {
        Lang::Fr => fr.to_owned(),
        Lang::En => en.to_owned(),
    };
    // Group and severity are set once every row is known.
    struct Base {
        name: &'static str,
        status: Status,
        detail: String,
        fix: Fix,
    }
    let mut rows = Vec::new();

    rows.push(Base {
        name: "weight-surgeon",
        status: Status::Ok,
        detail: format!(
            "{} {}",
            tr(
                "moteur de chirurgie intégré à l'application",
                "surgery engine built into the app"
            ),
            env!("CARGO_PKG_VERSION")
        ),
        fix: Fix::None,
    });

    let trusted: Option<Installed> = config.llama_cpp_dir.as_ref().and_then(|dir| {
        install::list_installed(components)
            .into_iter()
            .find(|i| &i.dir == dir)
    });
    let vulkan_build = trusted
        .as_ref()
        .is_some_and(|i| i.manifest.variant == "vulkan");
    rows.push(match (&config.llama_cpp_dir, trusted) {
        (None, _) => Base {
            name: "llama.cpp",
            status: Status::Missing,
            detail: tr(
                "non installé : aucun dossier de confiance",
                "not installed: no trusted folder",
            ),
            fix: Fix::LlamaCpp,
        },
        (Some(dir), None) if dir.join(install::server_name()).is_file() => Base {
            name: "llama.cpp",
            status: Status::Ok,
            detail: tr(
                "dossier de confiance externe (non géré par l'app)",
                "external trusted folder (not managed by the app)",
            ),
            fix: Fix::LlamaCpp,
        },
        (Some(_), None) => Base {
            name: "llama.cpp",
            status: Status::Missing,
            detail: tr(
                "le dossier de confiance est introuvable ou incomplet",
                "the trusted folder is missing or incomplete",
            ),
            fix: Fix::LlamaCpp,
        },
        (Some(_), Some(i)) => {
            let outdated = latest_tag.is_some_and(|l| is_outdated(&i.manifest.tag, l));
            Base {
                name: "llama.cpp",
                status: if outdated {
                    Status::Outdated
                } else {
                    Status::Ok
                },
                detail: format!(
                    "{} ({}){}",
                    i.manifest.tag,
                    i.manifest.variant,
                    latest_tag
                        .filter(|_| outdated)
                        .map(|l| format!(" -> {l}"))
                        .unwrap_or_default()
                ),
                fix: Fix::LlamaCpp,
            }
        }
    });

    rows.push(match system.and_then(|s| s.vulkan) {
        Some(true) => Base {
            name: "Vulkan",
            status: Status::Ok,
            detail: tr("chargeur Vulkan présent", "Vulkan loader present"),
            fix: Fix::None,
        },
        Some(false) => Base {
            name: "Vulkan",
            status: Status::Missing,
            detail: tr(
                "vulkan-1.dll absent : installez le pilote GPU du constructeur. Sans lui, le build GPU de llama.cpp ne démarre pas : utilisez le build CPU ou les moteurs natifs (l'application reste utilisable, son interface utilise OpenGL)",
                "vulkan-1.dll missing: install the GPU vendor driver. Without it the llama.cpp GPU build cannot start: use the CPU build or the native engines (the app stays usable, its interface uses OpenGL)",
            ),
            fix: Fix::None,
        },
        None => Base {
            name: "Vulkan",
            status: Status::Missing,
            detail: tr("diagnostic en cours ou indisponible", "diagnostic pending or unavailable"),
            fix: Fix::None,
        },
    });

    let cli = std::env::current_exe().ok().and_then(|exe| {
        exe.parent()
            .map(|d| d.join(format!("surgeon{}", std::env::consts::EXE_SUFFIX)))
    });
    rows.push(match cli.filter(|p| p.is_file()) {
        Some(path) => Base {
            name: "surgeon (CLI)",
            status: Status::Ok,
            detail: path.display().to_string(),
            fix: Fix::None,
        },
        None => Base {
            name: "surgeon (CLI)",
            status: Status::Missing,
            detail: tr("surgeon.exe absent à côté de l'application (utilisation de la bibliothèque intégrée)", "surgeon.exe not next to the app (the built-in library is used)"),
            fix: Fix::None,
        },
    });

    rows.push(match workspace {
        Some(dir) if dir.is_dir() => Base {
            name: "workspace",
            status: Status::Ok,
            detail: dir.display().to_string(),
            fix: Fix::Workspace,
        },
        _ => Base {
            name: "workspace",
            status: Status::Missing,
            detail: tr(
                "dossier d'espace de travail absent",
                "workspace folder missing",
            ),
            fix: Fix::Workspace,
        },
    });

    let sample = workspace.is_some_and(|w| QWEN.is_installed(w));
    rows.push(Base {
        name: "Qwen2.5-0.5B-Instruct",
        status: if sample { Status::Ok } else { Status::Missing },
        detail: tr("modèle d'exemple", "sample model"),
        fix: Fix::SampleModel,
    });

    rows.into_iter()
        .map(|b| {
            let (group, required, needed_by) = match b.name {
                "weight-surgeon" | "workspace" => (Group::Essential, true, None),
                "llama.cpp" => (
                    Group::Acceleration,
                    false,
                    uses.llama_backend.then_some("llama-server"),
                ),
                "Vulkan" => (
                    Group::Acceleration,
                    false,
                    (uses.llama_backend && vulkan_build).then_some("llama.cpp Vulkan"),
                ),
                "surgeon (CLI)" => (Group::Agents, false, uses.cli),
                _ => (Group::Samples, false, None),
            };
            Row {
                severity: severity(b.status, required, needed_by),
                name: b.name,
                status: b.status,
                detail: b.detail,
                fix: b.fix,
                group,
            }
        })
        .collect()
}

pub fn status_word(status: Status, lang: Lang) -> &'static str {
    match (status, lang) {
        (Status::Ok, _) => "OK",
        (Status::Missing, Lang::Fr) => "manquant",
        (Status::Missing, Lang::En) => "missing",
        (Status::Outdated, Lang::Fr) => "obsolète",
        (Status::Outdated, Lang::En) => "outdated",
    }
}

/// The text a user can attach to a bug report: no personal path, no secret.
pub fn export(system: Option<&SystemReport>, rows: &[Row], log_dir: &Path, lang: Lang) -> String {
    let mut text = format!("surgeon-gui {}\n", env!("CARGO_PKG_VERSION"));
    if let Some(s) = system {
        text.push_str(&format!(
            "os: {} {}\nram: {} MiB (free {})\ndisk free: {}\nvulkan: {}\n",
            s.os,
            s.arch,
            s.ram_total_mib,
            s.ram_free_mib,
            s.disk_free_mib.map_or("?".into(), |d| format!("{d} MiB")),
            s.vulkan.map_or("?", |v| if v { "yes" } else { "no" }),
        ));
        for gpu in &s.gpus {
            text.push_str(&format!(
                "gpu: {} | vram {} | driver {}\n",
                gpu.name,
                gpu.vram_mib.map_or("?".into(), |v| format!("{v} MiB")),
                gpu.driver.as_deref().unwrap_or("?")
            ));
        }
        for note in &s.notes {
            text.push_str(&format!("note: {note}\n"));
        }
    }
    text.push_str("\ndependencies:\n");
    for row in rows {
        text.push_str(&format!(
            "- {} [{}] {}\n",
            row.name,
            status_word(row.status, lang),
            row.detail
        ));
    }
    text.push_str("\nlog (tail):\n");
    for line in logging::tail(log_dir, 200) {
        text.push_str(&line);
        text.push('\n');
    }
    let user = std::env::var("USERNAME")
        .or_else(|_| std::env::var("USER"))
        .ok();
    logging::redact(&text, crate::paths::home_dir().as_deref(), user.as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_powershell_answer_is_parsed_with_one_or_many_gpus() {
        let many = r#"{"gpus":[{"name":"AMD Radeon RX 7600 XT","adapter_ram":4293918720,"qw":17163091968,"driver":"32.0.1"},{"name":"Basic","adapter_ram":0,"qw":null,"driver":null}],"disk_free":53687091200}"#;
        let (gpus, disk) = parse_probe(many).unwrap();
        assert_eq!(gpus.len(), 2);
        assert_eq!(
            gpus[0].vram_mib,
            Some(16368),
            "the 64-bit size wins over AdapterRAM"
        );
        assert_eq!(gpus[1].vram_mib, None);
        assert_eq!(disk, Some(51200));
        let one = r#"{"gpus":{"name":"X","adapter_ram":1048576},"disk_free":null}"#;
        assert_eq!(parse_probe(one).unwrap().0[0].vram_mib, Some(1));
        assert!(parse_probe("not json").is_err());
        assert!(parse_probe(r#"{"gpus":null,"disk_free":null}"#)
            .unwrap()
            .0
            .is_empty());
    }

    #[test]
    fn newer_builds_are_detected_by_number() {
        assert!(is_outdated("b9000", "b12360"));
        assert!(!is_outdated("b12360", "b12360"));
        assert!(!is_outdated("weird", "b1"));
    }

    #[test]
    fn rows_report_missing_then_ok_then_outdated() {
        let dir = tempfile::tempdir().unwrap();
        let none = BackendConfig::default();
        let rows = dependencies(
            dir.path(),
            &none,
            None,
            None,
            None,
            Uses::default(),
            Lang::En,
        );
        assert_eq!(rows[1].status, Status::Missing);
        assert_eq!(rows[1].fix, Fix::LlamaCpp);
        assert_eq!(rows[4].status, Status::Missing, "no workspace");

        let build = dir.path().join("llama.cpp").join("b10-vulkan");
        std::fs::create_dir_all(&build).unwrap();
        std::fs::write(build.join(install::server_name()), b"").unwrap();
        let manifest = install::Manifest {
            tag: "b10".into(),
            asset: "a".into(),
            variant: "vulkan".into(),
            sha256: "0".repeat(64),
            hash_source: install::HashSource::Published,
        };
        std::fs::write(
            build.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let cfg = BackendConfig {
            llama_cpp_dir: Some(build.clone()),
            ..Default::default()
        };
        let ws = dir.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let rows = dependencies(
            dir.path(),
            &cfg,
            Some(&ws),
            None,
            Some("b10"),
            Uses::default(),
            Lang::En,
        );
        assert_eq!(rows[1].status, Status::Ok);
        assert_eq!(rows[4].status, Status::Ok);
        let rows = dependencies(
            dir.path(),
            &cfg,
            Some(&ws),
            None,
            Some("b99"),
            Uses::default(),
            Lang::En,
        );
        assert_eq!(rows[1].status, Status::Outdated);
        std::fs::remove_file(build.join(install::server_name())).unwrap();
        let rows = dependencies(
            dir.path(),
            &cfg,
            Some(&ws),
            None,
            None,
            Uses::default(),
            Lang::En,
        );
        assert_eq!(
            rows[1].status,
            Status::Missing,
            "a deleted binary is reported"
        );
    }

    #[test]
    fn only_what_stops_the_work_is_blocking_and_choices_raise_optional_rows() {
        let dir = tempfile::tempdir().unwrap();
        let none = BackendConfig::default();
        let severity =
            |rows: &[Row], name: &str| rows.iter().find(|r| r.name == name).unwrap().severity;
        // A fresh profile: no workspace, nothing installed, nothing chosen.
        let rows = dependencies(
            dir.path(),
            &none,
            None,
            None,
            None,
            Uses::default(),
            Lang::En,
        );
        assert_eq!(severity(&rows, "workspace"), Severity::Blocking);
        assert_eq!(severity(&rows, "weight-surgeon"), Severity::Ok);
        for name in ["llama.cpp", "Vulkan", "Qwen2.5-0.5B-Instruct"] {
            assert_eq!(severity(&rows, name), Severity::Optional, "{name}");
        }
        assert_eq!(actions_needed(&rows), 1, "only the workspace");
        assert_eq!(
            rows.iter().filter(|r| r.group == Group::Essential).count(),
            2
        );

        // Claude Code configured and llama-server picked: their pieces become needed.
        let uses = Uses {
            llama_backend: true,
            cli: Some("Claude Code"),
        };
        let ws = dir.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let rows = dependencies(dir.path(), &none, Some(&ws), None, None, uses, Lang::En);
        assert_eq!(severity(&rows, "workspace"), Severity::Ok);
        assert_eq!(
            severity(&rows, "llama.cpp"),
            Severity::NeededBy("llama-server")
        );
        // No Vulkan build installed: a missing loader does not matter.
        assert_eq!(severity(&rows, "Vulkan"), Severity::Optional);
        let cli = severity(&rows, "surgeon (CLI)");
        // Tests run from target/, where the CLI may or may not sit next to the binary.
        assert!(
            matches!(cli, Severity::Ok | Severity::NeededBy("Claude Code")),
            "{cli:?}"
        );
    }

    #[test]
    fn the_export_holds_no_home_path() {
        let report = SystemReport {
            notes: vec![format!(
                "dir {}",
                crate::paths::home_dir().unwrap_or_default().display()
            )],
            ..Default::default()
        };
        let text = export(Some(&report), &[], Path::new("."), Lang::En);
        let home = crate::paths::home_dir()
            .unwrap_or_default()
            .display()
            .to_string();
        if home.len() > 3 {
            assert!(
                !text.to_lowercase().contains(&home.to_lowercase()),
                "{text}"
            );
        }
    }
}
