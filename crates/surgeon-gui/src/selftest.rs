//! `surgeon-gui --self-test`: proves the install works without anyone looking at a window.
//! Engine (the `weight_surgeon` library the GUI links, run on a bundled graph stream), a real
//! graphics initialisation (an invisible OpenGL window, closed after one frame), and a write
//! test of the data and workspace folders. Prints one line per check and exits 0 or 1.

use std::path::Path;
use std::sync::{Arc, Mutex};

use eframe::egui;

use crate::settings::Settings;
use crate::source::Timeline;
use weight_surgeon::graph::stream::GraphPatch;

pub struct Check {
    pub name: &'static str,
    pub ok: bool,
    pub detail: String,
}

fn check(name: &'static str, result: Result<String, String>) -> Check {
    match result {
        Ok(detail) => Check {
            name,
            ok: true,
            detail,
        },
        Err(detail) => Check {
            name,
            ok: false,
            detail,
        },
    }
}

/// A graph stream shipped with the crate, replayed through the engine's patch code.
const FIXTURE: &str = include_str!("../tests/fixtures/qwen2-mini.ndjson");

fn engine() -> Result<String, String> {
    let mut timeline = Timeline::default();
    let mut patches = 0;
    for line in FIXTURE.lines().filter(|l| !l.trim().is_empty()) {
        let patch: GraphPatch =
            serde_json::from_str(line).map_err(|e| format!("flux embarqué illisible : {e}"))?;
        timeline.push(&patch)?;
        patches += 1;
    }
    let step = timeline.current().ok_or("aucun état")?;
    Ok(format!(
        "weight-surgeon {} : {patches} patchs rejoués, {} nœuds",
        env!("CARGO_PKG_VERSION"),
        step.report.nodes.len()
    ))
}

/// The studio foundation (projects, copies, jobs, replay, ledger) on a throwaway workspace.
fn studio() -> Result<String, String> {
    let mut report = Vec::new();
    crate::studio::selftest::run(&mut report)?;
    let lines = String::from_utf8_lossy(&report).lines().count();
    Ok(format!("{lines} vérifications"))
}

/// Create, write, read back and delete a file in `dir`.
fn writable(dir: &Path) -> Result<String, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("{} : {e}", dir.display()))?;
    let probe = dir.join(format!(".selftest-{}", std::process::id()));
    std::fs::write(&probe, b"surgeon").map_err(|e| format!("{} : {e}", dir.display()))?;
    let back = std::fs::read(&probe).map_err(|e| format!("{} : {e}", probe.display()))?;
    let _ = std::fs::remove_file(&probe);
    match back == b"surgeon" {
        true => Ok(format!("{} inscriptible", dir.display())),
        false => Err(format!("{} : relecture différente", dir.display())),
    }
}

struct Probe {
    renderer: Arc<Mutex<Option<String>>>,
}

impl eframe::App for Probe {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        if let Some(gl) = frame.gl() {
            *self.renderer.lock().unwrap_or_else(|e| e.into_inner()) = Some(renderer_name(gl));
        }
        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
    }
}

#[allow(unsafe_code)]
fn renderer_name(gl: &eframe::glow::Context) -> String {
    use eframe::glow::HasContext;
    // SAFETY: a plain string query on the context eframe just created and made current.
    unsafe { gl.get_parameter_string(eframe::glow::RENDERER) }
}

/// Opens an invisible window with an OpenGL context for one frame.
fn graphics() -> Result<String, String> {
    let renderer = Arc::new(Mutex::new(None));
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_visible(false)
            .with_inner_size([64.0, 64.0]),
        ..Default::default()
    };
    let probe = Probe {
        renderer: renderer.clone(),
    };
    eframe::run_native(
        "surgeon-gui self-test",
        options,
        Box::new(move |_| Ok(Box::new(probe))),
    )
    .map_err(|e| format!("aucun contexte OpenGL : {e}"))?;
    let name = renderer.lock().unwrap_or_else(|e| e.into_inner()).take();
    match name {
        Some(name) if is_software(&name) => {
            Ok(format!("{name} (rendu logiciel : lent mais utilisable)"))
        }
        Some(name) => Ok(name),
        None => Err("la fenêtre s'est ouverte sans contexte OpenGL".into()),
    }
}

/// Renderer strings of the CPU implementations Windows and Mesa fall back to.
pub fn is_software(renderer: &str) -> bool {
    let r = renderer.to_ascii_lowercase();
    [
        "gdi generic",
        "llvmpipe",
        "softpipe",
        "swrast",
        "basic render",
    ]
    .iter()
    .any(|s| r.contains(s))
}

/// The color pairs of the theme against their WCAG thresholds.
fn interface() -> Result<String, String> {
    let failures = crate::ui::theme::contrast_failures();
    match failures.is_empty() {
        true => Ok(format!(
            "{} paires de couleurs au-dessus de leur seuil WCAG",
            crate::ui::theme::pairs().len()
        )),
        false => Err(failures.join(" ; ")),
    }
}

/// Runs every check and prints the report; `true` when all passed.
pub fn run(workspace: Option<&Path>) -> bool {
    let mut checks = vec![
        check("moteur", engine()),
        check("studio", studio()),
        check("agent", crate::agent::selftest::run()),
        check("données", writable(&crate::paths::data_dir())),
        check("interface", interface()),
    ];
    if let Some(dir) = workspace {
        checks.push(check("workspace", writable(dir)));
    }
    checks.push(check("graphique", graphics()));
    let all_ok = checks.iter().all(|c| c.ok);
    for c in &checks {
        println!(
            "[{}] {:<10} {}",
            if c.ok { " OK " } else { "FAIL" },
            c.name,
            c.detail
        );
    }
    println!(
        "surgeon-gui {} : {}",
        env!("CARGO_PKG_VERSION"),
        if all_ok {
            "self-test OK"
        } else {
            "self-test ECHEC"
        }
    );
    all_ok
}

/// The workspace folder the user configured, when there is one.
pub fn configured_workspace() -> Option<std::path::PathBuf> {
    Settings::load(&crate::paths::settings_path())
        .ok()?
        .workspace
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_engine_check_replays_the_bundled_stream() {
        let line = engine().unwrap();
        assert!(line.contains("patchs rejoués"), "{line}");
    }

    #[test]
    fn the_studio_check_runs_on_a_throwaway_workspace() {
        let line = studio().unwrap();
        assert!(line.contains("vérifications"), "{line}");
    }

    #[test]
    fn the_agent_check_runs_a_scripted_dialogue_on_the_engine() {
        let line = crate::agent::selftest::run().unwrap();
        assert!(line.contains("origine intacte"), "{line}");
    }

    #[test]
    fn a_writable_folder_passes_and_a_file_in_the_way_fails() {
        let dir = tempfile::tempdir().unwrap();
        assert!(writable(&dir.path().join("a")).is_ok());
        let file = dir.path().join("f");
        std::fs::write(&file, b"x").unwrap();
        assert!(writable(&file).is_err());
    }

    #[test]
    fn software_renderers_are_recognised() {
        assert!(is_software("GDI Generic"));
        assert!(is_software("llvmpipe (LLVM 15.0.7, 256 bits)"));
        assert!(!is_software("AMD Radeon RX 7600 XT"));
    }
}
