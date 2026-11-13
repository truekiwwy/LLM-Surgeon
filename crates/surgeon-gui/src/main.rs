//! surgeon-gui (LLM Surgery Studio): the desktop app. No argument opens the home screen (and
//! the first-run wizard on a fresh profile); `graph.json`, `graph.ndjson` and `-` still open
//! the 3D viewer directly; folders and `*.surgeon-studio.json` are accepted too.
//!
//! Release builds on Windows use the GUI subsystem (no console window); the process then
//! attaches to the parent console so that `--version`, `--help` and `--self-test` print.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use eframe::egui;
use surgeon_gui::app::SurgeonApp;
use surgeon_gui::cli::{self, Cli, Item};
use surgeon_gui::desktop::{Desktop, Screen};
use surgeon_gui::instance::{self, Role};
use surgeon_gui::settings::{Lang, Settings};
use surgeon_gui::source::Input;
use surgeon_gui::{actions, diag, logging, paths, raster, selftest, source};
use weight_surgeon::backend::config;

/// Reuses the console of the shell that launched the GUI-subsystem binary, if any.
#[cfg(all(windows, not(debug_assertions)))]
#[allow(unsafe_code)]
fn attach_parent_console() {
    use windows_sys::Win32::System::Console::{AttachConsole, ATTACH_PARENT_PROCESS};
    // SAFETY: no pointer is passed; failure (no parent console) is simply ignored.
    unsafe {
        AttachConsole(ATTACH_PARENT_PROCESS);
    }
}

#[cfg(not(all(windows, not(debug_assertions))))]
fn attach_parent_console() {}

const ICON: &[u8] = include_bytes!("../assets/icon.png");

fn window_icon() -> Option<egui::IconData> {
    let decoder = png::Decoder::new(std::io::Cursor::new(ICON));
    let mut reader = decoder.read_info().ok()?;
    let mut rgba = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut rgba).ok()?;
    rgba.truncate(info.buffer_size());
    Some(egui::IconData {
        rgba,
        width: info.width,
        height: info.height,
    })
}

fn main() -> ExitCode {
    attach_parent_console();
    let cli = Cli::parse();
    // Portable: the engine's trusted-toolchain record stays in data/ too, not in %APPDATA%.
    if paths::is_portable() && std::env::var_os(config::CONFIG_PATH_ENV).is_none() {
        std::env::set_var(
            config::CONFIG_PATH_ENV,
            paths::data_dir().join("config.toml"),
        );
    }
    logging::init(&paths::logs_dir());
    logging::info(&format!("surgeon-gui {} start", env!("CARGO_PKG_VERSION")));

    if cli.reset_settings {
        match Settings::reset(&paths::settings_path()) {
            Ok(()) => println!("réglages effacés : l'assistant se relancera au prochain démarrage"),
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
    }
    // The projects folder of the studio: the flag, then the settings, then the default.
    let chosen = cli
        .workspace
        .clone()
        .or_else(selftest::configured_workspace);
    if cli.self_test {
        return match selftest::run(chosen.as_deref()) {
            true => ExitCode::SUCCESS,
            false => ExitCode::FAILURE,
        };
    }
    if let Some(png) = &cli.screenshot {
        return match screenshot(&cli, png) {
            Ok(()) => {
                println!("{}", png.display());
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("{e}");
                ExitCode::FAILURE
            }
        };
    }

    let workspace = chosen.unwrap_or_else(paths::default_workspace);

    // A second launch hands its paths to the first and exits. `-` (stdin) cannot be handed
    // over, so such a launch always gets its own window.
    let stdin_input = cli.inputs.iter().any(|p| p.as_os_str() == "-");
    let inbox = match stdin_input {
        true => None,
        false => match instance::acquire(&paths::instance_path(), &absolute(&cli.inputs)) {
            Role::Forwarded => return ExitCode::SUCCESS,
            Role::Primary(inbox) => Some(inbox),
            Role::Alone => None,
        },
    };

    // The first graph opens the viewer at once; a failure shows on the home screen.
    let mut notice = None;
    let mut input: Option<Input> = None;
    let mut rest = Vec::new();
    for item in cli.inputs.iter().map(|p| cli::classify(p)) {
        match item {
            Item::Graph(path) if input.is_none() => match source::open(&path) {
                Ok(opened) => input = Some(opened),
                Err(e) => notice = Some(e),
            },
            other => rest.push(other),
        }
    }
    if let Some(e) = &notice {
        eprintln!("{e}");
    }
    let start = match input.is_some() {
        true => Screen::Workbench,
        false => Screen::Home,
    };
    let replay = cli.replay.clone();
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("LLM Surgery Studio")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([960.0, 600.0])
            .with_icon(window_icon().unwrap_or_default()),
        ..Default::default()
    };
    let outcome = eframe::run_native(
        "LLM Surgery Studio",
        options,
        Box::new(move |cc| {
            let mut desktop = Desktop::new(&cc.egui_ctx, inbox, start, false);
            if let Some(e) = notice {
                desktop.notify(e);
            }
            let mut app = SurgeonApp::with_shell(&cc.egui_ctx, input, desktop, workspace);
            app.open_items(&cc.egui_ctx, rest);
            if let Some(Err(e)) = replay.as_deref().map(|r| app.open_replay(r)) {
                logging::warn(&e);
                eprintln!("{e}");
            }
            Ok(Box::new(app))
        }),
    );
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            logging::warn(&format!("fenêtre : {e}"));
            eprintln!("surgeon-gui : {e}");
            ExitCode::FAILURE
        }
    }
}

/// Relative paths would not mean the same thing to the first instance (another cwd).
fn absolute(paths: &[PathBuf]) -> Vec<PathBuf> {
    paths
        .iter()
        .map(|p| std::path::absolute(p).unwrap_or_else(|_| p.clone()))
        .collect()
}

/// `--screenshot`: the current screen, rendered without a window or a GPU.
fn screenshot(cli: &Cli, png: &std::path::Path) -> Result<(), String> {
    let ctx = egui::Context::default();
    let screen = match cli.screen.as_str() {
        "home" | "wizard" => Screen::Home,
        "diagnostics" => Screen::Diagnostics,
        "agents" | "graph" => Screen::Workbench,
        other => {
            return Err(format!(
                "--screen {other} : home, wizard, diagnostics, agents ou graph"
            ))
        }
    };
    let mut settings = Settings::load(&paths::settings_path()).unwrap_or_default();
    settings.language.get_or_insert_with(Lang::system);
    let mut desktop = Desktop::detached(None, screen, settings, actions::Env::current());
    desktop.capture_system(diag::probe(&paths::data_dir()));
    if cli.screen == "wizard" {
        desktop.capture_wizard(diag::probe(&paths::data_dir()));
    }
    let input = match cli.inputs.first().map(|p| cli::classify(p)) {
        Some(Item::Graph(path)) => Some(source::open(&path)?),
        _ => None,
    };
    let workspace = cli
        .workspace
        .clone()
        .unwrap_or_else(paths::default_workspace);
    let mut app = SurgeonApp::with_shell(&ctx, input, desktop, workspace);
    if cli.screen == "agents" {
        app.open_agents();
    }
    if let Some(replay) = &cli.replay {
        app.open_replay(replay)?;
    }
    let frame = raster::render(&ctx, [1440, 900], 1.0, |ui| app.draw(ui));
    frame.image.save_png(png)
}
