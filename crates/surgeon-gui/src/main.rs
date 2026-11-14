//! surgeon-gui (Weight Surgeon): the desktop app, in a window without system frame. No
//! argument opens the Atelier (and the first-run wizard on a fresh profile); `graph.json`,
//! `graph.ndjson` and `-` open the 3D graph directly; folders and `*.surgeon-studio.json` are
//! accepted too. `--demo-data` fills every view with a throwaway workspace.
//!
//! Release builds on Windows use the GUI subsystem (no console window); the process then
//! attaches to the parent console so that `--version`, `--help` and `--self-test` print.
#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use eframe::egui;
use surgeon_gui::app::{Section, SurgeonApp, Tab};
use surgeon_gui::cli::{self, Cli, Item};
use surgeon_gui::desktop::Desktop;
use surgeon_gui::instance::{self, Role};
use surgeon_gui::settings::{Lang, Settings};
use surgeon_gui::source::Input;
use surgeon_gui::{actions, demo, diag, logging, paths, raster, selftest, source};
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

    // The first graph opens the 3D view at once; a failure shows in the status bar.
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
    let replay = cli.replay.clone();
    let [w, h] = cli.size;
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("Weight Surgeon")
            .with_decorations(false)
            .with_inner_size([w as f32, h as f32])
            .with_min_inner_size([960.0, 600.0])
            .with_icon(window_icon().unwrap_or_default()),
        ..Default::default()
    };
    let (demo_data, no_motion, scale) = (cli.demo_data, cli.no_motion, cli.scale);
    let tab = Tab::from_name(&cli.screen);
    let outcome = eframe::run_native(
        "Weight Surgeon",
        options,
        Box::new(move |cc| {
            let ctx = &cc.egui_ctx;
            let mut desktop = Desktop::new(ctx, inbox, false);
            if let Some(e) = notice {
                desktop.notify(e);
            }
            let mut app = SurgeonApp::with_shell(ctx, input, desktop, workspace);
            if demo_data {
                if let Err(e) = demo::install(ctx, &mut app) {
                    logging::warn(&e);
                    app.desktop.notify(e);
                }
            }
            if let Some(tab) = tab {
                app.tab = tab;
            }
            app.set_motion(ctx, !no_motion);
            ctx.set_zoom_factor(scale);
            app.open_items(ctx, rest);
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

/// `--screenshot`: one tab (or the wizard over the Atelier), rendered without a window or a
/// GPU at `--size` pixels and `--scale` pixels per point.
fn screenshot(cli: &Cli, png: &std::path::Path) -> Result<(), String> {
    let ctx = egui::Context::default();
    // `system:providers`: a tab and one of its parts.
    let (name, part) = cli
        .screen
        .split_once(':')
        .unwrap_or((cli.screen.as_str(), ""));
    let (tab, wizard) = match name {
        "wizard" => (Tab::Atelier, true),
        name => (
            Tab::from_name(name).ok_or_else(|| {
                format!("--screen {name} : atelier, graph, analyses, agents, system ou wizard")
            })?,
            false,
        ),
    };
    let section = match part {
        "" => None,
        part => Some(Section::from_name(part).ok_or_else(|| {
            format!("--screen system:{part} : machine, providers, updates ou journal")
        })?),
    };
    let mut settings = Settings::load(&paths::settings_path()).unwrap_or_default();
    settings.language.get_or_insert_with(Lang::system);
    let mut desktop = Desktop::detached(None, settings, actions::Env::current());
    desktop.capture_system(diag::probe(&paths::data_dir()));
    if wizard {
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
    if cli.demo_data {
        demo::install(&ctx, &mut app)?;
    }
    app.tab = tab;
    if let Some(section) = section {
        app.system_section = section;
        // One frame is rendered: the scan runs before it, not behind it.
        app.agents.scan = Some(surgeon_gui::agent::detect::scan());
    }
    // A capture shows the tab, not the first-run backend dialog over it.
    app.dialog.open = false;
    app.set_motion(&ctx, !cli.no_motion);
    if let Some(replay) = &cli.replay {
        app.open_replay(replay)?;
    }
    let frame = raster::render(&ctx, cli.size, cli.scale, |ui| app.draw(ui));
    frame.image.save_png(png)
}
