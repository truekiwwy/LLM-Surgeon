//! surgeon-gui (LLM Surgery Studio): the surgery card canvas and patch editor, and the
//! native 3D viewer of a weight graph (`surgeon graph --format json` or `ndjson`), followed
//! live when it is a stream of graph patches.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use eframe::egui;
use surgeon_gui::app::SurgeonApp;
use surgeon_gui::source;

#[derive(Debug, Parser)]
#[command(
    name = "surgeon-gui",
    version,
    about = "LLM Surgery Studio : canvas de cartes, éditeur de patchs et graphe des poids en 3D"
)]
struct Cli {
    /// graph.json, graph.ndjson (suivi en direct) ou `-` (NDJSON sur l'entrée standard).
    #[arg(value_name = "GRAPH")]
    graph: Option<PathBuf>,
    /// Dossier des projets (défaut : ~/surgeon-studio).
    #[arg(long, value_name = "DOSSIER")]
    workspace: Option<PathBuf>,
    /// Vérifie projets, copies, jobs, rejeu et journal sans fenêtre, puis quitte.
    #[arg(long)]
    self_test: bool,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    if cli.self_test {
        return match surgeon_gui::studio::selftest::run(&mut std::io::stdout()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("échec : {e}");
                ExitCode::FAILURE
            }
        };
    }
    let workspace = cli.workspace.unwrap_or_else(|| {
        let home = std::env::var_os("USERPROFILE").or_else(|| std::env::var_os("HOME"));
        home.map_or_else(|| PathBuf::from("."), PathBuf::from)
            .join("surgeon-studio")
    });
    // Read before opening the window: errors print normally.
    let input = match cli.graph.as_deref().map(source::open).transpose() {
        Ok(input) => input,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("surgeon-gui")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([960.0, 600.0]),
        ..Default::default()
    };
    let outcome = eframe::run_native(
        "surgeon-gui",
        options,
        Box::new(|cc| Ok(Box::new(SurgeonApp::new(&cc.egui_ctx, input, workspace)))),
    );
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("surgeon-gui : {e}");
            ExitCode::FAILURE
        }
    }
}
