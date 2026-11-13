//! surgeon-gui: open a weight graph (`surgeon graph --format json` or `ndjson`) in a native
//! 3D viewer, or follow a stream of graph patches live.

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
    about = "Visualiseur natif du graphe des poids (colonne, constellation, diff, flux GraphPatch)"
)]
struct Cli {
    /// graph.json, graph.ndjson (suivi en direct) ou `-` (NDJSON sur l'entrée standard).
    #[arg(value_name = "GRAPH")]
    graph: Option<PathBuf>,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
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
        Box::new(|cc| Ok(Box::new(SurgeonApp::new(&cc.egui_ctx, input)))),
    );
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("surgeon-gui : {e}");
            ExitCode::FAILURE
        }
    }
}
