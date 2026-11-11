//! surgeon-tui — inspect the spectral topology of LLM weights in the
//! terminal, and let AI coding agents drive the view over a local IPC bridge.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
use surgeon_tui::app::{self, App};
use surgeon_tui::ipc::IpcTarget;
use surgeon_tui::model::ModelIndex;
use surgeon_tui::terminal;

/// Command line of the inspector.
#[derive(Debug, Parser)]
#[command(
    name = "surgeon-tui",
    version,
    about = "Inspection interactive de la topologie spectrale des poids LLM (Safetensors / GGUF)"
)]
struct Cli {
    /// Model to inspect: a .safetensors file, a checkpoint directory or a .gguf file.
    #[arg(long, value_name = "PATH")]
    model: PathBuf,

    /// IPC endpoint for the agent bridge (default: platform socket / named pipe).
    #[arg(long, value_name = "ENDPOINT")]
    ipc_socket: Option<String>,

    /// Agent command template; {context} is replaced by the snippet path.
    #[arg(long, default_value = "claude {context}", value_name = "TEMPLATE")]
    agent_command: String,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let runtime = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(runtime) => runtime,
        Err(e) => {
            eprintln!("runtime tokio : {e}");
            return ExitCode::FAILURE;
        }
    };
    match runtime.block_on(run(cli)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

async fn run(cli: Cli) -> Result<(), surgeon_tui::app::AppError> {
    // Load before touching the terminal: errors print normally.
    let index = ModelIndex::open(&cli.model)
        .map_err(|e| surgeon_tui::app::AppError::Io(std::io::Error::other(e.to_string())))?;
    let target = cli
        .ipc_socket
        .as_deref()
        .map(IpcTarget::parse)
        .unwrap_or_else(IpcTarget::default_target);

    let mut terminal = terminal::init().map_err(surgeon_tui::app::AppError::Io)?;
    let app = App::new(index, cli.agent_command, target.display());
    let outcome = app::run(&mut terminal, app, target).await;
    terminal::cleanup(&mut terminal);
    outcome
}
