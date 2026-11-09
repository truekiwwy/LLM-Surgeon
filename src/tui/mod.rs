//! `surgeon tui`: terminal dashboard (ratatui) to inspect MLP weights and steering vectors.
//!
//! Exit codes: 0 normal exit (or `--snapshot` written); 1 not an interactive terminal;
//! 2 invalid arguments or files (including `--json`, which the TUI does not support);
//! 70 internal error (a panic, reported on stderr after the terminal is restored).
//!
//! (a) Call graph (graphify, checked by hand):
//!
//! ```text
//! main -> app::run_with -> tui::run_cli
//!   -> ModelStatsProvider::open (hardware::compute_model_size, engine::ReadOnlyModel per shard)
//!   -> VectorView::load (engine::load_direction_vector, header metadata, <vector>.stats.json)
//!   -> App::new
//!   --snapshot: provider.tensor_stats (synchronous) -> terminal::render_snapshot (TestBackend)
//!   interactive: ratatui::try_init -> terminal::run_session (Restore guard)
//!     -> run_loop: Worker::try_recv -> update(StatsReady)
//!                  app.dirty ? Terminal::draw(view) : nothing
//!                  EventSource::poll(16 ms) -> update(Key | Resize | Tick) -> effects
//!                  Effect::RequestStats -> Worker::request (cancels the previous job)
//!     Worker thread -> LRU cache | StatsProvider::tensor_stats -> engine::stats (rayon)
//! ```
//!
//! (b) State machine (`app.rs`):
//!
//! ```text
//! Tab: Inspector <-> Steering                      (Tab / Shift+Tab)
//! Selection { layer, proj }: arrows, PgUp/PgDn, Home/End; absent projections skipped
//! Computation: Idle --select--> Running{generation} --StatsReady(same gen)--> Done(result)
//!              Running{g} --select--> Running{g+1} (job g cancelled, its reply dropped)
//! Msg: Key | Resize | Tick | StatsReady{generation, key, result}
//! Effect: RequestStats{generation, key, tensor, bins}
//! dirty <- key, resize, accepted reply, blink flip (alert shown), spinner step (running)
//! ```

pub mod app;
pub mod event;
pub mod preview_args;
pub mod tabs;
pub mod terminal;
pub mod theme;
pub mod ui;
pub mod widgets;
pub mod worker;

use std::io::{IsTerminal, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub use app::{
    handle_key, update, App, Computation, Effect, ModelView, Msg, Selection, Tab, VectorView,
};
pub use event::{EventSource, ScriptedEventSource, TerminalEvents};
pub use terminal::{
    render_snapshot, run_loop, run_session, LoopStats, RatatuiGuard, Restore, TerminalGuard,
};
pub use theme::Theme;
pub use ui::view;
pub use worker::{
    Lru, ModelStatsProvider, StatsProvider, StatsReply, StatsRequest, TensorEntry, Worker,
    CACHE_CAPACITY,
};

use crate::app::{exit, AppDeps};
use crate::cli::TuiArgs;

pub const EXIT_NOT_INTERACTIVE: i32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum TuiError {
    #[error("argument invalide : {0}")]
    InvalidArgument(String),
    #[error("modèle {} : {message}", .path.display())]
    Model { path: PathBuf, message: String },
    #[error("vecteur {} : {message}", .path.display())]
    Vector { path: PathBuf, message: String },
    #[error("statistiques : {0}")]
    Stats(String),
    #[error("terminal : {0}")]
    Terminal(String),
}

/// `COLSxROWS`, e.g. `100x32`.
pub fn parse_size(value: &str) -> Result<(u16, u16), String> {
    let invalid = || format!("taille « {value} » invalide : attendu COLSxROWS, par exemple 100x32");
    let (cols, rows) = value.trim().split_once(['x', 'X']).ok_or_else(invalid)?;
    let cols: u16 = cols.trim().parse().map_err(|_| invalid())?;
    let rows: u16 = rows.trim().parse().map_err(|_| invalid())?;
    match cols > 0 && rows > 0 {
        true => Ok((cols, rows)),
        false => Err(invalid()),
    }
}

/// Loads the inputs: invalid paths or corrupt files fail here, before any terminal change.
pub fn load(
    args: &TuiArgs,
    theme: Theme,
    size: (u16, u16),
) -> Result<(App, Option<Arc<dyn StatsProvider>>), TuiError> {
    let provider: Option<Arc<dyn StatsProvider>> = match &args.model {
        Some(path) => Some(Arc::new(ModelStatsProvider::open(path)?)),
        None => None,
    };
    let model = match (&args.model, &provider) {
        (Some(path), Some(p)) => Some(ModelView::from_entries(path, &p.list_tensors())),
        _ => None,
    };
    let vector = args.vector.as_deref().map(VectorView::load).transpose()?;
    Ok((App::new(model, vector, theme, size), provider))
}

/// Entry point used by `run_with`. Nothing is written to `out` except a `--snapshot`.
pub fn run_cli(
    args: &TuiArgs,
    json: bool,
    deps: &AppDeps,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    if json {
        let _ = writeln!(
            err,
            "erreur : `surgeon tui` n'a pas de sortie JSON (retirer --json)"
        );
        return exit::INVALID_INPUT;
    }
    let theme = Theme::new(deps.ui.out);
    if args.snapshot {
        return match snapshot(args, theme) {
            Ok(text) => {
                let _ = out.write_all(text.as_bytes());
                let _ = out.flush();
                exit::OK
            }
            Err(e) => report(err, &e),
        };
    }
    let (app, provider) = match load(args, theme, (0, 0)) {
        Ok(loaded) => loaded,
        Err(e) => return report(err, &e),
    };
    // stderr too: the injected writers of tests are never terminals, and an interactive
    // session must never start on a terminal that `err` does not belong to.
    let interactive =
        deps.ui.err_is_tty && std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if !interactive {
        let _ = writeln!(
            err,
            "erreur : `surgeon tui` nécessite un terminal interactif ; utilisez --snapshot pour un rendu texte"
        );
        return EXIT_NOT_INTERACTIVE;
    }
    match catch_unwind(AssertUnwindSafe(|| interactive_session(app, provider))) {
        Ok(Ok(())) => exit::OK,
        Ok(Err(e)) => report(err, &e),
        Err(payload) => {
            let _ = writeln!(
                err,
                "erreur interne : {}",
                crate::app::panic_message(payload.as_ref())
            );
            exit::INTERNAL
        }
    }
}

fn report(err: &mut dyn Write, e: &TuiError) -> i32 {
    let _ = writeln!(err, "erreur : {e}");
    match e {
        TuiError::Terminal(_) => exit::INTERNAL,
        _ => exit::INVALID_INPUT,
    }
}

/// The selected tensor is computed synchronously (there is no loop to receive it).
pub fn snapshot(args: &TuiArgs, theme: Theme) -> Result<String, TuiError> {
    let (cols, rows) = args.size;
    let (mut app, provider) = load(args, theme, (cols, rows))?;
    if let (Some(provider), Some(entry)) = (&provider, app.current_entry().cloned()) {
        let never = AtomicBool::new(false);
        let result = provider
            .tensor_stats(&entry.name, app.bins, &never)
            .map_err(|e| e.to_string());
        let key = (app.selection.layer, app.selection.proj);
        if let Ok(stats) = &result {
            app.flagged.insert(key, stats.non_finite > 0);
        }
        app.computation = Computation::Done(result);
    }
    render_snapshot(&app, cols, rows)
}

fn interactive_session(
    mut app: App,
    provider: Option<Arc<dyn StatsProvider>>,
) -> Result<(), TuiError> {
    let mut terminal = ratatui::try_init().map_err(|e| TuiError::Terminal(e.to_string()))?;
    let mut guard = RatatuiGuard;
    let worker = provider.map(|p| Worker::spawn(p, CACHE_CAPACITY));
    if let Ok(size) = terminal.size() {
        app.size = (size.width, size.height);
    }
    run_session(
        &mut guard,
        &mut terminal,
        &mut app,
        &mut TerminalEvents,
        worker.as_ref(),
    )
    .map(|_| ())
}
