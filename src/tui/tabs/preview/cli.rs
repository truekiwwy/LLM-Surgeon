//! `surgeon tui --preview`: the whole command, so that `tui::run_cli` only delegates.
//! `--snapshot` generates synchronously (no thread, no timing) and prints the plain text
//! frame; the interactive session runs the model in a [`PreviewWorker`] thread.

use std::io::{IsTerminal, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use super::worker::LoadError;
use super::{
    update, Event, Loader, PreviewEffect, PreviewMsg, PreviewState, PreviewWorker, SystemClock,
};
use crate::app::{exit, AppDeps};
use crate::cli::TuiArgs;
use crate::tui::app::Tab;
use crate::tui::preview_args::missing_inputs;
use crate::tui::terminal::run_session_with;
use crate::tui::{
    load, render_snapshot, App, RatatuiGuard, StatsProvider, TerminalEvents, Theme, TuiError,
    Worker, CACHE_CAPACITY, EXIT_NOT_INTERACTIVE,
};

pub fn run(
    args: &TuiArgs,
    deps: &AppDeps,
    theme: Theme,
    out: &mut dyn Write,
    err: &mut dyn Write,
) -> i32 {
    if let Some(message) = missing_inputs(args) {
        let _ = writeln!(err, "erreur : {message}");
        return exit::INVALID_INPUT;
    }
    let size = match args.snapshot {
        true => args.size,
        false => (0, 0),
    };
    let prepared = load(args, theme, size).and_then(|(mut app, provider)| {
        let vector = app
            .vector
            .as_ref()
            .ok_or_else(|| TuiError::InvalidArgument("--preview nécessite --vector".into()))?;
        let launch = super::prepare(args, vector, &*deps.probe, Arc::new(SystemClock::new()))?;
        app.preview = Some(launch.state);
        app.tab = Tab::Preview;
        Ok((app, provider, launch.loader))
    });
    let (app, provider, loader) = match prepared {
        Ok(prepared) => prepared,
        Err(e) => return crate::tui::report(err, &e),
    };
    match args.snapshot {
        true => match snapshot(app, loader, args.size) {
            Ok(text) => {
                let _ = out.write_all(text.as_bytes());
                let _ = out.flush();
                exit::OK
            }
            Err(e) => crate::tui::report(err, &e),
        },
        false => interactive(app, provider, loader, deps, err),
    }
}

/// Loads the model on this thread and answers the first request in place.
fn snapshot(
    mut app: App,
    loader: Option<Loader>,
    (cols, rows): (u16, u16),
) -> Result<String, TuiError> {
    let path = app
        .model
        .as_ref()
        .map(|m| m.path.clone())
        .unwrap_or_default();
    if let (Some(state), Some(loader)) = (app.preview.as_mut(), loader) {
        generate_in_place(state, loader).map_err(|message| TuiError::Model { path, message })?;
    }
    render_snapshot(&app, cols, rows)
}

fn interactive(
    app: App,
    provider: Option<Arc<dyn StatsProvider>>,
    loader: Option<Loader>,
    deps: &AppDeps,
    err: &mut dyn Write,
) -> i32 {
    // stderr too: see `tui::run_cli`.
    let tty =
        deps.ui.err_is_tty && std::io::stdin().is_terminal() && std::io::stdout().is_terminal();
    if !tty {
        let _ = writeln!(
            err,
            "erreur : `surgeon tui` nécessite un terminal interactif ; utilisez --snapshot pour un rendu texte"
        );
        return EXIT_NOT_INTERACTIVE;
    }
    match catch_unwind(AssertUnwindSafe(|| session(app, provider, loader))) {
        Ok(Ok(())) => exit::OK,
        Ok(Err(e)) => crate::tui::report(err, &e),
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

fn session(
    mut app: App,
    provider: Option<Arc<dyn StatsProvider>>,
    loader: Option<Loader>,
) -> Result<(), TuiError> {
    let mut terminal = ratatui::try_init().map_err(|e| TuiError::Terminal(e.to_string()))?;
    let mut guard = RatatuiGuard;
    let worker = provider.map(|p| Worker::spawn(p, CACHE_CAPACITY));
    let preview = loader.map(PreviewWorker::spawn);
    if let Ok(size) = terminal.size() {
        app.size = (size.width, size.height);
    }
    run_session_with(
        &mut guard,
        &mut terminal,
        &mut app,
        &mut TerminalEvents,
        worker.as_ref(),
        preview.as_ref(),
    )
    .map(|_| ())
}

/// Synchronous twin of the worker thread: loads, then serves the first request.
/// `Err` is a model that cannot work at all; a machine that is too small only
/// puts a message in the tab.
pub fn generate_in_place(state: &mut PreviewState, loader: Loader) -> Result<(), String> {
    let mut session = match loader() {
        Ok(session) => session.untimed(),
        Err(LoadError::Insufficient(message)) => {
            update(state, PreviewMsg::Worker(Event::LoadFailed(message)));
            return Ok(());
        }
        Err(LoadError::Failed(message)) => return Err(message),
    };
    let warnings = session.warnings().to_vec();
    update(state, PreviewMsg::Worker(Event::Loaded { warnings }));
    let never = AtomicBool::new(false);
    for PreviewEffect::Generate(request) in state.start() {
        session.run(&request, &never, &mut |event| {
            update(state, PreviewMsg::Worker(event));
        });
    }
    Ok(())
}
