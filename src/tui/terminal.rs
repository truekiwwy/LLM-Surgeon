//! Terminal lifecycle and the main loop. The terminal is restored exactly once on every
//! exit path: normal quit, error (RAII guard), panic (ratatui's hook restores before the
//! previous hook prints, and the guard runs during unwinding).

use std::io;
use std::time::{Duration, Instant};

use ratatui::backend::{Backend, TestBackend};
use ratatui::crossterm::event::Event;
use ratatui::Terminal;

use super::app::{update, App, Effect, Msg};
use super::event::EventSource;
use super::tabs::preview::{PreviewEffect, PreviewMsg, PreviewWorker};
use super::ui::view;
use super::worker::Worker;
use super::TuiError;

/// 16 ms: a frame at 60 FPS. `poll` sleeps in the OS meanwhile.
pub const POLL: Duration = Duration::from_millis(16);

pub trait TerminalGuard {
    /// Leaves raw mode and the alternate screen.
    fn restore(&mut self) -> io::Result<()>;
}

/// The real terminal, entered with `ratatui::try_init` (raw mode, alternate screen, and a
/// panic hook that restores before chaining to the previous one).
pub struct RatatuiGuard;

impl TerminalGuard for RatatuiGuard {
    fn restore(&mut self) -> io::Result<()> {
        ratatui::try_restore()
    }
}

/// Calls `restore` once, when dropped.
pub struct Restore<'a> {
    guard: &'a mut dyn TerminalGuard,
}

impl<'a> Restore<'a> {
    pub fn new(guard: &'a mut dyn TerminalGuard) -> Self {
        Self { guard }
    }
}

impl Drop for Restore<'_> {
    fn drop(&mut self) {
        let _ = self.guard.restore();
    }
}

/// Counters of one run, for tests and metrics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoopStats {
    pub draws: usize,
    pub ticks: usize,
    pub draw_time: Duration,
}

fn run_effects(effects: Vec<Effect>, worker: Option<&Worker>, preview: Option<&PreviewWorker>) {
    for effect in effects {
        match (effect, worker) {
            (Effect::RequestStats(request), Some(worker)) => worker.request(request),
            (Effect::RequestStats(_), None) => {}
            (Effect::Preview(PreviewEffect::Generate(request)), _) => {
                if let Some(preview) = preview {
                    preview.request(request);
                }
            }
        }
    }
}

/// Draws only when the state is dirty; otherwise waits for input up to [`POLL`].
pub fn run_loop<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    events: &mut dyn EventSource,
    worker: Option<&Worker>,
) -> Result<LoopStats, TuiError> {
    run_loop_with(terminal, app, events, worker, None)
}

/// [`run_loop`] with the generation worker of the preview tab.
pub fn run_loop_with<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    events: &mut dyn EventSource,
    worker: Option<&Worker>,
    preview: Option<&PreviewWorker>,
) -> Result<LoopStats, TuiError> {
    let mut stats = LoopStats::default();
    run_effects(app.start(), worker, preview);
    loop {
        while let Some(reply) = worker.and_then(Worker::try_recv) {
            let msg = Msg::StatsReady {
                generation: reply.generation,
                key: reply.key,
                result: reply.result,
            };
            run_effects(update(app, msg), worker, preview);
        }
        while let Some(event) = preview.and_then(PreviewWorker::try_recv) {
            let msg = Msg::Preview(PreviewMsg::Worker(event));
            run_effects(update(app, msg), worker, preview);
        }
        if app.dirty {
            let started = Instant::now();
            terminal
                .draw(|frame| view(frame, app))
                .map_err(|e| TuiError::Terminal(e.to_string()))?;
            stats.draw_time += started.elapsed();
            stats.draws += 1;
            app.dirty = false;
        }
        if app.should_quit {
            return Ok(stats);
        }
        let msg = match events
            .poll(POLL)
            .map_err(|e| TuiError::Terminal(e.to_string()))?
        {
            Some(Event::Key(key)) => Msg::Key(key),
            Some(Event::Resize(width, height)) => Msg::Resize(width, height),
            Some(_) => continue,
            None => {
                stats.ticks += 1;
                Msg::Tick
            }
        };
        run_effects(update(app, msg), worker, preview);
    }
}

/// [`run_loop`] inside a guard that restores the terminal exactly once.
pub fn run_session<B: Backend>(
    guard: &mut dyn TerminalGuard,
    terminal: &mut Terminal<B>,
    app: &mut App,
    events: &mut dyn EventSource,
    worker: Option<&Worker>,
) -> Result<LoopStats, TuiError> {
    let _restore = Restore::new(guard);
    run_loop(terminal, app, events, worker)
}

/// [`run_session`] with the generation worker of the preview tab.
pub fn run_session_with<B: Backend>(
    guard: &mut dyn TerminalGuard,
    terminal: &mut Terminal<B>,
    app: &mut App,
    events: &mut dyn EventSource,
    worker: Option<&Worker>,
    preview: Option<&PreviewWorker>,
) -> Result<LoopStats, TuiError> {
    let _restore = Restore::new(guard);
    run_loop_with(terminal, app, events, worker, preview)
}

/// One frame as plain text (no ANSI at all): trailing spaces trimmed, one line per row.
pub fn render_snapshot(app: &App, cols: u16, rows: u16) -> Result<String, TuiError> {
    let mut terminal = Terminal::new(TestBackend::new(cols, rows))
        .map_err(|e| TuiError::Terminal(e.to_string()))?;
    terminal
        .draw(|frame| view(frame, app))
        .map_err(|e| TuiError::Terminal(e.to_string()))?;
    Ok(buffer_text(terminal.backend().buffer()))
}

pub fn buffer_text(buffer: &ratatui::buffer::Buffer) -> String {
    let width = buffer.area.width.max(1) as usize;
    buffer
        .content()
        .chunks(width)
        .map(|row| {
            let line: String = row.iter().map(|cell| cell.symbol()).collect();
            line.trim_end().to_owned() + "\n"
        })
        .collect()
}
