//! Input sources: the terminal (crossterm through ratatui) or a script for tests.

use std::collections::VecDeque;
use std::io;
use std::time::Duration;

use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

pub trait EventSource {
    /// Waits at most `timeout`; `None` means no input (a tick).
    fn poll(&mut self, timeout: Duration) -> io::Result<Option<Event>>;
}

/// The real terminal: `poll` blocks in the OS, so an idle TUI uses no CPU.
pub struct TerminalEvents;

impl EventSource for TerminalEvents {
    fn poll(&mut self, timeout: Duration) -> io::Result<Option<Event>> {
        match event::poll(timeout)? {
            true => event::read().map(Some),
            false => Ok(None),
        }
    }
}

/// Replays `Some(event)` / `None` (tick) in order, then sends `q` so a test can never hang.
pub struct ScriptedEventSource {
    script: VecDeque<Option<Event>>,
    pub polls: usize,
}

impl ScriptedEventSource {
    pub fn new(script: impl IntoIterator<Item = Option<Event>>) -> Self {
        Self {
            script: script.into_iter().collect(),
            polls: 0,
        }
    }

    pub fn keys(codes: impl IntoIterator<Item = KeyCode>) -> Self {
        Self::new(
            codes
                .into_iter()
                .map(|c| Some(Event::Key(KeyEvent::new(c, KeyModifiers::NONE)))),
        )
    }
}

impl EventSource for ScriptedEventSource {
    fn poll(&mut self, _timeout: Duration) -> io::Result<Option<Event>> {
        self.polls += 1;
        Ok(self.script.pop_front().unwrap_or_else(|| {
            Some(Event::Key(KeyEvent::new(
                KeyCode::Char('q'),
                KeyModifiers::NONE,
            )))
        }))
    }
}
