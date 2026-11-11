//! Terminal lifecycle: raw mode + alternate screen on entry, clean restore on
//! exit, and suspension around child processes (the agent launcher must run on
//! the real screen).

use std::io::{stdout, Stdout};

use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::ExecutableCommand;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

pub type Tui = Terminal<CrosstermBackend<Stdout>>;

/// Enters raw mode + alternate screen and builds the terminal.
pub fn init() -> Result<Tui, std::io::Error> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;
    Ok(terminal)
}

/// Restores the terminal fully.
pub fn cleanup(terminal: &mut Tui) {
    let _ = terminal.clear();
    let _ = terminal.show_cursor();
    let _ = stdout().execute(LeaveAlternateScreen);
    let _ = disable_raw_mode();
}

/// Leaves the alternate screen so a child process owns the real terminal.
pub fn suspend(terminal: &mut Tui) -> Result<(), std::io::Error> {
    terminal.clear()?;
    terminal.show_cursor()?;
    stdout().execute(LeaveAlternateScreen)?;
    disable_raw_mode()
}

/// Returns to the TUI after a suspended child process.
pub fn resume(terminal: &mut Tui) -> Result<(), std::io::Error> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    terminal.clear()
}
