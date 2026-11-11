//! Application state and the async event loop.
//!
//! ```text
//! tokio select! over
//!   - crossterm EventStream (keys)   -> AppEvent::Key
//!   - bridge::serve (agent IPC)      -> AppEvent::Ipc
//!   - spawn_blocking metrics jobs    -> AppEvent::Metrics
//!   - 60 Hz tick                     -> always redraw
//! ```
//! Metrics are computed off the UI thread (tokio blocking pool, which wraps
//! Rayon-sized work) on a deterministic sample, cached by tensor index; the
//! loop schedules the selected tensor and its neighbors so scrolling stays
//! instant.

use std::collections::{HashMap, HashSet, VecDeque};
use std::io::Stdout;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crossterm::event::{Event, EventStream, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use futures_util::StreamExt;
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use thiserror::Error;
use tokio::sync::mpsc;

use crate::context::{agent_argv, launch_agent, ContextInfo};
use crate::ipc::{IpcCommand, IpcRequest, IpcTarget, Snapshot};
use crate::model::metrics::{self, Stats};
use crate::model::roles::{Exactness, Role};
use crate::model::{ModelIndex, TensorEntry};
use crate::ui;

/// Events the loop consumes.
pub enum AppEvent {
    Key(KeyEvent),
    Ipc(IpcRequest),
    Metrics { index: usize, stats: Stats },
    AgentDone { detail: String },
}

/// Which panel owns the navigation keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Focus {
    Layers,
    Inspector,
    Bridge,
}

impl Focus {
    pub fn next(self) -> Focus {
        match self {
            Focus::Layers => Focus::Inspector,
            Focus::Inspector => Focus::Bridge,
            Focus::Bridge => Focus::Layers,
        }
    }
}

/// A patch an agent is previewing.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchPreview {
    pub tensor: String,
    pub alpha: f64,
    pub at: Instant,
}

#[derive(Debug, Error)]
pub enum AppError {
    #[error("io : {0}")]
    Io(#[from] std::io::Error),
    #[error("rendu : {0}")]
    Render(String),
}

/// The whole UI state.
pub struct App {
    pub index: ModelIndex,
    /// Indices into `tensors` matching both filters, in display order.
    pub filtered: Vec<usize>,
    /// Cursor position inside `filtered`.
    pub sel: usize,
    pub role_filter: Option<Role>,
    pub text_filter: String,
    pub searching: bool,
    pub focus: Focus,
    pub stats: HashMap<usize, Stats>,
    requested: HashSet<usize>,
    pub log: VecDeque<String>,
    pub highlight: Option<(u32, Instant)>,
    pub preview: Option<PatchPreview>,
    pub agent_command: String,
    pub ipc_display: String,
    pub quit: bool,
}

impl App {
    pub fn new(index: ModelIndex, agent_command: String, ipc_display: String) -> App {
        let mut app = App {
            filtered: (0..index.tensors.len()).collect(),
            index,
            sel: 0,
            role_filter: None,
            text_filter: String::new(),
            searching: false,
            focus: Focus::Layers,
            stats: HashMap::new(),
            requested: HashSet::new(),
            log: VecDeque::new(),
            highlight: None,
            preview: None,
            agent_command,
            ipc_display,
            quit: false,
        };
        app.log("prêt — Ctrl+E exporte le contexte, Ctrl+P lance l'agent, / filtre");
        app
    }

    pub fn log(&mut self, line: impl Into<String>) {
        self.log.push_back(format!("[{}] {}", stamp(), line.into()));
        while self.log.len() > 400 {
            self.log.pop_front();
        }
    }

    /// The tensor under the cursor.
    pub fn selected(&self) -> Option<&TensorEntry> {
        self.filtered.get(self.sel).and_then(|i| self.index.tensors.get(*i))
    }

    /// Selected index in `tensors`, if any.
    pub fn selected_index(&self) -> Option<usize> {
        self.filtered.get(self.sel).copied()
    }

    /// Reapplies both filters, keeping the selection when it stays visible.
    pub fn refilter(&mut self) {
        let text = self.text_filter.to_lowercase();
        let previous = self.selected_index();
        self.filtered = self
            .index
            .tensors
            .iter()
            .enumerate()
            .filter(|(_, t)| self.role_filter.is_none_or(|r| t.role == r))
            .filter(|(_, t)| text.is_empty() || t.name.to_lowercase().contains(&text))
            .map(|(i, _)| i)
            .collect();
        self.sel = previous
            .and_then(|p| self.filtered.iter().position(|i| *i == p))
            .unwrap_or(0);
        self.clamp_sel();
    }

    fn clamp_sel(&mut self) {
        self.sel = self.sel.min(self.filtered.len().saturating_sub(1));
    }

    pub fn move_sel(&mut self, delta: isize) {
        if self.filtered.is_empty() {
            return;
        }
        let target = self.sel as isize + delta;
        self.sel = target.clamp(0, self.filtered.len() as isize - 1) as usize;
    }

    pub fn jump_top(&mut self) {
        self.sel = 0;
    }

    pub fn jump_bottom(&mut self) {
        self.clamp_sel();
    }

    /// Cycles the role filter: none -> embed -> ... -> router -> none.
    pub fn cycle_role_filter(&mut self) {
        let present = self.index.roles_present();
        let candidates: Vec<Role> = present.into_iter().filter(|r| *r != Role::Other).collect();
        self.role_filter = match self.role_filter {
            None => candidates.first().copied(),
            Some(current) => {
                let position = candidates.iter().position(|r| *r == current);
                match position {
                    Some(p) if p + 1 < candidates.len() => Some(candidates[p + 1]),
                    _ => None,
                }
            }
        };
        self.refilter();
        self.log(match self.role_filter {
            Some(r) => format!("filtre rôle : {r}"),
            None => "filtre rôle : aucun".to_owned(),
        });
    }

    pub fn set_role_filter(&mut self, role: Option<Role>) {
        self.role_filter = role;
        self.refilter();
    }

    /// Handles one key; `launch_requested` reports Ctrl+P to the loop.
    pub fn on_key(&mut self, key: KeyEvent) -> Control {
        if key.kind != KeyEventKind::Press {
            return Control::Continue;
        }
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('c') => {
                    self.quit = true;
                    return Control::Quit;
                }
                KeyCode::Char('e') => {
                    self.export_context();
                    return Control::Continue;
                }
                KeyCode::Char('p') => {
                    return Control::LaunchAgent;
                }
                _ => {}
            }
        }
        if self.searching {
            return self.on_search_key(key);
        }
        match key.code {
            KeyCode::Char('q') => {
                self.quit = true;
                Control::Quit
            }
            KeyCode::Char('/') => {
                self.searching = true;
                self.focus = Focus::Layers;
                Control::Continue
            }
            KeyCode::Tab => {
                self.focus = self.focus.next();
                Control::Continue
            }
            KeyCode::Esc => {
                if self.preview.is_some() {
                    self.preview = None;
                    self.log("aperçu patch effacé");
                } else if self.role_filter.is_some() || !self.text_filter.is_empty() {
                    self.text_filter.clear();
                    self.role_filter = None;
                    self.refilter();
                    self.log("filtres effacés");
                } else {
                    self.quit = true;
                    return Control::Quit;
                }
                Control::Continue
            }
            KeyCode::Char('r') => {
                self.cycle_role_filter();
                Control::Continue
            }
            KeyCode::Up | KeyCode::Char('k') => {
                self.move_sel(-1);
                Control::Continue
            }
            KeyCode::Down | KeyCode::Char('j') => {
                self.move_sel(1);
                Control::Continue
            }
            KeyCode::PageUp => {
                self.move_sel(-10);
                Control::Continue
            }
            KeyCode::PageDown => {
                self.move_sel(10);
                Control::Continue
            }
            KeyCode::Home | KeyCode::Char('g') => {
                self.jump_top();
                Control::Continue
            }
            KeyCode::End | KeyCode::Char('G') => {
                self.jump_bottom();
                Control::Continue
            }
            KeyCode::Enter => {
                if self.focus == Focus::Layers {
                    self.log("tenseur épinglé pour l'inspecteur");
                }
                Control::Continue
            }
            _ => Control::Continue,
        }
    }

    fn on_search_key(&mut self, key: KeyEvent) -> Control {
        match key.code {
            KeyCode::Esc => {
                self.searching = false;
            }
            KeyCode::Enter => {
                self.searching = false;
                self.jump_top();
            }
            KeyCode::Backspace => {
                self.text_filter.pop();
                self.refilter();
            }
            KeyCode::Char(c) => {
                self.text_filter.push(c);
                self.refilter();
                self.jump_top();
            }
            _ => {}
        }
        Control::Continue
    }

    /// Applies one agent command received over the bridge.
    pub fn on_ipc(&mut self, request: IpcRequest) {
        let IpcRequest { command, reply } = request;
        match command {
            IpcCommand::Ping => self.log("agent : ping"),
            IpcCommand::HighlightLayer { layer } => {
                let layer_u32 = u32::try_from(layer).unwrap_or(u32::MAX);
                self.highlight = Some((layer_u32, Instant::now()));
                if let Some(position) = self
                    .filtered
                    .iter()
                    .position(|i| self.index.tensors[*i].layer == Some(layer_u32))
                {
                    self.sel = position;
                } else {
                    self.set_role_filter(None);
                    self.text_filter.clear();
                    self.refilter();
                    if let Some(position) = self
                        .filtered
                        .iter()
                        .position(|i| self.index.tensors[*i].layer == Some(layer_u32))
                    {
                        self.sel = position;
                    }
                }
                self.log(format!("agent : couche L{layer} mise en évidence"));
            }
            IpcCommand::SelectTensor { name } => match self.index.index_of(&name) {
                Some(index) => {
                    self.jump_to(index);
                    self.log(format!("agent : sélection {name}"));
                }
                None => self.log(format!("agent : tenseur inconnu {name}")),
            },
            IpcCommand::PreviewPatch { tensor, alpha } => {
                if let Some(index) = self.index.index_of(&tensor) {
                    self.jump_to(index);
                }
                self.preview = Some(PatchPreview {
                    tensor: tensor.clone(),
                    alpha,
                    at: Instant::now(),
                });
                self.log(format!("agent : aperçu {tensor} (α = {alpha:.4})"));
            }
            IpcCommand::ClearPreview => {
                self.preview = None;
                self.log("agent : aperçu effacé");
            }
            IpcCommand::SetFilter { role } => match role.as_deref().map(Role::from_filter) {
                None => {
                    self.set_role_filter(None);
                    self.log("agent : filtre désactivé");
                }
                Some(Some(r)) => {
                    self.set_role_filter(Some(r));
                    self.log(format!("agent : filtre {r}"));
                }
                Some(None) => self.log(format!(
                    "agent : rôle inconnu {:?}",
                    role.as_deref().unwrap_or("")
                )),
            },
            IpcCommand::Snapshot => {
                if let Some(sender) = reply {
                    let _ = sender.send(self.snapshot().to_value());
                }
            }
        }
    }

    fn jump_to(&mut self, tensor_index: usize) {
        if !self.filtered.contains(&tensor_index) {
            self.text_filter.clear();
            self.set_role_filter(None);
        }
        if let Some(position) = self.filtered.iter().position(|i| *i == tensor_index) {
            self.sel = position;
        }
    }

    /// The snapshot an agent receives for `snapshot`.
    pub fn snapshot(&self) -> Snapshot {
        let entry = self.selected();
        Snapshot {
            tensor: entry.map_or_else(|| "<aucun>".to_owned(), |e| e.name.clone()),
            role: entry.map_or_else(|| "none".to_owned(), |e| e.role.label().to_owned()),
            layer: entry.and_then(|e| e.layer),
            shape: entry.map_or_else(Vec::new, |e| e.shape.clone()),
            dtype: entry.map_or_else(|| "n/a".to_owned(), |e| e.dtype.name().to_owned()),
            byte_len: entry.map_or(0, |e| e.byte_len),
            filter: self.role_filter.map(|r| r.label().to_owned()),
        }
    }

    /// Writes the Markdown snippet for the selection; returns the path.
    pub fn export_context(&mut self) -> Option<String> {
        let Some(entry) = self.selected().cloned() else {
            self.log("rien à exporter");
            return None;
        };
        let info = ContextInfo {
            stats: self.selected_index().and_then(|i| self.stats.get(&i).cloned()),
            entry,
            family: self.index.family.clone(),
            format: self.index.format.to_string(),
        };
        match info.write_temp() {
            Ok(path) => {
                let shown = path.display().to_string();
                self.log(format!("contexte exporté : {shown}"));
                Some(shown)
            }
            Err(e) => {
                self.log(format!("export impossible : {e}"));
                None
            }
        }
    }

    /// Schedules missing metrics for the selection and its neighbors.
    pub fn schedule_metrics(&mut self, tx: &mpsc::UnboundedSender<AppEvent>) {
        if self.requested.len() >= 4 {
            return;
        }
        let mut want: Vec<usize> = Vec::new();
        let cursor = self.sel as isize;
        for offset in [0, 1, 2, 3] {
            let position = cursor + offset;
            if let Some(&index) = self.filtered.get(position.max(0) as usize) {
                if !self.stats.contains_key(&index) && self.requested.insert(index) {
                    want.push(index);
                }
            }
        }
        for index in want {
            let entry = self.index.tensors[index].clone();
            let tx = tx.clone();
            tokio::task::spawn_blocking(move || {
                let stats = compute_entry_stats(&entry);
                let _ = tx.send(AppEvent::Metrics { index, stats });
            });
        }
    }

    pub fn on_metrics(&mut self, index: usize, stats: Stats) {
        self.stats.insert(index, stats);
        self.requested.remove(&index);
    }

    /// Exactness badge of the selection (None for non-writers).
    pub fn selected_exactness(&self) -> Option<Exactness> {
        let entry = self.selected()?;
        Exactness::for_entry(self.index.family.as_deref(), entry)
    }

    pub fn context_of_selection(&self) -> Option<ContextInfo> {
        let index = self.selected_index()?;
        Some(ContextInfo {
            entry: self.index.tensors[index].clone(),
            family: self.index.family.clone(),
            format: self.index.format.to_string(),
            stats: self.stats.get(&index).cloned(),
        })
    }
}

/// What `on_key` asks the loop to do.
pub enum Control {
    Continue,
    LaunchAgent,
    Quit,
}

/// The metrics of one entry, computed with its bytes mmap-read in place.
fn compute_entry_stats(entry: &TensorEntry) -> Stats {
    match crate::model::read_entry_bytes(entry) {
        Ok(bytes) => metrics::compute(&entry.dtype, &entry.shape, &bytes),
        Err(_) => Stats {
            available: false,
            ..Stats::default()
        },
    }
}

/// Local wall-clock time as `HH:MM:SS` (UTC, no extra dependency).
fn stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{m:02}:{s:02}")
}

/// The event loop: keys + bridge + metrics + 60 Hz redraw.
pub async fn run(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    mut app: App,
    target: IpcTarget,
) -> Result<(), AppError> {
    let (tx, mut rx) = mpsc::unbounded_channel::<AppEvent>();

    // Keys, pushed from a dedicated task so the loop never blocks on input.
    let key_tx = tx.clone();
    tokio::spawn(async move {
        let mut events = EventStream::new();
        while let Some(result) = events.next().await {
            match result {
                Ok(Event::Key(key)) => {
                    if key_tx.send(AppEvent::Key(key)).is_err() {
                        break;
                    }
                }
                Ok(_) => {}
                Err(_) => break,
            }
        }
    });

    // The agent bridge, with a relay turning IpcRequests into app events.
    let (bridge_tx, mut bridge_rx) = mpsc::unbounded_channel::<IpcRequest>();
    tokio::spawn(async move {
        if let Err(_e) = crate::ipc::serve(target, bridge_tx).await {
            // No stdout here (alternate screen): the error only surfaces as a
            // dead bridge; the TUI keeps working.
        }
    });
    let relay_tx = tx.clone();
    tokio::spawn(async move {
        while let Some(request) = bridge_rx.recv().await {
            if relay_tx.send(AppEvent::Ipc(request)).is_err() {
                break;
            }
        }
    });

    let mut tick = tokio::time::interval(Duration::from_secs_f64(1.0 / 60.0));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    while !app.quit {
        tokio::select! {
            _ = tick.tick() => {
                app.schedule_metrics(&tx);
                ui::draw(terminal, &app).map_err(|e| AppError::Render(e.to_string()))?;
            }
            event = rx.recv() => {
                match event {
                    Some(AppEvent::Key(key)) => match app.on_key(key) {
                        Control::Continue => {}
                        Control::LaunchAgent => {
                            if let Some(info) = app.context_of_selection() {
                                let detail = run_agent_with_suspend(terminal, &app.agent_command, &info);
                                app.log(detail);
                            } else {
                                app.log("aucun tenseur sélectionné pour l'agent");
                            }
                        }
                        Control::Quit => app.quit = true,
                    },
                    Some(AppEvent::Ipc(request)) => app.on_ipc(request),
                    Some(AppEvent::Metrics { index, stats }) => app.on_metrics(index, stats),
                    Some(AppEvent::AgentDone { detail }) => app.log(format!("agent : {detail}")),
                    None => break,
                }
            }
        }
    }
    Ok(())
}

/// Runs the agent launcher: suspends the TUI, spawns the command, restores.
/// Returns a human line for the log.
pub fn run_agent_with_suspend(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    template: &str,
    info: &ContextInfo,
) -> String {
    let file = match info.write_temp() {
        Ok(path) => path,
        Err(e) => return format!("lancement impossible : {e}"),
    };
    let argv = agent_argv(template, &file);
    let shown = argv.join(" ");
    if crate::terminal::suspend(terminal).is_err() {
        return "lancement impossible : terminal non restaurable".into();
    }
    let result = launch_agent(&argv);
    let _ = crate::terminal::resume(terminal);
    match result {
        Ok(code) => format!("exécuté ({shown}) → code {code}"),
        Err(e) => format!("échec du lancement ({shown}) : {e}"),
    }
}
