//! Pure application state (Elm style): `update(app, msg) -> effects`. Nothing here draws
//! or computes statistics of a model; effects are executed by the loop.

use std::collections::{BTreeMap, HashMap};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use serde::Deserialize;

use super::tabs::preview::{self, PreviewEffect, PreviewMsg, PreviewState};
use super::theme::Theme;
use super::worker::{StatsKey, StatsRequest, TensorEntry};
use super::TuiError;
use crate::engine::stats::{self, TensorStats};
use crate::engine::{self, DType};

pub const PROJECTIONS: [&str; 3] = ["gate_proj", "up_proj", "down_proj"];
pub const DEFAULT_BINS: usize = 64;
/// Ticks are 16 ms polls without input: 31 of them make the 500 ms blink half-period.
pub const BLINK_TICKS: u64 = 31;
pub const SPINNER_TICKS: u64 = 6;
pub const MIN_WIDTH: u16 = 80;
pub const MIN_HEIGHT: u16 = 24;
const ZOOM_MAX: usize = 64;
const PAGE: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    Inspector,
    Steering,
    Preview,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Selection {
    pub layer: usize,
    pub proj: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Computation {
    Idle,
    Running { generation: u64 },
    Done(Result<TensorStats, String>),
}

#[derive(Debug, Clone)]
pub enum Msg {
    Key(KeyEvent),
    Resize(u16, u16),
    Tick,
    StatsReady {
        generation: u64,
        key: StatsKey,
        result: Result<TensorStats, String>,
    },
    Preview(PreviewMsg),
}

#[derive(Debug, Clone)]
pub enum Effect {
    RequestStats(StatsRequest),
    Preview(PreviewEffect),
}

/// MLP layout of the model, from the index only.
#[derive(Debug, Clone)]
pub struct ModelView {
    pub path: PathBuf,
    /// `layers[i][p]`: tensor of projection `p` in layer `i`, `None` when absent.
    pub layers: Vec<[Option<TensorEntry>; 3]>,
    pub hidden_size: Option<usize>,
}

impl ModelView {
    pub fn from_entries(path: &Path, entries: &[TensorEntry]) -> Self {
        let count = entries
            .iter()
            .filter(|e| e.name.contains(".mlp."))
            .filter_map(|e| engine::parse_layer_index(&e.name))
            .max()
            .map_or(0, |max| max + 1);
        let find = |name: String| entries.iter().find(|e| e.name == name).cloned();
        let layers = (0..count)
            .map(|i| PROJECTIONS.map(|p| find(format!("model.layers.{i}.mlp.{p}.weight"))))
            .collect::<Vec<_>>();
        let hidden_size = find("model.embed_tokens.weight".into())
            .and_then(|e| e.shape.get(1).copied())
            .or_else(|| {
                layers
                    .iter()
                    .flatten()
                    .flatten()
                    .find(|e| e.name.ends_with("down_proj.weight"))
                    .and_then(|e| e.shape.first().copied())
            });
        Self {
            path: path.to_path_buf(),
            layers,
            hidden_size,
        }
    }

    pub fn entry(&self, selection: Selection) -> Option<&TensorEntry> {
        self.layers
            .get(selection.layer)?
            .get(selection.proj)?
            .as_ref()
    }
}

/// `<vector>.stats.json` written by `surgeon extract` (schema 1).
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Companion {
    pub schema: u32,
    pub pos_projections: Vec<f64>,
    pub neg_projections: Vec<f64>,
    pub threshold: f64,
    pub d_prime: Option<f64>,
    pub threshold_accuracy: f64,
    pub cos_mean_pos_neg: f64,
    pub raw_diff_norm: f64,
    pub layer: usize,
    pub site: String,
    pub pooling: String,
    pub hidden_size: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CompanionState {
    Missing,
    Loaded(Companion),
    /// Corrupt or unknown schema: a warning, never a fatal error.
    Invalid(String),
}

#[derive(Debug, Clone)]
pub struct VectorView {
    pub path: PathBuf,
    pub values: Vec<f32>,
    pub stats: TensorStats,
    /// (index, value) of the 10 largest |v_i| among finite components.
    pub top: Vec<(usize, f32)>,
    pub metadata: BTreeMap<String, String>,
    pub companion: CompanionState,
}

impl VectorView {
    /// `direction` when present, else the single 1D tensor (the engine's own rule).
    pub fn load(path: &Path) -> Result<Self, TuiError> {
        let error = |message: String| TuiError::Vector {
            path: path.to_path_buf(),
            message,
        };
        let values = engine::load_direction_vector(path, Some(crate::extract::TENSOR_NAME))
            .or_else(|_| engine::load_direction_vector(path, None))
            .map_err(|e| error(e.to_string()))?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let never = AtomicBool::new(false);
        let stats = stats::tensor_stats(&bytes, &DType::F32, &[values.len()], 32, &never)
            .map_err(|e| error(e.to_string()))?;
        let mut order: Vec<usize> = (0..values.len())
            .filter(|&i| values[i].is_finite())
            .collect();
        order.sort_by(|&a, &b| values[b].abs().total_cmp(&values[a].abs()).then(a.cmp(&b)));
        let top = order.into_iter().take(10).map(|i| (i, values[i])).collect();
        Ok(Self {
            path: path.to_path_buf(),
            values,
            stats,
            top,
            metadata: read_metadata(path).unwrap_or_default(),
            companion: load_companion(&crate::extract::companion_path(path)),
        })
    }

    /// L2 norm of the finite components, in f64.
    pub fn norm(&self) -> f64 {
        self.stats.l2
    }

    pub fn is_normalized(&self) -> bool {
        (self.norm() - 1.0).abs() < 1e-3
    }
}

/// `__metadata__` of a SafeTensors header (strings only).
fn read_metadata(path: &Path) -> Option<BTreeMap<String, String>> {
    let mut file = std::fs::File::open(path).ok()?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len).ok()?;
    let len = usize::try_from(u64::from_le_bytes(len))
        .ok()
        .filter(|&n| n <= 100 << 20)?;
    let mut header = vec![0u8; len];
    file.read_exact(&mut header).ok()?;
    let value: serde_json::Value = serde_json::from_slice(&header).ok()?;
    let metadata = value.get("__metadata__")?.as_object()?;
    Some(
        metadata
            .iter()
            .filter_map(|(k, v)| Some((k.clone(), v.as_str()?.to_owned())))
            .collect(),
    )
}

pub fn load_companion(path: &Path) -> CompanionState {
    let raw = match std::fs::read_to_string(path) {
        Ok(raw) => raw,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return CompanionState::Missing,
        Err(e) => return CompanionState::Invalid(e.to_string()),
    };
    match serde_json::from_str::<Companion>(&raw) {
        Ok(c) if c.schema == crate::extract::COMPANION_SCHEMA => CompanionState::Loaded(c),
        Ok(c) => CompanionState::Invalid(format!("schema {} inconnu", c.schema)),
        Err(e) => CompanionState::Invalid(format!("fichier compagnon illisible : {e}")),
    }
}

#[derive(Debug, Clone)]
pub struct App {
    pub tab: Tab,
    pub selection: Selection,
    pub computation: Computation,
    pub generation: u64,
    pub model: Option<ModelView>,
    pub vector: Option<VectorView>,
    pub help: bool,
    pub log_scale: bool,
    pub pan: usize,
    pub zoom: usize,
    pub top_scroll: usize,
    pub ticks: u64,
    pub dirty: bool,
    pub should_quit: bool,
    pub size: (u16, u16),
    pub theme: Theme,
    pub bins: usize,
    /// Tensors already computed: `true` when they hold NaN/Inf (⚠ in the layer list).
    pub flagged: HashMap<StatsKey, bool>,
    /// Third tab (`--preview`); `None` keeps the two-tab behaviour.
    pub preview: Option<PreviewState>,
}

impl App {
    pub fn new(
        model: Option<ModelView>,
        vector: Option<VectorView>,
        theme: Theme,
        size: (u16, u16),
    ) -> Self {
        let tab = match (&model, &vector) {
            (None, Some(_)) => Tab::Steering,
            _ => Tab::Inspector,
        };
        let mut app = Self {
            tab,
            selection: Selection { layer: 0, proj: 0 },
            computation: Computation::Idle,
            generation: 0,
            model,
            vector,
            help: false,
            log_scale: false,
            pan: 0,
            zoom: 1,
            top_scroll: 0,
            ticks: 0,
            dirty: true,
            should_quit: false,
            size,
            theme,
            bins: DEFAULT_BINS,
            flagged: HashMap::new(),
            preview: None,
        };
        if let Some(proj) = app.first_available(0) {
            app.selection.proj = proj;
        }
        app
    }

    /// Request for the initial selection.
    pub fn start(&mut self) -> Vec<Effect> {
        let mut effects = self.request_selection();
        effects.extend(self.preview_start());
        effects
    }

    pub fn current_entry(&self) -> Option<&TensorEntry> {
        self.model.as_ref()?.entry(self.selection)
    }

    fn available(&self, layer: usize, proj: usize) -> bool {
        self.model
            .as_ref()
            .is_some_and(|m| m.entry(Selection { layer, proj }).is_some())
    }

    fn first_available(&self, layer: usize) -> Option<usize> {
        (0..PROJECTIONS.len()).find(|&p| self.available(layer, p))
    }

    fn layer_count(&self) -> usize {
        self.model.as_ref().map_or(0, |m| m.layers.len())
    }

    pub fn blink_on(&self) -> bool {
        (self.ticks / BLINK_TICKS).is_multiple_of(2)
    }

    pub fn spinner(&self) -> &'static str {
        const FRAMES: [&str; 8] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧"];
        FRAMES[(self.ticks / SPINNER_TICKS) as usize % FRAMES.len()]
    }

    /// Statistics shown in the inspector, when computed.
    pub fn current_stats(&self) -> Option<&TensorStats> {
        match &self.computation {
            Computation::Done(Ok(stats)) => Some(stats),
            _ => None,
        }
    }

    /// Whether a blinking alert is on screen (only then does blinking redraw).
    pub fn alert_visible(&self) -> bool {
        match self.tab {
            Tab::Inspector => self.current_stats().is_some_and(|s| s.non_finite > 0),
            Tab::Steering => self.vector.as_ref().is_some_and(|v| v.stats.non_finite > 0),
            Tab::Preview => false,
        }
    }

    fn request_selection(&mut self) -> Vec<Effect> {
        let Some(entry) = self.current_entry().cloned() else {
            self.computation = Computation::Idle;
            return Vec::new();
        };
        self.generation += 1;
        self.computation = Computation::Running {
            generation: self.generation,
        };
        vec![Effect::RequestStats(StatsRequest {
            generation: self.generation,
            key: (self.selection.layer, self.selection.proj),
            tensor: entry.name,
            bins: self.bins,
        })]
    }

    fn select_layer(&mut self, layer: usize) -> Vec<Effect> {
        let layer = layer.min(self.layer_count().saturating_sub(1));
        if layer == self.selection.layer {
            return Vec::new();
        }
        self.selection.layer = layer;
        if !self.available(layer, self.selection.proj) {
            if let Some(proj) = self.first_available(layer) {
                self.selection.proj = proj;
            }
        }
        self.request_selection()
    }

    /// Next available projection in `step` direction; stops at the ends (no wrap).
    fn step_projection(&mut self, forward: bool) -> Vec<Effect> {
        let layer = self.selection.layer;
        let candidates: Vec<usize> = match forward {
            true => (self.selection.proj + 1..PROJECTIONS.len()).collect(),
            false => (0..self.selection.proj).rev().collect(),
        };
        match candidates.into_iter().find(|&p| self.available(layer, p)) {
            Some(proj) => {
                self.selection.proj = proj;
                self.request_selection()
            }
            None => Vec::new(),
        }
    }

    fn top_scroll_max(&self) -> usize {
        self.vector
            .as_ref()
            .map_or(0, |v| v.top.len().saturating_sub(1))
    }

    fn pan_max(&self) -> usize {
        self.vector
            .as_ref()
            .map_or(0, |v| v.values.len().saturating_sub(1))
    }
}

impl App {
    /// Tab / Shift+Tab: the third tab only exists with `--preview`.
    fn cycle_tab(&mut self, forward: bool) {
        let tabs: &[Tab] = match self.preview {
            Some(_) => &[Tab::Inspector, Tab::Steering, Tab::Preview],
            None => &[Tab::Inspector, Tab::Steering],
        };
        let at = tabs.iter().position(|&t| t == self.tab).unwrap_or(0);
        let step = if forward { 1 } else { tabs.len() - 1 };
        self.tab = tabs[(at + step) % tabs.len()];
    }

    fn preview_update(&mut self, msg: PreviewMsg) -> Vec<Effect> {
        match &mut self.preview {
            Some(state) => preview::update(state, msg)
                .into_iter()
                .map(Effect::Preview)
                .collect(),
            None => Vec::new(),
        }
    }

    fn preview_start(&mut self) -> Vec<Effect> {
        let effects = match &mut self.preview {
            Some(state) => state.start(),
            None => Vec::new(),
        };
        effects.into_iter().map(Effect::Preview).collect()
    }

    /// Debounce and spinner of the preview; redraws only when the tab shows a change.
    fn preview_tick(&mut self) -> Vec<Effect> {
        let effects = self.preview_update(PreviewMsg::Tick);
        let animated = self.preview.as_ref().is_some_and(|p| p.busy())
            && self.ticks.is_multiple_of(SPINNER_TICKS);
        self.dirty |= self.tab == Tab::Preview && (animated || !effects.is_empty());
        effects
    }

    fn preview_input_focused(&self) -> bool {
        self.tab == Tab::Preview && self.preview.as_ref().is_some_and(|p| p.input_focus)
    }
}

/// Windows reports Press and Release: only Press counts, plus Repeat for held keys that
/// move (arrows, pages).
pub fn accepted(key: &KeyEvent) -> bool {
    match key.kind {
        KeyEventKind::Press => true,
        KeyEventKind::Repeat => matches!(
            key.code,
            KeyCode::Up
                | KeyCode::Down
                | KeyCode::Left
                | KeyCode::Right
                | KeyCode::PageUp
                | KeyCode::PageDown
        ),
        KeyEventKind::Release => false,
    }
}

pub fn handle_key(app: &mut App, key: KeyEvent) -> Vec<Effect> {
    if !accepted(&key) {
        return Vec::new();
    }
    // Raw mode turns Ctrl-C into a key: it quits like `q`.
    if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
        app.should_quit = true;
        return Vec::new();
    }
    // The prompt of the third tab takes every key but Tab: letters are text there.
    if app.preview_input_focused() && !matches!(key.code, KeyCode::Tab | KeyCode::BackTab) {
        app.dirty = true;
        return app.preview_update(PreviewMsg::Key(key));
    }
    match key.code {
        KeyCode::Char('q') => {
            app.should_quit = true;
            return Vec::new();
        }
        KeyCode::Esc if app.help => {
            app.help = false;
            app.dirty = true;
            return Vec::new();
        }
        KeyCode::Esc => {
            app.should_quit = true;
            return Vec::new();
        }
        KeyCode::Char('?') => {
            app.help = !app.help;
            app.dirty = true;
            return Vec::new();
        }
        _ => {}
    }
    // Quitting needs no redraw; anything else does.
    app.dirty = true;
    match key.code {
        _ if app.help => return Vec::new(),
        KeyCode::Tab | KeyCode::BackTab => {
            app.cycle_tab(key.code == KeyCode::Tab);
            return Vec::new();
        }
        _ => {}
    }
    match app.tab {
        Tab::Inspector => match key.code {
            KeyCode::Up => app.select_layer(app.selection.layer.saturating_sub(1)),
            KeyCode::Down => app.select_layer(app.selection.layer + 1),
            KeyCode::PageUp => app.select_layer(app.selection.layer.saturating_sub(PAGE)),
            KeyCode::PageDown => app.select_layer(app.selection.layer + PAGE),
            KeyCode::Home => app.select_layer(0),
            KeyCode::End => app.select_layer(usize::MAX),
            KeyCode::Left => app.step_projection(false),
            KeyCode::Right => app.step_projection(true),
            KeyCode::Char('l') => {
                app.log_scale = !app.log_scale;
                Vec::new()
            }
            _ => Vec::new(),
        },
        Tab::Steering => {
            match key.code {
                KeyCode::Left => app.pan = app.pan.saturating_sub(app.zoom.max(1)),
                KeyCode::Right => app.pan = (app.pan + app.zoom.max(1)).min(app.pan_max()),
                KeyCode::Char('+') => app.zoom = (app.zoom / 2).max(1),
                KeyCode::Char('-') => app.zoom = (app.zoom * 2).min(ZOOM_MAX),
                KeyCode::Up => app.top_scroll = app.top_scroll.saturating_sub(1),
                KeyCode::Down => app.top_scroll = (app.top_scroll + 1).min(app.top_scroll_max()),
                _ => {}
            }
            Vec::new()
        }
        Tab::Preview => app.preview_update(PreviewMsg::Key(key)),
    }
}

pub fn update(app: &mut App, msg: Msg) -> Vec<Effect> {
    match msg {
        Msg::Key(key) => handle_key(app, key),
        Msg::Resize(width, height) => {
            app.size = (width, height);
            app.dirty = true;
            Vec::new()
        }
        Msg::Tick => {
            app.ticks += 1;
            let blink_flip = app.ticks.is_multiple_of(BLINK_TICKS) && app.alert_visible();
            let spinner_step = app.ticks.is_multiple_of(SPINNER_TICKS)
                && matches!(app.computation, Computation::Running { .. });
            app.dirty |= blink_flip || spinner_step;
            app.preview_tick()
        }
        Msg::StatsReady {
            generation,
            key,
            result,
        } => {
            if let Ok(stats) = &result {
                app.flagged.insert(key, stats.non_finite > 0);
                app.dirty = true;
            }
            // A stale generation (the selection moved on) is ignored.
            if generation == app.generation {
                app.computation = Computation::Done(result);
                app.dirty = true;
            }
            Vec::new()
        }
        Msg::Preview(msg) => {
            app.dirty = true;
            app.preview_update(msg)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::ColorMode;

    pub(crate) fn entry(name: String) -> TensorEntry {
        TensorEntry {
            name,
            dtype: "F32".into(),
            shape: vec![4, 4],
            bytes: 64,
        }
    }

    fn model(layers: usize, skip: Option<(usize, usize)>) -> ModelView {
        let entries: Vec<TensorEntry> = (0..layers)
            .flat_map(|i| {
                PROJECTIONS
                    .iter()
                    .enumerate()
                    .map(move |(p, name)| (i, p, name))
            })
            .filter(|&(i, p, _)| Some((i, p)) != skip)
            .map(|(i, _, name)| entry(format!("model.layers.{i}.mlp.{name}.weight")))
            .collect();
        ModelView::from_entries(Path::new("m"), &entries)
    }

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn app(layers: usize) -> App {
        App::new(
            Some(model(layers, None)),
            None,
            Theme::new(ColorMode::Plain),
            (100, 32),
        )
    }

    #[test]
    fn navigation_is_bounded_and_requests_stats() {
        let mut a = app(3);
        assert_eq!(a.start().len(), 1);
        assert!(
            handle_key(&mut a, key(KeyCode::Up)).is_empty(),
            "already at layer 0"
        );
        assert_eq!(handle_key(&mut a, key(KeyCode::Down)).len(), 1);
        handle_key(&mut a, key(KeyCode::End));
        assert_eq!(a.selection.layer, 2);
        assert!(
            handle_key(&mut a, key(KeyCode::Down)).is_empty(),
            "bounded at the last layer"
        );
        handle_key(&mut a, key(KeyCode::Home));
        assert_eq!(a.selection.layer, 0);
        handle_key(&mut a, key(KeyCode::Right));
        handle_key(&mut a, key(KeyCode::Right));
        assert_eq!(a.selection.proj, 2);
        assert!(
            handle_key(&mut a, key(KeyCode::Right)).is_empty(),
            "no wrap"
        );
        handle_key(&mut a, key(KeyCode::PageDown));
        assert_eq!(a.selection.layer, 2);
    }

    #[test]
    fn missing_projection_is_skipped() {
        let mut a = App::new(
            Some(model(2, Some((0, 1)))),
            None,
            Theme::new(ColorMode::Plain),
            (100, 32),
        );
        handle_key(&mut a, key(KeyCode::Right));
        assert_eq!(
            a.selection.proj, 2,
            "up_proj of layer 0 is absent: jump to down_proj"
        );
    }

    #[test]
    fn quit_keys_help_tabs_and_key_kinds() {
        for code in [KeyCode::Char('q'), KeyCode::Esc] {
            let mut a = app(1);
            handle_key(&mut a, key(code));
            assert!(a.should_quit);
        }
        let mut a = app(1);
        handle_key(
            &mut a,
            KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
        );
        assert!(a.should_quit, "Ctrl-C quits");

        let mut a = app(1);
        handle_key(&mut a, key(KeyCode::Char('?')));
        assert!(a.help);
        handle_key(&mut a, key(KeyCode::Esc));
        assert!(!a.help && !a.should_quit, "Esc closes the help first");
        handle_key(&mut a, key(KeyCode::Tab));
        assert_eq!(a.tab, Tab::Steering);
        handle_key(&mut a, key(KeyCode::BackTab));
        assert_eq!(a.tab, Tab::Inspector);

        let mut a = app(3);
        let mut release = key(KeyCode::Down);
        release.kind = KeyEventKind::Release;
        handle_key(&mut a, release);
        assert_eq!(a.selection.layer, 0, "Release ignored (Windows sends both)");
        let mut repeat = key(KeyCode::Down);
        repeat.kind = KeyEventKind::Repeat;
        handle_key(&mut a, repeat);
        assert_eq!(a.selection.layer, 1, "Repeat accepted for arrows");
        let mut repeat_q = key(KeyCode::Char('q'));
        repeat_q.kind = KeyEventKind::Repeat;
        handle_key(&mut a, repeat_q);
        assert!(!a.should_quit);
    }

    #[test]
    fn stale_results_are_ignored_and_ticks_only_dirty_when_needed() {
        let mut a = app(2);
        a.start();
        handle_key(&mut a, key(KeyCode::Down));
        let stale = a.generation - 1;
        let never = AtomicBool::new(false);
        let stats = stats::tensor_stats(&[0u8; 4], &DType::F32, &[1], 4, &never).unwrap();
        update(
            &mut a,
            Msg::StatsReady {
                generation: stale,
                key: (0, 0),
                result: Ok(stats.clone()),
            },
        );
        assert!(
            matches!(a.computation, Computation::Running { .. }),
            "stale reply dropped"
        );
        let current = a.generation;
        update(
            &mut a,
            Msg::StatsReady {
                generation: current,
                key: (1, 0),
                result: Ok(stats),
            },
        );
        assert!(matches!(a.computation, Computation::Done(Ok(_))));
        a.dirty = false;
        for _ in 0..100 {
            update(&mut a, Msg::Tick);
        }
        assert!(!a.dirty, "no alert, no computation: ticks never redraw");
    }
}
