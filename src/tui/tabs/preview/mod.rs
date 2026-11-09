//! "Live Steering Preview" tab: edit a prompt, move the steering slider, and compare the
//! greedy continuation of the model without and with the steering vector injected.
//!
//! ```text
//! key / tick / worker event -> update(state, msg) -> Vec<PreviewEffect>    (pure)
//! PreviewEffect::Generate(Request) -> PreviewWorker::request               (the loop)
//! PreviewWorker thread: Session::run -> Event::{Scale, Token, Done, Error}  (worker.rs)
//! view(frame, area, state, theme): draws from the state only, never calls the worker
//! ```
//!
//! The slider is relative: `alpha_abs = alpha * s`, `s` being the mean residual norm at the
//! injection layer on the current prompt. It is an activation injection, an approximation
//! of the weight patch, not the patch itself.
//!
//! Stale protection: every emitted request bumps `gen_id`; replies of another `gen_id` are
//! dropped. Slider moves are debounced by [`DEBOUNCE_MS`] on an injectable [`Clock`].

pub mod cli;
pub mod editor;
mod launch;
mod render;
pub mod worker;

use std::fmt::Debug;
use std::sync::Arc;
use std::time::Instant;

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::extract::ChatTemplate;
use editor::LineEditor;
pub use launch::{prepare, Launch};
pub use render::view;
pub use worker::{Event, Loader, Pane, PreviewWorker, Request, Session};

pub const DEBOUNCE_MS: u64 = 250;
pub const ALPHA_STEP: f64 = 0.05;
pub const ALPHA_FINE_STEP: f64 = 0.01;
/// Below this speed the tab warns that the preview is too slow to be interactive.
pub const SLOW_TOK_PER_S: f64 = 1.0;
pub const NOTE: &str = "steering d'activation, approximation du patch de poids";
pub const DEFAULT_PROMPT: &str = "Raconte-moi une courte histoire.";

/// Millisecond time source, injected so that tests control the debounce.
pub trait Clock: Debug + Send + Sync {
    fn now_ms(&self) -> u64;
}

#[derive(Debug)]
pub struct SystemClock(Instant);

impl SystemClock {
    pub fn new() -> Self {
        Self(Instant::now())
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        u64::try_from(self.0.elapsed().as_millis()).unwrap_or(u64::MAX)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ModelStatus {
    Loading,
    Ready,
    /// Not enough resources, or the model could not be loaded: the message is shown in
    /// the tab, the other tabs are not affected.
    Unavailable(String),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum PaneStatus {
    #[default]
    Idle,
    Running,
    Done,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PaneState {
    pub text: String,
    pub tok_idx: usize,
    pub tok_per_s: f64,
    pub status: PaneStatus,
}

#[derive(Debug, Clone)]
pub struct PreviewConfig {
    pub prompt: String,
    pub alpha: f64,
    pub alpha_range: (f64, f64),
    pub tokens: usize,
    pub layer: usize,
    pub template: ChatTemplate,
}

#[derive(Debug, Clone)]
pub struct PreviewState {
    pub editor: LineEditor,
    pub input_focus: bool,
    /// Prompt of the last generation (the editor may hold an unvalidated edit).
    pub prompt: String,
    pub alpha: f64,
    pub alpha_range: (f64, f64),
    pub tokens: usize,
    pub layer: usize,
    pub template: ChatTemplate,
    pub model: ModelStatus,
    pub warnings: Vec<String>,
    /// Mean residual norm for the prompt of the last generation.
    pub scale: Option<f32>,
    pub baseline: PaneState,
    pub steered: PaneState,
    /// Relative alpha of the steered pane as last requested.
    pub shown_alpha: f64,
    pub notice: Option<String>,
    pub ticks: u64,
    gen_id: u64,
    /// (prompt, template) whose baseline is on screen.
    baseline_key: Option<(String, ChatTemplate)>,
    /// When the debounced regeneration is due.
    due_ms: Option<u64>,
    clock: Arc<dyn Clock>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PreviewMsg {
    Key(KeyEvent),
    /// One loop iteration without input: the debounce is checked here.
    Tick,
    Worker(Event),
}

#[derive(Debug, Clone, PartialEq)]
pub enum PreviewEffect {
    Generate(Request),
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

impl PreviewState {
    pub fn new(config: PreviewConfig, clock: Arc<dyn Clock>) -> Self {
        let alpha = config
            .alpha
            .clamp(config.alpha_range.0, config.alpha_range.1);
        Self {
            editor: LineEditor::new(&config.prompt),
            input_focus: false,
            prompt: config.prompt,
            alpha,
            alpha_range: config.alpha_range,
            tokens: config.tokens,
            layer: config.layer,
            template: config.template,
            model: ModelStatus::Loading,
            warnings: Vec::new(),
            scale: None,
            baseline: PaneState::default(),
            steered: PaneState::default(),
            shown_alpha: alpha,
            notice: None,
            ticks: 0,
            gen_id: 0,
            baseline_key: None,
            due_ms: None,
            clock,
        }
    }

    /// First generation, with the prompt and alpha given on the command line.
    pub fn start(&mut self) -> Vec<PreviewEffect> {
        self.generate()
    }

    pub fn unavailable(&mut self, message: String) {
        self.model = ModelStatus::Unavailable(message);
    }

    pub fn alpha_abs(&self) -> Option<f64> {
        self.scale.map(|s| self.alpha * f64::from(s))
    }

    /// A slider move waits for its debounce.
    pub fn pending(&self) -> bool {
        self.due_ms.is_some()
    }

    /// Something is loading, running or about to run: the loop redraws the spinner.
    pub fn busy(&self) -> bool {
        self.model == ModelStatus::Loading
            || self.pending()
            || [&self.baseline, &self.steered]
                .iter()
                .any(|p| p.status == PaneStatus::Running)
    }

    /// Lowest measured speed once a pane has a few tokens (the first one includes the
    /// prompt prefill), when it is below [`SLOW_TOK_PER_S`].
    pub fn slow(&self) -> Option<f64> {
        [&self.baseline, &self.steered]
            .iter()
            .filter(|p| p.tok_idx >= 3 && p.tok_per_s > 0.0 && p.tok_per_s < SLOW_TOK_PER_S)
            .map(|p| p.tok_per_s)
            .reduce(f64::min)
    }

    pub fn keys_hint(&self) -> &'static str {
        match self.input_focus {
            true => "Entrée générer · Esc quitter la saisie · Tab onglet",
            false => "Tab onglet · i saisie · ←→ alpha (Shift fin) · 0 reset · t gabarit · r relancer · q quitter",
        }
    }

    fn set_alpha(&mut self, alpha: f64) {
        let alpha = round2(alpha).clamp(self.alpha_range.0, self.alpha_range.1);
        if alpha != self.alpha {
            self.alpha = alpha;
            self.due_ms = Some(self.clock.now_ms() + DEBOUNCE_MS);
        }
    }

    /// Emits a request for the current prompt, template and alpha.
    fn generate(&mut self) -> Vec<PreviewEffect> {
        self.due_ms = None;
        if matches!(self.model, ModelStatus::Unavailable(_)) {
            return Vec::new();
        }
        if self.prompt.trim().is_empty() {
            self.notice = Some("prompt vide : rien à générer".into());
            return Vec::new();
        }
        self.notice = None;
        self.gen_id += 1;
        let key = (self.prompt.clone(), self.template);
        if self.baseline_key.as_ref() != Some(&key) {
            self.baseline = PaneState {
                status: PaneStatus::Running,
                ..PaneState::default()
            };
            self.scale = None;
            self.baseline_key = Some(key);
        }
        self.steered = PaneState {
            status: PaneStatus::Running,
            ..PaneState::default()
        };
        self.shown_alpha = self.alpha;
        vec![PreviewEffect::Generate(Request {
            gen_id: self.gen_id,
            prompt: self.prompt.clone(),
            alpha: self.alpha,
            tokens: self.tokens,
            template: self.template,
        })]
    }

    fn on_key(&mut self, key: KeyEvent) -> Vec<PreviewEffect> {
        if self.input_focus {
            return self.on_input_key(key);
        }
        let step = match key.modifiers.contains(KeyModifiers::SHIFT) {
            true => ALPHA_FINE_STEP,
            false => ALPHA_STEP,
        };
        match key.code {
            KeyCode::Char('i') | KeyCode::Enter => self.input_focus = true,
            KeyCode::Left => self.set_alpha(self.alpha - step),
            KeyCode::Right => self.set_alpha(self.alpha + step),
            KeyCode::Char('0') => self.set_alpha(0.0),
            KeyCode::Char('t') => {
                self.template = match self.template {
                    ChatTemplate::Chatml => ChatTemplate::NoTemplate,
                    ChatTemplate::NoTemplate => ChatTemplate::Chatml,
                };
                return self.generate();
            }
            KeyCode::Char('r') => return self.generate(),
            _ => {}
        }
        Vec::new()
    }

    fn on_input_key(&mut self, key: KeyEvent) -> Vec<PreviewEffect> {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Esc => self.input_focus = false,
            KeyCode::Enter => {
                self.prompt = self.editor.text().to_owned();
                return self.generate();
            }
            KeyCode::Char('u') if ctrl => self.editor.kill_to_start(),
            KeyCode::Char(c) if !ctrl && !key.modifiers.contains(KeyModifiers::ALT) => {
                self.editor.insert(c)
            }
            KeyCode::Backspace => self.editor.backspace(),
            KeyCode::Delete => self.editor.delete(),
            KeyCode::Left => self.editor.left(),
            KeyCode::Right => self.editor.right(),
            KeyCode::Home => self.editor.home(),
            KeyCode::End => self.editor.end(),
            _ => {}
        }
        Vec::new()
    }

    fn on_event(&mut self, event: Event) {
        match event {
            Event::Loaded { warnings } => {
                self.model = ModelStatus::Ready;
                self.warnings = warnings;
            }
            Event::LoadFailed(message) => {
                self.model = ModelStatus::Unavailable(message);
                self.baseline = PaneState::default();
                self.steered = PaneState::default();
            }
            // Everything below belongs to one request: only the latest counts.
            Event::Scale { gen_id, scale } if gen_id == self.gen_id => self.scale = Some(scale),
            Event::Token {
                gen_id,
                pane,
                text,
                tok_idx,
                tok_per_s,
            } if gen_id == self.gen_id => {
                *self.pane_mut(pane) = PaneState {
                    text,
                    tok_idx,
                    tok_per_s,
                    status: PaneStatus::Running,
                };
            }
            Event::Done { gen_id, pane } if gen_id == self.gen_id => {
                self.pane_mut(pane).status = PaneStatus::Done;
            }
            Event::Error { gen_id, message } if gen_id == self.gen_id => {
                for pane in [&mut self.baseline, &mut self.steered] {
                    if pane.status == PaneStatus::Running {
                        pane.status = PaneStatus::Failed(message.clone());
                    }
                }
            }
            Event::Scale { .. }
            | Event::Token { .. }
            | Event::Done { .. }
            | Event::Error { .. } => {}
        }
    }

    fn pane_mut(&mut self, pane: Pane) -> &mut PaneState {
        match pane {
            Pane::Baseline => &mut self.baseline,
            Pane::Steered => &mut self.steered,
        }
    }
}

/// Pure transition: the state changes, requests for the worker are returned.
pub fn update(state: &mut PreviewState, msg: PreviewMsg) -> Vec<PreviewEffect> {
    match msg {
        PreviewMsg::Key(key) => state.on_key(key),
        PreviewMsg::Tick => {
            state.ticks += 1;
            match state.due_ms {
                Some(due) if state.clock.now_ms() >= due => state.generate(),
                _ => Vec::new(),
            }
        }
        PreviewMsg::Worker(event) => {
            state.on_event(event);
            Vec::new()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[derive(Debug, Default)]
    struct FakeClock(AtomicU64);

    impl Clock for FakeClock {
        fn now_ms(&self) -> u64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    fn state(clock: &Arc<FakeClock>) -> PreviewState {
        PreviewState::new(
            PreviewConfig {
                prompt: "bonjour".into(),
                alpha: 0.0,
                alpha_range: (-2.0, 2.0),
                tokens: 8,
                layer: 2,
                template: ChatTemplate::Chatml,
            },
            clock.clone(),
        )
    }

    fn key(code: KeyCode) -> PreviewMsg {
        PreviewMsg::Key(KeyEvent::new(code, KeyModifiers::NONE))
    }

    fn requests(effects: &[PreviewEffect]) -> Vec<&Request> {
        effects.iter().map(|PreviewEffect::Generate(r)| r).collect()
    }

    #[test]
    fn slider_is_debounced_bounded_and_fine_with_shift() {
        let clock = Arc::new(FakeClock::default());
        let mut s = state(&clock);
        s.start();
        update(&mut s, key(KeyCode::Right));
        update(&mut s, key(KeyCode::Right));
        assert_eq!(s.alpha, 0.1);
        update(
            &mut s,
            PreviewMsg::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::SHIFT)),
        );
        assert_eq!(s.alpha, 0.09, "Shift steps by 0.01");
        assert!(
            update(&mut s, PreviewMsg::Tick).is_empty(),
            "debounce running"
        );
        clock.0.store(249, Ordering::SeqCst);
        assert!(update(&mut s, PreviewMsg::Tick).is_empty());
        clock.0.store(250, Ordering::SeqCst);
        let effects = update(&mut s, PreviewMsg::Tick);
        assert_eq!(requests(&effects).len(), 1, "one request after 250 ms");
        assert_eq!(requests(&effects)[0].alpha, 0.09);
        assert!(
            update(&mut s, PreviewMsg::Tick).is_empty(),
            "debounce consumed"
        );
        for _ in 0..100 {
            update(&mut s, key(KeyCode::Right));
        }
        assert_eq!(s.alpha, 2.0, "bounded by the range");
        update(&mut s, key(KeyCode::Char('0')));
        assert_eq!(s.alpha, 0.0);
    }

    #[test]
    fn debounce_restarts_on_each_move_and_reset_on_zero_is_a_noop() {
        let clock = Arc::new(FakeClock::default());
        let mut s = state(&clock);
        update(&mut s, key(KeyCode::Char('0')));
        assert_eq!(s.due_ms, None, "alpha already 0: nothing to regenerate");
        update(&mut s, key(KeyCode::Right));
        clock.0.store(200, Ordering::SeqCst);
        update(&mut s, key(KeyCode::Right));
        clock.0.store(400, Ordering::SeqCst);
        assert!(
            update(&mut s, PreviewMsg::Tick).is_empty(),
            "restarted at 200"
        );
        clock.0.store(450, Ordering::SeqCst);
        assert_eq!(update(&mut s, PreviewMsg::Tick).len(), 1);
    }

    #[test]
    fn range_not_containing_zero_clamps_reset() {
        let clock = Arc::new(FakeClock::default());
        let mut s = PreviewState::new(
            PreviewConfig {
                prompt: "p".into(),
                alpha: 9.0,
                alpha_range: (0.5, 1.0),
                tokens: 1,
                layer: 0,
                template: ChatTemplate::NoTemplate,
            },
            clock,
        );
        assert_eq!(s.alpha, 1.0, "initial alpha clamped");
        update(&mut s, key(KeyCode::Char('0')));
        assert_eq!(s.alpha, 0.5);
    }

    #[test]
    fn keys_outside_and_inside_the_input() {
        let clock = Arc::new(FakeClock::default());
        let mut s = state(&clock);
        s.start();
        update(&mut s, key(KeyCode::Char('i')));
        assert!(s.input_focus);
        // Letters that are commands outside the input are text inside it.
        for c in ['t', 'r', '0', 'q'] {
            update(&mut s, key(KeyCode::Char(c)));
        }
        assert_eq!(s.editor.text(), "bonjourtr0q");
        assert_eq!(s.template, ChatTemplate::Chatml);
        update(
            &mut s,
            PreviewMsg::Key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL)),
        );
        assert!(s.editor.is_empty());
        for c in "été".chars() {
            update(&mut s, key(KeyCode::Char(c)));
        }
        let effects = update(&mut s, key(KeyCode::Enter));
        assert_eq!(requests(&effects)[0].prompt, "été", "Enter generates");
        update(&mut s, key(KeyCode::Esc));
        assert!(!s.input_focus);
        let effects = update(&mut s, key(KeyCode::Char('t')));
        let request = requests(&effects)[0].clone();
        assert_eq!(request.template, ChatTemplate::NoTemplate);
        assert_eq!(request.gen_id, s.gen_id);
        assert_eq!(requests(&update(&mut s, key(KeyCode::Char('r')))).len(), 1);
    }

    #[test]
    fn empty_prompt_and_unavailable_model_emit_nothing() {
        let clock = Arc::new(FakeClock::default());
        let mut s = state(&clock);
        update(&mut s, key(KeyCode::Char('i')));
        s.editor.kill_to_start();
        assert!(update(&mut s, key(KeyCode::Enter)).is_empty());
        assert!(s.notice.is_some());
        let mut s = state(&clock);
        s.unavailable("RAM insuffisante".into());
        assert!(s.start().is_empty());
        update(&mut s, key(KeyCode::Char('r')));
        assert_eq!(s.gen_id, 0);
    }

    #[test]
    fn stale_events_are_ignored() {
        let clock = Arc::new(FakeClock::default());
        let mut s = state(&clock);
        s.start();
        update(
            &mut s,
            PreviewMsg::Worker(Event::Loaded { warnings: vec![] }),
        );
        update(&mut s, key(KeyCode::Char('r')));
        let token = |gen_id, text: &str| {
            PreviewMsg::Worker(Event::Token {
                gen_id,
                pane: Pane::Steered,
                text: text.into(),
                tok_idx: 1,
                tok_per_s: 5.0,
            })
        };
        update(&mut s, token(1, "vieux"));
        assert_eq!(s.steered.text, "", "gen 1 is stale after gen 2");
        update(&mut s, token(2, "neuf"));
        assert_eq!(s.steered.text, "neuf");
        update(
            &mut s,
            PreviewMsg::Worker(Event::Done {
                gen_id: 1,
                pane: Pane::Steered,
            }),
        );
        assert_eq!(s.steered.status, PaneStatus::Running);
        update(
            &mut s,
            PreviewMsg::Worker(Event::Scale {
                gen_id: 1,
                scale: 9.0,
            }),
        );
        assert_eq!(s.scale, None);
        update(
            &mut s,
            PreviewMsg::Worker(Event::Error {
                gen_id: 2,
                message: "boom".into(),
            }),
        );
        assert_eq!(s.steered.status, PaneStatus::Failed("boom".into()));
    }

    #[test]
    fn slow_generation_is_flagged_after_a_few_tokens() {
        let clock = Arc::new(FakeClock::default());
        let mut s = state(&clock);
        s.steered = PaneState {
            text: "x".into(),
            tok_idx: 1,
            tok_per_s: 0.2,
            status: PaneStatus::Running,
        };
        assert_eq!(s.slow(), None, "too early to judge");
        s.steered.tok_idx = 4;
        assert_eq!(s.slow(), Some(0.2));
        s.steered.tok_per_s = 3.0;
        assert_eq!(s.slow(), None);
    }
}
