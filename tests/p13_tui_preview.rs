//! S7: the live steering preview tab. Pure `update`, line editor, debounce on an injected
//! clock, stale replies and cancellation, the generation worker on a mock token source, the
//! tab rendering, the two-tab behaviour without `--preview`, and `--snapshot --preview`
//! end to end on a fake Qwen2 mini-model.

use std::collections::VecDeque;
use std::io;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{Event as TermEvent, KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;
use tokenizers::decoders::byte_level::ByteLevel;
use tokenizers::models::wordlevel::WordLevel;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::{AddedToken, Tokenizer};
use weight_surgeon::app::{self, exit, AppDeps};
use weight_surgeon::cli::{Commands, TuiArgs};
use weight_surgeon::eval::{CommandOutput, CommandRunner, EvalError};
use weight_surgeon::extract::generate::{
    eos_ids_from_dir, GenRequest, GenStats, Qwen2State, Steering, TokenSource,
};
use weight_surgeon::extract::{
    encode, load_tokenizer, parse_config, required_tensors_full, tokenize, ChatTemplate,
    DTypeChoice, ExtractError,
};
use weight_surgeon::hardware::{GpuInfo, HardwareProbe, MemoryInfo};
use weight_surgeon::tui::tabs::preview::worker::{load_session, PreviewModel};
use weight_surgeon::tui::tabs::preview::{
    cli::generate_in_place, prepare, update as preview_update, Clock, Event, Loader, ModelStatus,
    Pane, PaneStatus, PreviewConfig, PreviewEffect, PreviewMsg, PreviewState, PreviewWorker,
    Request, Session, DEBOUNCE_MS, NOTE,
};
use weight_surgeon::tui::terminal::run_loop_with;
use weight_surgeon::tui::{
    handle_key, load, render_snapshot, update, App, EventSource, Msg, Tab, Theme, VectorView,
};
use weight_surgeon::ui::{ColorMode, UiConfig};

const HIDDEN: usize = 16;
const VOCAB: usize = 64;
const LAYER: usize = 2;

// ------------------------------------------------------------------ fixtures

struct Lcg(u64);

impl Lcg {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    }
}

struct FakeProbe(u64);

impl HardwareProbe for FakeProbe {
    fn system_ram(&self) -> weight_surgeon::hardware::Result<MemoryInfo> {
        Ok(MemoryInfo {
            total_bytes: self.0,
            available_bytes: self.0,
        })
    }
    fn gpus(&self) -> weight_surgeon::hardware::Result<Vec<GpuInfo>> {
        Ok(vec![])
    }
}

struct NullRunner;

impl CommandRunner for NullRunner {
    fn run(
        &self,
        _program: &Path,
        _args: &[String],
        _cwd: Option<&Path>,
    ) -> weight_surgeon::eval::Result<CommandOutput> {
        Ok(CommandOutput {
            status_code: 0,
            stdout: String::new(),
            stderr: String::new(),
        })
    }
}

fn deps(available: u64) -> AppDeps {
    AppDeps {
        probe: Box::new(FakeProbe(available)),
        runner: Box::new(NullRunner),
        generator_factory: Box::new(|_, _| Err(EvalError::InvalidConfig("unused".into()))),
        interrupted: Arc::new(AtomicBool::new(false)),
        ui: UiConfig {
            out: ColorMode::Plain,
            err: ColorMode::Plain,
            err_is_tty: false,
        },
    }
}

const GIB64: u64 = 64 << 30;

fn run(deps: &AppDeps, args: &[&str]) -> (i32, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let argv = std::iter::once("weight-surgeon").chain(args.iter().copied());
    let code = match app::parse_cli(argv, &mut out, &mut err) {
        Ok(cli) => app::run_with(cli, deps, &mut out, &mut err),
        Err(code) => code,
    };
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

fn tui_args(args: &[&str]) -> TuiArgs {
    let argv = std::iter::once("weight-surgeon").chain(args.iter().copied());
    match app::parse_cli(argv, &mut Vec::new(), &mut Vec::new()) {
        Ok(cli) => match cli.command {
            Commands::Tui(args) => args,
            _ => panic!("not a tui command"),
        },
        Err(code) => panic!("parse failed: {code}"),
    }
}

fn s(path: &Path) -> &str {
    path.to_str().unwrap()
}

fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        values
            .iter()
            .for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = serde_json::to_string(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

const CONFIG: &str = r#"{"architectures":["Qwen2ForCausalLM"],"model_type":"qwen2","vocab_size":64,
    "hidden_size":16,"intermediate_size":32,"num_hidden_layers":4,"num_attention_heads":2,
    "num_key_value_heads":2,"max_position_embeddings":128,"rms_norm_eps":1e-6,
    "rope_theta":10000.0,"tie_word_embeddings":false,"sliding_window":128,
    "hidden_act":"silu","torch_dtype":"float32"}"#;

/// 64 words so that every id the model can emit decodes: `w5` .. `w63` after the specials.
fn write_fake_tokenizer(path: &Path) {
    let words: Vec<String> = ["[UNK]", "<|im_start|>", "<|im_end|>", "user", "assistant"]
        .iter()
        .map(|w| w.to_string())
        .chain((5..VOCAB).map(|i| format!("w{i}")))
        .collect();
    let model = WordLevel::builder()
        .vocab(
            words
                .into_iter()
                .enumerate()
                .map(|(i, w)| (w, i as u32))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer
        .add_special_tokens([
            AddedToken::from("<|im_start|>", true),
            AddedToken::from("<|im_end|>", true),
        ])
        .unwrap();
    tokenizer.save(path, false).unwrap();
}

/// Fake Qwen2 mini-model directory: config, tokenizer, one SafeTensors file.
fn write_model(dir: &Path) -> PathBuf {
    let model = dir.join("qwen2-mini");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), CONFIG).unwrap();
    write_fake_tokenizer(&model.join("tokenizer.json"));
    let cfg = parse_config(CONFIG).unwrap();
    let mut rng = Lcg(42);
    let tensors: Vec<_> = required_tensors_full(&cfg)
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let values = (0..n)
                .map(|_| match name.contains("norm") {
                    true => 1.0 + rng.next(0.1),
                    false => rng.next(0.4),
                })
                .collect();
            (name, shape, values)
        })
        .collect();
    write_safetensors(&model.join("model.safetensors"), &tensors);
    model
}

fn direction() -> Vec<f32> {
    let mut rng = Lcg(99);
    (0..HIDDEN).map(|_| rng.next(1.0)).collect()
}

/// Direction file as `surgeon extract` writes it (the layer is in the metadata).
fn write_vector(dir: &Path) -> PathBuf {
    let metadata = [("layer", LAYER.to_string()), ("site", "resid_post".into())]
        .into_iter()
        .map(|(k, v)| (k.to_owned(), v))
        .collect();
    let path = dir.join("direction.safetensors");
    std::fs::write(&path, encode(&direction(), &metadata).unwrap()).unwrap();
    path
}

struct Fixture {
    _dir: tempfile::TempDir,
    model: PathBuf,
    vector: PathBuf,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let model = write_model(dir.path());
    let vector = write_vector(dir.path());
    Fixture {
        _dir: dir,
        model,
        vector,
    }
}

// ------------------------------------------------------------------ mock model

#[derive(Debug, Default)]
struct FakeClock(AtomicU64);

impl Clock for FakeClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

/// Everything the tab asked of the model.
#[derive(Default)]
struct Spy {
    generates: Mutex<Vec<GenRequest>>,
    finished: AtomicUsize,
    cancelled: AtomicUsize,
    scales: AtomicUsize,
}

impl Spy {
    fn calls(&self) -> usize {
        self.generates.lock().unwrap().len() + self.scales.load(Ordering::SeqCst)
    }

    fn requests(&self, steered: bool) -> Vec<GenRequest> {
        self.generates
            .lock()
            .unwrap()
            .iter()
            .filter(|r| r.steering.is_some() == steered)
            .cloned()
            .collect()
    }
}

const SCALE: f32 = 2.0;

/// Emits `script` (steered or not), `delay` apart, honouring the cancel flag as the real
/// engine does: a cancelled run is `Ok` with the stats so far.
struct Mock {
    script: Vec<u32>,
    delay: Duration,
    spy: Arc<Spy>,
}

impl TokenSource for Mock {
    fn generate(
        &mut self,
        req: &GenRequest,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(u32) -> ControlFlow<()>,
    ) -> Result<GenStats, ExtractError> {
        self.spy.generates.lock().unwrap().push(req.clone());
        let mut generated = 0;
        for &id in self.script.iter().take(req.max_new_tokens) {
            if cancel.load(Ordering::Relaxed) {
                self.spy.cancelled.fetch_add(1, Ordering::SeqCst);
                break;
            }
            std::thread::sleep(self.delay);
            generated += 1;
            if on_token(id).is_break() {
                break;
            }
        }
        self.spy.finished.fetch_add(1, Ordering::SeqCst);
        Ok(GenStats {
            generated,
            ..GenStats::default()
        })
    }
}

impl PreviewModel for Mock {
    fn residual_scale(&mut self, _ids: &[u32], _layer: usize) -> Result<f32, ExtractError> {
        self.spy.scales.fetch_add(1, Ordering::SeqCst);
        Ok(SCALE)
    }
}

/// `a`, then the two byte-level pieces of `é`, then `a`.
const MULTIBYTE: [u32; 4] = [1, 4, 5, 1];

fn mock_tokenizer() -> Tokenizer {
    let vocab: Vec<(&str, u32)> = vec![
        ("[UNK]", 0),
        ("a", 1),
        ("b", 2),
        ("c", 3),
        ("Ã", 4),
        ("©", 5),
    ];
    let model = WordLevel::builder()
        .vocab(vocab.into_iter().map(|(w, i)| (w.to_owned(), i)).collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer.with_decoder(Some(ByteLevel::default()));
    tokenizer
}

fn mock_session(spy: &Arc<Spy>, script: &[u32], delay: Duration) -> Session {
    let model = Mock {
        script: script.to_vec(),
        delay,
        spy: Arc::clone(spy),
    };
    Session::new(
        Box::new(model),
        mock_tokenizer(),
        vec![],
        vec![0.5; 4],
        LAYER,
    )
}

fn mock_loader(spy: &Arc<Spy>, script: &[u32], delay: Duration) -> Loader {
    let (spy, script) = (Arc::clone(spy), script.to_vec());
    Box::new(move || Ok(mock_session(&spy, &script, delay)))
}

fn config(prompt: &str, alpha: f64) -> PreviewConfig {
    PreviewConfig {
        prompt: prompt.into(),
        alpha,
        alpha_range: (-2.0, 2.0),
        tokens: 8,
        layer: LAYER,
        template: ChatTemplate::NoTemplate,
    }
}

fn request(gen_id: u64, prompt: &str, alpha: f64, template: ChatTemplate) -> Request {
    Request {
        gen_id,
        prompt: prompt.into(),
        alpha,
        tokens: 8,
        template,
    }
}

fn run_session(session: &mut Session, req: &Request) -> Vec<Event> {
    let mut events = Vec::new();
    session.run(req, &AtomicBool::new(false), &mut |e| events.push(e));
    events
}

fn key(code: KeyCode) -> PreviewMsg {
    PreviewMsg::Key(KeyEvent::new(code, KeyModifiers::NONE))
}

fn plain() -> Theme {
    Theme::new(ColorMode::Plain)
}

/// The tab as the inspector would show it, on an app that has a vector (so that it is not
/// the welcome screen).
fn app_with_preview(fx: &Fixture, state: PreviewState) -> App {
    let vector = VectorView::load(&fx.vector).unwrap();
    let mut app = App::new(None, Some(vector), plain(), (100, 32));
    app.preview = Some(state);
    app.tab = Tab::Preview;
    app
}

fn frame_text(app: &App) -> String {
    render_snapshot(app, 100, 32).unwrap()
}

// ------------------------------------------------------------------ editor and slider

#[test]
fn editor_handles_unicode_through_pure_update() {
    let clock = Arc::new(FakeClock::default());
    let mut state = PreviewState::new(config("", 0.0), clock);
    preview_update(&mut state, key(KeyCode::Char('i')));
    for c in "héllo 日本e\u{301}".chars() {
        preview_update(&mut state, key(KeyCode::Char(c)));
    }
    assert_eq!(state.editor.text(), "héllo 日本e\u{301}");
    // "e" + combining acute leaves as one cluster, a wide char as one.
    preview_update(&mut state, key(KeyCode::Backspace));
    assert_eq!(state.editor.text(), "héllo 日本");
    preview_update(&mut state, key(KeyCode::Left));
    preview_update(&mut state, key(KeyCode::Backspace));
    assert_eq!(state.editor.text(), "héllo 本");
    preview_update(&mut state, key(KeyCode::Home));
    preview_update(&mut state, key(KeyCode::Delete));
    assert_eq!(state.editor.text(), "éllo 本");
    preview_update(&mut state, key(KeyCode::End));
    preview_update(&mut state, key(KeyCode::Left));
    preview_update(&mut state, key(KeyCode::Left));
    preview_update(
        &mut state,
        PreviewMsg::Key(KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL)),
    );
    assert_eq!(state.editor.text(), " 本", "Ctrl-U kills before the cursor");
    preview_update(&mut state, key(KeyCode::Esc));
    assert!(!state.input_focus);
}

#[test]
fn slider_is_bounded_with_coarse_and_fine_steps() {
    let clock = Arc::new(FakeClock::default());
    let mut state = PreviewState::new(config("p", 0.0), clock);
    let shift = |code| PreviewMsg::Key(KeyEvent::new(code, KeyModifiers::SHIFT));
    preview_update(&mut state, key(KeyCode::Right));
    assert_eq!(state.alpha, 0.05);
    preview_update(&mut state, shift(KeyCode::Right));
    assert_eq!(state.alpha, 0.06);
    preview_update(&mut state, key(KeyCode::Left));
    preview_update(&mut state, key(KeyCode::Left));
    assert_eq!(state.alpha, -0.04, "crosses zero without float drift");
    for _ in 0..200 {
        preview_update(&mut state, key(KeyCode::Left));
    }
    assert_eq!(state.alpha, -2.0);
    for _ in 0..500 {
        preview_update(&mut state, shift(KeyCode::Right));
    }
    assert_eq!(state.alpha, 2.0, "bounded by --alpha-range");
    preview_update(&mut state, key(KeyCode::Char('0')));
    assert_eq!(state.alpha, 0.0);
}

#[test]
fn debounce_follows_the_injected_clock() {
    let clock = Arc::new(FakeClock::default());
    let mut state = PreviewState::new(config("p", 0.0), clock.clone());
    state.start();
    preview_update(&mut state, key(KeyCode::Right));
    for now in [0, 100, DEBOUNCE_MS - 1] {
        clock.0.store(now, Ordering::SeqCst);
        assert!(preview_update(&mut state, PreviewMsg::Tick).is_empty());
    }
    preview_update(&mut state, key(KeyCode::Right));
    clock.0.store(DEBOUNCE_MS + 100, Ordering::SeqCst);
    assert!(
        preview_update(&mut state, PreviewMsg::Tick).is_empty(),
        "the second move restarted the wait"
    );
    clock.0.store(2 * DEBOUNCE_MS, Ordering::SeqCst);
    let effects = preview_update(&mut state, PreviewMsg::Tick);
    let [PreviewEffect::Generate(request)] = effects.as_slice() else {
        panic!("one request expected, got {effects:?}");
    };
    assert_eq!((request.alpha, request.gen_id), (0.1, 2));
}

// ------------------------------------------------------------------ worker

#[test]
fn baseline_is_computed_once_per_prompt_and_template() {
    let spy = Arc::new(Spy::default());
    let mut session = mock_session(&spy, &[1, 2, 3], Duration::ZERO);
    let chatml = ChatTemplate::Chatml;
    let first = run_session(&mut session, &request(1, "a b", 0.5, chatml));
    assert!(first.contains(&Event::Scale {
        gen_id: 1,
        scale: SCALE
    }));
    let steered = spy.requests(true);
    assert_eq!(steered.len(), 1);
    let steering = steered[0].steering.as_ref().unwrap();
    assert_eq!(
        (steering.layer, steering.alpha_abs),
        (LAYER, 0.5 * SCALE),
        "alpha_abs = alpha * s"
    );

    // Another alpha: the baseline is replayed from the cache, only the steered pane runs.
    let second = run_session(&mut session, &request(2, "a b", 0.8, chatml));
    assert_eq!(
        (spy.requests(false).len(), spy.requests(true).len()),
        (1, 2)
    );
    assert_eq!(spy.scales.load(Ordering::SeqCst), 1);
    let baseline_text = |events: &[Event]| {
        events.iter().rev().find_map(|e| match e {
            Event::Token {
                pane: Pane::Baseline,
                text,
                ..
            } => Some(text.clone()),
            _ => None,
        })
    };
    assert_eq!(baseline_text(&first), baseline_text(&second));
    assert!(
        baseline_text(&second).is_some(),
        "the replay reaches the pane"
    );

    // alpha = 0: the steered pane is the baseline, no new run at all.
    let zero = run_session(&mut session, &request(3, "a b", 0.0, chatml));
    assert_eq!(
        (spy.requests(false).len(), spy.requests(true).len()),
        (1, 2)
    );
    assert!(zero.contains(&Event::Done {
        gen_id: 3,
        pane: Pane::Steered
    }));

    // A new template or a new prompt is a new baseline.
    run_session(
        &mut session,
        &request(4, "a b", 0.0, ChatTemplate::NoTemplate),
    );
    run_session(
        &mut session,
        &request(5, "b c", 0.0, ChatTemplate::NoTemplate),
    );
    assert_eq!(spy.requests(false).len(), 3);
}

#[test]
fn multibyte_character_is_never_shown_half_decoded() {
    let spy = Arc::new(Spy::default());
    let mut session = mock_session(&spy, &MULTIBYTE, Duration::ZERO);
    let events = run_session(
        &mut session,
        &request(1, "a", 0.5, ChatTemplate::NoTemplate),
    );
    let texts: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            Event::Token {
                pane: Pane::Steered,
                text,
                ..
            } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        texts,
        ["a", "a", "aé", "aéa"],
        "the pending bytes stay hidden"
    );

    // Same through the whole tab: no frame ever contains U+FFFD.
    let fx = fixture();
    let mut state = PreviewState::new(config("a", 0.5), Arc::new(FakeClock::default()));
    state.start();
    let mut app = app_with_preview(&fx, state);
    for event in events {
        update(&mut app, Msg::Preview(PreviewMsg::Worker(event)));
        assert!(!frame_text(&app).contains('\u{FFFD}'));
    }
    let preview = app.preview.as_ref().unwrap();
    assert_eq!(preview.steered.text, "aéa");
    assert_eq!(preview.steered.status, PaneStatus::Done);
}

fn recv_until(
    worker: &PreviewWorker,
    seen: &mut Vec<Event>,
    done: impl Fn(&Event) -> bool,
) -> Event {
    let deadline = Instant::now() + Duration::from_secs(20);
    while Instant::now() < deadline {
        match worker.try_recv() {
            Some(event) => {
                seen.push(event.clone());
                if done(&event) {
                    return event;
                }
            }
            None => std::thread::sleep(Duration::from_millis(1)),
        }
    }
    panic!("timeout, events so far: {seen:?}");
}

#[test]
fn a_new_request_cancels_the_previous_one_and_stale_replies_are_ignored() {
    let spy = Arc::new(Spy::default());
    let slow = Duration::from_millis(4);
    let worker = PreviewWorker::spawn(mock_loader(&spy, &[1, 2, 3, 1, 2, 3, 1, 2], slow));
    let mut state = PreviewState::new(config("a b", 0.5), Arc::new(FakeClock::default()));
    let mut seen = Vec::new();
    let send = |effects: Vec<PreviewEffect>| {
        for PreviewEffect::Generate(request) in effects {
            worker.request(request);
        }
    };
    send(state.start());
    // The first request is in the baseline pane when the second arrives.
    recv_until(&worker, &mut seen, |e| {
        matches!(
            e,
            Event::Token {
                gen_id: 1,
                tok_idx: 2,
                ..
            }
        )
    });
    send(preview_update(&mut state, key(KeyCode::Char('r'))));
    recv_until(&worker, &mut seen, |e| {
        matches!(
            e,
            Event::Done {
                gen_id: 2,
                pane: Pane::Steered
            }
        )
    });
    assert!(
        !seen
            .iter()
            .any(|e| matches!(e, Event::Done { gen_id: 1, .. })),
        "a cancelled run is never closed: {seen:?}"
    );
    assert!(
        spy.cancelled.load(Ordering::SeqCst) >= 1,
        "the engine saw the flag"
    );
    let first_new = seen
        .iter()
        .position(|e| matches!(e, Event::Scale { gen_id: 2, .. }))
        .unwrap();
    assert!(
        seen[first_new..].iter().all(|e| !matches!(
            e,
            Event::Token { gen_id: 1, .. } | Event::Scale { gen_id: 1, .. }
        )),
        "no reply of the old request after the new one started"
    );
    // The state, fed with everything including the old request's tokens, shows the new one.
    for event in seen {
        preview_update(&mut state, PreviewMsg::Worker(event));
    }
    assert_eq!(state.steered.status, PaneStatus::Done);
    assert_eq!(state.steered.text, "abcabcab");
    assert_eq!(state.baseline.text, "abcabcab");
}

#[test]
fn view_never_calls_the_worker() {
    let fx = fixture();
    let spy = Arc::new(Spy::default());
    let worker = PreviewWorker::spawn(mock_loader(&spy, &[1, 2], Duration::ZERO));
    let mut state = PreviewState::new(config("a", 0.5), Arc::new(FakeClock::default()));
    state.start();
    let app = app_with_preview(&fx, state);
    for _ in 0..20 {
        frame_text(&app);
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(spy.calls(), 0, "drawing asks the model for nothing");
    drop(worker);
}

#[test]
fn model_failure_is_a_message_in_the_tab_only() {
    let worker = PreviewWorker::spawn(Box::new(|| {
        Err(
            weight_surgeon::tui::tabs::preview::worker::LoadError::Failed(
                "tokenizer.json introuvable".into(),
            ),
        )
    }));
    let mut seen = Vec::new();
    let event = recv_until(&worker, &mut seen, |e| matches!(e, Event::LoadFailed(_)));
    let mut state = PreviewState::new(config("a", 0.0), Arc::new(FakeClock::default()));
    preview_update(&mut state, PreviewMsg::Worker(event));
    assert_eq!(
        state.model,
        ModelStatus::Unavailable("tokenizer.json introuvable".into())
    );
    assert!(preview_update(&mut state, key(KeyCode::Char('r'))).is_empty());
}

// ------------------------------------------------------------------ tab and keys

fn tab_text_with_events(fx: &Fixture) -> String {
    let mut state = PreviewState::new(
        PreviewConfig {
            prompt: "Raconte-moi une courte histoire.".into(),
            alpha: 0.5,
            alpha_range: (-2.0, 2.0),
            tokens: 30,
            layer: LAYER,
            template: ChatTemplate::Chatml,
        },
        Arc::new(FakeClock::default()),
    );
    state.start();
    let mut app = app_with_preview(fx, state);
    let tokens = |pane, text: &str, n| {
        Msg::Preview(PreviewMsg::Worker(Event::Token {
            gen_id: 1,
            pane,
            text: text.into(),
            tok_idx: n,
            tok_per_s: 0.0,
        }))
    };
    let events = [
        Msg::Preview(PreviewMsg::Worker(Event::Loaded { warnings: vec![] })),
        Msg::Preview(PreviewMsg::Worker(Event::Scale {
            gen_id: 1,
            scale: 3.125,
        })),
        tokens(Pane::Baseline, "Il etait une fois un petit robot.", 8),
        Msg::Preview(PreviewMsg::Worker(Event::Done {
            gen_id: 1,
            pane: Pane::Baseline,
        })),
        tokens(Pane::Steered, "Il etait une fois un petit pirate.", 8),
        Msg::Preview(PreviewMsg::Worker(Event::Done {
            gen_id: 1,
            pane: Pane::Steered,
        })),
    ];
    for event in events {
        update(&mut app, event);
    }
    frame_text(&app)
}

#[test]
fn tab_matches_the_golden_frame_at_100x32() {
    let fx = fixture();
    let text = tab_text_with_events(&fx);
    // The last row is the global status bar, which prints the temporary vector path.
    let body: String = text.lines().take(31).map(|l| format!("{l}\n")).collect();
    let golden = include_str!("fixtures/p13/preview_tab_100x32.txt").replace("\r\n", "\n");
    assert_eq!(body, golden, "full frame:\n{text}");
    assert!(!text.contains('\x1b'));
}

#[test]
fn without_preview_the_tui_keeps_two_tabs() {
    let mut app = App::new(None, None, plain(), (100, 32));
    let press = |app: &mut App, code| {
        handle_key(app, KeyEvent::new(code, KeyModifiers::NONE));
    };
    let start = app.tab;
    press(&mut app, KeyCode::Tab);
    press(&mut app, KeyCode::Tab);
    assert_eq!(app.tab, start, "Tab alternates between two tabs");
    press(&mut app, KeyCode::BackTab);
    press(&mut app, KeyCode::BackTab);
    assert_eq!(app.tab, start);
    assert!(app.preview.is_none());

    let fx = fixture();
    let vector = VectorView::load(&fx.vector).unwrap();
    let app = App::new(None, Some(vector), plain(), (100, 32));
    let text = frame_text(&app);
    assert!(text.contains("1 Inspecteur de poids") && text.contains("2 Steering vector"));
    assert!(!text.contains("3 Preview live"));
}

#[test]
fn with_preview_tab_cycles_three_tabs_and_tab_stays_global_in_the_input() {
    let fx = fixture();
    let state = PreviewState::new(config("a", 0.0), Arc::new(FakeClock::default()));
    let mut app = app_with_preview(&fx, state);
    app.tab = Tab::Inspector;
    let press = |app: &mut App, code| {
        handle_key(app, KeyEvent::new(code, KeyModifiers::NONE));
    };
    press(&mut app, KeyCode::Tab);
    press(&mut app, KeyCode::Tab);
    assert_eq!(app.tab, Tab::Preview);
    press(&mut app, KeyCode::Tab);
    assert_eq!(app.tab, Tab::Inspector, "the cycle closes after three tabs");
    press(&mut app, KeyCode::BackTab);
    assert_eq!(app.tab, Tab::Preview, "Shift+Tab goes back");

    // In the input, q and Esc are not quit keys, but Tab still changes tab.
    press(&mut app, KeyCode::Char('i'));
    press(&mut app, KeyCode::Char('q'));
    assert!(!app.should_quit);
    assert!(app.preview.as_ref().unwrap().editor.text().ends_with('q'));
    press(&mut app, KeyCode::Tab);
    assert_eq!(app.tab, Tab::Inspector);
    press(&mut app, KeyCode::BackTab);
    press(&mut app, KeyCode::Esc);
    assert!(!app.should_quit, "Esc only leaves the input");
    press(&mut app, KeyCode::Esc);
    assert!(app.should_quit, "outside the input, Esc quits");
}

// ------------------------------------------------------------------ CLI

#[test]
fn preview_requires_model_and_vector() {
    let fx = fixture();
    for args in [
        vec!["tui", "--preview"],
        vec!["tui", "--preview", "--model", s(&fx.model)],
        vec!["tui", "--preview", "--vector", s(&fx.vector)],
    ] {
        let (code, out, err) = run(&deps(GIB64), &args);
        assert_eq!(code, exit::INVALID_INPUT, "{args:?}");
        assert!(
            out.is_empty() && err.contains("--preview nécessite"),
            "{err}"
        );
    }
    let (code, _, err) = run(
        &deps(GIB64),
        &[
            "tui",
            "--snapshot",
            "--preview",
            "--model",
            s(&fx.model),
            "--vector",
            s(&fx.vector),
            "--preview-alpha",
            "5",
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT);
    assert!(err.contains("hors de --alpha-range"), "{err}");
}

fn snapshot_args<'a>(fx: &'a Fixture, extra: &[&'a str]) -> Vec<&'a str> {
    let mut args = vec![
        "tui",
        "--snapshot",
        "--preview",
        "--model",
        s(&fx.model),
        "--vector",
        s(&fx.vector),
        "--preview-prompt",
        "w10 w11 w12",
        "--preview-tokens",
        "6",
    ];
    args.extend_from_slice(extra);
    args
}

/// Text of `ids` as the tab shows it.
fn shown(tokenizer: &Tokenizer, ids: &[u32]) -> String {
    let text = tokenizer.decode(ids, true).unwrap();
    text.trim_end_matches('\u{FFFD}').to_owned()
}

#[test]
fn snapshot_preview_is_deterministic_and_matches_the_engine() {
    let fx = fixture();
    let args = snapshot_args(&fx, &["--preview-alpha", "1.5"]);
    let (code, out, err) = run(&deps(GIB64), &args);
    assert_eq!(code, exit::OK, "{err}");
    assert!(err.is_empty() && !out.contains('\x1b'));
    let (_, again, _) = run(&deps(GIB64), &args);
    assert_eq!(out, again, "two snapshots are identical");
    for needle in [
        "3 Preview live",
        "Baseline (alpha=0)",
        "Steered (alpha=+1.50)",
        "alpha = +1.50",
        "couche 2 · gabarit chatml · 6 tokens max",
        NOTE,
    ] {
        assert!(out.contains(needle), "{needle} missing in\n{out}");
    }
    assert!(
        !out.contains("tok/s"),
        "no machine-dependent speed in a snapshot"
    );

    // Independent computation with the engine.
    let tokenizer = load_tokenizer(&fx.model).unwrap();
    let ids = tokenize(&tokenizer, &ChatTemplate::Chatml.apply("w10 w11 w12"), 512)
        .unwrap()
        .ids;
    let mut model = Qwen2State::load(&fx.model, DTypeChoice::Auto, &FakeProbe(GIB64), 600).unwrap();
    let eos = eos_ids_from_dir(&fx.model, &tokenizer).unwrap();
    let mut generate = |steering: Option<Steering>| {
        let mut out_ids = Vec::new();
        let request = GenRequest {
            prompt_ids: ids.clone(),
            max_new_tokens: 6,
            eos_ids: eos.clone(),
            steering,
        };
        model
            .generate(&request, &AtomicBool::new(false), &mut |id| {
                out_ids.push(id);
                ControlFlow::Continue(())
            })
            .unwrap();
        out_ids
    };
    let baseline = shown(&tokenizer, &generate(None));
    let scale = {
        let mut m = Qwen2State::load(&fx.model, DTypeChoice::Auto, &FakeProbe(GIB64), 600).unwrap();
        m.mean_residual_norm(&ids, LAYER).unwrap()
    };
    let steered = shown(
        &tokenizer,
        &generate(Some(Steering {
            layer: LAYER,
            direction: direction(),
            alpha_abs: (1.5 * f64::from(scale)) as f32,
        })),
    );
    assert!(!baseline.is_empty() && !steered.is_empty());
    assert_ne!(baseline, steered, "a strong steering changes the text");
    assert!(
        out.contains(&baseline),
        "baseline {baseline:?} not in\n{out}"
    );
    assert!(out.contains(&steered), "steered {steered:?} not in\n{out}");
    assert!(out.contains(&format!("alpha_abs = {:+.3}", 1.5 * f64::from(scale))));
}

#[test]
fn snapshot_preview_without_enough_memory_says_so_and_keeps_tabs_one_and_two() {
    let fx = fixture();
    let (code, out, err) = run(&deps(1 << 20), &snapshot_args(&fx, &[]));
    assert_eq!(code, exit::OK, "{err}");
    assert!(out.contains("Prévisualisation indisponible"), "{out}");
    assert!(out.contains("ressources insuffisantes"), "{out}");
    assert!(!out.contains("Baseline"), "no generation attempted");

    // The same options through `prepare`: no loader, and the inspector is untouched.
    let args = tui_args(&snapshot_args(&fx, &[]));
    let (mut app, _) = load(&args, plain(), (100, 32)).unwrap();
    let launch = prepare(
        &args,
        app.vector.as_ref().unwrap(),
        &FakeProbe(1 << 20),
        Arc::new(FakeClock::default()),
    )
    .unwrap();
    assert!(launch.loader.is_none());
    assert!(matches!(launch.state.model, ModelStatus::Unavailable(_)));
    app.preview = Some(launch.state);
    app.tab = Tab::Inspector;
    let inspector = frame_text(&app);
    assert!(inspector.contains("Couches (4)"), "{inspector}");
    handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    handle_key(&mut app, KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE));
    assert_eq!(app.tab, Tab::Preview);
    assert!(frame_text(&app).contains("Prévisualisation indisponible"));
}

#[test]
fn layer_comes_from_the_vector_unless_overridden() {
    let fx = fixture();
    let make = |extra: &[&str]| {
        let args = tui_args(&snapshot_args(&fx, extra));
        let vector = VectorView::load(&fx.vector).unwrap();
        prepare(
            &args,
            &vector,
            &FakeProbe(GIB64),
            Arc::new(FakeClock::default()),
        )
        .unwrap()
        .state
        .layer
    };
    assert_eq!(make(&[]), LAYER);
    assert_eq!(make(&["--preview-layer", "1"]), 1);
}

#[test]
fn out_of_range_layer_is_an_error_in_the_tab_not_a_crash() {
    let fx = fixture();
    let (code, out, err) = run(&deps(GIB64), &snapshot_args(&fx, &["--preview-layer", "9"]));
    assert_eq!(code, exit::OK, "{err}");
    assert!(out.contains("erreur :") && out.contains("bornes"), "{out}");
}

#[test]
fn generate_in_place_reports_unusable_models() {
    let fx = fixture();
    let dir = fx.model.clone();
    let broken: Loader = Box::new(move || {
        load_session(
            &dir.join("missing"),
            &FakeProbe(GIB64),
            direction(),
            LAYER,
            4,
        )
    });
    let mut state = PreviewState::new(config("w10", 0.0), Arc::new(FakeClock::default()));
    assert!(generate_in_place(&mut state, broken).is_err());
}

// ------------------------------------------------------------------ the loop

/// Plays keys, sleeping between polls and moving the fake clock; waits on the spy so
/// that the test does not depend on thread speed.
struct Driver {
    /// (event, wait for one more finished generation before the next one)
    script: VecDeque<(TermEvent, bool)>,
    clock: Arc<FakeClock>,
    spy: Arc<Spy>,
    waiting_for: usize,
    polls: usize,
}

impl EventSource for Driver {
    fn poll(&mut self, _timeout: Duration) -> io::Result<Option<TermEvent>> {
        self.polls += 1;
        std::thread::sleep(Duration::from_millis(2));
        self.clock.0.fetch_add(100, Ordering::SeqCst);
        let finished = self.spy.finished.load(Ordering::SeqCst);
        let quit = TermEvent::Key(KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE));
        if self.polls > 3000 {
            return Ok(Some(quit));
        }
        if finished < self.waiting_for {
            return Ok(None);
        }
        Ok(match self.script.pop_front() {
            Some((event, wait)) => {
                if wait {
                    self.waiting_for = finished + 1;
                }
                Some(event)
            }
            None => Some(quit),
        })
    }
}

#[test]
fn the_loop_wires_keys_debounce_worker_and_redraws() {
    let fx = fixture();
    let spy = Arc::new(Spy::default());
    let clock = Arc::new(FakeClock::default());
    let state = PreviewState::new(config("a b", 0.0), clock.clone());
    let mut app = app_with_preview(&fx, state);
    let worker = PreviewWorker::spawn(mock_loader(&spy, &[1, 2, 3], Duration::ZERO));
    let right = TermEvent::Key(KeyEvent::new(KeyCode::Right, KeyModifiers::NONE));
    let mut driver = Driver {
        // The first request (baseline only, alpha = 0) must finish before Right is sent;
        // the two Right keys are one debounced request.
        script: VecDeque::from([(right.clone(), false), (right, true)]),
        clock: clock.clone(),
        spy: spy.clone(),
        waiting_for: 1,
        polls: 0,
    };
    let mut terminal = Terminal::new(TestBackend::new(100, 32)).unwrap();
    let stats = run_loop_with(&mut terminal, &mut app, &mut driver, None, Some(&worker)).unwrap();
    assert!(stats.draws > 3, "the tab redrew as replies arrived");
    let preview = app.preview.as_ref().unwrap();
    assert_eq!(preview.alpha, 0.1);
    assert_eq!(
        preview.steered.status,
        PaneStatus::Done,
        "{:?}",
        preview.steered
    );
    assert_eq!(preview.steered.text, "abc");
    assert_eq!(preview.baseline.text, "abc");
    let steered = spy.requests(true);
    assert_eq!(steered.len(), 1, "two key presses, one debounced request");
    assert_eq!(steered[0].steering.as_ref().unwrap().alpha_abs, 0.1 * SCALE);
    assert_eq!(spy.requests(false).len(), 1, "baseline computed once");
}
