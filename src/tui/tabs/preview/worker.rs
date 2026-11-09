//! Generation off the render thread. One thread loads the model, then serves requests:
//! the latest wins, the previous one is cancelled through its own flag, and the replies
//! carry the `gen_id` of their request so the state can drop stale ones. The baseline
//! (alpha = 0) of a (prompt, template) pair is computed once and replayed from a cache.
//!
//! [`Session`] is the synchronous core (also used by `--snapshot`, without any thread) and
//! runs on any [`PreviewModel`], so tests drive it with a mock.

use std::collections::HashMap;
use std::ops::ControlFlow;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokenizers::Tokenizer;

use crate::extract::generate::{eos_ids_from_dir, GenRequest, Qwen2State, Steering, TokenSource};
use crate::extract::{load_tokenizer, tokenize, ChatTemplate, DTypeChoice, ExtractError};
use crate::hardware::{GpuInfo, HardwareProbe, MemoryInfo};

/// Prompts longer than this are truncated by the tokenizer step.
pub const MAX_PROMPT_TOKENS: usize = 512;
/// (prompt, template) pairs kept; the cache is dropped when it grows past this.
const CACHE_ENTRIES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pane {
    Baseline,
    Steered,
}

/// `alpha` is relative: the worker scales it by the mean residual norm of the prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub gen_id: u64,
    pub prompt: String,
    pub alpha: f64,
    pub tokens: usize,
    pub template: ChatTemplate,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// The model is in memory (`warnings`: e.g. f32 did not fit, bf16 used).
    Loaded {
        warnings: Vec<String>,
    },
    LoadFailed(String),
    /// Mean residual norm at the injection layer for the prompt of `gen_id`.
    Scale {
        gen_id: u64,
        scale: f32,
    },
    /// `text` is the whole pane so far, decoded again at each token.
    Token {
        gen_id: u64,
        pane: Pane,
        text: String,
        tok_idx: usize,
        tok_per_s: f64,
    },
    Done {
        gen_id: u64,
        pane: Pane,
    },
    Error {
        gen_id: u64,
        message: String,
    },
}

/// A model that streams greedy tokens and measures the residual scale (mocked in tests).
pub trait PreviewModel: TokenSource {
    fn residual_scale(&mut self, prompt_ids: &[u32], layer: usize) -> Result<f32, ExtractError>;
}

impl PreviewModel for Qwen2State {
    fn residual_scale(&mut self, prompt_ids: &[u32], layer: usize) -> Result<f32, ExtractError> {
        self.mean_residual_norm(prompt_ids, layer)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoadError {
    /// The machine cannot hold the model: shown in the tab, nothing is wrong with the inputs.
    Insufficient(String),
    Failed(String),
}

impl LoadError {
    pub fn message(&self) -> &str {
        match self {
            LoadError::Insufficient(m) | LoadError::Failed(m) => m,
        }
    }
}

/// One finished pane.
#[derive(Debug, Clone)]
struct Streamed {
    text: String,
    tok_idx: usize,
    tok_per_s: f64,
}

/// What the baseline run leaves behind, replayed on later requests.
#[derive(Debug, Clone)]
struct Cached {
    run: Streamed,
    scale: f32,
}

type CacheKey = (String, &'static str, usize);

pub struct Session {
    model: Box<dyn PreviewModel>,
    tokenizer: Tokenizer,
    eos_ids: Vec<u32>,
    direction: Vec<f32>,
    layer: usize,
    warnings: Vec<String>,
    timed: bool,
    cache: HashMap<CacheKey, Cached>,
}

/// `decode` of every id so far, minus a trailing U+FFFD: it is the start of a multi-byte
/// character whose remaining bytes are still to come (a genuine one is lost at the end).
fn decode(tokenizer: &Tokenizer, ids: &[u32]) -> Result<String, String> {
    let text = tokenizer.decode(ids, true).map_err(|e| e.to_string())?;
    Ok(text.trim_end_matches('\u{FFFD}').to_owned())
}

impl Session {
    pub fn new(
        model: Box<dyn PreviewModel>,
        tokenizer: Tokenizer,
        eos_ids: Vec<u32>,
        direction: Vec<f32>,
        layer: usize,
    ) -> Self {
        Self {
            model,
            tokenizer,
            eos_ids,
            direction,
            layer,
            warnings: Vec::new(),
            timed: true,
            cache: HashMap::new(),
        }
    }

    pub fn with_warnings(mut self, warnings: Vec<String>) -> Self {
        self.warnings = warnings;
        self
    }

    /// Reports `tok_per_s = 0` everywhere: a snapshot must not depend on the machine speed.
    pub fn untimed(mut self) -> Self {
        self.timed = false;
        self
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Serves one request: baseline (cached), then steered. A cancelled run (flag raised)
    /// ends silently: the newer request owns the screen. Failures become `Event::Error`.
    pub fn run(&mut self, req: &Request, cancel: &AtomicBool, emit: &mut dyn FnMut(Event)) {
        if let Err(message) = self.serve(req, cancel, emit) {
            emit(Event::Error {
                gen_id: req.gen_id,
                message,
            });
        }
    }

    fn serve(
        &mut self,
        req: &Request,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Event),
    ) -> Result<(), String> {
        let text = req.template.apply(&req.prompt);
        let ids = tokenize(&self.tokenizer, &text, MAX_PROMPT_TOKENS)
            .map_err(|e| e.to_string())?
            .ids;
        let key: CacheKey = (req.prompt.clone(), req.template.as_str(), req.tokens);
        let base = match self.cache.get(&key).cloned() {
            Some(hit) => {
                let gen_id = req.gen_id;
                emit(Event::Scale {
                    gen_id,
                    scale: hit.scale,
                });
                replay(gen_id, Pane::Baseline, &hit.run, emit);
                hit
            }
            None => {
                let scale = self
                    .model
                    .residual_scale(&ids, self.layer)
                    .map_err(|e| e.to_string())?;
                emit(Event::Scale {
                    gen_id: req.gen_id,
                    scale,
                });
                let Some(run) = self.stream(Pane::Baseline, req, &ids, None, cancel, emit)? else {
                    return Ok(());
                };
                let cached = Cached { run, scale };
                if self.cache.len() >= CACHE_ENTRIES {
                    self.cache.clear();
                }
                self.cache.insert(key, cached.clone());
                cached
            }
        };
        if req.alpha == 0.0 {
            // alpha_abs = 0 injects nothing: the steered run is the baseline, bit for bit.
            replay(req.gen_id, Pane::Steered, &base.run, emit);
            return Ok(());
        }
        let steering = Steering {
            layer: self.layer,
            direction: self.direction.clone(),
            alpha_abs: (req.alpha * f64::from(base.scale)) as f32,
        };
        self.stream(Pane::Steered, req, &ids, Some(steering), cancel, emit)?;
        Ok(())
    }

    /// Streams one pane. `None` when cancelled (partial output is never cached nor closed).
    fn stream(
        &mut self,
        pane: Pane,
        req: &Request,
        ids: &[u32],
        steering: Option<Steering>,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Event),
    ) -> Result<Option<Streamed>, String> {
        let gen = GenRequest {
            prompt_ids: ids.to_vec(),
            max_new_tokens: req.tokens,
            eos_ids: self.eos_ids.clone(),
            steering,
        };
        let (tokenizer, timed) = (&self.tokenizer, self.timed);
        let started = Instant::now();
        let (mut generated, mut text, mut failure) = (Vec::new(), String::new(), None);
        let mut speed = 0.0;
        self.model
            .generate(&gen, cancel, &mut |id| {
                generated.push(id);
                match decode(tokenizer, &generated) {
                    Ok(decoded) => text = decoded,
                    Err(message) => {
                        failure = Some(message);
                        return ControlFlow::Break(());
                    }
                }
                let seconds = started.elapsed().as_secs_f64();
                if timed && seconds > 0.0 {
                    speed = generated.len() as f64 / seconds;
                }
                emit(Event::Token {
                    gen_id: req.gen_id,
                    pane,
                    text: text.clone(),
                    tok_idx: generated.len(),
                    tok_per_s: speed,
                });
                ControlFlow::Continue(())
            })
            .map_err(|e| e.to_string())?;
        if let Some(message) = failure {
            return Err(message);
        }
        if cancel.load(Ordering::Relaxed) {
            return Ok(None);
        }
        emit(Event::Done {
            gen_id: req.gen_id,
            pane,
        });
        Ok(Some(Streamed {
            text,
            tok_idx: generated.len(),
            tok_per_s: speed,
        }))
    }
}

fn replay(gen_id: u64, pane: Pane, run: &Streamed, emit: &mut dyn FnMut(Event)) {
    emit(Event::Token {
        gen_id,
        pane,
        text: run.text.clone(),
        tok_idx: run.tok_idx,
        tok_per_s: run.tok_per_s,
    });
    emit(Event::Done { gen_id, pane });
}

/// Fixed answers of the probe of the main thread, so that the loading thread owns its
/// own copy (`AppDeps::probe` is not shareable).
pub struct SnapshotProbe {
    pub ram: MemoryInfo,
    pub gpus: Vec<GpuInfo>,
}

impl HardwareProbe for SnapshotProbe {
    fn system_ram(&self) -> crate::hardware::Result<MemoryInfo> {
        Ok(self.ram.clone())
    }

    fn gpus(&self) -> crate::hardware::Result<Vec<GpuInfo>> {
        Ok(self.gpus.clone())
    }
}

/// Qwen2 weights, tokenizer and stop ids of `dir`, steering with `direction` at `layer`.
pub fn load_session(
    dir: &Path,
    probe: &dyn HardwareProbe,
    direction: Vec<f32>,
    layer: usize,
    new_tokens: usize,
) -> Result<Session, LoadError> {
    let failed = |e: ExtractError| match e {
        ExtractError::InsufficientResources(m) => LoadError::Insufficient(m),
        other => LoadError::Failed(other.to_string()),
    };
    let model = Qwen2State::load(
        dir,
        DTypeChoice::Auto,
        probe,
        MAX_PROMPT_TOKENS + new_tokens,
    )
    .map_err(failed)?;
    let tokenizer = load_tokenizer(dir).map_err(failed)?;
    let eos_ids = eos_ids_from_dir(dir, &tokenizer).map_err(failed)?;
    let warnings = model.warnings().to_vec();
    Ok(Session::new(Box::new(model), tokenizer, eos_ids, direction, layer).with_warnings(warnings))
}

/// Builds the session inside the worker thread (loading is slow).
pub type Loader = Box<dyn FnOnce() -> Result<Session, LoadError> + Send>;

struct Job {
    request: Request,
    cancel: Arc<AtomicBool>,
}

pub struct PreviewWorker {
    jobs: Option<Sender<Job>>,
    events: Receiver<Event>,
    current: Mutex<Option<Arc<AtomicBool>>>,
}

impl PreviewWorker {
    /// The thread is detached: quitting while the model loads must not wait for it.
    pub fn spawn(loader: Loader) -> Self {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let (outbox, events) = mpsc::channel();
        std::thread::spawn(move || {
            let mut session = match loader() {
                Ok(session) => {
                    let warnings = session.warnings().to_vec();
                    let _ = outbox.send(Event::Loaded { warnings });
                    session
                }
                Err(e) => {
                    let _ = outbox.send(Event::LoadFailed(e.message().to_owned()));
                    return;
                }
            };
            while let Ok(mut job) = inbox.recv() {
                // Only the latest request matters: older queued ones are already stale.
                while let Ok(newer) = inbox.try_recv() {
                    job = newer;
                }
                if job.cancel.load(Ordering::Relaxed) {
                    continue;
                }
                let gen_id = job.request.gen_id;
                let ran = catch_unwind(AssertUnwindSafe(|| {
                    session.run(&job.request, &job.cancel, &mut |event| {
                        let _ = outbox.send(event);
                    })
                }));
                if ran.is_err() {
                    let _ = outbox.send(Event::Error {
                        gen_id,
                        message: "erreur interne pendant la génération".into(),
                    });
                }
            }
        });
        Self {
            jobs: Some(jobs),
            events,
            current: Mutex::new(None),
        }
    }

    /// Queues `request` and cancels the generation in progress, if any.
    pub fn request(&self, request: Request) {
        let cancel = Arc::new(AtomicBool::new(false));
        if let Ok(mut current) = self.current.lock() {
            if let Some(previous) = current.replace(Arc::clone(&cancel)) {
                previous.store(true, Ordering::Relaxed);
            }
        }
        if let Some(jobs) = &self.jobs {
            let _ = jobs.send(Job { request, cancel });
        }
    }

    pub fn try_recv(&self) -> Option<Event> {
        self.events.try_recv().ok()
    }
}

impl Drop for PreviewWorker {
    fn drop(&mut self) {
        if let Ok(current) = self.current.lock() {
            if let Some(flag) = current.as_ref() {
                flag.store(true, Ordering::Relaxed);
            }
        }
        // Closing the channel ends the thread once the current generation stops.
        self.jobs = None;
    }
}
