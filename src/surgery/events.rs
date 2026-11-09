//! Progress events of the `.surgery` and LoRA engines, and their bridge to `ProgressSink`.

use std::ops::ControlFlow;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::app::{ProgressEvent, ProgressSink};
use crate::ui::{self, Ui};

/// Carried by the single `ProgressEvent::Surgery` variant.
#[derive(Debug, Clone, PartialEq)]
pub enum SurgeryEvent {
    /// One tensor processed (diffed, applied, merged): `done` out of `total`.
    Tensor {
        done: usize,
        total: usize,
        name: String,
    },
    /// Base tensor hashed (SHA-256), to detect unchanged tensors or check `base_sha256`.
    Hashing {
        done: usize,
        total: usize,
        name: String,
    },
    /// Blob or global hash verified while reading a `.surgery`.
    Verify {
        done: usize,
        total: usize,
        name: String,
    },
    /// Dry-run line for one tensor.
    Preview {
        name: String,
        kind: String,
        rank: usize,
        rel_error: f64,
    },
}

/// Engines report through this trait and stop at the next checkpoint on `Break`.
pub trait SurgeryObserver {
    fn event(&mut self, ev: SurgeryEvent) -> ControlFlow<()>;
}

pub struct NullObserver;

impl SurgeryObserver for NullObserver {
    fn event(&mut self, _: SurgeryEvent) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

/// Forwards to a `ProgressSink` and breaks once the interrupt flag (Ctrl-C, MCP
/// cancellation) is raised.
pub struct SinkObserver<'a> {
    pub sink: &'a mut dyn ProgressSink,
    pub cancel: &'a AtomicBool,
}

impl SurgeryObserver for SinkObserver<'_> {
    fn event(&mut self, ev: SurgeryEvent) -> ControlFlow<()> {
        self.sink.event(ProgressEvent::Surgery(ev));
        match self.cancel.load(Ordering::SeqCst) {
            true => ControlFlow::Break(()),
            false => ControlFlow::Continue(()),
        }
    }
}

/// `(label, tensor, done, total)` of the events drawn as a progress bar by the CLI sink.
pub fn counter(ev: &SurgeryEvent) -> Option<(&'static str, &str, usize, usize)> {
    match ev {
        SurgeryEvent::Tensor { done, total, name } => Some(("surgery", name, *done, *total)),
        SurgeryEvent::Hashing { done, total, name } => Some(("hash", name, *done, *total)),
        SurgeryEvent::Verify { done, total, name } => Some(("verify", name, *done, *total)),
        SurgeryEvent::Preview { .. } => None,
    }
}

/// Terminal line of an event; the per-tensor counters are drawn as a bar by the CLI sink.
pub fn render(ui: &Ui, ev: &SurgeryEvent) -> Option<String> {
    match ev {
        SurgeryEvent::Preview {
            name,
            kind,
            rank,
            rel_error,
        } => Some(format!(
            "{} {} · {} · rang {} · erreur {}",
            ui::badge(ui, "DRY-RUN"),
            ui::metric(ui, name),
            ui::dim(ui, kind),
            ui::metric(ui, rank),
            ui::metric(ui, format!("{rel_error:.4}"))
        )),
        _ => None,
    }
}
