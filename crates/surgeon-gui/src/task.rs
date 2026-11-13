//! A background job with a progress readout and a cancel flag, polled from the UI thread.

use std::sync::atomic::AtomicBool;
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Default)]
pub struct Progress {
    pub label: String,
    pub done: u64,
    pub total: Option<u64>,
}

impl Progress {
    /// 0.0 to 1.0, `None` when the total is unknown.
    pub fn fraction(&self) -> Option<f32> {
        self.total
            .filter(|t| *t > 0)
            .map(|t| (self.done as f64 / t as f64).clamp(0.0, 1.0) as f32)
    }
}

/// What the job body gets: the flag to honor and where to report.
#[derive(Clone)]
pub struct Handle {
    pub cancel: Arc<AtomicBool>,
    progress: Arc<Mutex<Progress>>,
}

impl Handle {
    pub fn report(&self, label: &str, done: u64, total: Option<u64>) {
        let mut p = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        p.label.clear();
        p.label.push_str(label);
        p.done = done;
        p.total = total;
    }
}

pub struct Task<T> {
    rx: Receiver<T>,
    handle: Handle,
}

impl<T: Send + 'static> Task<T> {
    /// Runs `job` on a thread; `wake` asks the UI to repaint when it ends.
    pub fn spawn(
        wake: impl Fn() + Send + 'static,
        job: impl FnOnce(&Handle) -> T + Send + 'static,
    ) -> Self {
        let handle = Handle {
            cancel: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(Mutex::new(Progress::default())),
        };
        let (tx, rx) = mpsc::channel();
        let worker = handle.clone();
        std::thread::spawn(move || {
            let _ = tx.send(job(&worker));
            wake();
        });
        Self { rx, handle }
    }

    /// The result, once, when the job is done.
    pub fn poll(&self) -> Option<T> {
        self.rx.try_recv().ok()
    }

    pub fn progress(&self) -> Progress {
        self.handle
            .progress
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub fn cancel(&self) {
        self.handle
            .cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn a_job_reports_progress_and_returns() {
        let task = Task::spawn(
            || {},
            |h| {
                h.report("step", 5, Some(10));
                42
            },
        );
        let start = Instant::now();
        let result = loop {
            if let Some(r) = task.poll() {
                break r;
            }
            assert!(start.elapsed() < Duration::from_secs(5));
            std::thread::sleep(Duration::from_millis(5));
        };
        assert_eq!(result, 42);
        let p = task.progress();
        assert_eq!((p.label.as_str(), p.fraction()), ("step", Some(0.5)));
    }
}
