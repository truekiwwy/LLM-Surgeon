//! A background job with a progress readout and a cancel flag, polled from the UI thread.

use std::io::{BufRead, BufReader, Read};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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

/// One line of a program's output, readable whatever it wrote: UTF-16 output (`wsl.exe`'s own
/// messages) loses its NUL bytes, invalid UTF-8 is replaced.
pub fn output_line(bytes: &[u8]) -> String {
    let kept: Vec<u8> = bytes.iter().copied().filter(|b| *b != 0).collect();
    String::from_utf8_lossy(&kept).trim().to_owned()
}

/// Runs `cmd` to its end, each line it writes (stdout and stderr) shown as the progress;
/// the cancel flag kills it. The error carries the last line written.
pub fn run_streamed(mut cmd: Command, handle: &Handle) -> Result<(), String> {
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| e.to_string())?;
    let (tx, lines) = mpsc::channel();
    let pipes: [Option<Box<dyn Read + Send>>; 2] = [
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    ];
    for pipe in pipes.into_iter().flatten() {
        let tx = tx.clone();
        std::thread::spawn(move || {
            for line in BufReader::new(pipe).split(b'\n').map_while(Result::ok) {
                let line = output_line(&line);
                if !line.is_empty() {
                    let _ = tx.send(line);
                }
            }
        });
    }
    let mut last = String::new();
    let status = loop {
        for line in lines.try_iter() {
            handle.report(&line, 0, None);
            last = line;
        }
        if handle.cancel.load(Ordering::Relaxed) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(crate::ui::i18n::t("setup.cancelled").into());
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => std::thread::sleep(Duration::from_millis(200)),
            Err(e) => return Err(e.to_string()),
        }
    };
    for line in lines.try_iter() {
        last = line;
    }
    match status.success() {
        true => Ok(()),
        false => Err(last),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_lines_read_utf16_and_utf8_alike() {
        let utf16: Vec<u8> = "Ubuntu-24.04\r"
            .encode_utf16()
            .flat_map(|u| u.to_le_bytes())
            .collect();
        assert_eq!(output_line(&utf16), "Ubuntu-24.04");
        assert_eq!(
            output_line("[2/6] ROCm 7.2.1 ".as_bytes()),
            "[2/6] ROCm 7.2.1"
        );
    }

    #[test]
    fn a_streamed_command_reports_its_lines_and_its_last_one_on_failure() {
        let shell = |script: &str| {
            let mut c = Command::new(if cfg!(windows) { "cmd" } else { "sh" });
            c.args(if cfg!(windows) {
                ["/C", script]
            } else {
                ["-c", script]
            });
            c
        };
        let task = Task::spawn(
            || {},
            move |h| run_streamed(shell("echo un&& echo deux&& exit 3"), h),
        );
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let result = loop {
            if let Some(r) = task.poll() {
                break r;
            }
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(result, Err("deux".into()));
    }
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
