//! Background jobs of the studio. A step is either a `surgeon` command line, run in
//! process through `weight_surgeon::app::run_with` (the same validation, backups and
//! rollback as the CLI, its JSON envelope as the result), or the restore of the backups an
//! apply left. Steps run in order on one thread; the first failure stops the job, and
//! cancelling raises the flag every core of the library checks between tensors, prompts
//! and chunks (`AppDeps::interrupted`).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::Instant;

use serde_json::{json, Value};
use weight_surgeon::app::{self, AppDeps};
use weight_surgeon::eval::{EvalError, SystemCommandRunner};
use weight_surgeon::hardware::SystemProbe;
use weight_surgeon::ui::{ColorMode, UiConfig};

#[derive(Debug, Clone, PartialEq)]
pub enum Task {
    /// Arguments after the program name; `--json` is added.
    Cli(Vec<String>),
    /// Copies each backup (`<name>.bak-<stamp>[-n]`) over the file it was taken from.
    Restore(Vec<PathBuf>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    /// The canvas card the step belongs to (`None`: the editor).
    pub card: Option<u32>,
    pub label: String,
    pub task: Task,
}

/// A step that ran: its exit code and envelope (`{"ok", "exit_code", "data", "error", ...}`).
#[derive(Debug, Clone)]
pub struct Finished {
    pub card: Option<u32>,
    pub label: String,
    pub code: i32,
    pub envelope: Value,
    pub seconds: f64,
}

enum Event {
    Started(usize),
    Finished(Finished),
}

/// A running (or just finished) job. [`Job::poll`] hands over the steps that completed.
pub struct Job {
    pub labels: Vec<String>,
    /// The card of each step (`None`: the editor).
    pub cards: Vec<Option<u32>>,
    /// Index of the step running now.
    pub current: usize,
    pub started: Instant,
    pub finished: bool,
    cancel: Arc<AtomicBool>,
    rx: Receiver<Event>,
}

impl Job {
    pub fn spawn(steps: Vec<Step>, wake: impl Fn() + Send + 'static) -> Self {
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel();
        let labels = steps.iter().map(|s| s.label.clone()).collect();
        let cards = steps.iter().map(|s| s.card).collect();
        let flag = Arc::clone(&cancel);
        std::thread::spawn(move || {
            let deps = deps(Arc::clone(&flag));
            for (i, step) in steps.into_iter().enumerate() {
                if flag.load(Ordering::SeqCst) || tx.send(Event::Started(i)).is_err() {
                    break;
                }
                wake();
                let started = Instant::now();
                let (code, envelope) = match &step.task {
                    Task::Cli(args) => run_cli(args, &deps),
                    Task::Restore(backups) => restore_all(backups),
                };
                let done = Finished {
                    card: step.card,
                    label: step.label,
                    code,
                    envelope,
                    seconds: started.elapsed().as_secs_f64(),
                };
                if tx.send(Event::Finished(done)).is_err() || code != 0 {
                    break;
                }
                wake();
            }
            drop(tx);
            wake();
        });
        Self {
            labels,
            cards,
            current: 0,
            started: Instant::now(),
            finished: false,
            cancel,
            rx,
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::SeqCst);
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }

    /// The steps finished since the last call; sets `finished` once the thread is done.
    pub fn poll(&mut self) -> Vec<Finished> {
        let mut out = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(Event::Started(i)) => self.current = i,
                Ok(Event::Finished(f)) => out.push(f),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    self.finished = true;
                    break;
                }
            }
        }
        out
    }

    /// Blocks until the thread is done (tests only).
    #[cfg(test)]
    pub(crate) fn wait(mut self) -> Vec<Finished> {
        let mut done = Vec::new();
        while !self.finished {
            done.extend(self.poll());
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        done.extend(self.poll());
        done
    }
}

/// The dependencies of the CLI without its Ctrl-C handler: the window keeps the default
/// terminal behavior, and the job's own flag is the cancellation.
fn deps(interrupted: Arc<AtomicBool>) -> AppDeps {
    AppDeps {
        probe: Box::new(SystemProbe::detect()),
        runner: Box::new(SystemCommandRunner),
        // Only the per-generation llama-cli backend needs it; the studio offers the others.
        generator_factory: Box::new(|_, _| {
            Err(EvalError::InvalidConfig(
                "llama-cli n'est pas proposé par surgeon-gui : choisissez native, mistralrs, \
                 hipfire ou llama-server"
                    .into(),
            ))
        }),
        interrupted,
        ui: UiConfig {
            out: ColorMode::Off,
            err: ColorMode::Off,
            err_is_tty: false,
        },
    }
}

fn run_cli(args: &[String], deps: &AppDeps) -> (i32, Value) {
    let argv = ["surgeon", "--json"]
        .into_iter()
        .map(String::from)
        .chain(args.iter().cloned());
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = match app::parse_cli(argv, &mut out, &mut err) {
        Ok(cli) => app::run_with(cli, deps, &mut out, &mut err),
        Err(code) => code,
    };
    let envelope = serde_json::from_slice(&out).unwrap_or_else(|_| {
        let text = String::from_utf8_lossy(if out.is_empty() { &err } else { &out });
        json!({ "ok": code == 0, "exit_code": code, "error": { "message": text.trim() } })
    });
    (code, envelope)
}

/// The file a backup was taken from: `<dir>/<name>.bak-<stamp>[-n]` -> `<dir>/<name>`.
pub fn original_of(backup: &Path) -> Option<PathBuf> {
    let name = backup.file_name()?.to_str()?;
    let (original, stamp) = name.rsplit_once(".bak-")?;
    let digits = stamp
        .split('-')
        .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    (digits && !original.is_empty()).then(|| backup.with_file_name(original))
}

/// shortcut: copy then rename, without the SHA-256 check of `BackupGuard::rollback` (not
/// public for a committed backup); add it if backups ever come from untrusted places.
fn restore_all(backups: &[PathBuf]) -> (i32, Value) {
    let mut restored = Vec::new();
    for backup in backups {
        let result = original_of(backup)
            .ok_or_else(|| format!("{} n'est pas un backup .bak-<horodatage>", backup.display()))
            .and_then(|original| {
                let mut tmp = original.clone().into_os_string();
                tmp.push(".ws-restore");
                std::fs::copy(backup, &tmp)
                    .and_then(|_| std::fs::rename(&tmp, &original))
                    .map(|()| original)
                    .map_err(|e| format!("{} : {e}", backup.display()))
            });
        match result {
            Ok(original) => restored.push(original),
            Err(message) => {
                let envelope = json!({ "ok": false, "exit_code": 1,
                    "data": { "restored": restored }, "error": { "message": message } });
                return (1, envelope);
            }
        }
    }
    (
        0,
        json!({ "ok": true, "exit_code": 0, "data": { "restored": restored } }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_backup_name_gives_back_its_original() {
        let dir = Path::new("models");
        assert_eq!(
            original_of(&dir.join("model.safetensors.bak-1700000000")),
            Some(dir.join("model.safetensors"))
        );
        assert_eq!(
            original_of(&dir.join("config.json.bak-1700000000-2")),
            Some(dir.join("config.json"))
        );
        for bad in ["model.safetensors", "x.bak-", "x.bak-12a", ".bak-12"] {
            assert_eq!(original_of(&dir.join(bad)), None, "{bad}");
        }
    }

    #[test]
    fn a_usage_error_still_gives_an_envelope_and_stops_the_job() {
        let steps = vec![
            Step {
                card: Some(1),
                label: "bad".into(),
                task: Task::Cli(vec!["frobnicate".into()]),
            },
            Step {
                card: Some(2),
                label: "never".into(),
                task: Task::Cli(vec!["arch".into(), "list".into()]),
            },
        ];
        let done = Job::spawn(steps, || {}).wait();
        assert_eq!(done.len(), 1, "the failure stops the job");
        assert_eq!(done[0].code, 2);
        assert_eq!(done[0].envelope["ok"], false);
    }
}
