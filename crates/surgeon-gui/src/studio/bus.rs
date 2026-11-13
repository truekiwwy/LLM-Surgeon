//! The event stream of a run. The worker writes every line to `runs/<id>/events.ndjson`
//! (the engine's `EventWriter`: numbered, flushed per line) and hands it to the window over
//! a bounded channel. Progress lines may be dropped on the way to the window when it lags
//! (the file still has them); every other line waits for room, which slows the worker
//! rather than losing it. The window folds what arrives into a [`RunView`] at most 60 times
//! a second, and a finished run is rebuilt from its file, so what it shows is exact.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use weight_surgeon::app::{ProgressEvent, ProgressSink};
use weight_surgeon::events::{
    from_progress, parse_line, EventLine, EventWriter, LogLevel, NodeStatus, SurgeonEvent,
};

pub const EVENTS_FILE: &str = "events.ndjson";
/// Lines in flight between a worker and the window.
pub const CHANNEL_LINES: usize = 1024;
/// 60 Hz.
pub const FRAME: Duration = Duration::from_micros(16_667);
/// Log lines a view keeps (the file keeps them all).
pub const MAX_LOGS: usize = 500;

/// Worker side.
pub struct Recorder {
    writer: EventWriter,
    tx: SyncSender<EventLine>,
    sampled: Arc<AtomicU64>,
}

/// Window side.
pub struct Feed {
    rx: Receiver<EventLine>,
    last: Option<Instant>,
    sampled: Arc<AtomicU64>,
    pub file: PathBuf,
}

pub fn channel(file: &Path) -> Result<(Recorder, Feed), String> {
    let writer = EventWriter::open(file).map_err(|e| format!("{} : {e}", file.display()))?;
    let (tx, rx) = mpsc::sync_channel(CHANNEL_LINES);
    let sampled = Arc::new(AtomicU64::new(0));
    let recorder = Recorder {
        writer,
        tx,
        sampled: Arc::clone(&sampled),
    };
    let feed = Feed {
        rx,
        last: None,
        sampled,
        file: file.to_path_buf(),
    };
    Ok((recorder, feed))
}

impl Recorder {
    pub fn emit(&mut self, event: SurgeonEvent) {
        let line = self.writer.emit(event);
        if matches!(line.event, SurgeonEvent::Progress { .. }) {
            if let Err(TrySendError::Full(_)) = self.tx.try_send(line) {
                self.sampled.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            // The window gone (closed project) is not an error: the file has the line.
            let _ = self.tx.send(line);
        }
    }

    /// The first write error of the file, if any.
    pub fn take_error(&mut self) -> Option<std::io::Error> {
        self.writer.take_error()
    }
}

impl ProgressSink for Recorder {
    fn event(&mut self, event: ProgressEvent) {
        for e in from_progress(&event) {
            self.emit(e);
        }
    }
}

impl Feed {
    /// Folds what arrived into `view`, unless the last fold was less than a frame ago.
    /// Returns whether anything changed.
    pub fn drain(&mut self, now: Instant, view: &mut RunView) -> bool {
        if self.last.is_some_and(|t| now.duration_since(t) < FRAME) {
            return false;
        }
        self.last = Some(now);
        let mut changed = false;
        for line in self.rx.try_iter() {
            view.apply(&line);
            changed = true;
        }
        view.sampled = self.sampled.load(Ordering::Relaxed);
        changed
    }
}

/// What a run shows: built the same way live and from `events.ndjson`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RunView {
    pub command: Option<String>,
    /// Latest `(done, total)` per unit.
    pub progress: BTreeMap<String, (u64, Option<u64>)>,
    pub nodes: BTreeMap<String, NodeStatus>,
    /// Latest value per `(node, name)`.
    pub metrics: BTreeMap<(Option<String>, String), f64>,
    pub logs: VecDeque<(LogLevel, String)>,
    pub artifacts: Vec<(String, PathBuf, Option<String>)>,
    pub warnings: Vec<String>,
    /// The closing line.
    pub end: Option<SurgeonEvent>,
    pub lines: u64,
    /// Sequence numbers missing between two lines read (a file never has any).
    pub gaps: u64,
    last_seq: Option<u64>,
    /// Progress lines the window skipped (live only).
    pub sampled: u64,
}

impl RunView {
    pub fn apply(&mut self, line: &EventLine) {
        if let Some(prev) = self.last_seq {
            self.gaps += line.seq.saturating_sub(prev + 1);
        }
        self.last_seq = Some(line.seq);
        self.lines += 1;
        match &line.event {
            SurgeonEvent::JobStarted { command } => self.command = Some(command.clone()),
            SurgeonEvent::Progress { done, total, unit } => {
                self.progress.insert(unit.clone(), (*done, *total));
            }
            SurgeonEvent::NodeState { node_id, state } => {
                self.nodes.insert(node_id.clone(), *state);
            }
            SurgeonEvent::Metric {
                node_id,
                name,
                value,
            } => {
                self.metrics.insert((node_id.clone(), name.clone()), *value);
            }
            SurgeonEvent::Log { level, message } => self.log(*level, message.clone()),
            SurgeonEvent::Artifact { kind, path, sha256 } => {
                self.artifacts
                    .push((kind.clone(), path.clone(), sha256.clone()));
            }
            SurgeonEvent::Warning { message } => {
                self.warnings.push(message.clone());
                self.log(LogLevel::Warn, message.clone());
            }
            end => {
                if let SurgeonEvent::Failed { message, .. } = end {
                    self.log(LogLevel::Error, message.clone());
                }
                self.end = Some(end.clone());
            }
        }
    }

    fn log(&mut self, level: LogLevel, message: String) {
        if self.logs.len() == MAX_LOGS {
            self.logs.pop_front();
        }
        self.logs.push_back((level, message));
    }

    /// Rebuilds a run from its file. A torn last line (the process died mid-write) is
    /// skipped; any other unreadable line is an error.
    pub fn replay(file: &Path) -> Result<Self, String> {
        let text =
            std::fs::read_to_string(file).map_err(|e| format!("{} : {e}", file.display()))?;
        let mut view = Self::default();
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        let torn_tail = !text.ends_with('\n');
        for (i, line) in lines.iter().enumerate() {
            match parse_line(line) {
                Ok(parsed) => view.apply(&parsed),
                Err(_) if torn_tail && i + 1 == lines.len() => {}
                Err(e) => return Err(format!("{} ligne {} : {e}", file.display(), i + 1)),
            }
        }
        Ok(view)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn progress(done: u64) -> SurgeonEvent {
        SurgeonEvent::Progress {
            done,
            total: Some(10_000),
            unit: "tensor".into(),
        }
    }

    #[test]
    fn the_file_keeps_every_line_while_a_lagging_window_samples_progress() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(EVENTS_FILE);
        let (mut rec, mut feed) = channel(&file).unwrap();
        // The worker runs ahead of a window that does not drain.
        let worker = std::thread::spawn(move || {
            rec.emit(SurgeonEvent::JobStarted {
                command: "apply".into(),
            });
            for i in 1..=10_000 {
                rec.emit(progress(i));
            }
            rec.emit(SurgeonEvent::Finished { exit_code: 0 });
        });
        // Without a drain the channel fills up and progress starts being dropped.
        while feed.sampled.load(Ordering::Relaxed) == 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
        // The terminal line needs room: drain until the worker is done.
        let mut live = RunView::default();
        let mut t = Instant::now();
        while !worker.is_finished() {
            t += FRAME;
            feed.drain(t, &mut live);
        }
        worker.join().unwrap();
        t += FRAME;
        feed.drain(t, &mut live);
        let replayed = RunView::replay(&file).unwrap();
        assert_eq!(replayed.lines, 10_002, "nothing lost in the file");
        assert_eq!(replayed.gaps, 0);
        assert_eq!(replayed.progress["tensor"], (10_000, Some(10_000)));
        assert_eq!(
            live.lines + live.sampled,
            10_002,
            "skipped lines are counted"
        );
        assert!(live.sampled > 0, "the window was sampled");
        assert_eq!(live.end, replayed.end);
    }

    #[test]
    fn the_window_folds_at_most_once_per_frame() {
        let dir = tempfile::tempdir().unwrap();
        let (mut rec, mut feed) = channel(&dir.path().join(EVENTS_FILE)).unwrap();
        let mut view = RunView::default();
        let t0 = Instant::now();
        rec.emit(progress(1));
        assert!(feed.drain(t0, &mut view));
        rec.emit(progress(2));
        assert!(!feed.drain(t0 + FRAME / 2, &mut view), "same frame");
        assert_eq!(view.progress["tensor"].0, 1);
        assert!(feed.drain(t0 + FRAME, &mut view));
        assert_eq!(view.progress["tensor"].0, 2);
        // 1000 drain attempts over one second fold at most 60 times.
        let folds = (0..1000)
            .filter(|i| {
                rec.emit(progress(*i));
                feed.drain(t0 + FRAME * 2 + Duration::from_millis(*i), &mut view)
            })
            .count();
        assert!(folds <= 61, "{folds}");
    }

    #[test]
    fn replay_is_deterministic_and_matches_the_live_fold() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(EVENTS_FILE);
        let (mut rec, mut feed) = channel(&file).unwrap();
        let events = [
            SurgeonEvent::JobStarted {
                command: "graph".into(),
            },
            SurgeonEvent::NodeState {
                node_id: "L1.down".into(),
                state: NodeStatus::Done,
            },
            SurgeonEvent::Metric {
                node_id: None,
                name: "edges".into(),
                value: 3.0,
            },
            SurgeonEvent::Warning {
                message: "w".into(),
            },
            SurgeonEvent::Failed {
                kind: "io".into(),
                message: "disque plein".into(),
                exit_code: 1,
            },
        ];
        for e in events {
            rec.emit(e);
        }
        let mut live = RunView::default();
        feed.drain(Instant::now(), &mut live);
        let a = RunView::replay(&file).unwrap();
        let b = RunView::replay(&file).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, live);
        assert_eq!(a.logs.len(), 2, "warning and failure");
    }

    #[test]
    fn a_torn_last_line_is_skipped_but_a_corrupt_middle_line_is_not() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join(EVENTS_FILE);
        let ok = r#"{"v":1,"seq":0,"t_ms":0,"event":"job_started","command":"x"}"#;
        std::fs::write(&file, format!("{ok}\n{{\"v\":1,\"se")).unwrap();
        assert_eq!(RunView::replay(&file).unwrap().lines, 1);
        std::fs::write(&file, format!("garbage\n{ok}\n")).unwrap();
        assert!(RunView::replay(&file).unwrap_err().contains("ligne 1"));
    }
}
