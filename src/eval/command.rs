use std::io::Read;
use std::ops::ControlFlow;
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::Serialize;

use super::error::{EvalError, Result};

/// Each captured stream keeps only its last bytes: converters can be very chatty.
const CAPTURE_TAIL_BYTES: usize = 1 << 20;
/// How often a streamed child is checked against the timeout and the cancel flag, even
/// when it prints nothing (llama.cpp can stay silent for minutes while loading).
const STREAM_TICK: Duration = Duration::from_millis(50);
/// After a kill, how long the reader threads get to hand back their tails.
const STREAM_GRACE: Duration = Duration::from_secs(2);
const STREAM_CHANNEL_LINES: usize = 256;

/// Synchronous process runner, mockable. Implementations must never go through a shell.
pub trait CommandRunner {
    fn run(&self, program: &Path, args: &[String], cwd: Option<&Path>) -> Result<CommandOutput>;

    /// Like `run`, but every output line (stdout and stderr, in arrival order) goes to
    /// `on_line` while the child runs. `Break`, the cancel flag or the timeout kill and
    /// reap the child; the timeout is an `Err(Timeout)`, the other two return what was
    /// captured (the caller knows why it stopped). The default runs `run` and replays the
    /// lines afterwards, which is what a mock needs.
    fn run_streaming(
        &self,
        program: &Path,
        args: &[String],
        cwd: Option<&Path>,
        _control: StreamControl<'_>,
        on_line: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<CommandOutput> {
        let output = self.run(program, args, cwd)?;
        for line in output.stdout.lines().chain(output.stderr.lines()) {
            if on_line(line).is_break() {
                break;
            }
        }
        Ok(output)
    }
}

/// Stop conditions of [`CommandRunner::run_streaming`], polled every 50 ms.
#[derive(Debug, Clone, Copy, Default)]
pub struct StreamControl<'a> {
    pub timeout: Option<Duration>,
    /// Ctrl-C or an MCP cancellation.
    pub cancel: Option<&'a AtomicBool>,
}

/// Kills and reaps the child on every exit path, so no zombie survives an early return.
pub(crate) struct ChildGuard(pub(crate) Child);

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CommandOutput {
    /// -1 when the process was killed by a signal.
    pub status_code: i32,
    /// Last MiB of stdout (lossy UTF-8).
    pub stdout: String,
    /// Last MiB of stderr (lossy UTF-8).
    pub stderr: String,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SystemCommandRunner;

impl CommandRunner for SystemCommandRunner {
    // shortcut: no timeout, a hung converter blocks the caller; add one if that happens.
    fn run(&self, program: &Path, args: &[String], cwd: Option<&Path>) -> Result<CommandOutput> {
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let mut child = cmd.spawn().map_err(|e| spawn_error(program, e))?;
        // Both pipes are drained concurrently, otherwise a full stderr pipe deadlocks the child.
        let stdout = child
            .stdout
            .take()
            .map(|s| drain_tail(s, CAPTURE_TAIL_BYTES));
        let stderr = child
            .stderr
            .take()
            .map(|s| drain_tail(s, CAPTURE_TAIL_BYTES));
        let status = child.wait().map_err(|e| EvalError::Spawn {
            program: program.to_path_buf(),
            message: e.to_string(),
        })?;
        let collect =
            |h: Option<JoinHandle<String>>| h.and_then(|h| h.join().ok()).unwrap_or_default();
        Ok(CommandOutput {
            status_code: status.code().unwrap_or(-1),
            stdout: collect(stdout),
            stderr: collect(stderr),
        })
    }

    fn run_streaming(
        &self,
        program: &Path,
        args: &[String],
        cwd: Option<&Path>,
        control: StreamControl<'_>,
        on_line: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<CommandOutput> {
        let started = Instant::now();
        let mut cmd = Command::new(program);
        cmd.args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if let Some(dir) = cwd {
            cmd.current_dir(dir);
        }
        let mut child = ChildGuard(cmd.spawn().map_err(|e| spawn_error(program, e))?);
        let (tx, rx) = mpsc::sync_channel(STREAM_CHANNEL_LINES);
        let mut streams = 0;
        if let Some(out) = child.0.stdout.take() {
            stream_lines(out, 0, tx.clone());
            streams += 1;
        }
        if let Some(err) = child.0.stderr.take() {
            stream_lines(err, 1, tx.clone());
            streams += 1;
        }
        drop(tx);

        let mut tails = [String::new(), String::new()];
        let mut done = 0;
        let mut stopped_at: Option<Instant> = None;
        let mut timed_out = false;
        while done < streams {
            if let Some(at) = stopped_at {
                // A grandchild may keep a pipe open after the kill: never wait forever.
                if at.elapsed() > STREAM_GRACE {
                    break;
                }
            } else {
                let cancelled = control.cancel.is_some_and(|c| c.load(Ordering::SeqCst));
                timed_out = control.timeout.is_some_and(|t| started.elapsed() >= t);
                if cancelled || timed_out {
                    let _ = child.0.kill();
                    stopped_at = Some(Instant::now());
                }
            }
            match rx.recv_timeout(STREAM_TICK) {
                Ok(StreamMsg::Line(line)) if stopped_at.is_none() => {
                    if on_line(&line).is_break() {
                        let _ = child.0.kill();
                        stopped_at = Some(Instant::now());
                    }
                }
                Ok(StreamMsg::Line(_)) => {}
                Ok(StreamMsg::Done(stream, tail)) => {
                    tails[stream] = tail;
                    done += 1;
                }
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
        }
        if timed_out {
            return Err(EvalError::Timeout {
                after_ms: started.elapsed().as_millis(),
            });
        }
        let status = child.0.wait().map_err(|e| EvalError::Spawn {
            program: program.to_path_buf(),
            message: e.to_string(),
        })?;
        let [stdout, stderr] = tails;
        Ok(CommandOutput {
            status_code: status.code().unwrap_or(-1),
            stdout,
            stderr,
        })
    }
}

enum StreamMsg {
    Line(String),
    /// End of one stream (0 stdout, 1 stderr) with its last MiB.
    Done(usize, String),
}

/// Reads `reader` on a thread and sends its lines, then its tail. Keeps draining when the
/// receiver is gone, so a killed child never blocks on a full pipe.
fn stream_lines<R: Read + Send + 'static>(
    mut reader: R,
    stream: usize,
    tx: mpsc::SyncSender<StreamMsg>,
) {
    thread::spawn(move || {
        let mut splitter = LineSplitter::default();
        let mut kept = Vec::new();
        let mut buf = [0u8; 8192];
        let mut listening = true;
        loop {
            let n = match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            };
            kept.extend_from_slice(&buf[..n]);
            if kept.len() > 2 * CAPTURE_TAIL_BYTES {
                kept.drain(..kept.len() - CAPTURE_TAIL_BYTES);
            }
            for line in splitter.push(&buf[..n]) {
                listening = listening && tx.send(StreamMsg::Line(line)).is_ok();
            }
        }
        for line in splitter.finish() {
            listening = listening && tx.send(StreamMsg::Line(line)).is_ok();
        }
        let start = kept.len().saturating_sub(CAPTURE_TAIL_BYTES);
        let tail = String::from_utf8_lossy(&kept[start..]).into_owned();
        let _ = tx.send(StreamMsg::Done(stream, tail));
    });
}

/// Splits a byte stream on `\n` and `\r`. A fragment ending with `,` at the end of a read
/// is delivered at once: `llama-perplexity` writes `[1]5.4321,[2]...` without newlines.
#[derive(Debug, Default)]
pub(crate) struct LineSplitter {
    pending: Vec<u8>,
}

impl LineSplitter {
    pub(crate) fn push(&mut self, bytes: &[u8]) -> Vec<String> {
        let mut lines = Vec::new();
        for &b in bytes {
            match b {
                b'\n' | b'\r' => lines.extend(self.take()),
                _ => self.pending.push(b),
            }
        }
        if self.pending.last() == Some(&b',') {
            lines.extend(self.take());
        }
        lines
    }

    pub(crate) fn finish(&mut self) -> Vec<String> {
        self.take().into_iter().collect()
    }

    fn take(&mut self) -> Option<String> {
        let line = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        (!line.is_empty()).then_some(line)
    }
}

pub(crate) fn spawn_error(program: &Path, e: std::io::Error) -> EvalError {
    if e.kind() == std::io::ErrorKind::NotFound {
        EvalError::BinaryNotFound {
            name: program.display().to_string(),
            hint: "vérifiez le chemin, ou que le programme est dans le PATH".into(),
        }
    } else {
        EvalError::Spawn {
            program: program.to_path_buf(),
            message: e.to_string(),
        }
    }
}

/// Reads `reader` to EOF on a thread, keeping only the last `cap` bytes (amortized O(n)).
pub(crate) fn drain_tail<R: Read + Send + 'static>(
    mut reader: R,
    cap: usize,
) -> JoinHandle<String> {
    thread::spawn(move || {
        let mut kept = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    kept.extend_from_slice(&buf[..n]);
                    if kept.len() > 2 * cap {
                        kept.drain(..kept.len() - cap);
                    }
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(_) => break,
            }
        }
        let start = kept.len().saturating_sub(cap);
        String::from_utf8_lossy(&kept[start..]).into_owned()
    })
}

pub(crate) fn last_lines(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_helpers_keep_the_end() {
        let text: String = (1..=50).map(|i| format!("line {i}\n")).collect();
        let tail = last_lines(&text, 20);
        assert_eq!(tail.lines().count(), 20);
        assert!(tail.starts_with("line 31") && tail.ends_with("line 50"));
        assert_eq!(last_lines("a\nb", 20), "a\nb");

        let big = vec![b'x'; 100_000]
            .into_iter()
            .chain(*b"END")
            .collect::<Vec<_>>();
        let kept = drain_tail(std::io::Cursor::new(big), 1000).join().unwrap();
        assert_eq!(kept.len(), 1000);
        assert!(kept.ends_with("END"));
    }

    #[test]
    fn line_splitter_handles_crlf_and_comma_fragments() {
        let mut s = LineSplitter::default();
        assert_eq!(s.push(b"a\r\nb"), ["a"]);
        assert_eq!(s.push(b"c\n[1]5.4,"), ["bc", "[1]5.4,"]);
        assert_eq!(s.push(b"[2]6.1"), Vec::<String>::new());
        assert_eq!(s.push(b","), ["[2]6.1,"]);
        assert_eq!(s.push(b"tail"), Vec::<String>::new());
        assert_eq!(s.finish(), ["tail"]);
        assert!(s.finish().is_empty());
    }

    const CHILD_MODE: &str = "surgeon-child-mode=";

    /// Not a real test: the body of the child process the streaming tests spawn (this
    /// same test binary, filtered to this test).
    #[test]
    fn stream_child_helper() {
        let Some(mode) =
            std::env::args().find_map(|a| a.strip_prefix(CHILD_MODE).map(String::from))
        else {
            return;
        };
        use std::io::Write as _;
        let mut out = std::io::stdout();
        let mut err = std::io::stderr();
        match mode.as_str() {
            "lines" => {
                let _ = writeln!(out, "child-out-1");
                let _ = write!(err, "[1]5.0,[2]6.0,");
                let _ = err.flush();
                let _ = writeln!(err, "\nchild-err-last");
            }
            "ready-then-sleep" => {
                let _ = writeln!(out, "child-ready");
                let _ = out.flush();
                std::thread::sleep(Duration::from_secs(30));
            }
            _ => std::thread::sleep(Duration::from_secs(30)),
        }
    }

    fn child_run(
        mode: &str,
        control: StreamControl<'_>,
        on_line: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<CommandOutput> {
        let exe = std::env::current_exe().unwrap();
        let args = [
            "--exact",
            "eval::command::tests::stream_child_helper",
            "--nocapture",
            "--test-threads=1",
            &format!("{CHILD_MODE}{mode}"),
        ]
        .map(String::from);
        SystemCommandRunner.run_streaming(&exe, &args, None, control, on_line)
    }

    /// The child mode is an extra libtest filter argument, which matches no test name.
    #[test]
    fn streaming_delivers_lines_and_stops_on_break_timeout_and_cancel() {
        let mut lines = Vec::new();
        let out = child_run("lines", StreamControl::default(), &mut |l| {
            lines.push(l.to_owned());
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(out.status_code, 0);
        for expected in ["child-out-1", "[1]5.0,[2]6.0,", "child-err-last"] {
            assert!(
                lines.iter().any(|l| l.contains(expected)),
                "{expected}: {lines:?}"
            );
        }
        assert!(out.stderr.contains("child-err-last") && out.stdout.contains("child-out-1"));

        let started = Instant::now();
        let mut seen = 0;
        child_run("ready-then-sleep", StreamControl::default(), &mut |l| {
            seen += 1;
            match l.contains("child-ready") {
                true => ControlFlow::Break(()),
                false => ControlFlow::Continue(()),
            }
        })
        .unwrap();
        assert!(seen >= 1);
        assert!(started.elapsed() < Duration::from_secs(20), "Break kills");

        let started = Instant::now();
        let control = StreamControl {
            timeout: Some(Duration::from_millis(300)),
            cancel: None,
        };
        let err = child_run("silent", control, &mut |_| ControlFlow::Continue(())).unwrap_err();
        assert!(matches!(err, EvalError::Timeout { .. }), "{err:?}");
        assert!(started.elapsed() < Duration::from_secs(20), "timeout kills");

        let flag = std::sync::Arc::new(AtomicBool::new(false));
        let raiser = std::sync::Arc::clone(&flag);
        let raise = thread::spawn(move || {
            thread::sleep(Duration::from_millis(300));
            raiser.store(true, Ordering::SeqCst);
        });
        let started = Instant::now();
        let control = StreamControl {
            timeout: None,
            cancel: Some(&flag),
        };
        let out = child_run("silent", control, &mut |_| ControlFlow::Continue(())).unwrap();
        raise.join().unwrap();
        assert_ne!(out.status_code, 0, "killed");
        assert!(started.elapsed() < Duration::from_secs(20), "cancel kills");
    }

    #[test]
    fn default_run_streaming_replays_the_output() {
        struct Canned;
        impl CommandRunner for Canned {
            fn run(&self, _: &Path, _: &[String], _: Option<&Path>) -> Result<CommandOutput> {
                Ok(CommandOutput {
                    status_code: 0,
                    stdout: "o1\no2".into(),
                    stderr: "e1".into(),
                })
            }
        }
        let mut lines = Vec::new();
        Canned
            .run_streaming(
                Path::new("x"),
                &[],
                None,
                StreamControl::default(),
                &mut |l| {
                    lines.push(l.to_owned());
                    match lines.len() {
                        2 => ControlFlow::Break(()),
                        _ => ControlFlow::Continue(()),
                    }
                },
            )
            .unwrap();
        assert_eq!(lines, ["o1", "o2"]);
    }

    #[test]
    fn missing_program_is_binary_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let err = SystemCommandRunner
            .run(&dir.path().join("absent-tool"), &[], None)
            .unwrap_err();
        assert!(matches!(err, EvalError::BinaryNotFound { .. }), "{err:?}");
    }
}
