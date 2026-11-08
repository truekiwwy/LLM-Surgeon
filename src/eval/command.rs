use std::io::Read;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread::{self, JoinHandle};

use serde::Serialize;

use super::error::{EvalError, Result};

/// Each captured stream keeps only its last bytes: converters can be very chatty.
const CAPTURE_TAIL_BYTES: usize = 1 << 20;

/// Synchronous process runner, mockable. Implementations must never go through a shell.
pub trait CommandRunner {
    fn run(&self, program: &Path, args: &[String], cwd: Option<&Path>) -> Result<CommandOutput>;
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
    fn missing_program_is_binary_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let err = SystemCommandRunner
            .run(&dir.path().join("absent-tool"), &[], None)
            .unwrap_err();
        assert!(matches!(err, EvalError::BinaryNotFound { .. }), "{err:?}");
    }
}
