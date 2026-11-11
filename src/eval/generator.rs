use std::ffi::OsString;
use std::io::Read;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;

use super::command::{drain_tail, last_lines, spawn_error, ChildGuard};
use super::error::{EvalError, Result};

const STDERR_TAIL_BYTES: usize = 64 * 1024;
const READ_CHUNK_BYTES: usize = 4096;
/// Bounded so a flooding child is throttled instead of filling memory.
const CHANNEL_CHUNKS: usize = 64;
/// llama.cpp layouts: CMake single-config, multi-config (MSVC) and the legacy Makefile.
const LLAMA_SUBDIRS: [&str; 4] = ["", "build/bin", "build/bin/Release", "build"];

pub trait TextGenerator {
    /// Streams the output through `on_chunk`; returning `Break` stops the generation.
    fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: u32,
        on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<GenerationOutcome>;

    /// Switches to greedy, seeded decoding (call once, before the first `generate`).
    /// Returns false when the generator has no sampling to configure (mocks, scripts).
    fn configure(&mut self, _decoding: &Decoding) -> bool {
        false
    }

    /// Restricts the output to a GBNF grammar (call once, before the first `generate`).
    /// Returns false when the generator cannot constrain its output: the caller must then
    /// parse free text.
    fn constrain(&mut self, _gbnf: &str) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GenerationOutcome {
    pub text: String,
    pub stopped_early: bool,
    pub duration_ms: u128,
}

/// Runs `llama-cli -m <model.gguf> -p <prompt> -n <max_new_tokens> --no-display-prompt
/// <extra_args...>`, never through a shell.
///
/// The flags drift between llama.cpp releases (to verify against the installed build):
/// recent builds start in chat mode when the model has a template, so pass `-no-cnv`
/// (or `--single-turn`) in `extra_args`, or point `llama_cli_path` at `llama-completion`.
/// stdin is closed, so an interactive prompt sees EOF instead of hanging.
#[derive(Debug, Clone)]
pub struct LlamaCliGenerator {
    pub llama_cli_path: PathBuf,
    /// GGUF file.
    pub model_path: PathBuf,
    pub extra_args: Vec<String>,
    pub timeout: Duration,
}

/// Deterministic decoding requested by `surgeon bench eval`, passed as structured flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decoding {
    pub seed: u32,
    /// Per generation; `None` keeps the generator's own timeout.
    pub timeout: Option<Duration>,
}

impl LlamaCliGenerator {
    fn command(&self, prompt: &str, max_new_tokens: u32) -> Command {
        let mut cmd = Command::new(&self.llama_cli_path);
        cmd.arg("-m")
            .arg(&self.model_path)
            .arg("-p")
            .arg(prompt)
            .arg("-n")
            .arg(max_new_tokens.to_string())
            .arg("--no-display-prompt")
            .args(&self.extra_args);
        cmd
    }
}

impl TextGenerator for LlamaCliGenerator {
    fn generate(
        &mut self,
        prompt: &str,
        max_new_tokens: u32,
        on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> Result<GenerationOutcome> {
        stream_child(self.command(prompt, max_new_tokens), self.timeout, on_chunk)
    }

    /// `--temp 0 --top-k 1 --seed N`: same model, prompt and build give the same text.
    fn configure(&mut self, decoding: &Decoding) -> bool {
        let seed = decoding.seed.to_string();
        self.extra_args
            .extend(["--temp", "0", "--top-k", "1", "--seed", &seed].map(String::from));
        if let Some(timeout) = decoding.timeout {
            self.timeout = timeout;
        }
        true
    }

    /// `--grammar <GBNF>`, inline: no grammar file to write or clean up.
    fn constrain(&mut self, gbnf: &str) -> bool {
        self.extra_args
            .extend(["--grammar".to_owned(), gbnf.to_owned()]);
        true
    }
}

/// Spawns `cmd`, streams its stdout to `on_chunk`, and guarantees the child is reaped
/// on every path: normal exit, `Break`, timeout or read error.
fn stream_child(
    mut cmd: Command,
    timeout: Duration,
    on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
) -> Result<GenerationOutcome> {
    let start = Instant::now();
    let program = PathBuf::from(cmd.get_program());
    let child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| match spawn_error(&program, e) {
            EvalError::BinaryNotFound { name, .. } => EvalError::BinaryNotFound {
                name,
                hint: LLAMA_HINT.into(),
            },
            other => other,
        })?;
    let mut child = ChildGuard(child);
    let (Some(mut stdout), Some(stderr)) = (child.0.stdout.take(), child.0.stderr.take()) else {
        return Err(EvalError::Generator(
            "pipes de llama-cli indisponibles".into(),
        ));
    };
    let stderr = drain_tail(stderr, STDERR_TAIL_BYTES);

    let (tx, rx) = mpsc::sync_channel(CHANNEL_CHUNKS);
    thread::spawn(move || {
        let mut buf = [0u8; READ_CHUNK_BYTES];
        loop {
            let msg = match stdout.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => Ok(buf[..n].to_vec()),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => Err(e),
            };
            let failed = msg.is_err();
            // A send error means the generator stopped listening: stop reading.
            if tx.send(msg).is_err() || failed {
                break;
            }
        }
    });

    let deadline = start + timeout;
    let mut decoder = Utf8Decoder::default();
    let mut text = String::new();
    let mut emit = |chunk: String, text: &mut String| {
        text.push_str(&chunk);
        !chunk.is_empty() && on_chunk(&chunk).is_break()
    };
    let stopped_early = loop {
        // Checked explicitly: a child that never stops writing never lets recv time out.
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(EvalError::Timeout {
                after_ms: start.elapsed().as_millis(),
            });
        }
        match rx.recv_timeout(remaining) {
            Ok(Ok(bytes)) => {
                if emit(decoder.push(&bytes), &mut text) {
                    break true;
                }
            }
            Ok(Err(e)) => {
                return Err(EvalError::Generator(format!(
                    "lecture de la sortie de llama-cli : {e}"
                )))
            }
            Err(RecvTimeoutError::Timeout) => {
                return Err(EvalError::Timeout {
                    after_ms: start.elapsed().as_millis(),
                })
            }
            Err(RecvTimeoutError::Disconnected) => break emit(decoder.finish(), &mut text),
        }
    };
    drop(rx);

    if stopped_early {
        let _ = child.0.kill();
    }
    let status = child.0.wait().map_err(|e| EvalError::Spawn {
        program: program.clone(),
        message: e.to_string(),
    })?;
    let stderr = stderr.join().unwrap_or_default();
    if !stopped_early && !status.success() {
        return Err(EvalError::CommandFailed {
            step: "llama-cli".into(),
            code: status.code().unwrap_or(-1),
            stderr_tail: last_lines(&stderr, 20),
        });
    }
    Ok(GenerationOutcome {
        text,
        stopped_early,
        duration_ms: start.elapsed().as_millis(),
    })
}

/// Decodes a byte stream whose chunks may end in the middle of a multi-byte character.
#[derive(Debug, Default)]
struct Utf8Decoder {
    pending: Vec<u8>,
}

impl Utf8Decoder {
    fn push(&mut self, bytes: &[u8]) -> String {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(&String::from_utf8_lossy(&self.pending[..valid]));
                    match e.error_len() {
                        // Incomplete sequence at the end: wait for the next chunk.
                        None => {
                            self.pending.drain(..valid);
                            return out;
                        }
                        Some(bad) => {
                            out.push(char::REPLACEMENT_CHARACTER);
                            self.pending.drain(..valid + bad);
                        }
                    }
                }
            }
        }
    }

    fn finish(&mut self) -> String {
        let rest = String::from_utf8_lossy(&self.pending).into_owned();
        self.pending.clear();
        rest
    }
}

const LLAMA_HINT: &str = "installez llama.cpp puis passez --llama-cpp-dir <dossier> \
    ou définissez la variable LLAMA_CPP_DIR";

/// Looks in `hint_dir`, then `$LLAMA_CPP_DIR`, then `PATH`. Inside the two llama.cpp
/// directories, `build/bin`, `build/bin/Release` and `build` are searched too.
pub fn locate_binary(name: &str, hint_dir: Option<&Path>) -> Result<PathBuf> {
    let mut dirs: Vec<PathBuf> = hint_dir.map(Path::to_path_buf).into_iter().collect();
    dirs.extend(std::env::var_os("LLAMA_CPP_DIR").map(PathBuf::from));
    find_binary(&[name], &dirs, std::env::var_os("PATH")).ok_or_else(|| EvalError::BinaryNotFound {
        name: name.to_owned(),
        hint: LLAMA_HINT.into(),
    })
}

/// Tries every name in every llama.cpp directory, then on `path_var`, in order.
pub(crate) fn find_binary(
    names: &[&str],
    llama_dirs: &[PathBuf],
    path_var: Option<OsString>,
) -> Option<PathBuf> {
    let path_dirs: Vec<PathBuf> = path_var
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default();
    names.iter().find_map(|name| {
        let file_names = executable_names(name);
        let llama = llama_dirs
            .iter()
            .flat_map(|d| LLAMA_SUBDIRS.iter().map(move |sub| d.join(sub)));
        llama
            .chain(path_dirs.iter().cloned())
            .flat_map(|dir| file_names.iter().map(move |f| dir.join(f)))
            .find(|candidate| candidate.is_file())
    })
}

fn executable_names(name: &str) -> Vec<String> {
    let suffix = std::env::consts::EXE_SUFFIX;
    if suffix.is_empty() || name.ends_with(suffix) {
        vec![name.to_owned()]
    } else {
        vec![name.to_owned(), format!("{name}{suffix}")]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utf8_decoder_never_splits_a_character() {
        let mut d = Utf8Decoder::default();
        let emoji = "🦀".as_bytes();
        assert_eq!(d.push(&[b'a', 0xC3]), "a");
        assert_eq!(d.push(&[0xA9]), "é");
        assert_eq!(d.push(&emoji[..1]), "");
        assert_eq!(d.push(&emoji[1..3]), "");
        assert_eq!(d.push(&emoji[3..]), "🦀");
        assert_eq!(d.push(&[0xFF, b'z']), "\u{FFFD}z");
        assert_eq!(d.push(&[0xE2, 0x82]), "");
        assert_eq!(d.finish(), "\u{FFFD}");
    }

    #[test]
    fn find_binary_order_and_absence() {
        let dir = tempfile::tempdir().unwrap();
        let exe = format!("llama-cli{}", std::env::consts::EXE_SUFFIX);
        let bin = dir.path().join("build/bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join(&exe), "").unwrap();
        let llama = [dir.path().to_path_buf()];
        assert_eq!(
            find_binary(&["llama-cli"], &llama, None),
            Some(bin.join(&exe))
        );

        let on_path = tempfile::tempdir().unwrap();
        std::fs::write(on_path.path().join(&exe), "").unwrap();
        let path_var = Some(on_path.path().as_os_str().to_owned());
        assert_eq!(
            find_binary(&["llama-cli"], &[], path_var.clone()),
            Some(on_path.path().join(&exe))
        );
        assert_eq!(
            find_binary(&["llama-cli"], &llama, path_var),
            Some(bin.join(&exe)),
            "llama.cpp directory wins over PATH"
        );
        assert_eq!(find_binary(&["absent"], &llama, None), None);
    }

    #[test]
    fn locate_binary_uses_hint_dir_and_reports_absence() {
        let dir = tempfile::tempdir().unwrap();
        let exe = format!("fake-llama-tool{}", std::env::consts::EXE_SUFFIX);
        std::fs::write(dir.path().join(&exe), "").unwrap();
        assert_eq!(
            locate_binary("fake-llama-tool", Some(dir.path())).unwrap(),
            dir.path().join(&exe)
        );
        let err = locate_binary("weight-surgeon-absent-tool", Some(dir.path())).unwrap_err();
        let EvalError::BinaryNotFound { hint, .. } = &err else {
            panic!("{err:?}")
        };
        assert!(hint.contains("--llama-cpp-dir") && hint.contains("LLAMA_CPP_DIR"));
    }

    #[test]
    fn missing_llama_cli_is_binary_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let mut g = LlamaCliGenerator {
            llama_cli_path: dir.path().join("llama-cli-absent"),
            model_path: dir.path().join("m.gguf"),
            extra_args: vec![],
            timeout: Duration::from_secs(5),
        };
        let err = g
            .generate("x", 8, &mut |_| ControlFlow::Continue(()))
            .unwrap_err();
        assert!(
            matches!(err, EvalError::BinaryNotFound { ref hint, .. } if hint.contains("LLAMA_CPP_DIR"))
        );
    }

    #[test]
    fn llama_cli_arguments() {
        let g = LlamaCliGenerator {
            llama_cli_path: "llama-cli".into(),
            model_path: "m.gguf".into(),
            extra_args: vec!["-no-cnv".into()],
            timeout: Duration::from_secs(1),
        };
        let cmd = g.command("Bonjour; rm -rf /", 64);
        let args: Vec<_> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            args,
            [
                "-m",
                "m.gguf",
                "-p",
                "Bonjour; rm -rf /",
                "-n",
                "64",
                "--no-display-prompt",
                "-no-cnv"
            ]
        );
    }

    /// `yes` stands in for a looping llama-cli: it prints "y" forever.
    /// Skipped when `yes` is not on PATH (stock Windows without Git Bash).
    fn yes_command() -> Option<Command> {
        find_binary(&["yes"], &[], std::env::var_os("PATH")).map(Command::new)
    }

    #[test]
    fn break_kills_an_endless_child() {
        let Some(cmd) = yes_command() else {
            eprintln!("yes absent du PATH, test ignoré");
            return;
        };
        let mut chunks = 0;
        let started = Instant::now();
        let out = stream_child(cmd, Duration::from_secs(30), &mut |_| {
            chunks += 1;
            if chunks == 3 {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        })
        .unwrap();
        assert!(out.stopped_early);
        assert_eq!(chunks, 3);
        assert!(out.text.starts_with("y\ny\n"));
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn timeout_kills_the_child_and_errors() {
        let Some(cmd) = yes_command() else {
            eprintln!("yes absent du PATH, test ignoré");
            return;
        };
        let started = Instant::now();
        let err = stream_child(cmd, Duration::from_millis(300), &mut |_| {
            ControlFlow::Continue(())
        })
        .unwrap_err();
        assert!(matches!(err, EvalError::Timeout { .. }), "{err:?}");
        assert!(started.elapsed() < Duration::from_secs(10));
    }

    /// A child that writes to stderr and exits with code 3. Not `yes --option-inconnue`:
    /// only GNU `yes` rejects unknown options, BSD `yes` (macOS) prints them forever.
    /// Skipped when `sh` is not on PATH (stock Windows without Git Bash).
    fn failing_command() -> Option<Command> {
        let mut cmd = Command::new(find_binary(&["sh"], &[], std::env::var_os("PATH"))?);
        cmd.args(["-c", "echo 'option inconnue' >&2; exit 3"]);
        Some(cmd)
    }

    #[test]
    fn failing_child_reports_stderr_tail() {
        let Some(cmd) = failing_command() else {
            eprintln!("sh absent du PATH, test ignoré");
            return;
        };
        let err = stream_child(cmd, Duration::from_secs(10), &mut |_| {
            ControlFlow::Continue(())
        })
        .unwrap_err();
        assert!(
            matches!(&err, EvalError::CommandFailed { code, stderr_tail, .. } if *code != 0 && !stderr_tail.is_empty()),
            "{err:?}"
        );
    }
}
