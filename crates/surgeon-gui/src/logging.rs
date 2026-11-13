//! A small rotating log file under the user data folder, and the redaction shared with the
//! diagnostics export: personal paths and anything that looks like a secret never reach a
//! file the user may attach to a bug report.

use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// Rotated at this size; `log.txt` plus `KEEP` older files.
const MAX_BYTES: u64 = 512 * 1024;
const KEEP: usize = 3;

const SECRET_KEYS: [&str; 7] = [
    "token", "secret", "password", "passwd", "api_key", "api-key", "apikey",
];

static SINK: Mutex<Option<Sink>> = Mutex::new(None);

struct Sink {
    dir: PathBuf,
    file: File,
    written: u64,
}

fn log_path(dir: &Path, n: usize) -> PathBuf {
    match n {
        0 => dir.join("log.txt"),
        n => dir.join(format!("log.{n}.txt")),
    }
}

/// `log.txt` -> `log.1.txt` -> ... -> dropped.
fn rotate(dir: &Path) {
    let _ = fs::remove_file(log_path(dir, KEEP));
    for n in (0..KEEP).rev() {
        let _ = fs::rename(log_path(dir, n), log_path(dir, n + 1));
    }
}

fn open(dir: &Path) -> std::io::Result<Sink> {
    fs::create_dir_all(dir)?;
    if fs::metadata(log_path(dir, 0)).is_ok_and(|m| m.len() >= MAX_BYTES) {
        rotate(dir);
    }
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path(dir, 0))?;
    let written = file.metadata().map(|m| m.len()).unwrap_or(0);
    Ok(Sink {
        dir: dir.to_path_buf(),
        file,
        written,
    })
}

/// Starts logging into `dir`; failing to open it only disables the log.
pub fn init(dir: &Path) {
    if let Ok(sink) = open(dir) {
        *SINK.lock().unwrap_or_else(|e| e.into_inner()) = Some(sink);
    }
}

pub fn line(level: &str, message: &str) {
    let mut guard = SINK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(sink) = guard.as_mut() else { return };
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let text = format!("{secs} {level} {}\n", mask_secrets(message));
    if sink.written + text.len() as u64 > MAX_BYTES {
        rotate(&sink.dir);
        if let Ok(fresh) = open(&sink.dir) {
            *sink = fresh;
        }
    }
    if sink.file.write_all(text.as_bytes()).is_ok() {
        sink.written += text.len() as u64;
    }
}

pub fn info(message: &str) {
    line("INFO", message);
}

pub fn warn(message: &str) {
    line("WARN", message);
}

/// The last `max` lines of the current log, for the diagnostics export.
pub fn tail(dir: &Path, max: usize) -> Vec<String> {
    let text = fs::read_to_string(log_path(dir, 0)).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(max)..]
        .iter()
        .map(|l| (*l).to_owned())
        .collect()
}

/// `key: value`, `key=value`, `Authorization: Bearer x` -> the value is replaced by `***`.
pub fn mask_secrets(text: &str) -> String {
    text.lines()
        .map(|line| match secret_value_start(line) {
            Some(start) => format!("{}***", &line[..start]),
            None => line.to_owned(),
        })
        .collect::<Vec<_>>()
        .join(
            "
",
        )
}

/// Byte index where the secret's value starts: after `key =`, `key:` or `bearer `. A key
/// that is only part of a word (`tokenizer.json`) is not a secret.
fn secret_value_start(line: &str) -> Option<usize> {
    let lower = line.to_ascii_lowercase();
    let skip = |from: usize, chars: &[char]| {
        from + lower[from..]
            .find(|c: char| !chars.contains(&c))
            .unwrap_or(lower.len() - from)
    };
    let mut starts = Vec::new();
    for key in SECRET_KEYS.iter().chain(["authorization"].iter()) {
        for (at, _) in lower.match_indices(key) {
            let after = skip(at + key.len(), &['"', '\'', ' ']);
            if lower[after..].starts_with(['=', ':']) {
                starts.push(skip(after, &['=', ':', ' ', '"', '\'']));
            }
        }
    }
    if let Some(at) = lower.find("bearer ") {
        starts.push(skip(at + 7, &[' ']));
    }
    starts.into_iter().filter(|&s| s < lower.len()).min()
}

/// Replaces the home folder and the user name by neutral markers, then masks secrets.
pub fn redact(text: &str, home: Option<&Path>, user: Option<&str>) -> String {
    let mut out = text.to_owned();
    if let Some(home) = home
        .map(|h| h.display().to_string())
        .filter(|h| h.len() > 3)
    {
        for variant in [
            home.clone(),
            home.replace('\\', "/"),
            home.replace('/', "\\"),
        ] {
            out = replace_ci(&out, &variant, "~");
        }
    }
    if let Some(user) = user.filter(|u| u.len() >= 3) {
        out = replace_ci(&out, user, "<user>");
    }
    mask_secrets(&out)
}

/// ASCII case-insensitive replacement (Windows paths do not care about case).
fn replace_ci(text: &str, from: &str, to: &str) -> String {
    let (lower, needle) = (text.to_ascii_lowercase(), from.to_ascii_lowercase());
    let mut out = String::new();
    let mut last = 0;
    for (at, _) in lower.match_indices(&needle) {
        out.push_str(&text[last..at]);
        out.push_str(to);
        last = at + needle.len();
    }
    out.push_str(&text[last..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secrets_are_masked_and_the_rest_kept() {
        assert_eq!(mask_secrets("hf_token = abc123"), "hf_token = ***");
        assert_eq!(
            mask_secrets("Authorization: Bearer xyz"),
            "Authorization: ***"
        );
        assert_eq!(mask_secrets("API_KEY=\"k-1\" ok"), "API_KEY=\"***");
        assert_eq!(
            mask_secrets("download 12 of 40 MiB"),
            "download 12 of 40 MiB"
        );
    }

    #[test]
    fn personal_paths_and_names_are_redacted() {
        let home = Path::new(r"C:\Users\Jean Dupont");
        let text = r"C:\Users\Jean Dupont\AppData\x and c:/users/jean dupont/y, user=Dupont";
        let out = redact(text, Some(home), Some("Dupont"));
        assert!(
            !out.to_lowercase().contains("jean dupont") && !out.contains("Dupont"),
            "{out}"
        );
        assert!(out.contains(r"~\AppData\x"));
    }

    #[test]
    fn the_log_rotates_and_never_holds_a_secret() {
        let dir = tempfile::tempdir().unwrap();
        let mut sink = open(dir.path()).unwrap();
        sink.written = MAX_BYTES;
        *SINK.lock().unwrap() = Some(sink);
        line("INFO", "first");
        line("INFO", "password=hunter2");
        *SINK.lock().unwrap() = None;
        let current = fs::read_to_string(log_path(dir.path(), 0)).unwrap();
        assert!(current.contains("password=***") && !current.contains("hunter2"));
        assert!(log_path(dir.path(), 1).exists(), "the full log was rotated");
        assert_eq!(tail(dir.path(), 1).len(), 1);
    }
}
