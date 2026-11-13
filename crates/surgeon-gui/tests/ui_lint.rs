//! Guards of the design system (docs/design/DIRECTION.md): a color or a grouped frame is
//! written only in `ui/theme.rs` and `ui/components.rs`, and the shell and the views carry no
//! interface sentence of their own (every label goes through `i18n`).

use std::path::{Path, PathBuf};

const SRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn relative(path: &Path) -> String {
    path.strip_prefix(SRC)
        .unwrap()
        .to_string_lossy()
        .replace('\\', "/")
}

/// The code of a file before its test module, comment lines dropped.
fn code(path: &Path) -> Vec<(usize, String)> {
    let text = std::fs::read_to_string(path).unwrap();
    text.lines()
        .enumerate()
        .take_while(|(_, l)| !l.trim_start().starts_with("#[cfg(test)]"))
        .filter(|(_, l)| !l.trim_start().starts_with("//"))
        .map(|(n, l)| (n + 1, l.to_owned()))
        .collect()
}

#[test]
fn colors_and_frames_are_written_only_in_the_theme_and_the_components() {
    let mut files = Vec::new();
    rust_files(Path::new(SRC), &mut files);
    let allowed = ["ui/theme.rs", "ui/components.rs"];
    let mut found = Vec::new();
    for file in &files {
        let name = relative(file);
        if allowed.contains(&name.as_str()) {
            continue;
        }
        let text = std::fs::read_to_string(file).unwrap();
        for (n, line) in text.lines().enumerate() {
            if line.contains("Color32::") || line.contains("Frame::group") {
                found.push(format!("{name}:{}: {}", n + 1, line.trim()));
            }
        }
    }
    assert!(found.is_empty(), "outside the theme:\n{}", found.join("\n"));
}

/// String literals of one line (no escaped quotes in the views' labels).
fn literals(line: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(start) = rest.find('"') {
        let after = &rest[start + 1..];
        let Some(end) = after.find('"') else { break };
        out.push(&after[..end]);
        rest = &after[end + 1..];
    }
    out
}

/// Two words of three letters or more: a sentence, not a key, a format or a unit.
fn is_sentence(text: &str) -> bool {
    let words: Vec<&str> = text.split(' ').collect();
    words.windows(2).any(|w| {
        let long = |s: &str| {
            s.chars()
                .filter(|c| c.is_alphabetic() && c.is_ascii())
                .count()
                >= 3
                && s.chars().all(|c| c.is_alphabetic() || "'’".contains(c))
        };
        long(w[0]) && long(w[1])
    })
}

#[test]
fn the_shell_and_the_views_hold_no_sentence_of_their_own() {
    let mut files = Vec::new();
    rust_files(&Path::new(SRC).join("ui"), &mut files);
    let skip = ["ui/i18n.rs", "ui/fonts.rs", "ui/theme.rs"];
    let mut found = Vec::new();
    for file in &files {
        let name = relative(file);
        if skip.contains(&name.as_str()) {
            continue;
        }
        for (n, line) in code(file) {
            for lit in literals(&line) {
                if is_sentence(lit) {
                    found.push(format!("{name}:{n}: \"{lit}\""));
                }
            }
        }
    }
    assert!(found.is_empty(), "hard-coded text:\n{}", found.join("\n"));
}

#[test]
fn the_sentence_detector_tells_labels_from_keys_and_formats() {
    assert!(is_sentence("Lancer l'analyse"));
    assert!(is_sentence("Apply surgery now"));
    for ok in [
        "panel.inspector",
        "layers.{l}",
        "Δ erank",
        "κ (σ1/σmin)",
        "{name} *",
        "×{s}",
    ] {
        assert!(!is_sentence(ok), "{ok}");
    }
}
