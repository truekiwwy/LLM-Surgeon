//! The two locales carry exactly the same keys, with the same placeholders and no empty text;
//! every key the code asks for exists, and every key of the files is asked for somewhere.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

const SRC: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
const FR: &str = include_str!("../src/ui/locales/fr.json");
const EN: &str = include_str!("../src/ui/locales/en.json");

fn table(json: &str) -> BTreeMap<String, String> {
    serde_json::from_str(json).expect("a locale file parses as a flat object of strings")
}

fn placeholders(text: &str) -> BTreeSet<String> {
    text.split('{')
        .skip(1)
        .filter_map(|p| p.split_once('}').map(|(name, _)| name.to_owned()))
        .collect()
}

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

/// The first argument of every `t(` and `tf(` call before the file's test module, a call
/// split over several lines included.
fn keys_in(text: &str) -> Vec<String> {
    let code = text.split("#[cfg(test)]").next().unwrap_or("");
    let mut out = Vec::new();
    for call in ["t(", "tf("] {
        let mut rest = code;
        while let Some(at) = rest.find(call) {
            let before = rest[..at].chars().last();
            let after = rest[at + call.len()..].trim_start();
            rest = &rest[at + call.len()..];
            if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                continue;
            }
            if let Some(quoted) = after.strip_prefix('"') {
                if let Some(end) = quoted.find('"') {
                    out.push(quoted[..end].to_owned());
                }
            }
        }
    }
    out
}

#[test]
fn both_locales_have_the_same_keys_and_placeholders() {
    let (fr, en) = (table(FR), table(EN));
    let fr_keys: BTreeSet<_> = fr.keys().collect();
    let en_keys: BTreeSet<_> = en.keys().collect();
    assert_eq!(
        fr_keys.symmetric_difference(&en_keys).collect::<Vec<_>>(),
        Vec::<&&String>::new(),
        "keys in one locale only"
    );
    for (key, text) in &fr {
        assert!(
            !text.trim().is_empty() && !en[key].trim().is_empty(),
            "{key}: empty"
        );
        assert_eq!(
            placeholders(text),
            placeholders(&en[key]),
            "{key}: placeholders"
        );
        for t in [text, &en[key]] {
            assert!(!t.contains('\u{2014}'), "{key}: em dash");
        }
    }
}

#[test]
fn every_key_asked_for_exists_and_every_key_is_used() {
    let fr = table(FR);
    let mut files = Vec::new();
    rust_files(Path::new(SRC), &mut files);
    let mut used = BTreeSet::new();
    let mut missing = Vec::new();
    for file in &files {
        let text = std::fs::read_to_string(file).unwrap();
        for key in keys_in(&text) {
            if !fr.contains_key(&key) {
                missing.push(format!("{}: {key}", file.display()));
            }
            used.insert(key);
        }
    }
    assert!(missing.is_empty(), "missing keys:\n{}", missing.join("\n"));
    let unused: Vec<_> = fr.keys().filter(|k| !used.contains(*k)).collect();
    assert!(unused.is_empty(), "keys nobody asks for: {unused:?}");
}

#[test]
fn the_key_scanner_reads_split_calls_and_skips_lookalikes() {
    let text = "let a = t(\"x.y\");\nlet b = tf(\n    \"z.w\",\n &[]);\nlet c = fmt(\"no\");";
    assert_eq!(keys_in(text), ["x.y", "z.w"]);
}
