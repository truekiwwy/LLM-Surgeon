//! The experiment ledger: `ledger.jsonl`, one line per finished run, only ever appended.
//! A line records what is needed to judge and redo the run: parameters, hashes (model,
//! input files, studio binary, seed), versions, result and verdict.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Entry {
    pub run_id: String,
    pub tool: String,
    pub label: String,
    /// The tool arguments as submitted (`{out}` not yet resolved).
    pub params: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub resumed_from: Option<String>,
    pub started_unix: u64,
    pub finished_unix: u64,
    pub hashes: Hashes,
    pub versions: Versions,
    pub verdict: Verdict,
    pub exit_code: i32,
    /// `data` of the JSON envelope.
    pub result: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Hashes {
    /// The project's source SHA-256 when the run read it, else the engine's fast digest
    /// (`fast-...`) of the working copy, taken before the run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// SHA-256 of every input file named in the arguments, by argument name.
    pub inputs: BTreeMap<String, String>,
    /// SHA-256 of the studio executable (the engine is linked into it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub binary: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Versions {
    pub studio: String,
    pub events: u32,
    pub manifest: u32,
}

impl Default for Versions {
    fn default() -> Self {
        Self {
            studio: env!("CARGO_PKG_VERSION").into(),
            events: weight_surgeon::events::EVENTS_VERSION,
            manifest: super::project::MANIFEST_VERSION,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    Ok,
    Failed,
    Cancelled,
    /// The studio stopped while it ran.
    Interrupted,
}

/// Appends one line with a single write; earlier lines are never rewritten.
pub fn append(path: &Path, entry: &Entry) -> Result<(), String> {
    let mut line = serde_json::to_vec(entry).map_err(|e| e.to_string())?;
    line.push(b'\n');
    std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| f.write_all(&line).and_then(|()| f.sync_data()))
        .map_err(|e| format!("{} : {e}", path.display()))
}

pub fn read(path: &Path) -> Result<Vec<Entry>, String> {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{} : {e}", path.display())),
    };
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l).map_err(|e| format!("{} ligne {} : {e}", path.display(), i + 1))
        })
        .collect()
}

pub fn to_json(entries: &[Entry]) -> String {
    serde_json::to_string_pretty(entries).unwrap_or_default() + "\n"
}

pub fn to_markdown(entries: &[Entry]) -> String {
    let mut out = String::from(
        "| run | outil | libellé | verdict | code | modèle | début (unix) | durée (s) |\n\
         |---|---|---|---|---|---|---|---|\n",
    );
    for e in entries {
        let model = e.hashes.model.as_deref().unwrap_or("-");
        let short = model.get(..model.len().min(17)).unwrap_or(model);
        out.push_str(&format!(
            "| {} | {} | {} | {:?} | {} | `{short}` | {} | {} |\n",
            e.run_id,
            e.tool,
            e.label.replace('|', "\\|"),
            e.verdict,
            e.exit_code,
            e.started_unix,
            e.finished_unix.saturating_sub(e.started_unix)
        ));
    }
    out
}

/// SHA-256 of the running executable, computed once per session.
pub fn binary_sha256() -> Option<String> {
    static HASH: OnceLock<Option<String>> = OnceLock::new();
    HASH.get_or_init(|| {
        let exe = std::env::current_exe().ok()?;
        weight_surgeon::pipeline::job::sha256_file(&exe).ok()
    })
    .clone()
}

/// The hashes of a run's inputs, taken before it starts. Relative paths are read from `base`
/// (the project folder, as the engine's sandbox resolves them); `source` is the project's
/// model.
pub fn hash_inputs(args: &Value, base: &Path, source: Option<(&Path, &str)>) -> Hashes {
    use weight_surgeon::pipeline::job::sha256_file;
    // Large files (datasets are small, models are not) are left to the model hash.
    const MAX_INPUT_BYTES: u64 = 512 << 20;
    let mut hashes = Hashes {
        seed: args["seed"].as_u64(),
        binary: binary_sha256(),
        ..Hashes::default()
    };
    let Some(map) = args.as_object() else {
        return hashes;
    };
    for (key, value) in map {
        let Some(path) = value.as_str().map(|s| base.join(s)) else {
            continue;
        };
        if key == "model" {
            hashes.model = model_hash(&path, source);
            continue;
        }
        let small =
            std::fs::metadata(&path).is_ok_and(|m| m.is_file() && m.len() <= MAX_INPUT_BYTES);
        if let Some(sha) = small.then(|| sha256_file(&path).ok()).flatten() {
            hashes.inputs.insert(key.clone(), sha);
        }
    }
    hashes
}

fn model_hash(path: &Path, source: Option<(&Path, &str)>) -> Option<String> {
    let canonical = std::fs::canonicalize(path).ok()?;
    if let Some((src, sha)) = source {
        if canonical == src {
            return Some(sha.to_owned());
        }
    }
    let files = super::copy::plan(&canonical).ok()?.files;
    weight_surgeon::graph::model::hash_files_fast(&files).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: &str, verdict: Verdict) -> Entry {
        Entry {
            run_id: id.into(),
            tool: "surgeon_check".into(),
            label: "a | b".into(),
            params: serde_json::json!({"model": "m"}),
            resumed_from: None,
            started_unix: 10,
            finished_unix: 12,
            hashes: Hashes::default(),
            versions: Versions::default(),
            verdict,
            exit_code: 0,
            result: Value::Null,
            error: None,
        }
    }

    #[test]
    fn appending_never_rewrites_earlier_lines() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ledger.jsonl");
        append(&path, &entry("r1", Verdict::Ok)).unwrap();
        let first = std::fs::read(&path).unwrap();
        append(&path, &entry("r2", Verdict::Cancelled)).unwrap();
        let both = std::fs::read(&path).unwrap();
        assert_eq!(&both[..first.len()], &first[..]);
        let entries = read(&path).unwrap();
        assert_eq!(
            entries,
            vec![entry("r1", Verdict::Ok), entry("r2", Verdict::Cancelled)]
        );
    }

    #[test]
    fn exports_list_every_run() {
        let entries = [entry("r1", Verdict::Ok), entry("r2", Verdict::Failed)];
        let json: Value = serde_json::from_str(&to_json(&entries)).unwrap();
        assert_eq!(json.as_array().unwrap().len(), 2);
        let md = to_markdown(&entries);
        assert_eq!(md.lines().count(), 4);
        assert!(
            md.contains("| r2 | surgeon_check | a \\| b | Failed | 0 |"),
            "{md}"
        );
    }

    #[test]
    fn inputs_are_hashed_and_the_source_keeps_its_recorded_hash() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("d.jsonl");
        std::fs::write(&data, b"{}").unwrap();
        let model = dir.path().join("m.gguf");
        std::fs::write(&model, b"w").unwrap();
        let src = std::fs::canonicalize(&model).unwrap();
        let args = serde_json::json!({
            "model": model, "dataset": data, "seed": 7, "missing": "nope.txt", "limit": 3
        });
        let h = hash_inputs(&args, dir.path(), Some((&src, "recorded")));
        assert_eq!(h.model.as_deref(), Some("recorded"));
        assert_eq!(h.seed, Some(7));
        assert_eq!(h.inputs.keys().collect::<Vec<_>>(), ["dataset"]);
        let other = hash_inputs(&args, dir.path(), None);
        assert!(other.model.unwrap().starts_with("fast-"));
    }
}
