//! Persistent cache of the baseline of `bench eval` and `bench refusal`: the reference model
//! is not patched between two experiments, so its answers are generated once per set of
//! inputs and read back afterwards.
//!
//! ```text
//! key   = SHA-256 of {schema, kind, model files, dataset or prompts, every option that
//!         changes the answers, backend, surgeon version}           (canonical JSON)
//! entry = <dir>/<key>.json {schema, key, kind, model, created_at, timing_ms, items}
//! get   -> unreadable, other schema or other key: deleted, then a miss
//! put   -> temporary file then rename (two runs never see half an entry)
//! ```
//!
//! The directory is `$WEIGHT_SURGEON_CACHE_DIR`, else `%LOCALAPPDATA%\weight-surgeon\cache\
//! ref_bench` on Windows and `$XDG_CACHE_HOME/weight-surgeon/ref_bench` (else
//! `~/.cache/...`) elsewhere. Like the user configuration, the default location is ignored
//! under `WEIGHT_SURGEON_NO_USER_CONFIG=1` (the test harness): only an explicit directory
//! is used then. `--no-ref-cache` regenerates and overwrites the entry; deleting the
//! directory purges the cache.

use std::path::{Path, PathBuf};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::error::{BenchError, Result};
use super::{sha256_file, sha256_hex};

/// Changes with the layout of an entry or of its key: older entries are then misses.
pub const REF_CACHE_SCHEMA_VERSION: u32 = 1;
/// A cache directory at this exact path, whatever the platform.
pub const CACHE_DIR_ENV: &str = "WEIGHT_SURGEON_CACHE_DIR";

/// Whether the baseline came from the cache, for the report and the log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lookup {
    Hit,
    Miss,
}

impl Lookup {
    pub fn as_str(self) -> &'static str {
        match self {
            Lookup::Hit => "hit",
            Lookup::Miss => "miss",
        }
    }
}

/// The cache directory a run uses when nothing says otherwise; `None` when there is none.
pub fn default_dir() -> Option<PathBuf> {
    let env = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty());
    if let Some(dir) = env(CACHE_DIR_ENV) {
        return Some(PathBuf::from(dir));
    }
    if env(crate::backend::config::NO_USER_CONFIG_ENV).is_some_and(|v| v != "0") {
        return None;
    }
    let base = match cfg!(windows) {
        true => env("LOCALAPPDATA").map(|d| PathBuf::from(d).join("weight-surgeon").join("cache")),
        false => env("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| env("HOME").map(|h| PathBuf::from(h).join(".cache")))
            .map(|d| d.join("weight-surgeon")),
    };
    base.map(|b| b.join("ref_bench"))
}

/// SHA-256 of a model: of a file (GGUF), or of the regular files at the top of a folder,
/// names and contents in name order. Hidden files (`.hipfire-alias.json`...) do not change
/// what the model answers and are left out.
pub fn model_sha256(model: &Path) -> Result<String> {
    if model.is_file() {
        return sha256_file(model);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(model)
        .map_err(|e| BenchError::input(model, e))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            !p.file_name()
                .is_some_and(|n| n.to_string_lossy().starts_with('.'))
        })
        .collect();
    files.sort();
    let mut manifest = String::new();
    for file in &files {
        let name = file.file_name().unwrap_or_default().to_string_lossy();
        manifest.push_str(&format!("{name}\0{}\n", sha256_file(file)?));
    }
    Ok(sha256_hex(manifest.as_bytes()))
}

/// The key of an entry: `kind` (`eval`, `refusal`), the model's [`model_sha256`] and
/// `inputs`, everything else that changes the answers.
pub fn key(kind: &str, model_sha256: &str, inputs: Value) -> String {
    let canonical = serde_json::json!({
        "schema": REF_CACHE_SCHEMA_VERSION,
        "kind": kind,
        "model_sha256": model_sha256,
        "inputs": inputs,
        "surgeon_version": env!("CARGO_PKG_VERSION"),
    });
    // serde_json maps keep their keys sorted: the text, and so the digest, is canonical.
    sha256_hex(canonical.to_string().as_bytes())
}

#[derive(Serialize, Deserialize)]
struct Entry<T> {
    schema: u32,
    key: String,
    kind: String,
    model: PathBuf,
    created_at: String,
    /// What the generation took when it ran.
    timing_ms: u64,
    items: T,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefCache {
    dir: PathBuf,
    /// `--no-ref-cache`: nothing is read, every entry is generated again and overwritten.
    regenerate: bool,
}

impl RefCache {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            regenerate: false,
        }
    }

    /// The cache at [`default_dir`], if any.
    pub fn from_env() -> Option<Self> {
        default_dir().map(Self::new)
    }

    pub fn regenerating(self, regenerate: bool) -> Self {
        Self { regenerate, ..self }
    }

    fn path(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.json"))
    }

    /// The items stored under `key`. An entry that cannot be read back as such (truncated,
    /// edited, another schema) is deleted and said in `notes`: the caller regenerates it.
    pub fn get<T: DeserializeOwned>(&self, key: &str, notes: &mut Vec<String>) -> Option<T> {
        if self.regenerate {
            return None;
        }
        let path = self.path(key);
        let text = std::fs::read_to_string(&path).ok()?;
        match serde_json::from_str::<Entry<T>>(&text) {
            Ok(entry) if entry.schema == REF_CACHE_SCHEMA_VERSION && entry.key == key => {
                Some(entry.items)
            }
            _ => {
                let done = match std::fs::remove_file(&path) {
                    Ok(()) => "supprimée",
                    Err(_) => "ignorée",
                };
                notes.push(format!(
                    "cache des références : entrée illisible {} {done}, référence régénérée",
                    path.display()
                ));
                None
            }
        }
    }

    /// Writes `items` under `key`, atomically; the directory is created when missing.
    pub fn put<T: Serialize>(
        &self,
        key: &str,
        kind: &str,
        model: &Path,
        timing_ms: u64,
        items: &T,
    ) -> Result<()> {
        std::fs::create_dir_all(&self.dir).map_err(|e| io(&self.dir, e))?;
        let entry = Entry {
            schema: REF_CACHE_SCHEMA_VERSION,
            key: key.to_owned(),
            kind: kind.to_owned(),
            model: model.to_path_buf(),
            created_at: crate::pipeline::job::now_rfc3339(),
            timing_ms,
            items,
        };
        let bytes = serde_json::to_vec(&entry).map_err(|e| BenchError::Compute(e.to_string()))?;
        let path = self.path(key);
        crate::extract::write_atomic(&path, &bytes, true).map_err(|e| io(&path, e))
    }
}

/// What `surgeon doctor` finds in a cache directory.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct Scan {
    /// Entries a lookup can read back.
    pub entries: usize,
    /// Every file of the directory, in bytes.
    pub bytes: u64,
    /// Unreadable, or holding another key than its name: a lookup would delete it.
    pub corrupt: Vec<PathBuf>,
    /// Another schema version, or a temporary file a killed run left: never read again.
    pub obsolete: Vec<PathBuf>,
}

/// The entries of `dir` (a missing directory is an empty cache).
pub fn scan(dir: &Path) -> Scan {
    let mut scan = Scan::default();
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        if !meta.is_file() {
            continue;
        }
        scan.bytes += meta.len();
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".tmp") {
            scan.obsolete.push(path);
            continue;
        }
        let Some(stem) = name.strip_suffix(".json") else {
            continue;
        };
        let entry = std::fs::read_to_string(&path)
            .ok()
            .and_then(|text| serde_json::from_str::<Entry<Value>>(&text).ok());
        match entry {
            Some(e) if e.key != stem => scan.corrupt.push(path),
            Some(e) if e.schema != REF_CACHE_SCHEMA_VERSION => scan.obsolete.push(path),
            Some(_) => scan.entries += 1,
            None => scan.corrupt.push(path),
        }
    }
    scan.corrupt.sort();
    scan.obsolete.sort();
    scan
}

/// Removes what `scan` found unreadable or obsolete; the number of files removed.
pub fn purge(scan: &Scan) -> usize {
    scan.corrupt
        .iter()
        .chain(&scan.obsolete)
        .filter(|p| std::fs::remove_file(p).is_ok())
        .count()
}

fn io(path: &Path, e: impl std::fmt::Display) -> BenchError {
    BenchError::Io {
        path: path.to_path_buf(),
        message: e.to_string(),
    }
}

/// A baseline and where it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct Cached<T> {
    pub items: T,
    /// What the generation took; 0 on a hit (the caller times the lookup).
    pub timing_ms: u64,
    pub lookup: Lookup,
    /// For the report's warnings: an entry deleted, an entry not written.
    pub notes: Vec<String>,
}

/// The baseline under `key` from `cache`, else `generate()` then stored. Without a
/// cache, `generate()` and [`Lookup::Miss`]. A failed store is a note, not an error: the
/// measure itself succeeded.
pub fn cached<T: Serialize + DeserializeOwned>(
    cache: Option<&RefCache>,
    key: &str,
    kind: &str,
    model: &Path,
    generate: impl FnOnce() -> Result<(T, u64)>,
) -> Result<Cached<T>> {
    let mut notes = Vec::new();
    if let Some(items) = cache.and_then(|c| c.get(key, &mut notes)) {
        return Ok(Cached {
            items,
            timing_ms: 0,
            lookup: Lookup::Hit,
            notes,
        });
    }
    let (items, timing_ms) = generate()?;
    if let Some(Err(e)) = cache.map(|c| c.put(key, kind, model, timing_ms, &items)) {
        notes.push(format!("cache des références non écrit : {e}"));
    }
    Ok(Cached {
        items,
        timing_ms,
        lookup: Lookup::Miss,
        notes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn inputs(max_tokens: u32, dataset_sha256: &str) -> Value {
        serde_json::json!({"max_new_tokens": max_tokens, "dataset_sha256": dataset_sha256})
    }

    #[test]
    fn a_miss_is_written_then_read_back_and_any_input_change_is_a_new_key() {
        let dir = tempfile::tempdir().unwrap();
        let cache = RefCache::new(dir.path().join("ref_bench"));
        let model = dir.path().join("model");
        let runs = Cell::new(0);
        let generate = || {
            runs.set(runs.get() + 1);
            Ok((vec!["a".to_owned(), "b".to_owned()], 1234))
        };
        let base = key("eval", "m1", inputs(256, "d1"));

        let first = cached(Some(&cache), &base, "eval", &model, generate).unwrap();
        assert_eq!(
            (first.lookup, first.timing_ms, runs.get()),
            (Lookup::Miss, 1234, 1)
        );
        assert!(cache.path(&base).is_file(), "written on a miss");
        let again = cached(Some(&cache), &base, "eval", &model, generate).unwrap();
        assert_eq!(
            (again.lookup, runs.get()),
            (Lookup::Hit, 1),
            "not generated again"
        );
        assert_eq!(again.items, first.items);

        // Another token budget, dataset, model or kind: another entry.
        for other in [
            key("eval", "m1", inputs(40, "d1")),
            key("eval", "m1", inputs(256, "d2")),
            key("eval", "m2", inputs(256, "d1")),
            key("refusal", "m1", inputs(256, "d1")),
        ] {
            assert_ne!(other, base);
            let run = cached::<Vec<String>>(Some(&cache), &other, "eval", &model, generate);
            assert_eq!(run.unwrap().lookup, Lookup::Miss);
        }
        assert_eq!(runs.get(), 5);
        assert_eq!(base, key("eval", "m1", inputs(256, "d1")), "deterministic");
        assert_eq!(base.len(), 64);
    }

    #[test]
    fn a_corrupt_entry_is_deleted_and_regenerated() {
        let dir = tempfile::tempdir().unwrap();
        let cache = RefCache::new(dir.path());
        let model = dir.path().join("model");
        let k = key("refusal", "m", inputs(40, "p"));
        let generate = || Ok((vec![1u32, 2, 3], 7));
        for bad in [
            "{\"schema\": 1, \"items\": [1, 2".to_owned(),
            format!("{{\"schema\":99,\"key\":\"{k}\",\"kind\":\"refusal\",\"model\":\"m\",\"created_at\":\"x\",\"timing_ms\":1,\"items\":[9]}}"),
            "{\"schema\":1,\"key\":\"another\",\"kind\":\"refusal\",\"model\":\"m\",\"created_at\":\"x\",\"timing_ms\":1,\"items\":[9]}".to_owned(),
        ] {
            std::fs::write(cache.path(&k), bad).unwrap();
            let mut notes = Vec::new();
            assert_eq!(cache.get::<Vec<u32>>(&k, &mut notes), None);
            assert!(!cache.path(&k).exists(), "deleted");
            assert!(notes[0].contains("entrée illisible"), "{notes:?}");
            std::fs::write(cache.path(&k), b"{").unwrap();
            let run = cached(Some(&cache), &k, "refusal", &model, generate).unwrap();
            assert_eq!((run.items, run.lookup), (vec![1, 2, 3], Lookup::Miss));
            assert_eq!(run.notes.len(), 1, "the report says it");
            assert_eq!(cache.get::<Vec<u32>>(&k, &mut notes), Some(vec![1, 2, 3]));
        }
        // Without a cache: always generated, never stored.
        assert_eq!(
            cached(None, &k, "refusal", &model, generate)
                .unwrap()
                .lookup,
            Lookup::Miss
        );
        // --no-ref-cache: the entry is not read, but overwritten with the fresh answers.
        let fresh = cache.clone().regenerating(true);
        let run = cached(Some(&fresh), &k, "refusal", &model, || Ok((vec![4u32], 1))).unwrap();
        assert_eq!((run.items, run.lookup), (vec![4], Lookup::Miss));
        assert_eq!(cache.get::<Vec<u32>>(&k, &mut Vec::new()), Some(vec![4]));
    }

    #[test]
    fn the_model_fingerprint_covers_its_files_not_hidden_ones() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path();
        std::fs::write(model.join("model.safetensors"), b"w").unwrap();
        std::fs::write(model.join("config.json"), b"{}").unwrap();
        let first = model_sha256(model).unwrap();
        std::fs::write(model.join(".hipfire-alias.json"), b"{}").unwrap();
        assert_eq!(model_sha256(model).unwrap(), first);
        std::fs::write(model.join("config.json"), b"{\"a\":1}").unwrap();
        assert_ne!(model_sha256(model).unwrap(), first);
        let gguf = model.join("config.json");
        assert_eq!(model_sha256(&gguf).unwrap(), sha256_file(&gguf).unwrap());
    }
}
