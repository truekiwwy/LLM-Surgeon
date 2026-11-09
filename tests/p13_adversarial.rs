//! Adversarial tests of the P13 code: structured fuzzing of the `.surgery` reader and writer,
//! rSVD properties, hostile LoRA inputs, the apply kernel against an independent reference,
//! the preview line editor, and cancellation from another thread.
//!
//! Everything is deterministic: fixed seeds, an in-file xorshift generator, no new dependency.
//! A test that exposes a real defect is kept and marked `#[ignore = "BUG: ..."]`.

use std::fs;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use half::{bf16, f16};
use ratatui::backend::TestBackend;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Terminal;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use sysinfo::{Pid, ProcessesToUpdate, System};
use tempfile::TempDir;
use unicode_width::UnicodeWidthChar;
use weight_surgeon::engine::DType;
use weight_surgeon::extract::ChatTemplate;
use weight_surgeon::lora::cli_args::LoraExtractArgs;
use weight_surgeon::lora::config::AdapterConfig;
use weight_surgeon::lora::extract::run_extract;
use weight_surgeon::lora::keys::{
    adapter_key, is_dora_key, is_embedding_key, pair_keys, parse_key, Side,
};
use weight_surgeon::lora::merge::{run_merge, MergeReport, MergeRequest};
use weight_surgeon::patch::{BackupSet, RollbackHandle};
use weight_surgeon::surgery::apply::{run_apply, SurgeryApplyRequest};
use weight_surgeon::surgery::cli_args::{DiffArgs, OnHighError, VerifyBase};
use weight_surgeon::surgery::diff::run_diff;
use weight_surgeon::surgery::error::{Result as SurgeryResult, SurgeryError};
use weight_surgeon::surgery::events::{NullObserver, SurgeryEvent, SurgeryObserver};
use weight_surgeon::surgery::format::{
    sha256_hex, Compression, CompressionAlgo, FactorDtype, FormatError, LowRankFactors, MAGIC,
};
use weight_surgeon::surgery::lowrank::{apply_full_delta, apply_lowrank_rows, LowRankStats};
use weight_surgeon::surgery::reader::SurgeryReader;
use weight_surgeon::surgery::rsvd::{rsvd, RowSource, Rsvd, RsvdParams};
use weight_surgeon::surgery::target::resolve_shards;
use weight_surgeon::surgery::writer::{HeaderInit, SurgeryWriter, TensorMeta};
use weight_surgeon::tui::tabs::preview::editor::LineEditor;
use weight_surgeon::tui::tabs::preview::{
    update, view, Clock, Event, Pane, PreviewConfig, PreviewEffect, PreviewMsg, PreviewState,
};
use weight_surgeon::tui::Theme;
use weight_surgeon::ui::ColorMode;

// ------------------------------------------------------------------ shared helpers

/// xorshift64, seeded through a multiplicative hash so that small seeds differ.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `0..n` (`n` clamped to at least 1).
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() >> 11) as usize % n.max(1)
    }

    /// Uniform in `lo..=hi`.
    fn between(&mut self, lo: usize, hi: usize) -> usize {
        lo + self.below(hi - lo + 1)
    }

    /// Uniform in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform in `[-1, 1)`.
    fn signed(&mut self) -> f64 {
        self.unit() * 2.0 - 1.0
    }

    /// Standard normal (Box-Muller).
    fn gauss(&mut self) -> f64 {
        let u1 = 1.0 - self.unit();
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

/// Peak growth of the process commit charge while `f` runs, sampled by a watcher thread.
/// Other tests of this file run concurrently, but they allocate tens of MiB at most: the
/// callers only compare the result with a threshold of hundreds of MiB.
static MEMORY_PROBE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn peak_commit_growth<T>(f: impl FnOnce() -> T) -> (T, u64) {
    // One measurement at a time: a concurrent one would be charged for this one's memory.
    let _serial = MEMORY_PROBE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    fn commit(sys: &mut Option<System>, pid: Pid) -> u64 {
        let Some(sys) = sys.as_mut() else { return 0 };
        sys.refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        sys.process(pid).map_or(0, |p| p.virtual_memory())
    }
    let pid = Pid::from_u32(std::process::id());
    let base = commit(&mut System::new().ok(), pid);
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let watcher = {
        let (stop, peak) = (stop.clone(), peak.clone());
        std::thread::spawn(move || {
            let mut sys = System::new().ok();
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(commit(&mut sys, pid), Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(1));
            }
        })
    };
    let out = f();
    stop.store(true, Ordering::Relaxed);
    let _ = watcher.join();
    (out, peak.load(Ordering::Relaxed).saturating_sub(base))
}

const MEM_LIMIT: u64 = 400 << 20;

fn file_names(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

// ------------------------------------------------------------------ .surgery fixtures

const PRE: usize = 20;
const FOOT: usize = 36;

fn deflate(level: u8) -> Compression {
    Compression {
        algo: CompressionAlgo::Deflate,
        level,
    }
}

fn stored() -> Compression {
    Compression {
        algo: CompressionAlgo::None,
        level: 0,
    }
}

fn init(dtype: FactorDtype, compression: Compression) -> HeaderInit {
    HeaderInit {
        created_with: "adversary".into(),
        base_name: "base".into(),
        base_tensor_count: 3,
        tuned_name: "tuned".into(),
        rank: 4,
        factor_dtype: dtype,
        compression,
        timestamp: None,
    }
}

fn meta(name: &str, shape: &[usize]) -> TensorMeta {
    TensorMeta {
        name: name.into(),
        shape: shape.to_vec(),
        target_dtype: "BF16".into(),
        base_sha256: "ab".repeat(32),
        rel_error: 0.25,
        energy: 0.75,
    }
}

/// Multiples of 1/8 in [-1, 1]: exact in f16, bf16 and f32.
fn eighths(rng: &mut Rng, n: usize) -> Vec<f32> {
    (0..n).map(|_| rng.below(17) as f32 / 8.0 - 1.0).collect()
}

fn factors(rng: &mut Rng, m: usize, n: usize, r: usize) -> LowRankFactors {
    LowRankFactors {
        u: eighths(rng, m * r),
        s: eighths(rng, r).iter().map(|x| x.abs() + 0.5).collect(),
        v: eighths(rng, r * n),
    }
}

/// `a.weight` [6,5] rank 2 (U and V differ in size), `b.weight` [4,4] rank 3 (U and V have the
/// same size), `c.bias` [8] full delta. Header rank 4.
fn sample(dtype: FactorDtype, compression: Compression) -> Vec<u8> {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("s.surgery");
    let mut rng = Rng::new(7);
    let mut w = SurgeryWriter::create(&path, init(dtype, compression), false).unwrap();
    w.add_lowrank(meta("a.weight", &[6, 5]), &factors(&mut rng, 6, 5, 2))
        .unwrap();
    w.add_lowrank(meta("b.weight", &[4, 4]), &factors(&mut rng, 4, 4, 3))
        .unwrap();
    w.add_full_delta(meta("c.bias", &[8]), &eighths(&mut rng, 8))
        .unwrap();
    w.finish().unwrap();
    fs::read(path).unwrap()
}

fn header_len(bytes: &[u8]) -> usize {
    u64::from_le_bytes(bytes[12..20].try_into().unwrap()) as usize
}

/// Preamble, header, blobs and a fresh footer over everything before it.
fn assemble(json: &[u8], blobs: &[u8]) -> Vec<u8> {
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&[0u8; 6]);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(json);
    out.extend_from_slice(blobs);
    refresh_footer(&mut out);
    out
}

fn refresh_footer(bytes: &mut Vec<u8>) {
    let digest = Sha256::digest(&bytes[..]);
    bytes.extend_from_slice(&digest);
    bytes.extend_from_slice(&MAGIC);
}

/// Adds `delta` to every blob offset of a header value.
fn relocate(header: &mut Value, delta: i64) {
    for tensor in header["tensors"].as_array_mut().into_iter().flatten() {
        for blob in tensor["blobs"]
            .as_object_mut()
            .into_iter()
            .flat_map(|b| b.values_mut())
        {
            if let Some(off) = blob["offset"].as_u64() {
                blob["offset"] = json!((off as i64 + delta) as u64);
            }
        }
    }
}

type Mutation<'a> = &'a dyn Fn(&mut Value, &mut Vec<u8>, u64);

/// Rebuilds a file around a mutated header value and blob section with a VALID global hash, so
/// that only the mutation can make it fail. Blob offsets are rebased on the final header length
/// before `mutate` runs; `mutate` gets the absolute start of the blob section.
fn forge(orig: &[u8], mutate: Mutation) -> Vec<u8> {
    let hl = header_len(orig);
    let base: Value = serde_json::from_slice(&orig[PRE..PRE + hl]).unwrap();
    let blobs = &orig[PRE + hl..orig.len() - FOOT];
    let mut len = hl;
    for _ in 0..8 {
        let mut header = base.clone();
        relocate(&mut header, len as i64 - hl as i64);
        let mut blobs = blobs.to_vec();
        mutate(&mut header, &mut blobs, (PRE + len) as u64);
        let mut json = serde_json::to_vec(&header).unwrap();
        if json.len() <= len {
            json.resize(len, b' ');
            return assemble(&json, &blobs);
        }
        len = json.len();
    }
    panic!("forge: the header length does not converge");
}

/// Raw (inflated) bytes of one blob of the forged file.
fn blob_raw(header: &Value, blobs: &[u8], start: u64, t: usize, role: &str) -> Vec<u8> {
    let b = &header["tensors"][t]["blobs"][role];
    let from = (b["offset"].as_u64().unwrap() - start) as usize;
    let stored = &blobs[from..from + b["comp_len"].as_u64().unwrap() as usize];
    match header["compression"]["algo"].as_str() {
        Some("none") => stored.to_vec(),
        _ => miniz_oxide::inflate::decompress_to_vec(stored).unwrap(),
    }
}

/// Replaces the stored bytes of one blob (offsets after it are shifted) and sets `comp_len`,
/// `raw_len` and the blob hash (of `hashed`) so that only the intended defect remains.
#[allow(clippy::too_many_arguments)]
fn set_blob(
    header: &mut Value,
    blobs: &mut Vec<u8>,
    start: u64,
    (t, role): (usize, &str),
    stored: &[u8],
    raw_len: u64,
    hashed: &[u8],
) {
    let old = header["tensors"][t]["blobs"][role].clone();
    let (offset, old_len) = (
        old["offset"].as_u64().unwrap(),
        old["comp_len"].as_u64().unwrap(),
    );
    let from = (offset - start) as usize;
    blobs.splice(from..from + old_len as usize, stored.iter().copied());
    let shift = stored.len() as i64 - old_len as i64;
    for tensor in header["tensors"].as_array_mut().unwrap() {
        for blob in tensor["blobs"].as_object_mut().unwrap().values_mut() {
            let off = blob["offset"].as_u64().unwrap();
            if off > offset {
                blob["offset"] = json!((off as i64 + shift) as u64);
            }
        }
    }
    let b = &mut header["tensors"][t]["blobs"][role];
    b["comp_len"] = json!(stored.len());
    b["raw_len"] = json!(raw_len);
    b["sha256"] = json!(sha256_hex(hashed));
}

fn compress(raw: &[u8]) -> Vec<u8> {
    miniz_oxide::deflate::compress_to_vec(raw, 1)
}

/// Opens `bytes` and reads every tensor. Full deltas come back in the `v` field.
fn open_and_read(dir: &Path, bytes: &[u8]) -> Result<Vec<LowRankFactors>, SurgeryError> {
    let path = dir.join("x.surgery");
    fs::write(&path, bytes).unwrap();
    let mut reader = SurgeryReader::open(&path)?;
    let mut out = Vec::new();
    for entry in reader.header().tensors.clone() {
        match entry.rank {
            0 => out.push(LowRankFactors {
                v: reader.read_full_delta(&entry)?,
                ..Default::default()
            }),
            _ => out.push(reader.read_lowrank(&entry)?),
        }
    }
    Ok(out)
}

/// The reader must refuse with a TYPED format error (not an io error, not a panic), fast and
/// without allocating on the strength of an unverified size.
fn assert_format_error(label: &str, bytes: &[u8]) {
    let dir = TempDir::new().unwrap();
    let started = Instant::now();
    let result = open_and_read(dir.path(), bytes);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{label}: too slow"
    );
    match result {
        Err(SurgeryError::Format(_)) => {}
        Err(other) => panic!("{label}: expected a format error, got {other}"),
        Ok(_) => panic!("{label}: hostile file accepted"),
    }
}

/// Runs `cases` (hostile files) under one memory measurement: nothing may allocate on the
/// strength of an unverified size.
fn assert_no_balloon(label: &str, cases: impl FnOnce()) {
    let ((), growth) = peak_commit_growth(cases);
    assert!(
        growth < MEM_LIMIT,
        "{label}: commit grew by {} MiB",
        growth >> 20
    );
}

// ------------------------------------------------------------------ 1. .surgery format

#[test]
fn memory_sampler_sees_a_large_allocation() {
    // Guards the guard: without this the memory assertions below could pass vacuously.
    let (len, growth) = peak_commit_growth(|| {
        let big = vec![1u8; 1 << 30];
        std::thread::sleep(Duration::from_millis(40));
        big.len()
    });
    assert_eq!(len, 1 << 30);
    assert!(growth > 512 << 20, "sampler saw only {} MiB", growth >> 20);
}

type Boxed = Box<dyn Fn(&mut Value, &mut Vec<u8>, u64)>;

/// Sets `pointer` (a JSON pointer whose parent exists) to `value`.
fn put(pointer: &str, value: Value) -> Boxed {
    let pointer = pointer.to_owned();
    Box::new(move |h, _, _| {
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        let slot = h.pointer_mut(parent).unwrap();
        match slot {
            Value::Array(a) => a[key.parse::<usize>().unwrap()] = value.clone(),
            _ => slot[key] = value.clone(),
        }
    })
}

fn remove(parent: &str, key: &str) -> Boxed {
    let (parent, key) = (parent.to_owned(), key.to_owned());
    Box::new(move |h, _, _| {
        h.pointer_mut(&parent)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .remove(&key);
    })
}

const ROLES: [(usize, &str); 5] = [(0, "u"), (0, "s"), (0, "v"), (1, "u"), (2, "delta")];

/// Every structural defect of a header that keeps the global hash valid, so that the deeper
/// validation layers (fields, blob sets, sizes, layout, hashes) are the ones exercised.
fn header_mutations() -> Vec<(String, Boxed)> {
    let mut m: Vec<(String, Boxed)> = Vec::new();
    let mut add = |label: &str, f: Boxed| m.push((label.to_owned(), f));
    add("duplicate name", put("/tensors/1/name", json!("a.weight")));
    add(
        "duplicate entry with another name, same blobs",
        Box::new(|h, _, _| {
            let mut copy = h["tensors"][0].clone();
            copy["name"] = json!("z.weight");
            h["tensors"].as_array_mut().unwrap().push(copy);
        }),
    );
    add(
        "duplicate entry, same name",
        Box::new(|h, _, _| {
            let copy = h["tensors"][0].clone();
            h["tensors"].as_array_mut().unwrap().push(copy);
        }),
    );
    for (label, name) in [
        ("NUL in name", "a\u{0}.weight".to_owned()),
        ("empty name", String::new()),
        ("newline in name", "a\n.weight".to_owned()),
        ("1025 byte name", "x".repeat(1025)),
        ("1 MiB name", "y".repeat(1 << 20)),
    ] {
        add(label, put("/tensors/0/name", json!(name)));
    }
    for (label, rank) in [
        ("lowrank rank 0", json!(0)),
        ("tensor rank above header rank", json!(5)),
        ("tensor rank u64::MAX", json!(u64::MAX)),
        ("tensor rank negative", json!(-1)),
        ("tensor rank float", json!(2.5)),
    ] {
        add(label, put("/tensors/0/rank", rank));
    }
    for (label, rank) in [
        ("header rank 0", json!(0)),
        ("header rank u64::MAX", json!(u64::MAX)),
        ("header rank 65537", json!(65537)),
        ("header rank below tensor ranks", json!(1)),
    ] {
        add(label, put("/rank", rank));
    }
    for (label, shape) in [
        ("rank 0 shape", json!([])),
        ("zero dimension", json!([0, 5])),
        ("usize::MAX squared", json!([u64::MAX, u64::MAX])),
        ("2^32 x 2^32", json!([1u64 << 32, 1u64 << 32])),
        ("3D lowrank", json!([6, 5, 1])),
        ("nine dimensions", json!([1, 1, 1, 1, 1, 1, 1, 1, 1])),
        ("usize::MAX x 1", json!([u64::MAX, 1])),
        ("string dimension", json!(["a", 5])),
        ("negative dimension", json!([-1, 5])),
        ("float dimension", json!([6.5, 5])),
    ] {
        add(label, put("/tensors/0/shape", shape));
    }
    for (label, shape) in [
        ("delta shape u64::MAX", json!([u64::MAX])),
        ("delta shape 2^40", json!([1u64 << 40])),
        ("delta shape with a zero", json!([8, 0])),
        ("delta shape 2^62 x 4", json!([1u64 << 62, 4])),
    ] {
        add(label, put("/tensors/2/shape", shape));
    }
    for (t, role) in ROLES {
        let at = move |field: &str| format!("/tensors/{t}/blobs/{role}/{field}");
        add(
            &format!("{t}/{role} offset u64::MAX"),
            put(&at("offset"), json!(u64::MAX)),
        );
        add(
            &format!("{t}/{role} offset 0"),
            put(&at("offset"), json!(0)),
        );
        add(
            &format!("{t}/{role} offset before the blob section"),
            Box::new(move |h, _, start| {
                h["tensors"][t]["blobs"][role]["offset"] = json!(start - 1)
            }),
        );
        add(
            &format!("{t}/{role} offset + comp_len overflows"),
            Box::new(move |h, _, _| {
                let b = &mut h["tensors"][t]["blobs"][role];
                b["offset"] = json!(u64::MAX - b["comp_len"].as_u64().unwrap() + 1);
            }),
        );
        add(
            &format!("{t}/{role} offset + 1"),
            Box::new(move |h, _, _| {
                let b = &mut h["tensors"][t]["blobs"][role];
                b["offset"] = json!(b["offset"].as_u64().unwrap() + 1);
            }),
        );
        for len in [0, 1, u64::MAX, u64::MAX / 2, 1 << 31, 1 << 32] {
            add(
                &format!("{t}/{role} comp_len {len}"),
                put(&at("comp_len"), json!(len)),
            );
            add(
                &format!("{t}/{role} raw_len {len}"),
                put(&at("raw_len"), json!(len)),
            );
        }
        for (label, hash) in [
            ("uppercase", "AB".repeat(32)),
            ("63 characters", "a".repeat(63)),
            ("not hex", "zz".repeat(32)),
            ("wrong value", "00".repeat(32)),
            ("empty", String::new()),
        ] {
            add(
                &format!("{t}/{role} blob sha256 {label}"),
                put(&at("sha256"), json!(hash)),
            );
        }
    }
    for (label, value) in [
        ("uppercase", "AB".repeat(32)),
        ("empty", String::new()),
        ("65 characters", "a".repeat(65)),
    ] {
        add(
            &format!("base_sha256 {label}"),
            put("/tensors/0/base_sha256", json!(value)),
        );
    }
    for dtype in ["F8", "bf16", "", "F64"] {
        add(
            &format!("target_dtype {dtype:?}"),
            put("/tensors/0/target_dtype", json!(dtype)),
        );
    }
    for (label, field, value) in [
        ("negative rel_error", "rel_error", json!(-1.0)),
        ("null rel_error", "rel_error", Value::Null),
        ("string rel_error", "rel_error", json!("0.5")),
        ("negative energy", "energy", json!(-0.5)),
        ("null energy", "energy", Value::Null),
    ] {
        add(label, put(&format!("/tensors/0/{field}"), value));
    }
    for (label, compression) in [
        ("none with a level", json!({"algo": "none", "level": 3})),
        ("deflate level 11", json!({"algo": "deflate", "level": 11})),
        (
            "deflate level 255",
            json!({"algo": "deflate", "level": 255}),
        ),
        (
            "deflate level 256",
            json!({"algo": "deflate", "level": 256}),
        ),
        ("zstd", json!({"algo": "zstd", "level": 1})),
        ("no level", json!({"algo": "none"})),
        ("null", Value::Null),
    ] {
        add(label, put("/compression", compression));
    }
    add(
        "compression flipped against the blobs",
        Box::new(|h, _, _| {
            let none = h["compression"]["algo"] == "none";
            h["compression"] = match none {
                true => json!({"algo": "deflate", "level": 1}),
                false => json!({"algo": "none", "level": 0}),
            };
        }),
    );
    for version in [json!(0), json!(2), json!(65535), json!(65536), json!("1")] {
        add(
            &format!("format_version {version}"),
            put("/format_version", version),
        );
    }
    add("factor_dtype f64", put("/factor_dtype", json!("f64")));
    add(
        "factor_dtype F16 uppercase",
        put("/factor_dtype", json!("F16")),
    );
    add(
        "lowrank with an extra delta blob",
        Box::new(|h, _, _| {
            h["tensors"][0]["blobs"]["delta"] = h["tensors"][0]["blobs"]["s"].clone()
        }),
    );
    add("lowrank without s", remove("/tensors/0/blobs", "s"));
    add("lowrank without u", remove("/tensors/0/blobs", "u"));
    add(
        "full delta with a u blob",
        Box::new(|h, _, _| {
            h["tensors"][2]["blobs"]["u"] = h["tensors"][2]["blobs"]["delta"].clone()
        }),
    );
    add("empty blobs", put("/tensors/0/blobs", json!({})));
    add("null blobs", put("/tensors/0/blobs", Value::Null));
    add(
        "lowrank turned full_delta",
        put("/tensors/0/kind", json!("full_delta")),
    );
    add(
        "full_delta turned lowrank",
        put("/tensors/2/kind", json!("lowrank")),
    );
    add("unknown kind", put("/tensors/0/kind", json!("bogus")));
    add("tensors null", put("/tensors", Value::Null));
    add("tensors object", put("/tensors", json!({})));
    add("tensors of numbers", put("/tensors", json!([1, 2, 3])));
    add("base null", put("/base", Value::Null));
    add(
        "provenance without timestamp",
        put("/provenance", json!({})),
    );
    add("root array", Box::new(|h, _, _| *h = json!([])));
    add("root null", Box::new(|h, _, _| *h = Value::Null));
    add(
        "u and s swapped",
        Box::new(|h, _, _| {
            let blobs = &mut h["tensors"][0]["blobs"];
            let u = blobs["u"].clone();
            blobs["u"] = blobs["s"].clone();
            blobs["s"] = u;
        }),
    );
    add(
        "u and v swapped (different sizes)",
        Box::new(|h, _, _| {
            let blobs = &mut h["tensors"][0]["blobs"];
            let u = blobs["u"].clone();
            blobs["u"] = blobs["v"].clone();
            blobs["v"] = u;
        }),
    );
    m
}

#[test]
fn header_mutations_with_a_valid_global_hash_are_typed_errors() {
    for (dtype, compression) in [(FactorDtype::F16, deflate(1)), (FactorDtype::F32, stored())] {
        let orig = sample(dtype, compression);
        // Control: the forge itself must produce a file the reader accepts.
        let control = forge(&orig, &|_, _, _| {});
        let dir = TempDir::new().unwrap();
        let reference = open_and_read(dir.path(), &orig).unwrap();
        assert_eq!(open_and_read(dir.path(), &control).unwrap(), reference);
        assert_no_balloon("header mutations", || {
            for (label, mutate) in header_mutations() {
                assert_format_error(&format!("{label} [{dtype:?}]"), &forge(&orig, &*mutate));
            }
        });
    }
}

#[test]
fn swapped_blob_roles_are_caught_by_sizes_but_not_when_sizes_agree() {
    let orig = sample(FactorDtype::F16, deflate(1));
    let dir = TempDir::new().unwrap();
    let reference = open_and_read(dir.path(), &orig).unwrap();
    // b.weight is 4x4 rank 3: U and V have the same size, so only the hash covers the content,
    // not the role. The file is "valid" (the global hash is) and reads back with U and V swapped.
    let swapped = forge(&orig, &|h, _, _| {
        let blobs = &mut h["tensors"][1]["blobs"];
        let u = blobs["u"].clone();
        blobs["u"] = blobs["v"].clone();
        blobs["v"] = u;
    });
    let got = open_and_read(dir.path(), &swapped).unwrap();
    assert_eq!(got[1].u, reference[1].v);
    assert_eq!(got[1].v, reference[1].u);
    assert_eq!(got[0], reference[0]);
}

fn replace_stream(orig: &[u8], build: impl Fn(&[u8]) -> (Vec<u8>, u64, Vec<u8>)) -> Vec<u8> {
    forge(orig, &|h, blobs, start| {
        let raw = blob_raw(h, blobs, start, 0, "u");
        let (stored, declared, hashed) = build(&raw);
        set_blob(h, blobs, start, (0, "u"), &stored, declared, &hashed);
    })
}

#[test]
fn deflate_streams_longer_or_shorter_than_raw_len_are_refused() {
    let orig = sample(FactorDtype::F16, deflate(1));
    let declared = |raw: &[u8]| raw.len() as u64;
    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "one byte more than raw_len, hash of the long data",
            replace_stream(&orig, |raw| {
                let long = [raw, &[0u8]].concat();
                (compress(&long), declared(raw), long)
            }),
        ),
        (
            "one byte more than raw_len, hash of the declared length",
            replace_stream(&orig, |raw| {
                let long = [raw, &[0u8]].concat();
                (compress(&long), declared(raw), raw.to_vec())
            }),
        ),
        (
            "1 MiB of zeros under a 24 byte raw_len",
            replace_stream(&orig, |raw| {
                (compress(&vec![0u8; 1 << 20]), declared(raw), raw.to_vec())
            }),
        ),
        (
            "two bytes fewer than raw_len",
            replace_stream(&orig, |raw| {
                let short = raw[..raw.len() - 2].to_vec();
                (compress(&short), declared(raw), short)
            }),
        ),
        (
            "empty stream content",
            replace_stream(&orig, |raw| (compress(&[]), declared(raw), Vec::new())),
        ),
        (
            "raw_len shrunk to match the short stream (shape says otherwise)",
            replace_stream(&orig, |raw| {
                let short = raw[..raw.len() - 2].to_vec();
                (compress(&short), short.len() as u64, short)
            }),
        ),
    ];
    for (label, bytes) in cases {
        assert_format_error(label, &bytes);
    }
}

#[test]
fn garbage_after_a_valid_deflate_stream_inside_a_blob_never_changes_the_values() {
    let orig = sample(FactorDtype::F16, deflate(1));
    let dir = TempDir::new().unwrap();
    let reference = open_and_read(dir.path(), &orig).unwrap();
    let bytes = replace_stream(&orig, |raw| {
        let mut stored = compress(raw);
        stored.extend_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF, 0, 1, 2]);
        (stored, raw.len() as u64, raw.to_vec())
    });
    // Accepted or refused are both defensible; a different decoded value is not.
    match open_and_read(dir.path(), &bytes) {
        Ok(got) => assert_eq!(got, reference),
        Err(SurgeryError::Format(_)) => {}
        Err(other) => panic!("untyped error: {other}"),
    }
}

fn nonfinite_patterns(dtype: FactorDtype) -> Vec<Vec<u8>> {
    match dtype {
        FactorDtype::F16 => [0x7E00u16, 0x7C00, 0xFC00, 0x7C01]
            .map(|b| b.to_le_bytes().to_vec())
            .to_vec(),
        FactorDtype::Bf16 => [0x7FC0u16, 0x7F80, 0xFF80, 0x7F81]
            .map(|b| b.to_le_bytes().to_vec())
            .to_vec(),
        FactorDtype::F32 => [0x7FC0_0000u32, 0x7F80_0000, 0xFF80_0000, 0x7F80_0001]
            .map(|b| b.to_le_bytes().to_vec())
            .to_vec(),
    }
}

#[test]
fn valid_hash_nan_and_inf_payloads_are_refused_at_read_in_every_dtype() {
    for dtype in [FactorDtype::F16, FactorDtype::Bf16, FactorDtype::F32] {
        for compression in [deflate(1), stored()] {
            let orig = sample(dtype, compression);
            for (t, role) in ROLES {
                let patterns = match role {
                    "delta" => nonfinite_patterns(FactorDtype::F32),
                    _ => nonfinite_patterns(dtype),
                };
                for pattern in patterns {
                    for last in [false, true] {
                        let bytes = forge(&orig, &|h, blobs, start| {
                            let mut raw = blob_raw(h, blobs, start, t, role);
                            let at = if last { raw.len() - pattern.len() } else { 0 };
                            raw[at..at + pattern.len()].copy_from_slice(&pattern);
                            let stored = match compression.algo {
                                CompressionAlgo::None => raw.clone(),
                                CompressionAlgo::Deflate => compress(&raw),
                            };
                            let len = raw.len() as u64;
                            set_blob(h, blobs, start, (t, role), &stored, len, &raw);
                        });
                        let dir = TempDir::new().unwrap();
                        let got = open_and_read(dir.path(), &bytes);
                        assert!(
                            matches!(
                                got,
                                Err(SurgeryError::Format(FormatError::NonFiniteValue { .. }))
                            ),
                            "{dtype:?} {t}/{role} {pattern:02x?} last={last}: {got:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn trailing_leading_and_inserted_garbage_are_typed_errors() {
    let orig = sample(FactorDtype::F16, deflate(1));
    for extra in [1usize, 3, 4, 35, 36, 37, 100, 4096] {
        let mut bytes = orig.clone();
        bytes.extend(std::iter::repeat_n(0xA5u8, extra));
        assert_format_error(&format!("{extra} bytes after the footer"), &bytes);

        // Same, with a footer recomputed over the garbage: the old footer becomes payload.
        let mut hashed = bytes.clone();
        refresh_footer(&mut hashed);
        assert_format_error(
            &format!("{extra} bytes after the footer, rehashed"),
            &hashed,
        );

        let mut inside = orig[..orig.len() - FOOT].to_vec();
        inside.extend(std::iter::repeat_n(0x5Au8, extra));
        refresh_footer(&mut inside);
        assert_format_error(
            &format!("{extra} unreferenced bytes before the footer"),
            &inside,
        );

        let mut prefixed = vec![0xA5u8; extra];
        prefixed.extend_from_slice(&orig);
        assert_format_error(&format!("{extra} bytes before the magic"), &prefixed);
    }
}

/// Rebuilds the file with the header text post-processed (offsets rebased on the final length).
fn forge_text(orig: &[u8], post: impl Fn(Vec<u8>) -> Vec<u8>) -> Vec<u8> {
    let hl = header_len(orig);
    let base: Value = serde_json::from_slice(&orig[PRE..PRE + hl]).unwrap();
    let blobs = &orig[PRE + hl..orig.len() - FOOT];
    let mut len = hl;
    for _ in 0..8 {
        let mut header = base.clone();
        relocate(&mut header, len as i64 - hl as i64);
        let json = post(serde_json::to_vec(&header).unwrap());
        if json.len() == len {
            return assemble(&json, blobs);
        }
        len = json.len();
    }
    panic!("forge_text: the header length does not converge");
}

fn replace_first(json: Vec<u8>, from: &str, to: &str) -> Vec<u8> {
    let text = String::from_utf8(json).unwrap();
    assert!(text.contains(from), "{from} not found in the header");
    text.replacen(from, to, 1).into_bytes()
}

#[test]
fn json_level_oddities_in_the_header_are_typed_errors() {
    let orig = sample(FactorDtype::F16, deflate(1));
    let dir = TempDir::new().unwrap();
    let reference = open_and_read(dir.path(), &orig).unwrap();
    let rank_forms = [
        "4.0",
        "4e0",
        "+4",
        "04",
        "\"4\"",
        "[4]",
        "-4",
        "0x4",
        "18446744073709551616",
        "1e3",
        "4,\"rank\":4",
        "4,\"rank\":5",
        "NaN",
        "Infinity",
    ];
    for form in rank_forms {
        let bytes = forge_text(&orig, |j| {
            replace_first(j, "\"rank\":4", &format!("\"rank\":{form}"))
        });
        assert_format_error(&format!("header rank written as {form}"), &bytes);
    }
    let bytes = forge_text(&orig, |j| {
        replace_first(
            j,
            "\"format_version\":1",
            "\"format_version\":1,\"format_version\":1",
        )
    });
    assert_format_error("duplicate format_version key", &bytes);
    let with_bom = forge_text(&orig, |j| [b"\xEF\xBB\xBF".as_slice(), &j].concat());
    assert_format_error("UTF-8 byte order mark before the header", &with_bom);
    let invalid_utf8 = forge_text(&orig, |j| replace_first(j, "\"adversary\"", "\"\u{1}d\""));
    let mut bad = invalid_utf8.clone();
    let at = bad.windows(3).position(|w| w == b"\"\x01d").unwrap();
    bad[at + 1] = 0xFF;
    let digest = Sha256::digest(&bad[..bad.len() - FOOT]);
    let len = bad.len();
    bad[len - FOOT..len - 4].copy_from_slice(&digest);
    assert_format_error("invalid UTF-8 inside a header string", &bad);

    // serde_json skips unknown fields iteratively: a field nested 100 000 deep must neither
    // overflow the stack nor change what the reader returns.
    let nested = |depth: usize| {
        forge_text(&orig, |mut j| {
            j.pop();
            j.extend_from_slice(
                format!(",\"zz\":{}{}}}", "[".repeat(depth), "]".repeat(depth)).as_bytes(),
            );
            j
        })
    };
    for depth in [20, 1000, 100_000] {
        assert_eq!(
            open_and_read(dir.path(), &nested(depth)).unwrap(),
            reference,
            "{depth}"
        );
    }
}

// ---- random structured fuzz

fn all_paths(v: &Value, prefix: &str, out: &mut Vec<String>) {
    out.push(prefix.to_owned());
    match v {
        Value::Object(map) => map
            .iter()
            .for_each(|(k, child)| all_paths(child, &format!("{prefix}/{k}"), out)),
        Value::Array(items) => items
            .iter()
            .enumerate()
            .for_each(|(i, child)| all_paths(child, &format!("{prefix}/{i}"), out)),
        _ => {}
    }
}

fn random_value(rng: &mut Rng) -> Value {
    let numbers = [
        0,
        1,
        2,
        3,
        4,
        255,
        4096,
        1 << 31,
        1 << 32,
        1 << 63,
        u64::MAX - 1,
        u64::MAX,
    ];
    match rng.below(9) {
        0 => json!(*rng.pick(&numbers)),
        1 => json!(rng.next_u64() >> rng.below(64)),
        2 => json!(-(rng.below(1000) as i64)),
        3 => json!(*rng.pick(&[
            "",
            "a",
            "F32",
            "BF16",
            "deflate",
            "none",
            "lowrank",
            "full_delta",
            "\u{0}"
        ])),
        4 => json!("x".repeat(rng.below(3000))),
        5 => Value::Null,
        6 => json!(rng.below(2) == 0),
        7 => json!([]),
        _ => json!({}),
    }
}

/// One random edit of the header value: replace a node, delete it, duplicate an array
/// element, or swap two nodes.
fn mutate_randomly(header: &mut Value, rng: &mut Rng) {
    let mut paths = Vec::new();
    all_paths(header, "", &mut paths);
    paths.retain(|p| !p.is_empty());
    let path = rng.pick(&paths).clone();
    let (parent, key) = path.rsplit_once('/').unwrap();
    match rng.below(4) {
        0 | 1 => {
            let value = random_value(rng);
            put(&path, value)(header, &mut Vec::new(), 0);
        }
        2 => {
            let slot = header.pointer_mut(parent).unwrap();
            match slot {
                Value::Object(map) => {
                    map.remove(key);
                }
                Value::Array(items) => {
                    let copy = items[key.parse::<usize>().unwrap()].clone();
                    items.push(copy);
                }
                _ => {}
            }
        }
        _ => {
            let other = rng.pick(&paths).clone();
            let (a, b) = (
                header.pointer(&path).cloned(),
                header.pointer(&other).cloned(),
            );
            if let (Some(a), Some(b)) = (a, b) {
                if !other.starts_with(&path) && !path.starts_with(&other) {
                    put(&path, b)(header, &mut Vec::new(), 0);
                    put(&other, a)(header, &mut Vec::new(), 0);
                }
            }
        }
    }
}

/// Reads whatever the reader accepts and checks that its output is coherent.
fn assert_ok_or_format(label: &str, bytes: &[u8], dir: &Path) {
    match open_and_read(dir, bytes) {
        Ok(tensors) => {
            for t in tensors {
                assert!(
                    t.u.iter().chain(&t.s).chain(&t.v).all(|x| x.is_finite()),
                    "{label}"
                );
            }
        }
        Err(SurgeryError::Format(_)) => {}
        Err(other) => panic!("{label}: untyped error {other}"),
    }
}

#[test]
fn seeded_random_header_edits_with_a_valid_hash_never_panic_or_balloon() {
    let started = Instant::now();
    let (_, growth) = peak_commit_growth(|| {
        let dir = TempDir::new().unwrap();
        for (dtype, compression) in [(FactorDtype::F16, deflate(1)), (FactorDtype::F32, stored())] {
            let orig = sample(dtype, compression);
            let mut rng = Rng::new(0xF00D);
            for i in 0..200 {
                let edits = 1 + rng.below(3);
                let bytes = forge(&orig, &|h, _, _| {
                    // The generator is re-seeded per call: `forge` may call this several times
                    // and every call must make the same edits.
                    let mut local = Rng::new(0xF00D + i);
                    (0..edits).for_each(|_| mutate_randomly(h, &mut local));
                });
                assert_ok_or_format(&format!("edit {i} {dtype:?}"), &bytes, dir.path());
                rng.next_u64();
            }
        }
    });
    assert!(growth < MEM_LIMIT, "commit grew by {} MiB", growth >> 20);
    assert!(started.elapsed() < Duration::from_secs(60));
}

#[test]
fn seeded_random_byte_edits_in_the_header_with_a_valid_hash_never_panic() {
    let dir = TempDir::new().unwrap();
    let orig = sample(FactorDtype::F16, deflate(1));
    let hl = header_len(&orig);
    let mut rng = Rng::new(0xBEEF);
    let alphabet = b"0123456789{}[]\",:-.eEntulf \x00\xFF";
    for i in 0..600 {
        let mut bytes = orig.clone();
        for _ in 0..1 + rng.below(3) {
            let at = PRE + rng.below(hl);
            bytes[at] = match rng.below(3) {
                0 => *rng.pick(alphabet),
                1 => rng.next_u64() as u8,
                _ => bytes[at].wrapping_add(1),
            };
        }
        bytes.truncate(bytes.len() - FOOT);
        refresh_footer(&mut bytes);
        assert_ok_or_format(&format!("byte edit {i}"), &bytes, dir.path());
    }
}

#[test]
fn random_valid_files_round_trip_to_the_quantized_input_and_are_deterministic() {
    let dir = TempDir::new().unwrap();
    let quantize = |dtype: FactorDtype, x: f32| match dtype {
        FactorDtype::F16 => f16::from_f32(x).to_f32(),
        FactorDtype::Bf16 => bf16::from_f32(x).to_f32(),
        FactorDtype::F32 => x,
    };
    let mut rng = Rng::new(99);
    for case in 0..40 {
        let dtype = *rng.pick(&[FactorDtype::F16, FactorDtype::Bf16, FactorDtype::F32]);
        let compression = match rng.below(4) {
            0 => stored(),
            _ => deflate(rng.below(11) as u8),
        };
        let header_rank = rng.between(1, 6);
        let mut spec = init(dtype, compression);
        spec.rank = header_rank;
        let tensors = rng.below(4);
        let mut inputs = Vec::new();
        for t in 0..tensors {
            let (m, n) = (rng.between(1, 20), rng.between(1, 20));
            // The rank may exceed min(m, n): the format allows it.
            let r = rng.between(1, header_rank);
            let f = LowRankFactors {
                u: (0..m * r).map(|_| rng.signed() as f32 * 3.0).collect(),
                s: (0..r).map(|_| rng.unit() as f32 * 5.0).collect(),
                v: (0..r * n).map(|_| rng.signed() as f32 * 3.0).collect(),
            };
            inputs.push((format!("t{t}.weight"), m, n, f));
        }
        let write = |name: &str| {
            let path = dir.path().join(name);
            let mut w = SurgeryWriter::create(&path, spec.clone(), true).unwrap();
            for (name, m, n, f) in &inputs {
                w.add_lowrank(meta(name, &[*m, *n]), f).unwrap();
            }
            w.finish().unwrap();
            path
        };
        let (a, b) = (write("a.surgery"), write("b.surgery"));
        assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap(), "case {case}");
        let mut reader = SurgeryReader::open(&a).unwrap();
        assert_eq!(reader.header().tensors.len(), inputs.len());
        for (entry, (_, _, _, f)) in reader.header().tensors.clone().iter().zip(&inputs) {
            let got = reader.read_lowrank(entry).unwrap();
            let want = |v: &[f32]| v.iter().map(|&x| quantize(dtype, x)).collect::<Vec<_>>();
            assert_eq!(got.u, want(&f.u), "case {case}");
            assert_eq!(got.s, want(&f.s), "case {case}");
            assert_eq!(got.v, want(&f.v), "case {case}");
        }
    }
}

// ---- writer

fn lone_writer(dir: &Path, dtype: FactorDtype) -> SurgeryWriter {
    SurgeryWriter::create(&dir.join("o.surgery"), init(dtype, deflate(1)), false).unwrap()
}

#[test]
fn writer_refuses_hostile_tensors_and_keeps_working_afterwards() {
    let dir = TempDir::new().unwrap();
    let mut rng = Rng::new(3);
    let good = factors(&mut rng, 4, 3, 2);
    let hostile: Vec<(&str, TensorMeta, LowRankFactors)> = {
        let named = |name: &str| meta(name, &[4, 3]);
        let tweak = |f: fn(&mut TensorMeta)| {
            let mut m = named("t.weight");
            f(&mut m);
            m
        };
        let nan_in_v = LowRankFactors {
            v: [vec![f32::NAN], good.v[1..].to_vec()].concat(),
            ..good.clone()
        };
        let inf_in_u = LowRankFactors {
            u: [good.u[..7].to_vec(), vec![f32::INFINITY]].concat(),
            ..good.clone()
        };
        let big = LowRankFactors {
            s: vec![1e30, 1.0],
            ..good.clone()
        };
        vec![
            ("NUL name", named("a\u{0}b"), good.clone()),
            ("empty name", named(""), good.clone()),
            ("1025 byte name", named(&"n".repeat(1025)), good.clone()),
            ("zero rows", meta("t.weight", &[0, 3]), good.clone()),
            ("zero columns", meta("t.weight", &[4, 0]), good.clone()),
            (
                "shape product overflow",
                meta("t.weight", &[usize::MAX, usize::MAX]),
                good.clone(),
            ),
            ("1D lowrank", meta("t.weight", &[12]), good.clone()),
            (
                "rank 0 factors",
                named("t.weight"),
                LowRankFactors::default(),
            ),
            (
                "rank above header rank",
                named("t.weight"),
                factors(&mut Rng::new(1), 4, 3, 5),
            ),
            (
                "NaN rel_error",
                tweak(|m| m.rel_error = f64::NAN),
                good.clone(),
            ),
            (
                "infinite energy",
                tweak(|m| m.energy = f64::INFINITY),
                good.clone(),
            ),
            (
                "negative rel_error",
                tweak(|m| m.rel_error = -0.1),
                good.clone(),
            ),
            (
                "uppercase base hash",
                tweak(|m| m.base_sha256 = "AB".repeat(32)),
                good.clone(),
            ),
            (
                "target dtype F64",
                tweak(|m| m.target_dtype = "F64".into()),
                good.clone(),
            ),
            ("NaN factor late in V", named("t.weight"), nan_in_v),
            (
                "infinite factor at the end of U",
                named("t.weight"),
                inf_in_u,
            ),
            ("f16 overflow in S", named("t.weight"), big),
            (
                "U of the wrong size",
                named("t.weight"),
                LowRankFactors {
                    u: good.u[1..].to_vec(),
                    ..good.clone()
                },
            ),
        ]
    };
    let mut w = lone_writer(dir.path(), FactorDtype::F16);
    for (label, m, f) in &hostile {
        assert!(w.add_lowrank(m.clone(), f).is_err(), "{label}");
    }
    assert!(
        w.add_full_delta(meta("d.bias", &[0]), &[]).is_err(),
        "empty delta"
    );
    assert!(w
        .add_full_delta(meta("d.bias", &[3]), &[1.0, f32::NAN, 2.0])
        .is_err());
    assert!(w.add_full_delta(meta("d.bias", &[3]), &[1.0, 2.0]).is_err());
    // A failed add must not leave anything behind: the next tensor and the file are clean.
    w.add_lowrank(meta("ok.weight", &[4, 3]), &good).unwrap();
    assert!(
        w.add_lowrank(meta("ok.weight", &[4, 3]), &good).is_err(),
        "duplicate"
    );
    w.add_full_delta(meta("d.bias", &[3]), &[1.0, 2.0, 3.0])
        .unwrap();
    w.finish().unwrap();
    let mut reader = SurgeryReader::open(&dir.path().join("o.surgery")).unwrap();
    let names: Vec<String> = reader
        .header()
        .tensors
        .iter()
        .map(|t| t.name.clone())
        .collect();
    assert_eq!(names, ["ok.weight", "d.bias"]);
    let entry = reader.header().tensors[0].clone();
    assert_eq!(reader.read_lowrank(&entry).unwrap().u.len(), 8);
    assert_eq!(
        file_names(dir.path()),
        ["o.surgery"],
        "no spool or temporary file"
    );
}

#[test]
fn writer_with_no_tensor_and_dropped_writers_leave_no_files() {
    let dir = TempDir::new().unwrap();
    let w = lone_writer(dir.path(), FactorDtype::Bf16);
    w.finish().unwrap();
    let reader = SurgeryReader::open(&dir.path().join("o.surgery")).unwrap();
    assert!(reader.header().tensors.is_empty());
    assert!(
        open_and_read(dir.path(), &fs::read(dir.path().join("o.surgery")).unwrap())
            .unwrap()
            .is_empty()
    );
    let other = TempDir::new().unwrap();
    let mut dropped = lone_writer(other.path(), FactorDtype::F16);
    dropped
        .add_full_delta(meta("d.bias", &[2]), &[1.0, 2.0])
        .unwrap();
    drop(dropped);
    assert!(
        file_names(other.path()).is_empty(),
        "dropped writer left files"
    );
    let mut bad_init = init(FactorDtype::F16, deflate(1));
    bad_init.rank = 0;
    assert!(SurgeryWriter::create(&other.path().join("z.surgery"), bad_init, false).is_err());
    assert!(SurgeryWriter::create(Path::new(""), init(FactorDtype::F16, stored()), false).is_err());
    assert!(file_names(other.path()).is_empty());
}

// ------------------------------------------------------------------ 2. rSVD

type Hook = Box<dyn Fn(usize) -> Option<SurgeryResult<usize>> + Send + Sync>;

/// In-memory matrix served at most `chunk` rows per call, with an optional hook that sees the
/// 0-based call index and may replace the answer (errors, short or long reads, side effects).
struct MemSource {
    m: usize,
    n: usize,
    data: Vec<f32>,
    chunk: usize,
    calls: AtomicUsize,
    hook: Option<Hook>,
}

impl MemSource {
    fn new(m: usize, n: usize, data: Vec<f32>) -> Self {
        assert_eq!(m * n, data.len());
        Self {
            m,
            n,
            data,
            chunk: usize::MAX,
            calls: AtomicUsize::new(0),
            hook: None,
        }
    }

    fn chunked(mut self, chunk: usize) -> Self {
        self.chunk = chunk;
        self
    }

    fn hooked(mut self, hook: Hook) -> Self {
        self.hook = Some(hook);
        self
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl RowSource for MemSource {
    fn rows(&self) -> usize {
        self.m
    }

    fn cols(&self) -> usize {
        self.n
    }

    fn read_rows(&self, start: usize, out: &mut [f32]) -> SurgeryResult<usize> {
        let call = self.calls.fetch_add(1, Ordering::SeqCst);
        if let Some(answer) = self.hook.as_ref().and_then(|h| h(call)) {
            return answer;
        }
        let rows = (out.len() / self.n).min(self.m - start).min(self.chunk);
        out[..rows * self.n].copy_from_slice(&self.data[start * self.n..(start + rows) * self.n]);
        Ok(rows)
    }
}

fn run_rsvd(a: &dyn RowSource, rank: usize) -> Result<Rsvd, SurgeryError> {
    let p = RsvdParams {
        rank,
        ..RsvdParams::default()
    };
    rsvd(a, &p, &AtomicBool::new(false))
}

/// Orthonormal columns of a random `rows x cols` Gaussian matrix (row-major, f64), by modified
/// Gram-Schmidt applied twice.
fn orthonormal(rng: &mut Rng, rows: usize, cols: usize) -> Vec<f64> {
    let mut q: Vec<f64> = (0..rows * cols).map(|_| rng.gauss()).collect();
    for j in 0..cols {
        for _ in 0..2 {
            for k in 0..j {
                let dot: f64 = (0..rows).map(|i| q[i * cols + j] * q[i * cols + k]).sum();
                (0..rows).for_each(|i| q[i * cols + j] -= dot * q[i * cols + k]);
            }
        }
        let norm = (0..rows)
            .map(|i| q[i * cols + j].powi(2))
            .sum::<f64>()
            .sqrt();
        (0..rows).for_each(|i| q[i * cols + j] /= norm);
    }
    q
}

/// `U diag(sigma) V^T` with random orthonormal `U`, `V` and a geometric spectrum
/// `scale * decay^i`: the singular values are known exactly.
fn decaying(rng: &mut Rng, m: usize, n: usize, decay: f64, scale: f64) -> (Vec<f32>, Vec<f64>) {
    let k = m.min(n);
    let (u, v) = (orthonormal(rng, m, k), orthonormal(rng, n, k));
    let sigma: Vec<f64> = (0..k).map(|i| scale * decay.powi(i as i32)).collect();
    let a = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            (0..k)
                .map(|t| u[i * k + t] * sigma[t] * v[j * k + t])
                .sum::<f64>() as f32
        })
        .collect();
    (a, sigma)
}

fn random_matrix(rng: &mut Rng, m: usize, n: usize, scale: f64) -> Vec<f32> {
    (0..m * n).map(|_| (rng.gauss() * scale) as f32).collect()
}

fn frobenius(a: &[f32]) -> f64 {
    a.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt()
}

/// Largest deviation of `X^T X` (columns, `X` is `rows x cols`) from the identity restricted to
/// the `live` columns.
fn column_gram_deviation(x: &[f32], rows: usize, cols: usize, live: &[bool]) -> f64 {
    let mut worst = 0.0f64;
    for a in (0..cols).filter(|&a| live[a]) {
        for b in (0..cols).filter(|&b| live[b]) {
            let dot: f64 = (0..rows)
                .map(|i| f64::from(x[i * cols + a]) * f64::from(x[i * cols + b]))
                .sum();
            worst = worst.max((dot - f64::from(u8::from(a == b))).abs());
        }
    }
    worst
}

fn row_gram_deviation(x: &[f32], rows: usize, cols: usize, live: &[bool]) -> f64 {
    let mut worst = 0.0f64;
    for a in (0..rows).filter(|&a| live[a]) {
        for b in (0..rows).filter(|&b| live[b]) {
            let dot: f64 = (0..cols)
                .map(|j| f64::from(x[a * cols + j]) * f64::from(x[b * cols + j]))
                .sum();
            worst = worst.max((dot - f64::from(u8::from(a == b))).abs());
        }
    }
    worst
}

/// Invariants every successful result must satisfy whatever the input.
fn assert_sane(label: &str, out: &Rsvd, (m, n): (usize, usize)) {
    assert_eq!((out.m, out.n), (m, n), "{label}");
    assert_eq!(out.u.len(), m * out.r, "{label}");
    assert_eq!(out.s.len(), out.r, "{label}");
    assert_eq!(out.v.len(), out.r * n, "{label}");
    assert!(
        out.u
            .iter()
            .chain(&out.s)
            .chain(&out.v)
            .all(|x| x.is_finite()),
        "{label}: non finite factor"
    );
    assert!(
        out.rel_error.is_finite() && out.rel_error >= 0.0,
        "{label}: rel_error {}",
        out.rel_error
    );
    assert!(
        out.energy.is_finite() && (0.0..=1.0).contains(&out.energy),
        "{label}: energy {}",
        out.energy
    );
    assert!(out.s.iter().all(|&s| s >= 0.0), "{label}: negative sigma");
    assert!(
        out.s.windows(2).all(|w| w[0] >= w[1]),
        "{label}: sigma not sorted"
    );
    let s0 = out.s.first().copied().unwrap_or(0.0);
    let live: Vec<bool> = out.s.iter().map(|&s| s > 1e-4 * s0 && s > 0.0).collect();
    // The sign convention: the largest component of each live V row is positive.
    for (t, row) in out.v.chunks_exact(n).enumerate().filter(|(t, _)| live[*t]) {
        let big = row.iter().copied().fold(
            0.0f32,
            |best, x| if x.abs() > best.abs() { x } else { best },
        );
        assert!(
            big > 0.0,
            "{label}: V row {t} has a negative dominant entry"
        );
    }
}

#[test]
fn rsvd_is_within_1_5x_eckart_young_and_orthonormal_on_decaying_spectra() {
    let mut rng = Rng::new(2024);
    for case in 0..24 {
        let (m, n) = (rng.between(4, 70), rng.between(4, 70));
        let (decay, rank) = (0.4 + 0.35 * rng.unit(), rng.between(1, 10));
        let scale = 0.5 + 4.0 * rng.unit();
        let (a, sigma) = decaying(&mut rng, m, n, decay, scale);
        let total = frobenius(&a);
        let src = MemSource::new(m, n, a);
        let effective = rank.min(m).min(n);
        let tail = sigma[effective..].iter().map(|s| s * s).sum::<f64>().sqrt() / total;
        for seed in 0..3 {
            let p = RsvdParams {
                rank,
                seed,
                ..RsvdParams::default()
            };
            let out = rsvd(&src, &p, &AtomicBool::new(false)).unwrap();
            let label = format!("case {case} {m}x{n} r{rank} decay {decay:.2} seed {seed}");
            assert_sane(&label, &out, (m, n));
            assert_eq!(out.r, effective, "{label}");
            assert!(
                out.rel_error <= 1.5 * tail + 1e-5,
                "{label}: rel_error {} vs Eckart-Young {tail}",
                out.rel_error
            );
            // The error cannot beat the optimum (up to f32 noise).
            assert!(out.rel_error + 1e-5 >= tail, "{label}: below Eckart-Young");
            let live = vec![true; out.r];
            assert!(
                column_gram_deviation(&out.u, m, out.r, &live) < 2e-3,
                "{label}: U"
            );
            assert!(
                row_gram_deviation(&out.v, out.r, n, &live) < 2e-3,
                "{label}: V"
            );
            // Energy and error describe the same projection.
            assert!(
                (out.energy - (1.0 - out.rel_error * out.rel_error)).abs() < 1e-2,
                "{label}: energy {} vs 1 - rel^2 {}",
                out.energy,
                1.0 - out.rel_error * out.rel_error
            );
            for (t, &want) in sigma.iter().enumerate().take(out.r.min(3)) {
                assert!(
                    (f64::from(out.s[t]) - want).abs() <= 0.03 * want,
                    "{label}: sigma[{t}] {} vs {want}",
                    out.s[t]
                );
            }
        }
    }
}

#[test]
fn rsvd_degenerate_inputs_give_sane_results() {
    let mut rng = Rng::new(5);
    let ones = |m: usize, n: usize, x: f32| vec![x; m * n];
    let row = random_matrix(&mut rng, 1, 20, 1.0);
    let col = random_matrix(&mut rng, 20, 1, 1.0);
    let mut dup = Vec::new();
    let template = random_matrix(&mut rng, 1, 8, 1.0);
    (0..30).for_each(|_| dup.extend_from_slice(&template));
    let x = random_matrix(&mut rng, 25, 2, 1.0);
    let y = random_matrix(&mut rng, 2, 18, 1.0);
    let rank2: Vec<f32> = (0..25 * 18)
        .map(|idx| {
            (0..2)
                .map(|t| x[idx / 18 * 2 + t] * y[t * 18 + idx % 18])
                .sum()
        })
        .collect();
    let mut zero_rows = random_matrix(&mut rng, 12, 9, 1.0);
    zero_rows[..4 * 9].fill(0.0);
    zero_rows[8 * 9..].fill(0.0);
    type Degenerate<'a> = (&'a str, usize, usize, Vec<f32>, usize, f64);
    let cases: Vec<Degenerate> = vec![
        // (label, m, n, data, rank, maximum allowed rel_error)
        ("1x1", 1, 1, vec![5.0], 3, 1e-6),
        ("single row", 1, 20, row, 16, 1e-6),
        ("single column", 20, 1, col, 16, 1e-6),
        ("constant", 10, 6, ones(10, 6, 3.0), 4, 1e-6),
        ("negative constant", 7, 9, ones(7, 9, -0.25), 2, 1e-6),
        ("duplicate rows", 30, 8, dup, 5, 1e-6),
        ("rank 2 asked for rank 6", 25, 18, rank2, 6, 1e-5),
        ("zero rows inside", 12, 9, zero_rows, 9, 1e-5),
        (
            "rank above both dimensions",
            3,
            50,
            random_matrix(&mut rng, 3, 50, 1.0),
            16,
            1e-5,
        ),
        (
            "wide",
            5,
            400,
            random_matrix(&mut rng, 5, 400, 1.0),
            5,
            1e-5,
        ),
        (
            "tall",
            400,
            5,
            random_matrix(&mut rng, 400, 5, 1.0),
            5,
            1e-5,
        ),
    ];
    for (label, m, n, data, rank, limit) in cases {
        let out = run_rsvd(&MemSource::new(m, n, data.clone()), rank).unwrap();
        assert_sane(label, &out, (m, n));
        assert_eq!(out.r, rank.min(m).min(n), "{label}");
        assert!(
            out.rel_error <= limit,
            "{label}: rel_error {}",
            out.rel_error
        );
        // The dead components must stay orthonormal in U even though their V row is null.
        let live = vec![true; out.r];
        assert!(
            column_gram_deviation(&out.u, m, out.r, &live) < 2e-3,
            "{label}: U"
        );
        let alive: Vec<bool> = out.s.iter().map(|&s| s > 1e-4 * out.s[0]).collect();
        assert!(
            row_gram_deviation(&out.v, out.r, n, &alive) < 2e-3,
            "{label}: V"
        );
        assert!(frobenius(&data) > 0.0);
    }
    // Null matrix: zero factors, zero error, zero energy, never NaN.
    for (m, n) in [(1, 1), (5, 7), (70, 3)] {
        let out = run_rsvd(&MemSource::new(m, n, vec![0.0; m * n]), 4).unwrap();
        assert_sane("zero", &out, (m, n));
        assert!(out.u.iter().chain(&out.s).chain(&out.v).all(|&x| x == 0.0));
        assert_eq!((out.rel_error, out.energy), (0.0, 0.0));
    }
    // A single non-zero entry.
    let mut one = vec![0.0f32; 6 * 6];
    one[2 * 6 + 3] = -7.0;
    let out = run_rsvd(&MemSource::new(6, 6, one), 2).unwrap();
    assert_sane("single entry", &out, (6, 6));
    assert!(out.rel_error < 1e-6 && (out.s[0] - 7.0).abs() < 1e-4);
}

#[test]
fn rsvd_refuses_unusable_arguments_with_typed_errors() {
    let src = |m: usize, n: usize| MemSource::new(m, n, vec![1.0; m * n]);
    {
        assert!(matches!(
            run_rsvd(&src(4, 4), 0),
            Err(SurgeryError::Usage(_))
        ));
    }
    for (m, n) in [(0, 5), (5, 0), (0, 0)] {
        assert!(
            matches!(run_rsvd(&src(m, n), 2), Err(SurgeryError::Usage(_))),
            "{m}x{n}"
        );
    }
    // Parameters at their extremes are fine.
    for (oversample, power_iters, rank) in [(0, 0, 1), (0, 5, 3), (64, 0, 2), (3, 1, usize::MAX)] {
        let mut rng = Rng::new(1);
        let (a, _) = decaying(&mut rng, 20, 15, 0.5, 1.0);
        let p = RsvdParams {
            rank,
            oversample,
            power_iters,
            seed: u64::MAX,
        };
        let out = rsvd(&MemSource::new(20, 15, a), &p, &AtomicBool::new(false)).unwrap();
        assert_sane(
            &format!("o{oversample} q{power_iters} r{rank}"),
            &out,
            (20, 15),
        );
    }
}

#[test]
fn rsvd_survives_extreme_magnitudes() {
    let mut rng = Rng::new(11);
    for scale in [1e30f64, 1e-30, 1e20, 1e-20] {
        let (m, n) = (30, 22);
        let (a, sigma) = decaying(&mut rng, m, n, 0.5, scale);
        let tail = sigma[5..].iter().map(|s| s * s).sum::<f64>().sqrt() / frobenius(&a);
        let out = run_rsvd(&MemSource::new(m, n, a), 5).unwrap();
        let label = format!("scale {scale:e}");
        assert_sane(&label, &out, (m, n));
        assert!(
            out.rel_error <= 1.5 * tail + 1e-5,
            "{label}: rel_error {}",
            out.rel_error
        );
        let want = sigma[0] as f32;
        assert!(
            (out.s[0] - want).abs() <= 0.03 * want,
            "{label}: sigma0 {}",
            out.s[0]
        );
    }
    // Pure noise at 1e30: no structure to find, but still finite and in range.
    let noisy = random_matrix(&mut rng, 40, 25, 1e30);
    let out = run_rsvd(&MemSource::new(40, 25, noisy), 6).unwrap();
    assert_sane("noise 1e30", &out, (40, 25));
    // Subnormal range: precision is gone, finiteness and the ranges are not.
    for scale in [1e-37f64, 1e-42] {
        let noisy = random_matrix(&mut rng, 20, 12, scale);
        let out = run_rsvd(&MemSource::new(20, 12, noisy), 4).unwrap();
        assert_sane(&format!("subnormal {scale:e}"), &out, (20, 12));
    }
}

#[test]
#[ignore = "BUG: rsvd returns Ok with an infinite singular value (rsvd.rs:188 sigma as f32) when sigma exceeds f32::MAX"]
fn rsvd_entries_near_f32_max_give_a_finite_result_or_a_typed_error() {
    // sigma = 3e38 * sqrt(m n) does not fit an f32: the factors cannot be delivered, but the
    // answer must not be Ok with an infinite singular value or a NaN error.
    for (m, n, x) in [(2usize, 2usize, 3.0e38f32), (4, 4, 1.0e38), (3, 5, -3.3e38)] {
        match run_rsvd(&MemSource::new(m, n, vec![x; m * n]), 2) {
            Ok(out) => {
                eprintln!(
                    "{m}x{n} {x:e}: s={:?} rel_error={} energy={}",
                    out.s, out.rel_error, out.energy
                );
                assert_sane(&format!("{m}x{n} {x:e}"), &out, (m, n));
            }
            Err(SurgeryError::Failed(_) | SurgeryError::Usage(_)) => {}
            Err(other) => panic!("untyped error {other}"),
        }
    }
}

#[test]
fn rsvd_result_is_bit_identical_with_1_2_and_8_threads_and_any_read_chunking() {
    let mut rng = Rng::new(31);
    // 64-row blocks, 16 blocks per wave: sizes around the boundaries of both.
    for m in [63usize, 64, 65, 1023, 1024, 1025, 2100] {
        let n = 13;
        let data = random_matrix(&mut rng, m, n, 1.0);
        let run = |threads: usize, chunk: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    run_rsvd(&MemSource::new(m, n, data.clone()).chunked(chunk), 4).unwrap()
                })
        };
        let reference = run(1, usize::MAX);
        assert_sane(&format!("m={m}"), &reference, (m, n));
        for (threads, chunk) in [(2, usize::MAX), (8, usize::MAX), (8, 1), (3, 7), (8, 63)] {
            assert_eq!(
                run(threads, chunk),
                reference,
                "m={m} threads={threads} chunk={chunk}"
            );
        }
    }
}

#[test]
fn rsvd_rejects_non_finite_input_wherever_it_hides() {
    let mut rng = Rng::new(8);
    let (m, n) = (300, 7);
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for at in [0, 1, n - 1, 150 * n + 3, m * n - 1] {
            let mut data = random_matrix(&mut rng, m, n, 1.0);
            data[at] = bad;
            let got = run_rsvd(&MemSource::new(m, n, data), 3);
            assert!(
                matches!(got, Err(SurgeryError::Failed(_))),
                "{bad} at {at}: {got:?}"
            );
        }
    }
}

#[test]
fn rsvd_row_sources_that_misbehave_give_errors_not_panics_or_garbage() {
    let mut rng = Rng::new(17);
    let (m, n) = (200, 9);
    let data = random_matrix(&mut rng, m, n, 1.0);
    let make = || MemSource::new(m, n, data.clone());
    let total = {
        let counting = make();
        run_rsvd(&counting, 3).unwrap();
        counting.calls()
    };
    // An error at any call propagates unchanged.
    for fail_at in (0..total).step_by(3) {
        let src = make().hooked(Box::new(move |call| {
            (call == fail_at).then(|| Err(SurgeryError::Failed("boom".into())))
        }));
        let got = run_rsvd(&src, 3);
        assert!(
            matches!(&got, Err(SurgeryError::Failed(msg)) if msg == "boom"),
            "call {fail_at}: {got:?}"
        );
    }
    // Zero rows, or more rows than asked: refused (a zero would loop forever).
    for answer in [0usize, 10_000] {
        let src = make().hooked(Box::new(move |call| (call == 2).then_some(Ok(answer))));
        assert!(
            matches!(run_rsvd(&src, 3), Err(SurgeryError::Failed(_))),
            "answer {answer}"
        );
    }
    // One row per call everywhere: same numbers as the full reads.
    let slow = run_rsvd(&make().chunked(1), 3).unwrap();
    assert_eq!(slow, run_rsvd(&make(), 3).unwrap());
}

#[test]
fn rsvd_cancellation_at_every_read_is_clean() {
    let mut rng = Rng::new(23);
    // 200 rows = 4 blocks = one wave: 9 passes of 4 reads (mul, 3 x (A^T Q, A Z), A^T Q, residual).
    let (m, n) = (200, 9);
    let data = random_matrix(&mut rng, m, n, 1.0);
    let counting = MemSource::new(m, n, data.clone());
    run_rsvd(&counting, 3).unwrap();
    let total = counting.calls();
    assert_eq!(total, 36);
    for at in 0..total {
        let flag = Arc::new(AtomicBool::new(false));
        let raised = flag.clone();
        let src = MemSource::new(m, n, data.clone()).hooked(Box::new(move |call| {
            if call == at {
                raised.store(true, Ordering::SeqCst);
            }
            None
        }));
        let got = rsvd(
            &src,
            &RsvdParams {
                rank: 3,
                ..Default::default()
            },
            &flag,
        );
        match got {
            Err(SurgeryError::Cancelled) => {}
            // Raised during the last pass: every block may already be past its check.
            Ok(out) if at >= total - 4 => assert_sane("late cancel", &out, (m, n)),
            other => panic!("cancel at read {at}: {other:?}"),
        }
    }
    let flag = AtomicBool::new(true);
    let got = rsvd(&MemSource::new(m, n, data), &RsvdParams::default(), &flag);
    assert!(matches!(got, Err(SurgeryError::Cancelled)));
}

// ------------------------------------------------------------------ 3. LoRA

/// One tensor of a SafeTensors fixture.
struct St {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn encode_vals(dtype: &str, vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&x| match dtype {
            "BF16" => bf16::from_f32(x).to_le_bytes().to_vec(),
            "F16" => f16::from_f32(x).to_le_bytes().to_vec(),
            _ => x.to_le_bytes().to_vec(),
        })
        .collect()
}

fn st(name: &str, dtype: &'static str, shape: &[usize], vals: &[f32]) -> St {
    St {
        name: name.to_owned(),
        dtype,
        shape: shape.to_vec(),
        bytes: encode_vals(dtype, vals),
    }
}

fn write_st(path: &Path, tensors: &[St]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for t in tensors {
        let start = data.len();
        data.extend_from_slice(&t.bytes);
        header.insert(
            t.name.clone(),
            json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = serde_json::to_string(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    fs::write(path, file).unwrap();
}

fn gaussians(rng: &mut Rng, n: usize, scale: f64) -> Vec<f32> {
    (0..n).map(|_| (rng.gauss() * scale) as f32).collect()
}

const QP: &str = "model.layers.0.self_attn.q_proj";
const VP: &str = "model.layers.0.self_attn.v_proj";

/// PEFT pair of the module `path` (target `[m, n]`) of rank `r`.
fn lora_pair(path: &str, (m, n, r): (usize, usize, usize), rng: &mut Rng) -> [St; 2] {
    [
        st(
            &format!("base_model.model.{path}.lora_A.weight"),
            "F32",
            &[r, n],
            &gaussians(rng, r * n, 0.5),
        ),
        st(
            &format!("base_model.model.{path}.lora_B.weight"),
            "F32",
            &[m, r],
            &gaussians(rng, m * r, 0.5),
        ),
    ]
}

struct Fx {
    dir: TempDir,
    model: PathBuf,
    lora: PathBuf,
}

/// `model.safetensors` with q_proj [6,5] and v_proj [4,5] (F32), the adapter, and the config text.
fn fixture(adapter: &[St], config: Option<&str>) -> Fx {
    let dir = TempDir::new().unwrap();
    let mut rng = Rng::new(77);
    let model = dir.path().join("model.safetensors");
    write_st(
        &model,
        &[
            st(
                &format!("{QP}.weight"),
                "F32",
                &[6, 5],
                &gaussians(&mut rng, 30, 1.0),
            ),
            st(
                &format!("{VP}.weight"),
                "F32",
                &[4, 5],
                &gaussians(&mut rng, 20, 1.0),
            ),
            st(
                "model.norm.weight",
                "F32",
                &[5],
                &gaussians(&mut rng, 5, 1.0),
            ),
        ],
    );
    let lora = dir.path().join("adapter_model.safetensors");
    write_st(&lora, adapter);
    if let Some(config) = config {
        fs::write(dir.path().join("adapter_config.json"), config).unwrap();
    }
    Fx { dir, model, lora }
}

fn std_config() -> String {
    r#"{"peft_type": "LORA", "r": 2, "lora_alpha": 4}"#.to_owned()
}

fn good_adapter(rng: &mut Rng) -> Vec<St> {
    let [qa, qb] = lora_pair(QP, (6, 5, 2), rng);
    let [va, vb] = lora_pair(VP, (4, 5, 2), rng);
    vec![qa, qb, va, vb]
}

fn merge_request(fx: &Fx, alpha: Option<f64>, scale: f64, dry_run: bool) -> MergeRequest {
    MergeRequest {
        model_shards: resolve_shards(&fx.model).unwrap(),
        lora: fx.lora.clone(),
        alpha,
        adapter_config: None,
        scale,
        dry_run,
        cancel: Arc::new(AtomicBool::new(false)),
    }
}

/// Runs a merge under a real backup set and returns the outcome plus the model bytes before and
/// after (an error must leave them equal).
fn merge_outcome(
    fx: &Fx,
    alpha: Option<f64>,
    scale: f64,
    dry_run: bool,
) -> (Result<MergeReport, SurgeryError>, Vec<u8>, Vec<u8>) {
    let before = fs::read(&fx.model).unwrap();
    let req = merge_request(fx, alpha, scale, dry_run);
    let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
    let result = run_merge(&req, &mut backup, &mut NullObserver);
    (result, before, fs::read(&fx.model).unwrap())
}

fn assert_usage_and_untouched(
    label: &str,
    outcome: (Result<MergeReport, SurgeryError>, Vec<u8>, Vec<u8>),
) -> String {
    let (result, before, after) = outcome;
    assert_eq!(before, after, "{label}: model modified by a failed merge");
    match result {
        Err(SurgeryError::Usage(msg)) => msg,
        other => panic!("{label}: expected a usage error, got {other:?}"),
    }
}

const KEY_TOKENS: [&str; 22] = [
    "base_model.model.",
    "model.layers.0.",
    "q_proj",
    ".",
    "..",
    "lora_A",
    "lora_B",
    ".lora_A.",
    ".lora_B.",
    "weight",
    ".weight",
    "default",
    ".default",
    "lora_embedding_A",
    "lora_magnitude_vector",
    "\u{0}",
    "\u{e9}",
    "\u{65e5}\u{672c}",
    "\u{301}",
    " ",
    "/",
    "",
];

fn random_lora_key(rng: &mut Rng) -> String {
    if rng.below(2) == 0 {
        // Close to a real key, so that the valid shapes are hit often.
        let path: String = (0..1 + rng.below(4))
            .map(|_| *rng.pick(&KEY_TOKENS))
            .collect();
        let side = rng.pick(&["lora_A", "lora_B"]);
        let infix = rng.pick(&["", ".default", ".x", ".a.b"]);
        let prefix = rng.pick(&["", "base_model.model."]);
        return format!("{prefix}{path}.{side}{infix}.weight");
    }
    (0..rng.below(9)).map(|_| *rng.pick(&KEY_TOKENS)).collect()
}

#[test]
fn parse_key_never_panics_and_round_trips_through_adapter_key() {
    let mut rng = Rng::new(404);
    let mut parsed = 0;
    for _ in 0..4000 {
        let key = random_lora_key(&mut rng);
        let _ = (is_embedding_key(&key), is_dora_key(&key));
        let Some((target, side)) = parse_key(&key) else {
            continue;
        };
        parsed += 1;
        assert!(
            target.ends_with(".weight") && target.len() > ".weight".len(),
            "{key:?} -> {target:?}"
        );
        // A target that itself contains ".lora_" (never a real tensor name) can be re-read as the
        // other side: "p.lora_A.lora_B.weight" is the B of "p.lora_A.weight" or the A of
        // "p.weight" depending on the order tried. Not a defect for real checkpoints.
        if target.contains(".lora_") {
            continue;
        }
        let rebuilt = adapter_key(&target, side).unwrap();
        assert_eq!(
            parse_key(&rebuilt),
            Some((target.clone(), side)),
            "{key:?} -> {rebuilt:?}"
        );
    }
    assert!(
        parsed > 500,
        "the generator should hit valid keys often enough ({parsed})"
    );
    // Targets without the suffix, and absurd lengths.
    assert_eq!(adapter_key("q_proj.bias", Side::A), None);
    assert_eq!(adapter_key("", Side::B), None);
    let long = format!("{}.lora_A.weight", "x".repeat(1 << 20));
    assert_eq!(parse_key(&long).map(|(t, _)| t.len()), Some((1 << 20) + 7));
    assert_eq!(parse_key(&"\u{0}".repeat(1 << 16)), None);
    assert_eq!(parse_key(&".lora_A.".repeat(50_000)), None);
}

#[test]
fn pair_keys_reports_every_orphan_duplicate_and_unknown_key() {
    let mut rng = Rng::new(405);
    let pool: Vec<String> = ["a", "b.c", "x.lora_A.y"]
        .iter()
        .flat_map(|p| {
            [
                format!("base_model.model.{p}.lora_A.weight"),
                format!("base_model.model.{p}.lora_B.weight"),
                format!("base_model.model.{p}.lora_A.default.weight"),
                format!("{p}.lora_B.weight"),
                format!("{p}.lora_B.bias"),
                format!("{p}.weight"),
            ]
        })
        .collect();
    for case in 0..600 {
        let keys: Vec<&str> = (0..rng.below(10))
            .map(|_| rng.pick(&pool).as_str())
            .collect();
        // Oracle: group by (target, side) from parse_key.
        let mut a = std::collections::BTreeMap::<String, usize>::new();
        let mut b = std::collections::BTreeMap::<String, usize>::new();
        let mut unknown = 0;
        for k in &keys {
            match parse_key(k) {
                Some((t, Side::A)) => *a.entry(t).or_default() += 1,
                Some((t, Side::B)) => *b.entry(t).or_default() += 1,
                None => unknown += 1,
            }
        }
        let targets: std::collections::BTreeSet<&String> = a.keys().chain(b.keys()).collect();
        let clean = unknown == 0
            && targets
                .iter()
                .all(|t| a.get(*t) == Some(&1) && b.get(*t) == Some(&1));
        match (pair_keys(keys.iter().copied()), clean) {
            (Ok(pairs), true) => {
                assert_eq!(pairs.len(), targets.len(), "case {case}");
                assert!(
                    pairs.windows(2).all(|w| w[0].target < w[1].target),
                    "case {case}"
                );
                for p in &pairs {
                    assert_eq!(parse_key(&p.a), Some((p.target.clone(), Side::A)));
                    assert_eq!(parse_key(&p.b), Some((p.target.clone(), Side::B)));
                }
            }
            (Err(problems), false) => {
                for k in &keys {
                    let (target, side) = match parse_key(k) {
                        Some((t, s)) => (t, s),
                        None => {
                            assert!(
                                problems.iter().any(|p| p.contains(k)),
                                "case {case}: unknown {k} unreported"
                            );
                            continue;
                        }
                    };
                    let (mine, other) = match side {
                        Side::A => (&a, &b),
                        Side::B => (&b, &a),
                    };
                    let bad = mine.get(&target) != Some(&1) || other.get(&target).is_none();
                    if bad {
                        assert!(
                            problems.iter().any(|p| p.contains(k)),
                            "case {case}: {k} is not reported in {problems:?}"
                        );
                    }
                }
            }
            (got, clean) => panic!("case {case}: {keys:?}: clean={clean} but got {got:?}"),
        }
    }
}

#[test]
fn merge_lists_every_orphan_key_of_a_hostile_adapter_and_writes_nothing() {
    let mut rng = Rng::new(406);
    let junk = [
        "lora_A.weight",
        "base_model.model.q_proj.lora_A.weight",
        "weird\u{0}key",
        "x.lora_B.default.default.weight",
        "model.layers.9.mlp.up_proj.lora_B.weight",
        "\u{65e5}\u{672c}.lora_A.weight",
    ];
    for case in 0..20 {
        let mut adapter = good_adapter(&mut rng);
        let mut orphans = Vec::new();
        for _ in 0..1 + rng.below(4) {
            let key = format!("{}{}", rng.pick(&junk), rng.below(3));
            if orphans.contains(&key) {
                continue;
            }
            adapter.push(st(&key, "F32", &[2, 2], &[1.0; 4]));
            orphans.push(key);
        }
        let fx = fixture(&adapter, Some(&std_config()));
        let msg = assert_usage_and_untouched(
            &format!("case {case}"),
            merge_outcome(&fx, None, 1.0, false),
        );
        for key in &orphans {
            assert!(
                msg.contains(key.as_str()),
                "case {case}: {key:?} missing from {msg}"
            );
        }
    }
}

#[test]
fn merge_refuses_inconsistent_shapes_dtypes_and_non_finite_values_before_writing() {
    let mut rng = Rng::new(407);
    let name = |side: &str| format!("base_model.model.{QP}.lora_{side}.weight");
    let other = |side: &str| format!("base_model.model.{VP}.lora_{side}.weight");
    let g = |rng: &mut Rng, n: usize| gaussians(rng, n, 0.5);
    let cases: Vec<(&str, Vec<St>)> = vec![
        (
            "r of A and B disagree",
            vec![
                st(&name("A"), "F32", &[3, 5], &g(&mut rng, 15)),
                st(&name("B"), "F32", &[6, 2], &g(&mut rng, 12)),
            ],
        ),
        (
            "A columns differ from the model",
            vec![
                st(&name("A"), "F32", &[2, 6], &g(&mut rng, 12)),
                st(&name("B"), "F32", &[6, 2], &g(&mut rng, 12)),
            ],
        ),
        (
            "B rows differ from the model",
            vec![
                st(&name("A"), "F32", &[2, 5], &g(&mut rng, 10)),
                st(&name("B"), "F32", &[7, 2], &g(&mut rng, 14)),
            ],
        ),
        (
            "rank 0 factors",
            vec![
                st(&name("A"), "F32", &[0, 5], &[]),
                st(&name("B"), "F32", &[6, 0], &[]),
            ],
        ),
        (
            "A is 1D",
            vec![
                st(&name("A"), "F32", &[10], &g(&mut rng, 10)),
                st(&name("B"), "F32", &[6, 2], &g(&mut rng, 12)),
            ],
        ),
        (
            "B is 3D",
            vec![
                st(&name("A"), "F32", &[2, 5], &g(&mut rng, 10)),
                st(&name("B"), "F32", &[6, 2, 1], &g(&mut rng, 12)),
            ],
        ),
        (
            "two modules with different ranks",
            vec![
                st(&name("A"), "F32", &[2, 5], &g(&mut rng, 10)),
                st(&name("B"), "F32", &[6, 2], &g(&mut rng, 12)),
                st(&other("A"), "F32", &[3, 5], &g(&mut rng, 15)),
                st(&other("B"), "F32", &[4, 3], &g(&mut rng, 12)),
            ],
        ),
        (
            "module absent from the model",
            vec![
                st(
                    "base_model.model.model.layers.7.mlp.up_proj.lora_A.weight",
                    "F32",
                    &[2, 5],
                    &g(&mut rng, 10),
                ),
                st(
                    "base_model.model.model.layers.7.mlp.up_proj.lora_B.weight",
                    "F32",
                    &[6, 2],
                    &g(&mut rng, 12),
                ),
            ],
        ),
        (
            "target is the 1D norm",
            vec![
                st(
                    "base_model.model.model.norm.lora_A.weight",
                    "F32",
                    &[2, 5],
                    &g(&mut rng, 10),
                ),
                st(
                    "base_model.model.model.norm.lora_B.weight",
                    "F32",
                    &[5, 2],
                    &g(&mut rng, 10),
                ),
            ],
        ),
        (
            "A has an unsupported dtype",
            vec![
                St {
                    name: name("A"),
                    dtype: "I8",
                    shape: vec![2, 5],
                    bytes: vec![1; 10],
                },
                st(&name("B"), "F32", &[6, 2], &g(&mut rng, 12)),
            ],
        ),
        (
            "B is F64",
            vec![
                st(&name("A"), "F32", &[2, 5], &g(&mut rng, 10)),
                St {
                    name: name("B"),
                    dtype: "F64",
                    shape: vec![6, 2],
                    bytes: vec![0; 96],
                },
            ],
        ),
        (
            "NaN in A",
            vec![
                st(
                    &name("A"),
                    "F32",
                    &[2, 5],
                    &[0.5, f32::NAN, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                ),
                st(&name("B"), "F32", &[6, 2], &g(&mut rng, 12)),
            ],
        ),
        (
            "Inf in B (last element)",
            vec![
                st(&name("A"), "F32", &[2, 5], &g(&mut rng, 10)),
                st(
                    &name("B"),
                    "F32",
                    &[6, 2],
                    &[[0.5f32; 11].as_slice(), &[f32::NEG_INFINITY]].concat(),
                ),
            ],
        ),
        ("both factors empty adapter", vec![]),
    ];
    for (label, adapter) in cases {
        let fx = fixture(&adapter, Some(&std_config()));
        assert_usage_and_untouched(label, merge_outcome(&fx, None, 1.0, false));
        // The dry run must refuse the same adapters.
        assert_usage_and_untouched(label, merge_outcome(&fx, None, 1.0, true));
    }
}

#[test]
fn merge_with_zero_sized_targets_neither_panics_nor_writes() {
    let mut rng = Rng::new(408);
    let dir = TempDir::new().unwrap();
    let model = dir.path().join("model.safetensors");
    write_st(
        &model,
        &[
            st(&format!("{QP}.weight"), "F32", &[0, 5], &[]),
            st(&format!("{VP}.weight"), "F32", &[4, 0], &[]),
        ],
    );
    let lora = dir.path().join("adapter_model.safetensors");
    let [qa, qb] = lora_pair(QP, (0, 5, 2), &mut rng);
    let [va, vb] = lora_pair(VP, (4, 0, 2), &mut rng);
    write_st(&lora, &[qa, qb, va, vb]);
    fs::write(dir.path().join("adapter_config.json"), std_config()).unwrap();
    let fx = Fx { dir, model, lora };
    for dry_run in [true, false] {
        let (result, before, after) = merge_outcome(&fx, None, 1.0, dry_run);
        assert_eq!(before, after, "nothing to change");
        assert!(
            matches!(
                result,
                Ok(_) | Err(SurgeryError::Usage(_) | SurgeryError::Engine(_))
            ),
            "{result:?}"
        );
    }
}

#[test]
fn merge_scale_and_alpha_extremes_are_finite_or_usage_errors() {
    let mut rng = Rng::new(409);
    let adapter = good_adapter(&mut rng);
    // (alpha override, scale, must be refused)
    let cases = [
        (Some(f64::NAN), 1.0, true),
        (Some(f64::INFINITY), 1.0, true),
        (Some(f64::NEG_INFINITY), 1.0, true),
        (None, f64::NAN, true),
        (None, f64::INFINITY, true),
        (None, f64::NEG_INFINITY, true),
        (Some(1e308), 1.0, true),
        (Some(-1e308), 1.0, true),
        (Some(1e300), 1e300, true),
        (Some(1e-300), 1e-300, false),
        (Some(1e-320), 1.0, false),
        (Some(0.0), 1.0, false),
        (None, 0.0, false),
        (None, -1.0, false),
        (Some(-3.5), -2.0, false),
        (Some(1e38), 1.0, false),
    ];
    for (alpha, scale, refused) in cases {
        let fx = fixture(&adapter, Some(&std_config()));
        let outcome = merge_outcome(&fx, alpha, scale, true);
        let label = format!("alpha {alpha:?} scale {scale:e}");
        match refused {
            true => {
                assert_usage_and_untouched(&label, outcome);
            }
            false => {
                let (result, before, after) = outcome;
                assert_eq!(before, after, "{label}: dry run wrote");
                let report = result.unwrap_or_else(|e| panic!("{label}: {e}"));
                assert!(report.effective_scale.is_finite(), "{label}");
            }
        }
    }
    // The same finite extremes for real: a result that overflows is an error and a rollback,
    // never a model with Inf in it.
    for (alpha, scale) in [
        (Some(1e38), 1.0),
        (Some(1e-320), 1.0),
        (None, 0.0),
        (Some(-3.5), -2.0),
    ] {
        let fx = fixture(&adapter, Some(&std_config()));
        let (result, before, after) = merge_outcome(&fx, alpha, scale, false);
        match result {
            Ok(_) => {
                let new = fs::read(&fx.model).unwrap();
                let all_finite = new[new.len() - 4 * 55..]
                    .chunks_exact(4)
                    .all(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]).is_finite());
                assert!(
                    all_finite,
                    "alpha {alpha:?} scale {scale}: Inf or NaN written"
                );
            }
            Err(_) => assert_eq!(before, after, "alpha {alpha:?} scale {scale}: no rollback"),
        }
    }
}

#[test]
fn merge_overflow_in_the_middle_of_the_run_rolls_every_tensor_back() {
    let mut rng = Rng::new(410);
    let (qa, qb) = {
        let [a, b] = lora_pair(QP, (6, 5, 2), &mut rng);
        (a, b)
    };
    let a_key = format!("base_model.model.{VP}.lora_A.weight");
    let b_key = format!("base_model.model.{VP}.lora_B.weight");
    // v_proj sorts after q_proj: q_proj is written first, then the failure must undo it.
    let cases: Vec<(&str, f32, f32, f64)> = vec![
        // (label, A value, B value, alpha)
        ("B * scale overflows f32 before the kernel", 1.0, 1e30, 1e10),
        ("delta overflows to +Inf", 1e30, 1e30, 2.0),
        ("delta is NaN (Inf - Inf)", 3e30, 3e30, 2.0),
    ];
    for (label, a, b, alpha) in cases {
        let mut vb = vec![b; 8];
        let mut va = vec![a; 10];
        if label.contains("NaN") {
            // Row 0: +3e30 * 3e30 and -3e30 * 3e30 in the same sum.
            vb[1] = -b;
            va[5] = a;
        }
        let adapter = vec![
            St {
                name: qa.name.clone(),
                dtype: qa.dtype,
                shape: qa.shape.clone(),
                bytes: qa.bytes.clone(),
            },
            St {
                name: qb.name.clone(),
                dtype: qb.dtype,
                shape: qb.shape.clone(),
                bytes: qb.bytes.clone(),
            },
            st(&a_key, "F32", &[2, 5], &va),
            st(&b_key, "F32", &[4, 2], &vb),
        ];
        let fx = fixture(&adapter, Some(&std_config()));
        let (result, before, after) = merge_outcome(&fx, Some(alpha), 1.0, false);
        assert!(
            matches!(
                result,
                Err(SurgeryError::Usage(_) | SurgeryError::Failed(_))
            ),
            "{label}: {result:?}"
        );
        assert_eq!(before, after, "{label}: first tensor was not rolled back");
    }
}

fn config_cases() -> Vec<(&'static str, String, Option<bool>)> {
    // (label, adapter_config.json text, Some(true) = loads and is supported, Some(false) = loads
    // but a refusal applies, None = the file is refused)
    let nested = |depth: usize, key: &str| {
        format!("{{\"{key}\": {}{}}}", "[".repeat(depth), "]".repeat(depth))
    };
    vec![
        ("empty object", "{}".into(), Some(true)),
        ("alpha as a string", r#"{"lora_alpha": "16"}"#.into(), None),
        ("alpha NaN literal", r#"{"lora_alpha": NaN}"#.into(), None),
        (
            "alpha out of range",
            r#"{"lora_alpha": 1e999}"#.into(),
            None,
        ),
        ("alpha null", r#"{"lora_alpha": null}"#.into(), Some(true)),
        ("alpha negative", r#"{"lora_alpha": -8}"#.into(), Some(true)),
        ("alpha 1e308", r#"{"lora_alpha": 1e308}"#.into(), Some(true)),
        ("alpha array", r#"{"lora_alpha": [16]}"#.into(), None),
        ("r zero", r#"{"r": 0}"#.into(), Some(true)),
        ("r negative", r#"{"r": -1}"#.into(), None),
        ("r float", r#"{"r": 1.5}"#.into(), None),
        ("r string", r#"{"r": "4"}"#.into(), None),
        (
            "r u64::MAX",
            r#"{"r": 18446744073709551615}"#.into(),
            Some(true),
        ),
        ("r above u64", r#"{"r": 18446744073709551616}"#.into(), None),
        ("use_dora string", r#"{"use_dora": "yes"}"#.into(), None),
        ("use_dora number", r#"{"use_dora": 1}"#.into(), None),
        ("use_dora true", r#"{"use_dora": true}"#.into(), Some(false)),
        ("use_rslora null", r#"{"use_rslora": null}"#.into(), None),
        (
            "fan_in_fan_out true",
            r#"{"fan_in_fan_out": true}"#.into(),
            Some(false),
        ),
        (
            "rank_pattern filled",
            r#"{"rank_pattern": {"q": 2}}"#.into(),
            Some(false),
        ),
        (
            "rank_pattern empty array",
            r#"{"rank_pattern": []}"#.into(),
            Some(false),
        ),
        (
            "rank_pattern empty object",
            r#"{"rank_pattern": {}}"#.into(),
            Some(true),
        ),
        (
            "alpha_pattern number",
            r#"{"alpha_pattern": 0}"#.into(),
            Some(false),
        ),
        (
            "alpha_pattern null",
            r#"{"alpha_pattern": null}"#.into(),
            Some(true),
        ),
        // serde derives a positional sequence reader too: an array is accepted as a struct.
        ("root array", "[]".into(), Some(true)),
        ("root null", "null".into(), None),
        ("root number", "42".into(), None),
        ("empty file", String::new(), None),
        ("truncated", r#"{"r": 4, "lora_alpha": "#.into(), None),
        ("trailing garbage", r#"{"r": 4} x"#.into(), None),
        ("byte order mark", "\u{feff}{\"r\": 4}".into(), None),
        ("duplicate key", r#"{"r": 4, "r": 5}"#.into(), None),
        (
            "unknown field nested 100000 deep",
            nested(100_000, "zz"),
            Some(true),
        ),
        (
            "rank_pattern nested 100000 deep",
            nested(100_000, "rank_pattern"),
            None,
        ),
        (
            "modules_to_save nested 100000 deep",
            nested(100_000, "modules_to_save"),
            None,
        ),
        (
            "modules_to_save shallow",
            nested(20, "modules_to_save"),
            Some(true),
        ),
        (
            "1 MiB of padding",
            format!("{{\"r\": 4}}{}", " ".repeat(1 << 20)),
            Some(true),
        ),
    ]
}

#[test]
fn adapter_config_hostile_values_are_typed_refusals_or_supported() {
    let dir = TempDir::new().unwrap();
    for (label, text, expect) in config_cases() {
        let path = dir.path().join("cfg.json");
        fs::write(&path, &text).unwrap();
        let loaded = AdapterConfig::load(&path);
        match (expect, loaded) {
            (None, Err(SurgeryError::Usage(_))) => {}
            (Some(true), Ok(cfg)) => cfg
                .check_supported()
                .unwrap_or_else(|e| panic!("{label}: {e}")),
            (Some(false), Ok(cfg)) => assert!(
                matches!(cfg.check_supported(), Err(SurgeryError::Usage(_))),
                "{label}: should be refused"
            ),
            (expect, got) => panic!("{label}: expected {expect:?}, got {got:?}"),
        }
    }
    // Missing and non-UTF-8 files are usage errors too.
    assert!(matches!(
        AdapterConfig::load(&dir.path().join("nope.json")),
        Err(SurgeryError::Usage(_))
    ));
    fs::write(dir.path().join("bin.json"), [0xFFu8, 0xFE, b'{', b'}']).unwrap();
    assert!(matches!(
        AdapterConfig::load(&dir.path().join("bin.json")),
        Err(SurgeryError::Usage(_))
    ));
}

#[test]
fn merge_with_hostile_configs_is_a_typed_outcome_and_a_refusal_changes_nothing() {
    let mut rng = Rng::new(411);
    let adapter = good_adapter(&mut rng);
    for (label, text, expect) in config_cases() {
        if text.len() > 1 << 19 && expect.is_none() {
            continue;
        }
        let fx = fixture(&adapter, Some(&text));
        // Dry run: the whole preparation runs, nothing is written either way.
        let (result, before, after) = merge_outcome(&fx, None, 1.0, true);
        assert_eq!(before, after, "{label}");
        match (expect, &result) {
            (Some(true), Ok(_)) => {}
            // A supported file without an alpha is the documented usage error.
            (Some(true), Err(SurgeryError::Usage(msg))) => {
                assert!(
                    msg.contains("lora_alpha") || msg.contains("chelle"),
                    "{label}: {msg}"
                )
            }
            (_, Err(SurgeryError::Usage(_))) => {}
            (expect, got) => panic!("{label}: expected {expect:?}, got {got:?}"),
        }
    }
    // alpha 1e308 loads fine but its effective scale does not fit an f32.
    let fx = fixture(&adapter, Some(r#"{"lora_alpha": 1e308}"#));
    assert_usage_and_untouched(
        "alpha 1e308 in the config",
        merge_outcome(&fx, None, 1.0, false),
    );
    // r = 0 in the config only produces a warning; the real rank is the shape of A.
    let fx = fixture(&adapter, Some(r#"{"r": 0, "lora_alpha": 4}"#));
    let (result, ..) = merge_outcome(&fx, None, 1.0, true);
    let report = result.unwrap();
    assert_eq!(report.rank, 2);
    assert!(
        report.warnings.iter().any(|w| w.contains("r=")),
        "{:?}",
        report.warnings
    );
}

// ---- extract

struct Pair {
    dir: TempDir,
    base: PathBuf,
    patched: PathBuf,
}

/// Base and patched checkpoints (one tensor file each); `edit` receives the base tensors.
fn checkpoints(base: &[St], patched: &[St]) -> Pair {
    let dir = TempDir::new().unwrap();
    let (b, p) = (dir.path().join("base"), dir.path().join("patched"));
    fs::create_dir(&b).unwrap();
    fs::create_dir(&p).unwrap();
    write_st(&b.join("model.safetensors"), base);
    write_st(&p.join("model.safetensors"), patched);
    Pair {
        dir,
        base: b,
        patched: p,
    }
}

fn extract_args(p: &Pair, rank: usize, dtype: FactorDtype) -> LoraExtractArgs {
    LoraExtractArgs {
        base: p.base.clone(),
        patched: p.patched.clone(),
        out: p.dir.path().join("out").join("adapter_model.safetensors"),
        rank,
        dtype,
        target_modules: None,
        seed: 0,
        overwrite: false,
    }
}

fn run_extract_fresh(
    args: &LoraExtractArgs,
) -> Result<weight_surgeon::lora::extract::ExtractReport, SurgeryError> {
    fs::create_dir_all(args.out.parent().unwrap()).unwrap();
    run_extract(args, &mut NullObserver, &AtomicBool::new(false))
}

fn assert_no_output(args: &LoraExtractArgs) {
    let names = file_names(args.out.parent().unwrap());
    assert!(names.is_empty(), "partial output left behind: {names:?}");
}

#[test]
fn extract_with_a_rank_larger_than_every_dimension_is_clamped_and_merges_back() {
    let mut rng = Rng::new(500);
    let w = |rng: &mut Rng, n| gaussians(rng, n, 1.0);
    let base = [
        st(&format!("{QP}.weight"), "F32", &[4, 6], &w(&mut rng, 24)),
        st(&format!("{VP}.weight"), "F32", &[4, 9], &w(&mut rng, 36)),
    ];
    let patched: Vec<St> = base
        .iter()
        .map(|t| {
            let vals: Vec<f32> = t
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) + rng.signed() as f32)
                .collect();
            st(&t.name, "F32", &t.shape, &vals)
        })
        .collect();
    let pair = checkpoints(&base, &patched);
    let args = extract_args(&pair, 1000, FactorDtype::F32);
    let report = run_extract_fresh(&args).unwrap();
    assert_eq!(
        report.rank, 4,
        "clamped to the smallest dimension of the selected tensors"
    );
    assert!(
        report.warnings.iter().any(|w| w.contains("rang")),
        "{:?}",
        report.warnings
    );
    assert!(
        report.mean_rel_error < 1e-4,
        "full rank must reconstruct: {}",
        report.mean_rel_error
    );
    // Merge the adapter into a copy of the base: it must give the patched weights.
    let copy = pair.dir.path().join("copy");
    fs::create_dir(&copy).unwrap();
    fs::copy(
        pair.base.join("model.safetensors"),
        copy.join("model.safetensors"),
    )
    .unwrap();
    let req = MergeRequest {
        model_shards: resolve_shards(&copy).unwrap(),
        lora: args.out.clone(),
        alpha: None,
        adapter_config: None,
        scale: 1.0,
        dry_run: false,
        cancel: Arc::new(AtomicBool::new(false)),
    };
    let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
    run_merge(&req, &mut backup, &mut NullObserver).unwrap();
    let (merged, want) = (
        fs::read(copy.join("model.safetensors")).unwrap(),
        fs::read(pair.patched.join("model.safetensors")).unwrap(),
    );
    let to_f32 = |b: &[u8]| -> Vec<f32> {
        b[b.len() - 4 * 60..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let worst = to_f32(&merged)
        .iter()
        .zip(to_f32(&want))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 1e-3,
        "merged weights differ from the patched ones by {worst}"
    );
}

#[test]
fn extract_of_identical_checkpoints_gives_a_harmless_adapter() {
    let mut rng = Rng::new(501);
    let base = [st(
        &format!("{QP}.weight"),
        "F32",
        &[6, 5],
        &gaussians(&mut rng, 30, 1.0),
    )];
    let pair = checkpoints(&base, &base);
    let args = extract_args(&pair, 4, FactorDtype::F16);
    let report = run_extract_fresh(&args).unwrap();
    assert_eq!(report.mean_rel_error, 0.0);
    let bytes = fs::read(&args.out).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    for (name, view) in tensors.tensors() {
        let all_zero = view.data().iter().all(|&b| b == 0);
        assert!(all_zero, "{name}: a zero delta must give zero factors");
    }
}

#[test]
fn extract_refuses_unusable_tensors_with_typed_errors_and_leaves_no_output() {
    let mut rng = Rng::new(502);
    type Build = Box<dyn Fn(&mut Rng) -> (Vec<St>, Vec<St>)>;
    fn build(f: impl Fn(&mut Rng) -> (Vec<St>, Vec<St>) + 'static) -> Build {
        Box::new(f)
    }
    fn good(rng: &mut Rng) -> St {
        st(
            &format!("{QP}.weight"),
            "F32",
            &[4, 5],
            &gaussians(rng, 20, 1.0),
        )
    }
    let cases: Vec<(&str, Build, Option<&str>)> = vec![
        (
            "NaN in the base",
            build(|rng| {
                let mut v = gaussians(rng, 20, 1.0);
                v[7] = f32::NAN;
                (
                    vec![st(&format!("{QP}.weight"), "F32", &[4, 5], &v)],
                    vec![st(
                        &format!("{QP}.weight"),
                        "F32",
                        &[4, 5],
                        &gaussians(rng, 20, 1.0),
                    )],
                )
            }),
            None,
        ),
        (
            "difference overflows f32",
            build(|_| {
                (
                    vec![st(&format!("{QP}.weight"), "F32", &[2, 2], &[-3e38; 4])],
                    vec![st(&format!("{QP}.weight"), "F32", &[2, 2], &[3e38; 4])],
                )
            }),
            None,
        ),
        (
            "zero-row 2D tensor",
            build(|_| {
                (
                    vec![st(&format!("{QP}.weight"), "F32", &[0, 5], &[])],
                    vec![st(&format!("{QP}.weight"), "F32", &[0, 5], &[])],
                )
            }),
            None,
        ),
        (
            "zero-column 2D tensor",
            build(|_| {
                (
                    vec![st(&format!("{QP}.weight"), "F32", &[4, 0], &[])],
                    vec![st(&format!("{QP}.weight"), "F32", &[4, 0], &[])],
                )
            }),
            None,
        ),
        (
            "2D tensor without the .weight suffix",
            build(|rng| {
                let t = |rng: &mut Rng| {
                    st(
                        "model.layers.0.self_attn.q_proj.bias2d",
                        "F32",
                        &[3, 3],
                        &gaussians(rng, 9, 1.0),
                    )
                };
                (vec![t(rng)], vec![t(rng)])
            }),
            Some("q_proj"),
        ),
        (
            "unsupported dtype",
            build(|_| {
                let t = |b: u8| St {
                    name: format!("{QP}.weight"),
                    dtype: "I8",
                    shape: vec![3, 3],
                    bytes: vec![b; 9],
                };
                (vec![t(1)], vec![t(2)])
            }),
            None,
        ),
        (
            "dtype differs between the checkpoints",
            build(|rng| {
                (
                    vec![st(
                        &format!("{QP}.weight"),
                        "F32",
                        &[3, 3],
                        &gaussians(rng, 9, 1.0),
                    )],
                    vec![st(
                        &format!("{QP}.weight"),
                        "BF16",
                        &[3, 3],
                        &gaussians(rng, 9, 1.0),
                    )],
                )
            }),
            None,
        ),
        (
            "only 1D and 3D tensors",
            build(|rng| {
                let t = |rng: &mut Rng| {
                    vec![
                        st("model.norm.weight", "F32", &[5], &gaussians(rng, 5, 1.0)),
                        st(
                            &format!("{VP}.weight"),
                            "F32",
                            &[2, 2, 2],
                            &gaussians(rng, 8, 1.0),
                        ),
                    ]
                };
                (t(rng), t(rng))
            }),
            None,
        ),
        (
            "invalid regex",
            build(|rng| (vec![good(rng)], vec![good(rng)])),
            Some("("),
        ),
        (
            "regex selecting nothing",
            build(|rng| (vec![good(rng)], vec![good(rng)])),
            Some("^nothing$"),
        ),
    ];
    for (label, build, regex) in cases {
        let (base, patched) = build(&mut rng);
        let pair = checkpoints(&base, &patched);
        let mut args = extract_args(&pair, 4, FactorDtype::F32);
        args.target_modules = regex.map(str::to_owned);
        if label.contains("suffix") {
            args.target_modules = Some(".*".into());
        }
        let got = run_extract_fresh(&args);
        assert!(
            matches!(got, Err(SurgeryError::Usage(_) | SurgeryError::Failed(_))),
            "{label}: {got:?}"
        );
        assert_no_output(&args);
    }
    // A mix: the 1D and 3D tensors next to a good 2D one are simply not selected.
    let mixed = |rng: &mut Rng| {
        vec![
            good(rng),
            st("model.norm.weight", "F32", &[5], &gaussians(rng, 5, 1.0)),
            st(
                &format!("{VP}.weight"),
                "F32",
                &[2, 2, 2],
                &gaussians(rng, 8, 1.0),
            ),
        ]
    };
    let pair = checkpoints(&mixed(&mut rng), &mixed(&mut rng));
    let report = run_extract_fresh(&extract_args(&pair, 2, FactorDtype::F32)).unwrap();
    assert_eq!(report.tensors, 1);
}

#[test]
fn extract_output_path_oddities_are_usage_errors() {
    let mut rng = Rng::new(503);
    let t = |rng: &mut Rng| {
        [st(
            &format!("{QP}.weight"),
            "F32",
            &[4, 5],
            &gaussians(rng, 20, 1.0),
        )]
    };
    let pair = checkpoints(&t(&mut rng), &t(&mut rng));
    let missing_dir = {
        let mut a = extract_args(&pair, 2, FactorDtype::F32);
        a.out = pair.dir.path().join("no_such_dir").join("a.safetensors");
        a
    };
    let no_name = {
        let mut a = extract_args(&pair, 2, FactorDtype::F32);
        a.out = PathBuf::from("..");
        a
    };
    let empty = {
        let mut a = extract_args(&pair, 2, FactorDtype::F32);
        a.out = PathBuf::new();
        a
    };
    let zero_rank = extract_args(&pair, 0, FactorDtype::F32);
    for (label, args) in [
        ("missing directory", missing_dir),
        ("no file name", no_name),
        ("empty path", empty),
        ("rank 0", zero_rank),
    ] {
        let got = run_extract(&args, &mut NullObserver, &AtomicBool::new(false));
        assert!(
            matches!(got, Err(SurgeryError::Usage(_))),
            "{label}: {got:?}"
        );
    }
    // An existing output without --overwrite is refused and left byte for byte as it was.
    let args = extract_args(&pair, 2, FactorDtype::F32);
    run_extract_fresh(&args).unwrap();
    let before = fs::read(&args.out).unwrap();
    let again = run_extract(&args, &mut NullObserver, &AtomicBool::new(false));
    assert!(matches!(again, Err(SurgeryError::Usage(_))));
    assert_eq!(fs::read(&args.out).unwrap(), before);
}

// ------------------------------------------------------------------ 4. apply kernel

fn dt(name: &str) -> DType {
    match name {
        "BF16" => DType::BF16,
        "F16" => DType::F16,
        _ => DType::F32,
    }
}

fn load1(dtype: &str, b: &[u8]) -> f32 {
    match dtype {
        "BF16" => bf16::from_le_bytes([b[0], b[1]]).to_f32(),
        "F16" => f16::from_le_bytes([b[0], b[1]]).to_f32(),
        _ => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
    }
}

fn store1(dtype: &str, x: f32) -> Vec<u8> {
    encode_vals(dtype, &[x])
}

/// Straight loops over the documented semantics, with no blocks and no threads:
/// `W <- RNE(W + delta)` where a zero delta leaves the stored bits alone.
fn reference_apply(
    dtype: &str,
    w: &[u8],
    (rows, cols): (usize, usize),
    delta: impl Fn(usize, usize) -> f32,
) -> (Vec<u8>, LowRankStats) {
    let elem = if dtype == "F32" { 4 } else { 2 };
    let mut out = w.to_vec();
    let mut stats = LowRankStats::default();
    for i in 0..rows {
        for j in 0..cols {
            let d = delta(i, j);
            if d == 0.0 {
                continue;
            }
            let at = (i * cols + j) * elem;
            let old = load1(dtype, &out[at..at + elem]);
            let bits = store1(dtype, old + d);
            if bits != out[at..at + elem] {
                let new = load1(dtype, &bits);
                stats.elements_modified += 1;
                if new.is_finite() {
                    stats.max_abs_delta = stats.max_abs_delta.max((new - old).abs());
                } else if old.is_finite() {
                    stats.nan_or_inf_produced += 1;
                }
                out[at..at + elem].copy_from_slice(&bits);
            }
        }
    }
    (out, stats)
}

/// f32 accumulation in fixed `k` order, as documented.
fn lowrank_delta<'a>(
    u: &'a [f32],
    v: &'a [f32],
    cols: usize,
    r: usize,
) -> impl Fn(usize, usize) -> f32 + 'a {
    move |i, j| {
        let mut d = 0.0f32;
        for k in 0..r {
            d += u[i * r + k] * v[k * cols + j];
        }
        d
    }
}

/// Weights with some awkward values, and factors with exact zeros and a few magnitudes.
fn kernel_inputs(
    rng: &mut Rng,
    dtype: &str,
    (rows, cols, r): (usize, usize, usize),
) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let specials = [
        0.0f32,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        65504.0,
        1e-8,
        -1e30,
    ];
    let w: Vec<f32> = (0..rows * cols)
        .map(|_| match rng.below(50) {
            0 => *rng.pick(&specials),
            _ => rng.gauss() as f32,
        })
        .collect();
    let scale = *rng.pick(&[1.0f32, 1e-3, 100.0, 3e4]);
    let u = (0..rows * r)
        .map(|_| match rng.below(5) {
            0 => 0.0,
            _ => rng.gauss() as f32 * scale,
        })
        .collect();
    let v = (0..r * cols).map(|_| rng.gauss() as f32).collect();
    (encode_vals(dtype, &w), u, v)
}

fn run_kernel(
    dtype: &str,
    w: &[u8],
    shape: (usize, usize, usize),
    u: &[f32],
    v: &[f32],
) -> (Result<LowRankStats, SurgeryError>, Vec<u8>) {
    let mut bytes = w.to_vec();
    let out = apply_lowrank_rows(
        &mut bytes,
        dt(dtype),
        shape.0,
        shape.1,
        u,
        v,
        shape.2,
        &AtomicBool::new(false),
    );
    (out, bytes)
}

#[test]
fn apply_kernel_matches_a_plain_loop_on_shapes_around_the_block_size() {
    // BLOCK_ELEMS is 16384: sizes on both sides of one block, of two, and of one row per block.
    let mut shapes: Vec<(usize, usize, usize)> = vec![
        (1, 1, 1),
        (1, 1, 5),
        (1, 7, 3),
        (9, 1, 3),
        (3, 1, 0),
        (5, 5, 0),
        (2, 30, 7),
        (30, 2, 7),
        (1, 16384, 3),
        (2, 16384, 2),
        (3, 16383, 3),
        (2, 16385, 3),
        (3, 8192, 2),
        (5, 8192, 2),
        (3, 8193, 2),
        (3, 8191, 2),
        (16383, 1, 3),
        (16384, 1, 3),
        (16385, 1, 3),
        (32769, 1, 2),
        (8193, 2, 2),
        (5461, 3, 2),
        (5462, 3, 2),
        (129, 127, 2),
        (130, 127, 2),
        (128, 128, 2),
        (129, 128, 2),
        (127, 128, 2),
        (1, 40000, 1),
        (40000, 1, 1),
        (7, 4097, 9),
    ];
    let mut rng = Rng::new(0xA11);
    for _ in 0..40 {
        shapes.push((rng.between(1, 40), rng.between(1, 40), rng.between(0, 12)));
    }
    for shape in shapes {
        for dtype in ["BF16", "F16", "F32"] {
            let (w, u, v) = kernel_inputs(&mut rng, dtype, shape);
            let (want, want_stats) = reference_apply(
                dtype,
                &w,
                (shape.0, shape.1),
                lowrank_delta(&u, &v, shape.1, shape.2),
            );
            let (got, bytes) = run_kernel(dtype, &w, shape, &u, &v);
            assert_eq!(got.unwrap(), want_stats, "{shape:?} {dtype}: stats");
            assert!(
                bytes == want,
                "{shape:?} {dtype}: bytes differ from the reference"
            );
        }
    }
}

#[test]
fn apply_kernel_is_bit_identical_with_1_3_and_16_threads() {
    let mut rng = Rng::new(0xA12);
    for shape in [(257usize, 129usize, 5usize), (3, 16385, 4), (20000, 3, 3)] {
        let (w, u, v) = kernel_inputs(&mut rng, "BF16", shape);
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| run_kernel("BF16", &w, shape, &u, &v))
        };
        let (stats, bytes) = run(1);
        for threads in [3, 16] {
            let (s, b) = run(threads);
            assert_eq!(s.unwrap(), *stats.as_ref().unwrap(), "{shape:?} x{threads}");
            assert!(b == bytes, "{shape:?} x{threads}");
        }
    }
}

#[test]
fn apply_full_delta_matches_a_plain_loop_for_awkward_scales() {
    let mut rng = Rng::new(0xA13);
    for (rows, cols) in [
        (1usize, 1usize),
        (1, 33),
        (33, 1),
        (4, 16384),
        (5, 16385),
        (16385, 2),
        (7, 9),
    ] {
        for dtype in ["BF16", "F16", "F32"] {
            let (w, _, _) = kernel_inputs(&mut rng, dtype, (rows, cols, 0));
            let delta: Vec<f32> = (0..rows * cols)
                .map(|_| rng.gauss() as f32 * 50.0)
                .collect();
            for scale in [0.0f32, 1.0, -1.0, 0.1, 1e-30, 1e30, -3e4] {
                let (want, want_stats) =
                    reference_apply(dtype, &w, (rows, cols), |i, j| scale * delta[i * cols + j]);
                let mut bytes = w.clone();
                let got = apply_full_delta(
                    &mut bytes,
                    dt(dtype),
                    rows,
                    cols,
                    &delta,
                    scale,
                    &AtomicBool::new(false),
                )
                .unwrap();
                assert_eq!(got, want_stats, "{rows}x{cols} {dtype} scale {scale}");
                assert!(bytes == want, "{rows}x{cols} {dtype} scale {scale}");
            }
        }
    }
}

#[test]
fn apply_kernel_refuses_hostile_arguments_and_leaves_the_bytes_alone() {
    let mut rng = Rng::new(0xA14);
    let (rows, cols, r) = (6usize, 5usize, 3usize);
    let (w, u, v) = kernel_inputs(&mut rng, "F32", (rows, cols, r));
    let never = AtomicBool::new(false);
    let finite_u: Vec<f32> = u
        .iter()
        .map(|x| if x.is_finite() { *x } else { 1.0 })
        .collect();
    let check = |label: &str,
                 bytes: &[u8],
                 rows: usize,
                 cols: usize,
                 u: &[f32],
                 v: &[f32],
                 r: usize,
                 dtype: DType| {
        let mut copy = bytes.to_vec();
        let got = apply_lowrank_rows(&mut copy, dtype, rows, cols, u, v, r, &never);
        assert!(
            matches!(got, Err(SurgeryError::Usage(_))),
            "{label}: {got:?}"
        );
        assert!(copy == bytes, "{label}: bytes modified");
    };
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for at in [0, finite_u.len() / 2, finite_u.len() - 1] {
            let mut u2 = finite_u.clone();
            u2[at] = bad;
            check(
                &format!("{bad} in u at {at}"),
                &w,
                rows,
                cols,
                &u2,
                &v,
                r,
                DType::F32,
            );
        }
        for at in [0, v.len() - 1] {
            let mut v2 = v.clone();
            v2[at] = bad;
            check(
                &format!("{bad} in v at {at}"),
                &w,
                rows,
                cols,
                &finite_u,
                &v2,
                r,
                DType::F32,
            );
        }
    }
    let cases: Vec<(&str, usize, usize, usize, usize, usize)> = vec![
        // (label, rows, cols, r, bytes to cut from the tensor, extra u values)
        ("tensor one byte short", rows, cols, r, 1, 0),
        ("u one value too long", rows, cols, r, 0, 1),
    ];
    for (label, rows, cols, r, cut, extra) in cases {
        let mut u2 = finite_u.clone();
        u2.extend(std::iter::repeat_n(1.0, extra));
        check(
            label,
            &w[..w.len() - cut],
            rows,
            cols,
            &u2,
            &v,
            r,
            DType::F32,
        );
    }
    check(
        "v one value short",
        &w,
        rows,
        cols,
        &finite_u,
        &v[1..],
        r,
        DType::F32,
    );
    check(
        "r larger than the factors",
        &w,
        rows,
        cols,
        &finite_u,
        &v,
        r + 1,
        DType::F32,
    );
    check(
        "r = usize::MAX",
        &w,
        rows,
        cols,
        &finite_u,
        &v,
        usize::MAX,
        DType::F32,
    );
    check(
        "rows x cols overflows",
        &[],
        usize::MAX / 2,
        3,
        &[],
        &[],
        0,
        DType::F32,
    );
    check(
        "rows x cols x elem overflows",
        &[],
        usize::MAX / 4 + 1,
        1,
        &[],
        &[],
        0,
        DType::F32,
    );
    check(
        "rows x r overflows",
        &w,
        usize::MAX,
        0,
        &[],
        &[],
        3,
        DType::F32,
    );
    check(
        "unsupported dtype",
        &w,
        rows,
        cols,
        &finite_u,
        &v,
        r,
        DType::Other("I8".into()),
    );
    check(
        "dtype size mismatch",
        &w,
        rows,
        cols,
        &finite_u,
        &v,
        r,
        DType::BF16,
    );
    // Empty tensors and empty rank are fine and do nothing.
    for (rows, cols, r) in [(0usize, 5usize, 2usize), (5, 0, 2), (0, 0, 0), (4, 4, 0)] {
        let mut bytes = vec![7u8; rows * cols * 4];
        let before = bytes.clone();
        let u = vec![1.0f32; rows * r];
        let v = vec![1.0f32; r * cols];
        let got =
            apply_lowrank_rows(&mut bytes, DType::F32, rows, cols, &u, &v, r, &never).unwrap();
        assert_eq!(got, LowRankStats::default());
        assert_eq!(bytes, before);
    }
    // Full delta: scale and delta must be finite too.
    for scale in [f32::NAN, f32::INFINITY] {
        let mut bytes = w.clone();
        let got = apply_full_delta(
            &mut bytes,
            DType::F32,
            rows,
            cols,
            &vec![1.0; rows * cols],
            scale,
            &never,
        );
        assert!(matches!(got, Err(SurgeryError::Usage(_))) && bytes == w);
    }
    let mut bytes = w.clone();
    let got = apply_full_delta(
        &mut bytes,
        DType::F32,
        rows,
        cols,
        &vec![f32::NAN; rows * cols],
        1.0,
        &never,
    );
    assert!(matches!(got, Err(SurgeryError::Usage(_))) && bytes == w);
}

#[test]
fn apply_kernel_cancellation_from_another_thread_never_tears_a_block() {
    let mut rng = Rng::new(0xA15);
    // 16 rows per block (16384 / 1024): 32 blocks of 16 rows.
    let (rows, cols, r) = (2048usize, 1024usize, 8usize);
    let w = encode_vals("BF16", &gaussians(&mut rng, rows * cols, 1.0));
    let u = gaussians(&mut rng, rows * r, 1.0);
    let v = gaussians(&mut rng, r * cols, 1.0);
    let (new, _) = reference_apply("BF16", &w, (rows, cols), lowrank_delta(&u, &v, cols, r));
    let row_bytes = cols * 2;
    // Warm run first (thread pool start-up), then the timing the delays are drawn from.
    run_kernel("BF16", &w, (rows, cols, r), &u, &v).0.unwrap();
    let started = Instant::now();
    run_kernel("BF16", &w, (rows, cols, r), &u, &v).0.unwrap();
    let full = started.elapsed();
    let mut cancelled = 0;
    for i in 0..14 {
        let delay = full.mul_f64(rng.unit() * 1.1) * u32::from(i > 0);
        let flag = Arc::new(AtomicBool::new(false));
        let mut bytes = w.clone();
        let got = std::thread::scope(|scope| {
            let raiser = flag.clone();
            scope.spawn(move || {
                std::thread::sleep(delay);
                raiser.store(true, Ordering::SeqCst);
            });
            apply_lowrank_rows(&mut bytes, DType::BF16, rows, cols, &u, &v, r, &flag)
        });
        match got {
            Ok(_) => assert!(
                bytes == new,
                "run {i}: finished but differs from the reference"
            ),
            Err(SurgeryError::Cancelled) => {
                cancelled += 1;
                for block in 0..rows / 16 {
                    let range = block * 16 * row_bytes..(block + 1) * 16 * row_bytes;
                    let (done, untouched) = (
                        bytes[range.clone()] == new[range.clone()],
                        bytes[range.clone()] == w[range.clone()],
                    );
                    assert!(done || untouched, "run {i}: block {block} is torn");
                }
            }
            Err(other) => panic!("run {i}: {other}"),
        }
    }
    assert!(
        cancelled > 0,
        "no run was interrupted: raise the problem size (full run {full:?})"
    );
    // Raised before the first block: nothing may be written (documented).
    let mut bytes = w.clone();
    let raised = AtomicBool::new(true);
    let got = apply_lowrank_rows(&mut bytes, DType::BF16, rows, cols, &u, &v, r, &raised);
    assert!(matches!(got, Err(SurgeryError::Cancelled)) && bytes == w);
}

// ------------------------------------------------------------------ 5. preview editor and slider

fn display_width(s: &str) -> usize {
    s.chars().map(|c| c.width().unwrap_or(0)).sum()
}

const EDIT_CHARS: [char; 36] = [
    'a',
    'Z',
    ' ',
    '0',
    'é',
    '\u{301}',
    '\u{200d}',
    '\u{fe0f}',
    '\u{200b}',
    '\u{202e}',
    '👨',
    '👩',
    '👧',
    '❤',
    '日',
    '本',
    '語',
    'Ａ',
    '🇫',
    '🇷',
    '👍',
    '\u{1f3fd}',
    '\n',
    '\t',
    '\u{7}',
    '\u{7f}',
    '\u{85}',
    '\u{0}',
    '\u{feff}',
    '\u{20e3}',
    '\u{e01}',
    '\u{e31}',
    'ñ',
    '\u{0308}',
    '\u{2028}',
    '\u{1b}',
];

fn random_text(rng: &mut Rng, max: usize) -> String {
    (0..rng.below(max + 1))
        .map(|_| *rng.pick(&EDIT_CHARS))
        .collect()
}

/// Checks the invariants of the editor against the previous state after one operation.
fn check_visible(e: &LineEditor, rng: &mut Rng) {
    let total = display_width(e.text());
    assert!(
        e.cursor_col() <= total,
        "cursor beyond the text: {} > {total} in {:?}",
        e.cursor_col(),
        e.text()
    );
    let widths = [0usize, 1, 2, 3, 5, 8, 40, usize::MAX, rng.between(0, 30)];
    for width in widths {
        let (shown, col) = e.visible(width);
        assert!(
            display_width(&shown) <= width,
            "{shown:?} wider than {width}"
        );
        assert!(col < width.max(1), "cursor column {col} outside {width}");
        assert!(
            col <= display_width(&shown),
            "cursor {col} past what is shown {shown:?}"
        );
        assert!(
            e.text().contains(&shown),
            "{shown:?} is not a piece of {:?}",
            e.text()
        );
    }
}

#[test]
fn line_editor_survives_random_edits_on_hostile_text() {
    let mut rng = Rng::new(0xED17);
    for session in 0..300 {
        let mut e = match session % 3 {
            0 => LineEditor::default(),
            1 => LineEditor::new(&random_text(&mut rng, 12)),
            _ => LineEditor::new(""),
        };
        let clean_start = session % 3 != 1;
        assert_eq!(e.is_empty(), e.text().is_empty());
        if session % 3 == 1 {
            assert_eq!(
                e.cursor_col(),
                display_width(e.text()),
                "new() puts the cursor at the end"
            );
        }
        for step in 0..60 {
            let (text, col) = (e.text().to_owned(), e.cursor_col());
            let op = rng.below(9);
            let label = format!("session {session} step {step} op {op} on {text:?} (col {col})");
            match op {
                0..=2 => {
                    let c = *rng.pick(&EDIT_CHARS);
                    e.insert(c);
                    match c.is_control() {
                        true => assert_eq!(e.text(), text, "{label}: control char {c:?} inserted"),
                        false => {
                            assert_eq!(e.text().len(), text.len() + c.len_utf8(), "{label}");
                            assert_eq!(e.cursor_col(), col + c.width().unwrap_or(0), "{label}");
                        }
                    }
                }
                3 => {
                    e.backspace();
                    assert!(
                        e.text().len() <= text.len() && e.cursor_col() <= col,
                        "{label}"
                    );
                    if col == 0 && text.chars().all(|c| c.width().is_some_and(|w| w > 0)) {
                        assert_eq!(e.text(), text, "{label}: backspace at the start");
                    }
                }
                4 => {
                    e.delete();
                    assert!(
                        e.text().len() <= text.len() && e.cursor_col() == col,
                        "{label}"
                    );
                }
                5 => {
                    e.left();
                    assert!(e.cursor_col() <= col && e.text() == text, "{label}");
                }
                6 => {
                    e.right();
                    assert!(e.cursor_col() >= col && e.text() == text, "{label}");
                }
                7 => match rng.below(3) {
                    0 => {
                        e.home();
                        assert_eq!((e.cursor_col(), e.text()), (0, text.as_str()), "{label}");
                    }
                    1 => {
                        e.end();
                        assert_eq!(e.cursor_col(), display_width(&text), "{label}");
                    }
                    _ => {
                        e.kill_to_start();
                        assert_eq!(e.cursor_col(), 0, "{label}");
                        assert!(
                            text.ends_with(e.text()),
                            "{label}: kept text is not a suffix"
                        );
                    }
                },
                _ => {
                    // Home then right until the end must terminate within the char count.
                    let mut walker = e.clone();
                    walker.home();
                    for _ in 0..=text.chars().count() {
                        walker.right();
                    }
                    assert_eq!(
                        walker.cursor_col(),
                        display_width(&text),
                        "{label}: right never reaches the end"
                    );
                    walker.end();
                    for _ in 0..=text.chars().count() {
                        walker.left();
                    }
                    assert_eq!(
                        walker.cursor_col(),
                        0,
                        "{label}: left never reaches the start"
                    );
                }
            }
            if clean_start {
                assert!(
                    !e.text().chars().any(char::is_control),
                    "{label}: control char in the text"
                );
            }
            check_visible(&e, &mut rng);
        }
    }
}

#[test]
fn line_editor_boundaries_on_an_empty_line_and_with_marks() {
    let mut e = LineEditor::default();
    for op in 0..4 {
        match op {
            0 => e.backspace(),
            1 => e.delete(),
            2 => e.left(),
            _ => e.right(),
        }
        assert!(e.is_empty() && e.cursor_col() == 0);
    }
    e.home();
    e.end();
    e.kill_to_start();
    assert_eq!(e.visible(0), (String::new(), 0));
    assert_eq!(e.visible(usize::MAX), (String::new(), 0));
    // A combining mark and a variation selector stay with their base.
    for tail in ["e\u{301}", "\u{2764}\u{fe0f}", "n\u{303}\u{323}"] {
        let mut e = LineEditor::new(&format!("a{tail}"));
        e.backspace();
        assert_eq!(e.text(), "a", "{tail:?}");
        let mut e = LineEditor::new(&format!("{tail}z"));
        e.home();
        e.delete();
        assert_eq!(e.text(), "z", "{tail:?}");
    }
    // A text that starts with a mark has nothing to attach it to: still editable.
    let mut lone = LineEditor::new("\u{301}\u{301}");
    lone.backspace();
    lone.delete();
    assert!(lone.cursor_col() == 0);
    // Wide characters count two columns and the window never cuts one in half.
    let wide = LineEditor::new("日本語日本語");
    for width in 1..16 {
        let (shown, col) = wide.visible(width);
        assert!(
            display_width(&shown) <= width && col <= width,
            "{width}: {shown:?} {col}"
        );
    }
}

struct TestClock(AtomicU64);

impl std::fmt::Debug for TestClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TestClock({})", self.0.load(Ordering::SeqCst))
    }
}

impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.0.load(Ordering::SeqCst)
    }
}

fn preview_state(
    clock: &Arc<TestClock>,
    prompt: &str,
    alpha: f64,
    range: (f64, f64),
) -> PreviewState {
    PreviewState::new(
        PreviewConfig {
            prompt: prompt.to_owned(),
            alpha,
            alpha_range: range,
            tokens: 7,
            layer: 2,
            template: ChatTemplate::Chatml,
        },
        clock.clone(),
    )
}

fn key(code: KeyCode, modifiers: KeyModifiers) -> PreviewMsg {
    PreviewMsg::Key(KeyEvent::new(code, modifiers))
}

fn random_key_msg(rng: &mut Rng) -> PreviewMsg {
    let code = match rng.below(16) {
        0 => KeyCode::Left,
        1 => KeyCode::Right,
        2 => KeyCode::Char('0'),
        3 => KeyCode::Char('t'),
        4 => KeyCode::Char('r'),
        5 => KeyCode::Char('i'),
        6 => KeyCode::Enter,
        7 => KeyCode::Esc,
        8 => KeyCode::Backspace,
        9 => KeyCode::Delete,
        10 => KeyCode::Home,
        11 => KeyCode::End,
        12 => KeyCode::Tab,
        13 => KeyCode::F(1 + rng.below(12) as u8),
        14 => KeyCode::Up,
        _ => KeyCode::Char(*rng.pick(&EDIT_CHARS)),
    };
    let mut modifiers = KeyModifiers::NONE;
    for flag in [
        KeyModifiers::SHIFT,
        KeyModifiers::CONTROL,
        KeyModifiers::ALT,
    ] {
        if rng.below(4) == 0 {
            modifiers |= flag;
        }
    }
    key(code, modifiers)
}

#[test]
fn preview_slider_stays_in_range_on_the_grid_and_requests_match_the_state() {
    let ranges: [(f64, f64); 8] = [
        (-2.0, 2.0),
        (0.0, 1.0),
        (-0.5, 0.03),
        (0.5, 1.0),
        (-1e308, 1e308),
        (0.003, 0.007),
        (-1e-320, 1e-320),
        (1e300, 1e308),
    ];
    let mut rng = Rng::new(0x5111);
    for (n, range) in ranges.into_iter().enumerate() {
        let on_grid = n < 4;
        let clock = Arc::new(TestClock(AtomicU64::new(0)));
        // The initial alpha is clamped but not rounded (it comes from the command line).
        let raw = range.0 + (range.1 - range.0) * rng.unit();
        let start = if on_grid {
            (raw * 100.0).round() / 100.0
        } else {
            raw
        };
        let mut state = preview_state(&clock, "bonjour", start, range);
        let mut last_id = 0u64;
        for step in 0..1500 {
            let msg = match rng.below(6) {
                0 => {
                    clock.0.fetch_add(rng.below(600) as u64, Ordering::SeqCst);
                    PreviewMsg::Tick
                }
                _ => random_key_msg(&mut rng),
            };
            let is_tick = matches!(msg, PreviewMsg::Tick);
            let effects = update(&mut state, msg);
            let label = format!("range {range:?} step {step}");
            assert!(
                state.alpha.is_finite() && state.alpha >= range.0 && state.alpha <= range.1,
                "{label}: alpha {}",
                state.alpha
            );
            if on_grid {
                let cents = state.alpha * 100.0;
                assert!(
                    (cents - cents.round()).abs() < 1e-6,
                    "{label}: alpha {} is off the 0.01 grid",
                    state.alpha
                );
            }
            assert!(
                effects.len() <= 1,
                "{label}: several requests for one message"
            );
            assert!(!is_tick || effects.len() <= 1);
            for PreviewEffect::Generate(request) in effects {
                assert_eq!(request.alpha, state.alpha, "{label}");
                assert_eq!(request.prompt, state.prompt, "{label}");
                assert!(
                    !request.prompt.trim().is_empty(),
                    "{label}: blank prompt requested"
                );
                assert_eq!(
                    (request.tokens, request.gen_id),
                    (7, last_id + 1),
                    "{label}"
                );
                last_id = request.gen_id;
            }
            assert!(
                state.shown_alpha >= range.0 && state.shown_alpha <= range.1,
                "{label}"
            );
            assert!(
                !state.editor.text().chars().any(char::is_control),
                "{label}"
            );
        }
    }
}

#[test]
fn preview_slider_steps_do_not_drift_and_reach_the_bounds_exactly() {
    let clock = Arc::new(TestClock(AtomicU64::new(0)));
    let mut state = preview_state(&clock, "p", 0.0, (-2.0, 2.0));
    let none = KeyModifiers::NONE;
    for _ in 0..40 {
        update(&mut state, key(KeyCode::Right, none));
    }
    assert_eq!(state.alpha, 2.0, "40 x 0.05");
    for _ in 0..200 {
        update(&mut state, key(KeyCode::Right, none));
    }
    assert_eq!(state.alpha, 2.0, "bounded above");
    for _ in 0..200 {
        update(&mut state, key(KeyCode::Left, KeyModifiers::SHIFT));
    }
    assert_eq!(
        state.alpha, 0.0,
        "200 x 0.01 back from 2.0, no accumulated error"
    );
    for _ in 0..1000 {
        update(&mut state, key(KeyCode::Left, none));
    }
    assert_eq!(state.alpha, -2.0);
    update(&mut state, key(KeyCode::Char('0'), none));
    assert_eq!(state.alpha, 0.0);
    // Reset outside the range lands on the nearest bound.
    let mut narrow = preview_state(&clock, "p", 0.7, (0.5, 1.0));
    update(&mut narrow, key(KeyCode::Char('0'), none));
    assert_eq!(narrow.alpha, 0.5);
    // Debounce: many moves, one request, carrying the last value.
    let mut state = preview_state(&clock, "p", 0.0, (-2.0, 2.0));
    let mut requests = Vec::new();
    for _ in 0..30 {
        requests.extend(update(&mut state, key(KeyCode::Right, none)));
        clock.0.fetch_add(10, Ordering::SeqCst);
        requests.extend(update(&mut state, PreviewMsg::Tick));
    }
    assert!(
        requests.is_empty(),
        "moves inside the debounce window must not generate"
    );
    clock.0.fetch_add(300, Ordering::SeqCst);
    let fired = update(&mut state, PreviewMsg::Tick);
    assert!(
        matches!(fired.as_slice(), [PreviewEffect::Generate(r)] if r.alpha == 1.5),
        "{fired:?}"
    );
    assert!(
        update(&mut state, PreviewMsg::Tick).is_empty(),
        "fires once"
    );
}

#[test]
fn preview_ignores_stale_worker_events_and_never_generates_when_unavailable() {
    let clock = Arc::new(TestClock(AtomicU64::new(0)));
    let mut rng = Rng::new(0x5112);
    let mut state = preview_state(&clock, "bonjour", 0.0, (-2.0, 2.0));
    let first = state.start();
    let [PreviewEffect::Generate(current)] = first.as_slice() else {
        panic!("start must request one generation: {first:?}")
    };
    let id = current.gen_id;
    let (baseline, steered) = (state.baseline.clone(), state.steered.clone());
    for stale in [0, id + 1, id + 1000, u64::MAX, id.wrapping_sub(1)] {
        for pane in [Pane::Baseline, Pane::Steered] {
            let events = [
                Event::Token {
                    gen_id: stale,
                    pane,
                    text: "x".into(),
                    tok_idx: 3,
                    tok_per_s: 9.0,
                },
                Event::Done {
                    gen_id: stale,
                    pane,
                },
                Event::Error {
                    gen_id: stale,
                    message: "late".into(),
                },
                Event::Scale {
                    gen_id: stale,
                    scale: 4.0,
                },
            ];
            for event in events {
                update(&mut state, PreviewMsg::Worker(event));
            }
        }
    }
    assert_eq!(
        (&state.baseline, &state.steered),
        (&baseline, &steered),
        "stale events changed a pane"
    );
    assert_eq!(state.scale, None);
    // A burst of random worker traffic for the current id, with hostile text: no panic.
    for _ in 0..500 {
        let pane = if rng.below(2) == 0 {
            Pane::Baseline
        } else {
            Pane::Steered
        };
        let event = match rng.below(4) {
            0 => Event::Token {
                gen_id: id,
                pane,
                text: random_text(&mut rng, 30),
                tok_idx: rng.below(100),
                tok_per_s: rng.unit() * 50.0,
            },
            1 => Event::Done { gen_id: id, pane },
            2 => Event::Error {
                gen_id: id,
                message: random_text(&mut rng, 10),
            },
            _ => Event::Scale {
                gen_id: id,
                scale: rng.unit() as f32 * 10.0,
            },
        };
        update(&mut state, PreviewMsg::Worker(event));
        assert!(state.alpha_abs().is_none_or(f64::is_finite));
    }
    let mut down = preview_state(&clock, "bonjour", 0.5, (-2.0, 2.0));
    down.unavailable("not enough memory".into());
    for _ in 0..400 {
        let effects = update(&mut down, random_key_msg(&mut rng));
        clock.0.fetch_add(300, Ordering::SeqCst);
        assert!(
            effects.is_empty() && update(&mut down, PreviewMsg::Tick).is_empty(),
            "generated while unavailable"
        );
    }
}

#[test]
fn preview_view_renders_hostile_prompts_and_ranges_at_any_terminal_size() {
    let clock = Arc::new(TestClock(AtomicU64::new(0)));
    let theme = Theme::new(ColorMode::Plain);
    let mut rng = Rng::new(0x5113);
    let ranges = [(-2.0, 2.0), (-1e308, 1e308), (1e-320, 2e-320)];
    for case in 0..120 {
        let prompt = random_text(&mut rng, 80);
        let range = ranges[case % ranges.len()];
        let mut state = preview_state(&clock, &prompt, range.0, range);
        state.input_focus = rng.below(2) == 0;
        for _ in 0..rng.below(20) {
            update(&mut state, random_key_msg(&mut rng));
        }
        if case % 4 == 0 {
            state.unavailable(random_text(&mut rng, 60));
        }
        let id = state.start().len() as u64;
        update(
            &mut state,
            PreviewMsg::Worker(Event::Scale {
                gen_id: id,
                scale: 1e30,
            }),
        );
        for pane in [Pane::Baseline, Pane::Steered] {
            let text = format!("{}\u{fffd}", random_text(&mut rng, 400));
            update(
                &mut state,
                PreviewMsg::Worker(Event::Token {
                    gen_id: id,
                    pane,
                    text,
                    tok_idx: usize::MAX,
                    tok_per_s: 1e300,
                }),
            );
        }
        let (w, h) = (rng.between(0, 90) as u16, rng.between(0, 40) as u16);
        let mut terminal = Terminal::new(TestBackend::new(w, h)).unwrap();
        terminal
            .draw(|frame| view(frame, frame.area(), &state, &theme))
            .unwrap();
    }
}

// ------------------------------------------------------------------ 6. cancellation

/// Raises `flag` from another thread after `delay` while `work` runs.
fn with_cancel_after<T>(delay: Duration, flag: &AtomicBool, work: impl FnOnce() -> T) -> T {
    std::thread::scope(|scope| {
        scope.spawn(|| {
            std::thread::sleep(delay);
            flag.store(true, Ordering::SeqCst);
        });
        work()
    })
}

/// Delay of run `i`: none for the first one (cancelled at once), then random up to a little more
/// than a full run so that some runs complete.
fn random_delay(rng: &mut Rng, i: usize, full: Duration) -> Duration {
    match i {
        0 => Duration::ZERO,
        _ => full.mul_f64(rng.unit() * 1.15),
    }
}

/// `n` square F32 tensors `model.layers.i.self_attn.q_proj.weight` and a patched copy.
fn square_pair(n: usize, dim: usize, seed: u64) -> Pair {
    let mut rng = Rng::new(seed);
    let name = |i: usize| format!("model.layers.{i}.self_attn.q_proj.weight");
    let base: Vec<St> = (0..n)
        .map(|i| {
            st(
                &name(i),
                "F32",
                &[dim, dim],
                &gaussians(&mut rng, dim * dim, 1.0),
            )
        })
        .collect();
    let patched: Vec<St> = base
        .iter()
        .map(|t| {
            let vals: Vec<f32> = t
                .bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) + (rng.gauss() * 0.1) as f32)
                .collect();
            st(&t.name, "F32", &t.shape, &vals)
        })
        .collect();
    checkpoints(&base, &patched)
}

fn diff_args(pair: &Pair, out: PathBuf, rank: usize) -> DiffArgs {
    DiffArgs {
        base: pair.base.clone(),
        tuned: pair.patched.clone(),
        out,
        rank,
        factor_dtype: FactorDtype::F16,
        include: None,
        exclude: None,
        include_embeddings: false,
        max_rel_error: None,
        on_high_error: OnHighError::Warn,
        oversample: 8,
        power_iters: 3,
        seed: 0,
        timestamp: false,
        overwrite: false,
    }
}

/// Breaks at the `at`-th progress event (1-based), counting them all.
struct BreakAt {
    at: usize,
    seen: usize,
}

impl SurgeryObserver for BreakAt {
    fn event(&mut self, _: SurgeryEvent) -> ControlFlow<()> {
        self.seen += 1;
        match self.seen >= self.at {
            true => ControlFlow::Break(()),
            false => ControlFlow::Continue(()),
        }
    }
}

fn model_bytes(pair: &Pair) -> [Vec<u8>; 2] {
    [&pair.base, &pair.patched].map(|dir| fs::read(dir.join("model.safetensors")).unwrap())
}

#[test]
fn diff_cancelled_at_random_moments_leaves_no_output_and_the_checkpoints_intact() {
    let pair = square_pair(6, 160, 1);
    let out_dir = pair.dir.path().join("out");
    fs::create_dir(&out_dir).unwrap();
    let out = out_dir.join("patch.surgery");
    let args = diff_args(&pair, out.clone(), 8);
    let models = model_bytes(&pair);
    let started = Instant::now();
    run_diff(&args, &mut NullObserver, &AtomicBool::new(false)).unwrap();
    let full = started.elapsed();
    fs::remove_file(&out).unwrap();
    let mut rng = Rng::new(0xC0DE);
    let (mut cancelled, mut completed) = (0, 0);
    for i in 0..10 {
        let flag = AtomicBool::new(false);
        let delay = random_delay(&mut rng, i, full);
        let got = with_cancel_after(delay, &flag, || run_diff(&args, &mut NullObserver, &flag));
        match got {
            Ok(_) => {
                completed += 1;
                SurgeryReader::open(&out)
                    .unwrap_or_else(|e| panic!("run {i}: the finished patch is unreadable: {e}"));
                fs::remove_file(&out).unwrap();
            }
            Err(SurgeryError::Cancelled) => {
                cancelled += 1;
                assert_eq!(
                    file_names(&out_dir),
                    Vec::<String>::new(),
                    "run {i} (delay {delay:?}): output or temporary file left behind"
                );
            }
            Err(other) => panic!("run {i}: {other}"),
        }
        assert_eq!(model_bytes(&pair), models, "run {i}: a checkpoint changed");
    }
    assert!(
        cancelled > 0,
        "no run was cancelled (full run {full:?}, {completed} completed)"
    );
}

#[test]
fn diff_cancelled_at_every_progress_event_leaves_no_output_and_keeps_an_existing_file() {
    let pair = square_pair(4, 24, 2);
    let out_dir = pair.dir.path().join("out");
    fs::create_dir(&out_dir).unwrap();
    let out = out_dir.join("patch.surgery");
    let mut args = diff_args(&pair, out.clone(), 4);
    struct Count(usize);
    impl SurgeryObserver for Count {
        fn event(&mut self, _: SurgeryEvent) -> ControlFlow<()> {
            self.0 += 1;
            ControlFlow::Continue(())
        }
    }
    let mut counter = Count(0);
    run_diff(&args, &mut counter, &AtomicBool::new(false)).unwrap();
    let events = counter.0;
    assert!(events >= 4, "{events} events");
    fs::remove_file(&out).unwrap();
    args.overwrite = true;
    for existing in [false, true] {
        for at in 1..=events + 1 {
            if existing {
                fs::write(&out, b"OLD CONTENT").unwrap();
            }
            let mut obs = BreakAt { at, seen: 0 };
            let got = run_diff(&args, &mut obs, &AtomicBool::new(false));
            match got {
                Err(SurgeryError::Cancelled) => {
                    let names = file_names(&out_dir);
                    match existing {
                        true => assert_eq!(
                            (names, fs::read(&out).unwrap()),
                            (vec!["patch.surgery".to_owned()], b"OLD CONTENT".to_vec()),
                            "break at {at}"
                        ),
                        false => assert!(names.is_empty(), "break at {at}: {names:?}"),
                    }
                }
                Ok(_) => {
                    assert!(at >= events, "break at {at} of {events} was ignored");
                    SurgeryReader::open(&out).unwrap();
                    assert_eq!(file_names(&out_dir), ["patch.surgery"]);
                }
                Err(other) => panic!("break at {at}: {other}"),
            }
            let _ = fs::remove_file(&out);
        }
    }
    // The flag raised before the start: nothing at all.
    let raised = AtomicBool::new(true);
    assert!(matches!(
        run_diff(&args, &mut NullObserver, &raised),
        Err(SurgeryError::Cancelled)
    ));
    assert!(file_names(&out_dir).is_empty());
}

/// Model of `n` q_proj weights [dim, dim] and a rank-`r` adapter for all of them.
fn big_merge_fixture(n: usize, dim: usize, r: usize) -> Fx {
    let dir = TempDir::new().unwrap();
    let mut rng = Rng::new(0xB16);
    let paths: Vec<String> = (0..n)
        .map(|i| format!("model.layers.{i}.self_attn.q_proj"))
        .collect();
    let model = dir.path().join("model.safetensors");
    let weights: Vec<St> = paths
        .iter()
        .map(|p| {
            st(
                &format!("{p}.weight"),
                "F32",
                &[dim, dim],
                &gaussians(&mut rng, dim * dim, 1.0),
            )
        })
        .collect();
    write_st(&model, &weights);
    let adapter: Vec<St> = paths
        .iter()
        .flat_map(|p| lora_pair(p, (dim, dim, r), &mut rng))
        .collect();
    let lora = dir.path().join("adapter_model.safetensors");
    write_st(&lora, &adapter);
    fs::write(
        dir.path().join("adapter_config.json"),
        format!(r#"{{"r": {r}, "lora_alpha": {}}}"#, 2 * r),
    )
    .unwrap();
    Fx { dir, model, lora }
}

#[test]
fn merge_cancelled_at_random_moments_restores_the_model_byte_for_byte() {
    let fx = big_merge_fixture(8, 192, 8);
    let before = fs::read(&fx.model).unwrap();
    let started = Instant::now();
    merge_outcome(&fx, None, 1.0, false).0.unwrap();
    let full = started.elapsed();
    assert_ne!(
        fs::read(&fx.model).unwrap(),
        before,
        "the merge must change the model"
    );
    let mut rng = Rng::new(0xC0DF);
    let (mut cancelled, mut completed) = (0, 0);
    for i in 0..12 {
        fs::write(&fx.model, &before).unwrap();
        let req = merge_request(&fx, None, 1.0, false);
        let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
        let delay = random_delay(&mut rng, i, full);
        let got = with_cancel_after(delay, &req.cancel, || {
            run_merge(&req, &mut backup, &mut NullObserver)
        });
        match got {
            Ok(_) => completed += 1,
            Err(SurgeryError::Cancelled) => {
                cancelled += 1;
                assert!(
                    fs::read(&fx.model).unwrap() == before,
                    "run {i} (delay {delay:?}): the model was not rolled back"
                );
            }
            Err(other) => panic!("run {i}: {other}"),
        }
    }
    assert!(
        cancelled > 0,
        "no run was cancelled (full run {full:?}, {completed} completed)"
    );
}

#[test]
fn merge_stopped_by_the_observer_at_every_event_restores_the_model() {
    let fx = big_merge_fixture(5, 24, 2);
    let before = fs::read(&fx.model).unwrap();
    for at in 1..=6 {
        fs::write(&fx.model, &before).unwrap();
        let req = merge_request(&fx, None, 1.0, false);
        let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
        let got = run_merge(&req, &mut backup, &mut BreakAt { at, seen: 0 });
        match got {
            Err(SurgeryError::Cancelled) => {
                assert!(fs::read(&fx.model).unwrap() == before, "break at {at}")
            }
            Ok(_) => assert!(at >= 5, "break at {at} was ignored"),
            Err(other) => panic!("break at {at}: {other}"),
        }
    }
}

/// A `.surgery` with a rank-`r` patch for the `n` tensors of `big_merge_fixture`.
fn big_patch(fx: &Fx, n: usize, dim: usize, r: usize) -> PathBuf {
    let path = fx.dir.path().join("big.surgery");
    let mut spec = init(FactorDtype::F16, deflate(1));
    spec.rank = r;
    let mut writer = SurgeryWriter::create(&path, spec, true).unwrap();
    let mut rng = Rng::new(0x5A);
    for i in 0..n {
        let f = LowRankFactors {
            u: gaussians(&mut rng, dim * r, 0.5),
            s: (0..r).map(|t| 3.0 / (t + 1) as f32).collect(),
            v: gaussians(&mut rng, r * dim, 0.5),
        };
        let mut m = meta(
            &format!("model.layers.{i}.self_attn.q_proj.weight"),
            &[dim, dim],
        );
        m.target_dtype = "F32".into();
        writer.add_lowrank(m, &f).unwrap();
    }
    writer.finish().unwrap();
    path
}

fn apply_request(fx: &Fx, patch: &Path, verify_base: VerifyBase) -> SurgeryApplyRequest {
    SurgeryApplyRequest {
        model_shards: resolve_shards(&fx.model).unwrap(),
        patch: patch.to_path_buf(),
        scale: 1.0,
        verify_base,
        dry_run: false,
        cancel: Arc::new(AtomicBool::new(false)),
    }
}

#[test]
fn surgery_apply_cancelled_at_random_moments_restores_the_model_byte_for_byte() {
    let fx = big_merge_fixture(8, 192, 8);
    let patch = big_patch(&fx, 8, 192, 8);
    let before = fs::read(&fx.model).unwrap();
    let started = Instant::now();
    {
        let req = apply_request(&fx, &patch, VerifyBase::Warn);
        let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
        let report = run_apply(&req, &mut backup, &mut NullObserver).unwrap();
        assert_eq!(report.tensors.len(), 8);
        assert_eq!(
            report.warnings.len(),
            8,
            "every base hash differs from the fake one"
        );
    }
    let full = started.elapsed();
    assert_ne!(fs::read(&fx.model).unwrap(), before);
    let mut rng = Rng::new(0xC0E0);
    let (mut cancelled, mut completed) = (0, 0);
    for i in 0..12 {
        fs::write(&fx.model, &before).unwrap();
        let req = apply_request(&fx, &patch, VerifyBase::Warn);
        let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
        let delay = random_delay(&mut rng, i, full);
        let got = with_cancel_after(delay, &req.cancel, || {
            run_apply(&req, &mut backup, &mut NullObserver)
        });
        match got {
            Ok(_) => completed += 1,
            Err(SurgeryError::Cancelled) => {
                cancelled += 1;
                assert!(
                    fs::read(&fx.model).unwrap() == before,
                    "run {i} (delay {delay:?}): the model was not rolled back"
                );
            }
            Err(other) => panic!("run {i}: {other}"),
        }
    }
    assert!(
        cancelled > 0,
        "no run was cancelled (full run {full:?}, {completed} completed)"
    );
}

#[test]
fn surgery_apply_stopped_by_the_observer_and_strict_refusals_leave_the_model_alone() {
    let fx = big_merge_fixture(5, 24, 2);
    let patch = big_patch(&fx, 5, 24, 2);
    let before = fs::read(&fx.model).unwrap();
    // Hashing events come first (5), then one per written tensor (5).
    for at in 1..=11 {
        fs::write(&fx.model, &before).unwrap();
        let req = apply_request(&fx, &patch, VerifyBase::Warn);
        let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
        let got = run_apply(&req, &mut backup, &mut BreakAt { at, seen: 0 });
        match got {
            Err(SurgeryError::Cancelled) => {
                assert!(fs::read(&fx.model).unwrap() == before, "break at {at}")
            }
            Ok(_) => assert!(at >= 10, "break at {at} was ignored"),
            Err(other) => panic!("break at {at}: {other}"),
        }
    }
    fs::write(&fx.model, &before).unwrap();
    let req = apply_request(&fx, &patch, VerifyBase::Strict);
    let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
    let got = run_apply(&req, &mut backup, &mut NullObserver);
    assert!(matches!(got, Err(SurgeryError::Usage(_))), "{got:?}");
    assert!(fs::read(&fx.model).unwrap() == before);
    // A scale that overflows f32 once folded into the factors: refused or rolled back.
    for scale in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e300,
        -1e300,
        1e39,
        2e38,
    ] {
        let mut req = apply_request(&fx, &patch, VerifyBase::Off);
        req.scale = scale;
        let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
        let got = run_apply(&req, &mut backup, &mut NullObserver);
        assert!(
            matches!(got, Err(SurgeryError::Usage(_) | SurgeryError::Failed(_))),
            "scale {scale:e}: {got:?}"
        );
        assert!(
            fs::read(&fx.model).unwrap() == before,
            "scale {scale:e}: not rolled back"
        );
    }
}

#[test]
fn extract_cancelled_at_random_moments_leaves_no_adapter_and_no_config() {
    let pair = square_pair(6, 160, 3);
    let args = extract_args(&pair, 8, FactorDtype::F32);
    let out_dir = args.out.parent().unwrap().to_path_buf();
    let models = model_bytes(&pair);
    let started = Instant::now();
    run_extract_fresh(&args).unwrap();
    let full = started.elapsed();
    let mut again = args.clone();
    again.overwrite = true;
    let mut rng = Rng::new(0xC0E1);
    let (mut cancelled, mut completed) = (0, 0);
    for i in 0..10 {
        // The previous output (or none) must be exactly what is there afterwards.
        let _ = fs::remove_file(&args.out);
        let _ = fs::remove_file(out_dir.join("adapter_config.json"));
        let flag = AtomicBool::new(false);
        let delay = random_delay(&mut rng, i, full);
        let got = with_cancel_after(delay, &flag, || {
            run_extract(&again, &mut NullObserver, &flag)
        });
        match got {
            Ok(_) => completed += 1,
            Err(SurgeryError::Cancelled) => {
                cancelled += 1;
                assert_eq!(
                    file_names(&out_dir),
                    Vec::<String>::new(),
                    "run {i} (delay {delay:?}): partial output"
                );
            }
            Err(other) => panic!("run {i}: {other}"),
        }
        assert_eq!(model_bytes(&pair), models, "run {i}: a checkpoint changed");
    }
    assert!(
        cancelled > 0,
        "no run was cancelled (full run {full:?}, {completed} completed)"
    );
    // Cancelling over an existing adapter must not damage it.
    fs::write(&args.out, b"OLD ADAPTER").unwrap();
    let raised = AtomicBool::new(true);
    assert!(matches!(
        run_extract(&again, &mut NullObserver, &raised),
        Err(SurgeryError::Cancelled)
    ));
    assert_eq!(fs::read(&args.out).unwrap(), b"OLD ADAPTER");
}

// ------------------------------------------------------------------ 7. pipelines against references

fn tensor_bytes(file: &Path, name: &str) -> Vec<u8> {
    let bytes = fs::read(file).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    tensors.tensor(name).unwrap().data().to_vec()
}

fn decode_all(dtype: &str, bytes: &[u8]) -> Vec<f32> {
    let elem = if dtype == "F32" { 4 } else { 2 };
    bytes.chunks_exact(elem).map(|c| load1(dtype, c)).collect()
}

#[test]
fn merge_matches_a_plain_loop_for_every_dtype_combination_and_odd_shapes() {
    let mut rng = Rng::new(0x3E46E);
    let dtypes = ["BF16", "F16", "F32"];
    let mut combos: Vec<(&str, &str, &str)> = Vec::new();
    for m in dtypes {
        for a in dtypes {
            combos.extend(dtypes.map(|b| (m, a, b)));
        }
    }
    for (case, (model_dtype, a_dtype, b_dtype)) in combos.into_iter().enumerate() {
        let r = rng.between(1, 6);
        let (m1, n1, m2, n2) = (
            rng.between(1, 23),
            rng.between(1, 23),
            rng.between(1, 9),
            rng.between(1, 40),
        );
        let dir = TempDir::new().unwrap();
        let model = dir.path().join("model.safetensors");
        let q = st(
            &format!("{QP}.weight"),
            model_dtype,
            &[m1, n1],
            &gaussians(&mut rng, m1 * n1, 1.0),
        );
        let v = st(
            &format!("{VP}.weight"),
            model_dtype,
            &[m2, n2],
            &gaussians(&mut rng, m2 * n2, 1.0),
        );
        let norm = st("model.norm.weight", model_dtype, &[3], &[1.0, 2.0, 3.0]);
        write_st(
            &model,
            &[
                St {
                    name: q.name.clone(),
                    dtype: q.dtype,
                    shape: q.shape.clone(),
                    bytes: q.bytes.clone(),
                },
                St {
                    name: v.name.clone(),
                    dtype: v.dtype,
                    shape: v.shape.clone(),
                    bytes: v.bytes.clone(),
                },
                norm,
            ],
        );
        let mk = |path: &str, m: usize, n: usize, rng: &mut Rng| {
            let a = st(
                &format!("base_model.model.{path}.lora_A.weight"),
                a_dtype,
                &[r, n],
                &gaussians(rng, r * n, 0.5),
            );
            let b = st(
                &format!("base_model.model.{path}.lora_B.weight"),
                b_dtype,
                &[m, r],
                &gaussians(rng, m * r, 0.5),
            );
            (a, b)
        };
        let ((qa, qb), (va, vb)) = (mk(QP, m1, n1, &mut rng), mk(VP, m2, n2, &mut rng));
        let rslora = case % 3 == 0;
        let alpha = [3.0, 16.0, 0.5][case % 3];
        let scale = [1.0, 0.75, -1.5][(case / 3) % 3];
        let lora = dir.path().join("adapter_model.safetensors");
        let want_a: Vec<Vec<u8>> = [&qa, &qb, &va, &vb]
            .iter()
            .map(|t| t.bytes.clone())
            .collect();
        write_st(&lora, &[qa, qb, va, vb]);
        fs::write(
            dir.path().join("adapter_config.json"),
            format!(r#"{{"r": {r}, "lora_alpha": {alpha}, "use_rslora": {rslora}}}"#),
        )
        .unwrap();
        let fx = Fx { dir, model, lora };
        let (result, ..) = merge_outcome(&fx, None, scale, false);
        let report = result.unwrap_or_else(|e| panic!("case {case}: {e}"));
        let divisor = if rslora { (r as f64).sqrt() } else { r as f64 };
        let eff = (scale * alpha / divisor) as f32;
        assert_eq!(
            report.effective_scale,
            scale * alpha / divisor,
            "case {case}"
        );
        for (name, w, (a_bytes, b_bytes), (m, n)) in [
            (
                format!("{QP}.weight"),
                &q,
                (&want_a[0], &want_a[1]),
                (m1, n1),
            ),
            (
                format!("{VP}.weight"),
                &v,
                (&want_a[2], &want_a[3]),
                (m2, n2),
            ),
        ] {
            let a = decode_all(a_dtype, a_bytes);
            let u: Vec<f32> = decode_all(b_dtype, b_bytes)
                .iter()
                .map(|x| x * eff)
                .collect();
            let (want, stats) =
                reference_apply(model_dtype, &w.bytes, (m, n), lowrank_delta(&u, &a, n, r));
            let got = tensor_bytes(&fx.model, &name);
            assert!(got == want, "case {case} {model_dtype}/{a_dtype}/{b_dtype} {name} [{m}x{n}] r={r}: bytes differ from the reference");
            let reported = report.per_tensor.iter().find(|t| t.name == name).unwrap();
            assert_eq!(
                (reported.elements_modified, reported.max_abs_delta),
                (stats.elements_modified, stats.max_abs_delta),
                "case {case} {name}"
            );
        }
        assert_eq!(
            tensor_bytes(&fx.model, "model.norm.weight"),
            encode_vals(model_dtype, &[1.0, 2.0, 3.0]),
            "case {case}: untouched tensor changed"
        );
    }
}

#[test]
fn surgery_apply_matches_a_plain_loop_for_every_dtype_and_scale() {
    let mut rng = Rng::new(0x5E46E);
    let dtypes = ["BF16", "F16", "F32"];
    let quantize = |dtype: FactorDtype, x: f32| match dtype {
        FactorDtype::F16 => f16::from_f32(x).to_f32(),
        FactorDtype::Bf16 => bf16::from_f32(x).to_f32(),
        FactorDtype::F32 => x,
    };
    let combos: Vec<(&str, FactorDtype)> = dtypes
        .iter()
        .flat_map(|&m| [FactorDtype::F16, FactorDtype::Bf16, FactorDtype::F32].map(move |f| (m, f)))
        .collect();
    for (case, (model_dtype, factor_dtype)) in combos.into_iter().enumerate() {
        let dir = TempDir::new().unwrap();
        let model = dir.path().join("model.safetensors");
        let specs: [(&str, Vec<usize>); 4] = [
            ("a.weight", vec![rng.between(1, 20), rng.between(1, 20)]),
            ("b.weight", vec![rng.between(1, 9), rng.between(1, 40)]),
            ("c.bias", vec![5]),
            ("d.w3", vec![2, 3, 4]),
        ];
        let tensors: Vec<St> = specs
            .iter()
            .map(|(name, shape)| {
                st(
                    name,
                    model_dtype,
                    shape,
                    &gaussians(&mut rng, shape.iter().product(), 1.0),
                )
            })
            .collect();
        write_st(&model, &tensors);
        let patch = dir.path().join("p.surgery");
        let mut spec = init(factor_dtype, deflate(1));
        spec.rank = 6;
        let mut writer = SurgeryWriter::create(&patch, spec, false).unwrap();
        // (reference delta source per tensor)
        enum Delta {
            Low {
                u: Vec<f32>,
                s: Vec<f32>,
                v: Vec<f32>,
                r: usize,
            },
            Full(Vec<f32>),
        }
        let mut deltas = Vec::new();
        for (t, (name, shape)) in specs.iter().enumerate() {
            let mut m = meta(name, shape);
            m.target_dtype = model_dtype.to_owned();
            m.base_sha256 = sha256_hex(&tensors[t].bytes);
            match shape.len() {
                2 => {
                    // The rank may exceed both dimensions.
                    let r = rng.between(1, 6);
                    let f = LowRankFactors {
                        u: gaussians(&mut rng, shape[0] * r, 0.7),
                        s: (0..r).map(|_| rng.unit() as f32 * 3.0 + 0.1).collect(),
                        v: gaussians(&mut rng, r * shape[1], 0.7),
                    };
                    writer.add_lowrank(m, &f).unwrap();
                    let q = |x: &[f32]| {
                        x.iter()
                            .map(|&x| quantize(factor_dtype, x))
                            .collect::<Vec<_>>()
                    };
                    deltas.push(Delta::Low {
                        u: q(&f.u),
                        s: q(&f.s),
                        v: q(&f.v),
                        r,
                    });
                }
                _ => {
                    let delta = gaussians(&mut rng, shape.iter().product(), 2.0);
                    writer.add_full_delta(m, &delta).unwrap();
                    deltas.push(Delta::Full(delta));
                }
            }
        }
        writer.finish().unwrap();
        // Powers of two keep `scale * u * s` exact whatever the evaluation order.
        let scale = [1.0f64, 0.5, -2.0, 0.0, 4.0][case % 5];
        let fx = Fx {
            lora: patch.clone(),
            model: model.clone(),
            dir,
        };
        let before = fs::read(&model).unwrap();
        let mut req = apply_request(&fx, &patch, VerifyBase::Strict);
        req.scale = scale;
        let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
        let report = run_apply(&req, &mut backup, &mut NullObserver)
            .unwrap_or_else(|e| panic!("case {case}: {e}"));
        assert!(
            report.warnings.is_empty()
                && report.tensors.iter().all(|t| t.base_hash_ok == Some(true)),
            "case {case}"
        );
        let scale32 = scale as f32;
        for (t, ((name, shape), delta)) in specs.iter().zip(&deltas).enumerate() {
            let (rows, cols) = (
                shape[..shape.len() - 1].iter().product::<usize>(),
                shape[shape.len() - 1],
            );
            let (want, stats) = match delta {
                Delta::Low { u, s, v, r } => {
                    let scaled: Vec<f32> =
                        (0..rows * r).map(|i| u[i] * (s[i % r] * scale32)).collect();
                    reference_apply(
                        model_dtype,
                        &tensors[t].bytes,
                        (rows, cols),
                        lowrank_delta(&scaled, v, cols, *r),
                    )
                }
                Delta::Full(d) => {
                    reference_apply(model_dtype, &tensors[t].bytes, (rows, cols), |i, j| {
                        scale32 * d[i * cols + j]
                    })
                }
            };
            let got = tensor_bytes(&model, name);
            assert!(got == want, "case {case} {model_dtype}/{factor_dtype:?} scale {scale} {name} {shape:?}: differs from the reference");
            let reported = report.tensors.iter().find(|r| r.name == *name).unwrap();
            assert_eq!(
                (reported.elements_modified, reported.max_abs_delta),
                (stats.elements_modified, stats.max_abs_delta),
                "case {case} {name}"
            );
        }
        if scale == 0.0 {
            assert!(
                fs::read(&model).unwrap() == before,
                "case {case}: a zero scale must not change a byte"
            );
        }
    }
}

#[test]
fn diff_and_extract_outputs_are_bit_identical_with_1_2_and_8_threads() {
    let pair = square_pair(4, 80, 9);
    let out_dir = pair.dir.path().join("out");
    fs::create_dir(&out_dir).unwrap();
    let mut diff_runs = Vec::new();
    let mut extract_runs = Vec::new();
    for threads in [1usize, 2, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let out = out_dir.join(format!("p{threads}.surgery"));
        let args = diff_args(&pair, out.clone(), 6);
        pool.install(|| run_diff(&args, &mut NullObserver, &AtomicBool::new(false)))
            .unwrap();
        diff_runs.push(fs::read(&out).unwrap());
        let mut ex = extract_args(&pair, 6, FactorDtype::F32);
        fs::create_dir(out_dir.join(format!("t{threads}"))).unwrap();
        ex.out = out_dir
            .join(format!("t{threads}"))
            .join("adapter_model.safetensors");
        pool.install(|| run_extract(&ex, &mut NullObserver, &AtomicBool::new(false)))
            .unwrap();
        extract_runs.push(fs::read(&ex.out).unwrap());
    }
    assert!(
        diff_runs.windows(2).all(|w| w[0] == w[1]),
        ".surgery bytes depend on the thread count"
    );
    assert!(
        extract_runs.windows(2).all(|w| w[0] == w[1]),
        "adapter bytes depend on the thread count"
    );
}

#[test]
fn diff_falls_back_to_bf16_on_f16_overflow_and_reads_back() {
    let mut rng = Rng::new(0xDF);
    let base = [
        st(
            "model.layers.0.mlp.up_proj.weight",
            "F32",
            &[8, 8],
            &[0.0; 64],
        ),
        st(
            "model.layers.0.mlp.down_proj.weight",
            "F32",
            &[4, 4],
            &gaussians(&mut rng, 16, 1.0),
        ),
    ];
    let big = gaussians(&mut rng, 64, 1e5);
    let unchanged = St {
        name: base[1].name.clone(),
        dtype: base[1].dtype,
        shape: base[1].shape.clone(),
        bytes: base[1].bytes.clone(),
    };
    let patched = [
        st("model.layers.0.mlp.up_proj.weight", "F32", &[8, 8], &big),
        unchanged,
    ];
    let pair = checkpoints(&base, &patched);
    let out_dir = pair.dir.path().join("out");
    fs::create_dir(&out_dir).unwrap();
    let out = out_dir.join("p.surgery");
    let args = diff_args(&pair, out.clone(), 4);
    let report = run_diff(&args, &mut NullObserver, &AtomicBool::new(false)).unwrap();
    assert_eq!(report.factor_dtype, "bf16");
    assert!(
        report.warnings.iter().any(|w| w.contains("bf16")),
        "{:?}",
        report.warnings
    );
    assert_eq!((report.tensors.len(), report.skipped_identical), (1, 1));
    let mut reader = SurgeryReader::open(&out).unwrap();
    assert_eq!(reader.header().factor_dtype, FactorDtype::Bf16);
    let entry = reader.header().tensors[0].clone();
    let f = reader.read_lowrank(&entry).unwrap();
    assert!(
        f.s[0] > 65504.0,
        "the singular value that overflowed f16: {}",
        f.s[0]
    );
    assert_eq!(file_names(&out_dir), ["p.surgery"]);
}

fn tw(f: impl Fn(&mut DiffArgs) + 'static) -> Box<dyn Fn(&mut DiffArgs)> {
    Box::new(f)
}

#[test]
fn diff_hostile_checkpoints_and_options_are_typed_errors_with_no_output() {
    let mut rng = Rng::new(0xD1);
    let name = "model.layers.0.self_attn.q_proj.weight";
    let t2 = |rng: &mut Rng, vals: Option<Vec<f32>>| {
        st(
            name,
            "F32",
            &[6, 5],
            &vals.unwrap_or_else(|| gaussians(rng, 30, 1.0)),
        )
    };
    let with_nan = |rng: &mut Rng| {
        let mut v = gaussians(rng, 30, 1.0);
        v[29] = f32::NAN;
        v
    };
    let (nan_a, nan_b) = (with_nan(&mut rng), with_nan(&mut rng));
    let cases: Vec<(&str, Vec<St>, Vec<St>)> = vec![
        (
            "NaN in tuned (2D)",
            vec![t2(&mut rng, None)],
            vec![t2(&mut rng, Some(nan_a))],
        ),
        (
            "NaN in base (2D)",
            vec![t2(&mut rng, Some(nan_b))],
            vec![t2(&mut rng, None)],
        ),
        (
            "NaN in a 1D tensor",
            vec![st("model.norm.weight", "F32", &[4], &[1.0; 4])],
            vec![st(
                "model.norm.weight",
                "F32",
                &[4],
                &[1.0, f32::NAN, 1.0, 1.0],
            )],
        ),
        (
            "delta beyond f32",
            vec![st(name, "F32", &[2, 2], &[-3e38; 4])],
            vec![st(name, "F32", &[2, 2], &[3e38; 4])],
        ),
        (
            "shape mismatch",
            vec![t2(&mut rng, None)],
            vec![st(name, "F32", &[5, 6], &gaussians(&mut rng, 30, 1.0))],
        ),
        (
            "dtype mismatch",
            vec![t2(&mut rng, None)],
            vec![st(name, "BF16", &[6, 5], &gaussians(&mut rng, 30, 1.0))],
        ),
        (
            "tensor missing from tuned",
            vec![
                t2(&mut rng, None),
                st("extra.weight", "F32", &[2, 2], &[1.0; 4]),
            ],
            vec![t2(&mut rng, None)],
        ),
        (
            "extra tensor in tuned",
            vec![t2(&mut rng, None)],
            vec![
                t2(&mut rng, None),
                st("extra.weight", "F32", &[2, 2], &[1.0; 4]),
            ],
        ),
        (
            "unsupported dtype that differs",
            vec![St {
                name: name.into(),
                dtype: "I8",
                shape: vec![3, 3],
                bytes: vec![1; 9],
            }],
            vec![St {
                name: name.into(),
                dtype: "I8",
                shape: vec![3, 3],
                bytes: vec![2; 9],
            }],
        ),
        (
            "scalar tensor that differs",
            vec![st("logit_scale", "F32", &[], &[1.0])],
            vec![st("logit_scale", "F32", &[], &[2.0])],
        ),
    ];
    for (label, base, patched) in cases {
        let pair = checkpoints(&base, &patched);
        let out_dir = pair.dir.path().join("out");
        fs::create_dir(&out_dir).unwrap();
        let args = diff_args(&pair, out_dir.join("p.surgery"), 4);
        let got = run_diff(&args, &mut NullObserver, &AtomicBool::new(false));
        assert!(
            matches!(got, Err(SurgeryError::Usage(_))),
            "{label}: {got:?}"
        );
        assert!(
            file_names(&out_dir).is_empty(),
            "{label}: output left behind"
        );
    }
    // Option extremes on a good pair.
    let pair = checkpoints(&[t2(&mut rng, None)], &[t2(&mut rng, None)]);
    let out_dir = pair.dir.path().join("out");
    fs::create_dir(&out_dir).unwrap();
    let out = out_dir.join("p.surgery");
    type Tweak = Box<dyn Fn(&mut DiffArgs)>;
    let bad: Vec<(&str, Tweak)> = vec![
        ("rank 0", tw(|a| a.rank = 0)),
        ("rank 65537", tw(|a| a.rank = 65537)),
        ("rank usize::MAX", tw(|a| a.rank = usize::MAX)),
        (
            "max_rel_error NaN",
            tw(|a| a.max_rel_error = Some(f64::NAN)),
        ),
        (
            "max_rel_error negative",
            tw(|a| a.max_rel_error = Some(-0.5)),
        ),
        (
            "max_rel_error infinite",
            tw(|a| a.max_rel_error = Some(f64::INFINITY)),
        ),
        ("include regex", tw(|a| a.include = Some("(".into()))),
        ("exclude regex", tw(|a| a.exclude = Some("[".into()))),
    ];
    for (label, tweak) in bad {
        let mut args = diff_args(&pair, out.clone(), 4);
        tweak(&mut args);
        let got = run_diff(&args, &mut NullObserver, &AtomicBool::new(false));
        assert!(
            matches!(got, Err(SurgeryError::Usage(_))),
            "{label}: {got:?}"
        );
        assert!(file_names(&out_dir).is_empty(), "{label}");
    }
    // Extreme but valid: huge oversample and power_iters 0, and the high-error policies.
    for (label, tweak) in [
        ("oversample usize::MAX", tw(|a| a.oversample = usize::MAX)),
        ("no power iterations", tw(|a| a.power_iters = 0)),
        (
            "rank 1 and error limit 0 (warn)",
            tw(|a| {
                a.rank = 1;
                a.max_rel_error = Some(0.0);
            }),
        ),
        (
            "rank 1 and error limit 0 (full)",
            tw(|a| {
                a.rank = 1;
                a.max_rel_error = Some(0.0);
                a.on_high_error = OnHighError::Full;
            }),
        ),
        (
            "rank 1 and error limit 0 (skip)",
            tw(|a| {
                a.rank = 1;
                a.max_rel_error = Some(0.0);
                a.on_high_error = OnHighError::Skip;
            }),
        ),
        ("exclude everything", tw(|a| a.exclude = Some(".".into()))),
        ("include nothing", tw(|a| a.include = Some("^zzz$".into()))),
    ] {
        let mut args = diff_args(&pair, out.clone(), 4);
        args.overwrite = true;
        tweak(&mut args);
        let report = run_diff(&args, &mut NullObserver, &AtomicBool::new(false))
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        let mut reader = SurgeryReader::open(&out).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(
            reader.header().tensors.len(),
            report.tensors.len(),
            "{label}"
        );
        for entry in reader.header().tensors.clone() {
            match entry.rank {
                0 => drop(reader.read_full_delta(&entry).unwrap()),
                _ => drop(reader.read_lowrank(&entry).unwrap()),
            }
        }
        let kinds: Vec<&str> = report.tensors.iter().map(|t| t.kind.as_str()).collect();
        match label {
            l if l.ends_with("(full)") => assert_eq!(kinds, ["full_delta"], "{label}"),
            l if l.ends_with("(skip)") || l == "exclude everything" || l == "include nothing" => {
                assert!(kinds.is_empty(), "{label}")
            }
            _ => assert_eq!(kinds, ["lowrank"], "{label}"),
        }
        fs::remove_file(&out).unwrap();
    }
    // An existing --out without --overwrite is refused and untouched.
    fs::write(&out, b"KEEP").unwrap();
    let got = run_diff(
        &diff_args(&pair, out.clone(), 4),
        &mut NullObserver,
        &AtomicBool::new(false),
    );
    assert!(matches!(got, Err(SurgeryError::Usage(_))) && fs::read(&out).unwrap() == b"KEEP");
}

#[test]
fn merge_survives_random_corruption_of_the_adapter_and_of_the_model_files() {
    /// A dry run never touches the guard.
    struct NoGuard;
    impl RollbackHandle for NoGuard {
        fn rollback(&mut self) -> weight_surgeon::patch::Result<()> {
            Ok(())
        }
        fn commit(&mut self) {}
    }
    let mut rng = Rng::new(0xC0AA);
    let fx = fixture(&good_adapter(&mut rng), Some(&std_config()));
    let adapter = fs::read(&fx.lora).unwrap();
    let model = fs::read(&fx.model).unwrap();
    let req = merge_request(&fx, None, 1.0, true);
    let ((), growth) = peak_commit_growth(|| {
        for which in ["adapter", "model"] {
            let orig = if which == "adapter" { &adapter } else { &model };
            for i in 0..300 {
                let mut bytes = orig.clone();
                match rng.below(5) {
                    0 => {
                        for _ in 0..1 + rng.below(3) {
                            let at = rng.below(256.min(bytes.len()));
                            bytes[at] = rng.next_u64() as u8;
                        }
                    }
                    1 => {
                        let value = *rng.pick(&[
                            0u64,
                            1,
                            7,
                            1 << 20,
                            u64::MAX,
                            bytes.len() as u64,
                            bytes.len() as u64 - 8,
                            u64::MAX / 2,
                        ]);
                        bytes[..8].copy_from_slice(&value.to_le_bytes());
                    }
                    2 => bytes.truncate(rng.below(bytes.len())),
                    3 => {
                        let from = rng.below(bytes.len());
                        let to = (from + rng.between(1, 64)).min(bytes.len());
                        bytes[from..to].fill(*rng.pick(&[0u8, 0xFF, b'9', b'"']));
                    }
                    _ => {
                        // A digit in the JSON header replaced by another: shapes and offsets.
                        let header = 8 + u64::from_le_bytes(orig[..8].try_into().unwrap()) as usize;
                        let digits: Vec<usize> = (8..header.min(bytes.len()))
                            .filter(|&k| bytes[k].is_ascii_digit())
                            .collect();
                        if let Some(&k) = digits.get(rng.below(digits.len().max(1))) {
                            bytes[k] = b'0' + rng.below(10) as u8;
                        }
                    }
                }
                let target = if which == "adapter" {
                    &fx.lora
                } else {
                    &fx.model
                };
                fs::write(target, &bytes).unwrap();
                let _ = run_merge(&req, &mut NoGuard, &mut NullObserver).map_err(|e| (i, e.kind()));
                fs::write(target, orig).unwrap();
            }
        }
    });
    assert!(growth < MEM_LIMIT, "commit grew by {} MiB", growth >> 20);
}
