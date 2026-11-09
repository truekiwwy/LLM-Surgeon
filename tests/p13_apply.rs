//! `.surgery` apply: kernel semantics through `run_apply` and the CLI entry (`surgeon apply
//! --patch x.surgery`) on real SafeTensors files in temporary directories.
//!
//! Fixture values are multiples of 1/8 with few bits, so every f32 sum is exact and the
//! f64 reference (rounded once, to nearest even, with `half::*::from_f64`) must match the
//! kernel bit for bit.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use half::{bf16, f16};
use serde_json::Value;
use weight_surgeon::app::{self, exit, AppDeps};
use weight_surgeon::eval::{CommandOutput, CommandRunner, EvalError};
use weight_surgeon::hardware::{GpuInfo, HardwareProbe, MemoryInfo};
use weight_surgeon::patch::{BackupSet, RollbackHandle};
use weight_surgeon::surgery::apply::{run_apply, SurgeryApplyReport, SurgeryApplyRequest};
use weight_surgeon::surgery::cli_args::VerifyBase;
use weight_surgeon::surgery::error::SurgeryError;
use weight_surgeon::surgery::events::{SurgeryEvent, SurgeryObserver};
use weight_surgeon::surgery::format::{
    sha256_hex, Compression, CompressionAlgo, FactorDtype, LowRankFactors,
};
use weight_surgeon::surgery::target::resolve_shards;
use weight_surgeon::surgery::writer::{HeaderInit, SurgeryWriter, TensorMeta};

// ------------------------------------------------------------------ fixtures

#[derive(Clone)]
struct T {
    name: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    vals: Vec<f32>,
}

impl T {
    fn new(name: &'static str, dtype: &'static str, shape: &[usize], seed: u64) -> Self {
        let n = shape.iter().product();
        Self {
            name,
            dtype,
            shape: shape.to_vec(),
            vals: eighths(seed, n),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        self.vals
            .iter()
            .flat_map(|&x| match self.dtype {
                "BF16" => bf16::from_f32(x).to_le_bytes().to_vec(),
                "F16" => f16::from_f32(x).to_le_bytes().to_vec(),
                _ => x.to_le_bytes().to_vec(),
            })
            .collect()
    }

    /// Stored values (the dtype rounding of `vals`; exact for multiples of 1/8 here).
    fn stored(&self) -> Vec<f64> {
        decode(self.dtype, &self.bytes())
    }
}

/// Deterministic multiples of 1/8 in [-1, 1].
fn eighths(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 33) % 17) as f32 / 8.0 - 1.0
        })
        .collect()
}

fn decode(dtype: &str, bytes: &[u8]) -> Vec<f64> {
    match dtype {
        "BF16" => bytes
            .chunks(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f64())
            .collect(),
        "F16" => bytes
            .chunks(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f64())
            .collect(),
        _ => bytes
            .chunks(4)
            .map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            .collect(),
    }
}

fn encode_f64(dtype: &str, vals: &[f64]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&x| match dtype {
            "BF16" => bf16::from_f64(x).to_le_bytes().to_vec(),
            "F16" => f16::from_f64(x).to_le_bytes().to_vec(),
            _ => (x as f32).to_le_bytes().to_vec(),
        })
        .collect()
}

fn write_st(path: &Path, tensors: &[&T]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for t in tensors {
        let start = data.len();
        data.extend_from_slice(&t.bytes());
        header.insert(
            t.name.into(),
            serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = serde_json::to_string(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

fn read_tensor(path: &Path, name: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    st.tensor(name).unwrap().data().to_vec()
}

/// Factors of one low-rank tensor: `U` [m x r], `S` [r], `V` [r x n].
struct Factors {
    u: Vec<f32>,
    s: Vec<f32>,
    v: Vec<f32>,
}

fn factors(m: usize, n: usize, r: usize, seed: u64) -> Factors {
    Factors {
        u: eighths(seed, m * r),
        s: (0..r).map(|k| [0.5, 1.0, 2.0][k % 3]).collect(),
        v: eighths(seed + 100, r * n),
    }
}

/// f64 reference of `W + scale * U diag(S) V`, rounded once to the tensor dtype.
fn reference(t: &T, f: &Factors, scale: f64) -> Vec<u8> {
    let (m, n) = (t.shape[0], t.shape[1]);
    let r = f.s.len();
    let old = t.stored();
    let new: Vec<f64> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            let delta: f64 = (0..r)
                .map(|k| f64::from(f.u[i * r + k]) * f64::from(f.s[k]) * f64::from(f.v[k * n + j]))
                .sum();
            old[idx] + scale * delta
        })
        .collect();
    encode_f64(t.dtype, &new)
}

fn writer(path: &Path, rank: usize) -> SurgeryWriter {
    SurgeryWriter::create(
        path,
        HeaderInit {
            created_with: "test".into(),
            base_name: "base".into(),
            base_tensor_count: 0,
            tuned_name: "tuned".into(),
            rank,
            factor_dtype: FactorDtype::F32,
            compression: Compression {
                algo: CompressionAlgo::Deflate,
                level: 1,
            },
            timestamp: None,
        },
        true,
    )
    .unwrap()
}

fn meta(t: &T) -> TensorMeta {
    TensorMeta {
        name: t.name.into(),
        shape: t.shape.clone(),
        target_dtype: t.dtype.into(),
        base_sha256: sha256_hex(&t.bytes()),
        rel_error: 0.01,
        energy: 0.99,
    }
}

fn add_lowrank(w: &mut SurgeryWriter, t: &T, f: &Factors) {
    w.add_lowrank(
        meta(t),
        &LowRankFactors {
            u: f.u.clone(),
            s: f.s.clone(),
            v: f.v.clone(),
        },
    )
    .unwrap();
}

/// Observer that records every event and optionally stops after `break_after` events.
#[derive(Default)]
struct Recorder {
    events: Vec<SurgeryEvent>,
    break_after: Option<usize>,
}

impl SurgeryObserver for Recorder {
    fn event(&mut self, ev: SurgeryEvent) -> ControlFlow<()> {
        self.events.push(ev);
        match self.break_after {
            Some(n) if self.events.len() >= n => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        }
    }
}

/// Counts calls so tests can tell "nothing was written" from "written then undone".
#[derive(Default)]
struct CountingGuard {
    rollbacks: usize,
    commits: usize,
}

impl RollbackHandle for CountingGuard {
    fn rollback(&mut self) -> weight_surgeon::patch::Result<()> {
        self.rollbacks += 1;
        Ok(())
    }
    fn commit(&mut self) {
        self.commits += 1;
    }
}

fn request(model: &Path, patch: &Path) -> SurgeryApplyRequest {
    SurgeryApplyRequest {
        model_shards: resolve_shards(model).unwrap(),
        patch: patch.to_path_buf(),
        scale: 1.0,
        verify_base: VerifyBase::Strict,
        dry_run: false,
        cancel: Arc::new(AtomicBool::new(false)),
    }
}

fn apply(req: &SurgeryApplyRequest) -> (Result<SurgeryApplyReport, SurgeryError>, CountingGuard) {
    let mut guard = CountingGuard::default();
    let result = run_apply(req, &mut guard, &mut Recorder::default());
    (result, guard)
}

fn single_model(dir: &Path, tensors: &[&T]) -> PathBuf {
    let path = dir.join("model.safetensors");
    write_st(&path, tensors);
    path
}

// ------------------------------------------------------------------ kernel through run_apply

#[test]
fn lowrank_matches_f64_reference_for_every_dtype_and_scale() {
    for dtype in ["BF16", "F16", "F32"] {
        for scale in [1.0, 0.5, -2.0, 0.0] {
            let dir = tempfile::tempdir().unwrap();
            let t = T::new("w", dtype, &[19, 13], 7);
            let f = factors(19, 13, 3, 11);
            let model = single_model(dir.path(), &[&t]);
            let patch = dir.path().join("p.surgery");
            let mut w = writer(&patch, 3);
            add_lowrank(&mut w, &t, &f);
            w.finish().unwrap();

            let mut req = request(&model, &patch);
            req.scale = scale;
            let (report, guard) = apply(&req);
            let report = report.unwrap();
            assert_eq!((guard.commits, guard.rollbacks), (1, 0));

            let expected = reference(&t, &f, scale);
            assert_eq!(read_tensor(&model, "w"), expected, "{dtype} scale {scale}");
            let changed = expected
                .chunks(if dtype == "F32" { 4 } else { 2 })
                .zip(t.bytes().chunks(if dtype == "F32" { 4 } else { 2 }))
                .filter(|(a, b)| a != b)
                .count() as u64;
            assert_eq!(report.tensors[0].elements_modified, changed);
            assert_eq!(report.tensors[0].base_hash_ok, Some(true));
            assert_eq!(report.tensors[0].kind, "lowrank");
            assert_eq!(report.tensors[0].rank, 3);
        }
    }
}

#[test]
fn bytes_are_identical_with_1_2_and_8_threads() {
    let t = T::new("w", "BF16", &[200, 90], 5);
    let f = factors(200, 90, 4, 6);
    let dir = tempfile::tempdir().unwrap();
    let patch = dir.path().join("p.surgery");
    let mut w = writer(&patch, 4);
    add_lowrank(&mut w, &t, &f);
    w.finish().unwrap();

    let mut results = Vec::new();
    for threads in [1, 2, 8] {
        let sub = dir.path().join(format!("m{threads}"));
        std::fs::create_dir(&sub).unwrap();
        let model = single_model(&sub, &[&t]);
        let req = request(&model, &patch);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| apply(&req).0.unwrap());
        results.push(std::fs::read(&model).unwrap());
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0], results[2]);
}

#[test]
fn one_dimensional_full_delta_is_applied_and_scaled() {
    let dir = tempfile::tempdir().unwrap();
    let norm = T::new("norm.weight", "BF16", &[16], 3);
    let delta = eighths(4, 16);
    let model = single_model(dir.path(), &[&norm]);
    let patch = dir.path().join("p.surgery");
    let mut w = writer(&patch, 1);
    w.add_full_delta(meta(&norm), &delta).unwrap();
    w.finish().unwrap();

    let mut req = request(&model, &patch);
    req.scale = 0.5;
    let report = apply(&req).0.unwrap();
    let expected: Vec<f64> = norm
        .stored()
        .iter()
        .zip(&delta)
        .map(|(w, d)| w + 0.5 * f64::from(*d))
        .collect();
    assert_eq!(
        read_tensor(&model, "norm.weight"),
        encode_f64("BF16", &expected)
    );
    assert_eq!(report.tensors[0].kind, "full_delta");
}

// ------------------------------------------------------------------ base hash

fn hash_fixture(dir: &Path) -> (PathBuf, PathBuf, T) {
    let t = T::new("w", "F32", &[4, 5], 1);
    let patch = dir.join("p.surgery");
    let mut w = writer(&patch, 2);
    add_lowrank(&mut w, &t, &factors(4, 5, 2, 2));
    w.finish().unwrap();
    // The model is NOT the base the patch was computed against.
    let other = T::new("w", "F32", &[4, 5], 99);
    (single_model(dir, &[&other]), patch, other)
}

#[test]
fn verify_base_strict_refuses_before_any_write() {
    let dir = tempfile::tempdir().unwrap();
    let (model, patch, other) = hash_fixture(dir.path());
    let mut req = request(&model, &patch);
    req.verify_base = VerifyBase::Strict;
    let (result, guard) = apply(&req);
    let err = result.unwrap_err().to_string();
    assert!(err.contains("w") && err.contains("base du patch"), "{err}");
    assert_eq!(read_tensor(&model, "w"), other.bytes());
    assert_eq!((guard.commits, guard.rollbacks), (0, 0));
}

#[test]
fn verify_base_warn_applies_with_a_warning_and_off_skips_hashing() {
    let dir = tempfile::tempdir().unwrap();
    let (model, patch, other) = hash_fixture(dir.path());
    let before = other.bytes();

    let mut req = request(&model, &patch);
    req.verify_base = VerifyBase::Warn;
    let mut obs = Recorder::default();
    let report = run_apply(&req, &mut CountingGuard::default(), &mut obs).unwrap();
    assert_eq!(report.warnings.len(), 1);
    assert_eq!(report.tensors[0].base_hash_ok, Some(false));
    assert!(obs
        .events
        .iter()
        .any(|e| matches!(e, SurgeryEvent::Hashing { .. })));
    assert_ne!(read_tensor(&model, "w"), before);

    write_st(&model, &[&other]);
    req.verify_base = VerifyBase::Off;
    let mut obs = Recorder::default();
    let report = run_apply(&req, &mut CountingGuard::default(), &mut obs).unwrap();
    assert!(report.warnings.is_empty());
    assert_eq!(report.tensors[0].base_hash_ok, None);
    assert!(!obs
        .events
        .iter()
        .any(|e| matches!(e, SurgeryEvent::Hashing { .. })));
}

// ------------------------------------------------------------------ rollback and cancellation

/// Two tensors: the first applies fine, the second overflows f16 (60000 + 20000).
fn overflow_fixture(dir: &Path) -> (PathBuf, PathBuf) {
    let good = T::new("a", "F16", &[4, 4], 1);
    let mut hot = T::new("b", "F16", &[2, 2], 2);
    hot.vals = vec![60000.0; 4];
    let model = single_model(dir, &[&good, &hot]);
    let patch = dir.join("p.surgery");
    let mut w = writer(&patch, 1);
    add_lowrank(&mut w, &good, &factors(4, 4, 1, 3));
    let big = Factors {
        u: vec![1.0, 1.0],
        s: vec![1.0],
        v: vec![20000.0, 20000.0],
    };
    add_lowrank(&mut w, &hot, &big);
    w.finish().unwrap();
    (model, patch)
}

#[test]
fn nan_or_inf_rolls_back_to_identical_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let (model, patch) = overflow_fixture(dir.path());
    let before = std::fs::read(&model).unwrap();
    let req = request(&model, &patch);
    let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
    let err = run_apply(&req, &mut backup, &mut Recorder::default()).unwrap_err();
    assert!(err.to_string().contains("NaN"), "{err}");
    assert!(matches!(err, SurgeryError::Failed(_)));
    assert_eq!(std::fs::read(&model).unwrap(), before);
}

#[test]
fn cancellation_between_tensors_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let (model, patch) = overflow_fixture(dir.path());
    let before = std::fs::read(&model).unwrap();
    let mut req = request(&model, &patch);
    req.verify_base = VerifyBase::Off;
    let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
    let mut obs = Recorder {
        break_after: Some(1),
        ..Default::default()
    };
    let err = run_apply(&req, &mut backup, &mut obs).unwrap_err();
    assert!(matches!(err, SurgeryError::Cancelled), "{err}");
    assert_eq!(std::fs::read(&model).unwrap(), before);
}

#[test]
fn cancel_flag_raised_before_start_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let (model, patch) = overflow_fixture(dir.path());
    let before = std::fs::read(&model).unwrap();
    let mut req = request(&model, &patch);
    req.verify_base = VerifyBase::Off;
    req.cancel.store(true, Ordering::SeqCst);
    let (result, _) = apply(&req);
    assert!(matches!(result, Err(SurgeryError::Cancelled)));
    assert_eq!(std::fs::read(&model).unwrap(), before);
}

// ------------------------------------------------------------------ consistency and dry-run

#[test]
fn inconsistent_tensors_are_listed_and_nothing_is_written() {
    let dir = tempfile::tempdir().unwrap();
    let ok = T::new("ok", "BF16", &[4, 4], 1);
    let wrong_shape = T::new("shape", "BF16", &[4, 4], 2);
    let wrong_dtype = T::new("dtype", "BF16", &[4, 4], 3);
    let model = single_model(
        dir.path(),
        &[
            &ok,
            &T::new("shape", "BF16", &[4, 5], 2),
            &T::new("dtype", "F32", &[4, 4], 3),
        ],
    );
    let ghost = T::new("ghost", "BF16", &[4, 4], 4);
    let patch = dir.path().join("p.surgery");
    let mut w = writer(&patch, 1);
    for t in [&ok, &wrong_shape, &wrong_dtype, &ghost] {
        add_lowrank(&mut w, t, &factors(4, 4, 1, 5));
    }
    w.finish().unwrap();

    let before = std::fs::read(&model).unwrap();
    let (result, guard) = apply(&request(&model, &patch));
    let err = result.unwrap_err();
    assert!(matches!(err, SurgeryError::Usage(_)), "{err}");
    let msg = err.to_string();
    for needle in ["3 tenseur", "shape", "dtype", "ghost", "absent"] {
        assert!(msg.contains(needle), "{needle} missing from: {msg}");
    }
    assert!(!msg.contains("ok :"));
    assert_eq!(std::fs::read(&model).unwrap(), before);
    assert_eq!((guard.commits, guard.rollbacks), (0, 0));
}

#[test]
fn non_finite_scale_is_a_usage_error() {
    let dir = tempfile::tempdir().unwrap();
    let t = T::new("w", "F32", &[2, 2], 1);
    let model = single_model(dir.path(), &[&t]);
    let patch = dir.path().join("p.surgery");
    let mut w = writer(&patch, 1);
    add_lowrank(&mut w, &t, &factors(2, 2, 1, 1));
    w.finish().unwrap();
    let mut req = request(&model, &patch);
    req.scale = f64::NAN;
    assert!(matches!(apply(&req).0, Err(SurgeryError::Usage(_))));
}

#[test]
fn dry_run_previews_and_never_writes() {
    let dir = tempfile::tempdir().unwrap();
    let t = T::new("w", "BF16", &[6, 6], 1);
    let norm = T::new("n", "BF16", &[6], 2);
    let model = single_model(dir.path(), &[&t, &norm]);
    let patch = dir.path().join("p.surgery");
    let mut w = writer(&patch, 2);
    add_lowrank(&mut w, &t, &factors(6, 6, 2, 3));
    w.add_full_delta(meta(&norm), &eighths(9, 6)).unwrap();
    w.finish().unwrap();

    let before = std::fs::read(&model).unwrap();
    let mut req = request(&model, &patch);
    req.dry_run = true;
    let mut obs = Recorder::default();
    let mut guard = CountingGuard::default();
    let report = run_apply(&req, &mut guard, &mut obs).unwrap();
    assert!(report.dry_run);
    assert_eq!(report.tensors.len(), 2);
    assert!(report.tensors.iter().all(|t| t.elements_modified == 0));
    let previews = obs
        .events
        .iter()
        .filter(|e| matches!(e, SurgeryEvent::Preview { .. }))
        .count();
    assert_eq!(previews, 2);
    assert_eq!((guard.commits, guard.rollbacks), (0, 0));
    assert_eq!(std::fs::read(&model).unwrap(), before);
}

// ------------------------------------------------------------------ sharded model

#[test]
fn sharded_model_patches_tensors_in_every_shard() {
    let dir = tempfile::tempdir().unwrap();
    let a = T::new("a", "BF16", &[8, 8], 1);
    let b = T::new("b", "F16", &[8, 4], 2);
    write_st(&dir.path().join("model-00001.safetensors"), &[&a]);
    write_st(&dir.path().join("model-00002.safetensors"), &[&b]);
    let patch = dir.path().join("p.surgery");
    let (fa, fb) = (factors(8, 8, 2, 3), factors(8, 4, 2, 4));
    let mut w = writer(&patch, 2);
    add_lowrank(&mut w, &a, &fa);
    add_lowrank(&mut w, &b, &fb);
    w.finish().unwrap();

    let req = request(dir.path(), &patch);
    assert_eq!(req.model_shards.len(), 2);
    let (result, guard) = apply(&req);
    result.unwrap();
    assert_eq!(guard.commits, 1);
    assert_eq!(
        read_tensor(&dir.path().join("model-00001.safetensors"), "a"),
        reference(&a, &fa, 1.0)
    );
    assert_eq!(
        read_tensor(&dir.path().join("model-00002.safetensors"), "b"),
        reference(&b, &fb, 1.0)
    );
}

// ------------------------------------------------------------------ CLI (app::run_with)

struct FakeProbe;

impl HardwareProbe for FakeProbe {
    fn system_ram(&self) -> weight_surgeon::hardware::Result<MemoryInfo> {
        Ok(MemoryInfo {
            total_bytes: 64 << 30,
            available_bytes: 64 << 30,
        })
    }
    fn gpus(&self) -> weight_surgeon::hardware::Result<Vec<GpuInfo>> {
        Ok(vec![])
    }
}

struct NoRunner;

impl CommandRunner for NoRunner {
    fn run(
        &self,
        _: &Path,
        _: &[String],
        _: Option<&Path>,
    ) -> weight_surgeon::eval::Result<CommandOutput> {
        Err(EvalError::Generator("unused".into()))
    }
}

fn deps() -> AppDeps {
    AppDeps {
        probe: Box::new(FakeProbe),
        runner: Box::new(NoRunner),
        generator_factory: Box::new(|_, _| Err(EvalError::Generator("unused".into()))),
        interrupted: Arc::new(AtomicBool::new(false)),
        ui: weight_surgeon::ui::UiConfig {
            out: weight_surgeon::ui::ColorMode::Plain,
            err: weight_surgeon::ui::ColorMode::Plain,
            err_is_tty: false,
        },
    }
}

fn run_json(deps: &AppDeps, args: &[&str]) -> (i32, Value) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let argv = ["weight-surgeon", "--json"]
        .into_iter()
        .chain(args.iter().copied());
    let code = match app::parse_cli(argv, &mut out, &mut err) {
        Ok(cli) => app::run_with(cli, deps, &mut out, &mut err),
        Err(code) => code,
    };
    let out = String::from_utf8(out).unwrap();
    let value: Value = serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"));
    assert_eq!(value["exit_code"], code);
    (code, value)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

fn backups_in(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| Some(e.ok()?.path()))
        .filter(|p| p.to_string_lossy().contains(".bak-"))
        .collect()
}

struct Cli {
    dir: tempfile::TempDir,
    model: PathBuf,
    patch: PathBuf,
    base: T,
    f: Factors,
}

fn cli_fixture() -> Cli {
    let dir = tempfile::tempdir().unwrap();
    let base = T::new("w", "BF16", &[12, 10], 21);
    let f = factors(12, 10, 2, 22);
    let model = single_model(dir.path(), &[&base]);
    let patch = dir.path().join("x.surgery");
    let mut w = writer(&patch, 2);
    add_lowrank(&mut w, &base, &f);
    w.finish().unwrap();
    Cli {
        dir,
        model,
        patch,
        base,
        f,
    }
}

#[test]
fn cli_applies_a_surgery_file_with_backup_and_scale() {
    let c = cli_fixture();
    let (code, v) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&c.model),
            "--patch",
            s(&c.patch),
            "--scale",
            "0.5",
            "--verify-base",
            "strict",
        ],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["command"], "apply");
    assert_eq!(v["data"]["surgery"]["tensors"][0]["name"], "w");
    assert_eq!(v["data"]["surgery"]["scale"], 0.5);
    let backups = backups_in(c.dir.path());
    assert_eq!(backups.len(), 1);
    assert_eq!(v["data"]["backups"].as_array().unwrap().len(), 1);
    assert_eq!(read_tensor(&backups[0], "w"), c.base.bytes());
    assert_eq!(read_tensor(&c.model, "w"), reference(&c.base, &c.f, 0.5));
}

#[test]
fn cli_no_backup_and_backup_dir_are_honoured() {
    let c = cli_fixture();
    let (code, v) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&c.model),
            "--patch",
            s(&c.patch),
            "--no-backup",
        ],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert!(backups_in(c.dir.path()).is_empty());
    assert!(v["data"]["backups"].as_array().unwrap().is_empty());

    let c = cli_fixture();
    let elsewhere = tempfile::tempdir().unwrap();
    let (code, _) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&c.model),
            "--patch",
            s(&c.patch),
            "--backup-dir",
            s(elsewhere.path()),
        ],
    );
    assert_eq!(code, exit::OK);
    assert!(backups_in(c.dir.path()).is_empty());
    assert_eq!(backups_in(elsewhere.path()).len(), 1);
}

#[test]
fn cli_dry_run_changes_nothing_and_creates_no_backup() {
    let c = cli_fixture();
    let before = std::fs::read(&c.model).unwrap();
    let (code, v) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&c.model),
            "--patch",
            s(&c.patch),
            "--dry-run",
        ],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["data"]["surgery"]["dry_run"], true);
    assert_eq!(std::fs::read(&c.model).unwrap(), before);
    assert!(backups_in(c.dir.path()).is_empty());
}

#[test]
fn cli_strict_mismatch_is_an_input_error_without_backup() {
    let c = cli_fixture();
    let other = T::new("w", "BF16", &[12, 10], 77);
    write_st(&c.model, &[&other]);
    let (code, v) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&c.model),
            "--patch",
            s(&c.patch),
            "--verify-base",
            "strict",
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT, "{v}");
    assert_eq!(v["error"]["kind"], "invalid_arguments");
    assert_eq!(read_tensor(&c.model, "w"), other.bytes());
    assert!(backups_in(c.dir.path()).is_empty());

    // The default (warn) applies and reports the warning.
    let (code, v) = run_json(
        &deps(),
        &["apply", "--model", s(&c.model), "--patch", s(&c.patch)],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["warnings"].as_array().unwrap().len(), 1);
}

#[test]
fn cli_overflow_rolls_back_with_exit_3() {
    let dir = tempfile::tempdir().unwrap();
    let (model, patch) = overflow_fixture(dir.path());
    let before = std::fs::read(&model).unwrap();
    let (code, v) = run_json(
        &deps(),
        &["apply", "--model", s(&model), "--patch", s(&patch)],
    );
    assert_eq!(code, exit::EXECUTION_FAILED, "{v}");
    assert_eq!(v["error"]["kind"], "execution_failed");
    assert_eq!(std::fs::read(&model).unwrap(), before);
}

#[test]
fn cli_interrupt_before_start_exits_1_and_keeps_the_model() {
    let c = cli_fixture();
    let before = std::fs::read(&c.model).unwrap();
    let deps = deps();
    deps.interrupted.store(true, Ordering::SeqCst);
    let (code, v) = run_json(
        &deps,
        &["apply", "--model", s(&c.model), "--patch", s(&c.patch)],
    );
    assert_eq!(code, exit::GENERAL, "{v}");
    assert_eq!(v["error"]["kind"], "cancelled");
    assert_eq!(std::fs::read(&c.model).unwrap(), before);
    assert!(backups_in(c.dir.path()).is_empty());
}

#[test]
fn cli_json_patch_still_takes_the_legacy_path() {
    let c = cli_fixture();
    let legacy = c.dir.path().join("patch.json");
    std::fs::write(&legacy, "{not a patch").unwrap();
    let (code, v) = run_json(
        &deps(),
        &["apply", "--model", s(&c.model), "--patch", s(&legacy)],
    );
    assert_eq!(code, exit::INVALID_INPUT, "{v}");
    assert_eq!(v["error"]["kind"], "invalid_patch");
}

#[test]
fn cli_refuses_the_gguf_mode_for_a_surgery_patch() {
    let c = cli_fixture();
    let (code, v) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&c.model),
            "--patch",
            s(&c.patch),
            "--gguf",
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT, "{v}");
    assert_eq!(v["error"]["kind"], "invalid_arguments");
    assert!(backups_in(c.dir.path()).is_empty(), "nothing was touched");
}

#[test]
fn cli_warns_that_gguf_and_io_options_are_ignored_for_a_surgery_patch() {
    let c = cli_fixture();
    let (code, v) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&c.model),
            "--patch",
            s(&c.patch),
            "--no-prefetch",
        ],
    );
    assert_eq!(code, 0, "{v}");
    let warnings = v["warnings"].as_array().unwrap();
    assert!(
        warnings
            .iter()
            .any(|w| w.as_str().is_some_and(|w| w.contains("ignorées"))),
        "{warnings:?}"
    );
}
