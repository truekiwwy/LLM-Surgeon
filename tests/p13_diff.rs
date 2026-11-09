//! `surgeon diff`: engine (`run_diff`) and CLI (`app::run_with`) on small SafeTensors
//! checkpoints in temporary directories.
//!
//! Fixtures use values on a power-of-two grid, so `base + B*A` is exactly representable in
//! bf16: the difference of the two checkpoints has EXACT rank r and the f64 reference
//! `round(W_base + BA)` equals the tuned file bit for bit.

use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use half::{bf16, f16};
use serde_json::Value;
use weight_surgeon::app::{self, exit, AppDeps};
use weight_surgeon::eval::{CommandOutput, CommandRunner, EvalError};
use weight_surgeon::hardware::{GpuInfo, HardwareProbe, MemoryInfo};
use weight_surgeon::surgery::cli_args::{DiffArgs, OnHighError};
use weight_surgeon::surgery::diff::{run_diff, DiffReport};
use weight_surgeon::surgery::error::SurgeryError;
use weight_surgeon::surgery::events::{NullObserver, SurgeryEvent, SurgeryObserver};
use weight_surgeon::surgery::format::FactorDtype;
use weight_surgeon::surgery::reader::SurgeryReader;

// ------------------------------------------------------------------ fixtures

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// Multiples of `step` in `[-max_k * step, max_k * step]`.
    fn grid(&mut self, n: usize, step: f64, max_k: i64) -> Vec<f64> {
        (0..n)
            .map(|_| {
                let k = ((self.next() + 1.0) / 2.0 * (2 * max_k + 1) as f64) as i64 - max_k;
                k.min(max_k) as f64 * step
            })
            .collect()
    }

    fn noise(&mut self, n: usize, scale: f64) -> Vec<f64> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

#[derive(Clone)]
struct T {
    name: String,
    shape: Vec<usize>,
    dtype: &'static str,
    /// Values already representable in `dtype`.
    vals: Vec<f64>,
}

impl T {
    fn new(name: &str, shape: &[usize], dtype: &'static str, vals: Vec<f64>) -> Self {
        assert_eq!(vals.len(), shape.iter().product::<usize>());
        Self {
            name: name.into(),
            shape: shape.to_vec(),
            dtype,
            vals,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        encode(self.dtype, &self.vals)
    }
}

fn encode(dtype: &str, vals: &[f64]) -> Vec<u8> {
    vals.iter()
        .flat_map(|&x| match dtype {
            "BF16" => bf16::from_f64(x).to_le_bytes().to_vec(),
            "F16" => f16::from_f64(x).to_le_bytes().to_vec(),
            _ => (x as f32).to_le_bytes().to_vec(),
        })
        .collect()
}

fn round_bf16(x: f64) -> f64 {
    bf16::from_f64(x).to_f64()
}

fn header_json(tensors: &[T]) -> (Vec<u8>, Vec<u8>) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for t in tensors {
        let start = data.len();
        data.extend_from_slice(&t.bytes());
        header.insert(
            t.name.clone(),
            serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [start, data.len()]}),
        );
    }
    let mut header = serde_json::to_string(&header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    (file, data)
}

fn write_st(path: &Path, tensors: &[T]) {
    let (mut file, data) = header_json(tensors);
    file.extend_from_slice(&data);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, file).unwrap();
}

/// Two shards (even and odd tensors) plus `model.safetensors.index.json`.
fn write_sharded(dir: &Path, tensors: &[T]) {
    std::fs::create_dir_all(dir).unwrap();
    let mut weight_map = serde_json::Map::new();
    for (shard, name) in [
        (0, "model-00001-of-00002.safetensors"),
        (1, "model-00002-of-00002.safetensors"),
    ] {
        let part: Vec<T> = tensors.iter().skip(shard).step_by(2).cloned().collect();
        for t in &part {
            weight_map.insert(t.name.clone(), name.into());
        }
        write_st(&dir.join(name), &part);
    }
    let index = serde_json::json!({"metadata": {}, "weight_map": weight_map});
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
}

fn read_tensor(path: &Path, name: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    st.tensor(name).unwrap().data().to_vec()
}

fn decode_bf16(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks(2)
        .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f64())
        .collect()
}

fn bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// Order-preserving integer key of a bf16 bit pattern: neighbours differ by 1.
fn key(b: u16) -> i32 {
    let magnitude = i32::from(b & 0x7fff);
    if b & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }
}

const R: usize = 3;
const HIDDEN: usize = 16;
const INTER: usize = 24;

struct Fixture {
    base: Vec<T>,
    tuned: Vec<T>,
    /// `B` [m x r] and `A` [r x n] of every low-rank tensor.
    factors: BTreeMap<String, (Vec<f64>, Vec<f64>)>,
}

/// bf16 pair: 2D projections differ by exactly `B*A` (rank `R`), layer norms by a small
/// shift, `model.norm.weight` is identical, embeddings differ too.
fn fixture() -> Fixture {
    let mut rng = Lcg(7);
    let mut f = Fixture {
        base: Vec::new(),
        tuned: Vec::new(),
        factors: BTreeMap::new(),
    };
    let mut specs: Vec<(String, Vec<usize>)> =
        vec![("model.embed_tokens.weight".into(), vec![12, HIDDEN])];
    for l in 0..2 {
        specs.push((
            format!("model.layers.{l}.self_attn.q_proj.weight"),
            vec![HIDDEN, HIDDEN],
        ));
        specs.push((
            format!("model.layers.{l}.mlp.up_proj.weight"),
            vec![INTER, HIDDEN],
        ));
        specs.push((
            format!("model.layers.{l}.mlp.down_proj.weight"),
            vec![HIDDEN, INTER],
        ));
        specs.push((
            format!("model.layers.{l}.input_layernorm.weight"),
            vec![HIDDEN],
        ));
    }
    specs.push(("model.norm.weight".into(), vec![HIDDEN]));
    specs.push(("lm_head.weight".into(), vec![12, HIDDEN]));
    for (name, shape) in specs {
        let n: usize = shape.iter().product();
        // Odd multiples of 1/128: base + B*A (multiples of 1/64) is never 0, so a 1e-9 error of
        // the reconstruction cannot turn an exact zero into a different bf16 value.
        let w: Vec<f64> = rng
            .grid(n, 1.0 / 64.0, 32)
            .iter()
            .map(|x| x + 1.0 / 128.0)
            .collect();
        let new = if name == "model.norm.weight" {
            w.clone()
        } else if shape.len() == 1 {
            let shift = rng.grid(n, 1.0 / 64.0, 1);
            w.iter().zip(&shift).map(|(a, d)| a + d).collect()
        } else {
            let (m, cols) = (shape[0], shape[1]);
            let b = rng.grid(m * R, 0.125, 2);
            let a = rng.grid(R * cols, 0.125, 2);
            let delta = product(&b, &a, m, R, cols);
            f.factors.insert(name.clone(), (b, a));
            w.iter()
                .zip(&delta)
                .map(|(a, d)| round_bf16(a + d))
                .collect()
        };
        f.base.push(T::new(&name, &shape, "BF16", w));
        f.tuned.push(T::new(&name, &shape, "BF16", new));
    }
    f
}

fn product(b: &[f64], a: &[f64], m: usize, r: usize, n: usize) -> Vec<f64> {
    (0..m * n)
        .map(|i| (0..r).map(|k| b[(i / n) * r + k] * a[k * n + i % n]).sum())
        .collect()
}

/// `root/base` and `root/tuned` as single-file model directories.
fn write_pair(root: &Path, base: &[T], tuned: &[T]) -> (PathBuf, PathBuf) {
    let (b, t) = (root.join("base"), root.join("tuned"));
    write_st(&b.join("model.safetensors"), base);
    write_st(&t.join("model.safetensors"), tuned);
    (b, t)
}

fn args(base: &Path, tuned: &Path, out: &Path) -> DiffArgs {
    DiffArgs {
        base: base.into(),
        tuned: tuned.into(),
        out: out.into(),
        rank: R,
        // f32 factors: exact to ~1e-7, so exact-rank checks can be strict.
        factor_dtype: FactorDtype::F32,
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

fn diff(a: &DiffArgs) -> Result<DiffReport, SurgeryError> {
    run_diff(a, &mut NullObserver, &AtomicBool::new(false))
}

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

// ------------------------------------------------------------------ CLI plumbing

struct Probe(u64);

impl HardwareProbe for Probe {
    fn system_ram(&self) -> weight_surgeon::hardware::Result<MemoryInfo> {
        Ok(MemoryInfo {
            total_bytes: self.0,
            available_bytes: self.0,
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

fn deps_with_ram(ram: u64) -> AppDeps {
    AppDeps {
        probe: Box::new(Probe(ram)),
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

fn deps() -> AppDeps {
    deps_with_ram(64 << 30)
}

fn run_cli(deps: &AppDeps, json: bool, args: &[&str]) -> (i32, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let argv = ["weight-surgeon"]
        .into_iter()
        .chain(json.then_some("--json"))
        .chain(args.iter().copied());
    let code = match app::parse_cli(argv, &mut out, &mut err) {
        Ok(cli) => app::run_with(cli, deps, &mut out, &mut err),
        Err(code) => code,
    };
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

fn run_json(deps: &AppDeps, args: &[&str]) -> (i32, Value) {
    let (code, out, _) = run_cli(deps, true, args);
    let value: Value = serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"));
    assert_eq!(value["exit_code"], code);
    (code, value)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

/// `apply --patch` on a copy of `base`; returns the patched model file.
fn apply_to_copy(base_dir: &Path, patch: &Path, copy_dir: &Path) -> PathBuf {
    std::fs::create_dir_all(copy_dir).unwrap();
    let model = copy_dir.join("model.safetensors");
    std::fs::copy(base_dir.join("model.safetensors"), &model).unwrap();
    let (code, v) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&model),
            "--patch",
            s(patch),
            "--no-backup",
            "--verify-base",
            "strict",
        ],
    );
    assert_eq!(code, exit::OK, "{v}");
    model
}

// ------------------------------------------------------------------ exact rank round trip

#[test]
fn exact_rank_difference_round_trips_through_apply() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);
    for (factor, must_be_exact) in [(FactorDtype::F32, true), (FactorDtype::F16, false)] {
        let out = dir.path().join(format!("{}.surgery", factor.as_str()));
        let mut a = args(&base, &tuned, &out);
        a.factor_dtype = factor;
        let report = diff(&a).unwrap();
        assert_eq!(report.factor_dtype, factor.as_str());
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);

        let patched = apply_to_copy(
            &base,
            &out,
            &dir.path().join(format!("applied-{}", factor.as_str())),
        );
        let (mut total, mut exact) = (0usize, 0usize);
        for t in f
            .tuned
            .iter()
            .filter(|t| !t.name.contains("embed_tokens") && !t.name.contains("lm_head"))
        {
            let got = bits(&read_tensor(&patched, &t.name));
            // Independent f64 reference: round(W_base + B*A), or the tuned norm values.
            let base_vals = &f.base.iter().find(|b| b.name == t.name).unwrap().vals;
            let (expected, pre_rounding): (Vec<u16>, Vec<f64>) = match f.factors.get(&t.name) {
                Some((b, a)) => {
                    let delta = product(b, a, t.shape[0], R, t.shape[1]);
                    let pre: Vec<f64> = base_vals.iter().zip(&delta).map(|(w, d)| w + d).collect();
                    (
                        pre.iter().map(|&x| bf16::from_f64(x).to_bits()).collect(),
                        pre,
                    )
                }
                None => (bits(&t.bytes()), t.vals.clone()),
            };
            assert_eq!(expected, bits(&t.bytes()), "{}: fixture is exact", t.name);
            for ((g, e), pre) in got.iter().zip(&expected).zip(&pre_rounding) {
                total += 1;
                exact += usize::from(g == e);
                if must_be_exact {
                    // Documented tolerance: 1 ulp where the pre-rounding value sits on a rounding boundary.
                    assert!(
                        (key(*g) - key(*e)).abs() <= 1,
                        "{}: {g:#x} vs {e:#x}",
                        t.name
                    );
                } else {
                    // f16 factors carry ~5e-4 relative error: bf16 rounding no longer hides it
                    // for results near zero (fine ulp), so bound the absolute error instead.
                    let err = (bf16::from_bits(*g).to_f64() - pre).abs();
                    assert!(err < 2e-3, "{}: {err}", t.name);
                }
            }
        }
        println!(
            "{}: {exact}/{total} elements bit-exact after diff + apply",
            factor.as_str()
        );
        // f32 factors: 99% bit-exact required (100% observed). f16 factors measured 98.95%
        // here: they are 11-bit approximations, so only a looser floor is guaranteed.
        let floor = if must_be_exact { 99 } else { 95 };
        assert!(exact * 100 >= total * floor, "{exact}/{total}");
    }
}

#[test]
fn effective_rank_is_clamped_and_reported_per_tensor() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);
    let mut a = args(&base, &tuned, &dir.path().join("x.surgery"));
    a.rank = 100;
    let report = diff(&a).unwrap();
    assert_eq!(report.rank, 100);
    let q = report
        .tensors
        .iter()
        .find(|t| t.name.ends_with("0.self_attn.q_proj.weight"))
        .unwrap();
    assert_eq!((q.kind.as_str(), q.rank), ("lowrank", HIDDEN));
    let header = SurgeryReader::open(&a.out).unwrap();
    let entry = header
        .header()
        .tensors
        .iter()
        .find(|t| t.name == q.name)
        .unwrap();
    assert_eq!(entry.rank, HIDDEN);
    assert_eq!(header.header().rank, 100);
}

#[test]
fn full_rank_noise_reports_the_measured_reconstruction_error() {
    let mut f = fixture();
    let mut rng = Lcg(99);
    // Replace the tuned 2D tensors by base + noise: a full-rank difference.
    for (b, t) in f.base.iter().zip(f.tuned.iter_mut()) {
        if b.shape.len() == 2 {
            let noise = rng.noise(b.vals.len(), 0.05);
            t.vals = b
                .vals
                .iter()
                .zip(&noise)
                .map(|(w, n)| round_bf16(w + n))
                .collect();
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);
    let out = dir.path().join("x.surgery");
    let report = diff(&args(&base, &tuned, &out)).unwrap();

    let mut reader = SurgeryReader::open(&out).unwrap();
    let entries = reader.header().tensors.clone();
    let mut checked = 0;
    for t in report.tensors.iter().filter(|t| t.kind == "lowrank") {
        let entry = entries.iter().find(|e| e.name == t.name).unwrap();
        let factors = reader.read_lowrank(entry).unwrap();
        let (m, n, r) = (entry.shape[0], entry.shape[1], entry.rank);
        let (w, new) = (
            &f.base.iter().find(|b| b.name == t.name).unwrap().vals,
            &f.tuned.iter().find(|b| b.name == t.name).unwrap().vals,
        );
        let (mut err2, mut norm2) = (0.0f64, 0.0f64);
        for i in 0..m {
            for j in 0..n {
                let approx: f64 = (0..r)
                    .map(|k| {
                        f64::from(factors.u[i * r + k])
                            * f64::from(factors.s[k])
                            * f64::from(factors.v[k * n + j])
                    })
                    .sum();
                let d = new[i * n + j] - w[i * n + j];
                err2 += (d - approx).powi(2);
                norm2 += d * d;
            }
        }
        let measured = (err2 / norm2).sqrt();
        assert!(
            (measured - t.rel_error).abs() < 1e-4,
            "{}: {measured} vs {}",
            t.name,
            t.rel_error
        );
        assert!(
            t.rel_error > 0.1 && t.rel_error < 1.0,
            "{}: {}",
            t.name,
            t.rel_error
        );
        assert!(t.energy > 0.0 && t.energy < 1.0);
        assert!((t.energy - (1.0 - t.rel_error.powi(2))).abs() < 1e-3);
        checked += 1;
    }
    assert!(checked >= 6);
}

// ------------------------------------------------------------------ kinds, skips, mismatches

#[test]
fn one_dimensional_tensors_are_full_deltas_and_identical_ones_are_skipped() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);
    let out = dir.path().join("x.surgery");
    let report = diff(&args(&base, &tuned, &out)).unwrap();

    assert_eq!(report.skipped_identical, 1, "model.norm.weight");
    assert_eq!(report.skipped_filtered, 2, "embed_tokens and lm_head");
    assert!(report.tensors.iter().all(|t| t.name != "model.norm.weight"));
    let norm = report
        .tensors
        .iter()
        .find(|t| t.name.ends_with("0.input_layernorm.weight"))
        .unwrap();
    assert_eq!(
        (norm.kind.as_str(), norm.rank, norm.rel_error),
        ("full_delta", 0, 0.0)
    );
    assert_eq!(report.tensors.len(), 2 * 4);
    // Sorted by name, whatever the file order.
    let names: Vec<_> = report.tensors.iter().map(|t| &t.name).collect();
    assert!(names.windows(2).all(|w| w[0] < w[1]));

    // The exact f32 delta reproduces the tuned norm bytes.
    let patched = apply_to_copy(&base, &out, &dir.path().join("applied"));
    for t in f.tuned.iter().filter(|t| t.name.contains("layernorm")) {
        assert_eq!(read_tensor(&patched, &t.name), t.bytes(), "{}", t.name);
    }
    let reader = SurgeryReader::open(&out).unwrap();
    let h = reader.header();
    assert_eq!(h.base.tensor_count, f.base.len());
    assert!(h.tensors.iter().all(|e| !e.base_sha256.is_empty()));
}

#[test]
fn base_sha256_is_the_hash_of_the_base_tensor_bytes() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);
    let out = dir.path().join("x.surgery");
    diff(&args(&base, &tuned, &out)).unwrap();
    let reader = SurgeryReader::open(&out).unwrap();
    for e in &reader.header().tensors {
        let raw = f.base.iter().find(|b| b.name == e.name).unwrap().bytes();
        assert_eq!(
            e.base_sha256,
            weight_surgeon::surgery::format::sha256_hex(&raw),
            "{}",
            e.name
        );
        assert_eq!(e.target_dtype, "BF16");
    }
}

#[test]
fn every_difference_between_the_checkpoints_is_listed() {
    let f = fixture();
    let mut tuned = f.tuned.clone();
    tuned.retain(|t| t.name != "model.norm.weight"); // missing from tuned
    tuned.push(T::new("extra.weight", &[2], "BF16", vec![1.0, 2.0])); // extra in tuned
    let q = tuned
        .iter_mut()
        .find(|t| t.name.ends_with("0.self_attn.q_proj.weight"))
        .unwrap();
    *q = T::new(&q.name, &[HIDDEN * HIDDEN], "BF16", q.vals.clone()); // shape differs
    let up = tuned
        .iter_mut()
        .find(|t| t.name.ends_with("0.mlp.up_proj.weight"))
        .unwrap();
    *up = T::new(&up.name, &up.shape, "F16", up.vals.clone()); // dtype differs

    let dir = tempfile::tempdir().unwrap();
    let (base, tuned_dir) = write_pair(dir.path(), &f.base, &tuned);
    let out = dir.path().join("x.surgery");
    let message = diff(&args(&base, &tuned_dir, &out))
        .unwrap_err()
        .to_string();
    for needle in [
        "model.norm.weight",
        "extra.weight",
        "0.self_attn.q_proj.weight",
        "0.mlp.up_proj.weight",
        "4 écart",
    ] {
        assert!(message.contains(needle), "{needle} missing from: {message}");
    }
    assert!(!out.exists());

    let (code, v) = run_json(
        &deps(),
        &[
            "diff",
            "--base",
            s(&base),
            "--tuned",
            s(&tuned_dir),
            "--out",
            s(&out),
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT);
    assert_eq!(v["error"]["kind"], "invalid_arguments");
}

#[test]
fn non_finite_values_name_the_tensor() {
    let f = fixture();
    let mut tuned = f.tuned.clone();
    let t = tuned
        .iter_mut()
        .find(|t| t.name.ends_with("1.mlp.down_proj.weight"))
        .unwrap();
    t.vals[5] = f64::NAN;
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned_dir) = write_pair(dir.path(), &f.base, &tuned);
    let out = dir.path().join("x.surgery");
    let err = diff(&args(&base, &tuned_dir, &out)).unwrap_err();
    assert!(matches!(err, SurgeryError::Usage(_)));
    assert!(err.to_string().contains("1.mlp.down_proj.weight"), "{err}");
    assert!(!out.exists());
}

// ------------------------------------------------------------------ filters

#[test]
fn include_exclude_and_embeddings_select_tensors() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);
    let run = |edit: &dyn Fn(&mut DiffArgs)| {
        let mut a = args(&base, &tuned, &dir.path().join("x.surgery"));
        a.overwrite = true;
        edit(&mut a);
        diff(&a).unwrap()
    };

    let only_q = run(&|a| a.include = Some("q_proj".into()));
    assert_eq!(only_q.tensors.len(), 2);
    assert!(only_q.tensors.iter().all(|t| t.name.contains("q_proj")));
    assert_eq!(only_q.skipped_filtered, f.base.len() - 2);

    let no_layer1 = run(&|a| a.exclude = Some(r"layers\.1\.".into()));
    assert!(no_layer1
        .tensors
        .iter()
        .all(|t| !t.name.contains("layers.1.")));
    assert_eq!(no_layer1.tensors.len(), 4);

    let with_embeddings = run(&|a| a.include_embeddings = true);
    assert_eq!(with_embeddings.skipped_filtered, 0);
    for name in ["model.embed_tokens.weight", "lm_head.weight"] {
        assert!(
            with_embeddings.tensors.iter().any(|t| t.name == name),
            "{name}"
        );
    }

    let mut bad = args(&base, &tuned, &dir.path().join("y.surgery"));
    bad.include = Some("(".into());
    let err = diff(&bad).unwrap_err();
    assert!(matches!(err, SurgeryError::Usage(_)), "{err}");
    let (code, _) = run_json(
        &deps(),
        &[
            "diff",
            "--base",
            s(&base),
            "--tuned",
            s(&tuned),
            "--out",
            s(&dir.path().join("y.surgery")),
            "--exclude",
            "[",
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT);
}

#[test]
fn include_embeddings_checks_the_hardware_first() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);
    let out = dir.path().join("x.surgery");
    let cli = [
        "diff",
        "--base",
        s(&base),
        "--tuned",
        s(&tuned),
        "--out",
        s(&out),
        "--include-embeddings",
    ];

    let (code, v) = run_json(&deps_with_ram(1 << 20), &cli);
    assert_eq!(code, exit::GENERAL, "{v}");
    assert_eq!(v["error"]["kind"], "insufficient_resources");
    assert!(!out.exists());

    let (code, v) = run_json(&deps(), &cli);
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["data"]["surgery"]["skipped_filtered"], 0);
}

// ------------------------------------------------------------------ --max-rel-error

/// `hot` is full-rank noise (high error at rank 1), `ok` is exactly rank 1.
fn high_error_pair(dir: &Path) -> (PathBuf, PathBuf, Vec<T>) {
    let mut rng = Lcg(5);
    let w = rng.grid(HIDDEN * HIDDEN, 1.0 / 64.0, 32);
    let noise = rng.noise(HIDDEN * HIDDEN, 0.05);
    let hot_new: Vec<f64> = w
        .iter()
        .zip(&noise)
        .map(|(a, n)| round_bf16(a + n))
        .collect();
    let (b, a) = (rng.grid(HIDDEN, 0.125, 2), rng.grid(HIDDEN, 0.125, 2));
    let ok_new: Vec<f64> = w
        .iter()
        .zip(product(&b, &a, HIDDEN, 1, HIDDEN))
        .map(|(x, d)| x + d)
        .collect();
    let base = [
        T::new("hot.weight", &[HIDDEN, HIDDEN], "BF16", w.clone()),
        T::new("ok.weight", &[HIDDEN, HIDDEN], "BF16", w),
    ];
    let tuned = vec![
        T::new("hot.weight", &[HIDDEN, HIDDEN], "BF16", hot_new),
        T::new("ok.weight", &[HIDDEN, HIDDEN], "BF16", ok_new),
    ];
    let (bd, td) = write_pair(dir, &base, &tuned);
    (bd, td, tuned)
}

#[test]
fn max_rel_error_warn_skip_and_full() {
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned, tuned_tensors) = high_error_pair(dir.path());
    let run = |policy: OnHighError, name: &str| {
        let out = dir.path().join(name);
        let mut a = args(&base, &tuned, &out);
        a.rank = 1;
        a.max_rel_error = Some(0.5);
        a.on_high_error = policy;
        (diff(&a).unwrap(), out)
    };

    let (warn, _) = run(OnHighError::Warn, "warn.surgery");
    assert_eq!(warn.tensors.len(), 2);
    assert!(warn.tensors.iter().all(|t| t.kind == "lowrank"));
    assert_eq!(warn.warnings.len(), 1);
    assert!(
        warn.warnings[0].contains("hot.weight"),
        "{:?}",
        warn.warnings
    );
    let hot = warn
        .tensors
        .iter()
        .find(|t| t.name == "hot.weight")
        .unwrap();
    assert!(hot.rel_error > 0.5);

    let (skip, out) = run(OnHighError::Skip, "skip.surgery");
    assert_eq!(
        skip.tensors
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["ok.weight"]
    );
    assert_eq!(skip.warnings.len(), 1);
    assert!(SurgeryReader::open(&out)
        .unwrap()
        .header()
        .tensors
        .iter()
        .all(|e| e.name == "ok.weight"));

    let (full, out) = run(OnHighError::Full, "full.surgery");
    let hot = full
        .tensors
        .iter()
        .find(|t| t.name == "hot.weight")
        .unwrap();
    assert_eq!(
        (hot.kind.as_str(), hot.rank, hot.rel_error),
        ("full_delta", 0, 0.0)
    );
    assert_eq!(
        full.tensors
            .iter()
            .find(|t| t.name == "ok.weight")
            .unwrap()
            .kind,
        "lowrank"
    );
    let patched = apply_to_copy(&base, &out, &dir.path().join("applied"));
    assert_eq!(
        read_tensor(&patched, "hot.weight"),
        tuned_tensors[0].bytes(),
        "exact delta"
    );
}

// ------------------------------------------------------------------ f16 overflow

#[test]
fn f16_overflow_restarts_with_bf16_and_warns() {
    let mut rng = Lcg(3);
    // Rank-1 delta of 99840 everywhere: its singular value (~800000) does not fit f16.
    let base = [
        T::new(
            "small.weight",
            &[8, 8],
            "BF16",
            rng.grid(64, 1.0 / 64.0, 32),
        ),
        T::new("huge.weight", &[8, 8], "BF16", vec![0.0; 64]),
    ];
    let mut tuned = base.to_vec();
    tuned[1].vals = vec![99_840.0; 64];
    tuned[0].vals = base[0].vals.iter().map(|x| x + 0.125).collect();
    let dir = tempfile::tempdir().unwrap();
    let (b, t) = write_pair(dir.path(), &base, &tuned);

    let out = dir.path().join("x.surgery");
    let mut a = args(&b, &t, &out);
    a.factor_dtype = FactorDtype::F16;
    let report = diff(&a).unwrap();
    assert_eq!(report.factor_dtype, "bf16");
    assert_eq!(report.warnings.len(), 1);
    assert!(
        report.warnings[0].contains("huge.weight") && report.warnings[0].contains("bf16"),
        "{:?}",
        report.warnings
    );
    assert_eq!(report.tensors.len(), 2);
    assert_eq!(
        SurgeryReader::open(&out).unwrap().header().factor_dtype,
        FactorDtype::Bf16
    );
    assert!(
        report.tensors.iter().all(|t| t.rel_error < 0.02),
        "{:?}",
        report.tensors
    );

    // Asking for bf16 directly needs no fallback; f32 neither.
    for dtype in [FactorDtype::Bf16, FactorDtype::F32] {
        a.factor_dtype = dtype;
        a.overwrite = true;
        let report = diff(&a).unwrap();
        assert!(report.warnings.is_empty());
        assert_eq!(report.factor_dtype, dtype.as_str());
    }
    // No scratch file left next to the output.
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with('.'))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

// ------------------------------------------------------------------ sharded directories

#[test]
fn sharded_checkpoints_give_the_same_patch_as_single_files() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let single = dir.path().join("single");
    let (sb, st) = write_pair(&single, &f.base, &f.tuned);
    let sharded = dir.path().join("sharded");
    write_sharded(&sharded.join("base"), &f.base);
    write_sharded(&sharded.join("tuned"), &f.tuned);
    // Mixed: sharded base, single-file tuned (the file itself, not its directory).
    let mixed_tuned = st.join("model.safetensors");

    let out_single = dir.path().join("single.surgery");
    let r1 = diff(&args(&sb, &st, &out_single)).unwrap();
    let out_sharded = dir.path().join("sharded.surgery");
    let r2 = diff(&args(
        &sharded.join("base"),
        &sharded.join("tuned"),
        &out_sharded,
    ))
    .unwrap();
    let out_mixed = dir.path().join("mixed.surgery");
    let r3 = diff(&args(&sharded.join("base"), &mixed_tuned, &out_mixed)).unwrap();

    assert_eq!(r1.tensors.len(), r2.tensors.len());
    assert_eq!(r1.sha256, r2.sha256);
    assert_eq!(
        std::fs::read(&out_single).unwrap(),
        std::fs::read(&out_sharded).unwrap()
    );
    // Only the tuned name differs in the header of the mixed run.
    assert_eq!(r3.tensors.len(), r1.tensors.len());
    assert_eq!(
        SurgeryReader::open(&out_mixed).unwrap().header().tuned.name,
        "model.safetensors"
    );
}

// ------------------------------------------------------------------ determinism, header

#[test]
fn output_is_deterministic_across_runs_and_thread_counts() {
    let mut f = fixture();
    // Noise on one tensor so the randomized SVD really works.
    let mut rng = Lcg(11);
    let t = f
        .tuned
        .iter_mut()
        .find(|t| t.name.ends_with("0.mlp.up_proj.weight"))
        .unwrap();
    t.vals = t
        .vals
        .iter()
        .map(|x| round_bf16(x + rng.next() * 0.05))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);

    let mut outputs = Vec::new();
    for (i, threads) in [1usize, 1, 2, 8].into_iter().enumerate() {
        let out = dir.path().join(format!("run{i}.surgery"));
        let mut a = args(&base, &tuned, &out);
        a.factor_dtype = FactorDtype::F16;
        a.seed = 42;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| diff(&a).unwrap());
        outputs.push(std::fs::read(&out).unwrap());
    }
    assert!(outputs.iter().all(|o| *o == outputs[0]));
}

#[test]
fn header_has_names_only_and_timestamp_only_on_request() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);

    let plain = dir.path().join("plain.surgery");
    diff(&args(&base, &tuned, &plain)).unwrap();
    let reader = SurgeryReader::open(&plain).unwrap();
    assert_eq!(
        (
            reader.header().base.name.as_str(),
            reader.header().tuned.name.as_str()
        ),
        ("base", "tuned")
    );
    assert!(reader.header().provenance.is_none());
    let raw = std::fs::read(&plain).unwrap();
    let text = String::from_utf8_lossy(&raw);
    for personal in [dir.path().to_string_lossy().as_ref(), "kiwwy", "Users"] {
        assert!(!text.contains(personal), "{personal} leaked into the patch");
    }

    let stamped = dir.path().join("stamped.surgery");
    let mut a = args(&base, &tuned, &stamped);
    a.timestamp = true;
    diff(&a).unwrap();
    let reader = SurgeryReader::open(&stamped).unwrap();
    let ts = &reader.header().provenance.as_ref().unwrap().timestamp;
    let shape: String = ts
        .chars()
        .map(|c| if c.is_ascii_digit() { '9' } else { c })
        .collect();
    assert_eq!(shape, "9999-99-99T99:99:99Z", "{ts}");
    assert!(ts.as_str() >= "2025", "{ts}");
}

// ------------------------------------------------------------------ events, cancellation

#[test]
fn progress_events_cover_hashing_and_every_tensor() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);
    let mut rec = Recorder::default();
    let report = run_diff(
        &args(&base, &tuned, &dir.path().join("x.surgery")),
        &mut rec,
        &AtomicBool::new(false),
    )
    .unwrap();
    let hashing = rec
        .events
        .iter()
        .filter(|e| matches!(e, SurgeryEvent::Hashing { .. }))
        .count();
    assert_eq!(hashing, f.base.len() - report.skipped_filtered);
    let last = rec.events.iter().rev().find_map(|e| match e {
        SurgeryEvent::Tensor { done, total, .. } => Some((*done, *total)),
        _ => None,
    });
    assert_eq!(last, Some((report.tensors.len(), report.tensors.len())));
}

#[test]
fn cancellation_leaves_no_output_and_no_scratch_file() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);
    let out_dir = dir.path().join("out");
    std::fs::create_dir(&out_dir).unwrap();
    let out = out_dir.join("x.surgery");

    let err = run_diff(
        &args(&base, &tuned, &out),
        &mut NullObserver,
        &AtomicBool::new(true),
    )
    .unwrap_err();
    assert!(matches!(err, SurgeryError::Cancelled), "{err}");

    // The observer asks to stop in the middle of the write.
    let mut rec = Recorder {
        break_after: Some(f.base.len() - 2 + 3),
        ..Recorder::default()
    };
    let err = run_diff(
        &args(&base, &tuned, &out),
        &mut rec,
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(matches!(err, SurgeryError::Cancelled), "{err}");
    assert!(
        std::fs::read_dir(&out_dir).unwrap().next().is_none(),
        "output dir must stay empty"
    );

    // Same through the CLI flag (Ctrl-C): exit 1, kind cancelled.
    let d = deps();
    d.interrupted.store(true, Ordering::SeqCst);
    let (code, v) = run_json(
        &d,
        &[
            "diff",
            "--base",
            s(&base),
            "--tuned",
            s(&tuned),
            "--out",
            s(&out),
        ],
    );
    assert_eq!(
        (code, v["error"]["kind"].as_str()),
        (exit::GENERAL, Some("cancelled"))
    );
    assert!(std::fs::read_dir(&out_dir).unwrap().next().is_none());
}

// ------------------------------------------------------------------ CLI

#[test]
fn cli_json_envelope_and_human_output() {
    let f = fixture();
    let dir = tempfile::tempdir().unwrap();
    let (base, tuned) = write_pair(dir.path(), &f.base, &f.tuned);
    let out = dir.path().join("x.surgery");
    let (code, v) = run_json(
        &deps(),
        &[
            "diff",
            "--base",
            s(&base),
            "--tuned",
            s(&tuned),
            "--out",
            s(&out),
            "--rank",
            "3",
            "--factor-dtype",
            "f32",
            "--seed",
            "1",
        ],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["command"], "diff");
    let surgery = &v["data"]["surgery"];
    assert_eq!(surgery["rank"], 3);
    assert_eq!(surgery["factor_dtype"], "f32");
    assert_eq!(surgery["skipped_identical"], 1);
    assert_eq!(surgery["bytes"], std::fs::metadata(&out).unwrap().len());
    let first = &surgery["tensors"][0];
    for key in ["name", "kind", "rank", "rel_error", "energy"] {
        assert!(!first[key].is_null(), "{key}");
    }
    assert!(v["warnings"].as_array().unwrap().is_empty());

    // The same patch is accepted by `apply` (magic detection) and reproduces the tuned file.
    let patched = apply_to_copy(&base, &out, &dir.path().join("applied"));
    let q = "model.layers.0.self_attn.q_proj.weight";
    assert_eq!(
        decode_bf16(&read_tensor(&patched, q)),
        f.tuned.iter().find(|t| t.name == q).unwrap().vals
    );

    // Existing --out: exit 2 and the file is untouched; --overwrite replaces it.
    let before = std::fs::read(&out).unwrap();
    let cli = [
        "diff",
        "--base",
        s(&base),
        "--tuned",
        s(&tuned),
        "--out",
        s(&out),
        "--rank",
        "2",
    ];
    let (code, v) = run_json(&deps(), &cli);
    assert_eq!(
        (code, v["error"]["kind"].as_str()),
        (exit::INVALID_INPUT, Some("invalid_arguments"))
    );
    assert_eq!(std::fs::read(&out).unwrap(), before);
    let overwrite: Vec<&str> = cli.iter().copied().chain(["--overwrite"]).collect();
    let (code, _) = run_json(&deps(), &overwrite);
    assert_eq!(code, exit::OK);
    assert_ne!(std::fs::read(&out).unwrap(), before);

    // Human output carries the honesty note and the per-tensor errors.
    let (code, text, _) = run_cli(&deps(), false, &overwrite);
    assert_eq!(code, exit::OK);
    for needle in ["APPROXIMATION", "erreur relative", "q_proj", "MiB"] {
        assert!(text.contains(needle), "{needle} missing from:\n{text}");
    }
}

#[test]
fn cli_overflow_warning_reaches_the_envelope() {
    let base = [T::new("huge.weight", &[8, 8], "BF16", vec![0.0; 64])];
    let tuned = [T::new("huge.weight", &[8, 8], "BF16", vec![99_840.0; 64])];
    let dir = tempfile::tempdir().unwrap();
    let (b, t) = write_pair(dir.path(), &base, &tuned);
    let out = dir.path().join("x.surgery");
    let (code, v) = run_json(
        &deps(),
        &["diff", "--base", s(&b), "--tuned", s(&t), "--out", s(&out)],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["data"]["surgery"]["factor_dtype"], "bf16");
    assert_eq!(v["warnings"].as_array().unwrap().len(), 1);
}
