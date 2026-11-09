//! `lora merge` and `lora extract`: merge semantics against an independent f64 reference,
//! refusals, key mapping, rollback, cancellation, the extract -> merge round trip, and the
//! CLI envelope (`app::run_with`, `--json`).
//!
//! Merge fixtures use multiples of 1/8 and power-of-two scales, so every f32 sum is exact and
//! the kernel must match the f64 reference (rounded once, to nearest even) bit for bit.
//! Extract is numerical (rSVD): the round trip is checked with a documented tolerance.
//! The PEFT files written by `extract` are not checked against HuggingFace PEFT here.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use half::{bf16, f16};
use serde_json::Value;
use weight_surgeon::app::{self, exit, AppDeps};
use weight_surgeon::eval::{CommandOutput, CommandRunner, EvalError};
use weight_surgeon::hardware::{GpuInfo, HardwareProbe, MemoryInfo};
use weight_surgeon::lora::cli_args::LoraExtractArgs;
use weight_surgeon::lora::extract::run_extract;
use weight_surgeon::lora::merge::{run_merge, MergeReport, MergeRequest};
use weight_surgeon::patch::{BackupSet, RollbackHandle};
use weight_surgeon::surgery::error::SurgeryError;
use weight_surgeon::surgery::events::{NullObserver, SurgeryEvent, SurgeryObserver};
use weight_surgeon::surgery::format::FactorDtype;
use weight_surgeon::surgery::target::resolve_shards;

// ------------------------------------------------------------------ fixtures

#[derive(Clone)]
struct T {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    vals: Vec<f32>,
}

fn t(name: &str, dtype: &'static str, shape: &[usize], seed: u64) -> T {
    T {
        name: name.to_owned(),
        dtype,
        shape: shape.to_vec(),
        vals: eighths(seed, shape.iter().product()),
    }
}

impl T {
    fn with_vals(mut self, vals: Vec<f32>) -> Self {
        assert_eq!(vals.len(), self.vals.len());
        self.vals = vals;
        self
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
    file.extend_from_slice(&data);
    std::fs::write(path, file).unwrap();
}

fn read_tensor(path: &Path, name: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    st.tensor(name).unwrap().data().to_vec()
}

fn read_names(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut names: Vec<String> = st.names().into_iter().map(str::to_owned).collect();
    names.sort();
    names
}

const Q: &str = "model.layers.0.self_attn.q_proj";
const V: &str = "model.layers.0.self_attn.v_proj";

/// `lora_A` and `lora_B` of the module `path` (target `path.weight` is `[m, n]`), PEFT keys.
/// `default` selects the `.default.` key layout.
fn pair(
    path: &str,
    m: usize,
    n: usize,
    r: usize,
    dtype: &'static str,
    seed: u64,
    default: bool,
) -> [T; 2] {
    let infix = if default { ".default" } else { "" };
    [
        t(
            &format!("base_model.model.{path}.lora_A{infix}.weight"),
            dtype,
            &[r, n],
            seed,
        ),
        t(
            &format!("base_model.model.{path}.lora_B{infix}.weight"),
            dtype,
            &[m, r],
            seed + 50,
        ),
    ]
}

fn write_adapter(dir: &Path, tensors: &[&T], config: Option<&str>) -> PathBuf {
    let path = dir.join("adapter_model.safetensors");
    write_st(&path, tensors);
    if let Some(config) = config {
        std::fs::write(dir.join("adapter_config.json"), config).unwrap();
    }
    path
}

fn config(alpha: f64, r: usize, extra: &str) -> String {
    format!(r#"{{"peft_type": "LORA", "r": {r}, "lora_alpha": {alpha}{extra}}}"#)
}

/// f64 reference of `W + eff * B A`, rounded once to the tensor dtype.
fn reference(w: &T, a: &T, b: &T, eff: f64) -> Vec<u8> {
    let (m, n) = (w.shape[0], w.shape[1]);
    let r = a.shape[0];
    let (av, bv, old) = (a.stored(), b.stored(), w.stored());
    let new: Vec<f64> = (0..m * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            let delta: f64 = (0..r).map(|k| bv[i * r + k] * av[k * n + j]).sum();
            old[idx] + eff * delta
        })
        .collect();
    encode_f64(w.dtype, &new)
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

fn request(model: &Path, lora: &Path) -> MergeRequest {
    MergeRequest {
        model_shards: resolve_shards(model).unwrap(),
        lora: lora.to_path_buf(),
        alpha: None,
        adapter_config: None,
        scale: 1.0,
        dry_run: false,
        cancel: Arc::new(AtomicBool::new(false)),
    }
}

fn merge(req: &MergeRequest) -> (Result<MergeReport, SurgeryError>, CountingGuard) {
    let mut guard = CountingGuard::default();
    let result = run_merge(req, &mut guard, &mut Recorder::default());
    (result, guard)
}

/// Model with q [12, 10], v [6, 10] and an untouched 1D tensor, in `dtype`.
struct Fixture {
    dir: tempfile::TempDir,
    model: PathBuf,
    q: T,
    v: T,
    norm: T,
}

fn fixture(dtype: &'static str) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let (q, v) = (
        t(&format!("{Q}.weight"), dtype, &[12, 10], 1),
        t(&format!("{V}.weight"), dtype, &[6, 10], 2),
    );
    let norm = t("model.norm.weight", dtype, &[10], 3);
    let model = dir.path().join("model.safetensors");
    write_st(&model, &[&q, &v, &norm]);
    Fixture {
        dir,
        model,
        q,
        v,
        norm,
    }
}

/// Adapter of rank 4 for the fixture; `q` uses the `.default.` layout, `v` the plain one.
fn std_adapter(
    f: &Fixture,
    dtype: &'static str,
    config: Option<&str>,
) -> (PathBuf, [T; 2], [T; 2]) {
    let [qa, qb] = pair(Q, 12, 10, 4, dtype, 10, true);
    let [va, vb] = pair(V, 6, 10, 4, dtype, 20, false);
    let path = write_adapter(f.dir.path(), &[&qa, &qb, &va, &vb], config);
    (path, [qa, qb], [va, vb])
}

fn err_text(r: Result<MergeReport, SurgeryError>) -> String {
    r.unwrap_err().to_string()
}

// ------------------------------------------------------------------ merge semantics

#[test]
fn merge_matches_f64_reference_for_every_dtype() {
    for dtype in ["BF16", "F16", "F32"] {
        let f = fixture(dtype);
        let (lora, [qa, qb], [va, vb]) = std_adapter(&f, dtype, Some(&config(8.0, 4, "")));
        let norm_before = read_tensor(&f.model, &f.norm.name);
        let (result, guard) = merge(&request(&f.model, &lora));
        let report = result.unwrap();
        assert_eq!((guard.commits, guard.rollbacks), (1, 0));
        assert_eq!(report.rank, 4);
        assert_eq!(report.lora_alpha, 8.0);
        assert_eq!(report.effective_scale, 2.0);
        assert_eq!(report.tensors, 2);
        assert!(report.max_abs_delta > 0.0 && !report.dry_run);
        let names: Vec<_> = report.per_tensor.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, [f.q.name.as_str(), f.v.name.as_str()]);
        assert_eq!(
            read_tensor(&f.model, &f.q.name),
            reference(&f.q, &qa, &qb, 2.0),
            "{dtype} q"
        );
        assert_eq!(
            read_tensor(&f.model, &f.v.name),
            reference(&f.v, &va, &vb, 2.0),
            "{dtype} v"
        );
        assert_eq!(read_tensor(&f.model, &f.norm.name), norm_before);
    }
}

#[test]
fn scale_alpha_override_and_rslora_set_the_effective_scale() {
    // (config extra, alpha override, scale) -> effective scale. r = 4.
    let cases = [
        ("", None, 0.5, 1.0),                        // 0.5 * 8 / 4
        ("", Some(4.0), 1.0, 1.0),                   // --alpha wins over the config
        (r#", "use_rslora": true"#, None, 1.0, 4.0), // 8 / sqrt(4)
    ];
    for (extra, alpha, scale, eff) in cases {
        let f = fixture("F32");
        let (lora, [qa, qb], _) = std_adapter(&f, "F32", Some(&config(8.0, 4, extra)));
        let mut req = request(&f.model, &lora);
        req.alpha = alpha;
        req.scale = scale;
        let report = merge(&req).0.unwrap();
        assert_eq!(report.effective_scale, eff, "{extra} {alpha:?}");
        assert_eq!(
            read_tensor(&f.model, &f.q.name),
            reference(&f.q, &qa, &qb, eff)
        );
    }
}

#[test]
fn alpha_flag_replaces_a_missing_config_and_explicit_config_path_is_used() {
    let f = fixture("F32");
    let (lora, [qa, qb], _) = std_adapter(&f, "F32", None);
    let mut req = request(&f.model, &lora);
    req.alpha = Some(8.0);
    assert_eq!(merge(&req).0.unwrap().effective_scale, 2.0);
    assert_eq!(
        read_tensor(&f.model, &f.q.name),
        reference(&f.q, &qa, &qb, 2.0)
    );

    let g = fixture("F32");
    let (lora, ..) = std_adapter(&g, "F32", None);
    let other = g.dir.path().join("elsewhere.json");
    std::fs::write(&other, config(4.0, 4, "")).unwrap();
    let mut req = request(&g.model, &lora);
    req.adapter_config = Some(other);
    assert_eq!(merge(&req).0.unwrap().effective_scale, 1.0);
    let mut req = request(&g.model, &lora);
    req.adapter_config = Some(g.dir.path().join("missing.json"));
    assert!(matches!(merge(&req).0, Err(SurgeryError::Usage(_))));
}

#[test]
fn missing_alpha_is_a_usage_error_that_explains_the_fix() {
    let f = fixture("F32");
    let (lora, ..) = std_adapter(&f, "F32", None);
    let before = std::fs::read(&f.model).unwrap();
    let (result, guard) = merge(&request(&f.model, &lora));
    let err = result.unwrap_err();
    assert!(matches!(err, SurgeryError::Usage(_)), "{err}");
    let text = err.to_string();
    assert!(
        text.contains("--alpha") && text.contains("adapter_config.json"),
        "{text}"
    );
    assert_eq!((guard.commits, guard.rollbacks), (0, 0));
    assert_eq!(std::fs::read(&f.model).unwrap(), before);
    // A config without lora_alpha is the same case.
    std::fs::write(f.dir.path().join("adapter_config.json"), r#"{"r": 4}"#).unwrap();
    assert!(matches!(
        merge(&request(&f.model, &lora)).0,
        Err(SurgeryError::Usage(_))
    ));
}

#[test]
fn unsupported_adapters_are_refused_before_any_write() {
    let cases = [
        (r#", "use_dora": true"#, "use_dora"),
        (r#", "fan_in_fan_out": true"#, "fan_in_fan_out"),
        (r#", "rank_pattern": {"q_proj": 2}"#, "rank_pattern"),
        (r#", "alpha_pattern": {"q_proj": 2}"#, "alpha_pattern"),
    ];
    for (extra, what) in cases {
        let f = fixture("F32");
        let (lora, ..) = std_adapter(&f, "F32", Some(&config(8.0, 4, extra)));
        let before = std::fs::read(&f.model).unwrap();
        let err = merge(&request(&f.model, &lora)).0.unwrap_err();
        assert!(
            matches!(err, SurgeryError::Usage(_)) && err.to_string().contains(what),
            "{err}"
        );
        assert_eq!(std::fs::read(&f.model).unwrap(), before);
    }
    // Empty patterns are what PEFT writes by default: accepted.
    let f = fixture("F32");
    let (lora, ..) = std_adapter(
        &f,
        "F32",
        Some(&config(
            8.0,
            4,
            r#", "rank_pattern": {}, "alpha_pattern": {}, "use_dora": false"#,
        )),
    );
    merge(&request(&f.model, &lora)).0.unwrap();
}

#[test]
fn embedding_and_dora_keys_are_refused() {
    for key in [
        "base_model.model.model.embed_tokens.lora_embedding_A",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_magnitude_vector.default.weight",
    ] {
        let f = fixture("F32");
        let extra = t(key, "F32", &[4, 10], 7);
        let [qa, qb] = pair(Q, 12, 10, 4, "F32", 10, false);
        let lora = write_adapter(f.dir.path(), &[&qa, &qb, &extra], Some(&config(8.0, 4, "")));
        let before = std::fs::read(&f.model).unwrap();
        let err = merge(&request(&f.model, &lora)).0.unwrap_err();
        assert!(matches!(err, SurgeryError::Usage(_)), "{err}");
        assert!(err.to_string().contains(key), "{err}");
        assert_eq!(std::fs::read(&f.model).unwrap(), before);
    }
}

#[test]
fn modules_to_save_is_a_warning_only() {
    let f = fixture("F32");
    let (lora, ..) = std_adapter(
        &f,
        "F32",
        Some(&config(8.0, 4, r#", "modules_to_save": ["lm_head"]"#)),
    );
    let report = merge(&request(&f.model, &lora)).0.unwrap();
    assert_eq!(report.warnings.len(), 1, "{:?}", report.warnings);
    assert!(report.warnings[0].contains("modules_to_save"));
}

// ------------------------------------------------------------------ key mapping and shapes

#[test]
fn orphan_keys_are_all_listed() {
    let f = fixture("F32");
    let [qa, qb] = pair(Q, 12, 10, 4, "F32", 10, false);
    let [ga, gb] = pair("model.layers.9.mlp.gate_proj", 6, 10, 4, "F32", 30, false); // not in the model
    let [_, vb] = pair(V, 6, 10, 4, "F32", 20, false); // lora_A missing
    let stray = t("base_model.model.model.norm.bias", "F32", &[10], 5);
    let lora = write_adapter(
        f.dir.path(),
        &[&qa, &qb, &ga, &gb, &vb, &stray],
        Some(&config(8.0, 4, "")),
    );
    let text = err_text(merge(&request(&f.model, &lora)).0);
    assert!(
        text.contains(&format!("{V}.lora_B.weight : lora_A absent")),
        "{text}"
    );
    assert!(text.contains("norm.bias"), "{text}");
    // The pair whose target is absent is reported once the orphans are fixed.
    let lora = write_adapter(
        f.dir.path(),
        &[&qa, &qb, &ga, &gb],
        Some(&config(8.0, 4, "")),
    );
    let text = err_text(merge(&request(&f.model, &lora)).0);
    assert!(
        text.contains("gate_proj.weight") && text.contains("absent du modèle"),
        "{text}"
    );
}

#[test]
fn shape_rank_and_dtype_errors_are_listed_and_nothing_is_written() {
    let f = fixture("F32");
    let before = std::fs::read(&f.model).unwrap();
    // q: A has 11 columns (model n = 10). v: rank 3 instead of 4.
    let [qa, qb] = pair(Q, 12, 11, 4, "F32", 10, false);
    let [va, vb] = pair(V, 6, 10, 3, "F32", 20, false);
    let lora = write_adapter(
        f.dir.path(),
        &[&qa, &qb, &va, &vb],
        Some(&config(8.0, 4, "")),
    );
    let text = err_text(merge(&request(&f.model, &lora)).0);
    assert!(
        text.contains(&format!("{Q}.weight")) && text.contains("12x10"),
        "{text}"
    );
    assert!(
        text.contains(&format!("{V}.weight")) && text.contains("rang 3"),
        "{text}"
    );
    assert_eq!(std::fs::read(&f.model).unwrap(), before);

    // B rows and A rows disagree.
    let a = t(
        &format!("base_model.model.{Q}.lora_A.weight"),
        "F32",
        &[4, 10],
        1,
    );
    let b = t(
        &format!("base_model.model.{Q}.lora_B.weight"),
        "F32",
        &[12, 3],
        2,
    );
    let lora = write_adapter(f.dir.path(), &[&a, &b], Some(&config(8.0, 4, "")));
    assert!(err_text(merge(&request(&f.model, &lora)).0).contains("rangs incohérents"));

    // Unsupported target dtype.
    let g = fixture("F32");
    let model = g.dir.path().join("int.safetensors");
    write_i8(&model, &format!("{Q}.weight"), &[12, 10]);
    let [qa, qb] = pair(Q, 12, 10, 4, "F32", 10, false);
    let lora = write_adapter(g.dir.path(), &[&qa, &qb], Some(&config(8.0, 4, "")));
    let text = err_text(merge(&request(&model, &lora)).0);
    assert!(text.contains("I8"), "{text}");
}

fn write_i8(path: &Path, name: &str, shape: &[usize]) {
    let n: usize = shape.iter().product();
    let header = serde_json::json!({name: {"dtype": "I8", "shape": shape, "data_offsets": [0, n]}});
    let mut header = header.to_string();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend(std::iter::repeat_n(0u8, n));
    std::fs::write(path, file).unwrap();
}

#[test]
fn non_finite_adapter_values_are_refused() {
    let f = fixture("F32");
    let [mut qa, qb] = pair(Q, 12, 10, 4, "F32", 10, false);
    qa.vals[3] = f32::NAN;
    let lora = write_adapter(f.dir.path(), &[&qa, &qb], Some(&config(8.0, 4, "")));
    let before = std::fs::read(&f.model).unwrap();
    let text = err_text(merge(&request(&f.model, &lora)).0);
    assert!(text.contains("non finie"), "{text}");
    assert_eq!(std::fs::read(&f.model).unwrap(), before);
}

// ------------------------------------------------------------------ rollback, dry-run, cancel, shards

/// q applies fine (r = 1, alpha 1), v overflows f16: 60000 + 20000 > 65504.
fn overflow_fixture() -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let q = t(&format!("{Q}.weight"), "F16", &[4, 4], 1);
    let v = t(&format!("{V}.weight"), "F16", &[2, 2], 2).with_vals(vec![60000.0; 4]);
    let model = dir.path().join("model.safetensors");
    write_st(&model, &[&q, &v]);
    let [qa, qb] = pair(Q, 4, 4, 1, "F32", 3, false);
    let va = t(
        &format!("base_model.model.{V}.lora_A.weight"),
        "F32",
        &[1, 2],
        4,
    )
    .with_vals(vec![20000.0; 2]);
    let vb = t(
        &format!("base_model.model.{V}.lora_B.weight"),
        "F32",
        &[2, 1],
        5,
    )
    .with_vals(vec![1.0; 2]);
    let lora = write_adapter(dir.path(), &[&qa, &qb, &va, &vb], Some(&config(1.0, 1, "")));
    (dir, model, lora)
}

#[test]
fn nan_or_inf_rolls_back_to_identical_bytes() {
    let (_dir, model, lora) = overflow_fixture();
    let before = std::fs::read(&model).unwrap();
    let req = request(&model, &lora);
    let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
    let err = run_merge(&req, &mut backup, &mut NullObserver).unwrap_err();
    assert!(
        matches!(err, SurgeryError::Failed(_)) && err.to_string().contains("NaN"),
        "{err}"
    );
    assert_eq!(std::fs::read(&model).unwrap(), before);
    // With the recording guard: one rollback, no commit.
    let (result, guard) = merge(&req);
    assert!(result.is_err());
    assert_eq!((guard.commits, guard.rollbacks), (0, 1));
}

#[test]
fn cancellation_between_tensors_rolls_back() {
    let f = fixture("F32");
    let (lora, ..) = std_adapter(&f, "F32", Some(&config(8.0, 4, "")));
    let before = std::fs::read(&f.model).unwrap();
    let req = request(&f.model, &lora);
    let mut backup = BackupSet::create_many(&req.model_shards).unwrap();
    let mut obs = Recorder {
        break_after: Some(1),
        ..Default::default()
    };
    let err = run_merge(&req, &mut backup, &mut obs).unwrap_err();
    assert!(matches!(err, SurgeryError::Cancelled), "{err}");
    assert_eq!(std::fs::read(&f.model).unwrap(), before);
}

#[test]
fn cancel_flag_raised_before_start_writes_nothing() {
    let f = fixture("F32");
    let (lora, ..) = std_adapter(&f, "F32", Some(&config(8.0, 4, "")));
    let before = std::fs::read(&f.model).unwrap();
    let req = request(&f.model, &lora);
    req.cancel.store(true, Ordering::SeqCst);
    assert!(matches!(merge(&req).0, Err(SurgeryError::Cancelled)));
    assert_eq!(std::fs::read(&f.model).unwrap(), before);
}

#[test]
fn dry_run_validates_reports_and_never_writes() {
    let f = fixture("BF16");
    let (lora, ..) = std_adapter(&f, "BF16", Some(&config(8.0, 4, "")));
    let before = std::fs::read(&f.model).unwrap();
    let mut req = request(&f.model, &lora);
    req.dry_run = true;
    let mut obs = Recorder::default();
    let mut guard = CountingGuard::default();
    let report = run_merge(&req, &mut guard, &mut obs).unwrap();
    assert!(report.dry_run && report.tensors == 2 && report.effective_scale == 2.0);
    assert_eq!(obs.events.len(), 2);
    assert_eq!((guard.commits, guard.rollbacks), (0, 0));
    assert_eq!(std::fs::read(&f.model).unwrap(), before);
    // Validation still applies in a dry-run.
    let bad = write_adapter(
        f.dir.path(),
        &[&pair("nope", 2, 2, 1, "F32", 1, false)[0]],
        None,
    );
    req.lora = bad;
    req.alpha = Some(1.0);
    assert!(run_merge(&req, &mut guard, &mut obs).is_err());
}

#[test]
fn sharded_model_is_merged_in_every_shard() {
    let dir = tempfile::tempdir().unwrap();
    let q = t(&format!("{Q}.weight"), "BF16", &[12, 10], 1);
    let v = t(&format!("{V}.weight"), "F16", &[6, 10], 2);
    write_st(&dir.path().join("model-00001.safetensors"), &[&q]);
    write_st(&dir.path().join("model-00002.safetensors"), &[&v]);
    let adapter_dir = tempfile::tempdir().unwrap();
    let [qa, qb] = pair(Q, 12, 10, 4, "F32", 10, false);
    let [va, vb] = pair(V, 6, 10, 4, "F32", 20, true);
    let lora = write_adapter(
        adapter_dir.path(),
        &[&qa, &qb, &va, &vb],
        Some(&config(8.0, 4, "")),
    );
    let req = request(dir.path(), &lora);
    assert_eq!(req.model_shards.len(), 2);
    let (result, guard) = merge(&req);
    result.unwrap();
    assert_eq!(guard.commits, 1);
    assert_eq!(
        read_tensor(&dir.path().join("model-00001.safetensors"), &q.name),
        reference(&q, &qa, &qb, 2.0)
    );
    assert_eq!(
        read_tensor(&dir.path().join("model-00002.safetensors"), &v.name),
        reference(&v, &va, &vb, 2.0)
    );
}

// ------------------------------------------------------------------ extract and round trip

/// A rank-`r` matrix `P Q` (multiples of 1/64, exact in f32).
fn low_rank(m: usize, n: usize, r: usize, seed: u64) -> Vec<f32> {
    let (p, q) = (eighths(seed, m * r), eighths(seed + 7, r * n));
    (0..m * n)
        .map(|idx| {
            (0..r)
                .map(|k| p[(idx / n) * r + k] * q[k * n + idx % n])
                .sum()
        })
        .collect()
}

struct Pair {
    dir: tempfile::TempDir,
    base: PathBuf,
    patched: PathBuf,
    base_t: Vec<T>,
    patched_t: Vec<T>,
}

/// f32 checkpoints whose q and v weights differ by an exactly rank-2 matrix.
fn checkpoints(delta_scale: f32) -> Pair {
    let dir = tempfile::tempdir().unwrap();
    let (bd, pd) = (
        dir.path().join("base-model"),
        dir.path().join("patched-model"),
    );
    std::fs::create_dir_all(&bd).unwrap();
    std::fs::create_dir_all(&pd).unwrap();
    let mut base_t = vec![
        t(&format!("{Q}.weight"), "F32", &[12, 10], 1),
        t(&format!("{V}.weight"), "F32", &[6, 10], 2),
        t("model.embed_tokens.weight", "F32", &[20, 10], 3),
        t("model.norm.weight", "F32", &[10], 4),
    ];
    let mut patched_t = base_t.clone();
    for (i, (m, n)) in [(12, 10), (6, 10)].into_iter().enumerate() {
        let delta = low_rank(m, n, 2, 40 + i as u64);
        let vals = base_t[i]
            .vals
            .iter()
            .zip(&delta)
            .map(|(b, d)| b + delta_scale * d)
            .collect();
        patched_t[i] = patched_t[i].clone().with_vals(vals);
    }
    base_t.sort_by(|a, b| a.name.cmp(&b.name));
    patched_t.sort_by(|a, b| a.name.cmp(&b.name));
    let (base, patched) = (bd.join("model.safetensors"), pd.join("model.safetensors"));
    write_st(&base, &base_t.iter().collect::<Vec<_>>());
    write_st(&patched, &patched_t.iter().collect::<Vec<_>>());
    Pair {
        dir,
        base,
        patched,
        base_t,
        patched_t,
    }
}

fn extract_args(p: &Pair, out: &Path, rank: usize, dtype: FactorDtype) -> LoraExtractArgs {
    LoraExtractArgs {
        base: p.base.parent().unwrap().to_path_buf(),
        patched: p.patched.parent().unwrap().to_path_buf(),
        out: out.to_path_buf(),
        rank,
        dtype,
        target_modules: None,
        seed: 0,
        overwrite: false,
    }
}

fn extract(
    args: &LoraExtractArgs,
) -> Result<weight_surgeon::lora::extract::ExtractReport, SurgeryError> {
    run_extract(args, &mut NullObserver, &AtomicBool::new(false))
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Tolerance of the round trip: the difference is exactly rank 2, so the rank-2 rSVD is exact
/// up to f32 rounding (~1e-6 relative); 1e-4 absolute leaves a wide margin on values < 4.
const ROUND_TRIP_TOL: f64 = 1e-4;

#[test]
fn extract_then_merge_gives_the_patched_weights_up_to_rounding() {
    let p = checkpoints(1.0);
    let out_dir = p.dir.path().join("adapter");
    std::fs::create_dir_all(&out_dir).unwrap();
    let out = out_dir.join("adapter_model.safetensors");
    let report = extract(&extract_args(&p, &out, 2, FactorDtype::F32)).unwrap();
    assert_eq!(
        (report.rank, report.tensors, report.dtype.as_str()),
        (2, 2, "f32")
    );
    assert!(report.mean_rel_error < 1e-5, "{}", report.mean_rel_error);
    assert!(report
        .per_tensor
        .iter()
        .all(|t| t.rel_error < 1e-5 && t.energy > 0.999));

    let names = read_names(&out);
    assert_eq!(
        names,
        [
            format!("base_model.model.{Q}.lora_A.weight"),
            format!("base_model.model.{Q}.lora_B.weight"),
            format!("base_model.model.{V}.lora_A.weight"),
            format!("base_model.model.{V}.lora_B.weight"),
        ]
    );

    // The config written next to the adapter is valid and says what merge needs.
    let cfg: Value = serde_json::from_str(
        &std::fs::read_to_string(out_dir.join("adapter_config.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(cfg["peft_type"], "LORA");
    assert_eq!(
        (&cfg["r"], &cfg["lora_alpha"]),
        (&Value::from(2), &Value::from(2))
    );
    assert_eq!(
        cfg["target_modules"],
        serde_json::json!(["q_proj", "v_proj"])
    );
    assert_eq!(cfg["base_model_name_or_path"], "base-model");
    assert_eq!(cfg["task_type"], "CAUSAL_LM");
    assert_eq!(cfg["bias"], "none");
    for off in ["use_rslora", "use_dora", "fan_in_fan_out"] {
        assert_eq!(cfg[off], false, "{off}");
    }
    assert_eq!(cfg["lora_dropout"], 0.0);

    // Merge into a copy of the base (alpha comes from the generated config).
    let target = p.dir.path().join("target.safetensors");
    std::fs::copy(&p.base, &target).unwrap();
    let merged = merge(&request(&target, &out)).0.unwrap();
    assert_eq!((merged.rank, merged.effective_scale), (2, 1.0));
    for (base, patched) in p.base_t.iter().zip(&p.patched_t) {
        let got = decode("F32", &read_tensor(&target, &base.name));
        let want = patched.stored();
        let selected = base.name.contains("_proj");
        let tol = if selected { ROUND_TRIP_TOL } else { 0.0 };
        assert!(
            max_abs_diff(&got, &want) <= tol,
            "{}: {}",
            base.name,
            max_abs_diff(&got, &want)
        );
        if selected {
            assert!(
                max_abs_diff(&got, &base.stored()) > 1e-3,
                "{} was not modified",
                base.name
            );
        }
    }
}

#[test]
fn extract_is_deterministic_and_overwrite_is_explicit() {
    let p = checkpoints(1.0);
    let (o1, o2) = (
        p.dir.path().join("a.safetensors"),
        p.dir.path().join("b").join("a.safetensors"),
    );
    std::fs::create_dir_all(o2.parent().unwrap()).unwrap();
    extract(&extract_args(&p, &o1, 2, FactorDtype::F32)).unwrap();
    extract(&extract_args(&p, &o2, 2, FactorDtype::F32)).unwrap();
    assert_eq!(std::fs::read(&o1).unwrap(), std::fs::read(&o2).unwrap());
    assert_eq!(
        std::fs::read(o1.with_file_name("adapter_config.json")).unwrap(),
        std::fs::read(o2.with_file_name("adapter_config.json")).unwrap()
    );
    let before = std::fs::read(&o1).unwrap();
    let err = extract(&extract_args(&p, &o1, 2, FactorDtype::F32)).unwrap_err();
    assert!(
        matches!(err, SurgeryError::Usage(_)) && err.to_string().contains("--overwrite"),
        "{err}"
    );
    assert_eq!(std::fs::read(&o1).unwrap(), before);
    let mut args = extract_args(&p, &o1, 2, FactorDtype::F32);
    args.overwrite = true;
    args.seed = 5;
    extract(&args).unwrap();
    // No temporary file is left behind.
    let leftovers = std::fs::read_dir(p.dir.path())
        .unwrap()
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
        .count();
    assert_eq!(leftovers, 0);
}

#[test]
fn extract_in_half_precision_round_trips_within_its_precision() {
    for (dtype, st_name, tol) in [
        (FactorDtype::Bf16, "BF16", 5e-2),
        (FactorDtype::F16, "F16", 5e-3),
    ] {
        let p = checkpoints(1.0);
        let out = p.dir.path().join("a.safetensors");
        let report = extract(&extract_args(&p, &out, 2, dtype)).unwrap();
        assert_eq!(report.dtype, dtype.as_str());
        let bytes = std::fs::read(&out).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        for (_, view) in st.tensors() {
            assert_eq!(format!("{:?}", view.dtype()), st_name);
        }
        let target = p.dir.path().join("target.safetensors");
        std::fs::copy(&p.base, &target).unwrap();
        merge(&request(&target, &out)).0.unwrap();
        for (base, patched) in p
            .base_t
            .iter()
            .zip(&p.patched_t)
            .filter(|(b, _)| b.name.contains("_proj"))
        {
            let got = decode("F32", &read_tensor(&target, &base.name));
            let err = max_abs_diff(&got, &patched.stored());
            assert!(err < tol, "{} {st_name}: {err}", base.name);
        }
    }
}

#[test]
fn f16_overflow_in_extract_is_a_clear_error_with_a_bf16_hint() {
    let dir = tempfile::tempdir().unwrap();
    let (bd, pd) = (dir.path().join("b"), dir.path().join("p"));
    std::fs::create_dir_all(&bd).unwrap();
    std::fs::create_dir_all(&pd).unwrap();
    let base = t(&format!("{Q}.weight"), "F32", &[6, 6], 1).with_vals(vec![0.0; 36]);
    let patched = base.clone().with_vals(vec![1e11; 36]);
    write_st(&bd.join("model.safetensors"), &[&base]);
    write_st(&pd.join("model.safetensors"), &[&patched]);
    let args = LoraExtractArgs {
        base: bd,
        patched: pd,
        out: dir.path().join("a.safetensors"),
        rank: 1,
        dtype: FactorDtype::F16,
        target_modules: None,
        seed: 0,
        overwrite: false,
    };
    let err = extract(&args).unwrap_err();
    assert!(err.to_string().contains("--dtype bf16"), "{err}");
    assert!(!args.out.exists());
    extract(&LoraExtractArgs {
        dtype: FactorDtype::Bf16,
        ..args.clone()
    })
    .unwrap();
}

#[test]
fn extract_lists_layout_differences_and_empty_selections() {
    let p = checkpoints(1.0);
    let out = p.dir.path().join("a.safetensors");
    // --patched misses a tensor, changes a shape and has an extra one.
    let other = p.dir.path().join("other");
    std::fs::create_dir_all(&other).unwrap();
    let wrong = t(&format!("{Q}.weight"), "F32", &[10, 12], 1);
    let extra = t("extra.weight", "F32", &[2, 2], 1);
    write_st(&other.join("model.safetensors"), &[&wrong, &extra]);
    let mut args = extract_args(&p, &out, 2, FactorDtype::F32);
    args.patched = other;
    let text = err_text_x(extract(&args));
    assert!(
        text.contains(&format!("{V}.weight : absent de --patched")),
        "{text}"
    );
    assert!(
        text.contains(&format!("{Q}.weight")) && text.contains("[12, 10]"),
        "{text}"
    );
    assert!(text.contains("extra.weight : absent de --base"), "{text}");
    assert!(!out.exists());

    let mut args = extract_args(&p, &out, 2, FactorDtype::F32);
    args.target_modules = Some("nothing_matches".into());
    assert!(err_text_x(extract(&args)).contains("--target-modules"));
    args.target_modules = Some("(".into());
    assert!(matches!(extract(&args), Err(SurgeryError::Usage(_))));
    args.target_modules = Some("q_proj".into());
    assert_eq!(extract(&args).unwrap().tensors, 1);
    args.overwrite = true;
    args.rank = 0;
    assert!(matches!(extract(&args), Err(SurgeryError::Usage(_))));
    // A rank above the smallest dimension is lowered, with a warning.
    args.rank = 50;
    let report = extract(&args).unwrap();
    assert_eq!(report.rank, 10);
    assert_eq!(report.warnings.len(), 1);
}

fn err_text_x<T>(r: Result<T, SurgeryError>) -> String {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e.to_string(),
    }
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

#[test]
fn cli_merge_reports_data_lora_and_keeps_a_backup() {
    let f = fixture("BF16");
    let (lora, [qa, qb], _) = std_adapter(&f, "BF16", Some(&config(8.0, 4, "")));
    let before = std::fs::read(&f.model).unwrap();
    let (code, v) = run_json(
        &deps(),
        &[
            "lora",
            "merge",
            "--model",
            s(&f.model),
            "--lora",
            s(&lora),
            "--scale",
            "0.5",
        ],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["command"], "lora merge");
    let lora_data = &v["data"]["lora"];
    assert_eq!(
        (
            &lora_data["rank"],
            &lora_data["effective_scale"],
            &lora_data["tensors"]
        ),
        (&Value::from(4), &Value::from(1.0), &Value::from(2))
    );
    assert_eq!(lora_data["per_tensor"][0]["name"], f.q.name);
    let backups = backups_in(f.dir.path());
    assert_eq!(backups.len(), 1);
    assert_eq!(v["data"]["backups"].as_array().unwrap().len(), 1);
    assert_eq!(std::fs::read(&backups[0]).unwrap(), before);
    assert_eq!(
        read_tensor(&f.model, &f.q.name),
        reference(&f.q, &qa, &qb, 1.0)
    );
}

#[test]
fn cli_alpha_flag_no_backup_and_backup_dir() {
    let f = fixture("F32");
    let (lora, ..) = std_adapter(&f, "F32", None);
    let (code, v) = run_json(
        &deps(),
        &["lora", "merge", "--model", s(&f.model), "--lora", s(&lora)],
    );
    assert_eq!(code, exit::INVALID_INPUT, "{v}");
    assert_eq!(v["error"]["kind"], "invalid_arguments");
    assert!(v["error"].to_string().contains("--alpha"), "{v}");
    assert!(backups_in(f.dir.path()).is_empty());

    let (code, v) = run_json(
        &deps(),
        &[
            "lora",
            "merge",
            "--model",
            s(&f.model),
            "--lora",
            s(&lora),
            "--alpha",
            "8",
            "--no-backup",
        ],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert!(backups_in(f.dir.path()).is_empty());

    let g = fixture("F32");
    let (lora, ..) = std_adapter(&g, "F32", None);
    let bdir = g.dir.path().join("bk");
    std::fs::create_dir_all(&bdir).unwrap();
    let (code, v) = run_json(
        &deps(),
        &[
            "lora",
            "merge",
            "--model",
            s(&g.model),
            "--lora",
            s(&lora),
            "--alpha",
            "8",
            "--backup-dir",
            s(&bdir),
        ],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(backups_in(&bdir).len(), 1);
    assert!(backups_in(g.dir.path()).is_empty());
}

#[test]
fn cli_dry_run_changes_nothing_and_creates_no_backup() {
    let f = fixture("F32");
    let (lora, ..) = std_adapter(&f, "F32", Some(&config(8.0, 4, "")));
    let before = std::fs::read(&f.model).unwrap();
    let (code, v) = run_json(
        &deps(),
        &[
            "lora",
            "merge",
            "--model",
            s(&f.model),
            "--lora",
            s(&lora),
            "--dry-run",
        ],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["data"]["lora"]["dry_run"], true);
    assert_eq!(std::fs::read(&f.model).unwrap(), before);
    assert!(backups_in(f.dir.path()).is_empty());
}

#[test]
fn cli_overflow_rolls_back_with_exit_3_and_refusals_exit_2() {
    let (dir, model, lora) = overflow_fixture();
    let before = std::fs::read(&model).unwrap();
    let (code, v) = run_json(
        &deps(),
        &["lora", "merge", "--model", s(&model), "--lora", s(&lora)],
    );
    assert_eq!(code, exit::EXECUTION_FAILED, "{v}");
    assert_eq!(v["error"]["kind"], "execution_failed");
    assert_eq!(std::fs::read(&model).unwrap(), before);
    assert_eq!(backups_in(dir.path()).len(), 1);

    let f = fixture("F32");
    let (lora, ..) = std_adapter(&f, "F32", Some(&config(8.0, 4, r#", "use_dora": true"#)));
    let (code, v) = run_json(
        &deps(),
        &["lora", "merge", "--model", s(&f.model), "--lora", s(&lora)],
    );
    assert_eq!(code, exit::INVALID_INPUT, "{v}");
    assert_eq!(v["error"]["kind"], "invalid_arguments");
}

#[test]
fn cli_sharded_merge_backs_up_every_shard_and_interrupt_exits_1() {
    let dir = tempfile::tempdir().unwrap();
    let q = t(&format!("{Q}.weight"), "BF16", &[12, 10], 1);
    let v = t(&format!("{V}.weight"), "BF16", &[6, 10], 2);
    write_st(&dir.path().join("model-00001.safetensors"), &[&q]);
    write_st(&dir.path().join("model-00002.safetensors"), &[&v]);
    let adir = tempfile::tempdir().unwrap();
    let [qa, qb] = pair(Q, 12, 10, 4, "BF16", 10, false);
    let [va, vb] = pair(V, 6, 10, 4, "BF16", 20, false);
    let lora = write_adapter(
        adir.path(),
        &[&qa, &qb, &va, &vb],
        Some(&config(8.0, 4, "")),
    );

    let interrupted = deps();
    interrupted.interrupted.store(true, Ordering::SeqCst);
    let (code, out) = run_json(
        &interrupted,
        &[
            "lora",
            "merge",
            "--model",
            s(dir.path()),
            "--lora",
            s(&lora),
        ],
    );
    assert_eq!(code, exit::GENERAL, "{out}");
    assert_eq!(out["error"]["kind"], "cancelled");
    assert!(backups_in(dir.path()).is_empty());

    let (code, out) = run_json(
        &deps(),
        &[
            "lora",
            "merge",
            "--model",
            s(dir.path()),
            "--lora",
            s(&lora),
        ],
    );
    assert_eq!(code, exit::OK, "{out}");
    assert_eq!(backups_in(dir.path()).len(), 2);
    assert_eq!(
        read_tensor(&dir.path().join("model-00002.safetensors"), &v.name),
        reference(&v, &va, &vb, 2.0)
    );
}

#[test]
fn cli_extract_writes_the_adapter_and_honours_overwrite() {
    let p = checkpoints(1.0);
    let out = p.dir.path().join("adapter_model.safetensors");
    let (base, patched) = (p.base.parent().unwrap(), p.patched.parent().unwrap());
    let argv = [
        "lora",
        "extract",
        "--base",
        s(base),
        "--patched",
        s(patched),
        "--out",
        s(&out),
        "--rank",
        "2",
    ];
    let (code, v) = run_json(&deps(), &argv);
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["command"], "lora extract");
    assert_eq!(
        (&v["data"]["lora"]["rank"], &v["data"]["lora"]["tensors"]),
        (&Value::from(2), &Value::from(2))
    );
    assert_eq!(v["data"]["lora"]["dtype"], "f32");
    assert!(out.is_file() && p.dir.path().join("adapter_config.json").is_file());

    let (code, v) = run_json(&deps(), &argv);
    assert_eq!(code, exit::INVALID_INPUT, "{v}");
    assert_eq!(v["error"]["kind"], "invalid_arguments");
    let with = [&argv[..], &["--overwrite", "--dtype", "bf16"]].concat();
    let (code, v) = run_json(&deps(), &with);
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["data"]["lora"]["dtype"], "bf16");

    // The adapter just written merges through the CLI.
    let target = p.dir.path().join("target.safetensors");
    std::fs::copy(&p.base, &target).unwrap();
    let (code, v) = run_json(
        &deps(),
        &[
            "lora",
            "merge",
            "--model",
            s(&target),
            "--lora",
            s(&out),
            "--no-backup",
        ],
    );
    assert_eq!(code, exit::OK, "{v}");
    assert_eq!(v["data"]["lora"]["lora_alpha"], 2.0);
}

#[test]
fn cli_extract_layout_mismatch_is_exit_2_and_interrupt_exit_1() {
    let p = checkpoints(1.0);
    let out = p.dir.path().join("a.safetensors");
    let lone = p.dir.path().join("lone");
    std::fs::create_dir_all(&lone).unwrap();
    write_st(&lone.join("model.safetensors"), &[&p.base_t[0]]);
    let (base, patched) = (p.base.parent().unwrap(), p.patched.parent().unwrap());
    let (code, v) = run_json(
        &deps(),
        &[
            "lora",
            "extract",
            "--base",
            s(base),
            "--patched",
            s(&lone),
            "--out",
            s(&out),
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT, "{v}");
    assert!(!out.exists());

    let interrupted = deps();
    interrupted.interrupted.store(true, Ordering::SeqCst);
    let (code, v) = run_json(
        &interrupted,
        &[
            "lora",
            "extract",
            "--base",
            s(base),
            "--patched",
            s(patched),
            "--out",
            s(&out),
        ],
    );
    assert_eq!(code, exit::GENERAL, "{v}");
    assert!(!out.exists());
}
