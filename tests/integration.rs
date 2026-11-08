//! End-to-end flows through `app::run_with` with fake hardware, processes and generator,
//! plus smoke tests of the real binary.

use std::cell::RefCell;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde_json::{json, Value};
use weight_surgeon::app::{self, exit, AppDeps, EngineBackend};
use weight_surgeon::eval::{
    CommandOutput, CommandRunner, EvalError, GenerationOutcome, TextGenerator,
};
use weight_surgeon::hardware::{GpuInfo, HardwareProbe, MemoryInfo};
use weight_surgeon::patch::{Axis, OperationKind, PatchBackend};
use weight_surgeon::ui::{ColorMode, UiConfig};

// ------------------------------------------------------------------ fixtures

const GATE: &str = "model.layers.0.mlp.gate_proj.weight";
const DOWN: &str = "model.layers.0.mlp.down_proj.weight";
const GATE1: &str = "model.layers.1.mlp.gate_proj.weight";

/// F32 SafeTensors file written by hand (header padded to 8 bytes).
fn write_safetensors(path: &Path, tensors: &[(&str, [usize; 2], Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        values
            .iter()
            .for_each(|v| data.extend_from_slice(&v.to_le_bytes()));
        header.insert(
            (*name).into(),
            json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
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

/// Reads one F32 tensor back, independently of the engine.
fn read_f32(path: &Path, name: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap();
    let len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: Value = serde_json::from_slice(&bytes[8..8 + len]).unwrap();
    let offsets = &header[name]["data_offsets"];
    let (start, end) = (
        offsets[0].as_u64().unwrap() as usize,
        offsets[1].as_u64().unwrap() as usize,
    );
    bytes[8 + len + start..8 + len + end]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Single-shard model: gate [2,3] = 1..6, down [3,2] = 10..60.
fn model(dir: &Path) -> PathBuf {
    let path = dir.join("model.safetensors");
    write_safetensors(
        &path,
        &[
            (GATE, [2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            (DOWN, [3, 2], vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
        ],
    );
    path
}

fn write_patch(dir: &Path, ops: Value, backup: bool) -> PathBuf {
    std::fs::write(dir.join("v3.json"), "[1, 1, 1]").unwrap();
    std::fs::write(dir.join("v2.json"), "[1, 1]").unwrap();
    let path = dir.join("patch.json");
    let patch = json!({"schema_version": 1, "backup": backup, "operations": ops});
    std::fs::write(&path, patch.to_string()).unwrap();
    path
}

fn op(id: &str, layers: &str, proj: &str, vector: &str, axis: &str) -> Value {
    json!({"id": id, "layers": layers, "projections": [proj], "operation": "subtract",
           "vector_file": vector, "alpha": 0.5, "axis": axis})
}

fn backups_in(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.contains(".bak-"))
        .collect()
}

struct FakeProbe {
    available: u64,
}

impl HardwareProbe for FakeProbe {
    fn system_ram(&self) -> weight_surgeon::hardware::Result<MemoryInfo> {
        Ok(MemoryInfo {
            total_bytes: self.available,
            available_bytes: self.available,
        })
    }
    fn gpus(&self) -> weight_surgeon::hardware::Result<Vec<GpuInfo>> {
        Ok(vec![])
    }
}

/// Writes the GGUF each tool would produce and records the calls.
#[derive(Default)]
struct MockRunner {
    calls: Rc<RefCell<Vec<Vec<String>>>>,
    fail_call: Option<usize>,
}

impl CommandRunner for MockRunner {
    fn run(
        &self,
        _program: &Path,
        args: &[String],
        _cwd: Option<&Path>,
    ) -> weight_surgeon::eval::Result<CommandOutput> {
        let index = self.calls.borrow().len();
        self.calls.borrow_mut().push(args.to_vec());
        let status_code = if self.fail_call == Some(index) {
            1
        } else {
            let out = match args.iter().position(|a| a == "--outfile") {
                Some(i) => &args[i + 1],
                None => &args[1],
            };
            std::fs::write(out, b"GGUF\x03\x00\x00\x00").unwrap();
            0
        };
        Ok(CommandOutput {
            status_code,
            stdout: String::new(),
            stderr: "boom".into(),
        })
    }
}

struct Repeat(&'static str, usize);

impl TextGenerator for Repeat {
    fn generate(
        &mut self,
        _prompt: &str,
        _max_new_tokens: u32,
        on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> weight_surgeon::eval::Result<GenerationOutcome> {
        let mut text = String::new();
        for _ in 0..self.1 {
            text.push_str(self.0);
            if on_chunk(self.0).is_break() {
                return Ok(GenerationOutcome {
                    text,
                    stopped_early: true,
                    duration_ms: 0,
                });
            }
        }
        Ok(GenerationOutcome {
            text,
            stopped_early: false,
            duration_ms: 0,
        })
    }
}

fn deps_with(runner: MockRunner, generator: fn() -> Box<dyn TextGenerator>) -> AppDeps {
    AppDeps {
        probe: Box::new(FakeProbe {
            available: 64 << 30,
        }),
        runner: Box::new(runner),
        generator_factory: Box::new(move |_, _| Ok(generator())),
        interrupted: Arc::new(AtomicBool::new(false)),
        ui: UiConfig {
            out: ColorMode::Plain,
            err: ColorMode::Plain,
            err_is_tty: false,
        },
    }
}

fn healthy() -> Box<dyn TextGenerator> {
    Box::new(Repeat("La capitale de la France est Paris. ", 1))
}

fn deps() -> AppDeps {
    deps_with(MockRunner::default(), healthy)
}

/// Runs a command line; returns (exit code, stdout, stderr).
fn run(deps: &AppDeps, args: &[&str]) -> (i32, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let argv = std::iter::once("weight-surgeon").chain(args.iter().copied());
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
    let mut argv = vec!["--json"];
    argv.extend_from_slice(args);
    let (code, out, _) = run(deps, &argv);
    let value: Value = serde_json::from_str(&out).unwrap_or_else(|e| panic!("{e}: {out}"));
    assert_eq!(value["exit_code"], code, "envelope and exit code agree");
    assert_eq!(value["ok"], code == 0);
    (code, value)
}

fn s(p: &Path) -> &str {
    p.to_str().unwrap()
}

// ------------------------------------------------------------------ cli & output

#[test]
fn usage_errors_are_code_2_and_json_when_asked() {
    let (code, v) = run_json(&deps(), &["frobnicate"]);
    assert_eq!(code, exit::INVALID_INPUT);
    assert_eq!(v["command"], Value::Null);
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("frobnicate"));

    let (code, out, err) = run(&deps(), &["apply", "--model", "m"]);
    assert_eq!(code, 2);
    assert!(out.is_empty() && err.contains("--patch"));

    let (code, out, _) = run(&deps(), &["--help"]);
    assert_eq!(code, 0);
    assert!(out.contains("apply"));
}

struct BrokenPipe;

impl std::io::Write for BrokenPipe {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::BrokenPipe.into())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Err(std::io::ErrorKind::BrokenPipe.into())
    }
}

#[test]
fn closed_pipe_never_panics_and_keeps_the_exit_code() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let cli = app::parse_cli(
        ["ws", "--json", "inspect", "--model", s(&m)],
        &mut BrokenPipe,
        &mut BrokenPipe,
    )
    .unwrap();
    assert_eq!(
        app::run_with(cli, &deps(), &mut BrokenPipe, &mut BrokenPipe),
        0
    );
}

#[test]
fn panic_becomes_code_70_with_an_envelope() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("m.gguf"), "GGUF").unwrap();
    std::fs::write(dir.path().join("p.txt"), "Bonjour").unwrap();
    let mut deps = deps();
    deps.generator_factory = Box::new(|_, _| panic!("factory exploded"));
    let (code, v) = run_json(
        &deps,
        &[
            "eval",
            "--model",
            s(&dir.path().join("m.gguf")),
            "--prompts",
            s(&dir.path().join("p.txt")),
        ],
    );
    assert_eq!(code, exit::INTERNAL);
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("factory exploded"));
}

// ------------------------------------------------------------------ check

#[test]
fn check_verdict_drives_the_exit_code() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let (code, v) = run_json(&deps(), &["check", "--model", s(&m)]);
    assert_eq!(code, 0);
    assert_eq!(v["data"]["verdict"]["kind"], "fits_in_ram_only");

    let mut poor = deps();
    poor.probe = Box::new(FakeProbe { available: 0 });
    let (code, v) = run_json(&poor, &["check", "--model", s(&m)]);
    assert_eq!(code, exit::GENERAL);
    assert_eq!(v["data"]["verdict"]["kind"], "insufficient", "{v}");
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("insuffisantes"));

    let (code, _) = run_json(
        &deps(),
        &["check", "--model", s(&m), "--policy-ram-margin-gib", "-1"],
    );
    assert_eq!(code, exit::INVALID_INPUT);
    let (code, _, err) = run(
        &deps(),
        &["check", "--model", s(&dir.path().join("absent"))],
    );
    assert_eq!(code, exit::GENERAL);
    assert!(err.starts_with("erreur : "));
}

// ------------------------------------------------------------------ inspect

#[test]
fn inspect_filters_and_streams_stats() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let (code, v) = run_json(
        &deps(),
        &[
            "inspect",
            "--model",
            s(&m),
            "--proj",
            "gate",
            "--show-stats",
        ],
    );
    assert_eq!(code, 0);
    let tensors = v["data"]["tensors"].as_array().unwrap();
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0]["name"], GATE);
    assert_eq!(tensors[0]["shape"], json!([2, 3]));
    let stats = &tensors[0]["stats"];
    assert_eq!(
        (stats["min"].as_f64(), stats["max"].as_f64()),
        (Some(1.0), Some(6.0))
    );
    assert_eq!(stats["mean"], 3.5);
    assert!((stats["l2"].as_f64().unwrap() - 91f64.sqrt()).abs() < 1e-9);

    let (code, out, _) = run(&deps(), &["inspect", "--model", s(&m), "--layers", "5-9"]);
    assert_eq!(code, 0);
    assert!(out.contains("aucun tenseur"));
}

// ------------------------------------------------------------------ apply

#[test]
fn apply_shifts_exactly_and_keeps_a_backup() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let original = std::fs::read(&m).unwrap();
    let patch = write_patch(
        dir.path(),
        json!([
            op("g", "0", "gate_proj", "v3.json", "rows"),
            op("d", "all", "down_proj", "v3.json", "cols")
        ]),
        true,
    );

    let (code, v) = run_json(
        &deps(),
        &["apply", "--model", s(&m), "--patch", s(&patch), "--dry-run"],
    );
    assert_eq!(code, 0);
    assert_eq!(v["data"]["report"]["dry_run"], true);
    assert_eq!(
        std::fs::read(&m).unwrap(),
        original,
        "dry run writes nothing"
    );
    assert!(backups_in(dir.path()).is_empty());

    let (code, v) = run_json(&deps(), &["apply", "--model", s(&m), "--patch", s(&patch)]);
    assert_eq!(code, 0, "{v}");
    assert_eq!(read_f32(&m, GATE), [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]);
    assert_eq!(read_f32(&m, DOWN), [9.5, 19.5, 29.5, 39.5, 49.5, 59.5]);
    let backups = backups_in(dir.path());
    assert_eq!(backups.len(), 1);
    assert_eq!(
        std::fs::read(dir.path().join(&backups[0])).unwrap(),
        original
    );
    let listed = PathBuf::from(v["data"]["backups"][0].as_str().unwrap());
    assert!(
        listed.is_absolute() && listed.ends_with(&backups[0]),
        "{listed:?}"
    );
    assert_eq!(
        v["data"]["report"]["operations"][0]["tensors"][0]["stats"]["elements_modified"],
        6
    );
}

#[test]
fn failure_mid_patch_rolls_back_to_identical_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let original = std::fs::read(&m).unwrap();
    // The second operation has a vector of the wrong length for its axis.
    let patch = write_patch(
        dir.path(),
        json!([
            op("ok", "0", "gate_proj", "v3.json", "rows"),
            op("bad", "0", "down_proj", "v2.json", "cols")
        ]),
        true,
    );
    let (code, v) = run_json(&deps(), &["apply", "--model", s(&m), "--patch", s(&patch)]);
    assert_eq!(code, exit::EXECUTION_FAILED, "{v}");
    assert!(v["error"]["message"].as_str().unwrap().contains("bad"));
    assert_eq!(
        std::fs::read(&m).unwrap(),
        original,
        "restored bit for bit while still mapped"
    );
}

#[test]
fn failure_without_backup_is_critical() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let patch = write_patch(
        dir.path(),
        json!([
            op("ok", "0", "gate_proj", "v3.json", "rows"),
            op("bad", "0", "down_proj", "v2.json", "cols")
        ]),
        false,
    );
    let (code, _) = run_json(&deps(), &["apply", "--model", s(&m), "--patch", s(&patch)]);
    assert_eq!(code, exit::CRITICAL);
}

#[test]
fn invalid_or_conflicting_patches_are_code_2_with_issues() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let mut bad = op("g", "0", "gate_proj", "v3.json", "rows");
    bad["alpha"] = json!(500);
    let patch = write_patch(dir.path(), json!([bad]), true);
    let (code, v) = run_json(&deps(), &["apply", "--model", s(&m), "--patch", s(&patch)]);
    assert_eq!(code, exit::INVALID_INPUT);
    assert_eq!(v["error"]["issues"][0]["path"], "operations[0].alpha");

    let patch = write_patch(
        dir.path(),
        json!([op("g", "0", "gate_proj", "v3.json", "rows")]),
        true,
    );
    let (code, v) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&m),
            "--patch",
            s(&patch),
            "--no-backup",
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT);
    assert_eq!(v["error"]["issues"][0]["path"], "backup");
}

#[test]
fn patch_with_no_target_fails_before_any_backup() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let original = std::fs::read(&m).unwrap();
    let patch = write_patch(
        dir.path(),
        json!([op("g", "50-60", "gate_proj", "v3.json", "rows")]),
        true,
    );
    let (code, v) = run_json(&deps(), &["apply", "--model", s(&m), "--patch", s(&patch)]);
    assert_eq!(code, exit::EXECUTION_FAILED);
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("aucun tenseur"));
    assert!(
        backups_in(dir.path()).is_empty(),
        "no costly copy for a doomed patch"
    );
    assert_eq!(std::fs::read(&m).unwrap(), original);
}

#[test]
fn ctrl_c_stops_before_writing() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let original = std::fs::read(&m).unwrap();
    let patch = write_patch(
        dir.path(),
        json!([op("g", "0", "gate_proj", "v3.json", "rows")]),
        true,
    );
    let deps = deps();
    deps.interrupted.store(true, Ordering::SeqCst);
    let (code, v) = run_json(&deps, &["apply", "--model", s(&m), "--patch", s(&patch)]);
    assert_eq!(code, exit::EXECUTION_FAILED);
    assert!(v["error"]["message"].as_str().unwrap().contains("Ctrl-C"));
    assert_eq!(std::fs::read(&m).unwrap(), original);

    // Inside apply_patch, the backend refuses the next shift, which triggers the rollback.
    let flag = Arc::new(AtomicBool::new(false));
    let mut backend = EngineBackend::open(std::slice::from_ref(&m), Arc::clone(&flag)).unwrap();
    flag.store(true, Ordering::SeqCst);
    let err = backend
        .shift_tensor(
            GATE,
            &[1.0, 1.0, 1.0],
            0.5,
            OperationKind::Subtract,
            Axis::Rows,
        )
        .unwrap_err();
    assert!(err.to_string().contains("Ctrl-C"));
    drop(backend);
    assert_eq!(std::fs::read(&m).unwrap(), original);
}

#[test]
fn sharded_model_backs_up_only_touched_shards() {
    let dir = tempfile::tempdir().unwrap();
    let (a, b) = (
        dir.path().join("model-00001-of-00002.safetensors"),
        dir.path().join("model-00002-of-00002.safetensors"),
    );
    write_safetensors(&a, &[(GATE, [2, 3], vec![1.0; 6])]);
    write_safetensors(&b, &[(GATE1, [2, 3], vec![2.0; 6])]);
    let index = json!({"weight_map": {GATE: "model-00001-of-00002.safetensors", GATE1: "model-00002-of-00002.safetensors"}});
    std::fs::write(
        dir.path().join("model.safetensors.index.json"),
        index.to_string(),
    )
    .unwrap();
    let shard_a = std::fs::read(&a).unwrap();

    let patch = write_patch(
        dir.path(),
        json!([op("g1", "1", "gate_proj", "v3.json", "rows")]),
        true,
    );
    let (code, v) = run_json(
        &deps(),
        &["apply", "--model", s(dir.path()), "--patch", s(&patch)],
    );
    assert_eq!(code, 0, "{v}");
    assert_eq!(read_f32(&b, GATE1), [1.5; 6]);
    assert_eq!(std::fs::read(&a).unwrap(), shard_a);
    let backups = backups_in(dir.path());
    assert_eq!(backups.len(), 1);
    assert!(backups[0].starts_with("model-00002"), "{backups:?}");

    let (code, v) = run_json(&deps(), &["inspect", "--model", s(dir.path())]);
    assert_eq!(code, 0);
    assert_eq!(v["data"]["tensors"].as_array().unwrap().len(), 2);
}

// ------------------------------------------------------------------ eval

struct EvalFixture {
    dir: tempfile::TempDir,
    llama: PathBuf,
}

fn eval_fixture() -> EvalFixture {
    let dir = tempfile::tempdir().unwrap();
    model(dir.path());
    std::fs::write(
        dir.path().join("prompts.txt"),
        "Capitale de la France ?\nBonjour\n",
    )
    .unwrap();
    let llama = dir.path().join("llama.cpp");
    std::fs::create_dir_all(llama.join("build/bin")).unwrap();
    std::fs::write(llama.join("convert_hf_to_gguf.py"), "").unwrap();
    std::fs::write(
        llama.join(format!(
            "build/bin/llama-quantize{}",
            std::env::consts::EXE_SUFFIX
        )),
        "",
    )
    .unwrap();
    EvalFixture { dir, llama }
}

impl EvalFixture {
    fn args(&self, extra: &[&str]) -> Vec<String> {
        let d = self.dir.path();
        let mut args: Vec<String> = [
            "eval",
            "--model",
            s(&d.join("model.safetensors")),
            "--prompts",
            s(&d.join("prompts.txt")),
            "--llama-cpp-dir",
            s(&self.llama),
        ]
        .map(String::from)
        .to_vec();
        args.extend(extra.iter().map(|a| a.to_string()));
        args
    }
}

fn run_args(deps: &AppDeps, args: &[String]) -> (i32, Value) {
    run_json(deps, &args.iter().map(String::as_str).collect::<Vec<_>>())
}

fn no_temp_gguf_left(dir: &Path) -> bool {
    std::fs::read_dir(dir).unwrap().all(|e| {
        !e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".weight-surgeon-eval")
    })
}

#[test]
fn eval_healthy_model_converts_evaluates_and_exports() {
    let f = eval_fixture();
    let out = f.dir.path().join("final.gguf");
    let deps = deps();
    let (code, v) = run_args(
        &deps,
        &f.args(&["--export-gguf", s(&out), "--quant", "q4_k_m"]),
    );
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["report"]["verdict"], "healthy");
    assert_eq!(v["data"]["export"]["final_gguf"], s(&out));
    assert!(std::fs::read(&out).unwrap().starts_with(b"GGUF"));
    assert!(
        no_temp_gguf_left(f.dir.path()),
        "temporary f16 GGUF removed"
    );
}

#[test]
fn eval_broken_model_is_code_5_and_never_exported() {
    let f = eval_fixture();
    let out = f.dir.path().join("final.gguf");
    let deps = deps_with(MockRunner::default(), || {
        Box::new(Repeat("Wait... ", 10_000))
    });
    let (code, v) = run_args(
        &deps,
        &f.args(&["--export-gguf", s(&out), "--allow-degraded"]),
    );
    assert_eq!(code, exit::EVAL_UNHEALTHY);
    assert_eq!(v["data"]["report"]["verdict"], "broken");
    assert_eq!(v["data"]["export"], Value::Null);
    assert!(!out.exists());
    assert!(no_temp_gguf_left(f.dir.path()));
}

#[test]
fn eval_export_failure_is_code_6() {
    let f = eval_fixture();
    let runner = MockRunner {
        fail_call: Some(1),
        ..Default::default()
    };
    let deps = deps_with(runner, healthy);
    let (code, v) = run_args(
        &deps,
        &f.args(&["--export-gguf", s(&f.dir.path().join("final.gguf"))]),
    );
    assert_eq!(code, exit::EXPORT_FAILED, "{v}");
    assert_eq!(
        v["data"]["report"]["verdict"], "healthy",
        "the report survives the failure"
    );
}

#[test]
fn eval_input_errors_are_code_2_before_any_command() {
    let f = eval_fixture();
    let runner = MockRunner::default();
    let calls = Rc::clone(&runner.calls);
    let deps = deps_with(runner, healthy);
    let (code, _) = run_args(
        &deps,
        &f.args(&["--export-gguf", "x.gguf", "--quant", "Q4_K_M;rm -rf /"]),
    );
    assert_eq!(code, exit::INVALID_INPUT);

    let d = f.dir.path();
    let (model, prompts) = (d.join("model.safetensors"), d.join("prompts.txt"));
    let no_llama = ["eval", "--model", s(&model), "--prompts", s(&prompts)];
    let (code, v) = run_json(&deps, &no_llama);
    assert_eq!(code, exit::INVALID_INPUT, "{v}");

    std::fs::write(d.join("empty.txt"), "").unwrap();
    let (code, _) = run_json(
        &deps,
        &[
            "eval",
            "--model",
            s(&d.join("model.safetensors")),
            "--prompts",
            s(&d.join("empty.txt")),
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT);
    assert!(
        calls.borrow().is_empty(),
        "nothing converted for invalid input"
    );
}

#[test]
fn eval_gguf_model_skips_conversion_and_ctrl_c_aborts() {
    let f = eval_fixture();
    let gguf = f.dir.path().join("m.gguf");
    std::fs::write(&gguf, "GGUF").unwrap();
    let prompts = f.dir.path().join("prompts.txt");
    let args = ["eval", "--model", s(&gguf), "--prompts", s(&prompts)];
    let (code, v) = run_json(&deps(), &args);
    assert_eq!(code, 0, "{v}");

    let deps = deps();
    deps.interrupted.store(true, Ordering::SeqCst);
    let (code, v) = run_json(&deps, &args);
    assert_eq!(code, exit::GENERAL);
    assert!(v["error"]["message"].as_str().unwrap().contains("Ctrl-C"));
}

#[test]
fn generator_factory_errors_are_general_errors() {
    let f = eval_fixture();
    let gguf = f.dir.path().join("m.gguf");
    std::fs::write(&gguf, "GGUF").unwrap();
    let mut deps = deps();
    deps.generator_factory = Box::new(|_, _| {
        Err(EvalError::BinaryNotFound {
            name: "llama-cli".into(),
            hint: "LLAMA_CPP_DIR".into(),
        })
    });
    let (code, v) = run_json(
        &deps,
        &[
            "eval",
            "--model",
            s(&gguf),
            "--prompts",
            s(&f.dir.path().join("prompts.txt")),
        ],
    );
    assert_eq!(code, exit::GENERAL);
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("llama-cli"));
}

// ------------------------------------------------------------------ real binary

fn binary() -> Command {
    Command::new(env!("CARGO_BIN_EXE_surgeon"))
}

#[test]
fn binary_smoke() {
    let out = binary().args(["--json", "nope"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["ok"], false);

    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let out = binary()
        .args(["--json", "inspect", "--show-stats", "--model", s(&m)])
        .output()
        .unwrap();
    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["data"]["tensors"].as_array().unwrap().len(), 2);

    // No llama.cpp anywhere: a clear error, not a hang.
    std::fs::write(dir.path().join("m.gguf"), "GGUF").unwrap();
    std::fs::write(dir.path().join("p.txt"), "Bonjour").unwrap();
    let out = binary()
        .args([
            "eval",
            "--model",
            s(&dir.path().join("m.gguf")),
            "--prompts",
            s(&dir.path().join("p.txt")),
        ])
        .env("PATH", "")
        .env_remove("LLAMA_CPP_DIR")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("introuvable"));
}

// ------------------------------------------------------------------ terminal identity

use weight_surgeon::ui::{strip_ansi, Ui};

fn ui_deps(mode: ColorMode, err_is_tty: bool) -> AppDeps {
    let mut deps = deps();
    deps.ui = UiConfig {
        out: mode,
        err: mode,
        err_is_tty,
    };
    deps
}

fn separator_line() -> String {
    "─".repeat(Ui::new(ColorMode::Plain).width())
}

/// One model + patch per directory: (dir, model, patch). `fail` makes op 2 invalid.
fn apply_case(fail: bool) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let second = match fail {
        true => op("d", "0", "down_proj", "v2.json", "cols"),
        false => op("d", "0", "down_proj", "v3.json", "cols"),
    };
    let patch = write_patch(
        dir.path(),
        json!([op("g", "0", "gate_proj", "v3.json", "rows"), second]),
        true,
    );
    (dir, m, patch)
}

fn run_args_raw(deps: &AppDeps, args: &[String]) -> (i32, String, String) {
    run(deps, &args.iter().map(String::as_str).collect::<Vec<_>>())
}

#[test]
fn json_output_is_byte_clean_for_every_command_even_with_truecolor() {
    let deps = ui_deps(ColorMode::TrueColor, true);
    let (_ok_dir, ok_model, ok_patch) = apply_case(false);
    let (_bad_dir, bad_model, bad_patch) = apply_case(true);
    let f = eval_fixture();
    let model = f.dir.path().join("model.safetensors");
    let owned = |args: &[&str]| args.iter().map(|a| a.to_string()).collect::<Vec<_>>();
    let cases = vec![
        owned(&["check", "--model", s(&model)]),
        owned(&["inspect", "--show-stats", "--model", s(&model)]),
        owned(&[
            "apply",
            "--dry-run",
            "--model",
            s(&ok_model),
            "--patch",
            s(&ok_patch),
        ]),
        owned(&["apply", "--model", s(&ok_model), "--patch", s(&ok_patch)]),
        owned(&["apply", "--model", s(&bad_model), "--patch", s(&bad_patch)]),
        f.args(&[]),
    ];
    for case in cases {
        let mut argv = vec!["--json".to_owned()];
        argv.extend(case.iter().cloned());
        let (code, out, err) = run_args_raw(&deps, &argv);
        assert!(
            out.ends_with('\n') && out.matches('\n').count() == 1,
            "{case:?}: {out:?}"
        );
        let v: Value = serde_json::from_str(&out).unwrap_or_else(|e| panic!("{case:?}: {e}"));
        assert_eq!(v["exit_code"], code);
        for forbidden in [
            "\x1b",
            "█",
            "░",
            "─",
            "[CHECK]",
            "[APPLY]",
            "[VERDICT]",
            "[SHIFT]",
            "[BACKUP]",
            "[ROLLBACK]",
            "\r",
        ] {
            assert!(
                !out.contains(forbidden),
                "{case:?}: stdout has {forbidden:?}"
            );
        }
        assert!(
            err.is_empty(),
            "{case:?}: stderr must stay empty, got {err:?}"
        );
    }
}

#[test]
fn inspect_golden_in_truecolor() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let args = ["inspect", "--show-stats", "--model", s(&m)];
    let (code, colored, _) = run(&ui_deps(ColorMode::TrueColor, false), &args);
    assert_eq!(code, 0);
    assert!(
        colored.contains("\x1b[38;2;65;105;225m[L0]"),
        "blue layer brackets"
    );
    let expected = [
        "█▀█ █ █ ▀█▀  W E I G H T".to_owned(),
        "█ █ ▀▄▀  █   S U R G E O N".to_owned(),
        "[INSPECT]  2 tenseurs sélectionnés".to_owned(),
        separator_line(),
        "COUCHE   PROJ        DTYPE   SHAPE   TAILLE   MIN       MAX       MOYENNE   L2     "
            .to_owned(),
        "[L0]     gate_proj   f32     2 × 3   24 B     1.0000    6.0000    3.5000    9.5394 "
            .to_owned(),
        "[L0]     down_proj   f32     3 × 2   24 B     10.0000   60.0000   35.0000   95.3939"
            .to_owned(),
    ]
    .join("\n")
        + "\n";
    assert_eq!(strip_ansi(&colored), expected);
    let (_, plain, _) = run(&ui_deps(ColorMode::Plain, false), &args);
    assert_eq!(plain, expected, "Plain is exactly TrueColor without ANSI");
}

#[test]
fn check_golden_in_truecolor() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let len = std::fs::metadata(&m).unwrap().len();
    let (code, colored, _) = run(
        &ui_deps(ColorMode::TrueColor, false),
        &["check", "--model", s(&m)],
    );
    assert_eq!(code, 0);
    for rgb in ["38;2;65;105;225", "38;2;255;255;255", "38;2;136;136;136"] {
        assert!(colored.contains(rgb), "{rgb}");
    }
    let text = strip_ansi(&colored);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "[CHECK]    weight-surgeon");
    assert_eq!(lines[1], separator_line());
    assert!(
        lines[2].starts_with("  modèle        ") && lines[2].ends_with("model.safetensors"),
        "{}",
        lines[2]
    );
    assert_eq!(lines[3], format!("  taille        {len} B"));
    assert_eq!(lines[4], "  RAM           64.00 GiB libres");
    assert_eq!(lines[5], "  GPU           aucun");
    assert!(lines[6].starts_with("  besoin        RAM ") && lines[6].contains(" · VRAM "));
    assert_eq!(lines[7], "[VERDICT]  FITS IN RAM ONLY");
    assert!(
        lines[8..].iter().all(|l| l.starts_with("  ! ")),
        "warnings as gray ! lines"
    );

    let mut poor = ui_deps(ColorMode::TrueColor, true);
    poor.probe = Box::new(FakeProbe { available: 0 });
    let (code, out, err) = run(&poor, &["check", "--model", s(&m)]);
    assert_eq!(code, exit::GENERAL, "exit code unchanged by the identity");
    assert!(strip_ansi(&out).contains("[VERDICT]  INSUFFICIENT "));
    assert!(
        strip_ansi(&err).starts_with("[ERREUR]   ressources insuffisantes"),
        "{err:?}"
    );
}

fn positions(text: &str, badges: &[&str]) -> Vec<usize> {
    badges
        .iter()
        .map(|b| {
            text.find(b)
                .unwrap_or_else(|| panic!("{b} missing in\n{text}"))
        })
        .collect()
}

#[test]
fn apply_badges_follow_the_real_steps() {
    let deps = ui_deps(ColorMode::TrueColor, false);
    let (_dir, m, patch) = apply_case(false);
    let (code, out, err) = run(&deps, &["apply", "--model", s(&m), "--patch", s(&patch)]);
    assert_eq!(code, 0);
    let text = strip_ansi(&out);
    let p = positions(&text, &["[APPLY]", "[BACKUP]", "[SHIFT]", "[COMMIT]"]);
    assert!(p.windows(2).all(|w| w[0] < w[1]), "{text}");
    assert_eq!(
        text.matches("[SHIFT]").count(),
        2,
        "one line per finished operation"
    );
    assert!(
        text.contains("[SHIFT]    g terminée · Δmax 0.5000 · 1 tenseurs"),
        "{text}"
    );
    assert!(text.contains("✓ sha256 vérifié"));
    assert!(
        !err.contains('\r') && !out.contains('\r'),
        "no carriage return off a TTY"
    );

    let (_dir, m, patch) = apply_case(true);
    let (code, out, _) = run(&deps, &["apply", "--model", s(&m), "--patch", s(&patch)]);
    assert_eq!(code, exit::EXECUTION_FAILED);
    let text = strip_ansi(&out);
    assert!(text.contains("[ROLLBACK] restauré ✓"), "{text}");
    assert!(!text.contains("[COMMIT]"));

    let (_dir, m, patch) = apply_case(false);
    let (code, out, _) = run(
        &deps,
        &["apply", "--dry-run", "--model", s(&m), "--patch", s(&patch)],
    );
    assert_eq!(code, 0);
    let text = strip_ansi(&out);
    assert!(
        text.contains("[DRY-RUN]  g · 1 tenseurs · vecteur de 3"),
        "{text}"
    );
    assert!(!text.contains("[BACKUP]") && !text.contains("[SHIFT]") && !text.contains("[COMMIT]"));
}

#[test]
fn progress_bar_redraws_in_place_only_on_a_tty() {
    let (_dir, m, patch) = apply_case(false);
    let (code, out, err) = run(
        &ui_deps(ColorMode::TrueColor, true),
        &["apply", "--model", s(&m), "--patch", s(&patch)],
    );
    assert_eq!(code, 0);
    assert!(
        err.contains('\r') && strip_ansi(&err).contains("[SHIFT]"),
        "{err:?}"
    );
    assert!(err.contains('█'), "bar drawn");
    assert!(!out.contains('\r'), "stdout keeps whole lines");
}

#[test]
fn eval_golden_in_truecolor() {
    let f = eval_fixture();
    let out_gguf = f.dir.path().join("final.gguf");
    let (code, colored, _) = run_args_raw(
        &ui_deps(ColorMode::TrueColor, false),
        &f.args(&["--export-gguf", s(&out_gguf), "--quant", "q4_k_m"]),
    );
    assert_eq!(code, 0);
    let text = strip_ansi(&colored);
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines[0], "[EVAL]     2 prompts · max 256 tokens");
    assert_eq!(lines[1], separator_line());
    assert_eq!(
        lines[2],
        format!("PROMPT     STATUT   BOUCLE   SORTIE{}", " ".repeat(31))
    );
    assert_eq!(
        lines[3],
        "prompt-1   PASS     —        « La capitale de la France est Par… »"
    );
    assert_eq!(
        lines[5],
        format!(
            "[VERDICT]  HEALTHY · 2/2 PASS · 0 boucle(s)  {}",
            "█".repeat(20)
        )
    );
    assert!(
        lines[6].starts_with("[EXPORT]   ") && lines[6].ends_with(" · Q4_K_M · 8 B  ✓ magic GGUF"),
        "{}",
        lines[6]
    );

    let mut broken = deps_with(MockRunner::default(), || {
        Box::new(Repeat("Wait... ", 10_000))
    });
    broken.ui = UiConfig {
        out: ColorMode::Plain,
        err: ColorMode::Plain,
        err_is_tty: false,
    };
    let (code, out, _) = run_args_raw(&broken, &f.args(&["--export-gguf", s(&out_gguf)]));
    assert_eq!(code, exit::EVAL_UNHEALTHY);
    assert!(out.contains("LOOP     MarkerRepetition"), "{out}");
    assert!(out.contains("[EXPORT]   ignoré : verdict BROKEN"), "{out}");
}

#[test]
fn help_is_styled_only_when_asked_and_bannered_on_a_tty() {
    let (code, out, _) = run(&deps(), &["--help"]);
    assert_eq!(code, 0);
    assert!(
        !out.contains('\x1b') && !out.contains('█'),
        "no TTY: plain help, no banner"
    );

    let colored = UiConfig {
        out: ColorMode::TrueColor,
        err: ColorMode::TrueColor,
        err_is_tty: true,
    };
    let mut buf = Vec::new();
    let r = app::parse_cli_with(["ws", "--help"], &colored, true, &mut buf, &mut Vec::new());
    assert_eq!(r.err(), Some(0));
    let help = String::from_utf8(buf).unwrap();
    assert!(
        strip_ansi(&help).starts_with("█▀█ █ █ ▀█▀  W E I G H T\n"),
        "{help}"
    );
    assert!(help.contains("38;2;65;105;225") && strip_ansi(&help).contains("Usage"));

    let mut buf = Vec::new();
    let _ = app::parse_cli_with(
        ["ws", "--json", "--help"],
        &colored,
        true,
        &mut buf,
        &mut Vec::new(),
    );
    let help = String::from_utf8(buf).unwrap();
    assert!(
        !help.contains('\x1b') && !help.contains('█'),
        "--json disables every decoration"
    );
}

#[test]
fn closed_pipe_in_color_mode_never_panics() {
    let (_dir, m, patch) = apply_case(false);
    let deps = ui_deps(ColorMode::TrueColor, true);
    let cli = app::parse_cli(
        ["ws", "apply", "--model", s(&m), "--patch", s(&patch)],
        &mut Vec::new(),
        &mut Vec::new(),
    )
    .unwrap();
    assert_eq!(
        app::run_with(cli, &deps, &mut BrokenPipe, &mut BrokenPipe),
        0
    );
}

/// Every byte must go through the injected writers: no print macro outside tests.
#[test]
fn no_print_macros_outside_tests() {
    for file in ["src/app.rs", "src/ui.rs"] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(file);
        let source = std::fs::read_to_string(path).unwrap();
        let code = source.split("#[cfg(test)]").next().unwrap();
        for (n, line) in code.lines().enumerate() {
            let line = line.trim_start();
            if line.starts_with("//") {
                continue;
            }
            for banned in ["println!(", "print!(", "eprintln!(", "eprint!("] {
                assert!(!line.contains(banned), "{file}:{}: {banned}", n + 1);
            }
        }
    }
}

// ------------------------------------------------------------------ schema & MCP (prompt 7)

use std::io::Read as _;
use std::time::{Duration, Instant};

use weight_surgeon::app::{core_apply, ApplyRequest, ProgressEvent, ProgressSink, RollbackOutcome};
use weight_surgeon::cli::SchemaKind;
use weight_surgeon::mcp::{self, McpConfig};
use weight_surgeon::patch::{parse_patch, validate_patch, ValidationOptions};

fn patch_validator() -> jsonschema::Validator {
    jsonschema::draft7::new(&mcp::patch_schema()).unwrap()
}

fn one_op_patch(layers: Value) -> Value {
    json!({"schema_version": 1, "operations": [{"id": "a", "layers": layers,
        "projections": ["gate_proj"], "operation": "subtract", "vector_file": "v.json", "alpha": 0.5}]})
}

#[test]
fn schema_output_is_raw_deterministic_draft07() {
    let (code, plain, err) = run(&deps(), &["schema"]);
    assert_eq!(code, 0);
    assert!(err.is_empty());
    assert!(plain.ends_with('\n') && !plain.ends_with("\n\n"));
    assert!(!plain.contains('\x1b') && !plain.contains('█') && !plain.contains("[CHECK]"));
    assert_eq!(
        plain,
        mcp::render_schema(SchemaKind::Patch),
        "byte for byte"
    );
    assert_eq!(
        run(&deps(), &["--json", "schema"]).1,
        plain,
        "--json changes nothing"
    );
    let colored = ui_deps(ColorMode::TrueColor, true);
    assert_eq!(run(&colored, &["schema", "--kind", "patch"]).1, plain);
    let schema: Value = serde_json::from_str(&plain).unwrap();
    assert!(jsonschema::draft7::meta::is_valid(&schema));

    let out = binary()
        .arg("schema")
        .env("CLICOLOR_FORCE", "1")
        .env("COLORTERM", "truecolor")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), plain);
    assert_eq!(
        mcp::write_schema(SchemaKind::Patch, &mut BrokenPipe),
        0,
        "closed pipe"
    );
}

#[test]
fn schema_mirrors_serde_and_validation() {
    let schema = mcp::patch_schema();
    let text = schema.to_string();
    assert!(
        !text.contains("$ref") && !text.contains("definitions"),
        "fully inlined"
    );
    assert_eq!(schema["title"], "PatchFile");
    assert_eq!(schema["$schema"], "http://json-schema.org/draft-07/schema#");
    assert_eq!(schema["additionalProperties"], false);
    assert_eq!(schema["properties"]["schema_version"]["const"], 1);
    let ops = &schema["properties"]["operations"];
    assert_eq!(ops["minItems"], 1);
    let op = &ops["items"]["properties"];
    assert_eq!(ops["items"]["additionalProperties"], false);
    assert_eq!(op["operation"]["enum"], json!(["subtract", "add"]));
    assert_eq!(
        op["projections"]["items"]["enum"],
        json!(["gate_proj", "up_proj", "down_proj"])
    );
    assert_eq!(op["axis"]["enum"], json!(["rows", "cols"]));
    assert_eq!(op["axis"]["default"], "cols");
    assert_eq!(
        (
            op["alpha"]["minimum"].as_f64(),
            op["alpha"]["maximum"].as_f64()
        ),
        (Some(-100.0), Some(100.0))
    );
    assert!(op["axis"]["description"]
        .as_str()
        .unwrap()
        .contains("Défaut cols"));

    let v = patch_validator();
    for ok in [
        json!("all"),
        json!("ALL"),
        json!([1, 2]),
        json!({"start": 1, "end": 3}),
        json!("10-20"),
        json!(" 3, 5 ,7 "),
    ] {
        assert!(v.is_valid(&one_op_patch(ok.clone())), "{ok}");
    }
    for bad in [
        json!("foo"),
        json!([]),
        json!({"start": 3}),
        json!([1, 1]),
        json!({"start": 1, "end": 2, "step": 1}),
        json!(-1),
        json!("1-3,5"),
    ] {
        assert!(!v.is_valid(&one_op_patch(bad.clone())), "{bad}");
    }
}

/// "Accepted by Rust" means `parse_patch` (serde) then `validate_patch` without file
/// checks: serde alone accepts `operations: []`, validation refuses it.
fn rust_accepts(patch: &Value) -> bool {
    let opts = ValidationOptions {
        check_files: false,
        ..Default::default()
    };
    parse_patch(&patch.to_string()).is_ok_and(|p| validate_patch(&p, Path::new("."), &opts).is_ok())
}

#[test]
fn schema_and_rust_agree_on_valid_and_invalid_patches() {
    let v = patch_validator();
    let op = |extra: Value| {
        let mut base = json!({"id": "a", "layers": "all", "projections": ["gate_proj"],
            "operation": "subtract", "vector_file": "v.json", "alpha": 0.5});
        if let (Some(b), Some(e)) = (base.as_object_mut(), extra.as_object()) {
            b.extend(e.clone());
        }
        base
    };
    let valid = [
        json!({"schema_version": 1, "operations": [op(json!({}))]}),
        json!({"schema_version": 1, "description": "test", "backup": false, "operations": [op(json!({
            "layers": [10, 11], "axis": "rows", "vector_file": "d.safetensors", "vector_tensor": "layer_14"}))]}),
        json!({"schema_version": 1, "operations": [op(json!({"layers": {"start": 2, "end": 4}})),
            op(json!({"id": "b", "operation": "add", "alpha": -100}))]}),
        json!({"schema_version": 1, "operations": [op(json!({"alpha": 100, "projections": ["gate_proj", "up_proj", "down_proj"]}))]}),
        json!({"schema_version": 1, "operations": [op(json!({"layers": "10-20", "alpha": 0}))]}),
        json!({"schema_version": 1, "description": null, "operations": [op(json!({"layers": " 3,5,7 ", "axis": "cols"}))]}),
    ];
    for patch in &valid {
        assert!(v.is_valid(patch), "schema rejects {patch}");
        assert!(rust_accepts(patch), "Rust rejects {patch}");
    }
    let invalid = [
        json!({"schema_version": 1, "operations": [op(json!({}))], "bakup": true}),
        json!({"schema_version": 1, "operations": [op(json!({"alhpa": 1}))]}),
        json!({"schema_version": 1, "operations": [op(json!({"alpha": "0.5"}))]}),
        json!({"schema_version": 1, "operations": []}),
        json!({"schema_version": 1, "operations": [op(json!({"layers": {"start": 3}}))]}),
        json!({"operations": [op(json!({}))]}),
    ];
    for patch in &invalid {
        assert!(!v.is_valid(patch), "schema accepts {patch}");
        assert!(!rust_accepts(patch), "Rust accepts {patch}");
    }
}

// ---------------------------------------------------------------- MCP helpers

fn rpc(id: Value, method: &str, params: Value) -> String {
    json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string()
}

fn init(version: &str) -> String {
    rpc(
        json!(0),
        "initialize",
        json!({"protocolVersion": version, "capabilities": {},
        "clientInfo": {"name": "test", "version": "0"}}),
    )
}

const INITIALIZED: &str = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;

fn call(id: i64, tool: &str, arguments: Value) -> String {
    rpc(
        json!(id),
        "tools/call",
        json!({"name": tool, "arguments": arguments}),
    )
}

fn mcp_cfg(dir: &Path) -> McpConfig {
    McpConfig::new(&[dir.to_path_buf()], false, false).unwrap()
}

fn serve_bytes(deps: &AppDeps, cfg: &McpConfig, input: Vec<u8>) -> (i32, String, String) {
    let (mut out, mut err) = (Vec::new(), Vec::new());
    let code = mcp::serve(std::io::Cursor::new(input), &mut out, &mut err, deps, cfg);
    (
        code,
        String::from_utf8(out).unwrap(),
        String::from_utf8(err).unwrap(),
    )
}

/// Every stdout line must be one JSON-RPC 2.0 message; returns them in order.
fn session(deps: &AppDeps, cfg: &McpConfig, lines: &[String]) -> Vec<Value> {
    let (code, out, _) = serve_bytes(deps, cfg, (lines.join("\n") + "\n").into_bytes());
    assert_eq!(code, 0);
    out.lines()
        .map(|line| {
            assert!(
                !line.trim().is_empty() && !line.contains('\x1b') && !line.contains('█'),
                "{line:?}"
            );
            let v: Value = serde_json::from_str(line).unwrap_or_else(|e| panic!("{e}: {line}"));
            assert_eq!(v["jsonrpc"], "2.0");
            v
        })
        .collect()
}

fn reply(responses: &[Value], id: Value) -> &Value {
    responses
        .iter()
        .find(|r| r["id"] == id && (r.get("result").is_some() || r.get("error").is_some()))
        .unwrap_or_else(|| panic!("no reply for {id} in {responses:?}"))
}

/// The tool envelope of a successful JSON-RPC reply (`isError` may still be true).
fn envelope(responses: &[Value], id: i64) -> (bool, Value) {
    let result = &reply(responses, json!(id))["result"];
    assert!(result.get("structuredContent").is_some(), "{result}");
    let text: Value = serde_json::from_str(result["content"][0]["text"].as_str().unwrap()).unwrap();
    assert_eq!(
        text, result["structuredContent"],
        "text and structuredContent agree"
    );
    (
        result["isError"].as_bool().unwrap(),
        result["structuredContent"].clone(),
    )
}

fn error_code(responses: &[Value], id: Value) -> i64 {
    reply(responses, id)["error"]["code"].as_i64().unwrap()
}

/// bf16 SafeTensors: gate [2,3] = 1..6, down [3,2] = 10..60.
fn bf16_model(dir: &Path) -> PathBuf {
    let path = dir.join("model.safetensors");
    let encode = |values: &[f32]| {
        values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect::<Vec<u8>>()
    };
    let gate = encode(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let down = encode(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    let header = json!({
        GATE: {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, gate.len()]},
        DOWN: {"dtype": "BF16", "shape": [3, 2], "data_offsets": [gate.len(), gate.len() + down.len()]},
    });
    let mut header = header.to_string();
    while !header.len().is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&gate);
    file.extend_from_slice(&down);
    std::fs::write(&path, file).unwrap();
    path
}

fn mcp_patch(ops: Value, backup: bool) -> Value {
    json!({"schema_version": 1, "backup": backup, "operations": ops})
}

// ---------------------------------------------------------------- MCP, function level

#[test]
fn mcp_full_session_is_pure_json_rpc() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let lines = [
        init("2025-06-18"),
        INITIALIZED.to_owned(),
        rpc(json!(2), "tools/list", json!({})),
        call(3, "surgeon_check", json!({"model": "model.safetensors"})),
    ];
    let responses = session(&deps(), &mcp_cfg(dir.path()), &lines);
    assert_eq!(responses.len(), 3, "the notification gets no reply");
    assert_eq!(
        responses
            .iter()
            .map(|r| r["id"].clone())
            .collect::<Vec<_>>(),
        [json!(0), json!(2), json!(3)]
    );
    let init_result = &responses[0]["result"];
    assert_eq!(init_result["protocolVersion"], "2025-06-18");
    assert_eq!(init_result["capabilities"]["tools"]["listChanged"], false);
    assert_eq!(init_result["serverInfo"]["name"], "surgeon");
    assert!(init_result["instructions"]
        .as_str()
        .unwrap()
        .contains("sauvegarde"));
    assert_eq!(responses[1]["result"]["tools"].as_array().unwrap().len(), 4);
    let (is_error, env) = envelope(&responses, 3);
    assert!(!is_error, "{env}");
    assert_eq!(env["command"], "check");
    assert_eq!(env["ok"], true);
    assert!(env["warnings"].is_array());
    assert!(env["data"]["verdict"].is_object());
    let _ = m;
}

#[test]
fn mcp_version_negotiation_ping_and_id_types() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = mcp_cfg(dir.path());
    let old = session(&deps(), &cfg, &[init("2024-11-05")]);
    assert_eq!(old[0]["result"]["protocolVersion"], "2024-11-05");
    let latest = session(&deps(), &cfg, &[init("2025-11-25")]);
    assert_eq!(latest[0]["result"]["protocolVersion"], "2025-11-25");
    let unknown = session(&deps(), &cfg, &[init("1999-01-01")]);
    assert_eq!(
        unknown[0]["result"]["protocolVersion"],
        *mcp::SUPPORTED_PROTOCOL_VERSIONS.last().unwrap()
    );

    let pings = session(
        &deps(),
        &cfg,
        &[
            rpc(json!("abc"), "ping", json!({})),
            rpc(json!(7), "ping", json!({})),
            r#"{"jsonrpc":"2.0","id":null,"method":"ping"}"#.to_owned(),
        ],
    );
    assert_eq!(reply(&pings, json!("abc"))["result"], json!({}));
    assert_eq!(reply(&pings, json!(7))["result"], json!({}));
    assert_eq!(reply(&pings, Value::Null)["result"], json!({}));
}

#[test]
fn mcp_protocol_errors_never_stop_the_server() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = mcp_cfg(dir.path());
    let early = session(
        &deps(),
        &cfg,
        &[
            rpc(json!(1), "tools/list", json!({})),
            init("2025-06-18"),
            rpc(json!(2), "tools/list", json!({})),
            init("2025-06-18").replace("\"id\":0", "\"id\":3"),
        ],
    );
    assert_eq!(error_code(&early, json!(1)), mcp::SERVER_NOT_INITIALIZED);
    assert!(reply(&early, json!(2)).get("result").is_some());
    assert_eq!(
        error_code(&early, json!(3)),
        mcp::INVALID_REQUEST,
        "initialize twice"
    );

    let responses = session(
        &deps(),
        &cfg,
        &[
            init("2025-06-18"),
            "{not json".to_owned(),
            r#"{"jsonrpc":"2.0","id":10}"#.to_owned(),
            rpc(json!(11), "resources/list", json!({})),
            rpc(json!(12), "tools/call", json!({"arguments": {}})),
            call(13, "surgeon_nope", json!({})),
            r#"[{"jsonrpc":"2.0","id":14,"method":"ping"}]"#.to_owned(),
            rpc(json!(15), "ping", json!({})),
            r#"{"jsonrpc":"2.0","method":"notifications/whatever"}"#.to_owned(),
        ],
    );
    let parse = responses
        .iter()
        .find(|r| r["error"]["code"] == mcp::PARSE_ERROR)
        .unwrap();
    assert_eq!(parse["id"], Value::Null);
    assert_eq!(error_code(&responses, json!(10)), mcp::INVALID_REQUEST);
    assert_eq!(error_code(&responses, json!(11)), mcp::METHOD_NOT_FOUND);
    assert_eq!(error_code(&responses, json!(12)), mcp::INVALID_PARAMS);
    assert_eq!(error_code(&responses, json!(13)), mcp::INVALID_PARAMS);
    let batch = responses
        .iter()
        .filter(|r| r["error"]["code"] == mcp::INVALID_REQUEST && r["id"].is_null())
        .count();
    assert_eq!(batch, 1, "batch refused with id null");
    assert_eq!(
        reply(&responses, json!(15))["result"],
        json!({}),
        "still answering"
    );
    assert_eq!(responses.len(), 8, "unknown notification ignored");
}

#[test]
fn mcp_tool_errors_are_results_with_is_error() {
    let dir = tempfile::tempdir().unwrap();
    model(dir.path());
    write_patch(dir.path(), json!([]), true);
    let mut bad = op("a", "0", "gate_proj", "v3.json", "rows");
    bad["alpha"] = json!(500);
    let ops = json!([
        bad,
        op("a", "0", "gate_proj", "v3.json", "rows"),
        op("c", "0", "gate_proj", "v3.json", "rows")
    ]);
    let mut ops = ops;
    ops[2]["layers"] = json!([]);
    let lines = [
        init("2025-06-18"),
        call(1, "surgeon_check", json!({"model": "absent.safetensors"})),
        call(
            2,
            "surgeon_apply",
            json!({"model": "model.safetensors", "patch": mcp_patch(ops, true)}),
        ),
        call(
            3,
            "surgeon_check",
            json!({"model": "model.safetensors", "extra": 1}),
        ),
    ];
    let responses = session(&deps(), &mcp_cfg(dir.path()), &lines);
    let (is_error, env) = envelope(&responses, 1);
    assert!(is_error && env["ok"] == false);
    assert_eq!(env["error"]["kind"], "model_not_found");
    let (is_error, env) = envelope(&responses, 2);
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "invalid_patch");
    assert_eq!(env["error"]["issues"].as_array().unwrap().len(), 3, "{env}");
    assert_eq!(env["exit_code"], exit::INVALID_INPUT);
    assert_eq!(
        envelope(&responses, 3).1["error"]["kind"],
        "invalid_arguments"
    );
    assert!(
        responses.iter().all(|r| r.get("error").is_none()),
        "no JSON-RPC error"
    );

    let mut poor = deps();
    poor.probe = Box::new(FakeProbe { available: 0 });
    let responses = session(
        &poor,
        &mcp_cfg(dir.path()),
        &[
            init("2025-06-18"),
            call(1, "surgeon_check", json!({"model": "model.safetensors"})),
        ],
    );
    let (is_error, env) = envelope(&responses, 1);
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "insufficient_resources");
}

#[test]
fn mcp_apply_is_exact_and_always_backed_up() {
    let dir = tempfile::tempdir().unwrap();
    let m = bf16_model(dir.path());
    let original = std::fs::read(&m).unwrap();
    std::fs::write(dir.path().join("v3.json"), "[1, 1, 1]").unwrap();
    let gate_op = json!([op("g", "0", "gate_proj", "v3.json", "rows")]);
    let cfg = mcp_cfg(dir.path());

    let dry = session(
        &deps(),
        &cfg,
        &[
            init("2025-06-18"),
            call(
                1,
                "surgeon_apply",
                json!({"model": "model.safetensors", "patch": mcp_patch(gate_op.clone(), true), "dry_run": true}),
            ),
        ],
    );
    let (is_error, env) = envelope(&dry, 1);
    assert!(!is_error, "{env}");
    assert_eq!(env["data"]["report"]["dry_run"], true);
    assert_eq!(
        std::fs::read(&m).unwrap(),
        original,
        "dry run writes nothing"
    );
    assert!(backups_in(dir.path()).is_empty(), "dry run takes no backup");

    // backup: false in the patch, yet the server imposes one.
    let real = session(
        &deps(),
        &cfg,
        &[
            init("2025-06-18"),
            call(
                1,
                "surgeon_apply",
                json!({"model": "model.safetensors", "patch": mcp_patch(gate_op, false)}),
            ),
        ],
    );
    let (is_error, env) = envelope(&real, 1);
    assert!(!is_error, "{env}");
    let backups = backups_in(dir.path());
    assert_eq!(backups.len(), 1, "backup imposed without --allow-no-backup");
    assert_eq!(
        std::fs::read(dir.path().join(&backups[0])).unwrap(),
        original
    );

    // Reference: same file with gate - 0.5 * [1,1,1] per row, exact in bf16.
    let reference = tempfile::tempdir().unwrap();
    let expected_path = bf16_model(reference.path());
    let mut expected = std::fs::read(&expected_path).unwrap();
    let data_start = expected.len() - 24;
    for (i, v) in [0.5f32, 1.5, 2.5, 3.5, 4.5, 5.5].iter().enumerate() {
        expected[data_start + 2 * i..data_start + 2 * i + 2]
            .copy_from_slice(&half::bf16::from_f32(*v).to_le_bytes());
    }
    assert_eq!(std::fs::read(&m).unwrap(), expected, "byte for byte");
}

#[test]
fn mcp_honors_backup_false_only_with_allow_no_backup() {
    let dir = tempfile::tempdir().unwrap();
    model(dir.path());
    std::fs::write(dir.path().join("v3.json"), "[1, 1, 1]").unwrap();
    let cfg = McpConfig::new(&[dir.path().to_path_buf()], false, true).unwrap();
    let patch = mcp_patch(json!([op("g", "0", "gate_proj", "v3.json", "rows")]), false);
    let responses = session(
        &deps(),
        &cfg,
        &[
            init("2025-06-18"),
            call(
                1,
                "surgeon_apply",
                json!({"model": "model.safetensors", "patch": patch}),
            ),
        ],
    );
    assert!(!envelope(&responses, 1).0);
    assert!(backups_in(dir.path()).is_empty());
}

#[test]
fn mcp_failed_or_cancelled_apply_leaves_the_model_intact() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let original = std::fs::read(&m).unwrap();
    std::fs::write(dir.path().join("v3.json"), "[1, 1, 1]").unwrap();
    std::fs::write(dir.path().join("v2.json"), "[1, 1]").unwrap();
    let cfg = mcp_cfg(dir.path());
    let failing = mcp_patch(
        json!([
            op("ok", "0", "gate_proj", "v3.json", "rows"),
            op("bad", "0", "down_proj", "v2.json", "cols")
        ]),
        true,
    );
    let responses = session(
        &deps(),
        &cfg,
        &[
            init("2025-06-18"),
            call(
                1,
                "surgeon_apply",
                json!({"model": "model.safetensors", "patch": failing}),
            ),
        ],
    );
    let (is_error, env) = envelope(&responses, 1);
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "execution_failed");
    assert_eq!(env["exit_code"], exit::EXECUTION_FAILED);
    assert_eq!(std::fs::read(&m).unwrap(), original, "rolled back");

    // The cancellation precedes the call in the stream: deterministic, applied at start.
    let good = mcp_patch(json!([op("g", "0", "gate_proj", "v3.json", "rows")]), true);
    let cancelled = session(&deps(), &cfg, &[
        init("2025-06-18"),
        json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 5, "reason": "test"}}).to_string(),
        call(5, "surgeon_apply", json!({"model": "model.safetensors", "patch": good})),
    ]);
    let (is_error, env) = envelope(&cancelled, 5);
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "cancelled");
    assert_eq!(std::fs::read(&m).unwrap(), original);
}

/// Raises the cancellation flag after the first shifted tensor, as the reader thread does
/// when `notifications/cancelled` arrives mid-apply.
struct CancelAfterFirstShift<'a> {
    flag: &'a AtomicBool,
    events: Vec<ProgressEvent>,
}

impl ProgressSink for CancelAfterFirstShift<'_> {
    fn event(&mut self, event: ProgressEvent) {
        if matches!(event, ProgressEvent::Shift { .. }) {
            self.flag.store(true, Ordering::SeqCst);
        }
        self.events.push(event);
    }
}

#[test]
fn cancellation_mid_apply_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let original = std::fs::read(&m).unwrap();
    std::fs::write(dir.path().join("v3.json"), "[1, 1, 1]").unwrap();
    let patch = parse_patch(
        &mcp_patch(
            json!([
                op("g", "0", "gate_proj", "v3.json", "rows"),
                op("d", "0", "down_proj", "v3.json", "cols")
            ]),
            true,
        )
        .to_string(),
    )
    .unwrap();
    let deps = deps();
    let request = ApplyRequest {
        model: m.clone(),
        patch,
        patch_label: "patch".into(),
        base_dir: dir.path().to_path_buf(),
        dry_run: false,
        no_backup: false,
        backup_dir: None,
    };
    let mut sink = CancelAfterFirstShift {
        flag: &deps.interrupted,
        events: Vec::new(),
    };
    let failure = core_apply(&request, &deps, &mut sink).unwrap_err();
    assert_eq!(
        failure.error.unwrap().kind.as_deref(),
        Some("execution_failed")
    );
    assert!(
        sink.events
            .contains(&ProgressEvent::Rollback(RollbackOutcome::Restored)),
        "{:?}",
        sink.events
    );
    assert_eq!(
        std::fs::read(&m).unwrap(),
        original,
        "first tensor restored"
    );
}

#[test]
fn mcp_sandbox_refuses_paths_outside_without_leaking_them() {
    let allowed = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    model(allowed.path());
    let secret = model(outside.path());
    std::fs::write(outside.path().join("v3.json"), "[1, 1, 1]").unwrap();
    let escape = format!(
        "../{}/model.safetensors",
        outside.path().file_name().unwrap().to_string_lossy()
    );
    let outside_vector = outside.path().join("v3.json");
    let lines = [
        init("2025-06-18"),
        call(1, "surgeon_check", json!({"model": escape})),
        call(2, "surgeon_check", json!({"model": s(&secret)})),
        call(
            3,
            "surgeon_apply",
            json!({"model": "model.safetensors",
            "patch": mcp_patch(json!([op("g", "0", "gate_proj", s(&outside_vector), "rows")]), true)}),
        ),
        call(
            4,
            "surgeon_inspect",
            json!({"model": "missing.safetensors"}),
        ),
        call(
            5,
            "surgeon_eval",
            json!({"model": "model.safetensors", "prompts_file": s(&outside.path().join("p.txt"))}),
        ),
    ];
    let responses = session(&deps(), &mcp_cfg(allowed.path()), &lines);
    for id in [1, 2, 3, 5] {
        let (is_error, env) = envelope(&responses, id);
        assert!(is_error);
        assert_eq!(env["error"]["kind"], "path_not_allowed", "{id}: {env}");
        let message = env["error"]["message"].as_str().unwrap();
        assert!(
            !message.contains(&outside.path().display().to_string()),
            "{message}"
        );
    }
    let (_, env) = envelope(&responses, 4);
    assert_eq!(
        env["error"]["kind"], "model_not_found",
        "inside the sandbox: a normal error"
    );
}

#[test]
fn mcp_read_only_refuses_writes_only() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let original = std::fs::read(&m).unwrap();
    std::fs::write(dir.path().join("v3.json"), "[1, 1, 1]").unwrap();
    let cfg = McpConfig::new(&[dir.path().to_path_buf()], true, false).unwrap();
    let patch = mcp_patch(json!([op("g", "0", "gate_proj", "v3.json", "rows")]), true);
    let responses = session(
        &deps(),
        &cfg,
        &[
            init("2025-06-18"),
            call(
                1,
                "surgeon_apply",
                json!({"model": "model.safetensors", "patch": patch}),
            ),
            call(2, "surgeon_check", json!({"model": "model.safetensors"})),
            call(
                3,
                "surgeon_eval",
                json!({"model": "model.safetensors", "prompts": [{"text": "x"}], "export_gguf": "out.gguf"}),
            ),
        ],
    );
    assert_eq!(envelope(&responses, 1).1["error"]["kind"], "read_only");
    assert!(!envelope(&responses, 2).0);
    assert_eq!(envelope(&responses, 3).1["error"]["kind"], "read_only");
    assert_eq!(std::fs::read(&m).unwrap(), original);
}

#[test]
fn mcp_progress_only_with_a_token() {
    let dir = tempfile::tempdir().unwrap();
    model(dir.path());
    std::fs::write(dir.path().join("v3.json"), "[1, 1, 1]").unwrap();
    let patch = mcp_patch(
        json!([
            op("g", "0", "gate_proj", "v3.json", "rows"),
            op("d", "0", "down_proj", "v3.json", "cols")
        ]),
        true,
    );
    let with_token = rpc(
        json!(1),
        "tools/call",
        json!({"name": "surgeon_apply",
        "arguments": {"model": "model.safetensors", "patch": patch}, "_meta": {"progressToken": "tok"}}),
    );
    let responses = session(
        &deps(),
        &mcp_cfg(dir.path()),
        &[init("2025-06-18"), with_token],
    );
    let progress: Vec<&Value> = responses
        .iter()
        .filter(|r| r["method"] == "notifications/progress")
        .collect();
    assert_eq!(progress.len(), 2, "one per shifted tensor");
    for (i, p) in progress.iter().enumerate() {
        assert_eq!(p["params"]["progressToken"], "tok");
        assert_eq!(p["params"]["progress"], i + 1);
        assert_eq!(p["params"]["total"], 2);
        assert!(p.get("id").is_none());
    }
    assert!(!envelope(&responses, 1).0);

    let prompts =
        json!({"model": "m.gguf", "prompts": [{"text": "a"}, {"text": "b"}, {"text": "c"}]});
    std::fs::write(dir.path().join("m.gguf"), "GGUF").unwrap();
    let eval_with = rpc(
        json!(2),
        "tools/call",
        json!({"name": "surgeon_eval", "arguments": prompts.clone(), "_meta": {"progressToken": 9}}),
    );
    let responses = session(
        &deps(),
        &mcp_cfg(dir.path()),
        &[
            init("2025-06-18"),
            eval_with,
            call(3, "surgeon_eval", prompts),
        ],
    );
    let eval_progress: Vec<&Value> = responses
        .iter()
        .filter(|r| r["method"] == "notifications/progress")
        .collect();
    assert_eq!(
        eval_progress.len(),
        3,
        "one per prompt, none for the call without token"
    );
    assert!(eval_progress
        .iter()
        .all(|p| p["params"]["progressToken"] == 9));
}

#[test]
fn mcp_panic_in_a_tool_is_internal_error_and_server_lives_on() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("m.gguf"), "GGUF").unwrap();
    let mut deps = deps();
    deps.generator_factory = Box::new(|_, _| panic!("tool exploded"));
    let (code, out, err) = serve_bytes(
        &deps,
        &mcp_cfg(dir.path()),
        ([
            init("2025-06-18"),
            call(
                1,
                "surgeon_eval",
                json!({"model": "m.gguf", "prompts": [{"text": "x"}]}),
            ),
            rpc(json!(2), "ping", json!({})),
        ]
        .join("\n")
            + "\n")
            .into_bytes(),
    );
    assert_eq!(code, 0);
    let responses: Vec<Value> = out
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(error_code(&responses, json!(1)), mcp::INTERNAL_ERROR);
    assert!(
        !reply(&responses, json!(1))["error"]["message"]
            .as_str()
            .unwrap()
            .contains("exploded"),
        "neutral message"
    );
    assert!(err.contains("tool exploded"), "detail on stderr");
    assert_eq!(reply(&responses, json!(2))["result"], json!({}));
}

#[test]
fn mcp_input_framing_is_robust() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = mcp_cfg(dir.path());
    let mut input = Vec::new();
    input.extend_from_slice("\u{feff}".as_bytes());
    input.extend_from_slice(init("2025-06-18").as_bytes());
    input.extend_from_slice(b"\r\n\n   \n");
    input.extend_from_slice(rpc(json!(2), "ping", json!({})).as_bytes());
    input.extend_from_slice(b"\r\n");
    input.extend(std::iter::repeat_n(b'x', mcp::MAX_LINE_BYTES + 1));
    input.push(b'\n');
    input.extend_from_slice(rpc(json!(3), "ping", json!({})).as_bytes()); // EOF, no newline
    let (code, out, _) = serve_bytes(&deps(), &cfg, input);
    assert_eq!(code, 0, "clean EOF");
    let responses: Vec<Value> = out
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(responses.len(), 4, "{out}");
    assert!(
        reply(&responses, json!(0)).get("result").is_some(),
        "BOM tolerated"
    );
    assert_eq!(
        reply(&responses, json!(2))["result"],
        json!({}),
        "CRLF tolerated"
    );
    assert_eq!(
        responses[2]["error"]["code"],
        mcp::INVALID_REQUEST,
        "oversized line"
    );
    assert_eq!(
        reply(&responses, json!(3))["result"],
        json!({}),
        "resynchronized"
    );

    let (mut err, input) = (
        Vec::new(),
        init("2025-06-18") + "\n" + &rpc(json!(1), "ping", json!({})) + "\n",
    );
    let code = mcp::serve(
        std::io::Cursor::new(input.into_bytes()),
        BrokenPipe,
        &mut err,
        &deps(),
        &cfg,
    );
    assert_eq!(code, exit::GENERAL, "stdout closed: stop, no panic");
}

#[test]
fn schema_tools_kind_equals_tools_list() {
    let (code, out, _) = run(&deps(), &["schema", "--kind", "tools"]);
    assert_eq!(code, 0);
    let exported: Value = serde_json::from_str(&out).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let responses = session(
        &deps(),
        &mcp_cfg(dir.path()),
        &[init("2025-06-18"), rpc(json!(1), "tools/list", json!({}))],
    );
    assert_eq!(exported, reply(&responses, json!(1))["result"]["tools"]);
    let names: Vec<&str> = exported
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        [
            "surgeon_check",
            "surgeon_inspect",
            "surgeon_apply",
            "surgeon_eval"
        ]
    );
    for tool in exported.as_array().unwrap() {
        let schema = &tool["inputSchema"];
        assert_eq!(schema["type"], "object");
        assert!(
            jsonschema::draft7::meta::is_valid(schema),
            "{}",
            tool["name"]
        );
        assert!(tool["description"].as_str().unwrap().len() > 20);
        assert!(tool["annotations"]["readOnlyHint"].is_boolean());
    }
    assert_eq!(exported[2]["annotations"]["destructiveHint"], true);
    assert_eq!(
        exported[2]["inputSchema"]["properties"]["patch"]["additionalProperties"],
        false
    );
}

#[test]
fn mcp_clap_errors_go_to_stderr() {
    let (code, out, err) = run(&deps(), &["mcp", "--bogus"]);
    assert_eq!(code, exit::INVALID_INPUT);
    assert!(out.is_empty(), "stdout reserved for the protocol");
    assert!(err.contains("--bogus"));
}

/// Extends the prompt 6 guard to the MCP module.
#[test]
fn no_print_macros_in_mcp() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/mcp");
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let source = std::fs::read_to_string(&path).unwrap();
        let code = source.split("#[cfg(test)]").next().unwrap();
        for (n, line) in code.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            for banned in ["println!(", "print!(", "eprintln!(", "eprint!("] {
                assert!(
                    !line.contains(banned),
                    "{}:{}: {banned}",
                    path.display(),
                    n + 1
                );
            }
        }
    }
}

// ---------------------------------------------------------------- MCP, process level

#[test]
fn mcp_binary_speaks_only_json_rpc_on_stdout() {
    let dir = tempfile::tempdir().unwrap();
    model(dir.path());
    let mut child = binary()
        .args(["mcp", "--allow-dir", s(dir.path())])
        .env("CLICOLOR_FORCE", "1")
        .env("COLORTERM", "truecolor")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let input = [
        init("2025-06-18"),
        INITIALIZED.to_owned(),
        rpc(json!(2), "tools/list", json!({})),
        call(3, "surgeon_check", json!({"model": "model.safetensors"})),
    ]
    .join("\n")
        + "\n";
    {
        use std::io::Write as _;
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(input.as_bytes()).unwrap();
    } // stdin closed: EOF ends the server
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > Duration::from_secs(30) {
            let _ = child.kill();
            panic!("surgeon mcp did not exit within 30 s");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let (mut out, mut err) = (String::new(), String::new());
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut out)
        .unwrap();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut err)
        .unwrap();
    assert_eq!(status.code(), Some(0), "{err}");
    assert!(
        !out.contains('\x1b') && !err.contains('\x1b'),
        "no ANSI even with CLICOLOR_FORCE"
    );
    let responses: Vec<Value> = out
        .lines()
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l:?}"));
            assert_eq!(v["jsonrpc"], "2.0");
            v
        })
        .collect();
    assert_eq!(responses.len(), 3);
    assert!(!envelope(&responses, 3).0);
}
