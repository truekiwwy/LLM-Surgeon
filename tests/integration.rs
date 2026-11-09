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
    assert_eq!(
        op["operation"]["enum"],
        json!(["subtract", "add", "project_out"])
    );
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
    assert_eq!(responses[1]["result"]["tools"].as_array().unwrap().len(), 9);
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
            "surgeon_eval",
            "surgeon_extract",
            "surgeon_bench",
            "surgeon_autotune",
            "surgeon_diff",
            "surgeon_lora"
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

// ------------------------------------------------------------------ extract (prompt 8)

use std::cell::RefCell as ExtractCell;
use std::ops::ControlFlow as Flow;

use weight_surgeon::cli::ExtractArgs;
use weight_surgeon::extract::{
    self as ex, ExtractEvent, ExtractObserver, ExtractRequest, RawTensor, SafetensorsDir,
    TensorSource,
};
use weight_surgeon::hardware::BudgetPolicy;

const POS_WORDS: [&str; 12] = [
    "joie", "soleil", "rire", "fete", "danse", "musique", "ami", "lumiere", "chaleur", "sourire",
    "jardin", "plage",
];
const NEG_WORDS: [&str; 12] = [
    "pluie", "nuit", "froid", "brume", "silence", "gris", "orage", "vide", "hiver", "ombre",
    "perte", "peur",
];
const FILLER: [&str; 10] = [
    "le", "la", "un", "une", "est", "et", "tres", "il", "elle", "nous",
];
const HIDDEN: usize = 16;
const INTER: usize = 32;
const LAYERS: usize = 4;
const VOCAB: usize = 64;

struct Rng(u64);

impl Rng {
    fn next(&mut self, scale: f32) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    }
}

fn vocab() -> Vec<String> {
    ["[UNK]", "<|im_start|>", "<|im_end|>", "user", "assistant"]
        .iter()
        .chain(&POS_WORDS)
        .chain(&NEG_WORDS)
        .chain(&FILLER)
        .map(|w| w.to_string())
        .collect()
}

/// Every tensor of the full 4-layer model (final norm and lm_head included, so the tests
/// can prove they are never read).
fn qwen_tensors() -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let mut rng = Rng(42);
    let head = HIDDEN / 2;
    let mut shapes = vec![("model.embed_tokens.weight".to_owned(), vec![VOCAB, HIDDEN])];
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        for (suffix, shape) in [
            ("self_attn.q_proj.weight", vec![2 * head, HIDDEN]),
            ("self_attn.q_proj.bias", vec![2 * head]),
            ("self_attn.k_proj.weight", vec![2 * head, HIDDEN]),
            ("self_attn.k_proj.bias", vec![2 * head]),
            ("self_attn.v_proj.weight", vec![2 * head, HIDDEN]),
            ("self_attn.v_proj.bias", vec![2 * head]),
            ("self_attn.o_proj.weight", vec![HIDDEN, 2 * head]),
            ("mlp.gate_proj.weight", vec![INTER, HIDDEN]),
            ("mlp.up_proj.weight", vec![INTER, HIDDEN]),
            ("mlp.down_proj.weight", vec![HIDDEN, INTER]),
            ("input_layernorm.weight", vec![HIDDEN]),
            ("post_attention_layernorm.weight", vec![HIDDEN]),
        ] {
            shapes.push((format!("{p}.{suffix}"), shape));
        }
    }
    shapes.push(("model.norm.weight".into(), vec![HIDDEN]));
    shapes.push(("lm_head.weight".into(), vec![VOCAB, HIDDEN]));
    shapes
        .into_iter()
        .map(|(name, shape)| {
            let n: usize = shape.iter().product();
            let values = (0..n)
                .map(|_| {
                    if name.contains("norm") {
                        1.0 + rng.next(0.1)
                    } else {
                        rng.next(0.4)
                    }
                })
                .collect();
            (name, shape, values)
        })
        .collect()
}

/// F32 or BF16 SafeTensors with any shapes.
fn write_tensors(path: &Path, dtype: &str, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape, values) in tensors {
        let start = data.len();
        for v in values {
            match dtype {
                "BF16" => data.extend_from_slice(&half::bf16::from_f32(*v).to_le_bytes()),
                _ => data.extend_from_slice(&v.to_le_bytes()),
            }
        }
        header.insert(
            name.clone(),
            json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
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

fn write_tokenizer(path: &Path) {
    use tokenizers::models::wordlevel::WordLevel;
    use tokenizers::pre_tokenizers::whitespace::Whitespace;
    use tokenizers::{AddedToken, Tokenizer};
    let model = WordLevel::builder()
        .vocab(
            vocab()
                .into_iter()
                .enumerate()
                .map(|(i, w)| (w, i as u32))
                .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(Whitespace {}));
    tokenizer
        .add_special_tokens([
            AddedToken::from("<|im_start|>", true),
            AddedToken::from("<|im_end|>", true),
        ])
        .unwrap();
    tokenizer.save(path, false).unwrap();
}

fn qwen_config(model_type: &str) -> Value {
    json!({"architectures": ["Qwen2ForCausalLM"], "model_type": model_type, "vocab_size": VOCAB,
        "hidden_size": HIDDEN, "intermediate_size": INTER, "num_hidden_layers": LAYERS,
        "num_attention_heads": 2, "num_key_value_heads": 2, "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false,
        "sliding_window": 128, "hidden_act": "silu", "torch_dtype": "float32"})
}

/// Deterministic mini Qwen2: config.json, tokenizer.json, weights (one file or two shards
/// with an index).
fn write_fake_qwen2_dir(dir: &Path, dtype: &str, sharded: bool) -> PathBuf {
    let model = dir.join("qwen2-mini");
    std::fs::create_dir_all(&model).unwrap();
    std::fs::write(model.join("config.json"), qwen_config("qwen2").to_string()).unwrap();
    write_tokenizer(&model.join("tokenizer.json"));
    let tensors = qwen_tensors();
    if sharded {
        let (a, b) = tensors.split_at(tensors.len() / 2);
        let names = [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ];
        write_tensors(&model.join(names[0]), dtype, a);
        write_tensors(&model.join(names[1]), dtype, b);
        let weight_map: serde_json::Map<String, Value> = a
            .iter()
            .map(|t| (t.0.clone(), json!(names[0])))
            .chain(b.iter().map(|t| (t.0.clone(), json!(names[1]))))
            .collect();
        std::fs::write(
            model.join("model.safetensors.index.json"),
            json!({"weight_map": weight_map}).to_string(),
        )
        .unwrap();
    } else {
        write_tensors(&model.join("model.safetensors"), dtype, &tensors);
    }
    model
}

/// Six positive and six negative prompts made of vocabulary words.
fn write_prompt_files(dir: &Path) -> (PathBuf, PathBuf) {
    let make = |words: &[&str]| -> Vec<String> {
        (0..6)
            .map(|i| format!("{} {} est {}", FILLER[i % 4], words[i], words[(i + 3) % 12]))
            .collect()
    };
    let (pos, neg) = (dir.join("pos.json"), dir.join("neg.json"));
    std::fs::write(&pos, serde_json::to_string(&make(&POS_WORDS)).unwrap()).unwrap();
    std::fs::write(&neg, serde_json::to_string(&make(&NEG_WORDS)).unwrap()).unwrap();
    (pos, neg)
}

struct ExtractFixture {
    dir: tempfile::TempDir,
    model: PathBuf,
    pos: PathBuf,
    neg: PathBuf,
}

fn extract_fixture(dtype: &str, sharded: bool) -> ExtractFixture {
    let dir = tempfile::tempdir().unwrap();
    let model = write_fake_qwen2_dir(dir.path(), dtype, sharded);
    let (pos, neg) = write_prompt_files(dir.path());
    ExtractFixture {
        dir,
        model,
        pos,
        neg,
    }
}

impl ExtractFixture {
    fn args(&self, out: &str, extra: &[&str]) -> Vec<String> {
        let mut args: Vec<String> = [
            "extract",
            "--model",
            s(&self.model),
            "--pos-prompts",
            s(&self.pos),
            "--neg-prompts",
            s(&self.neg),
            "--layer",
            "2",
            "--out",
            s(&self.dir.path().join(out)),
        ]
        .map(String::from)
        .to_vec();
        args.extend(extra.iter().map(|a| a.to_string()));
        args
    }

    fn request(&self, out: &str, layer: usize) -> ExtractRequest {
        let load = |p: &Path| ex::load_prompt_texts(p).unwrap();
        ExtractRequest {
            model_dir: self.model.clone(),
            pos_prompts: load(&self.pos),
            neg_prompts: load(&self.neg),
            layer,
            out: self.dir.path().join(out),
            pooling: ex::Pooling::Last,
            site: ex::Site::ResidPost,
            chat_template: ex::ChatTemplate::NoTemplate,
            max_tokens: 512,
            dtype: ex::DTypeChoice::Auto,
            overwrite: false,
        }
    }
}

fn read_direction(path: &Path) -> (Vec<f32>, std::collections::HashMap<String, String>) {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let view = st.tensor("direction").unwrap();
    assert_eq!(view.dtype(), safetensors::Dtype::F32);
    assert_eq!(view.shape(), [HIDDEN]);
    let values = view
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    (values, meta.metadata().clone().unwrap())
}

#[test]
fn extract_end_to_end_is_deterministic_and_readable() {
    let f = extract_fixture("F32", false);
    let (code, out, err) = run_args_raw(
        &deps(),
        &["--json".to_owned()]
            .into_iter()
            .chain(f.args("a.safetensors", &[]))
            .collect::<Vec<_>>(),
    );
    assert_eq!(code, 0, "{out}{err}");
    assert!(err.is_empty() && !out.contains('\x1b') && !out.contains("[EXTRACT]"));
    assert!(out.ends_with('\n') && out.matches('\n').count() == 1);
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v["command"], "extract");
    assert_eq!(v["data"]["hidden_size"], HIDDEN);
    assert_eq!(
        (v["data"]["n_pos"].as_u64(), v["data"]["n_neg"].as_u64()),
        (Some(6), Some(6))
    );
    assert_eq!(v["data"]["dtype_used"], "f32");
    let few = v["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|w| w.as_str().unwrap().contains("recommandés"))
        .count();
    assert_eq!(few, 2, "fewer than 8 prompts per set");

    let (direction, meta) = read_direction(&f.dir.path().join("a.safetensors"));
    let norm: f64 = direction
        .iter()
        .map(|&x| f64::from(x).powi(2))
        .sum::<f64>()
        .sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "‖v‖ = {norm}");
    for (key, value) in [
        ("model", "qwen2-mini"),
        ("layer", "2"),
        ("site", "resid_post"),
        ("pooling", "last"),
        ("chat_template", "none"),
        ("n_pos", "6"),
        ("n_neg", "6"),
        ("dtype_used", "f32"),
        ("model_type", "qwen2"),
    ] {
        assert_eq!(meta[key], value, "{key}");
    }
    assert!(meta["raw_diff_norm"].parse::<f64>().unwrap() > 0.0);
    assert_eq!(meta["surgeon_version"], env!("CARGO_PKG_VERSION"));

    let (code, _, _) = run_args_raw(&deps(), &f.args("b.safetensors", &[]));
    assert_eq!(code, 0);
    assert_eq!(
        std::fs::read(f.dir.path().join("a.safetensors")).unwrap(),
        std::fs::read(f.dir.path().join("b.safetensors")).unwrap(),
        "deterministic"
    );
    let path = f.dir.path().join("a.safetensors");
    assert_eq!(
        weight_surgeon::engine::load_direction_vector(&path, Some("direction")).unwrap(),
        direction
    );
    assert_eq!(
        weight_surgeon::engine::load_direction_vector(&path, None).unwrap(),
        direction
    );
}

#[test]
fn extracted_vector_drives_apply_exactly() {
    let f = extract_fixture("F32", false);
    assert_eq!(run_args_raw(&deps(), &f.args("v.safetensors", &[])).0, 0);
    let vector = f.dir.path().join("v.safetensors");
    let v = weight_surgeon::engine::load_direction_vector(&vector, Some("direction")).unwrap();
    assert_eq!(v.len(), HIDDEN, "direction length == hidden_size");
    let weights = f.model.join("model.safetensors");
    let down = "model.layers.2.mlp.down_proj.weight";
    let before = read_f32(&weights, down);
    // down_proj is [hidden, intermediate]: v indexes its rows, so the engine axis is `cols`.
    let patch = json!({"schema_version": 1, "operations": [{"id": "steer", "layers": "2",
        "projections": ["down_proj"], "operation": "subtract", "vector_file": s(&vector),
        "vector_tensor": "direction", "alpha": 0.5, "axis": "cols"}]});
    let patch_path = f.dir.path().join("patch.json");
    std::fs::write(&patch_path, patch.to_string()).unwrap();
    let (code, v_json) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&weights),
            "--patch",
            s(&patch_path),
            "--dry-run",
        ],
    );
    assert_eq!(code, 0, "{v_json}");
    assert_eq!(read_f32(&weights, down), before, "dry run writes nothing");
    let (code, v_json) = run_json(
        &deps(),
        &["apply", "--model", s(&weights), "--patch", s(&patch_path)],
    );
    assert_eq!(code, 0, "{v_json}");
    let after = read_f32(&weights, down);
    for (i, (&new, &old)) in after.iter().zip(&before).enumerate() {
        let expected = old - 0.5 * v[i / INTER];
        assert_eq!(new.to_bits(), expected.to_bits(), "element {i}");
    }
}

/// Records every tensor read, delegating to the real memory-mapped source.
struct Spy {
    inner: SafetensorsDir,
    reads: ExtractCell<Vec<String>>,
}

impl TensorSource for Spy {
    fn meta(&self, name: &str) -> Option<(weight_surgeon::engine::DType, Vec<usize>)> {
        self.inner.meta(name)
    }
    fn read(&self, name: &str) -> ex::Result<RawTensor<'_>> {
        self.reads.borrow_mut().push(name.to_owned());
        self.inner.read(name)
    }
}

struct Quiet;

impl ExtractObserver for Quiet {
    fn on_event(&mut self, _: ExtractEvent) -> Flow<()> {
        Flow::Continue(())
    }
}

#[test]
fn extract_reads_nothing_past_layer_l_and_checks_resources_first() {
    let f = extract_fixture("BF16", true);
    let spy = Spy {
        inner: SafetensorsDir::open(&f.model).unwrap(),
        reads: ExtractCell::new(Vec::new()),
    };
    let report = ex::run_extraction_with(
        &f.request("l1.safetensors", 1),
        &spy,
        &FakeProbe {
            available: 64 << 30,
        },
        &mut Quiet,
    )
    .unwrap();
    assert_eq!(report.layer, 1);
    let reads = spy.reads.borrow();
    assert_eq!(reads.len(), 1 + 2 * 12);
    for name in reads.iter() {
        assert!(
            !name.contains("layers.2") && !name.contains("layers.3"),
            "{name}"
        );
        assert!(
            name != "lm_head.weight" && name != "model.norm.weight",
            "{name}"
        );
    }
    drop(reads);

    let spy = Spy {
        inner: SafetensorsDir::open(&f.model).unwrap(),
        reads: ExtractCell::new(Vec::new()),
    };
    let err = ex::run_extraction_with(
        &f.request("tiny.safetensors", 2),
        &spy,
        &FakeProbe { available: 1 << 20 },
        &mut Quiet,
    )
    .unwrap_err();
    assert!(
        matches!(err, ex::ExtractError::InsufficientResources(_)),
        "{err}"
    );
    assert!(
        spy.reads.borrow().is_empty(),
        "no weight read before the resource check"
    );
    assert!(!f.dir.path().join("tiny.safetensors").exists());

    let mut poor = deps();
    poor.probe = Box::new(FakeProbe { available: 1 << 20 });
    let (code, v) = run_args(&poor, &f.args("x.safetensors", &[]));
    assert_eq!(code, exit::GENERAL);
    assert_eq!(v["error"]["kind"], "insufficient_resources");
}

#[test]
fn extract_dtype_auto_falls_back_to_bf16() {
    let f = extract_fixture("F32", false);
    let cfg = ex::load_config(&f.model).unwrap();
    let source = SafetensorsDir::open(&f.model).unwrap();
    let totals = ex::check_tensors(&source, &ex::required_tensors(&cfg, 2)).unwrap();
    let plan = ex::plan_memory(&cfg, totals, 512);
    assert!(plan.bf16_bytes < plan.f32_bytes);
    let just_enough = ex::required_ram(plan.bf16_bytes, &BudgetPolicy::default());
    let report = ex::run_extraction(
        &f.request("auto.safetensors", 2),
        &FakeProbe {
            available: just_enough,
        },
        &mut Quiet,
    )
    .unwrap();
    assert_eq!(report.dtype_used, "bf16");
    assert!(report.warnings.iter().any(|w| w.contains("bf16")));
    let (_, meta) = read_direction(&f.dir.path().join("auto.safetensors"));
    assert_eq!(meta["dtype_used"], "bf16");
}

/// Cancels after the third prompt, as the MCP reader thread does on a cancellation.
struct CancelAtThird(usize);

impl ExtractObserver for CancelAtThird {
    fn on_event(&mut self, event: ExtractEvent) -> Flow<()> {
        if matches!(event, ExtractEvent::Prompt { .. }) {
            self.0 += 1;
        }
        match self.0 >= 3 {
            true => Flow::Break(()),
            false => Flow::Continue(()),
        }
    }
}

#[test]
fn extract_never_leaves_a_partial_or_clobbered_file() {
    let f = extract_fixture("F32", false);
    let existing = f.dir.path().join("taken.safetensors");
    std::fs::write(&existing, "keep me").unwrap();
    let (code, v) = run_args(&deps(), &f.args("taken.safetensors", &[]));
    assert_eq!(code, exit::INVALID_INPUT);
    assert_eq!(v["error"]["kind"], "output_exists");
    assert_eq!(std::fs::read(&existing).unwrap(), b"keep me");

    let err = ex::run_extraction(
        &f.request("cancel.safetensors", 2),
        &FakeProbe {
            available: 64 << 30,
        },
        &mut CancelAtThird(0),
    )
    .unwrap_err();
    assert!(matches!(err, ex::ExtractError::Cancelled));
    let names: Vec<String> = std::fs::read_dir(f.dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert!(
        !names.iter().any(|n| n.contains("cancel")),
        "no output nor temporary: {names:?}"
    );

    let (code, v) = run_args(&deps(), &f.args("missing/dir/out.safetensors", &[]));
    assert_eq!(code, exit::INVALID_INPUT);
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("inexistant"));

    let (code, _) = run_args(&deps(), &f.args("taken.safetensors", &["--overwrite"]));
    assert_eq!(code, 0, "--overwrite replaces it");
    assert_ne!(std::fs::read(&existing).unwrap(), b"keep me");
}

#[test]
fn extract_errors_are_typed() {
    let f = extract_fixture("BF16", true);
    let json_args = |out: &str, extra: &[&str]| f.args(out, extra);
    let mut out_of_range = json_args("o.safetensors", &[]);
    let layer = out_of_range.iter().position(|a| a == "--layer").unwrap();
    out_of_range[layer + 1] = "4".into();
    let (code, v) = run_args(&deps(), &out_of_range);
    assert_eq!(code, exit::INVALID_INPUT);
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("num_hidden_layers = 4"),
        "{v}"
    );

    let mut same = json_args("same.safetensors", &[]);
    let neg = same.iter().position(|a| a == "--neg-prompts").unwrap();
    same[neg + 1] = s(&f.pos).to_owned();
    let (code, v) = run_args(&deps(), &same);
    assert_eq!(code, exit::EXTRACT_FAILED);
    assert_eq!(v["error"]["kind"], "degenerate_direction");
    assert!(
        !f.dir.path().join("same.safetensors").exists(),
        "no zero vector written"
    );

    std::fs::write(
        f.model.join("config.json"),
        qwen_config("qwen3").to_string(),
    )
    .unwrap();
    let (code, v) = run_args(&deps(), &json_args("q3.safetensors", &[]));
    assert_eq!(code, exit::EXTRACT_FAILED);
    assert_eq!(v["error"]["kind"], "unsupported_architecture");
    assert!(v["error"]["message"].as_str().unwrap().contains("qwen3"));

    let g = extract_fixture("BF16", true);
    let (code, v) = run_args(
        &deps(),
        &g.args(
            "ok.safetensors",
            &[
                "--pooling",
                "mean",
                "--site",
                "mlp_out",
                "--chat-template",
                "chatml",
                "--max-tokens",
                "5",
            ],
        ),
    );
    assert_eq!(code, 0, "{v}");
    assert!(
        v["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .any(|w| w.as_str().unwrap().contains("tronqué")),
        "{v}"
    );
    let (_, meta) = read_direction(&g.dir.path().join("ok.safetensors"));
    assert_eq!(
        (
            meta["pooling"].as_str(),
            meta["site"].as_str(),
            meta["chat_template"].as_str()
        ),
        ("mean", "mlp_out", "chatml")
    );
}

#[derive(Default)]
struct Recorder(Vec<ProgressEvent>);

impl ProgressSink for Recorder {
    fn event(&mut self, event: ProgressEvent) {
        self.0.push(event);
    }
}

#[test]
fn extract_progress_one_event_per_prompt() {
    let f = extract_fixture("F32", false);
    let args = ExtractArgs {
        model: f.model.clone(),
        pos_prompts: f.pos.clone(),
        neg_prompts: f.neg.clone(),
        layer: 2,
        out: f.dir.path().join("p.safetensors"),
        pooling: ex::Pooling::Last,
        site: ex::Site::ResidPost,
        chat_template: ex::ChatTemplate::NoTemplate,
        max_tokens: 512,
        dtype: ex::DTypeChoice::F32,
        overwrite: false,
    };
    let mut recorder = Recorder::default();
    let outcome =
        weight_surgeon::app::core_extract(&args, &deps(), &Ui::off(), &mut recorder).unwrap();
    assert!(outcome.is_ok());
    let prompts: Vec<(ex::PromptSet, usize)> = recorder
        .0
        .iter()
        .filter_map(|e| match e {
            ProgressEvent::ExtractPrompt { set, done, .. } => Some((*set, *done)),
            _ => None,
        })
        .collect();
    assert_eq!(prompts.len(), 12);
    let expected: Vec<(ex::PromptSet, usize)> = (1..=6)
        .map(|d| (ex::PromptSet::Pos, d))
        .chain((1..=6).map(|d| (ex::PromptSet::Neg, d)))
        .collect();
    assert_eq!(prompts, expected);
    assert!(matches!(
        recorder.0.first(),
        Some(ProgressEvent::ExtractStart {
            n_pos: 6,
            n_neg: 6,
            ..
        })
    ));
    assert!(recorder.0.contains(&ProgressEvent::ExtractLoading {
        done: 37,
        total: 37
    }));
    assert_eq!(recorder.0.last(), Some(&ProgressEvent::ExtractWriting));
}

#[test]
fn extract_human_ui_golden_and_json_without_ui_bytes() {
    let f = extract_fixture("F32", false);
    let (_, json_out, _) = run_args_raw(
        &deps(),
        &["--json".to_owned()]
            .into_iter()
            .chain(f.args("j.safetensors", &[]))
            .collect::<Vec<_>>(),
    );
    let report: Value = serde_json::from_str(&json_out).unwrap();
    let stats = &report["data"]["stats"];

    let (code, colored, err) = run_args_raw(
        &ui_deps(ColorMode::TrueColor, false),
        &f.args("h.safetensors", &[]),
    );
    assert_eq!(code, 0, "{err}");
    assert!(colored.contains("\x1b[38;2;65;105;225m[EXTRACT]"));
    assert!(!err.contains('\r'), "no carriage return off a TTY");
    let width = Ui::new(ColorMode::Plain).width();
    let d_prime = stats["d_prime"]
        .as_f64()
        .map_or("∞".to_owned(), |d| format!("{d:.2}"));
    let expected = [
        format!(
            "[EXTRACT]  Qwen2 · couche 2 · resid_post · pooling last  {}",
            weight_surgeon::ui::truncate_middle(s(&f.model), width)
        ),
        "[LOAD]     poids chargés 37/37 tenseurs · f32 · 34.9 KiB".to_owned(),
        "[FORWARD]  pos 6/6 · neg 6/6".to_owned(),
        format!(
            "[VERDICT]  d′ {d_prime} · accuracy {:.0}% · ‖Δ‖ {:.3}",
            stats["threshold_accuracy"].as_f64().unwrap() * 100.0,
            stats["raw_diff_norm"].as_f64().unwrap()
        ),
        format!(
            "[WRITE]    {}",
            weight_surgeon::ui::truncate_middle(s(&f.dir.path().join("h.safetensors")), width)
        ),
    ]
    .into_iter()
    .chain(
        report["warnings"]
            .as_array()
            .unwrap()
            .iter()
            .map(|w| format!("  ! {}", w.as_str().unwrap())),
    )
    .collect::<Vec<_>>()
    .join("\n")
        + "\n";
    assert_eq!(strip_ansi(&colored), expected);

    let (code, tty_out, tty_err) = run_args_raw(
        &ui_deps(ColorMode::TrueColor, true),
        &f.args("t.safetensors", &[]),
    );
    assert_eq!(code, 0);
    assert!(
        tty_err.contains('\r') && strip_ansi(&tty_err).contains("[FORWARD]"),
        "in-place bar on a TTY"
    );
    assert!(strip_ansi(&tty_out).contains("[FORWARD]  pos 6/6 · neg 6/6"));

    let (code, out, err) = run_args_raw(
        &ui_deps(ColorMode::TrueColor, true),
        &["--json".to_owned()]
            .into_iter()
            .chain(f.args("k.safetensors", &[]))
            .collect::<Vec<_>>(),
    );
    assert_eq!(code, 0);
    assert!(err.is_empty(), "no ui byte on stderr under --json");
    for forbidden in [
        "\x1b",
        "█",
        "[EXTRACT]",
        "[LOAD]",
        "[FORWARD]",
        "[VERDICT]",
        "[WRITE]",
        "\r",
    ] {
        assert!(!out.contains(forbidden), "{forbidden:?}");
    }
    assert_eq!(out.matches('\n').count(), 1);
}

#[test]
fn mcp_extract_tool_end_to_end() {
    let f = extract_fixture("F32", false);
    let cfg = mcp_cfg(f.dir.path());
    let prompts = |words: &[&str]| -> Vec<String> {
        (0..6)
            .map(|i| format!("le {} est {}", words[i], words[i + 3]))
            .collect()
    };
    let args = |out: &str| json!({"model": "qwen2-mini", "pos_prompts": prompts(&POS_WORDS), "neg_prompts": prompts(&NEG_WORDS), "layer": 2, "out": out});
    let outside = tempfile::tempdir().unwrap();
    let with_token = rpc(
        json!(2),
        "tools/call",
        json!({"name": "surgeon_extract", "arguments": args("m.safetensors"), "_meta": {"progressToken": "x"}}),
    );
    let responses = session(
        &deps(),
        &cfg,
        &[
            init("2025-06-18"),
            rpc(json!(1), "tools/list", json!({})),
            with_token,
            call(3, "surgeon_extract", args("n.safetensors")),
            call(
                4,
                "surgeon_extract",
                args(s(&outside.path().join("o.safetensors"))),
            ),
            call(
                5,
                "surgeon_extract",
                json!({"model": "qwen2-mini", "pos_prompts": ["a"], "pos_prompts_file": "pos.json", "neg_prompts": ["b"], "out": "z.safetensors"}),
            ),
        ],
    );
    assert_eq!(
        reply(&responses, json!(1))["result"]["tools"]
            .as_array()
            .unwrap()
            .len(),
        9
    );
    let (is_error, env) = envelope(&responses, 2);
    assert!(!is_error, "{env}");
    assert_eq!(env["command"], "extract");
    assert_eq!(env["data"]["hidden_size"], HIDDEN);
    assert!(f.dir.path().join("m.safetensors").exists());
    let progress: Vec<&Value> = responses
        .iter()
        .filter(|r| r["method"] == "notifications/progress")
        .collect();
    assert_eq!(
        progress.len(),
        1 + 6 + 6,
        "weights, then one per prompt; none for the call without token"
    );
    for (i, p) in progress.iter().enumerate() {
        assert_eq!(p["params"]["progress"], i + 1);
        assert_eq!(p["params"]["total"], 13);
    }
    assert!(!envelope(&responses, 3).0);
    assert_eq!(
        envelope(&responses, 4).1["error"]["kind"],
        "path_not_allowed"
    );
    assert_eq!(
        envelope(&responses, 5).1["error"]["kind"],
        "invalid_arguments"
    );

    let read_only = McpConfig::new(&[f.dir.path().to_path_buf()], true, false).unwrap();
    let responses = session(
        &deps(),
        &read_only,
        &[
            init("2025-06-18"),
            call(1, "surgeon_extract", args("r.safetensors")),
        ],
    );
    assert_eq!(envelope(&responses, 1).1["error"]["kind"], "read_only");
    assert!(!f.dir.path().join("r.safetensors").exists());
}

#[test]
fn mcp_binary_extract_speaks_only_json_rpc() {
    let f = extract_fixture("BF16", false);
    let mut child = binary()
        .args(["mcp", "--allow-dir", s(f.dir.path())])
        .env("CLICOLOR_FORCE", "1")
        .env("COLORTERM", "truecolor")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    let input = [
        init("2025-06-18"),
        rpc(json!(1), "tools/list", json!({})),
        call(2, "surgeon_extract", json!({"model": "qwen2-mini", "pos_prompts_file": "pos.json", "neg_prompts_file": "neg.json", "layer": 1, "out": "bin.safetensors"})),
    ]
    .join("\n")
        + "\n";
    {
        use std::io::Write as _;
        child
            .stdin
            .take()
            .unwrap()
            .write_all(input.as_bytes())
            .unwrap();
    }
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if started.elapsed() > Duration::from_secs(60) {
            let _ = child.kill();
            panic!("surgeon mcp did not exit within 60 s");
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
    assert!(!out.contains('\x1b') && !err.contains('\x1b'));
    let responses: Vec<Value> = out
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
        .collect();
    assert_eq!(
        reply(&responses, json!(1))["result"]["tools"]
            .as_array()
            .unwrap()
            .len(),
        9
    );
    assert!(!envelope(&responses, 2).0, "{out}");
    assert!(f.dir.path().join("bin.safetensors").exists());
}

/// The extraction module never prints and never uses `unsafe` (the mmap stays in engine).
#[test]
fn extract_module_has_no_print_and_no_unsafe() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/extract");
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let source = std::fs::read_to_string(&path).unwrap();
        assert!(!source.contains("unsafe"), "{}: unsafe", path.display());
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

/// Writes the mini model and prompt files to `$SURGEON_FAKE_QWEN2_DIR` for a manual run
/// of the release binary: `cargo test --test integration make_fake_qwen2 -- --ignored`.
#[test]
#[ignore]
fn make_fake_qwen2_on_disk() {
    let dir = PathBuf::from(std::env::var("SURGEON_FAKE_QWEN2_DIR").unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    write_fake_qwen2_dir(&dir, "F32", false);
    write_prompt_files(&dir);
}

// ------------------------------------------------------------------ tui (prompt 9)

use std::sync::atomic::AtomicUsize;

use ratatui::backend::TestBackend as TuiBackend;
use ratatui::crossterm::event::{Event as TermEvent, KeyCode as TermKey};
use ratatui::style::{Color as TuiColor, Modifier as TuiModifier};
use ratatui::Terminal as TuiTerminal;
use weight_surgeon::engine::stats as estats;
use weight_surgeon::tui::{
    self, App as TuiApp, Computation as TuiComputation, EventSource, ModelView, Msg as TuiMsg,
    ScriptedEventSource, StatsProvider, StatsRequest, Tab as TuiTab, TensorEntry,
    Theme as TuiTheme, TuiError, VectorView, Worker,
};

fn f32_le(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn stats_of(values: &[f32], bins: usize) -> estats::TensorStats {
    let never = AtomicBool::new(false);
    estats::tensor_stats(
        &f32_le(values),
        &weight_surgeon::engine::DType::F32,
        &[values.len()],
        bins,
        &never,
    )
    .unwrap()
}

fn tui_entries(layers: usize, hidden: usize) -> Vec<TensorEntry> {
    let entry = |name: String, shape: Vec<usize>| TensorEntry {
        name,
        dtype: "F32".into(),
        bytes: shape.iter().product::<usize>() as u64 * 4,
        shape,
    };
    let mut entries = vec![entry("model.embed_tokens.weight".into(), vec![64, hidden])];
    for i in 0..layers {
        for p in ["gate_proj", "up_proj"] {
            entries.push(entry(
                format!("model.layers.{i}.mlp.{p}.weight"),
                vec![32, hidden],
            ));
        }
        entries.push(entry(
            format!("model.layers.{i}.mlp.down_proj.weight"),
            vec![hidden, 32],
        ));
    }
    entries
}

fn tui_app(mode: ColorMode, layers: usize) -> TuiApp {
    let model = ModelView::from_entries(Path::new("modele"), &tui_entries(layers, 16));
    TuiApp::new(Some(model), None, TuiTheme::new(mode), (100, 32))
}

fn tui_buffer(app: &TuiApp, w: u16, h: u16) -> ratatui::buffer::Buffer {
    let mut terminal = TuiTerminal::new(TuiBackend::new(w, h)).unwrap();
    terminal.draw(|f| tui::view(f, app)).unwrap();
    terminal.backend().buffer().clone()
}

/// (x, y) of the first char of `needle` in the buffer (every glyph used is one cell wide).
fn find_cell(buffer: &ratatui::buffer::Buffer, needle: &str) -> Option<(u16, u16)> {
    let width = buffer.area.width as usize;
    buffer
        .content()
        .chunks(width)
        .enumerate()
        .find_map(|(y, row)| {
            let line: Vec<&str> = row.iter().map(|c| c.symbol()).collect();
            let text = line.concat();
            text.find(needle)
                .map(|byte| (text[..byte].chars().count() as u16, y as u16))
        })
}

#[test]
fn tui_inspector_render_contains_list_metrics_and_histogram() {
    let mut app = tui_app(ColorMode::TrueColor, 4);
    app.computation =
        TuiComputation::Done(Ok(stats_of(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0], 64)));
    let text = tui::render_snapshot(&app, 100, 32).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 32);
    assert!(lines.iter().all(|l| l.chars().count() <= 100));
    assert!(lines[1].contains("1 Inspecteur de poids") && lines[1].contains("2 Steering vector"));
    for expected in [
        "Couches (4)",
        "[gate]",
        "▶L0",
        "tenseur         model.layers.0.mlp.gate_proj.weight",
        "dtype · shape   F32 · [32, 16] · 2048 octets",
        "min · max       0.000000 · 6.000000",
        "moyenne         2.625000e0",
        "écart-type      2.117634e0 (population)",
        "sparsité        25.000 % de zéros exacts",
        "NaN · ±Inf      0 NaN · 0 +Inf · 0 -Inf",
        "éléments finis  8 / 8",
        "Distribution des valeurs finies (linéaire)",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in\n{text}");
    }
    assert!(
        ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
            .iter()
            .any(|b| text.contains(b)),
        "histogram blocks"
    );
    assert!(lines[31].contains("↑↓ couche") && lines[31].contains("prêt"));
}

#[test]
fn tui_small_terminal_and_resize() {
    let mut app = tui_app(ColorMode::Plain, 2);
    let small = tui::render_snapshot(&app, 79, 23).unwrap();
    assert!(
        small.contains("Terminal trop petit (80×24 minimum)"),
        "{small}"
    );
    assert!(!tui::render_snapshot(&app, 80, 24)
        .unwrap()
        .contains("trop petit"));
    app.dirty = false;
    tui::update(&mut app, TuiMsg::Resize(120, 40));
    assert_eq!((app.size, app.dirty), ((120, 40), true));
    assert_eq!(
        tui::render_snapshot(&app, 120, 40).unwrap().lines().count(),
        40
    );
}

#[test]
fn tui_nan_alert_blinks_color_not_text() {
    let mut values = vec![f32::NAN; 12];
    values.extend([f32::INFINITY; 3]);
    values.extend([1.0, 2.0, 3.0]);
    let stats = stats_of(&values, 64);
    assert_eq!(
        stats.histogram.as_ref().unwrap().counts.iter().sum::<u64>(),
        3,
        "no NaN/Inf in the bins"
    );
    let alert = "⚠ NaN/Inf DÉTECTÉ : 12 NaN · 3 +Inf · 0 -Inf";
    for mode in [ColorMode::TrueColor, ColorMode::Plain] {
        let mut app = tui_app(mode, 2);
        app.computation = TuiComputation::Done(Ok(stats.clone()));
        app.flagged.insert((0, 0), true);
        app.ticks = 0;
        let on = tui_buffer(&app, 100, 32);
        app.ticks = weight_surgeon::tui::app::BLINK_TICKS;
        let off = tui_buffer(&app, 100, 32);
        let (x, y) =
            find_cell(&on, alert).unwrap_or_else(|| panic!("alert missing (on, {mode:?})"));
        assert!(
            find_cell(&off, alert).is_some(),
            "the text stays when the blink is off"
        );
        let (cell_on, cell_off) = (&on[(x, y)], &off[(x, y)]);
        match mode {
            ColorMode::TrueColor => {
                assert_eq!(cell_on.fg, TuiColor::Rgb(255, 85, 85));
                assert_ne!(cell_off.fg, TuiColor::Rgb(255, 85, 85));
            }
            _ => {
                assert!(
                    cell_on.modifier.contains(TuiModifier::REVERSED),
                    "no color: reversed instead"
                );
                assert!(!cell_off.modifier.contains(TuiModifier::REVERSED));
                assert_eq!(cell_on.fg, TuiColor::Reset);
            }
        }
        let text = tui::render_snapshot(&app, 100, 32).unwrap();
        assert!(
            text.lines().any(|l| l.contains("▶L0") && l.contains('⚠')),
            "⚠ marker in the layer list"
        );
    }
}

struct SpyProvider {
    entries: Vec<TensorEntry>,
    calls: Arc<AtomicUsize>,
    fail: bool,
}

impl StatsProvider for SpyProvider {
    fn list_tensors(&self) -> Vec<TensorEntry> {
        self.entries.clone()
    }
    fn tensor_stats(
        &self,
        name: &str,
        bins: usize,
        _cancel: &AtomicBool,
    ) -> Result<estats::TensorStats, TuiError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.fail {
            true => Err(TuiError::Stats(format!("tenseur introuvable : {name}"))),
            false => Ok(stats_of(&[1.0, -1.0, 0.5], bins)),
        }
    }
}

fn spy(fail: bool) -> (SpyProvider, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    (
        SpyProvider {
            entries: tui_entries(3, 16),
            calls: Arc::clone(&calls),
            fail,
        },
        calls,
    )
}

fn wait_reply(worker: &Worker) -> weight_surgeon::tui::StatsReply {
    let started = Instant::now();
    loop {
        if let Some(reply) = worker.try_recv() {
            return reply;
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "worker never answered"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn request(generation: u64, tensor: &str) -> StatsRequest {
    StatsRequest {
        generation,
        key: (0, 0),
        tensor: tensor.into(),
        bins: 16,
    }
}

/// Blocks on "slow" until cancelled; "fast" answers at once.
struct SlowProvider {
    started: Arc<AtomicBool>,
    saw_cancel: Arc<AtomicBool>,
}

impl StatsProvider for SlowProvider {
    fn list_tensors(&self) -> Vec<TensorEntry> {
        Vec::new()
    }
    fn tensor_stats(
        &self,
        name: &str,
        bins: usize,
        cancel: &AtomicBool,
    ) -> Result<estats::TensorStats, TuiError> {
        if name == "slow" {
            self.started.store(true, Ordering::SeqCst);
            let begin = Instant::now();
            while begin.elapsed() < Duration::from_secs(10) {
                if cancel.load(Ordering::SeqCst) {
                    self.saw_cancel.store(true, Ordering::SeqCst);
                    return Err(TuiError::Stats("annulé".into()));
                }
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        Ok(stats_of(&[2.0], bins))
    }
}

#[test]
fn tui_worker_caches_cancels_and_reports_errors() {
    let (provider, calls) = spy(false);
    let worker = Worker::spawn(Arc::new(provider), 64);
    worker.request(request(1, "t"));
    assert_eq!(wait_reply(&worker).generation, 1);
    worker.request(request(2, "t"));
    let second = wait_reply(&worker);
    assert!(second.result.is_ok() && second.generation == 2);
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "same (layer, projection): served from the cache"
    );

    let started = Arc::new(AtomicBool::new(false));
    let saw_cancel = Arc::new(AtomicBool::new(false));
    let worker = Worker::spawn(
        Arc::new(SlowProvider {
            started: Arc::clone(&started),
            saw_cancel: Arc::clone(&saw_cancel),
        }),
        64,
    );
    worker.request(StatsRequest {
        key: (5, 0),
        ..request(1, "slow")
    });
    let begin = Instant::now();
    while !started.load(Ordering::SeqCst) {
        assert!(begin.elapsed() < Duration::from_secs(10));
        std::thread::sleep(Duration::from_millis(1));
    }
    worker.request(StatsRequest {
        key: (6, 0),
        ..request(2, "fast")
    });
    let reply = wait_reply(&worker);
    assert_eq!(reply.generation, 2, "the cancelled job sends nothing");
    assert!(
        saw_cancel.load(Ordering::SeqCst),
        "selection change cancels the running job"
    );

    let dir = tempfile::tempdir().unwrap();
    let provider = weight_surgeon::tui::ModelStatsProvider::open(&model(dir.path())).unwrap();
    let worker = Worker::spawn(Arc::new(provider), 64);
    worker.request(request(1, "nope"));
    let reply = wait_reply(&worker);
    let message = reply.result.clone().unwrap_err();
    assert!(message.contains("introuvable"), "{message}");
    let mut app = tui_app(ColorMode::Plain, 2);
    app.start();
    let generation = app.generation;
    tui::update(
        &mut app,
        TuiMsg::StatsReady {
            generation,
            key: (0, 0),
            result: Err(message),
        },
    );
    assert!(tui::render_snapshot(&app, 100, 32)
        .unwrap()
        .contains("erreur : statistiques : tenseur introuvable : nope"));
}

/// Vector file (`direction` + metadata) and optional companion.
fn write_vector(dir: &Path, name: &str, values: &[f32], companion: Option<Value>) -> PathBuf {
    let path = dir.join(name);
    let data = f32_le(values);
    let metadata = json!({"model": "qwen2-mini", "layer": "2", "site": "resid_post", "pooling": "last",
        "n_pos": "4", "n_neg": "4", "d_prime": "2.5", "threshold_accuracy": "0.9",
        "pos_mean_proj": "1.15", "neg_mean_proj": "-0.35", "cos_mean_pos_neg": "0.1234"});
    let mut header = json!({"__metadata__": metadata, "direction": {"dtype": "F32", "shape": [values.len()], "data_offsets": [0, data.len()]}}).to_string();
    while !(8 + header.len()).is_multiple_of(8) {
        header.push(' ');
    }
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    file.extend_from_slice(&data);
    std::fs::write(&path, file).unwrap();
    if let Some(c) = companion {
        std::fs::write(
            weight_surgeon::extract::companion_path(&path),
            c.to_string(),
        )
        .unwrap();
    }
    path
}

fn unit_vector(n: usize) -> Vec<f32> {
    let raw: Vec<f64> = (0..n)
        .map(|i| (i as f64 + 1.0) * if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
    raw.iter().map(|x| (x / norm) as f32).collect()
}

fn companion_json(schema: u32) -> Value {
    json!({"schema": schema, "pos_projections": [1.0, 1.2, 0.9, 1.5], "neg_projections": [-0.5, -0.2, 0.1, -0.8],
        "threshold": 0.5, "d_prime": 2.5, "threshold_accuracy": 0.9, "cos_mean_pos_neg": 0.1234,
        "raw_diff_norm": 3.0, "layer": 2, "site": "resid_post", "pooling": "last", "hidden_size": 16})
}

fn steering_text(vector: VectorView, model: Option<ModelView>) -> String {
    let app = TuiApp::new(
        model,
        Some(vector),
        TuiTheme::new(ColorMode::Plain),
        (100, 32),
    );
    let app = TuiApp {
        tab: TuiTab::Steering,
        ..app
    };
    tui::render_snapshot(&app, 100, 32).unwrap()
}

#[test]
fn tui_steering_tab_with_and_without_companion() {
    let dir = tempfile::tempdir().unwrap();
    let values = unit_vector(16);
    let path = write_vector(
        dir.path(),
        "v.safetensors",
        &values,
        Some(companion_json(1)),
    );
    let text = steering_text(VectorView::load(&path).unwrap(), None);
    for expected in [
        "taille          16",
        "norme L2        1.000  normalisé ✓",
        "d′ (séparation)",
        "2.50",
        "90 %",
        "cos(μ₊, μ₋)     0.1234",
        "proj. μ₊ · μ₋   1.1500 · -0.3500",
        "couche · site   2 · resid_post",
        "seuil 0.5000",
    ] {
        assert!(text.contains(expected), "missing {expected:?} in\n{text}");
    }
    assert!(text.contains('┼'), "threshold marker on the butterfly axis");
    let mut order: Vec<usize> = (0..16).collect();
    order.sort_by(|&a, &b| values[b].abs().total_cmp(&values[a].abs()).then(a.cmp(&b)));
    for (rank, &i) in order.iter().take(10).enumerate() {
        let line = format!("{:>2}. dim {i:<5}{:+.4}", rank + 1, values[i]);
        assert!(text.contains(&line), "top-10 {line:?} in\n{text}");
    }

    let bare = write_vector(dir.path(), "bare.safetensors", &values, None);
    let text = steering_text(VectorView::load(&bare).unwrap(), None);
    assert!(
        text.contains("projections indisponibles : relancer `surgeon")
            && text.contains("extract` (v4.0+)"),
        "wrapped, not cut:\n{text}"
    );
    let unknown = write_vector(
        dir.path(),
        "future.safetensors",
        &values,
        Some(companion_json(2)),
    );
    assert!(steering_text(VectorView::load(&unknown).unwrap(), None)
        .contains("fichier compagnon ignoré : schema 2 inconnu"));
    std::fs::write(weight_surgeon::extract::companion_path(&unknown), "{oops").unwrap();
    assert!(steering_text(VectorView::load(&unknown).unwrap(), None)
        .contains("fichier compagnon ignoré"));

    let half: Vec<f32> = values.iter().map(|v| v * 0.5).collect();
    let half_path = write_vector(dir.path(), "half.safetensors", &half, None);
    assert!(steering_text(VectorView::load(&half_path).unwrap(), None)
        .contains("0.500  non normalisé ⚠"));
    let mut broken = values.clone();
    broken[3] = f32::NAN;
    let nan_path = write_vector(dir.path(), "nan.safetensors", &broken, None);
    assert!(steering_text(VectorView::load(&nan_path).unwrap(), None)
        .contains("⚠ NaN/Inf DÉTECTÉ : 1 NaN · 0 +Inf · 0 -Inf"));

    let big = write_vector(dir.path(), "big.safetensors", &vec![0.01; 4096], None);
    let model = ModelView::from_entries(Path::new("m"), &tui_entries(2, 16));
    assert!(
        steering_text(VectorView::load(&big).unwrap(), Some(model.clone()))
            .contains("incompatible ✗ (4096 ≠ 16)")
    );
    assert!(steering_text(VectorView::load(&path).unwrap(), Some(model)).contains("compatible ✓"));
}

#[test]
fn tui_view_is_pure_and_idle_loop_does_not_redraw() {
    let (provider, calls) = spy(false);
    let model = ModelView::from_entries(Path::new("m"), &provider.list_tensors());
    let mut app = TuiApp::new(
        Some(model),
        None,
        TuiTheme::new(ColorMode::TrueColor),
        (100, 32),
    );
    app.computation = TuiComputation::Done(Ok(stats_of(&[1.0, 2.0], 16)));
    for _ in 0..100 {
        tui::render_snapshot(&app, 100, 32).unwrap();
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0, "view never computes");
    drop(provider);

    let dir = tempfile::tempdir().unwrap();
    let mut broken = unit_vector(16);
    broken[0] = f32::NAN;
    let path = write_vector(dir.path(), "nan.safetensors", &broken, None);
    let ticks = 100usize;
    for (vector, alert) in [
        (Some(VectorView::load(&path).unwrap()), true),
        (None, false),
    ] {
        let mut app = TuiApp::new(None, vector, TuiTheme::new(ColorMode::TrueColor), (100, 32));
        let mut terminal = TuiTerminal::new(TuiBackend::new(100, 32)).unwrap();
        let mut events = ScriptedEventSource::new((0..ticks).map(|_| None));
        let stats = tui::run_loop(&mut terminal, &mut app, &mut events, None).unwrap();
        let flips = match alert {
            true => ticks / weight_surgeon::tui::app::BLINK_TICKS as usize,
            false => 0,
        };
        assert_eq!(stats.ticks, ticks);
        assert!(
            stats.draws <= flips + 1,
            "{} draws for {flips} blink flips",
            stats.draws
        );
        assert!(stats.draws >= 1);
    }
}

#[derive(Default)]
struct MockGuard {
    restores: usize,
}

impl weight_surgeon::tui::TerminalGuard for MockGuard {
    fn restore(&mut self) -> std::io::Result<()> {
        self.restores += 1;
        Ok(())
    }
}

struct PanickingEvents;

impl EventSource for PanickingEvents {
    fn poll(&mut self, _: Duration) -> std::io::Result<Option<TermEvent>> {
        panic!("simulated panic in the loop");
    }
}

#[test]
fn tui_full_loop_restores_terminal_once() {
    let dir = tempfile::tempdir().unwrap();
    let vector = VectorView::load(&write_vector(
        dir.path(),
        "v.safetensors",
        &unit_vector(16),
        None,
    ))
    .unwrap();
    for fail in [false, true] {
        let (provider, _) = spy(fail);
        let model = ModelView::from_entries(Path::new("m"), &provider.list_tensors());
        let worker = Worker::spawn(Arc::new(provider), 64);
        let mut app = TuiApp::new(
            Some(model),
            Some(vector.clone()),
            TuiTheme::new(ColorMode::Plain),
            (100, 32),
        );
        let mut terminal = TuiTerminal::new(TuiBackend::new(100, 32)).unwrap();
        let mut guard = MockGuard::default();
        let mut events = ScriptedEventSource::keys([
            TermKey::Tab,
            TermKey::Down,
            TermKey::Down,
            TermKey::Right,
            TermKey::Char('q'),
        ]);
        tui::run_session(
            &mut guard,
            &mut terminal,
            &mut app,
            &mut events,
            Some(&worker),
        )
        .unwrap();
        assert_eq!(
            (app.tab, app.top_scroll, app.pan, app.selection.layer),
            (TuiTab::Steering, 2, 1, 0)
        );
        assert!(app.should_quit);
        assert_eq!(guard.restores, 1, "fail={fail}");
    }

    let mut app = TuiApp::new(None, None, TuiTheme::new(ColorMode::Plain), (100, 32));
    let mut terminal = TuiTerminal::new(TuiBackend::new(100, 32)).unwrap();
    let mut guard = MockGuard::default();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tui::run_session(
            &mut guard,
            &mut terminal,
            &mut app,
            &mut PanickingEvents,
            None,
        )
    }));
    assert!(outcome.is_err());
    assert_eq!(guard.restores, 1, "restored during unwinding");
}

#[test]
fn tui_cli_snapshot_non_tty_json_and_bad_paths() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let (code, out, err) = run(&deps(), &["tui", "--model", s(&m)]);
    assert_eq!(code, 1, "{err}");
    assert!(out.is_empty());
    assert!(err.contains("nécessite un terminal interactif") && err.contains("--snapshot"));

    let (code, out, err) = run(
        &ui_deps(ColorMode::TrueColor, true),
        &["tui", "--snapshot", "--model", s(&m), "--size", "90x30"],
    );
    assert_eq!(code, 0, "{err}");
    assert!(!out.contains('\x1b'));
    assert_eq!(out.lines().count(), 30);
    assert!(out.lines().all(|l| l.chars().count() <= 90));
    assert!(out.contains("Couches (1)") && out.contains("min · max"));

    let (code, out, _) = run(&deps(), &["tui", "--snapshot"]);
    assert_eq!(code, 0);
    assert!(out.contains("Aucun modèle ni vecteur chargé"));

    let (code, out, err) = run(&deps(), &["--json", "tui", "--snapshot"]);
    assert_eq!(code, exit::INVALID_INPUT);
    assert!(out.is_empty() && err.contains("pas de sortie JSON"));
    let (code, _, err) = run(
        &deps(),
        &[
            "tui",
            "--snapshot",
            "--model",
            s(&dir.path().join("absent")),
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT, "{err}");
    std::fs::write(
        dir.path().join("corrupt.safetensors"),
        b"not a safetensors file",
    )
    .unwrap();
    let (code, _, _) = run(
        &deps(),
        &[
            "tui",
            "--snapshot",
            "--vector",
            s(&dir.path().join("corrupt.safetensors")),
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT);
    let (code, _, _) = run(&deps(), &["tui", "--snapshot", "--size", "12"]);
    assert_eq!(code, exit::INVALID_INPUT);
}

#[test]
fn tui_binary_snapshot_has_no_ansi_and_refuses_pipes() {
    let dir = tempfile::tempdir().unwrap();
    let m = model(dir.path());
    let out = binary()
        .args(["tui", "--snapshot", "--model", s(&m)])
        .env("CLICOLOR_FORCE", "1")
        .env("COLORTERM", "truecolor")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(!text.contains('\x1b') && text.lines().count() == 32);
    let piped = binary()
        .args(["tui"])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert_eq!(piped.status.code(), Some(1));
    assert!(piped.stdout.is_empty());
}

#[test]
fn extract_writes_new_metadata_and_companion() {
    let f = extract_fixture("F32", false);
    let (code, v) = run_args(&deps(), &f.args("c.safetensors", &[]));
    assert_eq!(code, 0, "{v}");
    let out = f.dir.path().join("c.safetensors");
    let companion_path = weight_surgeon::extract::companion_path(&out);
    assert_eq!(v["data"]["stats_path"], s(&companion_path));
    let companion: Value =
        serde_json::from_str(&std::fs::read_to_string(&companion_path).unwrap()).unwrap();
    assert_eq!(companion["schema"], 1);
    assert_eq!(companion["pos_projections"].as_array().unwrap().len(), 6);
    assert_eq!(companion["neg_projections"].as_array().unwrap().len(), 6);
    assert_eq!(
        (
            companion["layer"].as_u64(),
            companion["hidden_size"].as_u64()
        ),
        (Some(2), Some(HIDDEN as u64))
    );
    assert_eq!(
        (companion["site"].as_str(), companion["pooling"].as_str()),
        (Some("resid_post"), Some("last"))
    );
    assert_eq!(
        companion["threshold_accuracy"],
        v["data"]["stats"]["threshold_accuracy"]
    );
    assert_eq!(companion["threshold"], v["data"]["stats"]["threshold"]);
    let (_, meta) = read_direction(&out);
    for key in [
        "d_prime",
        "threshold_accuracy",
        "pos_mean_proj",
        "neg_mean_proj",
        "cos_mean_pos_neg",
    ] {
        let parsed = meta[key].parse::<f64>();
        assert!(
            parsed.is_ok() || meta[key] == "inf",
            "{key} = {:?}",
            meta[key]
        );
    }
    assert_eq!(
        meta["cos_mean_pos_neg"].parse::<f64>().unwrap(),
        v["data"]["stats"]["cosine_means"].as_f64().unwrap()
    );
    assert_eq!(
        weight_surgeon::engine::load_direction_vector(&out, None)
            .unwrap()
            .len(),
        HIDDEN,
        "still one 1D tensor"
    );

    let orphan = f.dir.path().join("o.safetensors");
    std::fs::write(weight_surgeon::extract::companion_path(&orphan), "old").unwrap();
    let (code, v) = run_args(&deps(), &f.args("o.safetensors", &[]));
    assert_eq!(
        (code, v["error"]["kind"].as_str()),
        (exit::INVALID_INPUT, Some("output_exists"))
    );
    assert!(!orphan.exists());
    let (code, _) = run_args(&deps(), &f.args("o.safetensors", &["--overwrite"]));
    assert_eq!(code, 0, "--overwrite covers the companion");
    assert_ne!(
        std::fs::read_to_string(weight_surgeon::extract::companion_path(&orphan)).unwrap(),
        "old"
    );

    let mut same = f.args("deg.safetensors", &[]);
    let neg = same.iter().position(|a| a == "--neg-prompts").unwrap();
    same[neg + 1] = s(&f.pos).to_owned();
    assert_eq!(run_args(&deps(), &same).0, exit::EXTRACT_FAILED);
    assert!(
        !weight_surgeon::extract::companion_path(&f.dir.path().join("deg.safetensors")).exists(),
        "no companion left"
    );
}

/// Extends the prompt 6 guard to the TUI: no print macro, no `unsafe`.
#[test]
fn tui_module_has_no_print_and_no_unsafe() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/tui");
    let mut files = Vec::new();
    for dir in [root.clone(), root.join("widgets")] {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    assert!(files.len() >= 14, "{files:?}");
    for path in files {
        let source = std::fs::read_to_string(&path).unwrap();
        assert!(!source.contains("unsafe"), "{}: unsafe", path.display());
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

/// Metrics for the deliverable (not a gate):
/// `cargo test --release --test integration tui_metrics -- --ignored --nocapture`.
#[test]
#[ignore]
fn tui_metrics() {
    let (rows, cols) = (4096usize, 11008usize);
    let bytes: Vec<u8> = (0..rows * cols)
        .flat_map(|i| half::bf16::from_f32(((i % 1999) as f32 - 999.0) / 997.0).to_le_bytes())
        .collect();
    let never = AtomicBool::new(false);
    let started = Instant::now();
    let stats = estats::tensor_stats(
        &bytes,
        &weight_surgeon::engine::DType::BF16,
        &[rows, cols],
        64,
        &never,
    )
    .unwrap();
    eprintln!(
        "METRIC stats 4096x11008 bf16 ({} MB): {:?}, mean {:.6}",
        bytes.len() / 1_000_000,
        started.elapsed(),
        stats.mean
    );

    let mut app = tui_app(ColorMode::TrueColor, 28);
    app.computation = TuiComputation::Done(Ok(stats));
    let mut terminal = TuiTerminal::new(TuiBackend::new(120, 40)).unwrap();
    let started = Instant::now();
    for _ in 0..200 {
        terminal.draw(|f| tui::view(f, &app)).unwrap();
    }
    eprintln!(
        "METRIC render 120x40: {:?} per frame",
        started.elapsed() / 200
    );

    // Real time: the poll really waits, and a worker answers the first request.
    let (provider, _) = spy(false);
    let model = ModelView::from_entries(Path::new("m"), &provider.list_tensors());
    let worker = Worker::spawn(Arc::new(provider), 64);
    let mut idle = TuiApp::new(
        Some(model),
        None,
        TuiTheme::new(ColorMode::TrueColor),
        (120, 40),
    );
    let mut events = IdleEvents(Instant::now() + Duration::from_secs(5));
    let loop_stats = tui::run_loop(&mut terminal, &mut idle, &mut events, Some(&worker)).unwrap();
    eprintln!(
        "METRIC redraws over 5 s idle ({} ticks, first frame and stats reply included): {}",
        loop_stats.ticks, loop_stats.draws
    );
}

/// Sleeps through each poll like a terminal with no input, then sends `q`.
struct IdleEvents(Instant);

impl EventSource for IdleEvents {
    fn poll(&mut self, timeout: Duration) -> std::io::Result<Option<TermEvent>> {
        if Instant::now() >= self.0 {
            return Ok(Some(TermEvent::Key(
                ratatui::crossterm::event::KeyEvent::from(TermKey::Char('q')),
            )));
        }
        std::thread::sleep(timeout);
        Ok(None)
    }
}

// ------------------------------------------------------------------ bench (prompt 10)

use std::collections::VecDeque;

use weight_surgeon::eval::StreamControl;

/// `logits` [rows, 8] in F32 (no `prompt_offsets`).
fn logits_dump(dir: &Path, name: &str, rows: &[[f32; 8]]) -> PathBuf {
    let path = dir.join(name);
    let values: Vec<f32> = rows.iter().flatten().copied().collect();
    write_safetensors(&path, &[("logits", [rows.len(), 8], values)]);
    path
}

/// Position 0 unchanged; position 1 goes from uniform to `[0, .., 0, ln 3]`: KL = 0.0858,
/// argmax and top-5 set changed, overlap 4/5, cos(p, q) = 0.8839.
fn golden_dumps(dir: &Path) -> (PathBuf, PathBuf) {
    let mut moved = [0.0f32; 8];
    moved[7] = 3f32.ln();
    (
        logits_dump(dir, "before.safetensors", &[[0.0; 8], [0.0; 8]]),
        logits_dump(dir, "after.safetensors", &[[0.0; 8], moved]),
    )
}

#[test]
fn bench_logits_cli_json_thresholds_and_exit_codes() {
    let dir = tempfile::tempdir().unwrap();
    let (before, after) = golden_dumps(dir.path());
    let args = [
        "bench",
        "logits",
        "--before",
        s(&before),
        "--after",
        s(&after),
    ];
    let (code, v) = run_json(&deps(), &args);
    assert_eq!(code, exit::OK);
    assert_eq!(v["command"], "bench logits");
    let d = &v["data"];
    assert!(
        (d["kl_mean"].as_f64().unwrap() - 0.042_908_6).abs() < 1e-6,
        "{d}"
    );
    assert_eq!(d["kl_max_position"], 1);
    assert_eq!(d["top1_shift_rate"], 0.5);
    assert_eq!(d["k"], 5);
    assert_eq!(d["topk_set_change_rate"], 0.5);
    assert_eq!(d["kl_direction"], "KL(before‖after)");
    assert_eq!(d["verdict"], "none");
    assert!(d["timing_ms"]["total"].is_u64());

    let mut warn = args.to_vec();
    warn.extend(["--warn-kl", "0.01"]);
    let (code, v) = run_json(&deps(), &warn);
    assert_eq!(code, exit::OK, "a warning alone never fails");
    assert_eq!(v["data"]["verdict"], "warn");
    assert!(v["warnings"][0].as_str().unwrap().contains("KL moyenne"));
    warn.push("--fail-on-warn");
    let (code, v) = run_json(&deps(), &warn);
    assert_eq!(code, 9);
    assert_eq!(v["error"]["kind"], "threshold_exceeded");
    assert_eq!(v["data"]["verdict"], "warn", "the report is kept");

    let small = logits_dump(dir.path(), "small.safetensors", &[[0.0; 8]]);
    let (code, v) = run_json(
        &deps(),
        &[
            "bench",
            "logits",
            "--before",
            s(&before),
            "--after",
            s(&small),
        ],
    );
    assert_eq!(code, 8);
    assert_eq!(v["error"]["kind"], "shape_mismatch");
    let (code, v) = run_json(
        &deps(),
        &[
            "bench",
            "logits",
            "--before",
            s(&before),
            "--after",
            "absent.safetensors",
        ],
    );
    assert_eq!(code, exit::INVALID_INPUT);
    assert_eq!(v["error"]["kind"], "invalid_input");
    let (code, _) = run_json(&deps(), &[&args[..], &["--top-k", "9"]].concat());
    assert_eq!(code, exit::INVALID_INPUT, "k above the vocabulary");

    let mut panicking = deps();
    panicking.generator_factory = Box::new(|_, _| panic!("factory exploded"));
    std::fs::write(dir.path().join("m.gguf"), b"GGUF").unwrap();
    std::fs::write(
        dir.path().join("d.json"),
        r#"[{"question":"q","answer":"1"}]"#,
    )
    .unwrap();
    let (code, v) = run_json(
        &panicking,
        &[
            "bench",
            "eval",
            "--model",
            s(&dir.path().join("m.gguf")),
            "--dataset",
            s(&dir.path().join("d.json")),
        ],
    );
    assert_eq!(code, exit::INTERNAL);
    assert_eq!(v["error"]["kind"], "internal");
}

#[test]
fn bench_human_output_golden_in_truecolor() {
    use weight_surgeon::ui::strip_ansi;
    let dir = tempfile::tempdir().unwrap();
    let (before, after) = golden_dumps(dir.path());
    let deps = ui_deps(ColorMode::TrueColor, false);
    let args = [
        "bench",
        "logits",
        "--before",
        s(&before),
        "--after",
        s(&after),
        "--warn-kl",
        "0.01",
    ];
    let (code, out, err) = run(&deps, &args);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\x1b[38;2;65;105;225m"), "royal blue badges");
    assert_eq!(
        strip_ansi(&out),
        "[BENCH]    logits · 2 positions · vocab 8\n\
         [KL]       moyenne 0.0429 · max 0.0858 @ 1 · p95 0.0858 · cos(p,q) 0.9419\n\
         [TOP-K]    top-1 changé 50.0 % · top-5 (ensemble) 50.0 % · recouvrement 0.90\n\
         [VERDICT]  avertissement : KL moyenne 0.0429 > 0.01\n"
    );

    let f = ppl_fixture();
    let deps = ppl_deps(
        ColorMode::TrueColor,
        vec![ppl_output("7.2000"), ppl_output("7.2500")],
    )
    .0;
    let (code, out, err) = run(
        &deps,
        &[
            "bench",
            "ppl",
            "--model",
            s(&f.after),
            "--text",
            s(&f.text),
            "--baseline-model",
            s(&f.before),
            "--llama-cpp-dir",
            s(&f.llama),
        ],
    );
    assert_eq!(code, 0, "{err}");
    let text = strip_ansi(&out);
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines[0].starts_with("[BENCH]    ppl · ctx 512  "), "{text}");
    assert_eq!(
        lines[1],
        "[PPL]      7.2000 ± 0.04 → 7.2500 ± 0.04 · Δ +0.0500 (dans le bruit)"
    );
    assert_eq!(lines[2], "[VERDICT]  ok · seuils respectés");

    let e = eval_fixture_bench();
    let mut deps = ui_deps(ColorMode::TrueColor, false);
    deps.generator_factory = Box::new(answer_by_model);
    let (code, out, err) = run(
        &deps,
        &[
            "bench",
            "eval",
            "--model",
            s(&e.after),
            "--dataset",
            s(&e.dataset),
        ],
    );
    assert_eq!(code, 0, "{err}");
    let text = strip_ansi(&out);
    let lines: Vec<&str> = text.lines().collect();
    assert!(
        lines[0].starts_with("[BENCH]    eval · 2 questions · auto  "),
        "{text}"
    );
    assert_eq!(
        lines[1],
        "[PASS@1]   1/2 · 50.0 % · IC95 [9.5 ; 90.5] · boucles 0"
    );
    assert_eq!(lines[2], "[VERDICT]  aucun seuil fourni");
    assert!(lines[3].contains("échantillon réduit (n = 2)"), "{text}");
}

#[test]
fn bench_json_on_the_binary_is_one_clean_document_even_with_forced_colors() {
    let dir = tempfile::tempdir().unwrap();
    let (before, after) = golden_dumps(dir.path());
    let out = binary()
        .args([
            "--json",
            "bench",
            "logits",
            "--before",
            s(&before),
            "--after",
            s(&after),
        ])
        .env("CLICOLOR_FORCE", "1")
        .env("COLORTERM", "truecolor")
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8(out.stdout).unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    assert!(
        !stdout.contains('\x1b') && !stderr.contains('\x1b'),
        "{stderr}"
    );
    let v: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["ok"], true);
    assert_eq!(v["command"], "bench logits");

    let human = binary()
        .args([
            "bench",
            "logits",
            "--before",
            s(&before),
            "--after",
            s(&after),
        ])
        .env("CLICOLOR_FORCE", "1")
        .env("COLORTERM", "truecolor")
        .output()
        .unwrap();
    assert!(
        String::from_utf8_lossy(&human.stdout).contains("\x1b["),
        "colors when asked"
    );
}

// -------------------------------------------------------------- bench dump

fn dump_args<'a>(model: &'a Path, prompts: &'a Path, out: &'a Path) -> Vec<&'a str> {
    vec![
        "bench",
        "dump",
        "--model",
        s(model),
        "--prompts",
        s(prompts),
        "--out",
        s(out),
    ]
}

fn temp_files(dir: &Path) -> Vec<String> {
    std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.contains(".tmp-"))
        .collect()
}

#[test]
fn bench_dump_writes_a_complete_file_and_the_same_model_compares_to_zero() {
    let dir = tempfile::tempdir().unwrap();
    let model = write_fake_qwen2_dir(dir.path(), "F32", true);
    let (prompts, _) = write_prompt_files(dir.path());
    let a = dir.path().join("a.safetensors");
    let (code, v) = run_json(&deps(), &dump_args(&model, &prompts, &a));
    assert_eq!(code, 0, "{v}");
    let d = &v["data"];
    assert_eq!(d["n_prompts"], 6);
    assert_eq!(d["vocab_size"], VOCAB);
    assert_eq!(d["dtype_used"], "f32");
    let n = d["n_positions"].as_u64().unwrap() as usize;

    let bytes = std::fs::read(&a).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let logits = st.tensor("logits").unwrap();
    assert_eq!(logits.shape(), [n, VOCAB]);
    let offsets: Vec<i64> = st
        .tensor("prompt_offsets")
        .unwrap()
        .data()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(offsets.len(), 7);
    assert_eq!((offsets[0], offsets[6]), (0, n as i64));
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let meta = meta.metadata().clone().unwrap();
    for key in [
        "model",
        "dtype_used",
        "n_prompts",
        "n_positions",
        "vocab_size",
        "prompts_sha256",
        "tokenizer_sha256",
        "max_tokens",
        "surgeon_version",
    ] {
        assert!(meta.contains_key(key), "{key}");
    }
    assert_eq!(meta["prompts_sha256"], d["prompts_sha256"]);

    let b = dir.path().join("b.safetensors");
    assert_eq!(run_json(&deps(), &dump_args(&model, &prompts, &b)).0, 0);
    assert_eq!(std::fs::read(&b).unwrap(), bytes, "bit-identical dumps");
    let (code, v) = run_json(
        &deps(),
        &["bench", "logits", "--before", s(&a), "--after", s(&b)],
    );
    assert_eq!(code, 0);
    assert_eq!(v["data"]["kl_mean"], 0.0);
    assert_eq!(v["data"]["top1_shift_rate"], 0.0);
    assert_eq!(v["data"]["prompts"], 6);
    assert!(v["warnings"].as_array().unwrap().is_empty());

    let (code, v) = run_json(&deps(), &dump_args(&model, &prompts, &a));
    assert_eq!(code, exit::INVALID_INPUT);
    assert_eq!(v["error"]["kind"], "output_exists");
    let mut again = dump_args(&model, &prompts, &a);
    again.extend(["--overwrite", "--out-dtype", "f16"]);
    assert_eq!(run_json(&deps(), &again).0, 0);
    assert!(
        std::fs::metadata(&a).unwrap().len() < bytes.len() as u64,
        "f16 is smaller"
    );
    assert!(temp_files(dir.path()).is_empty());
}

#[test]
fn bench_dump_refuses_before_computing_and_leaves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let model = write_fake_qwen2_dir(dir.path(), "F32", false);
    let (prompts, _) = write_prompt_files(dir.path());
    let out = dir.path().join("out.safetensors");

    let mut capped = dump_args(&model, &prompts, &out);
    capped.extend(["--max-total-positions", "3"]);
    let (code, v) = run_json(&deps(), &capped);
    assert_eq!(code, exit::INVALID_INPUT);
    assert!(
        v["error"]["message"].as_str().unwrap().contains("GiB"),
        "{v}"
    );

    let mut poor = deps();
    poor.probe = Box::new(FakeProbe { available: 0 });
    let (code, v) = run_json(&poor, &dump_args(&model, &prompts, &out));
    assert_eq!(code, exit::GENERAL);
    assert_eq!(v["error"]["kind"], "insufficient_resources");

    let cancelled = deps();
    cancelled.interrupted.store(true, Ordering::SeqCst);
    let (code, v) = run_json(&cancelled, &dump_args(&model, &prompts, &out));
    assert_eq!(code, exit::GENERAL);
    assert_eq!(v["error"]["kind"], "cancelled");

    std::fs::write(model.join("config.json"), qwen_config("qwen3").to_string()).unwrap();
    let (code, v) = run_json(&deps(), &dump_args(&model, &prompts, &out));
    assert_eq!(code, exit::INVALID_INPUT);
    assert!(v["error"]["message"].as_str().unwrap().contains("qwen3"));

    assert!(!out.exists());
    assert!(temp_files(dir.path()).is_empty());
}

/// The agent recipe: copy the model, patch the copy, dump both, compare.
#[test]
fn bench_dump_before_and_after_an_apply_on_a_copy() {
    let dir = tempfile::tempdir().unwrap();
    let model = write_fake_qwen2_dir(dir.path(), "F32", false);
    let copy = dir.path().join("qwen2-patched");
    std::fs::create_dir_all(&copy).unwrap();
    for entry in std::fs::read_dir(&model).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, copy.join(path.file_name().unwrap())).unwrap();
    }
    std::fs::write(
        dir.path().join("v16.json"),
        serde_json::to_string(&[1.0f32; HIDDEN]).unwrap(),
    )
    .unwrap();
    let patch = dir.path().join("patch.json");
    let ops = json!([op("g", "1", "gate_proj", "v16.json", "rows")]);
    std::fs::write(
        &patch,
        json!({"schema_version": 1, "backup": false, "operations": ops}).to_string(),
    )
    .unwrap();
    let weights = copy.join("model.safetensors");
    let (code, v) = run_json(
        &deps(),
        &[
            "apply",
            "--model",
            s(&weights),
            "--patch",
            s(&patch),
            "--no-backup",
        ],
    );
    assert_eq!(code, 0, "{v}");

    let (prompts, _) = write_prompt_files(dir.path());
    let (a, b) = (
        dir.path().join("a.safetensors"),
        dir.path().join("b.safetensors"),
    );
    assert_eq!(run_json(&deps(), &dump_args(&model, &prompts, &a)).0, 0);
    assert_eq!(run_json(&deps(), &dump_args(&copy, &prompts, &b)).0, 0);
    let (code, v) = run_json(
        &deps(),
        &["bench", "logits", "--before", s(&a), "--after", s(&b)],
    );
    assert_eq!(code, 0);
    assert!(v["data"]["kl_mean"].as_f64().unwrap() > 0.0, "{v}");
    assert_eq!(v["data"]["worst_prompts"].as_array().unwrap().len(), 5);
}

// -------------------------------------------------------------- bench ppl

/// Writes the GGUF a conversion would produce, otherwise replays perplexity outputs.
struct PplRunner {
    calls: Calls,
    outputs: RefCell<VecDeque<CommandOutput>>,
}

impl CommandRunner for PplRunner {
    fn run(
        &self,
        program: &Path,
        args: &[String],
        _cwd: Option<&Path>,
    ) -> weight_surgeon::eval::Result<CommandOutput> {
        self.calls
            .borrow_mut()
            .push((program.to_path_buf(), args.to_vec()));
        if let Some(i) = args.iter().position(|a| a == "--outfile") {
            std::fs::write(&args[i + 1], b"GGUF\x03\x00\x00\x00").unwrap();
            return Ok(CommandOutput {
                status_code: 0,
                stdout: String::new(),
                stderr: String::new(),
            });
        }
        Ok(self.outputs.borrow_mut().pop_front().unwrap())
    }
}

fn ppl_output(ppl: &str) -> CommandOutput {
    CommandOutput {
        status_code: 0,
        stdout: String::new(),
        stderr: format!(
            "perplexity: calculating perplexity over 2 chunks, n_ctx=512\n[1]6.1000,\n[2]{ppl},\nFinal estimate: PPL = {ppl} +/- 0.04000\n"
        ),
    }
}

type Calls = Rc<RefCell<Vec<(PathBuf, Vec<String>)>>>;

fn ppl_deps(mode: ColorMode, outputs: Vec<CommandOutput>) -> (AppDeps, Calls) {
    let mut deps = ui_deps(mode, false);
    let calls = Calls::default();
    deps.runner = Box::new(PplRunner {
        calls: Rc::clone(&calls),
        outputs: RefCell::new(outputs.into()),
    });
    (deps, calls)
}

struct PplFixture {
    dir: tempfile::TempDir,
    after: PathBuf,
    before: PathBuf,
    text: PathBuf,
    llama: PathBuf,
}

fn ppl_fixture() -> PplFixture {
    let dir = tempfile::tempdir().unwrap();
    let llama = dir.path().join("llama.cpp");
    std::fs::create_dir_all(&llama).unwrap();
    let exe = format!("llama-perplexity{}", std::env::consts::EXE_SUFFIX);
    std::fs::write(llama.join(exe), "").unwrap();
    std::fs::write(llama.join("convert_hf_to_gguf.py"), "").unwrap();
    let (after, before, text) = (
        dir.path().join("after.gguf"),
        dir.path().join("before.gguf"),
        dir.path().join("text.txt"),
    );
    for f in [&after, &before] {
        std::fs::write(f, b"GGUF").unwrap();
    }
    std::fs::write(&text, "Le texte de référence pour la perplexité.\n").unwrap();
    PplFixture {
        dir,
        after,
        before,
        text,
        llama,
    }
}

fn ppl_args<'a>(f: &'a PplFixture, extra: &[&'a str]) -> Vec<&'a str> {
    let mut args = vec![
        "bench",
        "ppl",
        "--model",
        s(&f.after),
        "--text",
        s(&f.text),
        "--llama-cpp-dir",
        s(&f.llama),
    ];
    args.extend_from_slice(extra);
    args
}

#[test]
fn bench_ppl_baselines_rounded_delta_and_exact_arguments() {
    let f = ppl_fixture();
    let (deps, calls) = ppl_deps(ColorMode::Plain, vec![ppl_output("7.4000")]);
    let (code, v) = run_json(&deps, &ppl_args(&f, &[]));
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["ppl"], 7.4);
    assert_eq!(v["data"]["ppl_err"], 0.04);
    assert!(v["data"]["delta"].is_null() && v["data"]["baseline"].is_null());
    assert_eq!(v["data"]["verdict"], "none");
    let recorded = calls.borrow();
    assert_eq!(recorded.len(), 1);
    let args = &recorded[0].1;
    assert_eq!(args.len(), 6);
    assert_eq!((args[0].as_str(), args[2].as_str()), ("-m", "-f"));
    assert!(Path::new(&args[1]).is_absolute() && Path::new(&args[3]).is_absolute());
    assert_eq!(&args[4..], ["-c", "512"]);
    drop(recorded);

    let (deps, _) = ppl_deps(ColorMode::Plain, vec![ppl_output("7.4000")]);
    let (code, v) = run_json(&deps, &ppl_args(&f, &["--baseline-ppl", "7.2"]));
    assert_eq!(code, 0);
    assert_eq!(v["data"]["delta"]["delta"], 0.2);
    assert_eq!(
        v["data"]["delta"]["exceeded"], false,
        "7.4 - 7.2 is not above 0.2"
    );
    assert!(v["warnings"].as_array().unwrap().is_empty());

    let (deps, _) = ppl_deps(ColorMode::Plain, vec![ppl_output("7.4001")]);
    let (code, v) = run_json(
        &deps,
        &ppl_args(&f, &["--baseline-ppl", "7.2", "--fail-on-warn"]),
    );
    assert_eq!(code, 9);
    assert_eq!(v["data"]["delta"]["exceeded"], true);

    let (deps, calls) = ppl_deps(
        ColorMode::Plain,
        vec![ppl_output("7.2000"), ppl_output("7.2500")],
    );
    let (code, v) = run_json(
        &deps,
        &ppl_args(
            &f,
            &[
                "--baseline-model",
                s(&f.before),
                "--llama-arg",
                "--batch-size=64",
            ],
        ),
    );
    assert_eq!(code, 0, "{v}");
    let order: Vec<String> = calls.borrow().iter().map(|(_, a)| a[1].clone()).collect();
    assert_eq!(order.len(), 2);
    assert!(
        order[0].ends_with("before.gguf") && order[1].ends_with("after.gguf"),
        "baseline first"
    );
    assert_eq!(v["data"]["baseline"]["source"], "model");
    assert_eq!(v["data"]["delta"]["within_noise"], true);
    assert!(calls.borrow()[1]
        .1
        .ends_with(&["--batch-size".to_owned(), "64".to_owned()]));

    let (code, _) = run_json(
        &deps,
        &ppl_args(
            &f,
            &["--baseline-model", s(&f.before), "--baseline-ppl", "7"],
        ),
    );
    assert_eq!(code, exit::INVALID_INPUT, "conflicting baselines");
    let (code, v) = run_json(&deps, &ppl_args(&f, &["--llama-arg", "--log-file=x.txt"]));
    assert_eq!(code, exit::INVALID_INPUT);
    assert!(v["error"]["message"]
        .as_str()
        .unwrap()
        .contains("--log-file"));
}

#[test]
fn bench_ppl_failures_are_typed_and_the_temporary_gguf_is_removed() {
    let f = ppl_fixture();
    let noisy: String = (1..=30).map(|i| format!("log line {i}\n")).collect();
    let failing = CommandOutput {
        status_code: 1,
        stdout: String::new(),
        stderr: noisy,
    };
    let (deps, _) = ppl_deps(ColorMode::Plain, vec![failing.clone()]);
    let (code, v) = run_json(&deps, &ppl_args(&f, &[]));
    assert_eq!(code, 8);
    assert_eq!(v["error"]["kind"], "process_failed");
    let message = v["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("log line 30") && !message.contains("log line 10\n"),
        "{message}"
    );

    let truncated = CommandOutput {
        status_code: 0,
        stdout: String::new(),
        stderr: "[1]6.1,[2]6.4,".into(),
    };
    let (deps, _) = ppl_deps(ColorMode::Plain, vec![truncated, ppl_output("nan")]);
    for _ in 0..2 {
        let (code, v) = run_json(&deps, &ppl_args(&f, &[]));
        assert_eq!(code, 8);
        assert_eq!(v["error"]["kind"], "ppl_parse_failed");
    }

    // A SafeTensors directory: converted to a temporary GGUF, removed on success and error.
    let model_dir = f.dir.path().join("hf");
    std::fs::create_dir_all(&model_dir).unwrap();
    bf16_model(&model_dir);
    let leftovers = || -> Vec<String> {
        std::fs::read_dir(&model_dir)
            .unwrap()
            .filter_map(|e| e.ok()?.file_name().into_string().ok())
            .filter(|n| n.ends_with(".gguf"))
            .collect()
    };
    let hf = ppl_args(&f, &[]);
    let hf: Vec<&str> = hf
        .iter()
        .map(|a| if *a == s(&f.after) { s(&model_dir) } else { *a })
        .collect();
    let (deps, calls) = ppl_deps(ColorMode::Plain, vec![ppl_output("8.0")]);
    assert_eq!(run_json(&deps, &hf).0, 0);
    assert_eq!(calls.borrow().len(), 2, "convert, then perplexity");
    assert!(calls.borrow()[1].1[1].contains(".weight-surgeon-bench-ppl-"));
    assert!(leftovers().is_empty());
    let (deps, _) = ppl_deps(ColorMode::Plain, vec![failing]);
    assert_eq!(run_json(&deps, &hf).0, 8);
    assert!(leftovers().is_empty(), "removed after an error too");
    let (code, v) = run_json(
        &deps,
        &[
            "bench",
            "ppl",
            "--model",
            s(&model_dir),
            "--text",
            s(&f.text),
        ],
    );
    if std::env::var_os("LLAMA_CPP_DIR").is_none() {
        assert_eq!(code, exit::GENERAL, "{v}");
        assert_eq!(v["error"]["kind"], "binary_not_found");
        assert!(v["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--llama-cpp-dir"));
    }
}

// -------------------------------------------------------------- bench eval

struct Fixed(&'static str);

impl TextGenerator for Fixed {
    fn generate(
        &mut self,
        _prompt: &str,
        _max_new_tokens: u32,
        on_chunk: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> weight_surgeon::eval::Result<GenerationOutcome> {
        let _ = on_chunk(self.0);
        Ok(GenerationOutcome {
            text: self.0.to_owned(),
            stopped_early: false,
            duration_ms: 0,
        })
    }
}

/// "before.gguf" answers 8, any other model answers 5.
fn answer_by_model(
    gguf: &Path,
    _: &weight_surgeon::cli::EvalArgs,
) -> Result<Box<dyn TextGenerator>, EvalError> {
    Ok(Box::new(Fixed(match gguf.ends_with("before.gguf") {
        true => "So it is 8.\n#### 8",
        false => "So it is 5.\n#### 5",
    })))
}

struct EvalBenchFixture {
    _dir: tempfile::TempDir,
    after: PathBuf,
    before: PathBuf,
    dataset: PathBuf,
}

fn eval_fixture_bench() -> EvalBenchFixture {
    let dir = tempfile::tempdir().unwrap();
    let (after, before, dataset) = (
        dir.path().join("after.gguf"),
        dir.path().join("before.gguf"),
        dir.path().join("dataset.json"),
    );
    for f in [&after, &before] {
        std::fs::write(f, b"GGUF").unwrap();
    }
    // Original problems written for these tests.
    std::fs::write(
        &dataset,
        "[{\"id\":\"q1\",\"question\":\"Two plus three?\",\"answer\":\"2 + 3 = 5\\n#### 5\"},\n \
         {\"id\":\"q2\",\"question\":\"Four plus four?\",\"answer\":\"4 + 4 = 8\\n#### 8\"}]",
    )
    .unwrap();
    EvalBenchFixture {
        _dir: dir,
        after,
        before,
        dataset,
    }
}

#[test]
fn bench_eval_scores_pairs_and_gates_on_min_accuracy() {
    use sha2::Digest as _;
    let e = eval_fixture_bench();
    let mut deps = deps();
    deps.generator_factory = Box::new(answer_by_model);
    let base = [
        "bench",
        "eval",
        "--model",
        s(&e.after),
        "--dataset",
        s(&e.dataset),
    ];
    let (code, v) = run_json(&deps, &base);
    assert_eq!(code, 0, "{v}");
    let d = &v["data"];
    assert_eq!((d["n"].as_u64(), d["correct"].as_u64()), (Some(2), Some(1)));
    assert_eq!(d["accuracy"], 0.5);
    let digest = sha2::Sha256::digest(std::fs::read(&e.dataset).unwrap());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(d["dataset_sha256"], hex);
    assert_eq!(d["items"][0]["reason"], "correct");
    assert_eq!(d["items"][1]["extracted"], "5");

    let (code, v) = run_json(
        &deps,
        &[&base[..], &["--baseline-model", s(&e.before)]].concat(),
    );
    assert_eq!(code, 0);
    assert_eq!(v["data"]["paired"]["lost"], json!(["q2"]));
    assert_eq!(v["data"]["paired"]["gained"], json!(["q1"]));
    assert_eq!(v["data"]["paired"]["delta_accuracy"], 0.0);

    let (code, v) = run_json(&deps, &[&base[..], &["--min-accuracy", "0.6"]].concat());
    assert_eq!(code, 0);
    assert_eq!(v["data"]["verdict"], "warn");
    let (code, v) = run_json(
        &deps,
        &[&base[..], &["--min-accuracy", "0.6", "--fail-on-warn"]].concat(),
    );
    assert_eq!(code, 9);
    assert_eq!(v["error"]["kind"], "threshold_exceeded");
    let (code, _) = run_json(
        &deps,
        &[&base[..], &["--prompt-template", "no variable"]].concat(),
    );
    assert_eq!(code, exit::INVALID_INPUT);
    let (code, _) = run_json(&deps, &[&base[..], &["--limit", "0"]].concat());
    assert_eq!(code, exit::INVALID_INPUT);
}

// -------------------------------------------------------------- bench over MCP

/// Blocks in `run_streaming` until the cancel flag rises, then reports a killed child.
struct CancelRunner {
    killed: Rc<RefCell<bool>>,
}

impl CommandRunner for CancelRunner {
    fn run(
        &self,
        _: &Path,
        _: &[String],
        _: Option<&Path>,
    ) -> weight_surgeon::eval::Result<CommandOutput> {
        unreachable!("only streaming runs here")
    }

    fn run_streaming(
        &self,
        _: &Path,
        _: &[String],
        _: Option<&Path>,
        control: StreamControl<'_>,
        _on_line: &mut dyn FnMut(&str) -> ControlFlow<()>,
    ) -> weight_surgeon::eval::Result<CommandOutput> {
        let started = Instant::now();
        while !control.cancel.unwrap().load(Ordering::SeqCst) {
            assert!(
                started.elapsed() < Duration::from_secs(20),
                "never cancelled"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        *self.killed.borrow_mut() = true;
        Ok(CommandOutput {
            status_code: -1,
            stdout: String::new(),
            stderr: String::new(),
        })
    }
}

fn copy_into(src: &Path, dir: &Path) -> PathBuf {
    let target = dir.join(src.file_name().unwrap());
    std::fs::copy(src, &target).unwrap();
    target
}

#[test]
fn mcp_bench_runs_every_mode_and_guards_its_arguments() {
    let f = ppl_fixture();
    let root = f.dir.path();
    let (before, after) = golden_dumps(root);
    let e = eval_fixture_bench();
    copy_into(&e.dataset, root);
    write_fake_qwen2_dir(root, "F32", false);
    let (prompts, _) = write_prompt_files(root);
    let (mut deps, _) = ppl_deps(ColorMode::Plain, vec![ppl_output("7.3000")]);
    deps.generator_factory = Box::new(answer_by_model);
    let lines = [
        init("2025-06-18"),
        call(
            1,
            "surgeon_bench",
            json!({"mode": "logits", "before": "before.safetensors", "after": "after.safetensors"}),
        ),
        call(
            2,
            "surgeon_bench",
            json!({"mode": "ppl", "model": "after.gguf", "text": "text.txt", "llama_cpp_dir": "llama.cpp", "baseline_ppl": 7.2}),
        ),
        call(
            3,
            "surgeon_bench",
            json!({"mode": "eval", "model": "after.gguf", "dataset": "dataset.json", "llama_cpp_dir": "llama.cpp"}),
        ),
        call(
            4,
            "surgeon_bench",
            json!({"mode": "dump", "model": "qwen2-mini", "prompts": s(&prompts), "out": "dump.safetensors"}),
        ),
        call(
            5,
            "surgeon_bench",
            json!({"mode": "logits", "before": "before.safetensors", "text": "text.txt"}),
        ),
        call(
            6,
            "surgeon_bench",
            json!({"mode": "logits", "before": "b", "after": "a", "frobnicate": 1}),
        ),
        call(
            7,
            "surgeon_bench",
            json!({"mode": "logits", "before": "../outside.safetensors", "after": "after.safetensors"}),
        ),
        call(8, "surgeon_bench", json!({"before": "before.safetensors"})),
        call(
            9,
            "surgeon_bench",
            json!({"mode": "ppl", "model": "after.gguf", "text": "text.txt", "baseline_ppl_err": 0.1}),
        ),
    ];
    let responses = session(&deps, &mcp_cfg(root), &lines);
    let (is_error, env) = envelope(&responses, 1);
    assert!(!is_error, "{env}");
    assert_eq!(env["command"], "bench logits");
    assert_eq!(env["data"]["top1_shift_rate"], 0.5);
    let (is_error, env) = envelope(&responses, 2);
    assert!(!is_error, "{env}");
    assert_eq!(env["data"]["delta"]["delta"], 0.1);
    let (is_error, env) = envelope(&responses, 3);
    assert!(!is_error, "{env}");
    assert_eq!(env["data"]["correct"], 1);
    let (is_error, env) = envelope(&responses, 4);
    assert!(!is_error, "{env}");
    assert!(root.join("dump.safetensors").exists());
    let (is_error, env) = envelope(&responses, 5);
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "invalid_arguments");
    let paths: Vec<&str> = env["error"]["issues"]
        .as_array()
        .unwrap()
        .iter()
        .map(|i| i["path"].as_str().unwrap())
        .collect();
    assert_eq!(paths, ["after", "text"]);
    assert_eq!(error_code(&responses, json!(6)), mcp::INVALID_PARAMS);
    assert_eq!(
        envelope(&responses, 7).1["error"]["kind"],
        "path_not_allowed"
    );
    assert_eq!(
        envelope(&responses, 8).1["error"]["kind"],
        "invalid_arguments"
    );
    assert_eq!(
        envelope(&responses, 9).1["error"]["issues"][0]["path"],
        "baseline_ppl_err"
    );

    let read_only = McpConfig::new(&[root.to_path_buf()], true, false).unwrap();
    let responses = session(
        &deps,
        &read_only,
        &[
            init("2025-06-18"),
            call(
                1,
                "surgeon_bench",
                json!({"mode": "dump", "model": "qwen2-mini", "prompts": s(&prompts), "out": "other.safetensors"}),
            ),
            call(
                2,
                "surgeon_bench",
                json!({"mode": "logits", "before": "before.safetensors", "after": "after.safetensors"}),
            ),
        ],
    );
    assert_eq!(envelope(&responses, 1).1["error"]["kind"], "read_only");
    assert!(!root.join("other.safetensors").exists());
    assert!(!envelope(&responses, 2).0, "reading stays allowed");
    let _ = (before, after);
}

#[test]
fn mcp_bench_progress_and_cancellation_kill_the_process() {
    let f = ppl_fixture();
    let root = f.dir.path();
    let (deps, _) = ppl_deps(ColorMode::Plain, vec![ppl_output("7.3000")]);
    let ppl = json!({"mode": "ppl", "model": "after.gguf", "text": "text.txt", "llama_cpp_dir": "llama.cpp"});
    let with_token = rpc(
        json!(1),
        "tools/call",
        json!({"name": "surgeon_bench", "arguments": ppl, "_meta": {"progressToken": "p"}}),
    );
    let responses = session(&deps, &mcp_cfg(root), &[init("2025-06-18"), with_token]);
    let progress: Vec<&Value> = responses
        .iter()
        .filter(|r| r["method"] == "notifications/progress")
        .collect();
    assert_eq!(progress.len(), 2, "{responses:?}");
    assert_eq!(progress[1]["params"]["progressToken"], "p");
    assert_eq!(progress[1]["params"]["progress"], 2);
    assert_eq!(progress[1]["params"]["total"], 2);

    let mut deps = deps;
    let killed = Rc::new(RefCell::new(false));
    deps.runner = Box::new(CancelRunner {
        killed: Rc::clone(&killed),
    });
    let cancel =
        json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 1}})
            .to_string();
    let responses = session(
        &deps,
        &mcp_cfg(root),
        &[init("2025-06-18"), call(1, "surgeon_bench", ppl), cancel],
    );
    let (is_error, env) = envelope(&responses, 1);
    assert!(is_error);
    assert_eq!(env["error"]["kind"], "cancelled");
    assert!(*killed.borrow(), "the llama-perplexity child was stopped");
}

#[test]
fn mcp_binary_bench_speaks_only_json_rpc() {
    let dir = tempfile::tempdir().unwrap();
    golden_dumps(dir.path());
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
        call(
            2,
            "surgeon_bench",
            json!({"mode": "logits", "before": "before.safetensors", "after": "after.safetensors"}),
        ),
    ]
    .join("\n")
        + "\n";
    {
        use std::io::Write as _;
        let mut stdin = child.stdin.take().unwrap();
        stdin.write_all(input.as_bytes()).unwrap();
    }
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(0));
    let out = String::from_utf8(output.stdout).unwrap();
    assert!(!out.contains('\x1b') && !String::from_utf8_lossy(&output.stderr).contains('\x1b'));
    let responses: Vec<Value> = out
        .lines()
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
        .collect();
    assert!(responses.iter().all(|r| r["jsonrpc"] == "2.0"));
    assert!(!envelope(&responses, 2).0);
}

#[test]
fn bench_module_has_no_print_and_no_unsafe() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bench");
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let source = std::fs::read_to_string(&path).unwrap();
        assert!(!source.contains("unsafe"), "{}: unsafe", path.display());
        let code = source.split("#[cfg(test)]").next().unwrap();
        for (n, line) in code.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            for banned in [
                "println!(",
                "print!(",
                "eprintln!(",
                "eprint!(",
                ".unwrap()",
                ".expect(",
                "panic!(",
            ] {
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

/// Two f16 dumps of 4096 positions × 151936 tokens (Qwen2 vocabulary, ~1.2 GB each) in
/// `$SURGEON_BENCH_DUMPS_DIR`, for the throughput and memory measures of `bench logits`:
/// `cargo test --release --test integration make_large_logits_dumps -- --ignored`.
#[test]
#[ignore]
fn make_large_logits_dumps() {
    use std::io::Write as _;
    let dir = PathBuf::from(std::env::var("SURGEON_BENCH_DUMPS_DIR").unwrap());
    std::fs::create_dir_all(&dir).unwrap();
    let (positions, vocab, prompts) = (4096usize, 151_936usize, 16usize);
    let offsets: Vec<i64> = (0..=prompts)
        .map(|p| (p * positions / prompts) as i64)
        .collect();
    for (name, seed, shift) in [
        ("large_before.safetensors", 1u64, 0.0f32),
        ("large_after.safetensors", 1, 0.05),
    ] {
        let logits_bytes = positions * vocab * 2;
        let mut header = json!({
            "logits": {"dtype": "F16", "shape": [positions, vocab], "data_offsets": [0, logits_bytes]},
            "prompt_offsets": {"dtype": "I64", "shape": [prompts + 1], "data_offsets": [logits_bytes, logits_bytes + 8 * (prompts + 1)]},
        })
        .to_string();
        while !(8 + header.len()).is_multiple_of(8) {
            header.push(' ');
        }
        let mut out = std::io::BufWriter::new(std::fs::File::create(dir.join(name)).unwrap());
        out.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        out.write_all(header.as_bytes()).unwrap();
        let mut state = seed;
        let mut row = Vec::with_capacity(vocab * 2);
        for _ in 0..positions {
            row.clear();
            for i in 0..vocab {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let base = ((state >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0;
                let v = base + if i % 7 == 0 { shift * base } else { 0.0 };
                row.extend(half::f16::from_f32(v).to_le_bytes());
            }
            out.write_all(&row).unwrap();
        }
        for o in &offsets {
            out.write_all(&o.to_le_bytes()).unwrap();
        }
        out.flush().unwrap();
    }
}

// ------------------------------------------------------------------ prompt 11: methods, project_out, autotune

fn tensor_names(path: &Path) -> Vec<String> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut names: Vec<String> = st.names().into_iter().map(str::to_owned).collect();
    names.sort();
    names
}

fn leace_of(out: &Path) -> PathBuf {
    let mut name = out.file_name().unwrap().to_os_string();
    name.push(".leace.safetensors");
    out.with_file_name(name)
}

#[test]
fn extract_svd_writes_one_direction_and_method_diagnostics() {
    let f = extract_fixture("F32", false);
    let out = f.dir.path().join("svd.safetensors");
    let (code, v) = run_args(&deps(), &f.args("svd.safetensors", &["--method", "svd"]));
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["method"], "svd");
    let diag = &v["data"]["method_diagnostics"];
    assert_eq!(diag["method"], "svd");
    assert_eq!(diag["n_pairs"], 6);
    assert!(diag["split_half_cosine"].as_f64().unwrap().abs() <= 1.0);
    assert_eq!(tensor_names(&out), ["direction"]);
    let (direction, meta) = read_direction(&out);
    let norm: f32 = direction.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
    assert_eq!(meta["method"], "svd");
    assert_eq!(meta["svd_center"], "false");
    let companion: Value =
        serde_json::from_str(&std::fs::read_to_string(ex::companion_path(&out)).unwrap()).unwrap();
    assert_eq!(companion["method_diagnostics"]["method"], "svd");
    assert!(v["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|w| w.as_str().unwrap().contains("paires")));

    // mean_diff stays as in prompt 8: no method key, no diagnostics in the companion.
    let (code, _) = run_args(&deps(), &f.args("md.safetensors", &[]));
    assert_eq!(code, 0);
    let (_, meta) = read_direction(&f.dir.path().join("md.safetensors"));
    assert!(!meta.contains_key("method"));
    let md_companion =
        std::fs::read_to_string(ex::companion_path(&f.dir.path().join("md.safetensors"))).unwrap();
    assert!(!md_companion.contains("method_diagnostics"));

    let five: Vec<String> = ex::load_prompt_texts(&f.neg).unwrap()[..5].to_vec();
    let short = f.dir.path().join("neg5.json");
    std::fs::write(&short, serde_json::to_string(&five).unwrap()).unwrap();
    let mut args = f.args("u.safetensors", &["--method", "svd"]);
    let at = args.iter().position(|a| a == "--neg-prompts").unwrap();
    args[at + 1] = s(&short).to_owned();
    let (code, v) = run_args(&deps(), &args);
    assert_eq!(code, exit::INVALID_INPUT);
    assert_eq!(v["error"]["kind"], "unpaired_prompts");
}

#[test]
fn extract_leace_writes_the_eraser_atomically_and_deterministically() {
    let f = extract_fixture("F32", false);
    let out = f.dir.path().join("e.safetensors");
    let args = f.args(
        "e.safetensors",
        &["--method", "leace", "--holdout-frac", "0.5"],
    );
    let (code, v) = run_args(&deps(), &args);
    assert_eq!(code, 0, "{v}");
    let diag = &v["data"]["method_diagnostics"];
    assert_eq!(diag["method"], "leace");
    assert!((diag["b_dot_a"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    assert!(
        diag["cross_cov_after"].as_f64().unwrap()
            < 1e-6 * diag["cross_cov_before"].as_f64().unwrap()
    );
    assert!(
        diag["holdout_cross_cov_ratio"].is_null(),
        "6 prompts per set: disabled"
    );
    let warnings = v["warnings"].as_array().unwrap();
    assert!(warnings
        .iter()
        .any(|w| w.as_str().unwrap().contains("--holdout-frac ignoré")));
    assert!(warnings
        .iter()
        .any(|w| w.as_str().unwrap().contains("généralise")));
    assert_eq!(tensor_names(&out), ["direction"]);
    let leace = leace_of(&out);
    assert_eq!(v["data"]["leace_path"], s(&leace));
    assert_eq!(tensor_names(&leace), ["a", "b", "mean", "u_whitened"]);
    let bytes = std::fs::read(&leace).unwrap();
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    let meta = meta.metadata().clone().unwrap();
    for key in [
        "shrinkage",
        "rank",
        "kappa",
        "n_pos",
        "n_neg",
        "layer",
        "site",
        "pooling",
        "bTa",
        "method",
    ] {
        assert!(meta.contains_key(key), "{key}");
    }
    let a = weight_surgeon::engine::load_direction_vector(&leace, Some("a")).unwrap();
    let (direction, _) = read_direction(&out);
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let cos: f32 = a.iter().zip(&direction).map(|(x, y)| x * y).sum::<f32>() / na;
    assert!(cos > 0.9999, "direction = a/‖a‖: {cos}");

    let (main_before, leace_before) = (std::fs::read(&out).unwrap(), bytes);
    let (code, v) = run_args(&deps(), &args);
    assert_eq!(
        code,
        exit::INVALID_INPUT,
        "existing outputs need --overwrite"
    );
    assert_eq!(v["error"]["kind"], "output_exists");
    let mut again = args.clone();
    again.push("--overwrite".into());
    assert_eq!(run_args(&deps(), &again).0, 0);
    assert_eq!(std::fs::read(&out).unwrap(), main_before, "byte for byte");
    assert_eq!(std::fs::read(&leace).unwrap(), leace_before);

    std::fs::remove_file(&out).unwrap();
    std::fs::remove_file(ex::companion_path(&out)).unwrap();
    let (code, _) = run_args(&deps(), &args);
    assert_eq!(
        code,
        exit::INVALID_INPUT,
        "a leftover .leace.safetensors alone also blocks"
    );

    let cancelled = deps();
    cancelled.interrupted.store(true, Ordering::SeqCst);
    let other = f.args("c.safetensors", &["--method", "leace"]);
    let (code, v) = run_args(&cancelled, &other);
    assert_eq!(code, exit::GENERAL);
    assert_eq!(v["error"]["kind"], "cancelled");
    let leftovers: Vec<String> = std::fs::read_dir(f.dir.path())
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .filter(|n| n.starts_with("c.") || n.starts_with(".c.") || n.contains(".tmp-"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

fn project_patch(dir: &Path, leace: &Path, edit: impl Fn(&mut Value)) -> PathBuf {
    let mut op = json!({"id": "erase", "layers": "1", "projections": ["down_proj"],
        "operation": "project_out", "vector_file": s(leace), "vector_tensor": "a",
        "vector_tensor_b": "b", "alpha": 1.0});
    edit(&mut op);
    let path = dir.join("project.json");
    std::fs::write(
        &path,
        json!({"schema_version": 1, "backup": false, "operations": [op]}).to_string(),
    )
    .unwrap();
    path
}

#[test]
fn project_out_applies_the_leace_eraser_exactly_and_is_validated() {
    let f = extract_fixture("F32", false);
    let out = f.dir.path().join("e.safetensors");
    assert_eq!(
        run_args(&deps(), &f.args("e.safetensors", &["--method", "leace"])).0,
        0
    );
    let leace = leace_of(&out);
    let weights = f.model.join("model.safetensors");
    let down = "model.layers.1.mlp.down_proj.weight";
    let before = read_f32(&weights, down);
    let a = weight_surgeon::engine::load_direction_vector(&leace, Some("a")).unwrap();
    let b = weight_surgeon::engine::load_direction_vector(&leace, Some("b")).unwrap();
    let mut reference: Vec<u8> = before.iter().flat_map(|v| v.to_le_bytes()).collect();
    weight_surgeon::engine::project_out_tensor(
        &mut reference,
        weight_surgeon::engine::DType::F32,
        &[HIDDEN, INTER],
        &a,
        &b,
        1.0,
    )
    .unwrap();

    for (edit, needle) in [
        (json!({"projections": ["gate_proj"]}), "down_proj"),
        (json!({"axis": "rows"}), "axis"),
        (json!({"alpha": 1.5}), "[0, 1]"),
        (json!({"vector_tensor_b": null}), "vector_tensor_b"),
    ] {
        let patch = project_patch(f.dir.path(), &leace, |op| {
            for (k, v) in edit.as_object().unwrap() {
                op[k] = v.clone();
            }
        });
        let (code, v) = run_json(
            &deps(),
            &[
                "apply",
                "--model",
                s(&weights),
                "--patch",
                s(&patch),
                "--dry-run",
            ],
        );
        assert_eq!(code, exit::INVALID_INPUT, "{edit}");
        assert!(
            v["error"]["message"].as_str().unwrap().contains(needle),
            "{edit}: {v}"
        );
    }

    let patch = project_patch(f.dir.path(), &leace, |_| {});
    let (code, out_text, err) = run(
        &deps(),
        &[
            "apply",
            "--model",
            s(&weights),
            "--patch",
            s(&patch),
            "--no-backup",
        ],
    );
    assert_eq!(code, 0, "{err}");
    assert!(out_text.contains("[PROJECT]"), "{out_text}");
    let after: Vec<u8> = read_f32(&weights, down)
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    assert_eq!(after, reference, "bytes equal the engine reference");
    let gate = "model.layers.1.mlp.gate_proj.weight";
    assert_eq!(
        read_f32(&weights, gate),
        read_f32(&f.model.join("model.safetensors"), gate)
    );

    let schema: Value =
        serde_json::from_str(&weight_surgeon::mcp::render_schema(SchemaKind::Patch)).unwrap();
    let op = &schema["properties"]["operations"]["items"]["properties"];
    assert!(op["operation"]["enum"]
        .as_array()
        .unwrap()
        .contains(&json!("project_out")));
    assert!(op.get("vector_tensor_b").is_some());
}

/// Readouts s_i = <pool(h_i^R), v> of a model directory, computed from the public API.
fn readouts(model: &Path, v: &[f32], prompts: &[String], layer: usize) -> Vec<f64> {
    let cfg = ex::load_config(model).unwrap();
    let src = SafetensorsDir::open(model).unwrap();
    let required = ex::required_tensors(&cfg, layer);
    let device = candle_core::Device::Cpu;
    let tensors = ex::load_tensors(
        &src,
        &required,
        candle_core::DType::F32,
        &device,
        &mut |_, _| Flow::Continue(()),
    )
    .unwrap();
    let hooked =
        ex::HookedQwen2::from_tensors(tensors, &cfg, layer, candle_core::DType::F32, &device)
            .unwrap();
    let tokenizer = ex::load_tokenizer(model).unwrap();
    prompts
        .iter()
        .map(|p| {
            let ids = ex::tokenize(&tokenizer, p, 256).unwrap().ids;
            let input = candle_core::Tensor::new(ids.as_slice(), &device)
                .unwrap()
                .unsqueeze(0)
                .unwrap();
            let act = hooked.forward_capture(&input, ex::Site::ResidPost).unwrap();
            let pooled = ex::pool(&act, ex::Pooling::Last, "p").unwrap();
            pooled.iter().zip(v).map(|(a, b)| a * f64::from(*b)).sum()
        })
        .collect()
}

struct AutotuneFixture {
    f: ExtractFixture,
    vector: PathBuf,
}

fn autotune_fixture(dtype: &str) -> AutotuneFixture {
    let f = extract_fixture(dtype, false);
    assert_eq!(run_args(&deps(), &f.args("v.safetensors", &[])).0, 0);
    let vector = f.dir.path().join("v.safetensors");
    AutotuneFixture { f, vector }
}

impl AutotuneFixture {
    fn args(&self, extra: &[&str]) -> Vec<String> {
        let mut args: Vec<String> = [
            "autotune",
            "--model",
            s(&self.f.model),
            "--vector",
            s(&self.vector),
            "--eval-dataset",
            s(&self.f.pos),
            "--target-layer-range",
            "0-3",
        ]
        .map(String::from)
        .to_vec();
        args.extend(extra.iter().map(|a| a.to_string()));
        args
    }
}

#[test]
fn autotune_response_matches_patching_each_layer_on_disk() {
    let t = autotune_fixture("BF16");
    let (code, v) = run_args(&deps(), &t.args(&[]));
    assert_eq!(code, 0, "{v}");
    let d = &v["data"];
    assert_eq!(d["readout_layer"], 2);
    assert_eq!(d["layers"], json!([0, 1, 2]));
    assert_eq!(d["excluded_layers"], json!([3]));
    assert_eq!(d["forwards"], 6 * 5 + 6 * 3);
    assert!(v["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .any(|w| w.as_str().unwrap().contains("exclues")));

    let prompts = ex::load_prompt_texts(&t.f.pos).unwrap();
    let (direction, _) = read_direction(&t.vector);
    let s0 = readouts(&t.f.model, &direction, &prompts, 2);
    for (col, layer) in [0usize, 1, 2].into_iter().enumerate() {
        let copy = t.f.dir.path().join(format!("copy-{layer}"));
        std::fs::create_dir_all(&copy).unwrap();
        for entry in std::fs::read_dir(&t.f.model).unwrap() {
            let p = entry.unwrap().path();
            std::fs::copy(&p, copy.join(p.file_name().unwrap())).unwrap();
        }
        let patch = copy.join("p.json");
        let op = json!({"id": "probe", "layers": [layer], "projections": ["down_proj"],
            "operation": "subtract", "vector_file": s(&t.vector), "vector_tensor": "direction",
            "alpha": 0.05, "axis": "cols"});
        std::fs::write(
            &patch,
            json!({"schema_version": 1, "backup": false, "operations": [op]}).to_string(),
        )
        .unwrap();
        let (code, _) = run_json(
            &deps(),
            &[
                "apply",
                "--model",
                s(&copy.join("model.safetensors")),
                "--patch",
                s(&patch),
                "--no-backup",
            ],
        );
        assert_eq!(code, 0);
        let s = readouts(&copy, &direction, &prompts, 2);
        for i in 0..prompts.len() {
            let manual = (s[i] - s0[i]) / 0.05;
            let reported = d["response_matrix"][i][col].as_f64().unwrap();
            assert!(
                (manual - reported).abs() < 1e-6,
                "layer {layer} prompt {i}: {manual} vs {reported}"
            );
        }
    }
}

#[test]
fn autotune_patch_is_valid_deterministic_and_guarded() {
    let t = autotune_fixture("F32");
    let dir = t.f.dir.path();
    let (pa, pb) = (dir.join("a.json"), dir.join("b.json"));
    let (code, v) = run_args(&deps(), &t.args(&["--out", s(&pa), "--l2-reg", "auto"]));
    assert_eq!(code, 0, "{v}");
    assert_eq!(v["data"]["lambda_grid"].as_array().unwrap().len(), 17);
    assert_eq!(v["data"]["out"], s(&pa));
    assert_eq!(
        run_args(&deps(), &t.args(&["--out", s(&pb), "--l2-reg", "auto"])).0,
        0
    );
    assert_eq!(
        std::fs::read(&pa).unwrap(),
        std::fs::read(&pb).unwrap(),
        "byte for byte"
    );
    let patch: Value = serde_json::from_str(&std::fs::read_to_string(&pa).unwrap()).unwrap();
    let op = &patch["operations"][0];
    assert!(op["id"].as_str().unwrap().starts_with("autotune-L"));
    assert_eq!(
        (op["operation"].as_str(), op["axis"].as_str()),
        (Some("subtract"), Some("cols"))
    );
    assert_eq!(op["vector_tensor"], "direction");
    assert_eq!(patch["backup"], true);
    let parsed = parse_patch(&patch.to_string()).unwrap();
    assert!(validate_patch(&parsed, dir, &ValidationOptions::default()).is_ok());
    let weights = t.f.model.join("model.safetensors");
    assert_eq!(
        run_json(
            &deps(),
            &[
                "apply",
                "--model",
                s(&weights),
                "--patch",
                s(&pa),
                "--dry-run"
            ]
        )
        .0,
        0
    );

    let (code, v) = run_args(&deps(), &t.args(&["--out", s(&pa)]));
    assert_eq!(code, exit::INVALID_INPUT);
    assert_eq!(v["error"]["kind"], "output_exists");
    let (code, v) = run_args(&deps(), &t.args(&["--max-forwards", "10"]));
    assert_eq!(code, exit::INVALID_INPUT);
    assert!(v["error"]["message"].as_str().unwrap().contains("forwards"));
    let mut poor = deps();
    poor.probe = Box::new(FakeProbe { available: 0 });
    let (code, v) = run_args(&poor, &t.args(&[]));
    assert_eq!(code, exit::GENERAL);
    assert_eq!(v["error"]["kind"], "insufficient_resources");
    let cancelled = deps();
    cancelled.interrupted.store(true, Ordering::SeqCst);
    let pc = dir.join("c.json");
    let (code, v) = run_args(&cancelled, &t.args(&["--out", s(&pc)]));
    assert_eq!(code, exit::GENERAL);
    assert_eq!(v["error"]["kind"], "cancelled");
    assert!(!pc.exists());
    let (code, _) = run_args(&deps(), &t.args(&["--objective", "target"]));
    assert_eq!(
        code,
        exit::INVALID_INPUT,
        "target needs --target-projection"
    );
}

#[test]
fn autotune_objectives_and_per_prompt_targets() {
    let t = autotune_fixture("F32");
    let (_, meta) = read_direction(&t.vector);
    let mean = |key: &str| meta[key].parse::<f64>().unwrap();
    let target_of = |extra: &[&str], dataset: Option<&Path>| -> f64 {
        let mut args = t.args(extra);
        if let Some(p) = dataset {
            let at = args.iter().position(|a| a == "--eval-dataset").unwrap();
            args[at + 1] = s(p).to_owned();
        }
        let (code, v) = run_args(&deps(), &args);
        assert_eq!(code, 0, "{v}");
        v["data"]["target_mean"].as_f64().unwrap()
    };
    assert!((target_of(&[], None) - mean("neg_mean_proj")).abs() < 1e-9);
    assert!((target_of(&["--objective", "amplify"], None) - mean("pos_mean_proj")).abs() < 1e-9);
    assert_eq!(
        target_of(
            &["--objective", "target", "--target-projection", "0.5"],
            None
        ),
        0.5
    );
    let prompts = ex::load_prompt_texts(&t.f.pos).unwrap();
    let with_targets: Vec<Value> = prompts
        .iter()
        .map(|p| json!({"text": p, "target": 1.25}))
        .collect();
    let ds = t.f.dir.path().join("targets.json");
    std::fs::write(&ds, serde_json::to_string(&with_targets).unwrap()).unwrap();
    assert_eq!(target_of(&[], Some(&ds)), 1.25, "per-prompt targets win");
}

#[test]
fn autotune_human_output_and_strict_json_on_the_binary() {
    let t = autotune_fixture("F32");
    let deps = ui_deps(ColorMode::TrueColor, false);
    let args: Vec<String> = t.args(&[]);
    let (code, out, err) = run_args_raw(&deps, &args);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\x1b[38;2;65;105;225m"));
    let text = strip_ansi(&out);
    let badges: Vec<&str> = text
        .lines()
        .filter_map(|l| l.split_whitespace().next())
        .filter(|w| w.starts_with('['))
        .collect();
    assert_eq!(
        badges,
        ["[AUTOTUNE]", "[RESPONSE]", "[RIDGE]", "[VERDICT]"],
        "{text}"
    );
    assert!(text.contains("patch dans data.patch"));

    let output = binary()
        .arg("--json")
        .args(&args)
        .env("CLICOLOR_FORCE", "1")
        .env("COLORTERM", "truecolor")
        .output()
        .unwrap();
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert_eq!(stdout.lines().count(), 1);
    assert!(!stdout.contains('\x1b') && !String::from_utf8_lossy(&output.stderr).contains('\x1b'));
    let v: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["command"], "autotune");
    assert!(v["data"]["patch"].is_object());
}

#[test]
fn mcp_extract_methods_and_autotune_tool() {
    let t = autotune_fixture("F32");
    let root = t.f.dir.path();
    let pos = ex::load_prompt_texts(&t.f.pos).unwrap();
    let neg = ex::load_prompt_texts(&t.f.neg).unwrap();
    let base = json!({"model": "qwen2-mini", "vector": "v.safetensors", "eval_dataset": "pos.json", "target_layer_range": "0-2"});
    let mut with_out = base.clone();
    with_out["out"] = json!("tuned.json");
    let mut outside = base.clone();
    outside["vector"] = json!("../v.safetensors");
    let progress_call = rpc(
        json!(5),
        "tools/call",
        json!({"name": "surgeon_autotune", "arguments": base, "_meta": {"progressToken": "t"}}),
    );
    let lines = [
        init("2025-06-18"),
        call(
            1,
            "surgeon_extract",
            json!({"model": "qwen2-mini", "pos_prompts": pos, "neg_prompts": neg, "layer": 2, "out": "m.safetensors", "method": "svd"}),
        ),
        call(2, "surgeon_autotune", base.clone()),
        call(3, "surgeon_autotune", with_out),
        call(4, "surgeon_autotune", outside),
        progress_call,
    ];
    let responses = session(&deps(), &mcp_cfg(root), &lines);
    let (is_error, env) = envelope(&responses, 1);
    assert!(!is_error, "{env}");
    assert_eq!(env["data"]["method"], "svd");
    let (is_error, env) = envelope(&responses, 2);
    assert!(!is_error, "{env}");
    assert_eq!(env["command"], "autotune");
    assert!(env["data"]["patch"].is_object() && env["data"]["out"].is_null());
    let (is_error, env) = envelope(&responses, 3);
    assert!(!is_error, "{env}");
    assert!(root.join("tuned.json").exists());
    assert_eq!(
        envelope(&responses, 4).1["error"]["kind"],
        "path_not_allowed"
    );
    let forwards = envelope(&responses, 5).1["data"]["forwards"]
        .as_u64()
        .unwrap();
    let ticks = responses
        .iter()
        .filter(|r| r["method"] == "notifications/progress" && r["params"]["progressToken"] == "t")
        .count();
    assert_eq!(ticks as u64, forwards);

    let mut read_only_out = base.clone();
    read_only_out["out"] = json!("ro.json");
    let read_only = McpConfig::new(&[root.to_path_buf()], true, false).unwrap();
    let cancel =
        json!({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {"requestId": 3}})
            .to_string();
    let responses = session(
        &deps(),
        &read_only,
        &[
            init("2025-06-18"),
            call(1, "surgeon_autotune", read_only_out),
            call(2, "surgeon_autotune", base.clone()),
            cancel,
            call(3, "surgeon_autotune", base),
        ],
    );
    assert_eq!(envelope(&responses, 1).1["error"]["kind"], "read_only");
    assert!(!root.join("ro.json").exists());
    assert!(
        !envelope(&responses, 2).0,
        "computing without out stays allowed"
    );
    assert_eq!(envelope(&responses, 3).1["error"]["kind"], "cancelled");
}

#[test]
fn linalg_module_has_no_print_and_no_unsafe() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/linalg");
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let source = std::fs::read_to_string(&path).unwrap();
        assert!(!source.contains("unsafe"), "{}: unsafe", path.display());
        let code = source.split("#[cfg(test)]").next().unwrap();
        for (n, line) in code.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            for banned in [
                "println!(",
                "print!(",
                "eprintln!(",
                "eprint!(",
                ".unwrap()",
                ".expect(",
                "panic!(",
            ] {
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
