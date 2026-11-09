//! Extension API v0: the registry through the public API, what the engine refuses from a
//! method, `--method` / `method` validated by the registry with the former values and
//! messages (CLI and MCP), `surgeon plugins list --json`, and the third-party example end
//! to end.

#[path = "../examples/plugin_extraction_example.rs"]
#[allow(dead_code)]
mod example;

use std::collections::BTreeMap;
use std::io::Write as _;
use std::ops::ControlFlow;
use std::path::Path;
use std::process::{Command, Stdio};
use std::sync::atomic::AtomicBool;

use example::fake_qwen2;
use serde_json::{json, Value};
use weight_surgeon::extract::{self as ex, ExtractError, ExtractEvent, ExtractObserver};
use weight_surgeon::hardware::SystemProbe;
use weight_surgeon::mcp::{ExtractToolArgs, MethodId};
use weight_surgeon::plugin::{
    ActivationSet, ExtractContext, ExtractionMethod, ExtractionOutput, MethodInfo, Origin,
    PluginError, Registry, Sidecar,
};

fn surgeon(dir: &Path, args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_surgeon"))
        .args(args)
        .current_dir(dir)
        .env("NO_COLOR", "1")
        .output()
        .unwrap();
    (
        out.status.code().unwrap(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

// ------------------------------------------------------------------ registry

#[test]
fn registry_lists_builtins_then_registered_and_refuses_a_duplicate() {
    let mut registry = Registry::with_builtin();
    let listed: Vec<(&str, Origin)> = registry.list().map(|(m, o)| (m.id(), o)).collect();
    assert_eq!(
        listed,
        [
            ("mean_diff", Origin::Builtin),
            ("svd", Origin::Builtin),
            ("leace", Origin::Builtin)
        ]
    );
    registry.register(Box::new(example::MedianDiff)).unwrap();
    let again = registry.register(Box::new(example::MedianDiff));
    assert!(matches!(again, Err(PluginError::DuplicateId(id)) if id == "median_diff"));
    assert_eq!(registry.ids(), ["mean_diff", "svd", "leace", "median_diff"]);
    assert_eq!(registry.get("median_diff").unwrap().describe().params, []);
}

// ------------------------------------------------------------------ engine guards

/// A method that returns one chosen mistake.
#[derive(Clone, Copy)]
enum Faulty {
    NotUnit,
    WrongSize,
    UndeclaredSidecar,
    EngineKey,
    Cancelled,
}

impl ExtractionMethod for Faulty {
    fn id(&self) -> &'static str {
        "faulty"
    }
    fn describe(&self) -> MethodInfo {
        MethodInfo {
            title: "faulty",
            summary: "faulty",
            params: vec![],
            requires_paired: false,
            sidecars: vec![],
        }
    }
    fn extract(
        &self,
        ctx: &ExtractContext,
        _: &ActivationSet,
        _: &AtomicBool,
    ) -> Result<ExtractionOutput, PluginError> {
        let mut unit = vec![0.0; ctx.hidden_size];
        unit[0] = 1.0;
        let mut out = ExtractionOutput::new(unit);
        match self {
            Faulty::NotUnit => out.direction[1] = 1.0,
            Faulty::WrongSize => out.direction.push(0.0),
            Faulty::UndeclaredSidecar => out.sidecars.push(Sidecar {
                suffix: "extra",
                tensors: vec![("x".into(), vec![1.0])],
                metadata: BTreeMap::new(),
            }),
            Faulty::EngineKey => {
                out.metadata.insert("layer".into(), "7".into());
            }
            Faulty::Cancelled => return Err(PluginError::Cancelled),
        }
        Ok(out)
    }
}

struct Quiet;

impl ExtractObserver for Quiet {
    fn on_event(&mut self, _: ExtractEvent) -> ControlFlow<()> {
        ControlFlow::Continue(())
    }
}

#[test]
fn the_engine_refuses_bad_method_outputs_and_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let model = fake_qwen2::write_fake_qwen2(dir.path()).unwrap();
    let (pos, neg) = fake_qwen2::write_prompt_files(dir.path()).unwrap();
    let request = ex::ExtractRequest {
        model_dir: model,
        pos_prompts: ex::load_prompt_texts(&pos).unwrap(),
        neg_prompts: ex::load_prompt_texts(&neg).unwrap(),
        layer: 1,
        out: dir.path().join("bad.safetensors"),
        pooling: ex::Pooling::Last,
        site: ex::Site::ResidPost,
        chat_template: ex::ChatTemplate::NoTemplate,
        max_tokens: 64,
        dtype: ex::DTypeChoice::Auto,
        overwrite: false,
    };
    let probe = SystemProbe::detect();
    for faulty in [
        Faulty::NotUnit,
        Faulty::WrongSize,
        Faulty::UndeclaredSidecar,
        Faulty::EngineKey,
        Faulty::Cancelled,
    ] {
        let never = AtomicBool::new(false);
        let err =
            ex::run_extraction_plugin(&request, &faulty, &Value::Null, &probe, &mut Quiet, &never)
                .unwrap_err();
        match faulty {
            Faulty::Cancelled => assert!(matches!(err, ExtractError::Cancelled)),
            _ => assert!(
                matches!(&err, ExtractError::Compute(m) if m.starts_with("sortie de la méthode invalide")),
                "{err}"
            ),
        }
        let mut names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        assert_eq!(
            names,
            ["neg.json", "pos.json", "qwen2-mini"],
            "nothing written"
        );
    }
}

// ------------------------------------------------------------------ --method and method

#[test]
fn an_unknown_cli_method_is_code_2_and_lists_the_registry() {
    let dir = tempfile::tempdir().unwrap();
    let args = [
        "extract",
        "--model",
        "m",
        "--pos-prompts",
        "p",
        "--neg-prompts",
        "n",
        "--out",
        "o.safetensors",
        "--method",
        "pca",
    ];
    let (code, out, err) = surgeon(dir.path(), &args);
    assert_eq!(code, 2);
    assert!(out.is_empty());
    assert!(
        err.contains("invalid value 'pca' for '--method <METHOD>'")
            && err.contains("[possible values: mean_diff, svd, leace]"),
        "{err}"
    );
    let mut json_args = vec!["--json"];
    json_args.extend(args);
    let (code, out, _) = surgeon(dir.path(), &json_args);
    let envelope: Value = serde_json::from_str(&out).unwrap();
    assert_eq!((code, &envelope["exit_code"]), (2, &json!(2)));
    let message = envelope["error"]["message"].as_str().unwrap();
    assert!(message.contains("[possible values: mean_diff, svd, leace]"));
    assert!(!dir.path().join("o.safetensors").exists());
}

fn extract_args(method: Value) -> Result<ExtractToolArgs, String> {
    let args = json!({"model": "m", "out": "o.safetensors", "method": method});
    serde_json::from_value(args).map_err(|e| e.to_string())
}

#[test]
fn the_mcp_method_field_accepts_and_refuses_like_the_former_enum() {
    for id in ["mean_diff", "svd", "leace"] {
        assert_eq!(
            extract_args(json!(id)).unwrap().method,
            Some(MethodId(id.into()))
        );
    }
    assert_eq!(
        extract_args(json!({"svd": null})).unwrap().method,
        Some(MethodId("svd".into()))
    );
    assert_eq!(extract_args(Value::Null).unwrap().method, None);
    // The serde_derive messages of `enum Method { MeanDiff, Svd, Leace }`.
    assert_eq!(
        extract_args(json!("pca")).unwrap_err(),
        "unknown variant `pca`, expected one of `mean_diff`, `svd`, `leace`"
    );
    assert_eq!(
        extract_args(json!(3)).unwrap_err(),
        "invalid type: integer `3`, expected string or map"
    );
}

#[test]
fn an_unknown_mcp_method_is_a_tool_error_with_code_2() {
    let dir = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_surgeon"))
        .args(["mcp", "--allow-dir", "."])
        .current_dir(dir.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let call = json!({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
        "name": "surgeon_extract", "arguments": {"model": "m", "out": "o.safetensors",
        "pos_prompts": ["a"], "neg_prompts": ["b"], "method": "pca"}}});
    let lines = [
        json!({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "t", "version": "0"}}}),
        json!({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        call,
    ];
    let mut stdin = child.stdin.take().unwrap();
    for line in lines {
        writeln!(stdin, "{line}").unwrap();
    }
    drop(stdin);
    let out = child.wait_with_output().unwrap();
    let responses: Vec<Value> = String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let result = &responses.iter().find(|r| r["id"] == 2).unwrap()["result"];
    assert_eq!(result["isError"], true);
    let envelope = &result["structuredContent"];
    assert_eq!(envelope["exit_code"], 2);
    assert_eq!(envelope["error"]["kind"], "invalid_arguments");
    assert_eq!(
        envelope["error"]["message"],
        "arguments invalides : unknown variant `pca`, expected one of `mean_diff`, `svd`, `leace`"
    );
}

// ------------------------------------------------------------------ plugins list

#[test]
fn plugins_list_json_is_stable() {
    let dir = tempfile::tempdir().unwrap();
    let (code, out, err) = surgeon(dir.path(), &["--json", "plugins", "list"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(surgeon(dir.path(), &["--json", "plugins", "list"]).1, out);
    let expected = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/plugins_list.json"),
    )
    .unwrap();
    let (got, want): (Value, Value) = (
        serde_json::from_str(&out).unwrap(),
        serde_json::from_str(&expected).unwrap(),
    );
    assert_eq!(got, want, "{out}");
    let (code, human, _) = surgeon(dir.path(), &["plugins", "list"]);
    assert_eq!(code, 0);
    assert!(human.contains("extraction_method") && human.contains("leace"));
}

// ------------------------------------------------------------------ the example

#[test]
fn the_third_party_example_runs_end_to_end_and_deterministically() {
    let dir = tempfile::tempdir().unwrap();
    let report = example::run(dir.path()).unwrap();
    assert_eq!(report.method, "median_diff");
    assert_eq!((report.n_pos, report.n_neg, report.layer), (6, 6, 2));
    let out = dir.path().join("median.safetensors");
    let bytes = std::fs::read(&out).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let direction: Vec<f32> = st
        .tensor("direction")
        .unwrap()
        .data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(direction.len(), fake_qwen2::HIDDEN);
    let norm: f32 = direction.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-5, "{norm}");
    let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
    assert_eq!(meta.metadata().as_ref().unwrap()["method"], "median_diff");
    let companion: Value =
        serde_json::from_slice(&std::fs::read(ex::companion_path(&out)).unwrap()).unwrap();
    let diagnostics = &companion["method_diagnostics"];
    assert_eq!(diagnostics["method"], "median_diff");
    assert!(diagnostics["raw_median_diff_norm"].as_f64().unwrap() > 0.0);

    example::run(dir.path()).unwrap();
    assert_eq!(std::fs::read(&out).unwrap(), bytes, "deterministic");
}

// ------------------------------------------------------------------ guard

/// Like `extract`, `bench` and `linalg`: no print, no `unsafe`, no panic path outside tests.
#[test]
fn plugin_module_has_no_print_no_unsafe_and_no_panic_path() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/plugin");
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
