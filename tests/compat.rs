//! Compatibility contract: the CLI, the MCP tools and the `--json` outputs of the reference
//! commands must stay what they were when `tests/fixtures/compat/` was generated (with the
//! steering extensions; every earlier output hash was unchanged by that rewrite except the
//! two schemas). Only human texts may change: clap help, MCP `description` / `title`.
//!
//! Last rewrites, additions only, checked entry by entry against the previous references:
//! - rank-k subspaces of `project_out`: `apply --max-rank`, `extract --rank` and
//!   `extract --labels-file`, the properties `rank` and `allow_non_projector` of a patch
//!   operation, `rank` and `labels_file` of `surgeon_extract`;
//! - the semantic judges of `bench refusal`: fifteen options of `bench refusal`, the
//!   subcommands `bench judge-sample` and `bench judge-eval`, twelve optional properties of
//!   `surgeon_bench`;
//! - the GGUF residual writers: `apply --allow-tied-embedding`, `--refine-passes` and
//!   `--residual-target`, the same three optional properties of `surgeon_apply` (only the
//!   tool schema hash moved).
//!
//! No entry changed or disappeared, and the only output hashes that moved are the two
//! schemas, which list those properties; every extraction file is still byte for byte what
//! it was.
//!
//! - `cli_surface.json`: every subcommand and option of `Cli::command()` (names, short and
//!   long flags, actions, value counts, defaults, possible values, required, global, env).
//!   Additive subcommands are listed in [`ADDITIVE_SUBCOMMANDS`] with the reason.
//! - `mcp_tools_surface.json`: every tool name, `inputSchema` and annotations, without the
//!   `description` / `title` keywords.
//! - `output_hashes.json`: SHA-256 of `exit code + "\n" + normalized stdout` of each reference
//!   command run by the real binary on the mini Qwen2 of `examples/make_fake_qwen2.rs`, and
//!   of the files written by `extract` for each built-in method (byte for byte).
//!
//! Normalization only removes what is not the tool's: hardware facts of `check` (RAM, GPU,
//! verdict, their warnings), `duration_ms`, and `\` path separators (Windows). The extraction
//! hashes come from floating-point code: they were taken on x86_64 Windows, and another CPU
//! family rounds differently (aarch64 does), so they are only compared on x86_64. `cargo test
//! --release --test compat` replays them with the release binary.
//!
//! The references are written by the ignored test `write_compat_references`, and only to
//! record a surface change that was decided on purpose, never to turn a red test green.

#[path = "../examples/make_fake_qwen2.rs"]
#[allow(dead_code)]
mod fake_qwen2;

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::CommandFactory;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use weight_surgeon::cli::Cli;

/// Subcommands added after the references, each one purely additive, with the reason.
/// Empty since the references were rewritten for the steering extensions (`o_proj` and
/// `embed_tokens` patch targets, `bench refusal`, `bench eval --backend`): they now
/// include `plugins`, the former entry.
const ADDITIVE_SUBCOMMANDS: [&str; 0] = [];

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/compat")
        .join(name)
}

fn read_fixture(name: &str) -> Value {
    let text = std::fs::read_to_string(fixture(name))
        .unwrap_or_else(|e| panic!("{name}: {e} (references missing)"));
    serde_json::from_str(&text).unwrap()
}

fn pretty(value: &Value) -> String {
    serde_json::to_string_pretty(value).unwrap() + "\n"
}

// ------------------------------------------------------------------ CLI

fn arg_surface(arg: &clap::Arg) -> Value {
    let strings = |v: Option<Vec<String>>| v.map(Value::from).unwrap_or(Value::Null);
    json!({
        "id": arg.get_id().as_str(),
        "long": arg.get_long(),
        "short": arg.get_short().map(String::from),
        "aliases": strings(arg.get_all_aliases().map(|a| a.iter().map(|s| s.to_string()).collect())),
        "positional": arg.is_positional(),
        "required": arg.is_required_set(),
        "global": arg.is_global_set(),
        "hidden": arg.is_hide_set(),
        "action": format!("{:?}", arg.get_action()),
        "num_args": arg.get_num_args().map(|r| format!("{r:?}")),
        "value_names": strings(arg.get_value_names().map(|v| v.iter().map(|s| s.to_string()).collect())),
        "default_values": arg.get_default_values().iter().map(|v| v.to_string_lossy().into_owned()).collect::<Vec<_>>(),
        "possible_values": arg.get_possible_values().iter().filter(|p| !p.is_hide_set()).map(|p| p.get_name().to_owned()).collect::<Vec<_>>(),
        "env": arg.get_env().map(|e| e.to_string_lossy().into_owned()),
    })
}

fn command_surface(cmd: &clap::Command) -> Value {
    json!({
        "name": cmd.get_name(),
        "aliases": cmd.get_all_aliases().collect::<Vec<_>>(),
        "hidden": cmd.is_hide_set(),
        "subcommand_required": cmd.is_subcommand_required_set(),
        "args": cmd.get_arguments().map(arg_surface).collect::<Vec<_>>(),
        "subcommands": cmd.get_subcommands().map(command_surface).collect::<Vec<_>>(),
    })
}

fn cli_surface() -> Value {
    let mut cli = Cli::command();
    cli.build();
    command_surface(&cli)
}

/// Removes the additive subcommands from a subcommand list; returns how many it removed.
fn drop_additive(command: &mut Value) -> usize {
    let subcommands = command["subcommands"].as_array_mut().unwrap();
    let before = subcommands.len();
    subcommands.retain(|s| !ADDITIVE_SUBCOMMANDS.contains(&s["name"].as_str().unwrap()));
    before - subcommands.len()
}

#[test]
fn cli_surface_is_unchanged() {
    let mut current = cli_surface();
    assert_eq!(
        drop_additive(&mut current),
        ADDITIVE_SUBCOMMANDS.len(),
        "every additive subcommand exists"
    );
    // clap's `help` subcommand mirrors every subcommand (`surgeon help <command>`).
    let help = current["subcommands"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|s| s["name"] == "help")
        .unwrap();
    assert_eq!(drop_additive(help), ADDITIVE_SUBCOMMANDS.len());
    let reference = read_fixture("cli_surface.json");
    assert!(
        current == reference,
        "CLI surface changed (only help texts may change):\n{}",
        pretty(&current)
    );
}

// ------------------------------------------------------------------ MCP

/// Drops the `description` and `title` keywords, never a property that bears that name.
fn strip_texts(value: &Value, in_properties: bool) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter(|(k, _)| in_properties || !matches!(k.as_str(), "description" | "title"))
                .map(|(k, v)| {
                    (
                        k.clone(),
                        strip_texts(v, !in_properties && k == "properties"),
                    )
                })
                .collect::<Map<_, _>>(),
        ),
        Value::Array(items) => Value::Array(items.iter().map(|v| strip_texts(v, false)).collect()),
        other => other.clone(),
    }
}

fn mcp_tools_surface() -> Value {
    let tools = weight_surgeon::mcp::tool_definitions();
    Value::Array(
        tools
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                json!({
                    "name": t["name"],
                    "inputSchema": strip_texts(&t["inputSchema"], false),
                    "annotations": strip_texts(&t["annotations"], false),
                })
            })
            .collect(),
    )
}

#[test]
fn mcp_tools_surface_is_unchanged() {
    let current = mcp_tools_surface();
    assert_eq!(current.as_array().unwrap().len(), 9, "nine MCP tools");
    let reference = read_fixture("mcp_tools_surface.json");
    assert!(
        current == reference,
        "MCP surface changed (only description/title may change):\n{}",
        pretty(&current)
    );
}

#[test]
fn the_stripper_keeps_properties_named_like_keywords() {
    let schema = json!({"title": "t", "description": "d", "properties": {
        "title": {"type": "string", "description": "x"}, "n": {"title": "y", "type": "integer"}}});
    assert_eq!(
        strip_texts(&schema, false),
        json!({"properties": {"title": {"type": "string"}, "n": {"type": "integer"}}})
    );
}

// ------------------------------------------------------------------ outputs

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `\` -> `/` in every string (Windows paths), `duration_ms` and the given hardware keys
/// of `data` replaced by a marker.
fn normalize(value: &mut Value, hardware: &[&str]) {
    fn walk(value: &mut Value) {
        match value {
            Value::String(s) => *s = s.replace('\\', "/"),
            Value::Array(items) => items.iter_mut().for_each(walk),
            Value::Object(map) => {
                if let Some(d) = map.get_mut("duration_ms") {
                    *d = json!("<duration>");
                }
                map.values_mut().for_each(walk);
            }
            _ => {}
        }
    }
    walk(value);
    for key in hardware {
        if let Some(v) = value["data"].get_mut(*key) {
            *v = json!("<hardware>");
        }
    }
}

struct Run {
    dir: tempfile::TempDir,
    hashes: BTreeMap<String, String>,
    /// Normalized outputs, printed when a hash differs.
    texts: BTreeMap<String, String>,
}

impl Run {
    fn surgeon(&mut self, name: &str, args: &[&str], json_out: bool, hardware: &[&str]) {
        let out = Command::new(env!("CARGO_BIN_EXE_surgeon"))
            .args(args)
            .current_dir(self.dir.path())
            .env("NO_COLOR", "1")
            .env_remove("CLICOLOR_FORCE")
            .output()
            .unwrap();
        let code = out.status.code().unwrap();
        let stdout = String::from_utf8(out.stdout).unwrap();
        let text = match json_out {
            true => {
                let mut v: Value = serde_json::from_str(&stdout)
                    .unwrap_or_else(|e| panic!("{name}: {e}: {stdout}"));
                normalize(&mut v, hardware);
                serde_json::to_string(&v).unwrap()
            }
            false => stdout,
        };
        let record = format!("{code}\n{text}");
        self.hashes.insert(name.into(), sha256(record.as_bytes()));
        self.texts.insert(name.into(), record);
    }

    fn file(&mut self, name: &str, path: &str) {
        let bytes = std::fs::read(self.dir.path().join(path)).unwrap();
        self.hashes.insert(name.into(), sha256(&bytes));
        self.texts
            .insert(name.into(), format!("{} bytes", bytes.len()));
    }
}

const VECTOR: &str =
    "[0.25,-0.25,0.5,-0.5,0.125,-0.125,0.0,1.0,0.25,-0.25,0.5,-0.5,0.125,-0.125,0.0,1.0]";
const PATCH: &str = r#"{"schema_version":1,"operations":[{"id":"ref","layers":"1-2","projections":["down_proj"],"operation":"subtract","vector_file":"vec.json","alpha":0.5,"axis":"cols"}]}"#;

fn reference_run() -> Run {
    let dir = tempfile::tempdir().unwrap();
    fake_qwen2::write_fake_qwen2(dir.path()).unwrap();
    fake_qwen2::write_prompt_files(dir.path()).unwrap();
    std::fs::write(dir.path().join("vec.json"), VECTOR).unwrap();
    std::fs::write(dir.path().join("patch.json"), PATCH).unwrap();
    let mut run = Run {
        dir,
        hashes: BTreeMap::new(),
        texts: BTreeMap::new(),
    };
    let check_hw = [
        "available_ram_bytes",
        "best_gpu",
        "required_vram_bytes",
        "verdict",
        "warnings",
    ];
    run.surgeon(
        "check",
        &["--json", "check", "--model", "qwen2-mini"],
        true,
        &check_hw,
    );
    run.surgeon(
        "inspect",
        &["--json", "inspect", "--model", "qwen2-mini", "--show-stats"],
        true,
        &[],
    );
    run.file("model_before_dry_run", "qwen2-mini/model.safetensors");
    run.surgeon(
        "apply_dry_run",
        &[
            "--json",
            "apply",
            "--model",
            "qwen2-mini",
            "--patch",
            "patch.json",
            "--dry-run",
        ],
        true,
        &[],
    );
    run.file("model_after_dry_run", "qwen2-mini/model.safetensors");
    run.surgeon("schema", &["schema"], false, &[]);
    let tools = Command::new(env!("CARGO_BIN_EXE_surgeon"))
        .args(["schema", "--kind", "tools"])
        .output()
        .unwrap();
    let tools: Value = serde_json::from_slice(&tools.stdout).unwrap();
    let stripped = serde_json::to_string(&strip_texts(&tools, false)).unwrap();
    run.hashes.insert(
        "schema_tools_without_texts".into(),
        sha256(stripped.as_bytes()),
    );
    run.texts
        .insert("schema_tools_without_texts".into(), stripped);
    for method in ["mean_diff", "svd", "leace"] {
        let out = format!("{method}.safetensors");
        run.surgeon(
            &format!("extract_{method}"),
            &[
                "--json",
                "extract",
                "--model",
                "qwen2-mini",
                "--pos-prompts",
                "pos.json",
                "--neg-prompts",
                "neg.json",
                "--layer",
                "2",
                "--out",
                &out,
                "--method",
                method,
            ],
            true,
            &[],
        );
        run.file(&format!("extract_{method}/direction"), &out);
        run.file(
            &format!("extract_{method}/stats_json"),
            &format!("{out}.stats.json"),
        );
    }
    run.file(
        "extract_leace/leace_safetensors",
        "leace.safetensors.leace.safetensors",
    );
    run
}

#[test]
fn reference_outputs_are_byte_identical() {
    let run = reference_run();
    assert_eq!(
        run.hashes["model_before_dry_run"], run.hashes["model_after_dry_run"],
        "apply --dry-run never writes"
    );
    let reference: BTreeMap<String, String> =
        serde_json::from_value(read_fixture("output_hashes.json")["sha256"].clone()).unwrap();
    let mut changed = Vec::new();
    for (name, hash) in &reference {
        // Taken on x86_64: the matmuls of the Qwen2 forward round differently elsewhere.
        if name.starts_with("extract_") && !cfg!(target_arch = "x86_64") {
            continue;
        }
        match run.hashes.get(name) {
            Some(h) if h == hash => {}
            Some(_) => changed.push(format!("{name}: {}", run.texts[name])),
            None => changed.push(format!("{name}: not produced")),
        }
    }
    assert_eq!(
        run.hashes.keys().collect::<Vec<_>>(),
        reference.keys().collect::<Vec<_>>()
    );
    assert!(
        changed.is_empty(),
        "outputs changed:\n{}",
        changed.join("\n")
    );
}

/// Writes the three references. Run on purpose only:
/// `cargo test --test compat write_compat_references -- --ignored`.
#[test]
#[ignore]
fn write_compat_references() {
    std::fs::create_dir_all(fixture("")).unwrap();
    std::fs::write(fixture("cli_surface.json"), pretty(&cli_surface())).unwrap();
    std::fs::write(
        fixture("mcp_tools_surface.json"),
        pretty(&mcp_tools_surface()),
    )
    .unwrap();
    let run = reference_run();
    let hashes = json!({
        "note": "sha256 of exit code + newline + normalized stdout (commands), or of the file bytes (extract_*/ and model_*); see tests/compat.rs",
        "binary": "surgeon",
        "sha256": run.hashes,
    });
    std::fs::write(fixture("output_hashes.json"), pretty(&hashes)).unwrap();
}
