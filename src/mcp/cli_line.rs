//! The `surgeon …` command that does what an MCP tool call did, so that any step of an agent
//! can be replayed or refined in a terminal. The tests check every flag it can write against
//! the CLI's own definition (`Cli::command()`), for every argument of every tool.

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::tools::{bench_backend, bench_command, BENCH_FIELDS, EVAL_BACKEND, REFUSAL_BACKEND};
use crate::bench::score::Backend;

/// A tool call written as a terminal command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliLine {
    pub command: String,
    /// hipfire was asked for a working copy, the bench runs on native ([`bench_backend`]).
    pub hipfire_set_aside: bool,
}

/// Arguments whose CLI option has another name: (command, argument, option).
const RENAMED: [(&str, &str, &str); 10] = [
    ("bench refusal-probe", "probe_method", "method"),
    ("bench refusal-probe", "direction_path", "vector"),
    ("inspect", "projections", "proj"),
    ("extract", "pos_prompts_file", "pos-prompts"),
    ("extract", "neg_prompts_file", "neg-prompts"),
    ("eval", "prompts_file", "prompts"),
    ("bench eval", "n_gpu_layers", "gpu-layers"),
    ("bench eval", "ctx", "ctx-size"),
    ("bench refusal", "n_gpu_layers", "gpu-layers"),
    ("bench refusal", "ctx", "ctx-size"),
];

/// Arguments given inline over MCP that the CLI reads from a file: (command, argument, option).
const INLINE: [(&str, &str, &str); 5] = [
    ("apply", "patch", "patch"),
    ("extract", "pos_prompts", "pos-prompts"),
    ("extract", "neg_prompts", "neg-prompts"),
    ("autotune", "eval_prompts", "eval-dataset"),
    ("eval", "prompts", "prompts"),
];

/// Limits of the server alone, with no CLI option.
const MCP_ONLY: [(&str, &str); 3] = [
    ("apply", "base_dir"),
    ("inspect", "max_tensors"),
    ("graph", "max_seconds"),
];

fn command(tool: &str, args: &Value) -> Option<&'static str> {
    Some(match tool {
        "surgeon_check" => "check",
        "surgeon_inspect" => "inspect",
        "surgeon_apply" => "apply",
        "surgeon_eval" => "eval",
        "surgeon_extract" => "extract",
        "surgeon_autotune" => "autotune",
        "surgeon_graph" => "graph",
        "surgeon_restore" => "restore",
        "surgeon_compose" => "compose",
        "surgeon_doctor" => "doctor",
        "surgeon_bench" => match bench_command(args) {
            "bench" => return None,
            sub => sub,
        },
        _ => return None,
    })
}

/// Quoted when a shell would split or read it: spaces, quotes, `$`, `&`, `;`...
fn shell(value: &str) -> String {
    let plain = !value.is_empty()
        && value
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "-_.,:/\\=+@%".contains(c));
    match plain {
        true => value.to_owned(),
        false => format!("\"{}\"", value.replace('"', "\\\"")),
    }
}

fn scalar(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

/// The command of a call to `tool` (an engine tool, `mcp__<server>__` prefix or not) with
/// `args`; `configured` is the backend of the user configuration, `base` the folder relative
/// paths start from (the server's first allowed folder). `None` for a tool without a CLI
/// command (files, backends, pipeline, interface tools).
pub fn cli_line(
    tool: &str,
    args: &Value,
    configured: Option<Backend>,
    base: Option<&Path>,
) -> Option<CliLine> {
    let tool = tool.rsplit("__").next().unwrap_or(tool);
    let command = command(tool, args)?;
    let fields = args.as_object()?;
    let bench_fields = command
        .strip_prefix("bench ")
        .and_then(|sub| BENCH_FIELDS.iter().find(|(mode, ..)| *mode == sub))
        .map(|(_, required, optional)| (*required, *optional));
    let mut line = format!("surgeon {command}");
    let mut files = Vec::new();
    let mut push = |option: &str, value: Option<&str>| {
        line.push_str(&format!(" --{option}"));
        if let Some(v) = value {
            line.push(' ');
            line.push_str(&shell(v));
        }
    };
    let default = match command {
        "bench eval" => Some(EVAL_BACKEND),
        "bench refusal" => Some(REFUSAL_BACKEND),
        _ => None,
    };
    let mut names: Vec<&String> = fields.keys().collect();
    names.sort_by_key(|n| n.as_str() != "model");
    for name in names {
        let value = &fields[name];
        let skipped = value.is_null()
            || value == &Value::Bool(false)
            || MCP_ONLY.contains(&(command, name.as_str()))
            // The bench mode is the subcommand; an eval or refusal backend is written after
            // the loop.
            || (bench_fields.is_some()
                && (name == "mode" || (name == "backend" && default.is_some())))
            || bench_fields.is_some_and(|(required, optional)| {
                !required.contains(&name.as_str()) && !optional.contains(&name.as_str())
            });
        if skipped {
            continue;
        }
        if let Some((_, _, option)) = INLINE.iter().find(|(c, a, _)| *c == command && a == name) {
            let file = format!("{name}.json");
            push(option, Some(&file));
            files.push(file);
            continue;
        }
        let option = RENAMED
            .iter()
            .find(|(c, a, _)| *c == command && a == name)
            .map_or_else(|| name.replace('_', "-"), |(_, _, o)| (*o).to_owned());
        match value {
            Value::Bool(_) => push(&option, None),
            // A repeatable option, the others take a comma-separated list.
            Value::Array(items) if matches!(option.as_str(), "proj" | "ablate" | "keep") => {
                for item in items {
                    push(&option, Some(&scalar(item)));
                }
            }
            Value::Array(items) => {
                let list: Vec<String> = items.iter().map(scalar).collect();
                push(&option, Some(&list.join(",")));
            }
            other => push(&option, Some(&scalar(other))),
        }
    }
    let mut hipfire_set_aside = false;
    // Written even when implicit: the terminal's own configuration must not change the run.
    if let Some(default) = default {
        let requested = fields
            .get("backend")
            .and_then(|b| serde_json::from_value::<Backend>(b.clone()).ok());
        let models: Vec<PathBuf> = ["model", "baseline_model"]
            .iter()
            .filter_map(|k| fields.get(*k).and_then(Value::as_str))
            .map(|m| base.map_or_else(|| PathBuf::from(m), |b| b.join(m)))
            .collect();
        let models: Vec<&Path> = models.iter().map(PathBuf::as_path).collect();
        // A refused GGUF copy keeps hipfire: the command fails just as the call did.
        let resolved = requested.or(configured);
        let (backend, set_aside) = bench_backend(resolved, default, &models)
            .unwrap_or((resolved.unwrap_or(default), None));
        hipfire_set_aside = set_aside.is_some();
        push("backend", Some(backend.label()));
    }
    if !files.is_empty() {
        line.push_str(&format!(
            "  # {} : champs de même nom de l'appel, à écrire en fichiers",
            files.join(", ")
        ));
    }
    Some(CliLine {
        command: line,
        hipfire_set_aside,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use clap::CommandFactory;
    use serde_json::json;

    use super::*;

    /// The long options of `surgeon <command>`.
    fn options(command: &str) -> BTreeSet<String> {
        let mut cmd = crate::cli::Cli::command();
        cmd.build();
        let mut sub = &cmd;
        for word in command.split(' ') {
            sub = sub
                .find_subcommand(word)
                .unwrap_or_else(|| panic!("no subcommand {command}"));
        }
        sub.get_arguments()
            .filter_map(|a| a.get_long().map(str::to_owned))
            .collect()
    }

    fn written(line: &str) -> BTreeSet<String> {
        line.split('#')
            .next()
            .unwrap_or_default()
            .split_whitespace()
            .filter_map(|w| w.strip_prefix("--").map(str::to_owned))
            .collect()
    }

    /// Every argument of every tool with a command becomes an option the CLI has.
    #[test]
    fn every_argument_maps_to_a_real_cli_option() {
        let tools = crate::mcp::tool_definitions();
        let mut checked = 0;
        for tool in tools.as_array().unwrap() {
            let name = tool["name"].as_str().unwrap();
            let props: Vec<&String> = tool["inputSchema"]["properties"]
                .as_object()
                .into_iter()
                .flat_map(|p| p.keys())
                .collect();
            let calls: Vec<Value> = match name {
                "surgeon_bench" => BENCH_FIELDS
                    .iter()
                    .map(|(mode, required, optional)| {
                        let mut args = json!({"mode": mode.replace('-', "_")});
                        for f in required.iter().chain(optional.iter()) {
                            args[*f] = json!(true);
                        }
                        args
                    })
                    .collect(),
                _ => vec![Value::Object(
                    props.iter().map(|p| ((*p).clone(), json!(true))).collect(),
                )],
            };
            for args in calls {
                let Some(line) = cli_line(name, &args, None, None) else {
                    continue;
                };
                let command = command(name, &args).unwrap();
                let known = options(command);
                let got = written(&line.command);
                for option in &got {
                    assert!(
                        known.contains(option),
                        "{name}: --{option} not in `surgeon {command}`"
                    );
                }
                let dropped = args.as_object().unwrap().keys().filter(|a| {
                    !(MCP_ONLY.contains(&(command, a.as_str())) || a.as_str() == "mode")
                        && !got.iter().any(|o| {
                            *o == a.replace('_', "-")
                                || RENAMED.contains(&(command, a.as_str(), o.as_str()))
                                || INLINE.contains(&(command, a.as_str(), o.as_str()))
                        })
                });
                assert_eq!(
                    dropped.count(),
                    0,
                    "{name}: an argument has no option in {line:?}"
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked,
            10 + BENCH_FIELDS.len(),
            "every tool with a command is covered"
        );
    }

    #[test]
    fn an_inline_patch_becomes_a_patch_file_and_the_flags_keep_their_values() {
        let args = json!({
            "model": r"C:\p\copies\copy-1",
            "dry_run": true,
            "gguf": false,
            "patch": {"operations": [{"operation": "project_out"}]},
        });
        let line = cli_line("mcp__surgeon__surgeon_apply", &args, None, None).unwrap();
        assert_eq!(
            line.command,
            r"surgeon apply --model C:\p\copies\copy-1 --dry-run --patch patch.json  # patch.json : champs de même nom de l'appel, à écrire en fichiers"
        );
        let args = json!({"model": "C:/My Models/m", "projections": ["o_proj", "down_proj"],
            "layers": "all", "max_tensors": 10});
        assert_eq!(
            cli_line("surgeon_inspect", &args, None, None).unwrap().command,
            "surgeon inspect --model \"C:/My Models/m\" --layers all --proj o_proj --proj down_proj"
        );
        assert!(cli_line("surgeon_files", &json!({}), None, None).is_none());
        assert!(cli_line("record_finding", &json!({}), None, None).is_none());
    }

    /// A surgeon-gui project: its manifest naming `source`, a SafeTensors working copy and a
    /// GGUF one.
    fn project() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        std::fs::create_dir_all(&source).unwrap();
        let manifest = json!({"source": {"path": source}});
        std::fs::write(
            dir.path().join("project.surgeon.json"),
            manifest.to_string(),
        )
        .unwrap();
        std::fs::create_dir_all(dir.path().join("copies/copy-1")).unwrap();
        std::fs::write(dir.path().join("copies/m.gguf"), b"GGUF").unwrap();
        dir
    }

    #[test]
    fn a_bench_names_its_backend_and_never_measures_a_working_copy_on_hipfire() {
        let p = project();
        let base = Some(p.path());
        let eval = json!({"mode": "eval", "model": "copies/copy-1", "dataset": "builtin:logic"});
        let line = cli_line("surgeon_bench", &eval, Some(Backend::Hipfire), base).unwrap();
        assert_eq!(
            line.command,
            "surgeon bench eval --model copies/copy-1 --dataset builtin:logic --backend native"
        );
        assert!(line.hipfire_set_aside);
        // Asked explicitly: set aside all the same.
        let mut refusal = json!({"mode": "refusal", "model": "copies/copy-1",
            "prompts": "builtin:refusals", "backend": "hipfire"});
        let line = cli_line("surgeon_bench", &refusal, None, base).unwrap();
        assert!(line.command.ends_with("--backend native") && line.hipfire_set_aside);
        refusal["backend"] = json!("mistralrs");
        let line = cli_line("surgeon_bench", &refusal, Some(Backend::Hipfire), base).unwrap();
        assert!(line.command.ends_with("--backend mistralrs") && !line.hipfire_set_aside);
        // The project's source is what hipfire's cache holds.
        let source = json!({"mode": "eval", "model": "source", "dataset": "builtin:math"});
        let line = cli_line("surgeon_bench", &source, Some(Backend::Hipfire), base).unwrap();
        assert!(line.command.ends_with("--backend hipfire") && !line.hipfire_set_aside);
        // A GGUF copy is refused by the call: the command keeps hipfire and fails the same.
        let gguf = json!({"mode": "eval", "model": "copies/m.gguf", "dataset": "builtin:math"});
        let line = cli_line("surgeon_bench", &gguf, Some(Backend::Hipfire), base).unwrap();
        assert!(line.command.ends_with("--backend hipfire") && !line.hipfire_set_aside);
        assert!(cli_line("surgeon_bench", &refusal, None, base)
            .unwrap()
            .command
            .contains("--prompts builtin:refusals"));
        assert_eq!(
            cli_line(
                "surgeon_bench",
                &json!({"mode": "eval", "model": "m"}),
                None,
                None
            )
            .unwrap()
            .command,
            "surgeon bench eval --model m --backend llama"
        );
        let ppl = json!({"mode": "ppl", "model": "copies/copy-1", "text": "builtin:wikitext",
            "backend": "native", "stride": 256});
        let line = cli_line("surgeon_bench", &ppl, Some(Backend::Hipfire), base).unwrap();
        assert_eq!(
            line.command,
            "surgeon bench ppl --model copies/copy-1 --backend native --stride 256 --text builtin:wikitext"
        );
        assert!(!line.hipfire_set_aside);
    }

    #[test]
    fn bench_backend_sets_a_working_copy_aside_and_refuses_a_gguf_one() {
        let p = project();
        let copy = p.path().join("copies/copy-1");
        let source = p.path().join("source");
        let hipfire = Some(Backend::Hipfire);
        let (backend, warning) = bench_backend(hipfire, Backend::Llama, &[&copy]).unwrap();
        assert_eq!(backend, Backend::Native);
        let warning = warning.unwrap();
        assert!(
            warning.starts_with(crate::mcp::HIPFIRE_SET_ASIDE),
            "{warning}"
        );
        assert!(
            warning.contains("copy-1"),
            "the reason names the copy: {warning}"
        );
        assert_eq!(
            bench_backend(None, Backend::Native, &[&copy]),
            Ok((Backend::Native, None))
        );
        assert_eq!(
            bench_backend(Some(Backend::LlamaServer), Backend::Llama, &[&copy]),
            Ok((Backend::LlamaServer, None))
        );
        assert_eq!(
            bench_backend(hipfire, Backend::Llama, &[&source]),
            Ok((Backend::Hipfire, None))
        );
        // A working-copy baseline moves the whole bench too.
        assert_eq!(
            bench_backend(hipfire, Backend::Llama, &[&source, &copy])
                .unwrap()
                .0,
            Backend::Native
        );
        let gguf = p.path().join("copies/m.gguf");
        let refused = bench_backend(hipfire, Backend::Llama, &[&gguf]).unwrap_err();
        assert!(refused.contains("copie de travail"), "{refused}");
    }
}
