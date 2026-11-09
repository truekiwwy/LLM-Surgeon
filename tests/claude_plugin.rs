//! The Claude Code plugin in `plugin/`, checked against the plugin reference read on
//! code.claude.com (manifest `.claude-plugin/plugin.json`, `.mcp.json` with `mcpServers`,
//! `skills/<name>/SKILL.md` with a YAML frontmatter, strict `userConfig` options,
//! `${user_config.KEY}` substitution in MCP server args).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use clap::Parser;
use serde_json::Value;
use weight_surgeon::cli::{Cli, Commands};

fn plugin_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("plugin")
}

fn json(rel: &str) -> Value {
    let text = std::fs::read_to_string(plugin_dir().join(rel)).unwrap();
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{rel}: {e}"))
}

#[test]
fn manifest_has_the_required_and_recommended_fields() {
    let manifest = json(".claude-plugin/plugin.json");
    let name = manifest["name"].as_str().unwrap();
    assert!(
        !name.is_empty()
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
        "kebab-case name: {name}"
    );
    for reserved in ["claude", "anthropic", "anthropics", "cc-plugin"] {
        assert!(!name.starts_with(reserved), "reserved prefix: {name}");
    }
    assert_eq!(manifest["version"], env!("CARGO_PKG_VERSION"));
    assert!(manifest["description"].as_str().unwrap().len() > 20);
    assert!(!manifest["author"]["name"].as_str().unwrap().is_empty());
    let allowed: BTreeSet<&str> = [
        "type",
        "title",
        "description",
        "required",
        "default",
        "options",
        "multiple",
        "sensitive",
        "min",
        "max",
    ]
    .into();
    let types = ["string", "number", "boolean", "directory", "file"];
    for (key, option) in manifest["userConfig"].as_object().unwrap() {
        assert!(key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_'));
        let option = option.as_object().unwrap();
        assert!(option.keys().all(|k| allowed.contains(k.as_str())), "{key}");
        assert!(types.contains(&option["type"].as_str().unwrap()), "{key}");
        for field in ["title", "description"] {
            assert!(!option[field].as_str().unwrap().is_empty(), "{key}.{field}");
        }
    }
}

#[test]
fn mcp_server_runs_the_crate_binary_with_real_options() {
    let manifest = json(".claude-plugin/plugin.json");
    let declared = manifest["userConfig"].as_object().unwrap();
    let servers = json(".mcp.json")["mcpServers"].clone();
    let servers = servers.as_object().unwrap();
    assert_eq!(servers.len(), 1);
    let server = &servers["surgeon"];
    let binary = Path::new(env!("CARGO_BIN_EXE_surgeon"))
        .file_stem()
        .unwrap();
    assert_eq!(
        server["command"],
        binary.to_str().unwrap(),
        "the crate's [[bin]]"
    );
    let args: Vec<&str> = server["args"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| a.as_str().unwrap())
        .collect();
    for arg in &args {
        if let Some(rest) = arg.strip_prefix("${") {
            let var = rest.strip_suffix('}').unwrap();
            let key = var.strip_prefix("user_config.").unwrap_or(var);
            assert!(
                declared.contains_key(key) || var.starts_with("CLAUDE_"),
                "undeclared {arg}"
            );
        }
    }
    // The command line `surgeon` parses once the placeholders are filled.
    let dir = tempfile::tempdir().unwrap();
    let filled = args.iter().map(|a| match a.starts_with("${") {
        true => dir.path().to_str().unwrap(),
        false => a,
    });
    let cli = Cli::try_parse_from(std::iter::once("surgeon").chain(filled)).unwrap();
    let Commands::Mcp(mcp) = cli.command else {
        panic!("not `surgeon mcp`")
    };
    assert_eq!(mcp.allow_dir, [dir.path()]);
    assert!(!mcp.allow_no_backup, "the backup stays imposed");
}

fn frontmatter(markdown: &str) -> Vec<(String, String)> {
    let mut lines = markdown.lines();
    assert_eq!(lines.next(), Some("---"), "frontmatter on the first line");
    lines
        .take_while(|l| *l != "---")
        .filter_map(|l| l.split_once(": "))
        .map(|(k, v)| (k.to_owned(), v.to_owned()))
        .collect()
}

#[test]
fn the_skill_imposes_the_safe_workflow_with_real_tools() {
    let skill = plugin_dir().join("skills/safe-surgery-workflow/SKILL.md");
    let markdown = std::fs::read_to_string(&skill).unwrap();
    let fields = frontmatter(&markdown);
    let get = |k: &str| fields.iter().find(|(key, _)| key == k).map(|(_, v)| v);
    assert_eq!(
        get("name").map(String::as_str),
        Some("safe-surgery-workflow")
    );
    let description = get("description").unwrap();
    assert!(description.chars().count() <= 1536 && description.len() > 40);
    for step in [
        "surgeon_check",
        "copie",
        "dry_run: true",
        "backup: false",
        "--allow-no-backup",
        "surgeon_bench",
        "non mesuré",
    ] {
        assert!(markdown.contains(step), "missing step: {step}");
    }
    let tools: BTreeSet<String> = weight_surgeon::mcp::tool_definitions()
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_owned())
        .collect();
    for word in markdown.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
        if word.starts_with("surgeon_") {
            assert!(tools.contains(word), "{word} is not an MCP tool");
        }
    }
}

#[test]
fn the_plugin_holds_no_secret_and_no_personal_path() {
    let mut stack = vec![plugin_dir()];
    let mut files = 0;
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            files += 1;
            let text = std::fs::read_to_string(&path).unwrap();
            let lower = text.to_lowercase();
            for banned in [
                "c:\\users",
                "c:/users",
                "/home/",
                "/users/",
                "password",
                "secret",
                "api_key",
                "apikey",
                "bearer ",
                "-----begin",
                "ghp_",
                "sk-ant-",
                "akia",
            ] {
                assert!(!lower.contains(banned), "{}: {banned}", path.display());
            }
            assert!(!text.contains('@'), "{}: e-mail address", path.display());
        }
    }
    assert_eq!(files, 3, "plugin.json, .mcp.json, SKILL.md");
}
