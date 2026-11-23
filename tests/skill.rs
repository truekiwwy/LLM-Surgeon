//! The distributable skill `skills/weight-surgeon/` (Agent Skills format: `SKILL.md` with a
//! YAML frontmatter, references beside it), checked against the engine it teaches: every
//! `surgeon_*` word is a real MCP tool, every `builtin:` set exists, every JSON block is an
//! argument some tool accepts, and the method of the studio (the project and its origin
//! first, then doctor, funnel, orthogonal composition, certification) is all there.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use serde_json::Value;

fn skill_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("skills/weight-surgeon")
}

/// Every file of the skill, `SKILL.md` first.
fn files() -> Vec<(PathBuf, String)> {
    let dir = skill_dir();
    let mut paths = vec![dir.join("SKILL.md")];
    for entry in std::fs::read_dir(dir.join("references")).unwrap() {
        paths.push(entry.unwrap().path());
    }
    paths
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).unwrap();
            (p, text)
        })
        .collect()
}

/// `key: value` lines of the frontmatter; `None` when the file does not open with one.
fn frontmatter(markdown: &str) -> Option<Vec<(String, String)>> {
    let mut lines = markdown.lines();
    (lines.next() == Some("---")).then_some(())?;
    let mut fields = Vec::new();
    for line in lines.by_ref() {
        if line == "---" {
            return Some(fields);
        }
        let (k, v) = line.split_once(": ")?;
        fields.push((k.to_owned(), v.to_owned()));
    }
    None
}

#[test]
fn the_frontmatter_follows_the_agent_skills_format() {
    let markdown = std::fs::read_to_string(skill_dir().join("SKILL.md")).unwrap();
    let fields = frontmatter(&markdown).expect("a YAML frontmatter on the first line");
    let keys: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["name", "description"], "one line each, nothing else");
    let name = &fields[0].1;
    assert_eq!(name, "weight-surgeon", "the folder name");
    assert!(
        name.len() <= 64
            && name
                .chars()
                .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-'),
        "{name}"
    );
    let description = &fields[1].1;
    let chars = description.chars().count();
    assert!((200..=1024).contains(&chars), "{chars} characters");
    // A plain YAML scalar: no quote to close, no `: ` that would start a mapping.
    assert!(!description.contains(": ") && !description.starts_with(['"', '\'', '>', '|']));
    for trigger in [
        "surgeon_doctor",
        "refusal_probe",
        "surgeon_compose",
        "surgeon-gui",
    ] {
        assert!(description.contains(trigger), "description: {trigger}");
    }
}

#[test]
fn the_method_of_the_studio_is_all_there() {
    let markdown = std::fs::read_to_string(skill_dir().join("SKILL.md")).unwrap();
    let init = markdown.find("surgeon_workspace").unwrap();
    let doctor = markdown.find("surgeon_doctor").unwrap();
    let extract = markdown.find("surgeon_extract").unwrap();
    let certify = markdown.find("## 3. Certification").unwrap();
    assert!(
        init < doctor && doctor < extract && extract < certify,
        "project, doctor, funnel, then proof"
    );
    for needle in [
        // Step 0: the project, its origin recorded by the engine.
        "`action: \"init\"`",
        "output_exists",
        // Step 1: the machine.
        "`f32`",
        "`bf16`",
        "ref_bench",
        "total_mb",
        "redémarrer",
        // Step 2: inspect, extract, funnel, compose.
        "surgeon_inspect",
        "refusal_probe",
        "first-token-prob",
        "surgeon_autotune",
        "refusal_prompts",
        "surgeon_compose",
        "orthogonal_preserve",
        "orthogonal-preserve",
        "dry_run: true",
        "surgeon_restore",
        // Step 3: certification.
        "is_heuristic: true",
        "baseline_model",
        "builtin:refusals",
        "builtin:math",
        "builtin:logic",
        "builtin:general",
        "pas certifié",
        // Step 4: the project the studio reopens.
        "project.surgeon.json",
        "copies/",
        "chat/mcp_runs.jsonl",
        "ledger.jsonl",
        "patch_",
        "record_finding",
        "run_id",
        "non mesuré",
    ] {
        assert!(markdown.contains(needle), "missing: {needle}");
    }
}

#[test]
fn every_tool_set_and_link_named_is_real() {
    let tools: BTreeSet<String> = weight_surgeon::mcp::tool_definitions()
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_owned())
        .collect();
    let sets = weight_surgeon::bench::builtin::names();
    let sets: BTreeSet<&str> = sets.split(", ").collect();
    let mut cited = BTreeSet::new();
    for (path, text) in files() {
        let words = text.split(|c: char| !(c.is_ascii_alphanumeric() || "_:-".contains(c)));
        for word in words {
            let word = word.trim_end_matches(':');
            // `surgeon_*` names the family, not a tool.
            if word.starts_with("surgeon_") && word != "surgeon_" {
                assert!(tools.contains(word), "{}: {word}", path.display());
                cited.insert(word.to_owned());
            }
            if word.starts_with("builtin:") {
                let text = weight_surgeon::bench::builtin::text(std::path::Path::new(word));
                assert!(
                    sets.contains(word) || text.is_some_and(|t| t.is_ok()),
                    "{}: {word}",
                    path.display()
                );
            }
        }
        // Relative links resolve from the file that holds them.
        for link in text.split("](").skip(1).filter_map(|r| r.split(')').next()) {
            if !link.contains("://") {
                let target = path.parent().unwrap().join(link);
                assert!(target.is_file(), "{}: {link}", path.display());
            }
        }
    }
    for needed in [
        "surgeon_doctor",
        "surgeon_check",
        "surgeon_inspect",
        "surgeon_extract",
        "surgeon_apply",
        "surgeon_bench",
        "surgeon_autotune",
        "surgeon_compose",
        "surgeon_restore",
        "surgeon_files",
        "surgeon_workspace",
    ] {
        assert!(cited.contains(needed), "never cited: {needed}");
    }
}

/// The ```json blocks of a markdown file.
fn json_blocks(markdown: &str) -> Vec<Value> {
    let mut blocks = Vec::new();
    let mut current: Option<String> = None;
    for line in markdown.lines() {
        match (line.trim_start().strip_prefix("```"), current.take()) {
            (Some("json"), None) => current = Some(String::new()),
            (Some(_), Some(block)) => {
                blocks.push(serde_json::from_str(&block).unwrap_or_else(|e| panic!("{e}: {block}")))
            }
            (None, Some(mut block)) => {
                block.push_str(line);
                block.push('\n');
                current = Some(block);
            }
            (_, None) => {}
        }
    }
    blocks
}

#[test]
fn every_json_block_is_accepted_by_a_tool() {
    let tools: Vec<(String, jsonschema::Validator)> = weight_surgeon::mcp::tool_definitions()
        .as_array()
        .unwrap()
        .iter()
        .map(|t| {
            let name = t["name"].as_str().unwrap().to_owned();
            (name, jsonschema::validator_for(&t["inputSchema"]).unwrap())
        })
        .collect();
    let typo = serde_json::json!({"mode": "refusal_probe", "modle": "copies/copy-1"});
    assert!(
        !tools.iter().any(|(_, v)| v.is_valid(&typo)),
        "a typo is caught"
    );
    let mut checked = 0;
    for (path, text) in files() {
        for block in json_blocks(&text) {
            checked += 1;
            if let Some(servers) = block.get("mcpServers") {
                assert_eq!(
                    servers["weight-surgeon"]["args"],
                    serde_json::json!(["mcp"])
                );
                continue;
            }
            assert!(
                tools.iter().any(|(_, v)| v.is_valid(&block)),
                "{}: no tool takes {block}",
                path.display()
            );
        }
    }
    assert!(checked >= 10, "{checked} blocks");
}

#[test]
fn the_skill_holds_no_secret_and_no_personal_path() {
    for (path, text) in files() {
        let lower = text.to_lowercase();
        for banned in [
            "c:\\users",
            "c:/users",
            "/home/",
            "password",
            "api_key",
            "bearer ",
            "-----begin",
            "ghp_",
            "sk-ant-",
        ] {
            assert!(!lower.contains(banned), "{}: {banned}", path.display());
        }
        assert!(!text.contains('@'), "{}: e-mail address", path.display());
        assert!(!text.contains('\u{2014}'), "{}: em dash", path.display());
    }
}
