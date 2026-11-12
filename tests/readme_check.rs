//! Keeps both READMEs honest: every `surgeon <command>` and `--option` quoted in their shell
//! blocks exists in the real CLI, relative links resolve, and both files cover the same commands.

use std::collections::BTreeSet;
use std::path::Path;

use clap::CommandFactory;
use weight_surgeon::cli::Cli;

const SHELL_BLOCKS: [&str; 6] = ["bash", "sh", "shell", "console", "powershell", "pwsh"];

fn read(name: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(name)).unwrap()
}

/// Logical lines of the fenced shell blocks (`\` and PowerShell backtick continuations joined).
fn shell_lines(markdown: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut in_shell = None;
    let mut pending = String::new();
    for line in markdown.lines() {
        let fence = line.trim_start().strip_prefix("```");
        match (fence, in_shell) {
            (Some(info), None) => in_shell = Some(SHELL_BLOCKS.contains(&info.trim())),
            (Some(_), Some(_)) => in_shell = None,
            (None, Some(true)) => {
                let trimmed = line.trim_end();
                match trimmed
                    .strip_suffix('\\')
                    .or_else(|| trimmed.strip_suffix('`'))
                {
                    Some(head) => pending.push_str(head),
                    None => lines.push(std::mem::take(&mut pending) + trimmed),
                }
            }
            _ => {}
        }
    }
    lines
}

/// `(subcommand, options)` for each `surgeon` invocation of a line; global options get "".
fn invocations(line: &str) -> Vec<(String, Vec<String>)> {
    let tokens: Vec<&str> = line
        .split_whitespace()
        .map(|t| t.trim_matches(['`', '"', '\'', ';']))
        .collect();
    let mut found = Vec::new();
    for (i, _) in tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == "surgeon" || **t == "surgeon.exe")
    {
        let mut rest = tokens[i + 1..]
            .iter()
            .take_while(|t| **t != "surgeon" && !["|", "&&", ">"].contains(t))
            .peekable();
        let mut globals = Vec::new();
        let sub = loop {
            match rest.next() {
                Some(t) if t.starts_with("--") && *t != "--" => globals.push(t.to_string()),
                Some(t) if t.starts_with('-') => break None,
                Some(t) => break Some(t.to_string()),
                None => break None,
            }
        };
        if !globals.is_empty() {
            found.push((String::new(), globals));
        }
        if let Some(mut sub) = sub.filter(|s| lowercase_word(s)) {
            // Nested subcommand: `surgeon bench ppl --warn-delta` -> "bench ppl".
            if let Some(leaf) = rest.next_if(|t| lowercase_word(t)) {
                sub = format!("{sub} {leaf}");
            }
            let options = rest
                .filter(|t| t.starts_with("--"))
                .map(|t| t.split('=').next().unwrap().to_string())
                .collect();
            found.push((sub, options));
        }
    }
    found
}

/// A subcommand name: lowercase letters, hyphens inside (`judge-sample`).
fn lowercase_word(token: &str) -> bool {
    token.starts_with(|c: char| c.is_ascii_lowercase())
        && token.chars().all(|c| c.is_ascii_lowercase() || c == '-')
}

fn long_options(command: &clap::Command) -> BTreeSet<String> {
    command
        .get_arguments()
        .filter_map(|a| a.get_long())
        .map(|l| format!("--{l}"))
        .collect()
}

fn check_readme(name: &str) -> BTreeSet<String> {
    let mut cli = Cli::command();
    cli.build();
    let markdown = read(name);
    let mut cited = BTreeSet::new();
    for line in shell_lines(&markdown) {
        for (sub, options) in invocations(&line) {
            let command = sub.split_whitespace().fold(&cli, |parent, word| {
                parent
                    .find_subcommand(word)
                    .unwrap_or_else(|| panic!("{name}: `surgeon {sub}` is not a command ({line})"))
            });
            let known = long_options(command);
            for option in options {
                assert!(
                    known.contains(&option),
                    "{name}: `{option}` is not an option of `surgeon {sub}` ({line})"
                );
            }
            if !sub.is_empty() {
                cited.insert(sub);
            }
        }
    }

    for target in markdown
        .split("](")
        .skip(1)
        .filter_map(|rest| rest.split(')').next())
    {
        let path = target.split('#').next().unwrap();
        if path.is_empty() || path.contains("://") || path.starts_with("mailto:") {
            continue;
        }
        let resolved = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        assert!(resolved.exists(), "{name}: broken relative link `{target}`");
    }
    assert!(
        !markdown.contains(r"C:\Users\"),
        "{name}: personal Windows path"
    );
    cited
}

#[test]
fn readme_commands_options_and_links_are_real() {
    let english = check_readme("README.md");
    let french = check_readme("README.fr.md");
    assert_eq!(
        english, french,
        "both READMEs must quote the same subcommands"
    );
    for command in [
        "check", "inspect", "extract", "apply", "eval", "tui", "mcp", "schema",
    ] {
        assert!(
            english.contains(command),
            "README.md never runs `surgeon {command}`"
        );
    }
    assert!(
        read("README.md").contains("](README.fr.md)")
            && read("README.fr.md").contains("](README.md)")
    );
}

#[test]
fn the_checker_catches_fakes() {
    assert_eq!(
        invocations("surgeon --json inspect --model m --layers=2"),
        [
            (String::new(), vec!["--json".into()]),
            ("inspect".into(), vec!["--model".into(), "--layers".into()]),
        ]
    );
    assert_eq!(
        invocations("claude mcp add surgeon -- surgeon mcp --allow-dir d"),
        [("mcp".into(), vec!["--allow-dir".into()])]
    );
    let lines =
        shell_lines("```bash\nsurgeon tui \\\n  --snapshot\n```\n```text\nsurgeon nope\n```\n");
    assert_eq!(lines, ["surgeon tui   --snapshot"]);
    let mut cli = Cli::command();
    cli.build();
    assert!(cli.find_subcommand("nope").is_none());
    assert!(!long_options(cli.find_subcommand("tui").unwrap()).contains("--layers"));
}

#[test]
fn nested_bench_commands_and_mcp_tool_count_are_checked() {
    assert_eq!(
        invocations("surgeon --json bench ppl --model m.gguf --warn-delta 0.3"),
        [
            (String::new(), vec!["--json".into()]),
            (
                "bench ppl".into(),
                vec!["--model".into(), "--warn-delta".into()]
            ),
        ]
    );
    let mut cli = Cli::command();
    cli.build();
    let ppl = cli
        .find_subcommand("bench")
        .and_then(|b| b.find_subcommand("ppl"))
        .unwrap();
    assert!(long_options(ppl).contains("--warn-delta"));
    assert!(!long_options(cli.find_subcommand("bench").unwrap()).contains("--warn-delta"));

    assert_eq!(
        invocations("surgeon bench judge-sample --responses run.json --n 6"),
        [(
            "bench judge-sample".into(),
            vec!["--responses".into(), "--n".into()]
        )]
    );

    let english = check_readme("README.md");
    for leaf in [
        "logits",
        "ppl",
        "eval",
        "dump",
        "judge-sample",
        "judge-eval",
    ] {
        assert!(
            english.contains(&format!("bench {leaf}")),
            "README.md never runs `surgeon bench {leaf}`"
        );
    }

    let tools = weight_surgeon::mcp::tool_definitions();
    let count = tools.as_array().unwrap().len();
    let words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven",
    ];
    let mots = [
        "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix",
        "onze",
    ];
    let (en, fr) = (read("README.md"), read("README.fr.md"));
    assert!(
        en.contains(&format!("{} tools", words[count])),
        "README.md: {count} tools"
    );
    assert!(
        fr.contains(&format!("{} outils", mots[count])),
        "README.fr.md: {count} outils"
    );
    for (n, (w, m)) in words
        .iter()
        .zip(mots)
        .enumerate()
        .filter(|(n, _)| *n != count && *n > 1)
    {
        assert!(
            !en.contains(&format!("serves {w} tools")),
            "README.md still says {n} tools"
        );
        assert!(
            !fr.contains(&format!("sert {m} outils")),
            "README.fr.md still says {n} outils"
        );
    }
}

// ---------------------------------------------------------------------------------------
// Documentation rewrite checks: parity between the two READMEs, accuracy against the real CLI
// and MCP tool list, coverage of every command, tool and option, and hygiene.
// ---------------------------------------------------------------------------------------

const READMES: [&str; 2] = ["README.md", "README.fr.md"];

/// Fenced blocks as `(info string, lines)`; fences inside a block never nest.
fn blocks(markdown: &str) -> Vec<(String, Vec<String>)> {
    let mut found = Vec::new();
    let mut current: Option<(String, Vec<String>)> = None;
    for line in markdown.lines() {
        match (line.trim_start().strip_prefix("```"), current.take()) {
            (Some(info), None) => current = Some((info.trim().to_string(), Vec::new())),
            (Some(_), Some(block)) => found.push(block),
            (None, Some(mut block)) => {
                block.1.push(line.to_string());
                current = Some(block);
            }
            (None, None) => {}
        }
    }
    assert!(current.is_none(), "unclosed code fence");
    found
}

/// Markdown outside fenced blocks.
fn prose(markdown: &str) -> Vec<&str> {
    let mut inside = false;
    let mut lines = Vec::new();
    for line in markdown.lines() {
        if line.trim_start().starts_with("```") {
            inside = !inside;
        } else if !inside {
            lines.push(line);
        }
    }
    lines
}

/// `(level, anchor)` of every heading; each one must be preceded by `<a id="..."></a>`.
fn outline(name: &str, markdown: &str) -> Vec<(usize, String)> {
    let mut anchor: Option<String> = None;
    let mut found = Vec::new();
    for line in prose(markdown) {
        if let Some(id) = line
            .strip_prefix("<a id=\"")
            .and_then(|r| r.strip_suffix("\"></a>"))
        {
            anchor = Some(id.to_string());
        } else if line.starts_with('#') {
            let level = line.chars().take_while(|c| *c == '#').count();
            let id = anchor
                .take()
                .unwrap_or_else(|| panic!("{name}: heading without anchor: {line}"));
            found.push((level, id));
        }
    }
    found
}

/// `(heading, body)` of every heading, fenced blocks included in the body.
fn sections(markdown: &str) -> Vec<(String, String)> {
    let mut inside = false;
    let mut out: Vec<(String, String)> = Vec::new();
    for line in markdown.lines() {
        if line.trim_start().starts_with("```") {
            inside = !inside;
        }
        if !inside && line.starts_with('#') {
            out.push((
                line.trim_start_matches('#').trim().to_string(),
                String::new(),
            ));
        } else if let Some(last) = out.last_mut() {
            last.1.push_str(line);
            last.1.push('\n');
        }
    }
    out
}

/// Every command path of the real CLI (`bench ppl`, `lora merge`...), parents included, with
/// its long options (help, version and the global ones left out).
fn command_paths(command: &clap::Command, prefix: &str, out: &mut Vec<(String, Vec<String>)>) {
    for sub in command.get_subcommands().filter(|s| s.get_name() != "help") {
        let path = format!("{prefix}{}", sub.get_name());
        let options: Vec<String> = sub
            .get_arguments()
            .filter_map(|a| a.get_long())
            .filter(|l| !["help", "version", "json", "verbose"].contains(l))
            .map(|l| format!("--{l}"))
            .collect();
        out.push((path.clone(), options));
        command_paths(sub, &format!("{path} "), out);
    }
}

fn real_commands() -> Vec<(String, Vec<String>)> {
    let mut cli = Cli::command();
    cli.build();
    let mut paths = Vec::new();
    command_paths(&cli, "", &mut paths);
    paths
}

fn tool_names() -> Vec<String> {
    weight_surgeon::mcp::tool_definitions()
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn both_readmes_have_the_same_structure() {
    let (en, fr) = (read(READMES[0]), read(READMES[1]));
    assert_eq!(
        outline(READMES[0], &en),
        outline(READMES[1], &fr),
        "headings and anchors differ"
    );
    let callouts = |markdown: &str| -> Vec<&'static str> {
        prose(markdown)
            .iter()
            .filter_map(|l| match *l {
                "> **For beginners**" | "> **Pour les débutants**" => Some("beginners"),
                "> **For experts**" | "> **Pour les experts**" => Some("experts"),
                _ => None,
            })
            .collect()
    };
    let (callouts_en, callouts_fr) = (callouts(&en), callouts(&fr));
    assert!(callouts_en.contains(&"beginners") && callouts_en.contains(&"experts"));
    assert_eq!(callouts_en, callouts_fr, "beginner/expert boxes differ");
    // Only the comments of shell blocks are translated: everything else is identical.
    let code = |markdown: &str| -> Vec<(String, Vec<String>)> {
        blocks(markdown)
            .into_iter()
            .map(|(info, lines)| {
                let is_shell = SHELL_BLOCKS.contains(&info.as_str());
                let lines = lines
                    .into_iter()
                    .filter(|l| !(is_shell && l.trim_start().starts_with('#')))
                    .collect();
                (info, lines)
            })
            .collect()
    };
    assert_eq!(code(&en), code(&fr), "code blocks differ");
}

#[test]
fn both_readmes_cite_the_same_options_and_tools() {
    let options = |name: &str| -> BTreeSet<(String, String)> {
        shell_lines(&read(name))
            .iter()
            .flat_map(|l| invocations(l))
            .flat_map(|(sub, opts)| opts.into_iter().map(move |o| (sub.clone(), o)))
            .collect()
    };
    assert_eq!(options(READMES[0]), options(READMES[1]));
    let mentioned = |name: &str| -> BTreeSet<String> {
        let text = read(name);
        let mut found = BTreeSet::new();
        for (at, _) in text.match_indices("surgeon_") {
            let word: String = text[at..]
                .chars()
                .take_while(|c| c.is_ascii_lowercase() || *c == '_')
                .collect();
            found.insert(word.trim_end_matches('_').to_string());
        }
        found
    };
    let real: BTreeSet<String> = tool_names().into_iter().collect();
    for name in READMES {
        assert_eq!(
            mentioned(name),
            real,
            "{name}: MCP tools cited vs tools/list"
        );
    }
}

#[test]
fn every_command_and_option_is_documented_and_real() {
    let commands = real_commands();
    for name in READMES {
        let markdown = read(name);
        let mut documented = BTreeSet::new();
        for (heading, body) in sections(&markdown) {
            let Some(path) = heading
                .strip_prefix("`surgeon ")
                .and_then(|h| h.strip_suffix('`'))
            else {
                continue;
            };
            let (_, real_options) = commands
                .iter()
                .find(|(p, _)| p == path)
                .unwrap_or_else(|| panic!("{name}: `surgeon {path}` is not a command"));
            let rows: BTreeSet<String> = body
                .lines()
                .filter_map(|l| l.strip_prefix("| `"))
                .filter(|r| r.starts_with("--"))
                .map(|r| r.split('`').next().unwrap().to_string())
                .collect();
            for row in &rows {
                assert!(
                    real_options.contains(row),
                    "{name}: `{row}` is not an option of `surgeon {path}`"
                );
            }
            for option in real_options {
                assert!(
                    rows.contains(option),
                    "{name}: `surgeon {path}` does not document {option}"
                );
            }
            documented.insert(path.to_string());
        }
        for (path, _) in &commands {
            assert!(
                documented.contains(path),
                "{name}: no reference card for `surgeon {path}`"
            );
        }
    }
}

#[test]
fn quoted_options_exist_somewhere_in_the_cli() {
    let mut known: BTreeSet<String> = real_commands().into_iter().flat_map(|(_, o)| o).collect();
    known.extend(["--json", "--verbose", "--help", "--version"].map(String::from));
    // Options of other tools quoted on purpose: Claude Code, cargo, llama-perplexity,
    // llama-server (its whitelisted flags).
    let foreign = [
        "--host",
        "--port",
        "--api-key",
        "--scope",
        "--temp",
        "--release",
        "--example",
        "--path",
        "--target",
        "--all-targets",
        "--batch-size",
        "--ubatch-size",
        "--ppl-stride",
        "--no-mmap",
        "--mlock",
    ];
    for name in READMES {
        for line in prose(&read(name)) {
            for token in line.split('`').skip(1).step_by(2) {
                for word in token.split_whitespace() {
                    let option = word.split(['=', ',', ')', '.']).next().unwrap();
                    let plain = option.len() > 2
                        && option.starts_with("--")
                        && option[2..]
                            .chars()
                            .all(|c| c.is_ascii_lowercase() || c == '-' || c.is_ascii_digit());
                    assert!(
                        !plain || known.contains(option) || foreign.contains(&option),
                        "{name}: unknown option `{option}` in: {line}"
                    );
                }
            }
        }
    }
}

#[test]
fn readmes_are_clean() {
    let license_file = Path::new(env!("CARGO_MANIFEST_DIR")).join("LICENSE");
    for name in READMES {
        let markdown = read(name);
        blocks(&markdown); // panics on an unclosed fence
        let fences = markdown
            .lines()
            .filter(|l| l.trim_start().starts_with("```"))
            .count();
        assert_eq!(fences % 2, 0, "{name}: unbalanced fences");
        for forbidden in [r"C:\Users", r"C:\surgery", "TODO", "lorem", "ollama"] {
            assert!(
                !markdown.to_lowercase().contains(&forbidden.to_lowercase()),
                "{name}: contains `{forbidden}`"
            );
        }
        if !license_file.exists() {
            let lower = markdown.to_lowercase();
            assert!(
                !lower.contains("badge/license") && !lower.contains("badge/licence"),
                "{name}: license badge without a LICENSE file"
            );
        }
    }
}
