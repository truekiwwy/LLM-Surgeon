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

fn lowercase_word(token: &str) -> bool {
    !token.is_empty() && token.chars().all(|c| c.is_ascii_lowercase())
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

    let english = check_readme("README.md");
    for leaf in ["logits", "ppl", "eval", "dump"] {
        assert!(
            english.contains(&format!("bench {leaf}")),
            "README.md never runs `surgeon bench {leaf}`"
        );
    }

    let tools = weight_surgeon::mcp::tool_definitions();
    let count = tools.as_array().unwrap().len();
    let words = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    let mots = [
        "zéro", "un", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf",
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
