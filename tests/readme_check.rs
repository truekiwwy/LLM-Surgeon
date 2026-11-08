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
            .take_while(|t| **t != "surgeon" && !["|", "&&", ">"].contains(t));
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
        if let Some(sub) = sub.filter(|s| s.chars().all(|c| c.is_ascii_lowercase())) {
            let options = rest
                .filter(|t| t.starts_with("--"))
                .map(|t| t.split('=').next().unwrap().to_string())
                .collect();
            found.push((sub, options));
        }
    }
    found
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
            let command = match sub.as_str() {
                "" => &cli,
                _ => cli
                    .find_subcommand(&sub)
                    .unwrap_or_else(|| panic!("{name}: `surgeon {sub}` is not a command ({line})")),
            };
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
