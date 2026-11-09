//! Keeps the P13 documentation fragments honest, with the same rules as `readme_check.rs`:
//! every `surgeon <command>` and `--option` quoted in a shell block or an inline code span
//! exists in the real CLI (nested `lora merge|extract` included), relative links resolve, and
//! the English and French fragments cite the same commands. Also checks that the MCP fragment
//! documents every argument of the two new tools.

use std::collections::BTreeSet;
use std::path::Path;

use clap::CommandFactory;
use weight_surgeon::cli::Cli;

const SHELL_BLOCKS: [&str; 6] = ["bash", "sh", "shell", "console", "powershell", "pwsh"];
const README_FRAGMENTS: [&str; 2] = ["p13-readme-en.md", "p13-readme-fr.md"];
const ALL_FRAGMENTS: [&str; 3] = ["p13-readme-en.md", "p13-readme-fr.md", "p13-mcp.md"];

fn fragment(name: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("docs")
        .join("fragments")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
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

/// Inline code spans of the prose (fenced blocks excluded).
fn inline_spans(markdown: &str) -> Vec<String> {
    let mut in_fence = false;
    let mut spans = Vec::new();
    for line in markdown.lines() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        } else if !in_fence {
            spans.extend(
                line.split('`')
                    .skip(1)
                    .step_by(2)
                    .map(|s| s.trim().to_string()),
            );
        }
    }
    spans
}

fn lowercase_word(token: &str) -> bool {
    !token.is_empty() && token.chars().all(|c| c.is_ascii_lowercase())
}

/// `(subcommand, options)` for each `surgeon` invocation of a line; global options get "".
fn invocations(line: &str) -> Vec<(String, Vec<String>)> {
    let tokens: Vec<&str> = line
        .split_whitespace()
        .map(|t| t.trim_matches(['`', '"', '\'', ';', ',']))
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
            // Nested subcommand: `surgeon lora merge --lora` -> "lora merge".
            if let Some(leaf) = rest.next_if(|t| lowercase_word(t)) {
                sub = format!("{sub} {leaf}");
            }
            let options = rest
                .filter(|t| t.starts_with("--"))
                .map(|t| t.split('=').next().unwrap_or_default().to_string())
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

/// Every long option of any (nested) command: for a bare `--option` quoted in prose.
fn all_options(command: &clap::Command, found: &mut BTreeSet<String>) {
    found.extend(long_options(command));
    for sub in command.get_subcommands() {
        all_options(sub, found);
    }
}

/// Checks one invocation against the CLI; returns the subcommand it names.
fn check_invocation(cli: &clap::Command, name: &str, sub: &str, options: &[String], line: &str) {
    let command = sub.split_whitespace().fold(cli, |parent, word| {
        parent
            .find_subcommand(word)
            .unwrap_or_else(|| panic!("{name}: `surgeon {sub}` is not a command ({line})"))
    });
    let known = long_options(command);
    for option in options {
        assert!(
            known.contains(option),
            "{name}: `{option}` is not an option of `surgeon {sub}` ({line})"
        );
    }
}

/// Subcommands cited by a fragment; panics on any fake command or option.
fn check_fragment(name: &str) -> BTreeSet<String> {
    let mut cli = Cli::command();
    cli.build();
    let mut options_anywhere = BTreeSet::new();
    all_options(&cli, &mut options_anywhere);
    let markdown = fragment(name);
    let mut cited = BTreeSet::new();
    let spans = inline_spans(&markdown);
    for line in shell_lines(&markdown).into_iter().chain(spans.clone()) {
        for (sub, options) in invocations(&line) {
            check_invocation(&cli, name, &sub, &options, &line);
            if !sub.is_empty() {
                cited.insert(sub);
            }
        }
    }
    // A bare `--option` in prose must exist somewhere in the CLI.
    for span in spans.iter().filter(|s| s.starts_with("--")) {
        let option = span
            .split_whitespace()
            .next()
            .and_then(|t| t.split('=').next())
            .unwrap_or_default();
        assert!(
            options_anywhere.contains(option),
            "{name}: `{option}` is not an option of any command"
        );
    }

    for target in markdown
        .split("](")
        .skip(1)
        .filter_map(|rest| rest.split(')').next())
    {
        let path = target.split('#').next().unwrap_or_default();
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
    assert!(!markdown.contains('\u{2014}'), "{name}: em dash");
    cited
}

#[test]
fn readme_fragments_quote_only_real_commands_and_options() {
    let english = check_fragment(README_FRAGMENTS[0]);
    let french = check_fragment(README_FRAGMENTS[1]);
    assert_eq!(english, french, "both fragments cite the same subcommands");
    for command in ["diff", "apply", "lora merge", "lora extract", "tui"] {
        assert!(
            english.contains(command),
            "fragments never run `surgeon {command}`"
        );
    }
}

#[test]
fn mcp_fragment_quotes_only_real_commands_and_options() {
    check_fragment("p13-mcp.md");
}

#[test]
fn the_checker_catches_fakes() {
    assert_eq!(
        invocations("surgeon lora merge --model m --lora=a.safetensors"),
        [(
            "lora merge".to_string(),
            vec!["--model".to_string(), "--lora".to_string()]
        )]
    );
    assert_eq!(
        invocations("run `surgeon diff --rank 4`, then"),
        [("diff".to_string(), vec!["--rank".to_string()])]
    );
    assert_eq!(
        inline_spans("a `surgeon diff` b\n```bash\nsurgeon nope\n```\nc `--seed`"),
        ["surgeon diff", "--seed"]
    );
    let mut cli = Cli::command();
    cli.build();
    let merge = cli
        .find_subcommand("lora")
        .and_then(|l| l.find_subcommand("merge"))
        .map(long_options)
        .unwrap_or_default();
    assert!(merge.contains("--adapter-config") && !merge.contains("--rank"));
    let nested = std::panic::catch_unwind(|| {
        check_invocation(
            &Cli::command(),
            "fake",
            "lora merge",
            &["--rank".to_string()],
            "surgeon lora merge --rank 4",
        )
    });
    assert!(
        nested.is_err(),
        "an option of `lora extract` is refused on `lora merge`"
    );
}

#[test]
fn fragments_have_no_em_dash_or_personal_path() {
    for name in ALL_FRAGMENTS {
        let markdown = fragment(name);
        assert!(!markdown.contains('\u{2014}'), "{name}: em dash");
        assert!(!markdown.contains(r"C:\Users\"), "{name}: personal path");
    }
}

#[test]
fn mcp_fragment_documents_every_argument_of_the_two_tools() {
    let markdown = fragment("p13-mcp.md");
    let tools = weight_surgeon::mcp::tool_definitions();
    for tool in tools
        .as_array()
        .into_iter()
        .flatten()
        .filter(|t| matches!(t["name"].as_str(), Some("surgeon_diff" | "surgeon_lora")))
    {
        let name = tool["name"].as_str().unwrap_or_default();
        assert!(
            markdown.contains(&format!("`{name}`")),
            "{name} not documented"
        );
        for field in tool["inputSchema"]["properties"]
            .as_object()
            .into_iter()
            .flatten()
            .map(|(k, _)| k)
        {
            assert!(
                markdown.contains(&format!("`{field}`")),
                "{name}: argument `{field}` is not documented in p13-mcp.md"
            );
        }
    }
}
