//! The official CLI runtimes, used as installed and signed in by the user (the way to use a
//! subscription account): Claude Code (`claude -p`) and Codex (`codex exec`). The studio
//! never reads, copies or stores their credentials; it starts them non-interactively with
//! `surgeon mcp` as their only tool server. Their own loop and approvals are theirs, not the
//! studio's: the barrier is the MCP server (`--read-only` in "Ask" mode, the project folder
//! as the only allowed folder in "Modify" mode, so the source model is out of reach) and an
//! explicit tool allow-list.
//!
//! Options checked against `claude --help` (2.1.x) and `codex exec --help` (0.166): a binary
//! whose help lacks one is reported and not used.

use std::ffi::{OsStr, OsString};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::config::ProviderKind;
use super::policy::AgentMode;

const CLAUDE_OPTIONS: [&str; 8] = [
    "--print",
    "--output-format",
    "stream-json",
    "--mcp-config",
    "--strict-mcp-config",
    "--allowedTools",
    "--permission-prompts",
    "--tools",
];
const CODEX_OPTIONS: [&str; 6] = [
    "--json",
    "--sandbox",
    "--config",
    "--cd",
    "--skip-git-repo-check",
    "--ephemeral",
];
/// The engine tools an "Ask" runtime may call (its server also runs `--read-only`).
const READ_TOOLS: [&str; 5] = [
    "surgeon_check",
    "surgeon_inspect",
    "surgeon_graph",
    "surgeon_bench",
    "surgeon_backends",
];

#[derive(Debug, Clone, PartialEq)]
pub struct Detected {
    pub kind: ProviderKind,
    pub path: PathBuf,
    pub version: String,
    /// Options the help does not mention: the runtime is not used.
    pub missing: Vec<&'static str>,
}

impl Detected {
    pub fn usable(&self) -> bool {
        self.missing.is_empty()
    }
}

fn binary_name(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::ClaudeCode => "claude",
        _ => "codex",
    }
}

/// `name` in the folders of `path` (`PATH`), with the `PATHEXT` extensions on Windows.
pub fn find_on_path(name: &str, path: Option<&OsStr>) -> Option<PathBuf> {
    let exts: Vec<String> = match cfg!(windows) {
        true => std::env::var("PATHEXT")
            .unwrap_or_else(|_| ".EXE;.CMD;.BAT".into())
            .split(';')
            .filter(|e| !e.is_empty())
            .map(|e| e.to_ascii_lowercase())
            .collect(),
        false => vec![String::new()],
    };
    std::env::split_paths(path?)
        .flat_map(|dir| exts.iter().map(move |e| dir.join(format!("{name}{e}"))))
        .find(|p| p.is_file())
}

/// The first line `bin --version` prints, empty when it prints nothing in time.
pub fn version(bin: &Path) -> String {
    capture(bin, &["--version"], Duration::from_secs(15))
        .and_then(|v| {
            v.lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .map(str::to_owned)
        })
        .unwrap_or_default()
}

/// Finds the CLI, reads its version, and checks that its help lists every option used.
pub fn detect(kind: ProviderKind, path: Option<&OsStr>) -> Option<Detected> {
    let bin = find_on_path(binary_name(kind), path)?;
    let version = version(&bin);
    let (help_args, needed): (&[&str], &[&'static str]) = match kind {
        ProviderKind::ClaudeCode => (&["--help"], &CLAUDE_OPTIONS),
        _ => (&["exec", "--help"], &CODEX_OPTIONS),
    };
    let help = capture(&bin, help_args, Duration::from_secs(15)).unwrap_or_default();
    Some(Detected {
        kind,
        path: bin,
        version,
        missing: needed
            .iter()
            .filter(|o| !help.contains(**o))
            .copied()
            .collect(),
    })
}

fn command(program: &Path) -> Command {
    let mut c = Command::new(program);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        c.creation_flags(CREATE_NO_WINDOW);
    }
    c
}

/// stdout (then stderr) of a short command, or None past `timeout`.
fn capture(program: &Path, args: &[&str], timeout: Duration) -> Option<String> {
    let mut child = command(program)
        .args(args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let (tx, rx) = mpsc::channel();
    for (k, pipe) in [
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    ]
    .into_iter()
    .enumerate()
    .filter_map(|(k, p)| Some((k, p?)))
    {
        let tx = tx.clone();
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = BufReader::new(pipe).take(1 << 20).read_to_string(&mut text);
            let _ = tx.send((k, text));
        });
    }
    drop(tx);
    let deadline = Instant::now() + timeout;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    // The two readers end in any order: stdout first, whichever came back first.
    let mut parts: Vec<(usize, String)> = rx.iter().collect();
    parts.sort_by_key(|(k, _)| *k);
    let texts: Vec<String> = parts.into_iter().map(|(_, text)| text).collect();
    Some(texts.join("\n"))
}

/// The `surgeon` engine binary the runtime launches as its MCP server: next to this app,
/// else on `PATH`.
pub fn surgeon_exe(path: Option<&OsStr>) -> Option<PathBuf> {
    let name = format!("surgeon{}", std::env::consts::EXE_SUFFIX);
    std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|d| d.join(&name)))
        .filter(|p| p.is_file())
        .or_else(|| find_on_path("surgeon", path))
}

/// Everything a run needs, built by the code from the mode (never by the agent).
#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub cwd: PathBuf,
    /// The prompt goes on stdin: no free text on a command line.
    pub stdin: String,
}

/// The `surgeon mcp` arguments for a mode.
pub fn mcp_args(mode: AgentMode, project: &Path, source_dir: Option<&Path>) -> Vec<String> {
    let mut args = vec![
        "mcp".to_owned(),
        "--allow-dir".into(),
        project.display().to_string(),
    ];
    match mode {
        AgentMode::Ask => {
            if let Some(dir) = source_dir {
                args.extend(["--allow-dir".into(), dir.display().to_string()]);
            }
            args.push("--read-only".into());
        }
        // The source model's folder is not allowed at all: only working copies are reachable.
        AgentMode::Modify => {}
    }
    args
}

/// The tools the runtime may call without asking.
pub fn allowed_tools(mode: AgentMode) -> Vec<String> {
    let all = super::tools::engine_definitions();
    all.into_iter()
        .map(|t| t.name)
        .filter(|n| match mode {
            AgentMode::Ask => READ_TOOLS.contains(&n.as_str()),
            // Pipelines can promote a result: human only, never through a runtime.
            AgentMode::Modify => n != "surgeon_pipeline",
        })
        .collect()
}

pub struct LaunchInput<'a> {
    pub detected: &'a Detected,
    pub surgeon: &'a Path,
    pub mode: AgentMode,
    pub project: &'a Path,
    pub source_dir: Option<&'a Path>,
    pub model: Option<&'a str>,
    pub prompt: &'a str,
    /// Where the MCP configuration file of a Claude Code run is written.
    pub scratch: &'a Path,
}

pub fn launch(input: &LaunchInput) -> Result<Launch, String> {
    let server = json!({
        "command": input.surgeon,
        "args": mcp_args(input.mode, input.project, input.source_dir),
    });
    let tools = allowed_tools(input.mode);
    let mut args: Vec<OsString> = Vec::new();
    match input.detected.kind {
        ProviderKind::ClaudeCode => {
            let cfg = input.scratch.join("claude-mcp.json");
            std::fs::create_dir_all(input.scratch).map_err(|e| e.to_string())?;
            std::fs::write(&cfg, json!({"mcpServers": {"surgeon": server}}).to_string())
                .map_err(|e| format!("{} : {e}", cfg.display()))?;
            let allowed: Vec<String> = tools.iter().map(|t| format!("mcp__surgeon__{t}")).collect();
            for a in [
                "-p",
                "--output-format",
                "stream-json",
                "--verbose",
                "--strict-mcp-config",
                "--permission-prompts",
                "none",
                "--no-session-persistence",
                "--mcp-config",
            ] {
                args.push(a.into());
            }
            args.push(cfg.into_os_string());
            // No built-in tool (shell, files, web): only the surgeon server.
            args.extend(["--tools".into(), "".into()]);
            args.push("--allowedTools".into());
            args.push(allowed.join(",").into());
            if let Some(m) = input.model {
                args.extend(["--model".into(), m.into()]);
            }
        }
        _ => {
            let toml_list = |v: &[String]| serde_json::to_string(v).unwrap_or_default();
            let mcp = mcp_args(input.mode, input.project, input.source_dir);
            for a in [
                "exec",
                "--json",
                "--ephemeral",
                "--skip-git-repo-check",
                "--sandbox",
                "read-only",
            ] {
                args.push(a.into());
            }
            args.push("--cd".into());
            args.push(input.project.as_os_str().to_owned());
            for kv in [
                "approval_policy=\"never\"".to_owned(),
                format!(
                    "mcp_servers.surgeon.command={}",
                    serde_json::to_string(&input.surgeon.display().to_string()).unwrap_or_default()
                ),
                format!("mcp_servers.surgeon.args={}", toml_list(&mcp)),
                format!("mcp_servers.surgeon.enabled_tools={}", toml_list(&tools)),
            ] {
                args.push("--config".into());
                args.push(kv.into());
            }
            if let Some(m) = input.model {
                args.extend(["--model".into(), m.into()]);
            }
            args.push("-".into());
        }
    }
    Ok(Launch {
        program: input.detected.path.clone(),
        args,
        cwd: input.project.to_path_buf(),
        stdin: input.prompt.to_owned(),
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    Text(String),
    Tool {
        name: String,
        input: Value,
    },
    ToolResult(String),
    Done {
        cost_usd: Option<f64>,
        summary: String,
    },
    Error(String),
    Exit(Option<i32>),
}

/// One line of `claude -p --output-format stream-json --verbose`.
pub fn parse_claude(line: &str) -> Vec<Event> {
    let Ok(v) = serde_json::from_str::<Value>(line) else {
        return Vec::new();
    };
    let content = || {
        v["message"]["content"]
            .as_array()
            .cloned()
            .unwrap_or_default()
    };
    match v["type"].as_str() {
        Some("assistant") => content()
            .iter()
            .filter_map(|b| match b["type"].as_str() {
                Some("text") => b["text"].as_str().map(|t| Event::Text(t.into())),
                Some("tool_use") => Some(Event::Tool {
                    name: b["name"].as_str().unwrap_or("").into(),
                    input: b["input"].clone(),
                }),
                _ => None,
            })
            .collect(),
        Some("user") => content()
            .iter()
            .filter(|b| b["type"] == "tool_result")
            .map(|b| Event::ToolResult(short(&b["content"])))
            .collect(),
        Some("result") => vec![Event::Done {
            cost_usd: v["total_cost_usd"].as_f64(),
            summary: v["subtype"].as_str().unwrap_or("").into(),
        }],
        _ => Vec::new(),
    }
}

/// One line of `codex exec --json`.
pub fn parse_codex(line: &str) -> Vec<Event> {
    let Ok(v) = serde_json::from_str::<Value>(line) else {
        return Vec::new();
    };
    let item = &v["item"];
    match (v["type"].as_str(), item["type"].as_str()) {
        (Some("item.completed"), Some("agent_message")) => item["text"]
            .as_str()
            .map(|t| Event::Text(t.into()))
            .into_iter()
            .collect(),
        (Some("item.started"), Some("mcp_tool_call")) => vec![Event::Tool {
            name: item["tool"].as_str().unwrap_or("").into(),
            input: item["arguments"].clone(),
        }],
        (Some("item.completed"), Some("mcp_tool_call")) => {
            vec![Event::ToolResult(short(&item["result"]))]
        }
        (Some("turn.completed"), _) => vec![Event::Done {
            cost_usd: None,
            summary: "turn.completed".into(),
        }],
        (Some("turn.failed") | Some("error"), _) => vec![Event::Error(
            v["error"]["message"]
                .as_str()
                .or(v["message"].as_str())
                .unwrap_or("erreur")
                .into(),
        )],
        _ => Vec::new(),
    }
}

fn short(v: &Value) -> String {
    let text = match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    };
    text.chars().take(2000).collect()
}

/// A running CLI: events as they come, `stop` kills it.
pub struct Run {
    pub rx: Receiver<Event>,
    child: Arc<Mutex<Child>>,
}

impl Run {
    pub fn start(
        l: &Launch,
        kind: ProviderKind,
        wake: Arc<dyn Fn() + Send + Sync>,
    ) -> Result<Self, String> {
        let mut child = command(&l.program)
            .args(&l.args)
            .current_dir(&l.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("{} : {e}", l.program.display()))?;
        if let Some(mut stdin) = child.stdin.take() {
            let prompt = l.stdin.clone();
            std::thread::spawn(move || {
                let _ = stdin.write_all(prompt.as_bytes());
            });
        }
        let (tx, rx) = mpsc::channel();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let child = Arc::new(Mutex::new(child));
        {
            let (tx, wake, child) = (tx.clone(), Arc::clone(&wake), Arc::clone(&child));
            std::thread::spawn(move || {
                for line in stdout.into_iter().flat_map(|s| BufReader::new(s).lines()) {
                    let Ok(line) = line else { break };
                    let events = match kind {
                        ProviderKind::ClaudeCode => parse_claude(&line),
                        _ => parse_codex(&line),
                    };
                    for e in events {
                        let _ = tx.send(e);
                    }
                    wake();
                }
                // try_wait, so that `stop` can still take the lock and kill.
                let code = loop {
                    match child.lock().map(|mut c| c.try_wait()) {
                        Ok(Ok(Some(status))) => break status.code(),
                        Ok(Ok(None)) => std::thread::sleep(Duration::from_millis(20)),
                        _ => break None,
                    }
                };
                let _ = tx.send(Event::Exit(code));
                wake();
            });
        }
        std::thread::spawn(move || {
            let mut text = String::new();
            if let Some(s) = stderr {
                let _ = BufReader::new(s).take(64 << 10).read_to_string(&mut text);
            }
            let text = crate::logging::mask_secrets(text.trim());
            if !text.is_empty() {
                let _ = tx.send(Event::Error(text));
                wake();
            }
        });
        Ok(Self { rx, child })
    }

    pub fn stop(&self) {
        if let Ok(mut c) = self.child.lock() {
            let _ = c.kill();
        }
    }
}
