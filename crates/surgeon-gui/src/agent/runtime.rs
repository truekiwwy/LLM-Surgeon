//! The official CLI runtimes, used as installed and signed in by the user (the way to use a
//! subscription account): Claude Code (`claude -p`), Codex (`codex exec`), OpenCode
//! (`opencode run`) and Antigravity (`agy -p`), started non-interactively with `surgeon mcp`
//! as their engine server. Their own loop and approvals are theirs, not the studio's.
//!
//! - Claude Code and Codex get `surgeon` as their only tool server and an explicit allow-list;
//!   their working files go through its `surgeon_files` tool, confined to `work/`.
//! - OpenCode and Antigravity keep their own tools (terminal, files) so as not to limit the
//!   model: OpenCode inside the project folder (`external_directory` denied), Antigravity in an
//!   isolated profile that only knows `surgeon` (the user's global configuration is never read
//!   nor written). For both, the source model is held by the kernel lock of
//!   [`crate::studio::lock`], and the engine server never reaches it in "Modify" mode.
//!
//! The studio reads no credential, except for Antigravity: its sign-in files
//! (`~/.gemini/oauth_creds.json`, `google_accounts.json`) are copied into the isolated
//! profile for the run and removed with it, on the user's explicit request. Claude Code never
//! inherits `ANTHROPIC_API_KEY`, so that it stays on the signed-in claude.ai account.
//!
//! Options checked against `claude --help` (2.1.x), `codex exec --help` (0.166),
//! `opencode run --help` (2.2.7) and `agy --help` (1.4.4): a binary whose help lacks one is
//! reported and not used.

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
const OPENCODE_OPTIONS: [&str; 4] = ["--format", "--auto", "--archived", "--standalone"];
const AGY_OPTIONS: [&str; 4] = ["--print", "--output-format", "--input-format", "--model"];
/// The sign-in files of Antigravity, in `~/.gemini`, copied into its isolated profile.
const AGY_SIGN_IN: [&str; 2] = ["oauth_creds.json", "google_accounts.json"];
/// The engine tools an "Ask" runtime may call (its server also runs `--read-only`).
const READ_TOOLS: [&str; 7] = [
    "surgeon_check",
    "surgeon_inspect",
    "surgeon_graph",
    "surgeon_bench",
    "surgeon_backends",
    "surgeon_autotune",
    "surgeon_files",
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
        ProviderKind::Codex => "codex",
        ProviderKind::OpenCode => "opencode",
        ProviderKind::Antigravity => "agy",
        ProviderKind::Anthropic | ProviderKind::OpenAi | ProviderKind::Compatible => "",
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
    capture(bin, &["--version"], &[], Duration::from_secs(15))
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
        ProviderKind::OpenCode => (&["run", "--help"], &OPENCODE_OPTIONS),
        ProviderKind::Antigravity => (&["--help"], &AGY_OPTIONS),
        _ => (&["exec", "--help"], &CODEX_OPTIONS),
    };
    let help = capture(&bin, help_args, &[], Duration::from_secs(15)).unwrap_or_default();
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

/// The process of a CLI runtime, without what would take over from its signed-in account:
/// `claude -p` prefers `ANTHROPIC_API_KEY` to the claude.ai session whenever it is set.
pub fn cli_command(kind: ProviderKind, program: &Path) -> Command {
    let mut c = command(program);
    if kind == ProviderKind::ClaudeCode {
        c.env_remove("ANTHROPIC_API_KEY");
    }
    c
}

/// The CLI's own sign-in check: it spends nothing (a status, or the model list) and the
/// studio never reads the credentials itself.
fn status_args(kind: ProviderKind) -> &'static [&'static str] {
    match kind {
        ProviderKind::ClaudeCode => &["auth", "status"],
        ProviderKind::OpenCode => &["models", "--standalone"],
        ProviderKind::Antigravity => &["models"],
        _ => &["login", "status"],
    }
}

/// What the user types to sign the CLI in.
pub fn login_command(kind: ProviderKind) -> &'static str {
    match kind {
        ProviderKind::ClaudeCode => "claude auth login",
        ProviderKind::OpenCode => "opencode auth login",
        ProviderKind::Antigravity => "agy",
        _ => "codex login",
    }
}

/// Whether the CLI found on `path` reports a signed-in account; the error says what failed.
/// Antigravity is checked inside its isolated profile, which is removed afterwards. `Ok`
/// carries the models the check listed (OpenCode, Antigravity), empty for the others.
pub fn signed_in(kind: ProviderKind, path: Option<&OsStr>) -> Result<Vec<String>, String> {
    let bin = find_on_path(binary_name(kind), path)
        .ok_or_else(|| format!("{} introuvable dans le PATH", kind.name()))?;
    let agy = match kind {
        ProviderKind::Antigravity => {
            Some(AgyHome::system().ok_or("dossier personnel introuvable")?)
        }
        _ => None,
    };
    let env = match &agy {
        Some(home) => home.prepare(None, None)?,
        None => Vec::new(),
    };
    let checked = run_short(
        cli_command(kind, &bin),
        status_args(kind),
        &env,
        Duration::from_secs(60),
    );
    if let Some(home) = &agy {
        home.clear();
    }
    match checked {
        Some((true, text)) => Ok(match status_args(kind).contains(&"models") {
            true => model_lines(&text),
            false => Vec::new(),
        }),
        Some((false, text)) => Err(crate::logging::mask_secrets(
            text.lines()
                .map(str::trim)
                .find(|l| !l.is_empty())
                .unwrap_or(""),
        )),
        None => Err(format!("{} ne répond pas", kind.name())),
    }
}

/// The ids of a `models` listing: one per line (`provider/model` for OpenCode). Lines with
/// spaces (headers, notices) are not ids.
pub fn model_lines(text: &str) -> Vec<String> {
    let mut ids: Vec<String> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.contains(char::is_whitespace) && !l.ends_with(':'))
        .map(str::to_owned)
        .collect();
    ids.dedup();
    ids
}

type Env = Vec<(OsString, OsString)>;

/// stdout (then stderr) of a short command, or None past `timeout`.
fn capture(
    program: &Path,
    args: &[&str],
    env: &[(OsString, OsString)],
    timeout: Duration,
) -> Option<String> {
    run_short(command(program), args, env, timeout).map(|(_, text)| text)
}

/// Whether a short command succeeded, with its stdout then stderr; None past `timeout`.
fn run_short(
    mut cmd: Command,
    args: &[&str],
    env: &[(OsString, OsString)],
    timeout: Duration,
) -> Option<(bool, String)> {
    let mut child = cmd
        .args(args)
        .envs(env.iter().map(|(k, v)| (k, v)))
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
    let success = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status.success(),
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(20)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    // The two readers end in any order: stdout first, whichever came back first.
    let mut parts: Vec<(usize, String)> = rx.iter().collect();
    parts.sort_by_key(|(k, _)| *k);
    let texts: Vec<String> = parts.into_iter().map(|(_, text)| text).collect();
    Some((success, texts.join("\n")))
}

/// The program a runtime launches as its engine MCP server: this app, which serves
/// `mcp …` itself (`runner::serve_mcp`), so the engine is always the window's own version.
pub fn engine_server() -> Result<PathBuf, String> {
    std::env::current_exe().map_err(|e| format!("chemin de l'application introuvable : {e}"))
}

/// Everything a run needs, built by the code from the mode (never by the agent).
#[derive(Debug, Clone, PartialEq)]
pub struct Launch {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub cwd: PathBuf,
    /// The prompt goes on stdin: no free text on a command line.
    pub stdin: String,
    /// Set for this run only (OpenCode's inline configuration, Antigravity's profile).
    pub env: Env,
}

/// The isolated profile Antigravity runs in: its own `~/.gemini` that knows `surgeon` as
/// its only MCP server, with the user's sign-in copied in. The user's own configuration
/// (global MCP servers, plugins, permission grants) is never read nor written.
#[derive(Debug, Clone, PartialEq)]
pub struct AgyHome {
    /// `%LOCALAPPDATA%\weight-surgeon\profiles\agy`, the `USERPROFILE` of the run.
    pub profile: PathBuf,
    /// The user's home, where `.gemini` holds the sign-in.
    pub user: PathBuf,
}

impl AgyHome {
    pub fn system() -> Option<Self> {
        Some(Self {
            profile: crate::paths::data_dir().join("profiles").join("agy"),
            user: crate::paths::home_dir()?,
        })
    }

    /// Recreates the profile from scratch: the sign-in files, `server` as the only MCP
    /// server, the tools allowed without asking on `project`. The environment of the run.
    pub fn prepare(&self, server: Option<&Value>, project: Option<&Path>) -> Result<Env, String> {
        let fail = |p: &Path, e: std::io::Error| format!("{} : {e}", p.display());
        let gemini = self.profile.join(".gemini");
        self.clear();
        for dir in [gemini.join("config"), gemini.join("antigravity-cli")] {
            std::fs::create_dir_all(&dir).map_err(|e| fail(&dir, e))?;
        }
        for name in AGY_SIGN_IN {
            let from = self.user.join(".gemini").join(name);
            if !from.is_file() {
                return Err(format!(
                    "Antigravity n'est pas connecté ({} absent) : lancez « agy » une fois",
                    from.display()
                ));
            }
            std::fs::copy(&from, gemini.join(name)).map_err(|e| fail(&from, e))?;
        }
        let servers = server.map_or(json!({}), |s| json!({ "surgeon": s }));
        let settings = json!({
            "toolPermission": "always-proceed",
            "trustedWorkspaces": project.map(|p| vec![p.display().to_string()]).unwrap_or_default(),
        });
        for (file, value) in [
            (
                gemini.join("config").join("mcp_config.json"),
                json!({ "mcpServers": servers }),
            ),
            (
                gemini.join("antigravity-cli").join("settings.json"),
                settings,
            ),
        ] {
            std::fs::write(&file, value.to_string()).map_err(|e| fail(&file, e))?;
        }
        let home = self.profile.clone().into_os_string();
        Ok(vec![
            ("USERPROFILE".into(), home.clone()),
            ("HOME".into(), home),
        ])
    }

    /// Removes the profile, and the copy of the sign-in with it.
    pub fn clear(&self) {
        let _ = std::fs::remove_dir_all(&self.profile);
    }
}

/// The `surgeon mcp` arguments for a mode: the project is read and written; `read_roots` (the
/// source model, the configured models folder) are read only, in every mode. The server
/// refuses any write there whatever the mode.
pub fn mcp_args(mode: AgentMode, project: &Path, read_roots: &[PathBuf]) -> Vec<String> {
    let mut args = vec![
        "mcp".to_owned(),
        "--allow-dir".into(),
        project.display().to_string(),
    ];
    for dir in read_roots {
        args.extend(["--allow-read".into(), dir.display().to_string()]);
    }
    if mode == AgentMode::Ask {
        args.push("--read-only".into());
    }
    args
}

/// The tools the runtime may call without asking: the whole MCP list, interface tools
/// (`record_finding`...) included since they only write under `chat/`.
pub fn allowed_tools(mode: AgentMode) -> Vec<String> {
    let all = weight_surgeon::mcp::tool_definitions();
    all.as_array()
        .into_iter()
        .flatten()
        .filter_map(|t| t["name"].as_str().map(str::to_owned))
        .filter(|n| match mode {
            AgentMode::Ask => {
                READ_TOOLS.contains(&n.as_str()) || super::tools::UI_TOOLS.contains(&n.as_str())
            }
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
    /// Folders the engine may read but never write.
    pub read_roots: &'a [PathBuf],
    pub model: Option<&'a str>,
    pub prompt: &'a str,
    /// Where the MCP configuration file of a Claude Code run is written.
    pub scratch: &'a Path,
    /// The isolated profile of an Antigravity run.
    pub agy: Option<&'a AgyHome>,
}

pub fn launch(input: &LaunchInput) -> Result<Launch, String> {
    let server = json!({
        "command": input.surgeon,
        "args": mcp_args(input.mode, input.project, input.read_roots),
    });
    let tools = allowed_tools(input.mode);
    let mut args: Vec<OsString> = Vec::new();
    let mut env = Env::new();
    let mut stdin = input.prompt.to_owned();
    match input.detected.kind {
        ProviderKind::OpenCode => {
            let mut command = vec![input.surgeon.display().to_string()];
            command.extend(mcp_args(input.mode, input.project, input.read_roots));
            let config = json!({
                "mcp": { "surgeon": { "type": "local", "command": command, "enabled": true } },
                // Its own tools stay, inside the project folder only.
                "permission": { "external_directory": "deny" },
            });
            // A private server: the background service would not read this configuration.
            for a in [
                "run",
                "--standalone",
                "--auto",
                "--format",
                "json",
                "--archived",
            ] {
                args.push(a.into());
            }
            if let Some(m) = input.model {
                args.extend(["--model".into(), m.into()]);
            }
            env = vec![
                ("OPENCODE_CONFIG_CONTENT".into(), config.to_string().into()),
                ("OPENCODE_DISABLE_PROJECT_CONFIG".into(), "1".into()),
                ("OPENCODE_DISABLE_AUTOUPDATE".into(), "1".into()),
            ];
        }
        ProviderKind::Antigravity => {
            let home = input.agy.ok_or("profil Antigravity absent")?;
            env = home.prepare(Some(&server), Some(input.project))?;
            for a in [
                "--input-format",
                "stream-json",
                "--output-format",
                "stream-json",
            ] {
                args.push(a.into());
            }
            if let Some(m) = input.model {
                args.extend(["--model".into(), m.into()]);
            }
            // The prompt is one stream message on stdin; `-p` takes a value, left empty.
            args.push("-p=".into());
            stdin = format!(
                "{}\n",
                json!({"event": "user", "message": {"role": "user", "content": input.prompt}})
            );
        }
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
            let mcp = mcp_args(input.mode, input.project, input.read_roots);
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
        stdin,
        env,
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

/// One line of `opencode run --format json` (2.2.7): `text` parts, `tool_use` parts (its
/// MCP tools are called from its `execute` tool), a `step_finish` with reason `stop` at the
/// end of the turn.
pub fn parse_opencode(line: &str) -> Vec<Event> {
    let Ok(v) = serde_json::from_str::<Value>(line) else {
        return Vec::new();
    };
    let part = &v["part"];
    match v["type"].as_str() {
        Some("text") => part["text"]
            .as_str()
            .map(|t| Event::Text(t.into()))
            .into_iter()
            .collect(),
        Some("tool_use") => {
            let state = &part["state"];
            let mut events = vec![Event::Tool {
                name: part["tool"].as_str().unwrap_or("").into(),
                input: state["input"].clone(),
            }];
            match state["status"].as_str() {
                Some("completed") => events.push(Event::ToolResult(short(&state["output"]))),
                Some("error") => events.push(Event::Error(short(&state["error"]))),
                _ => {}
            }
            events
        }
        Some("step_finish") if part["reason"] == "stop" => vec![Event::Done {
            cost_usd: None,
            summary: "stop".into(),
        }],
        Some("error") => vec![Event::Error(short(&v["error"]))],
        _ => Vec::new(),
    }
}

/// One line of `agy --output-format stream-json` (1.4.4): `step_update` events name the
/// steps, `result` carries the whole answer, its status and why the turn ended.
pub fn parse_agy(line: &str) -> Vec<Event> {
    let Ok(v) = serde_json::from_str::<Value>(line) else {
        return Vec::new();
    };
    match v["event"].as_str() {
        Some("step_update") => {
            let step = &v["step_update"];
            // shortcut: only the step's type is shown (its arguments are not in the observed
            // events); read them once a quota-free run shows a tool step's fields.
            let kind = step["step_type"].as_str().unwrap_or("");
            match (step["state"].as_str(), kind) {
                (_, "user_input" | "planner_response" | "error_message" | "") => Vec::new(),
                (Some("DONE"), tool) => vec![Event::Tool {
                    name: tool.into(),
                    input: Value::Null,
                }],
                _ => Vec::new(),
            }
        }
        Some("result") => {
            let result = &v["result"];
            let mut events = Vec::new();
            if let Some(text) = result["response"].as_str().filter(|t| !t.is_empty()) {
                events.push(Event::Text(text.into()));
            }
            match result["status"].as_str() {
                Some("ERROR") => events.push(Event::Error(
                    result["error"].as_str().unwrap_or("erreur").into(),
                )),
                _ => events.push(Event::Done {
                    cost_usd: None,
                    summary: result["finished_reason"].as_str().unwrap_or("").into(),
                }),
            }
            events
        }
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

/// The process of a run, its pipes still to set.
pub fn run_command(l: &Launch, kind: ProviderKind) -> Command {
    let mut c = cli_command(kind, &l.program);
    c.args(&l.args)
        .envs(l.env.iter().map(|(k, v)| (k, v)))
        .current_dir(&l.cwd);
    c
}

/// A running CLI: events as they come, `stop` (or the drop) kills it with what it started,
/// the `surgeon-gui mcp` servers first.
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
        let mut cmd = run_command(l, kind);
        // A group of its own: `stop` takes the MCP servers it starts with it (kill_tree).
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }
        let mut child = cmd
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
                        ProviderKind::OpenCode => parse_opencode(&line),
                        ProviderKind::Antigravity => parse_agy(&line),
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

    #[cfg(test)]
    pub fn pid(&self) -> u32 {
        self.child.lock().unwrap_or_else(|e| e.into_inner()).id()
    }

    /// Kills the CLI and its process tree. Killing the CLI alone left its `surgeon-gui mcp`
    /// servers running on Windows, a long benchmark with them.
    pub fn stop(&self) {
        let mut child = self.child.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(e) = weight_surgeon::process::kill_tree(&mut child) {
            crate::logging::warn(&format!("arrêt du runtime : {e}"));
        }
    }
}

impl Drop for Run {
    /// A run dropped while its CLI still works (project closed, window closed) stops it.
    fn drop(&mut self) {
        self.stop();
    }
}
