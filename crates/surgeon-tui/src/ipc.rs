//! The agent bridge: a local IPC endpoint (Unix domain socket on Linux/macOS,
//! Windows named pipe) that AI coding CLIs (Claude Code, Codex, Aider) write
//! JSON commands to, one JSON document per line:
//!
//! ```text
//! {"cmd":"ping"}
//! {"cmd":"highlight_layer","layer":14}
//! {"cmd":"select_tensor","name":"model.layers.14.mlp.down_proj.weight"}
//! {"cmd":"preview_patch","tensor":"model.layers.14.mlp.down_proj.weight","alpha":0.5}
//! {"cmd":"clear_preview"}
//! {"cmd":"set_filter","role":"gate"}          // role omitted or null = no filter
//! {"cmd":"snapshot"}                          // -> {"ok":true,"snapshot":{...}}
//! {"cmd":"graph_patch","patch":{"type":"snapshot","report":{...}}}  // -> {"ok":true,"patches":1}
//! {"cmd":"graph_patch","patch":{"type":"update",...}}               // -> {"ok":true,"patches":2}
//! {"cmd":"graph_stream"}                      // -> {"ok":true,"patches":[...]}
//! ```
//!
//! Every command gets exactly one reply line: `{"ok":true,...}` or
//! `{"ok":false,"error":"..."}`. Unknown commands and malformed payloads are
//! errors, never crashes; a closed connection is tolerated.
//!
//! `graph_patch` and `graph_stream` relay the `GraphPatch` lines of
//! `surgeon graph --format ndjson` (docs/GRAPH.md) from a producer (an
//! orchestrator) to consumers (a GUI): `graph_stream` returns the last snapshot
//! and the updates since, to apply in order.

use std::fmt;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use thiserror::Error;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, BufWriter};
use tokio::sync::{mpsc, oneshot};

/// Reply sent back to the agent, for the app to fill the snapshot with.
pub struct IpcRequest {
    pub command: IpcCommand,
    /// Present when [`IpcCommand::needs_reply`]: the snapshot document for `snapshot`,
    /// the whole reply line for the graph commands.
    pub reply: Option<oneshot::Sender<Value>>,
}

/// The wire protocol.
#[derive(Debug, Clone, PartialEq)]
pub enum IpcCommand {
    Ping,
    HighlightLayer {
        layer: u64,
    },
    SelectTensor {
        name: String,
    },
    PreviewPatch {
        tensor: String,
        alpha: f64,
    },
    ClearPreview,
    SetFilter {
        role: Option<String>,
    },
    Snapshot,
    /// One `GraphPatch` line, checked for its `type` only (the TUI does not read graphs).
    GraphPatch {
        patch: Value,
    },
    GraphStream,
}

impl IpcCommand {
    pub fn name(&self) -> &'static str {
        match self {
            IpcCommand::Ping => "ping",
            IpcCommand::HighlightLayer { .. } => "highlight_layer",
            IpcCommand::SelectTensor { .. } => "select_tensor",
            IpcCommand::PreviewPatch { .. } => "preview_patch",
            IpcCommand::ClearPreview => "clear_preview",
            IpcCommand::SetFilter { .. } => "set_filter",
            IpcCommand::Snapshot => "snapshot",
            IpcCommand::GraphPatch { .. } => "graph_patch",
            IpcCommand::GraphStream => "graph_stream",
        }
    }

    /// The app answers these itself (the bridge acks the others).
    pub fn needs_reply(&self) -> bool {
        matches!(
            self,
            IpcCommand::Snapshot | IpcCommand::GraphPatch { .. } | IpcCommand::GraphStream
        )
    }

    /// Parses one JSON document; a typed error string is returned to the agent.
    pub fn from_value(value: &Value) -> Result<IpcCommand, String> {
        let parsed: CommandWire = serde_json::from_value(value.clone())
            .map_err(|e| format!("commande invalide : {e}"))?;
        match parsed {
            CommandWire::Ping => Ok(IpcCommand::Ping),
            CommandWire::HighlightLayer { layer } => Ok(IpcCommand::HighlightLayer { layer }),
            CommandWire::SelectTensor { name } => Ok(IpcCommand::SelectTensor { name }),
            CommandWire::PreviewPatch { tensor, alpha } => {
                if !alpha.is_finite() {
                    return Err("alpha doit être fini".into());
                }
                Ok(IpcCommand::PreviewPatch { tensor, alpha })
            }
            CommandWire::ClearPreview => Ok(IpcCommand::ClearPreview),
            CommandWire::SetFilter { role } => Ok(IpcCommand::SetFilter { role }),
            CommandWire::Snapshot => Ok(IpcCommand::Snapshot),
            CommandWire::GraphPatch { patch } => match patch["type"].as_str() {
                Some("snapshot") if patch["report"].is_object() => {
                    Ok(IpcCommand::GraphPatch { patch })
                }
                Some("update") => Ok(IpcCommand::GraphPatch { patch }),
                _ => Err("graph_patch : patch de type snapshot (avec report) ou update".into()),
            },
            CommandWire::GraphStream => Ok(IpcCommand::GraphStream),
        }
    }

    /// The ack reply for commands that need no payload back.
    pub fn ack(&self) -> Value {
        match self {
            IpcCommand::Ping => json!({"ok": true, "pong": true}),
            _ => json!({"ok": true}),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(tag = "cmd", rename_all = "snake_case")]
enum CommandWire {
    Ping,
    HighlightLayer { layer: u64 },
    SelectTensor { name: String },
    PreviewPatch { tensor: String, alpha: f64 },
    ClearPreview,
    SetFilter { role: Option<String> },
    Snapshot,
    GraphPatch { patch: Value },
    GraphStream,
}

/// The graph stream relayed from `graph_patch` to `graph_stream`: the last snapshot and the
/// updates received since.
// shortcut: updates pile up until the next snapshot; bound them if a producer streams for hours.
#[derive(Debug, Default)]
pub struct GraphRelay {
    patches: Vec<Value>,
}

impl GraphRelay {
    /// Takes one patch and returns the reply line. An update before any snapshot is refused:
    /// a consumer could not apply it.
    pub fn push(&mut self, patch: Value) -> Value {
        match patch["type"].as_str() {
            Some("snapshot") => self.patches = vec![patch],
            Some("update") if !self.patches.is_empty() => self.patches.push(patch),
            _ => {
                return json!({"ok": false, "error": "graph_patch : update sans snapshot préalable"})
            }
        }
        json!({"ok": true, "patches": self.patches.len()})
    }

    /// The reply to `graph_stream`.
    pub fn stream(&self) -> Value {
        json!({"ok": true, "patches": self.patches})
    }
}

/// What the TUI answers to `snapshot`, describing the current selection.
#[derive(Debug, Serialize, PartialEq)]
pub struct Snapshot {
    pub tensor: String,
    pub role: String,
    pub layer: Option<u32>,
    pub shape: Vec<u64>,
    pub dtype: String,
    pub byte_len: u64,
    pub filter: Option<String>,
}

impl Snapshot {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).unwrap_or_else(|_| json!({"ok": true, "snapshot": null}))
    }
}

/// Which endpoint the bridge listens on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcTarget {
    /// Unix domain socket path.
    Unix(PathBuf),
    /// Windows named pipe: either the full `\\.\pipe\name` or the bare name.
    WindowsPipe(String),
}

impl IpcTarget {
    /// The default endpoint of the platform: `/tmp/surgeon-tui.sock` (via the
    /// system temp dir) or `\\.\pipe\surgeon-tui`.
    pub fn default_target() -> IpcTarget {
        if cfg!(windows) {
            IpcTarget::WindowsPipe("surgeon-tui".into())
        } else {
            IpcTarget::Unix(std::env::temp_dir().join("surgeon-tui.sock"))
        }
    }

    /// Parses the `--ipc-socket` value; a bare name on Windows becomes a pipe,
    /// anything starting with `\\.\pipe\` or ending in `.sock` keeps its shape.
    pub fn parse(arg: &str) -> IpcTarget {
        if arg.starts_with(r"\\.\pipe\") {
            IpcTarget::WindowsPipe(arg.trim_start_matches(r"\\.\pipe\").to_owned())
        } else if arg.contains('\\') || arg.contains('/') {
            IpcTarget::Unix(PathBuf::from(arg))
        } else if cfg!(windows) {
            IpcTarget::WindowsPipe(arg.to_owned())
        } else {
            IpcTarget::Unix(PathBuf::from(arg))
        }
    }

    /// Human-readable endpoint for the UI banner and the logs.
    pub fn display(&self) -> String {
        match self {
            IpcTarget::Unix(path) => path.display().to_string(),
            IpcTarget::WindowsPipe(name) => format!(r"\\.\pipe\{name}"),
        }
    }
}

impl fmt::Display for IpcTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display())
    }
}

#[derive(Debug, Error)]
pub enum BridgeError {
    #[error("io : {0}")]
    Io(#[from] std::io::Error),
    #[error("json : {0}")]
    Json(#[from] serde_json::Error),
}

/// Serves agents until the process ends. Every parsed command is forwarded to
/// the app through `tx`; `snapshot` waits for the app's answer.
#[cfg(unix)]
pub async fn serve(
    target: IpcTarget,
    tx: mpsc::UnboundedSender<IpcRequest>,
) -> Result<(), BridgeError> {
    match target {
        IpcTarget::Unix(path) => serve_unix(path, tx).await,
        IpcTarget::WindowsPipe(name) => serve_pipe(&name, tx).await,
    }
}

/// Serves agents until the process ends (Windows: named pipes only).
#[cfg(windows)]
pub async fn serve(
    target: IpcTarget,
    tx: mpsc::UnboundedSender<IpcRequest>,
) -> Result<(), BridgeError> {
    match target {
        IpcTarget::Unix(path) => Err(std::io::Error::other(format!(
            "socket unix indisponible sur Windows : {}",
            path.display()
        ))
        .into()),
        IpcTarget::WindowsPipe(name) => serve_pipe(&name, tx).await,
    }
}

#[cfg(unix)]
async fn serve_unix(
    path: PathBuf,
    tx: mpsc::UnboundedSender<IpcRequest>,
) -> Result<(), BridgeError> {
    // A stale socket file from a previous crash must not block the bind.
    let _ = std::fs::remove_file(&path);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let listener = tokio::net::UnixListener::bind(&path)?;
    loop {
        let (stream, _peer) = match listener.accept().await {
            Ok(pair) => pair,
            Err(_) => continue,
        };
        tokio::spawn(handle_connection(stream, tx.clone()));
    }
}

#[cfg(windows)]
async fn serve_pipe(name: &str, tx: mpsc::UnboundedSender<IpcRequest>) -> Result<(), BridgeError> {
    use tokio::net::windows::named_pipe::ServerOptions;
    let full = format!(r"\\.\pipe\{name}");
    let mut server = ServerOptions::new()
        .first_pipe_instance(true)
        .create(&full)?;
    loop {
        server.connect().await?;
        let connected = server;
        tokio::spawn(handle_connection(connected, tx.clone()));
        server = ServerOptions::new().create(&full)?;
    }
}

#[cfg(not(windows))]
async fn serve_pipe(
    _name: &str,
    _tx: mpsc::UnboundedSender<IpcRequest>,
) -> Result<(), BridgeError> {
    Err(std::io::Error::other("named pipes : Windows uniquement").into())
}

/// One connection: any number of JSON lines, one reply per line.
async fn handle_connection<S>(stream: S, tx: mpsc::UnboundedSender<IpcRequest>)
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let (read, write) = tokio::io::split(stream);
    let mut lines = BufReader::new(read).lines();
    let mut writer = BufWriter::new(write);
    while let Ok(Some(line)) = lines.next_line().await {
        let reply = process_line(&line, &tx).await;
        let mut text = reply.to_string();
        text.push('\n');
        if writer.write_all(text.as_bytes()).await.is_err() {
            break;
        }
        if writer.flush().await.is_err() {
            break;
        }
    }
}

async fn process_line(line: &str, tx: &mpsc::UnboundedSender<IpcRequest>) -> Value {
    let parsed: Result<Value, _> = serde_json::from_str(line.trim());
    let value = match parsed {
        Ok(value) => value,
        Err(e) => return json!({"ok": false, "error": format!("json invalide : {e}")}),
    };
    match IpcCommand::from_value(&value) {
        Ok(command) => {
            if command.needs_reply() {
                let snapshot = matches!(command, IpcCommand::Snapshot);
                let (sent, received) = oneshot::channel();
                if tx
                    .send(IpcRequest {
                        command,
                        reply: Some(sent),
                    })
                    .is_err()
                {
                    return json!({"ok": false, "error": "interface hors ligne"});
                }
                match tokio::time::timeout(std::time::Duration::from_secs(5), received).await {
                    Ok(Ok(answer)) if snapshot => json!({"ok": true, "snapshot": answer}),
                    Ok(Ok(reply)) => reply,
                    _ => json!({"ok": false, "error": "réponse de l'interface expirée"}),
                }
            } else {
                if tx
                    .send(IpcRequest {
                        command,
                        reply: None,
                    })
                    .is_err()
                {
                    return json!({"ok": false, "error": "interface hors ligne"});
                }
                value_ack(&value)
            }
        }
        Err(error) => json!({"ok": false, "error": error}),
    }
}

/// The ack of a command the bridge recognized and forwarded.
fn value_ack(value: &Value) -> Value {
    match IpcCommand::from_value(value) {
        Ok(command) => command.ack(),
        Err(error) => json!({"ok": false, "error": error}),
    }
}

/// Client helper: sends commands and collects the replies (tests, examples).
#[cfg(unix)]
pub async fn send_commands(
    target: &IpcTarget,
    commands: &[Value],
) -> Result<Vec<Value>, BridgeError> {
    match target {
        IpcTarget::Unix(path) => {
            let stream = tokio::net::UnixStream::connect(path).await?;
            exchange(stream, commands).await
        }
        IpcTarget::WindowsPipe(_) => {
            Err(std::io::Error::other("named pipes : Windows uniquement").into())
        }
    }
}

/// Client helper: sends commands and collects the replies (tests, examples).
#[cfg(windows)]
pub async fn send_commands(
    target: &IpcTarget,
    commands: &[Value],
) -> Result<Vec<Value>, BridgeError> {
    match target {
        IpcTarget::Unix(_) => {
            Err(std::io::Error::other("sockets unix : Linux/macOS uniquement").into())
        }
        IpcTarget::WindowsPipe(name) => {
            let stream = tokio::net::windows::named_pipe::ClientOptions::new()
                .open(format!(r"\\.\pipe\{name}"))?;
            exchange(stream, commands).await
        }
    }
}

async fn exchange<S>(stream: S, commands: &[Value]) -> Result<Vec<Value>, BridgeError>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    // The writer stays alive while reading: half-closing a named pipe tears
    // the whole connection down on Windows. One reply is expected per command,
    // so no EOF is needed.
    let (source, mut sink) = tokio::io::split(stream);
    for command in commands {
        let mut line = serde_json::to_string(command)?;
        line.push('\n');
        sink.write_all(line.as_bytes()).await?;
    }
    sink.flush().await?;
    let mut reader = BufReader::new(source);
    let mut replies = Vec::new();
    let mut buf = String::new();
    for _ in commands {
        buf.clear();
        if reader.read_line(&mut buf).await? == 0 {
            break;
        }
        replies.push(serde_json::from_str(buf.trim())?);
    }
    Ok(replies)
}
