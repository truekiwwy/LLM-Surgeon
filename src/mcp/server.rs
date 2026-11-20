//! The stdio loop: a reader thread frames stdin into messages (and handles cancellations
//! immediately), the calling thread executes requests one at a time and is the only one
//! that writes, through the `ProtocolWriter`.

use std::io::{self, BufRead, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::protocol::{
    classify, error, negotiate, notification, result, Incoming, ProtocolWriter, INTERNAL_ERROR,
    INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR, SERVER_NOT_INITIALIZED,
};
use super::sandbox::Sandbox;
use super::tools::{self, ToolContext};
use super::tools_session;
use crate::app::{exit, panic_message, AppDeps, ProgressEvent, ProgressSink};
use crate::cli::McpArgs;
use crate::extract::PromptSet;

/// A larger incoming line is answered with -32600 and skipped up to the next newline.
pub const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

const INSTRUCTIONS: &str = "surgeon est l'extension MCP du Surgery Engine : un moteur de \
chirurgie des poids de modèles de langage locaux (SafeTensors, GGUF), appelable par un agent \
avec des outils typés. Outils : surgeon_check (budget RAM/VRAM), surgeon_inspect (couches et \
matrices MLP), surgeon_apply (applique un patch EN PLACE ; faire d'abord dry_run: true), \
surgeon_restore (remet le modèle octet pour octet depuis la sauvegarde ou l'origine, SHA-256 \
vérifié : le rollback quand surgeon_bench montre une régression), \
surgeon_compose (une direction faite de plusieurs vecteurs, orthogonale aux directions de \
capacité à préserver, et le patch qui la retire), \
surgeon_eval (non-régression et export GGUF), surgeon_extract (vecteur de direction depuis \
une couche Qwen2 ; method est un identifiant du registre du moteur : mean_diff, svd, leace), \
surgeon_bench (mesure d'une chirurgie : mode dump, logits, ppl ou eval), surgeon_autotune \
(coefficients par couche ajustés par ridge, patch.json en sortie), surgeon_diff (patch \
.surgery de faible rang entre deux checkpoints), surgeon_lora (fusion ou extraction d'un \
adaptateur LoRA), surgeon_backends (moteurs d'inférence : CPU natif, GPU mistral.rs ou llama-server, hipfire). Démarche sûre : surgeon_check d'abord, travailler sur une copie du modèle, \
dry_run avant tout apply, garder la sauvegarde, mesurer avec surgeon_bench avant de conclure, \
surgeon_restore en cas de régression. \
Garde-fous : chemins limités aux dossiers autorisés, sauvegarde vérifiée imposée avant \
écriture, rollback automatique sur erreur ou annulation. Le format du patch est le schéma de \
l'argument patch de surgeon_apply.";

#[derive(Debug, Clone)]
pub struct McpConfig {
    pub sandbox: Sandbox,
    pub read_only: bool,
    pub allow_no_backup: bool,
}

impl McpConfig {
    /// No directory: the current directory, as when the client launched the server.
    pub fn new(allow_dirs: &[PathBuf], read_only: bool, allow_no_backup: bool) -> io::Result<Self> {
        let dirs = match allow_dirs.is_empty() {
            true => vec![std::env::current_dir()?],
            false => allow_dirs.to_vec(),
        };
        Ok(Self {
            sandbox: Sandbox::new(&dirs)?,
            read_only,
            allow_no_backup,
        })
    }

    /// Adds roots the tools may read but never write (`--allow-read`).
    pub fn with_read_roots(mut self, paths: &[PathBuf]) -> io::Result<Self> {
        self.sandbox = self.sandbox.with_read_roots(paths)?;
        Ok(self)
    }

    /// `--allow-dir`, else `$SURGEON_MCP_ALLOW_DIRS` split with the system separator, then
    /// the `--allow-read` roots.
    pub fn from_args(args: &McpArgs) -> io::Result<Self> {
        let dirs = match args.allow_dir.is_empty() {
            false => args.allow_dir.clone(),
            true => std::env::var_os("SURGEON_MCP_ALLOW_DIRS")
                .map(|v| {
                    std::env::split_paths(&v)
                        .filter(|p| !p.as_os_str().is_empty())
                        .collect()
                })
                .unwrap_or_default(),
        };
        Self::new(&dirs, args.read_only, args.allow_no_backup)?.with_read_roots(&args.allow_read)
    }
}

enum Event {
    Message(Value),
    ParseError,
    TooLong,
}

/// How long a tool call may take to hand control back once stdin is closed before
/// `surgeon mcp` exits anyway. The engine checks its flag between tensors, prompts and tokens:
/// a cooperative call returns well within it.
pub const EXIT_GRACE: Duration = Duration::from_secs(3);

/// What the end of the input means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Eof {
    /// A script (`serve`, tests, embedders): every message it held is answered, then the end.
    Drain,
    /// The client's shutdown (`surgeon mcp` on stdio, the MCP lifecycle): the running call is
    /// cancelled and nothing more is answered, even what was read before. With `grace`, the
    /// process exits that long after if the loop has not ended (a call that ignores the flag).
    Shutdown { grace: Option<Duration> },
}

/// Shared with the reader thread so a cancellation lands while the request runs.
#[derive(Default)]
struct CancelState {
    current: Option<Value>,
    /// Cancellations of requests that have not started yet.
    pending: Vec<Value>,
    /// The client shut down: `Some(whether a call was running)`.
    closed: Option<bool>,
    /// The loop has ended; what remains (the pipeline worker's stop) has its own grace.
    done: bool,
}

/// Serves until EOF on `reader` (0, every message read answered) or until `writer` fails
/// (1: the client is gone).
pub fn serve<R, W>(
    reader: R,
    writer: W,
    err: &mut dyn Write,
    deps: &AppDeps,
    cfg: &McpConfig,
) -> i32
where
    R: BufRead + Send + 'static,
    W: Write,
{
    serve_until(reader, writer, err, deps, cfg, Eof::Drain)
}

/// `surgeon mcp` on the process's stdin: [`serve`], except that the end of stdin is the
/// client's shutdown ([`Eof::Shutdown`]): a long benchmark does not outlive Claude Code.
pub(super) fn serve_stdio(
    out: &mut dyn Write,
    err: &mut dyn Write,
    deps: &AppDeps,
    cfg: &McpConfig,
) -> i32 {
    let stdin = io::BufReader::new(io::stdin());
    let eof = Eof::Shutdown {
        grace: Some(EXIT_GRACE),
    };
    serve_until(stdin, out, err, deps, cfg, eof)
}

fn serve_until<R, W>(
    reader: R,
    writer: W,
    err: &mut dyn Write,
    deps: &AppDeps,
    cfg: &McpConfig,
    eof: Eof,
) -> i32
where
    R: BufRead + Send + 'static,
    W: Write,
{
    let writer = ProtocolWriter::new(writer);
    let cancel = Arc::new(Mutex::new(CancelState::default()));
    let (tx, rx) = mpsc::channel();
    {
        let cancel = Arc::clone(&cancel);
        let interrupted = Arc::clone(&deps.interrupted);
        // Detached on purpose: after a write failure it may stay blocked on stdin.
        std::thread::spawn(move || read_loop(reader, tx, cancel, interrupted, eof));
    }
    // Declared before the session: dropped after it, it stops the pipeline worker.
    let pipeline = super::tools_pipeline::PipelineJobs::default();
    let mut session = Session {
        writer: &writer,
        deps,
        pipeline: &pipeline,
        cfg,
        cancel: &cancel,
        err,
        initialized: false,
    };
    let code = loop {
        let Ok(event) = rx.recv() else {
            break exit::OK;
        };
        // Read before the client shut down, never run after it.
        if closed(&cancel).is_some() {
            break exit::OK;
        }
        let reply = match event {
            Event::ParseError => Some(error(&Value::Null, PARSE_ERROR, "JSON invalide")),
            Event::TooLong => Some(error(
                &Value::Null,
                INVALID_REQUEST,
                format!("message plus long que {MAX_LINE_BYTES} octets"),
            )),
            Event::Message(message) => session.handle(&message),
        };
        if let Some(reply) = reply {
            writer.send(&reply);
        }
        if writer.is_broken() {
            let _ = writeln!(session.err, "surgeon mcp : stdout fermé, arrêt du serveur");
            break exit::GENERAL;
        }
    };
    if let Ok(mut state) = cancel.lock() {
        state.done = true;
    }
    if let Some(running) = closed(&cancel) {
        let what = match running {
            true => "stdin fermé pendant un outil : annulé, arrêt du serveur",
            false => "stdin fermé, arrêt du serveur",
        };
        let _ = writeln!(session.err, "surgeon mcp : {what}");
    }
    code
}

/// `Some(whether a call was running)` once the client shut down.
fn closed(cancel: &Mutex<CancelState>) -> Option<bool> {
    cancel.lock().map_or(Some(false), |s| s.closed)
}

/// The client shut down (EOF or a broken pipe): the running call is cancelled, and every
/// call that starts from now on sees the flag raised ([`Session::start`]). With `grace`, the
/// process exits if the loop has not ended by then.
fn shut_down(cancel: &Mutex<CancelState>, interrupted: &AtomicBool, grace: Option<Duration>) {
    if let Ok(mut state) = cancel.lock() {
        state.closed = Some(state.current.is_some());
        interrupted.store(true, Ordering::SeqCst);
    }
    let Some(grace) = grace else {
        return;
    };
    let deadline = Instant::now() + grace;
    while Instant::now() < deadline {
        if cancel.lock().map_or(true, |s| s.done) {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    // The session owns the injected stderr and is stuck in the call: the real one, then out.
    let _ = writeln!(
        io::stderr(),
        "surgeon mcp : stdin fermé, l'outil en cours ne rend pas la main après {} s : arrêt forcé",
        grace.as_secs()
    );
    std::process::exit(exit::GENERAL);
}

enum LineRead {
    Line,
    TooLong,
    Eof,
}

/// One `\n`-terminated line (or the unterminated rest at EOF) into `buf`, never holding
/// more than `max` bytes: an oversized line is drained and reported.
fn read_line_capped<R: BufRead>(
    reader: &mut R,
    buf: &mut Vec<u8>,
    max: usize,
) -> io::Result<LineRead> {
    buf.clear();
    let mut overflow = false;
    loop {
        let (used, complete) = {
            let chunk = match reader.fill_buf() {
                Ok(chunk) => chunk,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            if chunk.is_empty() {
                return Ok(match (overflow, buf.is_empty()) {
                    (true, _) => LineRead::TooLong,
                    (false, true) => LineRead::Eof,
                    (false, false) => LineRead::Line,
                });
            }
            let newline = chunk.iter().position(|&b| b == b'\n');
            let part = &chunk[..newline.unwrap_or(chunk.len())];
            if !overflow && buf.len() + part.len() > max {
                overflow = true;
                buf.clear();
            }
            if !overflow {
                buf.extend_from_slice(part);
            }
            (newline.map_or(chunk.len(), |i| i + 1), newline.is_some())
        };
        reader.consume(used);
        if complete {
            return Ok(match overflow {
                true => LineRead::TooLong,
                false => LineRead::Line,
            });
        }
    }
}

fn read_loop<R: BufRead>(
    mut reader: R,
    tx: Sender<Event>,
    cancel: Arc<Mutex<CancelState>>,
    interrupted: Arc<AtomicBool>,
    eof: Eof,
) {
    let mut buf = Vec::new();
    loop {
        let event = match read_line_capped(&mut reader, &mut buf, MAX_LINE_BYTES) {
            Ok(LineRead::Eof) | Err(_) => {
                if let Eof::Shutdown { grace } = eof {
                    // The loop wakes up (the channel closes) once the cancelled call returns.
                    drop(tx);
                    shut_down(&cancel, &interrupted, grace);
                }
                return;
            }
            Ok(LineRead::TooLong) => Event::TooLong,
            Ok(LineRead::Line) => {
                // Tolerates a UTF-8 BOM, `\r\n` and blank lines.
                let line = buf.strip_prefix("\u{feff}".as_bytes()).unwrap_or(&buf);
                let line = line.trim_ascii();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_slice::<Value>(line) {
                    Ok(message) => {
                        note_cancellation(&message, &cancel, &interrupted);
                        Event::Message(message)
                    }
                    Err(_) => Event::ParseError,
                }
            }
        };
        if tx.send(event).is_err() {
            return;
        }
    }
}

/// `notifications/cancelled` for the running request raises the flag at once (the core
/// rolls back at its next check); for a queued one it is remembered.
fn note_cancellation(message: &Value, cancel: &Mutex<CancelState>, interrupted: &AtomicBool) {
    if message.get("method").and_then(Value::as_str) != Some("notifications/cancelled") {
        return;
    }
    let Some(request_id) = message.pointer("/params/requestId").cloned() else {
        return;
    };
    if let Ok(mut state) = cancel.lock() {
        match state.current.as_ref() == Some(&request_id) {
            true => interrupted.store(true, Ordering::SeqCst),
            false => state.pending.push(request_id),
        }
    }
}

struct Session<'a, W: Write> {
    writer: &'a ProtocolWriter<W>,
    deps: &'a AppDeps,
    pipeline: &'a super::tools_pipeline::PipelineJobs,
    cfg: &'a McpConfig,
    cancel: &'a Mutex<CancelState>,
    err: &'a mut dyn Write,
    initialized: bool,
}

impl<W: Write> Session<'_, W> {
    fn handle(&mut self, message: &Value) -> Option<Value> {
        match classify(message) {
            Incoming::Invalid { id, message } => Some(error(&id, INVALID_REQUEST, message)),
            // `notifications/initialized` needs no answer, `notifications/cancelled` was
            // handled by the reader, any other notification is ignored.
            Incoming::Notification { .. } | Incoming::Response => None,
            Incoming::Request { id, method, params } => Some(self.request(&id, &method, params)),
        }
    }

    fn request(&mut self, id: &Value, method: &str, params: Option<Value>) -> Value {
        match method {
            "initialize" => self.initialize(id, params),
            "ping" => result(id, json!({})),
            // Lifecycle: tools are only served once the client has initialized. `ping`
            // stays available, as the spec allows it at any time.
            "tools/list" | "tools/call" if !self.initialized => error(
                id,
                SERVER_NOT_INITIALIZED,
                "serveur non initialisé : envoyer initialize d'abord",
            ),
            "tools/list" => result(id, json!({ "tools": tools::tool_definitions() })),
            "tools/call" => self.call(id, params),
            _ => error(id, METHOD_NOT_FOUND, format!("méthode inconnue : {method}")),
        }
    }

    fn initialize(&mut self, id: &Value, params: Option<Value>) -> Value {
        if self.initialized {
            return error(id, INVALID_REQUEST, "serveur déjà initialisé");
        }
        let requested = params
            .as_ref()
            .and_then(|p| p.get("protocolVersion"))
            .and_then(Value::as_str);
        let Some(requested) = requested else {
            return error(id, INVALID_PARAMS, "initialize : protocolVersion manquant");
        };
        self.initialized = true;
        result(
            id,
            json!({
                "protocolVersion": negotiate(requested),
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": { "name": "surgeon", "version": env!("CARGO_PKG_VERSION") },
                "instructions": INSTRUCTIONS,
            }),
        )
    }

    fn call(&mut self, id: &Value, params: Option<Value>) -> Value {
        let Some(params) = params.filter(Value::is_object) else {
            return error(id, INVALID_PARAMS, "tools/call : params doit être un objet");
        };
        let Some(name) = params.get("name").and_then(Value::as_str) else {
            return error(id, INVALID_PARAMS, "tools/call : name manquant");
        };
        if !tools::is_known(name) {
            return error(id, INVALID_PARAMS, format!("outil inconnu : {name}"));
        }
        let arguments = match params.get("arguments") {
            None | Some(Value::Null) => json!({}),
            Some(args @ Value::Object(_)) => args.clone(),
            Some(_) => {
                return error(
                    id,
                    INVALID_PARAMS,
                    "tools/call : arguments doit être un objet",
                )
            }
        };
        if let Some(message) = tools::unknown_property(name, &arguments) {
            return error(id, INVALID_PARAMS, message);
        }
        let token = params
            .pointer("/_meta/progressToken")
            .filter(|t| t.is_string() || t.is_number())
            .cloned();

        self.start(id);
        let mut sink = McpProgress {
            writer: self.writer,
            token,
            extract_sets: (0, 0),
        };
        let mut ctx = ToolContext {
            deps: self.deps,
            cfg: self.cfg,
            sink: &mut sink,
            pipeline: self.pipeline,
        };
        let started = tools_session::now_unix();
        let journaled = arguments.clone();
        // A panic unwinds through BackupSet::drop (rollback) and never kills the server.
        let outcome = catch_unwind(AssertUnwindSafe(|| tools::call(name, arguments, &mut ctx)));
        self.finish();
        match outcome {
            Ok(mut tool_result) => {
                tools_session::stamp(&ctx, name, &journaled, started, &mut tool_result);
                // A bench moved off hipfire says why in its result, and in the server's log.
                let warnings = &tool_result["structuredContent"]["warnings"];
                let set_aside = warnings
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str);
                for w in set_aside.filter(|w| w.starts_with(tools::HIPFIRE_SET_ASIDE)) {
                    let _ = writeln!(self.err, "surgeon mcp : {name} : {w}");
                }
                result(id, tool_result)
            }
            Err(payload) => {
                let _ = writeln!(
                    self.err,
                    "surgeon mcp : panic dans {name} : {}",
                    panic_message(payload.as_ref())
                );
                error(
                    id,
                    INTERNAL_ERROR,
                    "erreur interne pendant l'exécution de l'outil",
                )
            }
        }
    }

    /// Clears the flag left by a previous request, applies a cancellation that arrived
    /// while this one was queued, and publishes the id to the reader thread.
    fn start(&self, id: &Value) {
        if let Ok(mut state) = self.cancel.lock() {
            let queued_cancel = state.pending.iter().position(|p| p == id);
            if let Some(i) = queued_cancel {
                state.pending.remove(i);
            }
            // A call that starts after the client shut down is cancelled from the start.
            let cancelled = queued_cancel.is_some() || state.closed.is_some();
            self.deps.interrupted.store(cancelled, Ordering::SeqCst);
            state.current = Some(id.clone());
        }
    }

    fn finish(&self) {
        if let Ok(mut state) = self.cancel.lock() {
            state.current = None;
        }
    }
}

/// `notifications/progress` when the client sent a `progressToken`, nothing otherwise.
/// `progress` counts tensors (apply) or prompts (eval) across the whole call.
struct McpProgress<'a, W: Write> {
    writer: &'a ProtocolWriter<W>,
    token: Option<Value>,
    /// (positive, negative) prompt counts of a running extraction.
    extract_sets: (usize, usize),
}

impl<W: Write> ProgressSink for McpProgress<'_, W> {
    fn event(&mut self, event: ProgressEvent) {
        let Some(token) = &self.token else {
            return;
        };
        let (pos, neg) = self.extract_sets;
        // Extraction: one step for the weights, then one per prompt (pos, then neg).
        let extract_total = 1 + pos + neg;
        let (progress, total, message) = match event {
            ProgressEvent::ExtractStart { n_pos, n_neg, .. } => {
                self.extract_sets = (n_pos, n_neg);
                return;
            }
            ProgressEvent::ExtractLoading { done, total } if done == total => (
                1,
                extract_total,
                format!("poids chargés ({total} tenseurs)"),
            ),
            ProgressEvent::ExtractPrompt { set, done, total } => {
                let offset = match set {
                    PromptSet::Pos => 0,
                    PromptSet::Neg => pos,
                };
                (
                    1 + offset + done,
                    extract_total,
                    format!("{} {done}/{total}", set.as_str()),
                )
            }
            ProgressEvent::Shift {
                op_id,
                tensor,
                overall_done,
                overall_total,
                ..
            } => (overall_done, overall_total, format!("{op_id} · {tensor}")),
            ProgressEvent::EvalPrompt { id, done, total } => (done, total, format!("prompt {id}")),
            // Only the per-tensor counter: hashing and verification restart from zero.
            ProgressEvent::Surgery(crate::surgery::SurgeryEvent::Tensor { done, total, name }) => {
                (done, total, format!("{name} ({done}/{total})"))
            }
            ProgressEvent::BenchDumpPrompt { done, total } => {
                (done, total, format!("prompt {done}/{total}"))
            }
            ProgressEvent::BenchLogitsBlock { done, total } => {
                (done, total, format!("bloc {done}/{total}"))
            }
            ProgressEvent::BenchEvalQuestion { done, total } => {
                (done, total, format!("question {done}/{total}"))
            }
            ProgressEvent::BenchRefusalPrompt { done, total } => {
                (done, total, format!("prompt {done}/{total}"))
            }
            ProgressEvent::BenchJudgeVerdict { done, total } => {
                (done, total, format!("verdict {done}/{total}"))
            }
            ProgressEvent::AutotuneForward { done, total } => {
                (done, total, format!("forward {done}/{total}"))
            }
            ProgressEvent::GraphNode { id, done, total } => {
                (done, total, format!("{id} ({done}/{total})"))
            }
            // llama-perplexity announces its total late, or never: `total` is then omitted.
            ProgressEvent::BenchPplChunk { done, total } => {
                let mut params = json!({ "progressToken": token, "progress": done, "message": format!("chunk {done}") });
                if let Some(total) = total {
                    params["total"] = json!(total);
                }
                self.writer
                    .send(&notification("notifications/progress", params));
                return;
            }
            _ => return,
        };
        self.writer.send(&notification(
            "notifications/progress",
            json!({ "progressToken": token, "progress": progress, "total": total, "message": message }),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lines(input: &[u8], max: usize) -> Vec<String> {
        let mut reader = io::Cursor::new(input.to_vec());
        let mut buf = Vec::new();
        let mut out = Vec::new();
        loop {
            match read_line_capped(&mut reader, &mut buf, max).unwrap() {
                LineRead::Eof => return out,
                LineRead::TooLong => out.push("<too long>".into()),
                LineRead::Line => out.push(String::from_utf8_lossy(&buf).into_owned()),
            }
        }
    }

    #[test]
    fn framing_handles_crlf_overflow_and_unterminated_tail() {
        assert_eq!(lines(b"a\r\nb\n", 10), ["a\r", "b"]);
        assert_eq!(lines(b"0123456789AB\nok\n", 10), ["<too long>", "ok"]);
        assert_eq!(
            lines(b"x\nlast", 10),
            ["x", "last"],
            "EOF mid-line still yields it"
        );
        assert_eq!(lines(b"", 10), Vec::<String>::new());
        assert_eq!(lines(b"0123456789AB", 10), ["<too long>"]);
    }
}
