//! The provider layer: one normalized conversation format, a streaming trait every provider
//! implements (`LlmProvider`), typed errors, retries with backoff and the SSE reader shared
//! by the HTTP providers. Nothing here knows about the studio or the policy.

use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
    /// A provider block that must be sent back unchanged (Anthropic `thinking` with its
    /// signature); other providers drop it.
    Opaque {
        provider: String,
        raw: Value,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub blocks: Vec<Block>,
}

impl Message {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            blocks: vec![Block::Text { text: text.into() }],
        }
    }

    pub fn text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub schema: Value,
}

#[derive(Debug, Clone)]
pub struct Request {
    pub model: String,
    pub system: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub max_tokens: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl Usage {
    pub fn total(&self) -> u64 {
        self.input_tokens + self.output_tokens
    }

    pub fn add(&mut self, other: Usage) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    /// Refusal, content filter or anything the provider names otherwise.
    Other,
}

/// What a streaming call reports while it runs.
#[derive(Debug, Clone, PartialEq)]
pub enum Delta {
    Text(String),
    /// A retry starts over: what was streamed so far is void.
    Restart,
}

/// One complete assistant turn.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    pub blocks: Vec<Block>,
    pub usage: Usage,
    pub stop: StopReason,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProviderError {
    /// Bad, revoked or missing key (401, 403).
    Auth(String),
    RateLimited {
        retry_after: Option<Duration>,
        message: String,
    },
    /// 529, 503 or an `overloaded_error` in the stream.
    Overloaded(String),
    /// The prompt does not fit the model's context window.
    ContextOverflow(String),
    /// 5xx other than overload.
    Server(u16, String),
    /// Connection, TLS, DNS, or a stream cut before its end.
    Network(String),
    Timeout,
    Cancelled,
    /// Any other HTTP status (400, 404...).
    Http(u16, String),
    /// An answer this client cannot read.
    Protocol(String),
    /// Refused before any network access (local-only mode, bad URL).
    Blocked(String),
}

impl ProviderError {
    pub fn retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimited { .. }
                | Self::Overloaded(_)
                | Self::Server(..)
                | Self::Network(_)
                | Self::Timeout
        )
    }

    /// The status and body of a failed HTTP call, classified the way both APIs document
    /// them (Anthropic: 401/403/429/529, `prompt is too long`; OpenAI:
    /// `context_length_exceeded`).
    pub fn from_status(status: u16, body: &str, retry_after: Option<Duration>) -> Self {
        let message = error_message(body);
        let lower = body.to_lowercase();
        match status {
            401 | 403 => Self::Auth(message),
            429 => Self::RateLimited {
                retry_after,
                message,
            },
            503 | 529 => Self::Overloaded(message),
            413 => Self::ContextOverflow(message),
            400 if is_context_overflow(&lower) => Self::ContextOverflow(message),
            500..=599 => Self::Server(status, message),
            _ => Self::Http(status, message),
        }
    }
}

pub fn is_context_overflow(lower: &str) -> bool {
    [
        "context_length_exceeded",
        "prompt is too long",
        "maximum context length",
        "context window",
        "too many tokens",
    ]
    .iter()
    .any(|k| lower.contains(k))
}

/// `error.message` of a JSON error body (both APIs), else the start of the body.
fn error_message(body: &str) -> String {
    serde_json::from_str::<Value>(body)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().map(String::from))
        .unwrap_or_else(|| body.chars().take(300).collect())
}

impl std::fmt::Display for ProviderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auth(m) => write!(f, "authentification refusée : {m}"),
            Self::RateLimited { message, .. } => write!(f, "limite de débit atteinte : {message}"),
            Self::Overloaded(m) => write!(f, "fournisseur surchargé : {m}"),
            Self::ContextOverflow(m) => write!(f, "fenêtre de contexte dépassée : {m}"),
            Self::Server(code, m) => write!(f, "erreur serveur HTTP {code} : {m}"),
            Self::Network(m) => write!(f, "réseau : {m}"),
            Self::Timeout => f.write_str("délai dépassé"),
            Self::Cancelled => f.write_str("annulé"),
            Self::Http(code, m) => write!(f, "HTTP {code} : {m}"),
            Self::Protocol(m) => write!(f, "réponse illisible : {m}"),
            Self::Blocked(m) => write!(f, "refusé : {m}"),
        }
    }
}

/// A provider: a model list from its API (never hard-coded ids) and one streamed turn.
/// `cancel` is checked between stream events; `on` receives text as it arrives.
pub trait LlmProvider: Send + Sync {
    fn list_models(&self) -> Result<Vec<String>, ProviderError>;
    fn stream(
        &self,
        req: &Request,
        cancel: &AtomicBool,
        on: &mut dyn FnMut(Delta),
    ) -> Result<Turn, ProviderError>;
}

#[derive(Debug, Clone, Copy)]
pub struct RetryPolicy {
    pub attempts: u32,
    pub base: Duration,
    pub max: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: 4,
            base: Duration::from_millis(500),
            max: Duration::from_secs(30),
        }
    }
}

impl RetryPolicy {
    /// Exponential backoff, or the server's `retry-after` when it gives one (capped).
    pub fn delay(&self, attempt: u32, err: &ProviderError) -> Duration {
        if let ProviderError::RateLimited {
            retry_after: Some(d),
            ..
        } = err
        {
            return (*d).min(self.max);
        }
        self.base
            .saturating_mul(2u32.saturating_pow(attempt))
            .min(self.max)
    }
}

/// One streamed turn with retries on retryable errors; a retry after streamed text sends
/// [`Delta::Restart`]. The wait is cut into slices so that a cancellation stops it.
pub fn stream_with_retries(
    provider: &dyn LlmProvider,
    req: &Request,
    policy: RetryPolicy,
    cancel: &AtomicBool,
    on: &mut dyn FnMut(Delta),
) -> Result<Turn, ProviderError> {
    let mut attempt = 0;
    loop {
        let mut streamed = false;
        let result = provider.stream(req, cancel, &mut |d| {
            streamed = true;
            on(d)
        });
        match result {
            Err(e) if e.retryable() && attempt + 1 < policy.attempts => {
                let wait = policy.delay(attempt, &e);
                attempt += 1;
                if !sleep_unless(cancel, wait) {
                    return Err(ProviderError::Cancelled);
                }
                if streamed {
                    on(Delta::Restart);
                }
            }
            other => return other,
        }
    }
}

/// Sleeps `d` unless `cancel` is raised first (false then).
pub fn sleep_unless(cancel: &AtomicBool, d: Duration) -> bool {
    let step = Duration::from_millis(20);
    let mut left = d;
    while !left.is_zero() {
        if cancel.load(Ordering::SeqCst) {
            return false;
        }
        let s = left.min(step);
        std::thread::sleep(s);
        left -= s;
    }
    !cancel.load(Ordering::SeqCst)
}

/// One Server-Sent Event: its `event:` name (may be empty) and its joined `data:` lines.
#[derive(Debug, Clone, PartialEq)]
pub struct SseEvent {
    pub event: String,
    pub data: String,
}

/// Reads SSE events until the stream ends, `f` returns false or `cancel` is raised.
/// Comments (`:`) and unknown fields are ignored, per the SSE format.
pub fn read_sse(
    reader: &mut dyn BufRead,
    cancel: &AtomicBool,
    f: &mut dyn FnMut(SseEvent) -> Result<bool, ProviderError>,
) -> Result<(), ProviderError> {
    let mut event = String::new();
    let mut data: Vec<String> = Vec::new();
    let mut line = String::new();
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err(ProviderError::Cancelled);
        }
        line.clear();
        let n = reader.read_line(&mut line).map_err(|e| match e.kind() {
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock => ProviderError::Timeout,
            _ => ProviderError::Network(format!("flux coupé : {e}")),
        })?;
        let text = line.trim_end_matches(['\r', '\n']);
        if n == 0 || text.is_empty() {
            if !data.is_empty() || !event.is_empty() {
                let ev = SseEvent {
                    event: std::mem::take(&mut event),
                    data: data.join("\n"),
                };
                data.clear();
                if !f(ev)? {
                    return Ok(());
                }
            }
            if n == 0 {
                return Ok(());
            }
            continue;
        }
        if text.starts_with(':') {
            continue;
        }
        let (field, value) = text.split_once(':').unwrap_or((text, ""));
        let value = value.strip_prefix(' ').unwrap_or(value);
        match field {
            "event" => event = value.to_owned(),
            "data" => data.push(value.to_owned()),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sse_frames_join_data_lines_and_skip_comments() {
        let text =
            ": ping\nevent: a\ndata: {\"x\":1}\n\ndata: one\ndata: two\n\nevent: last\ndata: z";
        let mut seen = Vec::new();
        read_sse(
            &mut std::io::Cursor::new(text),
            &AtomicBool::new(false),
            &mut |e| {
                seen.push(e);
                Ok(true)
            },
        )
        .unwrap();
        assert_eq!(seen.len(), 3);
        assert_eq!(seen[0].event, "a");
        assert_eq!(seen[1].data, "one\ntwo");
        assert_eq!(seen[2].data, "z");
    }

    #[test]
    fn statuses_map_to_typed_errors() {
        assert!(matches!(
            ProviderError::from_status(401, "{}", None),
            ProviderError::Auth(_)
        ));
        assert!(matches!(
            ProviderError::from_status(529, "{}", None),
            ProviderError::Overloaded(_)
        ));
        let body = r#"{"error":{"type":"invalid_request_error","message":"prompt is too long: 300000 tokens"}}"#;
        assert!(matches!(
            ProviderError::from_status(400, body, None),
            ProviderError::ContextOverflow(_)
        ));
        let body = r#"{"error":{"code":"context_length_exceeded","message":"too long"}}"#;
        assert!(matches!(
            ProviderError::from_status(400, body, None),
            ProviderError::ContextOverflow(_)
        ));
        assert!(matches!(
            ProviderError::from_status(400, "{}", None),
            ProviderError::Http(400, _)
        ));
        let rl = ProviderError::from_status(429, "{}", Some(Duration::from_secs(2)));
        assert_eq!(RetryPolicy::default().delay(0, &rl), Duration::from_secs(2));
    }
}
