//! JSON-RPC 2.0 framing for the MCP stdio transport: message classification, error codes,
//! and the single writer that owns stdout.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use serde_json::{json, Value};

pub const PARSE_ERROR: i64 = -32700;
pub const INVALID_REQUEST: i64 = -32600;
pub const METHOD_NOT_FOUND: i64 = -32601;
pub const INVALID_PARAMS: i64 = -32602;
pub const INTERNAL_ERROR: i64 = -32603;
/// Server-defined range of JSON-RPC: a `tools/*` request before `initialize`.
pub const SERVER_NOT_INITIALIZED: i64 = -32002;

/// Handshake-based revisions (the "legacy" era of the spec), oldest first. Checked on
/// modelcontextprotocol.io: 2025-11-25 is the last revision with `initialize`; 2026-07-28
/// is handshake-free (`server/discover`, per-request version) and is not implemented. A
/// dual-era client probing `server/discover` gets -32601 and falls back to `initialize`.
pub const SUPPORTED_PROTOCOL_VERSIONS: [&str; 4] =
    ["2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"];

/// The requested version when supported, else the most recent one.
pub fn negotiate(requested: &str) -> &'static str {
    let latest = SUPPORTED_PROTOCOL_VERSIONS[SUPPORTED_PROTOCOL_VERSIONS.len() - 1];
    SUPPORTED_PROTOCOL_VERSIONS
        .iter()
        .copied()
        .find(|v| *v == requested)
        .unwrap_or(latest)
}

pub fn result(id: &Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

pub fn error(id: &Value, code: i64, message: impl Into<String>) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message.into() } })
}

pub fn notification(method: &str, params: Value) -> Value {
    json!({ "jsonrpc": "2.0", "method": method, "params": params })
}

#[derive(Debug, Clone, PartialEq)]
pub enum Incoming {
    Request {
        id: Value,
        method: String,
        params: Option<Value>,
    },
    Notification {
        method: String,
    },
    /// A response from the client: this server never sends requests, so it is ignored.
    Response,
    /// Answered with -32600 and this `id` (null when it could not be read).
    Invalid {
        id: Value,
        message: &'static str,
    },
}

pub fn classify(message: &Value) -> Incoming {
    let invalid = |id: Value, message| Incoming::Invalid { id, message };
    let object = match message {
        // Batches are optional in JSON-RPC and were removed from MCP in 2025-06-18.
        Value::Array(_) => {
            return invalid(Value::Null, "les requêtes batch ne sont pas supportées")
        }
        Value::Object(object) => object,
        _ => return invalid(Value::Null, "un message JSON-RPC doit être un objet"),
    };
    let id = match object.get("id") {
        None => None,
        Some(id @ (Value::String(_) | Value::Number(_) | Value::Null)) => Some(id.clone()),
        Some(_) => return invalid(Value::Null, "id doit être une chaîne, un nombre ou null"),
    };
    let reply_id = id.clone().unwrap_or(Value::Null);
    if object.get("jsonrpc").and_then(Value::as_str) != Some("2.0") {
        return invalid(reply_id, "jsonrpc doit valoir \"2.0\"");
    }
    match (object.get("method"), id) {
        (Some(Value::String(method)), Some(id)) => Incoming::Request {
            id,
            method: method.clone(),
            params: object.get("params").cloned(),
        },
        (Some(Value::String(method)), None) => Incoming::Notification {
            method: method.clone(),
        },
        (Some(_), _) => invalid(reply_id, "method doit être une chaîne"),
        (None, _) if object.contains_key("result") || object.contains_key("error") => {
            Incoming::Response
        }
        (None, _) => invalid(reply_id, "method manquant"),
    }
}

/// The only holder of stdout in `mcp` mode: one JSON message per line, flushed at once.
/// Once a write fails (client gone, pipe closed) every later send is dropped.
pub struct ProtocolWriter<W: Write> {
    inner: Mutex<W>,
    broken: AtomicBool,
}

impl<W: Write> ProtocolWriter<W> {
    pub fn new(inner: W) -> Self {
        Self {
            inner: Mutex::new(inner),
            broken: AtomicBool::new(false),
        }
    }

    pub fn send(&self, message: &Value) {
        if self.is_broken() {
            return;
        }
        // serde_json escapes control characters, so the line never contains a raw newline.
        let mut line = message.to_string();
        line.push('\n');
        let written = match self.inner.lock() {
            Ok(mut w) => w
                .write_all(line.as_bytes())
                .and_then(|()| w.flush())
                .is_ok(),
            Err(_) => false,
        };
        if !written {
            self.broken.store(true, Ordering::SeqCst);
        }
    }

    pub fn is_broken(&self) -> bool {
        self.broken.load(Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classification_covers_every_shape() {
        let req = json!({"jsonrpc":"2.0","id":"a","method":"ping"});
        assert!(matches!(classify(&req), Incoming::Request { id, .. } if id == "a"));
        let null_id = json!({"jsonrpc":"2.0","id":null,"method":"ping"});
        assert!(matches!(
            classify(&null_id),
            Incoming::Request {
                id: Value::Null,
                ..
            }
        ));
        let notif = json!({"jsonrpc":"2.0","method":"notifications/initialized"});
        assert!(matches!(classify(&notif), Incoming::Notification { .. }));
        assert_eq!(
            classify(&json!({"jsonrpc":"2.0","id":1,"result":{}})),
            Incoming::Response
        );
        for (bad, id) in [
            (json!([]), Value::Null),
            (json!(3), Value::Null),
            (json!({"jsonrpc":"2.0","id":{},"method":"x"}), Value::Null),
            (json!({"jsonrpc":"1.0","id":4,"method":"x"}), json!(4)),
            (json!({"jsonrpc":"2.0","id":5}), json!(5)),
            (json!({"jsonrpc":"2.0","id":6,"method":7}), json!(6)),
        ] {
            assert!(
                matches!(classify(&bad), Incoming::Invalid { id: got, .. } if got == id),
                "{bad}"
            );
        }
    }

    #[test]
    fn negotiation_echoes_or_falls_back_to_latest() {
        assert_eq!(negotiate("2025-03-26"), "2025-03-26");
        assert_eq!(negotiate("2024-11-05"), "2024-11-05");
        assert_eq!(negotiate("1999-01-01"), "2025-11-25");
        assert_eq!(negotiate("2026-07-28"), "2025-11-25");
    }

    struct Closed;
    impl Write for Closed {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::BrokenPipe.into())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn writer_emits_single_lines_and_survives_a_closed_pipe() {
        let w = ProtocolWriter::new(Vec::new());
        w.send(&json!({"text": "a\nb"}));
        let bytes = w.inner.into_inner().unwrap();
        assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), 1);
        let closed = ProtocolWriter::new(Closed);
        closed.send(&json!({}));
        closed.send(&json!({}));
        assert!(closed.is_broken());
    }
}
