//! Anthropic Messages API (`POST /v1/messages`, `stream: true`; `GET /v1/models`), from the
//! official reference: SSE events `message_start`, `content_block_start` / `_delta` /
//! `_stop` (`text_delta`, `input_json_delta`, `thinking_delta`, `signature_delta`),
//! `message_delta` (stop reason, output tokens), `message_stop`, `ping`, `error`. Several
//! `tool_use` blocks in one turn are parallel calls; their results go back in one user
//! message. `thinking` blocks are replayed unchanged, as the API requires in tool loops.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use super::http::{read_json, Endpoint, HeaderValue, Reach, Timeouts};
use super::provider::{
    read_sse, Block, Delta, LlmProvider, ProviderError, Request, Role, StopReason, Turn, Usage,
};
use super::vault::Secret;

pub const DEFAULT_BASE: &str = "https://api.anthropic.com";
/// The `anthropic-version` header of the current API.
const VERSION: &str = "2026-10-21";
const PROVIDER: &str = "anthropic";

pub struct Anthropic {
    endpoint: Endpoint,
    key: Secret,
}

impl Anthropic {
    pub fn new(base: &str, key: Secret, reach: Reach) -> Result<Self, ProviderError> {
        Ok(Self {
            endpoint: Endpoint::new(base, reach, Timeouts::default())?,
            key,
        })
    }

    fn headers(&self) -> [(&'static str, HeaderValue<'_>); 2] {
        [
            ("x-api-key", HeaderValue::Secret(&self.key, "")),
            ("anthropic-version", HeaderValue::Plain(VERSION)),
        ]
    }
}

pub fn body(req: &Request) -> Value {
    let messages: Vec<Value> = req
        .messages
        .iter()
        .map(|m| {
            let content: Vec<Value> = m
                .blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Text { text } => Some(json!({"type": "text", "text": text})),
                    Block::ToolUse { id, name, input } => {
                        Some(json!({"type": "tool_use", "id": id, "name": name, "input": input}))
                    }
                    Block::ToolResult {
                        id,
                        content,
                        is_error,
                    } => Some(json!({"type": "tool_result", "tool_use_id": id,
                        "content": content, "is_error": is_error})),
                    Block::Opaque { provider, raw } if provider == PROVIDER => Some(raw.clone()),
                    Block::Opaque { .. } => None,
                })
                .collect();
            let role = match m.role {
                Role::User => "user",
                Role::Assistant => "assistant",
            };
            json!({"role": role, "content": content})
        })
        .collect();
    let tools: Vec<Value> = req
        .tools
        .iter()
        .map(|t| json!({"name": t.name, "description": t.description, "input_schema": t.schema}))
        .collect();
    let mut body = json!({
        "model": req.model,
        "max_tokens": req.max_tokens,
        "system": req.system,
        "messages": messages,
        "stream": true,
    });
    if !tools.is_empty() {
        body["tools"] = Value::Array(tools);
    }
    body
}

enum Building {
    Text(String),
    Tool {
        id: String,
        name: String,
        json: String,
    },
    Raw(Value),
}

impl LlmProvider for Anthropic {
    fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let mut ids = Vec::new();
        let mut after: Option<String> = None;
        loop {
            let path = match &after {
                Some(a) => format!("/v1/models?limit=1000&after_id={a}"),
                None => "/v1/models?limit=1000".into(),
            };
            let page = read_json(self.endpoint.get(&path, &self.headers())?)?;
            let data = page["data"]
                .as_array()
                .ok_or_else(|| ProviderError::Protocol("/v1/models sans data".into()))?;
            ids.extend(
                data.iter()
                    .filter_map(|m| m["id"].as_str().map(String::from)),
            );
            match (page["has_more"].as_bool(), page["last_id"].as_str()) {
                (Some(true), Some(last)) if after.as_deref() != Some(last) => {
                    after = Some(last.into())
                }
                _ => return Ok(ids),
            }
        }
    }

    fn stream(
        &self,
        req: &Request,
        cancel: &AtomicBool,
        on: &mut dyn FnMut(Delta),
    ) -> Result<Turn, ProviderError> {
        let mut response = self
            .endpoint
            .post_json("/v1/messages", &self.headers(), &body(req))?;
        parse_stream(&mut response.reader, cancel, on)
    }
}

pub fn parse_stream(
    reader: &mut dyn std::io::BufRead,
    cancel: &AtomicBool,
    on: &mut dyn FnMut(Delta),
) -> Result<Turn, ProviderError> {
    let mut blocks: BTreeMap<u64, Building> = BTreeMap::new();
    let mut usage = Usage::default();
    let mut stop = StopReason::Other;
    let mut complete = false;
    read_sse(reader, cancel, &mut |ev| {
        let data: Value = serde_json::from_str(&ev.data)
            .map_err(|e| ProviderError::Protocol(format!("{e} : {}", ev.event)))?;
        let index = data["index"].as_u64().unwrap_or(0);
        match data["type"].as_str().unwrap_or(&ev.event) {
            "message_start" => {
                let u = &data["message"]["usage"];
                usage.input_tokens = [
                    "input_tokens",
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                ]
                .iter()
                .filter_map(|k| u[*k].as_u64())
                .sum();
                usage.output_tokens = u["output_tokens"].as_u64().unwrap_or(0);
            }
            "content_block_start" => {
                let b = &data["content_block"];
                let building = match b["type"].as_str() {
                    Some("text") => Building::Text(b["text"].as_str().unwrap_or("").into()),
                    Some("tool_use") => Building::Tool {
                        id: b["id"].as_str().unwrap_or_default().into(),
                        name: b["name"].as_str().unwrap_or_default().into(),
                        json: String::new(),
                    },
                    _ => Building::Raw(b.clone()),
                };
                if let Building::Text(t) = &building {
                    if !t.is_empty() {
                        on(Delta::Text(t.clone()));
                    }
                }
                blocks.insert(index, building);
            }
            "content_block_delta" => {
                let d = &data["delta"];
                match (blocks.get_mut(&index), d["type"].as_str()) {
                    (Some(Building::Text(t)), Some("text_delta")) => {
                        let piece = d["text"].as_str().unwrap_or("");
                        t.push_str(piece);
                        on(Delta::Text(piece.into()));
                    }
                    (Some(Building::Tool { json, .. }), Some("input_json_delta")) => {
                        json.push_str(d["partial_json"].as_str().unwrap_or(""));
                    }
                    (Some(Building::Raw(raw)), Some("thinking_delta")) => {
                        let mut t = raw["thinking"].as_str().unwrap_or("").to_owned();
                        t.push_str(d["thinking"].as_str().unwrap_or(""));
                        raw["thinking"] = json!(t);
                    }
                    (Some(Building::Raw(raw)), Some("signature_delta")) => {
                        raw["signature"] = d["signature"].clone();
                    }
                    _ => {}
                }
            }
            "message_delta" => {
                stop = match data["delta"]["stop_reason"].as_str() {
                    Some("end_turn") | Some("stop_sequence") => StopReason::EndTurn,
                    Some("tool_use") => StopReason::ToolUse,
                    Some("max_tokens") => StopReason::MaxTokens,
                    _ => StopReason::Other,
                };
                if let Some(out) = data["usage"]["output_tokens"].as_u64() {
                    usage.output_tokens = out;
                }
            }
            "message_stop" => {
                complete = true;
                return Ok(false);
            }
            "error" => {
                let kind = data["error"]["type"].as_str().unwrap_or("");
                let message = data["error"]["message"].as_str().unwrap_or("").to_owned();
                return Err(match kind {
                    "overloaded_error" => ProviderError::Overloaded(message),
                    "rate_limit_error" => ProviderError::RateLimited {
                        retry_after: None,
                        message,
                    },
                    "authentication_error" | "permission_error" => ProviderError::Auth(message),
                    "api_error" => ProviderError::Server(500, message),
                    _ if super::provider::is_context_overflow(&message.to_lowercase()) => {
                        ProviderError::ContextOverflow(message)
                    }
                    _ => ProviderError::Protocol(format!("{kind} : {message}")),
                });
            }
            _ => {}
        }
        Ok(true)
    })?;
    if !complete {
        return Err(ProviderError::Network(
            "flux coupé avant message_stop".into(),
        ));
    }
    let blocks = blocks
        .into_values()
        .filter_map(|b| match b {
            Building::Text(text) if text.is_empty() => None,
            Building::Text(text) => Some(Block::Text { text }),
            Building::Tool { id, name, json } => Some(Block::ToolUse {
                id,
                name,
                input: parse_input(&json),
            }),
            Building::Raw(raw) => Some(Block::Opaque {
                provider: PROVIDER.into(),
                raw,
            }),
        })
        .collect();
    Ok(Turn {
        blocks,
        usage,
        stop,
    })
}

/// The accumulated JSON of a tool call; `{}` when empty. Unreadable JSON stays a string, and
/// the session answers it with an error result instead of running anything.
pub fn parse_input(json: &str) -> Value {
    match json.trim() {
        "" => json!({}),
        text => serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.into())),
    }
}
