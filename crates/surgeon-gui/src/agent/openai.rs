//! OpenAI Chat Completions (`POST {base}/chat/completions`, `stream: true`,
//! `stream_options.include_usage`; `GET {base}/models`), the format OpenAI documents and that
//! OpenRouter, Ollama, LM Studio and `llama-server` also speak. A chunk carries
//! `choices[0].delta` (`content`, `tool_calls[]` keyed by `index` whose `id`, `name` and
//! `arguments` arrive in pieces) and `finish_reason`; the last chunk carries `usage`; the
//! stream ends with `data: [DONE]`.

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;

use serde_json::{json, Value};

use super::anthropic::parse_input;
use super::http::{read_json, Endpoint, HeaderValue, Reach, Timeouts};
use super::provider::{
    read_sse, Block, Delta, LlmProvider, ProviderError, Request, Role, StopReason, Turn, Usage,
};
use super::vault::Secret;

pub const DEFAULT_BASE: &str = "https://api.openai.com/v1";

pub struct OpenAi {
    endpoint: Endpoint,
    /// Optional for local servers.
    key: Option<Secret>,
}

impl OpenAi {
    pub fn new(base: &str, key: Option<Secret>, reach: Reach) -> Result<Self, ProviderError> {
        Ok(Self {
            endpoint: Endpoint::new(base, reach, Timeouts::default())?,
            key,
        })
    }

    fn headers(&self) -> Vec<(&'static str, HeaderValue<'_>)> {
        self.key
            .iter()
            .map(|k| ("authorization", HeaderValue::Secret(k, "Bearer ")))
            .collect()
    }
}

pub fn body(req: &Request) -> Value {
    let mut messages = vec![json!({"role": "system", "content": req.system})];
    for m in &req.messages {
        match m.role {
            Role::User => {
                for b in &m.blocks {
                    if let Block::ToolResult {
                        id,
                        content,
                        is_error,
                    } = b
                    {
                        let content = match is_error {
                            true => format!("ERREUR : {content}"),
                            false => content.clone(),
                        };
                        messages
                            .push(json!({"role": "tool", "tool_call_id": id, "content": content}));
                    }
                }
                let text = m.text();
                if !text.is_empty() {
                    messages.push(json!({"role": "user", "content": text}));
                }
            }
            Role::Assistant => {
                let calls: Vec<Value> = m
                    .blocks
                    .iter()
                    .filter_map(|b| match b {
                        Block::ToolUse { id, name, input } => Some(json!({"id": id,
                            "type": "function",
                            "function": {"name": name, "arguments": input.to_string()}})),
                        _ => None,
                    })
                    .collect();
                let text = m.text();
                let mut msg = json!({"role": "assistant",
                    "content": if text.is_empty() { Value::Null } else { json!(text) }});
                if !calls.is_empty() {
                    msg["tool_calls"] = Value::Array(calls);
                }
                messages.push(msg);
            }
        }
    }
    let mut body = json!({
        "model": req.model,
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
    });
    if !req.tools.is_empty() {
        body["tools"] = req
            .tools
            .iter()
            .map(|t| {
                json!({"type": "function", "function": {"name": t.name,
                "description": t.description, "parameters": t.schema}})
            })
            .collect();
    }
    body
}

#[derive(Default)]
struct Call {
    id: String,
    name: String,
    arguments: String,
}

impl LlmProvider for OpenAi {
    fn list_models(&self) -> Result<Vec<String>, ProviderError> {
        let page = read_json(self.endpoint.get("/models", &self.headers())?)?;
        let data = page["data"]
            .as_array()
            .ok_or_else(|| ProviderError::Protocol("/models sans data".into()))?;
        let mut ids: Vec<String> = data
            .iter()
            .filter_map(|m| m["id"].as_str().map(String::from))
            .collect();
        ids.sort();
        Ok(ids)
    }

    fn stream(
        &self,
        req: &Request,
        cancel: &AtomicBool,
        on: &mut dyn FnMut(Delta),
    ) -> Result<Turn, ProviderError> {
        let mut response =
            self.endpoint
                .post_json("/chat/completions", &self.headers(), &body(req))?;
        parse_stream(&mut response.reader, cancel, on)
    }
}

pub fn parse_stream(
    reader: &mut dyn std::io::BufRead,
    cancel: &AtomicBool,
    on: &mut dyn FnMut(Delta),
) -> Result<Turn, ProviderError> {
    let mut text = String::new();
    let mut calls: BTreeMap<u64, Call> = BTreeMap::new();
    let mut usage = Usage::default();
    let mut finish: Option<String> = None;
    let mut done = false;
    read_sse(reader, cancel, &mut |ev| {
        if ev.data.trim() == "[DONE]" {
            done = true;
            return Ok(false);
        }
        let chunk: Value =
            serde_json::from_str(&ev.data).map_err(|e| ProviderError::Protocol(e.to_string()))?;
        if chunk["error"].is_object() {
            let message = chunk["error"]["message"].as_str().unwrap_or("").to_owned();
            let code = chunk["error"]["code"].as_str().unwrap_or("");
            return Err(match code {
                "rate_limit_exceeded" => ProviderError::RateLimited {
                    retry_after: None,
                    message,
                },
                "context_length_exceeded" => ProviderError::ContextOverflow(message),
                _ => ProviderError::Server(500, message),
            });
        }
        if let Some(u) = chunk["usage"].as_object() {
            usage.input_tokens = u.get("prompt_tokens").and_then(Value::as_u64).unwrap_or(0);
            usage.output_tokens = u
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(0);
        }
        let Some(choice) = chunk["choices"].get(0) else {
            return Ok(true);
        };
        let delta = &choice["delta"];
        if let Some(piece) = delta["content"].as_str().filter(|p| !p.is_empty()) {
            text.push_str(piece);
            on(Delta::Text(piece.into()));
        }
        for tc in delta["tool_calls"].as_array().into_iter().flatten() {
            let call = calls.entry(tc["index"].as_u64().unwrap_or(0)).or_default();
            if let Some(id) = tc["id"].as_str() {
                call.id = id.into();
            }
            if let Some(name) = tc["function"]["name"].as_str() {
                call.name.push_str(name);
            }
            if let Some(args) = tc["function"]["arguments"].as_str() {
                call.arguments.push_str(args);
            }
        }
        if let Some(f) = choice["finish_reason"].as_str() {
            finish = Some(f.into());
        }
        Ok(true)
    })?;
    if !done && finish.is_none() {
        return Err(ProviderError::Network("flux coupé avant la fin".into()));
    }
    let mut blocks = Vec::new();
    if !text.is_empty() {
        blocks.push(Block::Text { text });
    }
    for (i, call) in calls {
        blocks.push(Block::ToolUse {
            // Some local servers send no id: one is made up so results can be matched.
            id: match call.id.is_empty() {
                true => format!("call_{i}"),
                false => call.id,
            },
            name: call.name,
            input: parse_input(&call.arguments),
        });
    }
    let stop = match finish.as_deref() {
        Some("tool_calls") => StopReason::ToolUse,
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        _ if blocks.iter().any(|b| matches!(b, Block::ToolUse { .. })) => StopReason::ToolUse,
        _ => StopReason::Other,
    };
    Ok(Turn {
        blocks,
        usage,
        stop,
    })
}
