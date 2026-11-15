//! The agent hub (docs/AGENTS.md): drive the engine with an AI agent without leaving the
//! app, with approvals computed by the code and an evidence log.
//!
//! - [`provider`]: the normalized conversation, the `LlmProvider` trait, typed errors,
//!   retries with backoff, the SSE reader.
//! - [`anthropic`], [`openai`]: the HTTP providers (OpenAI also covers compatible
//!   endpoints); [`http`]: endpoints (HTTPS or loopback, local-only mode); [`mock`]: a
//!   scripted provider for the tests and the self-test.
//! - [`vault`]: API keys in the Windows Credential Manager, else for the session only.
//! - [`runtime`]: Claude Code and Codex CLI, as installed and signed in by the user.
//! - [`detect`]: what the machine offers (CLIs, loopback servers, environment keys).
//! - [`tools`]: the engine's `tools/list` plus the interface tools; untrusted-data frames.
//! - [`policy`]: Safe / Approval / Forbidden, and the dated "always" rules.
//! - [`evidence`]: findings and their Measured / Inferred / Unknown status.
//! - [`mission`]: what a session is for (`Advance` full scan, `Scalping` one symptom).
//! - [`session`]: the loop, budgets, pause and stop, persistence in `chat/`.
//! - [`config`]: connections, consents, per-project settings (no secret).
//! - [`view`]: the Agents page; [`selftest`]: the `--self-test` dialogue.

pub mod anthropic;
pub mod config;
pub mod detect;
pub mod evidence;
pub mod http;
pub mod mission;
pub mod mock;
pub mod openai;
pub mod paths;
pub mod policy;
pub mod provider;
pub mod runtime;
pub mod selftest;
pub mod session;
pub mod tools;
pub mod vault;
pub mod view;

#[cfg(test)]
mod tests;
