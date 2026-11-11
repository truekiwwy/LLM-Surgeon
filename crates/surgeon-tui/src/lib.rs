//! surgeon-tui: fast terminal inspection of the spectral topology of LLM weights
//! (Safetensors and GGUF), with a local IPC bridge that AI coding agents
//! (Claude Code, Codex, Aider) can drive to highlight layers and preview
//! steering patches.
//!
//! Layout of the crate:
//! - [`model`]: zero-copy mmap readers, per-tensor static metrics, role taxonomy.
//! - [`ipc`]: the JSON-lines bridge (Unix socket / Windows named pipe) and its protocol.
//! - [`context`]: the Markdown context snippet exported for AI agents.
//! - [`app`]: state and the async event loop.
//! - [`ui`]: the three-panel Ratatui interface.
#![deny(unsafe_code)]

pub mod app;
pub mod context;
pub mod ipc;
pub mod model;
pub mod terminal;
pub mod ui;
