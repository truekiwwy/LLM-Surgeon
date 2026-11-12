// p12 modules first (merge with p13 stays a pure addition).
pub mod kernel;

pub mod app;
pub mod arch;
pub mod bench;
pub mod cli;
pub mod engine;
pub mod eval;
pub mod extract;
pub mod graph;
pub mod hardware;
pub mod linalg;
pub mod mcp;
pub mod patch;
pub mod pipeline;
pub mod plugin;
pub mod tui;
pub mod ui;

// --- P13 ---
pub mod lora;
pub mod surgery;
