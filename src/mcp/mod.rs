//! Native MCP (Model Context Protocol) server over stdio, and the JSON Schema exporter.
//! Hand-written on serde_json: the surface is small and no SDK version can drift.
//!
//! (b) Message flow (graphify, checked by hand):
//!
//! ```text
//! stdin -> read_loop thread: read_line_capped (16 MiB, resync on \n), BOM/CRLF/blank lines
//!          -> serde_json -> note_cancellation (requestId == running ? interrupted = true
//!                                              : queued for later)
//!          -> mpsc channel -> serve loop -> Session::handle -> classify
//!             -> initialize | ping | tools/list | tools/call | -32601 / -32600 / -32002
//!             tools/call -> Session::start (reset flag, queued cancel) -> catch_unwind
//!                -> tools::unknown_property (surgeon_bench: -32602)
//!                -> tools::call -> sandbox -> core_check | core_inspect | core_apply | core_eval
//!                   | core_extract_method (MethodId: built-in registry)
//!                   | core_bench_{logits,ppl,eval,dump} (mode checked first)
//!                   (McpProgress sink -> notifications/progress) -> envelope_of
//!          -> ProtocolWriter (Mutex, one line + flush) -> stdout
//! ```
//!
//! (c) Call graph shared by the CLI and the server: the command cores never receive a
//! writer, they return an `Outcome`; `envelope_of` is the single JSON rendering.
//!
//! ```text
//! CLI: main -> run_with -> dispatch -> core_check / core_inspect (Ui of the terminal)
//!                                   -> apply -> core_apply(CliSink: terminal UI)
//!                                   -> eval  -> core_eval(NullSink)
//!                       -> emit -> envelope_of (--json) | Outcome.human
//! MCP: main -> run_with -> run_stdio -> serve -> tools::call
//!                       -> core_check / core_inspect (Ui::off)
//!                       -> core_apply / core_eval / core_extract_method / core_bench_* (McpProgress)
//!                       -> envelope_of
//! ```
//!
//! stdout audit: in `mcp` mode `run_with` hands `out` to `serve`, which moves it into the
//! `ProtocolWriter` before anything else runs; cores get `Ui::off()` and a sink, never a
//! writer. Diagnostics go to the injected `err` (stderr). `run_with` routes `mcp` before
//! its panic catcher, so the panic hook prints on stderr, never on stdout. Remaining
//! direct writes in the crate are all on stderr (see `ui.rs`).

mod protocol;
mod sandbox;
mod server;
mod tools;
mod tools_graph;
mod tools_p13;
mod tools_pipeline;

use std::io::{BufReader, Write};

pub use protocol::{
    negotiate, INTERNAL_ERROR, INVALID_PARAMS, INVALID_REQUEST, METHOD_NOT_FOUND, PARSE_ERROR,
    SERVER_NOT_INITIALIZED, SUPPORTED_PROTOCOL_VERSIONS,
};
pub use sandbox::{Sandbox, SandboxError};
pub use server::{serve, McpConfig, MAX_LINE_BYTES};
pub use tools::{
    patch_schema, render_schema, tool_definitions, ApplyToolArgs, AutotuneToolArgs, BenchMode,
    BenchToolArgs, CheckToolArgs, EvalToolArgs, ExtractToolArgs, InspectToolArgs, MethodId,
    TOOL_APPLY, TOOL_AUTOTUNE, TOOL_BENCH, TOOL_CHECK, TOOL_EVAL, TOOL_EXTRACT, TOOL_INSPECT,
};

use crate::app::{exit, AppDeps};
use crate::cli::{McpArgs, SchemaKind};

/// `surgeon schema`: raw JSON even under `--json` or `CLICOLOR_FORCE`, since nothing here
/// goes through `ui`. A closed pipe is ignored.
pub fn write_schema(kind: SchemaKind, out: &mut dyn Write) -> i32 {
    let _ = out.write_all(render_schema(kind).as_bytes());
    let _ = out.flush();
    exit::OK
}

/// `surgeon mcp`: the real stdin, the injected stdout and stderr.
pub fn run_stdio(args: &McpArgs, deps: &AppDeps, out: &mut dyn Write, err: &mut dyn Write) -> i32 {
    let cfg = match McpConfig::from_args(args) {
        Ok(cfg) => cfg,
        Err(e) => {
            let _ = writeln!(err, "erreur : surgeon mcp : {e}");
            return exit::INVALID_INPUT;
        }
    };
    let _ = writeln!(
        err,
        "surgeon mcp {} : prêt sur stdio, {} dossier(s) autorisé(s){}",
        env!("CARGO_PKG_VERSION"),
        cfg.sandbox.roots().len(),
        if cfg.read_only { ", lecture seule" } else { "" }
    );
    serve(BufReader::new(std::io::stdin()), out, err, deps, &cfg)
}
