# surgeon-gui: LLM Surgery Studio

Native viewer of the weight graph written by `surgeon graph`, and a surgery workbench, in
pure Rust (egui, OpenGL through glow). No HTML, no browser, no network.

```bash
# the agent (and, on a fresh profile, the first-run wizard)
cargo run --release
# one state, opened on the 3D view
cargo run --release -- ../../graph-out/graph.json
# a GraphPatch stream (surgeon graph --format ndjson), followed live as lines are appended
cargo run --release -- ../../graph-out/graph.ndjson
# NDJSON on stdin (an orchestrator, or an agent driving the MCP server)
my-orchestrator | cargo run --release -- -
# projects in another folder than ~/surgeon-studio
cargo run --release -- --workspace D:/surgery
# projects, working copy, a real engine job, replay and ledger, without a window
cargo run --release -- --self-test
```

## Desktop app

One borderless window laid out like an IDE whose editor is an AI agent: an activity bar
(the agent, the 3D graph, the command palette `Ctrl+K`), the project and its sessions on the
left, the agent in the center, its evidence on the right, and the settings
(`Ctrl+,`, the gear of the title bar) in a dialog. Open a graph or a folder by drag and drop
or the palette; replay the **first-run wizard**, configure the AI providers and repair
dependencies from the settings; check for a new version there (manual, never silent). A
second launch hands its arguments to the running window. Design rules:
[../../docs/design/DIRECTION.md](../../docs/design/DIRECTION.md).

```bash
cargo run --release -- --self-test            # engine, OpenGL window, data folder; exit code 0 / 1
cargo run --release -- --screenshot a.png     # renders the agent off-screen (CPU, no window)
cargo run --release -- --demo-data --no-motion --screen agent:scalping --size 1440x900 --screenshot b.png
cargo run --release -- --demo-data --screen settings:providers --screenshot c.png
# the 3D view with a recorded run replayed over it, off-screen
cargo run --release -- --screenshot run.png --screen graph --replay runs/<id>/events.ndjson graph.json
cargo run --release -- --reset-settings       # the wizard runs again at the next start
```

Packaging (per-user installer, portable ZIP, checksums, optional signing) and troubleshooting:
[../../docs/DESKTOP.md](../../docs/DESKTOP.md). Crates and why: [../../docs/GUI_DEPENDENCIES.md](../../docs/GUI_DEPENDENCIES.md).

## The agent

A new session starts from a form, in one of two kinds:

- **Full scan (Advance)**: the agent checks and inspects the model, builds the complete
  surgery graph, looks for anomalies itself (norms and effective ranks out of their role's
  range, extreme conditioning, atypical layers or experts, differences against a reference,
  behavior measured read-only), reports the problems ranked by severity with their evidence,
  then proposes the fixes one by one.
- **Precise problem (Scalping)**: the user picks a symptom (excessive refusals, repetitions,
  hallucinations, language or format, reasoning, other) and describes it; the agent goes
  straight to the tensors most likely involved, confirms or rules out the hypothesis with a
  measure, then proposes the most targeted fix.

Either way every tool call becomes a job of the project, every write is an approval card
computed by the code, and every figure the agent claims is checked against the ledger
(Measured / Inferred / Unknown). Providers: Anthropic, OpenAI, any OpenAI-compatible endpoint
(OpenRouter, Ollama, LM Studio, `llama-server`), the Claude Code and Codex CLIs as signed in
by the user. Threat model and details: [docs/AGENTS.md](../../docs/AGENTS.md).

## 3D graph

Explorer (project, working copy, tensor tree; Up / Down walk it), 3D viewport, inspector and
chronology. Left drag orbits, right drag pans, wheel zooms, `R` resets. `Colonne` /
`Constellation`, `Diff (Δ)`, the replay of a run with a time cursor and, for a stream, the
patch timeline ([docs/GUI_3D.md](../../docs/GUI_3D.md)). The agent points at a tensor with
`show_in_3d`. Projects, jobs, runs and the event protocol: [docs/STUDIO.md](../../docs/STUDIO.md).

## Build

The crate has its own `Cargo.lock`, like `crates/surgeon-tui`: its windowing stack never
enters the root lockfile that `tests/deps_purity.rs` checks. Details in `docs/GRAPH.md`,
section 7.3.

```bash
cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --locked
```
