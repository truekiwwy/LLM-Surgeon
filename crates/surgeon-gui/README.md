# surgeon-gui

Native viewer of the weight graph written by `surgeon graph`, in pure Rust (egui, OpenGL
through glow). No HTML, no browser, no network.

```bash
# one state
cargo run --release -- ../../graph-out/graph.json
# a GraphPatch stream (surgeon graph --format ndjson), followed live as lines are appended
cargo run --release -- ../../graph-out/graph.ndjson
# NDJSON on stdin
my-orchestrator | cargo run --release -- -
```

- Top bar: model, `Colonne` / `Constellation`, `Diff (Δ)`, GPU seen by `llama-server`,
  configured backend, `Backend…`.
- Left: text filter, role filters (`attn`, `mlp`, `embed`), nodes by layer.
- Center: 3D viewport. Left drag orbits, right drag pans, wheel zooms, `R` resets.
- Right: inspector (shape, certified spectral bounds, warnings, `Baseline vs Actuel (Δ)`).
- Bottom (streams only): patch timeline; at the last state the view follows new patches.
- `Backend…`: the inference backend modal, saved to the user `config.toml` the CLI reads.
  It opens by itself when that file does not exist yet.

The crate has its own `Cargo.lock`, like `crates/surgeon-tui`: its windowing stack never
enters the root lockfile that `tests/deps_purity.rs` checks. Details in `docs/GRAPH.md`,
section 7.3.

```bash
cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --locked
```
