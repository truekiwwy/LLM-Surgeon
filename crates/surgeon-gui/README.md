# surgeon-gui: LLM Surgery Studio

Native viewer of the weight graph written by `surgeon graph`, and a surgery workbench, in
pure Rust (egui, OpenGL through glow). No HTML, no browser, no network.

```bash
# the studio, empty canvas
cargo run --release
# one state, opened on the 3D view
cargo run --release -- ../../graph-out/graph.json
# a GraphPatch stream (surgeon graph --format ndjson), followed live as lines are appended
cargo run --release -- ../../graph-out/graph.ndjson
# NDJSON on stdin (an orchestrator, or an agent driving the MCP server)
my-orchestrator | cargo run --release -- -
```

## Workspaces

The top bar switches between `Studio (Canvas)`, `Graphe 3D` and `Split (Canvas + 3D)`, and
opens the `Éditeur JSON` drawer on the right in any of them.

- `Graphe 3D`: layer list (left), 3D viewport (center), inspector (right). Left drag orbits,
  right drag pans, wheel zooms, `R` resets. `Colonne` / `Constellation`, `Diff (Δ)`, GPU
  seen by `llama-server`, configured backend, `Backend…` (user `config.toml`).
- `Studio (Canvas)`: the infinite card canvas. Wheel or Ctrl + wheel zooms on the pointer,
  middle drag or Space + drag pans, right click adds a card, `Suppr` deletes the selected
  one, `Recentrer` frames everything, the minimap (bottom right) moves the view.
- `Split (Canvas + 3D)`: both, side by side.
- Bottom: the running job (step, elapsed time, `Annuler`), and for a stream the patch
  timeline; at the last state the view follows new patches.

## Cards

Each card is one `surgeon` command, run in process through `weight_surgeon::app::run_with`:
same validation, same verified backups, same rollback as the CLI. A card's output port
(right) feeds the input port (left) of the next one; wires are checked (types, no loop).

| Card | Command | Shows |
| --- | --- | --- |
| Model | `inspect` (gate, up, down, o_proj, embed_tokens) | architecture, layers, targetable tensors |
| Extract | `extract --method <id> [--rank k]` | output file, layer |
| Patch | `apply --dry-run`, or `apply` when `Run écrit le modèle` is ticked | tensors touched, backups |
| Graph Diff | `graph --baseline` | nodes changed; `Voir en 3D` loads the `graph.json`, turns on the diff and frames the changes |
| Eval / Gate | `bench refusal --fail-on-warn --backend native/mistralrs/hipfire/llama-server` | GO when the bench passes and the refusal rate is at most `refus max`, else NO-GO |

`Run` runs the card after every card upstream of it, in wiring order; the first failure
stops the job, `Annuler` raises the flag the library checks between tensors and prompts. The
`...` menu duplicates a card, forks the branch below it (the copy keeps the same upstream, to
compare `alpha = 0.8` and `alpha = 1.2` side by side), unplugs its inputs or deletes it.
`Enregistrer` / `Ouvrir` write and read the workspace as `*.surgeon-studio.json` (cards,
wires, pan and zoom; results are not saved).

The patch operations are those of `patch.json`: `subtract`, `add`, `project_out`. Untying
`lm_head` happens by itself when a patch touches tied `embed_tokens`; merging a LoRA
adapter stays the `surgeon lora merge` command.

## Patch editor

- Monospace text with line numbers and sober JSON coloring; `Importer` / `Exporter` read
  and write the `fichier` field, `Ctrl+S` saves, `Ctrl+Enter` runs a dry run.
- 250 ms after the last keystroke the text goes through `patch::parse_patch` and
  `patch::validate_patch`; every error or warning is painted on the line its JSON path
  points at (`operations[0].alpha`).
- As soon as an operation's `layers` and `projections` parse, the matching nodes are ringed
  in the 3D view (the selected Patch card does the same when the drawer is closed).
- `Dry-Run` writes nothing; `Apply (avec backup)` refuses a patch with `"backup": false`;
  `Rollback` restores the files of the last apply from their `.bak-<timestamp>` copies.

## Build

The crate has its own `Cargo.lock`, like `crates/surgeon-tui`: its windowing stack never
enters the root lockfile that `tests/deps_purity.rs` checks. Details in `docs/GRAPH.md`,
section 7.3.

```bash
cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --locked
```
