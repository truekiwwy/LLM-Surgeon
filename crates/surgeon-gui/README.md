# surgeon-gui: LLM Surgery Studio

Native viewer of the weight graph written by `surgeon graph`, and a surgery workbench, in
pure Rust (egui, OpenGL through glow). No HTML, no browser, no network.

```bash
# the home screen (and, on a fresh profile, the first-run wizard)
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

No argument opens the **home screen**: open a graph, a workspace or a folder (typed path or
drag and drop), see the state of each dependency, replay the **first-run wizard** (machine
diagnostic, llama.cpp install with verified hashes, workspace, optional sample model), open
**Diagnostics** (repair, reinstall, redacted export) or check for a new version (manual, never
silent). A second launch hands its arguments to the running window.

```bash
cargo run --release -- --self-test            # engine, OpenGL window, data folder; exit code 0 / 1
cargo run --release -- --screenshot home.png  # renders the screen off-screen (CPU, no window)
cargo run --release -- --reset-settings       # the wizard runs again at the next start
```

Packaging (per-user installer, portable ZIP, checksums, optional signing) and troubleshooting:
[../../docs/DESKTOP.md](../../docs/DESKTOP.md). Crates and why: [../../docs/GUI_DEPENDENCIES.md](../../docs/GUI_DEPENDENCIES.md).

## Workspaces

The top bar switches between `Studio (Canvas)`, `Graphe 3D`, `Split (Canvas + 3D)`,
`Projets` and `Agents`, and opens the `Éditeur JSON` drawer on the right in any of them.

- `Graphe 3D`: layer list (left), 3D viewport (center), inspector (right). Left drag orbits,
  right drag pans, wheel zooms, `R` resets. `Colonne` / `Constellation`, `Diff (Δ)`, GPU
  seen by `llama-server`, configured backend, `Backend…` (user `config.toml`).
- `Studio (Canvas)`: the infinite card canvas. Wheel or Ctrl + wheel zooms on the pointer,
  middle drag or Space + drag pans, right click adds a card, `Suppr` deletes the selected
  one, `Recentrer` frames everything, the minimap (bottom right) moves the view.
- `Split (Canvas + 3D)`: both, side by side.
- `Projets`: the projects of the workspace (create one on a source model, which is hashed
  and never written; open one), the open project (state, source, `Copie de travail`, a
  resume or acknowledge banner after a crash or a failure), the launch form of any engine
  tool (generated from its MCP schema; `{out}` is the run's output folder; priority,
  `durée max`, `candidats max` for a pipeline), the running and queued jobs with `Annuler`,
  the followed run (progress, node states, metrics, artifacts, log, replayed from its
  `events.ndjson` once finished), the history and the ledger exports. `Graphe éphémère`
  opens a `graph.json` / `graph.ndjson` in the 3D view without a project. While a job runs,
  the 3D view rings the nodes it has reached. Model, guarantees and event protocol:
  [docs/STUDIO.md](../../docs/STUDIO.md).
- `Agents`: connections (Anthropic, OpenAI, any OpenAI-compatible endpoint such as
  OpenRouter, Ollama, LM Studio or `llama-server`; the Claude Code and Codex CLIs as signed
  in by the user), API keys in the Windows Credential Manager, a consent before the first
  request, and for the open project a conversation whose tool calls become jobs of the
  project, approval cards computed by the code, budgets, pause and stop, and an evidence log
  (Measured / Inferred / Unknown). Threat model and details: [docs/AGENTS.md](../../docs/AGENTS.md).
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
