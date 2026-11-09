# LLM Surgeon (`surgeon`)

[Version française](README.fr.md)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![rust](https://img.shields.io/badge/rust-2021_edition-orange)
![no C toolchain](https://img.shields.io/badge/dependencies-no_C_compiler-success)
![MCP](https://img.shields.io/badge/MCP-stdio_server-purple)

`surgeon` inspects, patches, evaluates and steers the MLP weights of local language models
stored as SafeTensors. It runs on the CPU, in pure Rust, and talks to humans (colored
terminal, dashboard) as well as to AI agents (`--json`, MCP server).

The terminal messages of the tool are in French. Every output below is a real run on a
tiny fake Qwen2 model (4 layers, hidden size 16), copied without ANSI codes.

## Architecture

Eight pillars, one binary:

```text
                          surgeon (src/main.rs -> app::run_with)
   ┌────────┬─────────┬──────────┬──────────┬─────────┬─────────┬────────┬──────────────┐
   check    inspect   apply      eval       extract   bench     tui      mcp / schema
   │        │         │          │          │         │         │        │
   hardware engine    patch      eval       extract   bench     tui      mcp
   RAM,VRAM mmap      patch.json llama.cpp  Qwen2     KL, PPL,  ratatui  JSON-RPC
   budget   stats     backup     loops,     forward   Pass@1,   worker   stdio,
            (rayon)   rollback   GGUF       (candle)  dumps     + LRU    sandbox
   └────────┴────┬────┴──────────┴──────────┴────┬────┴─────────┴────────┴──────────────┘
                 engine::stats (shared)          ui (terminal identity)
```

| Pillar | Module | What it guarantees |
| --- | --- | --- |
| Hardware budget | `src/hardware` | RAM and VRAM needs are computed before any load. |
| Read and shift | `src/engine` | Memory-mapped reads; statistics in two parallel passes, deterministic. |
| Patching | `src/patch` | `patch.json` validated first, SHA-256 checked backup, automatic rollback. |
| Evaluation | `src/eval` | Loop detection and expectations through llama.cpp, GGUF export only when healthy. |
| Extraction | `src/extract` | Steering vector from a Qwen2 layer, atomic output plus a companion stats file. |
| Measurement | `src/bench` | Logits dumps and KL/top-k shifts, perplexity delta, Pass@1; deterministic reports. |
| Linear algebra | `src/linalg` | Jacobi, thin SVD through the Gram matrix, Cholesky: pure Rust, f64, thread-count independent. |
| Dashboard | `src/tui` | Elm-style state, pure rendering, background worker with cancellation. |
| Agents | `src/mcp` | MCP server on stdio, paths confined to allowed directories. |

## Installation

Requirements: a Rust toolchain (tested with Rust 1.97) and the platform linker (MSVC Build
Tools on Windows). No dependency compiles C code: `tests/deps_purity.rs` fails if one does.

```bash
git clone https://github.com/truekiwwy/LLM-Surgeon.git
cd LLM-Surgeon
cargo install --path .
surgeon --version
```

```text
surgeon 0.1.0
```

The binary goes to `$HOME/.cargo/bin` (Linux, macOS) or `%USERPROFILE%\.cargo\bin`
(Windows). `surgeon eval` also needs a built llama.cpp (`llama-cli`, `llama-quantize`,
`convert_hf_to_gguf.py` and its Python requirements); `surgeon bench ppl` needs
`llama-perplexity`, `surgeon bench eval` needs `llama-completion` (or `llama-cli`).

## Tutorial

The commands run from `<MODELS_DIR>`, the folder that holds your models. The example model
is `qwen2-mini`; `pos.json` and `neg.json` are two JSON arrays of six short sentences.

```bash
cd <MODELS_DIR>
```

### 1. Check the machine

```bash
surgeon check --model qwen2-mini
```

```text
[CHECK]    weight-surgeon
────────────────────────────────────────────────────────────
  modèle        qwen2-mini
  taille        54.2 KiB
  RAM           4.66 GiB libres
  GPU           aucun
  besoin        RAM 2.00 GiB · VRAM n/a
[VERDICT]  FITS IN RAM ONLY
  ! NVML indisponible, GPU NVIDIA ignorés : the current user does not have permission to perform this operation
  ! aucun GPU détecté ; chargement en RAM (CPU), plus lent
```

The verdict is `FITS IN VRAM`, `FITS IN RAM ONLY` or insufficient (exit code 1).

### 2. Inspect the MLP weights

```bash
surgeon inspect --model qwen2-mini --layers 0-1 --show-stats
```

```text
█▀█ █ █ ▀█▀  W E I G H T
█ █ ▀▄▀  █   S U R G E O N
[INSPECT]  6 tenseurs sélectionnés
────────────────────────────────────────────────────────────
COUCHE   PROJ        DTYPE   SHAPE     TAILLE    MIN       MAX      MOYENNE   L2    
[L0]     gate_proj   f32     32 × 16   2.0 KiB   -0.3996   0.3994   0.0184    5.0572
[L0]     up_proj     f32     32 × 16   2.0 KiB   -0.3982   0.3991   -0.0120   5.4108
[L0]     down_proj   f32     16 × 32   2.0 KiB   -0.3996   0.3949   -0.0016   4.9931
[L1]     gate_proj   f32     32 × 16   2.0 KiB   -0.3998   0.3992   -0.0098   5.0053
[L1]     up_proj     f32     32 × 16   2.0 KiB   -0.3997   0.3984   0.0010    5.2173
[L1]     down_proj   f32     16 × 32   2.0 KiB   -0.3983   0.3968   -0.0043   5.3337
```

`--layers` takes `all`, `10-20` or `3,5,7`; `--proj` takes `gate`, `up` or `down` and can be
repeated. With `--json`, `check`, `inspect`, `apply`, `eval` and `extract` print a single
machine-readable envelope instead:

```bash
surgeon --json inspect --model qwen2-mini --layers 2 --proj down
```

```text
{"ok":true,"command":"inspect","exit_code":0,"data":{"shards":["qwen2-mini\\model.safetensors"],"tensors":[{"dtype":"F32","name":"model.layers.2.mlp.down_proj.weight","shape":[16,32],"shard":"qwen2-mini\\model.safetensors","stats":null}]},"error":null,"warnings":[]}
```

### 3. Extract a steering vector

`extract` runs the Qwen2 forward pass on both prompt sets and writes
`v = normalize(mean(pos) - mean(neg))` measured at one layer. Only Qwen2 and Qwen2.5
(`model_type: "qwen2"`) are supported.

```bash
surgeon extract --model qwen2-mini --pos-prompts pos.json --neg-prompts neg.json --layer 2 --out direction.safetensors
```

```text
[EXTRACT]  Qwen2 · couche 2 · resid_post · pooling last  qwen2-mini
[LOAD]     poids chargés 37/37 tenseurs · f32 · 34.9 KiB
[FORWARD]  pos 6/6 · neg 6/6
[VERDICT]  d′ 2.23 · accuracy 92% · ‖Δ‖ 3.378
[WRITE]    direction.safetensors
  ! seulement 6 prompts positifs (au moins 8 recommandés)
  ! seulement 6 prompts négatifs (au moins 8 recommandés)
```

`direction.safetensors` holds one 1D F32 tensor named `direction` and metadata (layer, site,
pooling, `d_prime`, `threshold_accuracy`, `pos_mean_proj`, `neg_mean_proj`,
`cos_mean_pos_neg`...). Next to it, `direction.safetensors.stats.json` keeps the projection of
every prompt, which the dashboard draws:

```text
{
  "schema": 1,
  "pos_projections": [
    1.3255219384032007,
    1.8297537791558263,
    1.7547985509917572,
    2.9448108869344285,
    1.5728345316552308,
    -1.6373556878039484
  ],
  "neg_projections": [
    -1.859846032281605,
    -3.9873652078601287,
    -0.9645802902590022,
    0.11087123832198972,
    -2.821489328449643,
    -2.9534084283447717
  ],
  "threshold": 0.7181965883625951,
  "d_prime": 2.2301346237037065,
  "threshold_accuracy": 0.9166666666666666,
  "cos_mean_pos_neg": 0.4886781739078529,
  "raw_diff_norm": 3.377697008034943,
  "layer": 2,
  "site": "resid_post",
  "pooling": "last",
  "hidden_size": 16
}
```

Both files are written atomically. An existing output, or an existing companion file, needs
`--overwrite`. On error or Ctrl-C, nothing is left behind.

### 4. Apply a patch

`patch.json` subtracts half of the vector from the `down_proj` of layers 1 and 2. The vector
lives in the residual stream (size `hidden_size`): it indexes the rows of `down_proj`
(`[hidden, intermediate]`), hence `axis: "cols"`; for `gate_proj` and `up_proj` use
`axis: "rows"`.

```json
{
  "schema_version": 1,
  "operations": [
    { "id": "steer-down", "layers": "1-2", "projections": ["down_proj"],
      "operation": "subtract", "vector_file": "direction.safetensors",
      "vector_tensor": "direction", "alpha": 0.5, "axis": "cols" }
  ]
}
```

Check first, then apply. The backup folder must already exist (default: next to the model).

```bash
surgeon apply --model qwen2-mini --patch patch.json --dry-run
surgeon apply --model qwen2-mini --patch patch.json --backup-dir backups
```

```text
[APPLY]    patch.json · 1 opérations
[DRY-RUN]  steer-down · 2 tenseurs · vecteur de 16
           model.layers.1.mlp.down_proj.weight
           model.layers.2.mlp.down_proj.weight
```

```text
[APPLY]    patch.json · 1 opérations
[BACKUP]   backups\model.safetensors.bak-1794165928  ✓ sha256 vérifié
[SHIFT]    steer-down terminée · Δmax 0.2570 · 2 tenseurs
[COMMIT]   modèle patché · 0.0 s
```

The full format: `surgeon schema` prints the JSON Schema (draft-07) of `patch.json`.

### 5. Evaluate, then export to GGUF

`eval` converts the model to GGUF with llama.cpp, runs each prompt through `llama-cli`,
detects loops and checks the expectations; the GGUF export only happens when the model is
healthy (exit code 5 otherwise). The prompts file:

```json
[{"id": "greet", "text": "Bonjour", "expect_not_contains": ["error"]}]
```

```bash
surgeon eval --model qwen2-mini --prompts prompts.json --llama-cpp-dir <LLAMA_CPP_DIR> --export-gguf qwen2-mini.gguf --quant Q4_K_M
```

This step was not run for this README: the machine had no llama.cpp. Without
`--llama-cpp-dir` (or `LLAMA_CPP_DIR`), `eval` stops before running anything:

```bash
surgeon eval --model qwen2-mini --prompts prompts.json
```

```text
erreur : --llama-cpp-dir (ou LLAMA_CPP_DIR) requis pour convertir en GGUF
```

### 6. Explore in the dashboard

```bash
surgeon tui --model qwen2-mini --vector direction.safetensors
```

| Key | Action |
| --- | --- |
| `Tab`, `Shift+Tab` | switch tab |
| `↑` `↓` | layer (tab 1), scroll the top-10 (tab 2) |
| `PgUp` `PgDn`, `Home` `End` | jump between layers (tab 1) |
| `←` `→` | projection (tab 1), pan the components (tab 2) |
| `+` `-` | zoom in or out on the components (tab 2) |
| `l` | linear or logarithmic histogram (tab 1) |
| `?` | help |
| `q`, `Esc`, `Ctrl-C` | quit |

Statistics are computed in a background thread (cancelled when the selection changes,
64 entries cached). A tensor with NaN or Inf shows a blinking red alert and a `⚠` in the
layer list. Without a terminal (CI, pipe), `--snapshot` prints one frame as plain text:

```bash
surgeon tui --snapshot --model qwen2-mini --vector direction.safetensors
```

```text
┌ surgeon tui ─────────────────────────────────────────────────────────────────────────────────────┐
│ 1 Inspecteur de poids │ 2 Steering vector                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
┌ Couches (4) ───────────┐┌ Métriques ─────────────────────────────────────────────────────────────┐
│[gate]│ up │ down │     ││tenseur         model.layers.0.mlp.gate_proj.weight                     │
│▶L0                     ││dtype · shape   F32 · [32, 16] · 2048 octets                            │
│ L1                     ││min · max       -0.399561 · 0.399415                                    │
│ L2                     ││moyenne         1.841966e-2                                             │
│ L3                     ││écart-type      2.227406e-1 (population)                                │
│                        ││sparsité        0.000 % de zéros exacts                                 │
│                        ││NaN · ±Inf      0 NaN · 0 +Inf · 0 -Inf                                 │
│                        ││éléments finis  512 / 512                                               │
│                        ││norme L2        5.057248                                                │
│                        │└────────────────────────────────────────────────────────────────────────┘
│                        │┌ Distribution des valeurs finies (linéaire) ────────────────────────────┐
│                        ││                                                   ██                   │
│                        ││                                                 ▁ ██                   │
│                        ││                 ▂    ▂                      ▂▂  █ ██         ▂      ▂  │
│                        ││               ▂▂█    █   ▂                  ██  █ ██   ▂     █      █  │
│                        ││               ███    █ ▃ █  ▃▃ ▃  ▃         ██  █ ██   █  ▃  █      █  │
│                        ││      ▃▃       ███    █ █ █  ██ █▃ █    ▃   ▃██  █ ██   █  █  █      █▃▃│
│                        ││   ▄▄ ██      ▄███    █ █▄█  ██ ██ █  ▄ █   ███ ▄█▄██▄▄▄█  █ ▄█▄▄    ███│
│                        ││   ██ ██   ▄  ████  ▄ █▄███▄▄██ ██ █  █▄█▄  ███ █████████  █ ████ ▄  ███│
│                        ││   ██ ██   █▅▅████  █▅█████████▅██▅█  ████▅ ███▅█████████▅▅█▅████ █▅▅███│
│                        ││   ██▅██   ███████  ████████████████  █████ █████████████████████▅██████│
│                        ││▆▆ █████▆  ███████  ████████████████  █████▆████████████████████████████│
│                        ││██▆██████  ███████▆▆████████████████▆▆██████████████████████████████████│
│                        ││█████████▇▇█████████████████████████████████████████████████████████████│
│                        ││████████████████████████████████████████████████████████████████████████│
│                        ││-3.9956e-1                                                     3.9942e-1│
└────────────────────────┘└────────────────────────────────────────────────────────────────────────┘
Tab onglet · ↑↓ couche · ←→ projection · l log · ? aide · q quitter  qwen2-mini  prêt
```

With `--vector` alone, the dashboard opens on tab 2 (vector, top-10, components, d′ and
accuracy gauges, projections of every prompt around the threshold):

```bash
surgeon tui --snapshot --vector direction.safetensors
```

```text
┌ surgeon tui ─────────────────────────────────────────────────────────────────────────────────────┐
│ 1 Inspecteur de poids │ 2 Steering vector                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
┌ Vecteur ───────────────────────────────────────┐┌ d′ (séparation) ───────────────────────────────┐
│taille          16                              ││██████████████████████2.23                      │
│norme L2        1.000  normalisé ✓              │└────────────────────────────────────────────────┘
│min · max       -0.5069 · 0.5141                │┌ accuracy au seuil ─────────────────────────────┐
│moyenne         -5.3450e-2                      ││██████████████████████92 % █████████████████    │
│                                                │└────────────────────────────────────────────────┘
└────────────────────────────────────────────────┘┌ Séparation ────────────────────────────────────┐
┌ TOP-10 |v_i| ──────────┐┌ Valeurs (linéaire) ──┐│proj. μ₊ · μ₋   1.2984 · -2.0793                │
│ 1. dim 15   +0.5141    ││        █             ││cos(μ₊, μ₋)     0.4887                          │
│ 2. dim 2    -0.5069    ││        █             ││couche · site   2 · resid_post                  │
│ 3. dim 4    -0.4161    ││        █             ││pooling         last                            │
│ 4. dim 6    -0.2963    ││        ███           ││n_pos · n_neg   6 · 6                           │
│ 5. dim 3    +0.2675    ││        ███           ││modèle source   qwen2-mini                      │
│ 6. dim 5    +0.2039    ││        ███           ││                                                │
│ 7. dim 9    -0.1790    ││█ █ █  █████ █ █ █   █│└────────────────────────────────────────────────┘
│ 8. dim 1    -0.1515    ││█ █ █  █████ █ █ █   █│┌ Projections : pos ▲ / neg ▼ (┼ seuil) ─────────┐
│ 9. dim 12   -0.1156    ││█ █ █  █████ █ █ █   █││                █                   █ ███      █│
│10. dim 10   -0.1128    ││-5.0694e-1   5.1405e-1││                █                   █ ███      █│
└────────────────────────┘└──────────────────────┘│                █                   █ ███      █│
┌ Composantes 0..16 (zoom ×1) ───────────────────┐│                █                   █ ███      █│
│               █                                ││────────────────────────────────┼───────────────│
│   █ █         █                                ││█      ██     █     █       █                   │
│█  █ █  █      █                                ││█      ██     █     █       █                   │
│────────────────────────────────────────────────││█      ██     █     █       █                   │
│ ██ █ ██ ██████                                 ││█      ██     █     █       █                   │
│  █ █ █  █                                      ││seuil 0.7182                                    │
│  █ █                                           ││                                                │
└────────────────────────────────────────────────┘└────────────────────────────────────────────────┘
Tab onglet · ←→ défiler · +/- zoom · ↑↓ top-10 · ? aide · q quitter  direction.safetensors
```

With both files, tab 2 also says `compatible ✓` or `incompatible ✗ (4096 ≠ 3584)` when the
vector size differs from the model hidden size. `--size 120x40` changes the snapshot size.
A vector extracted before this version has no companion file: tab 2 then explains that the
projections are unavailable.

### 7. Hand it to an AI agent (MCP)

`surgeon mcp` serves seven tools on stdio (`surgeon_check`, `surgeon_inspect`,
`surgeon_apply`, `surgeon_eval`, `surgeon_extract`, `surgeon_bench`, `surgeon_autotune`).
Every path must fall
under an allowed directory, and a verified backup is imposed before any write. The dashboard
is not exposed through MCP.

```bash
surgeon mcp --allow-dir <MODELS_DIR> --read-only
```

Claude Code:

```bash
claude mcp add surgeon -- surgeon mcp --allow-dir <MODELS_DIR>
```

Claude Desktop (`claude_desktop_config.json`):

```json
{ "mcpServers": { "surgeon": { "command": "surgeon", "args": ["mcp", "--allow-dir", "<MODELS_DIR>"] } } }
```

A real session: the client sends `initialize`, `notifications/initialized`, then a
`tools/call` of `surgeon_check` on `qwen2-mini`. The server writes one line on stderr, then
answers on stdout (the `initialize` answer is left out here):

```text
surgeon mcp 0.1.0 : prêt sur stdio, 1 dossier(s) autorisé(s), lecture seule
{"id":2,"jsonrpc":"2.0","result":{"content":[{"text":"{\"ok\":true,\"command\":\"check\",\"exit_code\":0,\"data\":{\"available_ram_bytes\":4780371968,\"best_gpu\":null,\"model_bytes\":55528,\"required_ram_bytes\":2147547506,\"required_vram_bytes\":null,\"verdict\":{\"kind\":\"fits_in_ram_only\"},\"warnings\":[\"NVML indisponible, GPU NVIDIA ignorés : the current user does not have permission to perform this operation\",\"aucun GPU détecté ; chargement en RAM (CPU), plus lent\"]},\"error\":null,\"warnings\":[]}","type":"text"}],"isError":false,"structuredContent":{"command":"check","data":{"available_ram_bytes":4780371968,"best_gpu":null,"model_bytes":55528,"required_ram_bytes":2147547506,"required_vram_bytes":null,"verdict":{"kind":"fits_in_ram_only"},"warnings":["NVML indisponible, GPU NVIDIA ignorés : the current user does not have permission to perform this operation","aucun GPU détecté ; chargement en RAM (CPU), plus lent"]},"error":null,"exit_code":0,"ok":true,"warnings":[]}}}
```

Tool schemas: `surgeon schema --kind tools`. Details, guards and protocol versions:
[docs/MCP.md](docs/MCP.md) (in French).

### 8. Measure a surgery (`surgeon bench`)

Four modes measure what a patch changed. The recipe an agent follows: extract a direction
(step 3), apply it **on a copy** of the model (step 4), dump the logits of both models on
the same prompts, compare them, then check the perplexity and an exact-match benchmark
before deciding. Here `patch.json` subtracts `0.5 × [1, …, 1]` from the `gate_proj` of
layer 1, without backup since it works on a copy:

```json
{"schema_version": 1, "backup": false, "operations": [{"id": "g1", "layers": "1", "projections": ["gate_proj"], "operation": "subtract", "vector_file": "v16.json", "alpha": 0.5, "axis": "rows"}]}
```

```bash
cp -r qwen2-mini qwen2-patched
surgeon apply --model qwen2-patched/model.safetensors --patch patch.json
surgeon bench dump --model qwen2-mini --prompts pos.json --out before.safetensors
surgeon bench dump --model qwen2-patched --prompts pos.json --out after.safetensors
surgeon bench logits --before before.safetensors --after after.safetensors --warn-kl 0.05
```

```text
[APPLY]    patch.json · 1 opérations
[SHIFT]    g1 terminée · Δmax 0.5000 · 1 tenseurs
[COMMIT]   modèle patché · 0.0 s
[BENCH]    dump · 6 prompts · 24 positions · vocab 64 · f32
[WRITE]    before.safetensors
[BENCH]    dump · 6 prompts · 24 positions · vocab 64 · f32
[WRITE]    after.safetensors
[BENCH]    logits · 24 positions · vocab 64
[KL]       moyenne 0.1648 · max 0.4785 @ 23 · p95 0.4187 · cos(p,q) 0.8684
[TOP-K]    top-1 changé 66.7 % · top-5 (ensemble) 95.8 % · recouvrement 0.51
  pires prompts : #1 0.2656 · #2 0.2572 · #3 0.1641 · #5 0.1311 · #4 0.0934
[VERDICT]  avertissement : KL moyenne 0.1648 > 0.05
```

`bench dump` runs the whole Qwen2 forward in pure Rust and writes `logits`
`[positions, vocab]`, `prompt_offsets` and metadata (`prompts_sha256`, `tokenizer_sha256`...)
atomically. `bench logits` also reads `[batch, seq, vocab]` tensors, f32/f16/bf16, `.npy`
and `.pt` files, and warns when the two dumps come from different prompts or tokenizers.

`bench ppl` and `bench eval` drive llama.cpp (a SafeTensors directory is converted to a
temporary GGUF, removed afterwards). The two outputs below come from
`examples/fake_llama.rs` copied as `llama-perplexity` and `llama-completion`: it prints a
perplexity derived from the model bytes and answers the sum of the numbers of each problem,
so these runs exercise the real process handling, not a real model.

```bash
surgeon bench ppl --model after.gguf --text text.txt --baseline-model before.gguf --llama-cpp-dir llama
surgeon bench eval --model after.gguf --dataset ds.json --baseline-model before.gguf --llama-cpp-dir llama --max-new-tokens 64 --min-accuracy 0.5
```

```text
[BENCH]    ppl · ctx 512  after.gguf
[PPL]      6.0930 ± 0.04 → 7.2870 ± 0.04 · Δ +1.1940 (hors du bruit)
[VERDICT]  avertissement : ΔPPL 1.1940 > 0.2
[BENCH]    eval · 4 questions · auto  ds.json
[PASS@1]   1/4 · 25.0 % · IC95 [4.6 ; 69.9] · boucles 0
[PASS@1]   référence 2/4 · 50.0 % · IC95 [15.0 ; 85.0] · boucles 0
  appariement : perdues 2 · gagnées 1 · Δ -25.0 points
[VERDICT]  avertissement : exactitude 25.0 % < minimum 50.0 %
  ! échantillon réduit (n = 4) : l'intervalle de confiance est large (± 32.7 points)
```

The dataset is a JSON array (or JSONL) of `question` (or `problem`), `answer` (or
`solution`) and an optional `id`; no GSM8K or MATH content ships with this repository:

```json
[{"id": "q1", "question": "What is 12 plus 30?", "answer": "12 + 30 = 42\n#### 42"},
 {"id": "q4", "question": "What is half of 9?", "answer": "Half of 9 is $\\boxed{\\frac{9}{2}}$."}]
```

Definitions used by the reports (`--json` gives every field):

- **KL**: `KL(before‖after) = Σ p_i (log p_i − log q_i)` with `p = softmax(before)` and
  `q = softmax(after)`, in f64 through log-sum-exp: the "before" distribution is the
  reference. Terms with `p_i = 0` are skipped; a token finite before and `−∞` after makes the
  position infinite, counted in `kl_infinite_positions` and kept out of `kl_mean`, `kl_max`
  and `kl_p95` (nearest rank).
- **Cosine**: `cosine_probs_mean` compares the probability vectors; `cosine_logits_mean`,
  labelled apart, the raw logits.
- **Top-k**: `top1_shift_rate` is the share of positions whose argmax changes (ties go to the
  lowest index); `topk_set_change_rate` the share whose set of `k` best tokens differs
  (`--top-k`, default 5), `topk_overlap_mean` the mean `|A∩B| / k`.
- **ΔPPL**: `ppl − ppl_baseline`, rounded to 4 decimals (what llama.cpp prints) before the
  strict comparison with `--warn-delta`: `7.4 − 7.2` is `0.20000000000000018` in floating
  point and does not exceed 0.2. `within_noise` when `|Δ| ≤ sqrt(err² + err_baseline²)`; a
  negative delta is an improvement, never a warning.
- **Pass@1**: exact match of one greedy sample (`--temp 0 --top-k 1 --seed 42`). The final
  answer is the first found of `#### X`, `\boxed{X}` (nested braces), the last `**X**`,
  "the answer is X", the last number; numbers and fractions compare with a relative
  tolerance of 1e-9 (`18` = `18.0` = `1,8e1`, `1/2` = `\frac{1}{2}`). The 95 % interval is
  Wilson's; with `--baseline-model` the comparison is paired (`lost`, `gained` ids).
- **Determinism**: reductions run in a fixed order, so `bench logits` gives the same bits
  with 1 or 12 threads; two runs give the same JSON apart from `timing_ms`.

A warning never changes the exit code, except a threshold (`--warn-kl`,
`--warn-top1-shift`, `--warn-delta`, `--min-accuracy`) strictly exceeded under
`--fail-on-warn`: exit code 9, the report still in `data`. Without a threshold the verdict
is `none`. Measured on a 12-thread laptop: `bench logits` on two f16 dumps of
4096 positions × 151,936 tokens takes 5.3 s with 49 MiB of private memory (the dumps are
memory-mapped); scoring 1,000 eval outputs takes 5 ms.

### 9. Representation engineering: `extract --method`, `project_out`, `autotune`

Theory first, with the exact formulas the code implements (all linear algebra is pure Rust,
f64, in `src/linalg`: cyclic Jacobi, thin SVD through the smaller Gram matrix, Cholesky).

- **`mean_diff`** (default, unchanged): `v = normalize(μ₊ − μ₋)` over the pooled activations.
- **`svd`** (contrastive PCA): prompts paired by index (`unpaired_prompts`, exit code 2,
  otherwise), `ΔX_i = x⁺_i − x⁻_i`, `v` = first right singular vector of ΔX, i.e. the
  dominant eigenvector of `(1/N) ΔXᵀΔX`, oriented so that `⟨v, μ₊ − μ₋⟩ ≥ 0`. ΔX is NOT
  centered by default: centering removes the mean delta, which is the concept axis itself.
  `--svd-center` is for pairs of arbitrary order (random signs). Diagnostics: σ₁..σ₅,
  `σ₁²/Σσ²`, cosine with `μ₊ − μ₋`, split-half cosine (even vs odd pairs); a warning when
  `σ₂/σ₁ > 0.9` or fewer than 8 pairs.
- **`leace`**: `r(x) = x − a bᵀ(x − μ)` with `Σ_λ = (1−λ)Σ + λ·tr(Σ)/d·I`,
  `w = Σ_λ^{-1/2}(μ₊ − μ₋)`, `u = w/‖w‖`, `a = Σ_λ^{1/2}u` (parallel to `μ₊ − μ₋`),
  `b = Σ_λ^{-1/2}u` (parallel to `Σ_λ⁻¹(μ₊ − μ₋)`), `bᵀa = 1`. `Σ^{-1/2}μ_Δ` alone is only the
  whitened direction `w`: the eraser needs both `a` (where to write) and `b` (what to read).
  `Σ_λ^p` is applied through the thin SVD of the centered data, never as a d×d matrix;
  `--leace-shrinkage 0` is the pseudo-inverse on the data subspace (`--leace-rcond`). The
  written `direction` is `a/‖a‖`, so it equals the `mean_diff` direction; `a`, `b`, `mean`
  and `u_whitened` go to `<out>.leace.safetensors`.
- **Honest limits of LEACE**: the cross-covariance `cov(r(x), z)` is zero in sample for any
  positive definite `Σ_λ`; "no linear predictor beats a constant" is a population guarantee.
  With n < d a classifier can still separate the erased sample: only `--holdout-frac`
  (refit on the rest, measure on the held-out prompts) says whether the erasure generalizes.
  Erasing a concept from the whole residual stream would need the eraser on every write
  into it; patching one layer's weights does not give that guarantee.
- **`project_out`** (patch operation): `W' = W − α·a(bᵀW)` on `down_proj`
  (`[hidden, intermediate]`, it writes into the residual stream), `α ∈ [0, 1]` (1 = full
  erasure of what this matrix writes along `b`). `o_proj` and the embeddings also write into
  the stream and are not covered.
- **`autotune`**: the naive `min ‖Y − Σ α_l W_l v_l‖²` leaves `Y` and `W_l v_l` undefined. The
  readout is `s_i = ⟨pool(h_i^R), v⟩` at layer R (site, pooling and template of the vector),
  the target `y_i = t_i − s_i⁰` (`neg_mean_proj` for `suppress`, `pos_mean_proj` for
  `amplify`, `--target-projection`, or a per-prompt `target`), the response
  `F_il = (s_i(δ_l) − s_i⁰)/δ` measured in memory with the engine's own shift on the stored
  bytes (bf16 rounding included), then `α = (FᵀF + λI)⁻¹Fᵀy` by Cholesky, with `λ` fixed or
  chosen by closed-form LOOCV (`e_i/(1 − h_ii)`, 17 points from 1e-6 to 1e2). R² is
  uncentered (`1 − Σe²/Σy²`, there is no intercept). The fitted `α` are applied together
  and measured again. Side effects are not measured: run `surgeon bench` afterwards.

Real runs on the tiny fake Qwen2 (6 prompts per set, hidden size 16):

```bash
surgeon extract --model qwen2-mini --pos-prompts pos.json --neg-prompts neg.json --layer 2 --method svd --out svd.safetensors
surgeon extract --model qwen2-mini --pos-prompts pos.json --neg-prompts neg.json --layer 2 --method leace --out erase.safetensors
```

```text
[VERDICT]  d′ 1.21 · accuracy 83% · ‖Δ‖ 3.378
[WRITE]    svd.safetensors
[METHOD]   svd · 6 paires · σ₁²/Σσ² 0.393 · cos(μ₊−μ₋) 0.812 · split-half 0.412
  ! seulement 6 paires (au moins 8 recommandées) : direction SVD peu fiable
[VERDICT]  d′ 2.23 · accuracy 92% · ‖Δ‖ 3.378
[WRITE]    erase.safetensors
[METHOD]   leace · rang 11 · λ 0.1 · κ 5.033e1 · cov(x,z) 8.444e-1 → 9.449e-16 · hors-échantillon —
[WRITE]    erase.safetensors.leace.safetensors
  ! n = 12 ≤ d = 16 : la covariance croisée s'annule en échantillon, mais un classifieur linéaire peut encore séparer ; seul holdout_cross_cov_ratio dit si l'effacement généralise
```

The `leace` d′ (2.23) is the `mean_diff` one: same direction, as the theory says. Applying
the eraser on a copy (`project.json` names the `a` and `b` tensors of the `.leace` file):

```json
{"schema_version": 1, "backup": false, "operations": [{"id": "erase-L1", "layers": "1", "projections": ["down_proj"], "operation": "project_out", "vector_file": "erase.safetensors.leace.safetensors", "vector_tensor": "a", "vector_tensor_b": "b", "alpha": 1.0}]}
```

```bash
surgeon apply --model qwen2-erased/model.safetensors --patch project.json --no-backup
surgeon autotune --model qwen2-mini --vector md.safetensors --eval-dataset pos.json --target-layer-range 0-3 --l2-reg auto --out tuned.json
```

```text
[PROJECT]  erase-L1 terminée · effacement a(bᵀW) · Δmax 0.3463 · 1 tenseurs
[AUTOTUNE] lecture L2 · resid_post · last · 6 prompts · 3 couches · 48 forwards
[RESPONSE] δ 0.05 · s⁰ moyen 1.2984 → cible -2.0793 · linéarité (écart max) 1.8 %
[RIDGE]    λ 1.0e2 · R² 0.0120 · RMSE 3.6374 · LOO 4.1374 · κ 1.32e0 · ‖α‖ 0.0658
  α : L0 +0.0183 · L1 -0.0596 · L2 -0.0210
[VERDICT]  décalage prévu -0.0371 · mesuré -0.0354 · écart à la cible 3.6594 → 3.6390
[WRITE]    tuned.json
  ! couches [3] au-delà de la lecture (couche 2) : sans effet sur elle, exclues
```

On this random mini model the responses do not generalize across prompts, so LOOCV picks
the strongest λ and the fit explains 1 % of the shift: that is the honest answer, and the
measured shift (−0.0354) matches the prediction (−0.0371). `tuned.json` is an ordinary
patch (`backup: true`, one `subtract` per layer) for `surgeon apply`.

Comparison of the three methods on SYNTHETIC data (`cargo run --release --example
bench_directions`): planted direction t, a "syntax" component shared by each pair,
anisotropic noise, concept strength `s = 0.5·√d` ("weak": `0.15·√d`), fixed seed; timings
are indicative. No gain on real models is claimed.

| scenario (synthetic) | pairs N | d | method | cos(v, t) | split-half | cos(b, W⁻¹t) | time (ms) |
|---|---:|---:|---|---:|---:|---:|---:|
| standard | 8 | 64 | mean_diff | 0.568 | 0.304 | — | 0.0 |
| standard | 8 | 64 | svd | 0.549 | -0.084 | — | 0.5 |
| standard | 8 | 64 | leace | 0.568 | 0.304 | 0.360 | 0.2 |
| standard | 32 | 64 | mean_diff | 0.889 | 0.689 | — | 0.0 |
| standard | 32 | 64 | svd | 0.828 | 0.497 | — | 1.0 |
| standard | 32 | 64 | leace | 0.889 | 0.689 | 0.597 | 4.1 |
| standard | 128 | 64 | mean_diff | 0.968 | 0.803 | — | 0.0 |
| standard | 128 | 64 | svd | 0.935 | 0.786 | — | 12.4 |
| standard | 128 | 64 | leace | 0.968 | 0.803 | 0.863 | 3.4 |
| standard | 32 | 1024 | mean_diff | 0.835 | 0.615 | — | 0.1 |
| standard | 32 | 1024 | svd | 0.842 | 0.622 | — | 2.7 |
| standard | 32 | 1024 | leace | 0.835 | 0.615 | 0.462 | 6.6 |
| standard | 128 | 1024 | mean_diff | 0.956 | 0.818 | — | 0.2 |
| standard | 128 | 1024 | svd | 0.957 | 0.821 | — | 61.0 |
| standard | 128 | 1024 | leace | 0.956 | 0.818 | 0.370 | 681.3 |
| weak | 32 | 1024 | mean_diff | 0.476 | 0.130 | — | 0.1 |
| weak | 32 | 1024 | svd | 0.406 | 0.054 | — | 2.5 |
| weak | 32 | 1024 | leace | 0.476 | 0.130 | 0.280 | 6.0 |
| weak | 128 | 1024 | mean_diff | 0.699 | 0.359 | — | 0.3 |
| weak | 128 | 1024 | svd | 0.655 | 0.211 | — | 98.2 |
| weak | 128 | 1024 | leace | 0.699 | 0.359 | 0.454 | 914.4 |

What this synthetic table shows: the pairing already cancels the shared "syntax" component
for `mean_diff`, and the uncentered SVD is pulled toward the noisiest axes, so `svd` does
not beat `mean_diff` here (they tie at d = 1024). The `leace` direction is the `mean_diff`
one by construction; its read direction `b` approaches the optimal `W⁻¹t` when N ≫ d and
stays far from it when n < d (shrinkage dominates).

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | Success |
| 1 | General failure: insufficient resources, unreadable model, missing llama.cpp binary, Ctrl-C, `tui` without an interactive terminal |
| 2 | Invalid input: arguments, patch, prompts, paths; nothing was touched |
| 3 | Execution failed; the model is intact (never touched, or restored from the backup) |
| 4 | Critical: rollback failed, or no backup and the model was left partially patched |
| 5 | `eval`: unhealthy model (loops, failed expectations); no export |
| 6 | `eval`: GGUF export failed |
| 7 | `extract` or `autotune`: invalid config, tokenizer or weights, degenerate direction, NaN/Inf, singular or ill-conditioned system |
| 8 | `bench`: run failed (llama.cpp process, unreadable output, incompatible dumps, generation) |
| 9 | `bench --fail-on-warn`: a threshold was exceeded (the report is in `data`) |
| 70 | Internal error (panic), reported after the terminal is restored |

## Commands

| Command | Role | Writes |
| --- | --- | --- |
| `surgeon check` | RAM and VRAM budget, verdict | nothing |
| `surgeon inspect` | MLP tensors, optional statistics | nothing |
| `surgeon apply` | Applies a `patch.json` in place (`--dry-run` first) | model, backup |
| `surgeon eval` | Non-regression through llama.cpp, conditional GGUF export | GGUF |
| `surgeon extract` | Steering vector from a Qwen2 layer | vector, companion `.stats.json` |
| `surgeon bench` | Logits dumps and KL/top-k, perplexity delta, Pass@1 | logits dump (`dump` only) |
| `surgeon autotune` | Per-layer coefficients by ridge on measured responses | patch.json (`--out` only) |
| `surgeon tui` | Dashboard: weight inspector and steering vector viewer | nothing |
| `surgeon schema` | JSON Schema of `patch.json`, or the MCP tool definitions | stdout only |
| `surgeon mcp` | MCP server on stdio | through its tools |

```bash
surgeon schema --kind patch
surgeon schema --kind tools
```

## Responsible use

Steering and weight surgery change what a model does, including its safety behaviors: a
direction can be used to remove refusals as easily as to add them. Work only on models whose
license allows modification, keep the backups, measure the effect with `surgeon eval` and
`surgeon bench` before sharing anything, and do not distribute a model whose safeguards you have removed.

## Known limits

- `extract` supports Qwen2 and Qwen2.5 only, on the CPU; large models need a lot of RAM
  (`--dtype bf16` halves it).
- `eval` and the GGUF export depend on an external llama.cpp build.
- `bench dump` supports Qwen2 only, on the CPU. `bench ppl` and `bench eval` depend on the
  flags of the installed llama.cpp (`llama-perplexity`, formerly `perplexity`; `--chunks`,
  `-ngl`, `--temp`, `--top-k`, `--seed`); GPU offload can change the last digits between runs.
- 50 questions (the `bench eval` default) make a wide interval: 40/50 gives
  [67.0 %, 88.8 %], so only large regressions are conclusive.
- `.pt` dumps are read on a best-effort basis: uncompressed `torch.save` archives,
  contiguous tensors only.
- `extract --method svd|leace` and `autotune` rely on Jacobi, limited to 1024 on the
  smaller side (prompts or hidden size). LEACE erasure is exact in sample only; with
  n < d check `holdout_cross_cov_ratio`. `project_out` covers `down_proj` only, so it does
  not erase a concept from the whole residual stream. `autotune` fits and measures on the
  same prompts and does not measure side effects.
- Only the MLP projections (`gate_proj`, `up_proj`, `down_proj`) are covered.
- The dashboard needs an interactive terminal of at least 80×24, with stdin, stdout and
  stderr attached to it; otherwise use `--snapshot`.
- Terminal messages are in French.
- No license file has been published yet: until one is added, default copyright applies
  (all rights reserved).

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `` `surgeon tui` nécessite un terminal interactif `` | Run it in a real terminal, or add `--snapshot`. |
| `dossier de backup inexistant` | Create the `--backup-dir` folder first. |
| `--llama-cpp-dir (ou LLAMA_CPP_DIR) requis` | Build llama.cpp and pass its folder. |
| `NVML indisponible` warning | Harmless: NVIDIA GPUs are ignored and the model is planned for RAM. |
| `cargo install` fails with "Access denied" on Windows | A `surgeon.exe` is still running: close it, or rename it to `surgeon.old.exe`, then reinstall. |
| Boxes or blocks look broken | Use a UTF-8 terminal with a font that has box-drawing characters (Windows Terminal, for example). |

## Contributing

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

`tests/deps_purity.rs` rejects any dependency that compiles C code, and
`tests/readme_check.rs` checks that every `surgeon` command and option quoted in both READMEs
exists. Commits follow `type: description` (`feat`, `fix`, `refactor`, `docs`, `chore`).

## Repository

```text
.
├── Cargo.toml
├── README.md, README.fr.md
├── docs/MCP.md            MCP server guide (French)
├── src/
│   ├── main.rs, lib.rs    entry point, modules
│   ├── app.rs, cli.rs     commands, exit codes, clap definitions
│   ├── ui.rs              terminal identity (colors, badges, tables)
│   ├── engine/            SafeTensors reads, shifts, shared statistics
│   ├── hardware/          RAM and VRAM probe, budget
│   ├── patch/             patch.json, validation, backup, rollback
│   ├── eval/              llama.cpp runner, loop detector, GGUF export
│   ├── extract/           Qwen2 forward, steering vector, companion file
│   ├── bench/             logits dumps and KL, perplexity, Pass@1 (GSM8K/MATH)
│   ├── linalg/            Jacobi, Gram SVD, power iteration, Cholesky (pure Rust)
│   ├── mcp/               JSON-RPC server, tools, sandbox
│   └── tui/               dashboard: app, worker, terminal, theme, widgets/
├── examples/              fake_llama.rs (stand-in llama.cpp), bench_directions.rs (synthetic)
└── tests/
    ├── integration.rs     end-to-end flows and binary smoke tests
    ├── deps_purity.rs     no C compiler in the dependency tree
    └── readme_check.rs    README commands and links
```
