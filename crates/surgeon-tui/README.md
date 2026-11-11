# surgeon-tui

Inspecteur terminal interactif de la **topologie spectrale des poids LLM** (Safetensors / GGUF),
et **dashboard visualiseur connecté aux CLI d'agents IA** (Claude Code, OpenAI Codex, Aider).

Crate autonome de [LLM-Surgeon](https://github.com/truekiwwy/LLM-Surgeon) : 100 % Rust pur,
mmap zéro-copie, aucune dépendance à un framework ML.

```text
┌───────────────────────────┬──────────────────────────────────────┬─────────────────────────┐
│ Couches (812)             │ model.layers.14.mlp.down_proj.weight │ Bridge agents           │
│  L14 · dwn blk.14.ffn…    │ rôle          down (dense)           │ [12:01:33] agent : ping │
│  L14 ◉ att blk.14.attn…   │ forme         [1024, 4096]           │ [12:01:35] agent : cou… │
│  L15 · gat …              │ dtype         Q6_K (6.56 bits/poids) │                         │
│                           │ ablation      Exact                  │ diff … α=0.5000         │
│  / recherche   r rôle     │ norme L2      31.42 (échantillon …)  │                         │
│                           │ r_eff         6.83 / 16              │ Ctrl+E · Ctrl+P         │
└───────────────────────────┴──────────────────────────────────────┴─────────────────────────┘
 ● blk.14.ffn_down.weight  down  Exact  L2 31.418 · kurt 0.31 · r_eff 6.83/16
```

## Installation et lancement

```bash
cargo install --path crates/surgeon-tui        # binaire surgeon-tui
surgeon-tui --model ~/models/qwen3-8b          # dossier checkpoint (.safetensors)
surgeon-tui --model ~/models/qwen3-8b.gguf     # fichier GGUF quantifié
surgeon-tui --model model.safetensors          # fichier unique
```

Options :

| Option | Rôle |
|---|---|
| `--model <PATH>` | `.safetensors`, dossier de checkpoint (shards + `config.json`) ou `.gguf` |
| `--ipc-socket <ENDPOINT>` | chemin de socket Unix ou nom de pipe Windows (défaut : `/tmp/surgeon-tui.sock` / `\\.\pipe\surgeon-tui`) |
| `--agent-command <TEMPLATE>` | commande de l'agent lancée par `Ctrl+P` ; `{context}` est remplacé par le fichier de contexte (défaut : `claude {context}`) |

## Interface

### Panneau gauche — arborescence
- Couches `L0..LN`, tag de rôle par tenseur (`emb`, `att`, `gat`, `up`, `dwn`, `rtr`),
  marqueurs MoE (`f` = experts fusionnés, `s` = expert partagé, `e` = expert individuel).
- `/` : filtre texte incrémental (le filtre suit la frappe). `Enter` valide, `Esc` efface.
- `r` : cycle de filtre par rôle (`embed`, `o_proj`, `gate`, `up`, `down`, `router`).

### Panneau central — inspecteur
- Carte du tenseur : rôle (embed, attn_out, mlp_gate, mlp_up, mlp_out, router), couche,
  expert, forme, dtype + bits/poids, octets et fichier.
- **Exactitude d'ablation** (`project_out`) : `Exact` (architectures pré-norm denses ou MoE),
  `Approximate` (post-norm apprises, type Gemma), `Unsupported` (famille inconnue). Les
  non-écrivains (gate, up, router) n'affichent pas de badge.
- Métriques statiques : norme L2 (remise à l'échelle), min/max, moyenne/|moyenne|,
  **kurtosis** (excès — poids des outliers), **rang effectif approximatif** `r_eff`
  (participation ratio des valeurs singulières² estimées par projection Rademacher
  déterministe sur un sous-échantillon, base 16) et **profil de magnitude des lignes**
  (sparkline). Chiffres calculés hors thread UI sur un échantillon déterministe
  (badge `échantillon n/m` quand il ne couvre pas tout le tenseur).
- Dtypes décodés : `F32`, `F16`, `BF16`, `Q8_0`, `Q4_K`, `Q5_K`, `Q6_K`. Les autres
  (IQ*, Q2_K, Q3_K…) sont listés avec leurs métadonnées, sans métriques.

### Panneau droit — bridge agents
- Journal d'événements horodaté de toutes les requêtes IPC reçues.
- Aperçu de patch proposé par un agent (`preview_patch`) : tenseur, α, rang 1.
- `Ctrl+E` : **export du contexte** — un snippet Markdown décrivant le tenseur sélectionné
  (identité, rôle, métriques, exactitude) écrit dans le répertoire temporaire, prêt à être
  collé dans Claude Code / Codex / Aider.
- `Ctrl+P` : **lancement de l'agent** — exécute la commande configurée (`--agent-command`)
  avec le fichier de contexte, en cédant l'écran au processus puis en restaurant la TUI.

### Raccourcis globaux
`↑/↓ j/k` naviguer · `PgUp/PgDn` ±10 · `g/G` début/fin · `Tab` changer de panneau ·
`/` recherche · `r` filtre rôle · `Esc` effacer les filtres / l'aperçu · `Ctrl+E` export ·
`Ctrl+P` agent · `q` / `Ctrl+C` quitter.

## Protocole du bridge (agents IA)

Socket Unix ou named pipe local, **une commande JSON par ligne**, une réponse par commande :

```text
{"cmd":"ping"}                                                -> {"ok":true,"pong":true}
{"cmd":"highlight_layer","layer":14}
{"cmd":"select_tensor","name":"model.layers.14.mlp.down_proj.weight"}
{"cmd":"preview_patch","tensor":"…","alpha":0.5}
{"cmd":"clear_preview"}
{"cmd":"set_filter","role":"gate"}          // "role" omis ou null : désactive le filtre
{"cmd":"snapshot"}                          // -> {"ok":true,"snapshot":{"tensor":"…", …}}
```

Les commandes inconnues ou malformées reçoivent `{"ok":false,"error":"…"}` — jamais de crash.

### Depuis un shell

```bash
# Linux / macOS
printf '{"cmd":"highlight_layer","layer":14}\n' | nc -U /tmp/surgeon-tui.sock
# Windows (PowerShell) — pipe nommé
[System.IO.Pipes.NamedPipeClientStream] ou l'exemple fourni :
cargo run -p surgeon-tui --example agent_ping -- highlight 14
cargo run -p surgeon-tui --example agent_ping -- preview blk.14.ffn_down.weight 0.5
cargo run -p surgeon-tui --example agent_ping -- snapshot
```

### Intégration Claude Code / Codex / Aider

1. `Ctrl+E` dans la TUI écrit `surgeon-context-….md`.
2. Lancez l'agent avec ce contexte :
   - Claude Code : `--agent-command "claude {context}"` (défaut) — `claude` reçoit le
     chemin du snippet en dernier argument ;
   - Codex : `--agent-command "codex {context}"` ;
   - Aider : `--agent-command "aider --message-file {context}"`.
3. Un agent peut ensuite piloter l'affichage en écrivant au socket, par exemple via un
   `curl --unix-socket` ou l'exemple `agent_ping`.

Le snippet contient les faits mesurés (forme, dtype, métriques, exactitude d'ablation) :
aucune mention d'outil, aucun texte décoratif — il se colle tel quel dans une conversation.

## Architecture de la crate

```text
src/model/   dtype.rs     dtypes + décodeurs exacts (Q8_0, Q4_K, Q5_K, Q6_K, demi-flottants)
             gguf.rs      parser GGUF v2/v3 autonome (KV, alignment, offsets alignés)
             safetensors.rs  fichier unique, dossier de shards, index.json
             metrics.rs   L2, min/max, moyenne, kurtosis, r_eff, profil de lignes
             roles.rs     taxonomie des rôles (embed…router), {layer}/{expert}, exactitude
src/ipc.rs   bridge JSON-lines : socket Unix / named pipe Windows, serveur + client
src/context.rs  snippet Markdown d'export + lancement de l'agent CLI
src/app.rs   état, boucle async (tokio + crossterm, 60 Hz), métriques hors thread UI
src/ui/      theme (ANSI 256), couches, inspecteur, bridge
```

## Qualité

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Determinisme : l'échantillonnage des métriques et la projection `r_eff` sont semés
fixes — deux exécutions sur le même fichier affichent exactement les mêmes chiffres.
Aucun `unsafe` (`#![forbid(unsafe_code)]`), mmap via `memmap2`.

Licence : identique au dépôt LLM-Surgeon.
