# LLM Surgeon (`surgeon`)

[English version](README.md)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![rust](https://img.shields.io/badge/rust-2021_edition-orange)
![no C toolchain](https://img.shields.io/badge/dependencies-no_C_compiler-success)
![MCP](https://img.shields.io/badge/MCP-stdio_server-purple)

`surgeon` inspecte, modifie, évalue et oriente (steering) les poids MLP de modèles de langage
locaux au format SafeTensors. Il tourne sur le CPU, en Rust pur, et s'adresse aux humains
(terminal en couleur, tableau de bord) comme aux agents IA (`--json`, serveur MCP).

Les messages du terminal sont en français. Toutes les sorties ci-dessous sont des exécutions
réelles sur un minuscule modèle Qwen2 factice (4 couches, taille cachée 16), copiées sans
codes ANSI.

## Architecture

Huit piliers, un seul binaire :

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

| Pilier | Module | Ce qu'il garantit |
| --- | --- | --- |
| Budget matériel | `src/hardware` | Besoins RAM et VRAM calculés avant tout chargement. |
| Lecture et décalage | `src/engine` | Lectures en mémoire mappée ; statistiques en deux passes parallèles, déterministes. |
| Patch | `src/patch` | `patch.json` validé d'abord, sauvegarde vérifiée par SHA-256, rollback automatique. |
| Évaluation | `src/eval` | Détection de boucles et attentes via llama.cpp, export GGUF seulement si le modèle est sain. |
| Extraction | `src/extract` | Vecteur de steering depuis une couche Qwen2, écriture atomique plus un fichier compagnon de statistiques. |
| Mesure | `src/bench` | Dumps de logits et écarts KL/top-k, delta de perplexité, Pass@1 ; rapports déterministes. |
| Noyaux et E/S | `src/kernel` | Noyaux SIMD de décalage (AVX2, AVX-512, NEON) identiques au bit près au chemin scalaire ; écriture par blocs avec prefetch de l'OS. |
| GGUF en place | `src/patch/gguf` | Cibles Q8_0, Q4_K, Q6_K, F16, BF16, F32 : backup, marqueur de crash, hachage de l'en-tête, relecture, rollback. |
| Algèbre linéaire | `src/linalg` | Jacobi, SVD fine par Gram, Cholesky : Rust pur, f64, indépendant du nombre de threads. |
| Tableau de bord | `src/tui` | État de style Elm, rendu pur, worker en arrière-plan avec annulation. |
| Agents | `src/mcp` | Serveur MCP sur stdio, chemins confinés aux dossiers autorisés. |

## Installation

Prérequis : une chaîne Rust (testé avec Rust 1.97) et l'éditeur de liens de la plateforme
(MSVC Build Tools sous Windows). Aucune dépendance ne compile de code C :
`tests/deps_purity.rs` échoue si c'est le cas.

```bash
git clone https://github.com/truekiwwy/LLM-Surgeon.git
cd LLM-Surgeon
cargo install --path .
surgeon --version
```

```text
surgeon 0.1.0
```

Le binaire est installé dans `$HOME/.cargo/bin` (Linux, macOS) ou `%USERPROFILE%\.cargo\bin`
(Windows). `surgeon eval` a aussi besoin d'un llama.cpp compilé (`llama-cli`,
`llama-quantize`, `convert_hf_to_gguf.py` et ses dépendances Python) ; `surgeon bench ppl`
a besoin de `llama-perplexity`, `surgeon bench eval` de `llama-completion` (ou `llama-cli`).

## Tutoriel

Les commandes se lancent depuis `<MODELS_DIR>`, le dossier qui contient vos modèles. Le
modèle d'exemple s'appelle `qwen2-mini` ; `pos.json` et `neg.json` sont deux tableaux JSON
de six phrases courtes.

```bash
cd <MODELS_DIR>
```

### 1. Vérifier la machine

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

Le verdict vaut `FITS IN VRAM`, `FITS IN RAM ONLY` ou insuffisant (code de sortie 1).

### 2. Inspecter les poids MLP

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

`--layers` accepte `all`, `10-20` ou `3,5,7` ; `--proj` accepte `gate`, `up` ou `down` et se
répète. Avec `--json`, `check`, `inspect`, `apply`, `eval` et `extract` écrivent à la place
une seule enveloppe lisible par une machine :

```bash
surgeon --json inspect --model qwen2-mini --layers 2 --proj down
```

```text
{"ok":true,"command":"inspect","exit_code":0,"data":{"shards":["qwen2-mini\\model.safetensors"],"tensors":[{"dtype":"F32","name":"model.layers.2.mlp.down_proj.weight","shape":[16,32],"shard":"qwen2-mini\\model.safetensors","stats":null}]},"error":null,"warnings":[]}
```

### 3. Extraire un vecteur de steering

`extract` exécute la passe avant Qwen2 sur les deux jeux de prompts et écrit
`v = normalize(mean(pos) - mean(neg))` mesuré sur une couche. Seuls Qwen2 et Qwen2.5
(`model_type: "qwen2"`) sont pris en charge.

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

`direction.safetensors` contient un tenseur 1D F32 nommé `direction` et des métadonnées
(couche, site, pooling, `d_prime`, `threshold_accuracy`, `pos_mean_proj`, `neg_mean_proj`,
`cos_mean_pos_neg`...). À côté, `direction.safetensors.stats.json` garde la projection de
chaque prompt, que le tableau de bord dessine :

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

Les deux fichiers sont écrits de façon atomique. Une sortie existante, ou un fichier
compagnon existant, exige `--overwrite`. En cas d'erreur ou de Ctrl-C, rien n'est laissé.

### 4. Appliquer un patch

`patch.json` soustrait la moitié du vecteur aux `down_proj` des couches 1 et 2. Le vecteur
vit dans le flux résiduel (taille `hidden_size`) : il indexe les lignes de `down_proj`
(`[hidden, intermediate]`), donc `axis: "cols"` ; pour `gate_proj` et `up_proj`, utilisez
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

Vérifier d'abord, puis appliquer. Le dossier de sauvegarde doit déjà exister (par défaut : à
côté du modèle).

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

Le format complet : `surgeon schema` écrit le JSON Schema (draft-07) de `patch.json`.

### 5. Évaluer, puis exporter en GGUF

`eval` convertit le modèle en GGUF avec llama.cpp, passe chaque prompt dans `llama-cli`,
détecte les boucles et vérifie les attentes ; l'export GGUF n'a lieu que si le modèle est
sain (code de sortie 5 sinon). Le fichier de prompts :

```json
[{"id": "greet", "text": "Bonjour", "expect_not_contains": ["error"]}]
```

```bash
surgeon eval --model qwen2-mini --prompts prompts.json --llama-cpp-dir <LLAMA_CPP_DIR> --export-gguf qwen2-mini.gguf --quant Q4_K_M
```

Cette étape n'a pas été exécutée pour ce README : la machine n'avait pas llama.cpp. Sans
`--llama-cpp-dir` (ni `LLAMA_CPP_DIR`), `eval` s'arrête avant de lancer quoi que ce soit :

```bash
surgeon eval --model qwen2-mini --prompts prompts.json
```

```text
erreur : --llama-cpp-dir (ou LLAMA_CPP_DIR) requis pour convertir en GGUF
```

### 6. Explorer dans le tableau de bord

```bash
surgeon tui --model qwen2-mini --vector direction.safetensors
```

| Touche | Action |
| --- | --- |
| `Tab`, `Shift+Tab` | changer d'onglet |
| `↑` `↓` | couche (onglet 1), défiler le top-10 (onglet 2) |
| `PgUp` `PgDn`, `Home` `End` | sauter entre les couches (onglet 1) |
| `←` `→` | projection (onglet 1), défiler les composantes (onglet 2) |
| `+` `-` | zoomer ou dézoomer sur les composantes (onglet 2) |
| `l` | histogramme linéaire ou logarithmique (onglet 1) |
| `?` | aide |
| `q`, `Esc`, `Ctrl-C` | quitter |

Les statistiques sont calculées dans un thread en arrière-plan (annulé quand la sélection
change, 64 entrées en cache). Un tenseur avec des NaN ou des Inf affiche une alerte rouge
clignotante et un `⚠` dans la liste des couches. Sans terminal (CI, pipe), `--snapshot`
écrit une image en texte brut :

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

Avec `--vector` seul, le tableau de bord s'ouvre sur l'onglet 2 (vecteur, top-10,
composantes, jauges d′ et accuracy, projection de chaque prompt autour du seuil) :

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

Avec les deux fichiers, l'onglet 2 indique aussi `compatible ✓` ou
`incompatible ✗ (4096 ≠ 3584)` quand la taille du vecteur diffère de la taille cachée du
modèle. `--size 120x40` change la taille de l'image. Un vecteur extrait avant cette version
n'a pas de fichier compagnon : l'onglet 2 explique alors que les projections sont
indisponibles.

### 7. Le confier à un agent IA (MCP)

`surgeon mcp` sert neuf outils sur stdio (`surgeon_check`, `surgeon_inspect`,
`surgeon_apply`, `surgeon_eval`, `surgeon_extract`, `surgeon_bench`, `surgeon_autotune`, `surgeon_diff`, `surgeon_lora`).
Chaque chemin doit
tomber sous un dossier autorisé, et une sauvegarde vérifiée est imposée avant toute
écriture. Le tableau de bord n'est pas exposé en MCP.

```bash
surgeon mcp --allow-dir <MODELS_DIR> --read-only
```

Claude Code :

```bash
claude mcp add surgeon -- surgeon mcp --allow-dir <MODELS_DIR>
```

Claude Desktop (`claude_desktop_config.json`) :

```json
{ "mcpServers": { "surgeon": { "command": "surgeon", "args": ["mcp", "--allow-dir", "<MODELS_DIR>"] } } }
```

Une session réelle : le client envoie `initialize`, `notifications/initialized`, puis un
`tools/call` de `surgeon_check` sur `qwen2-mini`. Le serveur écrit une ligne sur stderr,
puis répond sur stdout (la réponse à `initialize` est omise ici) :

```text
surgeon mcp 0.1.0 : prêt sur stdio, 1 dossier(s) autorisé(s), lecture seule
{"id":2,"jsonrpc":"2.0","result":{"content":[{"text":"{\"ok\":true,\"command\":\"check\",\"exit_code\":0,\"data\":{\"available_ram_bytes\":4780371968,\"best_gpu\":null,\"model_bytes\":55528,\"required_ram_bytes\":2147547506,\"required_vram_bytes\":null,\"verdict\":{\"kind\":\"fits_in_ram_only\"},\"warnings\":[\"NVML indisponible, GPU NVIDIA ignorés : the current user does not have permission to perform this operation\",\"aucun GPU détecté ; chargement en RAM (CPU), plus lent\"]},\"error\":null,\"warnings\":[]}","type":"text"}],"isError":false,"structuredContent":{"command":"check","data":{"available_ram_bytes":4780371968,"best_gpu":null,"model_bytes":55528,"required_ram_bytes":2147547506,"required_vram_bytes":null,"verdict":{"kind":"fits_in_ram_only"},"warnings":["NVML indisponible, GPU NVIDIA ignorés : the current user does not have permission to perform this operation","aucun GPU détecté ; chargement en RAM (CPU), plus lent"]},"error":null,"exit_code":0,"ok":true,"warnings":[]}}}
```

Schémas des outils : `surgeon schema --kind tools`. Détails, garde-fous et versions du
protocole : [docs/MCP.md](docs/MCP.md).

### 8. Mesurer une chirurgie (`surgeon bench`)

Quatre modes mesurent ce qu'un patch a changé. La recette d'un agent : extraire une
direction (étape 3), l'appliquer **sur une copie** du modèle (étape 4), écrire les logits des
deux modèles sur les mêmes prompts, les comparer, puis vérifier la perplexité et un benchmark
en exact match avant de décider. Ici `patch.json` retranche `0.5 × [1, …, 1]` au
`gate_proj` de la couche 1, sans sauvegarde puisqu'il travaille sur une copie :

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

`bench dump` exécute toute la passe avant Qwen2 en Rust pur et écrit de façon atomique
`logits` `[positions, vocab]`, `prompt_offsets` et des métadonnées (`prompts_sha256`,
`tokenizer_sha256`...). `bench logits` lit aussi les tenseurs `[batch, seq, vocab]`, en
f32/f16/bf16, les `.npy` et les `.pt`, et avertit quand les deux dumps viennent de prompts ou
de tokenizers différents.

`bench ppl` et `bench eval` pilotent llama.cpp (un dossier SafeTensors est converti en GGUF
temporaire, supprimé ensuite). Les deux sorties ci-dessous viennent de
`examples/fake_llama.rs` copié en `llama-perplexity` et `llama-completion` : il imprime une
perplexité dérivée des octets du modèle et répond la somme des nombres de chaque problème.
Ces runs exercent donc la vraie gestion des processus, pas un vrai modèle.

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

Le dataset est un tableau JSON (ou JSONL) de `question` (ou `problem`), `answer` (ou
`solution`) et d'un `id` facultatif ; aucun contenu de GSM8K ou de MATH n'est livré dans ce
dépôt :

```json
[{"id": "q1", "question": "What is 12 plus 30?", "answer": "12 + 30 = 42\n#### 42"},
 {"id": "q4", "question": "What is half of 9?", "answer": "Half of 9 is $\\boxed{\\frac{9}{2}}$."}]
```

Définitions utilisées par les rapports (`--json` donne tous les champs) :

- **KL** : `KL(before‖after) = Σ p_i (log p_i − log q_i)` avec `p = softmax(before)` et
  `q = softmax(after)`, en f64 via log-sum-exp : la distribution « avant » sert de
  référence. Les termes `p_i = 0` sont ignorés ; un token fini avant et à `−∞` après rend la
  position infinie, comptée dans `kl_infinite_positions` et exclue de `kl_mean`, `kl_max` et
  `kl_p95` (rang le plus proche).
- **Cosinus** : `cosine_probs_mean` compare les vecteurs de probabilités ;
  `cosine_logits_mean`, libellé à part, les logits bruts.
- **Top-k** : `top1_shift_rate` est la part des positions dont l'argmax change (égalités :
  plus petit index) ; `topk_set_change_rate` la part dont l'ensemble des `k` meilleurs tokens
  diffère (`--top-k`, 5 par défaut), `topk_overlap_mean` la moyenne de `|A∩B| / k`.
- **ΔPPL** : `ppl − ppl_baseline`, arrondi à 4 décimales (ce qu'imprime llama.cpp) avant la
  comparaison stricte avec `--warn-delta` : `7.4 − 7.2` vaut `0.20000000000000018` en
  flottant et ne dépasse pas 0.2. `within_noise` quand `|Δ| ≤ sqrt(err² + err_baseline²)` ;
  un delta négatif est une amélioration, jamais un avertissement.
- **Pass@1** : exact match d'un échantillon glouton (`--temp 0 --top-k 1 --seed 42`). La
  réponse finale est la première trouvée parmi `#### X`, `\boxed{X}` (accolades imbriquées),
  le dernier `**X**`, « the answer is X », le dernier nombre ; nombres et fractions se
  comparent avec une tolérance relative de 1e-9 (`18` = `18.0` = `1,8e1`,
  `1/2` = `\frac{1}{2}`). L'intervalle à 95 % est celui de Wilson ; avec `--baseline-model`
  la comparaison est appariée (ids `lost`, `gained`).
- **Déterminisme** : les réductions suivent un ordre fixe, donc `bench logits` donne les mêmes
  bits avec 1 ou 12 threads ; deux runs donnent le même JSON hors `timing_ms`.

Un avertissement ne change jamais le code de sortie, sauf un seuil (`--warn-kl`,
`--warn-top1-shift`, `--warn-delta`, `--min-accuracy`) strictement dépassé avec
`--fail-on-warn` : code 9, le rapport restant dans `data`. Sans seuil, le verdict vaut
`none`. Mesuré sur un portable à 12 threads : `bench logits` sur deux dumps f16 de
4096 positions × 151 936 tokens prend 5,3 s avec 49 Mio de mémoire privée (les dumps sont
mappés en mémoire) ; noter 1 000 sorties d'eval prend 5 ms.

### 9. Representation engineering : `extract --method`, `project_out`, `autotune`

La théorie d'abord, avec les formules exactes du code (toute l'algèbre linéaire est en Rust
pur, en f64, dans `src/linalg` : Jacobi cyclique, SVD fine par la matrice de Gram, Cholesky).

- **`mean_diff`** (défaut, inchangé) : `v = normalize(μ₊ − μ₋)` sur les activations poolées.
- **`svd`** (PCA contrastive) : prompts appariés par index (sinon `unpaired_prompts`, code
  2), `ΔX_i = x⁺_i − x⁻_i`, `v` = premier vecteur singulier droit de ΔX, c'est-à-dire le
  vecteur propre dominant de `(1/N) ΔXᵀΔX`, orienté pour que `⟨v, μ₊ − μ₋⟩ ≥ 0`. ΔX n'est PAS
  centrée par défaut : centrer retire le delta moyen, qui est l'axe du concept lui-même.
  `--svd-center` sert aux paires d'ordre arbitraire (signes aléatoires). Diagnostics :
  σ₁..σ₅, `σ₁²/Σσ²`, cosinus avec `μ₊ − μ₋`, cosinus split-half (paires paires contre
  impaires) ; avertissement si `σ₂/σ₁ > 0.9` ou moins de 8 paires.
- **`leace`** : `r(x) = x − a bᵀ(x − μ)` avec `Σ_λ = (1−λ)Σ + λ·tr(Σ)/d·I`,
  `w = Σ_λ^{-1/2}(μ₊ − μ₋)`, `u = w/‖w‖`, `a = Σ_λ^{1/2}u` (parallèle à `μ₊ − μ₋`),
  `b = Σ_λ^{-1/2}u` (parallèle à `Σ_λ⁻¹(μ₊ − μ₋)`), `bᵀa = 1`. `Σ^{-1/2}μ_Δ` seul n'est que la
  direction blanchie `w` : l'effaceur a besoin de `a` (où écrire) et de `b` (quoi lire).
  `Σ_λ^p` est appliquée via la SVD fine des données centrées, jamais comme matrice d×d ;
  `--leace-shrinkage 0` donne la pseudo-inverse sur le sous-espace des données
  (`--leace-rcond`). La `direction` écrite vaut `a/‖a‖`, donc la direction `mean_diff` ; `a`,
  `b`, `mean` et `u_whitened` vont dans `<out>.leace.safetensors`.
- **Limites honnêtes de LEACE** : la covariance croisée `cov(r(x), z)` s'annule en
  échantillon pour tout `Σ_λ` défini positif ; « aucun prédicteur linéaire ne fait mieux
  qu'une constante » est une garantie de population. Avec n < d, un classifieur peut encore
  séparer l'échantillon effacé : seul `--holdout-frac` (réajustement sur le reste, mesure sur
  les prompts retenus) dit si l'effacement généralise. Effacer un concept de tout le flux
  résiduel demanderait l'effaceur sur toutes les écritures dans ce flux ; patcher les poids
  d'une couche n'en donne pas la garantie.
- **`project_out`** (opération de patch) : `W' = W − α·a(bᵀW)` sur `down_proj`
  (`[hidden, intermediate]`, qui écrit dans le flux résiduel), `α ∈ [0, 1]` (1 = effacement
  complet de ce que cette matrice écrit selon `b`). `o_proj` et les embeddings écrivent aussi
  dans le flux et ne sont pas couverts.
- **`autotune`** : le naïf `min ‖Y − Σ α_l W_l v_l‖²` laisse `Y` et `W_l v_l` indéfinis. La
  lecture est `s_i = ⟨pool(h_i^R), v⟩` à la couche R (site, pooling et gabarit du vecteur),
  la cible `y_i = t_i − s_i⁰` (`neg_mean_proj` pour `suppress`, `pos_mean_proj` pour
  `amplify`, `--target-projection`, ou un `target` par prompt), la réponse
  `F_il = (s_i(δ_l) − s_i⁰)/δ` mesurée en mémoire avec le décalage du moteur sur les octets
  stockés (arrondi bf16 compris), puis `α = (FᵀF + λI)⁻¹Fᵀy` par Cholesky, avec `λ` fixe ou
  choisi par LOOCV en forme fermée (`e_i/(1 − h_ii)`, 17 points de 1e-6 à 1e2). Le R² est
  non centré (`1 − Σe²/Σy²`, il n'y a pas d'intercept). Les `α` ajustés sont appliqués
  ensemble et mesurés à nouveau. Les effets secondaires ne sont pas mesurés : lancer
  `surgeon bench` ensuite.

Exécutions réelles sur le mini Qwen2 factice (6 prompts par ensemble, hidden size 16) :

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

Le d′ de `leace` (2.23) est celui de `mean_diff` : même direction, comme le dit la théorie.
Application de l'effaceur sur une copie (`project.json` nomme les tenseurs `a` et `b` du
fichier `.leace`) :

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

Sur ce mini-modèle aléatoire, les réponses ne se généralisent pas d'un prompt à l'autre :
la LOOCV choisit le λ le plus fort et l'ajustement explique 1 % du décalage. C'est la réponse
honnête, et le décalage mesuré (−0.0354) colle à la prédiction (−0.0371). `tuned.json` est un
patch ordinaire (`backup: true`, un `subtract` par couche) pour `surgeon apply`.

Comparaison des trois méthodes sur données SYNTHÉTIQUES (`cargo run --release --example
bench_directions`) : direction t plantée, composante « syntaxe » partagée par chaque paire,
bruit anisotrope, force du concept `s = 0.5·√d` (« faible » : `0.15·√d`), graine fixe ; les
temps sont indicatifs. Aucun gain sur de vrais modèles n'est affirmé.

| scénario (synthétique) | paires N | d | méthode | cos(v, t) | split-half | cos(b, W⁻¹t) | temps (ms) |
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
| faible | 32 | 1024 | mean_diff | 0.476 | 0.130 | — | 0.1 |
| faible | 32 | 1024 | svd | 0.406 | 0.054 | — | 2.5 |
| faible | 32 | 1024 | leace | 0.476 | 0.130 | 0.280 | 6.0 |
| faible | 128 | 1024 | mean_diff | 0.699 | 0.359 | — | 0.3 |
| faible | 128 | 1024 | svd | 0.655 | 0.211 | — | 98.2 |
| faible | 128 | 1024 | leace | 0.699 | 0.359 | 0.454 | 914.4 |

Ce que montre ce tableau synthétique : l'appariement annule déjà la composante « syntaxe »
partagée pour `mean_diff`, et la SVD non centrée est tirée vers les axes les plus bruités :
`svd` ne bat donc pas `mean_diff` ici (égalité à d = 1024). La direction `leace` est celle de
`mean_diff` par construction ; sa direction de lecture `b` s'approche de l'optimum `W⁻¹t`
quand N ≫ d et en reste loin quand n < d (le rétrécissement domine).

### 10. Patcher un GGUF quantifié en place (`apply --gguf`)

Le même `patch.json` (même schéma, `vector_file` toujours un vecteur SafeTensors ou JSON)
s'applique directement à un GGUF mono-fichier. Les noms suivent llama.cpp : `{gate,up,down}_proj`
de la couche `i` devient `blk.{i}.ffn_{gate,up,down}.weight` ; `ne[0]` est la dimension des
colonnes, donc `rows` / `cols` gardent leur sens SafeTensors. Un `.gguf` donné sans `--gguf`
est refusé : la re-quantification perd de l'information et n'arrive jamais par surprise.

```bash
surgeon inspect --model qwen2.5-7b-q4_k_m.gguf
surgeon apply --gguf --model qwen2.5-7b-q4_k_m.gguf --patch patch.json --dry-run
surgeon apply --gguf --model qwen2.5-7b-q4_k_m.gguf --patch patch.json
surgeon apply --gguf --requant recompute --model qwen2.5-7b-q4_k_m.gguf --patch patch.json
```

Types de cibles supportés : F32, F16, BF16, Q8_0, Q4_K, Q6_K. Un fichier « Q4_K_M » mélange les
types par tenseur (`ffn_down` est souvent en Q6_K) ; les tenseurs non ciblés peuvent être de
n'importe quel type, une cible d'un autre type est refusée avec `unsupported_quant_type`.

| `--requant` | Ce qui change dans un bloc | À utiliser pour |
| --- | --- | --- |
| `keep-scales` (défaut) | Seuls les entiers des éléments dont la valeur patchée diffère ; `d`, `dmin` et les sous-échelles restent identiques à l'octet ; les valeurs hors grille sont écrêtées et comptées | les petits décalages (environ un pas de quantification) |
| `recompute` | De nouvelles échelles pour chaque bloc touché (Q8_0 : `quantize_row_q8_0_ref` ; Q4_K/Q6_K : min/max avec échelles arrondies vers le haut, sans écrêtage) | les décalages forts ; ce n'est pas l'algorithme itératif de `llama-quantize`, sans matrice d'importance |

Chaque tenseur reçoit un rapport de fidélité (`data.gguf.tensors[]`) : `intended_shift_norm`
(‖α·v‖), `realized_shift_norm` (‖déquant(après) − déquant(avant)‖), `realized_over_intended`,
`correlation_intended_realized`, `unchanged_fraction`, `saturated_elements`, `nmse_vs_exact`.
Un rapport sous 0,5 ou une corrélation sous 0,8 signifie que le décalage est plus petit qu'un
demi-pas et qu'il a disparu ou est devenu du bruit : augmentez alpha, ou patchez le SafeTensors
puis re-quantifiez avec llama.cpp. Un écrêtage au-delà de 0,1 % d'un tenseur suggère
`--requant recompute`. Mesuré sur des poids N(0, 0,02²), v ~ N(0, 1) : erreur du tenseur patché
divisée par l'erreur d'une quantification fraîche des mêmes poids exacts (`tests/p12_gguf.rs` ;
synthétique, pas un vrai modèle) :

| alpha | Q4_K keep-scales | Q8_0 keep-scales | Q6_K keep-scales | recompute (tous) |
| ---: | ---: | ---: | ---: | ---: |
| 0,0003 | 0,28 | 1,17 | 1,01 | 1,00 |
| 0,001 | 0,67 | 2,81 | 1,25 | 1,00 |
| 0,003 | 1,14 | 19,5 | 3,2 | 1,00 |
| 0,01 | 3,45 | 349 | 46 | 1,00 |

Q8_0 n'a aucune marge (`d = amax / 127`) : tout décalage qui pousse la plus grande valeur d'un
bloc vers l'extérieur est écrêté. Rien ici ne prouve qu'un patch faible est préservé : validez
avec `surgeon bench ppl` (et `surgeon bench eval`) contre le fichier d'origine.

Sûreté d'une écriture en place sur un fichier de plusieurs dizaines de Go :

1. tout est vérifié avant d'écrire (cibles, types, blocs entiers, longueurs de vecteur, espace
   disque libre pour la copie) ;
2. une copie complète est faite et son SHA-256 vérifié (`--no-backup` est refusé ; seul
   `--force-no-backup` s'en passe, et un échec laisse alors le fichier modifié) ;
3. un marqueur de crash `<modèle>.surgeon-inprogress` (chemin du backup, SHA-256 de l'original,
   horodatage) est écrit avant le premier octet ;
4. les tenseurs sont écrits bloc par bloc, chaque bloc calculé et contrôlé (échelles et valeurs
   finies) d'abord dans une copie ; l'en-tête, les métadonnées et la table des tenseurs ne sont
   jamais réécrits (leur SHA-256 est comparé avant et après) ;
5. après le flush, une seconde projection mémoire relit chaque tenseur patché et le compare au
   SHA-256 calculé pendant l'écriture (`--no-verify` saute cette étape) ;
6. toute erreur restaure le backup ; le marqueur est supprimé après le commit ou le rollback.

Si le marqueur est encore là (crash, coupure de courant), `apply` refuse de démarrer : copiez le
backup indiqué dans le marqueur sur le modèle, vérifiez son SHA-256 contre `original_sha256`,
puis supprimez le marqueur. La CI ne peut pas valider un fichier patché contre un vrai
llama.cpp : vérifiez qu'il se charge et lancez `surgeon bench ppl` avant de l'utiliser.

### 11. Livrer un fine-tune sous forme de patch (`surgeon diff`, `surgeon apply`)

`surgeon diff` compare un checkpoint de base à sa version affinée et écrit la différence dans un
fichier `.surgery` : pour chaque tenseur 2D, une approximation de rang faible `U diag(S) V` de
`tuned - base` (SVD randomisé, Rust pur), pour chaque tenseur 1D le delta exact. Les deux
checkpoints doivent être des SafeTensors (un fichier ou un dossier shardé) avec les mêmes noms,
formes et dtypes de tenseurs ; sinon une erreur liste les différences. Les tenseurs identiques
sont ignorés, et `embed_tokens` / `lm_head` sont exclus sauf demande explicite.

```bash
surgeon diff --base <DOSSIER_BASE> --tuned <DOSSIER_AFFINE> --out finetune.surgery --rank 16
surgeon diff --base <DOSSIER_BASE> --tuned <DOSSIER_AFFINE> --out finetune.surgery \
    --include 'mlp' --factor-dtype bf16 --max-rel-error 0.3 --on-high-error full
```

Sortie réelle sur la paire écrite par `cargo run --example make_fake_checkpoints -- --out <DOSSIER>
--kind rank --rank 4 --lora` (tenseurs abrégés ; la différence est de rang 4, donc l'erreur est nulle) :

```text
patch écrit : finetune.surgery (0.01 MiB, factor_dtype f16, rang demandé 4)
10 tenseur(s) dans le patch, 1 identique(s) ignoré(s), 2 filtré(s)
  model.layers.0.self_attn.q_proj.weight · lowrank · rang 4 · erreur relative 0.0000 · énergie 1.0000
  model.layers.0.input_layernorm.weight · full_delta · rang 0 · erreur relative 0.0000 · énergie 1.0000
  [...]
Note : le .surgery est une APPROXIMATION du fine-tune (erreur relative par tenseur), exacte seulement si la différence est de rang <= r. Sa taille vaut environ la somme de r(m+n) x octets par facteur : des centaines de Mo pour un 7B au rang 16.
```

La sortie est reproductible (mêmes entrées et même `--seed`, mêmes octets, quel que soit le
nombre de threads ; une date n'est écrite qu'avec `--timestamp`). `--max-rel-error` avec
`--on-high-error warn|skip|full` décide du sort d'un tenseur mal reconstruit par l'approximation.

`surgeon apply` reconnaît un `.surgery` à son magic `SURG` et ajoute le delta au modèle en
place, avec la même sauvegarde et le même rollback qu'un patch JSON
(`W <- W + scale * U diag(S) V`, arrondi au dtype du modèle) :

```bash
surgeon apply --model <DOSSIER_MODELE> --patch finetune.surgery --dry-run
surgeon apply --model <DOSSIER_MODELE> --patch finetune.surgery --scale 0.5 --verify-base strict
```

```text
[DRY-RUN]  model.layers.0.mlp.down_proj.weight · lowrank · rang 4 · erreur 0.0000
[DRY-RUN]  model.layers.0.input_layernorm.weight · full_delta · rang 0 · erreur 0.0000
[...]
dry-run : 10 tenseur(s) vérifiés, aucun octet écrit
```

Une vraie application se termine par `[BACKUP] ... ✓ sha256 vérifié`, `[COMMIT] modèle patché` et
`10 tenseur(s) patchés, 9124 éléments modifiés, max |delta| 0.218872`.

`--verify-base strict|warn|off` compare chaque tenseur du modèle au SHA-256 stocké dans le patch
(défaut `warn` : applique quand même, avec un avertissement). Le format binaire est spécifié
dans [docs/SPECIFICATION.md](docs/SPECIFICATION.md).

À quoi s'attendre, honnêtement :

- Un `.surgery` est une approximation du fine-tune complet. L'erreur relative de chaque tenseur
  est rapportée par `diff` ; le résultat n'est exact que si la vraie différence est de rang au
  plus `--rank`.
- Il n'est pas petit. Sa taille vaut environ la somme de `r (m + n)` fois les octets du dtype
  des facteurs sur tous les tenseurs : des centaines de Mo pour un 7B à rang 16, pas quelques Mo.
- Il n'est pas exposé par l'outil MCP `surgeon_apply` (`apply` sur un `.surgery` est réservé à
  la ligne de commande).
- `--gguf` est refusé pour un `.surgery` (il ne patche que des SafeTensors), et les autres options GGUF et d'écriture par blocs de `apply` sont ignorées avec un avertissement.

### 12. Adaptateurs LoRA (`surgeon lora merge`, `surgeon lora extract`)

`surgeon lora merge` fusionne un adaptateur LoRA au format PEFT dans les poids d'un modèle
SafeTensors, en place, avec une sauvegarde vérifiée et un rollback automatique sur erreur ou
Ctrl-C : `W <- W + scale * (lora_alpha / r) * B A`. `lora_alpha` est lu dans le
`adapter_config.json` voisin de l'adaptateur (ou dans `--adapter-config`) ; `--alpha` le
remplace. Sans l'un ni l'autre, la commande s'arrête sur une erreur d'usage au lieu de deviner.
DoRA, les adaptateurs d'embeddings, `fan_in_fan_out` et `rank_pattern` / `alpha_pattern` sont
refusés explicitement.

```bash
surgeon lora merge --model <DOSSIER_MODELE> --lora <DOSSIER_ADAPTATEUR>/adapter_model.safetensors --dry-run
surgeon lora merge --model <DOSSIER_MODELE> --lora <DOSSIER_ADAPTATEUR>/adapter_model.safetensors --scale 0.8
```

`surgeon lora extract` fait l'inverse : il approxime la différence entre deux checkpoints par un
adaptateur de rang `r` (`B = U sqrt(S)`, `A = sqrt(S) V`) et écrit un `adapter_model.safetensors`
au format PEFT, avec son `adapter_config.json` à côté.

```bash
surgeon lora extract --base <DOSSIER_BASE> --patched <DOSSIER_AFFINE> \
    --out adapter/adapter_model.safetensors --rank 16 --dtype f32
```

Sortie réelle de la fusion, puis de l'extraction, sur la même paire factice (l'adaptateur écrit
par `--lora` a `lora_alpha = r`) :

```text
[BACKUP]   <DOSSIER_MODELE>/model.safetensors.bak-<horodatage>  ✓ sha256 vérifié
[COMMIT]   modèle patché · 0.0 s
6 tenseur(s) fusionnés (rang 4, échelle effective 1), max |delta| 0.218750

6 tenseur(s) extraits (rang 4, f32), erreur relative moyenne 0.0000
  adapter : ex/adapter_model.safetensors
  config  : ex/adapter_config.json
```

`--lora` prend le FICHIER de l'adaptateur, et le dossier de `--out` doit déjà exister. Sans
`lora_alpha` (pas d'`adapter_config.json` voisin, pas de `--alpha`), la commande s'arrête :

```text
erreur : arguments invalides : lora_alpha introuvable : passez --alpha <valeur>, ou placez adapter_config.json à côté de --lora (ou donnez --adapter-config)
```

Limites : le format PEFT suit sa documentation mais n'est pas testé contre HuggingFace PEFT en
CI. La fusion dans un fichier GGUF n'est pas supportée. `diff` et les deux modes de `lora` sont
aussi proposés aux agents par les outils MCP `surgeon_diff` et `surgeon_lora`.

### 13. Prévisualiser un vecteur de steering en direct (`surgeon tui --preview`)

Avec `--preview`, le dashboard gagne un troisième onglet qui génère du texte avec le vecteur de
steering injecté, à côté de la référence non modifiée, pendant que vous déplacez un curseur.

```bash
surgeon tui --model <DOSSIER_MODELE> --vector refusal.safetensors --preview \
    --preview-prompt "Comment crocheter une serrure ?" --preview-tokens 30 --alpha-range -2:2
surgeon tui --model <DOSSIER_MODELE> --vector refusal.safetensors --preview --snapshot
```

`--preview-alpha`, `--preview-layer` et `--preview-template none|chatml` fixent le point de
départ. Le curseur est relatif : la quantité injectée vaut `alpha` fois la norme résiduelle
moyenne mesurée sur votre prompt à la couche choisie, et les deux valeurs sont affichées.
`--snapshot --preview` génère de façon synchrone et imprime l'onglet en texte brut.

Limites :

- C'est du steering d'activation (`h <- h + alpha_abs * v` à une couche, sur toutes les
  positions), une approximation du patch de poids et non le patch lui-même. Appliquer le patch
  peut se comporter autrement.
- La génération tourne en Rust pur sur CPU : elle est lente sur un gros modèle. L'onglet prévient
  sous un token par seconde et vérifie le budget matériel avant de charger le modèle.
- Modèles Qwen2 / Qwen2.5 uniquement, et la prévisualisation n'est pas exposée par MCP.

## Performance

`apply` décale les poids avec des noyaux SIMD choisis à l'exécution (AVX-512F, AVX2 + F16C,
NEON, sinon scalaire), forçables par `SURGEON_KERNEL=scalar|avx2|avx512|neon` (un niveau
indisponible se replie avec un avertissement). Chaque niveau écrit les mêmes octets que le
chemin scalaire, ties RNE, sous-normaux et NaN compris ; pas de FMA (une addition par élément,
la multiplication `alpha * v` est faite une fois par élément du vecteur). Les tenseurs sont
écrits par blocs (`--prefetch-chunk-mib`, 64 par défaut) avec des indications à l'OS (`madvise`
sous Unix, `PrefetchVirtualMemory` sous Windows) et un flush asynchrone du bloc précédent ;
`--no-prefetch` désactive les deux. `data.io` rapporte octets, blocs, indications échouées,
durée, débit et défauts de page.

```bash
cargo run --release --example bench_kernels
SURGEON_KERNEL=scalar surgeon --json apply --model qwen2-mini --patch patch.json
```

Mesuré avec `examples/bench_kernels.rs` sur un Intel Core i5-12400F (6 cœurs / 12 threads,
AVX2 + F16C, sans AVX-512), 16 Go de RAM, Windows 11 Pro, SSD NVMe Samsung 990 EVO, 12 threads
rayon. En mémoire, tampon de 1 Gio, 4096 colonnes, meilleur de 3 (`memcpy` parallèle du même
tampon : 18 565 Mio/s ; un thread : 14 951 Mio/s) :

| niveau | dtype | axe `rows` Mio/s | axe `cols` Mio/s | vs memcpy parallèle |
| --- | --- | ---: | ---: | ---: |
| scalar | bf16 | 4 186 | 4 103 | 0,22 |
| scalar | f16 | 2 740 | 2 825 | 0,15 |
| scalar | f32 | 13 368 | 13 301 | 0,72 |
| avx2 | bf16 | 18 270 | 18 117 | 0,98 |
| avx2 | f16 | 18 389 | 18 108 | 0,99 |
| avx2 | f32 | 17 669 | 17 602 | 0,95 |

AVX-512 et NEON compilent (vérifié avec `cargo check --target aarch64-unknown-linux-gnu` et
`aarch64-apple-darwin`) mais n'ont pas tourné sur cette machine. Les 2³² motifs f32 se
convertissent en bf16 et f16 exactement comme avec `half` au niveau AVX2 (test ignoré, 1,6 s
en release).

De bout en bout, `shift_tensor` + `flush` sur un fichier SafeTensors bf16 de 2 Gio déjà dans le
cache de pages (le cache froid demande `posix_fadvise`, Linux uniquement) :

| prefetch | calcul s | flush s | total Mio/s | défauts de page |
| --- | ---: | ---: | ---: | ---: |
| oui (run 1) | 8,21 | 0,41 | 237 | 525 573 |
| non (run 2) | 0,26 | 11,21 | 178 | 525 582 |
| oui (run 3) | 9,73 | 0,40 | 202 | 525 572 |
| non (run 4) | 0,26 | 11,00 | 182 | 525 572 |

Sur cette machine, c'est l'écriture des pages sales (environ 200 Mio/s à travers le fichier
mappé) qui domine, pas les noyaux : le flush par bloc ne fait que déplacer ce coût dans la
boucle (1,1 à 1,3x au total). Le prefetch ne supprime pas les défauts de page sous Windows (un
défaut mineur par page de 4 Kio dans les deux cas). L'E/S directe (`O_DIRECT`) n'est pas
implémentée : elle contourne le cache de pages dont une projection mémoire est faite. Ces
chiffres dépendent du disque, du cache de pages et de la politique d'écriture différée de l'OS ;
relancez l'exemple sur votre machine.

## Codes de sortie

| Code | Signification |
| --- | --- |
| 0 | Succès |
| 1 | Échec général : ressources insuffisantes, modèle illisible, binaire llama.cpp introuvable, Ctrl-C, `tui` sans terminal interactif |
| 2 | Entrée invalide : arguments, patch, prompts, chemins ; rien n'a été touché |
| 3 | Échec d'exécution ; le modèle est intact (jamais touché, ou restauré depuis la sauvegarde) |
| 4 | Critique : rollback raté, ou pas de sauvegarde et modèle laissé partiellement patché |
| 5 | `eval` : modèle malade (boucles, attentes non tenues) ; pas d'export |
| 6 | `eval` : échec de l'export GGUF |
| 7 | `extract` ou `autotune` : config, tokenizer ou poids invalides, direction dégénérée, NaN/Inf, système singulier ou mal conditionné |
| 8 | `bench` : échec du bench (processus llama.cpp, sortie illisible, dumps incompatibles, génération) |
| 9 | `bench --fail-on-warn` : un seuil a été dépassé (le rapport est dans `data`) |
| 70 | Erreur interne (panic), signalée après restauration du terminal |

## Commandes

| Commande | Rôle | Écrit |
| --- | --- | --- |
| `surgeon check` | Budget RAM et VRAM, verdict | rien |
| `surgeon inspect` | Tenseurs MLP, statistiques en option | rien |
| `surgeon apply` | Applique un `patch.json` ou un `.surgery` en place (`--dry-run` d'abord), `--gguf` pour un GGUF quantifié | modèle, sauvegarde |
| `surgeon eval` | Non-régression via llama.cpp, export GGUF conditionnel | GGUF |
| `surgeon extract` | Vecteur de steering depuis une couche Qwen2 | vecteur, compagnon `.stats.json` |
| `surgeon bench` | Dumps de logits et KL/top-k, delta de perplexité, Pass@1 | dump de logits (`dump` seulement) |
| `surgeon autotune` | Coefficients par couche par ridge sur des réponses mesurées | patch.json (`--out` seulement) |
| `surgeon diff` | Patch `.surgery` de bas rang entre deux checkpoints | le fichier `.surgery` |
| `surgeon lora` | `merge` d'un adaptateur LoRA dans les poids, ou `extract` depuis deux checkpoints | modèle, sauvegarde (`merge`) ou adaptateur (`extract`) |
| `surgeon tui` | Tableau de bord : inspecteur de poids et visualiseur de vecteur, onglet de prévisualisation avec `--preview` | rien |
| `surgeon schema` | JSON Schema de `patch.json`, de l'en-tête `.surgery`, ou définitions des outils MCP | stdout seulement |
| `surgeon mcp` | Serveur MCP sur stdio | via ses outils |

```bash
surgeon schema --kind patch
surgeon schema --kind tools
surgeon schema --kind surgery
```

## Usage responsable

Le steering et la chirurgie de poids changent ce que fait un modèle, y compris ses
comportements de sécurité : une direction peut retirer des refus aussi facilement qu'en
ajouter. Ne travaillez que sur des modèles dont la licence autorise la modification, gardez
les sauvegardes, mesurez l'effet avec `surgeon eval` et `surgeon bench` avant de partager
quoi que ce soit, et
ne distribuez pas un modèle dont vous avez retiré les garde-fous.

## Limites connues

- `extract` ne prend en charge que Qwen2 et Qwen2.5, sur le CPU ; les gros modèles demandent
  beaucoup de RAM (`--dtype bf16` la divise par deux).
- `eval` et l'export GGUF dépendent d'un llama.cpp compilé à part.
- `bench dump` ne prend en charge que Qwen2, sur le CPU. `bench ppl` et `bench eval`
  dépendent des options du llama.cpp installé (`llama-perplexity`, anciennement
  `perplexity` ; `--chunks`, `-ngl`, `--temp`, `--top-k`, `--seed`) ; le déchargement GPU
  peut faire varier les derniers chiffres d'un run à l'autre.
- 50 questions (le défaut de `bench eval`) donnent un intervalle large : 40/50 donne
  [67,0 % ; 88,8 %], seules les fortes régressions sont concluantes.
- Les dumps `.pt` sont lus au mieux : archives `torch.save` non compressées, tenseurs
  contigus seulement.
- `extract --method svd|leace` et `autotune` reposent sur Jacobi, limité à 1024 sur le petit
  côté (prompts ou hidden size). L'effacement LEACE n'est exact qu'en échantillon ; avec
  n < d, regarder `holdout_cross_cov_ratio`. `project_out` ne couvre que `down_proj` et
  n'efface donc pas un concept de tout le flux résiduel. `autotune` ajuste et mesure sur les
  mêmes prompts et ne mesure pas les effets secondaires.
- Seules les projections MLP (`gate_proj`, `up_proj`, `down_proj`) sont couvertes.
- `apply --gguf` : GGUF mono-fichier et little-endian uniquement ; cibles F32, F16, BF16,
  Q8_0, Q4_K et Q6_K ; pas de matrice d'importance ; un décalage sous le demi-pas est
  perdu, un décalage au-delà de la plage d'un bloc est écrêté (`keep-scales`) ; pas validé
  contre un vrai llama.cpp par la suite de tests. La re-quantification GGUF est scalaire
  (lignes en parallèle).
- La vitesse de bout en bout d'`apply` est limitée par l'écriture différée du fichier
  mappé par l'OS, pas par les noyaux (voir Performance) ; le prefetch ne supprime pas les
  défauts de page sous Windows.
- Le tableau de bord exige un terminal interactif d'au moins 80×24, avec stdin, stdout et
  stderr qui y sont rattachés ; sinon, utilisez `--snapshot`.
- Les messages du terminal sont en français.
- Aucun fichier de licence n'est encore publié : tant qu'il n'y en a pas, le droit d'auteur
  par défaut s'applique (tous droits réservés).

## Dépannage

| Symptôme | Solution |
| --- | --- |
| `` `surgeon tui` nécessite un terminal interactif `` | Lancez-le dans un vrai terminal, ou ajoutez `--snapshot`. |
| `dossier de backup inexistant` | Créez d'abord le dossier de `--backup-dir`. |
| `une application précédente a été interrompue` (`apply --gguf`) | Copiez le backup indiqué dans `<modèle>.surgeon-inprogress` sur le modèle, vérifiez son SHA-256, supprimez le marqueur. |
| `--llama-cpp-dir (ou LLAMA_CPP_DIR) requis` | Compilez llama.cpp et passez son dossier. |
| Avertissement `NVML indisponible` | Sans gravité : les GPU NVIDIA sont ignorés et le modèle est prévu en RAM. |
| `cargo install` échoue avec « Accès refusé » sous Windows | Un `surgeon.exe` tourne encore : fermez-le, ou renommez-le en `surgeon.old.exe`, puis réinstallez. |
| Cadres ou blocs mal affichés | Utilisez un terminal UTF-8 avec une police qui contient les caractères de cadre (Windows Terminal, par exemple). |

## Contribuer

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

`tests/deps_purity.rs` refuse toute dépendance qui compile du code C, et
`tests/readme_check.rs` vérifie que chaque commande et option `surgeon` citée dans les deux
README existe. Les commits suivent `type: description` (`feat`, `fix`, `refactor`, `docs`,
`chore`).

## Dépôt

```text
.
├── Cargo.toml
├── README.md, README.fr.md
├── docs/MCP.md            guide du serveur MCP
├── src/
│   ├── main.rs, lib.rs    point d'entrée, modules
│   ├── app.rs, cli.rs     commandes, codes de sortie, définitions clap
│   ├── ui.rs              identité du terminal (couleurs, badges, tableaux)
│   ├── engine/            lectures SafeTensors, décalages, statistiques partagées
│   ├── hardware/          sonde RAM et VRAM, budget
│   ├── patch/             patch.json, validation, sauvegarde, rollback
│   ├── eval/              exécution llama.cpp, détecteur de boucles, export GGUF
│   ├── extract/           passe avant Qwen2, vecteur de steering, fichier compagnon
│   ├── bench/             dumps de logits et KL, perplexité, Pass@1 (GSM8K/MATH)
│   ├── linalg/            Jacobi, SVD par Gram, itération de puissance, Cholesky (Rust pur)
│   ├── mcp/               serveur JSON-RPC, outils, sandbox
│   └── tui/               tableau de bord : app, worker, terminal, thème, widgets/
├── examples/              fake_llama.rs (faux llama.cpp), bench_directions.rs (synthétique)
└── tests/
    ├── integration.rs     parcours de bout en bout et tests du binaire
    ├── deps_purity.rs     aucun compilateur C dans l'arbre des dépendances
    └── readme_check.rs    commandes et liens des README
```
