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

Sept piliers, un seul binaire :

```text
                       surgeon (src/main.rs -> app::run_with)
   ┌────────┬─────────┬─────────┬────────┬─────────┬────────┬─────────┐
   check    inspect   apply     eval     extract   tui      mcp / schema
   │        │         │         │        │         │        │
   hardware engine    patch     eval     extract   tui      mcp
   RAM,VRAM mmap      patch.json llama.cpp Qwen2   ratatui  JSON-RPC
   budget   stats     backup    loops,   forward   worker   stdio,
            (rayon)   rollback  GGUF     (candle)  + LRU    sandbox
   └────────┴────┬────┴─────────┴────────┴────┬────┴────────┴─────────┘
                 engine::stats (shared)       ui (terminal identity)
```

| Pilier | Module | Ce qu'il garantit |
| --- | --- | --- |
| Budget matériel | `src/hardware` | Besoins RAM et VRAM calculés avant tout chargement. |
| Lecture et décalage | `src/engine` | Lectures en mémoire mappée ; statistiques en deux passes parallèles, déterministes. |
| Patch | `src/patch` | `patch.json` validé d'abord, sauvegarde vérifiée par SHA-256, rollback automatique. |
| Évaluation | `src/eval` | Détection de boucles et attentes via llama.cpp, export GGUF seulement si le modèle est sain. |
| Extraction | `src/extract` | Vecteur de steering depuis une couche Qwen2, écriture atomique plus un fichier compagnon de statistiques. |
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
`llama-quantize`, `convert_hf_to_gguf.py` et ses dépendances Python).

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

`surgeon mcp` sert cinq outils sur stdio (`surgeon_check`, `surgeon_inspect`,
`surgeon_apply`, `surgeon_eval`, `surgeon_extract`). Chaque chemin doit tomber sous un
dossier autorisé, et une sauvegarde vérifiée est imposée avant toute écriture. Le tableau de
bord n'est pas exposé en MCP.

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

## Codes de sortie

| Code | Signification |
| --- | --- |
| 0 | Succès |
| 1 | Échec général : ressources insuffisantes, modèle illisible, Ctrl-C, `tui` sans terminal interactif |
| 2 | Entrée invalide : arguments, patch, prompts, chemins ; rien n'a été touché |
| 3 | Échec d'exécution ; le modèle est intact (jamais touché, ou restauré depuis la sauvegarde) |
| 4 | Critique : rollback raté, ou pas de sauvegarde et modèle laissé partiellement patché |
| 5 | `eval` : modèle malade (boucles, attentes non tenues) ; pas d'export |
| 6 | `eval` : échec de l'export GGUF |
| 7 | `extract` : config, tokenizer ou poids invalides, direction dégénérée, NaN/Inf |
| 70 | Erreur interne (panic), signalée après restauration du terminal |

## Commandes

| Commande | Rôle | Écrit |
| --- | --- | --- |
| `surgeon check` | Budget RAM et VRAM, verdict | rien |
| `surgeon inspect` | Tenseurs MLP, statistiques en option | rien |
| `surgeon apply` | Applique un `patch.json` en place (`--dry-run` d'abord) | modèle, sauvegarde |
| `surgeon eval` | Non-régression via llama.cpp, export GGUF conditionnel | GGUF |
| `surgeon extract` | Vecteur de steering depuis une couche Qwen2 | vecteur, compagnon `.stats.json` |
| `surgeon tui` | Tableau de bord : inspecteur de poids et visualiseur de vecteur | rien |
| `surgeon schema` | JSON Schema de `patch.json`, ou définitions des outils MCP | stdout seulement |
| `surgeon mcp` | Serveur MCP sur stdio | via ses outils |

```bash
surgeon schema --kind patch
surgeon schema --kind tools
```

## Usage responsable

Le steering et la chirurgie de poids changent ce que fait un modèle, y compris ses
comportements de sécurité : une direction peut retirer des refus aussi facilement qu'en
ajouter. Ne travaillez que sur des modèles dont la licence autorise la modification, gardez
les sauvegardes, mesurez l'effet avec `surgeon eval` avant de partager quoi que ce soit, et
ne distribuez pas un modèle dont vous avez retiré les garde-fous.

## Limites connues

- `extract` ne prend en charge que Qwen2 et Qwen2.5, sur le CPU ; les gros modèles demandent
  beaucoup de RAM (`--dtype bf16` la divise par deux).
- `eval` et l'export GGUF dépendent d'un llama.cpp compilé à part.
- Seules les projections MLP (`gate_proj`, `up_proj`, `down_proj`) sont couvertes.
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
│   ├── mcp/               serveur JSON-RPC, outils, sandbox
│   └── tui/               tableau de bord : app, worker, terminal, thème, widgets/
└── tests/
    ├── integration.rs     parcours de bout en bout et tests du binaire
    ├── deps_purity.rs     aucun compilateur C dans l'arbre des dépendances
    └── readme_check.rs    commandes et liens des README
```
