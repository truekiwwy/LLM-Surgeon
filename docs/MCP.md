# surgeon : serveur MCP

`surgeon mcp` est l'**extension agent (MCP)** du Surgery Engine : il rend le moteur de
chirurgie de poids appelable par un agent IA, avec des outils typés, via le Model Context
Protocol, en JSON-RPC 2.0 sur stdio (un message par ligne). stdout ne transporte que le
protocole ; les journaux partent sur stderr, sans couleur. Le vocabulaire (Surgery Engine,
CLI front-end, extension agent, extension points) est défini dans
[docs/ARCHITECTURE.md](ARCHITECTURE.md).

## Lancement

```bash
surgeon mcp --allow-dir /chemin/vers/modeles
surgeon mcp --allow-dir /modeles --allow-dir /directions --read-only
```

| Option | Effet |
| --- | --- |
| `--allow-dir <dossier>` | Dossier autorisé, répétable. Défaut : `$SURGEON_MCP_ALLOW_DIRS` (séparateur du système : `;` sous Windows, `:` ailleurs), sinon le dossier courant. |
| `--read-only` | Refuse `surgeon_apply`, `surgeon_extract`, `export_gguf`, `surgeon_bench` en mode `dump`, `surgeon_autotune` avec `out`, `surgeon_diff` et `surgeon_lora` (les deux modes) (`error.kind = "read_only"`) ; `surgeon_autotune` sans `out` reste permis. |
| `--allow-no-backup` | Respecte `backup: false` dans un patch. Sans cette option, la sauvegarde est imposée. |

`$LLAMA_CPP_DIR`, défini par l'opérateur, sert de dossier llama.cpp par défaut pour
`surgeon_eval` et `surgeon_bench` (modes `ppl` et `eval`).

## Configuration des clients

Claude Desktop (`claude_desktop_config.json`) :

```json
{ "mcpServers": { "surgeon": { "command": "surgeon", "args": ["mcp", "--allow-dir", "C:\\modeles"] } } }
```

Claude Code :

```bash
claude mcp add surgeon -- surgeon mcp --allow-dir /chemin/vers/modeles
```

Claude Code, en plugin : le dossier [`plugin/`](../plugin) déclare le même serveur
(`allow_dir` demandé à l'activation) et un skill de démarche sûre ; voir
[docs/PLUGINS.md](PLUGINS.md#plugin-claude-code-plugin).

## Outils

| Outil | Rôle | Annotations |
| --- | --- | --- |
| `surgeon_check` | Budget RAM/VRAM du modèle | lecture seule |
| `surgeon_inspect` | Couches et matrices MLP, statistiques en option, `max_tensors` (défaut 500) | lecture seule |
| `surgeon_apply` | Applique un patch en place (`dry_run` pour vérifier d'abord) ; `gguf: true` pour un GGUF quantifié | destructif |
| `surgeon_eval` | Non-régression (boucles, attentes) et export GGUF conditionnel | écrit le GGUF |
| `surgeon_extract` | Vecteur de direction depuis une couche Qwen2 (Rust pur, CPU) | écrit le `.safetensors`, idempotent |
| `surgeon_bench` | Mesure d'une chirurgie : `mode` `dump`, `logits`, `ppl` ou `eval` | non destructif, idempotent |
| `surgeon_autotune` | Coefficients α par couche ajustés par ridge, patch.json en sortie | non destructif, idempotent ; écrit `out` seulement s'il est fourni |
| `surgeon_diff` | Différence de deux checkpoints, approximée à rang `r`, écrite dans un `.surgery` neuf | écrit un fichier, non destructif, idempotent |
| `surgeon_lora` | `mode` `merge` (fusionne un adaptateur LoRA en place) ou `extract` (écrit un adaptateur) | destructif |

Les schémas exacts : `surgeon schema --kind tools`. Le format du patch :
`surgeon schema` (JSON Schema draft-07, sans `$ref`).

Chaque résultat contient la même enveloppe que `surgeon --json` :
`{"ok", "command", "exit_code", "data", "error", "warnings"}`, en texte et en
`structuredContent`. `isError` vaut `!ok`.

## Extraction puis chirurgie

`surgeon_extract` (ou `surgeon extract` en ligne de commande) calcule
`v = normalize(mean(pos) - mean(neg))` sur les activations d'une couche, et écrit un
`.safetensors` avec un tenseur 1D F32 `direction` (taille `hidden_size`) et ses métadonnées
(couche, site, pooling, `d_prime`, `threshold_accuracy`, `pos_mean_proj`, `neg_mean_proj`,
`cos_mean_pos_neg`...). Un fichier compagnon `<out>.stats.json` (`"schema": 1`) garde la
projection de chaque prompt et le seuil ; `surgeon tui --vector` le dessine. Le résultat de
l'outil donne son chemin dans `data.stats_path`. Seuls Qwen2/Qwen2.5 sont supportés
(`model_type: "qwen2"`).

```bash
surgeon extract --model ./Qwen2.5-7B-Instruct --pos-prompts pos.json --neg-prompts neg.json \
    --layer 20 --chat-template chatml --out refusal.safetensors
```

Le vecteur se consomme ensuite dans un patch. Il vit dans le flux résiduel (taille
`hidden_size`) : pour `down_proj` (`[hidden, intermediate]`) il indexe les lignes, donc
`axis: "cols"` ; pour `gate_proj`/`up_proj` (`[intermediate, hidden]`) il indexe les
colonnes, donc `axis: "rows"`.

```json
{
  "schema_version": 1,
  "operations": [
    { "id": "refusal-down", "layers": "18-22", "projections": ["down_proj"],
      "operation": "subtract", "vector_file": "refusal.safetensors",
      "vector_tensor": "direction", "alpha": 0.5, "axis": "cols" }
  ]
}
```

## Mesure d'une chirurgie (`surgeon_bench`)

Un seul outil, un champ `mode` obligatoire et des champs typés optionnels (objet plat, sans
`oneOf`). Le serveur vérifie que les champs correspondent au mode : un champ requis absent ou
un champ d'un autre mode donne `isError: true`, `error.kind = "invalid_arguments"` et une
entrée par champ dans `error.issues` (`path`, `message`). Une propriété inconnue du schéma
est une erreur JSON-RPC `-32602` (les cinq autres outils la signalent en
`invalid_arguments`, comportement conservé).

| `mode` | Requis | Optionnels |
| --- | --- | --- |
| `logits` | `before`, `after` | `tensor`, `top_k` (5), `worst` (5), `warn_kl`, `warn_top1_shift`, `fail_on_warn` |
| `ppl` | `model`, `text` | `baseline_model` ou `baseline_ppl` (+ `baseline_ppl_err`), `ctx` (512), `chunks`, `threads`, `n_gpu_layers` (0), `llama_cpp_dir`, `warn_delta` (0.2), `timeout_secs`, `fail_on_warn` |
| `eval` | `model`, `dataset` | `baseline_model`, `limit` (50), `max_new_tokens` (512), `prompt_template` (texte avec `{question}`, jamais un chemin), `format` (`gsm8k`, `math`, `auto`), `llama_cpp_dir`, `timeout_secs`, `min_accuracy`, `fail_on_warn` |
| `dump` | `model`, `prompts` (fichier), `out` | `dtype`, `out_dtype` (`f32`, `f16`), `max_tokens` (256), `max_total_positions` (4096), `overwrite` |

`data` reçoit le rapport (`BenchLogitsReport`, `BenchPplReport`, `BenchEvalReport`,
`BenchDumpReport`), identique à `surgeon --json bench <mode>` ; les durées sont dans
`data.timing_ms`. Un seuil dépassé avec `fail_on_warn: true` donne `exit_code` 9 et
`isError: true`, le rapport restant dans `data`. Aucun argument libre n'atteint llama.cpp
par MCP (l'option `--llama-arg` n'existe qu'en ligne de commande, sur liste blanche).
`--read-only` refuse le mode `dump` mais laisse `logits`, `ppl` et `eval`.

Recette d'un agent :

1. `surgeon_extract` : la direction à retirer ou ajouter.
2. Copier le dossier du modèle (hors MCP), puis `surgeon_apply` **sur la copie**
   (`dry_run: true` d'abord).
3. `surgeon_bench` `mode: "dump"` sur l'original puis sur la copie, mêmes `prompts`.
4. `surgeon_bench` `mode: "logits"` : KL(before‖after), top-1, top-k ; `worst_prompts`
   montre où le modèle a le plus changé.
5. `surgeon_bench` `mode: "ppl"` avec `baseline_model` (l'original) : delta arrondi à 4
   décimales, `within_noise`.
6. `surgeon_bench` `mode: "eval"` avec `baseline_model` : Pass@1, intervalle de Wilson,
   questions perdues et gagnées.
7. Décider : garder la copie, ajuster `alpha` ou les couches, ou abandonner.

```json
{"name": "surgeon_bench", "arguments": {"mode": "logits", "before": "before.safetensors", "after": "after.safetensors", "warn_kl": 0.05}}
```

## Méthodes d'extraction, `project_out` et `surgeon_autotune`

`surgeon_extract` accepte `method` (`mean_diff` par défaut, sorties inchangées ; `svd` exige
autant de prompts positifs que négatifs, appariés par index, sinon `unpaired_prompts` ;
`leace` écrit en plus `<out>.leace.safetensors` avec les tenseurs `a`, `b`, `mean`,
`u_whitened`), `svd_center`, `leace_shrinkage` (défaut 0.1, 0 = pseudo-inverse),
`leace_rcond` (défaut 1e-10) et `holdout_frac` (diagnostic hors échantillon, 10 prompts par
classe au moins). `data.method_diagnostics` donne les diagnostics de la méthode
(`split_half_cosine`, `cross_cov_before`/`cross_cov_after`, `holdout_cross_cov_ratio`...).
L'effacement LEACE n'est exact qu'en échantillon : avec n < d, seul le ratio hors
échantillon dit s'il généralise.

`method` est un identifiant du registre des méthodes d'extraction du moteur (le même que
`surgeon extract --method` et `surgeon plugins list`). Le serveur n'utilise que le registre
intégré : il accepte exactement `mean_diff`, `svd` et `leace`, avec le schéma et les
messages d'erreur d'avant le registre (`unknown variant ...`, `invalid_arguments`). Une
méthode tierce (API `v0`) s'exécute par la bibliothèque, jamais par un argument d'outil :
voir [docs/PLUGINS.md](PLUGINS.md).

L'opération de patch `project_out` applique l'effaceur aux poids de `down_proj` :
`W ← W − alpha·a(bᵀW)`, `alpha` dans [0, 1], `vector_tensor` = `a` et `vector_tensor_b` = `b`
(obligatoires), pas d'`axis` autre que le défaut :

```json
{"id": "erase", "layers": "18-22", "projections": ["down_proj"], "operation": "project_out",
 "vector_file": "refusal.safetensors.leace.safetensors", "vector_tensor": "a",
 "vector_tensor_b": "b", "alpha": 1.0}
```

`surgeon_autotune` (`model`, `vector`, `eval_dataset` requis ; `target_layer_range` "A-B",
`readout_layer`, `proj`, `objective` `suppress`/`amplify`/`target`, `target_projection`,
`l2_reg` nombre ou "auto", `probe_alpha`, `alpha_max`, `min_alpha`, `max_forwards`,
`max_tokens`, `dtype`, `site`, `pooling`, `chat_template`, `out`, `overwrite`) mesure en
mémoire la réponse de la projection lue à la couche de lecture quand chaque couche candidate
est décalée, ajuste les α par ridge (λ fixe ou LOOCV), applique tous les α ensemble pour
vérifier, et renvoie le patch dans `data.patch` (écrit dans `out` seulement si fourni,
validé par `validate_patch` avant). Les couches au-delà de la lecture sont exclues avec un
avertissement ; `max_forwards` est vérifié avant tout calcul. Il ne mesure pas les effets
secondaires : enchaîner avec `surgeon_bench` sur une copie.

## GGUF quantifié (`surgeon_apply` avec `gguf: true`)

`surgeon_inspect` accepte un `.gguf` : `data.format = "gguf"`, et pour chaque tenseur MLP
(`blk.{i}.ffn_{gate,up,down}.weight`) son `ggml_type`, sa forme `[lignes, colonnes]`, sa
taille et `patchable` (F32, F16, BF16, Q8_0, Q4_K, Q6_K).

`surgeon_apply` gagne deux champs, le nombre d'outils ne change pas :

| Champ | Défaut | Rôle |
| --- | --- | --- |
| `gguf` | `false` | Obligatoire pour un `.gguf` : sans lui, `error.kind = "invalid_input"` (la re-quantification perd de l'information, jamais par surprise). |
| `requant` | `keep-scales` | `keep-scales` garde `d`, `dmin` et les sous-échelles ; `recompute` recalcule les échelles des blocs touchés. |

```json
{"name": "surgeon_apply", "arguments": {"model": "qwen2.5-7b-q4_k_m.gguf",
  "patch": {"schema_version": 1, "operations": [{"id": "d", "layers": "10-14",
  "projections": ["down_proj"], "operation": "subtract", "vector_file": "v.safetensors",
  "alpha": 0.5, "axis": "cols"}]}, "gguf": true, "dry_run": true}}
```

Mêmes garde-fous que pour les SafeTensors, en plus stricts :
- la sauvegarde est **toujours** créée, même avec `backup: false` et `--allow-no-backup` ;
- un marqueur `<modèle>.surgeon-inprogress` existe pendant l'écriture ; s'il est trouvé au
  départ, `error.kind = "apply_in_progress"` (restaurer le backup qu'il désigne) ;
- l'en-tête est haché avant et après, puis chaque tenseur patché est relu ;
- `--read-only` refuse l'outil, et `dry_run` calcule toutes les métriques sans écrire.

Erreurs propres au GGUF :

| `error.kind` | Signification |
| --- | --- |
| `unsupported_quant_type` | Une cible n'est pas en F32, F16, BF16, Q8_0, Q4_K ni Q6_K. |
| `invalid_gguf` | Fichier illisible, tronqué, big-endian ou multi-parties. |
| `invalid_target` | Cible non 2D, ou colonnes non multiples du bloc. |

`data.gguf.tensors[]` donne la fidélité par tenseur (`realized_over_intended`,
`correlation_intended_realized`, `unchanged_fraction`, `saturated_elements`,
`nmse_vs_exact`, `warning`), et `data.io` donne les blocs écrits, le débit et les défauts de
page (absent en `dry_run`). Un `warning` signale un patch perdu sous le demi-pas, ou des
valeurs écrêtées : il faut alors réessayer avec `requant: "recompute"`, puis valider avec
`surgeon_bench` (`ppl`) contre le fichier d'origine.

## Différence de checkpoints (`surgeon_diff`)

Équivalent de `surgeon diff` : pour chaque tenseur 2D, une approximation de rang `rank`
(SVD randomisé) de `tuned - base` ; pour chaque tenseur 1D, le delta exact. Le résultat est un
fichier `.surgery` neuf, jamais une modification en place. Un `out` existant est une erreur
sauf avec `overwrite: true`.

| Champ | Requis | Défaut | Rôle |
| --- | --- | --- | --- |
| `base` | oui | | Checkpoint de base (`.safetensors` ou dossier shardé) |
| `tuned` | oui | | Checkpoint affiné, mêmes noms, formes et dtypes |
| `out` | oui | | `.surgery` écrit |
| `rank` | non | 16 | Rang de l'approximation (au moins 1) |
| `factor_dtype` | non | `f16` | `f16`, `bf16` ou `f32` |
| `include`, `exclude` | non | | Regex sur le nom des tenseurs |
| `include_embeddings` | non | `false` | Inclut `embed_tokens` et `lm_head` |
| `max_rel_error` | non | | Seuil d'erreur relative de reconstruction |
| `on_high_error` | non | `warn` | `warn`, `skip` ou `full` (delta exact) au-delà du seuil |
| `seed` | non | 0 | Graine du SVD randomisé |
| `overwrite` | non | `false` | Remplace `out` |

Les options `--oversample`, `--power-iters` et `--timestamp` de la ligne de commande ne sont pas
exposées : les valeurs par défaut s'appliquent et l'en-tête reste reproductible.

```json
{ "name": "surgeon_diff",
  "arguments": { "base": "base/model.safetensors", "tuned": "tuned/model.safetensors",
                 "out": "finetune.surgery", "rank": 16 } }
```

Succès, `structuredContent` réel du serveur (liste des tenseurs abrégée à deux entrées, chemin
remplacé par `<MODELS_DIR>`, sur la paire factice de `cargo run --example make_fake_checkpoints`) :

```json
{"command":"diff","data":{"surgery":{"bytes":9908,"factor_dtype":"f16","out":"<MODELS_DIR>/mcp.surgery","rank":4,"sha256":"08780993d50f2d3efb5541b3ec433fd470adbe9957df5a59c239305a8bc16b36","skipped_filtered":2,"skipped_identical":1,"tensors":[{"energy":1.0,"kind":"full_delta","name":"model.layers.0.input_layernorm.weight","rank":0,"rel_error":0.0},{"energy":0.9999999737902472,"kind":"lowrank","name":"model.layers.0.mlp.down_proj.weight","rank":4,"rel_error":4.455349812212725e-8}],"warnings":[]}},"error":null,"exit_code":0,"ok":true,"warnings":[]}
```

`skipped_filtered` compte `embed_tokens` et `lm_head` (exclus par défaut), `skipped_identical` les
tenseurs strictement identiques. Ce que le résultat ne dit pas tout seul : un `.surgery` est une
APPROXIMATION du fine-tune complet (exact seulement si la différence a un rang au plus `rank`,
ce qui est le cas de cette paire factice) et sa taille vaut environ la somme de `r (m + n)` fois
les octets du dtype, soit des centaines de Mo pour un 7B à rang 16. Il ne s'applique pas via
`surgeon_apply` (`surgeon apply --patch x.surgery`, ligne de commande uniquement).

## Fusion et extraction LoRA (`surgeon_lora`)

Un seul outil, un champ `mode` obligatoire (`merge` ou `extract`) et des champs typés optionnels
(objet plat, sans `oneOf`), comme `surgeon_bench`. Le serveur vérifie que les champs
correspondent au mode : un champ requis absent ou un champ de l'autre mode donne
`isError: true`, `error.kind = "invalid_arguments"` et une entrée par champ dans `error.issues`.
Une propriété inconnue du schéma est aussi une `invalid_arguments`.

| `mode` | Requis | Optionnels |
| --- | --- | --- |
| `merge` | `model`, `lora` | `alpha`, `adapter_config`, `scale` (1), `dry_run`, `backup_dir` |
| `extract` | `base`, `patched`, `out` | `rank` (16), `dtype` (`f32`, `f16`, `bf16`), `target_modules` (regex), `seed` (0), `overwrite` |

`merge` calcule `W <- W + scale * (lora_alpha / r) * B A` en place. `lora_alpha` vient du
`adapter_config.json` voisin de `lora` (ou de `adapter_config`) ; `alpha` le remplace ; sans
l'un ni l'autre, c'est une erreur d'usage. La sauvegarde est TOUJOURS faite : il n'existe pas de
paramètre pour la sauter, et `--allow-no-backup` ne concerne que `backup: false` d'un patch
`surgeon_apply`. Une erreur ou une annulation (`notifications/cancelled`) restaure les fichiers
(`error.kind = "cancelled"` dans le second cas). Faire d'abord `dry_run: true`.

`extract` écrit un adaptateur de rang `rank` au format PEFT (`out`, plus `adapter_config.json`
à côté) sans toucher aux modèles. Limites : le format PEFT n'est pas testé contre HuggingFace en
CI ; la fusion dans un GGUF n'est pas supportée.

```json
{ "name": "surgeon_lora",
  "arguments": { "mode": "merge", "model": "model.safetensors",
                 "lora": "adapter/adapter_model.safetensors", "dry_run": true } }
```

```json
{ "name": "surgeon_lora",
  "arguments": { "mode": "extract", "base": "base/model.safetensors",
                 "patched": "tuned/model.safetensors",
                 "out": "adapter/adapter_model.safetensors", "rank": 16 } }
```

Succès de `merge` avec `dry_run: true` (rien n'est écrit, `elements_modified` vaut 0 ; liste
abrégée à deux entrées) :

```json
{"command":"lora merge","data":{"backups":[],"lora":{"dry_run":true,"effective_scale":1.0,"lora_alpha":4.0,"max_abs_delta":0.0,"per_tensor":[{"elements_modified":0,"max_abs_delta":0.0,"name":"model.layers.0.mlp.down_proj.weight"},{"elements_modified":0,"max_abs_delta":0.0,"name":"model.layers.0.mlp.up_proj.weight"}],"rank":4,"tensors":6,"warnings":[]},"warnings":[]},"error":null,"exit_code":0,"ok":true,"warnings":[]}
```

Succès de `extract` (le dossier de `out` doit déjà exister ; un `out` existant est refusé sans
`overwrite: true`) :

```json
{"command":"lora extract","data":{"lora":{"config":"<MODELS_DIR>/ex2/adapter_config.json","dtype":"f32","mean_rel_error":4.325878840151349e-8,"out":"<MODELS_DIR>/ex2/adapter_model.safetensors","per_tensor":[{"energy":0.9999999737902472,"name":"model.layers.0.mlp.down_proj.weight","rel_error":4.455349812212725e-8},{"energy":0.9999999835510746,"name":"model.layers.0.mlp.up_proj.weight","rel_error":3.924179396550512e-8}],"rank":4,"tensors":6,"warnings":[]},"warnings":[]},"error":null,"exit_code":0,"ok":true,"warnings":[]}
```

## Garde-fous communs aux deux outils

- Tous les chemins (`base`, `tuned`, `out`, `model`, `lora`, `adapter_config`, `backup_dir`,
  `patched`) sont confinés aux dossiers `--allow-dir`, relatifs au premier dossier autorisé.
- Avec un `progressToken` dans `_meta`, un `notifications/progress` est envoyé par tenseur traité.
- Les erreurs sont des résultats `isError: true` avec l'enveloppe habituelle ; `command` vaut
  `diff`, `lora merge` ou `lora extract`.

Exemples d'erreurs, sorties réelles du serveur (`structuredContent` de chaque réponse) :

Champ d'un autre mode (`surgeon_lora`, mode `merge` avec `out` et sans `lora`) :

```json
{"command":"lora merge","data":null,"error":{"issues":[{"message":"requis en mode merge","path":"lora"},{"message":"non utilisé en mode merge","path":"out"}],"kind":"invalid_arguments","message":"champs incohérents avec le mode merge"},"exit_code":2,"ok":false,"warnings":[]}
```

Chemin hors des dossiers autorisés (`surgeon_diff`, `base` hors `--allow-dir`) :

```json
{"command":"diff","data":null,"error":{"kind":"path_not_allowed","message":"« base » : chemin hors des dossiers autorisés (--allow-dir)"},"exit_code":2,"ok":false,"warnings":[]}
```

Serveur lancé avec `--read-only` :

```json
{"command":"diff","data":null,"error":{"kind":"read_only","message":"serveur lancé avec --read-only : surgeon_diff écrit des fichiers, il est désactivé"},"exit_code":2,"ok":false,"warnings":[]}
```

## Garde-fous

- **Chemins confinés** : chaque chemin (modèle, prompts, `vector_file`, `backup_dir`,
  `export_gguf`, `llama_cpp_dir`, `base_dir`, `out` de l'extraction, pour `surgeon_bench`
  `before`, `after`, `text`, `dataset`, `prompts`, `baseline_model`, `out`, et pour
  `surgeon_autotune` `vector`, `eval_dataset`, `out`)
  est résolu par le système (liens symboliques, `..`) et doit tomber sous un dossier
  autorisé, sinon `error.kind = "path_not_allowed"`, sans révéler la cible.
- **Sauvegarde imposée** avant toute écriture, vérifiée par SHA-256 ; une sauvegarde ratée
  bloque l'écriture.
- **Rollback automatique** sur erreur, NaN/Inf, annulation (`notifications/cancelled`,
  `error.kind = "cancelled"`) ou panic. Un panic devient une erreur `-32603` et le
  serveur continue.
- **Progression** : avec `_meta.progressToken`, une `notifications/progress` par tenseur
  (apply), par prompt (eval), ou une pour les poids puis une par prompt (extract) ; pour
  `surgeon_bench`, par prompt (`dump`), par bloc de 64 positions (`logits`), par chunk de
  llama-perplexity (`ppl`, `total` absent tant qu'il n'est pas annoncé) et par question
  (`eval`) ; pour `surgeon_autotune`, une par forward.
- **Annulation de `surgeon_bench`** : `notifications/cancelled` tue et attend le processus
  llama.cpp en cours (aucun zombie) ; un dump annulé ne laisse aucun fichier.
- **Extraction sans écriture partielle** : fichiers temporaires puis `rename`, vecteur et
  compagnon ensemble ; un `out` ou un compagnon existant exige `overwrite: true` ;
  annulation ou erreur, aucun fichier n'est laissé. `--read-only` refuse `surgeon_extract`.
- **Pas de tableau de bord** : `surgeon tui` est interactif et n'est pas exposé en MCP.

## Protocole

Versions acceptées à l'`initialize` : `2024-11-05`, `2025-03-26`, `2025-06-18`,
`2025-11-25` (révisions à handshake). Une version inconnue reçoit la plus récente. La
révision `2026-07-28`, sans handshake, n'est pas implémentée : un client récent qui sonde
`server/discover` reçoit `-32601` et revient à `initialize`.

| Code | Cas |
| --- | --- |
| `-32700` | JSON invalide (`id: null`) |
| `-32600` | Requête invalide, batch, `initialize` en double, ligne de plus de 16 Mio |
| `-32601` | Méthode inconnue |
| `-32602` | `tools/call` mal formé, outil inconnu, propriété inconnue pour `surgeon_bench` |
| `-32603` | Panic pendant un outil |
| `-32002` | `tools/*` avant `initialize` |
