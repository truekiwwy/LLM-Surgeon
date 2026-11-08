# surgeon : serveur MCP

`surgeon mcp` expose la chirurgie de poids à un agent IA via le Model Context Protocol, en
JSON-RPC 2.0 sur stdio (un message par ligne). stdout ne transporte que le protocole ; les
journaux partent sur stderr, sans couleur.

## Lancement

```bash
surgeon mcp --allow-dir /chemin/vers/modeles
surgeon mcp --allow-dir /modeles --allow-dir /directions --read-only
```

| Option | Effet |
| --- | --- |
| `--allow-dir <dossier>` | Dossier autorisé, répétable. Défaut : `$SURGEON_MCP_ALLOW_DIRS` (séparateur du système : `;` sous Windows, `:` ailleurs), sinon le dossier courant. |
| `--read-only` | Refuse `surgeon_apply`, `surgeon_extract`, `export_gguf` et `surgeon_bench` en mode `dump` (`error.kind = "read_only"`). |
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

## Outils

| Outil | Rôle | Annotations |
| --- | --- | --- |
| `surgeon_check` | Budget RAM/VRAM du modèle | lecture seule |
| `surgeon_inspect` | Couches et matrices MLP, statistiques en option, `max_tensors` (défaut 500) | lecture seule |
| `surgeon_apply` | Applique un patch en place (`dry_run` pour vérifier d'abord) | destructif |
| `surgeon_eval` | Non-régression (boucles, attentes) et export GGUF conditionnel | écrit le GGUF |
| `surgeon_extract` | Vecteur de direction depuis une couche Qwen2 (Rust pur, CPU) | écrit le `.safetensors`, idempotent |
| `surgeon_bench` | Mesure d'une chirurgie : `mode` `dump`, `logits`, `ppl` ou `eval` | non destructif, idempotent |

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

## Garde-fous

- **Chemins confinés** : chaque chemin (modèle, prompts, `vector_file`, `backup_dir`,
  `export_gguf`, `llama_cpp_dir`, `base_dir`, `out` de l'extraction, et pour
  `surgeon_bench` `before`, `after`, `text`, `dataset`, `prompts`, `baseline_model`, `out`)
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
  (`eval`).
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
