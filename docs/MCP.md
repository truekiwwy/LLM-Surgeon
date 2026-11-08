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
| `--read-only` | Refuse `surgeon_apply` et `export_gguf` (`error.kind = "read_only"`). |
| `--allow-no-backup` | Respecte `backup: false` dans un patch. Sans cette option, la sauvegarde est imposée. |

`$LLAMA_CPP_DIR`, défini par l'opérateur, sert de dossier llama.cpp par défaut pour
`surgeon_eval`.

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

Les schémas exacts : `surgeon schema --kind tools`. Le format du patch :
`surgeon schema` (JSON Schema draft-07, sans `$ref`).

Chaque résultat contient la même enveloppe que `surgeon --json` :
`{"ok", "command", "exit_code", "data", "error", "warnings"}`, en texte et en
`structuredContent`. `isError` vaut `!ok`.

## Extraction puis chirurgie

`surgeon_extract` (ou `surgeon extract` en ligne de commande) calcule
`v = normalize(mean(pos) - mean(neg))` sur les activations d'une couche, et écrit un
`.safetensors` avec un tenseur 1D F32 `direction` (taille `hidden_size`) et ses métadonnées
(couche, site, pooling...). Seuls Qwen2/Qwen2.5 sont supportés (`model_type: "qwen2"`).

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

## Garde-fous

- **Chemins confinés** : chaque chemin (modèle, prompts, `vector_file`, `backup_dir`,
  `export_gguf`, `llama_cpp_dir`, `base_dir`, `out` de l'extraction) est résolu par le système (liens
  symboliques, `..`) et doit tomber sous un dossier autorisé, sinon
  `error.kind = "path_not_allowed"`, sans révéler la cible.
- **Sauvegarde imposée** avant toute écriture, vérifiée par SHA-256 ; une sauvegarde ratée
  bloque l'écriture.
- **Rollback automatique** sur erreur, NaN/Inf, annulation (`notifications/cancelled`,
  `error.kind = "cancelled"`) ou panic. Un panic devient une erreur `-32603` et le
  serveur continue.
- **Progression** : avec `_meta.progressToken`, une `notifications/progress` par tenseur
  (apply), par prompt (eval), ou une pour les poids puis une par prompt (extract).
- **Extraction sans écriture partielle** : fichier temporaire puis `rename` ; un `out`
  existant exige `overwrite: true` ; annulation ou erreur, aucun fichier n'est laissé.
  `--read-only` refuse `surgeon_extract`.

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
| `-32602` | `tools/call` mal formé ou outil inconnu |
| `-32603` | Panic pendant un outil |
| `-32002` | `tools/*` avant `initialize` |
