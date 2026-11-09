<!-- Fragment fusionné dans docs/MCP.md par l'orchestrateur : lignes du tableau des outils, section dédiée, garde-fous. -->

Lignes à ajouter au tableau des outils :

| Outil | Rôle | Annotations |
| --- | --- | --- |
| `surgeon_diff` | Différence de deux checkpoints, approximée à rang `r`, écrite dans un `.surgery` neuf | écrit un fichier, non destructif, idempotent |
| `surgeon_lora` | `mode` `merge` (fusionne un adaptateur LoRA en place) ou `extract` (écrit un adaptateur) | destructif |

À ajouter à la ligne `--read-only` : `surgeon_diff` et `surgeon_lora` (les deux modes) sont
refusés avec `error.kind = "read_only"`.

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

Succès : `data.surgery` décrit le fichier écrit (taille, SHA-256, rang, dtype des facteurs,
erreur relative par tenseur, tenseurs ignorés, avertissements). Ce que le résultat ne dit pas
tout seul : un `.surgery` est une APPROXIMATION du fine-tune complet (exact seulement si la
différence a un rang au plus `rank`) et sa taille vaut environ la somme de `r (m + n)` fois les
octets du dtype, soit des centaines de Mo pour un 7B à rang 16. Il ne s'applique pas via
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

Succès : `data.lora` (rapport de la fusion ou de l'extraction).

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
