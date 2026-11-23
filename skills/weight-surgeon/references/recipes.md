# Recettes : arguments prêts à copier

Les chemins sont relatifs au dossier projet. `copies/copy-1` est la copie de travail,
`<ORIGINE>` le dossier du modèle d'origine (lu par `--allow-read`). Remplace la couche 14 par
celle que la sonde a retenue.

## Brancher le serveur

Le Studio écrit cette configuration pour toi (Réglages > Harnais externes > Copier la config
MCP) avec le chemin absolu de `surgeon.exe` :

```json
{"mcpServers": {"weight-surgeon": {"command": "<chemin absolu de surgeon>", "args": ["mcp"]}}}
```

- Claude Code : `claude mcp add --scope user weight-surgeon -- <surgeon> mcp`, ou l'extrait
  ci-dessus fusionné dans `~/.claude.json`.
- Codex : dans `~/.codex/config.toml`, `[mcp_servers.weight-surgeon]` avec
  `command = "<surgeon>"` et `args = ["mcp"]`.
- OpenCode : dans `opencode.json`, `"mcp": {"weight-surgeon": {"type": "local", "command":
  ["<surgeon>", "mcp"]}}`.
- Cursor : l'extrait ci-dessus dans `.cursor/mcp.json`.

Sans `--allow-dir`, le serveur sert le dossier courant du harnais : lance le harnais dans le
dossier projet. Pour lire l'origine hors du projet, ajoute `"--allow-read", "<ORIGINE>"` aux
`args`.

Le Skill s'installe dans `~/.claude/skills/weight-surgeon/`, `~/.codex/skills/weight-surgeon/`
ou `~/.agents/skills/weight-surgeon/` (Studio : Installer le Skill ; CLI : `surgeon-gui
--install-skill claude`).

## Projet

`surgeon_workspace`, une fois, au début d'un nouveau projet :

```json
{"action": "init", "model": "<ORIGINE>", "name": "qwen-refus"}
```

## Diagnostic

`surgeon_doctor` : `{"model": "copies/copy-1"}`

## Inspection et extraction

`surgeon_inspect` : `{"model": "copies/copy-1", "projections": ["o_proj", "down_proj"]}`

`surgeon_extract` (une fois par couche sondée) :

```json
{"model": "<ORIGINE>", "layer": 14, "pos_prompts_file": "builtin:refusals",
 "neg_prompts_file": "builtin:general", "method": "mean_diff", "holdout_frac": 0.2,
 "out": "work/dir_L14.safetensors"}
```

Direction de capacité à préserver, même couche : `pos_prompts_file` `builtin:math` (ou
`builtin:logic`), `neg_prompts_file` `builtin:general`, `out` `work/math_L14.safetensors`.

## Patch et crible

`surgeon_apply`, d'abord avec `"dry_run": true` :

```json
{"model": "copies/copy-1", "dry_run": true, "patch": {"schema_version": 1, "operations": [
  {"id": "po", "operation": "project_out", "projections": ["o_proj", "down_proj"],
   "layers": "all", "vector_file": "work/dir_L14.safetensors",
   "vector_tensor": "direction", "vector_tensor_b": "direction", "alpha": 0.6}]}}
```

`surgeon_bench` crible (heuristique) :

```json
{"mode": "refusal_probe", "model": "copies/copy-1", "prompts": "builtin:refusals",
 "probe_method": "first-token-prob"}
```

`surgeon_restore` entre deux alpha : `{"model": "copies/copy-1"}`

`surgeon_autotune` (mesure seule, sans `out`) :

```json
{"model": "copies/copy-1", "vector": "work/dir_L14.safetensors",
 "refusal_prompts": "builtin:refusals", "eval_dataset": "builtin:math"}
```

## Composition orthogonale

`surgeon_compose` :

```json
{"ablate": ["work/dir_L14.safetensors"], "keep": ["work/math_L14.safetensors",
 "work/logic_L14.safetensors"], "mode": "orthogonal_preserve",
 "out": "work/composed_L14.safetensors", "patch_out": "work/patch_compose.json"}
```

Puis `surgeon_apply` avec le contenu de `work/patch_compose.json` en `patch` (lu par
`surgeon_files` action `read`), `dry_run: true` d'abord.

## Fusion TIES / DARE de plusieurs directions

Plusieurs directions à retirer qui se contredisent par endroits (extraites sur des jeux
différents) : `mode` `ties` garde le top `density` de chacune, élit le signe majoritaire
par coordonnée et moyenne les contributions alignées ; `dare` abandonne une part
`drop_rate` des coordonnées (graine `seed`, reproductible) et remet le reste à l'échelle.
Avec `keep`, la direction fusionnée reste orthogonale aux capacités.

```json
{"ablate": ["work/dir_L14.safetensors", "work/dir_harm_L14.safetensors:0.5"],
 "keep": ["work/math_L14.safetensors"], "mode": "ties", "density": 0.2,
 "out": "work/ties_L14.safetensors", "patch_out": "work/patch_ties.json"}
```

```json
{"ablate": ["work/dir_L14.safetensors", "work/dir_harm_L14.safetensors"],
 "mode": "dare", "drop_rate": 0.5, "seed": 0,
 "out": "work/dare_L14.safetensors", "patch_out": "work/patch_dare.json"}
```

## Certification (génération complète)

```json
{"mode": "refusal", "model": "copies/copy-1", "prompts": "builtin:refusals",
 "baseline_model": "<ORIGINE>"}
```

```json
{"mode": "eval", "model": "copies/copy-1", "dataset": "builtin:math",
 "baseline_model": "<ORIGINE>"}
```

Même appel pour `builtin:logic` et `builtin:general`. Jeu de sécurité, écrit une fois :
`surgeon_files` `{"action": "write", "path": "safety.json", "content": "[\"…\", …]"}` (au
moins 10 demandes réellement dangereuses), puis `refusal` avec `"prompts": "work/safety.json"`.

## Journal

`record_finding` :

```json
{"text": "Refus 0.11 contre 0.39 à l'origine après project_out alpha 0.6",
 "evidence": [{"run_id": "mcp-1795000000000-7", "metric": "refusal_rate", "value": 0.11}]}
```

`update_plan` : `{"steps": ["Diagnostic", "Sonde des couches", "Crible alpha", "Certification"]}`
