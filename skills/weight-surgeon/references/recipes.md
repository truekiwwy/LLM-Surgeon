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

`surgeon_extract` en balayage (toutes les couches en un appel) :

```json
{"model": "<ORIGINE>", "layers": "all", "pos_prompts_file": "builtin:refusals",
 "neg_prompts_file": "builtin:general", "method": "mean_diff",
 "out": "work/refusal_sweep.safetensors"}
```

La couche retenue, seule, pour le d′ hors échantillon :

```json
{"model": "<ORIGINE>", "layer": 14, "pos_prompts_file": "builtin:refusals",
 "neg_prompts_file": "builtin:general", "method": "mean_diff", "holdout_frac": 0.2,
 "out": "work/dir_L14.safetensors"}
```

`surgeon_causal_trace` (paires appariées par index, autant de chaque côté) :

```json
{"model": "<ORIGINE>", "clean_prompts_file": "work/refused.json",
 "corrupted_prompts_file": "work/accepted.json", "component": "mlp", "layers": "all"}
```

`data.window.layers` (`"12-18"`) va tel quel dans `layers` d'une opération ou de
`surgeon_compose`.

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

## Balayage Pareto et checkpoints

`surgeon_checkpoint` avant une série d'essais :

```json
{"action": "create", "model": "copies/copy-1", "label": "avant patch"}
```

`surgeon_sweep` (la copie finit sur le meilleur compromis) :

```json
{"model": "copies/copy-1", "vectors": ["work/dir_L14.safetensors"],
 "operation": "project_out", "alphas": [0.4, 0.6, 0.8, 1.0], "layers": ["all", "10-14"],
 "max_drift": 0.05,
 "target": {"kind": "unblocking", "dataset": "builtin:refusals", "max_new_tokens": 40},
 "drift": {"kind": "kl", "dataset": "builtin:general"}}
```

Passer sur un autre point du front (`data.points[i].checkpoint`) :

```json
{"action": "restore", "model": "copies/copy-1", "id": "0003"}
```

`surgeon_checkpoint` `list` : `{"action": "list"}`

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

## Extraction en LoRA (après certification)

La copie certifiée devient un adaptateur PEFT de rang `rank` (8, 16, 32 ou 64) :
`surgeon_lora` mode `extract`, `base` = l'origine (lue, jamais écrite), `patched` = la
copie, `out` = un dossier existant qui reçoit `adapter_model.safetensors` et
`adapter_config.json`. `data.lora.per_tensor` donne l'erreur relative de chaque matrice ;
`1 − mean_rel_error` est la fidélité de l'adaptateur. Un `.safetensors` de la différence
elle-même passe en `delta` à la place de `patched`.

```json
{"mode": "extract", "base": "<ORIGINE>", "patched": "copies/copy-1",
 "out": "out/lora-r16", "rank": 16}
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

Contrôle de perplexité sans llama.cpp :

```json
{"mode": "ppl", "backend": "native", "model": "copies/copy-1", "text": "builtin:wikitext",
 "baseline_model": "<ORIGINE>"}
```

Lis `data.delta.delta_relative_pct` : jusqu'à +2 % correct, jusqu'à +5 % à surveiller, au-dessus
une régression à examiner. Cela ne remplace pas la génération complète ci-dessus.

## Journal

`record_finding` :

```json
{"text": "Refus 0.11 contre 0.39 à l'origine après project_out alpha 0.6",
 "evidence": [{"run_id": "mcp-1795000000000-7", "metric": "refusal_rate", "value": 0.11}]}
```

`update_plan` : `{"steps": ["Diagnostic", "Sonde des couches", "Crible alpha", "Certification"]}`
