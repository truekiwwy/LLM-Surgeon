# Déroulé détaillé et règles de décision

Ce fichier complète `SKILL.md` : ce qu'il faut lire dans chaque résultat et comment décider.
Les chemins relatifs partent du dossier projet (le premier dossier autorisé du serveur).

## Disposition d'un projet

| Chemin | Écrit par | Rôle |
| --- | --- | --- |
| `project.surgeon.json` | `surgeon_workspace` `init` (une fois) ou le Studio | marque le projet et son origine (chemin, SHA-256) ; sans lui, rien n'est journalisé |
| `copies/<id>/` | l'agent (copie du dossier du modèle) ou le Studio | la copie de travail, la plus récente sert par défaut |
| `work/` | `surgeon_files`, les sorties `out` / `patch_out` | prompts, jeux de sécurité, directions, `patch_*.json` |
| `work/checkpoints/` | `surgeon_checkpoint`, `surgeon_sweep` | instantanés de la copie par liens physiques (manifeste SHA-256, métriques, parent) ; aucun autre outil n'y écrit |
| `chat/mcp_runs.jsonl` | le serveur | un run par appel d'outil : `run_id`, outil, arguments, `ok`, `data` |
| `chat/findings.jsonl` | le serveur (`record_finding`) | le journal des preuves |
| `chat/plan.json`, `chat/focus.json` | le serveur (`update_plan`, `show_in_3d`) | plan et sélection 3D |
| `ledger.jsonl`, `runs/` | le Studio | ses propres jobs ; jamais l'agent |

Le serveur doit pouvoir lire le modèle d'origine : lance-le avec `--allow-read <dossier du
modèle>` (ou mets le modèle sous le dossier projet). `surgeon_files` action `list` donne
`writable_roots`, `read_only_roots` et `work_dir`.

## 1. Diagnostic (`surgeon_doctor`)

| Champ | Lecture |
| --- | --- |
| `data.ready` | `false` : un contrôle échoue, lis `data.checks[].message` |
| `data.memory.model.dtype` | `f32` normal ; `bf16` repli lent ; `none` le modèle ne se charge pas |
| `data.memory.model.batch_size` | séquences que la génération native décode ensemble |
| `data.cache.corrupt`, `data.cache.obsolete` | au-dessus de 0 : purge à proposer (l'agent ne purge pas) |
| `data.hipfire.up` | `/health` répond |
| `data.hipfire.gpu_memory.total_mb` | 0 : démon figé, redémarrage nécessaire avant toute mesure hipfire |

## 2. Sonde des couches (`surgeon_extract`)

- Un balayage (`layers` `"all"`) : une direction par couche en une passe avant par prompt,
  `auc`, `margin` et `d_prime` par couche dans `data.layers`, mesurés sur tous les prompts.
  Une AUC de 1 sur beaucoup de couches est normale en petit échantillon : départage par d′.
- La couche retenue (`data.best_layer` ou ton choix motivé) est refaite seule avec
  `holdout_frac` 0.2 : `d_prime` et `threshold_accuracy` hors échantillon. Un avertissement
  `d′ < 1` signale une direction peu fiable : dis-le.
- `surgeon_causal_trace` (lecture seule) dit où le comportement se joue : score de
  restauration par couche, `ranking`, `window.layers`. Un pic en `mlp` désigne `down_proj`,
  un pic en `attn` désigne `o_proj`. En `resid`, la dernière couche vaut 1 par construction :
  lis la montée, pas la fin.
- Ne choisis jamais une couche ou une plage sans cette sonde. Consigne le classement et la
  fenêtre (`record_finding`, une valeur par run cité).

## 3. Balayage d'alpha en entonnoir

1. `surgeon_restore` sur la copie (retour à l'état d'avant le dernier `surgeon_apply`).
2. `surgeon_apply` `dry_run: true` : vérifie la liste des tenseurs et `vector_len`.
3. `surgeon_apply` réel.
4. `surgeon_bench` `refusal_probe` : lis `data.probe_refusal_rate` (et
   `data.mean_refusal_prob`). C'est un criblage : il élimine, il ne prouve pas.
5. Garde les deux plus petits alpha dont le refus sondé baisse ; eux seuls passent en
   génération complète (section 5).

Garde le plus petit alpha qui atteint l'objectif. Après une baisse de rétention, n'augmente
pas alpha sur les mêmes couches : réduis-le, change de couches ou compose la direction.

## 3 bis. Balayage Pareto (`surgeon_sweep`)

- Cible (`target.kind`) : `unblocking` pour des refus excessifs (gain = plus de réponses),
  `safety` pour un comportement à renforcer, `gsm8k` pour une capacité. Le jeu de la cible
  et celui de la dérive (`drift.dataset`, des prompts neutres) sont distincts.
- `max_drift` : la dérive moyenne acceptée contre la copie de départ (KL par défaut). Un
  budget trop serré donne `drift_exceeded` partout : la copie revient à la base, aucun
  checkpoint n'est gardé, dis-le au lieu de relâcher le budget seul.
- Lis `data.points[].decision` (`kept`, `dominated`, `drift_exceeded`), `data.front` (indices
  par dérive croissante) et `data.best`. Consigne le front avec `record_finding`, une valeur
  par point cité, et le `run_id` du sweep.
- `operation` : `project_out` retire une direction, `add_direction` l'amplifie (`alphas` >= 0),
  `subtract` / `add` décalent par un vecteur. Dans les deux cas (meilleur point ou base si aucun
  gain), le sweep hache de nouveau la copie et la compare au SHA-256 du checkpoint
  `data.restored` : `checksum_mismatch` en cas d'écart.
- Le sweep a déjà mis la copie sur `data.restored`. Pour comparer un autre point du front :
  `surgeon_checkpoint` `restore` avec son `checkpoint`, puis certifie (section 5).
- `surgeon_checkpoint` `create` avant une série manuelle de `surgeon_apply` donne un retour
  sûr à cet état, quel que soit le nombre d'applies ; `list` donne `disk_bytes`, ce que
  l'arbre coûte vraiment sur disque.

## 4. Composition orthogonale (`surgeon_compose`)

- `ablate` : directions à retirer (`chemin` ou `chemin:alpha`), `keep` : directions de
  capacité extraites à la même couche.
- Lis `data.composition` (cosinus avant et après, ratio de norme) : un cosinus résiduel
  proche de 0 avec chaque direction gardée est attendu.
- `collinear_vectors` : les directions ne se séparent pas linéairement à cette couche.
- Plusieurs directions à retirer : `mode` `ties` (`density`) ou `dare` (`drop_rate`,
  `seed`) les fusionnent en limitant leurs interférences ; la recette du patch garde ces
  réglages. Une seule direction : reste en `orthogonal_preserve`.
- L'orthogonalité est linéaire : seule la génération complète dit si la capacité tient.

## 5. Certification

Sur chaque finaliste, puis sur la copie dans son état final, avec `baseline_model` = origine :

| Mesure | Lecture | Régression |
| --- | --- | --- |
| `refusal` sur `builtin:refusals` | `data.refusal_rate` contre `data.baseline.refusal_rate` | l'objectif n'est pas atteint |
| `eval` sur `builtin:math`, `builtin:logic`, `builtin:general` | `data.accuracy` et `data.ci95_low` / `data.ci95_high` contre `data.baseline`, `data.paired.lost` | baisse hors de l'intervalle rapporté |
| `refusal` sur le jeu de sécurité de `work/` | taux de refus sur des demandes dangereuses | toute baisse |
| `ppl` avec `backend: "native"` sur `builtin:wikitext` (sans llama.cpp) | delta relatif de perplexité (`data.delta.delta_relative_pct`, en %) | au-dessus de +5 % ; +2 % à +5 % à surveiller |

Une écriture du modèle après une mesure rend la mesure caduque. Un candidat qui régresse :
`surgeon_restore`, vérifie `data.identical` et que `sha256_after` égale `sha256_source`.

## 6. Backend et hipfire

hipfire sert un cache quantifié enregistré par nom et ne lit pas les poids modifiés : une
mesure d'une copie de travail passe sur `native` (avertissement « backend hipfire écarté »),
sauf si l'opérateur a activé `hipfire_requantize = "auto"` (la copie est alors requantifiée
sous l'alias `ws-<copie>-<sha8>`). L'agent ne requantifie jamais lui-même.
