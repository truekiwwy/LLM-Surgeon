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
surgeon mcp --allow-dir /projet --allow-read /modeles/origine
```

| Option | Effet |
| --- | --- |
| `--allow-dir <dossier>` | Dossier autorisé, répétable. Défaut : `$SURGEON_MCP_ALLOW_DIRS` (séparateur du système : `;` sous Windows, `:` ailleurs), sinon le dossier courant. |
| `--allow-read <chemin>` | Dossier ou fichier lisible mais jamais écrit (le modèle d'origine), répétable. Les chemins que les outils d'analyse et de mesure lisent y sont acceptés : `model` de `surgeon_check`, `surgeon_inspect`, `surgeon_eval`, `surgeon_extract`, `surgeon_autotune`, `surgeon_graph` ; `baseline` et `vector` de `surgeon_graph` ; les modèles, jeux, textes, `responses`, `labels` et `baseline_model` de `surgeon_bench` ; `from_origin`, `backup` et `backup_dir` de `surgeon_restore` ; `ablate` et `keep` de `surgeon_compose` ; `refusal_prompts` et `retention_dataset` de `surgeon_autotune` ; les fichiers de prompts, d'étiquettes et `arch_spec`. Tout chemin écrit ou exécuté (`out`, `out_dir`, `cache_dir`, `export_gguf`, `backup_dir` de `surgeon_apply`, `judge_cache`, `llama_cpp_dir`), le `model` de `surgeon_restore`, tout `surgeon_apply`, `surgeon_diff`, `surgeon_lora` et `surgeon_pipeline` restent limités à `--allow-dir` (`error.kind = "path_not_allowed"`, message « dossier en lecture seule »). |
| `--read-only` | Refuse `surgeon_apply` (sauf `dry_run`), `surgeon_restore`, `surgeon_compose`, l'action `write` de `surgeon_files`, `surgeon_extract`, `export_gguf`, `surgeon_bench` en modes `dump` et `judge_sample` (et `judge_eval` avec `out`), `surgeon_autotune` avec `out`, `surgeon_diff`, `surgeon_lora` (les deux modes), `surgeon_graph` avec `out_dir` ou `cache_dir` et les actions `start`, `cancel`, `promote_patch` de `surgeon_pipeline` (`error.kind = "read_only"`) ; `surgeon_autotune` sans `out` et `surgeon_graph` sans sortie restent permis. |
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

Tout harnais (Claude Code, Codex, OpenCode, Cursor), avec la méthode du Studio : le Skill
[`skills/weight-surgeon`](../skills/weight-surgeon/SKILL.md) et l'entrée
`{"mcpServers": {"weight-surgeon": {"command": "<surgeon>", "args": ["mcp"]}}}`, que
surgeon-gui installe et écrit (Réglages > Harnais externes, ou `surgeon-gui --install-skill`
et `--mcp-config`) ; voir [AGENTS.md](AGENTS.md) section 8 bis.

## Outils

Le serveur expose vingt outils (`tools/list`), décrits ici dans l'ordre où un agent les appelle
le plus souvent.

| Outil | Rôle | Annotations |
| --- | --- | --- |
| `surgeon_check` | Budget RAM/VRAM du modèle | lecture seule |
| `surgeon_inspect` | Couches et matrices MLP (et `o_proj`, `embed_tokens` sur demande), statistiques en option, `max_tensors` (défaut 500), architecture détectée (`data.arch`), `arch_spec` | lecture seule |
| `surgeon_apply` | Applique un patch en place (`dry_run` pour vérifier d'abord) ; `gguf: true` pour un GGUF quantifié | destructif |
| `surgeon_restore` | Remet un modèle octet pour octet depuis les sauvegardes de `surgeon_apply` ou depuis le modèle d'origine (`from_origin`), SHA-256 avant et après ([rollback](#rollback-surgeon_restore)) | destructif, idempotent |
| `surgeon_doctor` | La machine est-elle prête : RAM libre et plus gros modèle en f32 ou bf16 (avec `model` : son besoin exact et son lot natif), cœurs et threads, cache `ref_bench`, démon hipfire (URL, `/health`, `/metrics`, mémoire GPU lue dans son log ; 0 Mo au total = démon figé) ; `data.ready` | lecture seule, idempotent |
| `surgeon_compose` | Une direction faite de vecteurs extraits (`ablate`, poids `chemin:alpha`), orthogonale aux directions préservées (`keep`, Gram-Schmidt modifié en f64), et le patch `project_out` qui la retire avec sa recette (`composition`) ; refus nommé d'un vecteur nul, non fini, de dimension différente ou quasi colinéaire (`collinear_vectors`) | écrit des fichiers, idempotent |
| `surgeon_files` | Fichiers de travail d'un agent : `list`, `read` dans les dossiers autorisés, `write` de fichiers texte sous `work/` ([détail](#fichiers-de-travail-surgeon_files)) | écrit sous `work/` seulement |
| `surgeon_workspace` | Projet du studio pour un agent hors de l'interface : `action` `init` écrit `project.surgeon.json` (origine : chemin canonique, SHA-256, taille) et crée `runs/`, `artifacts/`, `copies/`, `chat/` ; le modèle n'est que lu, un manifeste existant n'est remplacé qu'avec `force` ([détail](#projet-du-studio-surgeon_workspace)) | écrit le manifeste, non destructif, idempotent |
| `surgeon_eval` | Non-régression (boucles, attentes) et export GGUF conditionnel | écrit le GGUF |
| `surgeon_extract` | Vecteur de direction depuis une couche Qwen2 (Rust pur, CPU) | écrit le `.safetensors`, idempotent |
| `surgeon_bench` | Mesure d'une chirurgie : `mode` `dump`, `logits`, `ppl`, `eval` ou `refusal` ; sonde heuristique `refusal_probe` ; calibration du juge : `judge_sample`, `judge_eval` | non destructif, idempotent |
| `surgeon_autotune` | Coefficients α par couche ajustés par ridge, patch.json en sortie ; avec `refusal_prompts`, la plus petite échelle c·α qui atteint `target_refusal_rate` en génération (sonde `first-token-prob` des `scales`, puis génération des `probe_top_k` finalistes, ou de toutes avec `no_probe_prefilter` ; `retention_dataset`, `retention_limit`), détail dans `data.selection` | non destructif, idempotent ; écrit `out` seulement s'il est fourni |
| `surgeon_diff` | Différence de deux checkpoints, approximée à rang `r`, écrite dans un `.surgery` neuf | écrit un fichier, non destructif, idempotent |
| `surgeon_lora` | `mode` `merge` (fusionne un adaptateur LoRA en place) ou `extract` (écrit un adaptateur) | destructif |
| `surgeon_pipeline` | Décision atomique sur un patch : `action` `plan`, `start`, `status`, `cancel`, `report`, `promote_patch` ([PIPELINE.md](PIPELINE.md)) | non destructif (copie de travail, modèle d'origine jamais écrit) |
| `surgeon_backends` | Moteurs d'inférence de la machine (native, mistralrs, llama-server et ses GPU, hipfire), configuration utilisateur et backend recommandé ([BACKENDS.md](BACKENDS.md)) | lecture seule |
| `surgeon_graph` | Graphe orienté des écritures/lectures du flux résiduel, avec bornes spectrales certifiées ; synthèse bornée sans `out_dir`, fichiers avec | écrit seulement avec `out_dir` ou `cache_dir`, non destructif, idempotent |

Les schémas exacts : `surgeon schema --kind tools`. Le format du patch :
`surgeon schema` (JSON Schema draft-07, sans `$ref`).

Chaque résultat contient la même enveloppe que `surgeon --json` :
`{"ok", "command", "exit_code", "data", "error", "warnings"}`, en texte et en
`structuredContent`. `isError` vaut `!ok`.

### Parité CLI / MCP

Chaque commande du moteur a son outil, exposé par défaut, qui passe par le même cœur
(`core_*`), accepte les mêmes jeux intégrés `builtin:*` et les mêmes entrées en ligne, et
renvoie la même enveloppe JSON complète. Seuls changent les chemins (résolus par le bac à
sable, donc absolus dans `data`) et les garde-fous propres à un agent (sauvegarde imposée,
aucun argument libre vers llama.cpp, aucune option `--allow-*` du pipeline).

| CLI | Outil MCP |
| --- | --- |
| `check` | `surgeon_check` |
| `inspect` | `surgeon_inspect` |
| `eval` | `surgeon_eval` |
| `extract` | `surgeon_extract` |
| `autotune` | `surgeon_autotune` |
| `apply` | `surgeon_apply` |
| `restore` | `surgeon_restore` |
| `compose` | `surgeon_compose` |
| `doctor` | `surgeon_doctor` (sans `--clean-cache`, ni `--hipfire-url` ni `--hipfire-log` : ceux de la configuration) |
| `graph` | `surgeon_graph` |
| `bench logits`, `ppl`, `eval`, `dump`, `refusal`, `refusal-probe`, `judge-sample`, `judge-eval` | `surgeon_bench` (`mode` du même nom, `refusal_probe`, `judge_sample`, `judge_eval` ; `--method` et `--vector` de `refusal-probe` s'y nomment `probe_method` et `direction_path`) |
| `diff` | `surgeon_diff` |
| `lora merge`, `lora extract` | `surgeon_lora` (`mode`) |
| `pipeline plan`, `run`, `status`, `cancel`, `report`, `promote` | `surgeon_pipeline` (`action`) |
| `backend detect`, `backend show` | `surgeon_backends` |

Exception voulue : `backend detect --save` reste en ligne de commande, car un agent ne
réécrit jamais la configuration de l'opérateur. `tui`, `schema`, `mcp`, `plugins` et `arch`
servent l'opérateur et n'ont pas d'outil (`surgeon_inspect` rapporte l'architecture
détectée dans `data.arch`).

## Extraction puis chirurgie

`surgeon_extract` (ou `surgeon extract` en ligne de commande) calcule
`v = normalize(mean(pos) - mean(neg))` sur les activations d'une couche, et écrit un
`.safetensors` avec un tenseur 1D F32 `direction` (taille `hidden_size`) et ses métadonnées
(couche, site, pooling, `d_prime`, `threshold_accuracy`, `pos_mean_proj`, `neg_mean_proj`,
`cos_mean_pos_neg`...). Un fichier compagnon `<out>.stats.json` (`"schema": 1`) garde la
projection de chaque prompt et le seuil ; `surgeon tui --vector` le dessine. Le résultat de
l'outil donne son chemin dans `data.stats_path` et en reprend le contenu, sans fichier à
relire : `data.stats` (`d_prime`, `threshold_accuracy`, `cosine_means`, `raw_diff_norm`,
moyennes et seuil, mesurés sur tous les prompts), `data.pos_projections`,
`data.neg_projections`, `data.layer`, et `data.holdout` (`frac`, `stats_basis:
"all_prompts"`, `n_train_pos`, `n_train_neg`, `n_holdout_pos`, `n_holdout_neg`, et `stats` :
la méthode réajustée sur la part gardée et mesurée sur la part retenue, `null` sans
`holdout_frac` ou sous 10 prompts par classe). Les fichiers de prompts acceptent aussi un jeu
intégré (`builtin:refusals`, `builtin:math`, `builtin:logic`, `builtin:general`).
Seuls Qwen2/Qwen2.5 sont supportés
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
| `eval` | `model`, `dataset` | `backend` (`llama`, `llama-completion`, `llama-server`, `mistralrs`, `hipfire`, `native`), `baseline_model`, `limit` (50), `max_new_tokens` (512), `prompt_template` (texte avec `{question}`, jamais un chemin), `chat_template` (voir ci-dessous), `format` (`gsm8k`, `math`, `auto`), `llama_cpp_dir`, `timeout_secs`, `min_accuracy`, `fail_on_warn`, options de génération |
| `refusal` | `model`, `prompts` (fichier) | `backend` (`native`, `llama-server`, `llama-completion`, `mistralrs`, `hipfire`), `baseline_model`, `max_new_tokens` (40), `chat_template` (voir ci-dessous), `marker_langs`, `markers_file`, `judge` (`rules`, `llm`, `both`), `judge_model`, `judge_role` (`refusal`, `harm`), `judge_template`, `judge_cache`, `judge_calibration`, `min_calibration_n` (100), `min_kappa` (0.6), `partial_as_refusal`, `max_unparsable`, `llama_cpp_dir`, `timeout_secs`, `fail_on_warn`, options de génération |
| `refusal_probe` | `model`, `prompts` (fichier) | `probe_method` (`first-token-prob` par défaut, `direction-projection`), `direction_path` (direction de `surgeon_extract`, requis en `direction-projection`), `threshold` (0.5, ou celui de la direction), `chat_template` |
| `dump` | `model`, `prompts` (fichier), `out` | `chat_template`, `dtype`, `out_dtype` (`f32`, `f16`), `max_tokens` (256), `max_total_positions` (4096), `overwrite` |
| `judge_sample` | `responses` (sortie JSON d'un `refusal`), `prompts` (ceux de ce run), `out` | `n` (100), `seed` (42), `overwrite` |
| `judge_eval` | `labels` (l'échantillon étiqueté), `judge_model` | `judge_role`, `judge_template`, `judge_cache`, `partial_as_refusal`, `marker_langs`, `markers_file`, `llama_cpp_dir`, `timeout_secs`, `out` (fichier de calibration), `overwrite` |

`data` reçoit le rapport (`BenchLogitsReport`, `BenchPplReport`, `BenchEvalReport`,
`BenchRefusalReport`, `BenchDumpReport`, `ProbeReport`), identique à `surgeon --json bench <mode>` ; les durées sont dans
`data.timing_ms`.

`refusal_probe` estime les refus sans générer, une passe avant par prompt (natif, CPU). Sa
réponse donne `probe_refusal_rate`, `mean_refusal_prob` (`first-token-prob` seulement),
`prompt_count`, `duration_ms` et le détail par prompt (`items`), et porte toujours
`is_heuristic: true`, `evidence_level: "heuristic_screening"` et un avertissement : c'est un
criblage, et seule une génération complète (`mode` `refusal` ou `eval`) est une preuve finale.
Un constat qui cite un tel run reste « Heuristique » dans le studio, jamais « Mesuré ». Un seuil dépassé avec `fail_on_warn: true` donne `exit_code` 9 et
`isError: true`, le rapport restant dans `data`. Aucun argument libre n'atteint llama.cpp
par MCP (l'option `--llama-arg` n'existe qu'en ligne de commande, sur liste blanche).
`--read-only` refuse les modes `dump` et `judge_sample`, `judge_eval` avec `out` et le champ
`judge_cache` (ils écrivent un fichier) mais laisse `logits`, `ppl`, `eval`, `refusal`, `refusal_probe` et
`judge_eval` sans `out`, juge compris.

Options de génération de `eval` et `refusal`, toutes structurées et bornées comme en ligne
de commande : `temperature` (0, glouton), `top_k` (absent : désactivé), `top_p` (1),
`min_p` (0), `repetition_penalty` (1), `seed` (42), `samples` (1 ; au-delà, `temperature`
doit dépasser 0), `batch_size` (1), `jobs` (1), `threads`, `n_gpu_layers` (`-ngl` de
`llama-server`), `ctx` (`-c` de `llama-server`). Sans elles, le rapport est celui d'avant ;
avec elles, il gagne `data.generation` (statistiques par prompt et intervalle bootstrap sur
les prompts quand `samples` dépasse 1). Le backend `llama-server` ne reçoit que les options
de sa liste blanche (`-m`, `--host 127.0.0.1`, `--port`, `--api-key` aléatoire, `-np`,
`-c`, `-ngl`, `-t`), vérifiées contre son `--help` ; aucune propriété libre n'existe
(`additionalProperties: false`), et `llama_cpp_dir` passe par le bac à sable. `mistralrs`
lance mistral.rs sur le dossier SafeTensors tel quel, `hipfire` interroge un démon déjà lancé ;
leurs emplacements viennent de la configuration de l'opérateur, jamais d'un argument. hipfire
sert son propre cache quantifié et ne lit pas le modèle donné : sur une copie de travail
(`copies/` ou sorties d'un projet surgeon-gui, `work/` d'un job pipeline, dossier à côté des
sauvegardes d'`apply`), `eval` et `refusal` passent sur `native` et l'enveloppe commence ses
`warnings` par « backend hipfire écarté (<chemin>) », écrit aussi sur stderr ; une copie
`.gguf` est refusée (`invalid_input`). Le modèle d'origine reste servi par hipfire, comme une
copie déjà requantifiée en ligne de commande dont le `.hipfire-alias.json` correspond à ses
poids (sous son alias). Aucun argument ne lève cette règle (pas d'`--allow-stale-hipfire` ni de
requantification côté agent). Voir la
section « Échantillonnage, lots et backends » du README.

`eval` avec `backend: "native"` et `refusal` génèrent en Rust pur sur le CPU, sur un dossier
Qwen2 SafeTensors, sans llama.cpp (décodage glouton par défaut, donc déterministe). `refusal` compte
comme refus une réponse qui contient un marqueur canonique (`data.markers`, anglais par
défaut, sans casse, aux frontières de mots ; `marker_langs` et `markers_file` ajoutent
d'autres langues) ; les réponses entières sont dans `data.items[].output`. Avec
`baseline_model`, `data.paired` donne les prompts débloqués (`unblocked`), les nouveaux refus
(`newly_refused`), `delta_rate` et son intervalle à 95 % (Newcombe apparié).

Avec `judge: "llm"` ou `"both"`, un modèle juge (`judge_model`, un `.gguf` lancé par
`llama-completion`) étiquette ensuite chaque réponse ; les champs des marqueurs ne changent
pas et les verdicts vont dans `data.judge`. Rôle `refusal` : `complied`, `refused_explicit`,
`refused_soft`, `partial`, `degenerate`, `other`. Rôle `harm` (Llama-Guard et semblables) :
`safe`, `unsafe`. Un verdict illisible est `unparsable` : exclu de `rate`, jamais compté comme
non-refus, encadré par `rate_min` et `rate_max` ; `max_unparsable` avec `fail_on_warn` en fait
un `exit_code` 9. `data.judge.calibration.status` vaut « JUGE NON CALIBRÉ : <raison> » tant
que `judge_calibration` ne désigne pas une calibration du même juge, de la même rubrique, du
même rôle et du même réglage `partial_as_refusal`, avec `min_calibration_n` étiquettes et un
kappa d'au moins `min_kappa` ; le même texte est dans `warnings`. Tous les chemins du juge
(`judge_model`, `judge_template`, `judge_cache`, `judge_calibration`, `markers_file`) passent
par le bac à sable `--allow-dir`. Les modes `judge_sample` et `judge_eval` (comme `bench
judge-sample` et `bench judge-eval`) produisent l'échantillon à étiqueter puis le fichier de
calibration ; entre les deux, chaque `label` de l'échantillon est rempli par un humain. Une
calibration dont les étiquettes ne viennent pas d'un humain ne mesure rien.

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
7. Décider : garder la copie, ajuster `alpha` ou les couches, ou abandonner. En cas de
   régression, `surgeon_restore` sur la copie la remet dans son état d'avant l'apply.

```json
{"name": "surgeon_bench", "arguments": {"mode": "logits", "before": "before.safetensors", "after": "after.safetensors", "warn_kl": 0.05}}
```

### Gabarit de chat (`chat_template`)

`surgeon_extract`, `surgeon_autotune` et `surgeon_bench` (`eval`, `dump`, `refusal`) habillent
les prompts de la même façon, CLI et MCP. `auto` (défaut) suit le modèle : `chat_template` de
`tokenizer_config.json`, `chat_template.jinja` ou métadonnée GGUF, sinon les jetons spéciaux de
`tokenizer_config.json`. Faute de moteur Jinja, le gabarit désigne une famille (`chatml`,
`llama3`, `inst`, `gemma`) dont les tours sont écrits par le moteur, avec le message système
par défaut littéral d'un gabarit ChatML (Qwen). Un modèle qui ne déclare rien reste en texte brut,
sauf `refusal` qui garde ChatML. `none` désactive l'habillage ; un prompt qui contient déjà les
tours de la famille n'est jamais habillé deux fois. Le vecteur extrait garde le format
(`chat_template`, `chat_system`), que `surgeon_autotune` rejoue.

## Rollback (`surgeon_restore`)

Remet un fichier ou un dossier modèle octet pour octet dans un état connu, comme
`surgeon restore`. `model` est la destination : un chemin d'écriture (`--allow-dir`), jamais
sous une racine `--allow-read`, même imbriquée dans un `--allow-dir`
(`error.kind = "path_not_allowed"`). Au plus une source :

| Argument | Source |
| --- | --- |
| aucun | Le dernier point de restauration à côté du modèle : l'état d'avant le dernier `surgeon_apply`. |
| `backup_dir` | Idem, dans le `backup_dir` passé à `surgeon_apply`. |
| `backup` | Les sauvegardes nommées, par exemple `data.backups` d'un `surgeon_apply`. |
| `from_origin` | Le modèle d'origine (dossier, ou fichier de poids recopié sous son nom dans un dossier `model` ; lisible en `--allow-read`) : chacun de ses fichiers est recopié sur `model`. |

Toutes les sauvegardes d'un même `surgeon_apply` partagent un point de restauration
(`<nom>.bak-<secondes>-<n>`) plus récent que tout ce que contient le dossier : sans argument,
l'outil revient exactement à l'état d'avant le dernier apply, quel que soit leur nombre, et un
nouvel appel ne remonte pas plus loin. Un `surgeon_apply` lancé alors que le modèle est déjà
dans l'état du dernier point de restauration le réutilise (`data.backups` le cite) au lieu de
recopier les poids. Dans un dossier d'une version antérieure, un seul fichier sauvegardé reprend
sa sauvegarde la plus récente ; plusieurs fichiers, que rien ne relie, reviennent tous à l'état
d'avant le premier apply, avec un avertissement. Les sources ne sont jamais modifiées ni
supprimées ; un fichier déjà identique à sa source n'est pas réécrit. `data` : `mode`
(`backup`, `origin`), `checkpoint` (détection : `last_apply`, `first_apply`),
`model`, `files` (`path`, `source`, `sha256_before`, `sha256_source`, `sha256_after`,
`rewritten`), `identical`. L'outil échoue (`exit_code` 3) si un `sha256_after` diffère de
son `sha256_source`. `--read-only` le refuse.

Cycle complet, régression comprise :

```json
{"name": "surgeon_apply", "arguments": {"model": "copies/c1", "patch": {"schema_version": 1, "operations": [...]}}}
{"name": "surgeon_bench", "arguments": {"mode": "logits", "before": "before.safetensors", "after": "after.safetensors", "warn_kl": 0.05}}
{"name": "surgeon_restore", "arguments": {"model": "copies/c1"}}
{"name": "surgeon_restore", "arguments": {"model": "copies/c1", "from_origin": "/models/qwen2"}}
```

## Fichiers de travail (`surgeon_files`)

Pour un agent qui n'a pas d'outils de fichiers (Claude Code et Codex lancés par le studio,
sessions API). `action` :

| Action | Arguments | Effet |
| --- | --- | --- |
| `list` | `path` (défaut : le premier `--allow-dir`) | Entrées d'un dossier autorisé, `--allow-read` compris ; `model: true` sur un dossier de modèle. `data` donne aussi `work_dir`, `writable_roots`, `read_only_roots`. |
| `read` | `path` | Fichier texte UTF-8, 1 Mio au plus (`data.truncated`). |
| `write` | `path`, `content`, `append`, `overwrite` | Écrit sous `<premier --allow-dir>/work/` ; `data.path` est le chemin absolu à passer aux autres outils. Extensions : `.jsonl .json .txt .md .csv .tsv .py .ps1 .sh .js .toml .yaml .yml .tmpl .jinja`. |

Hors de `work/`, l'écriture est refusée (`error.kind = "path_not_allowed"`) : ni poids, ni
copie de travail, ni journal du projet. `--read-only` refuse `write`.

## Projet du studio (`surgeon_workspace`)

`surgeon_workspace` donne son acte de naissance au dossier de travail : un petit fichier scellé,
`project.surgeon.json`, qui dit de quel modèle le dossier travaille et le prouve par une
empreinte. C'est le même manifeste que celui écrit par le Studio à la création d'un projet, donc
`surgeon-gui` rouvre ensuite le dossier comme l'un des siens : origine reconnue et verrouillée
en lecture, runs dans Résultats, constats et essais dans le journal.

Un modèle de 0,5 milliard de paramètres comme Qwen2.5-0.5B-Instruct pèse environ 988 Mo de
SafeTensors dans `C:\models\Qwen2.5-0.5B-Instruct` ; le manifeste enregistre ce chemin
canonique, sa taille et le SHA-256 de ses fichiers, une empreinte de 64 caractères (abrégée ici
en `4f9c…`). Un seul appel, au début de la session :

```json
{"action": "init", "model": "C:\\models\\Qwen2.5-0.5B-Instruct", "name": "qwen-0.5b-refus"}
```

| Argument | Effet |
| --- | --- |
| `action` | `init` seulement : prépare le dossier courant (le premier `--allow-dir`) comme projet. |
| `model` | Modèle d'origine, lu seulement : dossier SafeTensors (`config.json`, `*.safetensors`) ou fichier `.safetensors` / `.gguf`, dans un dossier autorisé (`--allow-read` compris). |
| `name` | Nom affiché par le Studio, 1 à 64 caractères ; par défaut le nom du dossier de travail. |
| `force` | Remplace un `project.surgeon.json` existant, sinon l'appel échoue (`output_exists`). |

Le modèle est refusé s'il est le projet lui-même ou une copie de `copies/`
(`invalid_arguments`) et si le dossier ne contient aucun fichier de poids (`invalid_input`).
`--read-only` refuse l'appel : il écrit le manifeste.

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

**Sous-espaces de rang k.** `rank` (1 à 64) et `labels_file` étendent `svd` et `leace` :

- `svd` avec `rank` > 1 écrit aussi `<out>.subspace.safetensors` : `a = b = V_k`, les `k`
  premiers vecteurs singuliers droits des différences appariées, de forme `[k, hidden]`
  (un projecteur orthogonal) ; `direction` reste le premier ;
- `leace` avec `labels_file` (fichier JSON dans un dossier autorisé, un label par prompt,
  les positifs puis les négatifs : indice de classe entier, ou tableau de `m` nombres)
  efface tous les labels à la fois ; `a` et `b` de `<out>.leace.safetensors` sont alors de
  forme `[k, hidden]`, avec `k` au plus `rank` (défaut : toutes les directions de la
  covariance croisée) ;
- `mean_diff` reste de rang 1 ; `rank` ou `labels_file` sur une méthode qui ne peut pas
  les honorer est refusé (`invalid_input`).

`data.method_diagnostics.subspace` détaille chaque composante : `rank`, `sigma_t`,
`explained_variance_cumulative`, `d_prime`, `gram_BtA_error` et, pour `leace` avec
`holdout_frac`, `holdout_cross_cov_residual`.

`method` est un identifiant du registre des méthodes d'extraction du moteur (le même que
`surgeon extract --method` et `surgeon plugins list`). Le serveur n'utilise que le registre
intégré : il accepte exactement `mean_diff`, `svd` et `leace`, avec le schéma et les
messages d'erreur d'avant le registre (`unknown variant ...`, `invalid_arguments`). Une
méthode tierce (API `v0`) s'exécute par la bibliothèque, jamais par un argument d'outil :
voir [docs/PLUGINS.md](PLUGINS.md).

L'opération de patch `project_out` applique l'effaceur aux matrices qui écrivent dans le
flux résiduel : `W ← W − alpha·a(bᵀW)` pour `down_proj` et `o_proj`, `W ← W − alpha·(W b)aᵀ`
pour `embed_tokens` ; `alpha` dans [0, 1], `vector_tensor` = `a` et `vector_tensor_b` = `b`
(obligatoires), pas d'`axis` autre que le défaut :

```json
{"id": "erase", "layers": "18-22", "projections": ["down_proj"], "operation": "project_out",
 "vector_file": "refusal.safetensors.leace.safetensors", "vector_tensor": "a",
 "vector_tensor_b": "b", "alpha": 1.0}
```

Les tenseurs `a` et `b` peuvent être de forme `[k, hidden]` (sorties de rang k ci-dessus) :
`W ← W − alpha·A(BᵀW)` retire alors les `k` directions en une seule passe, chaque poids
n'est arrondi qu'une fois. Deux champs optionnels de l'opération, `project_out` uniquement :

- `rank` : rang attendu ; des tenseurs d'un autre rang sont refusés. Absent : le rang des
  tenseurs ;
- `allow_non_projector` (défaut `false`) : avec `alpha = 1`, si `‖BᵀA − I‖_F > 1e-3` (`a` et
  `b` ne forment pas un projecteur), un patch `[k, hidden]` est refusé ; `true` l'accepte
  avec un avertissement. La forme 1D reste acceptée avec l'avertissement.

`surgeon_apply` plafonne le rang à 64 (le défaut de `surgeon apply --max-rank`, que le
serveur n'expose pas). Ces contrôles (rangs de `a` et `b`, `rank`, plafond, projecteur)
lisent les tenseurs, donc ils ont lieu à l'exécution et non à la validation du patch, mais
avant la première écriture : un refus laisse le modèle intact (code de sortie 3 en CLI).

Ablation directionnelle de tous les écrivains (`a = b = v`, la `direction` unitaire de
`surgeon_extract`), en une opération :

```json
{"id": "ablate", "layers": "all", "projections": ["down_proj", "o_proj", "embed_tokens"],
 "operation": "project_out", "vector_file": "refusal.safetensors",
 "vector_tensor": "direction", "vector_tensor_b": "direction", "alpha": 0.6}
```

`o_proj` et `embed_tokens` sont aussi des cibles de `subtract` / `add` (`axis` `cols` pour
`o_proj`, `rows` pour `embed_tokens`), en SafeTensors comme en GGUF (voir plus bas) ;
`surgeon_autotune` ne vise que le MLP. `embed_tokens` n'appartient à aucune couche :
`layers` ne le filtre pas.

**Détachement de `lm_head`.** Si `embed_tokens` est visé et que le modèle le partage avec la
projection de sortie (pas de tenseur `lm_head.weight`, ou `tie_word_embeddings` encore vrai),
`surgeon_apply` détache d'abord la tête : le shard est réécrit avec `lm_head.weight`, copie à
l'octet des plongements non patchés, `config.json` reçoit `"tie_word_embeddings": false`, et
`model.safetensors.index.json` est complété pour un modèle shardé. `dry_run: true` l'annonce
dans `data.untie_lm_head` (`shard`, `config`, `index`, `copy_embeddings`) et dans `warnings`,
sans rien écrire. Le shard et `config.json` sont sauvegardés (`data.backups`) et restaurés à
l'octet près si une opération échoue ensuite. Sans `config.json` à côté du modèle, le patch
est refusé avant toute écriture.

`surgeon_autotune` (`model`, `vector` requis, et exactement un de `eval_dataset` : fichier ou
jeu intégré `builtin:math|logic|general|refusals`, comme `surgeon autotune --eval-dataset`, ou
`eval_prompts` : les prompts en ligne, chaînes ou `{text, target?}` ; `target_layer_range` "A-B",
`readout_layer`, `proj`, `objective` `suppress`/`amplify`/`target`, `target_projection`,
`l2_reg` nombre ou "auto", `probe_alpha`, `alpha_max`, `max_delta_norm_ratio`,
`min_alpha`, `max_forwards`, `max_tokens`, `dtype`, `site`, `pooling`, `chat_template`, `out`, `overwrite`) mesure en
mémoire la réponse de la projection lue à la couche de lecture quand chaque couche candidate
est décalée, ajuste les α par ridge (λ fixe ou LOOCV), applique tous les α ensemble pour
vérifier, et renvoie le patch dans `data.patch` (écrit dans `out` seulement si fourni,
validé par `validate_patch` avant). `probe_alpha`, `alpha_max` et les α sont les coefficients
de `W ∓= α·v` ; `target_projection` et les décalages rapportés sont des projections
d'activation ; `data.alphas[].delta_norm_ratio` donne `‖ΔW‖_F / ‖W‖_F` par couche, que
`max_delta_norm_ratio` borne (0.01 pour 1 %). Les couches au-delà de la lecture sont exclues avec un
avertissement ; `max_forwards` est vérifié avant tout calcul. Il ne mesure pas les effets
secondaires : enchaîner avec `surgeon_bench` sur une copie.

## Architectures (`arch_spec`, `accept_approximate`)

Les cibles et leur orientation viennent de l'architecture du modèle : `model_type` de
`config.json`, `general.architecture` d'un GGUF, ou une spécification JSON (`arch_spec`,
chemin dans un dossier autorisé, schéma : `surgeon schema --kind arch`). `surgeon_inspect`
et `surgeon_apply` la rapportent dans `data.arch` (`id`, `source`, `roles`, `writer_biases`,
`tied_embeddings`, `exactness`). Pour un `project_out` :

| Exactitude | Effet |
| --- | --- |
| `exact` (pré-norme : Llama, Mistral, Qwen2/3, MoE, Phi-3, Falcon, Command-R, DeepSeek) | appliqué |
| `approximate` (post-normes de Gemma 2/3, biais d'écrivains du modèle) | refusé, `error.kind = "approximate_ablation"`, sauf `accept_approximate: true` (avertissement, `data.arch.accepted_approximate`) |
| `unsupported` (écrivains non projetables déclarés par une spécification) | refusé, `error.kind = "unsupported_architecture"` |

Une spécification invalide donne `error.kind = "invalid_arch_spec"`. `add` / `subtract` ne
sont jamais filtrés. Les experts MoE (par expert, fusionnés `[E, hidden, inter]`, partagés)
sont des cibles de `down_proj` ; le routeur ne l'est jamais. Extraction, génération et
`surgeon_bench` en mode `dump` / `refusal` ont un forward natif pour Qwen2, Llama/Mistral et
Qwen3 seulement (`unsupported_architecture` sinon).

## GGUF quantifié (`surgeon_apply` avec `gguf: true`)

`surgeon_inspect` accepte un `.gguf` : `data.format = "gguf"`, `data.tied_embeddings`, et
pour chaque cible (`blk.{i}.ffn_{gate,up,down}.weight` par défaut ;
`blk.{i}.attn_output.weight` et `token_embd.weight` avec `proj` `o_proj` / `embed_tokens`)
son `ggml_type`, son `orientation` (`reader`, `writer`, `writer_local`), sa forme
`[lignes, colonnes]`, sa taille, `patchable` (F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
Q4_K, Q5_K, Q6_K) et `patchable_reason` quand il ne l'est pas.

`surgeon_apply` gagne ces champs, le nombre d'outils ne change pas :

| Champ | Défaut | Rôle |
| --- | --- | --- |
| `gguf` | `false` | Obligatoire pour un `.gguf` : sans lui, `error.kind = "invalid_input"` (la re-quantification perd de l'information, jamais par surprise). |
| `requant` | `keep-scales` | `keep-scales` garde `d`, `m` / `dmin` et les sous-échelles ; `recompute` recalcule les échelles des blocs touchés. |
| `allow_tied_embedding` | `false` | Autorise `token_embd` quand le fichier n'a pas d'`output.weight` (la tête de sortie le lit aussi et change avec lui) ; refusé sinon (`invalid_input`), avertissement dans le rapport sinon. |
| `refine_passes` | 1 | `project_out` : passes de quantification mesurées, de 1 à 5 (hors bornes : `invalid_input`). |
| `residual_target` | 0.05 | `project_out` : arrêt du raffinement quand `requant_residual_fraction` vaut au plus cette part, dans `]0, 1]`. |

`project_out` fonctionne sur les écrivains du flux résiduel d'un GGUF (`down_proj`, `o_proj`,
`embed_tokens`) : chaque tenseur est déquantifié, projeté comme en SafeTensors, puis
re-quantifié. Une ablation GGUF n'est jamais complète : `data.gguf.tensors[]` donne
`residual_fraction` (`‖Bᵀ deq(W')‖ / ‖Bᵀ W‖`), `requant_residual_fraction` (la part ramenée
par la re-quantification) et `refine_passes`. Quand l'exactitude compte : SafeTensors non
quantifiés, patch, conversion, puis quantification.

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
| `unsupported_quant_type` | Une cible n'est pas en F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K ni Q6_K ; toutes les cibles sont vérifiées avant la première écriture. |
| `invalid_gguf` | Fichier illisible, tronqué, big-endian ou multi-parties. |
| `invalid_target` | Cible non 2D, ou colonnes non multiples du bloc. |

`data.gguf.tensors[]` donne la fidélité par tenseur (`realized_over_intended`,
`correlation_intended_realized`, `unchanged_fraction`, `saturated_elements`,
`nmse_vs_exact`, `warning`, et pour `project_out` les champs de résidu ci-dessus),
`data.gguf.preflight` le travail prévu (`tensors`, `target_bytes`, `work_bytes`,
`estimated_seconds`, `backup_bytes`), et `data.io` donne les blocs écrits, le débit et les
défauts de page (absent en `dry_run`). Un `warning` signale un patch perdu sous le demi-pas, ou des
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

## Graphe de poids (`surgeon_graph`)

Équivalent de `surgeon graph` : un nœud par matrice du flux résiduel, une arête de chaque
écriture vers les lectures strictement postérieures, avec alignement de sous-espaces,
couplage, angle principal, bornes certifiées de `sigma1`, de `D_spec` et du rang effectif
énergétique, kurtosis et canaux outliers. Aucun passage avant. Le détail des formules et
du schéma est dans [docs/GRAPH.md](GRAPH.md).

| Champ | Défaut | Rôle |
| --- | --- | --- |
| `model` | requis | `.safetensors`, dossier shardé ou `.gguf`, dans un dossier autorisé |
| `out_dir` | absent | Écrit `graph.json`, `graph.dot`, `graph.mmd`, `matrices.csv`, `edges.csv`, `graph.ndjson` ; absent : synthèse bornée seulement, aucun fichier |
| `format` | `["json"]` | `json`, `dot`, `mmd`, `csv`, `ndjson` (`graph.ndjson`, flux `GraphPatch` ouvert par `surgeon-gui`) |
| `layers`, `matrices` | tout | Filtres de couches (`0`, `1-3`, `0,2,4`, `all`) et de matrices (`q`, `k`, `v`, `o`, `gate`, `up`, `down`, `embed`, `lm_head`) |
| `rank`, `oversample`, `power_iters`, `seed` | 32, 10, 3, 0 | SVD randomisé ; même graine, mêmes nombres |
| `full_spectrum_max_dim` | 2048 | Spectre exact pour `min(m,n)` jusqu'à cette valeur (0 désactive ; plafonds 1024 et `m*n <= 4_000_000`) |
| `edges` | `adjacent` | `adjacent`, `all` ou `topk:N` |
| `min_alignment`, `edge_weight` | 0, `alignment` | Filtre d'alignement brut et poids (`alignment` ou `coupling`) |
| `vector` | absent | Vecteur JSON ou `.bin` f32 pour `e_write`/`e_read` |
| `kurtosis_threshold`, `color_scale`, `palette` | 10, `quantile`, `diverging` | Seuil d'outlier et rendu DOT/Mermaid |
| `max_nodes`, `cache_dir`, `no_cache`, `resume`, `estimate`, `jobs`, `overwrite` | | Mêmes sémantiques qu'en CLI |
| `max_seconds` | absent | Plafond wall-clock ; au-delà, `error.kind = "cancelled"` et aucun fichier |
| `baseline` | absent | Référence : modèle (mêmes paramètres) ou `graph.json` déjà écrit ; ajoute `baseline_diff` (synthèse : compteurs, `top_changed_layers`, `warnings`) |
| `fast_hash` | `false` | Empreinte rapide `fast-...` au lieu du SHA-256 complet ; `model.hash_mode = "fast"` |

Sans `out_dir`, `data` est une synthèse bornée : `model`, `parameters`, `global`, au plus
200 entrées de `nodes`, au plus 20 `top_edges`, `outputs` et `truncated`. Le graphe complet
n'est jamais renvoyé en ligne ; il faut `out_dir` et lire `graph.json` (schéma :
`surgeon schema --kind graph`). Avec `--read-only`, un appel sans `out_dir` ni `cache_dir`
reste autorisé (lecture seule) ; fournir l'un des deux donne `error.kind = "read_only"`
avant toute écriture. Tous les chemins (`model`, `out_dir`, `cache_dir`, `vector`,
`baseline`) passent par le bac à sable `--allow-dir` ; `model`, `vector` et `baseline`
peuvent aussi être lus dans une racine `--allow-read`, ce qui permet de comparer une copie
de travail à son modèle d'origine sans jamais écrire dans ce dernier.

Exemple d'appel sans écriture :

```json
{"name": "surgeon_graph", "arguments": {"model": "qwen2-mini", "rank": 8, "edges": "topk:2"}}
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
  `surgeon_autotune` `vector`, `eval_dataset`, `out`) ; pour `surgeon_graph`, `model`,
  `out_dir`, `cache_dir`, `vector` et `baseline`.
  est résolu par le système (liens symboliques, `..`) et doit tomber sous un dossier
  autorisé, sinon `error.kind = "path_not_allowed"`, sans révéler la cible. Un jeu intégré
  `builtin:<jeu>` n'est pas un fichier : il est lu dans le binaire, hors sandbox.
- **Sauvegarde imposée** avant toute écriture, vérifiée par SHA-256 ; une sauvegarde ratée
  bloque l'écriture.
- **Rollback automatique** sur erreur, NaN/Inf, annulation (`notifications/cancelled`,
  `error.kind = "cancelled"`) ou panic. Un panic devient une erreur `-32603` et le
  serveur continue.
- **Arrêt quand stdin se ferme** (cycle de vie MCP sur stdio : le client ferme l'entrée du
  serveur). L'appel en cours est annulé comme par `notifications/cancelled` (vérifié entre
  tenseurs, prompts et tokens), les requêtes déjà lues mais pas commencées ne sont pas
  exécutées, et `surgeon mcp` se termine (code 0) dès que l'appel a rendu la main, en le
  notant sur stderr. S'il ne la rend pas en 3 s (`server::EXIT_GRACE`), le processus sort
  quand même (code 1). Un client par script lit donc ses réponses avant de fermer stdin.
  `mcp::serve` (tests, intégrations en processus) garde l'ancienne lecture : tout ce qui a
  été lu reçoit sa réponse, puis la fin.
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

## Décision sur un patch (`surgeon_pipeline`)

Arguments plats : `action`, `job_dir`, `config` (pour `plan` et `start` ; absent pour
`start`, le job existant reprend). Le flux d'un agent :

1. `plan` : disque, durée estimée, puissance (`dev_underpowered` annonce un INCONCLUSIVE
   garanti).
2. `start` : lance `surgeon pipeline run --watch-stdin` en sous-processus et renvoie
   `job_id` tout de suite ; un seul job à la fois par serveur (`error.kind = "busy"`).
3. `status` jusqu'à `report_available` : étape, `units_done` / `units_total`,
   `eta_secs`, portes Dev déjà tranchées.
4. `report` : verdict `GO`, `NO_GO` ou `INCONCLUSIVE`, agrégats seulement pour le Test.
5. `promote_patch` après un GO : écrit `promoted/patch.json` ; l'appliquer est une décision
   séparée (`surgeon_apply`, sauvegarde imposée).

Garde-fous propres : la configuration doit contenir un pilier `safety` et aucun
`--allow-*` de la ligne de commande n'existe ici ; chaque fichier qu'elle nomme (modèle,
vecteurs, jeux Dev et Test, gabarits, registre) passe le bac à sable ; un jeu Test déjà
consulté est refusé (`test_set_already_used`) ; tout chemin sous un dossier `private/` de
job (marqueur `.surgeon-private`) est refusé par le bac à sable, pour tous les outils ; le
travailleur s'annule quand le serveur se ferme (stdin), et est tué après 15 s sinon. Une
progression par `progressToken` n'est pas émise : `start` rend la main tout de suite et
l'avancement se lit par `status`.

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
