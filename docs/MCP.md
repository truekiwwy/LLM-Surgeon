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
| `--read-only` | Refuse `surgeon_apply`, `surgeon_extract`, `export_gguf`, `surgeon_bench` en mode `dump`, `surgeon_autotune` avec `out`, `surgeon_diff`, `surgeon_lora` (les deux modes), `surgeon_graph` avec `out_dir` ou `cache_dir` et les actions `start`, `cancel`, `promote_patch` de `surgeon_pipeline` (`error.kind = "read_only"`) ; `surgeon_autotune` sans `out` et `surgeon_graph` sans sortie restent permis. |
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
| `surgeon_inspect` | Couches et matrices MLP (et `o_proj`, `embed_tokens` sur demande), statistiques en option, `max_tensors` (défaut 500), architecture détectée (`data.arch`), `arch_spec` | lecture seule |
| `surgeon_apply` | Applique un patch en place (`dry_run` pour vérifier d'abord) ; `gguf: true` pour un GGUF quantifié | destructif |
| `surgeon_eval` | Non-régression (boucles, attentes) et export GGUF conditionnel | écrit le GGUF |
| `surgeon_extract` | Vecteur de direction depuis une couche Qwen2 (Rust pur, CPU) | écrit le `.safetensors`, idempotent |
| `surgeon_bench` | Mesure d'une chirurgie : `mode` `dump`, `logits`, `ppl`, `eval` ou `refusal` | non destructif, idempotent |
| `surgeon_autotune` | Coefficients α par couche ajustés par ridge, patch.json en sortie | non destructif, idempotent ; écrit `out` seulement s'il est fourni |
| `surgeon_diff` | Différence de deux checkpoints, approximée à rang `r`, écrite dans un `.surgery` neuf | écrit un fichier, non destructif, idempotent |
| `surgeon_lora` | `mode` `merge` (fusionne un adaptateur LoRA en place) ou `extract` (écrit un adaptateur) | destructif |
| `surgeon_pipeline` | Décision atomique sur un patch : `action` `plan`, `start`, `status`, `cancel`, `report`, `promote_patch` ([PIPELINE.md](PIPELINE.md)) | non destructif (copie de travail, modèle d'origine jamais écrit) |
| `surgeon_backends` | Moteurs d'inférence de la machine (native, llama-server et ses GPU, hipfire), configuration utilisateur et backend recommandé ([BACKENDS.md](BACKENDS.md)) | lecture seule |
| `surgeon_graph` | Graphe orienté des écritures/lectures du flux résiduel, avec bornes spectrales certifiées ; synthèse bornée sans `out_dir`, fichiers avec | écrit seulement avec `out_dir` ou `cache_dir`, non destructif, idempotent |

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
| `eval` | `model`, `dataset` | `backend` (`llama`, `llama-completion`, `llama-server`, `native`), `baseline_model`, `limit` (50), `max_new_tokens` (512), `prompt_template` (texte avec `{question}`, jamais un chemin), `format` (`gsm8k`, `math`, `auto`), `llama_cpp_dir`, `timeout_secs`, `min_accuracy`, `fail_on_warn`, options de génération |
| `refusal` | `model`, `prompts` (fichier) | `backend` (`native`, `llama-server`, `llama-completion`), `baseline_model`, `max_new_tokens` (256), `chat_template` (`chatml`, `none`), `marker_langs`, `markers_file`, `judge` (`rules`, `llm`, `both`), `judge_model`, `judge_role` (`refusal`, `harm`), `judge_template`, `judge_cache`, `judge_calibration`, `min_calibration_n` (100), `min_kappa` (0.6), `partial_as_refusal`, `max_unparsable`, `llama_cpp_dir`, `timeout_secs`, `fail_on_warn`, options de génération |
| `dump` | `model`, `prompts` (fichier), `out` | `dtype`, `out_dtype` (`f32`, `f16`), `max_tokens` (256), `max_total_positions` (4096), `overwrite` |

`data` reçoit le rapport (`BenchLogitsReport`, `BenchPplReport`, `BenchEvalReport`,
`BenchRefusalReport`, `BenchDumpReport`), identique à `surgeon --json bench <mode>` ; les durées sont dans
`data.timing_ms`. Un seuil dépassé avec `fail_on_warn: true` donne `exit_code` 9 et
`isError: true`, le rapport restant dans `data`. Aucun argument libre n'atteint llama.cpp
par MCP (l'option `--llama-arg` n'existe qu'en ligne de commande, sur liste blanche).
`--read-only` refuse le mode `dump` et le champ `judge_cache` (il écrit un fichier) mais
laisse `logits`, `ppl`, `eval` et `refusal`, juge compris.

Options de génération de `eval` et `refusal`, toutes structurées et bornées comme en ligne
de commande : `temperature` (0, glouton), `top_k` (absent : désactivé), `top_p` (1),
`min_p` (0), `repetition_penalty` (1), `seed` (42), `samples` (1 ; au-delà, `temperature`
doit dépasser 0), `batch_size` (1), `jobs` (1), `threads`, `n_gpu_layers` (`-ngl` de
`llama-server`), `ctx` (`-c` de `llama-server`). Sans elles, le rapport est celui d'avant ;
avec elles, il gagne `data.generation` (statistiques par prompt et intervalle bootstrap sur
les prompts quand `samples` dépasse 1). Le backend `llama-server` ne reçoit que les options
de sa liste blanche (`-m`, `--host 127.0.0.1`, `--port`, `--api-key` aléatoire, `-np`,
`-c`, `-ngl`, `-t`), vérifiées contre son `--help` ; aucune propriété libre n'existe
(`additionalProperties: false`), et `llama_cpp_dir` passe par le bac à sable. Voir la
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
par le bac à sable `--allow-dir`. Les sous-commandes `bench judge-sample` et
`bench judge-eval`, qui produisent l'échantillon à étiqueter et le fichier de calibration, ne
sont pas exposées par MCP : la calibration reste un geste humain, en ligne de commande.

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
`baseline`) passent par le bac à sable `--allow-dir`.

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
