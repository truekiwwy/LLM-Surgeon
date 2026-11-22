---
name: weight-surgeon
description: Méthode du Studio Weight Surgeon pour un agent branché sur le serveur MCP « surgeon mcp » (Claude Code, Codex, OpenCode, Cursor) qui modifie les poids d'un LLM local. D'abord le dossier projet avec surgeon_workspace (origine enregistrée et vérifiée), puis diagnostic machine avec surgeon_doctor, inspection et extraction de directions, criblage en entonnoir (surgeon_bench refusal_probe, surgeon_autotune), composition orthogonale avec surgeon_compose pour préserver une capacité, certification obligatoire par génération complète, et un dossier projet que surgeon-gui rouvre. À suivre dès qu'on retire un comportement (refus excessifs, répétitions) d'un modèle SafeTensors ou GGUF, qu'on mesure ou corrige des poids avec les outils surgeon_*, ou qu'on prépare un projet à ouvrir dans le Studio.
---

# Weight Surgeon : la méthode du Studio

Le serveur MCP `surgeon` est le moteur de surgeon-gui : mêmes outils, mêmes garde-fous,
même format de projet. Suis les étapes dans l'ordre. Si un outil échoue, rapporte
`error.kind` et `exit_code` au lieu d'improviser. Un seul run lourd à la fois (extraction,
bench, autotune) : deux en parallèle se partagent la RAM et faussent les temps.

Détail de chaque étape et règles de décision : [references/workflow.md](references/workflow.md).
Arguments prêts à copier pour chaque appel : [references/recipes.md](references/recipes.md).

## 0. Le dossier projet

Le serveur sert le dossier courant du harnais (`surgeon mcp` sans `--allow-dir`), sinon son
premier `--allow-dir`. Ce dossier est le projet : surgeon-gui le rouvre tel quel.

- Nouveau projet : appelle d'abord `surgeon_workspace` avec `action: "init"` et `model` = le
  dossier du modèle d'origine (`name` facultatif). Il écrit `project.surgeon.json` avec
  l'origine comme le Studio l'enregistre (chemin, SHA-256, taille) : le Studio la reconnaît et
  la verrouille. Sans ce fichier, aucun run n'est journalisé et `record_finding` échoue
  (`no_project`). Une erreur `output_exists` veut dire que le dossier est déjà un projet :
  continue avec lui, ne passe `force: true` que si l'utilisateur le demande.
- `copies/<id>/` : la copie de travail, la seule que tu modifies. Ne patche jamais le modèle
  d'origine. Copie son dossier dans `copies/` avec ton outil shell, ou demande à
  l'utilisateur de le faire ; l'origine se lit par `--allow-read`.
- `work/` : tes fichiers (prompts, jeux de sécurité, directions, `patch_*.json`), écrits par
  `surgeon_files` ou en sortie des outils.
- `chat/` et `ledger.jsonl` : tenus par le serveur et par le Studio. N'y écris jamais à la
  main : une ligne invalide masque tout le journal.

## 1. Diagnostic machine : `surgeon_doctor` d'abord

Appelle `surgeon_doctor` (avec `model` = la copie) avant toute campagne et lis `data.ready`
et `data.checks`.

- RAM : `data.memory.model.dtype` vaut `f32` (normal), `bf16` (repli lent, environ 2,5 fois :
  préviens l'utilisateur et propose de libérer de la RAM) ou `none` (arrête-toi).
- Cache `ref_bench` : `data.cache.corrupt` ou `data.cache.obsolete` au-dessus de 0, dis-le ;
  la purge se fait dans le Studio (Réglages > Moteur) ou par `surgeon doctor --clean-cache`.
- hipfire : `data.hipfire.gpu_memory.total_mb` à 0 = démon figé sans GPU. Alerte
  l'utilisateur et demande-lui de le redémarrer (Studio : Réglages > Moteur, ou
  `start-hipfire.ps1` après l'avoir arrêté) avant toute mesure sur hipfire. Démon absent : les
  mesures tournent sur `native`.

## 2. Inspecter, extraire, cribler en entonnoir

1. `surgeon_check` puis `surgeon_inspect` (`projections` `["o_proj", "down_proj"]`).
2. `surgeon_extract` contrastive : `pos_prompts_file` `builtin:refusals`, `neg_prompts_file`
   `builtin:general`, `method` `mean_diff`, `holdout_frac` 0.2, `out` dans `work/`. Sonde
   toutes les couches (une sur deux au-delà de 32), classe-les par `d_prime` et
   `threshold_accuracy`, consigne le classement avec `record_finding`.
3. `surgeon_apply` sur la copie, `dry_run: true` puis réel : `project_out` sur `o_proj` et
   `down_proj`, `layers` `all` sauf si la sonde justifie une plage, alpha de 0.4 à 1.0,
   `surgeon_restore` entre deux essais.
4. Crible chaque alpha avec `surgeon_bench` mode `refusal_probe` (`probe_method`
   `first-token-prob`, `prompts` `builtin:refusals`) : écarte ceux dont le refus sondé ne
   baisse pas ; seuls les deux plus petits alpha retenus passent en génération complète.
5. `surgeon_autotune` avec `refusal_prompts` (sans `out`, il ne fait que mesurer) calibre de
   la même façon l'échelle d'un patch de steering.
6. Préserver une capacité (maths, logique, code) : extrais sa direction à la même couche
   (`builtin:math` ou `builtin:logic` contre des prompts neutres), puis `surgeon_compose` avec
   `ablate` = la direction du comportement, `keep` = les directions de capacité, `mode`
   `orthogonal_preserve` (CLI `--mode orthogonal-preserve`), `out` et `patch_out` dans
   `work/` (`patch_compose.json`). Applique ce patch (dry run d'abord). Un refus
   `collinear_vectors` : capacité et comportement ne se séparent pas à cette couche, change de
   couche.

## 3. Certification : règle non négociable

- Un résultat `refusal_probe` porte `is_heuristic: true` (`evidence_level`
  `heuristic_screening`) : il oriente, il ne conclut jamais. Le Studio le marque
  « Heuristique ».
- Avant de déclarer un gagnant ou de clore, certifie chaque finaliste puis la copie dans son
  état final par génération complète, chaque fois avec `baseline_model` = le modèle d'origine :
  `surgeon_bench` mode `refusal` sur `builtin:refusals` ; mode `eval` sur `builtin:math`,
  `builtin:logic` et `builtin:general` (rétention) ; contrôle de sécurité : mode `refusal` sur
  au moins 10 demandes réellement dangereuses écrites une fois dans `work/`, dont le taux de
  refus ne doit pas baisser.
- Toute écriture (`surgeon_apply` réel, `surgeon_restore`, `surgeon_lora` en `merge`) rend
  caduques les mesures d'avant : certifie à nouveau.
- Garde les défauts : `max_new_tokens` (refusal 40, eval 512) et `chat_template` `auto`.
- Si tu ne peux pas certifier, écris que le résultat n'est « pas certifié ».

## 4. Consigner pour le Studio

- Dans un projet, chaque outil `surgeon_*` renvoie un `run_id` (journalisé dans
  `chat/mcp_runs.jsonl`). Cite-le dans `record_finding` : `evidence` =
  `[{"run_id": …, "metric": …, "value": <valeur exacte de data>}]`. Un run de génération
  complète donne « Mesuré », une sonde « Heuristique », un raisonnement seul (`reasoning`)
  « Déduit ».
- `update_plan` affiche ton plan, `show_in_3d` sélectionne un tenseur dans la vue 3D.
- Régression mesurée : `surgeon_restore` (`model` = la copie) et vérifie `data.identical`.
- Aucune affirmation sans chiffre mesuré et l'outil qui l'a produit ; sinon « non mesuré ».
- À la fin, dis à l'utilisateur d'ouvrir le dossier parent du projet dans surgeon-gui (il
  devient l'espace de travail) : le projet y montre ses résultats, ses constats et leurs
  badges.
