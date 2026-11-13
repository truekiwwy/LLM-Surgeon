# Studio : projets, jobs et flux d'événements

Le socle de l'espace `Projets` de `crates/surgeon-gui` : des projets persistants, un
gestionnaire de jobs qui appelle le moteur comme bibliothèque, et un flux d'événements unique
(NDJSON versionné) que la vue 3D, le journal et un agent lisent de la même façon.

Code : `crates/surgeon-gui/src/studio/` (`project`, `copy`, `jobs`, `exec`, `bus`, `ledger`,
`forms`, `view`, `selftest`) et, côté moteur, `src/events.rs` et `weight_surgeon::mcp::call_tool`.

## 1. Projet

Un projet est un dossier `<workspace>/<nom>/` (workspace par défaut : `~/surgeon-studio`,
option `surgeon-gui --workspace <DOSSIER>`) :

| Chemin | Contenu |
| --- | --- |
| `project.surgeon.json` | Manifeste versionné (schéma : [schemas/project.surgeon.schema.json](schemas/project.surgeon.schema.json)) |
| `runs/<id>/` | `params.json` (outil, arguments, priorité, budgets), `status.json`, `events.ndjson`, `out/` |
| `artifacts/` | Vecteurs, graphes, patchs et dumps que l'utilisateur garde |
| `copies/<id>/` | Copies de travail, les seuls modèles que le studio écrit |
| `chat/` | Conversations, actions et journal des preuves de l'agent ([AGENTS.md](AGENTS.md)) |
| `ledger.jsonl` | Journal d'expériences, en ajout seul |
| `project.lock` | PID du studio qui a ouvert le projet |

Le nom de projet accepte 1 à 64 lettres, chiffres, `-`, `_` et `.`, sans point initial.

### Manifeste

```json
{
  "schema_version": 1,
  "name": "qwen-refus",
  "created_unix": 1794567672,
  "source": { "path": "C:\\models\\qwen2.5-0.5b", "sha256": "…", "bytes": 988097824 },
  "state": "ready"
}
```

- `source` référence le modèle d'origine : chemin canonique, SHA-256 (d'un fichier, ou des
  fichiers de premier niveau d'un dossier, noms et contenus, comme `surgeon pipeline`) et
  taille. Le hachage est fait à la création (quelques secondes par Go).
- Écriture atomique (fichier temporaire puis renommage) : un arrêt brutal laisse l'ancien
  manifeste ou le nouveau, jamais un mélange.
- Lecture stricte : JSON illisible, `schema_version` absent, champ inconnu ou version
  inconnue sont refusés avec la raison, et le fichier n'est pas touché. Le projet apparaît
  alors « illisible » dans la liste.
- Version 1 : c'est la première. Un manifeste plus récent est refusé (« mettez le studio à
  jour ») plutôt que réécrit. La v2 ajoutera sa migration depuis la v1 dans `load_manifest`.

### États

| État | Signification |
| --- | --- |
| `Prêt` (`ready`) | Aucun job ne tourne, rien à signaler |
| `En cours` (`running`) | Au moins un job tourne |
| `Interrompu` (`interrupted`) | Le studio s'est arrêté pendant un run |
| `Attention requise` (`needs_attention`) | Un run a échoué ou dépassé son budget |

Transitions permises (toute autre est refusée) :

| Depuis | Événement | Vers |
| --- | --- | --- |
| tout état | un job démarre | `En cours` |
| `En cours` | le dernier job se termine bien, ou est annulé | `Prêt` |
| `En cours` | un job échoue ou dépasse son budget | `Attention requise` |
| `En cours` | ouverture après un arrêt brutal | `Interrompu` |
| `Interrompu`, `Attention requise` | l'utilisateur acquitte (`Ignorer`, `Vu`) | `Prêt` |

Quand plusieurs jobs se terminent pendant que le projet est `En cours`, c'est la pire fin
qui compte (échec, puis annulation, puis succès).

## 2. Copies de travail : garanties

Le modèle d'origine n'est jamais écrit par le studio.

1. Il n'est jamais passé à un outil qui écrit un modèle en place. `surgeon_apply` (hors
   `dry_run: true`) et `surgeon_lora` en `mode: merge` exigent un `model` situé sous
   `copies/` du projet ; sinon le job est refusé avant d'être mis en file.
2. Aucun argument de sortie (`out`, `out_dir`, `backup_dir`, `export_gguf`, `judge_cache`,
   `cache_dir`, `job_dir`) ne peut viser le modèle d'origine ou son dossier.
3. La copie (`Copie de travail`, outil interne `studio_copy`) vérifie d'abord l'espace
   libre du volume de `copies/` : taille du modèle plus une marge de 256 Mio, sinon refus
   avant toute écriture. Elle remplit `copies/<id>.partial/`, puis renomme en
   `copies/<id>/` ; une erreur ou une annulation supprime le dossier partiel.
4. `std::fs::copy` laisse le système cloner le fichier quand il le sait (clonefile sur
   APFS, copy_file_range sur btrfs et XFS, CopyFileEx sur ReFS ou Dev Drive) et copie les
   octets sinon. Le studio ne vérifie pas lequel a eu lieu.
5. Le moteur reçoit comme dossiers autorisés (sandbox MCP) le dossier du projet et celui du
   modèle d'origine, et garde ses propres garanties : sauvegarde vérifiée imposée avant
   `apply`, rollback en cas d'erreur ou d'annulation.

Une copie est un dossier plat : seuls les fichiers de premier niveau d'un modèle sont copiés
(config, tokenizer, poids), comme pour le SHA-256 de la source. Promouvoir un résultat
reste une action explicite, hors de ce socle.

## 3. Gestionnaire de jobs

- **Job** : un appel d'outil MCP du moteur (`surgeon_check`, `surgeon_inspect`,
  `surgeon_apply`, `surgeon_extract`, `surgeon_bench` (logits, ppl, eval, refusal, dump,
  judge-sample, judge-eval), `surgeon_eval` (génération), `surgeon_graph`,
  `surgeon_autotune`, `surgeon_diff`, `surgeon_lora`, `surgeon_pipeline`,
  `surgeon_backends`) ou une copie de travail. L'appel passe par
  `weight_surgeon::mcp::call_tool`, le chemin exact de `surgeon mcp` : même schéma
  d'arguments, même sandbox, mêmes cœurs `core_*` que la CLI.
- **Estimation** : `Estimer` lance d'abord le mode aperçu de l'outil (`surgeon_graph`
  `estimate`, `surgeon_apply` et `surgeon_lora` `dry_run`, `surgeon_pipeline` `plan`) comme
  un run ordinaire ; avec une `durée max`, la mise en file exige l'estimation de ces
  paramètres exacts et la refuse si elle dépasse le budget ([GUI_3D.md](GUI_3D.md)).
- **Formulaires** : un par outil, générés depuis `surgeon schema --kind tools` (les
  `inputSchema` de `tools/list`) : mêmes noms, mêmes champs requis, mêmes choix. Un test
  vérifie l'égalité pour chaque outil.
- **File** : priorité décroissante, puis ordre d'arrivée. Un seul job lourd à la fois ;
  `surgeon_check`, `surgeon_inspect`, `surgeon_backends` et les aperçus qui n'écrivent aucun
  modèle (`estimate`, `dry_run`) sont légers et passent à côté ; un `plan` de pipeline reste
  lourd (il peut calibrer en faisant tourner le modèle).
- **`{out}`** : dans les arguments, remplacé par `runs/<id>/out.partial/`. Le dossier
  devient `runs/<id>/out/` seulement si le run réussit (les chemins de l'enveloppe sont
  réécrits) ; sinon il est supprimé. `params.json` garde `{out}` tel quel.
- **Annulation** : un job en file est retiré ; un job en cours reçoit le drapeau que le
  moteur consulte entre tenseurs, prompts et blocs (`AppDeps::interrupted`), et l'apply
  fait son rollback.
- **Budgets** : `max_wall_time_secs` annule le run au-delà de la durée et met le projet en
  `Attention requise`. `max_candidates` ne vaut que pour `surgeon_pipeline` : un
  `pipeline.json` dont `budgets.max_candidates` est plus élevé est refusé à la soumission.
- **Arrêt brutal** : à l'ouverture, les runs restés `queued` ou `running` passent
  `interrupted` (avec une entrée au journal), leur `out.partial` est supprimé et le projet
  passe `Interrompu`. `Reprendre` soumet à nouveau le même `params.json` comme un nouveau
  run (`resumed_from`), une seule fois par run interrompu.
- **Verrou** : `project.lock` contient le PID du studio. Un autre processus vivant bloque
  l'ouverture ; un PID mort (ou celui du processus courant) est repris.

## 4. Protocole d'événements `SurgeonEvent` (v1)

Une ligne JSON par événement :

```json
{"v":1,"seq":0,"t_ms":0,"event":"job_started","command":"apply"}
{"v":1,"seq":1,"t_ms":12,"event":"node_state","node_id":"model.layers.1.mlp.down_proj.weight","state":"done"}
{"v":1,"seq":2,"t_ms":12,"event":"progress","done":1,"total":2,"unit":"tensor"}
{"v":1,"seq":5,"t_ms":30,"event":"finished","exit_code":0}
```

| `event` | Champs |
| --- | --- |
| `job_started` | `command` |
| `progress` | `done`, `total` (absent si inconnu), `unit` (`tensor`, `matrix`, `prompt`, `question`, `chunk`, `block`, `verdict`, `forward`, `weight_file`, `file`…) |
| `node_state` | `node_id` (nom de tenseur ou id de nœud du graphe), `state` : `pending`, `running`, `done`, `failed` |
| `metric` | `node_id` (optionnel), `name`, `value` (fini) |
| `log` | `level` : `info`, `warn`, `error` ; `message` |
| `artifact` | `kind`, `path`, `sha256` (absent si non haché) |
| `warning` | `message` |
| `finished` | `exit_code` |
| `failed` | `kind` (le `error.kind` de l'enveloppe), `message`, `exit_code` |
| `cancelled` | — |

- `v` : version du format. Un lecteur refuse une version plus récente ; les champs ajoutés
  dans une même version sont optionnels et ignorés par un lecteur plus ancien.
- `seq` part de 0 sans trou : un trou signale une ligne perdue. `t_ms` compte depuis le
  début du flux. La première ligne est `job_started`, la dernière `finished`, `failed` ou
  `cancelled`.
- Schéma JSON d'une ligne : `surgeon schema --kind events`. Exemple de chaque événement :
  `tests/fixtures/events/v1.ndjson` (test de compatibilité dans les deux sens).
- Le moteur émet aujourd'hui `node_state` en `done` seulement (tenseur décalé, matrice du
  graphe analysée, prompt évalué) ; `pending`, `running` et `failed` font partie du format.
  Un tenseur décalé par `apply` porte aussi `metric` `max_abs_delta` (le plus grand |ΔW|
  de ce tenseur), que la vue 3D colore.

### Côté moteur : `--events`

```bash
surgeon --json --events run.ndjson apply --model copie/ --patch patch.json
surgeon --json --events - bench refusal ...   # les événements sur stderr
```

Option globale, additive : sans elle rien ne change, et avec elle l'enveloppe JSON de stdout
reste identique (test `tests/events.rs`). Les lignes viennent des `ProgressEvent` que les
cœurs envoient déjà à la CLI et au serveur MCP (`events::from_progress`) ; un fichier non
créable arrête la commande avant tout travail (`invalid_input`). Une erreur d'écriture en
cours de route ajoute un avertissement à l'enveloppe. Les commandes qui ne publient pas de
progression (`check`, `inspect`, `pipeline`…) n'écrivent que la première et la dernière
ligne. `schema`, `mcp` et `tui` l'ignorent.

### Côté studio : canal, coalescence, rejeu

- Le fil du job écrit chaque ligne dans `runs/<id>/events.ndjson` (une écriture et un flush
  par ligne) puis la passe à la fenêtre par un canal borné (1024 lignes).
- Contre-pression : une ligne `progress` qui ne trouve pas de place est sautée pour l'écran
  seulement (le fichier l'a, le nombre de lignes sautées est affiché) ; toute autre ligne
  attend de la place, ce qui ralentit le job au lieu de la perdre.
- La fenêtre replie le canal au plus 60 fois par seconde (`RunView`).
- Rejeu : `RunView::replay` reconstruit un run depuis son fichier ; à la fin d'un run, la
  vue affichée est celle du fichier. Une dernière ligne tronquée (arrêt brutal pendant
  l'écriture) est ignorée ; une ligne illisible ailleurs est une erreur.
- La vue 3D dessine l'état de chaque nœud qu'un run a atteint (id de nœud ou nom de
  tenseur), en direct ou en rejeu avec une barre de lecture : [GUI_3D.md](GUI_3D.md).

## 5. Journal d'expériences

`ledger.jsonl` : une ligne par run terminé, annulé ou interrompu, jamais réécrite (ajout
avec une seule écriture puis `sync`). Chaque entrée contient :

- `run_id`, `tool`, `label`, `params` (avec `{out}`), `resumed_from` ;
- `started_unix`, `finished_unix`, `verdict` (`ok`, `failed`, `cancelled`,
  `interrupted`), `exit_code`, `result` (le `data` de l'enveloppe), `error` ;
- `hashes` : `model` (le SHA-256 enregistré de la source si le run la lit, sinon
  l'empreinte rapide `fast-…` du moteur sur la copie, prise avant le run), `inputs`
  (SHA-256 des fichiers d'entrée de 512 Mio au plus, par nom d'argument), `binary`
  (SHA-256 de l'exécutable du studio, qui contient le moteur), `seed` ;
- `versions` : version du studio, du protocole d'événements et du manifeste.

`Exporter JSON` et `Exporter Markdown` écrivent `ledger.json` et `ledger.md` à la racine
du projet.

## 6. Graphe éphémère

`surgeon-gui graph.json`, `surgeon-gui graph.ndjson` ou le champ `Graphe éphémère` de
l'espace `Projets` ouvrent un graphe dans la vue 3D sans créer de projet.

## 7. Vérification

```bash
cargo test --test events --test compat --test deps_purity     # moteur
cd crates/surgeon-gui && cargo test                           # studio, sans GPU
cargo run -- --self-test                                       # bout en bout, sans fenêtre
```

`--self-test` crée un espace temporaire, un projet sur un petit modèle SafeTensors, une
copie de travail, lance `surgeon_check` par le moteur, vérifie le refus d'un apply sur
l'origine et son empreinte inchangée, le rejeu de chaque run, le journal et les
formulaires, puis supprime tout.

## Limites connues

- Pas de télémétrie, aucune donnée ne sort de la machine ; aucun secret n'est écrit par ce
  socle (les arguments des jobs sont journalisés tels quels : ne pas y mettre de secret).
- Les captures hors écran (`--screenshot`) n'existent pas : la fenêtre se vérifie à la main.
- Un job lancé par l'agent en mode « Demander » porte `"read_only": true` dans son
  `params.json` : le moteur le reçoit comme `surgeon mcp --read-only`.
- Un job `surgeon_pipeline start` lance un travailleur détaché : le run du studio se
  termine quand le travailleur est lancé, son avancement se suit par `action: status`.
