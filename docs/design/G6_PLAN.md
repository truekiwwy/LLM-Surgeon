# G6 : architecture d'information, premier démarrage, fournisseurs IA

Plan validé en entretien (questions 1 à 12). Il complète `DIRECTION.md` sans le remplacer :
la direction visuelle, les jetons et la composition restent ceux de G5. Ce document fixe
ce que les huit tranches doivent livrer, dans quel ordre, et comment on vérifie chacune.

## 1. Constat de départ

- La colonne `EXPLORATEUR` s'affiche dans quatre onglets sur cinq, y compris `Agents` où
  l'arbre des tenseurs ne sert à rien.
- L'Atelier empile six éléments : histogramme, formulaire, table des deltas, aperçu 3D,
  activité, approbations.
- `[ADVANCE]` ouvre un chat d'audit qui double `Agents` (D-07) au lieu de régler la
  profondeur de la chirurgie.
- La colonne droite d'`Agents` mélange connexions, clés, prix, consentement, ajout de
  fournisseur, mode, budgets et règles.
- `Diagnostics` affiche quatre pastilles rouges au premier lancement : `diag::Status` est
  binaire (`Ok | Missing | Outdated`), sans notion de requis ou de facultatif.
- `surgeon (CLI)` est marqué manquant alors que l'atelier utilise la bibliothèque
  intégrée. Il n'est requis que par les runtimes CLI, qui lancent `surgeon mcp`
  (`agent/runtime.rs`, `surgeon_exe`).
- Le wizard (`desktop/wizard.rs`) date d'avant G5 : modale egui brute de 620 px, huit
  étapes, `theme::GOOD`, chaînes `Lang::pick` hors de `ui_lint` (D-08). Son étape Agent
  annonce encore qu'aucun hub d'agents n'existe.
- La détection ne couvre que `claude` et `codex` dans le `PATH`. Rien pour les serveurs
  locaux, les variables d'environnement ni OpenCode.

## 2. Décisions

Elles prolongent le journal de `DIRECTION.md` (§ 7). D-07 et D-08 sont remplacées.

| # | Décision | Raison |
| --- | --- | --- |
| D-11 | Cinq onglets : `Atelier`, `Graphe 3D`, `Analyses`, `Agents`, `Système`. `Système` remplace `Diagnostics` et devient le seul espace de configuration (machine, dépendances, fournisseurs IA, mises à jour, journal). `--screen diagnostics` reste un alias de `system` | séparer « configurer une fois » de « travailler » |
| D-12 | La colonne gauche dépend de l'onglet, à largeur constante : Explorateur (projet, copie, arbre) dans `Atelier`, `Graphe 3D`, `Analyses` ; projet puis conversations dans `Agents` ; quatre sections dans `Système`, sans contexte projet | l'arbre seulement là où il sert ; aucun saut de mise en page |
| D-13 | La colonne gauche se replie (`Ctrl+B` et un chevron en tête de colonne), état global persistant | toute la largeur pour la 3D ou les formulaires |
| D-14 | `[SCALPING]` / `[ADVANCE]` ne règlent que la profondeur de l'Atelier. `[SCALPING]` : champs `required` et deux ou trois réglages clés, pas de sélecteur Cartes / Patch JSON. `[ADVANCE]` : tous les champs du schéma, sélecteurs fins (calques, rôles, experts), sélecteur Chirurgie / Cartes / Patch JSON. L'interrupteur quitte la barre de titre pour l'en-tête de l'Atelier. Le mode ne touche jamais l'agent | remplace D-07 : l'agent vit dans `Agents` seulement |
| D-15 | Atelier en trois zones : Explorateur ; Ciblage (histogramme et seuil, table des cibles, lignes ciblées en tête) ; Opération (outil, champs, estimation, Essai / Appliquer, puis progression et logs à la même place). Plus d'aperçu 3D (lien `Voir en 3D`), plus d'audit, plus d'approbations | « quoi toucher » au centre, « comment » à droite |
| D-16 | `Analyses` porte l'historique unique du projet (analyses et chirurgies, filtre par type). La colonne Opération de l'Atelier ne montre que le run en cours ou le dernier, avec « Voir dans l'historique » | une seule place pour l'historique |
| D-17 | `Voir en 3D` ouvre `Graphe 3D` cadré sur les cibles, surlignées comme les nœuds modifiés d'un diff | le 3D reste le moment visuel, sans second rendu |
| D-18 | `Agents` : session au centre (en-tête `connexion · modèle` qui ne liste que les fournisseurs prêts, plus « Configurer un fournisseur… » ; approbations dans le fil), Contrat (Demander / Modifier, me demander / décider seul, budgets, règles) et Preuves à droite. Clés, URL, prix et tests partent dans `Système` | usage et configuration séparés |
| D-19 | Une approbation en attente se signale par un compteur sur l'onglet `Agents` et une ligne cliquable dans la barre d'état | l'utilisateur n'est jamais forcé de surveiller `Agents` |
| D-20 | Fournisseur par défaut global (`agents.json`, `default_connection`), hérité par chaque projet, modifiable dans l'en-tête de session (`chat/agent.json`, existant) | un choix pour tous, une surcharge par projet |
| D-21 | Dépendances groupées par rôle : Essentiel (workspace, moteur intégré), Accélération (llama.cpp, Vulkan), Intégrations agents (surgeon CLI), Exemples (Qwen). Gravité selon l'usage : rouge = requis manquant (bloquant) ; orange = facultatif devenu nécessaire par un choix actif (« Requis par Claude Code ») ; gris = « Non installé · facultatif ». Bandeau « Prêt à opérer » / « N action(s) requise(s) » | le rouge veut dire « tu ne peux pas travailler », jamais « tu n'as pas tout » |
| D-22 | Wizard en page plein écran dans la fenêtre sans bordure, bâti sur `TOKENS` et `ui::components`, en quatre étapes : 01 Bienvenue, 02 Fournisseurs IA & Agents, 03 Espace de travail, 04 Composants optionnels. Visite guidée et Récapitulatif supprimés. Les étapes 02 à 04 réutilisent les panneaux de `Système` | un seul code pour configurer au premier lancement et plus tard |
| D-23 | Le wizard s'ouvre seulement si `wizard_done == false`. Chaque choix confirmé est enregistré tout de suite. `Passer la configuration` crée le workspace par défaut, met `[SCALPING]` et ouvre l'Atelier. Une fermeture en cours de route rouvre le wizard à l'étape 01, valeurs préremplies. `Terminer` laisse les installations continuer en arrière-plan (barre d'état, résultat dans `Système`) | aucune friction, aucun état perdu |
| D-24 | Prise en charge : CLI Claude Code, Codex, OpenCode (si son `--help` permet le non-interactif, une sortie JSON et un MCP limité à `surgeon`, sinon « détecté, non pris en charge ») ; serveurs locaux Ollama `:11434`, LM Studio `:1234`, llama-server `:8080` par `Compatible` ; cloud Anthropic, OpenAI, préréglage OpenRouter, endpoint personnalisé. Détectés seulement : `gemini` et les autres variables connues | ne prendre en charge que ce qu'on peut garantir |
| D-25 | Clés d'environnement (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`) : présence seule au scan ; copie dans le coffre Windows puis test `/models` au clic sur `Utiliser`. Clé saisie : enregistrée seulement après un `/models` réussi. Affichage masqué `••••…a1f3` | rien n'est lu ni stocké sans geste explicite |
| D-26 | CLI : « Installé · options OK » au scan, « Prêt » seulement après `Tester` (requête minimale en lecture seule). En cas d'échec d'authentification, la commande de connexion de la CLI est proposée à la copie. Jamais de quota consommé au scan, jamais de lecture de leurs identifiants | règle existante de `runtime.rs` conservée |
| D-27 | `local_only` unique dans `settings.json` (migration depuis `agents.json`) : aucune requête sortante, ni IA distante, ni CLI, ni téléchargement (instructions manuelles à la place). Badge `Local` dans la barre d'état quand il est actif | un concept, une source de vérité |
| D-28 | Sans fournisseur IA, toute la chirurgie, la 3D et les analyses marchent ; aucun rouge, aucun badge ; `Agents` montre un état vide guidé | l'IA est une option |
| D-29 | Le wizard passe sous `src/ui/` : `ui_lint` et `ui_i18n` le couvrent, `Lang::pick` disparaît de l'interface | remplace D-08 : plus d'exception |

## 3. Persistance

| Préférence | Où | Portée |
| --- | --- | --- |
| mode `[SCALPING]` / `[ADVANCE]` | `settings.json`, `mode` | globale |
| colonne gauche repliée | `settings.json`, `left_collapsed` | globale |
| `local_only` | `settings.json` (lu une fois depuis `agents.json`, puis retiré de l'écriture) | globale |
| fournisseur par défaut | `agents.json`, `default_connection` | globale |
| surcharge du fournisseur | `chat/agent.json`, `connection` + `model` (existant) | projet |
| filtre de l'historique, section de `Système` | mémoire | session |

Les champs ajoutés sont `#[serde(default)]` : un `settings.json` ou un `agents.json`
existant se relit sans erreur. `agents.json` garde `schema_version: 1` ; le champ
`local_only` y reste accepté en lecture pour la migration.

## 4. Invariants

- Toute surface passe par `TOKENS` et `ui::components`, aucune couleur ni taille en dur.
- `tests/ui_lint.rs` et `tests/ui_i18n.rs` passent, wizard compris ; chaque clé existe
  en `fr.json` et `en.json`.
- 0 warning : `cargo clippy -p surgeon-gui --all-targets -- -D warnings`, `cargo fmt --check`.
- `cargo test -p surgeon-gui` vert à chaque tranche.
- Les nouvelles fonctions pures ont leurs tests : gravité des dépendances, analyse des
  réponses `/v1/models`, détection des variables, migration de `local_only`, filtrage
  des champs par mode, héritage du fournisseur par défaut.
- Sécurité inchangée : `Secret` jamais affiché ni sérialisé, consentement avant la
  première requête distante, liste d'outils fermée pour les runtimes, `--read-only` en
  mode Demander, aucune auto-approbation.
- Le scan réseau ne vise que `127.0.0.1`, hors du fil d'interface, avec un délai court ;
  il reste permis en `local_only`, puisqu'il ne sort pas de la machine.
- Captures avant / après aux deux tailles de référence (1440 × 900, 1920 × 1080).

## 5. Tranches

Une tranche = un commit vérifiable (`feat:` / `refactor:`), tests, clippy et build verts
avant la suivante.

### Tranche 1 : shell

- `app.rs` : `Tab::Diagnostics` → `Tab::System`, nom d'écran `system`, alias
  `diagnostics` dans `Tab::from_name` ; `set_mode` ne touche plus `agents.agent.mode`.
- `ui/shell.rs` : retrait de l'interrupteur de mode de la barre de titre ; `Ctrl+B`.
- `ui/layout.rs` : `regions()` reçoit l'état replié (colonne gauche à 0, centre élargi),
  tests de grille étendus au cas replié.
- `ui/views/mod.rs` : chevron de repli dans l'en-tête de l'Explorateur.
- `ui/views/diagnostics.rs` → `ui/views/system.rs`, sections : Machine et dépendances,
  Fournisseurs IA (vide pour l'instant), Mises à jour, Journal ; plus de contexte projet.
- `settings.rs` : `mode`, `left_collapsed`.
- `ui/palette.rs`, `tests/ui_render.rs`, `tests/desktop.rs`, locales : suivre le renommage.

### Tranche 2 : Atelier et historique

- `ui/views/atelier.rs` : trois zones (D-15), interrupteur de mode dans l'en-tête de
  Ciblage, retrait de `audit()`, de l'aperçu et de l'approbation ; lien `Voir en 3D`.
- `ui/views/analyses.rs` : `fields()` filtre par mode (`required` en `[SCALPING]`, plus la
  liste courte des réglages clés par outil) ; la file devient l'historique unifié filtrable.
- `app.rs` : `targets()` sert aussi au cadrage `Voir en 3D` ; ouverture de `Graphe 3D`
  centrée sur la moyenne des cibles (même calcul que `show_graph`).
- Tests : filtrage des champs par mode, cadrage sur les cibles.

### Tranche 3 : Agents

- `ui/views/agents.rs` : colonne gauche projet puis conversations (Nouvelle, reprendre,
  exporter) ; en-tête de session `connexion · modèle` ; approbations dans le fil ;
  colonne droite Contrat puis Preuves ; `connections()` retiré.
- `agent/view.rs` : liste des conversations du projet, fournisseur effectif =
  surcharge du projet sinon défaut global.
- `ui/shell.rs` : compteur sur l'onglet `Agents`, ligne d'approbation dans la barre d'état.
- Tests : héritage du fournisseur par défaut.

### Tranche 4 : dépendances

- `diag.rs` : `Row` gagne un groupe (`Essential | Acceleration | Agents | Samples`) et une
  gravité calculée (`Blocking | NeededBy(&str) | Optional | Ok`) depuis l'état et les choix
  actifs (backend choisi, runtimes CLI configurés) ; ligne « moteur intégré » toujours Ok.
- `ui/views/system.rs` : bandeau d'état, groupes, trois couleurs, bouton Installer.
- Tests : les tables de gravité (workspace absent → bloquant ; Claude Code configuré sans
  `surgeon.exe` → `NeededBy` ; rien de configuré → facultatif).

### Tranche 5 : détection

- `agent/detect.rs` (nouveau) : scan parallèle de `127.0.0.1:{11434,1234,8080}/v1/models`
  (300 ms), variables d'environnement connues (présence seule), CLI `claude`, `codex`,
  `opencode`, `gemini`.
- `agent/runtime.rs` : OpenCode ajouté seulement après lecture de son `--help` réel ; ses
  options vérifiées comme celles de Claude Code et Codex ; sinon il reste « détecté ».
- `agent/config.rs` : `ProviderKind::OpenCode` si pris en charge ; préréglage OpenRouter
  (`Compatible` avec URL et clé).
- Tests : analyse des réponses `/v1/models` (Ollama, LM Studio, llama-server, réponse
  invalide, délai dépassé), détection des variables sur un environnement simulé.

### Tranche 6 : Fournisseurs IA, défaut global, local seulement

- `ui/views/system.rs` : section Fournisseurs IA (D-25, D-26) : liste groupée Agents (CLI),
  Serveurs locaux, Cloud ; détail à droite (clé masquée, modèle par défaut, prix,
  consentement, Tester, Supprimer) ; ★ défaut ; ajout OpenAI, OpenRouter, personnalisé.
- `agent/view.rs` : `Utiliser` (env → coffre → test), enregistrement après test réussi,
  test des CLI avec la commande de connexion en cas d'échec.
- `settings.rs`, `agent/config.rs` : `local_only` unifié et migration ; badge `Local`.
- Tests : migration de `local_only`, clé non enregistrée après un test refusé.

### Tranche 7 : wizard

- `desktop/wizard.rs` → `ui/wizard.rs`, page plein écran à la place des vues, quatre
  étapes (D-22), `Passer la configuration`, enregistrement immédiat de chaque choix.
- `desktop/mod.rs` : les `Task` d'installation (llama.cpp, Qwen) appartiennent à
  `Desktop`, pour survivre à `Terminer` ; progression dans la barre d'état.
- `main.rs` : `--screen wizard:N` (1 à 4) remplace `capture_wizard`.
- Locales : toutes les chaînes du wizard en clés i18n ; `Lang::pick` retiré de l'interface.
- Tests : `Passer` crée le workspace et ferme le wizard ; une fermeture sans `Terminer`
  laisse `wizard_done` faux ; consentement toujours requis avant un téléchargement.

### Tranche 8 : finition

- Captures avant / après des cinq onglets et des quatre étapes, aux deux tailles.
- Passe d'audit (`AUDIT.md` style G5) sur les nouveaux écrans.
- `DIRECTION.md` : D-07 et D-08 marquées remplacées, renvoi vers ce plan.
- Vault Obsidian : note d'architecture de `surgeon-gui` mise à jour (onglets, wizard,
  détection).

## 6. Hors périmètre

- Nouvel onglet, fusion d'`Analyses` dans l'Atelier.
- Gemini CLI, API Gemini, ports vLLM et Jan.
- Reprise du wizard à l'étape exacte.
- Audit IA dans l'Atelier.
- Test automatique des CLI au scan.
- Coffre de secrets hors Windows.

## 7. Risques

- **OpenCode** : sa CLI n'est pas encore vérifiée. Si elle ne permet pas de limiter les
  outils au seul serveur `surgeon`, elle reste « détectée » et la tranche 5 le dit.
- **`Task` passées dans `Desktop`** (tranche 7) : le wizard et `Système` lisent le même
  état ; une installation lancée deux fois doit être refusée, pas doublée.
- **Renommage `Diagnostics`** : les scripts de capture et la documentation qui citent
  `--screen diagnostics` gardent l'alias, à retirer plus tard.
- **Test des CLI** : consomme un peu du quota de l'utilisateur, ce qui est dit dans
  l'infobulle de `Tester`.
