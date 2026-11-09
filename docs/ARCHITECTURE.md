# Architecture : Surgery Engine, front-ends et points d'extension

Ce document décrit ce que le code contient aujourd'hui. Ce qui n'existe pas encore est
rangé dans la [feuille de route](#feuille-de-route-non-implémenté), séparée du reste.

## Vocabulaire

Ces termes ont le même sens dans tout le dépôt (aide du CLI, serveur MCP, documentation).

| Terme | Ce que c'est | Où |
| --- | --- | --- |
| **Surgery Engine** | La bibliothèque Rust `weight_surgeon` (paquet `weight-surgeon`) : moteur mmap, noyaux, formats, extraction, patch, LoRA, diff de faible rang, mesure, évaluation. | `src/lib.rs` et les modules ci-dessous |
| **CLI front-end** | Le binaire `surgeon` : analyse des arguments, rendu terminal, tableau de bord. | `src/main.rs`, `src/cli.rs`, `src/app.rs`, `src/ui.rs`, `src/tui/` |
| **Agent extension (MCP)** | Le serveur `surgeon mcp` : neuf outils typés en JSON-RPC sur stdio, dossiers autorisés, sauvegarde imposée. | `src/mcp/`, [docs/MCP.md](MCP.md) |
| **Extension point** | Un trait Rust public, versionné, qu'un tiers implémente et enregistre. Un seul aujourd'hui : `ExtractionMethod`. | `src/plugin/` |
| **Plugin** | Soit une implémentation d'un extension point compilée avec la bibliothèque, soit l'empaquetage `plugin/` pour Claude Code. Rien d'autre. | [docs/PLUGINS.md](PLUGINS.md) |

Le CLI et le serveur MCP sont deux front-ends du même moteur : tous deux appellent les
fonctions `app::core_*`, qui renvoient un `Outcome` rendu en texte (terminal) ou en
enveloppe JSON (`--json`, résultats MCP).

## Vue d'ensemble

```text
   surgeon (CLI front-end)                       surgeon mcp (Agent extension, 9 outils)
   cli.rs -> app::dispatch, ui.rs, tui/          mcp/ : sandbox, protocole, tools.rs
                 \                                  /
                  app::core_* -> Outcome -> envelope_of (JSON partagé)
                                    |
 ----------------------------- Surgery Engine (bibliothèque) ------------------------------
  extract  forward Qwen2 (candle, CPU) -> plugin::ExtractionMethod <- plugin::Registry
           (mean_diff, svd, leace intégrées ; méthodes tierces enregistrées)
  patch    patch.json, validation, sauvegarde, rollback, autotune ; patch::gguf (GGUF en place)
  engine   SafeTensors mmap, décalages de lignes ; kernel : noyaux SIMD, prefetch, E/S
  surgery  .surgery v1, SVD randomisée ; lora : fusion / extraction PEFT
  bench    KL et top-k entre dumps, perplexité (llama-perplexity), Pass@1 ; eval : llama.cpp
  hardware budget RAM/VRAM ; linalg : QR, Cholesky, Jacobi, Gram
```

Dépendances entre modules mesurées par graphify (graphe AST de `src/`, 3147 nœuds, arêtes
comptées entre fichiers de modules différents) : `app -> patch` 64, `mcp -> app` 62,
`mcp -> bench` 54, `patch -> extract` 45, `bench -> extract` 34, `extract -> engine` 24,
`cli -> extract` 15, `extract -> hardware` 10, `extract -> linalg` 4. Le CLI passe par
`app` ; le serveur MCP appelle `app` et, pour construire ses requêtes, les types de `bench`,
`patch` et `extract`. `extract` est le module le plus partagé du moteur.

## Composants : rôle et extensibilité

| Composant | Module | Rôle | Extensible aujourd'hui ? | Comment |
| --- | --- | --- | --- | --- |
| Méthodes d'extraction | `plugin`, `extract` | Direction (vecteur unitaire) depuis les activations poolées de deux ensembles de prompts | **Oui** (API `v0`, instable) | `impl ExtractionMethod`, `Registry::register`, `extract::run_extraction_plugin` ; voir [docs/PLUGINS.md](PLUGINS.md) |
| CLI front-end | `cli`, `app`, `ui`, `tui` | Commandes, rendu, codes de sortie | Non | `--method` accepte les identifiants du registre intégré |
| Agent extension (MCP) | `mcp` | 9 outils typés, sandbox de chemins | Non | Schémas figés, vérifiés par `tests/compat.rs` ; `method` accepte les identifiants du registre intégré |
| Forward Qwen2 et capture | `extract` | Activations d'une couche (site, pooling) | Non : interne, non remplaçable par un tiers à ce jour | aucun |
| Moteur mmap et décalages | `engine` | Lecture et écriture en place des SafeTensors | Non : interne, non remplaçable par un tiers à ce jour | aucun |
| Noyaux SIMD et prefetch | `kernel` | AVX-512, AVX2, NEON ou scalaire, choisis à l'exécution par détection du CPU (`SURGEON_KERNEL` pour forcer) ; compilés dans le binaire | Non : interne, non remplaçable par un tiers à ce jour | aucun |
| Patch SafeTensors | `patch` | `patch.json` (`add`, `subtract`, `project_out`), validation, sauvegarde, rollback | Non : interne, non remplaçable par un tiers à ce jour | Feuille de route |
| Patch GGUF en place | `patch::gguf` | Déquantification, décalage, requantification | Non : interne, non remplaçable par un tiers à ce jour | aucun |
| Formats d'adaptation | `engine`, `patch::gguf`, `surgery`, `lora` | SafeTensors, GGUF, `.surgery` v1, adaptateurs PEFT | Non : interne, non remplaçable par un tiers à ce jour | Feuille de route |
| Fusion et extraction LoRA | `lora` | `W += scale · (alpha / r) · B A`, extraction de rang `r` | Non : interne, non remplaçable par un tiers à ce jour | aucun |
| Diff de faible rang | `surgery` | Patch `.surgery` entre deux checkpoints | Non : interne, non remplaçable par un tiers à ce jour | aucun |
| Mesure | `bench` | KL, top-k, perplexité, Pass@1, dumps de logits ; lecture best-effort de `.npy` et `.pt`/`.pth` pour les seuls dumps de logits | Non : interne, non remplaçable par un tiers à ce jour | Feuille de route |
| Évaluation | `eval` | Génération via llama.cpp, détection de boucles, export GGUF | Non : interne, non remplaçable par un tiers à ce jour | aucun |
| Budget matériel | `hardware` | RAM, VRAM (NVML), verdict | Non : interne, non remplaçable par un tiers à ce jour | aucun |
| Algèbre linéaire | `linalg` | QR, Cholesky, Jacobi, Gram, en Rust pur | Non : interne, non remplaçable par un tiers à ce jour | aucun |

**Traits publics d'injection.** La bibliothèque expose d'autres traits publics
(`HardwareProbe`, `TensorSource`, `ExtractObserver`, `PatchBackend`, `RollbackHandle`,
`TextGenerator`, `CommandRunner`, `ProgressSink`, `PageHints`, ceux de `tui`...). Ils servent
à injecter des doublures dans les tests et à brancher les front-ends. Ce ne sont pas des
extension points : pas de registre, pas de versionnage, aucune promesse de stabilité.

## L'extension point `ExtractionMethod` (API `v0`)

Partage des responsabilités pendant une extraction :

| Étape | Qui | Détail |
| --- | --- | --- |
| Paramètres | méthode | `validate(params)` avant toute passe avant : une erreur ne coûte rien |
| Exigences | moteur | `requires_paired` (autant de prompts de chaque côté), fichiers annexes déclarés absents ou `overwrite` |
| Activations | moteur | Forward Qwen2, capture au site choisi, pooling : `ActivationSet { pos, neg }` |
| Direction | méthode | `extract(ctx, acts, cancel)` : vecteur unitaire, diagnostics, métadonnées, fichiers annexes |
| Contrôle | moteur | Taille `hidden_size`, valeurs finies, norme 1 à 1e-6 près ; sinon code 7, rien d'écrit |
| Statistiques | moteur | d′, seuil, projection de chaque prompt, le long de la direction rendue |
| Écriture | moteur | `<out>`, `<out>.stats.json` et chaque `<out>.<suffixe>.safetensors`, ensemble et atomiquement |

Décisions prises pour garder les sorties des méthodes intégrées identiques à l'octet près
(vérifié par `tests/compat.rs` sur les fichiers de `mean_diff`, `svd` et `leace`) :

- **Direction en f64.** Le moteur calcule les statistiques le long de la direction rendue,
  puis l'écrit en f32. Une direction déjà arrondie en f32 aurait changé les projections du
  `.stats.json`. La spécification initiale parlait d'un vecteur f32 ; elle a été adaptée.
- **Diagnostics typés pour les méthodes intégrées.** `serde_json` trie les clés d'un objet
  JSON générique ; les diagnostics de `svd` et `leace` gardent donc leur type
  (`Diagnostics::Builtin`) et leur ordre de champs. Une méthode tierce rend un objet JSON
  (`Diagnostics::Json`), dont le moteur fixe la clé `method`.
- **`mean_diff` sans clé `method`.** Ses fichiers restent ceux du prompt 8 (absence de la
  clé = `mean_diff`). Toute autre méthode reçoit `method = <identifiant>` dans l'en-tête.
- **Options partagées.** Les trois méthodes intégrées lisent les mêmes quatre options
  (`svd_center`, `leace_shrinkage`, `leace_rcond`, `holdout_frac`), toutes validées quelle
  que soit la méthode, comme le CLI l'a toujours fait.
- **Ordre de liste.** Méthodes intégrées dans l'ordre historique, puis méthodes
  enregistrées par identifiant. C'est l'ordre des valeurs possibles de `--method`, du
  `oneOf` du schéma MCP et du message d'erreur, inchangés.
- **Registre intégré dans le binaire.** Le binaire `surgeon` (CLI et MCP) n'utilise que
  `Registry::with_builtin()`. Une méthode tierce s'exécute par la bibliothèque, dans un
  programme qui l'enregistre. Raison : `AppDeps` est construit champ par champ dans six
  fichiers de tests ; y ajouter le registre aurait modifié des tests existants.

Codes de sortie : une `PluginError` devient une `ExtractError` et prend les codes de
`surgeon extract` (2 paramètres invalides, 1 annulation, 7 échec de calcul ou sortie
refusée).

## Compatibilité

`tests/compat.rs` compare l'état courant à trois références générées avant l'API `v0`
(`tests/fixtures/compat/`) : l'arbre complet des sous-commandes et options du CLI, les
schémas d'entrée des neuf outils MCP sans leurs textes (`description`, `title`), et les
SHA-256 des sorties `--json` normalisées de `check`, `inspect`, `apply --dry-run`, `schema`,
`schema --kind tools` et `extract` (enveloppe et fichiers) pour chaque méthode intégrée.
Seule addition déclarée : la sous-commande `surgeon plugins list`.

## Feuille de route (non implémenté)

Ces extension points n'existent pas. Ils sont listés avec ce qui les bloque aujourd'hui.

| Extension point envisagé | Pourquoi il n'existe pas encore |
| --- | --- |
| Opérations de patch | `operation` est l'enum `OperationKind` : il fixe le schéma de `patch.json` (`surgeon schema`), l'`inputSchema` de `surgeon_apply`, `validate_patch`, le décalage du moteur (`ShiftOp`) et la requantification GGUF. Une opération ouverte demande un `schema_version` 2 de `patch.json` et un changement de surface MCP. |
| Adaptateurs de format | Les chemins d'écriture (mmap en place, sauvegarde, rollback, écriture atomique, requantification) sont propres à chaque format. Seule la lecture de l'extraction passe déjà par un trait (`TensorSource`), sans registre. |
| Métriques de bench | Les modes sont l'enum `BenchMode` / `BenchCommand` : ils fixent l'`inputSchema` de `surgeon_bench` et les sous-commandes `surgeon bench`, et chaque rapport a un type figé. |
| Registre injectable dans le front-end | Permettrait à un binaire tiers de réutiliser tout le CLI et le serveur MCP avec ses méthodes. Demande un constructeur d'`AppDeps` (voir plus haut). |
| Plugins hors processus | Spécification proposée, non implémentée : [docs/PLUGINS.md](PLUGINS.md#proposition--plugins-hors-processus-non-implémenté). |

Hors périmètre, et absent du code : adaptateur PyTorch (seule existe la lecture best-effort
de dumps de logits `.pt`/`.npy` dans `bench`), noyaux SIMD chargeables dynamiquement (les
noyaux sont compilés dans le binaire), intégration Ollama.
