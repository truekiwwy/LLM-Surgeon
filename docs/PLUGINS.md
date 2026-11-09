# Plugins : étendre le Surgery Engine

Le mot « plugin » a deux sens ici, et seulement deux :

1. **Un plugin du moteur** : une implémentation d'un extension point, écrite en Rust et
   compilée avec la bibliothèque `weight_surgeon`. Un seul extension point existe
   aujourd'hui : `ExtractionMethod`, les méthodes de calcul d'une direction de `extract`.
2. **Le plugin Claude Code** : le dossier [`plugin/`](../plugin), qui branche le serveur
   `surgeon mcp` et un skill de démarche sûre dans Claude Code.

Le reste du moteur (noyaux, patch, formats, LoRA, mesure) est interne : voir le tableau de
[docs/ARCHITECTURE.md](ARCHITECTURE.md#composants--rôle-et-extensibilité).

## Modèle : compilation statique

Un plugin du moteur est du code Rust dans votre propre crate, qui dépend de
`weight-surgeon`. Il n'y a ni chargement dynamique de bibliothèque, ni code lu depuis un
fichier, ni découverte automatique : votre programme construit un `Registry`, y enregistre
sa méthode, et appelle l'extraction.

```toml
[dependencies]
weight-surgeon = { git = "https://github.com/truekiwwy/LLM-Surgeon", rev = "<commit>" }
serde_json = "1"
```

Le binaire `surgeon` lui-même n'exécute jamais de code tiers : son CLI (`--method`) et son
serveur MCP (`method`) n'acceptent que les méthodes intégrées `mean_diff`, `svd` et `leace`.

## Versionnage

- `weight_surgeon::plugin::API_VERSION` vaut `"v0"` et `surgeon plugins list` l'affiche.
- `v0` est **instable** : une version du moteur peut changer le trait, `MethodInfo`,
  `ExtractionOutput` ou les règles du moteur sans période de dépréciation. Épinglez un
  commit (`rev`) et relisez ce document à chaque mise à jour.
- Une API `v1` stable supposera au moins un registre injectable dans le front-end (voir la
  feuille de route de l'architecture).

## Pas à pas : `median_diff`

L'exemple complet est [`examples/plugin_extraction_example.rs`](../examples/plugin_extraction_example.rs).
Il calcule `normalize(médiane(pos) − médiane(neg))` coordonnée par coordonnée, l'enregistre
et lance une extraction réelle sur le mini-modèle Qwen2 de
[`examples/make_fake_qwen2.rs`](../examples/make_fake_qwen2.rs). `tests/plugin_api.rs`
l'exécute de bout en bout.

```bash
cargo run --example plugin_extraction_example -- ./median-demo
```

### 1. Implémenter le trait

```rust
use std::sync::atomic::AtomicBool;
use weight_surgeon::plugin::{
    ActivationSet, ExtractContext, ExtractionMethod, ExtractionOutput, MethodInfo, PluginError,
};

pub struct MedianDiff;

impl ExtractionMethod for MedianDiff {
    fn id(&self) -> &'static str {
        "median_diff"
    }

    fn describe(&self) -> MethodInfo {
        MethodInfo {
            title: "Différence des médianes",
            summary: "normalize(médiane(pos) − médiane(neg)), coordonnée par coordonnée.",
            params: vec![],
            requires_paired: false,
            sidecars: vec![],
        }
    }

    fn extract(
        &self,
        ctx: &ExtractContext,
        acts: &ActivationSet,
        _cancel: &AtomicBool,
    ) -> Result<ExtractionOutput, PluginError> {
        let diff: Vec<f64> = (0..ctx.hidden_size)
            .map(|j| median(acts.pos, j) - median(acts.neg, j))
            .collect();
        let norm = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !(norm.is_finite() && norm > 1e-12) {
            return Err(PluginError::Failed("médianes confondues".into()));
        }
        Ok(ExtractionOutput::new(diff.iter().map(|x| x / norm).collect()))
    }
}
```

`validate(&self, params)` a une implémentation par défaut qui accepte tout ; redéfinissez-la
dès que la méthode lit `ctx.params`, pour refuser une valeur avant la passe avant.

### 2. Enregistrer et lancer

```rust
let mut registry = Registry::with_builtin();
registry.register(Box::new(MedianDiff))?;
let method = registry.get("median_diff").ok_or("absent")?;
let report = run_extraction_plugin(
    &request,             // ExtractRequest : modèle, prompts, couche, sortie...
    method,
    &serde_json::Value::Null, // params de la méthode (null : ses valeurs par défaut)
    &SystemProbe::detect(),   // budget RAM
    &mut observer,            // progression ; ControlFlow::Break arrête entre deux étapes
    &cancel,                  // AtomicBool transmis à la méthode
)?;
```

`report` est l'`ExtractReport` de `surgeon extract` : `method` vaut `"median_diff"`, et les
fichiers `<out>` et `<out>.stats.json` portent `method = median_diff` et les diagnostics.

## Contrat

Ce que le moteur fait pour la méthode :

- il valide la requête, puis appelle `validate(params)`, vérifie `requires_paired` et
  refuse un fichier annexe déclaré déjà présent sans `overwrite`, **avant** toute passe
  avant ;
- il calcule les activations et les passe en `ActivationSet` (lignes de `hidden_size`
  valeurs f64, positifs puis négatifs) ;
- il contrôle la direction rendue (taille, valeurs finies, norme 1 à 1e-6 près), calcule
  les statistiques de séparation et écrit tous les fichiers ensemble, atomiquement ; une
  erreur n'écrit rien.

Ce que la méthode doit faire :

- rendre une direction **unitaire** en f64 (le fichier la stocke en f32) ;
- être déterministe : même entrée, même sortie à l'octet près ;
- rendre `PluginError::Cancelled` peu après que `cancel` passe à `true` si son calcul est
  long ;
- ne déclarer que les fichiers annexes qu'elle écrit (`MethodInfo::sidecars`).

| `PluginError` | Code de sortie | `error.kind` |
| --- | --- | --- |
| `InvalidParams` | 2 | `invalid_input` |
| `Cancelled` | 1 | `cancelled` |
| `Failed`, `InvalidOutput` | 7 | `extract_failed` |
| `Extract(e)` | celui de `e` | celui de `e` |

## Conventions d'identifiants

- Identifiant de méthode : `[a-z][a-z0-9_]{1,31}`. `Registry::register` refuse un
  identifiant invalide (`InvalidId`), celui d'une méthode intégrée (`ReservedId` :
  `mean_diff`, `svd`, `leace`) ou un doublon (`DuplicateId`) ; le registre reste inchangé.
  Préfixez vos identifiants par votre projet (`acme_median`) pour éviter les collisions.
- Suffixe de fichier annexe : même règle ; le fichier est `<out>.<suffixe>.safetensors`.
- Métadonnées (`ExtractionOutput::metadata`) : chaînes ajoutées à l'en-tête de `<out>`.
  Une clé que le moteur écrit déjà (`model`, `model_type`, `layer`, `site`, `pooling`,
  `chat_template`, `n_pos`, `n_neg`, `raw_diff_norm`, `d_prime`, `threshold_accuracy`,
  `pos_mean_proj`, `neg_mean_proj`, `cos_mean_pos_neg`, `dtype_used`, `surgeon_version`) ou
  la clé `method` sont refusées (code 7).
- Diagnostics : un objet JSON (`Diagnostics::Json`) ; le moteur y fixe `method`.
- Ordre de `Registry::list` : méthodes intégrées, puis méthodes enregistrées par identifiant.

## Sécurité d'un plugin statique

- Un plugin compilé est du code de votre programme : il a tous ses droits. Le moteur ne lui
  passe aucun chemin ni aucun descripteur de fichier, mais **ne le confine pas** : rien
  n'empêche son code d'ouvrir un fichier ou une connexion. Relisez-le et épinglez-le comme
  toute dépendance.
- Ce que le moteur garantit quoi qu'il rende : rien n'est écrit hors de `<out>`, de son
  compagnon et des fichiers annexes déclarés ; rien n'est écrasé sans `overwrite` ; une
  sortie invalide (taille, norme, valeur non finie, nom de tenseur vide ou en double,
  fichier annexe non déclaré, clé de métadonnées du moteur) est refusée sans rien écrire.
- Le binaire `surgeon` et son serveur MCP n'exécutent aucun plugin : un agent ne peut pas
  en charger un par un argument d'outil.

## Plugin Claude Code (`plugin/`)

Format suivi : la référence des plugins Claude Code (code.claude.com, consultée le
2026-11-09). `claude plugin validate ./plugin` répond « Validation passed » avec Claude Code
2.1.321.

```text
plugin/
├── .claude-plugin/plugin.json               nom llm-surgeon, version du crate, userConfig allow_dir
├── .mcp.json                                serveur "surgeon" : surgeon mcp --allow-dir ${user_config.allow_dir}
└── skills/safe-surgery-workflow/SKILL.md    démarche sûre imposée à l'agent
```

- **Prérequis** : le binaire `surgeon` dans le `PATH` (par exemple `cargo install --path .`
  depuis la racine du dépôt). Le plugin ne le fournit pas.
- **Dossier autorisé** : Claude Code demande `allow_dir` (type `directory`, obligatoire) à
  l'activation du plugin ; aucun chemin n'est écrit dans le dépôt. Le serveur ne lit et
  n'écrit que dans ce dossier, et la sauvegarde reste imposée (`--allow-no-backup` n'est
  pas passé).
- **Essai local** : `claude --plugin-dir ./plugin`.
- **Skill `safe-surgery-workflow`** : `surgeon_check` d'abord, travail sur une copie, dump
  de référence avant le patch, `dry_run` avant tout `surgeon_apply` ou `surgeon_lora`
  `merge`, sauvegarde jamais désactivée, mesure `surgeon_bench` après le patch, aucun
  résultat annoncé sans chiffre mesuré.
- **Publication dans l'annuaire Anthropic** : le validateur demande une licence (fichier
  `LICENSE` ou champ `license`) que le dépôt n'a pas encore, et conseille une icône.
- `tests/claude_plugin.rs` vérifie les JSON, les champs requis, les options strictes de
  `userConfig`, que `command` est le binaire du crate et que ses arguments sont des options
  réelles de `surgeon mcp`, que les outils cités par le skill existent, et l'absence de
  secret et de chemin personnel.

## Proposition : plugins hors processus (non implémenté)

> Statut : **proposition de conception**. Aucune ligne de ce protocole n'existe dans le code.

Objectif : utiliser une méthode tierce depuis le binaire `surgeon` lui-même (CLI et MCP),
sans recompiler, écrite dans n'importe quel langage.

### Exécutable et activation

- Un plugin est un exécutable nommé `surgeon-plugin-<nom>`, `<nom>` suivant la règle des
  identifiants.
- **Refusé par défaut.** Aucune découverte dans le `PATH`, aucun chargement depuis un
  fichier de configuration du modèle ou depuis un argument MCP. L'opérateur l'autorise
  explicitement, avec un chemin absolu et l'empreinte attendue :
  `--allow-plugin /opt/surgeon/surgeon-plugin-median#sha256=<hex>` (option de `surgeon
  extract` et de `surgeon mcp`). Sans cette option, `--method median` reste une méthode
  inconnue (code 2).
- Le moteur vérifie l'empreinte avant chaque lancement et refuse un exécutable modifié.
- Les méthodes ainsi chargées apparaissent dans `surgeon plugins list` avec
  `origin: "external"`.

### Transport et poignée de main

- JSON-RPC 2.0 sur stdio, un message par ligne, lignes limitées à 16 Mio comme le serveur
  MCP ; stderr du plugin recopié dans le journal, jamais sur stdout.
- Première requête `initialize` :
  `{"protocol_versions": ["surgeon-plugin/0"], "engine_version": "0.1.0", "api_version": "v0"}`.
  Réponse : `{"protocol_version": "surgeon-plugin/0", "plugin": {"name", "version"},
  "capabilities": {"extraction_methods": [<MethodInfo + id>]}}`. Le moteur choisit la plus
  haute version commune ; sans version commune, il arrête le processus et rend une erreur.
- Une capacité non annoncée n'est jamais appelée.

### Appels

| Méthode | Paramètres | Résultat |
| --- | --- | --- |
| `extraction/validate` | `id`, `params` | `{}` ou erreur `invalid_params` |
| `extraction/extract` | `id`, `ctx` (`hidden_size`, `layer`, `site`, `pooling`, `params`), `acts` (`pos`, `neg` : f64 petit-boutiste en base64, avec leur forme) | `direction` (f64 base64), `diagnostics`, `metadata`, `sidecars` (tenseurs f32 base64), `warnings` |
| `$/cancel` (notification) | aucun | le plugin doit répondre `cancelled` rapidement |
| `shutdown` | aucun | fin propre, puis fermeture de stdin |

La réponse passe par les mêmes contrôles que pour un plugin statique (taille, norme,
fichiers annexes déclarés, clés réservées) avant toute écriture.

### Confinement

- Le plugin ne reçoit aucun chemin : les activations arrivent par stdin, la direction
  repart par stdout, et le moteur écrit seul les fichiers, dans les dossiers déjà autorisés
  (sandbox de chemins du serveur MCP).
- Processus lancé dans un dossier temporaire vide, avec un environnement vidé (seules les
  variables système indispensables), délais bornés (poignée de main, chaque appel), arrêt
  forcé au dépassement, taille des messages bornée.

### Risques

| Risque | Ce que la proposition fait | Ce qui reste |
| --- | --- | --- |
| Exécution de code arbitraire | Opt-in de l'opérateur, chemin absolu, empreinte SHA-256, jamais activable par un agent via MCP | Le processus a les droits de l'utilisateur : ce n'est pas un bac à sable. Pour du code non relu, lancer `surgeon` sous un compte restreint ou dans un conteneur. |
| Fuite de données | Aucun chemin transmis | Les activations reflètent les prompts : un plugin malveillant peut les envoyer sur le réseau. |
| Déni de service | Délais, taille des messages, arrêt forcé | La mémoire du processus n'est pas bornée de façon portable. |
| Sortie corrompue | Contrôles du moteur avant écriture, écriture atomique | aucun |
| Chaîne d'approvisionnement | Empreinte épinglée | La mise à jour d'un plugin demande une nouvelle empreinte, donc une action de l'opérateur. |

Ce qu'il faudrait coder : un adaptateur `ExternalMethod` qui implémente `ExtractionMethod`
en parlant à l'exécutable, une `Origin::External`, l'option `--allow-plugin` sur `extract`
et `mcp`, et un registre injectable dans le front-end (voir l'architecture).
