# Dépendances de `crates/surgeon-gui`

Le GUI a son propre `Cargo.lock` : sa pile graphique n'entre jamais dans celui du moteur, que
`tests/deps_purity.rs` contrôle. Le moteur (`weight-surgeon`) n'a gagné aucune dépendance
avec le studio (`src/events.rs` n'utilise que `serde`, `serde_json` et `schemars`, déjà là).

| Crate | Pourquoi | Coût |
| --- | --- | --- |
| `eframe` (glow, accesskit) | Fenêtre, rendu OpenGL, accessibilité | Pile graphique, déjà présente |
| `clap` | Ligne de commande (`GRAPH`, `--workspace`, `--self-test`) | Déjà présente |
| `serde`, `serde_json` | Manifeste, `params.json`, `status.json`, journal | Déjà présentes |
| `weight-surgeon` | Le moteur, appelé en bibliothèque | — |
| `schemars` 0.8 | Schéma JSON du manifeste (`docs/schemas/project.surgeon.schema.json`) | Aucune nouvelle crate : déjà dans ce lockfile par le moteur, même version |
| `sysinfo` 0.40 (`disk`, `system`) | Espace libre du volume avant une copie de travail ; PID vivant derrière `project.lock` | Aucune nouvelle crate : déjà dans ce lockfile par le moteur ; seule la fonctionnalité `disk` s'ajoute |

Dépendances de test :

| Crate | Pourquoi |
| --- | --- |
| `tempfile` | Espaces de travail jetables des tests |
| `tokenizers` (mêmes fonctionnalités que le moteur) | Le mini Qwen2 de `examples/make_fake_qwen2.rs`, pour un job réel du moteur en test |

Ni TLS, ni HTTP, ni trousseau, ni télémétrie : le studio ne fait aucun accès réseau.
