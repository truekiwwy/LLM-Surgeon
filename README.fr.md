<a id="top"></a>
# LLM Surgeon (`surgeon`)

![version](https://img.shields.io/badge/version-0.1.0-blue)
![rust](https://img.shields.io/badge/rust-2021_edition-orange)
![pure rust](https://img.shields.io/badge/pure_Rust-no_C_build_dependency-success)
![MCP](https://img.shields.io/badge/MCP-stdio_server-purple)
![platforms](https://img.shields.io/badge/platforms-Windows_tested_%7C_Linux_and_macOS_compile--checked-lightgrey)

`surgeon` est un outil en ligne de commande et un serveur MCP qui inspecte et modifie les poids MLP de modèles de langage locaux (SafeTensors, et GGUF quantifié en place), sur le CPU, en Rust pur. Au lieu de ré-entraîner un modèle, il ajoute ou retranche un vecteur de direction, fusionne une LoRA ou applique un patch de rang faible, puis mesure ce qui a changé.

[English version](README.md)

Les messages du terminal sont en français. Toutes les sorties de ce fichier sont de vraies exécutions, copiées sans couleurs (`NO_COLOR=1`). Sauf mention contraire, un bloc a été exécuté sur un minuscule faux modèle Qwen2 (4 couches, taille cachée 16) généré par `examples/make_fake_qwen2.rs` : c'est un « exemple sur mini-modèle factice », pas un vrai modèle.

<a id="toc"></a>
## Table des matières

1. [Concept et cas d'usage](#concept)
2. [Démarrage rapide](#quick-start)
3. [Guide avancé et référence technique](#reference)
4. [Agents et MCP](#mcp)
5. [FAQ, sécurité et dépannage](#faq)

<a id="concept"></a>
## 1. Concept et cas d'usage

<a id="what-is-surgery"></a>
### 1.1 Qu'est-ce que la chirurgie de modèle ?

Un modèle de langage est un très grand ensemble de nombres, les poids. L'entraînement les ajuste lentement avec beaucoup de données. La chirurgie de poids en modifie quelques-uns directement, en une passe, pour pousser le comportement du modèle dans une direction choisie.

> **Pour les débutants**
> Pensez à une table de mixage avec des curseurs. Ré-entraîner un modèle, c'est demander au groupe de réapprendre le morceau. La chirurgie, c'est pousser un curseur d'une quantité mesurée. Ici le « curseur » est un vecteur (une liste de nombres) et la quantité dont vous le poussez est un seul nombre appelé `alpha`.

Ce qui est faux dans cette analogie, volontairement :

- Un modèle n'a pas de curseur étiqueté par concept. Une direction est estimée à partir d'exemples : elle peut être bruitée et mélange en général plusieurs choses.
- Les mêmes poids servent à de nombreux comportements. Bouger une chose peut en bouger d'autres, et l'effet n'est ni garanti linéaire ni garanti petit.
- Le vecteur vit dans un espace de centaines ou de milliers de dimensions (896 pour le modèle de 0,5 G de paramètres du démarrage rapide), pas sur un seul axe.
- Une vraie table se remet exactement en place. Ici « retrancher puis ajouter » n'est pas exact au bit près à cause des arrondis ; le retour en arrière, c'est la sauvegarde.

> **Pour les experts**
> L'objet modifiable est une matrice `W` : une projection MLP (`gate_proj`, `up_proj`, `down_proj`) ou, dans un modèle SafeTensors, la sortie d'attention `o_proj` ou les plongements d'entrée `embed_tokens`. `surgeon` diffuse un vecteur sur ses lignes ou ses colonnes (`W <- W - alpha v`), retire une direction des matrices qui écrivent dans le flux résiduel (`project_out`, effaceur LEACE ou simple ablation directionnelle), ajoute un delta de rang faible (`.surgery`, LoRA) ou arrondit de nouveau un bloc GGUF quantifié. Tout est une modification déterministe et auditable d'octets stockés ; rien n'est entraîné.

<a id="why-not-retrain"></a>
### 1.2 Pourquoi ne pas simplement ré-entraîner ?

Les deux approches ont leur place. Ce tableau est qualitatif : il ne contient aucun chiffre mesuré.

| | Fine-tuning classique | Chirurgie de poids avec `surgeon` |
| --- | --- | --- |
| Données nécessaires | Un jeu d'entraînement étiqueté ou trié | Quelques prompts contrastés pour trouver une direction, ou aucun si vous avez déjà un vecteur, une LoRA ou deux checkpoints |
| Calcul | Entraînement par gradient, souvent sur GPU | Passes avant sur quelques prompts, puis une écriture ; CPU seul |
| Réversibilité | Garder l'ancien checkpoint | Sauvegarde vérifiée par SHA-256, rollback automatique en cas d'échec |
| Précision | Apprend un comportement large et riche | Modification grossière et surtout linéaire ; des effets de bord sont possibles |
| Risques principaux | Oubli, coût, surapprentissage | Changement de comportement non voulu, directions faibles ou bruitées, modifications perdues par la quantification |

<a id="use-cases"></a>
### 1.3 Cas d'usage

- Essayer des idées de pilotage d'activations sous forme de patch de poids permanent (`extract`, puis `apply`), et comparer le résultat au modèle d'origine (`bench`).
- Transformer la différence entre un checkpoint de base et sa version affinée en patch compact de rang faible (`diff`) ou en adaptateur LoRA PEFT (`lora extract`).
- Fusionner un adaptateur LoRA PEFT dans les poids de base (`lora merge`).
- Patcher en place un GGUF quantifié, avec un rapport de fidélité par tenseur (`apply --gguf`).
- Laisser un agent IA faire tout cela via des outils MCP typés, confinés à des dossiers choisis et toujours sauvegardés (`surgeon mcp`).
- Comparer des méthodes d'extraction de direction (`mean_diff`, `svd`, `leace`) et brancher la vôtre par l'API d'extension Rust (`plugins list`).

<a id="non-goals"></a>
### 1.4 Ce que l'outil ne fait pas

- Il n'entraîne ni n'affine rien, et il ne télécharge pas de modèles.
- `patch.json` ne modifie que les projections MLP. `diff` et `lora` couvrent aussi les tenseurs d'attention.
- `extract`, `autotune`, `bench dump` et l'onglet de prévisualisation ne gèrent que Qwen2 / Qwen2.5 (`model_type: "qwen2"`), sur le CPU.
- Il ne fournit ni serveur de chat ni API d'inférence. `eval` et `bench ppl|eval` pilotent un llama.cpp externe que vous fournissez.
- Le format `.surgery` s'applique uniquement depuis le CLI, pas par MCP, et le tableau de bord n'est pas exposé par MCP non plus.
- Il ne peut pas garantir qu'une modification fait ce que vous voulez ni qu'elle garde le modèle sûr : il faut mesurer.

<a id="glossary"></a>
### 1.5 Glossaire

| Terme | Sens ici |
| --- | --- |
| Poids | Un nombre appris, stocké dans le fichier du modèle. |
| Activation | Le vecteur intermédiaire qu'une couche produit pendant que le modèle lit un prompt. |
| Couche | Un bloc répété du réseau ; les couches sont numérotées à partir de 0. |
| Vecteur de steering | Une direction de l'espace des activations (par exemple « positif moins négatif ») estimée à partir de prompts. |
| Quantification | Stocker les poids sur moins de bits (Q8_0, Q4_K...) pour économiser la mémoire ; les valeurs tombent sur une grille grossière. |
| SafeTensors | Un format de fichier de tenseurs simple et sûr, utilisé par les modèles Hugging Face. |
| GGUF | Le format de modèle en un seul fichier de llama.cpp, souvent quantifié. |
| LoRA | Un adaptateur de rang faible `B A` ajouté à une matrice de poids au lieu de ré-entraîner. |
| Perplexité | À quel point un modèle est surpris par un texte ; plus bas est mieux, comparée avant et après un patch. |
| Divergence KL | Une distance entre deux distributions de probabilité ; ici entre les distributions du token suivant avant et après un patch. |
| Pass@1 | Part des questions résolues par une réponse gloutonne unique. |

<a id="quick-start"></a>
## 2. Démarrage rapide

<a id="install"></a>
### 2.1 Installation

1. Installez la chaîne Rust depuis <https://rustup.rs> (testé avec Rust 1.97). Sous Windows il faut aussi les MSVC Build Tools ; sous Linux un éditeur de liens C (`cc`). Aucune dépendance ne compile de C : `tests/deps_purity.rs` échoue si une crate connue pour compiler du C apparaît dans `Cargo.lock`.
2. Récupérez le code et installez le binaire.

```bash
# récupère les sources
git clone https://github.com/truekiwwy/LLM-Surgeon.git
cd LLM-Surgeon
# installe le binaire dans $HOME/.cargo/bin (%USERPROFILE%\.cargo\bin sous Windows)
cargo install --path .
# vérifie qu'il s'exécute
surgeon --version
```

```text
surgeon 0.1.0
```

Si vous préférez ne pas installer, compilez avec `cargo build --release` et utilisez `target/release/surgeon` à la place de `surgeon`.

<a id="path-a"></a>
### 2.2 Chemin A : sans téléchargement, mini-modèle factice

Tout ce qui suit a été exécuté pour de vrai sur le faux modèle. Votre quantité de RAM et l'horodatage de la sauvegarde seront différents.

**Étape 1. Générer le faux modèle.** Ce que ça fait : écrit un Qwen2 de 55 Ko (4 couches, taille cachée 16) et deux fichiers de prompts de six phrases chacun.

```bash
# depuis le dossier LLM-Surgeon
cargo run --release --example make_fake_qwen2 -- demo
cd demo
```

```text
demo\qwen2-mini
demo\pos.json
demo\neg.json
```

En cas d'échec : lancez la commande depuis la racine du dépôt et vérifiez que `cargo --version` fonctionne.

**Étape 2. Vérifier que la machine peut le charger.**

```bash
# le modèle tient-il en mémoire ?
surgeon check --model qwen2-mini
```

```text
[CHECK]    weight-surgeon
────────────────────────────────────────────────────────────
  modèle        qwen2-mini
  taille        54.2 KiB
  RAM           5.01 GiB libres
  GPU           aucun
  besoin        RAM 2.00 GiB · VRAM n/a
[VERDICT]  FITS IN RAM ONLY
  ! NVML indisponible, GPU NVIDIA ignorés : the current user does not have permission to perform this operation
  ! aucun GPU détecté ; chargement en RAM (CPU), plus lent
```

Ce que ça veut dire : le modèle tient en RAM ; aucun GPU utilisable n'a été trouvé (l'avertissement NVML est sans danger). Le besoin de 2 GiB est la marge de sécurité, pas le modèle. Si le verdict est `INSUFFICIENT`, la commande sort avec le code 1 : voir [le budget mémoire](#hardware-budget).

**Étape 3. Inspecter les tenseurs MLP.**

```bash
# liste les tenseurs MLP des couches 0 et 1, avec statistiques
surgeon inspect --model qwen2-mini --layers 0-1 --show-stats
```

```text
█▀█ █ █ ▀█▀  W E I G H T
█ █ ▀▄▀  █   S U R G E O N
[INSPECT]  6 tenseurs sélectionnés
────────────────────────────────────────────────────────────
COUCHE   PROJ        DTYPE   SHAPE     TAILLE    MIN       MAX      MOYENNE   L2    
[L0]     gate_proj   f32     32 × 16   2.0 KiB   -0.3996   0.3994   0.0184    5.0572
[L0]     up_proj     f32     32 × 16   2.0 KiB   -0.3982   0.3991   -0.0120   5.4108
[L0]     down_proj   f32     16 × 32   2.0 KiB   -0.3996   0.3949   -0.0016   4.9931
[L1]     gate_proj   f32     32 × 16   2.0 KiB   -0.3998   0.3992   -0.0098   5.0053
[L1]     up_proj     f32     32 × 16   2.0 KiB   -0.3997   0.3984   0.0010    5.2173
[L1]     down_proj   f32     16 × 32   2.0 KiB   -0.3983   0.3968   -0.0043   5.3337
```

Ce que ça veut dire : trois projections par couche ; `gate_proj` et `up_proj` sont `[intermédiaire, caché]` = 32 × 16, `down_proj` est `[caché, intermédiaire]` = 16 × 32. La même commande avec `--json` affiche une enveloppe lisible par machine ([détails](#envelope)) :

```bash
# même liste dans une seule enveloppe JSON
surgeon --json inspect --model qwen2-mini --layers 2 --proj down
```

```text
{"ok":true,"command":"inspect","exit_code":0,"data":{"shards":["qwen2-mini\\model.safetensors"],"tensors":[{"dtype":"F32","name":"model.layers.2.mlp.down_proj.weight","shape":[16,32],"shard":"qwen2-mini\\model.safetensors","stats":null}]},"error":null,"warnings":[]}
```

**Étape 4. Extraire un vecteur de steering.** Ce que ça fait : exécute le modèle sur les deux jeux de prompts et écrit `normalize(mean(pos) - mean(neg))` mesuré à la couche 2.

```bash
# estime une direction à la couche 2 à partir des deux fichiers de prompts
surgeon extract --model qwen2-mini --pos-prompts pos.json --neg-prompts neg.json --layer 2 --out direction.safetensors
```

```text
[EXTRACT]  Qwen2 · couche 2 · resid_post · pooling last  qwen2-mini
[LOAD]     poids chargés 37/37 tenseurs · f32 · 34.9 KiB
[FORWARD]  pos 6/6 · neg 6/6
[VERDICT]  d′ 2.23 · accuracy 92% · ‖Δ‖ 3.378
[WRITE]    direction.safetensors
  ! seulement 6 prompts positifs (au moins 8 recommandés)
  ! seulement 6 prompts négatifs (au moins 8 recommandés)
```

Ce que ça veut dire : `d′` et `accuracy` disent à quel point la direction sépare les deux jeux de prompts (plus haut est mieux) ; les avertissements disent que six prompts par jeu, c'est trop peu pour une direction fiable (suffisant pour une démo). Deux fichiers ont été écrits : `direction.safetensors` (un tenseur f32 1D nommé `direction`, plus des métadonnées) et `direction.safetensors.stats.json` (la projection de chaque prompt). En cas d'échec avec le code 7, la config, le tokenizer ou les poids sont invalides ou la direction est dégénérée ; avec le code 2, un argument est faux (un `--out` existant demande `--overwrite`).

**Étape 5. Écrire le patch.** Enregistrez ceci sous `patch.json`. Il retranche la moitié du vecteur de `down_proj` aux couches 1 et 2. Le vecteur a la taille de la dimension cachée, qui indexe les lignes de `down_proj`, d'où `"axis": "cols"` ([pourquoi](#patch-format)).

```json
{
  "schema_version": 1,
  "operations": [
    { "id": "steer-down", "layers": "1-2", "projections": ["down_proj"],
      "operation": "subtract", "vector_file": "direction.safetensors",
      "vector_tensor": "direction", "alpha": 0.5, "axis": "cols" }
  ]
}
```

**Étape 6. Essai à blanc, puis application.** `--dry-run` valide tout et liste les tenseurs qui changeraient, sans rien écrire. Gardez une copie intacte pour comparer à l'étape 7, et créez le dossier de sauvegarde (il doit exister).

```bash
# garde une copie intacte du modèle pour la comparaison plus bas
cp -r qwen2-mini qwen2-orig
# essai à blanc : rien n'est écrit
surgeon apply --model qwen2-mini --patch patch.json --dry-run
# application réelle, avec la sauvegarde dans un dossier existant
mkdir backups
surgeon apply --model qwen2-mini --patch patch.json --backup-dir backups
```

```text
[APPLY]    patch.json · 1 opérations
[DRY-RUN]  steer-down · 2 tenseurs · vecteur de 16
           model.layers.1.mlp.down_proj.weight
           model.layers.2.mlp.down_proj.weight
```

```text
[APPLY]    patch.json · 1 opérations
[BACKUP]   backups\model.safetensors.bak-1794263372  ✓ sha256 vérifié
[SHIFT]    steer-down terminée · Δmax 0.2570 · 2 tenseurs
[COMMIT]   modèle patché · 0.0 s
```

Ce que ça veut dire : une sauvegarde du fichier a été copiée et son SHA-256 vérifié avant toute écriture ; deux tenseurs ont été décalés d'au plus 0,257 ; le commit a réussi. Si le dossier de sauvegarde n'existe pas, la commande s'arrête avec le code 3 et le modèle n'est pas touché (`erreur : backup : dossier de backup inexistant : nope`). Un patch mal formé donne le code 2.

**Étape 7. Vérifier l'effet.** Exportez les logits des deux modèles sur les mêmes prompts et comparez-les.

```bash
# logits du modèle d'origine et du modèle patché sur les mêmes prompts, puis comparaison
surgeon bench dump --model qwen2-orig --prompts pos.json --out before.safetensors
surgeon bench dump --model qwen2-mini --prompts pos.json --out after.safetensors
surgeon bench logits --before before.safetensors --after after.safetensors
```

```text
[BENCH]    dump · 6 prompts · 24 positions · vocab 64 · f32
[WRITE]    before.safetensors
[BENCH]    dump · 6 prompts · 24 positions · vocab 64 · f32
[WRITE]    after.safetensors
[BENCH]    logits · 24 positions · vocab 64
[KL]       moyenne 0.0492 · max 0.2433 @ 18 · p95 0.1731 · cos(p,q) 0.9565
[TOP-K]    top-1 changé 45.8 % · top-5 (ensemble) 87.5 % · recouvrement 0.70
  pires prompts : #4 0.1118 · #5 0.0692 · #1 0.0517 · #0 0.0229 · #3 0.0225
[VERDICT]  aucun seuil fourni
```

Ce que ça veut dire : le patch a mesurablement changé les distributions du token suivant (KL moyenne 0,049 ; le token top-1 a changé sur 45,8 % des positions). L'outil ne peut pas vous dire si ce changement est celui que vous vouliez : c'est à vous d'en juger. Ajoutez `--warn-kl 0.05 --fail-on-warn` pour en faire une barrière (code de sortie 9).

**Étape 8. Revenir en arrière.** Il n'y a pas de commande `undo` : restaurez la sauvegarde et vérifiez l'empreinte.

```bash
# remet le fichier sauvegardé en place
cp backups/model.safetensors.bak-* qwen2-mini/model.safetensors
# les deux empreintes doivent être identiques
sha256sum qwen2-orig/model.safetensors qwen2-mini/model.safetensors
```

```text
2f4f3e1b121db1c97307887c89c76569ec7ce3f8723cbbf8802fd63050373f88 *qwen2-orig/model.safetensors
2f4f3e1b121db1c97307887c89c76569ec7ce3f8723cbbf8802fd63050373f88 *qwen2-mini/model.safetensors
```

Sous Windows PowerShell : `Copy-Item backups\model.safetensors.bak-* qwen2-mini\model.safetensors`, puis `Get-FileHash` sur les deux fichiers.

<a id="path-b"></a>
### 2.3 Chemin B : un vrai petit modèle

Utilisez `Qwen/Qwen2.5-0.5B-Instruct` (`model_type: "qwen2"`, 24 couches, taille cachée 896, environ 0,9 GiB en bf16). La page officielle est <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct> ; à la date de rédaction ses métadonnées indiquent la licence Apache-2.0. Vérifiez toujours la licence d'un modèle avant de le modifier ou de le partager.

**Étape 1. Télécharger les trois fichiers** (les commandes ci-dessous ont été exécutées et les fichiers vérifiés le 2026-11-09) :

```bash
# les trois fichiers du modèle, depuis le dépôt officiel
mkdir -p qwen2.5-0.5b-instruct
cd qwen2.5-0.5b-instruct
curl -fLO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/config.json
curl -fLO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/tokenizer.json
curl -fLO https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/resolve/main/model.safetensors
cd ..
```

Si un téléchargement échoue, relancez-le (`-f` fait échouer `curl` sur une erreur HTTP au lieu d'enregistrer une page d'erreur).

**Étape 2. Vérifier et inspecter.** Sortie réelle sur un portable de 16 Go sans GPU (le chiffre après « RAM » dépend de votre machine) :

```bash
# budget mémoire, puis tenseurs MLP de la couche 12
surgeon check --model qwen2.5-0.5b-instruct
surgeon inspect --model qwen2.5-0.5b-instruct --layers 12 --show-stats
```

```text
[CHECK]    weight-surgeon
────────────────────────────────────────────────────────────
  modèle        qwen2.5-0.5b-instruct
  taille        942.3 MiB
  RAM           4.51 GiB libres
  GPU           aucun
  besoin        RAM 3.06 GiB · VRAM n/a
[VERDICT]  FITS IN RAM ONLY
  ! NVML indisponible, GPU NVIDIA ignorés : the current user does not have permission to perform this operation
  ! aucun GPU détecté ; chargement en RAM (CPU), plus lent
```

```text
█▀█ █ █ ▀█▀  W E I G H T
█ █ ▀▄▀  █   S U R G E O N
[INSPECT]  3 tenseurs sélectionnés
────────────────────────────────────────────────────────────
COUCHE   PROJ        DTYPE   SHAPE        TAILLE    MIN       MAX      MOYENNE   L2     
[L12]    gate_proj   bf16    4864 × 896   8.3 MiB   -0.2832   0.3359   0.0000    41.8239
[L12]    up_proj     bf16    4864 × 896   8.3 MiB   -0.3848   0.2539   0.0000    39.9329
[L12]    down_proj   bf16    896 × 4864   8.3 MiB   -0.4395   0.3652   0.0000    38.0406
```

**Étape 3. Extraire une direction.** Enregistrez deux fichiers de huit phrases courtes, par exemple `pos.json` (`["I love this, it is wonderful.", "What a fantastic day!", ...]`) et `neg.json` (`["I hate this, it is awful.", "What a terrible day!", ...]`), puis lancez :

```bash
# direction à partir des fichiers de prompts, avec le gabarit de chat du modèle
surgeon extract --model qwen2.5-0.5b-instruct --pos-prompts pos.json --neg-prompts neg.json --layer 12 --chat-template chatml --out sentiment.safetensors
```

```text
[EXTRACT]  Qwen2 · couche 12 · resid_post · pooling last  qwen2.5-0.5b-instruct
[LOAD]     poids chargés 157/157 tenseurs · f32 · 1.23 GiB
[FORWARD]  pos 8/8 · neg 8/8
[VERDICT]  d′ 6.62 · accuracy 100% · ‖Δ‖ 2.013
[WRITE]    sentiment.safetensors
```

Cela a pris environ deux secondes. Si `extract` s'arrête faute de mémoire (code de sortie 1), utilisez `--dtype bf16` (moitié moins de mémoire). Un `d′` aussi élevé sur huit paires décrit ces prompts-là, ce n'est pas une preuve que la direction veut dire « sentiment » partout.

**Étape 4. Patcher une copie et mesurer.** Ne patchez jamais votre unique copie. Enregistrez `patch.json` (couches 10 à 14, `down_proj`, `alpha` 0,5, `axis` `cols` ; la longueur 896 du vecteur est le nombre de lignes de `down_proj`) :

```json
{
  "schema_version": 1,
  "description": "Subtract half of the sentiment direction from down_proj, layers 10 to 14",
  "operations": [
    { "id": "less-positive", "layers": "10-14", "projections": ["down_proj"],
      "operation": "subtract", "vector_file": "sentiment.safetensors",
      "vector_tensor": "direction", "alpha": 0.5, "axis": "cols" }
  ]
}
```

```bash
# patche une copie, jamais l'original : essai à blanc, application, puis mesure
cp -r qwen2.5-0.5b-instruct qwen2.5-0.5b-patched
mkdir backups
surgeon apply --model qwen2.5-0.5b-patched --patch patch.json --dry-run
surgeon apply --model qwen2.5-0.5b-patched --patch patch.json --backup-dir backups
surgeon bench dump --model qwen2.5-0.5b-instruct --prompts pos.json --out before.safetensors
surgeon bench dump --model qwen2.5-0.5b-patched --prompts pos.json --out after.safetensors
surgeon bench logits --before before.safetensors --after after.safetensors --warn-kl 0.05
```

```text
[APPLY]    patch.json · 1 opérations
[DRY-RUN]  less-positive · 5 tenseurs · vecteur de 896
           model.layers.10.mlp.down_proj.weight
           model.layers.11.mlp.down_proj.weight
           model.layers.12.mlp.down_proj.weight
           model.layers.13.mlp.down_proj.weight
           model.layers.14.mlp.down_proj.weight
```

```text
[APPLY]    patch.json · 1 opérations
[BACKUP]   backups\model.safetensors.bak-1794263715  ✓ sha256 vérifié
[SHIFT]    less-positive terminée · Δmax 0.0791 · 5 tenseurs
[COMMIT]   modèle patché · 0.2 s
```

```text
[BENCH]    logits · 55 positions · vocab 151936
[KL]       moyenne 0.4102 · max 5.3584 @ 20 · p95 2.2612 · cos(p,q) 0.8711
[TOP-K]    top-1 changé 32.7 % · top-5 (ensemble) 63.6 % · recouvrement 0.79
  pires prompts : #2 1.0624 · #6 0.9839 · #7 0.3831 · #3 0.2215 · #5 0.1868
[VERDICT]  avertissement : KL moyenne 0.4102 > 0.05
```

Ce que ça veut dire : ce patch a beaucoup changé le modèle sur ces prompts (KL 0,41 face à un seuil de 0,05), l'avertissement est donc attendu. Dans cette exécution `bench dump` a affiché `f32 ne tient pas en RAM : calcul en bf16 (précision réduite)` : avec peu de RAM libre le dump est calculé en bf16 ; sur une machine avec plus de mémoire libre il utilise f32. Ensuite, évaluez le comportement qui vous intéresse avec vos propres prompts et essayez un `alpha` plus petit. Pour annuler, restaurez la sauvegarde comme dans le chemin A.

> **Pour les débutants**
> Si vous vouliez seulement apprendre l'outil, le chemin A suffit. Le chemin B montre que les mêmes commandes passent à l'échelle d'un vrai modèle sans changement.

<a id="reference"></a>
## 3. Guide avancé et référence technique

<a id="envelope"></a>
### 3.1 Conventions : enveloppe JSON, codes de sortie, environnement

Options globales, valables avant ou après toute sous-commande : `--json` (sortie lisible par machine), `-v`, `-vv`, `-vvv` (verbosité), `-h`, `-V`. Avec `--json`, chaque commande affiche une enveloppe sur stdout ; c'est la même enveloppe que renvoient les outils MCP.

```text
{"ok": true, "command": "check", "exit_code": 0, "data": {...}, "error": null, "warnings": []}
```

| Champ | Sens |
| --- | --- |
| `ok` | `true` quand `exit_code` vaut 0. |
| `command` | Nom de la commande (`bench logits`, `lora merge`...). |
| `exit_code` | Le code de sortie du processus, renseigné aussi en cas d'échec. |
| `data` | Le résultat propre à la commande, `null` en cas d'échec. |
| `error` | `null`, ou `{"kind": "...", "message": "..."}` (plus `issues` pour des arguments invalides). |
| `warnings` | Messages non fatals. |

Exemple d'échec : `{"ok":false,"command":"apply","exit_code":2,"data":null,"error":{"message":"patch illisible missing.json : ...","kind":"invalid_patch"},"warnings":[]}`.

Codes de sortie :

| Code | Sens |
| --- | --- |
| 0 | Succès |
| 1 | Échec général : ressources insuffisantes, modèle illisible, binaire llama.cpp absent, Ctrl-C, `tui` sans terminal interactif |
| 2 | Entrée invalide : arguments, patch, prompts, chemins ; rien n'a été touché |
| 3 | Échec d'exécution ; le modèle est intact (jamais touché, ou restauré depuis la sauvegarde) |
| 4 | Critique : le rollback a échoué, ou pas de sauvegarde et le modèle est resté partiellement patché |
| 5 | `eval` : modèle en mauvaise santé (boucles, attentes non satisfaites) ; pas d'export |
| 6 | `eval` : l'export GGUF a échoué |
| 7 | `extract` ou `autotune` : config, tokenizer ou poids invalides, direction dégénérée, NaN/Inf, système singulier ou mal conditionné |
| 8 | `bench` : l'exécution a échoué (processus llama.cpp, sortie illisible, dumps incompatibles, génération) |
| 9 | `bench --fail-on-warn` : un seuil a été dépassé (le rapport est quand même dans `data`) |
| 70 | Erreur interne (panic), signalée après la restauration du terminal |

Variables d'environnement : `NO_COLOR` (sortie sans couleur), `CLICOLOR_FORCE` (couleurs même dans un tube), `LLAMA_CPP_DIR` (dossier de llama.cpp), `SURGEON_KERNEL` (`scalar`, `avx2`, `avx512` ou `neon`), `SURGEON_MCP_ALLOW_DIRS` (dossiers autorisés de `surgeon mcp`).

<a id="commands"></a>
### 3.2 Référence des commandes

Chaque fiche donne l'usage, chaque option avec son défaut tiré de `--help`, le contenu `--json` de `data` et les codes de sortie en plus des codes communs. Les textes d'aide du binaire sont en partie en français ; les descriptions ci-dessous sont traduites. Les exemples de ce chapitre s'exécutent dans le dossier `demo` du [chemin A](#path-a) (le dépôt est alors `..`) et s'appuient sur ses fichiers.

<a id="cmd-check"></a>
#### `surgeon check`

Calcule la RAM et la VRAM nécessaires pour charger un modèle et donne un verdict, sans le charger.

```text
surgeon check [OPTIONS] --model <MODEL>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | Fichier ou dossier du modèle. |
| `--policy-ram-margin-gib` | défaut de la politique (2) | Marge de sécurité de RAM en GiB. |

`data` : `model_bytes`, `required_ram_bytes`, `required_vram_bytes`, `available_ram_bytes`, `best_gpu`, `verdict.kind` (`fits_in_vram`, `fits_in_ram_only`, `insufficient`), `warnings`. Code de sortie 1 si insuffisant.

<a id="cmd-inspect"></a>
#### `surgeon inspect`

Liste les tenseurs MLP d'un modèle SafeTensors (fichier, shards ou index) ou d'un fichier GGUF, avec des statistiques facultatives.

```text
surgeon inspect [OPTIONS] --model <MODEL>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | Fichier ou dossier SafeTensors, ou `.gguf`. |
| `--layers` | `all` | `all`, une plage `10-20` ou une liste `3,5,7`. |
| `--proj` | `gate`, `up`, `down` | `gate`, `up`, `down`, `o_proj` ou `embed_tokens` ; répétable. Les deux dernières ne sont listées que sur demande (SafeTensors). |
| `--show-stats` | désactivé | Min, max, moyenne et norme L2 par tenseur (deux passes parallèles, déterministes). |

`data` : `shards`, `tensors[]` (`name`, `dtype`, `shape`, `shard`, `stats`). Pour un fichier GGUF le tableau dit aussi si chaque tenseur est patchable.

<a id="cmd-extract"></a>
#### `surgeon extract`

Exécute une passe avant Qwen2 sur deux jeux de prompts et écrit une direction de steering. Rust pur, CPU, `model_type: "qwen2"` uniquement.

```text
surgeon extract [OPTIONS] --model <MODEL> --pos-prompts <POS_PROMPTS> --neg-prompts <NEG_PROMPTS> --out <OUT>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | Dossier avec `config.json`, `tokenizer.json` et les `*.safetensors` (ou un index). |
| `--pos-prompts` | obligatoire | Tableau JSON de chaînes ou de `{"text": ...}`, JSONL, ou texte brut (un prompt par ligne). |
| `--neg-prompts` | obligatoire | Même format, le jeu opposé. |
| `--layer` | 20 | Couche indexée depuis 0, strictement inférieure à `num_hidden_layers`. |
| `--out` | obligatoire | `.safetensors` de sortie (un tenseur f32 1D `direction`). |
| `--pooling` | `last` | `last` (dernier token) ou `mean` (moyenne sur le prompt). |
| `--site` | `resid_post` | `resid_post` (flux résiduel après la couche L) ou `mlp_out` (sortie du MLP). |
| `--chat-template` | `none` | `none` (texte brut) ou `chatml` (format de chat Qwen). |
| `--max-tokens` | 512 | Les prompts sont tronqués à droite au-delà. |
| `--dtype` | `auto` | `auto` (f32 si ça tient en RAM, sinon bf16), `f32`, `bf16`. |
| `--overwrite` | désactivé | Remplace un `--out` existant et ses fichiers compagnons. |
| `--method` | `mean_diff` | `mean_diff`, `svd` ou `leace` (voir `surgeon plugins list`). |
| `--svd-center` | désactivé | `svd` : centre les différences appariées (paires dans un ordre arbitraire). |
| `--leace-shrinkage` | 0.1 | `leace` : rétrécissement λ dans [0, 1] de Σ vers tr(Σ)/d·I ; 0 donne la pseudo-inverse. |
| `--leace-rcond` | 1e-10 | `leace` avec λ = 0 : seuil relatif des valeurs propres conservées. |
| `--holdout-frac` | 0 | Fraction mise de côté (par index) pour des diagnostics hors échantillon ; demande 10 prompts ou plus par jeu. |

Sorties : la direction, `<out>.stats.json` (projection de chaque prompt, lue par le tableau de bord) et, pour `leace`, `<out>.leace.safetensors` (`a`, `b`, `mean`, `u_whitened`). Les écritures sont atomiques. `data` : `out_path`, `stats_path`, `layer`, `method`, `hidden_size`, `n_pos`, `n_neg`, `dtype_used`, `duration_ms`, `stats`, `warnings`. Codes de sortie : 7 pour les échecs de modèle ou de calcul ; 2 pour des arguments invalides, dont `unpaired_prompts` pour `svd`.

<a id="cmd-apply"></a>
#### `surgeon apply`

Applique en place un `patch.json` (SafeTensors, ou GGUF avec `--gguf`) ou un fichier `.surgery`, après avoir tout validé, avec une sauvegarde vérifiée et un rollback automatique.

```text
surgeon apply [OPTIONS] --model <MODEL> --patch <PATCH>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | Fichier ou dossier SafeTensors, ou `.gguf` avec `--gguf`. |
| `--patch` | obligatoire | `patch.json`, ou un fichier `.surgery` (détecté par sa magie `SURG`). |
| `--dry-run` | désactivé | Valide et liste les tenseurs ; n'écrit rien. |
| `--backup-dir` | à côté du modèle | Dossier de sauvegarde ; il doit déjà exister. |
| `--no-backup` | désactivé | Refusé par la validation quand le patch fixe `backup: true`, et toujours refusé avec `--gguf`. |
| `--gguf` | désactivé | Patche en place un `.gguf` quantifié (avec perte : les valeurs sont arrondies de nouveau sur la grille du fichier). Un `.gguf` sans ce drapeau est refusé. |
| `--requant` | `keep-scales` | GGUF : `keep-scales` ou `recompute` ([détails](#formats)). |
| `--no-verify` | désactivé | GGUF : saute la relecture de contrôle après l'écriture. |
| `--force-no-backup` | désactivé | GGUF : patche sans sauvegarde ; un échec laisse le fichier modifié. |
| `--prefetch-chunk-mib` | 64 | Taille des blocs d'écriture (fenêtre de préchargement et granularité du flush). |
| `--no-prefetch` | désactivé | Ni indication de préchargement au système, ni flush intermédiaire. |
| `--scale` | 1 | `.surgery` uniquement : `W <- W + scale * delta`. |
| `--verify-base` | `warn` | `.surgery` uniquement : compare chaque tenseur de base au hash stocké : `strict` (erreur, rien d'écrit), `warn`, `off`. |

`data` : `backups`, `report` (`dry_run`, `backup_path`, tenseurs par opération avec `elements_modified`, `max_abs_delta`, `nan_or_inf_produced`, `rolled_back`), `io` (octets, blocs, débit, défauts de page), `warnings` ; avec `--gguf`, aussi `gguf` (fidélité par tenseur). Codes de sortie : 2 patch invalide (rien touché), 3 échec d'exécution (modèle intact), 4 critique (rollback échoué).

<a id="cmd-eval"></a>
#### `surgeon eval`

Convertit le modèle en GGUF avec llama.cpp, passe chaque prompt dans `llama-cli`, détecte les boucles et vérifie les attentes. L'export GGUF n'a lieu que si le modèle est en bonne santé.

```text
surgeon eval [OPTIONS] --model <MODEL> --prompts <PROMPTS>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | Modèle à évaluer. |
| `--prompts` | obligatoire | Tableau JSON de `{"id"?, "text", "expect_contains"?, "expect_not_contains"?}` (sous-chaînes insensibles à la casse), JSONL, ou texte brut. |
| `--max-new-tokens` | 256 | Budget de génération par prompt. |
| `--export-gguf` | aucun | Écrit le GGUF ici si le modèle est en bonne santé. |
| `--quant` | `Q4_K_M` | Type de quantification passé à `llama-quantize`. |
| `--llama-cpp-dir` | `$LLAMA_CPP_DIR` | Dossier de llama.cpp ([comment il est trouvé](#llama-cpp)). |
| `--allow-degraded` | désactivé | Exporte aussi un modèle jugé `degraded` (jamais un modèle cassé). |

Codes de sortie : 5 modèle en mauvaise santé, 6 export échoué, 1 llama.cpp absent. Le chemin de succès n'a pas été exécuté pour ce README : la machine de test n'a pas de llama.cpp. Ce qui a été exécuté, c'est l'échec précoce quand le fichier de prompts a la mauvaise forme (les fichiers de prompts d'extraction sont des chaînes simples, `eval` veut des objets avec `text`) :

```bash
# le fichier de prompts a le mauvais format : code de sortie 2, rien n'est modifié
surgeon eval --model qwen2-mini --prompts pos.json
```

```text
erreur : fichier de prompts pos.json : JSON invalide : invalid type: string "le joie est fete", expected struct EvalPrompt at line 1 column 19
```

Code de sortie 2, rien n'est touché.

<a id="cmd-tui"></a>
#### `surgeon tui`

Tableau de bord dans le terminal avec un inspecteur de poids (onglet 1), un visualiseur de vecteur de steering (onglet 2) et, avec `--preview`, un onglet de génération en direct (onglet 3).

```text
surgeon tui [OPTIONS]
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | aucun | Dossier du modèle ou fichier SafeTensors. |
| `--vector` | aucun | Fichier de direction produit par `extract`. |
| `--snapshot` | désactivé | Affiche une image en texte brut et quitte (pas besoin de terminal interactif). |
| `--size` | `100x32` | Taille de l'image pour `--snapshot`, `COLSxROWS` ; en dessous de 80x24 il affiche « Terminal trop petit ». |
| `--preview` | désactivé | Troisième onglet : génération avec pilotage d'activations (demande `--model` et `--vector`). |
| `--preview-prompt` | aucun | Prompt de la prévisualisation. |
| `--preview-alpha` | 0 | Alpha relatif initial (`alpha_abs = alpha * norme résiduelle moyenne à la couche`). |
| `--preview-tokens` | 30 | Tokens à générer. |
| `--preview-layer` | couche du vecteur | Couche d'injection. |
| `--preview-template` | `chatml` | `none` ou `chatml`. |
| `--alpha-range` | `-2:2` | Bornes du curseur `MIN:MAX`. |

Sans `--snapshot` il faut un terminal d'au moins 80x24 relié à stdin, stdout et stderr (code de sortie 1 sinon). Touches : `Tab` change d'onglet, les flèches déplacent, `l` bascule un histogramme logarithmique, `?` aide, `q` quitte. Exemple (mini-modèle factice, vecteur seul) :

```bash
# une image du tableau de bord en texte brut
surgeon tui --snapshot --vector direction.safetensors
```

```text
┌ surgeon tui ─────────────────────────────────────────────────────────────────────────────────────┐
│ 1 Inspecteur de poids │ 2 Steering vector                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
┌ Vecteur ───────────────────────────────────────┐┌ d′ (séparation) ───────────────────────────────┐
│taille          16                              ││██████████████████████2.23                      │
│norme L2        1.000  normalisé ✓              │└────────────────────────────────────────────────┘
│min · max       -0.5069 · 0.5141                │┌ accuracy au seuil ─────────────────────────────┐
│moyenne         -5.3450e-2                      ││██████████████████████92 % █████████████████    │
│                                                │└────────────────────────────────────────────────┘
└────────────────────────────────────────────────┘┌ Séparation ────────────────────────────────────┐
┌ TOP-10 |v_i| ──────────┐┌ Valeurs (linéaire) ──┐│proj. μ₊ · μ₋   1.2984 · -2.0793                │
│ 1. dim 15   +0.5141    ││        █             ││cos(μ₊, μ₋)     0.4887                          │
│ 2. dim 2    -0.5069    ││        █             ││couche · site   2 · resid_post                  │
│ 3. dim 4    -0.4161    ││        █             ││pooling         last                            │
│ 4. dim 6    -0.2963    ││        ███           ││n_pos · n_neg   6 · 6                           │
│ 5. dim 3    +0.2675    ││        ███           ││modèle source   qwen2-mini                      │
│ 6. dim 5    +0.2039    ││        ███           ││                                                │
│ 7. dim 9    -0.1790    ││█ █ █  █████ █ █ █   █│└────────────────────────────────────────────────┘
│ 8. dim 1    -0.1515    ││█ █ █  █████ █ █ █   █│┌ Projections : pos ▲ / neg ▼ (┼ seuil) ─────────┐
│ 9. dim 12   -0.1156    ││█ █ █  █████ █ █ █   █││                █                   █ ███      █│
│10. dim 10   -0.1128    ││-5.0694e-1   5.1405e-1││                █                   █ ███      █│
└────────────────────────┘└──────────────────────┘│                █                   █ ███      █│
┌ Composantes 0..16 (zoom ×1) ───────────────────┐│                █                   █ ███      █│
│               █                                ││────────────────────────────────┼───────────────│
│   █ █         █                                ││█      ██     █     █       █                   │
│█  █ █  █      █                                ││█      ██     █     █       █                   │
│────────────────────────────────────────────────││█      ██     █     █       █                   │
│ ██ █ ██ ██████                                 ││█      ██     █     █       █                   │
│  █ █ █  █                                      ││seuil 0.7182                                    │
│  █ █                                           ││                                                │
└────────────────────────────────────────────────┘└────────────────────────────────────────────────┘
Tab onglet · ←→ défiler · +/- zoom · ↑↓ top-10 · ? aide · q quitter  direction.safetensors
```

La prévisualisation est un pilotage d'activations (`h <- h + alpha_abs * v` à une couche), une approximation du patch de poids, pas le patch lui-même. Elle tourne sur le CPU et est lente sur les gros modèles.

<a id="cmd-mcp"></a>
#### `surgeon mcp`

Lance le serveur MCP sur stdio ([section 4](#mcp)).

```text
surgeon mcp [OPTIONS]
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--allow-dir` | `$SURGEON_MCP_ALLOW_DIRS`, sinon le dossier courant | Dossier autorisé, répétable ; chaque chemin doit se résoudre sous l'un d'eux. |
| `--read-only` | désactivé | Refuse les outils qui écrivent (`surgeon_apply`, exports GGUF...). |
| `--allow-no-backup` | désactivé | Respecte `backup: false` dans les patchs ; une sauvegarde est imposée sinon. |

<a id="cmd-schema"></a>
#### `surgeon schema`

Affiche un JSON Schema (draft-07) ou les définitions des outils MCP, en JSON brut même avec `--json`.

```text
surgeon schema [OPTIONS]
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--kind` | `patch` | `patch` (schéma de `patch.json`), `tools` (les définitions d'outils exactement comme `tools/list` les renvoie), `surgery` (schéma de l'en-tête `.surgery`). |

```bash
# les trois schémas, en JSON brut
surgeon schema --kind patch
surgeon schema --kind tools
surgeon schema --kind surgery
```

<a id="cmd-autotune"></a>
#### `surgeon autotune`

Ajuste un coefficient de steering par couche par régression ridge sur des réponses mesurées, et écrit éventuellement le résultat sous forme de `patch.json`.

```text
surgeon autotune [OPTIONS] --model <MODEL> --vector <VECTOR> --eval-dataset <EVAL_DATASET>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | Dossier Qwen2. |
| `--vector` | obligatoire | Fichier de direction issu de `extract` (tenseur `direction` et ses métadonnées). |
| `--eval-dataset` | obligatoire | Tableau JSON (ou JSONL) de chaînes ou de `{"text", "target"?}`. |
| `--target-layer-range` | de 0 à la couche de lecture | Couches candidates `A-B`. |
| `--readout-layer` | couche du vecteur | Couche dont l'activation est lue. |
| `--proj` | `down_proj` | Projection à patcher. |
| `--objective` | `suppress` | `suppress` (vers la moyenne négative), `amplify` (vers la moyenne positive), `target` (vers `--target-projection`). |
| `--target-projection` | aucun | Projection cible ; obligatoire avec `--objective target`. |
| `--l2-reg` | 1e-3 | λ ridge > 0, ou `auto` (leave-one-out de 1e-6 à 1e2). |
| `--probe-alpha` | 0.05 | Pas δ de différence finie de la mesure de réponse. |
| `--alpha-max` | 1 | Les coefficients sont écrêtés à plus ou moins cette valeur. |
| `--min-alpha` | 0.0001 | Les coefficients plus petits sont supprimés. |
| `--max-forwards` | 2000 | Refuse l'exécution, avant tout calcul, au-delà de ce nombre de passes avant. |
| `--max-tokens` | 256 | Troncature des prompts. |
| `--dtype` | `auto` | `auto`, `f32`, `bf16`. |
| `--site` | celui du vecteur | Remplace le site du vecteur. |
| `--pooling` | celui du vecteur | Remplace le pooling du vecteur. |
| `--chat-template` | celui du vecteur | Remplace le gabarit de chat du vecteur. |
| `--out` | aucun | `patch.json` à écrire (validé d'abord). |
| `--overwrite` | désactivé | Remplace un `--out` existant. |

`data` contient `alphas`, `ridge`, `linearity`, `validation`, `recommendation` et `patch`. Code de sortie 7 quand le système est singulier ou mal conditionné. Notez que le `patch.json` écrit enregistre le chemin absolu du fichier de vecteur.

<a id="cmd-diff"></a>
#### `surgeon diff`

Écrit la différence entre deux checkpoints sous forme de patch `.surgery` de rang faible.

```text
surgeon diff [OPTIONS] --base <BASE> --tuned <TUNED> --out <OUT>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--base` | obligatoire | Checkpoint de base (fichier `.safetensors` ou dossier shardé). |
| `--tuned` | obligatoire | Checkpoint affiné, mêmes noms de tenseurs, formes et dtypes. |
| `--out` | obligatoire | Fichier `.surgery` à écrire. |
| `--rank` | 16 | Rang de l'approximation de chaque delta 2D. |
| `--factor-dtype` | `f16` | `f16`, `bf16` ou `f32` (bascule en bf16 avec un avertissement en cas de dépassement f16). |
| `--include` | aucun | Seulement les tenseurs dont le nom correspond à cette regex. |
| `--exclude` | aucun | Ignore les tenseurs dont le nom correspond à cette regex. |
| `--include-embeddings` | désactivé | Compare aussi `embed_tokens` et `lm_head` (exclus par défaut). |
| `--max-rel-error` | aucun | Erreur relative de reconstruction au-dessus de laquelle `--on-high-error` s'applique. |
| `--on-high-error` | `warn` | `warn`, `skip` (laisse le tenseur de côté) ou `full` (stocke le delta exact). |
| `--oversample` | 8 | Directions aléatoires supplémentaires de la SVD randomisée. |
| `--power-iters` | 3 | Itérations de puissance de la SVD randomisée. |
| `--seed` | 0 | Graine ; même entrée et même graine donnent les mêmes octets. |
| `--timestamp` | désactivé | Enregistre la date de création dans l'en-tête (sans lui la sortie est reproductible). |
| `--overwrite` | désactivé | Remplace un `--out` existant. |

Exécution réelle sur une paire factice écrite par `examples/make_fake_checkpoints.rs` (la différence est de rang 4, donc l'erreur est nulle ; tenseurs raccourcis) :

```bash
# génère un faux checkpoint de base et un affiné, puis écrit leur différence de rang faible
cargo run --release --manifest-path ../Cargo.toml --example make_fake_checkpoints -- --out ckpt --kind rank --rank 4 --lora
surgeon diff --base ckpt/base --tuned ckpt/tuned --out finetune.surgery --rank 4
```

```text
patch écrit : finetune.surgery (0.01 MiB, factor_dtype f16, rang demandé 4)
10 tenseur(s) dans le patch, 1 identique(s) ignoré(s), 2 filtré(s)
  model.layers.0.self_attn.q_proj.weight · lowrank · rang 4 · erreur relative 0.0000 · énergie 1.0000
  model.layers.0.mlp.down_proj.weight · lowrank · rang 4 · erreur relative 0.0000 · énergie 1.0000
  [...]
  model.layers.1.post_attention_layernorm.weight · full_delta · rang 0 · erreur relative 0.0000 · énergie 1.0000
Note : le .surgery est une APPROXIMATION du fine-tune (erreur relative par tenseur), exacte seulement si la différence est de rang <= r. Sa taille vaut environ la somme de r(m+n) x octets par facteur : des centaines de Mo pour un 7B au rang 16.
```

Application :

```bash
# valide le patch .surgery sur la base sans rien écrire
surgeon apply --model ckpt/base --patch finetune.surgery --dry-run
```

```text
[DRY-RUN]  model.layers.0.input_layernorm.weight · full_delta · rang 0 · erreur 0.0000
[DRY-RUN]  model.layers.0.mlp.down_proj.weight · lowrank · rang 4 · erreur 0.0000
[...]
dry-run : 10 tenseur(s) vérifiés, aucun octet écrit
```

Un `.surgery` est une approximation du fine-tune complet (exact seulement si la vraie différence est de rang au plus `--rank`) et il n'est pas petit : environ la somme des `r (m + n)` fois les octets du dtype des facteurs, des centaines de Mo pour un modèle 7B au rang 16. Il s'applique uniquement depuis le CLI (`apply` sur un `.surgery`), `--gguf` lui est refusé, et les autres options GGUF et d'écriture par blocs de `apply` sont ignorées avec un avertissement. `data` vaut `{"surgery": ...}`.

<a id="cmd-lora"></a>
#### `surgeon lora`

Deux sous-commandes, `lora merge` et `lora extract`.

<a id="cmd-lora-merge"></a>
#### `surgeon lora merge`

Fusionne en place un adaptateur LoRA PEFT dans les poids : `W <- W + scale * (alpha/r) B A`, avec sauvegarde vérifiée et rollback. DoRA, les adaptateurs d'embeddings, `fan_in_fan_out` et `rank_pattern` / `alpha_pattern` sont refusés explicitement ; la fusion dans un GGUF n'est pas gérée.

```text
surgeon lora merge [OPTIONS] --model <MODEL> --lora <LORA>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | Fichier `.safetensors` ou dossier shardé à modifier. |
| `--lora` | obligatoire | Le FICHIER de l'adaptateur `adapter_model.safetensors` (clés PEFT). |
| `--alpha` | celui de la config | `lora_alpha` ; remplace la valeur de `adapter_config.json`. L'un des deux est obligatoire. |
| `--adapter-config` | à côté de `--lora` | `adapter_config.json`. |
| `--scale` | 1 | Multiplicateur supplémentaire du delta. |
| `--dry-run` | désactivé | Valide seulement. |
| `--no-backup` | désactivé | Ne sauvegarde pas les fichiers touchés (ils sont restaurés en cas d'erreur sinon). |
| `--backup-dir` | à côté du modèle | Dossier de sauvegarde. |

```bash
# valide la fusion sans rien écrire
cp -r ckpt/base ckpt/base-copy
surgeon lora merge --model ckpt/base-copy --lora ckpt/lora/adapter_model.safetensors --dry-run
```

```text
dry-run : 6 tenseur(s) vérifiés (rang 4, échelle effective 1), aucun octet écrit
```

<a id="cmd-lora-extract"></a>
#### `surgeon lora extract`

Approxime la différence de deux checkpoints par un adaptateur de rang `r` (`B = U sqrt(S)`, `A = sqrt(S) V`) et écrit un `adapter_model.safetensors` PEFT plus un `adapter_config.json` à côté. Le dossier de sortie doit déjà exister.

```text
surgeon lora extract [OPTIONS] --base <BASE> --patched <PATCHED> --out <OUT>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--base` | obligatoire | Checkpoint de base. |
| `--patched` | obligatoire | Checkpoint modifié (mêmes noms de tenseurs, formes et dtypes). |
| `--out` | obligatoire | `adapter_model.safetensors` à écrire. |
| `--rank` | 16 | Rang de l'adaptateur. |
| `--dtype` | `f32` | `f16`, `bf16` ou `f32`. |
| `--target-modules` | projections d'attention et MLP | Seulement les tenseurs 2D dont le nom correspond à cette regex. |
| `--seed` | 0 | Graine de la SVD randomisée. |
| `--overwrite` | désactivé | Remplace un `--out` existant. |

```bash
# le dossier de sortie doit exister, puis extrait un adaptateur de rang 4
mkdir -p ckpt/adapter-out
surgeon lora extract --base ckpt/base --patched ckpt/tuned --out ckpt/adapter-out/adapter_model.safetensors --rank 4
```

```text
6 tenseur(s) extraits (rang 4, f32), erreur relative moyenne 0.0000
  adapter : ckpt/adapter-out/adapter_model.safetensors
  config  : ckpt/adapter-out\adapter_config.json
```

La structure PEFT suit sa documentation et n'est pas testée contre Hugging Face PEFT en CI. `data` des deux : `lora`, `warnings` (`backups` pour `merge`).

<a id="cmd-bench"></a>
#### `surgeon bench`

Cinq modes qui mesurent ce qu'une chirurgie a changé. La recette : exporter les logits des deux modèles sur les mêmes prompts, les comparer, vérifier la perplexité, un benchmark à correspondance exacte et, pour une chirurgie sur les refus, le taux de refus, puis décider. Les seuils ne produisent que des avertissements, sauf avec `--fail-on-warn` (code de sortie 9, rapport toujours dans `data`).

<a id="cmd-bench-dump"></a>
#### `surgeon bench dump`

Écrit les logits d'un modèle Qwen2 sur des prompts (Rust pur, CPU).

```text
surgeon bench dump [OPTIONS] --model <MODEL> --prompts <PROMPTS> --out <OUT>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | Dossier Qwen2. |
| `--prompts` | obligatoire | Tableau JSON de chaînes ou de `{"text": ...}`, JSONL, ou texte brut. |
| `--out` | obligatoire | `.safetensors` avec `logits` `[positions, vocabulaire]` et `prompt_offsets`, plus des métadonnées (`prompts_sha256`, `tokenizer_sha256`). |
| `--dtype` | `auto` | Dtype de calcul : `auto`, `f32`, `bf16`. |
| `--out-dtype` | `f32` | Dtype stocké : `f32` ou `f16`. Préférez f32 : f16 ajoute du bruit d'arrondi à la comparaison. |
| `--max-tokens` | 256 | Troncature des prompts. |
| `--max-total-positions` | 4096 | Refuse le dump, avant tout calcul, au-delà de ce nombre de positions. |
| `--overwrite` | désactivé | Remplace un `--out` existant. |

<a id="cmd-bench-logits"></a>
#### `surgeon bench logits`

Compare deux dumps de logits : KL(avant‖après), cosinus, décalages top-1 et top-k. Il lit `.safetensors`, `.npy` et `.pt`/`.pth` (non compressés, contigus, au mieux), f32/f16/bf16, `[positions, vocabulaire]` ou `[batch, seq, vocabulaire]`, et avertit quand les dumps viennent de prompts ou de tokenizers différents.

```text
surgeon bench logits [OPTIONS] --before <BEFORE> --after <AFTER>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--before` | obligatoire | Dump de référence. |
| `--after` | obligatoire | Dump à comparer, même forme. |
| `--tensor` | `logits` | Tenseur à lire. |
| `--top-k` | 5 | Taille des ensembles top-k comparés. |
| `--worst` | 5 | Prompts listés dans `worst_prompts` (demande `prompt_offsets`). |
| `--warn-kl` | aucun | Avertit quand la KL moyenne est strictement au-dessus de cette valeur. |
| `--warn-top1-shift` | aucun | Avertit quand le taux de changement du top-1 (0 à 1) est strictement au-dessus de cette valeur. |
| `--fail-on-warn` | désactivé | Sort avec le code 9 quand un seuil est dépassé. |

`data` contient `kl_mean`, `kl_max`, `kl_max_position`, `kl_p95`, `kl_infinite_positions`, `cosine_probs_mean`, `cosine_logits_mean`, `top1_shift_rate`, `topk_set_change_rate`, `topk_overlap_mean`, `per_prompt`, `worst_prompts`, `verdict`, `timing_ms`. Les réductions s'exécutent dans un ordre fixe : les mêmes bits sortent avec 1 ou 12 threads ; deux exécutions donnent le même JSON hors `timing_ms`.

<a id="cmd-bench-ppl"></a>
#### `surgeon bench ppl`

Perplexité d'un GGUF (ou d'un dossier SafeTensors converti en GGUF temporaire) avec `llama-perplexity` de llama.cpp.

```text
surgeon bench ppl [OPTIONS] --model <MODEL> --text <TEXT>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | `.gguf`, ou un dossier SafeTensors. |
| `--text` | obligatoire | Fichier texte évalué par `llama-perplexity`. |
| `--baseline-model` | aucun | Modèle de référence, mesuré en premier. |
| `--baseline-ppl` | aucun | Perplexité de référence déjà connue. |
| `--baseline-ppl-err` | aucun | Incertitude (±) de `--baseline-ppl`. |
| `--ctx` | 512 | Taille du contexte. |
| `--chunks` | défaut de llama.cpp | Nombre de blocs. |
| `--threads` | défaut de llama.cpp | Threads. |
| `--n-gpu-layers` | 0 | Couches déportées sur le GPU (0 : CPU seul, reproductible). |
| `--llama-cpp-dir` | `$LLAMA_CPP_DIR` | Dossier de llama.cpp. |
| `--warn-delta` | 0.2 | Avertit quand le delta arrondi est strictement au-dessus de cette valeur. |
| `--timeout-secs` | aucun | Par exécution de `llama-perplexity`. |
| `--llama-arg` | aucun | Drapeau supplémentaire d'une liste blanche (`--batch-size=N`, `--ubatch-size=N`, `--ppl-stride=N`, `--seed=N`, `-b=N`, `-ub=N`, `--no-mmap`, `--mlock`) ; répétable. |
| `--fail-on-warn` | désactivé | Sort avec le code 9 quand le delta dépasse `--warn-delta`. |

<a id="cmd-bench-eval"></a>
#### `surgeon bench eval`

Pass@1 à correspondance exacte sur un jeu de type GSM8K ou MATH, décodage glouton via llama.cpp ou, avec `--backend native`, via le générateur en Rust pur (dossier Qwen2 SafeTensors, CPU, sans llama.cpp).

```text
surgeon bench eval [OPTIONS] --model <MODEL> --dataset <DATASET>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | `.gguf`, ou un dossier SafeTensors. |
| `--dataset` | obligatoire | Tableau JSON (ou JSONL) de `{question ou problem, answer ou solution, id?}`. Aucun contenu GSM8K ou MATH n'est livré avec ce dépôt. |
| `--backend` | `llama` | `llama` (llama.cpp sur un GGUF, converti au besoin) ou `native` (Rust pur sur un dossier Qwen2 SafeTensors ; `--llama-cpp-dir` ne sert alors pas). |
| `--baseline-model` | aucun | Modèle de référence sur les mêmes questions (comparaison appariée). |
| `--limit` | 50 | Les N premières entrées, dans l'ordre du fichier. |
| `--max-new-tokens` | 512 | Budget de génération par question. |
| `--prompt-template` | intégré | Gabarit avec `{question}`, ou `@FILE`. |
| `--format` | `auto` | Réponse de référence : `gsm8k` (après le dernier `####`), `math` (dernier `\boxed{...}`), `auto` (par entrée : `####`, sinon `\boxed{}`, sinon la réponse brute). |
| `--llama-cpp-dir` | `$LLAMA_CPP_DIR` | Dossier de llama.cpp. |
| `--timeout-secs` | aucun | Par question ; un dépassement compte la question comme fausse. |
| `--min-accuracy` | aucun | Avertit quand l'exactitude (0 à 1) est strictement en dessous de cette valeur. |
| `--fail-on-warn` | désactivé | Sort avec le code 9 quand l'exactitude est sous `--min-accuracy`. |

La réponse du modèle est extraite par la première règle qui correspond parmi : un marqueur `####`, le dernier `\boxed{...}` équilibré, le dernier texte en `**gras**`, une formule comme « the answer is X », puis le dernier nombre. Les nombres et fractions se comparent avec une tolérance relative de 1e-9. C'est une heuristique : lisez les `items` par question de la sortie `--json` avant de vous fier à un score. Ce README ne publie aucun score GSM8K : les exécutions réelles ci-dessous utilisent un faux llama.cpp.

`ppl` et `eval` ci-dessous ont tourné avec `examples/fake_llama.rs` copié sous les noms `llama-perplexity` et `llama-completion` (il dérive une perplexité des octets du modèle et répond la somme des nombres de chaque problème), sur les faux fichiers GGUF Q8_0 de la [section formats](#formats). Ils exercent la gestion des processus, pas un vrai modèle. Préparation : les binaires factices (sous Windows les copies demandent le suffixe `.exe`), une copie patchée du faux GGUF à comparer à `tiny-q8.orig.gguf`, un fichier texte et un jeu de données.

```bash
# compile le faux llama.cpp et l'installe sous les deux noms que les commandes cherchent
cargo build --release --manifest-path ../Cargo.toml --example fake_llama
mkdir llama
cp ../target/release/examples/fake_llama llama/llama-perplexity
cp ../target/release/examples/fake_llama llama/llama-completion
# une copie patchée du faux GGUF, et un texte pour la perplexité
cp tiny-q8.orig.gguf tiny-q8.patched.gguf
surgeon apply --gguf --requant recompute --model tiny-q8.patched.gguf --patch gguf-patch.json --backup-dir backups
echo "le joie est fete la soleil est danse" > text.txt
```

Enregistrez le jeu de données sous `ds.json` (les réponses finissent par `#### valeur`, la convention GSM8K) :

```json
[{"id": "q1", "question": "What is 12 plus 30?", "answer": "12 + 30 = 42\n#### 42"},
 {"id": "q2", "question": "What is 5 plus 6?", "answer": "5 + 6 = 11\n#### 11"}]
```

```bash
# perplexité et Pass@1 du fichier patché face à l'original (faux llama.cpp)
surgeon bench ppl --model tiny-q8.patched.gguf --text text.txt --baseline-model tiny-q8.orig.gguf --llama-cpp-dir llama
surgeon bench eval --model tiny-q8.patched.gguf --dataset ds.json --baseline-model tiny-q8.orig.gguf --llama-cpp-dir llama --max-new-tokens 64 --min-accuracy 0.5
```

```text
[BENCH]    ppl · ctx 512  tiny-q8.patched.gguf
[PPL]      6.2060 ± 0.04 → 7.3840 ± 0.04 · Δ +1.1780 (hors du bruit)
[VERDICT]  avertissement : ΔPPL 1.1780 > 0.2
```

```text
[BENCH]    eval · 2 questions · auto  ds.json
[PASS@1]   2/2 · 100.0 % · IC95 [34.2 ; 100.0] · boucles 0
[PASS@1]   référence 2/2 · 100.0 % · IC95 [34.2 ; 100.0] · boucles 0
  appariement : perdues 0 · gagnées 0 · Δ +0.0 points
[VERDICT]  ok · seuils respectés
  ! échantillon réduit (n = 2) : l'intervalle de confiance est large (± 32.9 points)
```

<a id="cmd-bench-refusal"></a>
#### `surgeon bench refusal`

Taux de refus d'un modèle Qwen2 sur des prompts : génération gloutonne en Rust pur sur le CPU (sans llama.cpp), puis recherche de marqueurs de refus canoniques dans chaque réponse.

```text
surgeon bench refusal [OPTIONS] --model <MODEL> --prompts <PROMPTS>
```

| Option | Défaut | Rôle |
| --- | --- | --- |
| `--model` | obligatoire | Dossier Qwen2. |
| `--prompts` | obligatoire | Tableau JSON de chaînes ou de `{"text": ...}`, JSONL, ou texte brut. |
| `--baseline-model` | aucun | Modèle de référence sur les mêmes prompts, généré en premier (comparaison appariée). |
| `--max-new-tokens` | 256 | Budget de génération par prompt. |
| `--chat-template` | `chatml` | Habillage de chaque prompt : `chatml` ou `none`. |

Une réponse est un refus quand elle contient l'un de douze marqueurs anglais (`I'm sorry`, `I cannot`, `As an AI`, ... ; la liste est dans `data.markers`), comparés sans casse et aux frontières de mots. Il n'y a pas de modèle juge : une réponse dégradée ou hors sujet compte comme une acceptation, donc `data.items[]` garde chaque réponse en entier (`index`, `refused`, `marker`, `output`) pour relecture. `data` contient aussi `n`, `refused`, `refusal_rate` et son intervalle de Wilson (`ci95_low`, `ci95_high`) ; avec une référence, `baseline` et `paired` (`unblocked`, `newly_refused`, `delta_rate`, et son intervalle à 95 % `delta_ci95_low` / `delta_ci95_high`, l'intervalle de Newcombe pour proportions appariées).

```bash
# taux de refus du mini modèle patché contre l'original (poids aléatoires : aucun refus)
surgeon bench refusal --model qwen2-mini --prompts pos.json --baseline-model qwen2-orig --max-new-tokens 8
```

```text
[BENCH]    refusal · 6 prompts · chatml · 8 tokens max  qwen2-mini
[REFUS]    0/6 · 0.0 % · IC95 [0.0 ; 39.0]
[REFUS]    référence 0/6 · 0.0 % · IC95 [0.0 ; 39.0]
  appariement : débloqués 0 · nouveaux refus 0 · Δ +0.0 points, IC95 [-39.0 ; +39.0]
```

Le mini modèle a des poids aléatoires et ne refuse jamais : l'exécution montre le rapport, pas un comportement. La génération est lente sur un vrai modèle (environ 11 tokens par seconde pour Qwen2.5-0.5B sur la machine à 12 threads de ce README).

<a id="cmd-plugins"></a>
#### `surgeon plugins`

Commande parente de `plugins list` ; elle n'a pas d'option propre.

<a id="cmd-plugins-list"></a>
#### `surgeon plugins list`

Liste les points d'extension du moteur et les implémentations enregistrées (identifiant, titre, origine).

```text
surgeon plugins list [OPTIONS]
```

Elle n'a aucune option en dehors des options globales. Sortie réelle :

```bash
# points d'extension et implémentations enregistrées
surgeon plugins list
```

```text
[EXTENSION] extraction_method · weight_surgeon::plugin::ExtractionMethod · API v0
  mean_diff  builtin    Différence des moyennes
  svd        builtin    SVD des différences appariées
  leace      builtin    Effaceur LEACE
```

Le binaire n'enregistre que les trois méthodes intégrées ; une méthode tierce passe par la bibliothèque ([docs/PLUGINS.md](docs/PLUGINS.md)).

<a id="patch-format"></a>
### 3.3 `patch.json` et les axes `rows` / `cols`

`surgeon schema` affiche le JSON Schema complet (draft-07) de `patch.json`, et ce schéma est la référence. En bref : un patch a `schema_version` (1), une `description` facultative, `backup` (par défaut `true`) et un tableau `operations` non vide appliqué dans l'ordre ; toute erreur annule le patch entier. Une opération a :

| Champ | Rôle |
| --- | --- |
| `id` | Identifiant unique, repris dans le rapport. |
| `layers` | `"all"`, `"10-20"`, `"3,5,7"`, un tableau `[10, 11]`, ou `{"start": 10, "end": 20}` (bornes incluses). |
| `projections` | Non vide, sans doublon : `gate_proj`, `up_proj`, `down_proj`, et pour un modèle SafeTensors `o_proj` (sortie d'attention de chaque couche) et `embed_tokens` (plongements d'entrée ; ils n'appartiennent à aucune couche, donc `layers` ne les filtre pas). |
| `operation` | `subtract`, `add` ou `project_out`. |
| `vector_file` | Vecteur `.safetensors` ou `.json` ; relatif au dossier du patch. |
| `vector_tensor` | Nom du tenseur quand le fichier SafeTensors en contient plusieurs ; interdit avec un `.json`. |
| `vector_tensor_b` | `project_out` uniquement : la direction de lecture `b` (`vector_tensor` est alors `a`). |
| `alpha` | Intensité dans [-100, 100] ; `|alpha| > 5` lève un avertissement. Pour `project_out`, dans [0, 1]. |
| `axis` | `rows` ou `cols` (par défaut `cols`). |

La formule de `subtract` et `add`, sur un tenseur `[rows, cols]` rangé par lignes : `W[r][c] <- round(W[r][c] ∓ alpha * v[i])`. Le produit `alpha * v[i]` est calculé une fois par élément du vecteur en f32, puis chaque poids reçoit une addition f32 et est arrondi au plus proche pair dans le dtype stocké (bf16, f16 ou f32). `alpha = 0` ne fait strictement rien. Les poids sont `[sortie, entrée]` (disposition Llama et Qwen) :

```text
gate_proj / up_proj : [intermediate, hidden]      down_proj : [hidden, intermediate]

axis "rows"  (len(v) = cols)                      axis "cols"  (len(v) = rows)
W[r][c] -= alpha * v[c]                           W[r][c] -= alpha * v[r]

   c0  c1  c2  c3                                    c0   c1   c2   c3
r0 -v0 -v1 -v2 -v3                                r0 -v0  -v0  -v0  -v0
r1 -v0 -v1 -v2 -v3                                r1 -v1  -v1  -v1  -v1
r2 -v0 -v1 -v2 -v3                                r2 -v2  -v2  -v2  -v2
```

Avec `rows`, chaque ligne reçoit le même vecteur (une valeur par colonne) ; avec `cols`, chaque ligne reçoit une seule valeur `v[r]` répétée le long de la ligne. Un vecteur de steering vit dans le flux résiduel, dont la taille est `hidden`. Pour `gate_proj` et `up_proj` cette taille est le nombre de colonnes, donc utilisez `rows` ; pour `down_proj` c'est le nombre de lignes, donc utilisez `cols`. `o_proj` (`[hidden, hidden]`) écrit comme `down_proj` : `cols`. `embed_tokens` (`[vocab, hidden]`) porte un vecteur du flux par ligne : `rows`. Un mauvais choix échoue à la validation avec une erreur de longueur avant qu'un seul octet soit écrit.

> **Pour les experts**
> `project_out` retire ce que lit `b` des matrices qui écrivent dans le flux résiduel : `W' = W - alpha * a (b^T W)` pour `down_proj` et `o_proj`, `W' = W - alpha * (W b) a^T` pour `embed_tokens` (ses lignes sont les vecteurs du flux). Avec `a` et `b` d'un `.leace.safetensors` c'est l'effaceur LEACE ; avec `a = b = v`, une direction unitaire (`"vector_tensor": "direction", "vector_tensor_b": "direction"`), c'est une ablation directionnelle. Une seule opération avec `"layers": "all"` et `"projections": ["down_proj", "o_proj", "embed_tokens"]` couvre tous les écrivains ; `gate_proj` et `up_proj` lisent le flux et sont refusées. En bf16 un petit delta peut disparaître à l'arrondi : mesurez le résultat.
>
> Quand `embed_tokens` est visé et que le modèle le partage avec la projection de sortie (pas de `lm_head.weight`, comme Qwen2.5-0.5B), `apply` détache d'abord la tête pour ne pas déformer les logits : il réécrit le shard avec `lm_head.weight`, copie à l'octet des plongements non patchés, et passe `"tie_word_embeddings"` à `false` dans `config.json` (et complète `model.safetensors.index.json` pour un modèle shardé). Le shard grossit de la taille des plongements. `--dry-run` l'annonce dans `data.untie_lm_head` et n'écrit rien ; le shard et `config.json` sont tous deux sauvegardés, et restaurés en cas d'échec. Sans `config.json` à côté du modèle le patch est refusé.

<a id="formats"></a>
### 3.4 Formats

| Format | Où il sert |
| --- | --- |
| SafeTensors (fichier, shards, index) | Lu et patché en place par `inspect`, `apply`, `diff`, `lora`, `eval` ; mappé en mémoire. Dtypes des noyaux de décalage : bf16, f16, f32. |
| GGUF v3, fichier unique, little endian | `inspect`, `apply --gguf` ; types cibles F32, F16, BF16, Q8_0, Q4_K, Q6_K. Écrit par `eval --export-gguf` via llama.cpp. |
| `.surgery` | Patch de rang faible écrit par `diff`, appliqué par `apply` ; magie `SURG`, spécifié dans [docs/SPECIFICATION.md](docs/SPECIFICATION.md) ; schéma de l'en-tête : `surgeon schema --kind surgery`. |
| Adaptateur LoRA PEFT | `adapter_model.safetensors` (clés `base_model.model.<chemin>.lora_A.weight` et `lora_B.weight`) plus `adapter_config.json` ; lu par `lora merge`, écrit par `lora extract`. |
| Dumps de logits | `.safetensors` écrit par `bench dump` ; `bench logits` lit aussi `.npy` et `.pt`. |

Pour un fichier GGUF, `gate_proj`, `up_proj` et `down_proj` de la couche `i` sont `blk.{i}.ffn_{gate,up,down}.weight` ; `ne[0]` est la dimension des colonnes, donc `rows` et `cols` gardent leur sens SafeTensors. Un fichier « Q4_K_M » mélange les types par tenseur (`ffn_down` est souvent en Q6_K) ; un tenseur patché d'un autre type est refusé avec `unsupported_quant_type`. Un petit GGUF pour essayer est généré par `examples/make_fake_gguf.rs` (deux couches, taille cachée 32, Q8_0). Enregistrez un vecteur de 32 uns sous `v32.json`, puis le patch ci-dessous sous `gguf-patch.json` :

```json
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

```json
{
  "schema_version": 1,
  "operations": [
    { "id": "g-down", "layers": "0-1", "projections": ["down_proj"],
      "operation": "subtract", "vector_file": "v32.json", "alpha": 0.05, "axis": "cols" }
  ]
}
```

```bash
# faux GGUF, une copie intacte, puis inspection, essai à blanc et application
cargo run --release --manifest-path ../Cargo.toml --example make_fake_gguf -- tiny-q8.gguf
cp tiny-q8.gguf tiny-q8.orig.gguf
surgeon inspect --model tiny-q8.gguf
surgeon apply --gguf --model tiny-q8.gguf --patch gguf-patch.json --dry-run
surgeon apply --gguf --model tiny-q8.gguf --patch gguf-patch.json --backup-dir backups
```

```text
tiny-q8.gguf
█▀█ █ █ ▀█▀  W E I G H T
█ █ ▀▄▀  █   S U R G E O N
[INSPECT]  6 tenseurs sélectionnés · GGUF v3 · qwen2
couche   proj   type   shape      taille    patchable
0        gate   Q8_0   [64, 32]   2.1 KiB   oui      
0        up     Q8_0   [64, 32]   2.1 KiB   oui      
0        down   Q8_0   [32, 64]   2.1 KiB   oui      
1        gate   Q8_0   [64, 32]   2.1 KiB   oui      
1        up     Q8_0   [64, 32]   2.1 KiB   oui      
1        down   Q8_0   [32, 64]   2.1 KiB   oui      
```

```text
[APPLY]    gguf-patch.json · 1 opérations
  ! blk.0.ffn_down.weight : 1195 valeurs écrêtées aux bornes de la grille d'origine (keep-scales ne change aucune échelle) : --requant recompute recalcule les échelles des blocs touchés
  ! blk.1.ffn_down.weight : 1152 valeurs écrêtées aux bornes de la grille d'origine (keep-scales ne change aucune échelle) : --requant recompute recalcule les échelles des blocs touchés
[DRY-RUN]  g-down · 2 tenseurs · vecteur de 32
           blk.0.ffn_down.weight
           blk.1.ffn_down.weight
[GGUF]     <WORKDIR>\tiny-q8.gguf · GGUF v3 · qwen2 · requant keep-scales · 2 tenseurs
tenseur                 type   réalisé/voulu   corr    inchangés   saturés   nmse   
blk.0.ffn_down.weight   Q8_0   0.833           0.949   1.4%        1195      9.98e-2
blk.1.ffn_down.weight   Q8_0   0.840           0.948   1.5%        1152      9.64e-2
[VERIFY]   dry-run : rien n'a été écrit
```

(`<WORKDIR>` remplace le chemin absolu que l'outil affiche sur cette ligne.) Un décalage de `0.05` est grand pour du Q8_0, d'où l'avertissement d'écrêtage. Le même patch avec `--requant recompute` sur une copie neuve :

```bash
# même patch avec les échelles de blocs recalculées (sur une copie neuve)
cp tiny-q8.orig.gguf tiny-q8.gguf
surgeon apply --gguf --requant recompute --model tiny-q8.gguf --patch gguf-patch.json --backup-dir backups
```

```text
[APPLY]    gguf-patch.json · 1 opérations
[BACKUP]   backups\tiny-q8.gguf.bak-1794263434  ✓ sha256 vérifié
[SHIFT]    g-down terminée · Δmax 0.0505 · 2 tenseurs
[COMMIT]   modèle patché · 0.0 s
[GGUF]     <WORKDIR>\tiny-q8.gguf · GGUF v3 · qwen2 · requant recompute · 2 tenseurs
tenseur                 type   réalisé/voulu   corr    inchangés   saturés   nmse   
blk.0.ffn_down.weight   Q8_0   1.000           1.000   0.0%        0         1.47e-5
blk.1.ffn_down.weight   Q8_0   1.000           1.000   0.0%        0         1.48e-5
[VERIFY]   relecture OK : octets, échelles et valeurs finies ; en-tête inchangé (sha256 a08facefb718…)
```

| `--requant` | Ce qui change dans un bloc | À utiliser quand |
| --- | --- | --- |
| `keep-scales` (défaut) | Seuls les entiers des éléments dont la valeur patchée diffère ; les échelles sont gardées octet pour octet ; les valeurs hors grille sont écrêtées et comptées | petits décalages, de l'ordre d'un pas de quantification |
| `recompute` | Nouvelles échelles pour chaque bloc touché (Q8_0 : `quantize_row_q8_0_ref` ; Q4_K et Q6_K : min/max avec échelles arrondies vers le haut, sans écrêtage) | forts décalages ; ce n'est pas le `llama-quantize` itératif, et il n'y a pas de matrice d'importance |

Chaque tenseur patché reçoit un rapport de fidélité : `intended_shift_norm`, `realized_shift_norm`, `realized_over_intended`, `correlation_intended_realized`, `unchanged_fraction`, `saturated_elements`, `nmse_vs_exact`. Un rapport sous 0,5 ou une corrélation sous 0,8 signifie que le décalage est inférieur à un demi-pas de quantification et qu'il a été perdu ou transformé en bruit.

> **Pour les experts**
> Une écriture en place sur un fichier de plusieurs dizaines de Go est protégée en six étapes : tout est vérifié avant d'écrire (cibles, types, blocs entiers, longueurs de vecteur, espace disque libre pour la copie) ; une sauvegarde complète est faite et son SHA-256 vérifié ; un marqueur de crash `<model>.surgeon-inprogress` (chemin de la sauvegarde, SHA-256 d'origine, heure) est écrit avant le premier octet ; les tenseurs sont écrits bloc par bloc, chaque bloc étant calculé et contrôlé (échelles et valeurs finies) dans une copie d'abord, et l'en-tête, les métadonnées et la table des tenseurs ne sont jamais réécrits (leur SHA-256 est comparé avant et après) ; après le flush, un second mapping relit chaque tenseur patché et le compare au SHA-256 calculé pendant l'écriture (`--no-verify` saute cette étape) ; toute erreur restaure la sauvegarde, et le marqueur est supprimé après le commit ou le rollback. La suite de tests ne peut pas valider un fichier patché contre un vrai llama.cpp : vérifiez qu'il se charge et lancez `surgeon bench ppl` avant usage.

<a id="theory"></a>
### 3.5 Théorie telle qu'implémentée

Toute l'algèbre linéaire est en Rust pur en f64 (`src/linalg` : Jacobi cyclique, SVD fine via la plus petite matrice de Gram, Cholesky) et indépendante du nombre de threads.

- **`mean_diff`** (défaut) : `v = normalize(mu+ - mu-)` sur les activations regroupées.
- **`svd`** (contrastif) : les prompts sont appariés par index (sinon code de sortie 2, `unpaired_prompts`) ; avec `dX_i = x+_i - x-_i`, `v` est le premier vecteur singulier droit de `dX`, soit le vecteur propre dominant de `(1/N) dX^T dX`, orienté pour que `<v, mu+ - mu-> >= 0`. `dX` n'est pas centré par défaut, parce que centrer supprime la différence moyenne, qui est l'axe du concept lui-même ; `--svd-center` sert pour des paires dans un ordre arbitraire. Diagnostics : les cinq premières valeurs singulières, `sigma1^2 / sum(sigma^2)`, le cosinus avec `mu+ - mu-` et un cosinus split-half (paires paires contre impaires). Un avertissement apparaît quand `sigma2 / sigma1 > 0.9` ou avec moins de 8 paires.
- **`leace`** : `r(x) = x - a b^T (x - mu)` avec `S_l = (1 - l) S + l tr(S)/d I`, `w = S_l^(-1/2) (mu+ - mu-)`, `u = w / |w|`, `a = S_l^(1/2) u` (parallèle à `mu+ - mu-`) et `b = S_l^(-1/2) u`, de sorte que `b^T a = 1`. `a` dit où écrire et `b` ce qu'il faut lire. La `direction` écrite est `a / |a|`, donc la même direction que `mean_diff` ; `a`, `b`, `mean` et `u_whitened` vont dans `<out>.leace.safetensors`. `--leace-shrinkage 0` est la pseudo-inverse sur le sous-espace des données (`--leace-rcond`).
- **Limites honnêtes de LEACE** : la covariance croisée entre `r(x)` et l'étiquette est nulle en échantillon pour tout `S_l` défini positif. « Aucun prédicteur linéaire ne bat une constante » est une garantie de population, pas d'échantillon : avec moins de prompts que de dimensions, un classifieur peut encore séparer l'échantillon effacé. Seul `--holdout-frac` (ajuster sur le reste, mesurer sur les prompts mis de côté, rapporté en `holdout_cross_cov_ratio`) dit si l'effacement généralise.
- **`autotune`** : la lecture est `s_i = <pool(h_i^R), v>` à la couche R (site, pooling et gabarit du vecteur) ; la cible est `y_i = t_i - s_i^0` (`neg_mean_proj` pour `suppress`, `pos_mean_proj` pour `amplify`, `--target-projection`, ou un `target` par prompt) ; la réponse `F_il = (s_i(delta_l) - s_i^0) / delta` est mesurée en mémoire avec le décalage propre du moteur sur les octets stockés (arrondi bf16 compris) ; puis `alpha = (F^T F + l I)^(-1) F^T y` par Cholesky, avec `l` fixé ou choisi par leave-one-out en forme fermée (`e_i / (1 - h_ii)`, 17 points de 1e-6 à 1e2). Le R² est non centré (il n'y a pas d'ordonnée à l'origine). Les `alpha` ajustés sont appliqués ensemble et mesurés de nouveau. Les effets de bord ne sont pas mesurés : lancez `surgeon bench` ensuite.

Définitions utilisées par les rapports de `bench` :

- **KL** : `KL(before||after) = sum p_i (log p_i - log q_i)` avec `p = softmax(before)` et `q = softmax(after)`, en f64 via log-sum-exp ; la distribution « avant » est la référence. Les termes avec `p_i = 0` sont ignorés. Un token fini avant et `-inf` après rend la position infinie : elle est comptée dans `kl_infinite_positions` et exclue de `kl_mean`, `kl_max` et `kl_p95` (rang le plus proche).
- **Cosinus** : `cosine_probs_mean` compare les vecteurs de probabilités ; `cosine_logits_mean`, étiqueté à part, compare les logits bruts.
- **Top-k** : `top1_shift_rate` est la part des positions dont l'argmax change (les égalités vont à l'indice le plus bas) ; `topk_set_change_rate` la part dont l'ensemble des `k` meilleurs tokens diffère ; `topk_overlap_mean` la moyenne de `|A ∩ B| / k`.
- **Delta de perplexité** : `ppl - ppl_baseline`, arrondi à 4 décimales (ce qu'affiche llama.cpp) avant la comparaison stricte avec `--warn-delta` : `7.4 - 7.2` vaut `0.20000000000000018` en virgule flottante et ne doit pas dépasser 0,2. Le delta est `within_noise` quand `|delta| <= sqrt(err^2 + err_baseline^2)` ; un delta négatif est une amélioration, jamais un avertissement.
- **Pass@1** : correspondance exacte d'un échantillon glouton (`--temp 0 --top-k 1 --seed 42`). L'intervalle à 95 % est celui de Wilson. Avec `--baseline-model` la comparaison est appariée (identifiants `lost` et `gained`). Avec les 50 questions par défaut l'intervalle est large (40 réussies sur 50 donnent [67,0 %, 88,8 %]), donc seules les grosses régressions sont concluantes ; le rapport avertit quand moins de 100 questions ont été notées.

Exemple d'ingénierie de représentation sur le mini-modèle factice (exécutions réelles ; six paires, c'est trop peu, les avertissements le disent) :

```bash
# les deux autres méthodes d'extraction
surgeon extract --model qwen2-mini --pos-prompts pos.json --neg-prompts neg.json --layer 2 --method svd --out svd.safetensors
surgeon extract --model qwen2-mini --pos-prompts pos.json --neg-prompts neg.json --layer 2 --method leace --out erase.safetensors
```

```text
[VERDICT]  d′ 1.21 · accuracy 83% · ‖Δ‖ 3.378
[WRITE]    svd.safetensors
[METHOD]   svd · 6 paires · σ₁²/Σσ² 0.393 · cos(μ₊−μ₋) 0.812 · split-half 0.412
  ! seulement 6 paires (au moins 8 recommandées) : direction SVD peu fiable
```

```text
[VERDICT]  d′ 2.23 · accuracy 92% · ‖Δ‖ 3.378
[WRITE]    erase.safetensors
[METHOD]   leace · rang 11 · λ 0.1 · κ 5.033e1 · cov(x,z) 8.444e-1 → 9.449e-16 · hors-échantillon —
[WRITE]    erase.safetensors.leace.safetensors
  ! n = 12 ≤ d = 16 : la covariance croisée s'annule en échantillon, mais un classifieur linéaire peut encore séparer ; seul holdout_cross_cov_ratio dit si l'effacement généralise
```

Le d′ de `leace` est égal à celui de `mean_diff` : même direction, comme le dit la théorie. La covariance `cov(x,z)` passe de 0,84 à environ 1e-15 en échantillon, tandis que `hors-échantillon —` dit qu'aucun jeu de côté n'a été demandé. Application de l'effaceur sur une copie (`project.json` nomme les tenseurs `a` et `b`) :

```json
{"schema_version": 1, "backup": false, "operations": [{"id": "erase-L1", "layers": "1", "projections": ["down_proj"], "operation": "project_out", "vector_file": "erase.safetensors.leace.safetensors", "vector_tensor": "a", "vector_tensor_b": "b", "alpha": 1.0}]}
```

```bash
# efface la direction dans une copie, puis ajuste des coefficients par couche
cp -r qwen2-orig qwen2-erased
surgeon apply --model qwen2-erased --patch project.json --no-backup
surgeon autotune --model qwen2-mini --vector direction.safetensors --eval-dataset pos.json --target-layer-range 0-3 --l2-reg auto --out tuned.json
```

```text
[APPLY]    project.json · 1 opérations
[PROJECT]  erase-L1 terminée · effacement a(bᵀW) · Δmax 0.3463 · 1 tenseurs
[COMMIT]   modèle patché · 0.0 s
```

```text
[AUTOTUNE] lecture L2 · resid_post · last · 6 prompts · 3 couches · 48 forwards
[RESPONSE] δ 0.05 · s⁰ moyen 1.2984 → cible -2.0793 · linéarité (écart max) 1.8 %
[RIDGE]    λ 1.0e2 · R² 0.0120 · RMSE 3.6374 · LOO 4.1374 · κ 1.32e0 · ‖α‖ 0.0658
  α : L0 +0.0183 · L1 -0.0596 · L2 -0.0210
[VERDICT]  décalage prévu -0.0371 · mesuré -0.0354 · écart à la cible 3.6594 → 3.6390
[WRITE]    tuned.json
  ! couches [3] au-delà de la lecture (couche 2) : sans effet sur elle, exclues
  effets secondaires non mesurés ici : appliquer sur une copie puis surgeon bench (dump, logits, ppl)
```

Sur ce mini-modèle aléatoire, les réponses ne généralisent pas d'un prompt à l'autre : la recherche leave-one-out choisit donc le `l` le plus fort et l'ajustement explique environ 1 % du décalage (R² 0,012). C'est une réponse honnête, pas un bug : le décalage mesuré (-0,0354) reste proche de la prédiction (-0,0371). `tuned.json` est un patch ordinaire (`backup: true`, un `subtract` par couche).

<a id="performance"></a>
### 3.6 Performances

`apply` décale avec des noyaux SIMD choisis à l'exécution : AVX-512F, AVX2 avec F16C (x86_64), NEON (aarch64), sinon scalaire. `SURGEON_KERNEL=scalar|avx2|avx512|neon` force le choix ; un niveau indisponible retombe sur le meilleur avec un avertissement, si bien qu'une instruction illégale est inatteignable. Chaque niveau est conçu pour écrire les mêmes octets que le chemin scalaire : chaque élément coûte une addition f32 `old + delta` suivie du même arrondi au plus proche pair vers bf16 ou f16, avec la règle NaN de `half`.

> **Pour les experts**
> Il n'y a aucun fused multiply-add nulle part. Le produit `alpha * v[i]` est calculé une fois par élément du vecteur, le noyau ne fait qu'additionner. Un fused multiply-add arrondit une fois au lieu de deux et dériverait de la référence scalaire d'un dernier bit ; et l'opération est de toute façon limitée par la mémoire, fusionner n'apporterait rien. Les tenseurs sont écrits par blocs (`--prefetch-chunk-mib`, 64 par défaut) avec des indications au système (`madvise` sous Unix, `PrefetchVirtualMemory` sous Windows) et un flush asynchrone du bloc précédent ; `--no-prefetch` désactive les deux. `data.io` de `apply` rapporte octets, blocs, indications échouées, durée, débit et défauts de page.

Les seuls chiffres cités ici viennent d'une exécution de `examples/bench_kernels.rs` faite pour ce README le 2026-11-09, avec cette commande (elle écrit un fichier temporaire de 2 GiB et dure environ 90 secondes) :

```bash
# banc des noyaux et des E/S, affiche des tableaux Markdown
cargo run --release --example bench_kernels
```

Machine : Intel Core i5-12400F (6 cœurs, 12 threads, AVX2 et F16C, pas d'AVX-512), 15,9 Go de RAM, Windows 11 Pro, 12 threads rayon. En mémoire, tampon de 1 GiB, 4096 colonnes ; un `memcpy` parallèle du même tampon a mesuré 17 463 MiB/s (13 676 MiB/s sur un thread) :

| niveau | dtype | axe `rows` MiB/s | axe `cols` MiB/s | face au memcpy parallèle (rows) |
| --- | --- | ---: | ---: | ---: |
| scalar | bf16 | 3 944 | 3 975 | 0.23 |
| scalar | f16 | 2 701 | 2 592 | 0.15 |
| scalar | f32 | 13 027 | 13 241 | 0.75 |
| avx2 | bf16 | 17 970 | 17 715 | 1.03 |
| avx2 | f16 | 17 601 | 18 077 | 1.01 |
| avx2 | f32 | 17 070 | 16 977 | 0.98 |

AVX-512 et NEON n'ont pas été exécutés : ce CPU n'a ni l'un ni l'autre. De bout en bout (`shift_tensor` plus flush sur un fichier bf16 de 2 GiB déjà dans le cache de pages) la même exécution a donné 99 à 119 MiB/s avec préchargement et 110 à 112 MiB/s sans, avec environ 525 000 défauts de page dans les deux cas : sur cette machine l'écriture différée des pages sales du fichier mappé domine, pas les noyaux, et le préchargement ne supprime pas les défauts de page sous Windows. Une exécution précédente du même exemple sur la même machine a donné environ 180 à 240 MiB/s : considérez donc les chiffres de bout en bout comme dépendants du disque, du cache de pages et du système ; relancez l'exemple sur votre machine. L'E/S directe (`O_DIRECT`) n'est pas implémentée.

Les rapports de `bench logits` et `bench eval` n'ont pas été mesurés pour ce README : aucun débit n'est annoncé pour eux.

<a id="mcp"></a>
## 4. Agents et MCP

<a id="what-is-mcp"></a>
### 4.1 Ce qu'est MCP

Le Model Context Protocol (MCP) permet à un assistant IA d'appeler des outils externes via une interface JSON-RPC standard. `surgeon mcp` est un serveur MCP local sur stdio : le client le lance comme sous-processus et appelle des outils typés. Il expose le même moteur que la ligne de commande, avec la même enveloppe `--json` comme résultat, mais seulement dans les dossiers que vous autorisez.

<a id="tools"></a>
### 4.2 Outils

`surgeon mcp` sert neuf outils (réponse réelle de `tools/list`, aussi affichée par `surgeon schema --kind tools`) :

| Outil | Rôle |
| --- | --- |
| `surgeon_check` | Budget RAM et VRAM d'un modèle. |
| `surgeon_inspect` | Couches et matrices MLP, statistiques facultatives. |
| `surgeon_apply` | Applique un patch en place (`dry_run` pour vérifier d'abord) ; `gguf: true` pour un GGUF quantifié. |
| `surgeon_eval` | Contrôles de boucles et d'attentes, export GGUF conditionnel. |
| `surgeon_extract` | Vecteur de steering depuis une couche Qwen2. |
| `surgeon_bench` | Mesure, avec `mode` `dump`, `logits`, `ppl` ou `eval`. |
| `surgeon_autotune` | Coefficients par couche par régression ridge. |
| `surgeon_diff` | Différence de deux checkpoints écrite dans un nouveau `.surgery`. |
| `surgeon_lora` | `mode` `merge` (adaptateur fusionné en place) ou `extract` (adaptateur écrit). |

Une vraie session sur le faux modèle : le client envoie `initialize`, `notifications/initialized`, puis un `tools/call`. Le serveur écrit une ligne sur stderr, puis répond sur stdout (la réponse à `initialize` est omise, et les lignes de réponse sont coupées) :

```bash
# démarre le serveur sur le dossier courant, en lecture seule
surgeon mcp --allow-dir . --read-only
```

```text
surgeon mcp 0.1.0 : prêt sur stdio, 1 dossier(s) autorisé(s), lecture seule
{"id":2,"jsonrpc":"2.0","result":{"content":[{"text":"{\"ok\":true,\"command\":\"check\",\"exit_code\":0,\"data\":{\"available_ram_bytes\":4787773440,\"best_gpu\":null,\"model_bytes\":55528,\"required_ram_bytes\":2147547506,\"required_vram_bytes\":null,\"verdict\":{\"kind\":\"fits_in_ram_only\"},[...]
{"id":3,"jsonrpc":"2.0","result":{"content":[{"text":"{\"ok\":false,\"command\":\"inspect\",\"exit_code\":2,\"data\":null,\"error\":{\"message\":\"« model » : chemin hors des dossiers autorisés (--allow-dir)\",\"kind\":\"path_not_allowed\"},\"warnings\":[]}","type":"text"}],"isError":true,[...]
```

Le premier appel (`surgeon_check` sur `qwen2-mini`) réussit ; le second demande `../elsewhere`, hors du dossier autorisé, et est refusé avec `path_not_allowed`.

<a id="guardrails"></a>
### 4.3 Garde-fous

- `--allow-dir` (répétable, sinon `SURGEON_MCP_ALLOW_DIRS`, sinon le dossier courant) : chaque chemin est résolu par le système (liens symboliques, `..`) et doit tomber sous un dossier autorisé ; les chemins relatifs le sont par rapport au premier.
- `--read-only` : refuse tout outil qui écrit (`surgeon_apply`, `surgeon_extract`, exports GGUF, `surgeon_bench` en mode `dump`, `surgeon_autotune` avec `out`, `surgeon_diff`, `surgeon_lora`).
- Sauvegarde imposée : une sauvegarde vérifiée par SHA-256 précède toute écriture, et une sauvegarde échouée bloque l'écriture. `backup: false` dans un patch n'est respecté qu'avec `--allow-no-backup`.
- Annulation : `notifications/cancelled` arrête l'appel en cours, annule un patch partiel et tue le processus llama.cpp en cours ; une extraction ou un dump annulé ne laisse aucun fichier.
- Progression : avec un `progressToken`, une `notifications/progress` est envoyée par tenseur, prompt ou bloc.
- Un panic pendant un outil devient une erreur JSON-RPC et le serveur continue de tourner.

Détails et versions du protocole acceptées : [docs/MCP.md](docs/MCP.md) (rédigé en français).

<a id="client-config"></a>
### 4.4 Configuration des clients

Les chemins ci-dessous viennent de la documentation officielle de chaque client, lue le 2026-11-09 ; vérifiez la documentation de la version de votre client si quelque chose diffère. Remplacez `<MODELS_DIR>` par le dossier absolu qui contient vos modèles.

Claude Code (portée utilisateur ici ; utilisez `--scope project` pour partager un `.mcp.json` avec une équipe) :

```bash
# enregistre le serveur ; tout ce qui suit le double tiret est la commande qui le lance
claude mcp add --scope user surgeon -- surgeon mcp --allow-dir <MODELS_DIR>
```

Claude Desktop, dans `claude_desktop_config.json` (`%APPDATA%\Claude\claude_desktop_config.json` sous Windows, `~/Library/Application Support/Claude/claude_desktop_config.json` sous macOS). Redémarrez l'application après l'enregistrement. Les antislashs des chemins Windows sont doublés en JSON (ici le dossier est `C:\models`) :

```json
{
  "mcpServers": {
    "surgeon": {
      "command": "surgeon",
      "args": ["mcp", "--allow-dir", "C:\\models"]
    }
  }
}
```

Cursor, dans `.cursor/mcp.json` (un projet) ou `~/.cursor/mcp.json` (global) :

```json
{
  "mcpServers": {
    "surgeon": {
      "type": "stdio",
      "command": "surgeon",
      "args": ["mcp", "--allow-dir", "<MODELS_DIR>"]
    }
  }
}
```

Si le client ne trouve pas `surgeon`, mettez le chemin absolu du binaire dans `command`. Claude Code peut aussi charger le dossier [`plugin/`](plugin) comme plugin (même serveur, `allow_dir` demandé à l'activation, plus un skill de démarche sûre ; voir [docs/PLUGINS.md](docs/PLUGINS.md)).

<a id="agent-recipe"></a>
### 4.5 Une recette d'agent sûre

1. Copiez le modèle dans un dossier autorisé. L'agent ne touche jamais l'unique copie.
2. `surgeon_check` sur la copie.
3. `surgeon_bench` avec `mode: "dump"` sur la copie : les logits de référence.
4. `surgeon_extract` d'une direction (au moins huit prompts par jeu).
5. `surgeon_apply` avec `dry_run: true`, lire les tenseurs listés.
6. `surgeon_apply` pour de bon ; la sauvegarde est imposée.
7. `surgeon_bench` `dump` de nouveau, puis `logits`, puis `ppl` et `eval` si llama.cpp est disponible.
8. Décider d'après les chiffres mesurés. Si c'est mauvais, restaurer la sauvegarde. N'annoncer aucun résultat qui n'a pas été mesuré.

C'est ce que le skill `safe-surgery-workflow` du plugin Claude Code dit à l'agent. La même recette depuis un shell, avec les commandes du [chemin A](#path-a) : `check`, `extract`, `apply --dry-run`, `apply`, `bench dump`, `bench logits`.

<a id="llama-cpp"></a>
### 4.6 llama.cpp : ce qui est utilisé et comment il est trouvé

llama.cpp est facultatif et jamais fourni. Il n'est utilisé que par `eval` et `bench ppl|eval`, et pour l'export GGUF. Les binaires et scripts recherchés par le code :

| Utilisé par | Fichier recherché |
| --- | --- |
| `eval` | `llama-cli` |
| `eval --export-gguf` | `convert_hf_to_gguf.py` (ou l'ancien `convert-hf-to-gguf.py`) et ses dépendances Python, puis `llama-quantize` (ou `quantize`) |
| `bench ppl` | `llama-perplexity` (ou l'ancien `perplexity`) |
| `bench eval` | `llama-completion`, sinon `llama-cli` |

Ordre de recherche : le dossier donné par `--llama-cpp-dir`, puis le dossier de la variable d'environnement `LLAMA_CPP_DIR` (dans les deux, les sous-dossiers `build/bin`, `build/bin/Release` et `build` sont aussi essayés), puis les dossiers du `PATH`.

```bash
# bash : passer le dossier à chaque commande, ou l'exporter pour la session
export LLAMA_CPP_DIR=$HOME/llama.cpp
surgeon bench ppl --model model.gguf --text text.txt --llama-cpp-dir $HOME/llama.cpp
```

```powershell
# PowerShell
$env:LLAMA_CPP_DIR = "$env:USERPROFILE\llama.cpp"
```

Par MCP, c'est l'opérateur qui définit `LLAMA_CPP_DIR` pour le serveur ; l'agent ne le choisit pas.

<a id="architecture"></a>
### 4.7 Architecture et plugins

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (en français) définit le vocabulaire : le Surgery Engine au centre, le CLI et le serveur MCP comme front-ends, et les points d'extension. En bref, `surgeon` est découpé en modules (`hardware`, `engine`, `patch`, `extract`, `bench`, `eval`, `kernel`, `linalg`, `lora`, `surgery`, `tui`, `mcp`, `plugin`). Un seul point d'extension existe aujourd'hui, `ExtractionMethod` (API `v0`, instable) : un trait Rust dont les implémentations sont enregistrées statiquement dans un programme qui lie la bibliothèque. Le binaire `surgeon` n'enregistre que les trois méthodes intégrées, et `surgeon plugins list` les montre. [docs/PLUGINS.md](docs/PLUGINS.md) (en français) explique pas à pas comment écrire une méthode et décrit une proposition de plugins hors processus qui n'est pas implémentée.

<a id="faq"></a>
## 5. FAQ, sécurité et dépannage

<a id="hardware-budget"></a>
### 5.1 Budget mémoire et verdicts de `check`

`surgeon check` estime, sans rien charger : RAM nécessaire = taille du modèle x 1,15 + 2 GiB de marge (`--policy-ram-margin-gib` change la marge), VRAM nécessaire = taille du modèle x 1,20. Verdicts : `FITS IN VRAM` (un GPU a assez de mémoire libre ; en mémoire unifiée, le plus strict des deux besoins est vérifié une fois), `FITS IN RAM ONLY` (repli CPU, plus lent), `INSUFFICIENT` (code de sortie 1, avec la raison). L'avertissement NVML signifie que la bibliothèque NVIDIA n'était pas utilisable, donc les GPU sont ignorés. La plupart des commandes, `apply` compris, travaillent par mappage mémoire et ont peu besoin de RAM ; `extract` et `bench dump` chargent les poids (`--dtype bf16` divise la mémoire par deux ; `auto` bascule seul en bf16, avec un avertissement).

<a id="backups"></a>
### 5.2 Sauvegardes et rollback automatique

- `apply`, `lora merge` et les outils MCP copient chaque fichier touché dans le dossier de sauvegarde et vérifient son SHA-256 avant d'écrire. Le dossier de sauvegarde doit exister.
- En cas d'erreur, de NaN ou d'Inf dans un résultat, ou d'annulation (Ctrl-C, ou `notifications/cancelled` en MCP), les octets d'origine sont restaurés et le code de sortie est 3 (4 si le rollback lui-même a échoué, ou s'il n'y avait pas de sauvegarde et que le modèle est partiellement patché).
- GGUF : un marqueur de crash `<model>.surgeon-inprogress` est écrit avant le premier octet et supprimé après le commit ou le rollback. S'il est toujours là (crash, coupure de courant), `apply` refuse de s'exécuter : copiez la sauvegarde nommée dans le marqueur par-dessus le modèle, comparez son SHA-256 à `original_sha256`, puis supprimez le marqueur.
- Il n'y a pas de commande `undo` : restaurer, c'est copier le fichier de sauvegarde, comme dans le [chemin A](#path-a). Appliquer l'`alpha` opposé n'est pas un retour exact, à cause des arrondis.

<a id="windows-lock"></a>
### 5.3 Windows : `surgeon.exe` verrouillé par un client MCP

Un client MCP garde `surgeon.exe` en cours d'exécution, donc `cargo install --path .` échoue avec « Access denied ». Fermez le client, ou renommez le binaire en cours d'exécution (Windows l'autorise) puis réinstallez :

```powershell
# Windows : renomme le binaire en cours d'exécution
Rename-Item "$env:USERPROFILE\.cargo\bin\surgeon.exe" surgeon.old.exe
```

<a id="limits"></a>
### 5.4 Limites connues

- `extract`, `autotune`, `bench dump` et l'onglet de prévisualisation ne gèrent que Qwen2 et Qwen2.5, sur le CPU ; les gros modèles demandent beaucoup de RAM.
- La qualité d'une direction dépend de vos prompts : avec peu de paires une direction mélange des concepts (l'outil avertit sous 8 prompts par jeu). Un bon `d′` sur les prompts d'extraction ne prouve pas que la direction généralise.
- Les patchs plus petits qu'un pas de quantification sont perdus dans les fichiers GGUF quantifiés ; les décalages hors de la plage du bloc sont écrêtés (`keep-scales`). Vérifiez `realized_over_intended`.
- `project_out` couvre les écrivains du flux résiduel d'un modèle SafeTensors (`down_proj`, `o_proj`, `embed_tokens`) ; dans un GGUF seules les projections MLP sont patchables. `bench refusal` n'a pas de modèle juge, seulement des marqueurs de refus anglais : lisez les réponses. L'effacement LEACE n'est exact qu'en échantillon ; avec moins de prompts que de dimensions, vérifiez `holdout_cross_cov_ratio`. `autotune` ajuste et mesure sur les mêmes prompts et ne mesure pas les effets de bord.
- `svd` et `leace` reposent sur Jacobi, limité à 1024 sur le plus petit côté (prompts ou taille cachée).
- Les échantillons de benchmark sont petits par défaut (`bench eval --limit 50`) : lisez l'intervalle de confiance.
- Les dumps de logits en f16 ajoutent du bruit d'arrondi ; préférez le f32 par défaut.
- Le déport GPU de `bench ppl` peut changer les derniers chiffres d'une exécution à l'autre ; le `--n-gpu-layers 0` par défaut est reproductible.
- `apply --gguf` : GGUF en un seul fichier little endian uniquement ; pas de matrice d'importance ; la requantification est scalaire (lignes en parallèle). Elle n'a pas été validée contre un vrai llama.cpp par la suite de tests.
- Le tableau de bord demande un terminal interactif d'au moins 80x24 ; sinon utilisez `--snapshot`.
- Les messages du terminal sont en français.
- Aucun fichier de licence n'est publié : tant qu'il n'en est pas ajouté un, le droit d'auteur par défaut s'applique (tous droits réservés). Le README ne porte aucun badge de licence pour cette raison.

<a id="responsible-use"></a>
### 5.5 Usage responsable

Le steering et la chirurgie de poids changent ce que fait un modèle, y compris ses comportements de sécurité : une direction peut retirer des refus aussi facilement qu'en ajouter. Travaillez sur des copies, évaluez après chaque patch, respectez la licence de chaque modèle que vous modifiez, gardez les sauvegardes, et ne distribuez pas un modèle dont vous avez retiré les garde-fous. Un modèle modifié peut se dégrader ou changer d'une façon que vos tests n'ont pas couverte.

<a id="troubleshooting"></a>
### 5.6 Dépannage

| Symptôme | Cause probable | Action |
| --- | --- | --- |
| `erreur : backup : dossier de backup inexistant` (code 3) | `--backup-dir` n'existe pas | Créez d'abord le dossier. |
| `est un fichier GGUF : relancez avec --gguf` (code 2) | Un `.gguf` a été donné sans `--gguf` | Ajoutez `--gguf` ; l'arrondi est avec perte et jamais silencieux. |
| `--no-backup est refusé avec --gguf` (code 2) | La copie de sauvegarde est obligatoire pour GGUF | Retirez-le, ou utilisez `--force-no-backup` en sachant qu'un échec laisse le fichier modifié. |
| `une application précédente a été interrompue` | Un marqueur de crash reste à côté du GGUF | Restaurez la sauvegarde nommée dans le marqueur, comparez le SHA-256, supprimez le marqueur. |
| `` `surgeon tui` nécessite un terminal interactif `` (code 1) | stdin, stdout ou stderr n'est pas un terminal | Lancez dans un vrai terminal, ou utilisez `--snapshot`. |
| `Terminal trop petit (80×24 minimum)` | Terminal ou `--size` sous 80x24 | Agrandissez-le. |
| `erreur : fichier de prompts ... JSON invalide` (code 2) | Les prompts d'`eval` doivent être des objets avec `text` | Utilisez `[{"id": "a", "text": "..."}]` ; les fichiers de prompts d'extraction sont des chaînes simples. |
| `--llama-cpp-dir (ou LLAMA_CPP_DIR) requis` (code 1) | llama.cpp introuvable | Compilez llama.cpp et passez son dossier ([ordre de recherche](#llama-cpp)). |
| `ressources insuffisantes` (code 1) | Le contrôle de budget a échoué | `--dtype bf16`, un modèle plus petit, ou fermez d'autres programmes. |
| Avertissement `NVML indisponible` | Bibliothèque NVIDIA inutilisable | Sans danger : les GPU sont ignorés et le modèle est prévu en RAM. |
| `f32 ne tient pas en RAM : calcul en bf16` | Peu de RAM libre | Attendu ; libérez de la mémoire pour un dump en pleine précision. |
| `Access denied` sur `cargo install` (Windows) | `surgeon.exe` toujours en cours d'exécution | Voir [5.3](#windows-lock). |
| Cadres ou blocs cassés | Police ou encodage | Utilisez un terminal UTF-8 avec les caractères de tracé de cadres (Windows Terminal). |
| Un seuil fait échouer l'exécution (code 9) | `--fail-on-warn` et un seuil dépassé | Lisez le rapport dans `data` ; assouplissez le seuil ou corrigez le patch. |

<a id="exit-codes-faq"></a>
### 5.7 Codes de sortie

Voir le [tableau de la section 3.1](#envelope).

<a id="contributing"></a>
### 5.8 Contribuer

```bash
# les quatre doivent passer avant de proposer un changement
cargo check --all-targets
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

`tests/deps_purity.rs` rejette les crates connues pour compiler du C dans `Cargo.lock`. `tests/readme_check.rs` garde les deux README honnêtes : les deux fichiers ont la même structure et les mêmes commandes, chaque commande, option et outil MCP cité existe, chaque commande et chaque outil est documenté, et les liens et blocs de code sont valides. Les commits suivent `type: description` (`feat`, `fix`, `refactor`, `docs`, `chore`).

<a id="roadmap"></a>
### 5.9 Non implémenté

Ces éléments ne sont pas disponibles aujourd'hui et ne sont pas documentés comme des fonctionnalités : plugins hors processus, opérations de patch et métriques de bench ouvertes, adaptateurs de format autres que ceux intégrés, adaptateur PyTorch, noyaux SIMD chargeables, et fusion d'une LoRA dans un fichier GGUF. Voir le tableau de feuille de route de [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
