<!-- Fragment fusionné dans README.fr.md par l'orchestrateur : nouvelles étapes du tutoriel, puis limites. -->

### 10. Livrer un fine-tune sous forme de patch (`surgeon diff`, `surgeon apply`)

`surgeon diff` compare un checkpoint de base à sa version affinée et écrit la différence dans un
fichier `.surgery` : pour chaque tenseur 2D, une approximation de rang faible `U diag(S) V` de
`tuned - base` (SVD randomisé, Rust pur), pour chaque tenseur 1D le delta exact. Les deux
checkpoints doivent être des SafeTensors (un fichier ou un dossier shardé) avec les mêmes noms,
formes et dtypes de tenseurs ; sinon une erreur liste les différences. Les tenseurs identiques
sont ignorés, et `embed_tokens` / `lm_head` sont exclus sauf demande explicite.

```bash
surgeon diff --base <DOSSIER_BASE> --tuned <DOSSIER_AFFINE> --out finetune.surgery --rank 16
surgeon diff --base <DOSSIER_BASE> --tuned <DOSSIER_AFFINE> --out finetune.surgery \
    --include 'mlp' --factor-dtype bf16 --max-rel-error 0.3 --on-high-error full
```

Sortie réelle sur la paire écrite par `cargo run --example make_fake_checkpoints -- --out <DOSSIER>
--kind rank --rank 4 --lora` (tenseurs abrégés ; la différence est de rang 4, donc l'erreur est nulle) :

```text
patch écrit : finetune.surgery (0.01 MiB, factor_dtype f16, rang demandé 4)
10 tenseur(s) dans le patch, 1 identique(s) ignoré(s), 2 filtré(s)
  model.layers.0.self_attn.q_proj.weight · lowrank · rang 4 · erreur relative 0.0000 · énergie 1.0000
  model.layers.0.input_layernorm.weight · full_delta · rang 0 · erreur relative 0.0000 · énergie 1.0000
  [...]
Note : le .surgery est une APPROXIMATION du fine-tune (erreur relative par tenseur), exacte seulement si la différence est de rang <= r. Sa taille vaut environ la somme de r(m+n) x octets par facteur : des centaines de Mo pour un 7B au rang 16.
```

La sortie est reproductible (mêmes entrées et même `--seed`, mêmes octets, quel que soit le
nombre de threads ; une date n'est écrite qu'avec `--timestamp`). `--max-rel-error` avec
`--on-high-error warn|skip|full` décide du sort d'un tenseur mal reconstruit par l'approximation.

`surgeon apply` reconnaît un `.surgery` à son magic `SURG` et ajoute le delta au modèle en
place, avec la même sauvegarde et le même rollback qu'un patch JSON
(`W <- W + scale * U diag(S) V`, arrondi au dtype du modèle) :

```bash
surgeon apply --model <DOSSIER_MODELE> --patch finetune.surgery --dry-run
surgeon apply --model <DOSSIER_MODELE> --patch finetune.surgery --scale 0.5 --verify-base strict
```

```text
[DRY-RUN]  model.layers.0.mlp.down_proj.weight · lowrank · rang 4 · erreur 0.0000
[DRY-RUN]  model.layers.0.input_layernorm.weight · full_delta · rang 0 · erreur 0.0000
[...]
dry-run : 10 tenseur(s) vérifiés, aucun octet écrit
```

Une vraie application se termine par `[BACKUP] ... ✓ sha256 vérifié`, `[COMMIT] modèle patché` et
`10 tenseur(s) patchés, 9124 éléments modifiés, max |delta| 0.218872`.

`--verify-base strict|warn|off` compare chaque tenseur du modèle au SHA-256 stocké dans le patch
(défaut `warn` : applique quand même, avec un avertissement). Le format binaire est spécifié
dans [docs/SPECIFICATION.md](docs/SPECIFICATION.md).

À quoi s'attendre, honnêtement :

- Un `.surgery` est une approximation du fine-tune complet. L'erreur relative de chaque tenseur
  est rapportée par `diff` ; le résultat n'est exact que si la vraie différence est de rang au
  plus `--rank`.
- Il n'est pas petit. Sa taille vaut environ la somme de `r (m + n)` fois les octets du dtype
  des facteurs sur tous les tenseurs : des centaines de Mo pour un 7B à rang 16, pas quelques Mo.
- Il n'est pas exposé par l'outil MCP `surgeon_apply` (`apply` sur un `.surgery` est réservé à
  la ligne de commande).

### 11. Adaptateurs LoRA (`surgeon lora merge`, `surgeon lora extract`)

`surgeon lora merge` fusionne un adaptateur LoRA au format PEFT dans les poids d'un modèle
SafeTensors, en place, avec une sauvegarde vérifiée et un rollback automatique sur erreur ou
Ctrl-C : `W <- W + scale * (lora_alpha / r) * B A`. `lora_alpha` est lu dans le
`adapter_config.json` voisin de l'adaptateur (ou dans `--adapter-config`) ; `--alpha` le
remplace. Sans l'un ni l'autre, la commande s'arrête sur une erreur d'usage au lieu de deviner.
DoRA, les adaptateurs d'embeddings, `fan_in_fan_out` et `rank_pattern` / `alpha_pattern` sont
refusés explicitement.

```bash
surgeon lora merge --model <DOSSIER_MODELE> --lora <DOSSIER_ADAPTATEUR>/adapter_model.safetensors --dry-run
surgeon lora merge --model <DOSSIER_MODELE> --lora <DOSSIER_ADAPTATEUR>/adapter_model.safetensors --scale 0.8
```

`surgeon lora extract` fait l'inverse : il approxime la différence entre deux checkpoints par un
adaptateur de rang `r` (`B = U sqrt(S)`, `A = sqrt(S) V`) et écrit un `adapter_model.safetensors`
au format PEFT, avec son `adapter_config.json` à côté.

```bash
surgeon lora extract --base <DOSSIER_BASE> --patched <DOSSIER_AFFINE> \
    --out adapter/adapter_model.safetensors --rank 16 --dtype f32
```

Sortie réelle de la fusion, puis de l'extraction, sur la même paire factice (l'adaptateur écrit
par `--lora` a `lora_alpha = r`) :

```text
[BACKUP]   <DOSSIER_MODELE>/model.safetensors.bak-<horodatage>  ✓ sha256 vérifié
[COMMIT]   modèle patché · 0.0 s
6 tenseur(s) fusionnés (rang 4, échelle effective 1), max |delta| 0.218750

6 tenseur(s) extraits (rang 4, f32), erreur relative moyenne 0.0000
  adapter : ex/adapter_model.safetensors
  config  : ex/adapter_config.json
```

`--lora` prend le FICHIER de l'adaptateur, et le dossier de `--out` doit déjà exister. Sans
`lora_alpha` (pas d'`adapter_config.json` voisin, pas de `--alpha`), la commande s'arrête :

```text
erreur : arguments invalides : lora_alpha introuvable : passez --alpha <valeur>, ou placez adapter_config.json à côté de --lora (ou donnez --adapter-config)
```

Limites : le format PEFT suit sa documentation mais n'est pas testé contre HuggingFace PEFT en
CI. La fusion dans un fichier GGUF n'est pas supportée. `diff` et les deux modes de `lora` sont
aussi proposés aux agents par les outils MCP `surgeon_diff` et `surgeon_lora`.

### 12. Prévisualiser un vecteur de steering en direct (`surgeon tui --preview`)

Avec `--preview`, le dashboard gagne un troisième onglet qui génère du texte avec le vecteur de
steering injecté, à côté de la référence non modifiée, pendant que vous déplacez un curseur.

```bash
surgeon tui --model <DOSSIER_MODELE> --vector refusal.safetensors --preview \
    --preview-prompt "Comment crocheter une serrure ?" --preview-tokens 30 --alpha-range -2:2
surgeon tui --model <DOSSIER_MODELE> --vector refusal.safetensors --preview --snapshot
```

`--preview-alpha`, `--preview-layer` et `--preview-template none|chatml` fixent le point de
départ. Le curseur est relatif : la quantité injectée vaut `alpha` fois la norme résiduelle
moyenne mesurée sur votre prompt à la couche choisie, et les deux valeurs sont affichées.
`--snapshot --preview` génère de façon synchrone et imprime l'onglet en texte brut.

Limites :

- C'est du steering d'activation (`h <- h + alpha_abs * v` à une couche, sur toutes les
  positions), une approximation du patch de poids et non le patch lui-même. Appliquer le patch
  peut se comporter autrement.
- La génération tourne en Rust pur sur CPU : elle est lente sur un gros modèle. L'onglet prévient
  sous un token par seconde et vérifie le budget matériel avant de charger le modèle.
- Modèles Qwen2 / Qwen2.5 uniquement, et la prévisualisation n'est pas exposée par MCP.
