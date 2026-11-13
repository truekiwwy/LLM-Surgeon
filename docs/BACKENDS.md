# Backends d'inférence et configuration utilisateur

Qui génère les réponses de `surgeon bench eval`, `bench refusal`, `surgeon pipeline` et des outils MCP
`surgeon_bench` / `surgeon_pipeline`, comment ce choix est mémorisé, et comment une interface
graphique s'en sert au premier démarrage. Code : `src/backend/`.

## 1. Les trois backends

| Backend | Matériel | Ce que c'est | État |
| --- | --- | --- | --- |
| `native` | CPU | Générateur en Rust pur de `surgeon` (Qwen2, Llama/Mistral, Qwen3 en SafeTensors). Aucun programme externe. | Repli, défaut historique de `bench refusal` et du pipeline. |
| `llama-server` | GPU (Vulkan, CUDA, Metal...) ou CPU | Un serveur `llama.cpp` local pour toute l'exécution. Un dossier SafeTensors Qwen2, Llama, Mistral, Gemma, Gemma 2 ou Phi-3 est converti par `surgeon` lui-même en GGUF (Rust pur, sans Python). | Backend GPU principal. |
| `hipfire` | GPU AMD, HIP/ROCm, Linux | Moteur externe [`warpfront/hipfire`](https://github.com/warpfront/hipfire). | Reconnu et vérifié, pilote non branché (section 6). |

Les trois partagent l'interface `TextGenerator` (`src/eval/generator.rs`) : un bench ne sait pas
lequel il utilise. `src/backend/` ajoute les réglages communs, leur ordre de priorité, la sonde de
la machine et les libellés des rapports. Le chemin de calcul n'a pas changé : `llama-server` est le
même pilote qu'avant (liste blanche d'options, port libre, clé aléatoire, serveur tué avec le
bench), le convertisseur est `src/eval/safetensors_gguf.rs`.

## 2. Réglages et priorité

`BackendConfig` (`src/backend/config.rs`) : `backend`, `gpu_layers`, `batch_size`, `llama_cpp_dir`,
`hipfire_bin`, `gguf_dtype`. Chaque champ est facultatif ; une couche ne dit que ce qu'elle sait.

```text
option de la ligne de commande  >  "backend" de pipeline.json  >  config.toml utilisateur  >  défaut
```

Les défauts sans configuration sont ceux d'avant : `bench eval` garde `llama`, `bench refusal` et
`pipeline` gardent `native`. Deux règles :

- `gpu_layers` et `batch_size` du fichier utilisateur ne valent que pour `llama-server` et
  `hipfire` : ils ne changent pas le découpage en lots du moteur natif. Une option de la ligne de
  commande, elle, s'applique toujours.
- `$LLAMA_CPP_DIR` reste entre l'option `--llama-cpp-dir` et le fichier utilisateur.

Options concernées : `--backend`, `--gpu-layers`, `--batch-size`, `--llama-cpp-dir`,
`--hipfire-bin` (`bench eval`, `bench refusal`, `pipeline plan`, `pipeline run`) et `--gguf-dtype`
(`bench eval`, `bench refusal`).

## 3. Configuration persistante

Fichier `config.toml` :

| Système | Emplacement |
| --- | --- |
| Windows | `%APPDATA%\weight-surgeon\config.toml` |
| Linux, macOS | `$XDG_CONFIG_HOME/weight-surgeon/config.toml`, sinon `~/.config/weight-surgeon/config.toml` |

```toml
# weight-surgeon: backend d'inférence
default_backend = "llama-server"
llama_cpp_dir = "C:\\Users\\<vous>\\tools\\llama-vulkan"
gpu_layers = 99
batch_size = 1
gguf_dtype = "f16"
```

Sous-ensemble plat de TOML (`clé = "texte"` ou `clé = 123`, commentaires `#`, chaînes `'...'`
sans échappement acceptées), lu et écrit sans crate TOML. Une clé inconnue ou une valeur invalide
est une erreur nommant le fichier et la ligne, et arrête les commandes qui génèrent (les autres,
`inspect` ou `check`, ne lisent pas le fichier). `surgeon backend show` affiche le fichier et le
backend auquel il aboutit.

Deux variables d'environnement :

- `WEIGHT_SURGEON_CONFIG=<fichier>` : un fichier à cet emplacement, quel que soit le système (une
  interface graphique, des scripts, des tests) ;
- `WEIGHT_SURGEON_NO_USER_CONFIG=1` : ignore l'emplacement par défaut.

**Tests.** `.cargo/config.toml` pose `WEIGHT_SURGEON_NO_USER_CONFIG=1` : `cargo test` ne lit jamais
la configuration de la machine. Il s'applique aussi à `cargo run` ; pour lancer `surgeon` avec votre
configuration, utilisez le binaire compilé (`target/release/surgeon`) ou
`WEIGHT_SURGEON_NO_USER_CONFIG=0 cargo run -- ...`.

## 4. Détection matérielle et premier démarrage

```bash
surgeon backend detect          # sonde et recommande, n'écrit rien
surgeon backend detect --save   # écrit la recommandation comme configuration utilisateur
surgeon backend show            # le fichier, et le backend qui en résulte
```

La sonde cherche `llama-server` (dossier configuré, `$LLAMA_CPP_DIR`, `PATH`, puis
`~/tools/llama-vulkan`), lance `llama-server --list-devices` (délai de 30 s) et cherche `hipfire`. Elle
recommande `llama-server` avec `gpu_layers = 99` quand un GPU est listé, sinon `native`. `hipfire`
n'est jamais recommandé. Exemple réel (Radeon RX 7600 XT) :

```text
[BACKEND]  llama-server C:\Users\<vous>\tools\llama-vulkan\llama-server.exe
  Vulkan0  AMD Radeon RX 7600 XT (16368 Mio)
[BACKEND]  hipfire : hipfire cible Linux avec ROCm (Windows et WSL2 non testés par ses auteurs)
[BACKEND]  recommandé : llama-server (Vulkan, ngl=99, batch=1) (llama-server voit AMD Radeon RX 7600 XT (Vulkan) : tous les calques sur le GPU)
```

**API pour une interface graphique.** Rien n'est propre au terminal :

- Rust : `weight_surgeon::backend::detect_hardware_and_backends(&BackendConfig) -> BackendReport`
  (`os`, `llama { server, dir, devices[], error }`, `hipfire { binary, supported_os, note }`,
  `recommended`, `reason`) ; `load_user_config()`, `save_to(path, &config)`, `user_config_path()` ;
  `BackendConfig` est `Serialize` / `Deserialize` (JSON : `{"backend":"llama-server",...}`, champs
  absents omis, champ inconnu refusé).
- Sans lier la bibliothèque : `surgeon --json backend detect` et `surgeon --json backend show`, ou
  l'outil MCP `surgeon_backends` (lecture seule, sans argument).

Parcours conseillé au premier lancement : `detect` ; montrer `report.llama.devices` et
`report.reason` ; sur confirmation, `detect --save` (ou `save_to` avec la configuration que
l'utilisateur a modifiée). Ensuite aucune option de backend n'est nécessaire.

## 5. `llama-server` : GPU, conversion, GGUF temporaire

- **GPU.** Il suffit d'un `llama-server` compilé pour le GPU (release `win-vulkan-x64` de
  `ggml-org/llama.cpp`) et de `gpu_layers = 99`. `surgeon` ne compile rien.
- **Conversion.** Un dossier SafeTensors est converti par `surgeon` (matrices en f16 par défaut,
  `gguf_dtype = "f32"` pour une conversion sans perte depuis bf16, normes et biais en f32,
  vocabulaire de `tokenizer.json`). Si le dossier llama.cpp contient `convert_hf_to_gguf.py`, c'est
  lui qui convertit (autres architectures, Python requis). Familles gérées, d'après
  `model_type` :

  | `model_type` | `general.architecture` | Particularités reproduites de `convert_hf_to_gguf.py` |
  | --- | --- | --- |
  | `qwen2` | `qwen2` | biais Q/K/V, vocabulaire byte-level `qwen2` |
  | `llama` (Llama 2, 3, 3.1, 3.2), `mistral` | `llama` | lignes Q/K regroupées par paire rotative ; `rope_freqs.weight` calculé pour `rope_scaling` `llama3` ; `rope.dimension_count` |
  | `gemma` | `gemma` | normes écrites `1 + w` ; `lm_head` ignoré (lié) ; `key_length` / `value_length` ; `add_space_prefix = false` |
  | `gemma2` | `gemma2` | comme `gemma`, plus les normes post-attention et post-FFN, les softcaps et `sliding_window` |
  | `phi3` | `phi3` | `qkv_proj` et `gate_up_proj` gardés fusionnés ; `rope_factors_long` / `rope_factors_short` et `rope.scaling.attn_factor` pour LongRoPE ; dimension rotative partielle |

  Vocabulaire : un BPE byte-level (Qwen2, Llama 3) devient `gpt2`, un BPE SentencePiece avec
  `byte_fallback` (Llama 2, Mistral, Gemma, Phi-3) devient `llama` avec des scores tirés de l'ordre
  des fusions, les pièces `<0xNN>` en type BYTE et le jeton inconnu en UNKNOWN. Sont refusés avec
  leur raison, pour ne pas produire un GGUF qui tourne faux : un autre `model_type`, un tokenizer
  qui n'est pas BPE, un BPE byte-level hors Qwen2 dont le pré-découpage n'est pas celui de Llama 3
  (Mistral Nemo, par exemple), un `rope_scaling` Llama ou Phi-3 autre que `llama3` / `longrope`.
- **Fichier temporaire.** Le GGUF est écrit dans `std::env::temp_dir()` sous le nom
  `weight-surgeon-<étiquette>-<pid>-<nanosecondes>.gguf`, jamais à côté du modèle. Une garde (`Drop`)
  le supprime avec le générateur, après une erreur, et sur une interruption par Ctrl-C (le
  gestionnaire lève un drapeau, le programme se termine normalement). Seul un `kill` brutal laisse un
  fichier : pointez `TMP` / `TEMP` / `TMPDIR` sur un disque assez grand pour un gros modèle.
- **Reproductibilité.** Un run GPU n'est pas reproductible au bit près (ordre des réductions des
  noyaux) : comparez des taux avec leurs intervalles. Dans un pipeline, avant et après le patch
  tournent sur le même moteur.

## 6. `hipfire` : ce qui est modélisé et ce qui ne l'est pas

Lu le 2026-11-13 dans le README du projet : binaire `hipfire`, sous-commandes `pull`, `run`, `chat`,
`serve`, `bench`, `quantize` ; démon OpenAI-compatible sur `127.0.0.1:11435`
(`/v1/chat/completions` avec `model`, `messages`, `max_tokens`) ; Linux avec ROCm 6, « Windows et WSL2
non testés sur GPU » ; formats quantifiés propres (`hipfire quantize`) ; Qwen2.5 absent de la liste.

`--backend hipfire` (ou `default_backend = "hipfire"`) est donc reconnu partout (CLI, `pipeline.json`,
MCP), vérifié avant tout calcul, et refusé avec un message qui dit pourquoi et quoi faire :

- hors Linux : « moteur HIP/ROCm pour Linux, non supporté sur windows... Utilisez --backend
  llama-server (Vulkan, llama.cpp) ou native » (une RX 7600 XT, `gfx1102`, passe par Vulkan) ;
- sous Linux sans le binaire : `--hipfire-bin` (ou `hipfire_bin`) ou le `PATH` ;
- avec le binaire : le pilote n'est pas branché. Le bench envoie des prompts déjà rendus par son
  gabarit, avec une graine et un décodage glouton ; un endpoint de chat appliquerait un second gabarit
  et la documentation de hipfire ne mentionne ni graine ni glouton, donc les mesures ne seraient pas
  comparables. Le pilote se branche sur `Backend::Hipfire` (`bench_generator` dans `src/app.rs`)
  dès qu'un endpoint de complétion brute pourra être validé sur une machine Linux ROCm.

## 7. Pipeline

`pipeline.json` accepte un bloc `backend` (mêmes champs que `config.toml`) ; les options de
`pipeline plan` et `pipeline run` passent devant, puis le fichier utilisateur. Le résultat est
écrit dans `job.json` : **un job est lié à son backend**. Relancer `--config` avec un autre backend
est refusé comme toute autre configuration différente ; une reprise sans `--config` refuse des
options de backend. Un job `native` ne porte aucun bloc `backend` (les jobs d'avant restent liés
à la même configuration).

Seuls les piliers binaires (`safety`, `unblocking`, `gsm8k`) génèrent ; `kl` et `logits` lisent
toujours des logits natifs. `plan` calibre chaque moteur sur son propre travail : le moteur natif
pour les piliers `kl` et `logits` (`calibration`), le backend du job pour les piliers binaires
(`backend_calibration`, dont le temps de chargement compte la conversion GGUF et le démarrage du
serveur, payés à chaque mesure). `estimated_secs` additionne les deux, chacun à son débit. Un
moteur sans travail n'est pas lancé ; `--no-calibration` ne lance rien. Le rapport donne le moteur réel :
`results.backend`, `manifest.backend` et `plan.backend`, par exemple
`llama-server (Vulkan, ngl=99, batch=1)`, `native-cpu`. Les rapports de `bench eval` et
`bench refusal` ajoutent `generation.backend_detail` quand ils portent un bloc `generation`.

**MCP.** Les outils `surgeon_bench` et `surgeon_pipeline` appliquent le fichier utilisateur de
l'opérateur, jamais un chemin de binaire donné par l'agent : `llama_cpp_dir` d'un argument ou d'un
`pipeline.json` reste soumis au bac à sable, `hipfire_bin` n'est pas un argument MCP.

## 8. Mesure réelle (2026-11-13)

Radeon RX 7600 XT (16 Go, Vulkan), Qwen2.5-0.5B-Instruct, configuration utilisateur écrite par
`surgeon backend detect --save`, aucune option de backend ni `$LLAMA_CPP_DIR` sur la ligne de commande.

```bash
surgeon pipeline plan --config pipeline.json --job-dir job   # [PLAN] ... llama-server (Vulkan, ngl=99, batch=1)
surgeon pipeline run  --config pipeline.json --job-dir job
```

Job complet : 2 candidats (alpha 0 et 0,5, `project_out` rang 2, couches 10 à 14), 4 piliers (`kl`,
`logits`, `gsm8k` Dev de 30 et Test de 20, `safety` Dev de 20 et Test de 20), Test consulté une fois.

| Mesure | Valeur |
| --- | --- |
| Durée du job | 316 s |
| Backend enregistré (`results.backend`, `manifest.backend`) | `llama-server (Vulkan, ngl=99, batch=1)` |
| Dossier llama.cpp enregistré (`manifest.llama_cpp`) | celui du fichier utilisateur |
| Dev `gsm8k`, alpha 0 (modèle inchangé) | 0,87 (26/30), 3 réponses coupées |
| Dev `gsm8k`, alpha 0,5 | 0,90, 2 réponses coupées |
| Portes Dev | toutes GO pour les deux candidats |
| Verdict | INCONCLUSIVE : une porte du Test n'est pas concluante (20 items) |
| GGUF temporaires restants dans `%TEMP%` et à côté du modèle | 0 |
| SHA-256 des fichiers du modèle d'origine | identiques avant et après |

Le 0,87 du modèle d'origine est celui du bench GPU ([validation](validation/qwen2.5-0.5b-validation.md),
section 11) : le pipeline et `bench eval` mesurent la même chose sur le même moteur. L'évaluation
`gsm8k` d'un modèle prend 35 à 40 s (voir la section 11), sous la minute.
