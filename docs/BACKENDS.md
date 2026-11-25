# Backends d'inférence et configuration utilisateur

Qui génère les réponses de `surgeon bench eval`, `bench refusal`, `surgeon pipeline` et des outils MCP
`surgeon_bench` / `surgeon_pipeline`, comment ce choix est mémorisé, et comment une interface
graphique s'en sert au premier démarrage. Code : `src/backend/`.

## 1. Les backends

`auto` n'est pas un moteur : c'est le défaut de `bench eval`, `bench refusal`, `pipeline`, de
`surgeon_bench` (modes `eval` et `refusal`) et de l'interface graphique. Il choisit au début de
chaque run d'après le GPU de la machine (section 4) : GPU AMD, `hipfire` ; GPU NVIDIA ou Apple
Silicon, `mistralrs` ; pas de GPU, `native` (CPU). `native` a aussi l'alias `cpu`.

| Backend | Matériel | Ce que c'est | État |
| --- | --- | --- | --- |
| `mistralrs` | GPU selon le build (CUDA, Metal...) ou CPU | Un serveur [mistral.rs](https://github.com/EricLBuehler/mistral.rs) local (`mistralrs-server`, ou `mistralrs serve`) qui charge le dossier SafeTensors tel quel, sans conversion (un `.gguf` est aussi accepté). | Moteur de `auto` pour NVIDIA et Apple Silicon ; sans GPU, recommandé par `backend detect` quand il est installé (section 4). Intégrable à surgeon par `--features cuda` ou `metal` (section 6). |
| `llama-server` | GPU (Vulkan, CUDA, Metal...) ou CPU | Un serveur `llama.cpp` local pour toute l'exécution. Un dossier SafeTensors Qwen2, Llama, Mistral, Gemma, Gemma 2 ou Phi-3 est converti par `surgeon` lui-même en GGUF (Rust pur, sans Python). | Jamais recommandé : seulement sur `--backend llama-server` ou un fichier qui le nomme. |
| `hipfire` | GPU AMD, HIP/ROCm | Démon externe [`warpfront/hipfire`](https://github.com/warpfront/hipfire), déjà lancé ou démarré par `surgeon-gui`. | Moteur de `auto` pour un GPU AMD ; tout système, Windows compris par WSL2 (section 7). |
| `native` | CPU | Générateur en Rust pur de `surgeon` (Qwen2, Llama/Mistral, Qwen3 en SafeTensors). Aucun programme externe. | Moteur de `auto` sans GPU, ou forcé par `--backend native` (alias `cpu`). Jamais un repli : voir la section 2. |

Tous partagent l'interface `TextGenerator` (`src/eval/generator.rs`) : un bench ne sait pas
lequel il utilise. Les trois serveurs passent par le même client (`src/eval/server.rs`), qui parle
le dialecte de chacun : `/completion` de llama.cpp, `/v1/completions` ou `/v1/chat/completions`
OpenAI. `src/backend/` ajoute les réglages communs, leur ordre de priorité, la sonde de la machine
et les libellés des rapports. Le convertisseur GGUF est `src/eval/safetensors_gguf.rs`.

## 2. Réglages et priorité

`BackendConfig` (`src/backend/config.rs`) : `backend`, `gpu_layers`, `batch_size`, `llama_cpp_dir`,
`mistralrs_dir`, `mistralrs_bin`, `hipfire_bin`, `hipfire_url`, `gguf_dtype`, `models_dir`
(dossier de modèles que surgeon-gui ouvre en lecture seule à ses agents), `hipfire_requantize`
(`auto` ou `off`, section 7). Chaque champ est
facultatif ; une couche ne dit que ce qu'elle sait.

```text
option de la ligne de commande  >  "backend" de pipeline.json  >  config.toml utilisateur  >  défaut
```

Sans configuration, le défaut de `bench eval`, `bench refusal`, `pipeline`, `surgeon_bench` et de
l'interface graphique est `auto` (`--backend auto`, `backend: "auto"`, `backend = "auto"` dans
`config.toml`). `auto` se résout au début de chaque run d'après le GPU : AMD, `hipfire` ; NVIDIA ou
Apple Silicon, `mistralrs` ; aucun GPU, `native`. Avant, `bench eval` partait sur `llama` et les
autres sur `native`. `bench ppl` ne change pas (`llama` ou `native` seulement). Un job repris garde
son backend. Quatre règles :

- **Aucun repli silencieux sur le CPU.** Sur une machine avec GPU, si le moteur GPU ne démarre pas
  (démon hipfire arrêté, mistral.rs absent), le bench échoue avec une erreur qui finit par « aucun
  repli sur le CPU ; diagnostic : surgeon doctor (ou --backend cpu pour le forcer) ». Le CPU ne sert
  que sans GPU, ou quand `--backend native` (alias `cpu`) est passé explicitement.

- `gpu_layers` et `batch_size` du fichier utilisateur ne valent que pour les serveurs
  (`mistralrs`, `llama-server`, `hipfire`) : ils ne changent pas le découpage en lots du moteur
  natif. Une option de la ligne de commande, elle, s'applique toujours. `gpu_layers` ne concerne
  que `llama-server`.
- `$LLAMA_CPP_DIR` reste entre l'option `--llama-cpp-dir` et le fichier utilisateur, comme
  `$MISTRALRS_DIR` pour `--mistralrs-dir`.
- `mistralrs_bin` donné, c'est ce fichier et lui seul : absent, la commande s'arrête au lieu d'en
  prendre un autre dans le `PATH`.

Options concernées : `--backend`, `--gpu-layers`, `--batch-size`, `--llama-cpp-dir`,
`--mistralrs-dir`, `--mistralrs-bin`, `--hipfire-bin`, `--hipfire-url`, `--hipfire-requantize` (`bench eval`,
`bench refusal`, `pipeline plan`, `pipeline run`) et `--gguf-dtype` (`bench eval`, `bench refusal`).

## 3. Configuration persistante

Fichier `config.toml` :

| Système | Emplacement |
| --- | --- |
| Windows | `%APPDATA%\weight-surgeon\config.toml` |
| Linux, macOS | `$XDG_CONFIG_HOME/weight-surgeon/config.toml`, sinon `~/.config/weight-surgeon/config.toml` |

```toml
# weight-surgeon: backend d'inférence
default_backend = "mistralrs"
llama_cpp_dir = "C:\\Users\\<vous>\\tools\\llama-vulkan"
mistralrs_bin = "C:\\Users\\<vous>\\tools\\mistral.rs\\mistralrs-server.exe"
hipfire_url = "http://127.0.0.1:11435"
gpu_layers = 99
batch_size = 1
gguf_dtype = "f16"
models_dir = "C:\\models"
hipfire_requantize = "auto"
```

Sous-ensemble plat de TOML (`clé = "texte"` ou `clé = 123`, commentaires `#`, chaînes `'...'`
sans échappement acceptées), lu et écrit sans crate TOML. Une clé inconnue ou une valeur invalide
est une erreur nommant le fichier et la ligne, et arrête les commandes qui génèrent (les autres,
`inspect` ou `check`, ne lisent pas le fichier). `surgeon backend show` affiche le fichier et le
backend auquel il aboutit.

Trois variables d'environnement :

- `WEIGHT_SURGEON_CONFIG=<fichier>` : un fichier à cet emplacement, quel que soit le système (une
  interface graphique, des scripts, des tests) ;
- `WEIGHT_SURGEON_NO_USER_CONFIG=1` : ignore l'emplacement par défaut.

Une troisième, `WEIGHT_SURGEON_GPU=none|amd|nvidia|apple`, force le fabricant de GPU détecté (pour une
machine dont le GPU n'est pas reconnu) ; elle décide donc de ce que `auto` devient.

**Tests.** `.cargo/config.toml` pose `WEIGHT_SURGEON_NO_USER_CONFIG=1` et `WEIGHT_SURGEON_GPU=none` :
`cargo test` ne lit jamais la configuration de la machine et `auto` y vaut `native`. Ils s'appliquent
aussi à `cargo run` ; pour lancer `surgeon` avec votre configuration et votre vrai GPU, utilisez le
binaire compilé (`target/release/surgeon`) ou
`WEIGHT_SURGEON_NO_USER_CONFIG=0 WEIGHT_SURGEON_GPU= cargo run -- ...`.

## 4. Détection matérielle et premier démarrage

```bash
surgeon backend detect          # sonde et recommande, n'écrit rien
surgeon backend detect --save   # écrit la recommandation comme configuration utilisateur
surgeon backend show            # le fichier, et le backend qui en résulte
```

La sonde ne connaît que les moteurs Rust. Elle lit les GPU (NVML pour NVIDIA, sysfs pour AMD
sous Linux, Apple Silicon, et sous Windows les adaptateurs d'affichage), en déduit le fabricant
qui décide (NVIDIA, puis AMD, puis Apple ; sinon le CPU), cherche mistral.rs (`mistralrs_bin`,
sinon `mistralrs-server` puis `mistralrs` dans `mistralrs_dir`, `$MISTRALRS_DIR`, puis le
`PATH`) et interroge le démon hipfire (section 7 ; sur une machine AMD sans `hipfire_url`, à
l'adresse par défaut `127.0.0.1:11435`). La cible est `hipfire` pour un GPU AMD et `mistralrs`
sinon ; sur une machine avec GPU, `target` et `recommended` sont toujours le moteur du GPU, même pas
encore prêt : `reason` dit ce qui manque et renvoie à `surgeon doctor`. `native` n'est recommandé
que sans GPU (`mistralrs` s'il est installé sur une machine sans GPU). llama.cpp n'est ni cherché ni recommandé ; `--backend llama-server` reste
utilisable à la main. Exemple (Radeon RX 7600 XT, sans mistral.rs ni démon hipfire) :

```text
[BACKEND]  GPU : AMD Radeon RX 7600 XT
[BACKEND]  mistral.rs introuvable
[BACKEND]  hipfire : hipfire injoignable sur 127.0.0.1:11435 : lancez `hipfire serve` ...
[BACKEND]  recommandé : hipfire (démon injoignable : lancez `hipfire serve` ; diagnostic : surgeon doctor)
```

`surgeon backend show` ajoute un avertissement, sans échouer, quand le backend configuré ne pourra
pas tourner (mistral.rs introuvable, démon hipfire injoignable) ou faussera les mesures (hipfire
qui n'expose que `/v1/chat/completions`).

**API pour une interface graphique.** Rien n'est propre au terminal :

- Rust : `weight_surgeon::backend::detect_hardware_and_backends(&BackendConfig) -> BackendReport`
  (`os`, `gpus[]`, `vendor` (`nvidia`, `amd`, `apple`, `cpu`), `target`,
  `mistralrs { binary }`, `hipfire { binary, up, note }`, `recommended`, `reason`) ;
  `load_user_config()`, `save_to(path, &config)`, `user_config_path()` ;
  `BackendConfig` est `Serialize` / `Deserialize` (JSON : `{"backend":"mistralrs",...}`, champs
  absents omis, champ inconnu refusé).
- Sans lier la bibliothèque : `surgeon --json backend detect` et `surgeon --json backend show`, ou
  l'outil MCP `surgeon_backends` (lecture seule, sans argument, même rapport).

`crates/surgeon-gui` applique la même recommandation au premier lancement (`backend_choice`) et
l'écrit dans la configuration utilisateur tant que l'utilisateur n'a pas choisi à la main ; il ne
retombe plus sur `native` ni sur un mistral.rs CPU quand un GPU AMD est là.

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

## 6. `mistralrs` : le dossier SafeTensors sans conversion

Doc lue le 2026-11-13 : deux lignes de commande existent selon la version, et `surgeon` choisit
d'après le `--help` du binaire installé (sous-commande `serve`, sinon `plain` / `gguf`), chaque
option venant de l'aide du niveau qui la liste. Rien n'est deviné ; une version qui ne liste pas
ces options est refusée avec son nom.

```text
mistralrs        --seed 42 serve -p <port> --host 127.0.0.1 --max-seqs <batch> -m <dossier>
mistralrs-server --port <port> --serve-ip 127.0.0.1 --seed 42 --max-seqs <batch> plain -m <dossier>
un GGUF :        ... serve|gguf ... -m <dossier parent> -f <fichier.gguf>
```

- **Sans conversion.** Le dossier SafeTensors (le modèle patché, une copie de travail du
  pipeline) est passé tel quel : aucun GGUF temporaire, aucun `gguf_dtype`.
- **GPU.** Celui du build de mistral.rs (CUDA, Metal...) ; `--cpu` n'est jamais passé. Le serveur
  écoute en boucle locale seulement ; il est tué avec le bench.
- **Embarqué (features Cargo facultatives, aucune par défaut).** `--features mistralrs` : mistral.rs
  dans le processus, sur le CPU (surtout un contrôle de l'API) ; `--features cuda` : NVIDIA, avec le
  CUDA toolkit (`nvcc`) ; `--features metal` : Apple Silicon, avec les Xcode CLT. Les mêmes noms
  existent sur le crate `surgeon-gui` et sont transmis au moteur. Compilé avec `cuda` ou `metal`,
  `--backend mistralrs` charge dans le processus le dossier SafeTensors de `--model`, depuis ce
  chemin exact ; sinon il passe par le binaire externe `mistralrs-server` / `mistralrs` comme avant
  (avec `--features mistralrs` seul, le moteur CPU embarqué ne sert que si aucun binaire n'est
  trouvé). Il décode en glouton seulement (son SDK ne prend pas de graine) : `--temperature` > 0
  y est refusé ; un GGUF passe toujours par le binaire. Il n'existe pas de feature `hipfire` ni
  `rocm` : hipfire est un démon externe joint en HTTP sur la boucle locale, rien à compiler.

  ```bash
  cargo build --release --workspace --features cuda    # ou metal, ou mistralrs
  ```
- **Décodage.** Requêtes `/v1/completions` (le prompt brut : le gabarit du bench reste le seul),
  `temperature = 0`, `top_p = 1`, graine 42 globale et par requête, `model = "default"`. Les réglages
  hors OpenAI (`top_k`, `min_p`, `repetition_penalty`) ne sont envoyés que s'ils sont actifs.
- **Non vérifié sur une vraie installation.** Les deux formes sont testées contre un faux serveur
  (`examples/fake_llama_server.rs`), pas contre mistral.rs lui-même : à valider sur la machine
  cible avec `bench eval` sur quelques questions avant un pipeline.

## 7. `hipfire` : un démon, sur tout système

Lu le 2026-11-13 dans le README du projet : binaire `hipfire`, sous-commandes `pull`, `run`, `chat`,
`serve`, `bench`, `quantize` ; démon OpenAI-compatible sur `127.0.0.1:11435`
(`/v1/chat/completions` avec `model`, `messages`, `max_tokens`, et `/v1/models`) ; Linux avec
ROCm 6, « Windows et WSL2 non testés sur GPU » ; modèles nommés par étiquette ou produits par
`hipfire quantize`.

`--backend hipfire` (ou `default_backend = "hipfire"`) est accepté sur tout système, Windows
compris. `surgeon` ne lance pas le démon, il s'y connecte :

1. **Où.** `hipfire_url` (`--hipfire-url`, boucle locale seulement : `http://127.0.0.1:<port>` ou
   `http://localhost:<port>`), sinon `127.0.0.1:11435` quand le binaire est trouvé
   (`hipfire_bin` / `--hipfire-bin`, sinon le dossier de l'exécutable, sinon le `PATH`). Ni l'un
   ni l'autre : erreur qui nomme ces réglages et `--backend mistralrs, llama-server ou native`.
2. **Quoi.** Une requête vide sur `/v1/completions`, puis sur `/v1/chat/completions` : une route
   qui ne répond pas 404 ou 405 existe. Démon injoignable : « lancez `hipfire serve` ».
3. **Comment.** `/v1/completions` garde le prompt brut. Si seul `/v1/chat/completions` existe, le
   run continue, avec un avertissement (`backend show` et le bench) : le prompt rendu est
   redécoupé en messages (`system`, `user`... : `extract::chat_messages`), si bien que le gabarit
   de hipfire est le seul appliqué, et la requête part en `temperature = 0` quel que soit le
   réglage, faute de graine documentée. Un bench hipfire avec `--temperature > 0` est refusé
   (code 2). Un prompt hors des familles connues, ou avec du texte hors de ses tours (`<bos>`
   en tête, réponse amorcée), part tel quel en un seul message `user`. Les scores restent à
   comparer sur hipfire seulement : `bench eval --baseline-model` mesure les deux modèles sur
   le même backend.

Le champ `model` des requêtes est le nom du modèle donné au bench (son dossier, ou son fichier sans
extension) : le démon doit connaître un modèle sous ce nom. Comme pour mistral.rs, seul un faux
démon l'a vérifié ici.

Le démon ne lit donc jamais les poids du dossier, GGUF compris : un dossier patché doit être
re-quantifié et réenregistré, sinon la mesure porte sur d'anciens poids. Une seule règle
(`backend::hipfire::validate_hipfire_target`) vaut pour la CLI, le pipeline et MCP : avec
`hipfire_requantize = "auto"` (défaut), tout dossier SafeTensors, l'original compris, passe sous
l'alias de son contenu ; avec `off`, l'original passe sous son nom et une copie de travail passe
sous son alias quand elle a été requantifiée pour ses poids actuels, sinon elle est refusée. Est une copie de travail un chemin
d'un projet surgeon-gui autre que la source de son manifeste (`copies/`, sorties des runs), le
`work/` d'un job pipeline, ou un dossier à côté des sauvegardes d'`apply` (`<fichier>.bak-<s>-<n>`).

**Alias de contenu.** `hipfire_requantize` vaut `auto` par défaut (config.toml, `"backend"` de
pipeline.json) ; la ligne de commande prend `--hipfire-requantize [auto|off]` (le drapeau seul
vaut `auto`) sur `bench eval`, `bench refusal`, `pipeline plan` et `pipeline run`. Sous `auto`,
tout dossier SafeTensors mesuré sur hipfire, l'original compris, est servi sous l'alias
`ws-<dossier>-<sha8>` de ses poids ; surgeon le requantifie d'abord si le catalogue de hipfire ne
l'a pas. Un bench pairé (`--model` copie patchée contre `--baseline-model` original) demande donc
à hipfire deux noms qui valent chacun les poids sur disque. Le fichier `.hipfire-alias.json` n'est
écrit que dans les copies de travail, jamais dans le dossier de l'original. Rien ne se passe après
`apply` : seul un bench ou un pipeline hipfire déclenche la requantification.

1. SHA-256 des `*.safetensors` du dossier (noms et empreintes, dans l'ordre des noms ; les
   sauvegardes d'`apply` n'en font pas partie), alias `ws-<copie>-<sha8>` (nom du dossier,
   caractères hors `[A-Za-z0-9_-]` remplacés par `-`, puis les 8 premiers chiffres hexadécimaux).
2. Si `models.toml` de hipfire (`$HIPFIRE_HOME`, sinon `~/.hipfire`) a déjà cet alias, mêmes
   poids : pas de quantification. Sinon `hipfire quantize <copie> --format q8f16 --stem <alias>
   --output-dir <home>/models --install --register <alias>`, sortie sur stderr. `q8f16` : MQ4
   change des réponses de Qwen2.5-0.5B sur cette configuration (17 + 25 = 32).
3. Les poids sont rehachés (modifiés pendant la quantification : erreur), puis
   `.hipfire-alias.json` est écrit dans la copie de travail (pas dans l'original) : `{"alias", "weights_sha256", "created_at"}`.

À chaque mesure, le hash est recalculé : égal à `weights_sha256`, les requêtes portent
`model = <alias>` ; différent (un shard réécrit après coup), la copie est refusée, ou
requantifiée sous un nouvel alias en mode `auto`. Sous Windows, le script tourne dans WSL
(`wsl.exe -d Ubuntu-24.04 -e bash -lc`, chemin traduit par `wslpath`) en tant qu'utilisateur par
défaut de la distribution : son catalogue doit être celui que lit le démon (`start-hipfire.ps1`
le lance sous ce même utilisateur ; le démon root de surgeon-gui lit `/root/.hipfire`). Ce qui
dépend de l'installation : l'emplacement de `models.toml` et du dossier `models`, la forme des
clés `[aliases]` et la prise en compte d'un nouvel alias par un démon déjà lancé, vérifiés sur
hipfire 0.4.6 seulement. Les anciens alias ne sont jamais supprimés : `hipfire rm` les retire.

Sous Windows, quand le démon hipfire ne répond pas, `surgeon doctor` (et l'onglet « Moteur et
machine » de surgeon-gui, qui lance le même contrôle) exécute `wsl.exe -d Ubuntu-24.04 -e true` :
une sortie contenant `HCS_E_SERVICE_NOT_AVAILABLE`, `WSL_E_WSL2_NEEDED` ou `CreateVm` remplace le
conseil par un diagnostic (virtualisation du CPU dans le BIOS, SVM Mode ou VT-x ;
`bcdedit /set hypervisorlaunchtype auto` ; redémarrage de Windows après l'activation de
VirtualMachinePlatform ; llama.cpp (Vulkan) comme moteur en attendant). Le bouton « Démarrer
hipfire » ne s'arrête plus sur un arrêt préalable en échec : si `wsl.exe` se termine avant que le
démon réponde, l'erreur affichée est cette sortie (ou le même diagnostic).

- **CLI.** `bench eval` et `bench refusal` refusent avant la première génération (modèle et
  `--baseline-model`), code 2 ; `--allow-stale-hipfire` laisse passer, avec un avertissement dans
  le rapport, une copie enregistrée à la main sous son nom. Avec `--baseline-model`, l'original et
  la copie sont mesurés sur le même démon.
- **Pipeline.** Les candidats sont toujours mesurés sur la copie de travail : le job échoue avant
  le premier candidat et avant toute lecture du Test Set, sauf en mode `auto` (chaque candidat est
  requantifié avant sa mesure) ; `pipeline run --allow-stale-hipfire`
  (ligne de commande seulement) passe outre et l'écrit dans `notes` et `report.json`.
- **MCP (agents, studio).** `surgeon_bench` `eval` et `refusal` mesurent sur le backend résolu
  (`auto` par défaut) et préparent la cible comme la CLI : sous `auto`, une copie ou l'original sans
  alias à jour est requantifié avant la mesure. L'ancien déplacement vers `native`, avec son
  avertissement et son drapeau `hipfire_set_aside`, n'existe plus (c'était un repli silencieux sur
  le CPU). Une copie `.gguf` est refusée. Aucune dérogation (`--allow-stale-hipfire`) côté agent.

**Avec `surgeon-gui`.** Sur une machine AMD, ou quand `hipfire` est le backend enregistré, et sauf
si `native` a été enregistré à la main, la fenêtre démarre le démon si rien ne répond sur
`hipfire_url` et si le binaire est trouvé : `hipfire serve 127.0.0.1:<port> --no-prewarm`, sans
console, sortie dans `hipfire.log` (dossier des journaux). Elle attend le port 5 s au plus sans
bloquer l'interface, puis relance la détection. À la fermeture, elle arrête ce démon et son
processus GPU, ce qui libère la VRAM ; un démon qu'elle n'a pas lancé n'est jamais touché. Sans
binaire, rien ne change : le moteur de repli continue, et `surgeon_backends` donne
`hipfire.binary: null`. Une fois la fenêtre fermée, un `surgeon` en ligne de commande réglé sur
`hipfire` doit trouver un démon lancé autrement.

hipfire sous WSL2 comparé à `llama-server` Vulkan sur une RX 7600 XT, et la recommandation qui en
découle : [GPU_AMD_WINDOWS_EVAL.md](GPU_AMD_WINDOWS_EVAL.md).

## 8. Pipeline

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
`llama-server (Vulkan, ngl=99, batch=1)`, `mistralrs (batch=1)`, `native-cpu`. Avant tout calcul,
`pipeline run` refuse un backend introuvable (mistral.rs absent, hipfire sans binaire ni URL).
Les rapports de `bench eval` et
`bench refusal` ajoutent `generation.backend_detail` quand ils portent un bloc `generation`.

**MCP.** Les outils `surgeon_bench` et `surgeon_pipeline` appliquent le fichier utilisateur de
l'opérateur, jamais un chemin de binaire donné par l'agent : `llama_cpp_dir` d'un argument ou d'un
`pipeline.json` reste soumis au bac à sable ; `mistralrs_dir`, `mistralrs_bin`, `hipfire_bin` et
`hipfire_url` ne sont pas des arguments MCP. L'agent laisse `backend` vide (`auto`) et ne passe jamais `native` sur une machine avec GPU ; il peut
choisir `backend: "mistralrs"`.

## 9. Mesure réelle (2026-11-13)

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
