# GPU AMD sous Windows : hipfire (WSL2) ou llama-server Vulkan

Évaluation du 2026-11-20, branche `feat/amd-rx7600xt-pipeline`. But : mesurer une copie de travail
patchée sur le GPU d'une machine Windows. Machine de référence : Intel Core i5-12400F, AMD Radeon
RX 7600 XT (16 Go, RDNA 3, gfx1102), Windows 11, pilote Adrenalin. Ce document compare deux voies
et recommande ; la voie B n'est pas implémentée ici.

## Les deux voies

**Voie A : hipfire sous WSL2, copie requantifiée.** Ubuntu 24.04 dans WSL2, ROCm 7.2.1,
librocdxg 1.2.2, hipfire 0.4.6 en démon. Le démon ne lit jamais le dossier SafeTensors : il sert le
fichier quantifié enregistré sous un nom. Cette branche ajoute la requantification
(`hipfire_requantize = "auto"`, défaut, ou `--hipfire-requantize`) : SHA-256 des `*.safetensors`,
`hipfire quantize --format q8f16 --register ws-<copie>-<sha8>`, `.hipfire-alias.json` dans la
copie, puis mesure sous cet alias tant que le hash ne bouge pas ([BACKENDS.md](BACKENDS.md),
section 7).

**Voie B : llama-server Vulkan natif, GGUF converti en Rust.** `--backend llama-server` existe
déjà : `surgeon` convertit le dossier SafeTensors en GGUF temporaire f16 (ou f32) avec son propre
convertisseur (`src/eval/safetensors_gguf.rs`, sans Python), démarre `llama-server -ngl 99` (build
`win-vulkan-x64` de llama.cpp) le temps du bench, puis l'arrête et supprime le GGUF.

## Pourquoi `detect.rs` n'en recommande qu'une

`recommend()` (`src/backend/detect.rs:165`) n'a que trois issues : `hipfire` (GPU AMD dont le
démon répond), `mistralrs` (installé), `native`. llama.cpp n'est ni cherché ni recommandé ;
`--backend llama-server` reste utilisable à la main ([BACKENDS.md](BACKENDS.md), section 4).
C'est une règle de produit posée par le commit `9c463e5` (« Rust engines only », en tête du
module), pas un défaut constaté de la voie Vulkan : la section 11 de la validation la mesure
26 fois plus rapide que le CPU, avec la même exactitude. Ce qui la tient à l'écart en pratique :

- `llama-server` est un binaire C++ externe que la sonde ne cherche pas et que `surgeon-gui`
  n'installe pas (dossier à donner par `LLAMA_CPP_DIR` ou `llama_cpp_dir`) ;
- le convertisseur Rust couvre Qwen2, Llama 2 et 3, Mistral, Gemma, Gemma 2 et Phi-3, pas Qwen3
  (que le moteur natif lit) ;
- il n'écrit que du f16 ou du f32 : aucune quantification en Rust pur.

## Comparaison

| Critère | Voie A : hipfire (WSL2) | Voie B : llama-server Vulkan |
| --- | --- | --- |
| Préparer une copie patchée | SHA-256 (environ 1 à 2 s par Go ici) puis `hipfire quantize` q8f16. Pour Qwen2.5-0.5B (630 M paramètres, 1,26 Go lus depuis `/mnt/c`, 678 Mo écrits), la commande entière (deux hachages, quantification, première requête) a pris 29 s. Une fois par état des poids : l'alias est réutilisé tant que le hash ne change pas. | Conversion f16 en Rust : environ 1 s pour Qwen2.5-0.5B (section 11). Refaite à chaque bench et à chaque mesure d'un pipeline (fichier temporaire), plus le démarrage du serveur. |
| Entre le patch et la mesure | Poids en 8 bits (q8f16). MQ4 changeait déjà des réponses (17 + 25 donnait 32) : une étape de quantification sépare ce qui est mesuré de ce qui a été patché. | f16 depuis bf16 : 29 extraits sur 30 identiques au CPU, l'item restant aboutit à la même réponse (section 11). |
| Débit mesuré | 110 à 200 tok/s en décodage dans le log du démon (requêtes de 2 à 119 tokens, `--spec ngram`, modèle d'origine). Aucun bench complet de 30 questions : le démon avait perdu le GPU (ligne suivante). | 30 questions GSM8K, 256 tokens : 38,9 s en `--batch-size 1`, 16,9 s en `--batch-size 6` (section 11) ; job pipeline de 2 candidats en 316 s (section 12). |
| Stabilité des pilotes | gfx1102 est hors de la matrice de librocdxg 1.2 : ligne `0x7480,11,0,2` dans `dids.conf`, `LD_PRELOAD` contre un SIGSEGV à la sortie, `--spec ngram` contre une route Qwen2 qui répond 500. hipfire se dit « non testé sur GPU » sous Windows et WSL2. Constaté le 2026-11-20 : après 1 j 16 h de fonctionnement, le démon voit `0 MB free / 0 MB total`, tout chargement échoue (`hipMalloc: out of memory`, `[CpuWait] fail 1`) jusqu'à son redémarrage. | Vulkan d'Adrenalin, sans contournement : 4 runs identiques, 80 à 88 % de calcul GPU, jamais de repli CPU (section 11). Un serveur par bench, arrêté à la fin : rien ne vit assez longtemps pour se dégrader. |
| VRAM (16 Go) | q8f16 : environ 1,07 octet par paramètre, donc des modèles jusqu'à 12 ou 13 milliards de paramètres environ avec le contexte (estimation, non mesurée). Un modèle chargé à la fois (le précédent est déchargé avant). Sous WSL, KV en mode `legacy` (le VMM n'est pas certifié par hipfire sous ROCDXG). | f16 : 2 octets par paramètre, donc environ 6 milliards au plus avec le contexte : Qwen2.5-7B (15,2 Go en f16) ne tient pas. f32 divise encore par deux. La VRAM est rendue à l'arrêt du serveur. |
| Disque | Chaque alias est un fichier q8f16 complet dans le disque virtuel de WSL, qui ne rétrécit pas tout seul ; rien n'est supprimé automatiquement (`hipfire rm`). | GGUF temporaire supprimé à la fin (0 restant après un pipeline, section 12). |
| Installation | WSL2, Ubuntu, ROCm (plusieurs Go), librocdxg, Rust, compilation de hipfire ; le catalogue des alias doit être celui de l'utilisateur qui lance le démon. | Une archive llama.cpp `win-vulkan-x64` décompressée, `LLAMA_CPP_DIR`. |
| Requête | `/v1/chat/completions` seulement : tours du gabarit renvoyés en messages, gabarit de hipfire, glouton imposé, aucune graine documentée. | `/completion` avec le prompt brut du bench, sa graine et ses réglages d'échantillonnage. |

## Recommandation

**Voie B par défaut pour mesurer des copies de travail sur cette machine, voie A en complément.**

1. **Comparer une copie à son original : llama-server Vulkan.** Rien ne s'intercale entre le patch
   et la mesure : du f16, pas de quantification 8 bits. La préparation coûte environ une seconde
   par mesure, contre une requantification par état des poids. La pile pilote est celle que
   Windows supporte. Les mesures de référence (sections 11 et 12) sont faites avec. Le démon hipfire
   s'est en revanche dégradé en cours de session, ce qu'un serveur par bench évite.
2. **Lever la règle « moteurs Rust seulement » pour un cas précis** de `recommend()` : machine AMD
   sous Windows, démon hipfire injoignable, `llama-server --list-devices` qui annonce un
   périphérique `Vulkan`. Recommander alors `llama-server` plutôt que `native`. `list_devices`
   et `parse_devices` existent déjà dans `detect.rs`. À faire dans une branche dédiée.
3. **Garder hipfire pour ce que Vulkan f16 ne peut pas faire** : un modèle de plus de 6 milliards
   de paramètres environ, qui ne tient en VRAM que quantifié. La requantification par alias de
   cette branche en est le prérequis. Avant un long bench, redémarrer le démon
   (`start-hipfire.ps1`) et vérifier `/health`, puis comparer son débit à celui de Vulkan sur les
   mêmes 30 questions, mesure que cette évaluation n'a pas pu faire.

À revoir si : gfx1102 entre dans la matrice de librocdxg, ROCm 7.13 ou plus récent remplace les
réglages DXG, hipfire corrige sa route Qwen2 en HTTP, ou une quantification Q8_0 arrive dans le
convertisseur Rust (la voie B ouvrirait alors les modèles de 7 à 13 milliards).
