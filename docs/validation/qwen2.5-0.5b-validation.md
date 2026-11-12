# Validation du pipeline sur Qwen2.5-0.5B-Instruct (2026-11-12)

Validation réelle de `surgeon pipeline`, de l'outil MCP `surgeon_pipeline` et des métriques
qu'ils utilisent, sur un vrai modèle, avec correction des écarts révélés. Voir
[PIPELINE.md](../PIPELINE.md) pour le fonctionnement du pipeline.

Ce document ne contient ni poids, ni sortie par item d'un jeu de Test. Les artefacts (jeux,
configurations, journaux, répertoires de job) sont dans `.jobs/qwen-test/`, hors dépôt.

## 1. Cadre

| Élément | Valeur |
| --- | --- |
| Modèle | `C:\models\Qwen2.5-0.5B-Instruct` (dossier, lecture seule) |
| SHA-256 de `model.safetensors` avant | `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe` |
| SHA-256 de `model.safetensors` après | identique, revérifié après chaque run |
| Binaire | `target/release/surgeon.exe`, `surgeon 0.1.0`, construit sur la branche `test/qwen-validation` (base `c02666e`) |
| Machine | 6 cœurs / 12 threads, 45 Gio de RAM libres, pas de GPU, backend natif (CPU, glouton) |
| llama.cpp | absent de la machine : aucune comparaison llama.cpp |

Dimensions lues dans `config.json` et vérifiées dans le checkpoint (290 tenseurs) :

| Paramètre | Valeur | Vérification |
| --- | --- | --- |
| `hidden_size` | 896 | `down_proj` [896, 4864], `o_proj` [896, 896] |
| `num_hidden_layers` | 24 | couches 0 à 23 présentes |
| `num_attention_heads` / `num_key_value_heads` | 14 / 2 | `k_proj` et `v_proj` [128, 896] = 2 têtes × 64 |
| `intermediate_size` | 4864 | `up_proj` et `gate_proj` [4864, 896] |
| `vocab_size` | 151936 | `embed_tokens` [151936, 896] |
| `tie_word_embeddings` | true | pas de `lm_head.weight` ; logits natifs par l'embedding |
| `rope_theta` | 1e6 | |
| EOS | 151645 (`<|im_end|>`), 151643 (`<|endoftext|>`) | `generation_config.json` |

## 2. Données et sous-espace

Jeux écrits pour cette validation (problèmes originaux), jamais commités :

| Jeu | Taille |
| --- | --- |
| GSM8K-like Dev / Test | 30 / 20 problèmes à réponse entière |
| Texte neutre Dev / Test (KL, top-1) | 12 / 12 phrases, 271 et 255 positions à 40 tokens au plus |
| Contraste formel / familier | 24 paires |
| Sécurité Dev / Test | 20 / 20 prompts, copies de sous-ensembles de jeux existants de l'utilisateur, sans intersection avec son Test Set ; aucun prompt nuisible écrit |

Extraction : `surgeon extract`, méthode `svd`, rang 2, couche 12 (milieu de 24), site
`resid_post`, pooling `last`, gabarit `chatml`. Vecteur de longueur 896, norme 1,0000000 ;
sous-espace [2, 896] orthonormé (produit scalaire des lignes −9,4e-10) ; d' = 4,41, cosinus
avec la différence des moyennes 0,984. Témoin : sous-espace aléatoire de même forme et de même
norme, graine fixe.

Patch testé : `project_out` de rang 2 sur `down_proj`, couches 10 à 14. Le concept est
inoffensif : il s'agit de valider l'outillage.

Portes, fixées avant toute mesure du pipeline :

- `kl` et `top1` : `max_drift` 0,1 à 90 % (les niveaux d'avertissement déjà utilisés par
  l'expérience de l'utilisateur) ;
- `gsm` et `safety` : non-infériorité, marge 0,15 à 90 %. Cette marge vient de la puissance
  seule : avec 20 à 30 prompts, une marge plus étroite ne peut pas conclure. Ce n'est pas une
  marge de décision recommandée ;
- contrôles de fumée (n = 5) : marge 0,4, uniquement pour traverser toutes les étapes.

## 3. Contrôles avant le vrai run (n = 5, un candidat)

| Contrôle | Résultat | Verdict |
| --- | --- | --- |
| a. Identité (α = 0) | KL exactement 0, intervalle [0 ; 0] ; top-1 0 ; aucune paire discordante sur GSM et Sécurité, Dev et Test | OK |
| b. Déterminisme | deux exécutions : 12 fichiers de mesure par item identiques octet pour octet ; `results.json` ne diffère que par les identifiants dérivés du chemin de la configuration | OK |
| c. Témoin négatif | sous-espace aléatoire : KL 0,0014, top-1 0, aucune réponse changée. Vrai sous-espace : KL 0,0027, top-1 0,010, 1 refus perdu sur 5 | OK |
| d. Témoin positif | `subtract` α = 20 : KL 10,39 [10,23 ; 10,50], `NO_GO` en 8 s, arrêt au pilier `kl` | OK |
| e. Reprise | travailleur tué pendant la mesure GSM, `run --job-dir` : résultats identiques à l'exécution ininterrompue | OK |
| e. Annulation | `cancel` pendant la copie de travail puis pendant une mesure : statut `cancelled`, pas de `work/`, 0 fichier partiel | OK après BUG-4 |
| f. Registre du Test Set | seconde consultation refusée (`test_set_already_used`) ; `--allow-test-reuse` : `test_contaminated`, verdict `INCONCLUSIVE` | OK |
| g. Modèle d'origine | SHA-256 inchangé après chaque run | OK |

Remarque sur f : le refus arrive après la mesure du Dev (115 s ici), car le registre n'est lu
qu'au moment de consulter le Test. C'est conforme à la conception ; `plan` ne prévient pas.

## 4. Run réel

Deux runs : Dev réduit (10 problèmes, α = 1), puis Dev complet (30 problèmes, α ∈ {0 ; 0,5 ; 1},
256 tokens au plus par réponse).

### 4.1 Résultats sur le Dev (30 problèmes GSM, 20 prompts Sécurité, 12 textes neutres)

| α | KL moyenne [IC 90 %] | Top-1 changé [IC 90 %] | GSM avant → après, différence [IC 90 %] | Sécurité avant → après, différence [IC 90 %] | Dev |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 [0 ; 0] | 0 [0 ; 0] | 0,600 → 0,600, 0 [−0,055 ; 0,055] | 0,850 → 0,850, 0 [−0,114 ; 0,114] | passe |
| 0,5 | 0,00063 [0,00052 ; 0,00075] | 0,011 [0,003 ; 0,022] | 0,600 → 0,533, −0,067 [−0,181 ; 0,052] | non mesurée | `INCONCLUSIVE` à `gsm` (n requis 44) |
| 1 | 0,0031 [0,0026 ; 0,0037] | 0,030 [0,015 ; 0,046] | 0,600 → 0,633, +0,033 [−0,074 ; 0,139] | 0,850 → 0,850, 0 [−0,158 ; 0,158] | `INCONCLUSIVE` à `safety` (n requis 22) |

Intervalles appariés de Newcombe (piliers binaires) et bootstrap sur les prompts (dérives).

Test, consulté une fois pour le seul candidat retenu (α = 0), agrégats seulement : KL 0,
top-1 0, GSM 0,600 → 0,600 [−0,081 ; 0,081], Sécurité 0,800 → 0,800 [−0,105 ; 0,105].

**Verdict : `GO` pour α = 0**, c'est-à-dire pour le patch sans effet. Les deux candidats qui
modifient le modèle sont `INCONCLUSIVE` : l'intervalle contient la marge, les échantillons
sont trop petits. C'est le comportement attendu, pas un bug : le pipeline ne conclut pas en
faveur d'un patch quand la puissance manque. Ce résultat valide l'outillage, pas le patch.

### 4.2 Vérifications

| Point | Preuve | Verdict |
| --- | --- | --- |
| 1. Dimensions | tableau de la section 1 ; vecteur de longueur 896 ; couche 12 < 24 | OK |
| 2. Gabarit de chat | gabarit officiel = 21 tokens de message système + la séquence exacte du `chatml` natif, sur 3 prompts | ÉCART-4 |
| 3. Fin de génération | 15 sorties brutes lues : 14 finissent sur EOS, 1 atteint le budget ; aucun marqueur résiduel | OK, avec ÉCART-1 et ÉCART-3 |
| 4. Scores GSM8K | 0,600 mesuré ; 26/30 à la relecture manuelle des sorties | ÉCART-3 |
| 5. Statistiques | 9 portes binaires recalculées hors dépôt : intervalle, McNemar exact, verdict, n requis ; erreur max 2,5e-10 ; tables 2×2 reconstruites depuis les items | OK |
| 6. KL et logits | recalcul en float64 depuis les logits bruts de deux dumps : KL par prompt à 8,9e-15, top-1 identique | OK |
| 7. Portes et ordre | α = 0,5 arrêté à `gsm` : aucun point de reprise `safety` ; témoin positif arrêté à `kl` en 8 s ; verdicts sur la borne basse | OK |
| 8. Coût | section 4.5 | ÉCART documenté, estimation disque corrigée |

### 4.3 Gabarit, génération, extraction

**Gabarit (ÉCART-4).** Le gabarit officiel de `tokenizer_config.json` ajoute « You are Qwen,
created by Alibaba Cloud. You are a helpful assistant. » quand aucun message système n'est
fourni. Le `chatml` natif ne l'ajoute pas, et il n'existe pas d'option `--system-prompt`.

| Mesure sur le Dev, modèle d'origine | Avec message système | Sans | Différence [IC 95 % apparié] |
| --- | --- | --- | --- |
| GSM, exactitude | 18/30 | 16/30 | −0,067 [−0,247 ; 0,121], 8 réponses changées sur 30 |
| Sécurité, taux de refus | 19/20 | 17/20 | −0,100 [−0,305 ; 0,090] |

Aucune différence n'est significative à ces tailles, mais le message système change la réponse
d'un item sur quatre. Contournements sans modifier le code :

- GSM8K : `prompt_template` avec le gabarit complet (ids identiques au rendu officiel) ;
- refus : jeu pré-rendu avec le gabarit officiel et `chat_template: none`.

Les runs de cette validation utilisent le gabarit officiel pour GSM et le `chatml` natif pour
la Sécurité.

**Fin de génération.** Sur 10 sorties brutes du Dev : 143 à 245 tokens puis EOS pour 9
d'entre elles, 256 pour une. Le compteur de troncatures n'existait pas ; il a été ajouté
(section 5). Sur le run complet : 1 troncature sur 30 au Dev, 3 sur 20 au Test.

**Extraction (ÉCART-1, non corrigé).** Deux faux négatifs sur 30 :

- le modèle écrit `#### 62` puis recopie `#### N` de la consigne ; la règle `hashes` prend la
  dernière occurrence ;
- le modèle écrit `#### Final Line: 3000`.

L'extracteur fait ce que sa spécification dit. Ni l'extracteur ni le gabarit ne sont ajustés.
Aucun `no_answer`.

### 4.4 Détecteur de boucles (ÉCART-3, non corrigé) : le constat principal

Sur le Dev, 7 réponses sur 30 du modèle d'origine sont coupées par le détecteur de boucles et
comptées fausses. Les 7 sorties brutes, régénérées sans détecteur, ont été relues : **aucune
n'est une boucle**. Six sont des solutions complètes et justes, la septième atteint le budget
de 256 tokens après le bon calcul.

Cause : la règle n-gramme signale tout groupe de 4 mots présent plus de 4 fois dans une
fenêtre de 4000 caractères, plus longue que la réponse entière. Une solution rédigée étape par
étape répète « the total number of » cinq fois.

Effet sur la mesure :

| Lecture | Exactitude du modèle d'origine sur le Dev |
| --- | --- |
| Pipeline (`bench eval`) | 18/30 = 0,600 |
| Relecture manuelle des sorties brutes | 26/30 = 0,867 |

L'écart vient de 6 fausses boucles et de 2 faux négatifs d'extraction. Le biais touche les
deux côtés d'une comparaison appariée, mais il transforme des paires discordantes possibles en
échecs concordants et réduit donc la puissance des portes GSM.

Pourquoi ce n'est pas corrigé ici : le détecteur est une heuristique partagée par `eval` et
`bench eval`, et tout nouveau critère demanderait un seuil réglé sur les données qui ont révélé
le défaut. La règle de cette validation l'interdit. Ce qui est livré :

- un test de reproduction synthétique, laissé rouge et ignoré :
  `a_step_by_step_answer_reusing_a_stock_phrase_is_not_a_loop` ;
- les compteurs `loops_before` / `loops_after` dans le résultat du pilier GSM, pour que l'effet
  se voie dans chaque rapport.

La décision de revoir la règle revient au mainteneur, avec un jeu de calibration indépendant.

Corrigé depuis, sur un jeu de calibration indépendant : voir la section 9.

### 4.5 Coût : plan contre réalité

| Run | Durée estimée | Durée réelle | Écart | Unités prévues / comptées |
| --- | --- | --- | --- | --- |
| Fumée (identité) | 383 s | 272 s | +41 % | 3264 / 3264 |
| Dev réduit | 1073 s | 388 s | +177 % | 11440 / 11440 |
| Dev complet | 4882 s | 3576 s | +37 % | 49600 / 49120 |

L'estimation de durée est une borne haute et dépasse 25 %. Causes mesurées :

- une génération compte pour `max_new_tokens`, alors qu'elle finit sur EOS vers 200 tokens en
  moyenne, ou plus tôt quand le détecteur de boucles la coupe ;
- un candidat arrêté à une porte ne mesure pas les piliers suivants, et le Test n'est pas
  consulté si aucun candidat ne passe (Dev réduit) ;
- la calibration repose sur 8 tokens : 9,8 à 11,4 tokens/s machine au repos (débit réel
  mesuré : 11,3), mais 6,1 juste après une compilation, ce qui a porté une estimation à 8137 s.

Elle n'est pas corrigée : la longueur des réponses n'est pas connue avant de générer. Le
pipeline n'enregistre pas le nombre de tokens réellement générés, les « unités » sont
nominales. Chargements du modèle : 28 prévus, 21 réels d'après les points de reprise.

Disque : 2166 Mo prévus ; 1319 Mo observés (taille du dossier du job échantillonnée toutes les
5 s : copie de travail et deux dumps de référence), 1474 Mo au pic calculé, quand le dump
candidat du Test s'y ajoute quelques secondes. Deux erreurs dans le calcul ont été corrigées
(BUG-6) ; le plan corrigé annonce 1484 Mo.

Mémoire d'un dump : 271 positions × 151936 × 4 = 164 698 624 octets de logits, fichier de
164 699 240 octets. Perplexité native sur le texte neutre (259 tokens) : 15,56 pour le modèle
d'origine, 15,52 pour le modèle patché à α = 1.

## 5. Bugs corrigés

Chaque correctif a été précédé d'un test rouge sur données synthétiques (mini-modèle factice
ou générateur scripté).

| Bug | Symptôme | Cause | Test de régression |
| --- | --- | --- | --- |
| BUG-1 | `NO_GO` en 2 s sans aucune mesure quand le patch garde `backup` par défaut | le dry-run passait `no_backup` avec le `backup` du gabarit | `a_patch_template_with_the_default_backup_passes_the_static_checks` |
| BUG-2 | raison « aucun candidat ne passe le Dev » alors que le Dev n'a pas été mesuré | pas de raison propre aux contrôles statiques | `candidates_refused_by_the_static_checks_are_not_reported_as_a_dev_failure` |
| BUG-3 | `plan` annonce un `INCONCLUSIVE` garanti qu'une table 5/5 à n = 10 réfute | la table la plus favorable est celle à taux 1/2, pas « tout du même côté » | `no_table_without_a_discordant_pair_is_go_below_the_lowest_n`, `plan_does_not_announce_a_guaranteed_inconclusive_that_a_table_can_refute` |
| BUG-4 | une annulation pendant `apply` est rapportée comme un échec | l'interruption remontait comme une erreur de patch | `a_cancellation_that_lands_during_apply_is_a_cancellation_not_a_failure` |
| BUG-5 | MCP : juste après `start`, `status` répond « aucun travailleur » et `cancel` est perdu | l'état du travailleur n'était lu que dans le verrou, écrit après son démarrage | `mcp_status_and_cancel_see_the_worker_as_soon_as_start_returns` |
| BUG-6 | `plan` : estimation disque des dumps fausse (4 dumps de 480 positions prévus, 3 de 271 et 255 réels ; sous-estimation possible avec un seul pilier de dérive) | dumps comptés par pilier à `max_tokens` positions, sans le dump de référence du Test | `plan_counts_the_dumps_that_coexist_at_their_real_size` |
| Compteur de troncatures | impossible de savoir si `max_new_tokens` a coupé une réponse | la raison de fin de génération était jetée | `generations_cut_by_the_token_budget_are_counted`, assertions ajoutées au test de décision complète |

Ajouts au JSON, tous additifs : `truncated` par item et au score de `bench eval` (avec un
avertissement), `truncated_*`, `loops_*` et `no_answer_*` par pilier `gsm8k` du pipeline,
`min_n_possible` dans le plan. Le drapeau `truncated` n'est renseigné que par le backend natif.

BUG-5 touche `src/mcp/tools_pipeline.rs`, hors de la liste des fichiers prévus : le défaut est
dans l'outil MCP lui-même.

## 6. Flux MCP

Échange JSON-RPC scripté avec `surgeon mcp --allow-dir` (dossier des modèles et dossier des
jobs), après BUG-5 :

| Vérification | Résultat |
| --- | --- |
| `plan` | estimations renvoyées |
| Chemin hors bac à sable (`job_dir`, `config`, fichier nommé par la configuration) | refusé, `path_not_allowed` |
| Configuration sans pilier `safety` | refusée à `plan` et à `start` |
| Arguments `allow_disable_safety`, `allow_test_reuse` | refusés, champ inconnu |
| `start` | rend la main en 12 à 15 ms ; second `start` refusé (`busy`) |
| `status` puis `cancel` | travailleur vu en cours, annulé sans résidu |
| `report` | 18 items des jeux de Test cherchés dans la réponse, 0 trouvé ; pas de clé `items` |
| `private/` | refusé comme `job_dir` et comme `config` |

## 7. Ce qui n'a pas été validé

- Le pilier Sécurité avec le message système officiel dans le pipeline, ni avec un juge : seuls
  les marqueurs de refus sont utilisés.
- Les backends llama.cpp et GGUF, `llama-perplexity`, le nettoyage des marqueurs de fin de
  texte côté llama.cpp : llama.cpp n'est pas installé.
- Le GPU, les modèles plus gros, les autres architectures.
- Le vrai Test Set de l'utilisateur, jamais lu.
- Un patch à effet réel : les échantillons de 20 à 30 prompts ne permettent pas de conclure
  sur α = 0,5 ou α = 1.
- La puissance de détection en dessous du témoin positif : seul un patch destructeur
  (KL 10,4) a été montré détecté.

## 8. Reproduire

```bash
cargo build --release
surgeon pipeline plan --config pipeline.json --job-dir runs/p4-full
surgeon pipeline run  --config pipeline.json --job-dir runs/p4-full --json
```

Validation du dépôt : `cargo check --all-targets`, `cargo test`,
`cargo clippy --all-targets -- -D warnings`, `cargo fmt --check`.

Checklist avant de se fier à un verdict du pipeline sur un autre modèle :

- [ ] `plan` : `dev_underpowered` et `test_underpowered` à false, durée sous le budget ;
- [ ] contrôle d'identité (α = 0) : KL exactement 0, aucune paire discordante ;
- [ ] témoin positif : un patch destructeur donne `NO_GO` ;
- [ ] `truncated_*` et `loops_*` du pilier GSM proches de 0, sinon relire des sorties brutes ;
- [ ] gabarit de chat comparé au gabarit officiel du modèle (message système) ;
- [ ] `model.unchanged` à true dans le rapport.

## 9. Addendum du 2026-11-12 : recalibration du détecteur de boucles (ÉCART-3)

Branche `fix/loop-detector`. Rapport complet, hors dépôt : `outputs/reports/loop-calibration.md`.

**Protocole.** Aucun réglage sur les données de cette validation. Calibration sur 50 problèmes
du split `train` officiel de GSM8K (un tous les 149), sans doublon avec les jeux de
`.jobs/qwen-test/data/` (SHA-256 des questions normalisées, 0 collision sur 55 empreintes,
aucun énoncé affiché), et sur 29 boucles : 20 synthétiques (1-2, 4-8 et 12-20 tokens,
ponctuation, balises) et 9 sorties réelles d'une copie du modèle détruite par un `subtract`
sur `down_proj` des couches 10 à 12, les écrivains les plus couplés du graphe. Le Dev GSM n'a
été exécuté qu'une fois, après gel des paramètres et tests au vert.

**Nouvelle règle** (`src/eval/loop_detector.rs`) :

- blocs de mots : un bloc de 1 à 64 mots doit se répéter d'affilée en fin de flux, au moins
  3 fois et sur au moins 24 mots après la première copie ; l'ancien compte d'un 4-gramme dans
  une fenêtre de 4000 caractères disparaît (`ngram_size` et `max_ngram_repeats` retirés, raison
  JSON `ngram_repetition` gardée) ;
- périodicité de caractères : blancs réduits à un espace, 9 copies au lieu de 6 ; un motif
  sans lettre ni chiffre compte au-delà de 160 caractères ;
- marqueurs : « Wait » et les autres doivent être suivis d'une ponctuation ou d'un saut de
  ligne (« wait time » ne compte plus) ;
- coût : O(104) comparaisons d'entiers par caractère, aucune allocation par mot.

**Calibration.**

| Jeu | Avant | Après |
| --- | --- | --- |
| 50 solutions saines | 11 faux positifs (10 n-grammes, 1 « wait time ») | 0 |
| 29 boucles | 23 détectées | 25 détectées (20/20 synthétiques) |

Les 4 boucles manquées sont des rafales du modèle détruit : deux dans les 80 premiers
caractères (contrôles muets par conception), un nombre à 13 chiffres mal étiqueté par
l'oracle, une rafale de 7 copies sous le seuil de 9.

**Dev GSM, modèle d'origine, passe unique** (mêmes réglages que le pilier `gsm`) :

| Mesure | Avant | Après |
| --- | --- | --- |
| Boucles détectées | 7 | 0 |
| Exactitude mesurée | 18/30 = 0,600 | 23/30 = 0,767 [0,591 ; 0,882] |
| Réponses coupées par le budget de 256 tokens | 1 | 3 |
| Durée de `bench eval` | non mesurée isolément | 1022 s |

Les 7 échecs restants : 3 faux négatifs d'extraction (ÉCART-1, toujours ouvert), 2 réponses
coupées par le budget, 2 erreurs du modèle. Sur le contenu, 26/30, comme la relecture
manuelle de la section 4.4.

**Tests.** `a_step_by_step_answer_reusing_a_stock_phrase_is_not_a_loop` réactivé sans
changer son assertion. Ajoutés : structures mathématiques répétitives (étapes, doublements,
`2^8` écrit en entier, listes, tableau, « wait time »), boucles courtes, moyennes et longues
avec délai d'arrêt borné, découpage du flux, et cohérence de la raison `loop` avec
`score.loops` relu par `loops_before` / `loops_after`. `repeated_ngram_with_varying_tail`
devient `a_template_with_changing_numbers_is_not_a_word_block_loop` : une phrase répétée avec
des nombres qui changent a la forme d'un raisonnement sain et n'est plus coupée.

SHA-256 des fichiers du modèle d'origine inchangés.
