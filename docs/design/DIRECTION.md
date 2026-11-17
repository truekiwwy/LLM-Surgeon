# Direction : Weight Surgeon, interface native

Depuis G7, la référence visuelle est **Craft Anything** (thème « Studio ») : même famille de
fenêtre, mêmes surfaces, même police, mêmes contrôles. Ce document fixe ce que toute vue doit
respecter. Les décisions de G5 et G6 restent au journal (§ 7) ; celles que G7 remplace sont
barrées.

## 1. Les cinq lignes

- **Produit.** « Vibe-éditer » un modèle open weight : on parle à un agent IA qui inspecte le
  modèle, trouve ou cible les tenseurs en cause et propose des corrections mesurées, sur une
  copie, jamais sur l'original.
- **Caractère.** Un IDE sans éditeur de code : calme, net, l'agent au centre. Les outils
  d'ingénierie (graphe 3D, preuves) restent à portée sans prendre la place.
- **Monde.** La fenêtre d'IDE sombre et douce de Craft Anything : surfaces presque noires,
  contrôles arrondis, un seul bouton clair par écran.
- **Moments.** Deux : le formulaire de nouvelle session (une question, des choix en tuiles,
  `Démarrer`) et le graphe 3D, seul visuel riche, où l'agent montre un tenseur.
- **Signature.** Le logomark carré en grille de blocs, et le bouton principal blanc cassé.

## 2. Règle de l'accent

- `action` (`#F2F1EE`, texte `#101014`) remplit **une seule action principale par écran** :
  `Démarrer`, `Autoriser une fois`, `Suivant`. Les autres boutons sont secondaires (surface
  et contour) ou fantômes.
- `accent` (`#2F74E0`) marque **un seul focus d'attention par région** : onglet actif
  (soulignement), activité active, choix sélectionné (contour et coche d'une tuile), section
  active des réglages, curseur et anneau de focus, interrupteur actif. Jamais du texte courant.
- États : OK, avertissement et erreur sont toujours un point dessiné plus un mot.
- **Plafond d'accent** : part des pixels à moins de 24 (Tchebychev, RVB) de `accent`, hors
  canevas 3D, sous **3 %** par écran (test `ui_render`). Le bouton principal clair n'y entre
  pas : il est mesuré par la règle « un seul par écran ».

## 3. Volontairement absent

Dégradés, halos, ombres, verre, emojis, badges encadrés, cartes imbriquées, spinners,
animations de plus de 120 ms, texte en dégradé, titres en capitales, barre d'état. Les seules
illustrations sont des données (le graphe 3D).

## 4. Jetons

| Jeton | Valeur | Usage |
| --- | --- | --- |
| `window` | `#101014` | barre de titre, barre d'activité, colonnes latérales |
| `canvas` | `#121316` | centre de la fenêtre, canevas 3D |
| `surface` | `#1B1C1F` | champs, cartes, zone de saisie, menus, réglages |
| `surface_hover` | `#222325` | survol, piste d'un contrôle segmenté, bulle de l'utilisateur |
| `selection` | `#1E2C47` | ligne, tuile, activité ou section sélectionnée |
| `border` | `#2B2B2F` | contour d'un contrôle ou d'une carte, 1 px physique |
| `separator` | `#262729` | traits entre les zones et dans un panneau |
| `accent` | `#2F74E0` | focus d'attention (≥ 3:1 sur toutes les surfaces) |
| `action` | `#F2F1EE` | action principale (survol `#FFFFFF`), texte `on_action` `#101014` |
| `text` | `#F2F1EE` | texte principal (≥ 7:1) |
| `text_secondary` | `#949494` | libellés, méta (≥ 4,5:1) |
| `text_disabled` | `#757575` | désactivé, indices (≥ 3:1) |
| `ok` / `warn` / `error` | `#3FB950` / `#D29922` / `#FF6A61` | états (≥ 4,5:1), toujours avec un mot |
| `data[0..7]` | sept paliers d'un seul bleu | échelles de données du graphe 3D |

Valeurs relevées à la pipette sur trois captures de Craft Anything (barre de titre, fonds,
champs, tuile sélectionnée, réglages). Les ratios sont calculés par `theme_contrast` et
`--self-test` (WCAG 2.x).

**Rayons.** `radius` 6 pour les contrôles (boutons, champs, lignes, segments, tuiles),
`radius_card` 10 pour les cartes, la zone de saisie et les dialogues.

**Espacements.** Dimensions de mise en page (colonnes, barres, lignes) multiples de **8** ;
paddings internes dans **{4, 8, 12, 16, 24}**. Barre de titre et barre d'activité de 48,
lignes et contrôles de 32, action principale de 40, en-tête de panneau de 40, colonne de
lecture du centre plafonnée à 800.

**Typographie.** **DM Sans** (Regular, Medium, Bold ; SIL OFL 1.1), la police de Craft
Anything, et **JetBrains Mono** pour les tenseurs, formes, normes et journaux. Six tailles :
12 (méta), 12 (mono), 14 (corps), 15 (titres de panneau, onglets, chapeau), 16 (marque,
étapes), 24 (question d'une page, en gras). Casse de phrase partout.

## 5. Composition

Une fenêtre sans bordure système, organisée comme un IDE dont l'éditeur est l'agent.

- **Barre de titre (48).** Logomark au-dessus de la barre d'activité, `Weight Surgeon`. À
  droite : le travail en cours (installation, jobs), l'interrupteur `Local`, le sélecteur
  `connexion · modèle`, la roue des réglages, les boutons de fenêtre.
- **Barre d'activité (48).** `Agent` (badge du nombre d'approbations en attente), `Graphe 3D` ;
  en bas, la palette (`Ctrl+K`). Recliquer l'activité active replie la colonne gauche
  (`Ctrl+B`).
- **Trois zones bord à bord**, séparées par un trait et redimensionnables par lui.

| Vue | Gauche | Centre | Droite |
| --- | --- | --- | --- |
| Agent | `Nouvelle session`, projet et copie, sessions | sans projet : création ; sans session : formulaire (Advance / Scalping) ; sinon le fil et la zone de saisie | `Preuves` |
| Graphe 3D | explorateur (projet, copie, arbre) | vue 3D, rejeu, chronologie des flux | inspecteur, chronologie |

**Réglages** (`Ctrl+,`) : dialogue de 1120 × 760 au plus, sections à gauche (`Général`, `IA`,
`Moteur et machine`, `Mises à jour`, `Journal`), contenu à droite ; Échap ou un clic dehors
le ferme. **Assistant** de premier lancement : page sur tout le corps, quatre étapes.

**Formulaire de nouvelle session** (calqué sur « Que voulez-vous créer ? ») : le choix
`Scan complet (Advance)` / `Problème précis (Scalping)` en contrôle segmenté, la question en
24 gras et son chapeau, puis une carte en étapes numérotées : 1 le modèle et sa copie, 2 le
symptôme en six tuiles (Scalping), 3 la description (ou des précisions facultatives), des
idées en puces, ce que l'agent peut faire, le fournisseur, et `Démarrer` à droite.

## 6. Approches considérées (G7)

1. **Garder la direction G5 et réorganiser seulement** (agent en premier onglet). Écarté :
   la demande vise l'allure de Craft Anything (rayons, police, bouton clair) autant que la
   disposition.
2. **Agent au centre, Graphe 3D en vue secondaire, Atelier et Analyses retirés** (retenu).
   Tout passe par l'agent ; le 3D reste le moment visuel et la preuve de ce que l'agent
   désigne.
3. **Agent seul, graphe dans l'inspecteur.** Écarté : une vue 3D dans une colonne de 320 ne
   se lit pas.

## 7. Journal des décisions

| # | Décision | Raison |
| --- | --- | --- |
| D-01 | Aucune question posée en G5 ; les étapes d'entretien remplacées par ce document | consigne du brief G5 |
| D-02 | ~~Plus d'écran d'accueil~~ remplacée par D-31 | |
| D-03 | ~~Onglet `Projets` réparti entre Atelier et Analyses~~ remplacée par D-31 | |
| D-04 | ~~Cartes et éditeur de patch dans l'Atelier~~ remplacée par D-32 | |
| D-05 | ~~Le rôle d'un nœud 3D se lit à sa forme, pas à sa couleur~~ remplacée par D-40 | |
| D-06 | Copie active = dossier le plus récent de `copies/` ; « modifiée » = au moins une écriture réussie du ledger sur cette copie | calculé depuis le ledger, jamais déclaré |
| D-07 | ~~`[ADVANCE]` lance l'agent~~ remplacée par D-14, puis D-30 | |
| D-08 | ~~L'assistant garde `Lang::pick`~~ remplacée par D-29 | |
| D-09 | ~~Police Barlow~~ remplacée par D-33 | |
| D-10 | Pas de MSAA demandé au contexte GL | l'antialiasing vient du feathering d'epaint |
| D-11 à D-29 | Décisions G6 : [G6_PLAN.md](G6_PLAN.md). D-11, D-12, D-14, D-15, D-16, D-18 et D-19 sont remplacées par D-30 à D-34 ; les autres (dépendances, détection, fournisseurs, `local_only`, assistant) restent | |
| D-30 | `Advance` / `Scalping` sont les deux missions de l'agent : scan complet où l'agent trouve lui-même les problèmes, ou symptôme précis décrit par l'utilisateur. Le choix est enregistré avec la conversation et rappelé à l'agent dans son prompt système ; il ne touche ni la politique ni le contrat | demande G7 : « vibe-éditer » un modèle en discutant avec un agent |
| D-31 | Deux vues seulement, `Agent` et `Graphe 3D`, dans une barre d'activité ; l'agent ouvre l'application | l'interface principale est un IDE sans fichiers dont l'éditeur est l'agent |
| D-32 | Atelier (histogramme, cibles, formulaires d'outils), Analyses (historique), cartes et éditeur de patch retirés de l'interface et du code | choix de l'utilisateur : tout passe par l'agent, seul le 3D reste |
| D-33 | Thème « Studio » de Craft Anything : palette relevée à la pipette, DM Sans extraite de l'application (OFL 1.1), rayons 6 et 10, bouton principal blanc cassé, plus de capitales ni de barre d'état | coller à Craft Anything, plus propre et minimaliste |
| D-34 | `Système` devient le dialogue `Réglages` (Général, IA, Moteur et machine, Mises à jour, Journal) ; langue, `Local` et moteur d'inférence y vont avec la barre d'état ; les approbations en attente se lisent au badge de l'activité `Agent` | configurer une fois sans quitter le travail |
| D-35 | Plus de panneau Contrat : ce que l'agent peut faire (Demander / Modifier) se choisit dans le formulaire ; questions, budgets et règles gardent leurs valeurs par défaut ; plus de bouton « Toujours pour ce projet » sur les cartes | demande de l'utilisateur ; sans panneau, une règle « toujours » ne pourrait plus être révoquée |
| D-36 | OpenCode et Antigravity (`agy`) deviennent des runtimes avec leurs outils natifs (terminal, fichiers), pour ne pas brider les modèles. La garantie sur le modèle d'origine ne vient plus de la liste d'outils mais du verrou noyau Windows (`FILE_SHARE_READ`) tenu tant que le projet est ouvert ; OpenCode reste dans le dossier du projet (`external_directory: deny`), Antigravity tourne dans un profil isolé avec la connexion copiée le temps du run | choix de l'utilisateur : outils natifs gardés, origine protégée par le système |
| D-37 | Backend d'inférence : Rust seulement dans l'application (plus de llama.cpp : ni option, ni installation, ni lignes de dépendances). Au premier lancement, `backend_choice` écrit le backend d'après le GPU : mistral.rs (NVIDIA, Apple Silicon, CPU), hipfire (AMD) ; en attendant, mistral.rs puis le moteur natif. Le choix suit la machine jusqu'à un enregistrement à la main ; un fichier qui nomme encore llama-server est repris | demande de l'utilisateur : « pas de llama.cpp, que du Rust » |
| D-38 | Un agent CLI travaille seul sur le projet ouvert : chemins exacts et consigne d'autonomie dans son prompt, copie de travail créée par « Démarrer », jeux de sondes `builtin:*` dans le moteur, et `record_finding` / `show_in_3d` / `update_plan` servis par le MCP `surgeon` avec un journal `chat/mcp_runs.jsonl` qui rend ses constats vérifiables. La sonde du moteur (`surgeon_backends`, `backend detect`) suit la règle D-37 | un vrai test avec Claude Code : l'agent demandait des chemins et un dataset, et ni les Preuves ni la vue 3D ne bougeaient |
| D-39 | Passe de finition sur Craft Anything (`crates/ui-egui/src/theme.rs` du dépôt : palette Studio identique à la nôtre, on la garde) : barre de titre en capsule segmentée bordée (`components::capsule`, `[projet · état de la copie ● \| Local ●]`), zone de saisie en un seul cadre avec le bouton d'envoi imbriqué et une ligne dédiée dessous pour le message et le raccourci, appels d'outils repliés sur une ligne (`components::tool_row`, les CLI produisent de vrais blocs `ToolUse` / `ToolResult`), carte Plan sans double numérotation, états vides centrés (badge rond bordé de 36, titre, deux lignes) | retour d'un test réel : textes superposés, pavés JSON bruts, état vide en haut à gauche |
| D-40 | La vue 3D quitte le filaire : nœuds ombrés colorés par rôle (embeddings et tête cyan, attention indigo, MLP ambre), écart Δ du corail au magenta, la forme gardée pour qui ne distingue pas les couleurs. Disposition calculée par le GUI (`scene.rs`) depuis rôle et couche : Colonne en grille (attention à gauche, MLP à droite, flux résiduel au centre), Constellation en hélice ; les dispositions spectrales du rapport ne sont plus dessinées. Repères `L0…Ln`, anneau pulsant et carte de la sélection, couches de l'outil en cours de l'agent qui ondulent sous un bandeau | demande de l'utilisateur : le filaire gris se lisait mal et ne montrait pas l'agent au travail |
